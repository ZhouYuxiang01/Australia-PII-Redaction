from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pii_prep.qwen_spancls_smoke import (
    SpanClsJsonlDataset,
    SpanCollator,
    load_backbone_and_tokenizer,
    write_json,
)


def build_cache_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": row.get("id"),
        "split": row.get("split"),
        "source": row.get("source"),
        "start": int(row.get("start")),
        "end": int(row.get("end")),
        "value": row.get("value"),
        "top_type": row.get("top_type"),
        "target_distribution": row.get("target_distribution", {}),
        "training_weight": float(row.get("training_weight", 1.0)),
    }


def pool_span_embeddings(hidden: torch.Tensor, span_token_ranges: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_embeddings = []
    first_embeddings = []
    last_embeddings = []
    for batch_index, (start, end) in enumerate(span_token_ranges):
        span_hidden = hidden[batch_index, start:end]
        mean_embeddings.append(span_hidden.mean(dim=0))
        first_embeddings.append(span_hidden[0])
        last_embeddings.append(span_hidden[-1])
    if not mean_embeddings:
        empty = torch.empty((0, hidden.shape[-1]), device=hidden.device, dtype=hidden.dtype)
        return empty, empty, empty
    return (
        torch.stack(mean_embeddings, dim=0),
        torch.stack(first_embeddings, dim=0),
        torch.stack(last_embeddings, dim=0),
    )


class CacheWriter:
    def __init__(self, final_path: Path, *, split: str, chunk_size: int = 5000, dtype: torch.dtype = torch.float16):
        self.final_path = final_path
        self.split = split
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.chunks_dir = final_path.with_suffix(".chunks")
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.mean_parts: list[torch.Tensor] = []
        self.first_parts: list[torch.Tensor] = []
        self.last_parts: list[torch.Tensor] = []
        self.next_chunk_index = self._existing_chunk_count()

    def completed_count(self) -> int:
        total = 0
        for chunk_path in self._chunk_paths():
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            total += len(chunk["records"])
        return total

    def append(self, rows: list[dict[str, Any]], mean: torch.Tensor, first: torch.Tensor, last: torch.Tensor) -> None:
        for index, row in enumerate(rows):
            self.records.append(build_cache_record(row))
            self.mean_parts.append(mean[index].detach().cpu().to(self.dtype))
            self.first_parts.append(first[index].detach().cpu().to(self.dtype))
            self.last_parts.append(last[index].detach().cpu().to(self.dtype))
            if len(self.records) >= self.chunk_size:
                self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        chunk_path = self.chunks_dir / f"chunk_{self.next_chunk_index:06d}.pt"
        if chunk_path.exists():
            raise FileExistsError(f"refusing to overwrite existing cache chunk: {chunk_path}")
        torch.save(
            {
                "split": self.split,
                "records": self.records,
                "mean_embeddings": torch.stack(self.mean_parts, dim=0),
                "first_embeddings": torch.stack(self.first_parts, dim=0),
                "last_embeddings": torch.stack(self.last_parts, dim=0),
                "dtype": str(self.dtype).replace("torch.", ""),
            },
            chunk_path,
        )
        print(f"saved {chunk_path} ({len(self.records)} examples)", flush=True)
        self.next_chunk_index += 1
        self.records = []
        self.mean_parts = []
        self.first_parts = []
        self.last_parts = []

    def close(self) -> dict[str, Any]:
        self.flush()
        chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in self._chunk_paths()]
        records = [record for chunk in chunks for record in chunk["records"]]
        if records:
            mean = torch.cat([chunk["mean_embeddings"] for chunk in chunks], dim=0)
            first = torch.cat([chunk["first_embeddings"] for chunk in chunks], dim=0)
            last = torch.cat([chunk["last_embeddings"] for chunk in chunks], dim=0)
            embedding_dim = int(mean.shape[1])
        else:
            mean = first = last = torch.empty((0, 0), dtype=self.dtype)
            embedding_dim = 0
        payload = {
            "split": self.split,
            "records": records,
            "mean_embeddings": mean,
            "first_embeddings": first,
            "last_embeddings": last,
            "embedding_dim": embedding_dim,
            "dtype": str(self.dtype).replace("torch.", ""),
            "chunk_size": self.chunk_size,
            "chunk_count": len(chunks),
            "schema": {
                "records": "metadata for each cached example",
                "mean_embeddings": "[num_examples, embedding_dim], aligned with records by index",
                "first_embeddings": "[num_examples, embedding_dim], aligned with records by index",
                "last_embeddings": "[num_examples, embedding_dim], aligned with records by index",
            },
        }
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.final_path)
        return {
            "final_path": str(self.final_path),
            "chunk_dir": str(self.chunks_dir),
            "chunk_count": len(chunks),
            "final_size_bytes": self.final_path.stat().st_size,
            "chunk_size_bytes": sum(path.stat().st_size for path in self._chunk_paths()),
            "examples": len(records),
            "embedding_dim": embedding_dim,
        }

    def _existing_chunk_count(self) -> int:
        return len(self._chunk_paths())

    def _chunk_paths(self) -> list[Path]:
        return sorted(self.chunks_dir.glob("chunk_*.pt"))


def extract_hidden(backbone: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": False,
        "return_dict": True,
    }
    try:
        outputs = backbone(**kwargs, use_cache=False)
    except TypeError:
        outputs = backbone(**kwargs)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    kwargs["output_hidden_states"] = True
    try:
        outputs = backbone(**kwargs, use_cache=False)
    except TypeError:
        outputs = backbone(**kwargs)
    return outputs.hidden_states[-1]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported cache dtype: {name}")


def cache_split(
    *,
    root: Path,
    split: str,
    model_path: str,
    labels: list[str],
    backbone: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
    chunk_size: int,
    cache_dtype: torch.dtype,
    cache_name_prefix: str = "qwen_spancls_embeddings",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_path = root / "data" / "train" / f"qwen_spancls_{split}.jsonl"
    cache_path = root / "data" / "cache" / f"{cache_name_prefix}_{split}.pt"
    dataset = SpanClsJsonlDataset(data_path, labels)
    writer = CacheWriter(cache_path, split=split, chunk_size=chunk_size, dtype=cache_dtype)
    completed = writer.completed_count()
    dataset.rows = dataset.rows[completed:]
    collator = SpanCollator(tokenizer, labels, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    mapping_errors: list[dict[str, Any]] = []
    processed = 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            mapping_errors.extend(batch.mapping_errors)
            if not batch.rows:
                continue
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            hidden = extract_hidden(backbone, input_ids, attention_mask)
            mean, first, last = pool_span_embeddings(hidden, batch.span_token_ranges)
            writer.append(batch.rows, mean, first, last)
            processed += len(batch.rows)
    cache_stats = writer.close()
    elapsed = max(time.time() - started, 1e-6)
    report = {
        "split": split,
        "input_path": str(data_path),
        "cache_path": str(cache_path),
        "total_examples": completed + len(dataset.rows),
        "resumed_completed_examples": completed,
        "newly_processed_examples": processed,
        "processed_examples": cache_stats["examples"],
        "cached_examples": cache_stats["examples"],
        "mapping_failure_count": len(mapping_errors),
        "skipped_examples": len(mapping_errors),
        "labels_outside_training_space": dict(dataset.labels_outside),
        "embedding_dim": cache_stats["embedding_dim"],
        "dtype": str(cache_dtype).replace("torch.", ""),
        "disk_size_bytes": {
            "final": cache_stats["final_size_bytes"],
            "chunks": cache_stats["chunk_size_bytes"],
            "total": cache_stats["final_size_bytes"] + cache_stats["chunk_size_bytes"],
        },
        "throughput_examples_per_min": round(processed / elapsed * 60, 3),
        "chunk_count": cache_stats["chunk_count"],
    }
    return report, mapping_errors


def cache_all_splits(
    root: Path | str = ".",
    *,
    model_path: str = "/home/admin/model/Qwen3.5-9B-Base",
    splits: list[str] | None = None,
    batch_size: int = 1,
    max_length: int = 1536,
    chunk_size: int = 5000,
    cache_dtype_name: str = "float16",
    cache_name_prefix: str = "qwen_spancls_embeddings",
) -> dict[str, Any]:
    root = Path(root)
    splits = splits or ["train", "dev", "test"]
    labels = json.loads((root / "pii_schema" / "training_label_space_80.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(model_path, device, use_bf16)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    cache_dtype = dtype_from_name(cache_dtype_name)
    split_reports: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    labels_outside = Counter()
    started = time.time()
    for split in splits:
        split_report, split_errors = cache_split(
            root=root,
            split=split,
            model_path=model_path,
            labels=labels,
            backbone=backbone,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            chunk_size=chunk_size,
            cache_dtype=cache_dtype,
            cache_name_prefix=cache_name_prefix,
        )
        split_reports[split] = split_report
        all_errors.extend({"split": split, **error} for error in split_errors)
        labels_outside.update(split_report["labels_outside_training_space"])
    trainable_params = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
    report = {
        "stage": "3A.2",
        "model_path": model_path,
        "model_loaded_successfully": True,
        "label_count": len(labels),
        "hidden_size": hidden_size,
        "embedding_dim": {split: split_reports[split]["embedding_dim"] for split in split_reports},
        "dtype": cache_dtype_name,
        "device": str(device),
        "bf16_backbone": use_bf16,
        "batch_size": batch_size,
        "max_length": max_length,
        "chunk_size": chunk_size,
        "split_reports": split_reports,
        "total_examples_per_split": {split: split_reports[split]["total_examples"] for split in split_reports},
        "processed_examples_per_split": {split: split_reports[split]["processed_examples"] for split in split_reports},
        "mapping_failure_count": len(all_errors),
        "skipped_examples": sum(split_reports[split]["skipped_examples"] for split in split_reports),
        "labels_outside_training_space": dict(labels_outside),
        "qwen_trainable_parameter_count": trainable_params,
        "no_qwen_parameters_trainable": trainable_params == 0,
        "lora_started": False,
        "opf_started": False,
        "classifier_full_training_started": False,
        "wall_time_seconds": round(time.time() - started, 3),
    }
    reports_dir = root / "reports"
    write_json(reports_dir / "stage3a_qwen_embedding_cache_report.json", report)
    write_json(reports_dir / "stage3a_qwen_embedding_cache_errors.json", {"mapping_failure_count": len(all_errors), "errors": all_errors[:500]})
    if labels_outside:
        raise SystemExit("labels outside training space found")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-path", default="/home/admin/model/Qwen3.5-9B-Base")
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--cache-name-prefix", default="qwen_spancls_embeddings",
                        help="output file/dir prefix under data/cache, e.g. qwen4b_spancls_embeddings")
    args = parser.parse_args(argv)
    report = cache_all_splits(
        args.root,
        model_path=args.model_path,
        splits=[split.strip() for split in args.splits.split(",") if split.strip()],
        batch_size=args.batch_size,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        cache_dtype_name=args.cache_dtype,
        cache_name_prefix=args.cache_name_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["no_qwen_parameters_trainable"]:
        raise SystemExit("unexpected trainable Qwen parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
