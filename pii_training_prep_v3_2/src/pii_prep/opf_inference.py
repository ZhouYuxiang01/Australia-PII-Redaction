from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


OPF_ROOT = "/home/admin/ZYX/opf_au_pii/privacy-filter"
CONDA_SH = "/home/admin/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "opf"


def _build_opf_command(
    text: str,
    checkpoint: str,
    device: str = "cuda",
) -> list[str]:
    return [
        "bash",
        "-c",
        (
            f'source {CONDA_SH} && conda activate {CONDA_ENV}'
            f' && cd {OPF_ROOT} && export PYTHONPATH=.'
            f' && python -m opf redact'
            f' --format json'
            f' --output-mode typed'
            f' --checkpoint {checkpoint}'
            f' --device {device}'
            f' --no-print-color-coded-text'
            f" {_shell_quote(text)}"
        ),
    ]


def _shell_quote(text: str) -> str:
    escaped = text.replace("'", "'\\''")
    return f"'{escaped}'"


class OPFDetector:
    def __init__(
        self,
        checkpoint: str,
        *,
        opf_root: str = OPF_ROOT,
        conda_sh: str = CONDA_SH,
        conda_env: str = CONDA_ENV,
        device: str = "cuda",
        timeout_s: float = 120.0,
    ):
        self.checkpoint = str(checkpoint)
        self.opf_root = str(opf_root)
        self.conda_sh = str(conda_sh)
        self.conda_env = str(conda_env)
        self.device = device
        self.timeout_s = timeout_s

    def detect_spans(self, text: str) -> dict[str, Any]:
        cmd = _build_opf_command(text, self.checkpoint, device=self.device)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.opf_root,
            )
        except subprocess.TimeoutExpired:
            return {
                "text": text,
                "error": "opf_timeout",
                "candidate_spans": [],
                "summary": {"span_count": 0},
            }

        if result.returncode != 0:
            return {
                "text": text,
                "error": f"opf_exit_{result.returncode}",
                "stderr": result.stderr[:2000],
                "candidate_spans": [],
                "summary": {"span_count": 0},
            }

        output = result.stdout.strip()
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            json_start = output.find('{"')
            if json_start >= 0:
                json_end = output.rfind('}') + 1
                try:
                    data = json.loads(output[json_start:json_end])
                except json.JSONDecodeError:
                    return {
                        "text": text,
                        "error": "opf_json_parse_failed",
                        "raw_output": output[:2000],
                        "candidate_spans": [],
                        "summary": {"span_count": 0},
                    }
            else:
                return {
                    "text": text,
                    "error": "opf_json_parse_failed",
                    "raw_output": output[:2000],
                    "candidate_spans": [],
                    "summary": {"span_count": 0},
                }

        detected = data.get("detected_spans", [])
        spans = []
        for ds in detected:
            spans.append({
                "start": int(ds["start"]),
                "end": int(ds["end"]),
                "value": str(ds.get("text", text[int(ds["start"]):int(ds["end"])])),
                "opf_top_type": str(ds["label"]),
                "opf_confidence": ds.get("confidence"),
            })

        return {
            "text": text,
            "candidate_spans": spans,
            "summary": {
                "span_count": len(spans),
                "by_label": data.get("summary", {}).get("by_label", {}),
                "redacted_text": data.get("redacted_text", ""),
            },
        }

    @classmethod
    def from_project_root(
        cls,
        root: str | Path = ".",
        *,
        run_name: str = "opf_hard_79",
        device: str = "cuda",
    ) -> OPFDetector:
        root = Path(root)
        checkpoint = str(root / "runs" / run_name)
        return cls(checkpoint, device=device)
