"""Regression tests for OPF char-span evaluation helpers."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_char_spans_v2.py"


def load_module():
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda text: {}))
    spec = importlib.util.spec_from_file_location("eval_char_spans_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_partial_overlap_and_cost_metrics() -> None:
    mod = load_module()
    gold = {"example": [("AU_TFN", 0, 9), ("PERSON", 20, 25)]}
    pred = {"example": [("AU_TFN", 0, 8), ("PERSON", 30, 35)]}

    metrics = mod.evaluate(
        gold,
        pred,
        parent={},
        diag={"strategy": "unit"},
        cost_weights={"AU_TFN": 5, "PERSON": 3},
    )

    assert metrics["typed"]["tp"] == 0
    assert metrics["typed"]["fp"] == 2
    assert metrics["typed"]["fn"] == 2
    assert metrics["typed"]["partial"]["tp"] == 1
    assert metrics["typed"]["partial"]["fp"] == 1
    assert metrics["typed"]["partial"]["fn"] == 1
    assert metrics["typed"]["over_redaction_cost"] == 13
    assert metrics["typed"]["under_redaction_cost"] == 60


if __name__ == "__main__":
    test_partial_overlap_and_cost_metrics()
    print("PASS test_partial_overlap_and_cost_metrics")
