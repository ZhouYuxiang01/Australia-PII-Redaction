"""Tests for Viterbi calibration search helpers."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_viterbi_biases.py"


def load_module():
    spec = importlib.util.spec_from_file_location("calibrate_viterbi_biases", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_biases_are_complete_and_include_baseline() -> None:
    mod = load_module()
    candidates = mod.default_candidates()

    assert candidates[0].name == "baseline_zero"
    assert len(candidates) >= 5
    for candidate in candidates:
        assert set(candidate.biases) == set(mod.BIAS_KEYS)
        assert all(isinstance(value, float) for value in candidate.biases.values())


def test_calibration_artifact_uses_opf_schema(tmp_path: Path) -> None:
    mod = load_module()
    biases = {key: 0.0 for key in mod.BIAS_KEYS}
    biases["transition_bias_background_to_start"] = 0.05
    out = tmp_path / "viterbi_calibration.json"

    mod.write_calibration_artifact(out, biases)

    assert json.loads(out.read_text(encoding="utf-8")) == {
        "operating_points": {
            "default": {
                "biases": biases,
            },
        },
    }


def test_select_best_prefers_objective_then_precision() -> None:
    mod = load_module()
    rows = [
        {
            "candidate": "lower_precision",
            "typed_f1": 0.8,
            "typed_precision": 0.7,
            "typed_recall": 0.95,
        },
        {
            "candidate": "higher_precision",
            "typed_f1": 0.8,
            "typed_precision": 0.9,
            "typed_recall": 0.8,
        },
    ]

    assert mod.select_best(rows, "typed_f1")["candidate"] == "higher_precision"


def test_wide_candidate_set_includes_stronger_probe_biases() -> None:
    mod = load_module()
    candidates = mod.candidate_subset("wide")

    assert candidates[0].name == "baseline_zero"
    assert any(
        abs(value) >= 0.25
        for candidate in candidates
        for value in candidate.biases.values()
    )


if __name__ == "__main__":
    test_candidate_biases_are_complete_and_include_baseline()
    test_calibration_artifact_uses_opf_schema(Path("/tmp"))
    test_select_best_prefers_objective_then_precision()
    test_wide_candidate_set_includes_stronger_probe_biases()
    print("PASS test_calibrate_viterbi_biases")
