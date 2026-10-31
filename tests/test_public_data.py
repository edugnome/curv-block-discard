"""Direct contracts for scientific public data and protected export paths."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from paper_data import exports
from paper_data.exports import (
    public_crossing_pairs,
    reconcile_crossing_pairs,
    scientific_checkpoint_epochs,
)
from paper_data.manifests import _validate_granularity_restore, _validate_pilot
from paper_data.residual_ablation import (
    _validate_loading_preflight,
    _validate_scientific_context,
    export_ablation,
)


def pilot_bundle() -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {
        name: {"loss_d0": 1.0, "loss": 1.0, "actual_gain": 0.0}
        for name in (
            "original",
            "gradient_refresh",
            "curvature_refresh",
            "both_refresh",
            *(f"scale_{index}" for index in range(7)),
        )
    }
    candidates["original"]["original_gates"] = {
        "ratio": True,
        "model": True,
        "loss": False,
        "rejected_by": "loss",
        "accepted": False,
    }
    row = {
        "native": {
            "candidates": candidates,
            "scale_control": [
                {"alpha": alpha, "candidate": f"scale_{index}", "gain": 0.0}
                for index, alpha in enumerate(
                    (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
                )
            ],
            "best_scale_gain": 0.0,
            "comparison": {"predicted_endpoint_slope": 0.25, "endpoint_slope": -0.5},
        },
        "interventions": {
            name: {"loss_d0": 1.0, "loss": 1.0, "actual_gain": 0.0}
            for name in ("relu", "pool")
        },
    }
    return {
        "schema": 1,
        "format": "curv-scientific-pilot-v1",
        "thresholds": {
            "schema": 1,
            "tolerances": {"gain_floor": 0.0001},
            "gain_floor": 0.0001,
            "repeatability_error": 0.0,
            "repeats": 3,
        },
        "rows": [
            {**copy.deepcopy(row), "seed": seed, "step": step}
            for seed in (42, 43, 44)
            for step in (20, 60, 200)
        ],
    }


def test_public_pilot_validates_cohort_and_gain():
    bundle = pilot_bundle()
    _validate_pilot(bundle)
    missing = copy.deepcopy(bundle)
    missing["rows"].pop()
    with pytest.raises(ValueError):
        _validate_pilot(missing)
    inconsistent = copy.deepcopy(bundle)
    inconsistent["rows"][0]["native"]["candidates"]["original"]["actual_gain"] = 0.125
    with pytest.raises(ValueError):
        _validate_pilot(inconsistent)


def test_missing_author_receipt_is_not_public_format():
    bundle = pilot_bundle()
    del bundle["format"]
    bundle["thresholds"]["calibration_capture_sha256"] = "a" * 64
    with pytest.raises((KeyError, ValueError)):
        _validate_pilot(bundle)
    bundle["format"] = "unknown-format"
    with pytest.raises((KeyError, ValueError)):
        _validate_pilot(bundle)


def pair_rows():
    for seed in (42, 43, 44):
        for model in ("resnet18", "plain_resnet18"):
            for batch in ("B0", "B1"):
                cell = f"s{seed}-{model}-{batch}"
                for index in range(30):
                    pair = {"schema": 1, "cell": cell, "replicate": index}
                    for partition in ("stage", "layer"):
                        count = (
                            6
                            if partition == "stage"
                            else (22 if model == "resnet18" else 19)
                        )
                        pair[partition] = {
                            "schema": 1,
                            "cell": cell,
                            "replicate": index,
                            "partition": partition,
                            "n_probes": 100,
                            "groups": [
                                {
                                    "name": f"g{i}",
                                    "size": i + 1,
                                    "parameters": [f"p{i}"],
                                }
                                for i in range(count)
                            ],
                            "diag_sq": [0.12345678901234566] * count,
                            "full_sq": 2.3,
                            "diag_frob_returned": {
                                f"g{i}": math.sqrt(0.12345678901234566)
                                for i in range(count)
                            },
                            "phi_returned": 1
                            - math.fsum([0.12345678901234566] * count) / 2.3,
                            "stream": {
                                "cell": cell,
                                "replicate": index,
                                "partition": partition,
                                "seed": index,
                            },
                        }
                    yield {
                        "format": "curv-scientific-crossing-pair-v1",
                        "bundle": "exp8_crossing_check",
                        "cell": cell,
                        "seed": seed,
                        "model": model,
                        "batch": batch,
                        "index": index,
                        "observation": pair,
                    }


def write_pairs(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_public_pairs_retain_identity_and_precision(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = list(pair_rows())
    write_pairs(path, rows)
    read = list(public_crossing_pairs(path))
    assert [(r["cell"], r["index"]) for r in read] == [
        (r["cell"], r["index"]) for r in rows
    ]
    assert read[0]["observation"]["stage"]["diag_sq"][0] == 0.12345678901234566
    assert read[0]["observation"]["layer"]["stream"]["seed"] == 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "partition", "nonfinite"])
def test_public_pairs_reject_invalid_cohort(tmp_path, defect):
    rows = list(pair_rows())
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif defect == "partition":
        rows[0]["observation"]["stage"]["partition"] = "layer"
    else:
        rows[0]["observation"]["stage"]["full_sq"] = float("nan")
    path = tmp_path / "pairs.jsonl"
    write_pairs(path, rows)
    with pytest.raises(ValueError):
        list(public_crossing_pairs(path))


def test_residual_export_rejects_protected_root_overlap(tmp_path):
    runs = {
        (s, t): tmp_path / f"run-{s}-{t}" for s in (42, 43, 44) for t in (20, 60, 200)
    }
    protected = tmp_path / "protected"
    for output in (protected, protected / "result", tmp_path):
        with pytest.raises(ValueError):
            export_ablation(
                tmp_path / "registration",
                tmp_path / "thresholds",
                runs,
                output,
                protected_root=protected,
            )


def test_residual_export_checks_input_overlap_without_optional_root(tmp_path):
    runs = {
        (s, t): tmp_path / f"run-{s}-{t}" for s in (42, 43, 44) for t in (20, 60, 200)
    }
    with pytest.raises(ValueError):
        export_ablation(
            tmp_path / "registration",
            tmp_path / "thresholds",
            runs,
            runs[42, 20] / "output",
        )


def test_scientific_residual_protocol_rejects_inconsistent_threshold():
    tolerances = {
        "grad_rtol": 1e-4,
        "hvp_rtol": 1e-3,
        "atol": 1e-6,
        "gain_floor": 1e-4,
        "correction_relative_floor": 1e-3,
    }
    registration = {
        "measurement_schema": 2,
        "historical_probe_replay": False,
        "d0_recomputed": False,
        "partition": ["stem", "layer1", "layer2", "layer3", "layer4", "head"],
        "seeds": [42, 43, 44],
        "steps": [20, 60, 200],
        "drawsets": {"primary": [100003, 200003], "sensitivity": [300003, 400003]},
        "alpha_grid": [0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0],
        "lambda": 1.0,
        "coupling_probes": 10,
        "rank": 4,
        "power_iterations": 8,
        "tau": 0.1,
        "max_pairs": 3,
        "norm_ratio": 2.0,
        "tolerances": tolerances,
        "precision": "saved_runtime_float32_no_autocast_change",
        "cohorts": "smoke_and_final_separate;final_remeasures_all_nine_states",
        "capture_roster": [
            {"seed": s, "step": t, "status": "bytes_checked_not_loaded"}
            for s in (42, 43, 44)
            for t in (20, 60, 200)
        ],
    }
    fixed = {
        "measurement_schema": 2,
        "capture_seed": 42,
        "capture_step": 60,
        "tolerances": tolerances,
        "role": "technical_smoke_only_not_final_data",
        "same_candidate_repeatability_spread": 0.00002,
        "gain_floor": 0.0002,
    }
    _validate_scientific_context(registration, fixed)
    fixed["gain_floor"] = 0.0001
    with pytest.raises(ValueError):
        _validate_scientific_context(registration, fixed)


def test_public_smoke_loading_preflight_requires_roster_and_operations():
    roster = [
        {"seed": seed, "step": step, "status": "bytes_checked_not_loaded"}
        for seed in (42, 43, 44)
        for step in (20, 60, 200)
    ]
    preflight: dict[str, Any] = {
        "measurement_schema": 2,
        "status": "complete",
        "capture_roster": roster,
        "expected": 9,
        "attempted": 9,
        "valid": 9,
        "technical_failure": 0,
        "not_attempted": 0,
        "device": "cpu",
        "hvp_run": False,
        "gradient_run": False,
        "training_run": False,
        "pretrained_requested": False,
        "entries": [
            {
                "seed": row["seed"],
                "step": row["step"],
                "status": "compatible_loaded",
                "state_preserved": True,
            }
            for row in roster
        ],
    }
    _validate_loading_preflight(preflight, roster)
    for change in ("missing", "identity", "preservation", "flag", "count"):
        bad = copy.deepcopy(preflight)
        if change == "missing":
            bad["entries"].pop()
        elif change == "identity":
            bad["entries"][0]["step"] = 60
        elif change == "preservation":
            bad["entries"][0]["state_preserved"] = False
        elif change == "flag":
            bad["hvp_run"] = True
        else:
            bad["valid"] = 8
        with pytest.raises(ValueError):
            _validate_loading_preflight(bad, roster)


def test_public_granularity_rejects_mixed_author_restore_field():
    public: dict[str, Any] = {
        "meta": {
            "format": "curv-scientific-granularity-v1",
            "config": {
                "restored_from": [
                    {"experiment": "exp8_a1_long", "seed": seed, "epoch": 300}
                    for seed in (42, 43, 44)
                ]
            },
        }
    }
    _validate_granularity_restore(public)
    public["meta"]["config"]["restore_checkpoints_dir"] = ["incorrect"]
    with pytest.raises(ValueError):
        _validate_granularity_restore(public)


def test_checkpoint_projection_retains_epoch_identity_and_observations():
    original: dict[str, Any] = {
        "checkpoints": {
            "0": "checkpoints/model_ep0.pt",
            "100": "checkpoints/model_ep100.pt",
        },
        "phi": 0.12345678901234566,
        "status": "complete",
    }
    projected = scientific_checkpoint_epochs(original)
    assert projected == {
        "checkpoint_epochs": ["0", "100"],
        "phi": original["phi"],
        "status": "complete",
    }
    assert original["checkpoints"]["0"] == "checkpoints/model_ep0.pt"


def test_paired_pool_rejects_same_factor_rescaling():
    import numpy as np

    rows = list(pair_rows())[:30]
    cell_id = rows[0]["cell"]
    pool = {}
    for part in ("stage", "layer"):
        components = [r["observation"][part] for r in rows]
        pool[part] = {
            "diag_sq": np.asarray([c["diag_sq"] for c in components], dtype=np.float64)
            .mean(axis=0)
            .tolist(),
            "full_sq": float(
                np.asarray([c["full_sq"] for c in components], dtype=np.float64).mean()
            ),
        }
    bundle = {
        "source_analysis": {
            "analysis": {"cells": [{"id": cell_id, "pooled_components": pool}]}
        }
    }
    reconcile_crossing_pairs(rows, bundle)
    changed = copy.deepcopy(rows)
    component = changed[0]["observation"]["stage"]
    component["diag_sq"] = [2 * value for value in component["diag_sq"]]
    component["full_sq"] *= 2
    component["diag_frob_returned"] = {
        key: value * math.sqrt(2)
        for key, value in component["diag_frob_returned"].items()
    }
    assert component["phi_returned"] == rows[0]["observation"]["stage"]["phi_returned"]
    with pytest.raises(ValueError):
        reconcile_crossing_pairs(changed, bundle)


@pytest.mark.parametrize("public", [True, False])
def test_export_selects_pair_source_by_bundle_identity(tmp_path, monkeypatch, public):
    class Selected(Exception):
        pass

    marker = {"format": "curv-scientific-crossing-v1"} if public else {}
    monkeypatch.setattr(
        exports, "preflight_b", lambda _: {"exp8_crossing_check": marker}
    )

    def selected_public(_):
        if public:
            raise Selected
        raise AssertionError("wrong source selected")

    def selected_archive(_):
        if not public:
            raise Selected
        raise AssertionError("wrong source selected")

    monkeypatch.setattr(exports, "public_crossing_pairs", selected_public)
    monkeypatch.setattr(exports, "crossing_pairs", selected_archive)
    with pytest.raises(Selected):
        exports.export_tables(tmp_path, tmp_path / "output")
