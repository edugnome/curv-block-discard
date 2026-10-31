"""Corrected schema-2 scalar reporting, without loading captures or geometry."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

SEEDS = (42, 43, 44)
STEPS = (20, 60, 200)
ARMS = ("full", "off", "rest")
GROUPS = ("stem", "layer1", "layer2", "layer3", "layer4", "head")
GRID = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
DRAWS = {"primary": (100003, 200003), "sensitivity": (300003, 400003)}
PAIRS = ("full__off", "full__rest", "off__rest")
TOLERANCES = {
    "grad_rtol": 1e-4,
    "hvp_rtol": 1e-3,
    "atol": 1e-6,
    "gain_floor": 1e-4,
    "correction_relative_floor": 1e-3,
}
ENVIRONMENT = {
    "python": "3.12.13",
    "python_executable": "/synthetic/env/bin/python",
    "sys_prefix": "/synthetic/env",
    "torch": "synthetic",
    "torch_cuda_build": None,
    "numpy": "synthetic",
    "torchvision": "synthetic",
}


def _bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _ref(path, value):
    return {"path": str(path), "sha256": _hash(_bytes(value))}


def _labels(arm):
    return {
        **{f"alpha_{i}": a for i, a in enumerate(GRID)},
        **{f"matched_{p}": None for p in PAIRS if arm in p.split("__")},
    }


def _counts(rows, status="status"):
    values = [r[status] for r in rows.values()]
    interpretation = (
        "applied_interpretation" if status == "applied_status" else "interpretation"
    )
    return {
        "expected": len(values),
        "attempted": sum(v != "not_attempted" for v in values),
        "valid": values.count("finite"),
        "technical_failure": values.count("technical_failure"),
        "not_attempted": values.count("not_attempted"),
        "unresolved": sum(
            r[status] == "finite" and r.get(interpretation) == "unresolved_preference"
            for r in rows.values()
        ),
    }


def _recount(result):
    for draw in result["drawsets"].values():
        measurements = draw["measurements"]
        measurements["unresolved_is_subset_of_valid"] = True
        for field, key in (
            ("status", "counts"),
            ("raw_status", "raw_counts"),
            ("applied_status", "applied_counts"),
        ):
            measurements[key] = {
                arm: _counts(rows, field) for arm, rows in measurements["arms"].items()
            }
    result["drawset_counts"] = {
        "expected": 2,
        "attempted": sum(
            d["status"] != "not_attempted" for d in result["drawsets"].values()
        ),
        "valid": sum(d["status"] == "complete" for d in result["drawsets"].values()),
        "technical_failure": sum(
            d["status"] == "technical_failure" for d in result["drawsets"].values()
        ),
        "not_attempted": sum(
            d["status"] == "not_attempted" for d in result["drawsets"].values()
        ),
    }
    for key in ("counts", "raw_counts", "applied_counts"):
        result[key] = {
            arm: {
                field: sum(
                    d["measurements"][key][arm][field]
                    for d in result["drawsets"].values()
                )
                for field in (
                    "expected",
                    "attempted",
                    "valid",
                    "technical_failure",
                    "not_attempted",
                    "unresolved",
                )
            }
            for arm in ARMS
        }


def _candidate(arm, alpha, label):
    norms = {"full": 0.3, "off": 0.2, "rest": 0.4}
    scale = (
        alpha
        if alpha is not None
        else min(norms[a] for a in label[8:].split("__")) / norms[arm]
    )
    correction = norms[arm] * scale
    loss = 2.0 + {"full": -0.02, "off": 0.01, "rest": -0.03}[arm] * scale
    delta = loss - 2.0
    model = -0.1 * scale
    accepted = loss <= 2.0
    gain = -delta if accepted else 0.0
    return {
        "status": "finite",
        "raw_status": "finite",
        "applied_status": "finite",
        "alpha": alpha,
        "kind": "scale" if alpha is not None else "matched_down_only",
        "loss_theta_repeats": [3.0] * 3,
        "loss_d0_repeats": [2.0] * 3,
        "loss_theta": 3.0,
        "loss_d0": 2.0,
        "model_baseline": 0.0,
        "loss_repeats": [loss] * 3,
        "loss": loss,
        "raw_delta_loss": delta,
        "actual_gain": -delta,
        "strict_sign": "improvement"
        if delta < 0
        else "harm"
        if delta > 0
        else "exact_tie",
        "interpretation": "unresolved_preference"
        if abs(delta) <= 1e-4
        else "improvement"
        if delta < 0
        else "harm",
        "step_norm": 1.0 + correction,
        "correction_norm": correction,
        "near_zero_correction": correction < 1e-3,
        "norm_match_scale": scale if alpha is None else None,
        "model": {
            "undamped": model - 0.5 * (1 + correction) ** 2,
            "damping": 0.5 * (1 + correction) ** 2,
            "damped": model,
            "model_delta": model,
        },
        "gates": {
            "ratio": True,
            "model": True,
            "loss": accepted,
            "accepted": accepted,
            "rejected_by": None if accepted else "loss",
        },
        "applied_direction": "raw" if accepted else "d0",
        "applied_gain": gain,
        "applied_strict_sign": "improvement"
        if gain > 0
        else "harm"
        if gain < 0
        else "exact_tie",
        "applied_interpretation": "unresolved_preference"
        if abs(gain) <= 1e-4
        else "improvement"
        if gain > 0
        else "harm",
    }


def _result(seed, step, phase, registration, reference, thresholds):
    capture = next(
        r
        for r in registration["capture_roster"]
        if (r["seed"], r["step"]) == (seed, step)
    )
    result = {
        "measurement_schema": 2,
        "phase": phase,
        "status": "complete",
        "requested_capture": capture["capture_path"],
        "seed": seed,
        "step": step,
        "capture_sha256": capture["capture_sha256"],
        "historical_source_sha256": capture["historical_source_sha256"],
        "current_source_sha256": registration["current_source_sha256"],
        "registration": {**reference, "phase": phase},
        "historical_probe_replay": False,
        "d0_recomputed": False,
        "device": "cpu",
        "dtype": "torch.float32",
        "runtime_flags": {},
        "environment": {},
        "environment_identity": dict(ENVIRONMENT),
        "capture_status": "complete",
        "technical_validation": "measured_requires_external_smoke_or_result_review",
        "state_preservation_check": {"passed": True},
        "thresholds": thresholds,
        "drawsets": {},
        "count_semantics": "unresolved is a subset of valid, not an additional outcome; steps/draws are not independent training seeds",
    }
    if phase == "smoke":
        result["loading_preflight"] = {
            "measurement_schema": 2,
            "status": "complete",
            "registration_sha256": reference["sha256"],
            "current_source_sha256": registration["current_source_sha256"],
            "environment_identity": dict(ENVIRONMENT),
            "capture_roster": registration["capture_roster"],
            "expected": 9,
            "attempted": 9,
            "valid": 9,
            "technical_failure": 0,
            "not_attempted": 0,
            "entries": [
                {**r, "status": "compatible_loaded", "state_preserved": True}
                for r in registration["capture_roster"]
            ],
            "device": "cpu",
            "hvp_run": False,
            "gradient_run": False,
            "training_run": False,
            "pretrained_requested": False,
        }
    else:
        result["loading_preflight"] = {
            "status": "bound_to_frozen_smoke",
            "smoke_results": registration["smoke_calibration"]["smoke_results"],
        }
    for name, offsets in DRAWS.items():
        arms = {
            arm: {label: _candidate(arm, a, label) for label, a in _labels(arm).items()}
            for arm in ARMS
        }
        check = {
            "absolute_error": 0.0,
            "component_norm_sum": 1.0,
            "scaled_error": 0.0,
            "threshold": 0.001,
            "passed": True,
        }
        geometry_bytes = (
            f"synthetic geometry bytes {seed}/{step}/{phase}/{name}".encode()
        )
        result["drawsets"][name] = {
            "drawset": name,
            "status": "complete",
            "seed": seed,
            "step": step,
            "seeds": {"coupling": seed + offsets[0], "basis": seed + offsets[1]},
            "replay_checks": {key: {"passed": True} for key in ("loss", "gradient")},
            "split_checks": {
                **{k: dict(check) for k in ("masked_sum", "off_receivers", "residual")},
                "passed": True,
            },
            "source_checks": [{"group": group, "action_norm": 1.0} for group in GROUPS],
            "hvp_residual_resolution": 0.001,
            "residual_norms": {a: 1.0 for a in ARMS},
            "projected_residual_norms": {a: 0.5 for a in ARMS},
            "coverage": {
                "numerator": 1.0,
                "denominator": 4.0,
                "resolution_norm": 0.001,
                "eta_action": 0.25,
                "status": "resolved",
                "orthonormality_error": 0.0,
            },
            "selected_pairs": [["stem", "head"]],
            "coupling": [{"pair": ["stem", "head"], "value": 0.2}],
            "correction_additivity": {
                **check,
                "native_full_check": {
                    **check,
                    "comparison_space": "direction",
                    "bitwise_equal": True,
                },
                "recovered_correction_check": {
                    **check,
                    "passed": False,
                    "diagnostic_only": True,
                },
            },
            "correction_norms": {"full": 0.3, "off": 0.2, "rest": 0.4},
            "off_rest_dot": -0.055,
            "independent_flat_ad": {
                key: {"passed": True} for key in ("loss", "gradient", "hvp_d0")
            },
            "factored_geometry": {
                "measurement_schema": 2,
                "capture_sha256": result["capture_sha256"],
                "current_source_sha256": result["current_source_sha256"],
                "registration_sha256": reference["sha256"],
                "seed": seed,
                "step": step,
                "drawset": name,
                "seeds": {"coupling": seed + offsets[0], "basis": seed + offsets[1]},
                "filename": f"factored_{name}.pt",
                "sha256": _hash(geometry_bytes),
            },
            "measurements": {
                "arms": arms,
                "baselines": {
                    "theta": [3.0] * 3,
                    "d0": [2.0] * 3,
                    "model_d0": 0.0,
                    "norm_d0": 1.0,
                    "status": "finite",
                    "numerical_failures": [],
                },
                "norm_matched_pairs": list(PAIRS),
                "loss_interaction": {
                    f"alpha_{i}": {
                        "status": "finite",
                        "delta_loss_full_minus_components": arms["full"][f"alpha_{i}"][
                            "raw_delta_loss"
                        ]
                        - arms["off"][f"alpha_{i}"]["raw_delta_loss"]
                        - arms["rest"][f"alpha_{i}"]["raw_delta_loss"],
                    }
                    for i in range(7)
                },
            },
        }
    _recount(result)
    return result


def _status(result):
    return {
        "schema": 1,
        "kind": f"residual_ablation_v2_{result['phase']}",
        "status": result["status"],
        "measurement_identity": {
            key: result[key]
            for key in (
                "measurement_schema",
                "phase",
                "seed",
                "step",
                "requested_capture",
                "capture_sha256",
                "current_source_sha256",
                "registration",
            )
        },
        **(
            {"error": {"message": result.get("technical_error", "synthetic failure")}}
            if result["status"] != "complete"
            else {}
        ),
    }


def _write_run(path, result):
    path.mkdir(parents=True)
    (path / "results.json").write_bytes(_bytes(result))
    (path / "status.json").write_bytes(_bytes(_status(result)))
    for name, draw in result["drawsets"].items():
        if "factored_geometry" in draw:
            (path / f"factored_{name}.pt").write_bytes(
                f"synthetic geometry bytes {result['seed']}/{result['step']}/{result['phase']}/{name}".encode()
            )


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    smoke: dict[str, Any] = {
        "measurement_schema": 2,
        "registration_phase": "smoke",
        "current_source_sha256": "a" * 64,
        "installed_source_root": "/synthetic/src",
        "environment_identity": dict(ENVIRONMENT),
        "capture_roster": [
            {
                "seed": seed,
                "step": step,
                "status": "bytes_checked_not_loaded",
                "capture_sha256": _hash(f"capture {seed}/{step}".encode()),
                "historical_source_sha256": "historical-source-identity",
                "capture_path": str(
                    root / "captures" / f"seed_{seed}" / f"step_{step:06d}"
                ),
            }
            for seed in SEEDS
            for step in STEPS
        ],
        "historical_probe_replay": False,
        "d0_recomputed": False,
        "partition": list(GROUPS),
        "seeds": list(SEEDS),
        "steps": list(STEPS),
        "drawsets": {k: list(v) for k, v in DRAWS.items()},
        "alpha_grid": list(GRID),
        "lambda": 1.0,
        "coupling_probes": 10,
        "rank": 4,
        "power_iterations": 8,
        "tau": 0.1,
        "max_pairs": 3,
        "norm_ratio": 2.0,
        "tolerances": dict(TOLERANCES),
        "precision": "saved_runtime_float32_no_autocast_change",
        "cohorts": "smoke_and_final_separate;final_remeasures_all_nine_states",
    }
    smoke_path = root / "smoke-registration.json"
    smoke_path.write_bytes(_bytes(smoke))
    smoke_ref = _ref(smoke_path, smoke)
    thresholds = {
        "measurement_schema": 2,
        "source_sha256": "a" * 64,
        "capture_seed": 42,
        "capture_step": 60,
        "capture_sha256": smoke["capture_roster"][1]["capture_sha256"],
        "tolerances": dict(TOLERANCES),
        "gain_floor": 1e-4,
        "same_candidate_repeatability_spread": 0.0,
        "role": "technical_smoke_only_not_final_data",
        "smoke_registration_sha256": smoke_ref["sha256"],
        "environment_identity": dict(ENVIRONMENT),
    }
    smoke_result = _result(42, 60, "smoke", smoke, smoke_ref, thresholds)
    _write_run(root / "smoke", smoke_result)
    (root / "smoke" / "thresholds.json").write_bytes(_bytes(thresholds))
    binding = {
        "smoke_registration": smoke_ref,
        "thresholds": _ref(root / "smoke/thresholds.json", thresholds),
        "smoke_results": _ref(root / "smoke/results.json", smoke_result),
        "smoke_status": _ref(root / "smoke/status.json", _status(smoke_result)),
    }
    final = {**smoke, "registration_phase": "final", "smoke_calibration": binding}
    final_path = root / "final-registration.json"
    final_path.write_bytes(_bytes(final))
    states, inputs = [], []
    final_ref = _ref(final_path, final)
    for seed in SEEDS:
        for step in STEPS:
            result = _result(seed, step, "final", final, final_ref, thresholds)
            path = root / f"s{seed}-t{step}"
            _write_run(path, result)
            geometry = {
                name: draw["factored_geometry"]["sha256"]
                for name, draw in result["drawsets"].items()
            }
            states.append(
                {
                    "seed": seed,
                    "step": step,
                    "status": "complete",
                    "reason": None,
                    "result": result,
                    "status_record": _status(result),
                    "geometry_sha256": geometry,
                }
            )
            inputs.append(
                {
                    "seed": seed,
                    "step": step,
                    "path": str(path),
                    "sha256": {
                        "results.json": _hash(_bytes(result)),
                        "status.json": _hash(_bytes(_status(result))),
                    },
                    "geometry_sha256": geometry,
                }
            )
    return {
        "schema": 1,
        "bundle": "residual_ablation_v2",
        "phase": "final",
        "registration": final,
        "thresholds": thresholds,
        "calibration": {
            "smoke_registration": smoke,
            "smoke_results": smoke_result,
            "smoke_status": _status(smoke_result),
            "geometry_sha256": {
                name: d["factored_geometry"]["sha256"]
                for name, d in smoke_result["drawsets"].items()
            },
        },
        "states": states,
        "provenance": {
            "registration": final_ref,
            "calibration": binding,
            "inputs": inputs,
        },
    }


def _seal_state_artifacts(bundle):
    """Refresh synthetic file receipts so mutations test scalar contracts, not hashes."""
    indices = {(r["seed"], r["step"]): r for r in bundle["provenance"]["inputs"]}
    for state in bundle["states"]:
        index = indices[state["seed"], state["step"]]
        index["sha256"] = {
            "results.json": _hash(_bytes(state["result"])),
            "status.json": _hash(_bytes(state["status_record"])),
        }
        index["geometry_sha256"] = dict(state["geometry_sha256"])


def test_complete_raw_applied_counts_and_seed_hierarchy(bundle):
    from paper_data.residual_ablation import ablation_summary, candidate_rows

    original = copy.deepcopy(bundle)
    rows = list(candidate_rows(bundle))
    summary = ablation_summary(bundle)
    assert bundle == original
    assert (
        len(rows)
        == len(
            {
                (r["seed"], r["step"], r["drawset"], r["arm"], r["candidate"])
                for r in rows
            }
        )
        == 486
    )
    assert (
        summary["counts"]["valid"]
        == summary["raw_counts"]["valid"]
        == summary["applied_counts"]["valid"]
        == 486
    )
    assert summary["raw_counts"]["unresolved"] == 54
    assert summary["applied_counts"]["unresolved"] == 198
    assert summary["per_seed"]["42"]["primary"]["off"]["raw"]["harm"] == 3
    assert summary["drawsets"][0]["matched"]["full__off"]["left_minus_right"] < 0


@pytest.mark.parametrize(
    "case",
    [
        "smoke",
        "source",
        "registration_hash",
        "requested_capture",
        "environment",
        "threshold",
        "calibration",
        "missing_state",
        "duplicate_state",
        "draw_roster",
        "candidate_roster",
        "raw_count",
        "applied_count",
        "gain",
        "gate",
        "geometry_identity",
        "geometry_receipt",
        "nan",
        "action",
    ],
)
def test_rejects_inconsistent_corrected_envelope(bundle, case):
    from paper_data.residual_ablation import ablation_summary

    result = bundle["states"][0]["result"]
    draw = result["drawsets"]["primary"]
    row = draw["measurements"]["arms"]["off"]["alpha_6"]
    if case == "smoke":
        result["phase"] = "smoke"
    elif case == "source":
        result["current_source_sha256"] = "c" * 64
    elif case == "registration_hash":
        result["registration"]["sha256"] = "c" * 64
    elif case == "requested_capture":
        result["requested_capture"] = "/wrong/capture"
    elif case == "environment":
        result["environment_identity"]["sys_prefix"] = "/wrong/env"
    elif case == "threshold":
        result["thresholds"] = {**result["thresholds"], "gain_floor": 1.0}
    elif case == "calibration":
        bundle["calibration"]["smoke_results"]["loading_preflight"]["valid"] = 8
    elif case == "missing_state":
        bundle["states"].pop()
    elif case == "duplicate_state":
        bundle["states"][-1] = copy.deepcopy(bundle["states"][0])
    elif case == "draw_roster":
        result["drawsets"].pop("sensitivity")
    elif case == "candidate_roster":
        draw["measurements"]["arms"]["off"].pop("alpha_6")
    elif case == "raw_count":
        result["raw_counts"]["off"]["valid"] -= 1
    elif case == "applied_count":
        draw["measurements"]["applied_counts"]["off"]["unresolved"] -= 1
    elif case == "gain":
        row["actual_gain"] = row["raw_delta_loss"]
    elif case == "gate":
        row["gates"]["accepted"] = True
    elif case == "geometry_identity":
        draw["factored_geometry"]["registration_sha256"] = "c" * 64
    elif case == "geometry_receipt":
        bundle["states"][0]["geometry_sha256"]["primary"] = "c" * 64
    elif case == "nan":
        row["raw_delta_loss"] = float("nan")
    elif case == "action":
        draw["coverage"]["eta_action"] = 1.0
    if case != "nan":
        _seal_state_artifacts(bundle)
    with pytest.raises(ValueError):
        ablation_summary(bundle)


def _fail_state(state, reason):
    state.update(status="technical_failure", reason=reason)
    state["result"].update(status="technical_failure", technical_error=reason)
    state["result"].pop("thresholds", None)
    state["status_record"] = _status(state["result"])


def test_not_attempted_draw_has_no_geometry_obligation(bundle):
    from paper_data.residual_ablation import ablation_summary

    state = bundle["states"][0]
    _fail_state(state, "synthetic early failure")
    state["result"].update(
        capture_status="technical_failure", dtype=None, runtime_flags=None
    )
    for draw in state["result"]["drawsets"].values():
        identity = {key: draw[key] for key in ("drawset", "seed", "step", "seeds")}
        draw.clear()
        draw.update(
            **identity,
            status="not_attempted",
            not_run_reason="blocked_by_prior_technical_failure",
            measurements={
                "arms": {
                    arm: {
                        label: {
                            "status": "not_attempted",
                            "raw_status": "not_attempted",
                            "applied_status": "not_attempted",
                            "alpha": a,
                            "kind": "scale" if a is not None else "matched_down_only",
                        }
                        for label, a in _labels(arm).items()
                    }
                    for arm in ARMS
                }
            },
        )
    state["geometry_sha256"].clear()
    _recount(state["result"])
    _seal_state_artifacts(bundle)
    summary = ablation_summary(bundle)
    assert summary["raw_counts"]["not_attempted"] == 54
    assert summary["drawsets"][0]["coverage"] is None
    assert summary["per_seed"]["42"]["primary"]["off"]["raw"]["not_attempted"] == 1


def test_overflow_model_retains_finite_raw_loss_but_not_applied_zero(bundle):
    from paper_data.residual_ablation import ablation_summary

    state = bundle["states"][0]
    _fail_state(state, "synthetic model overflow")
    draw = state["result"]["drawsets"]["primary"]
    draw.update(status="technical_failure", technical_error="model overflow")
    row = draw["measurements"]["arms"]["off"]["alpha_6"]
    row.update(
        status="technical_failure",
        applied_status="technical_failure",
        model={"undamped": None, "damped": None, "damping": 0.72, "model_delta": None},
        applied_gain=None,
        applied_direction=None,
        gates_status="unavailable_technical_failure",
        numerical_failures=["model.damped: nonfinite inf"],
        technical_error="model overflow",
    )
    row.pop("gates")
    _recount(state["result"])
    _seal_state_artifacts(bundle)
    summary = ablation_summary(bundle)
    assert summary["raw_counts"]["valid"] == 486
    assert summary["counts"]["technical_failure"] == 1
    assert summary["applied_counts"]["technical_failure"] == 1
    primary = summary["drawsets"][0]["primary"]["off"]
    assert primary["raw_delta_loss"] > 0 and primary["applied_gain"] is None
    assert summary["per_seed"]["42"]["primary"]["off"]["raw"]["harm"] == 3


def test_undefined_action_and_diagnostic_only_recovery_do_not_become_failures(bundle):
    from paper_data.residual_ablation import ablation_summary

    coverage = bundle["states"][0]["result"]["drawsets"]["primary"]["coverage"]
    coverage.update(
        status="zero_denominator",
        numerator=None,
        denominator=0.0,
        eta_action=None,
        orthonormality_error=None,
    )
    _seal_state_artifacts(bundle)
    summary = ablation_summary(bundle)
    assert summary["drawsets"][0]["coverage"]["eta_action"] is None


def test_exact_artifact_bytes_and_output_guards(bundle, tmp_path):
    from paper_data.residual_ablation import export_ablation

    registration = Path(bundle["provenance"]["registration"]["path"])
    thresholds = Path(bundle["registration"]["smoke_calibration"]["thresholds"]["path"])
    runs = {
        (r["seed"], r["step"]): Path(r["path"]) for r in bundle["provenance"]["inputs"]
    }
    output = tmp_path / "export"
    assert export_ablation(registration, thresholds, runs, output) == {
        "states": 9,
        "drawsets": 18,
        "candidates": 486,
    }
    geometry = next(iter(runs.values())) / "factored_primary.pt"
    geometry.write_bytes(b"changed bytes, never unpickle")
    with pytest.raises(ValueError):
        export_ablation(registration, thresholds, runs, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_bound_smoke_artifact_mutation_is_rejected_before_outputs(bundle, tmp_path):
    from paper_data.residual_ablation import export_ablation

    registration = Path(bundle["provenance"]["registration"]["path"])
    thresholds = Path(bundle["registration"]["smoke_calibration"]["thresholds"]["path"])
    thresholds.write_bytes(thresholds.read_bytes() + b" ")
    runs = {
        (r["seed"], r["step"]): Path(r["path"]) for r in bundle["provenance"]["inputs"]
    }
    with pytest.raises(ValueError):
        export_ablation(registration, thresholds, runs, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_tables_macros_and_standalone_preflight(bundle, tmp_path):
    from paper_data.config import DataPaths
    from paper_data.main import generate_all

    path = tmp_path / "bundle.json"
    bundle["states"].pop()
    path.write_bytes(_bytes(bundle))
    with pytest.raises(ValueError):
        generate_all(
            DataPaths(tmp_path, {"residual_ablation_v2": path}),
            tmp_path / "never",
            only=["tab:residual-ablation"],
        )
    assert not (tmp_path / "never").exists()


def _scalar_geometry_bundle():
    return {
        "rows": [
            {
                "seed": seed,
                "step": step,
                "drawset": draw,
                "native_screen": {
                    "status": "resolved",
                    "max_entry_error": 1e-5,
                    "eta_action": 0.4,
                },
                "represented_factor": {
                    "status": "available",
                    "raw_columns": 30,
                    "retained_columns": 28,
                    "gram_spectral_error": 0.0002,
                    "gram_eigen_min": 0.9999,
                    "gram_eigen_max": 1.0002,
                    "gram_condition": 1.0003,
                    "map_relative_difference": 0.00025,
                },
            }
            for seed in SEEDS
            for step in STEPS
            for draw in DRAWS
        ]
    }


def test_relocated_reporting_uses_scalar_bundle_without_original_files(
    bundle, tmp_path, monkeypatch
):
    from paper_data import residual_ablation
    from paper_data.config import DataPaths
    from paper_data.main import generate_all

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    path = relocated / "results.json"
    path.write_bytes(_bytes(bundle))
    geometry_path = relocated / "geometry.json"
    geometry_path.write_bytes(_bytes(_scalar_geometry_bundle()))
    original_open = Path.open

    def scalar_only_open(self, *args, **kwargs):
        if not self.resolve().is_relative_to(relocated):
            pytest.fail(f"reporting opened an original artifact: {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", scalar_only_open)
    monkeypatch.setattr(
        residual_ablation,
        "_file_hash",
        lambda p: pytest.fail("reporting hashed geometry"),
    )
    output = relocated / "generated"
    generate_all(
        DataPaths(
            relocated,
            {"residual_ablation_v2": path, "residual_basis_geometry": geometry_path},
        ),
        output,
        track="b-ablation",
    )


def test_overflow_baseline_keeps_raw_validity_separate_from_joint_norm_model(bundle):
    from paper_data.residual_ablation import ablation_summary

    state = bundle["states"][0]
    _fail_state(state, "synthetic baseline overflow")
    draw = state["result"]["drawsets"]["primary"]
    draw.update(status="technical_failure", technical_error="baseline overflow")
    baselines = draw["measurements"]["baselines"]
    baselines.update(
        model_d0=None,
        norm_d0=None,
        status="technical_failure",
        numerical_failures=["model_d0: nonfinite inf", "norm_d0: nonfinite inf"],
    )
    for rows in draw["measurements"]["arms"].values():
        for row in rows.values():
            row.update(
                status="technical_failure",
                applied_status="technical_failure",
                model_baseline=None,
                step_norm=None,
                near_zero_correction=None,
                model={
                    "undamped": None,
                    "damped": None,
                    "damping": None,
                    "model_delta": None,
                },
                applied_gain=None,
                applied_direction=None,
                numerical_failures=list(baselines["numerical_failures"]),
                technical_error="baseline overflow",
                gates_status="unavailable_technical_failure",
            )
            row.pop("gates")
    _recount(state["result"])
    _seal_state_artifacts(bundle)
    summary = ablation_summary(bundle)
    assert summary["raw_counts"]["valid"] == 486
    assert (
        summary["counts"]["technical_failure"]
        == summary["applied_counts"]["technical_failure"]
        == 27
    )
    assert (
        summary["drawsets"][0]["matched"]["full__off"]["status"] == "norm_unavailable"
    )


def test_incomplete_raw_grid_is_not_an_oracle_over_the_finite_subset(bundle):
    from paper_data.residual_ablation import ablation_summary

    state = bundle["states"][0]
    _fail_state(state, "synthetic partial loss failure")
    draw = state["result"]["drawsets"]["primary"]
    draw.update(status="technical_failure", technical_error="loss failure")
    row = draw["measurements"]["arms"]["off"]["alpha_6"]
    row.update(
        status="technical_failure",
        raw_status="technical_failure",
        applied_status="technical_failure",
        loss=None,
        loss_repeats=[None] * 3,
        raw_delta_loss=None,
        actual_gain=None,
        applied_gain=None,
        applied_direction=None,
        technical_error="nonfinite loss",
        numerical_failures=["loss[0]: nonfinite nan"],
    )
    row.pop("gates")
    draw["measurements"]["loss_interaction"]["alpha_6"] = {
        "status": "technical_unavailable"
    }
    _recount(state["result"])
    _seal_state_artifacts(bundle)
    summary = ablation_summary(bundle)
    assert (
        summary["drawsets"][0]["scale_min"]["off"]["status"]
        == "unavailable_incomplete_raw_grid"
    )
    assert summary["raw_counts"]["technical_failure"] == 1


def test_measured_alpha_zero_loss_variation_is_retained(bundle):
    import torch

    from experiments.residual_ablation import measure_candidates
    from paper_data.residual_ablation import ablation_summary, candidate_rows

    d0 = torch.full((6,), 0.1)
    zero = torch.zeros_like(d0)
    solved = {arm: {"delta": zero, "direction": d0} for arm in ARMS}
    calls = 0

    def noisy_loss(_direction: torch.Tensor) -> float:
        nonlocal calls
        calls += 1
        if calls <= 3:
            return 3.0
        return 2.0 if calls <= 6 else 2.0 + 1e-7

    measured = measure_candidates(
        d0,
        -d0,
        zero,
        lambda _: zero,
        solved,
        noisy_loss,
        floor=1e-4,
        reverse_order_check=True,
    )
    state = bundle["states"][0]
    draw = state["result"]["drawsets"]["primary"]
    draw["measurements"] = measured
    draw["correction_norms"] = dict.fromkeys(ARMS, 0.0)
    _recount(state["result"])
    _seal_state_artifacts(bundle)

    summary = ablation_summary(bundle)
    rows = list(candidate_rows(bundle))
    row = measured["arms"]["full"]["alpha_0"]
    assert len(rows) == summary["raw_counts"]["valid"] == 486
    assert row["correction_norm"] == 0
    assert row["raw_delta_loss"] == pytest.approx(1e-7)
    assert row["strict_sign"] == "harm"
    assert row["interpretation"] == "unresolved_preference"
    assert row["applied_gain"] == 0.0
    assert row["gates"]["accepted"] is False
    assert row["raw_delta_loss"] in [r["raw_delta_loss"] for r in rows]


@pytest.mark.parametrize("field", ["seed", "step"])
def test_state_identities_require_canonical_integers(bundle, field):
    from paper_data.residual_ablation import validated_ablation

    state = bundle["states"][0]
    state[field] = float(state[field])
    with pytest.raises(ValueError):
        validated_ablation(bundle)


def test_retrospective_contrast_export_and_figure(bundle):
    from paper_data.residual_ablation import (
        contrast_summary,
        residual_contrast_rows,
    )

    rows = list(residual_contrast_rows(bundle))
    scale = [r for r in rows if r["type"] == "scale"]
    matched = [r for r in rows if r["type"] == "matched_full_rest"]
    assert len(scale) == 126 and len(matched) == 18
    assert len({(r["seed"], r["step"], r["drawset"]) for r in matched}) == 18
    assert all("alpha" not in r and "j" not in r for r in matched)
    assert all(r["status"] == "finite" for r in rows)
    summary = contrast_summary(bundle)
    assert summary["primary"]["all_states"]["total"] == 9
    assert summary["primary"]["42"]["total"] == 3
