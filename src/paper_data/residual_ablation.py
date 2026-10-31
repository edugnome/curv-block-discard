"""Scalar reporting and validation for the schema-2 residual ablation.

Author bundles include frozen artifact receipts. Public scientific bundles retain
measurement identities and numerical records without requiring those receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .formatting import sci

_SEEDS = (42, 43, 44)
_STEPS = (20, 60, 200)
_ARMS = ("full", "off", "rest")
_GROUPS = ("stem", "layer1", "layer2", "layer3", "layer4", "head")
_GRID = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
_DRAWS = {"primary": (100003, 200003), "sensitivity": (300003, 400003)}
_PAIRS = ("full__off", "full__rest", "off__rest")
_COUNT_FIELDS = (
    "expected",
    "attempted",
    "valid",
    "technical_failure",
    "not_attempted",
    "unresolved",
)
_STATUS_FIELDS = {
    "counts": "status",
    "raw_counts": "raw_status",
    "applied_counts": "applied_status",
}
_TOLERANCES = {
    "grad_rtol": 1e-4,
    "hvp_rtol": 1e-3,
    "atol": 1e-6,
    "gain_floor": 1e-4,
    "correction_relative_floor": 1e-3,
}


def _require(condition: bool, context: str) -> None:
    if not condition:
        raise ValueError(f"residual ablation: {context}")


def _number(value: Any, context: str, *, nonnegative: bool = False) -> float:
    _require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (not nonnegative or value >= 0),
        f"{context}: finite scalar required",
    )
    return float(value)


def _json_bytes(value: Any) -> bytes:
    # Artifact checksums use strict, sorted JSON serialization.
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "ascii"
    )


def _hash(value: Any) -> None:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "invalid SHA-256",
    )


def _reference(reference: dict[str, Any], value: Any) -> None:
    _require(
        set(reference) == {"path", "sha256"} and Path(reference["path"]).is_absolute(),
        "artifact reference",
    )
    _hash(reference["sha256"])
    _require(
        hashlib.sha256(_json_bytes(value)).hexdigest() == reference["sha256"],
        "recorded JSON checksum",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(arm: str) -> dict[str, float | None]:
    return {
        **{f"alpha_{i}": a for i, a in enumerate(_GRID)},
        **{f"matched_{p}": None for p in _PAIRS if arm in p.split("__")},
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [row[field] for row in rows]
    _require(
        all(v in ("finite", "technical_failure", "not_attempted") for v in values),
        f"{field}: outcome status",
    )
    interpretation = (
        "applied_interpretation" if field == "applied_status" else "interpretation"
    )
    return {
        "expected": len(rows),
        "attempted": sum(v != "not_attempted" for v in values),
        "valid": values.count("finite"),
        "technical_failure": values.count("technical_failure"),
        "not_attempted": values.count("not_attempted"),
        "unresolved": sum(
            r[field] == "finite" and r.get(interpretation) == "unresolved_preference"
            for r in rows
        ),
    }


def _sign(delta: float) -> str:
    return "improvement" if delta < 0 else "harm" if delta > 0 else "exact_tie"


def _interpret(delta: float, floor: float) -> str:
    return "unresolved_preference" if abs(delta) <= floor else _sign(delta)


def _coverage(row: dict[str, Any], failed: bool, nonfinite_reasons: list[str]) -> None:
    status = row["status"]
    _require(
        status
        in {
            "resolved",
            "zero_denominator",
            "unresolved_denominator",
            "nonorthogonal_basis",
            "projection_contract_unresolved",
            "nonfinite_action_or_resolution",
        },
        "action status",
    )
    for field in (
        "numerator",
        "denominator",
        "resolution_norm",
        "orthonormality_error",
    ):
        value = row[field]
        if value is not None:
            _number(value, f"action/{field}", nonnegative=True)
    if row["denominator"] is None or row["resolution_norm"] is None:
        _require(
            failed
            and bool(nonfinite_reasons)
            and status == "nonfinite_action_or_resolution"
            and row["eta_action"] is None,
            "nonfinite action reasons",
        )
        return
    denominator, resolution = row["denominator"], row["resolution_norm"]
    if status == "resolved":
        eta = _number(row["eta_action"], "eta_action", nonnegative=True)
        numerator, error = row["numerator"], row["orthonormality_error"]
        _require(
            math.sqrt(denominator) > resolution
            and numerator is not None
            and eta == numerator / denominator
            and eta <= 1 + 1e-4,
            "action projection arithmetic (no clipping)",
        )
        _require(
            (error is None and numerator == 0 and eta == 0)
            or (error is not None and error <= 1e-4),
            "checked orthonormal projection",
        )
    else:
        _require(row["eta_action"] is None, "unavailable eta is not zero")
        if status == "zero_denominator":
            _require(denominator == 0, "zero action denominator")
        elif status == "unresolved_denominator":
            _require(
                0 < math.sqrt(denominator) <= resolution,
                "unresolved action denominator",
            )


def _candidate(
    row: dict[str, Any],
    label: str,
    alpha: float | None,
    baselines: dict[str, Any] | None,
    floor: float,
) -> None:
    _require(
        row["alpha"] == alpha
        and row["kind"] == ("scale" if alpha is not None else "matched_down_only"),
        f"{label}: registered candidate",
    )
    for field in _STATUS_FIELDS.values():
        _require(
            row[field] in ("finite", "technical_failure", "not_attempted"),
            f"{field}: status",
        )
    if row["status"] == "not_attempted":
        _require(
            row["raw_status"] == row["applied_status"] == "not_attempted",
            "unattempted row statuses",
        )
        return
    _require(baselines is not None, "attempted candidate missing baseline")
    assert baselines is not None
    for field, key in (("loss_theta_repeats", "theta"), ("loss_d0_repeats", "d0")):
        _require(row[field] == baselines[key], "same-pass baseline repetitions")
    _require(
        row["loss_theta"] == baselines["theta"][0]
        and row["loss_d0"] == baselines["d0"][0]
        and row["model_baseline"] == baselines["model_d0"],
        "same-pass baseline scalars",
    )
    delta = row["raw_delta_loss"]
    if delta is not None:
        _number(delta, "raw delta loss")
        _require(
            row["loss"] == row["loss_repeats"][0]
            and delta == row["loss"] - row["loss_d0"]
            and row["actual_gain"] == -delta,
            "raw loss/gain arithmetic",
        )
        _require(
            row["strict_sign"] == _sign(delta)
            and row["interpretation"] == _interpret(delta, floor),
            "strict sign and frozen-floor interpretation",
        )
    else:
        _require(
            row["actual_gain"] is None and row["raw_status"] != "finite",
            "undefined raw gain",
        )
    if row["raw_status"] == "finite":
        _require(delta is not None, "finite raw endpoint")
        for values in (row["loss_repeats"], baselines["d0"]):
            _require(len(values) == 3, "three raw-loss repetitions")
            for value in values:
                _number(value, "raw-loss repetition")
    if row["status"] == "technical_failure":
        _require(
            bool(row.get("technical_error") or row.get("numerical_failures")),
            "technical row reasons",
        )
        _require(
            row["applied_status"] == "technical_failure"
            and row["applied_gain"] is None
            and row["applied_direction"] is None,
            "technical applied outcome remains null",
        )
        return
    _require(
        row["raw_status"] == row["applied_status"] == "finite",
        "finite joint/raw/applied status",
    )
    _require(baselines["status"] == "finite", "finite candidate baseline")
    norm0 = _number(baselines["norm_d0"], "baseline norm", nonnegative=True)
    norm = _number(row["step_norm"], "step norm", nonnegative=True)
    correction = _number(row["correction_norm"], "correction norm", nonnegative=True)
    _require(
        type(row["near_zero_correction"]) is bool
        and row["near_zero_correction"]
        == (correction == 0 or correction < 1e-3 * norm0),
        "near-zero correction",
    )
    model = row["model"]
    for key in ("undamped", "damping", "damped", "model_delta"):
        _number(model[key], f"model/{key}", nonnegative=key == "damping")
    _require(
        model["model_delta"] == model["damped"] - baselines["model_d0"], "model delta"
    )
    gates = row["gates"]
    expected = {
        "ratio": norm <= 2 * norm0,
        "model": model["model_delta"] <= 0,
        "loss": row["loss"] <= row["loss_d0"],
    }
    for key, value in expected.items():
        _require(
            type(gates[key]) is bool and gates[key] == value,
            f"strict native gate/{key}",
        )
    accepted = all(expected.values())
    _require(
        type(gates["accepted"]) is bool
        and gates["accepted"] == accepted
        and gates["rejected_by"]
        == next((k for k, ok in expected.items() if not ok), None),
        "native acceptance partition",
    )
    gain = row["applied_gain"]
    _require(
        gain == (row["actual_gain"] if accepted else 0.0)
        and row["applied_direction"] == ("raw" if accepted else "d0"),
        "separate applied outcome",
    )
    _require(
        row["applied_strict_sign"] == _sign(-gain)
        and row["applied_interpretation"] == _interpret(-gain, floor),
        "applied interpretation",
    )
    if alpha == 0:
        # Separate loss calls may vary; only the zero direction is exact here.
        _require(correction == 0 and norm == norm0, "alpha zero direction")
    if alpha is None:
        _number(row["norm_match_scale"], "norm-match scale", nonnegative=True)
        _require(row["norm_match_scale"] <= 1, "norm matching only decreases")
    else:
        _require(row["norm_match_scale"] is None, "scale-grid identity")


def _draw(
    draw: dict[str, Any],
    result: dict[str, Any],
    name: str,
    floor: float,
    geometry_receipts: dict[str, str],
) -> None:
    seed, step = result["seed"], result["step"]
    _require(
        (draw["seed"], draw["step"], draw["drawset"]) == (seed, step, name),
        "draw identity",
    )
    _require(
        draw["seeds"]
        == {"coupling": seed + _DRAWS[name][0], "basis": seed + _DRAWS[name][1]},
        "registered geometry seeds",
    )
    _require(
        draw["status"] in ("complete", "technical_failure", "not_attempted"),
        "terminal draw status",
    )
    measurements = draw["measurements"]
    _require(set(measurements["arms"]) == set(_ARMS), "three residual arms")
    baselines = measurements.get("baselines")
    if baselines is not None:
        _require(
            baselines["status"] in ("finite", "technical_failure"), "baseline status"
        )
        for key in ("theta", "d0"):
            _require(len(baselines[key]) == 3, "three baseline repetitions")
            for value in baselines[key]:
                if value is not None:
                    _number(value, f"baseline/{key}")
        if baselines["status"] == "finite":
            for key in ("model_d0", "norm_d0"):
                _number(baselines[key], f"baseline/{key}", nonnegative=key == "norm_d0")
    for arm, rows in measurements["arms"].items():
        _require(set(rows) == set(_labels(arm)), "complete alpha/matched roster")
        for label, alpha in _labels(arm).items():
            _candidate(rows[label], label, alpha, baselines, floor)
        for key, field in _STATUS_FIELDS.items():
            _require(
                measurements[key][arm] == _counts(list(rows.values()), field),
                f"{key}/{arm}: valid/failure/not-attempted/unresolved counts",
            )
    _require(
        measurements["unresolved_is_subset_of_valid"] is True,
        "unresolved count semantics",
    )
    if draw["status"] == "not_attempted":
        _require(
            all(
                c["not_attempted"] == 9
                for key in _STATUS_FIELDS
                for c in measurements[key].values()
            ),
            "unattempted draw accounting",
        )
        _require(name not in geometry_receipts, "unattempted draw has geometry receipt")
        return
    if "factored_geometry" in draw:
        artifact = draw["factored_geometry"]
        expected = {
            "measurement_schema": 2,
            "seed": seed,
            "step": step,
            "drawset": name,
            "seeds": draw["seeds"],
        }
        if result.get("format") != "curv-scientific-residual-record-v1":
            expected.update(
                {
                    "capture_sha256": result["capture_sha256"],
                    "current_source_sha256": result["current_source_sha256"],
                    "registration_sha256": result["registration"]["sha256"],
                    "filename": f"factored_{name}.pt",
                }
            )
        for key, value in expected.items():
            _require(artifact[key] == value, f"geometry identity/{key}")
        if result.get("format") != "curv-scientific-residual-record-v1":
            _hash(artifact["sha256"])
            _require(
                geometry_receipts.get(name) == artifact["sha256"],
                "geometry checksum receipt",
            )
    else:
        _require(
            draw["status"] == "technical_failure" and name not in geometry_receipts,
            "complete geometry artifact missing",
        )
    if "coverage" in draw:
        _coverage(
            draw["coverage"],
            result["status"] != "complete",
            result.get("nonfinite_fields", []),
        )
        _require(
            draw["coverage"]["resolution_norm"] == draw["hvp_residual_resolution"],
            "action resolution source",
        )
    if draw["status"] == "complete":
        _require(
            "factored_geometry" in draw
            and "coverage" in draw
            and all(
                c["valid"] == 9
                for key in _STATUS_FIELDS
                for c in measurements[key].values()
            ),
            "complete draw accounting",
        )
        _require(
            draw["split_checks"]["passed"] is True
            and [r["group"] for r in draw["source_checks"]] == list(_GROUPS),
            "six-source residual checks",
        )
        correction = draw["correction_additivity"]
        _require(
            correction["passed"] is True
            and correction["native_full_check"]["passed"] is True
            and correction["native_full_check"]["bitwise_equal"] is True
            and correction["native_full_check"]["comparison_space"] == "direction",
            "native direction/additivity checks",
        )
        _require(
            correction["recovered_correction_check"]["diagnostic_only"] is True,
            "recovered correction is diagnostic only",
        )
        _require(
            all(
                draw["independent_flat_ad"][k]["passed"] is True
                for k in ("loss", "gradient", "hvp_d0")
            ),
            "independent flat-AD checks",
        )
        _require(
            all(
                draw["replay_checks"][k]["passed"] is True for k in ("loss", "gradient")
            ),
            "recorded reference checks",
        )
    if baselines is not None:
        _require(
            measurements["norm_matched_pairs"] == list(_PAIRS),
            "three pairwise controls",
        )
        for pair in _PAIRS:
            left, right = pair.split("__")
            norms = draw.get("correction_norms", {})
            for arm in (left, right):
                candidate = measurements["arms"][arm][f"matched_{pair}"]
                if candidate["status"] == "finite":
                    expected_scale = (
                        min(norms[left], norms[right]) / norms[arm]
                        if norms[arm]
                        else 0.0
                    )
                    _require(
                        candidate["norm_match_scale"] == expected_scale,
                        "same pairwise down-only target",
                    )
        _require(
            set(measurements["loss_interaction"]) == {f"alpha_{i}" for i in range(7)},
            "interaction grid",
        )
        for label, interaction in measurements["loss_interaction"].items():
            rows = [measurements["arms"][a][label] for a in _ARMS]
            if all(r["raw_status"] == "finite" for r in rows):
                value = (
                    rows[0]["raw_delta_loss"]
                    - rows[1]["raw_delta_loss"]
                    - rows[2]["raw_delta_loss"]
                )
                _require(
                    interaction["status"]
                    == ("finite" if math.isfinite(value) else "nonfinite")
                    and interaction["delta_loss_full_minus_components"]
                    == (value if math.isfinite(value) else None),
                    "raw loss interaction arithmetic",
                )
            else:
                _require(
                    interaction["status"] == "technical_unavailable",
                    "unavailable interaction",
                )


def _validate_record(
    result: dict[str, Any],
    status: dict[str, Any],
    context: dict[str, Any],
    geometry: dict[str, str],
    *,
    phase: str,
) -> None:
    public = context.get("format") == "curv-scientific-residual-v1"
    _require(
        (result.get("format") == "curv-scientific-residual-record-v1") == public,
        "record format identity",
    )
    if public:
        registration = (
            context["protocol"]
            if phase == "final"
            else context["calibration"]["smoke_protocol"]
        )
    else:
        registration = (
            context["registration"]
            if phase == "final"
            else context["calibration"]["smoke_registration"]
        )
    capture = next(
        (
            r
            for r in registration["capture_roster"]
            if (r["seed"], r["step"]) == (result["seed"], result["step"])
        ),
        None,
    )
    _require(capture is not None, "registered state identity")
    assert capture is not None
    expected_identity = {
        "measurement_schema": 2,
        "phase": phase,
        "historical_probe_replay": False,
        "d0_recomputed": False,
    }
    if not public:
        reference = (
            context["provenance"]["registration"]
            if phase == "final"
            else context["registration"]["smoke_calibration"]["smoke_registration"]
        )
        expected_identity.update(
            {
                "requested_capture": capture["capture_path"],
                "capture_sha256": capture["capture_sha256"],
                "historical_source_sha256": capture["historical_source_sha256"],
                "current_source_sha256": registration["current_source_sha256"],
                "registration": {**reference, "phase": phase},
                "environment_identity": registration["environment_identity"],
            }
        )
    for key, expected in expected_identity.items():
        _require(result[key] == expected, f"record/{key}")
    _require(
        result["status"] in ("complete", "technical_failure"), "terminal result status"
    )
    _require(
        status["schema"] == 1
        and status["kind"] == f"residual_ablation_v2_{phase}"
        and status["status"] == result["status"],
        "terminal status/result envelope",
    )
    identity_fields = ["measurement_schema", "phase", "seed", "step"]
    if not public:
        identity_fields.extend(
            (
                "requested_capture",
                "capture_sha256",
                "current_source_sha256",
                "registration",
            )
        )
    _require(
        status["measurement_identity"] == {k: result[k] for k in identity_fields},
        "status measurement identity",
    )
    if result["status"] == "complete":
        _require(
            result["capture_status"] == "complete"
            and result["dtype"] == "torch.float32"
            and isinstance(result["runtime_flags"], dict)
            and result["state_preservation_check"]["passed"] is True
            and result["technical_validation"]
            == "measured_requires_external_smoke_or_result_review",
            "completed technical envelope",
        )
        _require(result["thresholds"] == context["thresholds"], "accepted thresholds")
    else:
        _require(bool(result.get("technical_error")), "technical state reason")
        if "thresholds" in result:
            _require(
                result["thresholds"] == context["thresholds"], "failed-state thresholds"
            )
    if phase == "final" and not public:
        _require(
            result["loading_preflight"]
            == {
                "status": "bound_to_frozen_smoke",
                "smoke_results": registration["smoke_calibration"]["smoke_results"],
            },
            "final loading-preflight binding",
        )
    _require(set(result["drawsets"]) == set(_DRAWS), "both drawsets predeclared")
    if not public:
        _require(
            set(geometry)
            == {
                name
                for name, d in result["drawsets"].items()
                if "factored_geometry" in d
            },
            "geometry index roster",
        )
    for name, draw in result["drawsets"].items():
        _draw(draw, result, name, context["thresholds"]["gain_floor"], geometry)
    expected_draws = {
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
    _require(result["drawset_counts"] == expected_draws, "drawset accounting")
    if result["status"] == "complete":
        _require(expected_draws["valid"] == 2, "complete state has two completed draws")
    for key in _STATUS_FIELDS:
        expected = {
            arm: {
                field: sum(
                    d["measurements"][key][arm][field]
                    for d in result["drawsets"].values()
                )
                for field in _COUNT_FIELDS
            }
            for arm in _ARMS
        }
        _require(result[key] == expected, f"state/{key}: 18 expected rows per arm")
    _require(
        result["count_semantics"]
        == "unresolved is a subset of valid, not an additional outcome; steps/draws are not independent training seeds",
        "count semantics",
    )


def _validate_scientific_context(
    registration: dict[str, Any], fixed: dict[str, Any]
) -> None:
    for key, expected in {
        "measurement_schema": 2,
        "historical_probe_replay": False,
        "d0_recomputed": False,
        "partition": list(_GROUPS),
        "seeds": list(_SEEDS),
        "steps": list(_STEPS),
        "drawsets": {k: list(v) for k, v in _DRAWS.items()},
        "alpha_grid": list(_GRID),
        "lambda": 1.0,
        "coupling_probes": 10,
        "rank": 4,
        "power_iterations": 8,
        "tau": 0.1,
        "max_pairs": 3,
        "norm_ratio": 2.0,
        "tolerances": _TOLERANCES,
        "precision": "saved_runtime_float32_no_autocast_change",
        "cohorts": "smoke_and_final_separate;final_remeasures_all_nine_states",
    }.items():
        _require(registration[key] == expected, f"registration/{key}")
    roster = registration["capture_roster"]
    _require(
        [(r["seed"], r["step"]) for r in roster]
        == [(s, t) for s in _SEEDS for t in _STEPS]
        and all(r["status"] == "bytes_checked_not_loaded" for r in roster),
        "nine registered states",
    )
    for key, expected in {
        "measurement_schema": 2,
        "capture_seed": 42,
        "capture_step": 60,
        "tolerances": _TOLERANCES,
        "role": "technical_smoke_only_not_final_data",
    }.items():
        _require(fixed[key] == expected, f"accepted calibration/{key}")
    spread = _number(
        fixed["same_candidate_repeatability_spread"], "smoke spread", nonnegative=True
    )
    _require(
        _number(fixed["gain_floor"], "gain floor") == max(1e-4, 10 * spread),
        "accepted smoke floor formula",
    )


def _validate_loading_preflight(
    preflight: dict[str, Any], roster: list[dict[str, Any]]
) -> None:
    """Validate the complete smoke loading cohort and operation flags."""
    for key, expected in {
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
    }.items():
        _require(
            preflight[key] == expected
            and (
                type(expected) not in (bool, int)
                or type(preflight[key]) is type(expected)
            ),
            f"all-nine load preflight/{key}",
        )
    _require(len(preflight["entries"]) == len(roster) == 9, "preflight entries")
    for registered, entry in zip(roster, preflight["entries"], strict=True):
        _require(
            all(entry[k] == v for k, v in registered.items() if k != "status")
            and entry["status"] == "compatible_loaded"
            and entry["state_preserved"] is True,
            "loaded capture roster identity",
        )


def _validate_context(context: dict[str, Any]) -> None:
    _json_bytes(
        context
    )  # Reject nonfinite JSON anywhere, including failure diagnostics.
    registration, fixed = context["registration"], context["thresholds"]
    smoke = context["calibration"]["smoke_registration"]
    binding = registration["smoke_calibration"]
    _reference(context["provenance"]["registration"], registration)
    _require(
        registration
        == {**smoke, "registration_phase": "final", "smoke_calibration": binding}
        and smoke["registration_phase"] == "smoke",
        "exact final/smoke registration binding",
    )
    _require(
        set(binding)
        == {"thresholds", "smoke_registration", "smoke_results", "smoke_status"}
        and context["provenance"]["calibration"] == binding,
        "four frozen calibration artifacts",
    )
    for key in binding:
        _reference(
            binding[key], fixed if key == "thresholds" else context["calibration"][key]
        )
    _validate_scientific_context(registration, fixed)
    _hash(registration["current_source_sha256"])
    environment = registration["environment_identity"]
    _require(
        set(environment)
        == {
            "python",
            "python_executable",
            "sys_prefix",
            "torch",
            "torch_cuda_build",
            "numpy",
            "torchvision",
        },
        "recorded environment fields",
    )
    _require(
        all(
            isinstance(environment[k], str) and environment[k]
            for k in (
                "python",
                "python_executable",
                "sys_prefix",
                "torch",
                "numpy",
                "torchvision",
            )
        )
        and all(
            Path(environment[k]).is_absolute()
            for k in ("python_executable", "sys_prefix")
        ),
        "recorded environment identity",
    )
    roster = registration["capture_roster"]
    _require(
        [(r["seed"], r["step"]) for r in roster]
        == [(s, t) for s in _SEEDS for t in _STEPS]
        and len({r["capture_path"] for r in roster}) == 9,
        "nine registered captures",
    )
    for row in roster:
        _require(
            row["status"] == "bytes_checked_not_loaded"
            and Path(row["capture_path"]).is_absolute()
            and isinstance(row["historical_source_sha256"], str)
            and bool(row["historical_source_sha256"]),
            "recorded capture identity",
        )
        _hash(row["capture_sha256"])
    for key, expected in {
        "measurement_schema": 2,
        "source_sha256": registration["current_source_sha256"],
        "capture_seed": 42,
        "capture_step": 60,
        "capture_sha256": roster[1]["capture_sha256"],
        "tolerances": _TOLERANCES,
        "role": "technical_smoke_only_not_final_data",
        "smoke_registration_sha256": binding["smoke_registration"]["sha256"],
        "environment_identity": environment,
    }.items():
        _require(fixed[key] == expected, f"accepted calibration/{key}")
    result = context["calibration"]["smoke_results"]
    _require(
        (result["seed"], result["step"], result["status"]) == (42, 60, "complete"),
        "completed calibration state",
    )
    preflight = result["loading_preflight"]
    _validate_loading_preflight(preflight, roster)
    for key, expected in {
        "registration_sha256": binding["smoke_registration"]["sha256"],
        "current_source_sha256": registration["current_source_sha256"],
        "environment_identity": environment,
    }.items():
        _require(preflight[key] == expected, f"frozen all-nine load preflight/{key}")
    _validate_record(
        result,
        context["calibration"]["smoke_status"],
        context,
        context["calibration"]["geometry_sha256"],
        phase="smoke",
    )


def _validate_public_context(context: dict[str, Any]) -> None:
    _json_bytes(context)
    _require("provenance" not in context, "public envelope")
    registration = context["protocol"]
    smoke = context["calibration"]["smoke_protocol"]
    _require(
        smoke["registration_phase"] == "smoke"
        and registration == {**smoke, "registration_phase": "final"},
        "smoke/final scientific protocol",
    )
    _validate_scientific_context(registration, context["thresholds"])
    result = context["calibration"]["smoke_results"]
    _require(
        (result["seed"], result["step"], result["status"]) == (42, 60, "complete"),
        "completed calibration state",
    )
    _validate_loading_preflight(
        result["loading_preflight"], registration["capture_roster"]
    )
    _validate_record(
        result, context["calibration"]["smoke_status"], context, {}, phase="smoke"
    )


def validated_ablation(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate scientific records and, for author bundles, frozen receipts."""
    try:
        _require(
            bundle["schema"] == 1
            and bundle["bundle"] == "residual_ablation_v2"
            and bundle["phase"] == "final",
            "final bundle identity",
        )
        public = bundle.get("format") == "curv-scientific-residual-v1"
        if public:
            _validate_public_context(bundle)
        else:
            _validate_context(bundle)
        states = bundle["states"]
        _require(
            all(
                type(state["seed"]) is int and type(state["step"]) is int
                for state in states
            ),
            "canonical integer state identity",
        )
        identities = [(s, t) for s in _SEEDS for t in _STEPS]
        _require(
            sorted((s["seed"], s["step"]) for s in states) == identities,
            "nine unique final states",
        )
        if public:
            inputs = [None] * len(states)
        else:
            inputs = bundle["provenance"]["inputs"]
            _require(
                [(r["seed"], r["step"]) for r in inputs] == identities
                and len({r["path"] for r in inputs}) == 9,
                "nine input artifact indices",
            )
        for state, index in zip(
            sorted(states, key=lambda r: (r["seed"], r["step"])), inputs, strict=True
        ):
            result = state["result"]
            _require(
                (result["seed"], result["step"], result["status"])
                == (state["seed"], state["step"], state["status"]),
                "state/result identity",
            )
            geometry: dict[str, str] = {}
            if not public:
                assert index is not None
                geometry = state["geometry_sha256"]
                _require(
                    index["geometry_sha256"] == geometry, "indexed geometry receipts"
                )
                for name, value in (
                    ("results.json", result),
                    ("status.json", state["status_record"]),
                ):
                    _reference(
                        {
                            "path": str(Path(index["path"]) / name),
                            "sha256": index["sha256"][name],
                        },
                        value,
                    )
            _validate_record(
                result,
                state["status_record"],
                bundle,
                geometry,
                phase="final",
            )
        return sorted(states, key=lambda r: (r["seed"], r["step"]))
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        AttributeError,
        StopIteration,
    ) as exc:
        if str(exc).startswith("residual ablation:"):
            raise
        raise ValueError(f"residual ablation: invalid scalar records: {exc}") from exc


def candidate_rows(bundle: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """All 486 predeclared slots, with separate joint/raw/applied statuses."""
    for state in validated_ablation(bundle):
        for name, draw in state["result"]["drawsets"].items():
            for arm in _ARMS:
                for label in _labels(arm):
                    yield {
                        **draw["measurements"]["arms"][arm][label],
                        "seed": state["seed"],
                        "step": state["step"],
                        "drawset": name,
                        "arm": arm,
                        "candidate": label,
                        "state_status": state["status"],
                        "drawset_status": draw["status"],
                    }


def contrast_values(
    candidates: dict[str, dict[str, Any]], *, matched: bool = False
) -> dict[str, Any]:
    """Raw loss contrasts and observed repeat extrema, not error bounds or CIs."""
    names = ("full", "rest") if matched else ("full", "off", "rest")
    keys = ("y",) if matched else ("x", "y", "j")
    rows = [candidates.get(name) for name in names]
    if any(row is None or row.get("raw_status") != "finite" for row in rows):
        return {
            "status": "technical_unavailable",
            **dict.fromkeys(keys),
            "envelopes": dict.fromkeys(keys),
        }
    data = {name: candidates[name] for name in names}
    baseline = data["full"]["loss_d0"]
    repeats = data["full"]["loss_d0_repeats"]
    _require(len(repeats) == 3, "baseline repeat count")
    for value in repeats:
        _number(value, "baseline repeat")
    _number(baseline, "baseline")
    _require(repeats[0] == baseline, "baseline first repeat")
    for row in data.values():
        _require(
            row["loss_d0"] == baseline and row["loss_d0_repeats"] == repeats,
            "common baseline and repeats required",
        )
        _number(row["loss"], "candidate loss")
        values = row["loss_repeats"]
        _require(len(values) == 3, "candidate repeat count")
        for value in values:
            _number(value, "candidate repeat")
        _require(values[0] == row["loss"], "candidate first repeat")
    full, rest = data["full"], data["rest"]
    y = full["loss"] - rest["loss"]

    def bounds(row: dict[str, Any]) -> tuple[float, float]:
        return min(row["loss_repeats"]), max(row["loss_repeats"])

    fl, fh = bounds(full)
    rl, rh = bounds(rest)
    envelopes = {"y": [fl - rh, fh - rl]}
    result: dict[str, Any] = {"status": "finite", "y": y, "envelopes": envelopes}
    if not matched:
        off = data["off"]
        ol, oh = bounds(off)
        bl, bh = min(repeats), max(repeats)
        x = off["loss"] - baseline
        # Match the producer's stored delta arithmetic and its binary rounding.
        j = (full["loss"] - baseline) - x - (rest["loss"] - baseline)
        envelopes.update(
            {"x": [ol - bh, oh - bl], "j": [fl - oh - rh + bl, fh - ol - rl + bh]}
        )
        result.update({"x": x, "j": j})
    for value in (result[key] for key in keys):
        _number(value, "contrast")
    return result


def residual_contrast_rows(bundle: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Seven scale contrasts and one separate matched pair per state/draw."""
    for state in validated_ablation(bundle):
        for name in _DRAWS:
            draw = state["result"]["drawsets"][name]
            measurements = draw["measurements"]
            arms = measurements["arms"]
            identity = {
                "bundle": "residual_ablation_v2",
                "seed": state["seed"],
                "step": state["step"],
                "drawset": name,
                "state_status": state["status"],
                "drawset_status": draw["status"],
            }
            for i, alpha in enumerate(_GRID):
                label = f"alpha_{i}"
                candidates = {arm: arms[arm][label] for arm in _ARMS}
                values = contrast_values(candidates)
                stored = measurements.get("loss_interaction", {}).get(label)
                if values["status"] == "finite":
                    _require(
                        stored is not None
                        and stored["status"] == "finite"
                        and stored["delta_loss_full_minus_components"] == values["j"],
                        "stored loss_interaction mismatch",
                    )
                elif stored is not None:
                    _require(
                        stored["status"] == "technical_unavailable",
                        "unavailable interaction",
                    )
                yield {
                    **identity,
                    "type": "scale",
                    "alpha": alpha,
                    "candidates": {arm: label for arm in _ARMS},
                    "raw_statuses": {
                        arm: candidates[arm]["raw_status"] for arm in _ARMS
                    },
                    "delta_full": candidates["full"].get("raw_delta_loss"),
                    "delta_rest": candidates["rest"].get("raw_delta_loss"),
                    **values,
                }
            label = "matched_full__rest"
            candidates = {arm: arms[arm][label] for arm in ("full", "rest")}
            yield {
                **identity,
                "type": "matched_full_rest",
                "candidates": {arm: label for arm in candidates},
                "raw_statuses": {
                    arm: candidates[arm]["raw_status"] for arm in candidates
                },
                "norm_match_scales": {
                    arm: candidates[arm].get("norm_match_scale") for arm in candidates
                },
                **contrast_values(candidates, matched=True),
            }


def contrast_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """Strict descriptive counts by draw and training seed, with unavailable slots."""
    rows = list(residual_contrast_rows(bundle))
    matched = {
        (r["seed"], r["step"], r["drawset"]): r
        for r in rows
        if r["type"] == "matched_full_rest"
    }
    result: dict[str, Any] = {}
    for draw in _DRAWS:
        groups: dict[str, Any] = {}
        for seed in (*_SEEDS, None):
            selected = [
                r
                for r in rows
                if r["type"] == "scale"
                and r["alpha"] == 1
                and r["drawset"] == draw
                and (seed is None or r["seed"] == seed)
            ]
            counts = {
                key: 0
                for key in (
                    "total",
                    "unavailable",
                    "off_harm",
                    "rest_harm",
                    "full_harm",
                    "full_improves",
                    "full_over_rest",
                    "off_harm_full_over_rest",
                    "both_harm_full_improves",
                    "j_negative",
                    "matched_unavailable",
                    "matched_reversal",
                    "matched_persists",
                    "matched_tie",
                    "preference_full_harm",
                )
            }
            for row in selected:
                counts["total"] += 1
                if row["status"] != "finite":
                    counts["unavailable"] += 1
                    continue
                x, y, full, rest = (
                    row["x"],
                    row["y"],
                    row["delta_full"],
                    row["delta_rest"],
                )
                counts["off_harm"] += x > 0
                counts["rest_harm"] += rest > 0
                counts["full_harm"] += full > 0
                counts["full_improves"] += full < 0
                counts["full_over_rest"] += y < 0
                counts["off_harm_full_over_rest"] += x > 0 and y < 0
                counts["both_harm_full_improves"] += x > 0 and rest > 0 and full < 0
                counts["j_negative"] += row["j"] < 0
                if y < 0:
                    counts["preference_full_harm"] += full > 0
                    pair = matched[(row["seed"], row["step"], draw)]
                    if pair["status"] != "finite":
                        counts["matched_unavailable"] += 1
                    elif pair["y"] > 0:
                        counts["matched_reversal"] += 1
                    elif pair["y"] < 0:
                        counts["matched_persists"] += 1
                    else:
                        counts["matched_tie"] += 1
            groups[str(seed) if seed is not None else "all_states"] = counts
        result[draw] = groups
    return result


def ablation_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """Descriptive seed-first outcomes; raw finite losses survive model failures."""
    states = validated_ablation(bundle)
    rows = list(candidate_rows(bundle))
    summary: dict[str, Any] = {
        "states": 9,
        "complete_states": sum(s["status"] == "complete" for s in states),
        "per_seed": {},
        "drawsets": [],
    }
    summary.update({key: _counts(rows, field) for key, field in _STATUS_FIELDS.items()})
    for state in states:
        for name in _DRAWS:
            draw = state["result"]["drawsets"][name]
            arms = draw["measurements"]["arms"]
            primary = {arm: arms[arm]["alpha_6"] for arm in _ARMS}
            tally = (
                summary["per_seed"]
                .setdefault(str(state["seed"]), {})
                .setdefault(name, {})
            )
            for arm, row in primary.items():
                counts = tally.setdefault(
                    arm,
                    {
                        kind: {
                            k: 0
                            for k in (
                                "improvement",
                                "harm",
                                "unresolved_preference",
                                "technical_failure",
                                "not_attempted",
                                "strict_exact_ties",
                                "technical_state",
                            )
                        }
                        for kind in ("raw", "applied")
                    },
                )
                for kind, field, interpretation, strict in (
                    ("raw", "raw_status", "interpretation", "strict_sign"),
                    (
                        "applied",
                        "applied_status",
                        "applied_interpretation",
                        "applied_strict_sign",
                    ),
                ):
                    status = row[field]
                    counts[kind][
                        row[interpretation] if status == "finite" else status
                    ] += 1
                    counts[kind]["strict_exact_ties"] += (
                        status == "finite" and row[strict] == "exact_tie"
                    )
                    counts[kind]["technical_state"] += (
                        status == "finite" and state["status"] != "complete"
                    )
            scale_min, matched = {}, {}
            for arm in _ARMS:
                scale = [arms[arm][f"alpha_{i}"] for i in range(7)]
                if all(r["raw_status"] == "finite" for r in scale):
                    best = min(scale, key=lambda r: r["raw_delta_loss"])
                    scale_min[arm] = {
                        "status": "finite",
                        "raw_delta_loss": best["raw_delta_loss"],
                        "alpha": best["alpha"],
                    }
                else:
                    scale_min[arm] = {"status": "unavailable_incomplete_raw_grid"}
            for pair in _PAIRS:
                a, b = (arms[arm][f"matched_{pair}"] for arm in pair.split("__"))
                if a["raw_status"] == b["raw_status"] == "finite":
                    status = (
                        "norm_unavailable"
                        if a.get("near_zero_correction") is None
                        or b.get("near_zero_correction") is None
                        else "near_zero_correction"
                        if a["near_zero_correction"] or b["near_zero_correction"]
                        else "finite"
                    )
                    matched[pair] = {
                        "status": status,
                        "left_minus_right": a["raw_delta_loss"] - b["raw_delta_loss"],
                        "left": a["raw_delta_loss"],
                        "right": b["raw_delta_loss"],
                    }
                else:
                    matched[pair] = {"status": "unavailable_raw_pair"}
            summary["drawsets"].append(
                {
                    "seed": state["seed"],
                    "step": state["step"],
                    "drawset": name,
                    "state_status": state["status"],
                    "drawset_status": draw["status"],
                    "primary": primary,
                    "coverage": draw.get("coverage"),
                    "scale_min": scale_min,
                    "matched": matched,
                }
            )
    summary["interpretation"] = (
        "Recorded finite raw losses are separate from joint technical validity and applied outcomes. Descriptions stay within training seed; scale minima are retrospective grid oracles, not iid inference or numerical certification."
    )
    return summary


def _scalar(value: float | None) -> str:
    return "---" if value is None else f"${sci(value, 2)}$"


def _cell(row: dict[str, Any], applied: bool, state_failed: bool) -> str:
    status = row["applied_status" if applied else "raw_status"]
    value = (
        -row["applied_gain"]
        if applied and row.get("applied_gain") is not None
        else row.get("raw_delta_loss")
        if not applied
        else None
    )
    flags = []
    if status != "finite":
        flags.append("NA" if status == "not_attempted" else "T")
    elif (
        row["applied_interpretation" if applied else "interpretation"]
        == "unresolved_preference"
    ):
        flags.append("U")
    if state_failed:
        flags.append("Tstate")
    if not applied and row.get("gates") is not None:
        flags.append("A" if row["gates"]["accepted"] else "R")
    if row.get("near_zero_correction"):
        flags.append("Z")
    return _scalar(value) + (" (" + ",".join(flags) + ")" if flags else "")


def ablation_table(residual_ablation_v2: dict[str, Any]) -> str:
    """All state/draw primary raw losses and separate applied endpoints."""
    summary = ablation_summary(residual_ablation_v2)
    lines = []
    shown_flags: set[str] = set()
    coverage_codes = {
        "zero_denominator": "zero denom.",
        "unresolved_denominator": "unresolved",
        "nonorthogonal_basis": "nonorthog.",
        "projection_contract_unresolved": "projection",
        "nonfinite_action_or_resolution": "nonfinite",
    }
    for row in summary["drawsets"]:
        for applied in (False, True):
            cells = [
                str(row["seed"]) if not applied else "",
                str(row["step"]) if not applied else "",
                ("P" if row["drawset"] == "primary" else "S") if not applied else "",
                "applied" if applied else "raw",
            ]
            for arm in _ARMS:
                cell = _cell(
                    row["primary"][arm], applied, row["state_status"] != "complete"
                )
                cells.append(cell)
                flags = cell.partition(" (")[2].removesuffix(")")
                if flags:
                    shown_flags.update(flags.split(","))
            coverage = row["coverage"]
            cells.append(
                ""
                if applied
                else _scalar(coverage["eta_action"])
                if coverage is not None and coverage["status"] == "resolved"
                else coverage_codes[coverage["status"]]
                if coverage is not None
                else "not attempted"
            )
            lines.append(
                "    "
                + " & ".join(cells)
                + r" \\"
                + (
                    f" % state {row['seed']}/{row['step']}/{row['drawset']}"
                    if not applied
                    else ""
                )
            )
    body = "\n".join(lines)
    extra_flags = "".join(
        note
        for present, note in (
            ("Z" in shown_flags, " Z denotes a near-zero correction."),
            ("T" in shown_flags, " T denotes technical failure."),
            ("NA" in shown_flags, " NA denotes not attempted."),
            (
                "Tstate" in shown_flags,
                " Tstate flags an invalid state even when a finite raw loss was retained.",
            ),
        )
        if present
    )
    return rf"""\begin{{table*}}[htbp]
  \centering
  \caption{{Residual-input ablation at $\alpha=1$, all nine states and both
    registered geometry draws (P/S). Raw $\Delta L$ precedes every gate; applied
    $\Delta L$ is separate and is zero only for an actual finite fallback/tie.
    Negative is improvement. A/R denotes native acceptance/rejection; U denotes
    the frozen operational preference band.{extra_flags} Missing values are never zeros.
    $\eta_{{\rm action}}$ is the checked projection of summed off action, not
    Frobenius coverage; nonorthog. denotes a non-orthogonal basis, and undefined
    reasons remain distinct. These are descriptive
    within-seed observations, not independent draws or numerical certificates.}}
  \label{{tab:residual-ablation}}
  \scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{@{{}}rrclllll@{{}}}}
    \toprule
    Seed & Step & Draw & Endpoint & Full $\Delta L$ & Off $\Delta L$ & Rest $\Delta L$ & $\eta_{{\rm action}}$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}"""


def ablation_controls(residual_ablation_v2: dict[str, Any]) -> str:
    """Scale-grid minima and pairwise down-only comparisons, retaining unavailable."""
    summary = ablation_summary(residual_ablation_v2)
    lines = []
    has_zero = False
    has_technical_state = False
    for row in summary["drawsets"]:
        cells = [
            str(row["seed"]),
            str(row["step"]),
            "P" if row["drawset"] == "primary" else "S",
        ]
        cells.extend(
            _scalar(row["scale_min"][arm].get("raw_delta_loss")) for arm in _ARMS
        )
        for pair in _PAIRS:
            comparison = row["matched"][pair]
            has_zero |= comparison["status"] == "near_zero_correction"
            cells.append(
                _scalar(comparison.get("left_minus_right"))
                if comparison["status"] == "finite"
                else "Z"
                if comparison["status"] == "near_zero_correction"
                else "---"
            )
        if row["state_status"] != "complete":
            has_technical_state = True
            cells[2] += "/T"
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    extra_flags = (
        " Z denotes a near-zero correction, not a directional win." if has_zero else ""
    ) + (" /T flags a technical state." if has_technical_state else "")
    return rf"""\begin{{table*}}[htbp]
  \centering
  \caption{{Scale and pairwise down-only norm diagnostics in fixed geometry.
    Minima use the entire registered seven-point raw grid, including zero;
    incomplete grids are unavailable. They are retrospective oracles, not an
    online line search. Pair columns are left-minus-right raw $\Delta L$ at the
    matched correction norms; missing raw/norm evidence is unavailable.{extra_flags}}}
  \label{{tab:residual-controls}}
  \scriptsize
  \setlength{{\tabcolsep}}{{2pt}}
  \begin{{tabular}}{{@{{}}rrcrrrrrr@{{}}}}
    \toprule
    Seed & Step & Draw & min full & min off & min rest & full--off & full--rest & off--rest \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}"""


def ablation_numbers(residual_ablation_v2: dict[str, Any]) -> str:
    """Accounting plus seed-first raw/applied macros; never pooled iid inference."""
    summary = ablation_summary(residual_ablation_v2)
    values = {
        "AblationStates": summary["states"],
        "AblationCompleteStates": summary["complete_states"],
    }
    for field, prefix in (
        ("counts", ""),
        ("raw_counts", "Raw"),
        ("applied_counts", "Applied"),
    ):
        values.update(
            {
                f"Ablation{prefix}{k.title().replace('_', '')}": v
                for k, v in summary[field].items()
            }
        )
    letters = {"42": "A", "43": "B", "44": "C"}
    for seed, draws in summary["per_seed"].items():
        for draw, arms in draws.items():
            for arm, outcomes in arms.items():
                for kind, counts in outcomes.items():
                    for outcome, count in counts.items():
                        values[
                            f"AblationSeed{letters[seed]}{draw.title()}{arm.title()}{kind.title()}{outcome.title().replace('_', '')}"
                        ] = count
    for draw, groups in contrast_summary(residual_ablation_v2).items():
        for group, counts in groups.items():
            suffix = "All" if group == "all_states" else f"Seed{letters[group]}"
            for outcome, count in counts.items():
                name = outcome.title().replace("_", "")
                values[f"Contrast{draw.title()}{suffix}{name}"] = count
    return "\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()
    )


def _geometry_receipts(root: Path, result: dict[str, Any]) -> dict[str, str]:
    receipts = {}
    for name, draw in result["drawsets"].items():
        if "factored_geometry" not in draw:
            continue
        artifact = draw["factored_geometry"]
        _require(
            name in _DRAWS and artifact["filename"] == f"factored_{name}.pt",
            "geometry filename",
        )
        checksum = _file_hash(root / artifact["filename"])
        _require(checksum == artifact["sha256"], "geometry artifact checksum")
        receipts[name] = checksum
    return receipts


def load_final_context(registration: Path, thresholds: Path) -> dict[str, Any]:
    """Verify bound original smoke JSON/geometry bytes once, without tensor loading."""
    raw = registration.read_bytes()
    final = json.loads(raw)
    _require(final["registration_phase"] == "final", "final registration required")
    binding = final["smoke_calibration"]
    _require(
        str(thresholds.resolve()) == binding["thresholds"]["path"],
        "exact accepted threshold path",
    )
    artifacts = {}
    for name, reference in binding.items():
        blob = Path(reference["path"]).read_bytes()
        _require(
            hashlib.sha256(blob).hexdigest() == reference["sha256"],
            f"frozen {name} checksum",
        )
        artifacts[name] = json.loads(blob)
    context = {
        "registration": final,
        "thresholds": artifacts.pop("thresholds"),
        "calibration": artifacts,
        "provenance": {
            "registration": {
                "path": str(registration.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "calibration": binding,
        },
    }
    context["calibration"]["geometry_sha256"] = _geometry_receipts(
        Path(binding["smoke_results"]["path"]).parent, artifacts["smoke_results"]
    )
    _validate_context(context)
    return context


def read_final_run(
    root: Path, context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one explicitly selected final run; verify geometry bytes, never tensors."""
    blobs = {
        name: (root / name).read_bytes() for name in ("results.json", "status.json")
    }
    result, status = (
        json.loads(blobs[name]) for name in ("results.json", "status.json")
    )
    geometry = _geometry_receipts(root, result)
    _validate_record(result, status, context, geometry, phase="final")
    state = {
        "seed": result["seed"],
        "step": result["step"],
        "status": result["status"],
        "reason": result.get("technical_error"),
        "result": result,
        "status_record": status,
        "geometry_sha256": geometry,
    }
    index = {
        "seed": state["seed"],
        "step": state["step"],
        "path": str(root.resolve()),
        "sha256": {
            name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()
        },
        "geometry_sha256": geometry,
    }
    return state, index


def export_ablation(
    registration: Path,
    thresholds: Path,
    runs: Mapping[tuple[int, int], Path],
    output: Path,
    *,
    protected_root: Path | None = None,
) -> dict[str, int]:
    """Aggregate exactly nine final runs into a new, portable scalar staging bundle."""
    _require(
        set(runs) == {(s, t) for s in _SEEDS for t in _STEPS}
        and len({p.resolve() for p in runs.values()}) == 9,
        "nine distinct explicit final run paths",
    )
    protected_paths = [*runs.values(), registration, thresholds]
    if protected_root is not None:
        protected_paths.append(protected_root)
    for protected in protected_paths:
        a, b = output.resolve(), protected.resolve()
        _require(
            not (a == b or a.is_relative_to(b) or b.is_relative_to(a)),
            "output/input overlap",
        )
    context = load_final_context(registration, thresholds)
    states, indices = [], []
    for identity, root in sorted(runs.items()):
        state, index = read_final_run(root, context)
        _require(
            (state["seed"], state["step"]) == identity, "selected final run identity"
        )
        states.append(state)
        indices.append(index)
    bundle = {
        **context,
        "schema": 1,
        "bundle": "residual_ablation_v2",
        "phase": "final",
        "states": states,
        "provenance": {**context["provenance"], "inputs": indices},
    }
    summary = ablation_summary(bundle)
    candidates = list(candidate_rows(bundle))
    encoded = {
        "results.json": _json_bytes(bundle),
        "summary.json": _json_bytes(summary),
        "candidates.jsonl": b"".join(
            (json.dumps(r, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
            for r in candidates
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in encoded.items():
        (output / name).write_bytes(blob)
    return {"states": 9, "drawsets": 18, "candidates": 486}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument(
        "--state", action="append", required=True, metavar="SEED:STEP=RUN_DIR"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new staging directory"
    )
    parser.add_argument("--protected-root", type=Path)
    args = parser.parse_args()
    runs = {}
    for value in args.state:
        identity, separator, path = value.partition("=")
        if not separator:
            parser.error("--state requires SEED:STEP=RUN_DIR")
        try:
            seed, step = (int(part) for part in identity.split(":"))
        except ValueError:
            parser.error("--state requires integer SEED:STEP")
        if (seed, step) in runs:
            parser.error("duplicate --state identity")
        runs[seed, step] = Path(path)
    print(
        json.dumps(
            export_ablation(
                args.registration,
                args.thresholds,
                runs,
                args.output_dir,
                protected_root=args.protected_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
