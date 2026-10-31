"""Retrospective, same-record scalar diagnostics; no model or trajectory replay.

The proposal comparison is between a base step and its *proposal*, never the
safeguarded step actually applied or a different method trajectory. Every raw
refresh record is retained, including records ineligible for gamma. Strict signs
refer to arithmetic on logged scalars, not certified signs of historical exact
quantities. The optional binary32 screens neither attest historical dtype nor
bound loss/AD/HVP error.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

import numpy as np

from .analysis import seed_number

_PROPOSAL_ARMS = ("kfac+overlay", "ekfac+overlay")
_DELTAS = ("delta_m", "delta_penalty", "delta_q", "delta_loss")
_TESTS = ("ratio", "model", "loss")
_LATE_STEPS = (200, 600, 1500)
_SEEDS = ("seed_42", "seed_43", "seed_44")

# Legacy K-FAC records omit damping; use their fixed protocol value rather than
# Exp7Config's current default. EKFAC damping must be present in each record.
_LEGACY_KFAC_DAMPING = 1.0

PROPOSAL_SIGN_SCREEN = (
    "16 * 2**(-23) * max(1, abs(m_proposal), abs(m_kfac), "
    "lambda/2 * norm_d0**2, lambda/2 * norm_d1**2); "
    "a binary32-scale sensitivity screen, not a certified arithmetic bound, "
    "and not evidence of historical dtype. Strict delta_q < 0 counts are unfiltered."
)
VALIDITY_RESOLUTION_SCREEN = (
    "abs(spacing(binary32(loss_theta))) is an illustrative subtraction-resolution "
    "scale at the rounded logged loss, not a certified arithmetic bound or "
    "evidence of historical dtype. Ratios use absolute signed margin or loss "
    "change divided by that scale; unavailable scales/overflowing ratios are null."
)


def _require(condition: bool, context: str) -> None:
    if not condition:
        raise ValueError(f"scalar diagnostics: {context}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context}: expected mapping")
    return value


def _field(record: dict[str, Any], key: str, context: str) -> Any:
    _require(key in record, f"{context}: missing {key}")
    return record[key]


def _number(value: Any, context: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context}: expected finite scalar",
    )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"scalar diagnostics: {context}: overflow") from exc
    _require(math.isfinite(number), f"{context}: expected finite scalar")
    _require(not nonnegative or number >= 0, f"{context}: must be nonnegative")
    return number


def _integer(value: Any, context: str, *, positive: bool = False) -> int:
    _require(
        type(value) is int and value >= (1 if positive else 0),
        f"{context}: expected {'positive' if positive else 'nonnegative'} integer",
    )
    return value


def _nonempty_list(value: Any, context: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{context}: empty/invalid list")
    return value


def _steps(records: list[Any], context: str) -> list[int]:
    steps = [
        _integer(_field(_mapping(r, context), "step", context), f"{context}/step")
        for r in records
    ]
    _require(
        all(b > a for a, b in zip(steps, steps[1:], strict=False)),
        f"{context}: duplicate or unordered step",
    )
    return steps


def _proposal_record(record: dict[str, Any], arm: str, seed: str) -> dict[str, Any]:
    context = f"exp7/{arm}/{seed}/step={record['step']}"
    values = {
        key: _number(
            _field(record, key, context),
            f"{context}/{key}",
            nonnegative=key.startswith("norm_"),
        )
        for key in ("m_kfac", "m_proposal", "norm_d0", "norm_d1", "loss_d0", "loss_d1")
    }
    accepted = _field(record, "accepted", context)
    verdict = _field(record, "rejected_by", context)
    _require(type(accepted) is bool, f"{context}/accepted: expected bool")
    _require(
        (accepted and verdict is None) or (not accepted and verdict in _TESTS),
        f"{context}/rejected_by: inconsistent/invalid verdict",
    )
    recorded_damping = "damping" in record
    _require(recorded_damping or arm == "kfac+overlay", f"{context}: missing damping")
    damping = _number(
        record["damping"] if recorded_damping else _LEGACY_KFAC_DAMPING,
        f"{context}/damping",
        nonnegative=True,
    )
    try:
        norm0_sq = _number(values["norm_d0"] ** 2, f"{context}/norm_d0 squared")
        norm1_sq = _number(values["norm_d1"] ** 2, f"{context}/norm_d1 squared")
    except OverflowError as exc:
        raise ValueError(
            f"scalar diagnostics: {context}: norm_d1/norm_d0 squared overflow"
        ) from exc
    delta_m = values["m_proposal"] - values["m_kfac"]
    delta_penalty = 0.5 * damping * (norm1_sq - norm0_sq)
    delta_q = delta_m - delta_penalty
    delta_loss = values["loss_d1"] - values["loss_d0"]
    deltas = dict(
        zip(_DELTAS, (delta_m, delta_penalty, delta_q, delta_loss), strict=True)
    )
    for key, value in deltas.items():
        _number(value, f"{context}/{key}")
    penalty0 = _number(0.5 * damping * norm0_sq, f"{context}/base penalty")
    penalty1 = _number(0.5 * damping * norm1_sq, f"{context}/proposal penalty")
    screen = (
        16
        * 2**-23
        * max(1.0, abs(values["m_proposal"]), abs(values["m_kfac"]), penalty0, penalty1)
    )
    return {
        "input_bundle": "exp7",
        "arm": arm,
        "seed": seed,
        "step": record["step"],
        "raw": dict(record),
        "accepted": accepted,
        "rejected_by": verdict,
        "damping": damping,
        "damping_source": (
            "record" if recorded_damping else "historical_fixed_protocol_not_recorded"
        ),
        **deltas,
        "binary32_sign_screen": screen,
        "borderline_q": abs(delta_q) <= screen,
    }


def _proposal_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [r for r in records if not r["accepted"]]
    loss = [r for r in rejected if r["rejected_by"] == "loss"]
    by_test = {test: sum(r["rejected_by"] == test for r in rejected) for test in _TESTS}
    accepted = sum(r["accepted"] for r in records)
    _require(accepted + len(rejected) == len(records), "proposal count partition")
    _require(
        sum(by_test.values()) == len(rejected), "proposal rejection count partition"
    )
    medians = {
        key: (
            _number(median(r[key] for r in loss), f"proposal median/{key}")
            if loss
            else None
        )
        for key in _DELTAS
    }
    return {
        "n_candidates": len(records),
        "n_accepted": accepted,
        "n_rejected": len(rejected),
        "n_rejected_lower_q": sum(r["delta_q"] < 0 for r in rejected),
        "n_rejected_by": by_test,
        "n_lower_m": sum(r["delta_m"] < 0 for r in records),
        "n_loss_rejected": len(loss),
        "n_loss_rejected_lower_q": sum(r["delta_q"] < 0 for r in loss),
        "n_loss_rejected_nonimproving_q": sum(r["delta_q"] >= 0 for r in loss),
        "n_loss_rejected_borderline_q": sum(r["borderline_q"] for r in loss),
        "loss_rejected_medians": medians,
    }


def proposal_decomposition(exp7: dict[str, Any] | None) -> dict[str, Any]:
    """Every same-record candidate pair, per-seed and pooled counts/medians.

    Denominators are refresh records, not all training-step safeguard proposals.
    The four legacy K-FAC non-improving cases remain individually accessible.
    Neither gamma eligibility, loss rejection nor a rounding screen filters records.
    """
    bundle = _mapping(exp7, "exp7")
    arms = {}
    for arm in _PROPOSAL_ARMS:
        method = _mapping(_field(bundle, arm, "exp7"), f"exp7/{arm}")
        seeds = _mapping(_field(method, "seeds", arm), f"exp7/{arm}/seeds")
        _require(
            bool(seeds) and all(isinstance(s, str) for s in seeds),
            f"exp7/{arm}: seed cohort",
        )
        records = []
        per_seed = {}
        for seed in sorted(seeds, key=seed_number):
            context = f"exp7/{arm}/{seed}"
            payload = _mapping(seeds[seed], context)
            raw = _nonempty_list(_field(payload, "model_decrease", context), context)
            _steps(raw, context)
            rows = [_proposal_record(r, arm, seed) for r in raw]
            per_seed[seed] = _proposal_counts(rows)
            records.extend(rows)
        arms[arm] = {
            "per_seed": per_seed,
            "pooled": _proposal_counts(records),
            "records": records,
            "loss_rejected_nonimproving_q": [
                r for r in records if r["rejected_by"] == "loss" and r["delta_q"] >= 0
            ],
        }
    return {
        "interpretation": "Retrospective, non-causal arithmetic on same-record logged scalars; no training-trajectory intervention or error certification.",
        "definitions": {
            "delta_m": "m_proposal - m_kfac (damped model)",
            "delta_penalty": "lambda/2 * (norm_d1**2 - norm_d0**2)",
            "delta_q": "delta_m - delta_penalty (undamped local quadratic, NOT native loss)",
            "delta_loss": "loss_d1 - loss_d0 (native minibatch loss)",
            "lower_q": "strict delta_q < 0, including screened/borderline records",
            "denominator": "all retained raw model_decrease refresh records, not gamma-eligible records or all optimiser steps",
            "legacy_damping": "K-FAC lambda=1 from the stated fixed historical protocol, absent in the raw records; not inferred from current defaults. Damping still affects candidate construction.",
            "binary32_sign_screen": PROPOSAL_SIGN_SCREEN,
        },
        "arms": arms,
    }


def complete_proposal_decomposition(exp7: dict[str, Any]) -> dict[str, Any]:
    """Require all historical refresh identities for the figure and flat export."""
    result = proposal_decomposition(exp7)
    expected = {"seed_42", "seed_43", "seed_44"}
    for arm, payload in result["arms"].items():
        _require(set(payload["per_seed"]) == expected, f"{arm}: seed cohort")
        _require(
            all(
                counts["n_candidates"] == 150 for counts in payload["per_seed"].values()
            )
            and len(payload["records"]) == 450,
            f"{arm}: expected 150 refreshes per seed and 450 records",
        )
    return result


def validity_prefix(
    curve: list[dict[str, Any]],
    lambda_norm_d0_sq: float,
    *,
    model_kind: str,
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """First failed point and longest valid *prefix*, not the last passing point.

    Recompute signed margins, without dividing by observed loss change: zero/zero
    succeeds and zero/nonzero fails. A producer's legitimate infinite rel_err is
    not consumed or rejected. Empty data is invalid; an undefined prefix means the
    first measured point failed, distinct from no failure on the measured grid.
    """
    _require(model_kind in ("damped", "undamped"), "validity/model_kind")
    penalty = _number(lambda_norm_d0_sq, "validity/lambda_norm_d0_sq", nonnegative=True)
    tolerance = _number(tolerance, "validity/tolerance", nonnegative=True)
    points = []
    extent = None
    failed = None
    previous_t = 0.0
    for raw in _nonempty_list(curve, "validity/curve_d0"):
        raw = _mapping(raw, "validity/curve_d0 point")
        t = _number(_field(raw, "t", "validity/curve_d0"), "validity/t")
        _require(
            t > previous_t, "validity/t: grid must be strictly increasing and positive"
        )
        previous_t = t
        dl = _number(_field(raw, "dL", "validity/curve_d0"), f"validity/t={t}/dL")
        model = _number(_field(raw, "m", "validity/curve_d0"), f"validity/t={t}/m")
        if model_kind == "undamped":
            model = _number(
                model - 0.5 * t * t * penalty, f"validity/t={t}/undamped model"
            )
        discrepancy = _number(abs(dl - model), f"validity/t={t}/discrepancy")
        margin = _number(
            discrepancy - tolerance * abs(dl), f"validity/t={t}/signed_margin"
        )
        point = {
            "t": t,
            "delta_loss": dl,
            "model": model,
            "discrepancy": discrepancy,
            "signed_margin": margin,
            "passes": margin <= 0,
        }
        points.append(point)
        if failed is None:
            if point["passes"]:
                extent = t
            else:
                failed = point
    return {
        "status": (
            "no_failure"
            if failed is None
            else "undefined_prefix"
            if extent is None
            else "failed_after_prefix"
        ),
        "prefix_extent": extent,
        "first_failure": failed,
        "first_grid_point": points[0],
        "decisive_point": failed if failed is not None else points[-1],
        "points": points,
    }


def _binary32_loss_spacing(loss: float) -> float | None:
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = abs(float(np.spacing(np.float32(loss))))
    return spacing if math.isfinite(spacing) and spacing > 0 else None


def _resolution_ratio(value: float, spacing: float | None) -> float | None:
    ratio = abs(value) / spacing if spacing is not None else math.inf
    return ratio if math.isfinite(ratio) else None


def _validity_meta(bundle: dict[str, Any], name: str) -> dict[str, Any]:
    meta = _mapping(_field(bundle, "meta", name), f"{name}/meta")
    for key in ("model", "dataset"):
        _require(
            isinstance(meta.get(key), str) and bool(meta[key]), f"{name}/meta/{key}"
        )
    _integer(_field(meta, "batch_size", name), f"{name}/batch_size", positive=True)
    _number(_field(meta, "damping", name), f"{name}/damping", nonnegative=True)
    _require(
        _number(_field(meta, "tolerance", name), f"{name}/tolerance") == 0.1,
        f"{name}/tolerance: expected 0.1",
    )
    grid = _nonempty_list(_field(meta, "t_grid", name), f"{name}/t_grid")
    previous = 0.0
    for t in grid:
        t = _number(t, f"{name}/t_grid")
        _require(
            t > previous, f"{name}/t_grid: expected strictly increasing positive grid"
        )
        previous = t
    _require(
        meta.get("step_source", "kfac" if name == "exp11" else None) == "kfac",
        f"{name}/step_source: expected kfac",
    )
    return meta


def validity_resolution(
    exp11: dict[str, Any] | None, exp11_kfac_seam: dict[str, Any] | None
) -> dict[str, Any]:
    """Both models on the original nine late cells and separate three-cell repeat.

    Identity is the input bundle/seed/checkpoint and its own pre-update minibatch;
    sample IDs are not retained. No early checkpoints or independent seeds are
    invented for the repeat. All grid points remain accessible per cell/model.
    """
    bundles = {"exp11": exp11, "exp11_kfac_seam": exp11_kfac_seam}
    cells = []
    reference = None
    for name, value in bundles.items():
        bundle = _mapping(value, name)
        meta = _validity_meta(bundle, name)
        protocol = {
            key: meta[key]
            for key in (
                "model",
                "dataset",
                "batch_size",
                "damping",
                "tolerance",
                "t_grid",
            )
        }
        _require(
            reference is None or protocol == reference,
            f"{name}: validity protocol differs from exp11",
        )
        reference = protocol
        seeds = _mapping(_field(bundle, "seeds", name), f"{name}/seeds")
        expected_seeds = _SEEDS if name == "exp11" else ("seed_42",)
        _require(set(seeds) == set(expected_seeds), f"{name}: seed cohort")
        for seed in expected_seeds:
            context = f"{name}/{seed}"
            payload = _mapping(seeds[seed], context)
            probes = _nonempty_list(
                _field(payload, "probes", context), f"{context}/probes"
            )
            steps = _steps(probes, context)
            _require(
                set(_LATE_STEPS).issubset(steps), f"{context}: missing late step cohort"
            )
            by_step = dict(zip(steps, probes, strict=True))
            for step in _LATE_STEPS:
                probe = by_step[step]
                cell_context = f"{context}/step={step}"
                curve = _nonempty_list(
                    _field(probe, "curve_d0", cell_context), cell_context
                )
                _require(
                    [_mapping(p, cell_context).get("t") for p in curve]
                    == meta["t_grid"],
                    f"{cell_context}: curve_d0 grid",
                )
                decomposition = _mapping(
                    _field(probe, "decomposition", cell_context), cell_context
                )
                penalty = _number(
                    _field(decomposition, "lambda_norm_d0_sq", cell_context),
                    f"{cell_context}/lambda_norm_d0_sq",
                    nonnegative=True,
                )
                payoff = _mapping(_field(probe, "payoff", cell_context), cell_context)
                loss = _number(
                    _field(payoff, "loss_theta", cell_context),
                    f"{cell_context}/loss_theta",
                )
                spacing = _binary32_loss_spacing(loss)
                for kind in ("damped", "undamped"):
                    prefix = validity_prefix(
                        curve, penalty, model_kind=kind, tolerance=meta["tolerance"]
                    )
                    for point in prefix["points"]:
                        point["absolute_delta_over_binary32_loss_spacing"] = (
                            _resolution_ratio(point["delta_loss"], spacing)
                        )
                        point["absolute_margin_over_binary32_loss_spacing"] = (
                            _resolution_ratio(point["signed_margin"], spacing)
                        )
                    cells.append(
                        {
                            "input_bundle": name,
                            "cohort": (
                                "original_late_kfac"
                                if name == "exp11"
                                else "kfac_seam_repeat"
                            ),
                            "input_record": f"{name}/seeds/{seed}/probes/step={step}",
                            "seed": seed,
                            "step": step,
                            "input": {
                                **protocol,
                                "sample_ids": None,
                                "batch_identity": "the record's own pre-update training minibatch; sample IDs not retained",
                            },
                            "model_kind": kind,
                            "loss_theta": loss,
                            "lambda_norm_d0_sq": penalty,
                            "binary32_loss_spacing": spacing,
                            **prefix,
                        }
                    )
    return {
        "interpretation": "Retrospective, non-causal grid-prefix diagnostic, not a continuous radius or a precision replay. Native loss change is distinct from either model.",
        "definitions": {
            "tolerance": 0.1,
            "late_steps": list(_LATE_STEPS),
            "undamped_model": "curve_d0.m - 0.5 * t**2 * lambda_norm_d0_sq",
            "signed_margin": "abs(delta_loss - model) - 0.1 * abs(delta_loss); <= 0 passes",
            "undefined_prefix": "the first grid point failed; prefix_extent is null",
            "no_failure": "all sampled grid points passed; first_failure is null",
            "binary32_loss_spacing": VALIDITY_RESOLUTION_SCREEN,
        },
        "cells": cells,
    }
