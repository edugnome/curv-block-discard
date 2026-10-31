"""Validate and summarise the reviewed finite check; no bootstrap or tensor access."""

from __future__ import annotations

import math
import re
from typing import Any

_STATISTICS = {
    "resamples": 10000,
    "seed": 20261020,
    "rng": "numpy.PCG64",
    "arithmetic": "float64 CPU",
    "resampling": "complete paired vectors; independent cells",
    "marginal": "percentile",
    "probabilities": [0.025, 0.975],
    "quantile": "linear at (n-1)*p",
    "sd_ddof": 1,
    "family_size": 36,
    "screen_level": 0.95,
    "screen": "max(abs(bootstrap-observed)/fixed bootstrap SD)",
    "jackknife": "diagnostic only; no bias correction",
    "invalid_family": "whole family unavailable",
}
_ROSTER = [
    (f"s{seed}-{model}-{batch}", seed, model, batch)
    for seed in (42, 43, 44)
    for model in ("resnet18", "plain_resnet18")
    for batch in ("B0", "B1")
]
_FAMILY = ("S_stage", "S_layer", "DeltaS")
_METRICS = ("S", "Phi", "Phi_sep")


def _require(condition: bool, context: str) -> None:
    if not condition:
        raise ValueError(f"crossing check: {context}")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _close(actual: Any, expected: float, context: str) -> None:
    _require(
        _number(actual)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        context,
    )


def _interval(value: Any, context: str) -> None:
    _require(
        isinstance(value, list)
        and len(value) == 2
        and all(_number(v) for v in value)
        and value[0] <= value[1],
        context,
    )


def _hash_record(record: dict[str, Any], context: str) -> None:
    _require(isinstance(record["path"], str) and bool(record["path"]), context)
    _require(
        isinstance(record["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
        context,
    )


def _marginal(record: dict[str, Any], context: str) -> None:
    _require(_number(record["estimate"]), f"{context}: estimate")
    _require(
        _number(record["bootstrap_mc_se"]) and record["bootstrap_mc_se"] >= 0,
        f"{context}: bootstrap SD",
    )
    _interval(record["percentile_95"], f"{context}: marginal interval")
    lo, hi = record["percentile_95"]
    _close(record["half_width"], (hi - lo) / 2, f"{context}: half width")
    _require(record["descriptive_half_width_target"] == 0.03, f"{context}: target")
    _require(
        type(record["target_met"]) is bool
        and record["target_met"] == (record["half_width"] <= 0.03),
        f"{context}: target flag (unmet is valid)",
    )


def _validate_receipts(bundle: dict[str, Any]) -> None:
    source, provenance = bundle["source_analysis"], bundle["provenance"]
    references = provenance["references"]
    for key in (
        "manifest",
        "analysis",
        "campaign",
        "numerical_review",
        "review_integrity",
        "review_statistics",
        "routing_clarification",
    ):
        _hash_record(references[key], f"provenance/{key}")
    for key in ("manifest", "campaign"):
        _require(
            source[f"{key}_sha256"] == references[key]["sha256"], f"{key} identity"
        )
    for key in ("scientific_source_sha256", "implementation_sha256"):
        _require(bool(provenance[key]), f"provenance/{key}")
        for path, sha256 in provenance[key].items():
            _hash_record({"path": path, "sha256": sha256}, f"provenance/{key}")
    verification = provenance["verification"]
    _require(
        verification["metadata"] == "PASS"
        and verification["independent_numerical_review"]
        == "PASS_FOR_NEW_CONDITIONAL_DIAGNOSTIC_EVIDENCE_WITH_REPORTING_CONDITIONS"
        and verification["required_scientific_corrections"] == []
        and bool(verification["reporting_conditions"]),
        "verification status",
    )
    _require(bool(provenance["limitations"]), "recorded limitations")
    metadata = source["metadata_verification"]
    _require(
        metadata["status"] == "PASS"
        and metadata["errors"] == []
        and metadata["cell_count"] == 12
        and metadata["checkpoint_count"] == 6
        and metadata["stream_count"] == 720,
        "source metadata verification",
    )
    for checkpoint in provenance["protocol"]["checkpoints"]:
        _hash_record(checkpoint, "checkpoint provenance")


def _validated(bundle: dict[str, Any]) -> dict[str, Any]:
    _require(bundle["schema"] == 1, "bundle schema")
    source = bundle["source_analysis"]
    public = bundle.get("format") == "curv-scientific-crossing-v1"
    if public:
        _require("provenance" not in bundle, "public envelope")
        protocol = bundle["protocol"]
    else:
        _validate_receipts(bundle)
        protocol = bundle["provenance"]["protocol"]
    _require(
        protocol["replicates"] == 30 and protocol["probes"] == 100, "30 by 100 budget"
    )
    _require(
        protocol["batches"]
        == [
            {"id": "B0", "indices": list(range(32))},
            {"id": "B1", "indices": list(range(32, 64))},
        ],
        "fixed disjoint input indices",
    )
    dataset, arithmetic, environment = (
        protocol["dataset"],
        protocol["arithmetic"],
        protocol["environment"],
    )
    _require(
        dataset["name"] == "CIFAR-100"
        and dataset["train"] is False
        and dataset["download"] is False
        and dataset["augmentation"] is False
        and dataset["batch_size"] == 32
        and dataset["target"] == "fine_labels"
        and dataset["mean"] == [0.5071, 0.4865, 0.4409]
        and dataset["std"] == [0.2673, 0.2564, 0.2762],
        "offline fixed-input protocol",
    )
    _require(
        all(
            arithmetic[key] == "torch.float32" for key in ("parameters", "input", "hvp")
        )
        and arithmetic["deterministic_algorithms"] is True
        and all(
            arithmetic[key] is False for key in ("autocast", "tf32", "cudnn_benchmark")
        )
        and arithmetic["loss"] == "eval mean cross_entropy"
        and environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        and environment["NVIDIA_TF32_OVERRIDE"] == "0",
        "recorded arithmetic protocol",
    )
    adaptation = protocol["adaptation"]
    _require(
        adaptation["enabled"] is True
        and adaptation["internal_retry_count"] == "UNAVAILABLE"
        and adaptation["internal_chunk_history"] == "UNAVAILABLE"
        and adaptation["exact_draw_replay_claimed"] is False,
        "adaptation limitations",
    )
    checkpoints = protocol["checkpoints"]
    _require(
        len(checkpoints) == 6
        and {(c["arch"], c["seed"]) for c in checkpoints}
        == {(model, seed) for _, seed, model, _ in _ROSTER},
        "six saved checkpoint roster",
    )
    for checkpoint in checkpoints:
        _require(checkpoint["epoch"] == 300, "checkpoint epoch")
    analysis = source["analysis"]
    _require(analysis["complete"] is True, "incomplete analysis")
    _require(
        analysis["statistics"] == protocol["statistics"] == _STATISTICS,
        "fixed bootstrap/family contract",
    )
    cells, family = analysis["cells"], analysis["family"]
    _require(
        [(c["id"], c["seed"], c["model"], c["batch"]) for c in cells] == _ROSTER,
        "fixed twelve-cell roster/order",
    )
    _require(
        family["status"] == "available"
        and family["metrics"]
        == [f"{cell}:{metric}" for cell, *_ in _ROSTER for metric in _FAMILY]
        and len(family["intervals"]) == len(family["fixed_sd"]) == 36,
        "complete 36-metric family",
    )
    critical = family["critical_value"]
    _require(_number(critical) and critical > 0, "family critical value")
    for index, cell in enumerate(cells):
        context = cell["id"]
        _require(
            cell["status"] == "complete"
            and cell["complete_pairs"] == 30
            and cell["reasons"] == [],
            f"{context}: 30 complete pairs",
        )
        for partition in ("stage", "layer"):
            pooled = cell["pooled_components"][partition]
            diag, full = pooled["diag_sq"], pooled["full_sq"]
            groups = (
                6
                if partition == "stage"
                else (22 if cell["model"] == "resnet18" else 19)
            )
            _require(
                isinstance(diag, list)
                and len(diag) == groups
                and all(_number(v) and v > 0 for v in diag)
                and _number(full)
                and full > 0,
                f"{context}/{partition}: pooled domain/group count",
            )
            diagonal = math.fsum(diag)
            separable = math.fsum(math.sqrt(v) for v in diag) ** 2
            expected = (separable / full, 1 - diagonal / full, 1 - diagonal / separable)
            for metric, point in zip(_METRICS, expected, strict=True):
                record = cell["partitions"][partition][metric]
                _marginal(record, f"{context}/{partition}/{metric}")
                _close(
                    record["estimate"],
                    point,
                    f"{context}/{partition}/{metric}: pooled estimate",
                )
        _marginal(cell["DeltaS"], f"{context}/DeltaS")
        points = [cell["partitions"][p]["S"]["estimate"] for p in ("stage", "layer")]
        _close(
            cell["DeltaS"]["estimate"],
            points[0] - points[1],
            f"{context}: paired DeltaS",
        )
        points.append(cell["DeltaS"]["estimate"])
        for offset, (metric, point) in enumerate(zip(_FAMILY, points, strict=True)):
            j = 3 * index + offset
            interval, sd = cell["screen_95"][metric], family["fixed_sd"][j]
            _interval(interval, f"{context}/{metric}: family endpoints")
            _interval(
                family["intervals"][j], f"{context}/{metric}: family registry endpoints"
            )
            _require(
                interval == family["intervals"][j],
                f"{context}/{metric}: endpoint registry",
            )
            _require(_number(sd) and sd > 0, f"{context}/{metric}: positive family SD")
            record = (
                cell["DeltaS"]
                if metric == "DeltaS"
                else cell["partitions"][metric.removeprefix("S_")]["S"]
            )
            _close(sd, record["bootstrap_mc_se"], f"{context}/{metric}: fixed SD")
            for endpoint, expected_endpoint in zip(
                interval, (point - critical * sd, point + critical * sd), strict=True
            ):
                _close(
                    endpoint, expected_endpoint, f"{context}/{metric}: centered screen"
                )
        stage, layer = cell["screen_95"]["S_stage"], cell["screen_95"]["S_layer"]
        classification = (
            "resolved_crossing"
            if stage[0] > 1 and layer[1] < 1
            else (
                "resolved_reverse_crossing"
                if stage[1] < 1 and layer[0] > 1
                else "unresolved"
            )
        )
        # Decisions use unrounded endpoints, never displayed decimals or DeltaS signs.
        _require(cell["crossing"] == classification, f"{context}: classification")
        for partition in ("stage", "layer", "DeltaS"):
            diagnostics = cell["jackknife"][partition]
            if "unavailable" in diagnostics:
                _require(
                    isinstance(diagnostics["unavailable"], str)
                    and bool(diagnostics["unavailable"]),
                    f"{context}: diagnostic reason",
                )
                continue
            for diagnostic in (
                [diagnostics]
                if partition == "DeltaS"
                else [diagnostics[m] for m in _METRICS]
            ):
                _require(
                    _number(diagnostic["bias"])
                    and _number(diagnostic["se"])
                    and diagnostic["se"] >= 0,
                    f"{context}: jackknife diagnostic",
                )
    comparisons = analysis["input_comparisons"]
    _require(
        [(c["seed"], c["model"]) for c in comparisons]
        == [(seed, model) for _, seed, model, batch in _ROSTER if batch == "B0"],
        "input-comparison roster",
    )
    for i, comparison in enumerate(comparisons):
        a, b = cells[2 * i : 2 * i + 2]
        for partition in ("stage", "layer"):
            _close(
                comparison["B1_minus_B0_S"][partition],
                b["partitions"][partition]["S"]["estimate"]
                - a["partitions"][partition]["S"]["estimate"],
                "input comparison S",
            )
        _close(
            comparison["B1_minus_B0_DeltaS"],
            b["DeltaS"]["estimate"] - a["DeltaS"]["estimate"],
            "input comparison DeltaS",
        )
    return analysis


def crossing_summary(exp8_crossing_check: dict[str, Any]) -> dict[str, Any]:
    """Consistency checks, not a coverage certificate; keep unresolved and contrary rows."""
    try:
        analysis = _validated(exp8_crossing_check)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        if str(exc).startswith("crossing check:"):
            raise
        raise ValueError(
            f"crossing check: missing/invalid retained evidence: {exc}"
        ) from exc
    cells = analysis["cells"]
    widths, unmet_s, unmet_family = [], [], []
    for cell in cells:
        for partition in ("stage", "layer"):
            record = cell["partitions"][partition]["S"]
            widths.append(record["half_width"])
            if not record["target_met"]:
                unmet_s.append(
                    {
                        "cell": cell["id"],
                        "metric": f"S_{partition}",
                        "half_width": record["half_width"],
                    }
                )
        if not cell["DeltaS"]["target_met"]:
            unmet_family.append(
                {
                    "cell": cell["id"],
                    "metric": "DeltaS",
                    "half_width": cell["DeltaS"]["half_width"],
                }
            )
    return {
        "cells": cells,
        "resolved": {
            model: sum(
                c["model"] == model and c["crossing"] == "resolved_crossing"
                for c in cells
            )
            for model in ("resnet18", "plain_resnet18")
        },
        "critical_value": analysis["family"]["critical_value"],
        "S_half_width_range": [min(widths), max(widths)],
        "unmet_S": unmet_s,
        "unmet_family": unmet_s + unmet_family,
    }
