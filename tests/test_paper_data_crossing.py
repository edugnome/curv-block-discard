"""Render crossing-check outcomes while preserving the other paper outputs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from paper_data.config import DEFAULT_RESULTS_DIR, DataPaths


@pytest.fixture
def crossing() -> dict[str, Any]:
    from experiments.crossing.measure import ADAPTATION
    from experiments.crossing.verify import manifest_contract, roster

    protocol = manifest_contract(
        "/synthetic/data",
        [
            {
                "arch": arch,
                "seed": seed,
                "epoch": 300,
                "path": f"synthetic-{seed}-{arch}",
                "sha256": "0" * 64,
            }
            for seed in (42, 43, 44)
            for arch in ("resnet18", "plain_resnet18")
        ],
    )
    protocol["adaptation"] = copy.deepcopy(ADAPTATION)
    cells = []
    intervals: list[list[float]] = []
    fixed_sd = []

    def marginal(point):
        return {
            "estimate": point,
            "bootstrap_mc_se": 0.01,
            "percentile_95": [point - 0.02, point + 0.02],
            "half_width": 0.02,
            "descriptive_half_width_target": 0.03,
            "target_met": True,
        }

    for c in roster():
        partitions, pooled = {}, {}
        for name, count, point in (
            ("stage", 6, 1.2),
            ("layer", 22 if c["model"] == "resnet18" else 19, 0.8),
        ):
            diag = [1.0] * count
            full = count * count / point
            pooled[name] = {"diag_sq": diag, "full_sq": full}
            partitions[name] = {
                "S": marginal(point),
                "Phi": marginal(1 - count / full),
                "Phi_sep": marginal(1 - 1 / count),
            }
        screen = {
            key: [value - 0.03, value + 0.03]
            for key, value in (("S_stage", 1.2), ("S_layer", 0.8), ("DeltaS", 0.4))
        }
        intervals.extend(screen.values())
        fixed_sd.extend([0.01] * 3)
        cells.append(
            dict(
                c,
                status="complete",
                complete_pairs=30,
                reasons=[],
                crossing="resolved_crossing",
                partitions=partitions,
                pooled_components=pooled,
                DeltaS=marginal(0.4),
                screen_95=screen,
                jackknife={
                    p: {"unavailable": "Synthetic renderer fixture"}
                    for p in ("stage", "layer", "DeltaS")
                },
            )
        )
    analysis = {
        "complete": True,
        "statistics": protocol["statistics"],
        "cells": cells,
        "family": {
            "status": "available",
            "metrics": [
                f"{c['id']}:{key}"
                for c in roster()
                for key in ("S_stage", "S_layer", "DeltaS")
            ],
            "intervals": intervals,
            "fixed_sd": fixed_sd,
            "critical_value": 3.0,
        },
        "input_comparisons": [
            {
                "seed": seed,
                "model": model,
                "B1_minus_B0_S": {"stage": 0.0, "layer": 0.0},
                "B1_minus_B0_DeltaS": 0.0,
            }
            for seed in (42, 43, 44)
            for model in ("resnet18", "plain_resnet18")
        ],
    }
    references = {
        key: {"path": f"synthetic-{key}", "sha256": "0" * 64}
        for key in (
            "manifest",
            "analysis",
            "campaign",
            "numerical_review",
            "review_integrity",
            "review_statistics",
            "routing_clarification",
        )
    }
    return {
        "schema": 1,
        "source_analysis": {
            "manifest_sha256": "0" * 64,
            "campaign_sha256": "0" * 64,
            "metadata_verification": {
                "status": "PASS",
                "errors": [],
                "cell_count": 12,
                "checkpoint_count": 6,
                "stream_count": 720,
            },
            "analysis": analysis,
        },
        "provenance": {
            "references": references,
            "scientific_source_sha256": {"synthetic": "0" * 64},
            "implementation_sha256": {"synthetic": "0" * 64},
            "verification": {
                "metadata": "PASS",
                "independent_numerical_review": "PASS_FOR_NEW_CONDITIONAL_DIAGNOSTIC_EVIDENCE_WITH_REPORTING_CONDITIONS",
                "required_scientific_corrections": [],
                "reporting_conditions": ["Synthetic fixture, not measured evidence"],
            },
            "protocol": protocol,
            "limitations": ["Synthetic fixture only"],
        },
    }


def _analysis(bundle: dict[str, Any]) -> dict[str, Any]:
    return bundle["source_analysis"]["analysis"]


def test_synthetic_summary_preserves_all_cells_and_widths(crossing):
    from paper_data.crossing import crossing_summary

    original = copy.deepcopy(crossing)
    summary = crossing_summary(crossing)
    assert crossing == original
    assert len(summary["cells"]) == 12
    assert summary["resolved"] == {"resnet18": 6, "plain_resnet18": 6}
    assert summary["unmet_S"] == []
    assert summary["unmet_family"] == []
    assert summary["S_half_width_range"] == pytest.approx([0.02, 0.02])
    assert summary["critical_value"] == 3.0


_INVALID = [
    "missing_provenance",
    "missing_hash",
    "bad_hash",
    "verification",
    "budget",
    "batch_indices",
    "checkpoint_epoch",
    "checkpoint_roster",
    "incomplete",
    "missing_cell",
    "duplicate_cell",
    "wrong_seed",
    "missing_metric",
    "nonfinite",
    "bad_component",
    "wrong_group_count",
    "point_inconsistent",
    "missing_endpoint",
    "nonfinite_endpoint",
    "reversed_endpoint",
    "wrong_endpoint",
    "wrong_family_roster",
    "missing_family_sd",
    "zero_family_sd",
    "classification",
    "width_flag",
    "bootstrap_contract",
    "missing_jackknife",
    "comparison",
    "arithmetic",
    "dataset",
    "adaptation",
    "missing_protocol",
]


def _invalidate(bundle: dict[str, Any], case: str) -> None:
    a = _analysis(bundle)
    cell = a["cells"][0]
    p = bundle["provenance"]
    if case == "missing_provenance":
        del bundle["provenance"]
    elif case == "missing_hash":
        del p["references"]["analysis"]["sha256"]
    elif case == "bad_hash":
        p["references"]["manifest"]["sha256"] = "z" * 64
    elif case == "verification":
        p["verification"]["independent_numerical_review"] = "pending"
    elif case == "budget":
        p["protocol"]["probes"] = 99
    elif case == "batch_indices":
        p["protocol"]["batches"][1]["indices"][0] = 0
    elif case == "checkpoint_epoch":
        p["protocol"]["checkpoints"][0]["epoch"] = 299
    elif case == "checkpoint_roster":
        p["protocol"]["checkpoints"].pop()
    elif case == "incomplete":
        cell["complete_pairs"] = 29
    elif case == "missing_cell":
        a["cells"].pop()
    elif case == "duplicate_cell":
        a["cells"][1] = copy.deepcopy(cell)
    elif case == "wrong_seed":
        cell["seed"] = 99
    elif case == "missing_metric":
        del cell["partitions"]["layer"]["Phi_sep"]
    elif case == "nonfinite":
        cell["partitions"]["layer"]["S"]["estimate"] = math.nan
    elif case == "bad_component":
        cell["pooled_components"]["stage"]["full_sq"] = 0
    elif case == "wrong_group_count":
        cell["pooled_components"]["layer"]["diag_sq"].pop()
    elif case == "point_inconsistent":
        cell["DeltaS"]["estimate"] += 0.1
    elif case == "missing_endpoint":
        del cell["screen_95"]["DeltaS"]
    elif case == "nonfinite_endpoint":
        cell["screen_95"]["S_layer"][1] = math.inf
    elif case == "reversed_endpoint":
        cell["screen_95"]["S_layer"].reverse()
    elif case == "wrong_endpoint":
        cell["screen_95"]["S_layer"][1] += 0.1
    elif case == "wrong_family_roster":
        a["family"]["metrics"].pop()
    elif case == "missing_family_sd":
        a["family"]["fixed_sd"].pop()
    elif case == "zero_family_sd":
        a["family"]["fixed_sd"][0] = 0
    elif case == "classification":
        cell["crossing"] = "unresolved"
    elif case == "width_flag":
        cell["partitions"]["stage"]["S"]["target_met"] = False
    elif case == "bootstrap_contract":
        a["statistics"]["resamples"] = 9999
    elif case == "missing_jackknife":
        del cell["jackknife"]["DeltaS"]
    elif case == "arithmetic":
        p["protocol"]["arithmetic"]["tf32"] = True
    elif case == "dataset":
        p["protocol"]["dataset"]["train"] = True
    elif case == "adaptation":
        p["protocol"]["adaptation"]["exact_draw_replay_claimed"] = True
    elif case == "missing_protocol":
        del p["protocol"]["arithmetic"]
    elif case == "comparison":
        a["input_comparisons"][0]["B1_minus_B0_DeltaS"] += 0.1
    else:
        raise AssertionError(case)


def _preflight_paths(root: Path, override: Path) -> DataPaths:
    """Isolate the first detailed crossing gate with non-observational JSON stubs."""
    from paper_data.manifests import track_inputs

    for name in track_inputs("b"):
        if name == "exp8_crossing_check":
            continue
        path = root / name / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return DataPaths(root, {"exp8_crossing_check": override})


@pytest.mark.parametrize("case", _INVALID)
def test_invalid_crossing_fails_before_any_output_mkdir(
    crossing, tmp_path, monkeypatch, case
):
    from paper_data.main import generate_all

    _invalidate(crossing, case)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(crossing))
    paths = _preflight_paths(tmp_path / "results", invalid)
    output = tmp_path / "never-created"
    calls = []
    real_mkdir = Path.mkdir

    def record_mkdir(self, *args, **kwargs):
        calls.append(self)
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    with pytest.raises(ValueError):
        generate_all(paths, output, only=["tab:crossing-check"])
    assert not output.exists()
    assert output not in calls


def test_missing_crossing_fails_before_any_output(crossing, tmp_path):
    from paper_data.main import generate_all

    paths = _preflight_paths(tmp_path / "results", tmp_path / "missing.json")
    output = tmp_path / "never-created"
    with pytest.raises((FileNotFoundError, ValueError)):
        generate_all(paths, output, only=["tab:crossing-check"])
    assert not output.exists()


def _set_points(bundle, stage, layer):
    """Synthetic scalar counterexample, never a retained scientific record."""
    a = _analysis(bundle)
    cell = a["cells"][0]
    for partition, point in (("stage", stage), ("layer", layer)):
        pooled = cell["pooled_components"][partition]
        diag = pooled["diag_sq"]
        pooled["full_sq"] = math.fsum(math.sqrt(v) for v in diag) ** 2 / point
        points = {"S": point, "Phi": 1 - math.fsum(diag) / pooled["full_sq"]}
        for metric, value in points.items():
            record = cell["partitions"][partition][metric]
            shift = value - record["estimate"]
            record["estimate"] = value
            record["percentile_95"] = [v + shift for v in record["percentile_95"]]
            record["half_width"] = (
                record["percentile_95"][1] - record["percentile_95"][0]
            ) / 2
    record = cell["DeltaS"]
    shift = stage - layer - record["estimate"]
    record["estimate"] = stage - layer
    record["percentile_95"] = [v + shift for v in record["percentile_95"]]
    record["half_width"] = (record["percentile_95"][1] - record["percentile_95"][0]) / 2
    for i, metric in enumerate(("S_stage", "S_layer", "DeltaS")):
        value = (stage, layer, stage - layer)[i]
        half = a["family"]["critical_value"] * a["family"]["fixed_sd"][i]
        interval = [value - half, value + half]
        cell["screen_95"][metric] = interval
        a["family"]["intervals"][i] = interval.copy()
    other = a["cells"][1]
    comparison = a["input_comparisons"][0]
    comparison["B1_minus_B0_S"] = {
        p: other["partitions"][p]["S"]["estimate"]
        - cell["partitions"][p]["S"]["estimate"]
        for p in ("stage", "layer")
    }
    comparison["B1_minus_B0_DeltaS"] = other["DeltaS"]["estimate"] - record["estimate"]


def test_honest_reverse_crossing_is_retained(crossing):
    from paper_data.crossing import crossing_summary

    _set_points(crossing, 0.8, 1.2)
    _analysis(crossing)["cells"][0]["crossing"] = "resolved_reverse_crossing"
    assert (
        crossing_summary(crossing)["cells"][0]["crossing"]
        == "resolved_reverse_crossing"
    )


def test_classification_uses_unrounded_strict_endpoints(crossing):
    from paper_data.crossing import crossing_summary

    family = _analysis(crossing)["family"]
    stage = 1 + family["critical_value"] * family["fixed_sd"][0]
    _set_points(crossing, stage, 0.8)
    cell = _analysis(crossing)["cells"][0]
    assert cell["screen_95"]["S_stage"][0] == 1
    cell["crossing"] = "unresolved"
    assert crossing_summary(crossing)["cells"][0]["crossing"] == "unresolved"
    _set_points(crossing, stage + 1e-9, 0.8)
    cell["crossing"] = "resolved_crossing"
    assert crossing_summary(crossing)["cells"][0]["crossing"] == "resolved_crossing"


@pytest.mark.parametrize("label", ["tab:crossing-check", "num:crossing-check"])
@pytest.mark.parametrize("missing", [False, True])
def test_selected_crossing_fails_before_output_without_a_track(
    crossing, tmp_path, missing, label
):
    from paper_data.main import generate_all

    path = tmp_path / "crossing.json"
    if not missing:
        _invalidate(crossing, "classification")
        path.write_text(json.dumps(crossing))
    output = tmp_path / "no-output"
    paths = DataPaths(DEFAULT_RESULTS_DIR, {"exp8_crossing_check": path})
    with pytest.raises((ValueError, FileNotFoundError)):
        generate_all(paths, output, only=[label])
    assert not output.exists()


def test_new_renderer_reads_no_files_and_imports_no_scientific_stack(
    crossing, monkeypatch
):
    import builtins

    from paper_data.tables.crossing import num_crossing_check, tab_crossing_check

    real_import = builtins.__import__

    def no_scientific_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "numpy", "science"}
        return real_import(name, *args, **kwargs)

    def no_file_access(*args, **kwargs):
        raise AssertionError("renderer must use its retained scalar argument only")

    monkeypatch.setattr(builtins, "__import__", no_scientific_import)
    monkeypatch.setattr(Path, "open", no_file_access)
    tab_crossing_check(crossing)
    num_crossing_check(crossing)


def test_standalone_crossing_figure_preflights_before_output(crossing, tmp_path):
    from paper_data.main import generate_all

    _invalidate(crossing, "missing_family_sd")
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(crossing))
    target = tmp_path / "missing-output"
    with pytest.raises(ValueError):
        generate_all(
            _preflight_paths(tmp_path / "results", invalid),
            target,
            only=["fig:crossing-screen"],
        )
    assert not target.exists()
