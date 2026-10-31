"""The paper-only causal-pilot export: JSON in, no checkpoints or GPU needed."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from paper_data import causal_pilot


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _candidate(gain: float, base: float = 0.0) -> dict[str, Any]:
    return {"loss_d0": base, "loss": base - gain, "actual_gain": gain}


def _raw(seed: int, step: int) -> dict[str, Any]:
    candidates = {
        "original": _candidate(-0.5),
        "gradient_refresh": _candidate(0.5),
        "curvature_refresh": _candidate(-1.0),
        "both_refresh": _candidate(0.0),
        **{f"scale_{i}": _candidate(0.0) for i in range(7)},
    }
    candidates["scale_6"] = _candidate(-0.5)
    candidates["original"]["original_gates"] = {
        "ratio": True,
        "model": True,
        "loss": False,
        "accepted": False,
        "rejected_by": "loss",
    }
    comparison = {
        "actual_gain": -0.5,
        "predicted_endpoint_slope": -2.0,
        "endpoint_slope": 1.0,
    }
    return {
        "seed": seed,
        "step": step,
        "capture_sha256": f"{seed * 1000 + step:064x}",
        "tests": {"architecture_intervention": "complete", "refresh": "complete"},
        "original_model_replay": {"passed": True},
        "production_acceptance_check": {"passed": True},
        "thresholds": _thresholds(),
        "native": {
            "candidates": candidates,
            "best_scale_gain": 0.0,
            "scale_control": [
                {
                    "alpha": alpha,
                    "candidate": f"scale_{i}",
                    "gain": candidates[f"scale_{i}"]["actual_gain"],
                }
                for i, alpha in enumerate((0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1))
            ],
            "comparison": comparison,
        },
        "interventions": {
            # Both losses rise, but the within-surrogate ranking flips.
            "relu": {
                "measurements": {"candidates": {"original": _candidate(0.5, 2.0)}}
            },
            "pool": {"measurements": {"candidates": {"original": _candidate(-0.5)}}},
        },
    }


def _thresholds() -> dict[str, Any]:
    return {
        "schema": 1,
        "gain_floor": 0.0001,
        "tolerances": {"gain_floor": 0.0001, "atol": 1e-6},
        "calibration_capture_sha256": "a" * 64,
        "repeatability_error": 0.0,
        "repeats": {"d0": [0.0, 0.0, 0.0]},
        "role": "fixed_from_smoke",
        "this_repeatability_error": 0.0,
        "this_repeats": {"d0": [0.0, 0.0, 0.0]},
    }


def _reference(data: dict[str, Any], path: str) -> dict[str, Any]:
    original = data["native"]["candidates"]["original"]
    return {
        "seed": data["seed"],
        "step": data["step"],
        "path": path,
        "capture_sha256": data["capture_sha256"],
        "original_gain": original["actual_gain"],
        "original_gates": original["original_gates"],
        "best_scale_gain": 0.0,
        "comparison": data["native"]["comparison"],
        "refresh": {
            name: {"gain": gain, "extra_vs_scale": gain}
            for name, gain in (
                ("gradient_refresh", 0.5),
                ("curvature_refresh", -1.0),
                ("both_refresh", 0.0),
            )
        },
        "interventions": {
            "relu": {
                "gain": 0.5,
                "surrogate_base_loss_shift": 2.0,
                "surrogate_candidate_loss_shift": 1.0,
            },
            "pool": {
                "gain": -0.5,
                "surrogate_base_loss_shift": 0.0,
                "surrogate_candidate_loss_shift": 0.0,
            },
        },
    }


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "causal_pilot_run"
    refs = []
    for seed in (42, 43, 44):
        for step in (20, 60, 200):
            data = _raw(seed, step)
            rel = f"science-v2/probes/s{seed}_t{step}_main"
            directory = root / rel
            _write(directory / "results.json", data)
            _write(directory / "thresholds.json", data["thresholds"])
            _write(directory / "status.json", {"status": "complete"})
            _write(
                directory / "provenance.json",
                {
                    "source": {"sha256": "b" * 64},
                    "capture_manifest": {
                        "seed": seed,
                        "step": step,
                        "sha256": data["capture_sha256"],
                        "provenance": {"source": {"sha256": "b" * 64}},
                    },
                },
            )
            refs.append(_reference(data, f"results/{root.name}/{rel}"))
    _write(
        root / "verification/v2-complete-cohort-summary.json",
        {
            "complete_fixed_cohort": True,
            "n_primary": 9,
            "missing_primary": [],
            "source_sha256": ["b" * 64],
            "rows": refs,
        },
    )
    return root


def _change(path: Path, change: Any) -> None:
    data = json.loads(path.read_text())
    change(data)
    _write(path, data)


def test_export_is_derived_from_primary_json_with_hashed_provenance(
    run_root: Path,
) -> None:
    bundle = causal_pilot.export_cohort(run_root)
    assert len(bundle["rows"]) == 9
    provenance = bundle["provenance"]
    assert provenance["source_sha256"] == "b" * 64
    assert len(provenance["inputs"]) == 9
    first = provenance["inputs"][0]
    assert set(first["sha256"]) == {
        "results.json",
        "status.json",
        "thresholds.json",
        "provenance.json",
    }
    path = run_root / first["path"] / "results.json"
    assert (
        first["sha256"]["results.json"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["rows"].pop(),
        lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])),
        lambda d: d["rows"][0].update(seed=45),
    ],
)
def test_export_rejects_bad_primary_cohorts(run_root: Path, mutation: Any) -> None:
    _change(run_root / "verification/v2-complete-cohort-summary.json", mutation)
    with pytest.raises(ValueError):
        causal_pilot.export_cohort(run_root)


@pytest.mark.parametrize(
    "file, mutation",
    [
        ("status.json", lambda d: d.update(status="incomplete")),
        ("provenance.json", lambda d: d["source"].update(sha256="c" * 64)),
        ("provenance.json", lambda d: d["capture_manifest"].update(seed=43)),
        (
            "results.json",
            lambda d: d["native"]["candidates"]["gradient_refresh"].update(
                actual_gain=0.6
            ),
        ),
        ("results.json", lambda d: d["native"].update(best_scale_gain=1.0)),
        (
            "results.json",
            lambda d: d["native"]["comparison"].update(endpoint_slope=math.nan),
        ),
        (
            "results.json",
            lambda d: d["native"]["candidates"]["original"]["original_gates"].update(
                loss=True
            ),
        ),
    ],
)
def test_export_rejects_inconsistent_primary_artifacts(
    run_root: Path, file: str, mutation: Any
) -> None:
    _change(run_root / "science-v2/probes/s42_t20_main" / file, mutation)
    with pytest.raises(ValueError):
        causal_pilot.export_cohort(run_root)


@pytest.mark.parametrize(
    "field, value",
    [
        ("gain_floor", 0.001),
        ("calibration_capture_sha256", "d" * 64),
        ("tolerances", {"gain_floor": 0.0001, "atol": 1e-5}),
    ],
)
def test_export_rejects_mixed_frozen_thresholds(
    run_root: Path, field: str, value: Any
) -> None:
    directory = run_root / "science-v2/probes/s43_t60_main"
    _change(directory / "thresholds.json", lambda d: d.update({field: value}))
    _change(
        directory / "results.json", lambda d: d["thresholds"].update({field: value})
    )
    with pytest.raises(ValueError):
        causal_pilot.export_cohort(run_root)


def test_export_checks_embedded_thresholds_against_the_sidecar(run_root: Path) -> None:
    path = run_root / "science-v2/probes/s43_t60_main/thresholds.json"
    _change(path, lambda d: d.update(gain_floor=0.001))
    with pytest.raises(ValueError):
        causal_pilot.export_cohort(run_root)


def test_export_does_not_trust_a_stale_verified_summary(run_root: Path) -> None:
    _change(
        run_root / "verification/v2-complete-cohort-summary.json",
        lambda d: d["rows"][0].update(original_gain=1.0),
    )
    with pytest.raises(ValueError):
        causal_pilot.export_cohort(run_root)


def test_only_the_verified_primary_paths_are_observations(run_root: Path) -> None:
    # Repeats and v1 can have the same state; neither is in the primary-path inventory.
    _write(run_root / "science-v2/probes/s42_t60_repeat/results.json", _raw(42, 60))
    _write(run_root / "science/probes/s42_t60_main/results.json", _raw(42, 60))
    assert len(causal_pilot.export_cohort(run_root)["rows"]) == 9


def test_render_revalidates_gains_and_complete_cohort(run_root: Path) -> None:
    bundle = causal_pilot.export_cohort(run_root)
    bundle["rows"][0]["interventions"]["pool"]["actual_gain"] = 7.0
    with pytest.raises(ValueError):
        causal_pilot.causal_pilot_table(bundle)
    bundle = causal_pilot.export_cohort(run_root)
    bundle["rows"].pop()
    with pytest.raises(ValueError):
        causal_pilot.causal_pilot_numbers(bundle)


def test_reconstruct_summary_is_honest_scalar_packaging(run_root: Path) -> None:
    path = run_root / causal_pilot._SUMMARY
    reference = json.loads(path.read_text())
    path.unlink()
    primary = sorted(
        Path("science-v2/probes").joinpath(p.name)
        for p in (run_root / "science-v2/probes").iterdir()
    )
    summary = causal_pilot.reconstruct_summary(run_root, primary)
    for key in (
        "complete_fixed_cohort",
        "n_primary",
        "missing_primary",
        "source_sha256",
    ):
        assert summary[key] == reference[key]
    assert summary["packaging"]["kind"] == "scalar_reconstruction"
    assert summary["packaging"]["independent_gpu_verification"] is False
    assert len(summary["packaging"]["inputs"]) == 9
    _write(path, summary)
    bundle = causal_pilot.export_cohort(run_root)
    assert len(bundle["rows"]) == 9
    assert bundle["provenance"]["summary"]["kind"] == "scalar_reconstruction"
    assert bundle["provenance"]["summary"]["independent_gpu_verification"] is False


def test_reconstruct_summary_requires_explicit_complete_primary_selection(
    run_root: Path,
) -> None:
    primary = sorted(
        Path("science-v2/probes").joinpath(p.name)
        for p in (run_root / "science-v2/probes").iterdir()
    )
    with pytest.raises(ValueError):
        causal_pilot.reconstruct_summary(run_root, primary[:-1])
    with pytest.raises(ValueError):
        causal_pilot.reconstruct_summary(run_root, primary + primary[:1])
    with pytest.raises(ValueError):
        causal_pilot.reconstruct_summary(run_root, [Path("../outside")])


@pytest.mark.parametrize(
    ("gain", "expected"),
    [(-0.00011, -1), (-0.0001, 0), (0.0, 0), (0.0001, 0), (0.00011, 1)],
)
def test_rank_uses_a_closed_gain_floor(gain: float, expected: int) -> None:
    assert causal_pilot._rank(gain, 0.0001) == expected
