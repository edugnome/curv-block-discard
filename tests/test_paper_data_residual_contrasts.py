"""Retrospective scalar contrasts from recorded candidate losses only."""

import math

import pytest

from paper_data.residual_ablation import contrast_values


def candidate(loss, baseline=10.0, repeats=None, baselines=None):
    return {
        "raw_status": "finite",
        "loss": loss,
        "loss_d0": baseline,
        "loss_repeats": repeats or [loss] * 3,
        "loss_d0_repeats": baselines or [baseline] * 3,
    }


def test_additive_quadratic_and_strict_signs():
    # L(d)=10+d^2, off=1, rest=2, full=3: J=2*1*2.
    result = contrast_values(
        {"off": candidate(11), "rest": candidate(14), "full": candidate(19)}
    )
    assert (result["x"], result["y"], result["j"]) == (1, 5, 4)
    additive = contrast_values(
        {"off": candidate(11), "rest": candidate(12), "full": candidate(13)}
    )
    assert (additive["x"], additive["y"], additive["j"]) == (1, 1, 0)
    reversal = contrast_values(
        {"off": candidate(11), "rest": candidate(12), "full": candidate(9)}
    )
    assert reversal["x"] > 0 and reversal["y"] < 0 and reversal["j"] < 0


def test_repeat_envelopes_keep_shared_baseline_coefficient():
    b = [10, 11, 12]
    result = contrast_values(
        {
            "off": candidate(14, 10, [14, 15, 16], b),
            "rest": candidate(20, 10, [20, 22, 24], b),
            "full": candidate(25, 10, [25, 27, 29], b),
        }
    )
    assert result["envelopes"] == {"x": [2, 6], "y": [1, 9], "j": [-5, 7]}


def test_zero_alpha_is_not_forced_to_zero():
    result = contrast_values(
        {"off": candidate(10.01), "rest": candidate(10.02), "full": candidate(10.03)}
    )
    assert result["x"] != 0 and result["y"] != 0


def test_unequal_baseline_and_nonfinite_are_not_silently_cancelled():
    with pytest.raises(ValueError):
        contrast_values(
            {"off": candidate(11, 9), "rest": candidate(12), "full": candidate(13)}
        )
    with pytest.raises(ValueError):
        contrast_values(
            {"off": candidate(math.nan), "rest": candidate(12), "full": candidate(13)}
        )


def test_unavailable_is_not_a_zero_and_matching_has_no_j():
    failed = candidate(11)
    failed["raw_status"] = "technical_failure"
    result = contrast_values(
        {"off": failed, "rest": candidate(12), "full": candidate(13)}
    )
    assert result["status"] == "technical_unavailable" and result["x"] is None
    matched = contrast_values(
        {"rest": candidate(12), "full": candidate(11)}, matched=True
    )
    assert matched["y"] == -1 and "j" not in matched and "x" not in matched


def test_repeat_baseline_disagreement_nonfinite_and_missing_are_distinct():
    valid = {"off": candidate(11), "rest": candidate(12), "full": candidate(13)}
    valid["off"]["loss_d0_repeats"] = [10, 10, 10.1]
    with pytest.raises(ValueError):
        contrast_values(valid)
    valid["off"]["loss_d0_repeats"] = [10, 10, 10]
    valid["off"]["loss_repeats"] = [11, math.inf, 11]
    with pytest.raises(ValueError):
        contrast_values(valid)
    del valid["off"]
    assert contrast_values(valid)["status"] == "technical_unavailable"


def test_matching_reversal_and_persistence_count_once(monkeypatch):
    from paper_data import residual_ablation as module

    scales = [
        {
            "type": "scale",
            "alpha": 1,
            "drawset": "primary",
            "seed": 42,
            "step": step,
            "status": "finite",
            "x": 1.0,
            "y": -1.0,
            "j": -2.0,
            "delta_full": 1.0,
            "delta_rest": 2.0,
        }
        for step in (20, 60)
    ]
    matched = [
        {
            "type": "matched_full_rest",
            "drawset": "primary",
            "seed": 42,
            "step": step,
            "status": "finite",
            "y": y,
        }
        for step, y in ((20, 0.1), (60, -0.1))
    ]
    monkeypatch.setattr(
        module, "residual_contrast_rows", lambda _: iter(scales + matched)
    )
    counts = module.contrast_summary({})["primary"]["42"]
    assert counts["full_over_rest"] == 2
    assert counts["matched_reversal"] == counts["matched_persists"] == 1
    assert counts["preference_full_harm"] == 2
