"""Retrospective diagnostics use every same-record pair and exact late cohorts."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

_ARMS = ("kfac+overlay", "ekfac+overlay")


def _record(**changes: Any) -> dict[str, Any]:
    row = {
        "step": 0,
        "damping": 2.0,
        "m_kfac": -3.0,
        "m_proposal": -4.0,
        "norm_d0": 2.0,
        "norm_d1": 1.0,
        "loss_d0": 1.0,
        "loss_d1": 1.25,
        "accepted": False,
        "rejected_by": "loss",
        # Deliberately gamma-ineligible, but still a same-record candidate pair.
        "m_newton": -2.0,
    }
    row.update(changes)
    return row


def _proposals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        arm: {"seeds": {"seed_42": {"model_decrease": copy.deepcopy(rows)}}}
        for arm in _ARMS
    }


def test_same_record_decomposition_does_not_filter_on_gamma_or_modify_input():
    from paper_data.diagnostics import proposal_decomposition

    rows = [_record(), _record(step=20, accepted=True, rejected_by=None)]
    data = _proposals(rows)
    original = copy.deepcopy(data)
    result = proposal_decomposition(data)
    assert data == original
    for arm in _ARMS:
        records = result["arms"][arm]["records"]
        assert len(records) == 2
        row = records[0]
        assert row["raw"] == rows[0]
        assert (row["arm"], row["seed"], row["step"]) == (arm, "seed_42", 0)
        assert row["delta_m"] == -1.0
        assert row["delta_penalty"] == -3.0
        assert row["delta_q"] == 2.0
        assert row["delta_loss"] == 0.25
        counts = result["arms"][arm]["pooled"]
        assert counts["n_candidates"] == 2
        assert counts["n_accepted"] == counts["n_rejected"] == 1
        assert counts["n_loss_rejected"] == 1
        assert counts["n_loss_rejected_lower_q"] == 0


def test_strict_q_sign_is_not_replaced_by_binary32_screen():
    from paper_data.diagnostics import proposal_decomposition

    rows = [
        _record(m_kfac=0.0, m_proposal=-1e-12, norm_d0=1.0, norm_d1=1.0),
        _record(step=20, m_kfac=0.0, m_proposal=0.0, norm_d0=1.0, norm_d1=1.0),
        _record(step=40, m_kfac=0.0, m_proposal=1e-12, norm_d0=1.0, norm_d1=1.0),
    ]
    arm = proposal_decomposition(_proposals(rows))["arms"][_ARMS[0]]
    assert arm["pooled"]["n_loss_rejected_lower_q"] == 1
    assert arm["pooled"]["n_loss_rejected_nonimproving_q"] == 2
    assert arm["pooled"]["n_loss_rejected_borderline_q"] == 3
    assert all(r["binary32_sign_screen"] == 16 * 2**-23 for r in arm["records"])


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"m_proposal": math.nan}, "m_proposal"),
        ({"loss_d1": math.inf}, "loss_d1"),
        ({"norm_d0": -1.0}, "norm_d0"),
        ({"norm_d1": True}, "norm_d1"),
        ({"norm_d1": 1e308}, "norm_d1|penalty"),
        ({"damping": -1.0}, "damping"),
        ({"accepted": 1}, "accepted"),
        ({"accepted": True}, "rejected_by"),
        ({"rejected_by": "unknown"}, "rejected_by"),
        ({"step": 0.5}, "step"),
    ],
)
def test_proposal_schema_and_finite_derived_domains(changes, match):
    from paper_data.diagnostics import proposal_decomposition

    with pytest.raises(ValueError):
        proposal_decomposition(_proposals([_record(**changes)]))


@pytest.mark.parametrize("kind", ("missing", "duplicate", "empty", "ekfac_damping"))
def test_proposal_missing_fields_and_record_counts(kind):
    from paper_data.diagnostics import proposal_decomposition

    rows = [_record()]
    if kind == "missing":
        rows[0].pop("loss_d1")
    elif kind == "duplicate":
        rows.append(_record())
    elif kind == "empty":
        rows.clear()
    else:
        rows[0].pop("damping")
    with pytest.raises(ValueError):
        proposal_decomposition(_proposals(rows))


def test_all_verdict_counts_and_empty_loss_reject_medians_are_defined():
    from paper_data.diagnostics import proposal_decomposition

    rows = [
        _record(accepted=True, rejected_by=None),
        _record(step=20, rejected_by="ratio"),
        _record(step=40, rejected_by="model"),
    ]
    result = proposal_decomposition(_proposals(rows))["arms"][_ARMS[0]]["pooled"]
    assert result["n_rejected_by"] == {"ratio": 1, "model": 1, "loss": 0}
    assert result["n_loss_rejected_lower_q"] == 0
    assert all(value is None for value in result["loss_rejected_medians"].values())


def _curve(*pairs: tuple[float, float]) -> list[dict[str, float]]:
    return [
        {"t": t, "dL": dl, "m": m}
        for t, (dl, m) in zip((0.01, 0.1, 1.0), pairs, strict=False)
    ]


def test_prefix_uses_first_failure_not_last_success_and_separates_models():
    from paper_data.diagnostics import validity_prefix

    curve = _curve((-1.0, -1.0), (-1.0, -0.8), (-1.0, -1.0))
    result = validity_prefix(curve, 0.0, model_kind="damped")
    assert result["status"] == "failed_after_prefix"
    assert result["prefix_extent"] == 0.01
    assert result["first_failure"]["t"] == 0.1
    assert result["first_failure"]["signed_margin"] == pytest.approx(0.1)
    assert result["points"][-1]["passes"] is True
    curve = _curve((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0))
    damped = validity_prefix(curve, 100.0, model_kind="damped")
    undamped = validity_prefix(curve, 100.0, model_kind="undamped")
    assert damped["status"] == "no_failure"
    assert damped["prefix_extent"] == 1.0
    assert damped["first_failure"] is None
    assert undamped["first_failure"]["t"] == 0.1
    assert undamped["first_failure"]["model"] == -1.5


def test_zero_zero_succeeds_zero_loss_nonzero_model_has_undefined_prefix():
    from paper_data.diagnostics import validity_prefix

    zero = validity_prefix(_curve((0.0, 0.0)), 0.0, model_kind="damped")
    assert zero["status"] == "no_failure" and zero["prefix_extent"] == 0.01
    curve = _curve((0.0, 1e-8), (-1.0, -1.0))
    curve[0]["rel_err"] = math.inf  # Legitimate producer value, not an invalid scalar.
    fail = validity_prefix(curve, 0.0, model_kind="damped")
    assert fail["status"] == "undefined_prefix"
    assert fail["prefix_extent"] is None
    assert fail["first_failure"]["signed_margin"] == 1e-8
    # The inclusive boundary is evaluated by signed margin, not rounded relative error.
    boundary = validity_prefix(_curve((10.0, 9.0)), 0.0, model_kind="damped")
    assert boundary["points"][0]["signed_margin"] == 0.0
    assert boundary["status"] == "no_failure"


@pytest.mark.parametrize(
    "kind",
    (
        "empty",
        "duplicate",
        "unsorted",
        "nan",
        "inf",
        "zero_t",
        "negative_penalty",
        "bad_model",
        "overflow",
    ),
)
def test_prefix_rejects_invalid_consumed_domains(kind):
    from paper_data.diagnostics import validity_prefix

    curve = _curve((-1.0, -1.0), (-1.0, -1.0))
    penalty = 0.0
    model = "damped"
    if kind == "empty":
        curve.clear()
    elif kind == "duplicate":
        curve[1]["t"] = curve[0]["t"]
    elif kind == "unsorted":
        curve.reverse()
    elif kind == "nan":
        curve[0]["dL"] = math.nan
    elif kind == "inf":
        curve[0]["m"] = math.inf
    elif kind == "zero_t":
        curve[0]["t"] = 0.0
    elif kind == "negative_penalty":
        penalty = -1.0
    elif kind == "bad_model":
        model = "other"
    else:
        curve[0].update(dL=1e308, m=-1e308)
    with pytest.raises(ValueError):
        validity_prefix(curve, penalty, model_kind=model)


@pytest.mark.parametrize(
    "field", ("m_kfac", "m_proposal", "norm_d0", "norm_d1", "loss_d0", "loss_d1")
)
@pytest.mark.parametrize("invalid", ("missing", "nan", "text", "null"))
def test_every_new_proposal_scalar_has_a_schema_and_finite_check(field, invalid):
    from paper_data.diagnostics import proposal_decomposition

    row = _record()
    if invalid == "missing":
        row.pop(field)
    else:
        row[field] = {"nan": math.nan, "text": "1.0", "null": None}[invalid]
    with pytest.raises(ValueError):
        proposal_decomposition(_proposals([row]))


def test_nonfinite_derived_median_fails_validation():
    from paper_data.diagnostics import proposal_decomposition

    rows = [
        _record(
            step=step,
            m_kfac=-1e308,
            m_proposal=0.0,
            norm_d0=0.0,
            norm_d1=0.0,
            loss_d0=0.0,
            loss_d1=1e308,
        )
        for step in (0, 20)
    ]
    with pytest.raises(ValueError):
        proposal_decomposition(_proposals(rows))


def test_infinite_relative_quotient_does_not_invalidate_finite_margin():
    from paper_data.diagnostics import validity_prefix

    curve = _curve((1e-320, 1.0))
    curve[0]["rel_err"] = math.inf
    result = validity_prefix(curve, 0.0, model_kind="damped")
    assert result["status"] == "undefined_prefix"
    assert result["first_failure"]["signed_margin"] == 1.0
    json.dumps(result, allow_nan=False)


def _full_proposals():
    rows = [
        _record(
            step=20 * i,
            m_kfac=0.0,
            m_proposal=(-1e-12, 0.0, 1.0)[i % 3],
            m_overlay=999.0,
            norm_d0=1.0,
            norm_d1=1.0,
            loss_d0=1.0,
            loss_d1=(1.25, 1.0, 0.5)[i % 3],
            accepted=i % 2 == 0,
            rejected_by=None if i % 2 == 0 else "loss",
        )
        for i in range(150)
    ]
    return {
        arm: {
            "seeds": {
                f"seed_{seed}": {"model_decrease": copy.deepcopy(rows)}
                for seed in (42, 43, 44)
            }
        }
        for arm in _ARMS
    }


def test_proposal_display_transform_preserves_signed_zeros_and_small_values():
    from paper_data.figures.exp789 import _proposal_scale

    values = [-100.0, -1.0, -1e-12, 0.0, 1e-12, 1.0, 100.0]
    transformed = [_proposal_scale(value) for value in values]
    assert transformed == sorted(transformed)
    assert transformed[3] == 0
    assert transformed[2] < 0 < transformed[4]
    assert transformed[0] == -transformed[-1]


def test_proposal_figure_rejects_incomplete_cohort():
    from paper_data.figures.exp789 import fig_proposal_model_loss

    with pytest.raises(ValueError):
        fig_proposal_model_loss(_proposals([_record()]))
