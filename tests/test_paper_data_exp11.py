"""Validity extent and input-validation contracts."""

from __future__ import annotations

import math
from typing import Any

import pytest

from experiments.runner_exp11 import _aggregate  # noqa: E402
from paper_data.figures.exp11 import _far_shorter_from_step, _sharpness_value
from paper_data.tables.exp11 import tab_validity_arms

_T_GRID = [0.01, 0.1, 1.0]
_META: dict[str, Any] = {
    "model": "resnet50",
    "dataset": "stanford_cars",
    "pretrained": True,
    "batch_size": 64,
    "damping": 1.0,
    "lr": 1.0,
    "fisher_type": "mc",
    "kfac_refresh_period": 20,
    "probe_steps": [0, 60],
    "t_grid": _T_GRID,
    "kink_t_grid": [0.01, 1.0],
    "tolerance": 0.1,
    "n_lanczos_iter": 60,
    "seeds": [42],
}


def _probe(
    step: int,
    *,
    norm_d0: float,
    g_dot_d0: float,
    d0hd0: float,
    dl_1: float,
    lam_max: float,
    lam_min: float,
    t_star: float,
    kink_1: float,
) -> dict[str, Any]:
    """A probe record whose curve is the damped model of the given coefficients but for
    its t = 1 point, where the loss change is ``dl_1``."""
    lam_sq = _META["damping"] * norm_d0**2
    curve = []
    for t in _T_GRID:
        m = t * g_dot_d0 + 0.5 * t * t * (d0hd0 + lam_sq)
        dl = dl_1 if t == 1.0 else m
        curve.append(
            {
                "t": t,
                "dL": dl,
                "m": m,
                "secant": 2.0 * (dl - t * g_dot_d0) / (t * t),
                "rel_err": abs(dl - m) / abs(dl),
            }
        )
    eta_eff = norm_d0 / 5.0
    return {
        "step": step,
        "decomposition": {
            "g_dot_d0": g_dot_d0,
            "d0_H_d0": d0hd0,
            "lambda_norm_d0_sq": lam_sq,
            "m_d0": g_dot_d0 + 0.5 * (d0hd0 + lam_sq),
            "norm_d0": norm_d0,
            "norm_g": 5.0,
            "eta_eff": eta_eff,
            "eta_g": -g_dot_d0 / 25.0,
        },
        "curve_d0": curve,
        "t_star": t_star,
        "curve_vmax": curve,
        "t_star_vmax": t_star,
        "sharpness": {
            "sigma": lam_max,
            "lambda_max": lam_max,
            "lambda_min": lam_min,
            "curv_vmax": lam_max,
            "g_dot_vmax": -1.0,
            "cos_d0_vmax": 0.1,
            "curv_along_g": 28.0,
            "curv_along_d0": d0hd0 / norm_d0**2,
            "lambda_max_x_eta_eff": lam_max * eta_eff,
            "lambda_max_x_eta_g": lam_max * -g_dot_d0 / 25.0,
        },
        "kinks": {
            "n_relu": 1000,
            "n_pool": 100,
            "rows": [
                {"t": 0.01, "relu_flips": 1, "relu_frac": 0.001, "pool_changes": 0, "pool_frac": 0.0},
                {"t": 1.0, "relu_flips": int(1000 * kink_1), "relu_frac": kink_1, "pool_changes": 6, "pool_frac": 0.06},
            ],
        },
        "payoff": {
            "m_kfac": g_dot_d0 + 0.5 * (d0hd0 + lam_sq),
            "loss_theta": 5.0,
            "loss_d0": 5.0 + dl_1,
            "rho": -dl_1 / -(g_dot_d0 + 0.5 * (d0hd0 + lam_sq)),
            "norm_d0": norm_d0,
        },
    }  # fmt: skip


def _seed() -> dict[str, Any]:
    return {
        "probes": [
            _probe(0, norm_d0=2.0, g_dot_d0=-5.0, d0hd0=50.0, dl_1=-1.4, lam_max=280.0, lam_min=-30.0, t_star=0.01, kink_1=0.0432),
            _probe(60, norm_d0=4.0, g_dot_d0=-20.0, d0hd0=800.0, dl_1=-1.8, lam_max=920.3, lam_min=-4.5, t_star=math.nan, kink_1=0.0365),
        ],
        "loss_trajectory": [5.0] * 61,
        "val_acc": [0.1],
        "n_steps": 61,
    }  # fmt: skip


def _bundle() -> dict[str, Any]:
    seeds = {"seed_42": _seed()}
    return {"meta": dict(_META), "seeds": seeds, "aggregate": _aggregate(seeds)}


# ---------------------------------------------------------------------------
# tab:validity-arms
# ---------------------------------------------------------------------------

# The six probe steps of the b1 protocol (`_DEFAULT_STEPS` in `tables/exp11.py`).
_ARM_STEPS: tuple[int, ...] = (0, 20, 60, 200, 600, 1500)


def _arm_seed_payload(t_stars: list[float]) -> dict[str, Any]:
    """One seed of one arm: a probe per step of `_ARM_STEPS`, holding only what
    `tab_validity_arms` and `runner_exp11._aggregate` need from a probe record -
    `step` (a coordinate, kept verbatim by `_aggregate`) and `t_star` (read raw, per
    seed, by `tab_validity_arms`; reduced to a `{mean, std}` no one here reads)."""
    return {
        "probes": [
            {"step": step, "t_star": t_star}
            for step, t_star in zip(_ARM_STEPS, t_stars, strict=True)
        ]
    }


def _arm_bundle(
    *, step_source: str | None = None, **t_stars_by_seed: list[float]
) -> dict[str, Any]:
    """Aggregate one arm with an optional step-source identity."""
    seeds = {
        seed: _arm_seed_payload(t_stars) for seed, t_stars in t_stars_by_seed.items()
    }
    meta = dict(_META) if step_source is None else {**_META, "step_source": step_source}
    return {"meta": meta, "seeds": seeds, "aggregate": _aggregate(seeds)}


def _arms_bundles() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Three arms on three seeds and the same probe steps."""
    kfac = _arm_bundle(
        seed_42=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        seed_43=[0.01, 0.05, 0.05, 0.05, 0.05, 0.05],
        seed_44=[0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
    )
    sgd = _arm_bundle(
        step_source="sgd",
        seed_42=[0.05] * 6,
        seed_43=[0.05] * 6,
        seed_44=[0.05] * 6,
    )
    adam = _arm_bundle(
        step_source="adam",
        seed_42=[0.05] * 6,
        seed_43=[0.05] * 6,
        seed_44=[0.05] * 6,
    )
    return kfac, sgd, adam


@pytest.mark.parametrize(
    ("late_extent", "expected"),
    [(0.1, True), (0.25, False), (math.nan, False), (math.inf, False)],
)
def test_late_extent_guard_checks_every_seed(
    late_extent: float, expected: bool
) -> None:
    bundle = _arm_bundle(
        seed_42=[1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
        seed_43=[1.0, 0.5, 0.25, late_extent, 0.05, 0.01],
    )
    assert _far_shorter_from_step(bundle) is expected


def test_secant_curvature_averages_normalised_seed_values() -> None:
    seeds = {
        "seed_42": {
            "probes": [
                {
                    "decomposition": {"norm_d0": 1.0},
                    "curve_d0": [{"t": 1.0, "secant": 7.0}],
                }
            ]
        },
        "seed_43": {
            "probes": [
                {
                    "decomposition": {"norm_d0": 3.0},
                    "curve_d0": [{"t": 1.0, "secant": 9.0}],
                }
            ]
        },
    }
    assert _sharpness_value({}, "secant", seeds, 0) == pytest.approx(4.0)


def test_late_extent_guard_requires_late_probes() -> None:
    assert _far_shorter_from_step(_bundle()) is False


def test_tab_validity_arms_refuses_a_bundle_that_is_not_its_columns_arm() -> None:
    """Recorded step sources must match their respective arms."""
    kfac, sgd, adam = _arms_bundles()
    tab_validity_arms(kfac, sgd, adam)
    explicit = {**kfac, "meta": {**kfac["meta"], "step_source": "kfac"}}
    tab_validity_arms(explicit, sgd, adam)

    with pytest.raises(ValueError):
        tab_validity_arms(kfac, adam, adam)
    with pytest.raises(ValueError):
        tab_validity_arms(kfac, sgd, sgd)
    with pytest.raises(ValueError):
        tab_validity_arms(sgd, sgd, adam)
    unrecorded = {
        **sgd,
        "meta": {k: v for k, v in sgd["meta"].items() if k != "step_source"},
    }
    with pytest.raises(ValueError):
        tab_validity_arms(kfac, unrecorded, adam)


def test_tab_validity_arms_refuses_arms_run_under_different_protocols() -> None:
    """Arms with mismatched protocol fields are rejected."""
    kfac, sgd, adam = _arms_bundles()
    for field, other in (("damping", 0.1), ("lr", 0.5), ("batch_size", 32)):
        changed = {**adam, "meta": {**adam["meta"], field: other}}
        with pytest.raises(ValueError):
            tab_validity_arms(kfac, sgd, changed)


def test_tab_validity_arms_missing_seed_raises() -> None:
    kfac, sgd, adam = _arms_bundles()
    del sgd["seeds"]["seed_44"]
    with pytest.raises(ValueError):
        tab_validity_arms(kfac, sgd, adam)


def test_tab_validity_arms_step_mismatch_raises() -> None:
    kfac, sgd, adam = _arms_bundles()
    adam["aggregate"]["probes"][0]["step"] = 1
    with pytest.raises(ValueError):
        tab_validity_arms(kfac, sgd, adam)


def test_tab_validity_arms_missing_bundle_raises() -> None:
    """`exp11` (unlike exp11_sgd/exp11_adam) stays in `OPTIONAL_EXPERIMENTS`, so
    `generate_all` can still pass it through as None; tab_validity_arms has no
    placeholder form, so that must be a `ValueError` naming the absent bundle."""
    _, sgd, adam = _arms_bundles()
    with pytest.raises(ValueError):
        tab_validity_arms(None, sgd, adam)
