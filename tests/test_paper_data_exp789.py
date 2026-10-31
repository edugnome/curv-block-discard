"""Numerical contracts for exp7–9 analyses and scientific gates."""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import pytest
from scipy import stats

from experiments.runner_exp7 import aggregate_method  # noqa: E402
from experiments.runner_exp8 import aggregate as _aggregate_exp8  # noqa: E402
from experiments.runner_exp9 import aggregate as _aggregate_exp9  # noqa: E402
from paper_data.analysis import (  # noqa: E402
    _phi_estimate,
    _spearman_rows,
    accuracy_matched_delta,
    accuracy_matched_delta_phi,
    coupling_eps_by_pair,
    coupling_eps_records,
    coupling_field_values,
    eps_variance_decomposition,
    gamma_records,
    gap_closed,
    gate_a,
    gate_a_replicated,
    gate_b,
    gate_c,
    gate_d,
    negative_curvature_share,
    pair_role,
    reject_breakdown,
    same_label_delta_phi,
    spearman_coupling_eps,
    validity_share,
)
from paper_data.tables.exp789 import tab_phi_long


# Retrospective reporting contracts.
def _sensitivity_method() -> dict[str, Any]:
    rng = np.random.default_rng(8102)
    groups = ["a", "b", "c", "d"]
    records = []
    for refresh in range(3):
        field = rng.normal(size=(4, 4))
        field = (field + field.T) / 2
        row, col = rng.normal(size=(2, 4)) * (refresh + 2)
        scale = rng.normal(size=4) * 3
        pairs = {}
        for i, v in enumerate(groups):
            for j, w in enumerate(groups):
                if i != j:
                    pairs[f"{v}->{w}"] = {
                        "coupling": math.exp(field[i, j] + scale[i] + scale[j]),
                        "eps": math.exp(
                            0.6 * field[i, j] + row[i] + col[j] + rng.normal()
                        ),
                    }
        records.append({"step": refresh * 20, "pairs": pairs})
    return {
        "seeds": {
            "seed_42": {
                "cross_term": records,
                "model_decrease": [_record(1.0, 1.0, 2.0, r * 20) for r in range(3)],
            }
        }
    }


def _sensitivity_reference(records):
    # Independent treatment-coded OLS reference, matching the frozen recheck gauge.
    # Symmetric C residuals can have numerical ties: a different projection gauge
    # can perturb their rank order at roundoff despite identical fitted values.
    pooled: list[list[float]] = [[], []]
    record_rhos = []
    for record in records:
        keys = list(record["pairs"])
        groups = sorted({g for k in keys for g in k.split("->")})
        design = np.array(
            [
                [1]
                + [int(k.split("->")[0] == g) for g in groups[1:]]
                + [int(k.split("->")[1] == g) for g in groups[1:]]
                for k in keys
            ],
            dtype=float,
        )
        residuals = []
        for field in ("coupling", "eps"):
            y = np.array([math.log(record["pairs"][k][field]) for k in keys])
            residuals.append(y - design @ np.linalg.lstsq(design, y, rcond=None)[0])
        record_rhos.append(stats.spearmanr(*residuals).statistic)
        for destination, values in zip(pooled, residuals, strict=True):
            destination.extend(values)
    return stats.spearmanr(*pooled).statistic, np.mean(record_rhos)


def test_retrospective_adjusts_both_logs_per_refresh_and_pools_within_seed():
    from paper_data.analysis import coupling_sensitivity

    method = _sensitivity_method()
    payload = method["seeds"]["seed_42"]
    expected, mean_rho = _sensitivity_reference(payload["cross_term"])
    assert abs(expected - mean_rho) > 0.01, "pool ranks, not per-refresh rhos"
    keys = list(payload["cross_term"][0]["pairs"])
    groups = sorted({g for k in keys for g in k.split("->")})
    design = np.array(
        [
            [1]
            + [int(k.split("->")[0] == g) for g in groups[1:]]
            + [int(k.split("->")[1] == g) for g in groups[1:]]
            for k in keys
        ]
        * 3,
        dtype=float,
    )
    lc, le = [
        np.array(
            [math.log(r["pairs"][k][f]) for r in payload["cross_term"] for k in keys]
        )
        for f in ("coupling", "eps")
    ]
    pooled_c = lc - design @ np.linalg.lstsq(design, lc, rcond=None)[0]
    pooled_e = le - design @ np.linalg.lstsq(design, le, rcond=None)[0]
    assert abs(expected - stats.spearmanr(pooled_c, pooled_e).statistic) > 0.1
    assert abs(expected - stats.spearmanr(lc, pooled_e).statistic) > 0.1

    result = coupling_sensitivity(method)["seed_42"]
    assert gamma_records(payload)[1] == 3  # all reference gaps ineligible
    assert result["adjusted"]["rho"] == pytest.approx(expected, abs=1e-12)
    assert result["raw"]["n_pairs"] == result["adjusted"]["n_pairs"] == 36
    assert result["symmetric"]["n_pairs"] == 18
    assert result["raw"]["n_records"] == result["adjusted"]["n_records"] == 3
    assert result["adjusted"]["n_skipped"] == 0
    rows = [x for r in payload["cross_term"] for x in r["pairs"].values()]
    assert result["raw"]["rho"] == pytest.approx(
        stats.spearmanr(
            [x["coupling"] for x in rows], [x["eps"] for x in rows]
        ).statistic
    )
    import copy

    second = copy.deepcopy(payload)
    for record in second["cross_term"]:
        for values in record["pairs"].values():
            values["eps"] = 1 / values["eps"]
    method["seeds"]["seed_43"] = second
    both = coupling_sensitivity(method)
    assert both["seed_42"] == result
    assert both["seed_43"]["adjusted"]["rho"] == pytest.approx(-expected, abs=1e-12)


def test_retrospective_domains_missing_reverse_zero_negative_and_nonfinite():
    from paper_data.analysis import coupling_sensitivity

    observations = [
        ("a->b", 2, 0),
        ("b->a", 2, 4),
        ("a->c", 3, 9),
        ("c->a", 3, 16),
        ("b->c", 4, -1),
        ("c->b", 4, 25),
        ("a->d", 5, 2),
        ("b->d", 6, math.nan),
        ("d->b", 6, 36),
        ("c->d", math.nan, 49),
        ("d->c", 7, 64),
    ]
    pairs = {key: {"coupling": c, "eps": e} for key, c, e in observations}
    method = {"seeds": {"seed_42": {"cross_term": [{"step": 0, "pairs": pairs}]}}}
    row = coupling_sensitivity(method)["seed_42"]
    assert (row["raw"]["n_pairs"], row["raw"]["n_skipped"]) == (9, 2)
    assert (row["adjusted"]["n_pairs"], row["adjusted"]["n_skipped"]) == (7, 4)
    symmetric = row["symmetric"]
    assert symmetric["n_candidates"] == 6
    assert symmetric["n_pairs"] == 2  # sqrt(0 * 4) = 0 is defined
    assert symmetric["n_missing_reverse"] == 1
    assert symmetric["n_skipped"] == 4
    assert math.isnan(symmetric["rho"])  # fewer than 3 observations
    reversed_method = {
        "seeds": {
            "seed_42": {
                "cross_term": [
                    {"step": 0, "pairs": dict(reversed(list(pairs.items())))}
                ]
            }
        }
    }
    other = coupling_sensitivity(reversed_method)["seed_42"]
    for domain in ("raw", "symmetric", "adjusted"):
        assert other[domain]["n_pairs"] == row[domain]["n_pairs"]
    assert other["raw"]["rho"] == row["raw"]["rho"]


def test_retrospective_symmetric_target_and_reverse_consistency():
    from paper_data.analysis import coupling_sensitivity

    pairs = {
        "a->b": {"coupling": 1.0, "eps": 1.0},
        "b->a": {"coupling": 1.0, "eps": 4.0},
        "a->c": {"coupling": 2.0, "eps": 9.0},
        "c->a": {"coupling": 2.0, "eps": 1.0},
        "b->c": {"coupling": 3.0, "eps": 1.0},
        "c->b": {"coupling": 3.0, "eps": 1.0},
    }
    method = {"seeds": {"seed_42": {"cross_term": [{"pairs": pairs}]}}}
    row = coupling_sensitivity(method)["seed_42"]
    assert row["symmetric"]["rho"] == pytest.approx(
        stats.spearmanr([1, 2, 3], [2, 3, 1]).statistic
    )
    pairs["b->a"]["coupling"] = 4.0
    with pytest.raises(ValueError):
        coupling_sensitivity(method)


def test_retrospective_constant_and_empty_domains_are_undefined_without_warnings():
    from paper_data.analysis import coupling_sensitivity

    pairs = {
        f"{v}->{w}": {"coupling": 1.0, "eps": 2.0}
        for v in "abc"
        for w in "abc"
        if v != w
    }
    method = {
        "seeds": {
            "seed_42": {"cross_term": [{"pairs": pairs}]},
            "seed_43": {"cross_term": []},
        }
    }
    for row in coupling_sensitivity(method).values():
        for domain in ("raw", "symmetric", "adjusted"):
            assert math.isnan(row[domain]["rho"])
    constant = coupling_sensitivity(method)["seed_42"]
    assert constant["adjusted"]["n_pairs"] == 6
    assert constant["adjusted"]["n_skipped"] == 0


# Bootstrap replicates for the hand-built checks (the paper default is 10000).
_B = 400
_STEPS = (0, 5, 10)


# ======================================================================
# Hand-built bundles
# ======================================================================


def _record(m_kfac: float, m_overlay: float, m_newton: float, step: int) -> dict:
    return {
        "step": step,
        "m_kfac": m_kfac,
        "m_overlay": m_overlay,
        "m_newton": m_newton,
        "norm_d0": 1.0,
        "norm_d1": 1.0,
        "accepted": True,
        "n_pairs": 1,
        "n_hvp_newton": 5,
    }


def _cross(step: int, couplings: list[float], epsilons: list[float]) -> dict:
    return {
        "step": step,
        "pairs": {
            f"b{i}->b{i + 1}": {"eps": e, "coupling": c}
            for i, (c, e) in enumerate(zip(couplings, epsilons, strict=True))
        },
    }


def _payload(
    *,
    records: list[dict] | None = None,
    cross: list[dict] | None = None,
    val_acc: tuple[float, ...] = (0.5, 0.8),
    val_loss: tuple[float, ...] = (1.0, 0.5),
    epochs_to_target: int | None = 2,
    n_steps: int = 100,
    n_refresh_failed: int = 0,
    n_proposed: int = 4,
    n_reject_by: dict[str, int] | None = None,
    newton_cg: tuple[int, int] = (10, 4),
) -> dict[str, Any]:
    """One exp7 seed payload: 4 proposals, 1 model rejection and 10 CG solves,
    including 4 stopped on negative curvature.
    """
    reject_by = (
        {"ratio": 0, "model": 1, "loss": 0} if n_reject_by is None else n_reject_by
    )
    return {
        "val_acc": list(val_acc),
        "val_loss": list(val_loss),
        "final_acc": val_acc[-1],
        "epochs_to_target": epochs_to_target,
        "n_steps": n_steps,
        "hessian_batch_size": 8,
        "wall_sec_per_epoch": [1.0, 1.0],
        "hvp_per_epoch": [10, 10],
        "safeguard": {
            "accept_rate_per_epoch": [0.5, 1.0],
            "n_proposed": n_proposed,
            "n_reject": sum(reject_by.values()),
            "n_reject_by": reject_by,
            "n_refresh_failed": n_refresh_failed,
        },
        "newton_cg": {"n_solves": newton_cg[0], "n_negative_curvature": newton_cg[1]},
        "model_decrease": records or [],
        "cross_term": cross or [],
        "phi": {
            c: {"phi": 0.3, "diag_frob": {"a": 1.0}, "coupling": {"a->b": 0.1}}
            for c in ("init", "mid", "final")
        },
    }


def _method(seeds: dict[str, dict[str, Any]], target: float = 0.7) -> dict[str, Any]:
    return aggregate_method(seeds, target)


_COUPLINGS = [0.1 * i for i in range(1, 11)]


def _stage_b_bundle(
    *,
    m_overlay: tuple[float, ...] = (-1.4, -1.5, -1.6),
    epochs_overlay: int | None = 2,
    acc_overlay: float = 0.8,
    monotone: bool = True,
) -> dict[str, Any]:
    """kfac vs kfac+overlay over three seeds: gamma = 0.4/0.5/0.6, rho_s = +-1.

    m_kfac = -1, m_newton = -2 on every record, so gamma_s = (m_overlay + 1) / (-1).
    """
    epsilons = [c * c if monotone else 1.0 / c for c in _COUPLINGS]
    overlay_records = [
        _record(-1.0, m, -2.0, step) for m, step in zip(m_overlay, _STEPS, strict=True)
    ]
    cross = [_cross(step, _COUPLINGS, epsilons) for step in _STEPS]
    base_records = [_record(-1.0, -1.0, -2.0, step) for step in _STEPS]
    seeds = ("seed_42", "seed_43", "seed_44")
    return {
        "kfac": _method(
            {s: _payload(records=base_records, epochs_to_target=3) for s in seeds}
        ),
        "kfac+overlay": _method(
            {
                s: _payload(
                    records=overlay_records,
                    cross=cross,
                    val_acc=(0.5, acc_overlay),
                    epochs_to_target=epochs_overlay,
                )
                for s in seeds
            }
        ),
    }


def _ekfac_bundle(*, n_refresh_failed: int = 0, nan_loss: bool = False) -> dict:
    val_loss = (1.0, math.nan) if nan_loss else (1.0, 0.5)
    payload = _payload(
        n_steps=100, n_refresh_failed=n_refresh_failed, val_loss=val_loss
    )
    return {
        "ekfac": _method({"seed_42": payload}),
        "ekfac+overlay": _method({"seed_42": payload}),
    }


def _exp8_bundle(
    phi_resnet: tuple[float, ...], phi_plain: tuple[float, ...]
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for arch, phis in (("resnet18", phi_resnet), ("plain_resnet18", phi_plain)):
        raw[arch] = {}
        for c, ckpt in enumerate(("init", "final")):
            raw[arch][ckpt] = {
                f"seed_{42 + i}": {
                    "phi": phi if c else 0.1,
                    "diag_frob": {"stem": 1.0, "head": 2.0},
                    "coupling": {"stem->head": 0.3},
                    "test_acc": 0.5,
                }
                for i, phi in enumerate(phis)
            }
    return _aggregate_exp8(raw)


def _exp8_long_bundle() -> dict[str, Any]:
    """A tiny 2-seed exp8-shaped bundle, checkpoints ``init``/``50``/``100``/``final``
    (named and numeric labels mixed, as the real ``exp8_a1_long`` has), built so the
    accuracy-matched pairing has one dropped plain checkpoint, one
    kept pair whose nearest resnet checkpoint is *not* the same label, and both kept
    pairs' matches share both seeds (same accuracies) while their Phi differs - so the
    per-seed means differ and the t-interval is exercised on two distinct numbers.

    test_acc: resnet {init:.05, 50:.50, 100:.70, final:.86}, plain {init:.05,
    50:.30, 100:.505, final:.70}. Plain ``50`` is nearest resnet ``50`` (|.20| >
    tol.02): dropped. Plain ``100`` is nearest resnet ``50`` (|.005|): kept, matched
    to a different label. Plain ``final`` is nearest resnet ``100`` (|0|): kept,
    likewise a different label. Plain ``init`` is never queried (excluded by
    definition), even though its accuracy would match resnet ``init`` exactly.
    """
    acc = {
        "resnet18": {"init": 0.05, "50": 0.50, "100": 0.70, "final": 0.86},
        "plain_resnet18": {"init": 0.05, "50": 0.30, "100": 0.505, "final": 0.70},
    }
    # phi_by_probes["300"] per seed; only resnet 50/100 and plain 100/final matter to
    # the kept pairs, but every leaf gets one so the bundle shape is uniform.
    phi300 = {
        "seed_1": {
            "resnet18": {"init": 0.90, "50": 0.60, "100": 0.66, "final": 0.68},
            "plain_resnet18": {"init": 0.10, "50": 0.40, "100": 0.50, "final": 0.55},
        },
        "seed_2": {
            "resnet18": {"init": 0.91, "50": 0.62, "100": 0.70, "final": 0.71},
            "plain_resnet18": {"init": 0.11, "50": 0.42, "100": 0.50, "final": 0.53},
        },
    }
    raw: dict[str, Any] = {}
    for arch in ("resnet18", "plain_resnet18"):
        raw[arch] = {}
        for label in ("init", "50", "100", "final"):
            raw[arch][label] = {
                seed: {
                    "phi": phi300[seed][arch][label],
                    "phi_by_probes": {"300": phi300[seed][arch][label]},
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": acc[arch][label],
                }
                for seed in ("seed_1", "seed_2")
            }
    return _aggregate_exp8(raw)


def _exp9_bundle(models: dict[str, tuple[int, dict[str, float]]]) -> dict[str, Any]:
    raw = {
        model: {
            "seed_42": {"phi": 0.4, "diag_frob": {"embed": 1.0}, "coupling": coupling}
        }
        for model, (_, coupling) in models.items()
    }
    return _aggregate_exp9(raw, {m: n for m, (n, _) in models.items()})


_TOY = _exp9_bundle(
    {
        "tiny": (100, {"L0.attn_v->L0.attn_o": 0.5, "embed->L0.attn_q": 0.9}),
        "base": (
            1000,
            {
                "L0.attn_v->L0.attn_o": 0.9,
                "L0.attn_q->L0.attn_k": 0.8,
                "L0.attn_o->L0.mlp": 0.7,
                "embed->L0.attn_q": 0.1,
            },
        ),
    }
)
_LLM_PASS = _exp9_bundle(
    {
        "Qwen/Qwen2.5-0.5B": (
            500_000_000,
            {
                "L5.attn_v->L5.attn_o": 0.95,
                "L5.attn_q->L5.attn_k": 0.85,
                "L2.mlp->L3.attn_q": 0.6,
                "embed->L0.attn_q": 0.2,
            },
        )
    }
)
_LLM_FAIL = _exp9_bundle(
    {
        "Qwen/Qwen2.5-0.5B": (
            500_000_000,
            {
                "L5.attn_v->L5.attn_o": 0.95,
                "L0.attn_q->L0.attn_v": 0.9,
                "embed->L0.attn_q": 0.8,
                "L5.attn_q->L5.attn_k": 0.1,
            },
        )
    }
)
# The case of the real bundles: the three strongest pairs of the largest toy model and of
# each language model are all v->o, in three layers.
_TOY_V_TO_O = _exp9_bundle(
    {
        "tiny": (100, {"L0.attn_v->L0.attn_o": 0.5, "embed->L0.attn_q": 0.9}),
        "base": (
            1000,
            {
                "L0.attn_v->L0.attn_o": 0.9,
                "L1.attn_v->L1.attn_o": 0.8,
                "L2.attn_v->L2.attn_o": 0.7,
                "L1.attn_q->L1.attn_k": 0.6,
            },
        ),
    }
)
_LLM_V_TO_O = _exp9_bundle(
    {
        "Qwen/Qwen2.5-0.5B": (
            500_000_000,
            {
                "L5.attn_v->L5.attn_o": 0.95,
                "L6.attn_v->L6.attn_o": 0.9,
                "L7.attn_v->L7.attn_o": 0.85,
                "L5.attn_q->L5.attn_k": 0.8,
            },
        ),
        "meta-llama/Llama-3.2-1B": (
            1_200_000_000,
            {
                "L9.attn_v->L9.attn_o": 0.7,
                "L10.attn_v->L10.attn_o": 0.65,
                "L11.attn_v->L11.attn_o": 0.6,
                "L9.attn_q->L9.attn_k": 0.5,
            },
        ),
    }
)


# ======================================================================
# Gap closed
# ======================================================================


def test_gamma_records_rule() -> None:
    payload = _payload(
        records=[
            _record(-1.0, -1.5, -2.0, 0),  # gamma = 0.5
            _record(-1.0, -1.25, -2.0, 1),  # gamma = 0.25
            _record(-1.0, -1.0, -1.0, 2),  # denominator 0: skipped
            _record(-1.0, -2.2, -2.0, 3),  # gamma = 1.2 (not clipped)
            _record(-1.0, -1.0, -1.0 - 1e-9, 4),  # below 1e-8 * max(1, |m_N|): skipped
            _record(-100.0, -100.5, -100.0 - 5e-7, 5),  # 5e-7 < 1e-8 * 100: skipped
        ]
    )
    gammas, n_skipped = gamma_records(payload)
    assert n_skipped == 3, n_skipped
    assert np.allclose(gammas, [0.5, 0.25, 1.2]), gammas


def test_gap_closed_known_values() -> None:
    seed_a = _payload(
        records=[
            _record(-1.0, -1.5, -2.0, 0),
            _record(-1.0, -1.25, -2.0, 1),
            _record(-1.0, -1.0, -1.0, 2),
            _record(-1.0, -2.2, -2.0, 3),
        ]
    )
    seed_b = _payload(records=[_record(-1.0, -1.8, -2.0, 0)])  # gamma = 0.8
    result = gap_closed(_method({"seed_2": seed_b, "seed_1": seed_a}), n_boot=_B)

    assert result["n_records"] == 4 and result["n_skipped"] == 1
    assert math.isclose(result["median"], 0.65), "median of {0.25, 0.5, 0.8, 1.2}"
    assert math.isclose(result["frac_above_one"], 0.25)
    assert list(result["per_seed_median"]) == ["seed_1", "seed_2"], "numeric order"
    assert math.isclose(result["per_seed_median"]["seed_1"], 0.5)
    assert math.isclose(result["per_seed_median"]["seed_2"], 0.8)
    assert 0.25 - 1e-12 <= result["ci_low"] <= result["median"]
    assert result["median"] <= result["ci_high"] <= 1.2 + 1e-12
    # stratified: seed_2 contributes exactly one 0.8 to every replicate
    again = gap_closed(_method({"seed_2": seed_b, "seed_1": seed_a}), n_boot=_B)
    assert again["ci_low"] == result["ci_low"], "rng seed 0: reproducible"

    empty = gap_closed(_method({"seed_1": _payload()}), n_boot=_B)
    assert math.isnan(empty["median"]) and empty["n_records"] == 0


# ======================================================================
# Validity share, verdicts, negative curvature
# ======================================================================


def _bare_record(step: int) -> dict:
    """A ``newton_cg``-method record: no ``m_kfac``."""
    return {"step": step, "m_newton": -1.0, "n_hvp_newton": 3}


def test_validity_share() -> None:
    seed_a = _payload(
        records=[
            _record(-1.0, -1.0, -2.0, 0),
            _record(0.5, 0.5, -2.0, 1),  # m(d_0) > 0
            _record(2.0, 2.0, -2.0, 2),  # m(d_0) > 0
            _bare_record(3),  # skipped
            _record(math.nan, math.nan, -2.0, 4),  # skipped
        ]
    )
    seed_b = _payload(
        records=[_record(-1.0, -1.0, -2.0, 0), _record(-3.0, -3.0, -4.0, 1)]
    )
    result = validity_share(_method({"seed_2": seed_b, "seed_1": seed_a}))
    assert list(result["per_seed"]) == ["seed_1", "seed_2"], "numeric order"
    assert math.isclose(result["per_seed"]["seed_1"], 2 / 3)
    assert result["per_seed"]["seed_2"] == 0.0
    assert math.isclose(result["pooled"], 0.4) and result["n_records"] == 5

    only_bare = validity_share(_method({"seed_1": _payload(records=[_bare_record(0)])}))
    assert only_bare["n_records"] == 0 and math.isnan(only_bare["pooled"])
    assert math.isnan(only_bare["per_seed"]["seed_1"])

    stage_b = validity_share(_stage_b_bundle()["kfac+overlay"])
    assert stage_b["pooled"] == 0.0 and stage_b["n_records"] == 9, (
        "m_kfac = -1 throughout"
    )


def test_reject_breakdown() -> None:
    # three seeds x (4 proposed, 1 rejected by the model test)
    result = reject_breakdown(_stage_b_bundle()["kfac+overlay"])
    assert (result["n_proposed"], result["n_accepted"], result["n_reject"]) == (
        12,
        9,
        3,
    )
    assert result["n_reject_by"] == {"ratio": 0, "model": 3, "loss": 0}
    assert math.isclose(result["accept_share"], 0.75)
    assert math.isclose(result["reject_share_by"]["model"], 0.25)
    assert result["reject_share_by"]["loss"] == 0.0

    mixed = _method(
        {
            "seed_1": _payload(
                n_proposed=10, n_reject_by={"ratio": 2, "model": 3, "loss": 4}
            ),
            "seed_2": _payload(
                n_proposed=5, n_reject_by={"ratio": 0, "model": 0, "loss": 1}
            ),
        }
    )
    result = reject_breakdown(mixed)
    assert result["n_proposed"] == 15 and result["n_accepted"] == 5
    assert result["n_reject_by"] == {"ratio": 2, "model": 3, "loss": 5}
    assert math.isclose(result["accept_share"], 1 / 3)
    assert math.isclose(result["reject_share_by"]["loss"], 1 / 3)

    nothing = _payload(n_proposed=0, n_reject_by={"ratio": 0, "model": 0, "loss": 0})
    none = reject_breakdown(_method({"seed_1": nothing}))
    assert none["n_proposed"] == 0 and math.isnan(none["accept_share"])
    assert all(math.isnan(v) for v in none["reject_share_by"].values())

    legacy = _payload()  # Missing rejection categories remain unattributed.
    del legacy["safeguard"]["n_reject_by"]
    result = reject_breakdown(_method({"seed_1": legacy}))
    assert result["n_reject"] == 1 and result["n_accepted"] == 3
    assert result["n_reject_by"] == {"ratio": 0, "model": 0, "loss": 0}, "unattributed"

    for method in _ekfac_bundle().values():
        assert reject_breakdown(method)["n_reject_by"]["model"] == 1


def test_negative_curvature_share() -> None:
    result = negative_curvature_share(_stage_b_bundle()["kfac"])  # 3 x (10 solves, 4)
    assert result["n_solves"] == 30 and result["n_negative_curvature"] == 12
    assert math.isclose(result["share"], 0.4)

    none = negative_curvature_share(_method({"seed_1": _payload(newton_cg=(0, 0))}))
    assert none["n_solves"] == 0 and math.isnan(none["share"])

    legacy = _payload()
    del legacy["newton_cg"]
    assert negative_curvature_share(_method({"seed_1": legacy}))["n_solves"] == 0


# ======================================================================
# Spearman
# ======================================================================


def test_spearman_rows_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    c = rng.integers(0, 5, size=40).astype(float)
    e = np.round(c + rng.normal(size=40))
    expected = stats.spearmanr(c, e)[0]
    assert math.isclose(_spearman_rows(c[None], e[None])[0], expected, abs_tol=1e-12)
    constant = _spearman_rows(np.ones((1, 5)), c[None, :5])
    assert math.isnan(constant[0]), "a constant row has no rank correlation"


def test_spearman_monotone_relation() -> None:
    epsilons = [c * c for c in _COUPLINGS]
    method = _method(
        {"seed_1": _payload(cross=[_cross(s, _COUPLINGS, epsilons) for s in _STEPS])}
    )
    result = spearman_coupling_eps(method, n_boot=_B)
    assert math.isclose(result["rho"], 1.0) and result["p"] < 0.01
    assert result["n_pairs"] == 30 and result["n_records"] == 3
    assert result["ci_low"] == result["ci_high"] == 1.0

    anti = _method({"seed_1": _payload(cross=[_cross(0, _COUPLINGS, epsilons[::-1])])})
    assert math.isclose(spearman_coupling_eps(anti, n_boot=_B)["rho"], -1.0)

    # NaNs are dropped pair by pair; records of unequal length still bootstrap
    with_nan = _payload(
        cross=[
            _cross(0, _COUPLINGS, epsilons),
            _cross(5, [math.nan] + _COUPLINGS[1:], epsilons),
        ]
    )
    result = spearman_coupling_eps(_method({"seed_1": with_nan}), n_boot=_B)
    assert result["n_pairs"] == 19 and math.isclose(result["rho"], 1.0)

    empty = spearman_coupling_eps(_method({"seed_1": _payload()}), n_boot=_B)
    assert math.isnan(empty["rho"]) and empty["n_pairs"] == 0


# ======================================================================
# Eps variance decomposition
# ======================================================================


def _eps_pairs(values: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    return {
        f"{v}->{w}": {"eps": math.exp(value), "coupling": 1.0}
        for (v, w), value in values.items()
    }


def test_eps_variance_decomposition_is_order_invariant() -> None:
    # The diagonal-free grid (v == w never observed) is an incomplete design, so a
    # sequential (row-then-column) decomposition depends on which factor is demeaned
    # first. The least-squares fit does not: shuffling the pairs, or transposing which
    # axis is called row and which is column, must leave the residual share unchanged.
    values = {
        ("v1", "v2"): 0.9,
        ("v1", "v3"): -0.4,
        ("v2", "v1"): 1.7,
        ("v2", "v3"): 0.2,
        ("v3", "v1"): -1.1,
        ("v3", "v2"): 0.6,
    }
    original = eps_variance_decomposition([{"step": 0, "pairs": _eps_pairs(values)}])

    shuffled_pairs = dict(reversed(list(_eps_pairs(values).items())))
    shuffled = eps_variance_decomposition([{"step": 0, "pairs": shuffled_pairs}])
    assert math.isclose(
        original["residual_share"], shuffled["residual_share"], abs_tol=1e-9
    )

    transposed_values = {(w, v): value for (v, w), value in values.items()}
    transposed = eps_variance_decomposition(
        [{"step": 0, "pairs": _eps_pairs(transposed_values)}]
    )
    assert math.isclose(
        original["residual_share"], transposed["residual_share"], abs_tol=1e-9
    )


def test_eps_variance_decomposition_additive_data_has_zero_residual() -> None:
    # log eps(v, w) = a_v + b_w exactly on the diagonal-free grid (v == w excluded):
    # the additive model explains everything, so nothing is left for the residual.
    a = {"v1": 2.0, "v2": -1.0, "v3": 0.5}
    b = {"v1": 0.3, "v2": -0.7, "v3": 1.1}
    values = {(v, w): a[v] + b[w] for v in a for w in b if v != w}
    out = eps_variance_decomposition([{"step": 0, "pairs": _eps_pairs(values)}])
    assert math.isclose(out["residual_share"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["additive_share"], 1.0, abs_tol=1e-9)
    assert out["n_used"] == 6 and out["n_skipped"] == 0


def test_eps_variance_decomposition_pair_specific_term_raises_residual() -> None:
    # One cell pushed away from mu + a_v + b_w: the additive model can no longer fit it
    # exactly, so the residual share must move above zero.
    a = {"v1": 2.0, "v2": -1.0, "v3": 0.5}
    b = {"v1": 0.3, "v2": -0.7, "v3": 1.1}
    additive = {(v, w): a[v] + b[w] for v in a for w in b if v != w}
    bumped = dict(additive)
    bumped[("v1", "v2")] += 5.0

    plain = eps_variance_decomposition([{"step": 0, "pairs": _eps_pairs(additive)}])
    out = eps_variance_decomposition([{"step": 0, "pairs": _eps_pairs(bumped)}])
    assert math.isclose(plain["residual_share"], 0.0, abs_tol=1e-9)
    assert out["residual_share"] > 1e-6


def test_eps_variance_decomposition_skips_nan_and_nonpositive_eps() -> None:
    a = {"v1": 2.0, "v2": -1.0, "v3": 0.5}
    b = {"v1": 0.3, "v2": -0.7, "v3": 1.1}
    values = {(v, w): a[v] + b[w] for v in a for w in b if v != w}
    pairs = _eps_pairs(values)
    pairs["v1->v2"] = {"eps": math.nan, "coupling": 1.0}  # undefined: dropped
    pairs["v2->v3"] = {"eps": 0.0, "coupling": 1.0}  # log undefined: dropped
    pairs["v3->v1"] = {"eps": -2.0, "coupling": 1.0}  # log undefined: dropped
    out = eps_variance_decomposition([{"step": 0, "pairs": pairs}])
    assert out["n_used"] == 3 and out["n_skipped"] == 3
    assert math.isfinite(out["residual_share"]) and math.isfinite(out["additive_share"])
    assert math.isclose(
        out["additive_share"] + out["residual_share"], 1.0, abs_tol=1e-9
    )


def test_eps_variance_decomposition_empty_is_nan() -> None:
    out = eps_variance_decomposition([{"step": 0, "pairs": {}}])
    assert out["n_used"] == 0 and out["n_skipped"] == 0
    assert math.isnan(out["residual_share"])
    assert math.isnan(out["additive_share"])
    assert math.isnan(out["row_effect_sd"])
    assert math.isnan(out["col_effect_sd"])
    # Verify the joint fit's output fields.
    assert set(out) == {
        "residual_share",
        "additive_share",
        "row_effect_sd",
        "col_effect_sd",
        "n_used",
        "n_skipped",
    }


# ======================================================================
# Gates
# ======================================================================


def test_coupling_eps_by_pair() -> None:
    """Per ordered pair over all records and seeds, in order of first appearance, NaNs
    dropped (a pair without a finite record is absent)."""
    by_pair = coupling_eps_by_pair(_stage_b_bundle()["kfac+overlay"])
    assert list(by_pair) == [f"b{i}->b{i + 1}" for i in range(10)]
    c, e = by_pair["b2->b3"]
    assert c.shape == e.shape == (9,), "3 seeds x 3 records"
    assert np.allclose(c, 0.3) and np.allclose(e, 0.09)

    with_nan = _payload(
        cross=[
            _cross(0, [0.5, math.nan], [2.0, 1.0]),
            _cross(5, [0.7, 0.4], [math.nan, 3.0]),
        ]
    )
    by_pair = coupling_eps_by_pair({"seeds": {"seed_1": with_nan}})
    assert {k: (list(c), list(e)) for k, (c, e) in by_pair.items()} == {
        "b0->b1": ([0.5], [2.0]),
        "b1->b2": ([0.4], [3.0]),
    }


def test_coupling_field_values() -> None:
    """Every finite coupling, pooled flat over all seeds/records/ordered pairs
     -- 3 seeds x 3 steps x
    10 pairs of `_stage_b_bundle`'s ``kfac+overlay``, all finite."""
    values = coupling_field_values(_stage_b_bundle()["kfac+overlay"])
    assert len(values) == 90
    assert statistics.median(values) == pytest.approx(0.55)
    assert max(values) == pytest.approx(1.0)


def test_coupling_field_values_does_not_consult_eps() -> None:
    """A pair with a finite coupling and a non-finite eps is counted by
    `coupling_field_values` but dropped by `coupling_eps_records`/
    `coupling_eps_by_pair`, which require both finite."""
    method = {
        "seeds": {"seed_1": _payload(cross=[_cross(0, [0.5, 0.9], [2.0, math.nan])])}
    }
    assert coupling_field_values(method) == [0.5, 0.9]

    records = coupling_eps_records(method)
    assert len(records) == 1
    c_arr, e_arr = records[0]
    assert list(c_arr) == [0.5]
    assert list(e_arr) == [2.0]

    by_pair = coupling_eps_by_pair(method)
    assert list(by_pair) == ["b0->b1"], "b1->b2 dropped: its eps is non-finite"


def test_gate_a_pass_and_fail() -> None:
    passing = gate_a(_exp8_bundle((0.50, 0.52, 0.48), (0.30, 0.31, 0.29)))
    assert passing["pass"] and not passing["missing"]
    assert math.isclose(passing["mean_delta"], 0.2) and passing["n_seeds"] == 3
    assert math.isnan(passing["ci_low"]), "no bootstrap CI below 5 seeds"

    one_negative = gate_a(_exp8_bundle((0.50, 0.52, 0.28), (0.30, 0.31, 0.29)))
    assert not one_negative["pass"], "every delta_s must be positive"
    small = gate_a(_exp8_bundle((0.32, 0.33, 0.34), (0.30, 0.31, 0.29)))
    assert not small["pass"] and math.isclose(small["mean_delta"], 0.03)

    five = gate_a(
        _exp8_bundle((0.5, 0.52, 0.48, 0.51, 0.49), (0.3, 0.31, 0.29, 0.3, 0.3)),
        n_boot=_B,
    )
    assert five["pass"] and five["ci_low"] <= five["mean_delta"] <= five["ci_high"]

    missing = gate_a({"resnet18": _exp8_bundle((0.5,), (0.3,))["resnet18"]})
    assert not missing["pass"] and missing["missing"] == ["plain_resnet18/final"]


def test_gate_a_replicated_pools_two_runs() -> None:
    published = _exp8_bundle((0.50, 0.52, 0.48), (0.30, 0.31, 0.29))  # delta_s mean 0.2
    replication = _exp8_bundle((0.40, 0.42, 0.38), (0.30, 0.31, 0.29))  # mean 0.1
    pooled = gate_a_replicated([published, replication])
    assert not pooled["missing"] and pooled["n"] == 6
    assert math.isclose(pooled["mean"], 0.15)
    means = [g["mean_delta"] for g in pooled["per_bundle"]]
    assert math.isclose(means[0], 0.2) and math.isclose(means[1], 0.1)
    assert pooled["ci_low"] < pooled["mean"] < pooled["ci_high"]
    # gate_a's own semantics are untouched: each per-bundle entry still passes on its
    # own pre-registered terms (every delta_s > 0, mean >= 0.05).
    assert all(g["pass"] for g in pooled["per_bundle"])

    missing = gate_a_replicated([published, {"resnet18": replication["resnet18"]}])
    assert missing["missing"] == ["plain_resnet18/final"]
    assert missing["n"] == 0 and math.isnan(missing["mean"])

    single = gate_a_replicated([_exp8_bundle((0.5,), (0.3,))])
    assert single["n"] == 1 and math.isnan(single["ci_low"]), "n < 2: no t-interval"
    assert math.isclose(single["mean"], 0.2)
    assert single["n_seeds"] == 1 and math.isnan(single["ci_low_seeds"])
    # No intraclass correlation is reported.
    assert "icc" not in single


def test_gate_a_replicated_interval_over_seeds_is_the_t_interval_of_the_means() -> None:
    """Same two seeds-42-44 runs, but with no run-to-run noise at all (run 2 is run 1
    plus a fixed offset, tiny next to the seed-to-seed spread): the per-seed-mean
    interval is exactly the plain t-interval over the three per-seed means (n=3,
    df=2)."""
    run1 = _exp8_bundle((0.40, 0.60, 0.80), (0.30, 0.30, 0.30))  # delta .10/.30/.50
    run2 = _exp8_bundle((0.42, 0.62, 0.82), (0.30, 0.30, 0.30))  # delta .12/.32/.52
    pooled = gate_a_replicated([run1, run2])

    assert pooled["n_seeds"] == 3
    assert pooled["per_seed_mean"] == pytest.approx(
        {"seed_42": 0.11, "seed_43": 0.31, "seed_44": 0.51}
    )
    seed_means = np.array([0.11, 0.31, 0.51])
    mean_s, sd_s = float(seed_means.mean()), float(seed_means.std(ddof=1))
    se_s = sd_s / math.sqrt(3)
    t_crit = float(stats.t.ppf(0.975, df=2))
    assert pooled["ci_low_seeds"] == pytest.approx(mean_s - t_crit * se_s)
    assert pooled["ci_high_seeds"] == pytest.approx(mean_s + t_crit * se_s)


def test_gate_a_replicated_per_seed_means_average_the_runs() -> None:
    """Identical per-seed means (all three seeds average to the same delta) with all
    of the spread coming from run-to-run noise: each seed's mean is taken over the
    runs, and the interval over the three equal means is a point."""
    noisy1 = _exp8_bundle((0.40, 0.35, 0.30), (0.30, 0.30, 0.30))  # delta .10/.05/.00
    noisy2 = _exp8_bundle((0.50, 0.55, 0.60), (0.30, 0.30, 0.30))  # delta .20/.25/.30
    pooled = gate_a_replicated([noisy1, noisy2])

    assert pooled["per_seed_mean"] == pytest.approx(
        {"seed_42": 0.15, "seed_43": 0.15, "seed_44": 0.15}
    )
    assert pooled["ci_low_seeds"] == pytest.approx(0.15)
    assert pooled["ci_high_seeds"] == pytest.approx(0.15)


# ======================================================================
# Accuracy-matched Delta Phi
# ======================================================================


def test_accuracy_matched_delta_phi_pairing_and_ci() -> None:
    exp8 = _exp8_long_bundle()
    result = accuracy_matched_delta_phi(exp8, tol=0.02, probes="300")

    for seed in ("seed_1", "seed_2"):
        by_plain_label = {p[0]: p for p in result["pairs"][seed]}
        assert set(by_plain_label) == {"100", "final"}, "50 dropped, init excluded"
        assert by_plain_label["100"][1] == "50", "nearest by accuracy, not same label"
        assert by_plain_label["final"][1] == "100", (
            "nearest by accuracy, not same label"
        )
        # (plain_label, resnet_label, acc_plain, acc_resnet, delta)
        assert by_plain_label["100"][2:4] == (0.505, 0.50)
        assert by_plain_label["final"][2:4] == (0.70, 0.70)

    assert result["n_pairs"] == 4
    assert math.isclose(result["pairs"]["seed_1"][0][4], 0.10), "resnet .60 - plain .50"
    assert math.isclose(result["pairs"]["seed_1"][1][4], 0.11), "resnet .66 - plain .55"
    assert math.isclose(result["pairs"]["seed_2"][0][4], 0.12), "resnet .62 - plain .50"
    assert math.isclose(result["pairs"]["seed_2"][1][4], 0.17), "resnet .70 - plain .53"

    assert math.isclose(result["per_seed_mean"]["seed_1"], 0.105)
    assert math.isclose(result["per_seed_mean"]["seed_2"], 0.145)
    assert result["n_seeds"] == 2

    # the CI arithmetic against scipy directly, on the same per-seed means
    values = np.array([0.105, 0.145])
    mean, sd = values.mean(), values.std(ddof=1)
    se = sd / math.sqrt(2)
    t_crit = stats.t.ppf(0.975, df=1)
    assert math.isclose(result["mean"], mean)
    assert math.isclose(result["sd"], sd)
    assert math.isclose(result["ci_low"], mean - t_crit * se)
    assert math.isclose(result["ci_high"], mean + t_crit * se)

    # a tolerance no accuracy gap can meet drops every pair: NaN mean, no seeds
    # contribute (the seed is still a key of ``pairs``, just with an empty list)
    empty = accuracy_matched_delta_phi(exp8, tol=-1.0, probes="300")
    assert empty["n_pairs"] == 0 and empty["n_seeds"] == 0
    assert math.isnan(empty["mean"]) and not empty["per_seed_mean"]
    assert empty["pairs"]["seed_1"] == []


def test_accuracy_matched_delta_phi_skips_a_seed_one_arch_never_measured() -> None:
    """A seed a merge left out of ``resnet`` entirely - present under ``plain``, absent
    from every ``resnet`` checkpoint - is skipped for that seed, not a ``KeyError``:
    the nearest-checkpoint search only considers resnet checkpoints that carry the
    seed being matched, and a seed with no such checkpoint contributes no pair. A
    normal, fully-shared seed is unaffected."""
    raw: dict[str, Any] = {
        "resnet18": {
            "init": {
                "seed_42": {
                    "phi": 0.9,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.05,
                }
            },
            "final": {
                "seed_42": {
                    "phi": 0.6,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.70,
                }
            },
        },
        "plain_resnet18": {
            "init": {
                "seed_42": {
                    "phi": 0.1,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.05,
                },
                "seed_43": {
                    "phi": 0.1,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.05,
                },
            },
            "final": {
                "seed_42": {
                    "phi": 0.5,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.50,
                },
                # seed_43: resnet18 has no shard for this seed at any checkpoint.
                "seed_43": {
                    "phi": 0.55,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.50,
                },
            },
        },
    }
    exp8 = _aggregate_exp8(raw)

    result = accuracy_matched_delta_phi(exp8, tol=1.0)
    assert result["pairs"]["seed_43"] == [], "resnet never measured seed_43"
    assert "seed_43" not in result["per_seed_mean"]
    (pair_42,) = result["pairs"]["seed_42"]
    assert pair_42[:4] == ("final", "final", 0.5, 0.7)
    assert math.isclose(pair_42[4], 0.1)
    assert result["n_seeds"] == 1 and result["n_pairs"] == 1


def test_same_label_delta_phi_on_synthetic_bundle() -> None:
    exp8 = _exp8_long_bundle()
    result = same_label_delta_phi(exp8, probes="300")

    assert set(result) == {"init", "50", "100", "final"}
    at_100 = result["100"]
    assert at_100["n"] == 2 and at_100["deltas"] == [0.66 - 0.50, 0.70 - 0.50]
    assert math.isclose(at_100["mean"], 0.18)
    assert math.isclose(at_100["sd"], np.array([0.16, 0.20]).std(ddof=1))

    at_init = result["init"]
    assert math.isclose(at_init["mean"], ((0.90 - 0.10) + (0.91 - 0.11)) / 2)


def _hundred_epoch_bundle(resnet_final_phi: dict[str, float]) -> dict[str, Any]:
    """A short-run bundle with one ResNet Phi value per seed."""
    raw = {
        arch: {
            "final": {
                seed: {
                    "phi": phi if arch == "resnet18" else 0.5,
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.6,
                }
                for seed, phi in resnet_final_phi.items()
            }
        }
        for arch in ("resnet18", "plain_resnet18")
    }
    return _aggregate_exp8(raw)


def _long_bundle_with_final(resnet_final_phi300: dict[str, float]) -> dict[str, Any]:
    """A long-run bundle with distinct headline and 300-probe Phi values."""
    raw: dict[str, Any] = {}
    for arch in ("resnet18", "plain_resnet18"):
        raw[arch] = {}
        for label in ("init", "final"):
            raw[arch][label] = {}
            for seed, phi in resnet_final_phi300.items():
                value = phi if (arch, label) == ("resnet18", "final") else 0.5
                raw[arch][label][seed] = {
                    "phi": value + 0.05,
                    "phi_by_probes": {"300": value},
                    "diag_frob": {"a": 1.0},
                    "coupling": {"a->b": 0.1},
                    "test_acc": 0.7,
                }
    return _aggregate_exp8(raw)


_LONG_FINAL = {"seed_42": 0.72, "seed_43": 0.73, "seed_44": 0.74}  # mean 0.73
_HUNDRED = {"seed_42": 0.64, "seed_43": 0.66, "seed_44": 0.65}
_REPLICATION = {"seed_42": 0.58, "seed_43": 0.62, "seed_44": 0.60}


def test_tab_phi_long_requires_the_long_schedule_above_every_hundred_epoch_seed() -> (
    None
):
    """The long-schedule mean must strictly exceed every individual short-run seed,
    including the replication. Equality and higher short-run values are rejected."""
    long_bundle = _long_bundle_with_final(_LONG_FINAL)
    hundred = _hundred_epoch_bundle(_HUNDRED)
    replication = _hundred_epoch_bundle(_REPLICATION)
    tab_phi_long(long_bundle, hundred, replication)  # the data satisfy it

    above = _hundred_epoch_bundle({"seed_42": 0.64, "seed_43": 0.74, "seed_44": 0.65})
    with pytest.raises(
        ValueError,
    ):
        tab_phi_long(long_bundle, above, replication)

    above_replication = _hundred_epoch_bundle(
        {"seed_42": 0.58, "seed_43": 0.62, "seed_44": 0.80}
    )
    with pytest.raises(ValueError):
        tab_phi_long(long_bundle, hundred, above_replication)

    flat_long = _long_bundle_with_final(
        {"seed_42": 0.5, "seed_43": 0.5, "seed_44": 0.5}
    )
    equal = _hundred_epoch_bundle({"seed_42": 0.4, "seed_43": 0.5, "seed_44": 0.45})
    low = _hundred_epoch_bundle({"seed_42": 0.3, "seed_43": 0.35, "seed_44": 0.4})
    with pytest.raises(ValueError):
        tab_phi_long(flat_long, equal, low)


def test_accuracy_matched_delta_generalizes_the_pairing_to_any_quantity() -> None:
    """Matching uses the same checkpoint pairs for either selected quantity."""
    exp8 = _exp8_long_bundle()

    as_phi = accuracy_matched_delta(
        exp8, value=lambda payload: _phi_estimate(payload, "300")
    )
    via_wrapper = accuracy_matched_delta_phi(exp8, tol=0.02, probes="300")
    assert as_phi["pairs"] == via_wrapper["pairs"]
    assert as_phi["per_seed_mean"] == via_wrapper["per_seed_mean"]
    assert as_phi["mean"] == via_wrapper["mean"]
    assert as_phi["ci_low"] == via_wrapper["ci_low"]
    assert as_phi["ci_high"] == via_wrapper["ci_high"]

    by_acc = accuracy_matched_delta(
        exp8, value=lambda payload: float(payload["test_acc"])
    )
    pair_labels = {
        seed: [pair[:2] for pair in plist] for seed, plist in by_acc["pairs"].items()
    }
    assert pair_labels == {
        "seed_1": [("100", "50"), ("final", "100")],
        "seed_2": [("100", "50"), ("final", "100")],
    }
    # Both kept pairs' accuracies are shared across seeds (the fixture's docstring:
    # resnet .50/.70, plain .505/.70), so both seeds' mean of (acc_resnet - acc_plain)
    # over the two pairs is identical: ((.50-.505) + (.70-.70)) / 2 = -0.0025.
    assert math.isclose(by_acc["per_seed_mean"]["seed_1"], -0.0025)
    assert math.isclose(by_acc["per_seed_mean"]["seed_2"], -0.0025)
    assert math.isclose(by_acc["mean"], -0.0025)
    assert by_acc["sd"] == 0.0
    assert math.isclose(by_acc["ci_low"], -0.0025)
    assert math.isclose(by_acc["ci_high"], -0.0025)


def test_gate_b_pass_and_fail() -> None:
    passing = gate_b(_stage_b_bundle(), n_boot=_B)
    assert passing["pass"] and all(passing["checks"].values()), passing["checks"]
    assert math.isclose(passing["gap_closed"]["median"], 0.5)
    assert passing["epochs_to_target_median"] == {"kfac": 3.0, "kfac+overlay": 2.0}
    assert math.isclose(passing["min_final_acc_delta"], 0.0)

    low_gamma = gate_b(_stage_b_bundle(m_overlay=(-1.1, -1.1, -1.1)), n_boot=_B)
    assert not low_gamma["pass"] and not low_gamma["checks"]["gap_closed"]
    assert low_gamma["checks"]["spearman"], "only the gap check fails"

    slower = gate_b(_stage_b_bundle(epochs_overlay=None), n_boot=_B)
    assert not slower["checks"]["epochs_to_target"], "not reached counts as +inf"
    assert math.isinf(slower["epochs_to_target_median"]["kfac+overlay"])

    regressed = gate_b(_stage_b_bundle(acc_overlay=0.79), n_boot=_B)
    assert not regressed["checks"]["final_acc"], "-1 pp is below the -0.5 pp tolerance"
    within = gate_b(_stage_b_bundle(acc_overlay=0.796), n_boot=_B)
    assert within["checks"]["final_acc"]

    anti = gate_b(_stage_b_bundle(monotone=False), n_boot=_B)
    assert not anti["checks"]["spearman"] and anti["spearman"]["rho"] < 0

    missing = gate_b({"kfac": _stage_b_bundle()["kfac"]}, n_boot=_B)
    assert not missing["pass"] and missing["missing"] == ["kfac+overlay"]

    only_overlay = gate_b(_stage_b_bundle(), n_boot=_B)  # no ekfac+overlay present
    assert set(only_overlay["spearman_by_arm"]) == {"kfac+overlay"}, (
        "missing arm skipped"
    )


def test_gate_b_reports_other_arms_rho() -> None:
    # The pre-registered rho_s > 0, p < 0.01 sub-criterion names no method (module
    # docstring): gate_b evaluates it on the primary overlay alone (pass/fail
    # unchanged), but reports every arm's own rho_s in spearman_by_arm.
    bundle = _stage_b_bundle()  # kfac+overlay: rho_s = +1 (monotone C, eps)
    anti_epsilons = [1.0 / c for c in _COUPLINGS]  # anti-monotone: rho_s = -1
    cross = [_cross(step, _COUPLINGS, anti_epsilons) for step in _STEPS]
    bundle["ekfac+overlay"] = _method(
        {s: _payload(cross=cross) for s in ("seed_42", "seed_43", "seed_44")}
    )
    gate = gate_b(bundle, n_boot=_B)
    assert gate["pass"], "B1/B3 still fail/pass on the primary overlay alone"
    assert set(gate["spearman_by_arm"]) == {"kfac+overlay", "ekfac+overlay"}
    assert math.isclose(gate["spearman_by_arm"]["kfac+overlay"]["rho"], 1.0)
    assert math.isclose(gate["spearman_by_arm"]["ekfac+overlay"]["rho"], -1.0)
    assert gate["spearman_by_arm"]["kfac+overlay"] is gate["spearman"], "same object"


def test_gate_c_pass_and_fail() -> None:
    passing = gate_c(_ekfac_bundle())
    assert passing["pass"] and passing["max_failed_ratio"] == 0.0
    assert set(passing["methods"]) == {"ekfac", "ekfac+overlay"}

    failed = gate_c(_ekfac_bundle(n_refresh_failed=1))  # 1 / (100 / 10) = 0.1
    assert not failed["pass"] and math.isclose(failed["max_failed_ratio"], 0.1)
    assert gate_c(_ekfac_bundle(n_refresh_failed=1), refresh_period=1)["pass"]

    nan_loss = gate_c(_ekfac_bundle(nan_loss=True))
    assert not nan_loss["pass"] and nan_loss["any_nan_val_loss"]

    none = gate_c(_stage_b_bundle())
    assert not none["pass"] and none["missing"] == ["ekfac"]


def test_pair_role() -> None:
    assert pair_role("L3.attn_v->L3.attn_o") == "v->o"
    assert pair_role("L3.attn_k->L3.attn_q") == "q<->k"
    assert pair_role("L3.attn_o->L3.mlp") == "o->mlp"
    assert pair_role("L2.mlp->L3.attn_q") == "mlp->next.attn_q"
    assert pair_role("L2.attn_v->L3.attn_v") == "adjacent.attn_v"
    assert pair_role("L2.attn_v->L4.attn_v") is None, "not adjacent"
    assert pair_role("embed->L0.attn_q") is None, "embed has no layer"
    assert pair_role("L0.attn_q->L0.attn_v") is None, "not a named role"
    assert pair_role("L2.mlp->L3.mlp") == "adjacent.mlp"


def test_gate_d_pass_and_fail() -> None:
    passing = gate_d(_TOY, _LLM_PASS)
    assert passing["pass"] and passing["toy_model"] == "base"
    assert passing["toy_role_counts"] == {"v->o": 1, "q<->k": 1, "o->mlp": 1}
    llm = passing["llm"]["Qwen/Qwen2.5-0.5B"]
    assert llm["role_counts"] == {"v->o": 1, "q<->k": 1, "mlp->next.attn_q": 1}
    assert llm["matched_roles"] == {"v->o": 1, "q<->k": 1}
    assert llm["n_matched"] == 2 and llm["J"] == "2/3" and llm["pass"]
    assert "toy_roles" not in passing
    assert not {"roles", "shared_roles", "n_shared"} & set(llm)

    failing = gate_d(_TOY, _LLM_FAIL)
    assert not failing["pass"]
    llm = failing["llm"]["Qwen/Qwen2.5-0.5B"]
    assert llm["n_matched"] == 1 and llm["J"] == "1/3" and not llm["pass"]
    # the two pairs of its top three without a role are matched with none
    assert llm["role_counts"] == {"v->o": 1}

    missing = gate_d(_TOY, None)
    assert not missing["pass"] and missing["missing"] == ["exp9_a3_llm"]


def test_gate_d_matches_pairs_not_distinct_roles() -> None:
    """Gate D matches top-three pairs one-to-one by structural role.

    Three pairs of the same role match three pairs, not just one distinct role.
    """
    result = gate_d(_TOY_V_TO_O, _LLM_V_TO_O)
    assert result["pass"] and result["toy_model"] == "base"
    assert result["toy_role_counts"] == {"v->o": 3}
    assert set(result["llm"]) == {"Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"}
    for entry in result["llm"].values():
        assert entry["role_counts"] == {"v->o": 3}
        assert entry["matched_roles"] == {"v->o": 3}
        assert entry["n_matched"] == 3 and entry["J"] == "3/3" and entry["pass"]


def _vo(layer: int) -> str:
    return f"L{layer}.attn_v->L{layer}.attn_o"


def _qk(layer: int) -> str:
    return f"L{layer}.attn_q->L{layer}.attn_k"


def _om(layer: int) -> str:
    return f"L{layer}.attn_o->L{layer}.mlp"


# three pairs `pair_role` gives no role
_NO_ROLE = ["embed->L0.attn_q", "L0.attn_q->L0.attn_v", "L1.attn_q->L1.attn_v"]


def test_gate_d_counts_the_smaller_side_of_each_role() -> None:
    """The pairs matched are the sum over roles of min(pairs of that role in the toy
    model's top three, in the language model's); a pair without a role is matched with
    none, an identical roleless pair on the other side included."""
    cases = [
        # (toy's top three, language model's top three, pairs matched)
        ([_vo(0), _vo(1), _vo(2)], [_vo(5), _vo(6), _qk(5)], 2),
        ([_vo(0), _vo(1), _qk(0)], [_vo(5), _qk(5), _qk(6)], 2),
        ([_vo(0), _vo(1), _vo(2)], [_vo(5), _qk(5), _om(5)], 1),
        ([_vo(0), _vo(1), _vo(2)], [_qk(5), _qk(6), _qk(7)], 0),
        (_NO_ROLE, _NO_ROLE, 0),
        ([_vo(0), *_NO_ROLE[:2]], [_vo(5), *_NO_ROLE[1:]], 1),
    ]
    for toy_pairs, llm_pairs, expected in cases:
        toy_coupling = {p: 0.9 - 0.1 * i for i, p in enumerate(toy_pairs)}
        llm_coupling = {p: 0.9 - 0.1 * i for i, p in enumerate(llm_pairs)}
        toy = _exp9_bundle({"base": (1000, toy_coupling)})
        llm = _exp9_bundle({"Qwen/Qwen2.5-0.5B": (5, llm_coupling)})
        entry = gate_d(toy, llm)["llm"]["Qwen/Qwen2.5-0.5B"]
        assert entry["n_matched"] == expected, (toy_pairs, llm_pairs)
        assert entry["J"] == f"{expected}/3"
        assert entry["pass"] == (expected >= 2)


def test_gate_d_requires_every_language_model() -> None:
    llm = _exp9_bundle(
        {
            "Qwen/Qwen2.5-0.5B": (
                500_000_000,
                {
                    "L5.attn_v->L5.attn_o": 0.95,
                    "L6.attn_v->L6.attn_o": 0.9,
                    "L7.attn_v->L7.attn_o": 0.85,
                },
            ),
            "meta-llama/Llama-3.2-1B": (
                1_200_000_000,
                {
                    "L9.attn_v->L9.attn_o": 0.7,
                    "embed->L0.attn_q": 0.65,
                    "L0.attn_q->L0.attn_v": 0.6,
                },
            ),
        }
    )
    result = gate_d(_TOY_V_TO_O, llm)
    assert result["llm"]["Qwen/Qwen2.5-0.5B"]["pass"]
    assert not result["llm"]["meta-llama/Llama-3.2-1B"]["pass"]
    assert not result["pass"]


# ======================================================================
# Tables and figures
# ======================================================================

_STAGE_B_HEADER = (
    r"Method & Final acc.\,(\%) & Ep.\ to target & $\gamma$ [95\,\% CI]"
    "\n"
    r"    & $m(d_0)>0$ & Accept & \begin{tabular}[c]{@{}c@{}}Verdicts\\acc.; rej.\ by"
    r"\\ratio/model/loss\end{tabular} & $\rho_s$ \\"
)


_PAYOFF_HEADER = r"Method & HVP/step & HVP/solve & Non-pos.\ curv. & s/step \\"


def _newton_record(step: int, loss: float | None = None) -> dict[str, Any]:
    """A ``newton_cg``-method record with a ``loss_theta`` (none when ``loss`` is None)."""
    record: dict[str, Any] = {"step": step, "m_newton": -1.0, "n_hvp_newton": 3}
    if loss is not None:
        record["loss_theta"] = loss
    return record


def test_nonfinite_record_steps_lists_the_seeds_that_went_non_finite() -> None:
    """`nonfinite_record_steps`: per seed, the steps of the payoff records whose
    ``loss_theta`` is NaN or infinite -- the states a run reached after it diverged. Seeds with none
    are left out, in step order; a record without a ``loss_theta`` is not counted."""
    from paper_data.analysis import nonfinite_record_steps

    method = _method(
        {
            "seed_2": _payload(
                records=[
                    _newton_record(0, 5.0),
                    _newton_record(20, 6.0),
                    _newton_record(40, math.nan),
                    _newton_record(60, math.inf),
                ]
            ),
            "seed_1": _payload(
                records=[_newton_record(0, 5.0), _newton_record(20, 5.5)]
            ),
            "seed_3": _payload(records=[_newton_record(0), _newton_record(20)]),
        }
    )
    assert nonfinite_record_steps(method) == {"seed_2": [40, 60]}
    assert nonfinite_record_steps(_method({"seed_1": _payload()})) == {}


def test_validity_share_reports_counted_records_whose_loss_does_not_fall() -> None:
    """`validity_share` also lists, as ``loss_not_falling``, the (seed, step) of the records
    it counts (m_kfac > 0) on which the minibatch loss does not demonstrably fall after
    the base step: ``loss_d0`` not below ``loss_theta``, or either missing or not finite. Records with m_kfac <= 0 are not
    counted, so a rising loss on them is not listed."""

    def record(m: float, step: int, theta: float | None, d0: float | None) -> dict:
        out = _record(m, m, -2.0, step)
        if theta is not None and d0 is not None:
            out["loss_theta"], out["loss_d0"] = theta, d0
        return out

    method = _method(
        {
            "seed_2": _payload(
                records=[
                    record(0.5, 0, 5.0, 4.0),  # counted, falls
                    record(0.5, 1, 5.0, 5.5),  # counted, rises
                    record(-1.0, 2, 5.0, 6.0),  # not counted
                    record(0.5, 3, None, None),  # counted, no loss recorded
                    record(0.5, 4, 5.0, math.nan),  # counted, not finite
                ]
            ),
            "seed_1": _payload(records=[record(0.5, 0, 5.0, 5.0)]),  # counted, equal
        }
    )
    result = validity_share(method)
    expected = [("seed_1", 0), ("seed_2", 1), ("seed_2", 3), ("seed_2", 4)]
    assert result["loss_not_falling"] == expected
    falls = _method({"seed_1": _payload(records=[record(0.5, 0, 5.0, 4.0)])})
    assert validity_share(falls)["loss_not_falling"] == []


@pytest.mark.parametrize("field", ["coupling", "eps"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_retrospective_nonpositive_domains_apply_to_both_variables(field, value):
    from paper_data.analysis import coupling_sensitivity

    pairs = {
        f"{v}->{w}": {"coupling": 1.0, "eps": 2.0}
        for v in "abc"
        for w in "abc"
        if v != w
    }
    for key in ("a->b", "b->a"):
        pairs[key][field] = value
    row = coupling_sensitivity(
        {"seeds": {"seed_42": {"cross_term": [{"pairs": pairs}]}}}
    )["seed_42"]
    assert row["raw"]["n_pairs"] == (6 if math.isfinite(value) else 4)
    assert row["symmetric"]["n_pairs"] == (3 if value == 0 else 2)
    assert row["adjusted"]["n_pairs"] == 4
    assert row["adjusted"]["n_skipped"] == 2
