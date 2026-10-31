"""Exp11 runner: the validity probe, the kink count, the hook, the bundle, the CLI.

The probe is checked on an explicit quadratic - where the undamped model is the loss, so
every grid point is valid, the secant equals the pointwise curvature and the damping
term is the whole model error, hence t* is known in closed form - and its spectral part
against `torch.linalg.eigvalsh`. The kink count is checked on a one-layer MLP whose
biases are displaced by hand (a known number of units cross zero at a known t) and on a
real ResNet-18 (max-pool windows counted, nothing changes at t = 0). The hook is checked
to take exactly the K-FAC step of Exp. 7 (`build_couplefac_hook` at tau = inf) and to
record the same payoff numbers, so the two experiments cross-check by construction; a
tiny CPU run on the synthetic MLP protocol checks the documented bundle shape, the
aggregate on a hand-built pair of seeds, and the CLI overrides on parsed arguments.
Multi-seed loader RNG isolation checks both persistent-loader reuse and fresh loader
construction per seed, including the runner's per-seed behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from dataclasses import replace
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments.config import Exp7Config, Exp11Config  # noqa: E402
from experiments.models import SegmentedMLP, SegmentedResNet18  # noqa: E402
from experiments.param_ops import group_major_params, trial_forward  # noqa: E402
from experiments.provenance import environment_block  # noqa: E402
from experiments.runner_exp7 import MethodTrace, build_couplefac_hook  # noqa: E402
from experiments.runner_exp11 import (  # noqa: E402
    STEP_SOURCES,
    Exp11Runner,
    ValidityTrace,
    _aggregate,
    add_args,
    build_config,
    build_validity_hook,
    count_kink_crossings,
    model_curve,
    probe_model_validity,
    validity_radius,
)
from experiments.training import StepCtx  # noqa: E402
from experiments.utils import set_seed  # noqa: E402
from hessian import ParamBlockEstimator  # noqa: E402
from secondorder.kfac_adapter import CurvlinopsKFACProvider  # noqa: E402
from secondorder.safeguard import model_value  # noqa: E402

_DEVICE = torch.device("cpu")
_PROBE_KEYS = {
    "step",
    "decomposition",
    "curve_d0",
    "t_star",
    "curve_vmax",
    "t_star_vmax",
    "sharpness",
    "payoff",
    "kinks",
}
_DECOMPOSITION_KEYS = {
    "g_dot_d0",
    "d0_H_d0",
    "lambda_norm_d0_sq",
    "m_d0",
    "norm_d0",
    "norm_g",
    "eta_eff",
    "eta_g",
}
_SHARPNESS_KEYS = {
    "sigma",
    "lambda_max",
    "lambda_min",
    "ritz_max",
    "ritz_min",
    "residual_max",
    "residual_min",
    "converged",
    "n_iter_used",
    "curv_vmax",
    "g_dot_vmax",
    "cos_d0_vmax",
    "curv_along_g",
    "curv_along_d0",
    "lambda_max_x_eta_eff",
    "lambda_max_x_eta_g",
}
_POINT_KEYS = {"t", "dL", "m", "secant", "rel_err"}
_KINK_ROW_KEYS = {"t", "relu_flips", "relu_frac", "pool_changes", "pool_frac"}
_PAYOFF_KEYS = {"m_kfac", "loss_theta", "loss_d0", "rho", "norm_d0"}


# ======================================================================
# Helpers
# ======================================================================


def _quadratic(n: int = 12, seed: int = 0) -> tuple[Tensor, Tensor]:
    """(g, H) with H = Q diag(linspace(0.1, 3)) Q^T (SPD, a clear top eigenvalue)."""
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen))
    h = q @ torch.diag(torch.linspace(0.1, 3.0, n)) @ q.T
    return torch.randn(n, generator=gen), (h + h.T) / 2


def _quadratic_loss(g: Tensor, h: Tensor, loss_0: float):
    """L(theta + d) = loss_0 + g.d + 1/2 d.Hd - the loss whose undamped model is exact."""
    return lambda d: loss_0 + (g.dot(d) + 0.5 * d.dot(h @ d)).item()


def _tiny_synthetic_loaders() -> tuple[DataLoader, DataLoader]:
    """In-memory loaders shaped like the synthetic MLP task (64 inputs, 10 classes)."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(64, 64, generator=gen)
    y = torch.randint(0, 10, (64,), generator=gen)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    val = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=16, shuffle=False)
    return train, val


def _probe(step: int, norm_d0: float, t_star: float, flips: int) -> dict[str, Any]:
    """A minimal hand-built probe record (the aggregate walks whatever it is given)."""
    return {
        "step": step,
        "decomposition": {"norm_d0": norm_d0},
        "curve_d0": [{"t": 0.01, "dL": -norm_d0, "rel_err": 0.0}],
        "t_star": t_star,
        "kinks": {"n_relu": 10, "n_pool": 0, "rows": [{"t": 1.0, "relu_flips": flips}]},
    }


# ======================================================================
# The probe on an explicit quadratic
# ======================================================================


def test_validity_radius_is_a_radius() -> None:
    points = [
        {"t": 0.01, "rel_err": 0.02},
        {"t": 0.1, "rel_err": 0.3},
        {"t": 1.0, "rel_err": 0.05},  # passes again, but the model already failed
    ]
    assert validity_radius(points, 0.1) == 0.01, "the scan ends at the first failure"
    assert validity_radius(points, 0.5) == 1.0
    assert math.isnan(validity_radius(points, 0.01)), "the smallest t fails: NaN"
    assert math.isnan(validity_radius([{"t": 0.01, "rel_err": math.nan}], 0.1))
    assert math.isnan(validity_radius([], 0.1))


def test_model_curve_exact_on_quadratic_and_damping_is_the_only_error() -> None:
    g, h = _quadratic()
    d = torch.randn(g.numel(), generator=torch.Generator().manual_seed(1))
    g_dot, dhd, norm_sq = g.dot(d).item(), d.dot(h @ d).item(), d.dot(d).item()
    loss_at = _quadratic_loss(g, h, loss_0=2.0)
    t_grid = [1.0, 0.1, 0.5]  # unsorted on purpose

    points, t_star = model_curve(
        loss_at, d, 2.0, g_dot=g_dot, dhd=dhd, norm_sq=norm_sq,
        damping=0.0, t_grid=t_grid, tolerance=1e-4,
    )  # fmt: skip
    assert [p["t"] for p in points] == [0.1, 0.5, 1.0], "points in increasing t"
    assert t_star == 1.0, "the undamped model of a quadratic is the loss itself"
    for p in points:
        assert p["rel_err"] < 1e-4 and set(p) == _POINT_KEYS
        assert math.isclose(p["secant"], dhd, rel_tol=1e-3), "secant = d.Hd everywhere"
        assert math.isclose(
            p["dL"], p["t"] * g_dot + 0.5 * p["t"] ** 2 * dhd, rel_tol=1e-4
        )

    # along steepest descent the damping term's share grows with t (~ t lambda / 2 at
    # small t), so with lambda = 0.7 the radius at tolerance 0.1 is the interior t = 0.1
    damping = 0.7
    descent = -g
    damped, t_star_damped = model_curve(
        loss_at, descent, 2.0, g_dot=-g.dot(g).item(), dhd=g.dot(h @ g).item(),
        norm_sq=g.dot(g).item(), damping=damping, t_grid=t_grid, tolerance=0.1,
    )  # fmt: skip
    expected = []
    for p in damped:
        err = 0.5 * p["t"] ** 2 * damping * g.dot(g).item() / abs(p["dL"])
        assert math.isclose(p["rel_err"], err, rel_tol=1e-3), (
            "the damping term is the error"
        )
        expected.append({"t": p["t"], "rel_err": err})
    assert [p["rel_err"] < 0.1 for p in damped] == [True, False, False]
    assert t_star_damped == 0.1 == validity_radius(expected, 0.1)


def test_probe_model_validity_on_dense_quadratic() -> None:
    g, h = _quadratic()
    damping = 0.1
    d0 = torch.linalg.solve(h + damping * torch.eye(g.numel()), -g)
    loss_0 = 3.0
    record = probe_model_validity(
        g,
        lambda z: h @ z,
        d0,
        _quadratic_loss(g, h, loss_0),
        damping=damping,
        t_grid=[0.25, 0.5, 1.0],
        tolerance=0.1,
        n_lanczos_iter=300,
        generator=torch.Generator().manual_seed(0),
    )
    assert set(record) == _PROBE_KEYS - {"step", "kinks"}

    dec = record["decomposition"]
    assert set(dec) == _DECOMPOSITION_KEYS
    hd0 = h @ d0
    assert math.isclose(dec["g_dot_d0"], g.dot(d0).item(), rel_tol=1e-5)
    assert math.isclose(dec["d0_H_d0"], d0.dot(hd0).item(), rel_tol=1e-5)
    assert math.isclose(
        dec["lambda_norm_d0_sq"], damping * d0.dot(d0).item(), rel_tol=1e-5
    )
    assert math.isclose(
        dec["m_d0"], model_value(g, d0, hd0, damping=damping), rel_tol=1e-6
    )
    assert math.isclose(dec["eta_eff"], d0.norm().item() / g.norm().item())
    assert math.isclose(dec["eta_g"], -g.dot(d0).item() / g.dot(g).item(), rel_tol=1e-5)

    # the payoff numbers are the Exp. 7 record: actual over predicted decrease
    pay = record["payoff"]
    assert set(pay) == _PAYOFF_KEYS
    assert pay["m_kfac"] == dec["m_d0"] and pay["norm_d0"] == dec["norm_d0"]
    assert pay["loss_theta"] == loss_0
    assert math.isclose(pay["loss_d0"] - loss_0, record["curve_d0"][-1]["dL"])
    assert math.isclose(pay["rho"], (loss_0 - pay["loss_d0"]) / -pay["m_kfac"])

    # spectral part against eigvalsh: v_max downhill, its curvature the top eigenvalue
    sharp = record["sharpness"]
    assert set(sharp) == _SHARPNESS_KEYS
    lam_max = torch.linalg.eigvalsh(h.double()).max().item()
    for key in ("sigma", "lambda_max", "curv_vmax"):
        assert abs(sharp[key] - lam_max) <= 1e-3 * lam_max, (key, sharp[key], lam_max)
    assert sharp["g_dot_vmax"] <= 0.0 and abs(sharp["cos_d0_vmax"]) <= 1.0 + 1e-6
    g_hat = g / g.norm()
    assert math.isclose(
        sharp["curv_along_g"], g_hat.dot(h @ g_hat).item(), rel_tol=1e-4
    )
    assert math.isclose(sharp["curv_along_d0"], dec["d0_H_d0"] / dec["norm_d0"] ** 2)
    assert math.isclose(
        sharp["lambda_max_x_eta_eff"], sharp["lambda_max"] * dec["eta_eff"]
    )
    assert math.isclose(sharp["lambda_max_x_eta_g"], sharp["lambda_max"] * dec["eta_g"])

    # the curve along ||d0|| v_max: same t axis, the quadratic's model exact but for lambda
    assert [p["t"] for p in record["curve_vmax"]] == [0.25, 0.5, 1.0]
    for p in record["curve_vmax"]:
        undamped = p["m"] - 0.5 * p["t"] ** 2 * damping * dec["norm_d0"] ** 2
        assert math.isclose(p["dL"], undamped, rel_tol=1e-3, abs_tol=1e-6)

    # without damping the model is the loss: every grid point valid along both lines
    exact = probe_model_validity(
        g,
        lambda z: h @ z,
        d0,
        _quadratic_loss(g, h, loss_0),
        damping=0.0,
        t_grid=[0.25, 0.5, 1.0],
        tolerance=1e-3,
        n_lanczos_iter=300,
        generator=torch.Generator().manual_seed(0),
    )
    assert exact["t_star"] == 1.0 and exact["t_star_vmax"] == 1.0
    assert math.isclose(exact["payoff"]["rho"], 1.0, rel_tol=1e-4)


# ======================================================================
# Kinks
# ======================================================================


def test_count_kink_crossings_hand_built_bias_shift() -> None:
    """x = 0, pre-activations = biases [-1, -1, 1, 1]: shifting the biases by 2 t flips
    two units per example once t > 1/2 and none before."""
    torch.manual_seed(0)
    model = SegmentedMLP(in_dim=4, hidden=4, depth=1, num_classes=2).eval()
    fc0 = cast(nn.Linear, model.blocks[0])
    with torch.no_grad():
        fc0.bias.copy_(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    params = group_major_params(model)
    assert [p.numel() for p in params] == [16, 4, 8, 2], "fc0.weight, fc0.bias, head"
    d0 = torch.zeros(30)
    d0[16:20] = 2.0  # the fc0 bias slice of the group-major flat vector
    x = torch.zeros(3, 4)

    kinks = count_kink_crossings(
        model, trial_forward(model, params, 1.0), d0, x, [1.0, 0.0, 0.25]
    )
    assert kinks["n_relu"] == 3 * 4 and kinks["n_pool"] == 0
    rows = kinks["rows"]
    assert [r["t"] for r in rows] == [0.0, 0.25, 1.0] and all(
        set(r) == _KINK_ROW_KEYS for r in rows
    )
    assert [r["relu_flips"] for r in rows] == [0, 0, 6], "2 units x 3 examples at t = 1"
    assert rows[2]["relu_frac"] == 0.5
    assert all(r["pool_changes"] == 0 and math.isnan(r["pool_frac"]) for r in rows)
    assert torch.equal(fc0.bias, torch.tensor([-1.0, -1.0, 1.0, 1.0])), (
        "theta untouched"
    )


def test_count_kink_crossings_on_resnet18_counts_pool_windows() -> None:
    torch.manual_seed(1)
    model = SegmentedResNet18(num_classes=10).eval()
    params = group_major_params(model)
    dim = sum(p.numel() for p in params)
    d0 = torch.randn(dim, generator=torch.Generator().manual_seed(2))
    d0 = 10.0 * d0 / d0.norm()
    x = torch.randn(2, 3, 32, 32)

    kinks = count_kink_crossings(
        model, trial_forward(model, params, 1.0), d0, x, [0.0, 1e-3, 1.0]
    )
    assert kinks["n_pool"] == 2 * 64 * 8 * 8, (
        "the stem max-pool: 64 x 8 x 8 windows per image"
    )
    assert kinks["n_relu"] > kinks["n_pool"]
    rows = kinks["rows"]
    assert rows[0]["relu_flips"] == 0 == rows[0]["pool_changes"], (
        "nothing moves at t = 0"
    )
    assert rows[2]["relu_flips"] > rows[1]["relu_flips"] > 0, "more kinks further along"
    assert 0.0 < rows[2]["relu_frac"] < 1.0 and 0.0 <= rows[2]["pool_frac"] <= 1.0
    assert rows[2]["pool_changes"] > 0, "a displacement of norm 10 moves some argmax"


# ======================================================================
# The hook is the Exp. 7 K-FAC step
# ======================================================================


def test_validity_hook_takes_the_exp7_kfac_step_and_records_its_payoff() -> None:
    """Same model, same minibatch: the exp11 hook moves theta exactly as the ``kfac``
    method of Exp. 7 does and records the payoff numbers of its record."""
    protocol = replace(
        Exp7Config.b1_mlp(), measure_period=1, coupling_n_probes_online=1
    )
    cfg = Exp11Config(
        protocol=protocol, probe_steps=[0], n_lanczos_iter=2, t_grid=[0.1, 1.0]
    )

    def fresh() -> SegmentedMLP:
        torch.manual_seed(0)
        return SegmentedMLP(in_dim=16, hidden=16, depth=2, num_classes=4)

    torch.manual_seed(3)
    x, y = torch.randn(32, 16), torch.randint(0, 4, (32,))

    exp7_model, exp11_model = fresh(), fresh()
    trace7 = MethodTrace()
    hook7 = build_couplefac_hook(
        exp7_model, protocol, tau=math.inf, seed=0, device=_DEVICE, trace=trace7
    )
    hook7(StepCtx(model=exp7_model, x=x, y=y, step=0, device=_DEVICE))

    trace11 = ValidityTrace()
    hook11 = build_validity_hook(
        exp11_model, cfg, seed=0, device=_DEVICE, trace=trace11
    )
    hook11(StepCtx(model=exp11_model, x=x, y=y, step=0, device=_DEVICE))

    for a, b in zip(exp7_model.parameters(), exp11_model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), "the same K-FAC step was applied"
    (record7,) = trace7.model_decrease
    (probe,) = trace11.probes
    for key in ("m_kfac", "loss_theta", "loss_d0", "norm_d0"):
        assert math.isclose(probe["payoff"][key], record7[key], rel_tol=1e-5), key
    assert math.isclose(probe["payoff"]["rho"], record7["rho"], rel_tol=1e-4)
    assert len(trace11.losses) == 1
    assert math.isclose(trace11.losses[0], probe["payoff"]["loss_theta"], rel_tol=1e-5)

    # a non-probe step records only the loss and still takes the step
    before = torch.cat([p.detach().flatten() for p in exp11_model.parameters()]).clone()
    hook11(StepCtx(model=exp11_model, x=x, y=y, step=1, device=_DEVICE))
    after = torch.cat([p.detach().flatten() for p in exp11_model.parameters()])
    assert len(trace11.probes) == 1 and len(trace11.losses) == 2
    assert not torch.allclose(before, after)


# ======================================================================
# Step sources (`--step-source`)
# ======================================================================


def test_kfac_step_source_matches_the_provider_directly() -> None:
    """`STEP_SOURCES["kfac"]` is `CurvlinopsKFACProvider.update` + `base_direction`,
    wired without changing either call: same model, same minibatch, same direction.

    ``fisher_type="mc"`` (`Exp7Config.b1_mlp`) draws its Monte-Carlo Fisher labels on
    curvlinops' own `KFACLinearOperator`, which reseeds a private generator to a fixed
    constant before every refresh instead of drawing off the global RNG, and our
    adapter never overrides that seed. Both sides' labels therefore agree regardless
    of the global RNG state; the `torch.manual_seed(123)` calls below are ordinary
    test hygiene, not a requirement for that agreement.
    """
    torch.manual_seed(0)
    model = SegmentedMLP(in_dim=16, hidden=16, depth=2, num_classes=4)
    protocol = Exp7Config.b1_mlp()
    loss_fn = nn.CrossEntropyLoss()
    x, y = torch.randn(8, 16), torch.randint(0, 4, (8,))
    cur = ParamBlockEstimator(model, loss_fn).curvature(x, y)

    torch.manual_seed(123)
    d0 = STEP_SOURCES["kfac"](model, protocol).direction(cur, x, y)

    provider = CurvlinopsKFACProvider(
        model,
        loss_fn,
        damping=protocol.damping,
        fisher_type=protocol.fisher_type,
        refresh_period=protocol.kfac_refresh_period,
    )
    torch.manual_seed(123)
    provider.update(x, y)
    expected = provider.base_direction(cur.flat_grad)
    assert torch.equal(d0, expected)


def test_step_source_unknown_choice_is_rejected() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            parser.parse_args(["--step-source", "nosuch"])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for an unknown --step-source")


def test_sgd_step_source_is_minus_gradient() -> None:
    torch.manual_seed(0)
    model = SegmentedMLP(in_dim=8, hidden=8, depth=1, num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    x, y = torch.randn(4, 8), torch.randint(0, 3, (4,))
    cur = ParamBlockEstimator(model, loss_fn).curvature(x, y)

    d0 = STEP_SOURCES["sgd"](model, Exp7Config.b1_mlp()).direction(cur, x, y)
    assert torch.equal(d0, -cur.flat_grad)


def test_adam_step_source_accumulates_state_along_the_trajectory() -> None:
    """The trajectory's second direction differs from what a fresh, state-free
    ``AdamStep`` gives the same gradient: the first/second moments carry forward
    (Kingma & Ba, 2015), this is not a per-step rescaling of that step's own gradient.
    """
    torch.manual_seed(0)
    model = SegmentedMLP(in_dim=8, hidden=8, depth=1, num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    estimator = ParamBlockEstimator(model, loss_fn)
    protocol = Exp7Config.b1_mlp()
    x1, y1 = torch.randn(4, 8), torch.randint(0, 3, (4,))
    x2, y2 = torch.randn(4, 8), torch.randint(0, 3, (4,))

    source = STEP_SOURCES["adam"](model, protocol)
    source.direction(estimator.curvature(x1, y1), x1, y1)  # step 1 seeds the moments
    cur2 = estimator.curvature(x2, y2)
    d0_2 = source.direction(cur2, x2, y2)

    stateless_d0_2 = STEP_SOURCES["adam"](model, protocol).direction(cur2, x2, y2)
    assert not torch.allclose(d0_2, stateless_d0_2)


def test_exp11_smoke_synthetic_sgd_and_adam_step_sources() -> None:
    """The full runner - CLI dispatch through the aggregate - completes end to end on
    the two first-order references, not just `StepSource.direction` in isolation."""
    base = Exp11Config.b1_mlp()
    for source in ("sgd", "adam"):
        cfg = replace(
            base,
            probe_steps=[0, 2],
            t_grid=[1e-2, 1.0],
            kink_t_grid=[1.0],
            n_lanczos_iter=3,
            step_source=source,
        )
        results = Exp11Runner(cfg, _DEVICE, loaders=_tiny_synthetic_loaders).run()
        json.dumps(results)
        assert results["meta"]["step_source"] == source
        payload = results["seeds"]["seed_42"]
        assert payload["n_steps"] == 3 == len(payload["loss_trajectory"])
        assert all(math.isfinite(loss) for loss in payload["loss_trajectory"])
        assert [p["step"] for p in payload["probes"]] == [0, 2]


# ======================================================================
# Multi-seed loader RNG isolation
# ======================================================================


class _PerturbedTrainDataset(Dataset):
    """8 examples shaped for the synthetic MLP task (64 in, 10 classes).

    ``__getitem__`` perturbs x with a ``torch.rand`` draw and a ``random.random()``
    draw - the same two independent RNG sources `_cars_transforms`'s
    RandomResizedCrop / RandomHorizontalFlip touch on the real Stanford Cars loader -
    so a worker's own RNG state, not just the sampler's permutation, shows up in the
    returned batch.
    """

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        x = torch.full((64,), float(idx)) + torch.rand(64) + random.random()
        return x, idx % 10


def _perturbed_loaders() -> tuple[DataLoader, DataLoader]:
    """A fresh pair every call (the `LoaderFactory` shape): a persistent, multi-worker,
    shuffled train loader configured like `get_stanford_cars_loaders` (``shuffle``,
    ``num_workers``, ``persistent_workers``, ``drop_last``) at a CPU-testable scale. The
    val loader is a plain, unshuffled, RNG-free stand-in (like the production eval
    split): only the train loader's reuse across seeds is under test.
    """
    train = DataLoader(
        _PerturbedTrainDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        drop_last=True,
    )
    val = DataLoader(
        TensorDataset(torch.zeros(4, 64), torch.zeros(4, dtype=torch.long)),
        batch_size=4,
    )
    return train, val


def test_reused_persistent_loader_leaks_rng_state_between_seeds() -> None:
    """Compare first batches from fresh and reused persistent multi-worker loaders.

    A fresh loader after seeding yields the same batch for seed 44 independently of
    seed 43. Reuse retains worker state and yields a different batch in this example.
    """

    def first_batch(loader: DataLoader) -> tuple[Tensor, Tensor]:
        idx, x = next(iter(loader))
        return idx.clone(), x.clone()

    # A fresh loader per seed.
    set_seed(44)
    solo_idx, solo_x = first_batch(_perturbed_loaders()[0])

    set_seed(43)
    next(iter(_perturbed_loaders()[0]))
    set_seed(44)
    fixed_idx, fixed_x = first_batch(_perturbed_loaders()[0])
    assert torch.equal(fixed_idx, solo_idx) and torch.equal(fixed_x, solo_x), (
        "fixed shape: seed 44's first batch must not depend on seed 43 having run"
    )

    # Reuse one persistent loader across both seeds.
    shared_loader, _ = _perturbed_loaders()
    set_seed(43)
    next(iter(shared_loader))
    set_seed(44)
    broken_idx, broken_x = first_batch(shared_loader)
    assert not (torch.equal(broken_idx, solo_idx) and torch.equal(broken_x, solo_x)), (
        "sanity check that this test would have caught the defect: reusing one "
        "persistent loader across seeds must reproduce it (seed 44 after seed 43 "
        "must differ from seed 44 run alone)"
    )


def _run_seeds_on_perturbed_loader(seeds: list[int]) -> dict[str, Any]:
    """`Exp11Runner.run()` on the synthetic MLP protocol with `_perturbed_loaders`,
    probing only step 0 with the plain ``sgd`` step source (d0 = -g, no state carried
    across steps to obscure the comparison) - the run touches exactly the first
    training batch of every seed.
    """
    base = Exp11Config.b1_mlp()
    cfg = replace(
        base,
        protocol=replace(
            base.protocol, training=replace(base.protocol.training, seeds=seeds)
        ),
        probe_steps=[0],
        t_grid=[1.0],
        kink_t_grid=[1.0],
        n_lanczos_iter=2,
        step_source="sgd",
    )
    return Exp11Runner(cfg, _DEVICE, loaders=_perturbed_loaders).run()


def test_exp11_runner_seed_44_unaffected_by_a_shared_persistent_loader() -> None:
    """Compare seed 44's step-0 loss and direction norm when run alone or after
    seed 43, using the perturbed persistent-worker loader through `Exp11Runner`.
    """
    solo = _run_seeds_on_perturbed_loader([44])["seeds"]["seed_44"]
    shared = _run_seeds_on_perturbed_loader([43, 44])["seeds"]["seed_44"]

    assert solo["loss_trajectory"][0] == shared["loss_trajectory"][0], (
        "seed 44's step-0 loss must not depend on whether seed 43 ran first"
    )
    assert (
        solo["probes"][0]["decomposition"]["norm_d0"]
        == shared["probes"][0]["decomposition"]["norm_d0"]
    ), "seed 44's step-0 gradient norm (d0 = -g for sgd) must likewise be unaffected"


# ======================================================================
# Runner smoke, aggregate, CLI
# ======================================================================


def test_exp11_smoke_synthetic() -> None:
    """Three K-FAC steps on the synthetic MLP with probes at steps 0 and 2."""
    base = Exp11Config.b1_mlp()
    cfg = replace(
        base,
        probe_steps=[2, 0],
        t_grid=[1.0, 1e-2, 0.1],
        kink_t_grid=[1.0, 0.1],
        n_lanczos_iter=3,
    )
    results = Exp11Runner(cfg, _DEVICE, loaders=_tiny_synthetic_loaders).run()
    json.dumps(results)  # JSON-serializable (NaN allowed as a float)

    assert set(results) == {"meta", "seeds", "aggregate"}
    meta = results["meta"]
    assert meta["model"] == "mlp" and meta["dataset"] == "synthetic"
    assert meta["damping"] == base.protocol.damping and meta["lr"] == 1.0
    assert meta["probe_steps"] == [2, 0] and meta["t_grid"] == [1e-2, 0.1, 1.0]
    assert meta["seeds"] == [42] and meta["batch_size"] == 256
    assert meta["step_source"] == "kfac"

    assert list(results["seeds"]) == ["seed_42"]
    payload = results["seeds"]["seed_42"]
    assert set(payload) == {"probes", "loss_trajectory", "val_acc", "n_steps"}
    assert payload["n_steps"] == 3 == len(payload["loss_trajectory"])
    assert len(payload["val_acc"]) == 1 and 0.0 <= payload["val_acc"][0] <= 1.0
    assert all(math.isfinite(loss) for loss in payload["loss_trajectory"])

    probes = payload["probes"]
    assert [p["step"] for p in probes] == [0, 2]
    for probe, step in zip(probes, (0, 2), strict=True):
        assert set(probe) == _PROBE_KEYS
        assert set(probe["decomposition"]) == _DECOMPOSITION_KEYS
        assert set(probe["sharpness"]) == _SHARPNESS_KEYS
        assert set(probe["payoff"]) == _PAYOFF_KEYS
        for curve in (probe["curve_d0"], probe["curve_vmax"]):
            assert [p["t"] for p in curve] == [1e-2, 0.1, 1.0]
            assert all(set(p) == _POINT_KEYS for p in curve)
        dec, pay = probe["decomposition"], probe["payoff"]
        assert pay["m_kfac"] == dec["m_d0"]
        assert math.isclose(
            dec["m_d0"],
            dec["g_dot_d0"] + 0.5 * (dec["d0_H_d0"] + dec["lambda_norm_d0_sq"]),
            rel_tol=1e-6,
        )
        assert math.isclose(
            pay["loss_d0"] - pay["loss_theta"],
            probe["curve_d0"][-1]["dL"],
            abs_tol=1e-9,
        ), "the t = 1 point of the curve is the step's loss change"
        assert math.isclose(
            pay["loss_theta"], payload["loss_trajectory"][step], rel_tol=1e-5
        ), "L(theta) on the step's minibatch, both ways"
        for p in probe["curve_d0"]:
            assert math.isclose(p["rel_err"], abs(p["dL"] - p["m"]) / abs(p["dL"]))
        assert math.isnan(probe["t_star"]) or probe["t_star"] in (1e-2, 0.1, 1.0)
        kinks = probe["kinks"]
        assert kinks["n_relu"] > 0 and kinks["n_pool"] == 0
        assert [r["t"] for r in kinks["rows"]] == [0.1, 1.0]
        assert all(0.0 <= r["relu_frac"] <= 1.0 for r in kinks["rows"])
        assert all(math.isnan(r["pool_frac"]) for r in kinks["rows"])
        assert probe["sharpness"]["lambda_max"] > 0.0

    agg = results["aggregate"]
    assert agg["n_seeds"] == 1 and [p["step"] for p in agg["probes"]] == [0, 2]
    first, seed_first = agg["probes"][0], probes[0]
    assert first["sharpness"]["lambda_max"] == {
        "mean": seed_first["sharpness"]["lambda_max"],
        "std": 0.0,
    }
    assert first["curve_d0"][0]["t"] == 1e-2 and "mean" in first["curve_d0"][0]["dL"]
    assert first["kinks"]["n_relu"] == seed_first["kinks"]["n_relu"], "a coordinate"


def test_exp11_meta_carries_the_environment_keys() -> None:
    """Flat metadata includes the live environment and protocol fields."""
    meta = Exp11Runner(Exp11Config.b1_mlp(), _DEVICE)._meta()
    env = environment_block()
    assert env, "a torch version is always reported"
    for key, value in env.items():
        assert meta[key] == value, key
    # Every field the runner already wrote is untouched.
    assert meta["model"] == "mlp" and meta["step_source"] == "kfac"
    assert meta["seeds"] == [42]


def test_aggregate_over_two_seeds() -> None:
    seeds = {
        "seed_42": {"probes": [_probe(0, 1.0, 0.01, 4), _probe(20, 2.0, 0.1, 8)]},
        "seed_43": {"probes": [_probe(0, 3.0, math.nan, 6), _probe(20, 2.0, 0.1, 8)]},
    }
    agg = _aggregate(seeds)
    assert agg["n_seeds"] == 2 and [p["step"] for p in agg["probes"]] == [0, 20]
    first = agg["probes"][0]
    assert first["decomposition"]["norm_d0"] == {"mean": 2.0, "std": 1.0}
    assert math.isnan(first["t_star"]["mean"]) and math.isnan(first["t_star"]["std"])
    assert first["curve_d0"] == [
        {
            "t": 0.01,
            "dL": {"mean": -2.0, "std": 1.0},
            "rel_err": {"mean": 0.0, "std": 0.0},
        }
    ]
    assert first["kinks"]["n_relu"] == 10 and first["kinks"]["n_pool"] == 0
    assert first["kinks"]["rows"] == [
        {"t": 1.0, "relu_flips": {"mean": 5.0, "std": 1.0}}
    ]
    assert agg["probes"][1]["t_star"] == {"mean": 0.1, "std": 0.0}


def test_exp11_config_profiles_and_cli_overrides() -> None:
    default = Exp11Config.b1_cars()
    assert default.protocol.training.seeds == [42], "exp11 probes one trajectory"
    assert Exp7Config.b1_cars().training.seeds == [42, 43, 44], "Exp. 7 untouched"
    assert default.protocol.model == "resnet50" and default.protocol.damping == 1.0
    assert default.protocol.training.batch_size == 64
    assert default.probe_steps == [0, 20, 60, 200, 600, 1500]
    assert Exp11Config.b1_mlp().protocol.model == "mlp"
    assert default.step_source == "kfac"

    parser = argparse.ArgumentParser()
    add_args(parser)
    assert build_config(parser.parse_args([])) == default

    cfg = build_config(
        parser.parse_args(
            "--profile b1_mlp --batch-size 8 --probe-steps 0 20 --seeds 1 2 "
            "--damping 0.5 --tolerance 0.2 --n-lanczos-iter 5 "
            "--step-source kfac".split()
        )
    )
    assert cfg.protocol.model == "mlp" and cfg.protocol.dataset == "synthetic"
    assert cfg.protocol.training.batch_size == 8 and cfg.protocol.training.seeds == [
        1,
        2,
    ]
    assert cfg.protocol.damping == 0.5
    assert cfg.probe_steps == [0, 20] and cfg.tolerance == 0.2
    assert cfg.n_lanczos_iter == 5
    assert cfg.t_grid == default.t_grid, "untouched default"
    assert cfg.step_source == "kfac"
