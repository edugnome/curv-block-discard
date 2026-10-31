"""Exp7 runner wiring: second-order step hooks, the Newton-CG oracle, and metrics.

White-box, mostly dataset-free checks that the COUPLE-FAC / Newton-CG step hooks compose
the provider, the parameter-space estimator, the overlay, and the safeguard into a finite
in-place update on a real ResNet-18, that the truncated-CG oracle solves a known damped
system, that the payoff bookkeeping (HVP counter, cross-term ratios, per-seed result
schema) is exact, and that the backbone dispatch and speed-to-target metrics behave. The
mathematical correctness of the overlay and the safeguard is covered by their own tests;
here we verify the integration seam - including that the run's one damping (the
Levenberg-Marquardt-adapted `Damping`) is the value the provider, the overlay, the
safeguard and the oracle all read, and that its trajectory is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import experiments.runner_exp7 as runner_exp7  # noqa: E402
from experiments.config import Exp7Config  # noqa: E402
from experiments.models import (  # noqa: E402
    SegmentedMLP,
    SegmentedResNet18,
    SegmentedResNet50,
)
from experiments.param_ops import group_major_params  # noqa: E402
from experiments.runner_exp7 import (  # noqa: E402
    Exp7Runner,
    MethodTrace,
    _cross_term_ratios,
    _damping_of,
    _epochs_to_target,
    _newton_cg,
    _record_payoff,
    add_args,
    aggregate_method,
    build_config,
    build_couplefac_hook,
    build_model,
    build_newton_cg_hook,
)
from experiments.training import StepCtx  # noqa: E402
from hessian import ParamBlockEstimator, ParamGroupedModel  # noqa: E402
from secondorder import CoupleFacOptimizer, CoupleFacOverlay, Damping  # noqa: E402
from secondorder.damping import LM_RHO_HIGH, LM_RHO_LOW  # noqa: E402
from secondorder.kfac_adapter import CurvlinopsKFACProvider  # noqa: E402
from secondorder.safeguard import model_value  # noqa: E402
from tests.test_secondorder import (  # noqa: E402
    _build_quadratic,
    _coupling,
    _ExactKFAC,
    _hvp_of,
)

_DEVICE = torch.device("cpu")


def _tiny_cfg() -> Exp7Config:
    """An Exp7 config with the cheapest curvature knobs (for fast CPU tests)."""
    return Exp7Config(
        coupling_n_probes=1,
        overlay_rank=1,
        overlay_n_power_iter=1,
        measure_period=1,
        damping=1e-1,
    )


def _resnet_batch(seed: int = 0) -> tuple[SegmentedResNet18, Tensor, Tensor]:
    """A real ResNet-18 (`ParamGroupedModel`) and a CIFAR-shaped classification minibatch.

    This is the Exp7 headline regime - a deep convolutional backbone in float32 - so
    the step hooks are exercised against the architecture and dtype the real run uses.
    Factored Tikhonov damping keeps the K-FAC factor inversion stable here, where the
    raw-factor eigendecomposition of exact damping fails to converge in float32.
    """
    torch.manual_seed(seed)
    model = SegmentedResNet18(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    return model, x, y


def _flat(model: ParamGroupedModel) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in group_major_params(model)])


def _assert_hook_updates(
    model: ParamGroupedModel, ctx: StepCtx, hook: Callable[[StepCtx], None]
) -> None:
    before = _flat(model).clone()
    hook(ctx)
    after = _flat(model)
    assert torch.isfinite(after).all(), "update produced non-finite parameters"
    assert not torch.allclose(before, after), "hook did not change any parameter"


def _assert_payoff_recorded(trace: MethodTrace, model: ParamGroupedModel) -> None:
    """One step at measure_period = 1: a full payoff record on top of the counted HVPs."""
    n_blocks = len(model.get_param_groups())
    assert trace.n_hvp > 0, "every exact HVP must be counted"
    assert len(trace.model_decrease) == 1 and len(trace.cross_term) == 1
    entry = trace.model_decrease[0]
    assert entry["step"] == 0 and math.isfinite(entry["m_newton"])
    assert math.isfinite(entry["m_kfac"]) and math.isfinite(entry["m_overlay"])
    assert entry["m_overlay"] <= entry["m_kfac"] + 1e-6, "no-harm on the record"
    assert len(trace.cross_term[0]["pairs"]) == n_blocks * (n_blocks - 1)
    assert len(trace.cross_term[0]["denom"]) == n_blocks


# What the ``newton_cg`` method records on a payoff step (its own Newton step is the
# base step the LM rule judges).
_NEWTON_RECORD_KEYS = {
    "step",
    "damping",
    "rho",
    "m_newton",
    "loss_theta",
    "loss_newton",
    "n_hvp_newton",
    "newton_negative_curvature",
}


def _mlp_batch(seed: int = 0) -> tuple[SegmentedMLP, Tensor, Tensor]:
    """A full-rank MLP (`ParamGroupedModel`) + low-dim batch - the clean EKFAC regime.

    Low input dimension and batch >= width keep every Kronecker factor full rank, so
    EKFAC's eigenbasis is well posed - the setting where the optimizer-agnostic overlay is
    demonstrated alongside the (rank-deficient) convolutional headline.
    """
    torch.manual_seed(seed)
    model = SegmentedMLP(in_dim=16, hidden=16, depth=2, num_classes=4).to(_DEVICE)
    x = torch.randn(32, 16, device=_DEVICE)
    y = torch.randint(0, 4, (32,), device=_DEVICE)
    return model, x, y


def test_newton_cg_solves_damped_system() -> None:
    torch.manual_seed(0)
    n = 12
    root = torch.randn(n, n)
    a = root @ root.t() + torch.eye(n)  # SPD
    damping = 0.1
    grad = torch.randn(n)

    solve = _newton_cg(
        grad, lambda v: a @ v, damping=damping, max_iter=n + 10, tol=1e-12
    )
    expected = torch.linalg.solve(a + damping * torch.eye(n), -grad)
    assert torch.allclose(solve.direction, expected, atol=1e-4), (
        f"Newton-CG residual {(solve.direction - expected).norm().item():.2e}"
    )
    assert not solve.negative_curvature and 1 <= solve.n_iter <= n + 10


def test_newton_cg_reports_negative_curvature() -> None:
    """On an indefinite damped system CG stops at p^T A p <= 0 and says so."""
    n = 6
    a = -torch.eye(n)  # every direction has curvature -1 < -damping
    grad = torch.ones(n)
    solve = _newton_cg(grad, lambda v: a @ v, damping=0.1, max_iter=20)
    assert solve.negative_curvature and solve.n_iter == 1
    assert torch.equal(solve.direction, -grad), "first-step exit: steepest descent"


def test_couplefac_hook_kfac_updates() -> None:
    model, x, y = _resnet_batch()
    trace = MethodTrace()
    hook = build_couplefac_hook(
        model, _tiny_cfg(), tau=float("inf"), seed=0, device=_DEVICE, trace=trace
    )
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)
    _assert_payoff_recorded(trace, model)
    assert trace.n_proposed == trace.n_accepted == trace.n_reject == 0, "no verdicts"
    entry = trace.model_decrease[0]
    assert entry["accepted"] and entry["n_pairs"] == 0
    assert entry["m_overlay"] == entry["m_kfac"], "tau = inf: the step is d0 itself"
    assert all(
        math.isnan(pair["coupling"]) for pair in trace.cross_term[0]["pairs"].values()
    ), "no coupling is measured when the overlay is off"


def test_couplefac_hook_overlay_updates() -> None:
    model, x, y = _resnet_batch()
    cfg = _tiny_cfg()
    trace = MethodTrace()
    hook = build_couplefac_hook(
        model, cfg, tau=cfg.tau, seed=0, device=_DEVICE, trace=trace
    )
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)
    _assert_payoff_recorded(trace, model)
    entry = trace.model_decrease[0]
    assert entry["n_pairs"] > 0, "tau = 0.1 on a ResNet-18 selects a coupled pair"
    assert trace.n_proposed == 1, "one verdict: a correction was proposed"
    assert trace.n_accepted + trace.n_reject == trace.n_proposed
    assert entry["accepted"] == (trace.n_accepted == 1)
    assert all(
        math.isfinite(pair["coupling"])
        for pair in trace.cross_term[0]["pairs"].values()
    ), "the coupling that gated selection is recorded next to eps"


def test_couplefac_hook_ekfac_updates() -> None:
    """EKFAC base step (overlay off) on the full-rank MLP: a finite in-place update."""
    model, x, y = _mlp_batch()
    trace = MethodTrace()
    hook = build_couplefac_hook(
        model,
        _tiny_cfg(),
        tau=float("inf"),
        kind="ekfac",
        seed=0,
        device=_DEVICE,
        trace=trace,
    )
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)
    _assert_payoff_recorded(trace, model)
    assert trace.n_refresh_failed == 0, "a full-rank eigenbasis converges"


def test_couplefac_hook_ekfac_overlay_updates() -> None:
    """EKFAC + coupling-gated overlay on the full-rank MLP (the agnostic-overlay path)."""
    model, x, y = _mlp_batch()
    cfg = _tiny_cfg()
    trace = MethodTrace()
    hook = build_couplefac_hook(
        model, cfg, tau=cfg.tau, kind="ekfac", seed=0, device=_DEVICE, trace=trace
    )
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)
    _assert_payoff_recorded(trace, model)


def test_newton_cg_hook_updates() -> None:
    model, x, y = _resnet_batch()
    trace = MethodTrace()
    hook = build_newton_cg_hook(model, _tiny_cfg(), device=_DEVICE, trace=trace)
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)
    assert trace.n_hvp > 0 and trace.cross_term == []
    (entry,) = trace.model_decrease
    assert set(entry) == _NEWTON_RECORD_KEYS
    assert entry["step"] == 0 and math.isfinite(entry["m_newton"])
    assert 1 <= entry["n_hvp_newton"] <= trace.n_hvp - 1, "the solve, then m_newton"
    assert trace.n_newton_solves == 1
    assert trace.n_newton_negative_curvature == int(entry["newton_negative_curvature"])
    assert entry["damping"] == _tiny_cfg().damping, "taken with lambda_0"
    assert math.isclose(
        entry["rho"], (entry["loss_theta"] - entry["loss_newton"]) / -entry["m_newton"]
    ), "rho of the Newton step itself"
    assert trace.damping_trajectory[0][0] == 0, "one LM update, at the record"


def test_hvp_counter_matches_oracle_calls() -> None:
    """`MethodTrace.counting` sees every product the optimizer issues; none go around."""
    gen = torch.Generator().manual_seed(21)
    g, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.7, gen=gen)
    calls: list[int] = []

    def oracle(u: Tensor) -> Tensor:
        calls.append(1)
        return h @ u

    opt = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges),
        overlay=CoupleFacOverlay(rank=2, n_power_iter=4),
        tau=0.0,
    )
    trace = MethodTrace()
    opt.compute_direction(
        g, trace.counting(oracle), _coupling(h, ranges), refresh=True, generator=gen
    )
    assert trace.n_hvp == len(calls) > 2, f"counter {trace.n_hvp} vs {len(calls)}"

    # empty selection: H d0 only - d0 itself is accepted without a second HVP
    calls.clear()
    trace = MethodTrace()
    report = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges), tau=math.inf
    ).compute_direction(
        g, trace.counting(oracle), _coupling(h, ranges), refresh=True, generator=gen
    )
    assert trace.n_hvp == len(calls) == 1
    assert report.decision.accepted and report.n_selected_pairs == 0


def test_cross_term_ratios_match_dense() -> None:
    """eps(v, w) = ||H[s_v, s_w] d0[s_w]|| / ||H[s_v, s_v] d0[s_v]|| on the explicit H."""
    gen = torch.Generator().manual_seed(22)
    g, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.5, gen=gen)
    d0 = _ExactKFAC(h, ranges).base_direction(g)
    ratios, denoms = _cross_term_ratios(_hvp_of(h), d0, ranges)

    names = list(ranges)
    assert set(ratios) == {(v, w) for v in names for w in names if v != w}
    assert set(denoms) == set(names)
    for (v, w), eps in ratios.items():
        sv, sw = ranges[v], ranges[w]
        expected = (h[sv, sw] @ d0[sw]).norm().item() / (
            h[sv, sv] @ d0[sv]
        ).norm().item()
        assert abs(eps - expected) < 1e-6, f"eps({v},{w}) = {eps} != {expected}"
    for v, denom in denoms.items():
        sv = ranges[v]
        expected = (h[sv, sv] @ d0[sv]).norm().item()
        assert abs(denom - expected) < 1e-6, f"denom({v}) = {denom} != {expected}"

    zero = torch.zeros_like(d0)  # vanishing diagonal action -> undefined ratio
    zero_ratios, zero_denoms = _cross_term_ratios(_hvp_of(h), zero, ranges)
    assert all(math.isnan(eps) for eps in zero_ratios.values())
    assert all(denom == 0.0 for denom in zero_denoms.values())


def test_exp7_smoke_synthetic() -> None:
    """A 25-step exp7 on the synthetic MLP task produces the full per-seed schema."""
    base = Exp7Config.b1_mlp()
    cfg = replace(
        base,
        methods=["kfac", "kfac+overlay", "newton_cg", "sgd"],
        measure_period=5,
        coupling_n_probes=1,
        overlay_rank=1,
        overlay_n_power_iter=1,
        hessian_batch_size=8,
        training=replace(base.training, epochs=1, max_steps=25, seeds=[42]),
    )
    results = Exp7Runner(cfg, _DEVICE).run()
    json.dumps(results)  # JSON-serializable (NaN allowed as a float)

    expected_keys = {
        "val_acc",
        "val_loss",
        "final_acc",
        "epochs_to_target",
        "n_steps",
        "hessian_batch_size",
        "wall_sec_per_epoch",
        "hvp_per_epoch",
        "safeguard",
        "newton_cg",
        "model_decrease",
        "cross_term",
        "damping_trajectory",
        "phi",
    }
    for method in cfg.methods:
        payload = results[method]["seeds"]["seed_42"]
        assert expected_keys <= set(payload), f"{method}: missing keys"
        assert payload["n_steps"] == 25 and payload["hessian_batch_size"] == 8
        assert len(payload["val_acc"]) == 1 == len(payload["wall_sec_per_epoch"])
        assert payload["wall_sec_per_epoch"][0] > 0.0
        assert set(payload["phi"]) == {"init", "mid", "final"}
        assert all(
            set(report) == {"phi", "diag_frob", "coupling", "group_sizes"}
            for report in payload["phi"].values()
        )
        assert set(payload["safeguard"]) == {
            "accept_rate_per_epoch",
            "n_proposed",
            "n_reject",
            "n_reject_by",
            "n_refresh_failed",
        }
        assert payload["safeguard"]["n_proposed"] >= payload["safeguard"]["n_reject"]
        assert set(payload["safeguard"]["n_reject_by"]) == {"ratio", "model", "loss"}
        assert (
            sum(payload["safeguard"]["n_reject_by"].values())
            == payload["safeguard"]["n_reject"]
        )

    n_blocks = len(build_model(cfg).get_param_groups())
    overlay = results["kfac+overlay"]["seeds"]["seed_42"]
    assert [e["step"] for e in overlay["model_decrease"]] == [0, 5, 10, 15, 20]
    assert [e["step"] for e in overlay["cross_term"]] == [0, 5, 10, 15, 20]
    assert all(
        len(e["pairs"]) == n_blocks * (n_blocks - 1) for e in overlay["cross_term"]
    )
    assert all(len(e["denom"]) == n_blocks for e in overlay["cross_term"])
    assert overlay["hvp_per_epoch"][0] > 0
    n_proposed = overlay["safeguard"]["n_proposed"]
    assert n_proposed > 0, "the overlay proposed corrections on the synthetic MLP"
    # the cache built on a refresh step (T = 5) serves that step and the next four
    refreshes_with_pairs = sum(1 for e in overlay["model_decrease"] if e["n_pairs"] > 0)
    assert n_proposed == 5 * refreshes_with_pairs, "one proposal per cached step"
    assert math.isfinite(overlay["safeguard"]["accept_rate_per_epoch"][0])
    assert 0.0 <= overlay["safeguard"]["accept_rate_per_epoch"][0] <= 1.0

    kfac = results["kfac"]["seeds"]["seed_42"]
    assert len(kfac["model_decrease"]) == 5
    assert kfac["safeguard"]["n_proposed"] == 0, "no overlay: nothing proposed"
    assert math.isnan(kfac["safeguard"]["accept_rate_per_epoch"][0])

    newton = results["newton_cg"]["seeds"]["seed_42"]
    assert newton["cross_term"] == []
    assert newton["newton_cg"]["n_solves"] == 25
    assert 0 <= newton["newton_cg"]["n_negative_curvature"] <= 25
    assert [set(e) for e in newton["model_decrease"]] == [_NEWTON_RECORD_KEYS] * 5

    sgd = results["sgd"]["seeds"]["seed_42"]
    assert sgd["model_decrease"] == [] and sgd["cross_term"] == []
    assert sgd["damping_trajectory"] == [], "no damping in a first-order method"
    assert sgd["hvp_per_epoch"] == [0]
    assert math.isnan(sgd["safeguard"]["accept_rate_per_epoch"][0])
    assert sgd["safeguard"]["n_proposed"] == 0 == sgd["safeguard"]["n_reject"]
    assert sgd["safeguard"]["n_refresh_failed"] == 0


def _tiny_synthetic_pair() -> tuple[DataLoader, DataLoader]:
    """A fresh tiny (train, val) pair each call, shaped for `Exp7Config.b1_mlp()`."""
    x, y = torch.randn(8, 64), torch.randint(0, 10, (8,))
    return (
        DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, drop_last=True),
        DataLoader(TensorDataset(x, y), batch_size=4),
    )


def test_exp7_rebuilds_the_train_loader_once_per_seed() -> None:
    """Check loader construction count and identity for the validation batch and
    each seed. Separate training loaders isolate persistent-worker RNG state.
    """
    calls: list[tuple[DataLoader, DataLoader]] = []

    def fake_build_loaders(cfg: Exp7Config) -> tuple[DataLoader, DataLoader]:
        pair = _tiny_synthetic_pair()
        calls.append(pair)
        return pair

    with patch.object(runner_exp7, "build_loaders", fake_build_loaders):
        base = Exp7Config.b1_mlp()
        cfg = replace(
            base,
            methods=["sgd"],
            coupling_n_probes=1,
            hessian_batch_size=4,
            training=replace(base.training, epochs=1, max_steps=1, seeds=[43, 44]),
        )
        Exp7Runner(cfg, _DEVICE).run()

    assert len(calls) == 3, (
        "one `build_loaders` call for the shared val loader / phi batch, plus one per "
        f"seed - got {len(calls)}"
    )
    seed_43_loader, seed_44_loader = calls[1][0], calls[2][0]
    assert seed_43_loader is not seed_44_loader, (
        "seed 43 and seed 44 must not share the same train DataLoader object"
    )


def test_exp7_overlay_stable_on_synthetic() -> None:
    """60 steps of K-FAC vs COUPLE-FAC on the synthetic MLP: finite, inside the region.

    Check finite losses and recorded model values and norms, and that each accepted
    correction stays within tr_ratio times the base-step norm. The synthetic epoch is
    32 steps, so the 60-step budget spans two epochs.
    """
    base = Exp7Config.b1_mlp()
    cfg = replace(
        base,
        methods=["kfac", "kfac+overlay"],
        measure_period=10,
        coupling_n_probes_online=2,
        coupling_n_probes=1,
        overlay_rank=2,
        overlay_n_power_iter=2,
        hessian_batch_size=8,
        training=replace(base.training, epochs=2, max_steps=60, seeds=[42]),
    )
    results = Exp7Runner(cfg, _DEVICE).run()

    for method in cfg.methods:
        payload = results[method]["seeds"]["seed_42"]
        assert payload["n_steps"] == 60 and len(payload["val_loss"]) == 2
        assert all(math.isfinite(loss) for loss in payload["val_loss"]), method
        assert [e["step"] for e in payload["model_decrease"]] == [0, 10, 20, 30, 40, 50]
        for entry in payload["model_decrease"]:
            for key in ("m_kfac", "m_overlay", "m_newton", "norm_d0", "norm_d1"):
                assert math.isfinite(entry[key]), (
                    f"{method} step {entry['step']}: {key}"
                )
            assert entry["m_overlay"] <= entry["m_kfac"] + 1e-6, "no-harm on the record"
            if entry["accepted"]:
                assert entry["norm_d1"] <= cfg.tr_ratio * entry["norm_d0"] + 1e-9
                assert entry["rejected_by"] is None
                if entry["n_pairs"] > 0:  # a real proposal: the loss test ran
                    assert entry["loss_d1"] <= entry["loss_d0"], "no-harm on the loss"
            else:
                assert entry["rejected_by"] in {"ratio", "model", "loss"}

    overlay = results["kfac+overlay"]["seeds"]["seed_42"]
    assert overlay["safeguard"]["n_proposed"] == 60, "a cached pair on every step"
    assert all(e["n_pairs"] > 0 for e in overlay["model_decrease"])


def test_damping_shared_by_provider_overlay_and_oracle() -> None:
    """One `Damping`: provider (at refresh), overlay, safeguard and oracle read one lambda.

    The value is moved by hand between three refresh steps on the full-rank MLP. At each
    step the provider's d0 equals that of a fresh provider built with the plain float
    (the inverse is rebuilt with the lambda in force, not the one at construction), the
    overlay's correction equals a fresh float-damped overlay's on the same cached
    subspace, the safeguard's m(d0) is the model damped by that lambda, and the payoff
    record carries it as ``damping`` with m_newton solved under it.
    """
    model, x, y = _mlp_batch()
    loss_fn = nn.CrossEntropyLoss()
    damping = Damping.fixed(0.1)  # frozen: moved by hand, so the trajectory is known
    provider = CurvlinopsKFACProvider(model, loss_fn, damping=damping, refresh_period=1)
    overlay = CoupleFacOverlay(rank=1, n_power_iter=1, damping=damping)
    optimizer = CoupleFacOptimizer(
        provider=provider, overlay=overlay, tau=0.0, params=group_major_params(model)
    )
    estimator = ParamBlockEstimator(model, loss_fn, n_probes=1)
    trace = MethodTrace()
    gen = torch.Generator().manual_seed(0)
    lambdas = (0.1, 0.4, 0.05)
    for step, lam in enumerate(lambdas):
        damping.value = lam
        provider.update(x, y)  # a refresh: rebuilt with the lambda now in force
        cur = estimator.curvature(x, y)
        g = cur.flat_grad
        report = optimizer.compute_direction(
            g, cur.hvp, cur.coupling(), refresh=True, generator=gen
        )
        fresh = CurvlinopsKFACProvider(model, loss_fn, damping=lam).update(x, y)
        assert torch.allclose(report.d0, fresh.base_direction(g)), (
            f"provider at lambda = {lam}"
        )
        hd0 = cur.hvp(report.d0)
        subspace = optimizer._subspace
        assert subspace is not None and report.n_selected_pairs > 0
        d1 = overlay.correct(g, report.d0, hd0, subspace)
        d1_float = CoupleFacOverlay(rank=1, n_power_iter=1, damping=lam).correct(
            g, report.d0, hd0, subspace
        )
        assert torch.equal(d1, d1_float), f"overlay at lambda = {lam}"
        assert math.isclose(
            report.decision.m0,
            model_value(g, report.d0, hd0, damping=lam),
            rel_tol=1e-6,
        ), f"safeguard at lambda = {lam}"
        _record_payoff(
            trace,
            step,
            g,
            cur.hvp,
            report,
            provider.block_ranges(),
            {},
            damping=damping,
            loss_at=lambda direction: 0.0,
        )
        entry = trace.model_decrease[-1]
        assert entry["damping"] == lam
        d_newton = _newton_cg(g, cur.hvp, damping=lam, max_iter=20).direction
        assert math.isclose(
            entry["m_newton"],
            model_value(g, d_newton, cur.hvp(d_newton), damping=lam),
            rel_tol=1e-6,
        ), f"oracle at lambda = {lam}"
        del cur
    assert [lam for _, lam in trace.damping_trajectory] == list(lambdas)


def test_exp7_damping_adapts_on_synthetic() -> None:
    """The LM rule moves the shared lambda over a 25-step synthetic run and records it.

    Every payoff record carries the lambda it was taken with and its rho (the recorded
    losses over the recorded model value); each trajectory entry is the lambda the rule
    set at that record - by the factor, or a clamp, in the direction the rule prescribes
    from the recorded (m0, rho) - and is the lambda the next record is taken with. The
    default fixed-lambda protocol (``damping_adapt=False``) keeps lambda_0 on every
    record while still reporting rho.
    """
    base = Exp7Config.b1_mlp()
    cfg = replace(
        base,
        methods=["kfac+overlay", "newton_cg"],
        damping_adapt=True,
        measure_period=5,
        coupling_n_probes=1,
        overlay_rank=1,
        overlay_n_power_iter=1,
        hessian_batch_size=8,
        training=replace(base.training, epochs=1, max_steps=25, seeds=[42]),
    )
    results = Exp7Runner(cfg, _DEVICE).run()
    low, high = cfg.damping_bounds

    for method in cfg.methods:
        payload = results[method]["seeds"]["seed_42"]
        records = payload["model_decrease"]
        trajectory = payload["damping_trajectory"]
        assert [step for step, _ in trajectory] == [e["step"] for e in records]
        assert [e["step"] for e in records] == [0, 5, 10, 15, 20]
        assert records[0]["damping"] == cfg.damping, "the first record: lambda_0"
        newton = method == "newton_cg"
        m0_key, after_key = (
            ("m_newton", "loss_newton") if newton else ("m_kfac", "loss_d0")
        )
        for entry, (_, after) in zip(records, trajectory, strict=True):
            lam, rho, m0 = entry["damping"], entry["rho"], entry[m0_key]
            assert math.isclose(
                rho, (entry["loss_theta"] - entry[after_key]) / -m0, rel_tol=1e-9
            ), "rho is the recorded actual over the recorded predicted reduction"
            if m0 >= 0.0 or math.isnan(rho) or rho < LM_RHO_LOW:
                expected = min(lam * cfg.damping_factor, high)
            elif rho > LM_RHO_HIGH:
                expected = max(lam / cfg.damping_factor, low)
            else:
                expected = lam
            assert math.isclose(after, expected), (
                f"{method} step {entry['step']}: {lam} -> {after} at rho = {rho}"
            )
        for (_, previous), entry in zip(trajectory[:-1], records[1:], strict=True):
            assert entry["damping"] == previous, "set at a record, in force at the next"
        assert len({e["damping"] for e in records}) > 1, (
            f"{method}: the rule never moved lambda"
        )

    frozen = replace(
        cfg,
        methods=["kfac"],
        damping_adapt=False,
        training=replace(cfg.training, max_steps=15),
    )
    payload = Exp7Runner(frozen, _DEVICE).run()["kfac"]["seeds"]["seed_42"]
    assert [e["damping"] for e in payload["model_decrease"]] == [cfg.damping] * 3
    assert payload["damping_trajectory"] == [(s, cfg.damping) for s in (0, 5, 10)]
    assert all(math.isfinite(e["rho"]) for e in payload["model_decrease"])


def test_damping_adapt_flag_turns_the_rule_on() -> None:
    """``--damping`` sets lambda_0, fixed by default; ``--damping-adapt`` opts in."""
    parser = argparse.ArgumentParser()
    add_args(parser)
    frozen = build_config(
        parser.parse_args(["--profile", "b1_mlp", "--damping", "0.5"])
    )
    assert frozen.damping == 0.5 and not frozen.damping_adapt
    assert _damping_of(frozen) == Damping.fixed(0.5)
    adaptive = build_config(
        parser.parse_args(
            ["--profile", "b1_mlp", "--damping", "0.5", "--damping-adapt"]
        )
    )
    assert adaptive.damping == 0.5 and adaptive.damping_adapt
    assert _damping_of(adaptive) == Damping(0.5)


def test_group_major_params_is_group_major() -> None:
    model = SegmentedResNet18(num_classes=10)
    expected = [p for params in model.get_param_groups().values() for p in params]
    flat = group_major_params(model)
    assert len(flat) == len(expected) and all(
        a is b for a, b in zip(flat, expected, strict=True)
    ), "params must follow get_param_groups (group-major) order"
    assert sum(p.numel() for p in flat) == sum(p.numel() for p in model.parameters())


def test_build_model_dispatch() -> None:
    m50 = build_model(Exp7Config(model="resnet50", num_classes=10, pretrained=False))
    assert isinstance(m50, SegmentedResNet50), (
        "resnet50 key must build SegmentedResNet50"
    )

    m18 = build_model(Exp7Config(model="resnet18", num_classes=10))
    assert isinstance(m18, SegmentedResNet18) and not isinstance(m18, SegmentedResNet50)

    mlp = build_model(Exp7Config(model="mlp", num_classes=10))
    assert isinstance(mlp, SegmentedMLP), "mlp key must build SegmentedMLP"

    raised = False
    try:
        build_model(Exp7Config(model="vit"))
    except ValueError:
        raised = True
    assert raised, "unknown backbone must raise ValueError"


def test_epochs_to_target_and_aggregate() -> None:
    assert _epochs_to_target([0.1, 0.5, 0.9, 0.95], 0.9) == 3, (
        "first crossing at epoch 3"
    )
    assert _epochs_to_target([0.1, 0.2], 0.9) is None, "never reaching target -> None"

    per_seed = {
        "seed_1": {"final_acc": 0.8, "epochs_to_target": 3},
        "seed_2": {"final_acc": 0.9, "epochs_to_target": None},
    }
    agg = aggregate_method(per_seed, target=0.9)
    assert abs(agg["final_acc_mean"] - 0.85) < 1e-9, "final-acc mean"
    assert agg["n_reached_target"] == 1, "one seed reached target"
    assert agg["epochs_to_target_mean"] == 3.0, "mean over reached seeds only"
    assert agg["n_seeds"] == 2
