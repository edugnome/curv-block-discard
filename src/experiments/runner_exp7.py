"""Exp7: compare first- and second-order optimizers on classification tasks.

This runner *uses* the inter-layer curvature the rest of the project *measures*: it
trains one backbone with several optimizers and compares them by how fast they reach a
target validation accuracy and by their final accuracy. The comparison set is

  - ``sgd`` / ``adam``: first-order references (the built-in Trainer optimizer);
  - ``kfac``: the block-diagonal K-FAC Newton step (overlay disabled);
  - ``kfac+overlay``: COUPLE-FAC - K-FAC plus the coupling-gated, trust-region
                            safeguarded rank-r cross-block correction (the method);
  - ``newton_cg``: a truncated-CG reference using exact Hessian-vector products
                   for the damped system; it is not a performance upper bound.

The three second-order methods all run through the `Trainer` step-hook seam:
each step builds the exact parameter-space bundle (gradient, HVP, deferred inter-block
coupling) with `hessian.param_space.ParamBlockEstimator` and applies one accepted
update. ``kfac`` and ``kfac+overlay`` differ only in the selection threshold tau
(tau = +inf disables the overlay, so S = {C_{vw} > tau} is empty and d = d_KFAC);
``newton_cg`` approximately solves (H + lambda I) d = -g by truncated CG. The method is amortized over two
refresh periods (see `secondorder.overlay`, `secondorder.kfac_adapter`):

  - T   = ``measure_period``: the overlay refresh period. On the steps with
          t mod T == 0 the coupling C is re-measured (``coupling_n_probes_online``
          Hutchinson probes; selection also uses their threshold), the pairs are re-selected,
          and the joint subspace U and its reduced Hessian are rebuilt; in between the
          overlay solves on the cached systems with the fresh gradient and H d0;
  - T_k = ``kfac_refresh_period``: the K-FAC factors and their inverse are rebuilt
          every T_k steps (EKFAC: ``ekfac_refresh_period``); the gradient they
          precondition is fresh every step.

The safeguard compares proposals on the damped minibatch quadratic and optionally the
minibatch loss. For finite values it accepts a proposal only within the model tolerance
and the relative norm bound; this is not a guarantee of improved training or test accuracy.

The Tikhonov damping lambda is one value per run (`secondorder.Damping`, lambda_0 =
``damping``), shared by reference between the K-FAC / EKFAC inverse (read at its
refresh), the overlay's reduced solve, the safeguard's damped model and the Newton-CG
oracle; with ``damping_adapt`` it is adapted at every payoff record by the
Levenberg-Marquardt rule on the base step's actual-to-predicted reduction
rho = (L(theta) - L(theta + d0)) / (-m(d0)). Adaptation does not guarantee model
validity (off by default: fixed lambda_0, rho still recorded). ``newton_cg`` adapts on its own Newton step; ``sgd`` / ``adam`` have
no damping.

Beyond the accuracy curves, every seed records the *payoff* of the correction on the
damped local quadratic model of the current minibatch,
m(d) = g^T d + 1/2 d^T (H + lambda I) d:

  - every T steps (the overlay refresh steps, step 0 included), on the step's own
    minibatch and at the pre-update weights: m(d_KFAC), m(d_overlay) and m(d_Newton) -
    the model values of the block-diagonal step, the safeguarded corrected step,
    and the truncated-CG damped reference - together with the cross-term ratio
        eps(v, w) = ||H_vw d0_w|| / ||H_vv d0_v||,
    the inter-block curvature the block-diagonal step ignores relative to the diagonal
    curvature it accounts for, next to the coupling C(v, w) that gated pair selection;
  - on the same records: the ``damping`` in force and the ratio ``rho`` the LM rule
    judged it on, and per seed the ``damping_trajectory`` of (step, lambda) after each
    update;
  - per epoch: the training wall-clock, the number of exact HVPs, the safeguard's
    accept rate over the steps on which a correction was proposed (plus the totals of
    proposals, rejections and failed EKFAC eigenbasis refreshes);
  - Phi, the discarded-coupling fraction, on one fixed validation minibatch at init /
    mid / final - for every method, the diagnosis alongside the repair.

The payoff records and Phi run under `deterministic_mode`, whatever kernels training
uses. The online coupling field does not: C is re-measured every T steps on the kernels
training uses, and its Hutchinson probes come from the global generator, not from the
per-seed one (that one drives the overlay's subspace iteration).

All methods share one fixed train/val split and the same backbone factory, so the only
varying factor is the optimizer; ``training.seeds`` controls model init and minibatch
order. The result bundle is
JSON-serializable:
per-seed curves and traces plus mean/std final accuracy and mean epochs-to-target.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.cli import Experiment, add_device_args, override
from experiments.config import Exp7Config, TrainingConfig
from experiments.data import (
    DATASET_NUM_CLASSES,
    get_cifar10_image_loaders,
    get_cifar100_loaders,
    get_imagenet32_loaders,
    get_stanford_cars_loaders,
    get_synthetic_classification_loaders,
)
from experiments.models import SegmentedMLP, SegmentedResNet18, SegmentedResNet50
from experiments.param_ops import (
    SECOND_ORDER_LR,
    apply_flat,
    group_major_params,
    trial_loss,
)
from experiments.phi import measure_phi
from experiments.protocol import sample_hessian_batch
from experiments.provenance import META_KEY, provenance
from experiments.training import StepCtx, Trainer
from experiments.utils import deterministic_mode, set_seed
from hessian import ParamBlockEstimator, ParamGroupedModel
from hessian.memory import run_peak_allocated
from hessian.types import LayerID
from secondorder import (
    CoupleFacOptimizer,
    CoupleFacOverlay,
    Damping,
    HessianOracle,
    Pair,
    StepReport,
)
from secondorder.kfac_adapter import CurvlinopsEKFACProvider, CurvlinopsKFACProvider
from secondorder.safeguard import LossAt, model_value

logger = logging.getLogger(__name__)

# Reference first-order Adam learning rate (the Trainer config carries the SGD lr).
_ADAM_LR = 1e-3
# Truncated-CG iteration budget for the Newton-CG oracle (HVPs per step).
_NEWTON_CG_ITERS = 20
# Fixed seed for the train/val split, so every method and seed sees the same data.
_SPLIT_SEED = 0
# Input dimension of the synthetic full-rank MLP task (`b1_mlp`); low enough that every
# Kronecker factor is full rank at the configured batch, so EKFAC is well conditioned.
_MLP_IN_DIM = 64
# Below this diagonal action ||H_vv d0_v|| the cross-term ratio eps(v, w) is undefined.
_EPS_DENOM_TOL = 1e-12


# ======================================================================
# Update direction helpers
# ======================================================================


def _damping_of(cfg: Exp7Config) -> Damping:
    """The run's one damping: lambda_0 = ``cfg.damping``, LM-adapted as ``cfg`` says."""
    return Damping(
        cfg.damping,
        factor=cfg.damping_factor,
        bounds=cfg.damping_bounds,
        adapt=cfg.damping_adapt,
    )


@dataclass(frozen=True)
class NewtonCGSolve:
    """One truncated-CG solve: the direction, its iteration count, and how it ended.

    ``negative_curvature`` records that CG met a direction with p^T (H + lambda I) p <= 0
    and stopped there. On an indefinite Hessian the damped model is unbounded below
    along eigendirections with eigenvalue below -lambda, so such a step is a *truncated*
    descent step whose model value says nothing about the loss - the flag makes that
    visible in the payoff records (Newton-CG diverged at lambda = 1 on b1 with every
    m_newton negative).
    """

    direction: Tensor
    n_iter: int
    negative_curvature: bool


def _newton_cg(
    grad: Tensor,
    hvp: HessianOracle,
    *,
    damping: float,
    max_iter: int,
    tol: float = 1e-6,
) -> NewtonCGSolve:
    """Truncated-CG solve of (H + lambda I) d = -grad (a truncated-CG reference step).

    Linear conjugate gradients on the damped exact Hessian, terminated early on a
    negative-curvature direction (p^T A p <= 0) - the standard Newton-CG safeguard for a
    non-convex H. If the very first direction has negative curvature, it falls back to
    steepest descent d = -grad. Each iteration costs one exact HVP; the report carries
    the iteration count and whether the negative-curvature exit was taken.
    """
    b = -grad
    d = torch.zeros_like(grad)
    r = b.clone()
    p = r.clone()
    rs = float(r.dot(r))
    if rs <= tol * tol:
        return NewtonCGSolve(d, 0, False)
    n_iter = 0
    negative = False
    for _ in range(max_iter):
        n_iter += 1
        ap = hvp(p) + damping * p
        pap = float(p.dot(ap))
        if pap <= 0.0:
            negative = True
            if float(d.norm()) == 0.0:
                d = b  # negative curvature on the first step -> steepest descent
            break
        alpha = rs / pap
        d = d + alpha * p
        r = r - alpha * ap
        rs_new = float(r.dot(r))
        if rs_new <= tol * tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return NewtonCGSolve(d, n_iter, negative)


# ======================================================================
# Payoff bookkeeping (what the hooks record, what the runner serializes)
# ======================================================================


@dataclass
class MethodTrace:
    """Per-seed accumulator the step hooks write and the runner serializes.

    The cumulative counters are read at every epoch end and differenced into per-epoch
    series; the measurement records are appended at the ``measure_period`` cadence. A
    method without a hook (sgd / adam) leaves its fresh trace untouched, which
    serializes to zeros and empty lists.

      - ``n_hvp``: exact HVPs issued through `counting` - optimizer, overlay, safeguard
        and measurements alike (the Hutchinson coupling probes run through their own
        block-wise closures inside `ParamBlockEstimator` and are not counted);
      - ``n_proposed``: steps on which the overlay proposed a correction (a cached pair
        was in use), i.e. the safeguard had a verdict to give;
      - ``n_accepted`` / ``n_reject``: those verdicts (``n_accepted + n_reject ==
        n_proposed``);
      - ``n_refresh_failed``: EKFAC eigenbasis refreshes that did not converge;
      - ``n_newton_solves`` / ``n_newton_negative_curvature``: truncated-CG solves
        (the oracle's on the payoff records, plus every step of the ``newton_cg``
        method) and how many of them stopped on negative curvature;
      - ``model_decrease`` / ``cross_term``: the periodic payoff records;
      - ``damping_trajectory``: (step, lambda) after the Levenberg-Marquardt update
        taken at each payoff record - the lambda in force until the next record.
    """

    n_hvp: int = 0
    n_proposed: int = 0
    n_accepted: int = 0
    n_reject: int = 0
    n_refresh_failed: int = 0
    n_newton_solves: int = 0
    n_newton_negative_curvature: int = 0
    n_reject_by: dict[str, int] = field(
        default_factory=lambda: {"ratio": 0, "model": 0, "loss": 0}
    )
    model_decrease: list[dict[str, Any]] = field(default_factory=list)
    cross_term: list[dict[str, Any]] = field(default_factory=list)
    damping_trajectory: list[tuple[int, float]] = field(default_factory=list)

    def counting(self, hvp: HessianOracle) -> HessianOracle:
        """Wrap an exact HVP oracle so every product it serves increments ``n_hvp``."""

        def counted(z: Tensor) -> Tensor:
            self.n_hvp += 1
            return hvp(z)

        return counted

    def newton_solve(self, solve: NewtonCGSolve) -> NewtonCGSolve:
        """Count one truncated-CG solve and its negative-curvature exit."""
        self.n_newton_solves += 1
        self.n_newton_negative_curvature += int(solve.negative_curvature)
        return solve

    def damping_update(
        self, step: int, damping: Damping, m0: float, actual: float
    ) -> float:
        """One LM update of the run's damping at ``step``; logs the new lambda, returns rho.

        ``m0`` is the damped model's predicted change for the step taken (m(d0), or
        m(d_Newton) for the ``newton_cg`` method) and ``actual`` the realized decrease
        L(theta) - L(theta + d) on the same minibatch.
        """
        rho = damping.update(m0, actual)
        self.damping_trajectory.append((step, damping.value))
        return rho


def _cross_term_ratios(
    hvp: HessianOracle, d0: Tensor, block_ranges: dict[LayerID, slice]
) -> tuple[dict[Pair, float], dict[LayerID, float]]:
    """eps(v, w) = ||H_vw d0_w|| / ||H_vv d0_v|| for every ordered pair v != w, plus
    the per-block denominator ||H_vv d0_v|| itself.

    The block actions are read off the full HVP: y_w = H (P_w d0), with P_w d0 the base
    direction zeroed outside block w, has y_w[s_v] = H_vw d0_w - one HVP per block, n_b
    in total. The ratio is the cross-block curvature the block-diagonal step ignores,
    relative to the diagonal curvature it accounts for; NaN where the latter vanishes.
    """
    images: dict[LayerID, Tensor] = {}
    for w, sw in block_ranges.items():
        masked = torch.zeros_like(d0)
        masked[sw] = d0[sw]
        images[w] = hvp(masked)
    ratios: dict[Pair, float] = {}
    denoms: dict[LayerID, float] = {}
    for v, sv in block_ranges.items():
        denom = images[v][sv].norm().item()
        denoms[v] = denom
        for w in block_ranges:
            if w == v:
                continue
            num = images[w][sv].norm().item()
            ratios[(v, w)] = num / denom if denom >= _EPS_DENOM_TOL else math.nan
    return ratios, denoms


def _coupling_at(coupling: dict[Pair, float], v: LayerID, w: LayerID) -> float:
    """C(v, w) as last measured (keyed v-before-w in group order); NaN if never."""
    return coupling.get((v, w), coupling.get((w, v), math.nan))


def _record_payoff(
    trace: MethodTrace,
    step: int,
    grad: Tensor,
    hvp: HessianOracle,
    report: StepReport,
    block_ranges: dict[LayerID, slice],
    coupling: dict[Pair, float],
    *,
    damping: Damping,
    loss_at: LossAt,
) -> None:
    """One payoff record: m(d_KFAC), m(d_overlay), m(d_Newton), eps(v, w) and the losses.

    Same minibatch, same exact oracle, pre-update weights; costs the Newton-CG solve
    (<= ``_NEWTON_CG_ITERS`` HVPs), one HVP for m(d_Newton), n_b HVPs for eps, and up
    to two forward passes for the losses the safeguard did not already evaluate:
    ``loss_theta`` (before the step) and ``loss_d0`` (after the base step) give ``rho``,
    the actual-reduction ratio of the base step, i.e. whether the quadratic model is
    predictive at the step length in use. ``m_proposal`` is the model value of the
    overlay's proposal whether or not it was taken (NaN when it never got a model
    evaluation); ``m_overlay`` is that of the step actually taken.

    The record is taken with the ``damping`` in force on this step - the same lambda
    the base step, the overlay and the safeguard just used - and then that lambda is
    updated by the Levenberg-Marquardt rule on ``rho`` (`Damping.update`), so the next
    steps, and the next factor refresh, run on the adapted value.
    """
    decision = report.decision
    lam = float(damping)  # the lambda this record - and this step - was taken with
    with deterministic_mode():
        n_before = trace.n_hvp
        solve = trace.newton_solve(
            _newton_cg(grad, hvp, damping=lam, max_iter=_NEWTON_CG_ITERS)
        )
        d_newton = solve.direction
        n_hvp_newton = trace.n_hvp - n_before
        m_newton = model_value(grad, d_newton, hvp(d_newton), damping=lam)
        ratios, denoms = _cross_term_ratios(hvp, report.d0, block_ranges)
        loss_theta = loss_at(torch.zeros_like(report.d0))
        loss_d0 = decision.loss_d0
        if math.isnan(loss_d0):
            loss_d0 = loss_at(report.d0)
    rho = trace.damping_update(step, damping, decision.m0, loss_theta - loss_d0)
    trace.model_decrease.append(
        {
            "step": step,
            "damping": lam,
            "rho": rho,
            "m_kfac": decision.m0,
            "m_overlay": decision.m1 if decision.accepted else decision.m0,
            "m_proposal": decision.m1,
            "m_newton": m_newton,
            "norm_d0": decision.norm_d0,
            "norm_d1": decision.norm_d1,
            "loss_theta": loss_theta,
            "loss_d0": loss_d0,
            "loss_d1": decision.loss_d1,
            "accepted": decision.accepted,
            "rejected_by": decision.rejected_by,
            "n_pairs": report.n_selected_pairs,
            "n_hvp_newton": n_hvp_newton,
            "newton_negative_curvature": solve.negative_curvature,
        }
    )
    trace.cross_term.append(
        {
            "step": step,
            "pairs": {
                f"{v}->{w}": {"eps": eps, "coupling": _coupling_at(coupling, v, w)}
                for (v, w), eps in ratios.items()
            },
            "denom": denoms,
        }
    )


# ======================================================================
# Step-hook factories (the COUPLE-FAC / Newton-CG seam into Trainer)
# ======================================================================


def build_couplefac_hook(
    model: ParamGroupedModel,
    cfg: Exp7Config,
    *,
    tau: float,
    kind: str = "kfac",
    seed: int,
    device: torch.device,
    trace: MethodTrace,
) -> Callable[[StepCtx], None]:
    """A Trainer step hook running one COUPLE-FAC update per minibatch.

    ``kind`` selects the block-diagonal base preconditioner (``"kfac"`` or ``"ekfac"``);
    the overlay machinery is identical for both. With ``tau = +inf`` the selection set
    S = {C_{vw} > tau} is empty and the step is the plain base direction (the ``kfac`` /
    ``ekfac`` baseline); with a finite ``tau`` the coupling-gated rank-r overlay is added
    and the safeguard tests its norm, damped local quadratic model, and minibatch loss.
    For finite values, acceptance requires the model value to be within the configured
    tolerance of the base value and the norm within ``tr_ratio`` times the base norm. Every
    ``measure_period`` steps (T) the coupling C is re-measured
    and the optimizer refreshes its selection and subspace cache on it; on those same
    steps the payoff record is taken (see `_record_payoff`) before the direction is
    applied, and the shared damping is LM-updated on it. The provider refreshes its
    factors on its own period, reading the damping then in force. A safeguard verdict is
    counted only when a correction was actually proposed. Every exact HVP goes through
    ``trace.counting``.
    """
    loss_fn = nn.CrossEntropyLoss()
    damping = _damping_of(cfg)  # the one lambda: provider, overlay, safeguard, oracle
    provider: CurvlinopsKFACProvider | CurvlinopsEKFACProvider
    if kind == "ekfac":
        provider = CurvlinopsEKFACProvider(
            model,
            loss_fn,
            damping=damping,
            fisher_type=cfg.fisher_type,
            refresh_period=cfg.ekfac_refresh_period,
        )
    else:
        provider = CurvlinopsKFACProvider(
            model,
            loss_fn,
            damping=damping,
            fisher_type=cfg.fisher_type,
            refresh_period=cfg.kfac_refresh_period,
        )
    overlay = CoupleFacOverlay(
        rank=cfg.overlay_rank,
        n_power_iter=cfg.overlay_n_power_iter,
        # the base step's lambda, the same value: the K-FAC step damps its Kronecker
        # factors, the EKFAC step its eigenbasis, the correction |lambda_i| + lambda
        damping=damping,
        max_pairs=cfg.overlay_max_pairs,
    )
    optimizer = CoupleFacOptimizer(
        provider=provider,
        overlay=overlay,
        tau=tau,
        tr_ratio=cfg.tr_ratio,
        params=group_major_params(model),
        lr=SECOND_ORDER_LR,
    )
    estimator = ParamBlockEstimator(
        model, loss_fn, n_probes=cfg.coupling_n_probes_online
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    overlay_active = math.isfinite(tau)
    period = max(1, cfg.measure_period)
    last_coupling: dict[Pair, float] = {}
    loss_at = trial_loss(model, loss_fn, optimizer.params or [], optimizer.lr)

    def hook(ctx: StepCtx) -> None:
        nonlocal last_coupling
        provider.update(ctx.x, ctx.y)  # K-FAC curvature: rebuilt every T_k steps
        if isinstance(provider, CurvlinopsEKFACProvider):
            trace.n_refresh_failed = provider.n_refresh_failed
        cur = estimator.curvature(ctx.x, ctx.y)  # exact grad + HVP + deferred coupling
        hvp = trace.counting(cur.hvp)
        measuring = ctx.step % period == 0  # the overlay refresh step (t mod T == 0)
        if overlay_active and measuring:
            last_coupling = cur.coupling()  # amortized: re-measure C every T steps
        coupling = last_coupling if overlay_active else {}
        loss_here = loss_at(ctx.x, ctx.y)
        report = optimizer.compute_direction(
            cur.flat_grad,
            hvp,
            coupling,
            refresh=measuring,
            generator=generator,
            loss_at=loss_here,
        )
        if report.n_selected_pairs > 0:  # a correction was proposed: a verdict
            trace.n_proposed += 1
            if report.decision.accepted:
                trace.n_accepted += 1
            else:
                trace.n_reject += 1
                trace.n_reject_by[report.decision.rejected_by or "model"] += 1
        if measuring:
            _record_payoff(
                trace,
                ctx.step,
                cur.flat_grad,
                hvp,
                report,
                provider.block_ranges(),
                coupling,
                damping=damping,
                loss_at=loss_here,
            )
        optimizer.apply(report.direction)  # after every HVP: the oracle is pre-update
        del cur  # release the create_graph backward graph held by the bundle

    return hook


def build_newton_cg_hook(
    model: ParamGroupedModel,
    cfg: Exp7Config,
    *,
    device: torch.device,
    trace: MethodTrace,
) -> Callable[[StepCtx], None]:
    """A Trainer step hook taking a truncated-CG damped-system step.

    Every ``measure_period`` steps (and at step 0) the step's own solve doubles as the
    payoff measurement: m(d_Newton), the HVPs the solve took and whether it stopped on
    negative curvature are recorded; every solve is counted in the trace. The Newton
    step is this method's base step, so the same record measures L(theta + d_Newton)
    (two forward passes, theta untouched) and LM-updates the damping on its rho.
    """
    loss_fn = nn.CrossEntropyLoss()
    damping = _damping_of(cfg)
    estimator = ParamBlockEstimator(model, loss_fn, n_probes=cfg.coupling_n_probes)
    flat_params = group_major_params(model)
    period = max(1, cfg.measure_period)
    loss_at = trial_loss(model, loss_fn, flat_params, SECOND_ORDER_LR)

    def solve(grad: Tensor, hvp: HessianOracle) -> NewtonCGSolve:
        return trace.newton_solve(
            _newton_cg(grad, hvp, damping=float(damping), max_iter=_NEWTON_CG_ITERS)
        )

    def hook(ctx: StepCtx) -> None:
        cur = estimator.curvature(ctx.x, ctx.y)
        hvp = trace.counting(cur.hvp)
        if ctx.step % period == 0:
            lam = float(damping)
            with deterministic_mode():
                n_before = trace.n_hvp
                solved = solve(cur.flat_grad, hvp)
                n_hvp_newton = trace.n_hvp - n_before
                m_newton = model_value(
                    cur.flat_grad, solved.direction, hvp(solved.direction), damping=lam
                )
                loss_here = loss_at(ctx.x, ctx.y)
                loss_theta = loss_here(torch.zeros_like(solved.direction))
                loss_newton = loss_here(solved.direction)
            rho = trace.damping_update(
                ctx.step, damping, m_newton, loss_theta - loss_newton
            )
            trace.model_decrease.append(
                {
                    "step": ctx.step,
                    "damping": lam,
                    "rho": rho,
                    "m_newton": m_newton,
                    "loss_theta": loss_theta,
                    "loss_newton": loss_newton,
                    "n_hvp_newton": n_hvp_newton,
                    "newton_negative_curvature": solved.negative_curvature,
                }
            )
        else:
            solved = solve(cur.flat_grad, hvp)
        apply_flat(flat_params, solved.direction, SECOND_ORDER_LR)
        del cur

    return hook


# ======================================================================
# Dataset / backbone construction
# ======================================================================


def build_loaders(cfg: Exp7Config) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) for the configured dataset (fixed split)."""
    batch_size = cfg.training.batch_size
    if cfg.dataset == "stanford_cars":
        return get_stanford_cars_loaders(
            batch_size,
            image_size=cfg.image_size,
            augment=cfg.augment,
            seed=_SPLIT_SEED,
        )
    if cfg.dataset == "cifar10":
        return get_cifar10_image_loaders(batch_size)
    if cfg.dataset == "cifar100":
        return get_cifar100_loaders(batch_size)
    if cfg.dataset == "synthetic":
        return get_synthetic_classification_loaders(
            batch_size, in_dim=_MLP_IN_DIM, num_classes=cfg.num_classes
        )
    if cfg.dataset == "imagenet32":
        return get_imagenet32_loaders(batch_size, augment=cfg.augment)
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def build_model(cfg: Exp7Config) -> SegmentedResNet18 | SegmentedMLP:
    """Backbone for the configured registry key."""
    if cfg.model == "resnet50":
        return SegmentedResNet50(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    if cfg.model == "resnet18":
        return SegmentedResNet18(num_classes=cfg.num_classes)
    if cfg.model == "mlp":
        return SegmentedMLP(in_dim=_MLP_IN_DIM, num_classes=cfg.num_classes)
    raise ValueError(
        f"unknown model {cfg.model!r}; expected 'resnet50', 'resnet18', or 'mlp'"
    )


# ======================================================================
# Metrics
# ======================================================================


def _has_overlay(method: str) -> bool:
    """Whether ``method`` runs the coupling-gated overlay (so the safeguard decides)."""
    return method.endswith("+overlay")


def _epochs_planned(training: TrainingConfig, steps_per_epoch: int) -> int:
    """Epochs the Trainer will actually run: ``epochs``, cut by a ``max_steps`` budget."""
    if training.max_steps is None:
        return training.epochs
    return min(training.epochs, math.ceil(training.max_steps / max(1, steps_per_epoch)))


def _epochs_to_target(curve: list[float], target: float) -> int | None:
    """First 1-indexed epoch whose validation accuracy reaches ``target`` (else None)."""
    for epoch, acc in enumerate(curve, start=1):
        if acc >= target:
            return epoch
    return None


def aggregate_method(per_seed: dict[str, Any], target: float) -> dict[str, Any]:
    """Mean/std final accuracy and mean epochs-to-target over seeds."""
    finals = [payload["final_acc"] for payload in per_seed.values()]
    reached = [
        payload["epochs_to_target"]
        for payload in per_seed.values()
        if payload["epochs_to_target"] is not None
    ]
    return {
        "seeds": per_seed,
        "target_acc": target,
        "final_acc_mean": float(np.mean(finals)),
        "final_acc_std": float(np.std(finals)),
        "epochs_to_target_mean": float(np.mean(reached)) if reached else None,
        "n_reached_target": len(reached),
        "n_seeds": len(per_seed),
    }


# ======================================================================
# Runner
# ======================================================================


class Exp7Runner:
    """Runner for Exp7 (COUPLE-FAC overlay vs. first-/second-order baselines)."""

    def __init__(self, config: Exp7Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {meta, method: {seeds: {...}, final_acc_mean, epochs_to_target_mean,...}}."""
        cfg = self._cfg
        _, val_loader = build_loaders(cfg)
        # One fixed validation minibatch: Phi is read off it for every method and seed.
        phi_batch = sample_hessian_batch(
            val_loader, cfg.hessian_batch_size, self._device
        )
        results: dict[str, Any] = {META_KEY: provenance(cfg, experiment="exp7")}

        for method in cfg.methods:
            logger.info("=== Method: %s ===", method)
            per_seed: dict[str, Any] = {}

            for seed in cfg.training.seeds:
                logger.info("  Seed %d", seed)
                set_seed(seed, deterministic=cfg.training.deterministic)
                # Build a fresh training loader after seeding each method and seed,
                # isolating persistent-worker augmentation RNG state. The shared
                # validation loader does not shuffle or augment.
                train_loader, _ = build_loaders(cfg)

                model = build_model(cfg).to(self._device)
                trace = MethodTrace()
                trainer = self._make_trainer(method, model, seed, trace)
                per_seed[f"seed_{seed}"] = self._train_with_curve(
                    trainer, model, trace, train_loader, val_loader, phi_batch
                )

            results[method] = aggregate_method(per_seed, cfg.target_acc)

        return results

    # ------------------------------------------------------------------
    # Per-method wiring
    # ------------------------------------------------------------------

    def _make_trainer(
        self,
        method: str,
        model: SegmentedResNet18 | SegmentedMLP,
        seed: int,
        trace: MethodTrace,
    ) -> Trainer:
        """Builds the Trainer for ``method``: optimizer slot or a second-order hook."""
        cfg = self._cfg
        if method == "sgd":
            return Trainer(model, replace(cfg.training, optimizer="sgd"), self._device)
        if method == "adam":
            tcfg = replace(cfg.training, optimizer="adam", lr=_ADAM_LR)
            return Trainer(model, tcfg, self._device)
        if method in ("kfac", "kfac+overlay", "ekfac", "ekfac+overlay"):
            tau = cfg.tau if _has_overlay(method) else math.inf
            kind = "ekfac" if method.startswith("ekfac") else "kfac"
            hook = build_couplefac_hook(
                model,
                cfg,
                tau=tau,
                kind=kind,
                seed=seed,
                device=self._device,
                trace=trace,
            )
            return Trainer(model, cfg.training, self._device, step_hook=hook)
        if method == "newton_cg":
            hook = build_newton_cg_hook(model, cfg, device=self._device, trace=trace)
            return Trainer(model, cfg.training, self._device, step_hook=hook)
        raise ValueError(
            f"unknown method {method!r}; expected one of {{'sgd', 'adam', 'kfac', "
            "'ekfac', 'kfac+overlay', 'ekfac+overlay', 'newton_cg'}}"
        )

    def _train_with_curve(
        self,
        trainer: Trainer,
        model: SegmentedResNet18 | SegmentedMLP,
        trace: MethodTrace,
        train_loader: DataLoader,
        val_loader: DataLoader,
        phi_batch: tuple[Tensor, Tensor],
    ) -> dict[str, Any]:
        """Trains and records the per-seed payload: curves, per-epoch bookkeeping, Phi.

        Phi is measured at init, after epoch ``epochs // 2`` and after the last epoch;
        with a one-epoch budget the midpoint is the start, so ``mid`` repeats ``init``.
        The per-epoch wall-clock covers training only (it is read before the validation
        pass and the Phi measurement); the HVP count and the accept rate are the
        epoch's increments of the trace's cumulative counters - the accept rate is
        accepted / proposed within the epoch, NaN when no correction was proposed
        (methods without the overlay, or an epoch on which no pair was selected).
        """
        cfg = self._cfg
        x_phi, y_phi = phi_batch

        def phi_now() -> dict[str, Any]:
            model.eval()  # as exp8: the deployed network (BatchNorm running statistics)
            return measure_phi(
                model, x_phi, y_phi, trainer.loss_fn, cfg.coupling_n_probes
            )

        mid_epoch = _epochs_planned(cfg.training, len(train_loader)) // 2
        phi: dict[str, Any] = {"init": phi_now()}
        if mid_epoch == 0:
            phi["mid"] = phi["init"]

        accs: list[float] = []
        losses: list[float] = []
        wall: list[float] = []
        hvps: list[int] = []
        accept_rates: list[float] = []
        prev_hvp = prev_accepted = prev_proposed = 0
        epoch_start = time.perf_counter()

        def record(epoch: int) -> None:
            nonlocal epoch_start, prev_hvp, prev_accepted, prev_proposed
            wall.append(time.perf_counter() - epoch_start)
            proposed = trace.n_proposed - prev_proposed
            hvps.append(trace.n_hvp - prev_hvp)
            accept_rates.append(
                (trace.n_accepted - prev_accepted) / proposed
                if proposed > 0
                else math.nan
            )
            prev_hvp, prev_accepted, prev_proposed = (
                trace.n_hvp,
                trace.n_accepted,
                trace.n_proposed,
            )
            evaluation = trainer.evaluate(val_loader)
            accs.append(evaluation["test_acc"])
            losses.append(evaluation["test_loss"])
            if epoch == mid_epoch:
                phi["mid"] = phi_now()
            epoch_start = time.perf_counter()

        trainer.train(train_loader, checkpoint_epochs=[], on_epoch_end=record)
        if self._device.type == "cuda":  # peak GPU allocation for this run
            # Through `run_peak_allocated`, not torch's counter: the estimator resets
            # that counter to calibrate its probe chunk (`hessian.memory`).
            logger.info(
                "  peak GPU memory so far: %.2f GiB allocated, %.2f GiB reserved",
                run_peak_allocated(self._device) / 2**30,
                torch.cuda.max_memory_reserved(self._device) / 2**30,
            )
        phi["final"] = phi_now()

        return {
            "val_acc": accs,
            "val_loss": losses,
            "final_acc": accs[-1] if accs else float("nan"),
            "epochs_to_target": _epochs_to_target(accs, cfg.target_acc),
            "n_steps": trainer.step_count,
            "hessian_batch_size": cfg.hessian_batch_size,
            "wall_sec_per_epoch": wall,
            "hvp_per_epoch": hvps,
            "safeguard": {
                "accept_rate_per_epoch": accept_rates,
                "n_proposed": trace.n_proposed,
                "n_reject": trace.n_reject,
                "n_reject_by": dict(trace.n_reject_by),
                "n_refresh_failed": trace.n_refresh_failed,
            },
            "newton_cg": {
                "n_solves": trace.n_newton_solves,
                "n_negative_curvature": trace.n_newton_negative_curvature,
            },
            "model_decrease": trace.model_decrease,
            "cross_term": trace.cross_term,
            "damping_trajectory": trace.damping_trajectory,
            "phi": phi,
        }


# ======================================================================
# CLI: exp7 (COUPLE-FAC overlay optimization, the repair experiment)
# ======================================================================


def add_args(parser: argparse.ArgumentParser) -> None:
    add_device_args(parser)
    parser.add_argument(
        "--profile",
        type=str,
        default="b1",
        choices=["b1", "b1_mlp"],
        help="b1: pretrained ResNet-50 finetune on Stanford Cars (headline); "
        "b1_mlp: optimizer-agnostic control on a full-rank synthetic MLP (clean EKFAC).",
    )
    # Optional overrides; a value of None leaves the profile default untouched.
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["stanford_cars", "cifar10", "cifar100", "imagenet32", "synthetic"],
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=["resnet50", "resnet18", "mlp"]
    )
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--overlay-rank", type=int, default=None)
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="lambda_0, the initial Tikhonov damping shared by K-FAC, overlay, "
        "safeguard and the Newton-CG oracle",
    )
    parser.add_argument(
        "--damping-adapt",
        action="store_true",
        help="adapt the damping from lambda_0 by the Levenberg-Marquardt rule at "
        "every payoff record (default: fixed at lambda_0)",
    )
    parser.add_argument("--measure-period", type=int, default=None)
    parser.add_argument("--target-acc", type=float, default=None)
    parser.add_argument(
        "--tr-ratio",
        type=float,
        default=None,
        help="relative trust region of the safeguard, ||d1|| <= tr_ratio ||d0||",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps in total (benchmarks and smokes)",
    )


def build_config(args: argparse.Namespace) -> Exp7Config:
    """Selects the regime profile, then applies any explicit CLI overrides."""
    if args.profile == "b1":
        base = Exp7Config.b1_cars()
    else:
        base = Exp7Config.b1_mlp()

    training = override(
        base.training,
        epochs=args.epochs,
        seeds=args.seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
    )

    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "model": args.model,
            "methods": args.methods,
            "tau": args.tau,
            "overlay_rank": args.overlay_rank,
            "damping": args.damping,
            "measure_period": args.measure_period,
            "target_acc": args.target_acc,
            "tr_ratio": args.tr_ratio,
        }.items()
        if value is not None
    }
    # A dataset switch implies its class count (keeps the head consistent).
    if "dataset" in overrides:
        overrides["num_classes"] = DATASET_NUM_CLASSES[overrides["dataset"]]
    if args.damping_adapt:
        overrides["damping_adapt"] = True

    return replace(base, training=training, **overrides)


SPEC = Experiment(
    name="exp7",
    help="COUPLE-FAC overlay optimization (repair experiment)",
    add_args=add_args,
    build_config=build_config,
    run=Exp7Runner,
    default_output_dir=lambda args: f"results/exp7_{args.profile}",
)
