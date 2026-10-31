"""Exp11: model validity radius along an optimization trajectory.

The runner probes K-FAC, SGD or Adam directions at pre-update weights. On each
selected minibatch it compares the observed loss change along a direction with the
damped quadratic model m(t d0) = t g^T d0 + t^2/2 d0^T (H + lambda I) d0.
The damping term has no counterpart in the loss, so a discrepancy does not by itself
measure error in the undamped local Hessian approximation.

Outputs include the model decomposition, the loss and model curves, the grid-based
relative-error validity radius, estimated Hessian extremes and a scaled approximate
top direction, as well as ReLU and max-pool region changes. The reported stability
number is a diagnostic, not a universal threshold for these nonlinear optimizers.
The selected step source defines the trajectory between probes; the default is K-FAC.

"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from experiments.cli import Experiment, add_device_args, override
from experiments.config import Exp7Config, Exp11Config, TrainingConfig
from experiments.param_ops import (
    SECOND_ORDER_LR,
    apply_flat,
    group_major_params,
    trial_forward,
    trial_loss,
)
from experiments.protocol import mean_std
from experiments.provenance import environment_block
from experiments.runner_exp7 import build_loaders, build_model
from experiments.training import StepCtx, Trainer
from experiments.utils import deterministic_mode, set_seed
from hessian import ParamBlockEstimator, ParamGroupedModel
from hessian.memory import run_peak_allocated
from hessian.param_space import ParamCurvature
from hessian.spectrum import HVPClosure, extreme_ritz
from secondorder.kfac_adapter import CurvlinopsKFACProvider
from secondorder.safeguard import LossAt, model_value

logger = logging.getLogger(__name__)

# (direction, x) -> logits at theta + direction, theta untouched (`trial_forward`).
TrialForward = Callable[[Tensor, Tensor], Tensor]
# () -> (train_loader, val_loader); the runner's injectable data seam.
LoaderFactory = Callable[[], tuple[DataLoader, DataLoader]]

# Entries of a probe record that are coordinates shared by every seed (the step, the
# grid point, the unit counts); the aggregate keeps them verbatim instead of averaging.
_COORDINATES = frozenset({"step", "t", "n_relu", "n_pool"})


# ======================================================================
# (a)-(d): the model against the loss along a direction
# ======================================================================


def _rel_err(actual: float, predicted: float) -> float:
    """|actual - predicted| / |actual|; 0 when both vanish, inf when only the actual."""
    if actual == 0.0:
        return 0.0 if predicted == 0.0 else math.inf
    return abs(actual - predicted) / abs(actual)


def validity_radius(points: list[dict[str, float]], tolerance: float) -> float:
    """t* = the largest grid t up to which every point's ``rel_err`` <= ``tolerance``.

    ``points`` in increasing t. A radius, not the last point that happens to pass: the
    model is declared valid *up to* t*, and a failure at a smaller t ends the scan.
    NaN when the smallest t already fails.
    """
    t_star = math.nan
    for point in points:
        if not point["rel_err"] <= tolerance:  # NaN / inf fail too
            break
        t_star = point["t"]
    return t_star


def model_curve(
    loss_at: LossAt,
    direction: Tensor,
    loss_0: float,
    *,
    g_dot: float,
    dhd: float,
    norm_sq: float,
    damping: float,
    t_grid: list[float],
    tolerance: float,
) -> tuple[list[dict[str, float]], float]:
    """The loss against the damped model along ``direction``, and the validity radius.

    For every t of ``t_grid`` (one forward each): Delta L = L(theta + t d) - ``loss_0``,
    m = t g.d + t^2/2 (d.Hd + lambda ||d||^2) from the given coefficients, the secant
    curvature 2 (Delta L - t g.d) / t^2 - the curvature the loss actually has at that
    scale, against the pointwise d.Hd - and the relative error |Delta L - m| / |Delta L|.
    Returns the points (increasing t) and t* (`validity_radius`).
    """
    points: list[dict[str, float]] = []
    for t in sorted(t_grid):
        dl = loss_at(t * direction) - loss_0
        m = t * g_dot + 0.5 * t * t * (dhd + damping * norm_sq)
        points.append(
            {
                "t": t,
                "dL": dl,
                "m": m,
                "secant": 2.0 * (dl - t * g_dot) / (t * t),
                "rel_err": _rel_err(dl, m),
            }
        )
    return points, validity_radius(points, tolerance)


def probe_model_validity(
    grad: Tensor,
    hvp: HVPClosure,
    d0: Tensor,
    loss_at: LossAt,
    *,
    damping: float,
    t_grid: list[float],
    tolerance: float,
    n_lanczos_iter: int,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """(a)-(d) and the payoff numbers of one step, on the step's own objects.

    ``grad`` / ``hvp`` are the exact gradient and Hessian-vector product of the step's
    minibatch at the pre-update weights, ``d0`` the base direction about to be applied
    (eta = 1), ``loss_at`` the minibatch loss after a trial displacement.

    The spectral part is one Lanczos run of ``n_lanczos_iter`` steps (`extreme_ritz`),
    which returns both ends of the spectrum, the top Ritz vector (reconstructed by a
    second pass, since the basis is not stored) and the residual bounds. The two
    directions the probe measures anyway - d0 and g - are passed as candidates, so
    lambda_max is floored by their Rayleigh quotients and cannot be reported below a
    curvature measured on the same Hessian. The gradient's quotient uses the same HVP and is the
    reported ``curv_along_g`` (``curv_along_d0`` stays the exact d0^T H d0 / ||d0||^2
    of the model's own H d0).

    Costs 2 ``n_lanczos_iter`` + 13 HVPs (the Lanczos run and its replay, the two
    candidates, the 10 power steps that polish the reconstructed Ritz vector and that
    vector's own Rayleigh quotient - the reported ``curv_vmax`` - plus H d0) and
    2 |``t_grid``| + 2 forwards. The top eigenvector is
    oriented downhill (g^T v <= 0), like the step, and the curve along it is taken at
    ||d0|| v so its t is the step's.
    """
    dim, device = grad.numel(), grad.device
    hd0 = hvp(d0)
    g_dot_d0 = grad.dot(d0).item()
    d0hd0 = d0.dot(hd0).item()
    norm_d0, norm_g = d0.norm().item(), grad.norm().item()
    m_d0 = model_value(grad, d0, hd0, damping=damping)
    eta_eff = norm_d0 / norm_g
    eta_g = -g_dot_d0 / norm_g**2

    loss_theta = loss_at(torch.zeros_like(d0))
    curve_kwargs: dict[str, Any] = {
        "damping": damping,
        "t_grid": t_grid,
        "tolerance": tolerance,
    }
    curve_d0, t_star = model_curve(
        loss_at,
        d0,
        loss_theta,
        g_dot=g_dot_d0,
        dhd=d0hd0,
        norm_sq=norm_d0**2,
        **curve_kwargs,
    )
    loss_d0 = loss_at(d0)

    g_hat = grad / norm_g
    ritz = extreme_ritz(
        hvp,
        dim,
        n_lanczos_iter,
        device,
        generator=generator,
        candidates=[d0 / norm_d0, g_hat],
        with_eigenvector=True,
    )
    # The gradient's quotient comes back from the candidate HVP; d0's is already exact
    # from the model's own H d0, and keeping it makes d0_H_d0 / ||d0||^2 the record's
    # curv_along_d0 to the last bit.
    _, curv_along_g = ritz.quotients
    v, vhv = ritz.eigenvector, ritz.eigenvector_quotient
    assert v is not None and vhv is not None  # with_eigenvector
    if grad.dot(v) > 0:
        v = -v  # downhill, as the step is (the Rayleigh quotient is sign-blind)
    g_dot_v = grad.dot(v).item()
    curve_vmax, t_star_vmax = model_curve(
        loss_at,
        norm_d0 * v,
        loss_theta,
        g_dot=norm_d0 * g_dot_v,
        dhd=norm_d0**2 * vhv,
        norm_sq=norm_d0**2,
        **curve_kwargs,
    )

    return {
        "decomposition": {
            "g_dot_d0": g_dot_d0,
            "d0_H_d0": d0hd0,
            "lambda_norm_d0_sq": damping * norm_d0**2,
            "m_d0": m_d0,
            "norm_d0": norm_d0,
            "norm_g": norm_g,
            "eta_eff": eta_eff,
            "eta_g": eta_g,
        },
        "curve_d0": curve_d0,
        "t_star": t_star,
        "curve_vmax": curve_vmax,
        "t_star_vmax": t_star_vmax,
        "sharpness": {
            "sigma": ritz.sigma,
            "lambda_max": ritz.lambda_max,
            "lambda_min": ritz.lambda_min,
            "ritz_max": ritz.ritz_max,
            "ritz_min": ritz.ritz_min,
            "residual_max": ritz.residual_max,
            "residual_min": ritz.residual_min,
            "converged": ritz.converged,
            "n_iter_used": ritz.n_iter_used,
            "curv_vmax": vhv,
            "g_dot_vmax": g_dot_v,
            "cos_d0_vmax": d0.dot(v).item() / norm_d0,
            "curv_along_g": curv_along_g,
            "curv_along_d0": d0hd0 / norm_d0**2,
            "lambda_max_x_eta_eff": ritz.lambda_max * eta_eff,
            "lambda_max_x_eta_g": ritz.lambda_max * eta_g,
        },
        "payoff": {
            "m_kfac": m_d0,
            "loss_theta": loss_theta,
            "loss_d0": loss_d0,
            "rho": (loss_theta - loss_d0) / -m_d0 if m_d0 != 0.0 else math.nan,
            "norm_d0": norm_d0,
        },
    }


# ======================================================================
# (e): the kinks a displacement crosses
# ======================================================================


class _KinkPatterns:
    """Activation patterns of every ReLU / max-pool of a module, and their changes.

    A ReLU's pattern is the sign of its pre-activation, read as the positivity of its
    output (right for the in-place ReLU of torchvision's ResNets too); a max-pool's is
    the winning position of every window, re-derived from the hook's input. Between two
    forwards on the same minibatch a differing entry is a kink of the piecewise-linear
    network crossed by the parameter displacement - a boundary between linear regions,
    across which the pointwise Hessian changes. `record` holds the reference pattern
    (one bool / int64 per unit); `compare` streams a later forward's pattern through
    the hooks and accumulates the counts, so no second copy is ever held.
    """

    def __init__(self, module: nn.Module) -> None:
        self._reference: list[Tensor] = []
        self._recording = True
        self._cursor = 0
        self._changes = {"relu": 0, "pool": 0}
        self._units = {"relu": 0, "pool": 0}
        self._handles: list[RemovableHandle] = []
        for m in module.modules():
            if isinstance(m, nn.ReLU):
                self._handles.append(m.register_forward_hook(self._on_relu))
            elif isinstance(m, nn.MaxPool2d):
                self._handles.append(m.register_forward_hook(self._on_pool))

    def __enter__(self) -> _KinkPatterns:
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._reference.clear()

    @property
    def n_relu(self) -> int:
        """ReLU evaluations per forward (units x minibatch)."""
        return self._units["relu"]

    @property
    def n_pool(self) -> int:
        """Max-pool windows per forward."""
        return self._units["pool"]

    def record(self, run: Callable[[], Any]) -> None:
        """Run the reference forward and hold its pattern."""
        self._recording = True
        self._reference.clear()
        self._units = {"relu": 0, "pool": 0}
        with torch.no_grad():
            run()

    def compare(self, run: Callable[[], Any]) -> tuple[int, int]:
        """Run a forward; (ReLU flips, max-pool changes) against the reference."""
        self._recording = False
        self._cursor = 0
        self._changes = {"relu": 0, "pool": 0}
        with torch.no_grad():
            run()
        if self._cursor != len(self._reference):
            raise RuntimeError("the compared forward visited a different set of kinks")
        return self._changes["relu"], self._changes["pool"]

    def _observe(self, kind: str, pattern: Tensor) -> None:
        if self._recording:
            self._reference.append(pattern)
            self._units[kind] += pattern.numel()
            return
        reference = self._reference[self._cursor]
        self._cursor += 1
        self._changes[kind] += int((pattern != reference).sum().item())

    def _on_relu(
        self, module: nn.Module, inputs: tuple[Tensor, ...], out: Tensor
    ) -> None:
        self._observe("relu", out > 0)

    def _on_pool(
        self, module: nn.Module, inputs: tuple[Tensor, ...], out: Tensor
    ) -> None:
        pool = cast(nn.MaxPool2d, module)
        _, indices = F.max_pool2d(
            inputs[0],
            pool.kernel_size,
            pool.stride,
            pool.padding,
            pool.dilation,
            pool.ceil_mode,
            return_indices=True,
        )
        self._observe("pool", indices)


def count_kink_crossings(
    model: ParamGroupedModel,
    forward: TrialForward,
    d0: Tensor,
    x: Tensor,
    t_grid: list[float],
) -> dict[str, Any]:
    """Kinks crossed between theta and theta + t d0 for every t of ``t_grid``.

    One forward per t plus the reference, all through ``forward`` (a `trial_forward`
    closure, so theta and the buffers stay put) on the same minibatch ``x``. Returns the
    unit counts and, per t, the number and the fraction of ReLU sign flips and max-pool
    argmax changes (the fraction is NaN for a network without that module type).
    """

    def displaced(t: float) -> Callable[[], Tensor]:
        return lambda: forward(t * d0, x)

    with _KinkPatterns(cast(nn.Module, model)) as patterns:
        patterns.record(displaced(0.0))
        rows: list[dict[str, float]] = []
        for t in sorted(t_grid):
            relu, pool = patterns.compare(displaced(t))
            rows.append(
                {
                    "t": t,
                    "relu_flips": relu,
                    "relu_frac": (
                        relu / patterns.n_relu if patterns.n_relu else math.nan
                    ),
                    "pool_changes": pool,
                    "pool_frac": (
                        pool / patterns.n_pool if patterns.n_pool else math.nan
                    ),
                }
            )
        return {"n_relu": patterns.n_relu, "n_pool": patterns.n_pool, "rows": rows}


# ======================================================================
# Step sources: the direction d0 the probed step takes (`--step-source`)
# ======================================================================


class StepSource(Protocol):
    """A pluggable choice of the step direction the runner probes (`--step-source`).

    One call per training step, in step order; an implementation may hold state
    accumulated across those calls (K-FAC's periodic Kronecker factors, Adam's running
    moments). The returned direction is applied at eta = 1
    (`experiments.param_ops.SECOND_ORDER_LR`), like every second-order step of Exp. 7.
    """

    def direction(self, curvature: ParamCurvature, x: Tensor, y: Tensor) -> Tensor:
        """d0, flat and group-major (`experiments.param_ops.group_major_params`).

        ``curvature`` is the exact bundle at the pre-update weights
        (`hessian.param_space.ParamCurvature`); ``x``/``y`` are the step's own
        minibatch, for a source whose curvature - unlike the gradient - is not fresh
        every step.
        """
        ...


class KFACStep:
    """d0 of Exp. 7's ``kfac`` method (`build_couplefac_hook` at tau = inf), moved out
    of the runner unchanged; see `CurvlinopsKFACProvider` for the block-diagonal solve.
    """

    def __init__(self, provider: CurvlinopsKFACProvider) -> None:
        self._provider = provider

    def direction(self, curvature: ParamCurvature, x: Tensor, y: Tensor) -> Tensor:
        self._provider.update(x, y)  # Kronecker factors rebuilt every T_k steps
        return self._provider.base_direction(curvature.flat_grad)


def _build_kfac_step(model: ParamGroupedModel, protocol: Exp7Config) -> StepSource:
    provider = CurvlinopsKFACProvider(
        model,
        nn.CrossEntropyLoss(),
        damping=protocol.damping,
        fisher_type=protocol.fisher_type,
        refresh_period=protocol.kfac_refresh_period,
    )
    return KFACStep(provider)


class SGDStep:
    """d0 = -g: plain steepest descent, no state - Exp. 7's ``sgd`` reference undecayed
    (that method's own momentum/lr/weight-decay are not reproduced, only its direction's
    shape; see `AdamStep` for the source that does carry Exp. 7's hyperparameters).
    """

    def direction(self, curvature: ParamCurvature, x: Tensor, y: Tensor) -> Tensor:
        return -curvature.flat_grad


def _build_sgd_step(model: ParamGroupedModel, protocol: Exp7Config) -> StepSource:
    return SGDStep()


# Exp. 7's ``adam`` method (`experiments.runner_exp7._ADAM_LR`) sets only ``lr``; betas
# and eps are torch.optim.Adam's un-overridden defaults (`experiments.training.Trainer`
# passes just ``lr``/``weight_decay``). Duplicated rather than imported (the source is a
# private module constant; `runner_exp8.py` does the same for its own `_SPLIT_SEED`).
_ADAM_LR = 1e-3
_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPS = 1e-8


class AdamStep:
    """Adam's diagonal-preconditioned direction, state accumulated along the trajectory.

    d0 = -lr * m_hat / (sqrt(v_hat) + eps) (Kingma & Ba, 2015), the update
    ``torch.optim.Adam`` would apply to ``curvature.flat_grad`` directly: first/second
    moments m, v with exponential decay ``betas``, bias-corrected by the call count t
    (m_hat = m / (1 - beta1^t), v_hat analogous). No weight decay - like `SGDStep`,
    this reproduces the shape of Exp. 7's update, not every training hyperparameter.
    """

    def __init__(
        self,
        lr: float = _ADAM_LR,
        betas: tuple[float, float] = _ADAM_BETAS,
        eps: float = _ADAM_EPS,
    ) -> None:
        self._lr = lr
        self._beta1, self._beta2 = betas
        self._eps = eps
        self._m: Tensor | None = None
        self._v: Tensor | None = None
        self._t = 0

    def direction(self, curvature: ParamCurvature, x: Tensor, y: Tensor) -> Tensor:
        g = curvature.flat_grad
        # ``m``/``v`` are always set together, but mypy narrows `Tensor | None` only
        # from a condition that mentions both attributes.
        if self._m is None or self._v is None:
            m, v = torch.zeros_like(g), torch.zeros_like(g)
        else:
            m, v = self._m, self._v
        self._t += 1
        m.mul_(self._beta1).add_(g, alpha=1.0 - self._beta1)
        v.mul_(self._beta2).addcmul_(g, g, value=1.0 - self._beta2)
        self._m, self._v = m, v
        m_hat = m / (1.0 - self._beta1**self._t)
        v_hat = v / (1.0 - self._beta2**self._t)
        return -self._lr * m_hat / (v_hat.sqrt() + self._eps)


def _build_adam_step(model: ParamGroupedModel, protocol: Exp7Config) -> StepSource:
    return AdamStep()


# The single source of truth for `--step-source`: its argparse ``choices`` and the
# dispatch in `build_validity_hook` both read this mapping, so neither can name an
# option the other lacks.
STEP_SOURCES: dict[str, Callable[[ParamGroupedModel, Exp7Config], StepSource]] = {
    "kfac": _build_kfac_step,
    "sgd": _build_sgd_step,
    "adam": _build_adam_step,
}


# ======================================================================
# The probing step (a Trainer step hook)
# ======================================================================


@dataclass
class ValidityTrace:
    """Per-seed accumulator the validity hook writes and the runner serializes.

    ``probes`` are the records of the probe steps in step order; ``losses`` the
    minibatch loss L(theta) before every step (read off the estimator's graph, so the
    trajectory costs no extra forward), ``losses[s]`` for step ``s``.
    """

    probes: list[dict[str, Any]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)


def build_validity_hook(
    model: ParamGroupedModel,
    cfg: Exp11Config,
    *,
    seed: int,
    device: torch.device,
    trace: ValidityTrace,
) -> Callable[[StepCtx], None]:
    """A Trainer step hook probing the step ``cfg.step_source`` selects (`STEP_SOURCES`).

    The default, ``kfac`` (`KFACStep`), is the ``kfac`` method of Exp. 7
    (`build_couplefac_hook` at tau = inf) without its payoff bookkeeping. Every source
    shares the exact gradient off the estimator's graph and theta <- theta + d0
    (eta = 1). On a probe step the measurement runs between the direction and its
    application - pre-update weights, the step's own minibatch and exact oracle,
    deterministic kernels - in two parts: the curvature part while the estimator's
    graph is alive, the kink count after it is released (the count needs only
    forwards, and the graph is the step's largest allocation).
    """
    protocol = cfg.protocol
    loss_fn = nn.CrossEntropyLoss()
    source = STEP_SOURCES[cfg.step_source](model, protocol)
    estimator = ParamBlockEstimator(model, loss_fn)
    params = group_major_params(model)
    loss_at = trial_loss(model, loss_fn, params, SECOND_ORDER_LR)
    forward = trial_forward(model, params, SECOND_ORDER_LR)
    generator = torch.Generator(device=device).manual_seed(seed)
    probe_steps = set(cfg.probe_steps)

    def hook(ctx: StepCtx) -> None:
        cur = estimator.curvature(ctx.x, ctx.y)  # exact grad + HVP, and the loss
        trace.losses.append(cur.loss)
        d0 = source.direction(cur, ctx.x, ctx.y)
        if ctx.step in probe_steps:
            with deterministic_mode():
                record = probe_model_validity(
                    cur.flat_grad,
                    cur.hvp,
                    d0,
                    loss_at(ctx.x, ctx.y),
                    damping=protocol.damping,
                    t_grid=cfg.t_grid,
                    tolerance=cfg.tolerance,
                    n_lanczos_iter=cfg.n_lanczos_iter,
                    generator=generator,
                )
                del cur  # release the graph before the pattern forwards
                record["kinks"] = count_kink_crossings(
                    model, forward, d0, ctx.x, cfg.kink_t_grid
                )
            trace.probes.append({"step": ctx.step, **record})
            sharp = record["sharpness"]
            logger.info(
                "  probe step %d: ||d0|| %.3f, m(d0) %+.3f, rho %+.3f, t* %s, "
                "lambda_max %.4g, lambda_min %.4g (Lanczos %d steps, residual "
                "%.2g / %.2g, converged %s), lambda_max*eta_eff %.1f, "
                "ReLU flips at t=1 %.2e",
                ctx.step,
                record["decomposition"]["norm_d0"],
                record["payoff"]["m_kfac"],
                record["payoff"]["rho"],
                record["t_star"],
                sharp["lambda_max"],
                sharp["lambda_min"],
                sharp["n_iter_used"],
                sharp["residual_max"],
                sharp["residual_min"],
                sharp["converged"],
                sharp["lambda_max_x_eta_eff"],
                record["kinks"]["rows"][-1]["relu_frac"],
            )
        else:
            del cur
        apply_flat(params, d0, SECOND_ORDER_LR)

    return hook


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_tree(records: list[Any]) -> Any:
    """mean/std over seeds of every number of aligned records; coordinates verbatim."""
    ref = records[0]
    if isinstance(ref, dict):
        return {
            key: (
                ref[key]
                if key in _COORDINATES
                else _aggregate_tree([r[key] for r in records])
            )
            for key in ref
        }
    if isinstance(ref, list):
        return [_aggregate_tree(list(items)) for items in zip(*records, strict=True)]
    return mean_std([float(r) for r in records])


def _aggregate(seeds: dict[str, Any]) -> dict[str, Any]:
    """The probe records averaged over seeds, probe by probe.

    The probe steps are the same for every seed, so the records align by position;
    the coordinates (`_COORDINATES`) are kept, every other number becomes
    ``{mean, std}`` (a NaN t* - no valid grid point - propagates to the mean). The
    loss trajectories stay per seed.
    """
    payloads = list(seeds.values())
    return {
        "n_seeds": len(payloads),
        "probes": [
            _aggregate_tree(list(records))
            for records in zip(*(p["probes"] for p in payloads), strict=True)
        ],
    }


# ======================================================================
# Runner
# ======================================================================


class Exp11Runner:
    """Runner for Exp11 (the model validity radius along the K-FAC trajectory)."""

    def __init__(
        self,
        config: Exp11Config,
        device: torch.device,
        *,
        loaders: LoaderFactory | None = None,
    ) -> None:
        self._cfg = config
        self._device = device
        # Injectable data seam (tests pass tiny in-memory loaders); the default is the
        # protocol's own dataset through the Exp. 7 loaders (fixed split).
        self._loaders: LoaderFactory = loaders or self._protocol_loaders

    def _protocol_loaders(self) -> tuple[DataLoader, DataLoader]:
        return build_loaders(self._cfg.protocol)

    def run(self) -> dict[str, Any]:
        """Returns {meta, seeds: {seed_N: payload}, aggregate: {n_seeds, probes}}.

        A per-seed payload holds the probe records, the loss trajectory (one entry per
        step, the last probe step included), the validation accuracy after every epoch
        and the step count. The run trains for ``max(probe_steps) + 1`` steps - up to
        and including the last probe step - in as many epochs as that takes.
        """
        cfg = self._cfg
        train_loader, val_loader = self._loaders()
        n_steps = max(cfg.probe_steps) + 1
        epochs = max(1, math.ceil(n_steps / max(1, len(train_loader))))
        training = replace(cfg.protocol.training, epochs=epochs, max_steps=n_steps)

        seeds: dict[str, Any] = {}
        for seed in training.seeds:
            logger.info("=== Seed %d: %d K-FAC steps ===", seed, n_steps)
            seeds[f"seed_{seed}"] = self._run_seed(seed, training, val_loader)
        return {"meta": self._meta(), "seeds": seeds, "aggregate": _aggregate(seeds)}

    def _run_seed(
        self,
        seed: int,
        training: TrainingConfig,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """One K-FAC trajectory with its probes: the per-seed payload."""
        cfg = self._cfg
        set_seed(seed, deterministic=training.deterministic)
        # Build a fresh training loader after seeding each trajectory to isolate
        # persistent-worker RNG state. The shared validation loader does not shuffle
        # or augment.
        train_loader, _ = self._loaders()
        model = build_model(cfg.protocol).to(self._device)
        trace = ValidityTrace()
        hook = build_validity_hook(
            model, cfg, seed=seed, device=self._device, trace=trace
        )
        trainer = Trainer(model, training, self._device, step_hook=hook)
        val_acc: list[float] = []

        def record(epoch: int) -> None:
            val_acc.append(trainer.evaluate(val_loader)["test_acc"])
            logger.info("  epoch %d: val acc %.4f", epoch, val_acc[-1])

        trainer.train(train_loader, checkpoint_epochs=[], on_epoch_end=record)
        if self._device.type == "cuda":
            logger.info(
                "  peak GPU memory: %.2f GiB allocated",
                run_peak_allocated(self._device) / 2**30,
            )
        return {
            "probes": trace.probes,
            "loss_trajectory": trace.losses,
            "val_acc": val_acc,
            "n_steps": trainer.step_count,
        }

    def _meta(self) -> dict[str, Any]:
        """Flat protocol metadata with environment and code provenance."""
        cfg = self._cfg
        protocol = cfg.protocol
        return {
            "model": protocol.model,
            "dataset": protocol.dataset,
            "pretrained": protocol.pretrained,
            "batch_size": protocol.training.batch_size,
            "damping": protocol.damping,
            "lr": SECOND_ORDER_LR,
            "fisher_type": protocol.fisher_type,
            "kfac_refresh_period": protocol.kfac_refresh_period,
            "probe_steps": list(cfg.probe_steps),
            "t_grid": sorted(cfg.t_grid),
            "kink_t_grid": sorted(cfg.kink_t_grid),
            "tolerance": cfg.tolerance,
            "n_lanczos_iter": cfg.n_lanczos_iter,
            "step_source": cfg.step_source,
            "seeds": list(protocol.training.seeds),
            **environment_block(),
        }


# ======================================================================
# CLI: exp11 (model validity radius along the K-FAC trajectory)
# ======================================================================


def add_args(parser: argparse.ArgumentParser) -> None:
    add_device_args(parser)
    parser.add_argument(
        "--profile",
        type=str,
        default="b1",
        choices=["b1", "b1_mlp"],
        help="b1: the K-FAC run of Exp. 7 on the pretrained ResNet-50 / Stanford Cars "
        "(the headline); b1_mlp: the same probes on the synthetic full-rank MLP "
        "control (cheap, local).",
    )
    # Optional overrides; a value of None leaves the profile default untouched. The
    # protocol knobs (seeds, batch, lambda) go to the embedded Exp. 7 config, the probe
    # knobs to the exp11 config itself.
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="training minibatch (the probes run on it); the protocol's is 64",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="the fixed Tikhonov damping lambda of the K-FAC step and the model",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        nargs="+",
        default=None,
        help="steps probed before the step is applied; training ends after the last",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="relative model error |dL - m| / |dL| allowed inside the validity radius",
    )
    parser.add_argument("--n-lanczos-iter", type=int, default=None)
    parser.add_argument(
        "--step-source",
        type=str,
        default="kfac",
        choices=sorted(STEP_SOURCES),
        help="which trajectory's step the probes measure; kfac (default) is the b1 "
        "K-FAC step of Exp. 7, sgd is d0 = -g, adam is Exp. 7's adam "
        "hyperparameters with state accumulated along this trajectory",
    )


def build_config(args: argparse.Namespace) -> Exp11Config:
    """Selects the profile, then applies any explicit CLI overrides."""
    base = Exp11Config.b1_cars() if args.profile == "b1" else Exp11Config.b1_mlp()

    training = override(
        base.protocol.training,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )
    protocol = replace(
        base.protocol,
        training=training,
        **({} if args.damping is None else {"damping": args.damping}),
    )
    return override(
        base,
        protocol=protocol,
        probe_steps=args.probe_steps,
        tolerance=args.tolerance,
        n_lanczos_iter=args.n_lanczos_iter,
        step_source=args.step_source,
    )


SPEC = Experiment(
    name="exp11",
    help="Model validity radius along an optimization trajectory",
    add_args=add_args,
    build_config=build_config,
    run=Exp11Runner,
    default_output_dir=lambda args: f"results/exp11_{args.profile}",
)
