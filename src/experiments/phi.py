"""Phi at a checkpoint: the blind-spot measurement the training runners share.

Both the diagnosis (exp8) and the repair (exp7) read, at fixed points of a training run,
the discarded-coupling fraction

  Phi = 1 - sum_v ||H_{theta_v, theta_v}||_F^2 / ||H||_F^2

together with the per-block diagonal norms ||H_{vv}||_F and the coupling field C(v, w)
all from one parameter-space backward graph
(`hessian.param_space.ParamBlockEstimator.estimate_phi_report`). The measurement is
taken on one fixed validation minibatch and, whatever kernels training runs on, under
`deterministic_mode` so the Hutchinson estimate is bit-reproducible given its probes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn
from torch import Generator, Tensor

from experiments.utils import deterministic_mode
from hessian import ParamBlockEstimator, ParamGroupedModel, PhiReport, phi_rank1
from hessian.spectrum import DEFAULT_LANCZOS_ITER, extreme_ritz


def group_sizes_of(model: ParamGroupedModel) -> list[int]:
    """Parameter counts of ``model``'s groups, in `get_param_groups`'s own order.

    This is the architecture's partition {p_v} that `phi_null` and `phi_rank1` need -
    fixed by the model's shape alone, never by the trained weights or the minibatch a
    Phi report is taken on - so it is exactly reproducible from the model object
    whenever a caller has one (`measure_phi` below; `experiments.group_sizes` rebuilds
    a model on CPU just to call this). Kept in `get_param_groups` order (not sorted) so
    a caller zipping it against ``diag_frob`` - whose keys come from the same
    dict, in the same order - or indexing into a Lanczos eigenvector by cumulative
    group size (`phi_rank1`) lines each size up with the right block.
    """
    return [sum(p.numel() for p in ps) for ps in model.get_param_groups().values()]


def measure_phi(
    model: ParamGroupedModel,
    x: Tensor,
    y: Tensor,
    loss_fn: nn.Module,
    n_probes: int,
    *,
    with_rank1: bool = False,
    n_lanczos_iter: int = DEFAULT_LANCZOS_ITER,
    generator: Generator | None = None,
    coupling_pairs: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """JSON-safe Phi report on one minibatch: Phi, diagonal norms, coupling field, and
    the group sizes behind them.

    The caller owns the model mode (the runners measure in ``eval`` so BatchNorm uses
    its running statistics and the curvature is that of the deployed network).

    ``with_rank1`` adds ``phi_rank1`` for the approximate top direction returned by
    `hessian.spectrum.extreme_ritz` and ``lambda_max``, the largest of its Ritz and
    measured Rayleigh estimates. The reported value need not equal the direction's
    Rayleigh quotient.
    A close match to ``phi_rank1`` alone does not establish that the top direction
    explains the measured Phi or identify the source of coupling. Costs one Lanczos run of
    ``n_lanczos_iter`` steps (`hessian.spectrum.extreme_ritz`) on top of the probes.

    ``generator`` drives the Hutchinson probes (passed straight through to
    `ParamBlockEstimator`; it must live on ``model``'s device). ``None`` (the default)
    draws from the ambient global RNG, so a caller comparing two measurements at
    different ``n_probes`` needs its own generator, seeded identically, to isolate the
    probe count as the only variable - otherwise the two draws differ by wherever the
    global RNG happened to be as well.

    ``coupling_pairs`` restricts the coupling field to a caller-chosen subset of (v, w)
    pairs, passed straight through to `ParamBlockEstimator.estimate_phi_report`. ``None``
    (the default) measures every v-before-w pair, a blockwise estimate of
    ``||H||_F^2``; a subset - e.g. a depth band on a many-block transformer
    (`experiments.runner_exp9._coupling_pairs`) - limits the coupling field and makes
    that estimator fall back to a full-parameter Hutchinson probe for ``||H||_F^2``
    instead (see its own docstring for why).
    """
    with deterministic_mode():
        estimator = ParamBlockEstimator(
            model, loss_fn, n_probes=n_probes, generator=generator
        )
        out = phi_report_to_dict(
            estimator.estimate_phi_report(x, y, coupling_pairs=coupling_pairs)
        )
        sizes = group_sizes_of(model)
        out["group_sizes"] = sizes
        if with_rank1:
            cur = estimator.curvature(x, y)
            dim, device = cur.flat_grad.numel(), cur.flat_grad.device
            ritz = extreme_ritz(
                cur.hvp, dim, n_lanczos_iter, device, with_eigenvector=True
            )
            v = ritz.eigenvector
            assert v is not None  # with_eigenvector=True
            out["phi_rank1"] = phi_rank1(v, sizes)
            out["lambda_max"] = ritz.lambda_max
    return out


def measure_phi_only(
    model: ParamGroupedModel,
    x: Tensor,
    y: Tensor,
    loss_fn: nn.Module,
    n_probes: int,
    *,
    generator: Generator | None = None,
) -> float:
    """Phi alone (`ParamBlockEstimator.estimate_phi`), none of `measure_phi`'s coupling
    field or diagonal norms.

    ``estimate_phi`` gets ||H||_F^2 from one dedicated full-parameter probe set rather
    than from the blockwise estimate `measure_phi` reconstructs out of every
    off-diagonal pair. The estimates may differ due to probe noise; and this one skips the O(groups^2) coupling pairs
    entirely, the term that makes a fine granularity level (e.g. one group per
    Conv2d/Linear) expensive for a report nothing reads. Same `deterministic_mode` and
    ``generator`` passthrough as `measure_phi`.
    """
    with deterministic_mode():
        estimator = ParamBlockEstimator(
            model, loss_fn, n_probes=n_probes, generator=generator
        )
        return estimator.estimate_phi(x, y)


def measure_phi_and_diag_frob(
    model: ParamGroupedModel,
    x: Tensor,
    y: Tensor,
    loss_fn: nn.Module,
    n_probes: int,
    *,
    generator: Generator | None = None,
) -> dict[str, Any]:
    """Phi plus the per-group diagonal norms ||H_{vv}||_F, none of `measure_phi`'s
    coupling field or its optional extras (``group_sizes``, ``phi_rank1``, ``lambda_max``).

    Built on `ParamBlockEstimator.estimate_phi_report` with ``coupling_pairs=[]``: the
    empty subset measures no coupling pair, so ``||H||_F^2`` takes the same fallback
    `estimate_phi` uses - one dedicated full-parameter Hutchinson probe - instead of the
    blockwise estimate `measure_phi` reconstructs out of every off-diagonal pair.
    That makes this call cost exactly what `measure_phi_only` costs (one full-parameter
    probe set plus one group-localized set per diagonal block) and nothing more: the
    coupling field itself is what needs the O(groups^2) pairs a fine granularity level
    cannot afford (see `measure_phi_only`'s docstring and `runner_exp8`'s granularity
    axis), and this sibling exists precisely so a caller can have the diagonal norms
    without paying for it.

    The norms alone do not carry Phi's off-diagonal information, but together with the
    returned ``phi`` they determine everything `measure_phi`'s coupling field would
    additionally locate in aggregate: with ``a_v = diag_frob[v]``,
    ``||H||_F^2 = sum_v a_v^2 / (1 - phi)``, hence ``Phi_sep(a) = 1 - sum_v a_v^2 /
    (sum_v a_v)^2`` (the reference at the same diagonal profile ``a`` when every
    off-diagonal squared block norm equals ``a_v * a_w``) and
    ``S = (sum_v a_v)^2 / ||H||_F^2``
    (the profile-envelope to full-mass ratio) - the two references not determined by
    Phi alone. Neither is stored here: both are one line from
    ``phi`` and ``diag_frob``, so a stored ``||H||_F^2`` would just be a redundant,
    independently-rounding-error-prone copy of the same quantity.

    Same `deterministic_mode` and ``generator`` passthrough as `measure_phi_only`.
    """
    with deterministic_mode():
        estimator = ParamBlockEstimator(
            model, loss_fn, n_probes=n_probes, generator=generator
        )
        report = estimator.estimate_phi_report(x, y, coupling_pairs=[])
        return _phi_and_diag_frob_to_dict(report)


def _phi_and_diag_frob_to_dict(report: PhiReport) -> dict[str, Any]:
    """The ``phi`` + ``diag_frob`` prefix every JSON view of a `PhiReport` shares.

    ``diag_frob`` is copied out of the (already group-major) ``dict`` the report
    carries, so its key order - `get_param_groups` order - survives into JSON.
    """
    return {"phi": report.phi, "diag_frob": dict(report.diag_frob)}


def phi_report_to_dict(report: PhiReport) -> dict[str, Any]:
    """JSON-safe view of a `PhiReport`: scalar Phi, diagonal norms, coupling field.

    Coupling keys are flattened from ``(v, w)`` tuples to ``"v->w"`` strings so the bundle
    serializes directly. Shared by every Phi runner (exp7, exp8, exp9).
    """
    return {
        **_phi_and_diag_frob_to_dict(report),
        "coupling": {f"{v}->{w}": c for (v, w), c in report.coupling.items()},
    }
