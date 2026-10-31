"""
Trust-region safeguard for the COUPLE-FAC overlay.

The overlay proposes a corrected direction d1 = d0 + sum_vw Delta_vw on top of the
block-diagonal (K-FAC) direction d0. The safeguard accepts d1 only if it stays inside a
trust region *relative* to the base step (d1 is a correction of d0, not a replacement),
does not worsen the *exact* damped local quadratic model the base step targets, and -
when the caller can afford two forward passes - does not lose to d0 on the minibatch
loss itself; otherwise it falls back to d0. With the damping lambda of the base step,

  m(d)  = g^T d + 1/2 d^T (H + lambda I) d          (the damped model: its minimizer is
                                                      the damped Newton step K-FAC
                                                      approximates),
  m0    = m(d0)                                     (H d0 is handed in: the overlay
                                                      already needed it for its residual),
  m1    = m(d1)                                     (one HVP; skipped when
                                                      ||d1|| > tr_ratio * ||d0||),
  L0, L1 = loss(theta + lr d0), loss(theta + lr d1)  (two forward passes on the same
                                                      minibatch; optional),
  accepted  iff  ||d1|| <= tr_ratio * ||d0||  and  m1 <= m0 + tol  and  L1 <= L0,
  d = d1 if accepted else d0
  =>  m(d) <= m0 + tol and ||d|| <= max(1, tr_ratio) * ||d0|| for finite values.
      L(d) <= L(d0) applies only when the optional loss test is supplied.

The three tests answer three different failures. The norm bound keeps the accepted step
on the scale the base preconditioner set. The model test is the Newton-type guarantee,
on the damped model, exact H + lambda I with the base step's lambda, so that it
penalizes length by lambda/2 ||d||^2 (the K-FAC step damps its Kronecker factors and
the EKFAC step its eigenbasis, so neither is this model's lambda I on the exact H).
The optional loss test compares the two candidate minibatch losses directly. The
model contains a damping penalty absent from the loss, and a model comparison alone
does not guarantee lower loss. The loss test costs two forward passes.

For an indefinite damped Hessian, the quadratic model can be unbounded below, so
m1 <= m0 alone can admit an arbitrarily long proposal. The norm bound limits accepted
proposals to the scale set by the base step; it does not guarantee training stability.

When the overlay returns d0 itself (no pair selected, so d1 *is* d0) there is nothing
to decide: the step is accepted trivially with m1 = m0 and no second HVP.

H enters only through the exact HessianOracle; the decision is returned with the
quantities it was made on (`SafeguardDecision`) so a runner can record them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from .types import HessianOracle


def model_value(grad: Tensor, d: Tensor, hd: Tensor, *, damping: float) -> float:
    """m(d) = g^T d + 1/2 d^T (H d) + damping/2 ||d||^2, given hd = H d (the damped model)."""
    return (grad.dot(d) + 0.5 * d.dot(hd) + 0.5 * damping * d.dot(d)).item()


LossAt = Callable[[Tensor], float]
"""Minibatch loss at theta + lr * direction, without moving the parameters."""


@dataclass(frozen=True)
class SafeguardDecision:
    """Outcome of one safeguard call: the direction taken and what it was judged on.

    - ``direction``: d1 if accepted else d0;
    - ``accepted``: whether the corrected step passed (trivially True when d1 is d0);
    - ``m0``, ``m1``: exact damped-model values m(d0), m(d1) - ``m1`` is NaN when d1
      fell outside the relative trust region (it was rejected before its HVP was spent);
    - ``norm_d0``, ``norm_d1``: Euclidean norms of the two candidates;
    - ``loss_d0``, ``loss_d1``: minibatch losses after the two candidate steps, NaN when
      the loss test was not reached or no ``loss_at`` was supplied;
    - ``rejected_by``: which test rejected d1 (``"ratio"``, ``"model"``, ``"loss"``), or
      None when accepted.
    """

    direction: Tensor
    accepted: bool
    m0: float
    m1: float
    norm_d0: float
    norm_d1: float
    loss_d0: float = math.nan
    loss_d1: float = math.nan
    rejected_by: str | None = None


@dataclass(frozen=True)
class TrustRegionSafeguard:
    """Apply the norm, damped-model and optional minibatch-loss tests."""

    tol: float = 0.0  # acceptance slack: accept when m(d1) <= m(d0) + tol

    def accept(
        self,
        grad: Tensor,
        d0: Tensor,
        d1: Tensor,
        hvp: HessianOracle,
        *,
        hd0: Tensor,
        tr_ratio: float,
        damping: float,
        loss_at: LossAt | None = None,
    ) -> SafeguardDecision:
        """Decide between d1 and d0: ratio test, damped-model test, then the loss test.

        ``hd0`` is H d0, computed once by the caller and shared with the overlay;
        ``tr_ratio`` is the trust region's radius in units of ||d0||; ``damping`` is the
        base step's Tikhonov damping, which defines the model; ``loss_at``, when given,
        evaluates the minibatch loss after a candidate step (two forward passes, spent
        only when the first two tests pass).
        """
        norm_d0 = d0.norm().item()
        norm_d1 = d1.norm().item()
        m0 = model_value(grad, d0, hd0, damping=damping)
        if d1 is d0:  # the overlay left d0 untouched: nothing to judge, no HVP
            return SafeguardDecision(d0, True, m0, m0, norm_d0, norm_d1)
        if norm_d1 > tr_ratio * norm_d0:
            return SafeguardDecision(
                d0, False, m0, math.nan, norm_d0, norm_d1, rejected_by="ratio"
            )
        m1 = model_value(grad, d1, hvp(d1), damping=damping)
        if m1 > m0 + self.tol:
            return SafeguardDecision(
                d0, False, m0, m1, norm_d0, norm_d1, rejected_by="model"
            )
        if loss_at is None:
            return SafeguardDecision(d1, True, m0, m1, norm_d0, norm_d1)
        loss_d0, loss_d1 = loss_at(d0), loss_at(d1)
        if loss_d1 > loss_d0:
            return SafeguardDecision(
                d0, False, m0, m1, norm_d0, norm_d1, loss_d0, loss_d1, "loss"
            )
        return SafeguardDecision(d1, True, m0, m1, norm_d0, norm_d1, loss_d0, loss_d1)
