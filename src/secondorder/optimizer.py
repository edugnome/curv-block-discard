"""
CoupleFacOptimizer: composition of a K-FAC provider, the COUPLE-FAC overlay, and the
trust-region safeguard.

DIP: the optimizer depends on the `KFACProvider` abstraction and an exact
`HessianOracle`, never on a concrete K-FAC library. Per step, the accepted update
direction is

  d0  = provider.base_direction(g)                              # K-FAC / EKFAC step
  hd0 = H d0                                                    # one HVP, shared
  on a refresh step (or with an empty cache):
    S   = overlay.select(coupling, tau)                         # coupled block pairs
    U   = overlay.build(d0, hvp, provider.block_ranges(), S)    # cached OverlaySubspace
  d1  = overlay.correct(g, d0, hd0, U)                          # cached reduced solves
  d   = safeguard.accept(g, d0, d1, hvp, hd0=hd0, tr_ratio=tr_ratio)  # no-harm

The subspace U (joint basis and reduced-Hessian eigenfactors) is the expensive part;
it is rebuilt only when the caller passes ``refresh=True`` - the exp7 hook does so
on the same cadence it re-measures the coupling - and reused in between, so a
non-refresh step costs at most two HVPs (H d0, H d1) whatever the selection. H d0 is
computed once and handed to both the overlay (its damped residual g + H d0 + lambda d0) and the safeguard (m(d0));
the safeguard spends one more HVP on m(d1) only when there is a distinct d1 inside the
relative trust region ||d1|| <= tr_ratio ||d0||, and evaluates the exact model on every
step. For finite values, the accepted direction satisfies m(d) <= m(d0) + tol and
||d|| <= max(1, tr_ratio) ||d0||; the loss check is optional. The step returns a `StepReport` - the direction with the
safeguard's decision and the number of cached pairs in use - so a runner can account
for what the overlay did without recomputing anything.

The model-free direction algebra (`compute_direction`) is validated against exact
random quadratics. `apply` then adds that direction in place to a held parameter list
- the group-major `block_ranges` layout - and `step` is the two in sequence, given the
per-batch gradient, exact Hessian-vector product, and parameter-space coupling the
caller supplies (the experiment runner builds these from the model with
`hessian.param_space.ParamBlockEstimator` and refreshes the provider's curvature on the
same minibatch). The concrete K-FAC backend lives behind the `KFACProvider` seam
(`secondorder.kfac_adapter.CurvlinopsKFACProvider`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import Parameter

from hessian.types import LayerID

from .overlay import CoupleFacOverlay, OverlaySubspace
from .safeguard import LossAt, SafeguardDecision, TrustRegionSafeguard
from .types import HessianOracle, Pair


class KFACProvider(Protocol):
    """A block-diagonal second-order backend (K-FAC / EKFAC).

    Supplies the base preconditioned direction and the parameter-group block layout
    the overlay needs. It owns no training logic (SRP) and is the only seam the
    optimizer binds to a concrete library (DIP).
    """

    def base_direction(self, grad: Tensor) -> Tensor:
        """d0 = -(blkdiag H_vv + lambda I)^{-1} grad  (flat, full-parameter)."""
        ...

    def block_ranges(self) -> dict[LayerID, slice]:
        """Contiguous slice of the flat parameter vector owned by each block."""
        ...


@dataclass(frozen=True)
class StepReport:
    """One COUPLE-FAC step: the direction taken, how it was decided, and on what.

    - ``direction``: the accepted update d (== ``decision.direction``);
    - ``d0``: the block-diagonal base direction the overlay corrected (kept so a
      runner can measure the cross-block curvature it ignores, H_vw d0_w);
    - ``decision``: the safeguard's verdict with m(d0), m(d1) and the candidate norms;
    - ``n_selected_pairs``: |S|, the cached coupled pairs the overlay corrected with.
    """

    direction: Tensor
    d0: Tensor
    decision: SafeguardDecision
    n_selected_pairs: int


@dataclass
class CoupleFacOptimizer:
    """Coupling-gated, safeguarded overlay over a block-diagonal K-FAC backend.

    Owns the overlay's subspace cache: `compute_direction` rebuilds it on a refresh
    step (select + build) and reuses it otherwise.
    """

    provider: KFACProvider
    overlay: CoupleFacOverlay = field(default_factory=CoupleFacOverlay)
    safeguard: TrustRegionSafeguard = field(default_factory=TrustRegionSafeguard)
    tau: float = 0.1
    tr_ratio: float = 2.0  # relative trust region: accept d1 only if ||d1|| <= 2 ||d0||
    params: list[Parameter] | None = None
    lr: float = 1.0
    _subspace: OverlaySubspace | None = field(default=None, init=False, repr=False)

    def compute_direction(
        self,
        grad: Tensor,
        hvp: HessianOracle,
        coupling: dict[Pair, float],
        *,
        refresh: bool,
        generator: torch.Generator | None = None,
        loss_at: LossAt | None = None,
    ) -> StepReport:
        """Report the accepted direction and safeguard decision.

        For finite values the model comparison allows the safeguard's tolerance; the
        optional ``loss_at`` also compares the two minibatch losses.

        With ``refresh`` (or before the first build) the coupled pairs are re-selected
        from ``coupling`` and their subspaces rebuilt on this step's d0 and H; otherwise
        the cached subspaces are reused and ``coupling`` is not consulted. The damped
        model and the residual use the overlay's damping - the base step's, by
        construction of the runner (one shared `Damping`, read here at its current value).
        """
        d0 = self.provider.base_direction(grad)
        hd0 = hvp(d0)
        if refresh or self._subspace is None:
            selected = self.overlay.select(coupling, self.tau)
            self._subspace = self.overlay.build(
                d0, hvp, self.provider.block_ranges(), selected, generator=generator
            )
        subspace = self._subspace
        d1 = self.overlay.correct(grad, d0, hd0, subspace)
        decision = self.safeguard.accept(
            grad,
            d0,
            d1,
            hvp,
            hd0=hd0,
            tr_ratio=self.tr_ratio,
            damping=float(self.overlay.damping),
            loss_at=loss_at,
        )
        return StepReport(
            direction=decision.direction,
            d0=d0,
            decision=decision,
            n_selected_pairs=len(subspace.pairs) if subspace is not None else 0,
        )

    def step(
        self,
        grad: Tensor,
        hvp: HessianOracle,
        coupling: dict[Pair, float],
        *,
        refresh: bool,
        generator: torch.Generator | None = None,
        loss_at: LossAt | None = None,
    ) -> StepReport:
        """Apply one accepted COUPLE-FAC update in place; return its report.

        ``grad``, ``hvp`` and ``coupling`` are the per-batch parameter-space gradient,
        exact Hessian-vector product, and block coupling - the caller builds them on the
        minibatch used to refresh the provider's curvature (e.g. via
        `hessian.param_space.ParamBlockEstimator`). `compute_direction` followed by
        `apply`; a caller that must inspect the direction before the parameters move
        (the exact HVP oracle is tied to the pre-update weights) calls the two itself.
        """
        report = self.compute_direction(
            grad, hvp, coupling, refresh=refresh, generator=generator, loss_at=loss_at
        )
        self.apply(report.direction)
        return report

    def apply(self, direction: Tensor) -> None:
        """theta <- theta + lr * direction over the held group-major parameter list."""
        params = self.params
        if params is None:
            raise RuntimeError(
                "CoupleFacOptimizer.apply needs `params` (the group-major parameter "
                "list matching the provider's block layout); it was not set"
            )
        with torch.no_grad():
            offset = 0
            for p in params:
                n = p.numel()
                p.add_(self.lr * direction[offset : offset + n].view_as(p))
                offset += n
