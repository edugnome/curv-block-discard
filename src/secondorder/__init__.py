"""
Second-order optimization: the COUPLE-FAC overlay over K-FAC.

Where `hessian/` *measures* inter-layer curvature, this package *acts* on it: a
coupling-gated, trust-region-safeguarded rank-r saddle-free correction applied on top
of a block-diagonal K-FAC / EKFAC direction. The correction uses selected cross-block
actions; it does not in general recover all off-diagonal curvature. For finite values,
the safeguard limits the damped quadratic model value to the base value plus tolerance
and bounds the accepted norm relative to the base norm. A loss test is optional.

Public API:
  TrustRegionSafeguard - accept/reject a correction by the exact local model inside
                         the relative trust region ||d1|| <= tr_ratio ||d0||.
  SafeguardDecision    - the verdict with the model values it was made on.
  CoupleFacOverlay     - select coupled blocks; build a joint reduced system; correct
                         (saddle-free) on cached eigenfactors every step.
  OverlaySubspace      - the cached joint basis U over the selected pairs and the
                         eigenfactors of U^T H U.
  CoupleFacOptimizer   - composition (K-FAC provider + overlay + safeguard); owns the
                         subspace cache, rebuilt on refresh steps.
  StepReport           - one step's direction, base direction, decision, |S|.
  KFACProvider         - abstraction a K-FAC / EKFAC backend must satisfy (DIP).
  Damping              - the run's one Tikhonov damping lambda, shared by reference by
                         the provider, the overlay, the safeguard and the oracle, and
                         adapted by the Levenberg-Marquardt rule.
  HessianOracle, Pair  - the matrix-free HVP closure and a coupled block pair.

The core above is library-independent (DIP). The concrete curvlinops-backed provider
lives in the opt-in submodule ``secondorder.kfac_adapter`` (``CurvlinopsKFACProvider``)
so that importing the overlay never pulls in a K-FAC library.
"""

from __future__ import annotations

from .damping import Damping
from .optimizer import CoupleFacOptimizer, KFACProvider, StepReport
from .overlay import CoupleFacOverlay, OverlaySubspace
from .safeguard import LossAt, SafeguardDecision, TrustRegionSafeguard
from .types import HessianOracle, Pair

__all__ = [
    "CoupleFacOptimizer",
    "CoupleFacOverlay",
    "Damping",
    "HessianOracle",
    "KFACProvider",
    "Pair",
    "OverlaySubspace",
    "LossAt",
    "SafeguardDecision",
    "StepReport",
    "TrustRegionSafeguard",
]
