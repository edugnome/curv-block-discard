"""
Inter-layer Hessian structure: the block Hessian in node-value (activation) and
parameter coordinates, its GN/tensor split, the discarded-coupling fraction Phi, and
matrix-free spectrum/conditioning tools built on top of either.

Two coordinate systems for the same inter-layer curvature, each with an exact and a
stochastic estimator:
  - activation space H^f_{v,w} = d^2L / df_v df_w (`exact.py`, `stochastic.py`), and its
    H = H^GN + H^T decomposition (`gn_decomposition.py`) into the Gauss-Newton part and
    the tensor remainder the GN-Gap measures;
  - parameter space H_{theta_v, theta_w} (`param_space.py`) - the curvature between
    parameter groups a block-diagonal preconditioner (K-FAC / EKFAC) actually sees - and
    the discarded-coupling fraction Phi it throws away.

`spectrum.py` measures the conditioning of any such block, or the full Hessian, from an
exact spectrum or matrix-free through an HVP closure: extreme eigenvalues, the damped
condition number a Newton solve sees, and the stable rank.

Public API:
  HessianBlock, BlockHessianResult - a single inter-layer block H^f_{v,w} and the
                         collection of blocks `ExactBlockHessian` returns.
  GNGapResult          - the H = H^GN + H^T decomposition's result for one block.
  LayerID              - a layer/activation name, the key both coordinate systems index
                         blocks by.
  ExactBlockHessian    - exact H^f_{v,w} via nested double-backward.
  hessian_of_loss_wrt_tensor, cross_hessian - the diagonal and cross blocks
                         d^2L/df_v^2 and d^2L/df_v df_w from raw tensors and a loss,
                         what `ExactBlockHessian` composes, or what a non-chain graph
                         without SegmentedModel calls directly.
  StochasticHessianEstimator - ||H^f_{v,w}||_F and its stable rank via Hutchinson + HVP,
                         without building the full matrix.
  ParamBlockEstimator  - Hutchinson estimation of the parameter-space blocks
                         H_{theta_v, theta_w} and of Phi, sharing one backward graph.
  ParamGroupedModel    - the protocol a model must satisfy to be estimated in parameter
                         space (trainable parameters partitioned into named groups).
  PhiReport            - Phi plus its diagonal-mass and coupling localization, for one
                         minibatch.
  phi_null             - Phi's block-size-only floor, 1 - sum_v p_v^2 / (sum_v p_v)^2;
                         equal groups give the 1 - 1/n_b ceiling a measured Phi reads
                         against.
  phi_sep              - the matched separable reference computed from one
                         measurement's own diagonal block norms; moves with the
                         weights (not a null), and phi_null is its a proportional to p
                         special case.
  n_eff                - the effective number of blocks carrying the diagonal
                         curvature mass, from the same diagonal norms as phi_sep.
  phi_rank1            - Phi of a rank-1 H = lambda u u^T for a given unit eigenvector;
                         a measured Phi near this value is reading eigenvector spread,
                         not block coupling.
  ExactGNDecomposition - exact H^GN_{v,w} via Jacobians, and the GN-Gap it implies.
  StochasticGNGapEstimator - stochastic GN-Gap via Hutchinson with shared probe vectors.
  compute_resonance, compute_coupling, compute_dimension, coupling_from_norms -
                         per-block metrics (Frobenius norm, inter-block coupling,
                         stable rank) derived from a `HessianBlock`.
  HVPClosure           - the z |-> H z closure type every matrix-free tool here takes.
  ExtremeRitz          - both extreme eigenvalues of a symmetric operator from one
                         Lanczos run; `extreme_ritz`'s result.
  extreme_ritz         - both extreme eigenvalues by Lanczos, floored by the Rayleigh
                         quotients of caller-supplied candidate directions.
  matrix_free_conditioning, conditioning_from_spectrum - the conditioning summary
                         (extreme eigenvalues, damped kappa_lambda, stable rank) from an
                         HVP closure or from a dense spectrum, respectively.

`_estimators.py` (the Hutchinson/power-iteration kernels shared by the activation- and
parameter-space estimators) and `memory.py`'s peak-memory bookkeeping are reached by
their own module paths rather than hoisted here: the former is exercised only by
white-box tests, and the latter is a small utility, not part of the measured maths.
`spectrum.py`'s power-iteration route is reached the same way, by its own module
path - documented single-eigenvalue kernels the summaries in that module do not
call (see `spectrum.py` for why).
"""

from __future__ import annotations

from .exact import ExactBlockHessian, cross_hessian, hessian_of_loss_wrt_tensor
from .gn_decomposition import ExactGNDecomposition, StochasticGNGapEstimator
from .metrics import (
    compute_coupling,
    compute_dimension,
    compute_resonance,
    coupling_from_norms,
)
from .param_space import ParamBlockEstimator, ParamGroupedModel, PhiReport
from .phi_baselines import n_eff, phi_null, phi_rank1, phi_sep
from .spectrum import (
    ExtremeRitz,
    HVPClosure,
    conditioning_from_spectrum,
    extreme_ritz,
    matrix_free_conditioning,
)
from .stochastic import StochasticHessianEstimator
from .types import BlockHessianResult, GNGapResult, HessianBlock, LayerID

__all__ = [
    "HessianBlock",
    "BlockHessianResult",
    "GNGapResult",
    "LayerID",
    "ExactBlockHessian",
    "hessian_of_loss_wrt_tensor",
    "cross_hessian",
    "StochasticHessianEstimator",
    "ParamBlockEstimator",
    "ParamGroupedModel",
    "PhiReport",
    "n_eff",
    "phi_null",
    "phi_rank1",
    "phi_sep",
    "ExactGNDecomposition",
    "StochasticGNGapEstimator",
    "compute_resonance",
    "compute_coupling",
    "compute_dimension",
    "coupling_from_norms",
    "HVPClosure",
    "ExtremeRitz",
    "extreme_ritz",
    "matrix_free_conditioning",
    "conditioning_from_spectrum",
]
