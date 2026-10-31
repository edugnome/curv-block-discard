"""Reference values for Phi: what the discarded-coupling fraction reads against.

Phi = 1 - sum_v ||H_vv||_F^2 / ||H||_F^2 is the relative squared block-diagonal
projection error at a specified partition. The references below support comparative
interpretation of that error. All three share one
formula, 1 - sum_v x_v^2 / (sum_v x_v)^2, on a different x:

  phi_null   - a null model: the block sizes and nothing else. Every entry of H drawn
               from one distribution makes ||H_vw||_F^2 proportional to p_v p_w, so
               x = p_v (the group sizes) and the diagonal mass is
               sum_v p_v^2 / (sum_v p_v)^2. Equal groups give phi_null = 1 - 1/n_b; the
               same number is the ceiling of phi_rank1 (Cauchy-Schwarz), not of Phi,
               which the partition does not bound; a ResNet-18's group sizes put
               phi_null at 0.404. Fixed per architecture - it never reads a
               measurement.
  phi_sep    - NOT a null: a matched reference computed from one measurement's own
               diagonal block norms, x = a_v = ||H_vv||_F (a bundle's diag_frob, taken
               plain, not squared - see param_space.PhiReport.diag_frob), so it moves
               with the weights instead of standing still (e.g. 0.745 at init, 0.771
               mid-training and 0.698 at final, measured on a ResNet-18).
               Phi = phi_sep(a) exactly iff the (a_v a_w)-weighted mean of ||H_vw||_F^2
               / (a_v a_w) over v != w equals 1 - an aggregate condition on the
               off-diagonal mass as a whole. Pointwise separability, ||H_vw||_F =
               sqrt(a_v a_w) for every v != w, is sufficient for this but not
               necessary: a shortfall in some pairs can be offset by an excess in
               others while the weighted mean still lands on 1. A rank-1 H satisfies
               the pointwise case exactly and an iid-entry H satisfies it in
               expectation, which is exactly where phi_null lives: phi_null is
               phi_sep's special case at a proportional to p. Also Phi > phi_sep iff
               ||H||_F > sum_v ||H_vv||_F.
  phi_rank1  - the other extreme: H = lambda u u^T, one direction and no block
               structure at all, for which ||H_vw||_F^2 = lambda^2 ||u_v||^2 ||u_w||^2
               and Phi = 1 - sum_v ||u_v||^4 for a unit u. A measured Phi close to this
               says Phi is reading how spread the top eigenvector is rather than how
               coupled the blocks are.

phi_null and phi_sep use available group sizes and measured individual diagonal
norms, respectively. Computing these references adds no HVP once those inputs are
available; obtaining the norms is not free. phi_rank1 additionally needs an
eigenvector, whose optional Lanczos calculation incurs separate HVPs.

n_eff (1 / sum_v share_v^2, share_v = ||H_vv||_F^2 / sum_w ||H_ww||_F^2) is not a
reference for Phi - it does not compare against ||H||_F^2 at all - but a companion
summary of the same diagonal norms, from the same plain ||H_vv||_F as phi_sep: the
effective number of blocks carrying the diagonal curvature mass, from 1 (one block has
it all) to n_b (every block has an equal share).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor


def _as_values(x: Sequence[float] | Mapping[str, float]) -> list[float]:
    """``x``'s values in order, whether given as a sequence or a mapping."""
    return list(x.values()) if isinstance(x, Mapping) else list(x)


def _one_minus_ratio(x: Sequence[float]) -> float:
    """1 - sum_v x_v^2 / (sum_v x_v)^2: the diagonal-mass identity phi_null and phi_sep
    share, ``x`` being the group sizes or the diagonal block scales respectively.
    """
    total = float(sum(x))
    diag = sum(float(v) ** 2 for v in x)
    return 1.0 - diag / total**2


def phi_null(group_sizes: Sequence[int]) -> float:
    """1 - sum_v p_v^2 / (sum_v p_v)^2: Phi with no structure beyond the block sizes."""
    if not group_sizes:
        raise ValueError("phi_null needs a non-empty list of group sizes")
    if any(p <= 0 for p in group_sizes):
        raise ValueError("every group size must be positive")
    return _one_minus_ratio(group_sizes)


def phi_sep(block_norms: Sequence[float] | Mapping[str, float]) -> float:
    r"""1 - sum_v a_v^2 / (sum_v a_v)^2 for a_v = block_norms_v: the matched
    separable reference (see module docstring), computed from one measurement's own
    diagonal block norms rather than from the partition alone.

    ``block_norms`` is a sequence of nonnegative ``||H_vv||_F`` - the plain per-block
    Frobenius norm, not squared, exactly as a bundle stores it
    (`hessian.param_space.PhiReport.diag_frob`) - or a mapping block -> ``||H_vv||_F``
    (its values taken in insertion order, e.g. the mapping form the exp8/exp9 bundles
    use directly).
    """
    values = _as_values(block_norms)
    if not values:
        raise ValueError("phi_sep needs a non-empty list of diagonal block norms")
    if any(v < 0 for v in values):
        raise ValueError("every diagonal block norm must be nonnegative")
    return _one_minus_ratio(values)


def n_eff(block_norms: Sequence[float] | Mapping[str, float]) -> float:
    """1 / sum_v share_v^2, share_v = a_v^2 / sum_w a_w^2: the effective number of
    blocks carrying the diagonal curvature mass (see module docstring).

    share_v is a share of *squared* mass - Phi's own diagonal term is a ratio of
    squared norms - even though, like `phi_sep`, ``block_norms`` takes the plain
    ||H_vv||_F. Same input convention as `phi_sep`.
    """
    values = _as_values(block_norms)
    if not values:
        raise ValueError("n_eff needs a non-empty list of diagonal block norms")
    if any(v < 0 for v in values):
        raise ValueError("every diagonal block norm must be nonnegative")
    squares = [v**2 for v in values]
    total_sq = float(sum(squares))
    shares_sq = sum((s / total_sq) ** 2 for s in squares)
    return 1.0 / shares_sq


def phi_rank1(vector: Tensor, group_sizes: Sequence[int]) -> float:
    """1 - sum_v ||u_v||^4 for the unit u: Phi of a rank-1 H = lambda u u^T.

    ``vector`` is the flat top eigenvector in the same group-major layout the estimator
    flattens parameters in; it is normalized here, so an unnormalized Ritz vector is
    fine. ``lambda`` cancels, so only the direction matters.
    """
    if vector.ndim != 1:
        raise ValueError(f"expected a flat vector, got shape {tuple(vector.shape)}")
    if int(vector.numel()) != sum(group_sizes):
        raise ValueError(
            f"vector of {int(vector.numel())} entries does not match "
            f"{sum(group_sizes)} grouped parameters"
        )
    u = vector.double()
    u = u / u.norm()
    out = 0.0
    start = 0
    for p in group_sizes:
        out += float(u[start : start + p].pow(2).sum()) ** 2
        start += p
    return 1.0 - out
