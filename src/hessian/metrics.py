"""Metrics derived from inter-layer Hessian blocks."""

from __future__ import annotations

import math

from .types import HessianBlock


def compute_resonance(block: HessianBlock) -> float:
    """R(v, w) = ||H^f_{v,w}||_F."""
    return block.frobenius_norm


def compute_coupling(
    block_vw: HessianBlock,
    block_vv: HessianBlock,
    block_ww: HessianBlock,
) -> float:
    """C(v, w) = ||H^f_{v,w}||_F / sqrt(||H^f_{v,v}||_F * ||H^f_{w,w}||_F).

    C > 1 on at least one pair is possible only if H^f is not positive semi-definite
    (`coupling_from_norms` has the argument).
    """
    return coupling_from_norms(
        block_vw.frobenius_norm,
        block_vv.frobenius_norm,
        block_ww.frobenius_norm,
    )


def coupling_from_norms(
    norm_vw: float,
    norm_vv: float,
    norm_ww: float,
) -> float:
    """C(v,w) from raw Frobenius norms.

    C > 1 on at least one pair is possible only if H is not positive semi-definite: for
    PSD H = X^T X, with X_v the columns of group v,
    ||H_vw||_F^2 = tr(X_v X_v^T X_w X_w^T) <= ||H_vv||_F ||H_ww||_F (Cauchy-Schwarz for
    the trace inner product), so C <= 1 on every pair, with equality for a rank-one H.
    Clamping with min(., 1) would hide violations of this PSD bound.
    """
    denom = (norm_vv * norm_ww) ** 0.5
    if denom < 1e-12:
        return math.nan
    return norm_vw / denom


def compute_dimension(block: HessianBlock) -> float:
    """D(v, w) = ||H^f_{v,w}||_F^2 / ||H^f_{v,w}||_2^2 (stable rank)."""
    return block.stable_rank
