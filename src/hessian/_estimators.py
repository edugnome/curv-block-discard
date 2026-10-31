"""
Shared matrix-free estimation kernels (Hutchinson trace + power iteration).

These kernels operate on an abstract linear operator H given only through its
action z |-> H z (a "HVP closure"). Both estimators call into these kernels for
their numerical core:
  - the activation-space estimator (`stochastic.StochasticHessianEstimator`), and
  - the parameter-space estimator (`param_space.ParamBlockEstimator`).

Frobenius norm (Hutchinson):
  ||H||_F^2 = tr(H^T H) = E_z[||H z||^2],  z ~ Rademacher(dim),
estimated by the empirical mean over `n_probes` probes.

Spectral norm (power iteration on H^T H):
  v <- H^T(H v) / ||H^T(H v)||  converges to the top right-singular vector;
  sigma_1^2 = ||H v||^2  at convergence.

A "HVP closure" maps a probe vector with `dim` elements to H z; the closure owns
any internal reshaping. The kernels only read the squared L2 norm of the returned
tensor, so the closure may return H z in any shape.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

HVPClosure = Callable[[Tensor], Tensor]

_ZERO_NORM_TOL = 1e-12


def rademacher_signs(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Rademacher entries of ``shape`` in ``dtype`` (float32), drawn as int8 bits.

    The bits are drawn in int8 and mapped to {-1, +1} in place before the one float
    copy, so the draw peaks at 5 bytes per element. Drawing int64 and applying the
    float temporaries of ``bits.float() * 2 - 1`` instead peaks at 16 bytes per
    element: for a 0.5B-parameter model (2 GB of float) that is an 8 GB allocation,
    more than a 24 GB card holds alongside the rest of the working set (exp9
    ``a3_llm``). ``dtype`` lets a probe slice match a lower-precision gradient
    without a second copy.
    """
    bits = torch.randint(
        0, 2, shape, dtype=torch.int8, device=device, generator=generator
    )
    return bits.mul_(2).sub_(1).to(dtype)


def _rademacher(
    dim: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Random Rademacher vector z in {-1, +1}^dim."""
    return rademacher_signs((dim,), device, generator)


def rademacher_batch(
    n: int,
    dim: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``n`` stacked Rademacher vectors as one (n, dim) tensor in {-1, +1}.

    The batched counterpart of `_rademacher`, for vectorized (``is_grads_batched``)
    Hutchinson probes where the whole probe block is drawn and applied at once.
    """
    return rademacher_signs((n, dim), device, generator)


def hutchinson_frob_sq(
    hvp: HVPClosure,
    dim: int,
    n_probes: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> float:
    """Estimates ||H||_F^2 = E_z[||H z||^2] via Hutchinson with Rademacher probes.

    Args:
        hvp: closure z |-> H z, where z has `dim` elements.
        dim: dimension of the probe (the domain of H).
        n_probes: number of Rademacher probes m.
        device: device for probe generation.
        generator: optional RNG for reproducibility (default: global RNG).

    Returns:
        (1/m) sum_k ||H z_k||^2 - an unbiased estimate of ||H||_F^2.
    """
    frob_sq = torch.zeros((), device=device)
    for _ in range(n_probes):
        z = _rademacher(dim, device, generator)
        hz = hvp(z)
        frob_sq = frob_sq + (hz**2).sum()
    return (frob_sq / n_probes).item()


def power_iteration_spectral_sq(
    hvp: HVPClosure,
    hvp_adj: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> float:
    """Estimates sigma_1^2 = ||H||_2^2 via power iteration on H^T H.

    Args:
        hvp: closure z |-> H z (z has `dim` elements, the domain of H).
        hvp_adj: adjoint closure u |-> H^T u.
        dim: dimension of the domain of H (the start vector lives here).
        n_iter: number of power-iteration steps.
        device: device for the start vector.
        generator: optional RNG for reproducibility (default: global RNG).

    Returns:
        sigma_1^2 = ||H v||^2 at convergence (0.0 if H is numerically zero).
    """
    v = torch.randn(dim, device=device, generator=generator)
    v = v / v.norm()

    for _ in range(n_iter):
        w = hvp(v)  # H v        (lives in the codomain of H)
        v_new = hvp_adj(w)  # H^T (H v)  (lives in the domain of H)
        nrm = v_new.reshape(-1).norm()
        if nrm < _ZERO_NORM_TOL:
            return 0.0
        v = (v_new / nrm).reshape(-1)

    hv = hvp(v)
    return (hv**2).sum().item()
