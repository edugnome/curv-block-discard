"""
Spectral summaries of a symmetric operator: extreme eigenvalues, damped condition
number, stable rank - from an exact spectrum or matrix-free through an HVP closure.

For a symmetric A (a diagonal parameter block H_{theta_v, theta_v}, an activation-space
block H^f_{v,v}, or the full Hessian) the appendix "Conditioning of the blocks" reports

  lambda_max(A), lambda_min(A)      extreme (algebraic) eigenvalues
  sigma(A) = ||A||_2                spectral norm = max(|lambda_max|, |lambda_min|)
  kappa_lambda(A) = (lambda_max + lambda) / (lambda_min + lambda)
                                    if lambda_min + lambda > 0, inf otherwise
                                    (the damped condition number a solve of
                                    (A + lambda I) d = -g sees)
  D(A) = ||A||_F^2 / ||A||_2^2      stable rank
  neg_frac(A) = #{i: lambda_i < -tol * max_j |lambda_j|} / n
                                    fraction of negative eigenvalues (exact spectra only)

Exact route (`conditioning_from_spectrum`): the eigenvalues of the dense symmetrized
matrix (torch.linalg.eigvalsh, float64) give every quantity directly.

Matrix-free route (`matrix_free_conditioning`, `extreme_eigenvalues`), given only
z |-> A z: Lanczos (`lanczos_tridiagonal`, `extreme_ritz`).
  m steps of the symmetric three-term recurrence on A itself - no shift - build the
  tridiagonal T_m = V_m^T A V_m of the Krylov space K_m(A, q_1). Its extreme
  eigenvalues (Ritz values, taken in float64) estimate both ends at once,
    lambda_max ~ theta_1 = lambda_max(T_m),   lambda_min ~ theta_m = lambda_min(T_m),
  with the standard residual bounds beta_m |y_i[m]| and sigma = max(|theta_1|,
  |theta_m|). The one-sided guarantees the appendix claims survive: in exact arithmetic
  theta_1 = max_{y} (V y)^T A (V y) / ||V y||^2 <= lambda_max(A) and, mirrored,
  theta_m >= lambda_min(A), so lambda_max is still estimated from below, lambda_min
  from above, and every kappa_lambda stays conservative.
  The basis is *not* stored: three vectors and the recurrence coefficients are all that
  is held (a 2.4e7-dimensional probe next to an 11 GB retained backward graph cannot
  afford a 60-vector basis - 5.7 GB - so full reorthogonalisation is out), and
  orthogonality is kept only locally, against the previous two vectors. The price of
  that is spurious *copies* of already converged eigenvalues, not Ritz values above
  lambda_max(A).
  ||A||_F^2 is Hutchinson's E_z ||A z||^2 over Rademacher probes, unchanged.
  The extreme Ritz *values* are indifferent to that lost orthogonality; the top Ritz
  *vector*, reconstructed by a second pass over the recurrence, is not, so it is
  polished by a few power steps on the PSD shift (`DEFAULT_RITZ_REFINE`).
  Budget: m HVPs, one more per candidate direction and, for `with_eigenvector`,
  m - 1 for the second pass plus `DEFAULT_RITZ_REFINE` + 1 for the polish and its
  Rayleigh quotient; plus n_probes for ||A||_F^2.

Why not the power iteration. sigma = ||A||_2 does converge by power iteration on A,
but reading lambda_max off a power iteration on the PSD shift B = A + sigma I fails
exactly where it matters. When the dominant eigenvalue is the negative
one (|lambda_min| = sigma >> lambda_max), the spectrum of B is [0, lambda_max + sigma]
and the bulk of A near zero maps to ~sigma, so the ratio between the wanted top and the
bulk is 1 + lambda_max / sigma ~ 1: the iteration converges nowhere and the Rayleigh
quotient returned is essentially the bulk mean - a lower bound so weak that it can fall
below the curvature measured along an explicit direction, which is impossible for a
correct lambda_max. Evidence: exp11 step 0 on ResNet-50 / Stanford Cars gave
lambda_max = 1.09e-3 with sigma = 56.8 while d0^T H d0 / ||d0||^2 = 0.101 and
g^T H g / ||g||^2 = 0.176 on the same Hessian (results/exp11_lambdamax_unfixed). A
smaller shift does not help - the obstruction is the near-zero bulk, not the dominant
vector - so the summaries use Lanczos, whose Krylov space resolves both ends at once.

Power-iteration route (`spectral_norm_power`, `lambda_max_power`,
`lambda_max_eigenpair`, `lambda_min_power`): documented single-eigenvalue kernels,
with their shift identities exact for any shift value; the summaries in this module
use Lanczos instead (see "Why not the power iteration" above).

Candidate floor. `extreme_ritz` also accepts unit directions the caller already holds
(the K-FAC step d0, the gradient g) and returns
  lambda_max = max(theta_1, max_v v^T A v / v^T v),
mirrored for lambda_min. Both terms are lower bounds on lambda_max(A), so their maximum
is the better bound - and the bound violation above becomes structurally impossible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ._estimators import HVPClosure, hutchinson_frob_sq

__all__ = [
    "DEFAULT_LANCZOS_ITER",
    "DEFAULT_RITZ_REFINE",
    "ExtremeRitz",
    "HVPClosure",
    "conditioning_from_extremes",
    "conditioning_from_spectrum",
    "extreme_eigenvalues",
    "extreme_ritz",
    "hutchinson_frob_sq",
    "lambda_key",
    "lambda_max_eigenpair",
    "lambda_max_power",
    "lambda_min_power",
    "lanczos_tridiagonal",
    "matrix_free_conditioning",
    "spectral_norm_power",
]

_ZERO_NORM_TOL = 1e-12
# Lanczos steps the matrix-free summaries take by default (matches the 3 x 20 HVP
# budget of three power iterations).
DEFAULT_LANCZOS_ITER = 60
# beta_j at or below this, relative to the largest coefficient seen, is a breakdown:
# the Krylov space is exhausted (an exact invariant subspace) and the run stops there.
_BREAKDOWN_REL_TOL = 1e-10
# Residual bound beta_m |y_i[m]| below rel_tol * max(|theta_i|, 1) counts as converged.
_RITZ_REL_TOL = 1e-2
# Power steps that polish the reconstructed top Ritz *vector* (`extreme_ritz`). Without
# reorthogonalisation the reconstruction V_m y_1 is contaminated at scale - on the exp11
# ResNet-50 probe (dim 2.4e7, m = 60) its own Rayleigh quotient came out 239 against a
# Ritz value of 278.84, a 14 % shortfall - and these steps, taken on the PSD shift from
# that vector, close the gap monotonically: 239 -> 278.545 (5) -> 278.838 (10).
DEFAULT_RITZ_REFINE = 10


def lambda_key(lam: float) -> str:
    """JSON key of a damping value: ``lambda_key(1e-2) == "0.01"``."""
    return f"{lam:g}"


# ----------------------------------------------------------------------
# Power iterations
# ----------------------------------------------------------------------


def _unit_start(
    dim: int, device: torch.device, generator: torch.Generator | None
) -> Tensor:
    v = torch.randn(dim, device=device, generator=generator)
    return v / v.norm()


def spectral_norm_power(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> float:
    """sigma = ||A||_2 of a symmetric A: ||A v|| after ``n_iter`` power steps on A.

    Returns 0.0 if A is numerically zero on the iterate.
    """
    v = _unit_start(dim, device, generator)
    nrm = torch.zeros((), device=device)
    for _ in range(n_iter):
        av = hvp(v).reshape(-1)
        nrm = av.norm()
        if nrm < _ZERO_NORM_TOL:
            return 0.0
        v = av / nrm
    return nrm.item()


def _rayleigh_power(
    op: Callable[[Tensor], Tensor],
    dim: int,
    n_iter: int,
    device: torch.device,
    generator: torch.Generator | None,
    start: Tensor | None = None,
) -> tuple[float, Tensor]:
    """Rayleigh quotient v^T B v of the last iterate of ``n_iter`` power steps on B, and
    the unit iterate that product feeds.

    For a PSD B the quotient converges (from below) to lambda_max(B) and the iterate to
    its eigenvector. Exactly ``n_iter`` applications of ``op``: the quotient is read off
    the product that also forms the returned iterate, so that iterate is one power step
    ahead of the quotient - its own Rayleigh quotient is at least as close to
    lambda_max(B). On a numerically zero B the quotient is 0.0.

    ``start`` (normalized here) replaces the random draw, for a caller that already has
    a good direction to polish: for a PSD B the Rayleigh quotients of the power sequence
    are non-decreasing (m_k m_{k+2} >= m_{k+1}^2 for the moments m_k = v^T B^k v), so
    such a polish can only improve the direction it was given.
    """
    v = _unit_start(dim, device, generator) if start is None else start / start.norm()
    rq = 0.0
    for _ in range(n_iter):
        bv = op(v).reshape(-1)
        rq = v.dot(bv).item()
        nrm = bv.norm()
        if nrm < _ZERO_NORM_TOL:
            return 0.0, v
        v = bv / nrm
    return rq, v


def lambda_max_power(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    shift: float,
    generator: torch.Generator | None = None,
) -> float:
    """lambda_max(A) = lambda_max(A + shift I) - shift by power iteration on the shift.

    ``shift >= ||A||_2`` makes A + shift I PSD, so its top eigenvalue is the one the
    power iteration finds; the subtraction is exact for any shift value.

    Convergence, however, is governed by the *shifted* gap: with a near-zero bulk and
    |lambda_min| = shift >> lambda_max the ratio of the wanted top to the bulk is
    1 + lambda_max / shift ~ 1 and the quotient returned is the bulk mean (see the
    module docstring). `extreme_ritz` is what the summaries use; this kernel is correct
    for the PSD / positive-dominant case.
    """
    return lambda_max_eigenpair(
        hvp, dim, n_iter, device, shift=shift, generator=generator
    )[0]


def lambda_max_eigenpair(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    shift: float,
    generator: torch.Generator | None = None,
) -> tuple[float, Tensor]:
    """(lambda_max(A), its unit eigenvector) by power iteration on A + shift I.

    The eigenvalue is `lambda_max_power`'s Rayleigh quotient; the vector is the last
    power iterate (one step ahead of that quotient, see `_rayleigh_power`), for a
    caller that needs the *direction* of the sharpest curvature - the line along which
    a quadratic model of the loss has the shortest range.
    """

    def shifted(z: Tensor) -> Tensor:
        return hvp(z).reshape(-1) + shift * z

    rq, v = _rayleigh_power(shifted, dim, n_iter, device, generator)
    return rq - shift, v


def lambda_min_power(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    shift: float,
    generator: torch.Generator | None = None,
) -> float:
    """lambda_min(A) = shift - lambda_max(shift I - A) by power iteration on the shift."""

    def reflected(z: Tensor) -> Tensor:
        return shift * z - hvp(z).reshape(-1)

    return shift - _rayleigh_power(reflected, dim, n_iter, device, generator)[0]


# ----------------------------------------------------------------------
# Lanczos
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExtremeRitz:
    """Both ends of the spectrum of a symmetric A from one Lanczos run.

    ``lambda_max`` / ``lambda_min`` are the extreme Ritz values ``ritz_max`` /
    ``ritz_min`` after the candidate floor (`extreme_ritz`), ``sigma`` =
    max(|lambda_max|, |lambda_min|) the spectral norm they imply; ``residual_max`` /
    ``residual_min`` are the residual bounds beta_m |y_i[m]| of the two Ritz pairs and
    ``converged`` says both fell below the relative tolerance. ``n_iter_used`` is the
    number of Lanczos steps actually taken (a breakdown or the dimension can cut the
    run short), ``n_hvp`` the total operator applications spent, candidates and the
    eigenvector reconstruction included. ``quotients`` holds the Rayleigh quotient of
    every candidate direction, in the order they were passed, so a caller that needs
    those curvatures anyway spends no second HVP on them. ``eigenvector`` (only with
    ``with_eigenvector``) is the unit direction attaining ``lambda_max``: the top Ritz
    vector, or the candidate that floored it, in either case polished by the power steps
    `extreme_ritz` takes on the PSD shift; ``eigenvector_quotient`` is that vector's own
    Rayleigh quotient, measured and used as a floor, so ``lambda_max >=
    eigenvector_quotient`` holds exactly and a caller needs no HVP of its own for it.
    """

    lambda_max: float
    lambda_min: float
    sigma: float
    ritz_max: float
    ritz_min: float
    residual_max: float
    residual_min: float
    converged: bool
    n_iter_used: int
    n_hvp: int
    quotients: tuple[float, ...] = ()
    eigenvector: Tensor | None = None
    eigenvector_quotient: float | None = None


def _lanczos_recurrence(
    hvp: HVPClosure,
    q1: Tensor,
    n_iter: int,
    *,
    weights: Sequence[float] | None = None,
) -> tuple[list[float], list[float], Tensor | None]:
    """The symmetric three-term recurrence on A from ``q1``; (alpha, beta, combination).

    Step j (1-based) forms w = A q_j - beta_{j-1} q_{j-1}, alpha_j = q_j^T w,
    w <- w - alpha_j q_j, locally reorthogonalises w against q_j and q_{j-1} (the basis
    is not kept, so nothing else is available), sets beta_j = ||w|| and
    q_{j+1} = w / beta_j. Only q_{j-1}, q_j and w are alive at any time, and the HVP's
    output tensor is consumed in place, so the recurrence costs three vectors on top of
    whatever the closure itself holds.

    Stops early on breakdown - beta_j <= `_BREAKDOWN_REL_TOL` times the largest
    coefficient seen, i.e. K_j(A, q1) is an invariant subspace (exactly what a zero
    operator does at j = 1) - so ``len(alpha)`` is the number of steps actually taken.
    ``beta[-1]`` is beta_m, the residual coefficient of the Ritz bounds, not part of the
    tridiagonal; ``beta[:-1]`` is its subdiagonal.

    With ``weights`` (one per step, from a first pass) the same recurrence is replayed
    and sum_j weights_j q_j accumulated on the fly - the way a Ritz vector is recovered
    without ever storing the basis. The replay needs one HVP fewer than there are
    weights: the last vector is accumulated before its product would be formed.
    """
    alpha: list[float] = []
    beta: list[float] = []
    coefficients = list(weights) if weights is not None else []
    combination = torch.zeros_like(q1) if weights is not None else None
    scale = 0.0
    q_prev: Tensor | None = None
    q = q1
    beta_prev = 0.0

    for j in range(n_iter):
        if combination is not None:
            if j >= len(coefficients):
                break
            combination.add_(q, alpha=coefficients[j])
            if j + 1 >= len(coefficients):
                break  # the last wanted vector: no further product needed
        w = hvp(q).reshape(-1)
        if w.requires_grad:  # never for the estimators here; cheap insurance
            w = w.detach().clone()
        if q_prev is not None:
            w.sub_(q_prev, alpha=beta_prev)
        a = q.dot(w).item()
        w.sub_(q, alpha=a)
        # Local reorthogonalisation: the two vectors we still have.
        w.sub_(q, alpha=q.dot(w).item())
        if q_prev is not None:
            w.sub_(q_prev, alpha=q_prev.dot(w).item())
        b = w.norm().item()
        alpha.append(a)
        beta.append(b)
        scale = max(scale, abs(a), b)
        if b <= max(_ZERO_NORM_TOL, _BREAKDOWN_REL_TOL * scale):
            break  # breakdown: an invariant subspace, T_j is exact
        q_prev, q, beta_prev = q, w / b, b

    return alpha, beta, combination


def lanczos_tridiagonal(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    start: Tensor | None = None,
) -> tuple[Tensor, Tensor, int]:
    """(alpha, beta, n_iter_used) of ``n_iter`` Lanczos steps on a symmetric A.

    ``start`` (normalized here) or a random unit vector from ``generator`` seeds the
    Krylov space; ``n_iter`` is capped at ``dim``, beyond which the space is exhausted.
    ``alpha`` (n_iter_used,) is the diagonal of T_m and ``beta`` (n_iter_used,) holds
    the subdiagonal in ``beta[:-1]`` plus the residual coefficient beta_m in
    ``beta[-1]``, both float64 on the CPU - the tridiagonal is tiny and its eigenvalues
    are wanted in double. ``n_iter_used`` < ``n_iter`` means a breakdown stopped the
    run (`_lanczos_recurrence`). One HVP per step.
    """
    q1 = _unit_start(dim, device, generator) if start is None else start / start.norm()
    alpha, beta, _ = _lanczos_recurrence(hvp, q1, max(1, min(n_iter, dim)))
    return (
        torch.tensor(alpha, dtype=torch.float64),
        torch.tensor(beta, dtype=torch.float64),
        len(alpha),
    )


def _tridiagonal_eigh(alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """(ascending Ritz values, their eigenvectors) of T_m from `lanczos_tridiagonal`."""
    m = alpha.numel()
    t = torch.diag(alpha)
    if m > 1:
        idx = torch.arange(m - 1)
        sub = beta[: m - 1]
        t[idx, idx + 1] = sub
        t[idx + 1, idx] = sub
    values, vectors = torch.linalg.eigh(t)
    return values, vectors


def _rayleigh(hvp: HVPClosure, v: Tensor) -> float:
    """v^T A v / v^T v, 0.0 for a numerically zero v. One HVP."""
    flat = v.reshape(-1)
    norm_sq = flat.dot(flat).item()
    if norm_sq <= _ZERO_NORM_TOL:
        return 0.0
    return flat.dot(hvp(flat).reshape(-1)).item() / norm_sq


def extreme_ritz(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    candidates: Sequence[Tensor] | None = None,
    with_eigenvector: bool = False,
    n_refine: int = DEFAULT_RITZ_REFINE,
    rel_tol: float = _RITZ_REL_TOL,
    start: Tensor | None = None,
) -> ExtremeRitz:
    """Both extreme eigenvalues of a symmetric A by ``n_iter`` Lanczos steps.

    The extreme eigenvalues of the tridiagonal (float64) are the estimates, their
    residual bounds beta_m |y_i[m]| the accuracy, and - in exact arithmetic - they are
    one-sided: theta_1 <= lambda_max(A), theta_m >= lambda_min(A) (the Ritz values are
    Rayleigh quotients over the Krylov space).

    ``candidates`` are directions the caller already holds - the step it is about to
    take, the gradient. Their Rayleigh quotients are lower bounds on lambda_max(A) and
    upper bounds on lambda_min(A) just as the Ritz values are, so the result is floored
    by them: ``lambda_max = max(theta_1, max_v v^T A v / ||v||^2)``, mirrored for
    ``lambda_min``. One HVP each, counted in ``n_hvp``, and each quotient is returned
    (``quotients``, in order) for a caller that wants the curvature along those
    directions too. Whatever Lanczos does, the reported lambda_max can then never fall
    below a curvature the caller measured.

    ``with_eigenvector`` returns the unit direction attaining ``lambda_max``. When the
    top Ritz value is the largest, the vector is recovered by *replaying* the recurrence
    from the same start vector and accumulating V_m y_1 on the fly (the basis is never
    stored): ``n_iter_used - 1`` extra HVPs, i.e. close to twice the cost of the
    eigenvalue alone. When a candidate floored lambda_max instead, that candidate is the
    seed and no replay is needed - the top Ritz vector would be meaningless there.

    That reconstruction is the one place where dropping reorthogonalisation costs
    accuracy in a way the Ritz *values* do not feel: V_m is not orthonormal any more, so
    V_m y_1 is not quite the Ritz vector, and at scale the shortfall is large (see
    `DEFAULT_RITZ_REFINE`). ``n_refine`` power steps on the PSD shift A + sigma I,
    seeded with that vector, repair it - the Rayleigh quotient of a PSD power sequence
    is non-decreasing, so they can only improve the seed, whichever branch produced it -
    and the quotient they reach floors ``lambda_max`` in turn. The returned vector's own
    quotient is then measured (one HVP, ``eigenvector_quotient``) and floors
    ``lambda_max`` as well, so ``lambda_max >= eigenvector_quotient`` holds exactly - the
    caller reads that curvature off the result instead of spending an HVP on it.
    ``n_refine`` + 1 extra HVPs; ``n_refine = 0`` gives the bare reconstruction.

    ``start`` fixes the Krylov start vector (normalized here) instead of drawing it from
    ``generator``: a reproducible probe, and the seam the tests use.
    """
    q1 = _unit_start(dim, device, generator) if start is None else start / start.norm()
    alpha, beta, used = lanczos_tridiagonal(hvp, dim, n_iter, device, start=q1)
    values, vectors = _tridiagonal_eigh(alpha, beta)
    n_hvp = used

    ritz_max, ritz_min = values[-1].item(), values[0].item()
    beta_m = beta[-1].item()
    residual_max = beta_m * abs(vectors[-1, -1].item())
    residual_min = beta_m * abs(vectors[-1, 0].item())
    converged = residual_max <= rel_tol * max(
        abs(ritz_max), 1.0
    ) and residual_min <= rel_tol * max(abs(ritz_min), 1.0)

    lam_max, lam_min = ritz_max, ritz_min
    best: Tensor | None = None
    quotients: list[float] = []
    for v in candidates or ():
        quotient = _rayleigh(hvp, v)
        n_hvp += 1
        quotients.append(quotient)
        if quotient > lam_max:
            lam_max, best = quotient, v.reshape(-1)
        lam_min = min(lam_min, quotient)

    eigenvector: Tensor | None = None
    eig_quotient: float | None = None
    if with_eigenvector:
        if best is not None:
            seed = best
        else:
            _, _, combination = _lanczos_recurrence(
                hvp, q1, used, weights=vectors[:, -1].tolist()
            )
            n_hvp += max(0, used - 1)
            assert combination is not None
            seed = combination if combination.norm().item() > _ZERO_NORM_TOL else q1
        if n_refine > 0:
            shift = max(abs(lam_max), abs(lam_min))

            def shifted(z: Tensor) -> Tensor:
                return hvp(z).reshape(-1) + shift * z

            polished, eigenvector = _rayleigh_power(
                shifted, dim, n_refine, device, None, start=seed
            )
            n_hvp += n_refine
            lam_max = max(lam_max, polished - shift)
        else:
            eigenvector = seed / seed.norm()
        eig_quotient = _rayleigh(hvp, eigenvector)
        n_hvp += 1
        lam_max = max(lam_max, eig_quotient)

    return ExtremeRitz(
        lambda_max=lam_max,
        lambda_min=lam_min,
        sigma=max(abs(lam_max), abs(lam_min)),
        ritz_max=ritz_max,
        ritz_min=ritz_min,
        residual_max=residual_max,
        residual_min=residual_min,
        converged=bool(converged),
        n_iter_used=used,
        n_hvp=n_hvp,
        quotients=tuple(quotients),
        eigenvector=eigenvector,
        eigenvector_quotient=eig_quotient,
    )


def extreme_eigenvalues(
    hvp: HVPClosure,
    dim: int,
    n_iter: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> tuple[float, float, float]:
    """(sigma, lambda_max, lambda_min) of a symmetric A by one Lanczos run.

    `extreme_ritz` uses at most ``n_iter`` Lanczos HVPs plus HVPs for candidate
    Rayleigh quotients, and estimates both algebraic extremes.
    """
    ritz = extreme_ritz(hvp, dim, n_iter, device, generator=generator)
    return ritz.sigma, ritz.lambda_max, ritz.lambda_min


# ----------------------------------------------------------------------
# Summaries (pure)
# ----------------------------------------------------------------------


def conditioning_from_extremes(
    lambda_max: float,
    lambda_min: float,
    sigma: float,
    frob_sq: float,
    lambdas: Sequence[float],
) -> dict[str, Any]:
    """kappa_lambda for every damping in ``lambdas`` and the stable rank D.

    Returns ``{lambda_max, lambda_min, sigma, frob_sq, kappa: {lambda_key: kappa},
    stable_rank}``; kappa is +inf where ``lambda_min + lambda <= 0`` and the stable
    rank NaN for a numerically zero operator.
    """
    kappa: dict[str, float] = {}
    for lam in lambdas:
        denom = lambda_min + lam
        kappa[lambda_key(lam)] = (lambda_max + lam) / denom if denom > 0 else math.inf
    stable_rank = frob_sq / sigma**2 if sigma > _ZERO_NORM_TOL else math.nan
    return {
        "lambda_max": float(lambda_max),
        "lambda_min": float(lambda_min),
        "sigma": float(sigma),
        "frob_sq": float(frob_sq),
        "kappa": kappa,
        "stable_rank": float(stable_rank),
    }


def conditioning_from_spectrum(
    eigenvalues: Tensor,
    lambdas: Sequence[float],
    *,
    neg_tol: float = 1e-8,
) -> dict[str, Any]:
    """Exact summary from a full spectrum (``torch.linalg.eigvalsh`` output).

    Adds ``neg_frac``, the fraction of eigenvalues below ``-neg_tol * max|lambda|``
    (0.0 for a zero spectrum), to `conditioning_from_extremes`.
    """
    e = eigenvalues.detach().reshape(-1).to(torch.float64)
    if e.numel() == 0:
        raise ValueError("empty spectrum")
    sigma = e.abs().max().item()
    record = conditioning_from_extremes(
        e.max().item(), e.min().item(), sigma, (e**2).sum().item(), lambdas
    )
    record["neg_frac"] = (
        (e < -neg_tol * sigma).double().mean().item() if sigma > 0 else 0.0
    )
    return record


def matrix_free_conditioning(
    hvp: HVPClosure,
    dim: int,
    lambdas: Sequence[float],
    *,
    n_iter: int = DEFAULT_LANCZOS_ITER,
    n_probes: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    candidates: Sequence[Tensor] | None = None,
) -> dict[str, Any]:
    """HVP-only summary: the extremes by Lanczos, ||A||_F^2 by Hutchinson.

    ``n_iter`` + ``n_probes`` HVPs (one more per candidate direction). No ``neg_frac``
    (the spectrum is never formed). On top of `conditioning_from_extremes` the record
    carries the estimator's own diagnostics - ``n_iter_used``, the two residual bounds
    ``residual_max`` / ``residual_min`` and ``converged`` - so a reader of the appendix
    can tell a resolved end from an unresolved one.
    """
    ritz = extreme_ritz(
        hvp, dim, n_iter, device, generator=generator, candidates=candidates
    )
    frob_sq = hutchinson_frob_sq(hvp, dim, n_probes, device, generator=generator)
    record = conditioning_from_extremes(
        ritz.lambda_max, ritz.lambda_min, ritz.sigma, frob_sq, lambdas
    )
    record["n_iter_used"] = ritz.n_iter_used
    record["residual_max"] = ritz.residual_max
    record["residual_min"] = ritz.residual_min
    record["converged"] = ritz.converged
    return record
