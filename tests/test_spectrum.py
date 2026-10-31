"""`hessian.spectrum`: Lanczos, the power iterations and Hutchinson against dense spectra.

Dense symmetric matrices with a prescribed spectrum (Q diag(e) Q^T, Q a random
orthogonal) - indefinite, PSD, negative-definite and one with lambda_max = -lambda_min -
pin sigma, lambda_max and lambda_min of the power-iteration kernels to
`torch.linalg.eigvalsh` within 1e-3 relative after enough iterations. A diagonal matrix
makes Hutchinson with Rademacher probes exact; a dense one checks it statistically. The
pure summaries are checked on hand-picked spectra (the kappa = inf rule, the stable rank
from the eigenvalues, the negative fraction and its threshold), and the matrix-free
summary against the exact one on the same dense operator.

The Lanczos estimator the matrix-free summaries use gets four 200-dimensional reference
spectra: PSD, negative-dominant (|lambda_min| = 50 lambda_max), a nearly degenerate top,
and a dense bulk near zero with a moderate positive top and a large negative bottom.
The latter compares shifted power iteration with Lanczos and a measured directional
curvature. The tests also cover the candidate floor (a
supplied direction's Rayleigh quotient is never violated, however tight the iteration
cap), the residual / `converged` flag at a tight cap, the breakdown bookkeeping and the
second-pass reconstruction of the top Ritz vector.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from hessian.spectrum import (  # noqa: E402
    DEFAULT_LANCZOS_ITER,
    DEFAULT_RITZ_REFINE,
    conditioning_from_extremes,
    conditioning_from_spectrum,
    extreme_eigenvalues,
    extreme_ritz,
    hutchinson_frob_sq,
    lambda_key,
    lambda_max_eigenpair,
    lambda_max_power,
    lambda_min_power,
    lanczos_tridiagonal,
    matrix_free_conditioning,
    spectral_norm_power,
)

_DEVICE = torch.device("cpu")
_LAMBDAS = [1e-2, 1e-1]
_N_ITER = 500  # "enough" for the linspace spectra below (gap ratio <= 0.99)


def _dense_symmetric(eigs: Tensor, seed: int) -> Tensor:
    """Q diag(eigs) Q^T for a random orthogonal Q (float32, like a model's HVP)."""
    gen = torch.Generator().manual_seed(seed)
    n = eigs.numel()
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    return (q @ torch.diag(eigs.to(torch.float64)) @ q.T).to(torch.float32)


def _hvp_of(a: Tensor) -> Callable[[Tensor], Tensor]:
    return lambda z: a @ z


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device=_DEVICE).manual_seed(seed)


def _assert_rel(est: float, true: float, tol: float, what: str) -> None:
    assert abs(est - true) <= tol * max(abs(true), 1e-12), f"{what}: {est} != {true}"


_SPECTRA = {
    "indefinite": torch.linspace(-1.0, 2.0, 40),
    "psd": torch.linspace(0.1, 3.0, 40),
    "negative_definite": torch.linspace(-3.0, -0.1, 40),
    "symmetric": torch.linspace(-2.0, 2.0, 40),  # lambda_max = -lambda_min
}


def test_power_iterations_match_eigvalsh() -> None:
    for name, eigs in _SPECTRA.items():
        a = _dense_symmetric(eigs, seed=1)
        true = torch.linalg.eigvalsh(a.double())
        hvp = _hvp_of(a)
        n = a.shape[0]

        sigma = spectral_norm_power(hvp, n, _N_ITER, _DEVICE, generator=_gen())
        _assert_rel(sigma, true.abs().max().item(), 1e-3, f"{name}: sigma")

        lam_max = lambda_max_power(
            hvp, n, _N_ITER, _DEVICE, shift=sigma, generator=_gen()
        )
        _assert_rel(lam_max, true.max().item(), 1e-3, f"{name}: lambda_max")

        lam_min = lambda_min_power(
            hvp, n, _N_ITER, _DEVICE, shift=sigma, generator=_gen()
        )
        _assert_rel(lam_min, true.min().item(), 1e-3, f"{name}: lambda_min")

        s, mx, mn = extreme_eigenvalues(hvp, n, _N_ITER, _DEVICE, generator=_gen())
        _assert_rel(s, true.abs().max().item(), 1e-3, f"{name}: sigma (combined)")
        _assert_rel(mx, true.max().item(), 1e-3, f"{name}: lambda_max (combined)")
        _assert_rel(mn, true.min().item(), 1e-3, f"{name}: lambda_min (combined)")


def test_shift_identities_hold_for_any_shift() -> None:
    """lambda_max / lambda_min are exact whatever shift is passed (it only sets the
    convergence rate), so an under-converged sigma cannot bias them."""
    a = _dense_symmetric(_SPECTRA["indefinite"], seed=2)
    true = torch.linalg.eigvalsh(a.double())
    n = a.shape[0]
    for shift in (2.0, 5.0, 20.0):  # sigma = 2 exactly, then over-estimates
        mx = lambda_max_power(
            _hvp_of(a), n, _N_ITER * 4, _DEVICE, shift=shift, generator=_gen()
        )
        mn = lambda_min_power(
            _hvp_of(a), n, _N_ITER * 4, _DEVICE, shift=shift, generator=_gen()
        )
        _assert_rel(mx, true.max().item(), 1e-3, f"shift {shift}: lambda_max")
        _assert_rel(mn, true.min().item(), 1e-3, f"shift {shift}: lambda_min")


def test_power_iteration_on_zero_operator() -> None:
    zero = torch.zeros(6, 6)
    assert spectral_norm_power(_hvp_of(zero), 6, 5, _DEVICE, generator=_gen()) == 0.0
    sigma, mx, mn = extreme_eigenvalues(_hvp_of(zero), 6, 5, _DEVICE, generator=_gen())
    assert sigma == 0.0 and mx == 0.0 and mn == 0.0
    lam, v = lambda_max_eigenpair(
        _hvp_of(zero), 6, 5, _DEVICE, shift=0.0, generator=_gen()
    )
    assert lam == 0.0 and math.isclose(v.norm().item(), 1.0, rel_tol=1e-6)


def test_lambda_max_eigenpair_returns_the_top_eigenvector() -> None:
    """The eigenvalue is `lambda_max_power`'s (same draws, same arithmetic); the vector
    is a unit vector whose Rayleigh quotient is lambda_max and which H maps to itself.
    """
    for name, eigs in _SPECTRA.items():
        a = _dense_symmetric(eigs, seed=4)
        true = torch.linalg.eigvalsh(a.double())
        n = a.shape[0]
        sigma = spectral_norm_power(_hvp_of(a), n, _N_ITER, _DEVICE, generator=_gen())
        lam, v = lambda_max_eigenpair(
            _hvp_of(a), n, _N_ITER, _DEVICE, shift=sigma, generator=_gen()
        )
        assert lam == lambda_max_power(
            _hvp_of(a), n, _N_ITER, _DEVICE, shift=sigma, generator=_gen()
        ), f"{name}: the eigenvalue is the one lambda_max_power reports"
        assert math.isclose(v.norm().item(), 1.0, rel_tol=1e-5), f"{name}: unit"
        _assert_rel(v.dot(a @ v).item(), true.max().item(), 1e-3, f"{name}: v.Av")
        residual = (a @ v - true.max().item() * v).norm().item()
        assert residual <= 1e-2 * true.abs().max().item(), f"{name}: A v = lambda v"


def test_hutchinson_exact_on_diagonal_and_close_on_dense() -> None:
    # Diagonal A: ||A z||^2 = sum_i a_i^2 z_i^2 = sum_i a_i^2 for every Rademacher z.
    diag = torch.tensor([3.0, -1.0, 0.5, 0.0, 2.0])
    a = torch.diag(diag)
    est = hutchinson_frob_sq(_hvp_of(a), 5, 3, _DEVICE, generator=_gen())
    assert math.isclose(est, (diag**2).sum().item(), rel_tol=1e-5), est

    dense = _dense_symmetric(_SPECTRA["indefinite"], seed=3)
    true = (dense**2).sum().item()
    est = hutchinson_frob_sq(_hvp_of(dense), 40, 4000, _DEVICE, generator=_gen())
    assert abs(est - true) <= 0.1 * true, f"{est} vs {true}"


def test_lambda_key() -> None:
    assert lambda_key(1e-2) == "0.01" and lambda_key(1e-1) == "0.1"
    assert lambda_key(1.0) == "1"


def test_conditioning_from_extremes_kappa_rule_and_stable_rank() -> None:
    rec = conditioning_from_extremes(2.0, -0.05, 2.0, 8.0, _LAMBDAS)
    assert math.isinf(rec["kappa"]["0.01"]), "lambda_min + 0.01 < 0 -> inf"
    assert math.isclose(rec["kappa"]["0.1"], 2.1 / 0.05), "damped ratio"
    assert math.isclose(rec["stable_rank"], 8.0 / 4.0), "||A||_F^2 / ||A||_2^2"
    assert set(rec) == {
        "lambda_max",
        "lambda_min",
        "sigma",
        "frob_sq",
        "kappa",
        "stable_rank",
    }

    boundary = conditioning_from_extremes(1.0, -0.01, 1.0, 1.0, _LAMBDAS)
    assert math.isinf(boundary["kappa"]["0.01"]), "lambda_min + lambda == 0 -> inf"
    assert math.isclose(boundary["kappa"]["0.1"], 1.1 / 0.09)

    psd = conditioning_from_extremes(3.0, 0.5, 3.0, 12.0, _LAMBDAS)
    assert math.isclose(psd["kappa"]["0.01"], 3.01 / 0.51)
    assert math.isclose(psd["kappa"]["0.1"], 3.1 / 0.6)

    zero = conditioning_from_extremes(0.0, 0.0, 0.0, 0.0, _LAMBDAS)
    assert math.isnan(zero["stable_rank"]), "D undefined for a zero operator"
    assert math.isclose(zero["kappa"]["0.1"], 1.0)


def test_conditioning_from_spectrum() -> None:
    eigs = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    rec = conditioning_from_spectrum(eigs, _LAMBDAS)
    assert rec["lambda_max"] == 2.0 and rec["lambda_min"] == -1.0
    assert rec["sigma"] == 2.0 and math.isclose(rec["frob_sq"], 5.25)
    assert math.isclose(rec["stable_rank"], 5.25 / 4.0)
    assert math.isclose(rec["neg_frac"], 0.25)
    assert math.isinf(rec["kappa"]["0.01"]) and math.isinf(rec["kappa"]["0.1"])

    # the negative-eigenvalue threshold is relative to max|lambda|
    assert conditioning_from_spectrum(torch.tensor([-1e-12, 1.0]), _LAMBDAS)[
        "neg_frac"
    ] == (0.0)
    assert conditioning_from_spectrum(torch.tensor([-1e-6, 1.0]), _LAMBDAS)[
        "neg_frac"
    ] == (0.5)
    assert conditioning_from_spectrum(torch.zeros(3), _LAMBDAS)["neg_frac"] == 0.0

    raised = False
    try:
        conditioning_from_spectrum(torch.zeros(0), _LAMBDAS)
    except ValueError:
        raised = True
    assert raised, "an empty spectrum is rejected"


def test_matrix_free_matches_exact_on_dense_operator() -> None:
    for name in ("psd", "indefinite"):
        a = _dense_symmetric(_SPECTRA[name], seed=4)
        exact = conditioning_from_spectrum(torch.linalg.eigvalsh(a.double()), _LAMBDAS)
        est = matrix_free_conditioning(
            _hvp_of(a),
            a.shape[0],
            _LAMBDAS,
            n_iter=_N_ITER,
            n_probes=4000,
            device=_DEVICE,
            generator=_gen(),
        )
        assert "neg_frac" not in est, "the spectrum is never formed"
        for key in ("lambda_max", "lambda_min", "sigma"):
            _assert_rel(est[key], exact[key], 1e-3, f"{name}: {key}")
        for lam in _LAMBDAS:
            k = lambda_key(lam)
            if math.isinf(exact["kappa"][k]):
                assert math.isinf(est["kappa"][k]), f"{name}: kappa_{k} must be inf"
            else:
                _assert_rel(est["kappa"][k], exact["kappa"][k], 1e-2, f"kappa_{k}")
        _assert_rel(est["stable_rank"], exact["stable_rank"], 0.1, f"{name}: D")


# ======================================================================
# Lanczos (the matrix-free route)
# ======================================================================

_BIG = 200  # dimension of the Lanczos reference operators
_LANCZOS_SPECTRA = {
    # (a) PSD, a spread bulk
    "psd_bulk": torch.linspace(0.01, 5.0, _BIG),
    # (b) indefinite, |lambda_min| = 50 lambda_max
    "negative_dominant": torch.linspace(-50.0, 1.0, _BIG),
    # (c) dense bulk near zero, moderate positive top and large negative bottom;
    # shifted power iteration converges slowly for this spectrum
    "near_zero_bulk": torch.cat(
        [
            torch.linspace(-1e-4, 1e-4, _BIG - 2),
            torch.tensor([1.0]),
            torch.tensor([-500.0]),
        ]
    ),
    # (d) a nearly degenerate top
    "degenerate_top": torch.cat(
        [torch.linspace(-1.0, 2.0, _BIG - 2), torch.tensor([3.0, 3.0 - 1e-6])]
    ),
}
_LANCZOS_SEED = 5
_RITZ_TOL = 2e-2  # "within a few percent" of eigvalsh at 60 steps


def _rayleigh_of(a: Tensor, v: Tensor) -> float:
    return v.dot(a @ v).item() / v.dot(v).item()


def _eigenvectors(a: Tensor) -> Tensor:
    return torch.linalg.eigh(a.double()).eigenvectors.to(torch.float32)


def test_extreme_ritz_matches_eigvalsh_on_reference_spectra() -> None:
    """Both ends within a few percent at `DEFAULT_LANCZOS_ITER` steps, one HVP each,
    and the one-sided guarantee the conditioning appendix claims (lambda_max from
    below, lambda_min from above, up to the residual bound)."""
    for name, eigs in _LANCZOS_SPECTRA.items():
        a = _dense_symmetric(eigs, seed=_LANCZOS_SEED)
        true = torch.linalg.eigvalsh(a.double())
        ritz = extreme_ritz(
            _hvp_of(a), _BIG, DEFAULT_LANCZOS_ITER, _DEVICE, generator=_gen()
        )
        _assert_rel(ritz.lambda_max, true.max().item(), _RITZ_TOL, f"{name}: max")
        _assert_rel(ritz.lambda_min, true.min().item(), _RITZ_TOL, f"{name}: min")
        _assert_rel(ritz.sigma, true.abs().max().item(), _RITZ_TOL, f"{name}: sigma")
        assert ritz.converged, f"{name}: 60 steps resolve both ends"
        assert ritz.n_iter_used == DEFAULT_LANCZOS_ITER, name
        assert ritz.n_hvp == DEFAULT_LANCZOS_ITER, f"{name}: one HVP per step"
        assert ritz.eigenvector is None, f"{name}: no vector unless asked"

        slack = max(ritz.residual_max, ritz.residual_min, 1e-5 * ritz.sigma)
        assert ritz.ritz_max <= true.max().item() + slack, f"{name}: max from below"
        assert ritz.ritz_min >= true.min().item() - slack, f"{name}: min from above"


def test_lanczos_fixes_the_near_zero_bulk_defect() -> None:
    """Compare Lanczos and shifted power iteration for a near-zero bulk spectrum.

    With 20 power iterations, the shifted top and bulk are close, and the estimated
    maximum falls below a measured directional curvature. The test also compares the
    Lanczos estimate against the dense spectrum at the same HVP budget.
    """
    a = _dense_symmetric(_LANCZOS_SPECTRA["near_zero_bulk"], seed=_LANCZOS_SEED)
    true = torch.linalg.eigvalsh(a.double())
    true_max, true_min = true.max().item(), true.min().item()
    hvp = _hvp_of(a)

    lanczos = extreme_ritz(hvp, _BIG, DEFAULT_LANCZOS_ITER, _DEVICE, generator=_gen())
    _assert_rel(lanczos.lambda_max, true_max, _RITZ_TOL, "Lanczos lambda_max")
    _assert_rel(lanczos.lambda_min, true_min, _RITZ_TOL, "Lanczos lambda_min")
    lanczos_err = abs(lanczos.lambda_max - true_max) / abs(true_max)

    # Shifted power iteration with 3 x 20 HVPs.
    sigma = spectral_norm_power(hvp, _BIG, 20, _DEVICE, generator=_gen())
    power_max = lambda_max_power(hvp, _BIG, 20, _DEVICE, shift=sigma, generator=_gen())
    power_min = lambda_min_power(hvp, _BIG, 20, _DEVICE, shift=sigma, generator=_gen())
    _assert_rel(sigma, abs(true_min), 1e-3, "sigma: the dominant end is found")
    _assert_rel(power_min, true_min, 1e-3, "lambda_min: the dominant end is found")
    assert power_max <= 1e-2 * true_max, (
        "the power route must be orders of magnitude below lambda_max here "
        f"(got {power_max} for {true_max})"
    )
    power_err = abs(power_max - true_max) / abs(true_max)
    assert power_err >= 50 * max(lanczos_err, 1e-6), (
        "and far worse than Lanczos at the same budget: relative errors "
        f"{power_err} (power) against {lanczos_err} (Lanczos)"
    )

    # A direction with a little of the top
    # eigenvector in it has more curvature than the power route's lambda_max
    vectors = _eigenvectors(a)
    mixed = 0.3 * vectors[:, -1] + math.sqrt(1.0 - 0.09) * vectors[:, _BIG // 2]
    measured = _rayleigh_of(a, mixed)
    assert measured > power_max, "this is the bound violation that exposed the defect"
    assert lanczos.lambda_max >= measured, "Lanczos respects the bound"


def test_candidate_floor_never_below_a_supplied_direction() -> None:
    """With a deliberately tiny iteration cap the Ritz values are useless, but a
    supplied direction's Rayleigh quotient floors lambda_max (and ceils lambda_min), so
    the bound violation is structurally impossible."""
    a = _dense_symmetric(_LANCZOS_SPECTRA["near_zero_bulk"], seed=_LANCZOS_SEED)
    hvp = _hvp_of(a)
    vectors = _eigenvectors(a)
    mixed = 0.3 * vectors[:, -1] + math.sqrt(1.0 - 0.09) * vectors[:, _BIG // 2]
    bottom = vectors[:, 0]
    up, down = _rayleigh_of(a, mixed), _rayleigh_of(a, bottom)

    for n_iter in (1, 2, 3, 8):
        ritz = extreme_ritz(
            hvp,
            _BIG,
            n_iter,
            _DEVICE,
            generator=_gen(),
            candidates=[mixed, bottom],
            with_eigenvector=True,
        )
        assert ritz.lambda_max >= up - 1e-6, (n_iter, ritz.lambda_max, up)
        assert ritz.lambda_min <= down + 1e-6, (n_iter, ritz.lambda_min, down)
        assert len(ritz.quotients) == 2, "every candidate's quotient is returned"
        assert math.isclose(ritz.quotients[0], up, rel_tol=1e-6)
        assert math.isclose(ritz.quotients[1], down, rel_tol=1e-6)
        assert ritz.lambda_max >= ritz.ritz_max and ritz.lambda_min <= ritz.ritz_min
        assert ritz.sigma == max(abs(ritz.lambda_max), abs(ritz.lambda_min))
        # every candidate costs one HVP, honestly counted
        assert ritz.n_hvp >= ritz.n_iter_used + 2
        # the returned direction attains the reported lambda_max
        vec = ritz.eigenvector
        assert vec is not None and math.isclose(vec.norm().item(), 1.0, rel_tol=1e-5)
        slack = max(ritz.residual_max, 1e-4 * abs(ritz.lambda_max))
        assert _rayleigh_of(a, vec) >= ritz.lambda_max - slack, n_iter
        assert ritz.eigenvector_quotient is not None
        assert ritz.lambda_max >= ritz.eigenvector_quotient, n_iter


def test_extreme_ritz_reports_convergence_only_when_the_residual_is_small() -> None:
    """converged is False when the cap is hit on a hard spectrum, True at 60 steps."""
    for name in ("psd_bulk", "negative_dominant", "degenerate_top"):
        a = _dense_symmetric(_LANCZOS_SPECTRA[name], seed=_LANCZOS_SEED)
        capped = extreme_ritz(_hvp_of(a), _BIG, 3, _DEVICE, generator=_gen())
        assert capped.n_iter_used == 3, name
        assert not capped.converged, f"{name}: 3 steps cannot resolve this"
        assert capped.residual_max > 0.0 and capped.residual_min > 0.0, name
        assert extreme_ritz(
            _hvp_of(a), _BIG, DEFAULT_LANCZOS_ITER, _DEVICE, generator=_gen()
        ).converged, name


def test_top_ritz_vector_is_reconstructed_by_a_second_pass() -> None:
    """The basis is never stored, so the top Ritz vector is recovered by replaying the
    recurrence from the same start vector: a unit vector whose own Rayleigh quotient is
    theta_1 up to the residual bound, at the price of n_iter_used - 1 extra HVPs."""
    for name, eigs in _LANCZOS_SPECTRA.items():
        a = _dense_symmetric(eigs, seed=_LANCZOS_SEED)
        ritz = extreme_ritz(
            _hvp_of(a),
            _BIG,
            DEFAULT_LANCZOS_ITER,
            _DEVICE,
            generator=_gen(),
            with_eigenvector=True,
        )
        vec = ritz.eigenvector
        assert vec is not None
        assert math.isclose(vec.norm().item(), 1.0, rel_tol=1e-5), f"{name}: unit"
        quotient = _rayleigh_of(a, vec)
        slack = max(ritz.residual_max, 1e-4 * ritz.sigma)
        assert quotient >= ritz.ritz_max - slack, (name, quotient, ritz.ritz_max)
        _assert_rel(quotient, ritz.ritz_max, _RITZ_TOL, f"{name}: v^T A v = theta_1")
        assert ritz.n_hvp == 2 * DEFAULT_LANCZOS_ITER + DEFAULT_RITZ_REFINE, (
            f"{name}: the second pass, the polish, the vector's own quotient"
        )
        assert ritz.eigenvector_quotient is not None
        assert math.isclose(ritz.eigenvector_quotient, quotient, rel_tol=1e-5), name
        assert ritz.lambda_max >= ritz.eigenvector_quotient, f"{name}: exact floor"

        # the bare reconstruction, and the polish that can only improve it
        bare = extreme_ritz(
            _hvp_of(a),
            _BIG,
            DEFAULT_LANCZOS_ITER,
            _DEVICE,
            generator=_gen(),
            with_eigenvector=True,
            n_refine=0,
        )
        assert bare.n_hvp == 2 * DEFAULT_LANCZOS_ITER, f"{name}: no polish"
        assert bare.eigenvector is not None
        assert quotient >= _rayleigh_of(a, bare.eigenvector) - slack, name
        assert ritz.lambda_max >= bare.lambda_max, f"{name}: the polish also floors"


def test_lanczos_tridiagonal_coefficients_start_and_breakdown() -> None:
    """The recurrence's bookkeeping: float64 coefficients (``beta[-1]`` = beta_m, the
    residual coefficient of the Ritz bounds, ``beta[:-1]`` the subdiagonal), the cap at
    ``dim``, a caller-supplied start vector, and the early stop on breakdown.

    Breakdown is tested in float64, where "beta_j is numerically zero" is unambiguous.
    In the float32 of a model's HVP an exhausted Krylov space leaves beta_j at roundoff
    level - ~1e-7 relative, the same size as a legitimately small beta on a spectrum
    with a near-zero bulk - so the recurrence deliberately does *not* stop there: it
    continues on a ghost vector, which costs a spurious copy of a converged eigenvalue
    and leaves both ends where they were.
    """
    a = _dense_symmetric(_LANCZOS_SPECTRA["psd_bulk"], seed=_LANCZOS_SEED)
    alpha, beta, used = lanczos_tridiagonal(
        _hvp_of(a), _BIG, 12, _DEVICE, generator=_gen()
    )
    assert used == 12 and alpha.shape == beta.shape == (12,)
    assert alpha.dtype == beta.dtype == torch.float64
    assert (beta > 0).all(), "no breakdown on a full-rank bulk"

    # the same start vector gives the same coefficients
    start = _fixed_unit(_BIG)
    first = lanczos_tridiagonal(_hvp_of(a), _BIG, 6, _DEVICE, start=start)
    second = lanczos_tridiagonal(_hvp_of(a), _BIG, 6, _DEVICE, start=start)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])

    # the cap at dim
    _, _, used = lanczos_tridiagonal(
        _hvp_of(a), _BIG, 10 * _BIG, _DEVICE, generator=_gen()
    )
    assert used <= _BIG, "never more steps than the dimension"

    # float64, started from an eigenvector: K_1 is already invariant
    a64 = torch.diag(torch.linspace(-2.0, 4.0, 50, dtype=torch.float64))
    unit = torch.zeros(50, dtype=torch.float64)
    unit[-1] = 1.0
    alpha, beta, used = lanczos_tridiagonal(_hvp_of(a64), 50, 20, _DEVICE, start=unit)
    assert used == 1, "breakdown at the first step"
    _assert_rel(alpha[0].item(), 4.0, 1e-12, "alpha_1 = the eigenvalue")
    assert beta[0].item() <= 1e-12

    # float64 rank-one projector: K(A, q1) = span{q1, v}, so beta_2 = 0
    v = _fixed_unit(50, seed=7).to(torch.float64)
    rank_one = 3.0 * torch.outer(v, v)
    q1 = _fixed_unit(50, seed=8).to(torch.float64)
    alpha, beta, used = lanczos_tridiagonal(
        _hvp_of(rank_one), 50, 20, _DEVICE, start=q1
    )
    assert used == 2, f"a rank-one Krylov space is 2-dimensional, not {used}"
    ritz = extreme_ritz(_hvp_of(rank_one), 50, 20, _DEVICE, generator=_gen(), start=q1)
    assert ritz.n_iter_used == 2 and ritz.converged, "an exact invariant subspace"
    _assert_rel(ritz.lambda_max, 3.0, 1e-6, "rank-one lambda_max")
    assert abs(ritz.lambda_min) <= 1e-6, "and 0 on the orthogonal complement"

    # the zero operator breaks down immediately
    zero = torch.zeros(6, 6)
    ritz = extreme_ritz(_hvp_of(zero), 6, 5, _DEVICE, generator=_gen())
    assert (ritz.lambda_max, ritz.lambda_min, ritz.sigma) == (0.0, 0.0, 0.0)
    assert ritz.n_iter_used == 1 and ritz.n_hvp == 1 and ritz.converged


def _fixed_unit(dim: int, seed: int = 11) -> Tensor:
    """A deterministic unit start vector (float32)."""
    v = torch.randn(dim, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def test_extreme_eigenvalues_and_matrix_free_run_on_lanczos() -> None:
    """The two summaries take the Lanczos route: `extreme_eigenvalues` is
    `extreme_ritz`'s triple, and the matrix-free record carries the estimator's
    diagnostics next to the conditioning quantities."""
    a = _dense_symmetric(_LANCZOS_SPECTRA["near_zero_bulk"], seed=_LANCZOS_SEED)
    ritz = extreme_ritz(_hvp_of(a), _BIG, 40, _DEVICE, generator=_gen())
    assert extreme_eigenvalues(_hvp_of(a), _BIG, 40, _DEVICE, generator=_gen()) == (
        ritz.sigma,
        ritz.lambda_max,
        ritz.lambda_min,
    )

    est = matrix_free_conditioning(
        _hvp_of(a),
        _BIG,
        _LAMBDAS,
        n_iter=40,
        n_probes=8,
        device=_DEVICE,
        generator=_gen(),
    )
    exact = conditioning_from_spectrum(torch.linalg.eigvalsh(a.double()), _LAMBDAS)
    for key in ("lambda_max", "lambda_min", "sigma"):
        _assert_rel(est[key], exact[key], _RITZ_TOL, key)
    assert est["n_iter_used"] == 40 and est["converged"] is True
    assert est["residual_max"] >= 0.0 and est["residual_min"] >= 0.0
    assert "neg_frac" not in est
