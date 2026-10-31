"""
Validates the COUPLE-FAC overlay on random quadratics, where the exact
Hessian, the block-diagonal (K-FAC) optimum, and the full-Newton optimum are all known
in closed form. This is a fully independent reference for the overlay math.

Checked:
  - no-harm: m(d_accepted) <= m(d0) and ||d_accepted|| <= max(1, tr_ratio) ||d0|| for
               every random quadratic (the safeguard guarantee), across PD and
               indefinite curvature;
  - benefit: with real inter-block coupling and PD curvature, the overlay's proposal
               strictly decreases the local model and never beats the exact Newton
               optimum; the optimizer takes it exactly when it is inside the relative
               trust region (PD: the model test alone always passes);
  - fallback: the safeguard rejects a model-increasing step, and a step outside the
               relative trust region ||d1|| > tr_ratio ||d0|| without evaluating it;
  - decision: the reported m(d0), m(d1), norms and verdict match the explicit model;
               out of the trust region m(d1) is never evaluated (NaN); with nothing
               selected d0 is accepted trivially on the single shared HVP;
  - select: coupling-thresholded pair selection is correct and capped;
  - build: at a refresh step the cached eigenfactors reconstruct U^T H U and
               `correct` is the dense saddle-free reduced-Newton solve on them - on a
               PD reduced Hessian above the floor, the undamped reduced Newton step;
  - saddle: on an indefinite reduced Hessian the correction is bounded by
               ||U^T r|| / lambda and descends the residual along every negative-curvature
               eigendirection, where the plain damped solve would climb;
  - stale: subspaces built on one gradient, reused on another, give the closed-form
               correction with the cached U, H^ and the fresh residual, and the
               optimizer's verdict on that step still satisfies no-harm;
  - cadence: a refresh step costs 1 + sum_pairs (2 n_it r + r + k) HVPs, a non-refresh
               step 1 (H d0), plus one (H d1) iff the proposal is inside the trust
               region;
  - step: `.step` applies theta <- theta + lr*d (d == compute_direction) to the
               held parameters in block layout, and requires a parameter list.
"""

from __future__ import annotations

import math

import torch

from secondorder import (  # noqa: E402
    CoupleFacOptimizer,
    CoupleFacOverlay,
    OverlaySubspace,
    TrustRegionSafeguard,
)

_DTYPE = torch.float64
_DAMPING = float(CoupleFacOverlay().damping)  # the damping every test's model uses


def _spd(n: int, gen: torch.Generator) -> torch.Tensor:
    """A symmetric positive-definite n x n block."""
    a = torch.randn(n, n, dtype=_DTYPE, generator=gen)
    return a @ a.t() / n + torch.eye(n, dtype=_DTYPE)


def _build_quadratic(
    sizes: list[int],
    coupling_scale: float,
    gen: torch.Generator,
    *,
    make_pd: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
    """Random quadratic m(d) = g^T d + 1/2 d^T H d with a known block structure.

    Diagonal blocks are SPD (so K-FAC is always well-defined); off-diagonal blocks are
    scaled by `coupling_scale`. When `make_pd`, H is shifted to be PD overall.
    """
    p = sum(sizes)
    h = torch.zeros(p, p, dtype=_DTYPE)
    ranges: dict[str, slice] = {}
    offset = 0
    for i, s in enumerate(sizes):
        sl = slice(offset, offset + s)
        ranges[str(i)] = sl
        h[sl, sl] = _spd(s, gen)
        offset += s

    names = list(ranges)
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            si, sj = ranges[names[i]], ranges[names[j]]
            block = coupling_scale * torch.randn(
                sizes[i], sizes[j], dtype=_DTYPE, generator=gen
            )
            h[si, sj] = block
            h[sj, si] = block.t()

    h = 0.5 * (h + h.t())
    if make_pd:
        lo = torch.linalg.eigvalsh(h)[0].item()
        if lo <= 1e-3:
            h = h + (1e-3 - lo) * torch.eye(p, dtype=_DTYPE)
    g = torch.randn(p, dtype=_DTYPE, generator=gen)
    return g, h, ranges


class _ExactKFAC:
    """Block-diagonal Newton provider: d0 = -blkdiag(H)^{-1} g (the K-FAC oracle)."""

    def __init__(self, h: torch.Tensor, ranges: dict[str, slice]) -> None:
        self._h = h
        self._ranges = ranges

    def base_direction(self, grad: torch.Tensor) -> torch.Tensor:
        d0 = torch.zeros_like(grad)
        for sl in self._ranges.values():
            d0[sl] = torch.linalg.solve(self._h[sl, sl], -grad[sl])
        return d0

    def block_ranges(self) -> dict[str, slice]:
        return self._ranges


def _hvp_of(h: torch.Tensor):
    return lambda u: h @ u


def _model(
    g: torch.Tensor, h: torch.Tensor, d: torch.Tensor, damping: float = _DAMPING
) -> float:
    """The damped model m(d) = g.d + 1/2 d.(H d) + damping/2 ||d||^2."""
    return (g.dot(d) + 0.5 * d.dot(h @ d) + 0.5 * damping * d.dot(d)).item()


def _coupling(
    h: torch.Tensor, ranges: dict[str, slice]
) -> dict[tuple[str, str], float]:
    names = list(ranges)
    out: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            si, sj = ranges[names[i]], ranges[names[j]]
            f_vw = h[si, sj].norm().item()
            f_vv = h[si, si].norm().item()
            f_ww = h[sj, sj].norm().item()
            out[(names[i], names[j])] = f_vw / (f_vv * f_ww) ** 0.5
    return out


def test_no_harm_random_quadratics() -> None:
    """The safeguard guarantees m(d) <= m(d0) and ||d|| <= max(1, tr_ratio) ||d0||."""
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        g, h, ranges = _build_quadratic(
            [4, 5, 3], coupling_scale=0.3, gen=gen, make_pd=(seed % 2 == 0)
        )
        opt = CoupleFacOptimizer(provider=_ExactKFAC(h, ranges), tau=0.0)
        d0 = opt.provider.base_direction(g)
        d = opt.compute_direction(
            g, _hvp_of(h), _coupling(h, ranges), refresh=True, generator=gen
        ).direction
        m0, md = _model(g, h, d0), _model(g, h, d)
        assert md <= m0 + 1e-7, f"no-harm violated (seed={seed}): m(d)={md} m(d0)={m0}"
        bound = max(1.0, opt.tr_ratio) * d0.norm().item()
        assert d.norm().item() <= bound + 1e-9, f"trust region violated (seed={seed})"


def test_overlay_strict_benefit() -> None:
    """PD + coupling: the proposal has m* <= m(d1) < m0; taken iff inside the region."""
    for seed in range(10):
        gen = torch.Generator().manual_seed(100 + seed)
        g, h, ranges = _build_quadratic([5, 5], coupling_scale=0.7, gen=gen)
        hvp = _hvp_of(h)
        provider = _ExactKFAC(h, ranges)
        overlay = CoupleFacOverlay(rank=2, n_power_iter=12)
        coupling = _coupling(h, ranges)
        d0 = provider.base_direction(g)
        subspaces = overlay.build(
            d0,
            hvp,
            ranges,
            overlay.select(coupling, tau=0.0),
            generator=torch.Generator().manual_seed(3),
        )
        d1 = overlay.correct(g, d0, hvp(d0), subspaces)
        eye = torch.eye(h.shape[0], dtype=_DTYPE)
        d_newton = torch.linalg.solve(h + _DAMPING * eye, -g)  # damped-model optimum
        m0, m1, m_star = _model(g, h, d0), _model(g, h, d1), _model(g, h, d_newton)
        assert m1 < m0 - 1e-6, f"overlay gave no benefit (seed={seed}): m={m1} m0={m0}"
        assert m1 >= m_star - 1e-7, (
            f"beat the Newton optimum (seed={seed}): {m1} < {m_star}"
        )

        # the optimizer's verdict on the same proposal: PD curvature makes the model
        # test pass, so the decision is exactly the relative trust-region test
        opt = CoupleFacOptimizer(provider=provider, overlay=overlay, tau=0.0)
        decision = opt.compute_direction(
            g, hvp, coupling, refresh=True, generator=torch.Generator().manual_seed(3)
        ).decision
        inside = d1.norm().item() <= opt.tr_ratio * d0.norm().item()
        assert decision.accepted == inside, (
            f"verdict != trust-region test (seed={seed})"
        )
        assert torch.allclose(decision.direction, d1 if inside else d0, atol=1e-12)


def test_safeguard_rejects_harmful_and_far_steps() -> None:
    """The safeguard falls back to d0 on a model-increasing or out-of-region step.

    d0 is the K-FAC step of a PD quadratic, so g^T d0 < 0: its mirror -d0 has the same
    norm (inside any tr_ratio >= 1) and a larger model value - rejected by the model
    test, on one HVP; 3 d0 is outside tr_ratio = 2 - rejected before any HVP, m1 NaN.
    """
    gen = torch.Generator().manual_seed(7)
    g, h, ranges = _build_quadratic([4, 4], coupling_scale=0.5, gen=gen)
    calls: list[int] = []

    def counting_hvp(u: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return h @ u

    sg = TrustRegionSafeguard()
    d0 = _ExactKFAC(h, ranges).base_direction(g)
    hd0 = h @ d0
    m0 = _model(g, h, d0)

    harmful = -d0
    assert _model(g, h, harmful) > m0, "the mirror step is model-increasing"
    verdict = sg.accept(
        g, d0, harmful, counting_hvp, hd0=hd0, tr_ratio=2.0, damping=_DAMPING
    )
    assert torch.equal(verdict.direction, d0) and not verdict.accepted
    assert abs(verdict.m1 - _model(g, h, harmful)) < 1e-9 and len(calls) == 1

    calls.clear()
    far = 3.0 * d0
    verdict = sg.accept(
        g, d0, far, counting_hvp, hd0=hd0, tr_ratio=2.0, damping=_DAMPING
    )
    assert torch.equal(verdict.direction, d0) and not verdict.accepted
    assert math.isnan(verdict.m1) and len(calls) == 0, "no HVP outside the region"
    assert abs(verdict.norm_d1 - 3.0 * verdict.norm_d0) < 1e-9


def test_safeguard_tiny_ratio_rejects_every_distinct_proposal() -> None:
    """tr_ratio small enough rejects any distinct d1 of comparable length, HVP-free."""
    gen = torch.Generator().manual_seed(8)
    g, h, ranges = _build_quadratic([4, 4], coupling_scale=0.5, gen=gen)
    calls: list[int] = []

    def counting_hvp(u: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return h @ u

    sg = TrustRegionSafeguard()
    d0 = _ExactKFAC(h, ranges).base_direction(g)
    hd0 = h @ d0
    for d1 in (0.5 * d0, d0.clone(), 3.0 * d0, torch.linalg.solve(h, -g)):
        verdict = sg.accept(
            g, d0, d1, counting_hvp, hd0=hd0, tr_ratio=1e-6, damping=_DAMPING
        )
        assert not verdict.accepted and math.isnan(verdict.m1)
        assert torch.equal(verdict.direction, d0)
    assert len(calls) == 0, "no HVP is spent on a proposal outside the region"

    # d1 *is* d0: nothing to judge, accepted trivially whatever the ratio
    verdict = sg.accept(
        g, d0, d0, counting_hvp, hd0=hd0, tr_ratio=1e-6, damping=_DAMPING
    )
    assert verdict.accepted and verdict.m1 == verdict.m0 and len(calls) == 0


def test_decision_reports_exact_model_values() -> None:
    """m0, m1, the norms and the verdict in the decision match the explicit quadratic."""
    gen = torch.Generator().manual_seed(12)
    g, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.2, gen=gen)
    opt = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges),
        overlay=CoupleFacOverlay(rank=2, n_power_iter=12),
        tau=0.0,
    )
    d0 = opt.provider.base_direction(g)
    report = opt.compute_direction(
        g, _hvp_of(h), _coupling(h, ranges), refresh=True, generator=gen
    )
    decision = report.decision

    assert torch.equal(report.d0, d0), "the report carries the base direction"
    assert report.n_selected_pairs == 3, "tau = 0 selects every coupled pair"
    assert abs(decision.m0 - _model(g, h, d0)) < 1e-6, "m0 = g.d0 + 1/2 d0.(H d0)"
    assert abs(decision.norm_d0 - d0.norm().item()) < 1e-9
    assert decision.norm_d1 <= opt.tr_ratio * decision.norm_d0, (
        "this proposal is inside the relative trust region, so m1 is evaluated"
    )
    assert decision.accepted == (decision.m1 <= decision.m0 + opt.safeguard.tol), (
        "verdict is the model comparison"
    )
    assert decision.accepted, "PD + real coupling: the correction is taken"
    d1 = report.direction
    assert abs(decision.m1 - _model(g, h, d1)) < 1e-6, "m1 = g.d1 + 1/2 d1.(H d1)"
    assert abs(decision.norm_d1 - d1.norm().item()) < 1e-9
    assert torch.equal(decision.direction, d1)


def test_decision_outside_trust_ratio_skips_m1() -> None:
    """||d1|| > tr_ratio ||d0||: rejected before its HVP is spent; m1 NaN and d = d0."""
    gen = torch.Generator().manual_seed(13)
    g, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.7, gen=gen)
    calls: list[int] = []

    def counting_hvp(u: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return h @ u

    opt = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges),
        overlay=CoupleFacOverlay(rank=2, n_power_iter=4),
        tau=0.0,
        tr_ratio=1.0,
    )
    d0 = opt.provider.base_direction(g)
    report = opt.compute_direction(
        g, counting_hvp, _coupling(h, ranges), refresh=True, generator=gen
    )
    decision = report.decision

    assert not decision.accepted and math.isnan(decision.m1)
    assert decision.norm_d1 > opt.tr_ratio * decision.norm_d0, (
        "the overlay proposed a distinct d1 outside the region"
    )
    assert torch.equal(report.direction, d0) and torch.equal(decision.direction, d0)
    assert abs(decision.m0 - _model(g, h, d0)) < 1e-6

    # reusing the cache on the same gradient: only H d0 is computed, never H d1
    calls.clear()
    report = opt.compute_direction(g, counting_hvp, {}, refresh=False)
    assert len(calls) == 1 and not report.decision.accepted
    assert math.isnan(report.decision.m1)


def test_empty_selection_accepts_trivially_on_one_hvp() -> None:
    """Nothing selected: d1 is d0, accepted with m1 = m0, and only H d0 is computed."""
    gen = torch.Generator().manual_seed(14)
    g, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.7, gen=gen)
    calls: list[int] = []

    def counting_hvp(u: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return h @ u

    opt = CoupleFacOptimizer(provider=_ExactKFAC(h, ranges), tau=math.inf)
    d0 = opt.provider.base_direction(g)
    report = opt.compute_direction(
        g, counting_hvp, _coupling(h, ranges), refresh=True, generator=gen
    )
    decision = report.decision

    assert len(calls) == 1, f"empty selection must cost exactly one HVP, got {calls}"
    assert report.n_selected_pairs == 0
    assert decision.accepted and decision.m1 == decision.m0
    assert abs(decision.m0 - _model(g, h, d0)) < 1e-6
    assert torch.equal(report.direction, d0) and decision.norm_d1 == decision.norm_d0


def test_select_threshold_and_cap() -> None:
    """select keeps pairs above tau, strongest first, capped at max_pairs."""
    overlay = CoupleFacOverlay(max_pairs=2)
    coupling = {("0", "1"): 0.9, ("0", "2"): 0.05, ("1", "2"): 0.5}
    assert overlay.select(coupling, tau=0.1) == [("0", "1"), ("1", "2")]


def _dense_correction(
    g: torch.Tensor,
    h: torch.Tensor,
    d0: torch.Tensor,
    sub: OverlaySubspace,
    damping: float,
) -> torch.Tensor:
    """d0 + U c*,  c* = -V diag(1 / (|lambda| + damping)) V^T U^T r,  r the damped residual.

    The dense saddle-free reduced-Newton solve on the cached eigenfactors (U, V, lambda).
    """
    residual = g + h @ d0 + damping * d0  # the damped system's residual
    u, lam, vecs = sub.basis, sub.eigenvalues, sub.eigenvectors
    curvature = lam.abs() + damping
    h_sf_inv = vecs @ torch.diag(1.0 / curvature) @ vecs.t()
    return d0 - u @ (h_sf_inv @ (u.t() @ residual))


def test_build_and_correct_match_closed_form() -> None:
    """Refresh: the eigenfactors give U^T H U; correct == the dense saddle-free solve.

    On this PD quadratic every reduced eigenvalue is positive, so the saddle-free step
    is the Tikhonov-damped reduced Newton step -U (U^T H U + lambda I)^{-1} U^T r.
    """
    gen = torch.Generator().manual_seed(31)
    g, h, ranges = _build_quadratic([4, 5, 3], coupling_scale=0.7, gen=gen)
    hvp = _hvp_of(h)
    overlay = CoupleFacOverlay(rank=2, n_power_iter=6)
    d0 = _ExactKFAC(h, ranges).base_direction(g)
    hd0 = hvp(d0)
    selected = overlay.select(_coupling(h, ranges), tau=0.0)

    sub = overlay.build(d0, hvp, ranges, selected, generator=gen)

    assert sub is not None and sub.pairs == tuple(selected), "one joint subspace"
    u, lam, vecs = sub.basis, sub.eigenvalues, sub.eigenvectors
    k = u.shape[1]
    assert 1 <= k <= len(selected) * (2 + 2 * overlay.rank)
    assert torch.allclose(u.t() @ u, torch.eye(k, dtype=_DTYPE), atol=1e-10)
    assert lam.shape == (k,) and vecs.shape == (k, k)
    assert torch.all(lam[:-1] <= lam[1:]), "eigenvalues ascend (eigh order)"
    assert torch.allclose(vecs.t() @ vecs, torch.eye(k, dtype=_DTYPE), atol=1e-10)
    assert torch.allclose(sub.reduced, u.t() @ h @ u, atol=1e-10), "H^ = U^T H U"
    assert torch.allclose(sub.reduced, sub.reduced.t(), atol=1e-12), "symmetric"
    d1 = overlay.correct(g, d0, hd0, sub)
    expected = _dense_correction(g, h, d0, sub, float(overlay.damping))
    assert torch.allclose(d1, expected, atol=1e-8), "correct != dense saddle-free solve"
    assert overlay.correct(g, d0, hd0, None) is d0, "no subspace: d0 itself"
    assert overlay.build(d0, hvp, ranges, [], generator=gen) is None

    residual = g + hd0 + float(overlay.damping) * d0
    assert lam.min().item() > 0, "PD reduced Hessian: the precondition of the identity"
    damped = sub.reduced + float(overlay.damping) * torch.eye(k, dtype=_DTYPE)
    newton = d0 - u @ torch.linalg.solve(damped, u.t() @ residual)
    assert torch.allclose(d1, newton, atol=1e-8), "PD: damped reduced Newton"


def _reference_columns(
    overlay: CoupleFacOverlay,
    d0: torch.Tensor,
    h: torch.Tensor,
    ranges: dict[str, slice],
    selected: list[tuple[str, str]],
    gen: torch.Generator,
) -> torch.Tensor:
    """The original dense column stack: embedded d0 slices and singular directions."""
    from secondorder.overlay import _embed

    hvp = _hvp_of(h)
    dim = d0.numel()
    cols = []
    for v, w in selected:
        sv, sw = ranges[v], ranges[w]
        cols += [_embed(d0[sv], sv, dim), _embed(d0[sw], sw, dim)]
        left, right = overlay._cross_singular(d0, hvp, sv, sw, gen)
        for i in range(right.shape[1]):
            cols += [_embed(left[:, i], sv, dim), _embed(right[:, i], sw, dim)]
    return torch.stack(cols, dim=1)


def _exact_basis(columns: torch.Tensor) -> torch.Tensor:
    """An exact orthonormal basis of the column space (SVD, dense reference)."""
    u, s, _ = torch.linalg.svd(columns, full_matrices=False)
    return u[:, s > 1e-8 * s[0]]


def test_lean_build_matches_the_dense_path() -> None:
    """In-place Gram orthonormalization + column-wise H^ reproduce the dense build.

    The reference is the dense stack of the same columns with an exact SVD basis of
    its column space. Two orthonormal bases of one space differ by a rotation, so the
    invariants are compared: the projector U U^T, the reduced spectrum, and the
    correction (basis-independent by construction of the saddle-free solve). With
    three pairs over three blocks the stack is rank deficient (d0|_v repeats), the
    case the in-place path must get right.
    """
    from secondorder.overlay import _lift, _orthonormalizing_transform

    g, h, ranges = _build_quadratic(
        [6, 5, 4], coupling_scale=0.8, gen=torch.Generator().manual_seed(77)
    )
    hvp = _hvp_of(h)
    overlay = CoupleFacOverlay(rank=2, n_power_iter=5)
    d0 = _ExactKFAC(h, ranges).base_direction(g)
    hd0 = hvp(d0)
    selected = overlay.select(_coupling(h, ranges), tau=0.0)
    assert len(selected) == 3

    sub = overlay.build(
        d0, hvp, ranges, selected, generator=torch.Generator().manual_seed(5)
    )
    columns = _reference_columns(
        overlay, d0, h, ranges, selected, torch.Generator().manual_seed(5)
    )
    q_ref = _exact_basis(columns)
    assert q_ref.shape[1] < columns.shape[1], "the stack is rank deficient"

    assert sub is not None
    u = sub.basis
    k = u.shape[1]
    assert k == q_ref.shape[1], "exactly the null directions are dropped"
    assert torch.allclose(u.t() @ u, torch.eye(k, dtype=_DTYPE), atol=1e-10)
    assert torch.allclose(u @ u.t(), q_ref @ q_ref.t(), atol=1e-5), "same subspace"
    h_ref = q_ref.t() @ h @ q_ref
    assert torch.allclose(sub.eigenvalues, torch.linalg.eigvalsh(h_ref), atol=1e-5)
    assert torch.allclose(sub.reduced, u.t() @ h @ u, atol=1e-8), "H^ = U^T H U"
    d1 = overlay.correct(g, d0, hd0, sub)
    lam_ref, vecs_ref = torch.linalg.eigh(h_ref)
    dense = OverlaySubspace(
        pairs=tuple(selected),
        columns=((slice(0, d0.numel()), q_ref),),
        transform=torch.eye(k, dtype=_DTYPE),
        dim=d0.numel(),
        eigenvalues=lam_ref,
        eigenvectors=vecs_ref,
    )
    assert torch.allclose(dense.basis, q_ref), "factored form with T = I is U itself"
    ref = _dense_correction(g, h, d0, dense, float(overlay.damping))
    assert torch.allclose(d1, ref, atol=1e-5), "same correction"

    # Repeated and zero columns on one block, another column on a second block: only
    # the null directions go, and U = C T is orthonormal across both blocks.
    a = torch.randn(5, dtype=_DTYPE, generator=torch.Generator().manual_seed(1))
    b = torch.randn(5, dtype=_DTYPE, generator=torch.Generator().manual_seed(2))
    c = torch.randn(4, dtype=_DTYPE, generator=torch.Generator().manual_seed(3))
    raw = (
        (slice(0, 5), torch.stack([a, a.clone(), b, torch.zeros(5, dtype=_DTYPE)], 1)),
        (slice(5, 9), c[:, None]),
    )
    transform = _orthonormalizing_transform(raw)
    assert transform is not None and transform.shape == (5, 3)
    basis = torch.stack([_lift(raw, transform[:, j], 9) for j in range(3)], dim=1)
    assert torch.allclose(basis.t() @ basis, torch.eye(3, dtype=_DTYPE), atol=1e-10)
    stack = torch.stack([_lift(raw, e, 9) for e in torch.eye(5, dtype=_DTYPE)], dim=1)
    q_stack = _exact_basis(stack)
    assert torch.allclose(basis @ basis.t(), q_stack @ q_stack.t(), atol=1e-8)
    empty = ((slice(0, 5), torch.zeros(5, 2, dtype=_DTYPE)),)
    assert _orthonormalizing_transform(empty) is None


def test_saddle_free_correction_on_indefinite_reduced_hessian() -> None:
    """Indefinite H^: ||c*|| <= ||U^T r|| / lambda, and descent along negative curvature.

    A strongly scaled cross block makes the quadratic indefinite with negative curvature
    inside the pair subspace. Along every eigenvector with lambda_i < 0 the saddle-free
    coefficient of c* has the sign opposite to the residual's - it moves down the
    residual - whereas the plain damped solve -(H^ + lambda I)^{-1} U^T r keeps the residual's
    sign there (toward the saddle, where the model is unbounded below).
    """
    gen = torch.Generator().manual_seed(42)
    g, h, ranges = _build_quadratic([4, 4], coupling_scale=2.0, gen=gen, make_pd=False)
    hvp = _hvp_of(h)
    overlay = CoupleFacOverlay(rank=2, n_power_iter=8)
    provider = _ExactKFAC(h, ranges)
    coupling = _coupling(h, ranges)
    d0 = provider.base_direction(g)
    hd0 = hvp(d0)
    sub = overlay.build(
        d0,
        hvp,
        ranges,
        overlay.select(coupling, tau=0.0),
        generator=torch.Generator().manual_seed(3),
    )
    assert sub is not None
    u, lam, vecs = sub.basis, sub.eigenvalues, sub.eigenvectors
    damping = float(overlay.damping)
    negative = lam < 0
    assert negative.any() and (lam[negative] + damping < 0).all(), (
        "indefinite H^ (below -lambda)"
    )

    residual = g + hd0 + damping * d0
    u_r = u.t() @ residual
    delta = overlay.correct(g, d0, hd0, sub) - d0
    c = u.t() @ delta
    assert torch.allclose(u @ c, delta, atol=1e-10), "the correction lies in span U"
    assert c.norm().item() <= u_r.norm().item() / damping * (1 + 1e-12), "||c*|| bound"

    coef_c, coef_r = vecs.t() @ c, vecs.t() @ u_r
    assert (coef_r[negative].abs() > 1e-6).all(), "residual mass on negative curvature"
    assert (coef_c[negative] * coef_r[negative] < 0).all(), (
        "saddle-free: descent on the residual along negative curvature"
    )
    assert torch.allclose(coef_c, -coef_r / (lam.abs() + damping), atol=1e-10), (
        "c* = -V diag(1 / (|lambda| + lambda_d)) V^T U^T r, coefficient by coefficient"
    )

    k = u.shape[1]
    plain = torch.linalg.solve(sub.reduced + damping * torch.eye(k, dtype=_DTYPE), -u_r)
    coef_plain = vecs.t() @ plain
    assert (coef_plain[negative] * coef_r[negative] > 0).all(), (
        "the plain damped solve climbs toward the saddle"
    )

    # through the optimizer: no-harm and the relative trust region hold here too
    opt = CoupleFacOptimizer(provider=provider, overlay=overlay, tau=0.0)
    d = opt.compute_direction(
        g, hvp, coupling, refresh=True, generator=torch.Generator().manual_seed(3)
    ).direction
    assert _model(g, h, d) <= _model(g, h, d0) + 1e-7
    assert d.norm().item() <= max(1.0, opt.tr_ratio) * d0.norm().item() + 1e-9


def test_stale_subspaces_reuse_closed_form_and_no_harm() -> None:
    """Built on (g1, d0_1), reused on (g2, d0_2): cached U, H^ with the fresh residual."""
    gen = torch.Generator().manual_seed(32)
    g1, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.7, gen=gen)
    g2 = torch.randn(g1.numel(), dtype=_DTYPE, generator=gen)
    hvp = _hvp_of(h)
    provider = _ExactKFAC(h, ranges)
    overlay = CoupleFacOverlay(rank=2, n_power_iter=6)
    coupling = _coupling(h, ranges)

    d0_1 = provider.base_direction(g1)
    sub = overlay.build(
        d0_1,
        hvp,
        ranges,
        overlay.select(coupling, tau=0.0),
        generator=torch.Generator().manual_seed(3),
    )
    assert sub is not None
    d0_2 = provider.base_direction(g2)
    expected = _dense_correction(g2, h, d0_2, sub, float(overlay.damping))

    d1 = overlay.correct(g2, d0_2, hvp(d0_2), sub)
    assert torch.allclose(d1, expected, atol=1e-8), "stale reuse != closed form"
    assert not torch.allclose(d1, d0_2), "the stale subspaces still propose a step"

    # the same two steps through the optimizer: refresh on g1, reuse on g2 (no coupling
    # is consulted on the reuse step) - the safeguard's verdict keeps no-harm
    opt = CoupleFacOptimizer(provider=provider, overlay=overlay, tau=0.0)
    opt.compute_direction(
        g1, hvp, coupling, refresh=True, generator=torch.Generator().manual_seed(3)
    )
    report = opt.compute_direction(g2, hvp, {}, refresh=False)
    decision = report.decision
    assert report.n_selected_pairs == len(sub.pairs) == 3
    assert abs(decision.norm_d1 - expected.norm().item()) < 1e-8, "same stale d1"
    assert not decision.accepted or decision.m1 <= decision.m0 + opt.safeguard.tol
    assert _model(g2, h, report.direction) <= _model(g2, h, d0_2) + 1e-7, "no-harm"


def test_refresh_cadence_hvp_costs() -> None:
    """Refresh: 1 + sum_pairs (2 n_it r + r) + k HVPs; else 1; +1 (H d1) inside the region.

    m(d1) is evaluated - one HVP - only for a proposal inside the relative trust region
    (``m1`` finite); outside it the safeguard rejects on the norms alone.
    """
    gen = torch.Generator().manual_seed(33)
    g, h, ranges = _build_quadratic([4, 5, 3], coupling_scale=0.7, gen=gen)
    calls: list[int] = []

    def counting_hvp(u: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return h @ u

    rank, n_it = 2, 3
    opt = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges),
        overlay=CoupleFacOverlay(rank=rank, n_power_iter=n_it),
        tau=0.0,
    )
    coupling = _coupling(h, ranges)
    sizes = {name: sl.stop - sl.start for name, sl in ranges.items()}

    evaluated: list[bool] = []
    for t in range(5):
        grad = torch.randn(g.numel(), dtype=_DTYPE, generator=gen)
        calls.clear()
        opt.tr_ratio = 1e6 if t % 2 else 2.0  # exercise both branches deterministically
        report = opt.compute_direction(
            grad, counting_hvp, coupling, refresh=(t % 2 == 0), generator=gen
        )
        assert report.n_selected_pairs == 3, "every coupled pair is cached"
        assert opt._subspace is not None
        build = opt._subspace.basis.shape[1]  # k HVPs for the joint reduced Hessian
        for v, w in opt._subspace.pairs:
            r = min(rank, sizes[v], sizes[w])
            build += 2 * n_it * r + r
        decision = report.decision
        inside = decision.norm_d1 <= opt.tr_ratio * decision.norm_d0
        assert inside == math.isfinite(decision.m1), "m1 is evaluated iff inside"
        evaluated.append(inside)
        expected = (1 + build if t % 2 == 0 else 1) + int(inside)
        assert len(calls) == expected, f"step {t}: {len(calls)} HVPs != {expected}"
    assert any(evaluated) and not all(evaluated), "both branches were exercised"


def test_step_applies_accepted_direction() -> None:
    """`.step` applies theta <- theta + lr*d in block layout, with d == compute_direction."""
    gen = torch.Generator().manual_seed(11)
    g, h, ranges = _build_quadratic([4, 5], coupling_scale=0.7, gen=gen)
    lr = 0.5
    params = [
        torch.nn.Parameter(torch.randn(4, dtype=_DTYPE, generator=gen)),
        torch.nn.Parameter(torch.randn(5, dtype=_DTYPE, generator=gen)),
    ]
    opt = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges),
        overlay=CoupleFacOverlay(rank=2, n_power_iter=12),
        params=params,
        lr=lr,
        tau=0.0,
    )

    # the overlay draws its subspace from `generator`; seed both calls identically so the
    # reference direction matches the one `.step` applies. compute_direction mutates nothing.
    d_expected = opt.compute_direction(
        g,
        _hvp_of(h),
        _coupling(h, ranges),
        refresh=True,
        generator=torch.Generator().manual_seed(3),
    ).direction
    before = [p.detach().clone() for p in params]

    d_step = opt.step(
        g,
        _hvp_of(h),
        _coupling(h, ranges),
        refresh=True,
        generator=torch.Generator().manual_seed(3),
    ).direction

    assert torch.allclose(d_step, d_expected), "step direction != compute_direction"
    offset = 0
    for p, p0 in zip(params, before, strict=True):
        n = p.numel()
        moved = p.detach() - p0
        assert torch.allclose(
            moved, lr * d_expected[offset : offset + n].view_as(p), atol=1e-12
        ), "parameter not updated by lr * d in block layout"
        offset += n


def test_step_requires_params() -> None:
    """`.step` without a held parameter list is a usage error, not a silent no-op."""
    gen = torch.Generator().manual_seed(5)
    g, h, ranges = _build_quadratic([3, 3], coupling_scale=0.3, gen=gen)
    opt = CoupleFacOptimizer(provider=_ExactKFAC(h, ranges), tau=0.0)
    raised = False
    try:
        opt.step(g, _hvp_of(h), _coupling(h, ranges), refresh=True)
    except RuntimeError:
        raised = True
    assert raised, "step must require params to be set"


def test_loss_gate_decides_after_ratio_and_model() -> None:
    """The loss test runs last, only on proposals that passed the other two, and its
    verdict and both losses are recorded; without ``loss_at`` nothing changes."""
    gen = torch.Generator().manual_seed(15)
    g, h, ranges = _build_quadratic([4, 4, 4], coupling_scale=0.7, gen=gen)
    sg = TrustRegionSafeguard()
    d0 = _ExactKFAC(h, ranges).base_direction(g)
    hd0 = h @ d0
    # a short exact line-search step from d0 along -grad m: inside the ratio (at most
    # 1.5 ||d0||) and model-decreasing by construction
    grad_m = g + hd0 + _DAMPING * d0
    unit = -grad_m / grad_m.norm()
    curv = unit.dot(h @ unit) + _DAMPING
    step = min((grad_m.norm() / curv).item(), 0.5 * d0.norm().item())
    d1 = d0 + step * unit
    assert _model(g, h, d1) < _model(g, h, d0) and d1.norm() <= 1.5 * d0.norm()
    evaluated: list[torch.Tensor] = []

    def loss_up(direction: torch.Tensor) -> float:  # d1 loses on the minibatch
        evaluated.append(direction)
        return 1.0 if direction is d1 else 0.5

    def loss_down(direction: torch.Tensor) -> float:
        return 0.2 if direction is d1 else 0.5

    verdict = sg.accept(
        g, d0, d1, _hvp_of(h), hd0=hd0, tr_ratio=2.0, damping=_DAMPING, loss_at=loss_up
    )
    assert not verdict.accepted and verdict.rejected_by == "loss"
    assert torch.equal(verdict.direction, d0)
    assert (verdict.loss_d0, verdict.loss_d1) == (0.5, 1.0) and len(evaluated) == 2

    verdict = sg.accept(
        g,
        d0,
        d1,
        _hvp_of(h),
        hd0=hd0,
        tr_ratio=2.0,
        damping=_DAMPING,
        loss_at=loss_down,
    )
    assert verdict.accepted and verdict.rejected_by is None
    assert (verdict.loss_d0, verdict.loss_d1) == (0.5, 0.2)

    evaluated.clear()  # rejected by the ratio: the loss is never evaluated
    verdict = sg.accept(
        g,
        d0,
        3.0 * d0,
        _hvp_of(h),
        hd0=hd0,
        tr_ratio=2.0,
        damping=_DAMPING,
        loss_at=loss_up,
    )
    assert verdict.rejected_by == "ratio" and math.isnan(verdict.loss_d0)
    assert evaluated == []

    verdict = sg.accept(g, d0, d1, _hvp_of(h), hd0=hd0, tr_ratio=2.0, damping=_DAMPING)
    assert verdict.accepted and math.isnan(verdict.loss_d1), (
        "no loss_at: model test only"
    )
