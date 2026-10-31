"""Coupling bounds for positive semidefinite and indefinite matrices."""

from __future__ import annotations

import numpy as np

from hessian.metrics import coupling_from_norms


def _couplings(h: np.ndarray, sizes: list[int]) -> list[float]:
    """C(v, w) of every unordered pair of the groups of ``sizes``, from the Frobenius
    norms of the blocks of ``h``."""
    edges = np.cumsum([0, *sizes])

    def norm(v: int, w: int) -> float:
        block = h[edges[v] : edges[v + 1], edges[w] : edges[w + 1]]
        return float(np.linalg.norm(block))

    count = len(sizes)
    return [
        coupling_from_norms(norm(v, w), norm(v, v), norm(w, w))
        for v in range(count)
        for w in range(v + 1, count)
    ]


def test_coupling_is_at_most_one_for_a_positive_semi_definite_matrix() -> None:
    """H = X^T X: C <= 1 on every pair, whatever the rank (a wide X gives a full-rank
    H, a short one a rank-deficient H), and C = 1 on every pair of a rank-one matrix."""
    rng = np.random.default_rng(0)
    sizes = [3, 5, 2, 4]
    for rows in (1, 2, 6, 14, 40):
        for _ in range(10):
            x = rng.standard_normal((rows, sum(sizes)))
            assert max(_couplings(x.T @ x, sizes)) <= 1.0 + 1e-12, rows

    u = rng.standard_normal(sum(sizes))
    ones = _couplings(np.outer(u, u), sizes)
    assert all(abs(c - 1.0) < 1e-12 for c in ones), ones


def test_coupling_exceeds_one_for_an_indefinite_matrix() -> None:
    """The converse direction, by example: [[0.1, 1], [1, 0.1]] has the eigenvalues 1.1
    and -0.9, and C = 1 / sqrt(0.1 * 0.1) = 10 on its one pair. Adding a multiple of the
    identity that makes the matrix positive definite (0.9 + 0.1 on the diagonal, 1.0
    off it) brings C back to at most one."""
    indefinite = np.array([[0.1, 1.0], [1.0, 0.1]])
    assert np.linalg.eigvalsh(indefinite).min() < 0.0
    (c,) = _couplings(indefinite, [1, 1])
    assert abs(c - 10.0) < 1e-12

    shifted = indefinite + 0.95 * np.eye(2)
    assert np.linalg.eigvalsh(shifted).min() > 0.0
    (c_shifted,) = _couplings(shifted, [1, 1])
    assert c_shifted <= 1.0
