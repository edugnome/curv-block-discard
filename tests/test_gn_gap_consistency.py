"""Consistency test for exact and stochastic GN-Gap.

Checks:
  1. Exact vs stochastic GN-Gap estimates converge (for a smooth activation).
  2. ReLU: GN-Gap ~= 0 (exact and stochastic).
  3. _assert_mean_reduction catches incorrect reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.models import PlainMLP  # noqa: E402
from hessian import ExactGNDecomposition, StochasticGNGapEstimator  # noqa: E402


def _make_data(
    batch_size: int = 4,
    input_dim: int = 32,
    num_classes: int = 10,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random input + labels."""
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


# ------------------------------------------------------------------
# 1. Exact vs stochastic (GELU - smooth, gap > 0)
# ------------------------------------------------------------------


def test_exact_vs_stochastic_gelu() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    model = PlainMLP(input_dim=32, width=8, depth=3, activation="gelu").to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v, w = names[0], names[1]

    exact = ExactGNDecomposition(model, loss_fn)
    gap_exact = exact.compute_gn_gap(x, y, v, w)

    stoch = StochasticGNGapEstimator(model, loss_fn, n_probes=500)
    gap_stoch = stoch.estimate_gn_gap(x, y, v, w)

    tol = 0.15
    diff = abs(gap_exact.gap - gap_stoch.gap)
    assert diff < tol, (
        f"Exact vs stochastic GN-Gap diverge: "
        f"exact={gap_exact.gap:.4f}, stochastic={gap_stoch.gap:.4f}, diff={diff:.4f}"
    )

    # Frobenius norm should also be close
    frob_diff = abs(gap_exact.frob_gn - gap_stoch.frob_gn) / (gap_exact.frob_gn + 1e-12)
    assert frob_diff < 0.2, (
        f"||H^GN|| diverges: exact={gap_exact.frob_gn:.4f}, "
        f"stochastic={gap_stoch.frob_gn:.4f}"
    )
    print(f"  GELU gap exact={gap_exact.gap:.4f}, stochastic={gap_stoch.gap:.4f} - OK")


# ------------------------------------------------------------------
# 2. ReLU: gap ~= 0
# ------------------------------------------------------------------


def test_relu_gap_zero() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    # use_layernorm=False: network is piecewise-linear => H^T = 0 a.e.
    model = PlainMLP(
        input_dim=32, width=8, depth=3, activation="relu", use_layernorm=False
    ).to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v, w = names[0], names[1]

    exact = ExactGNDecomposition(model, loss_fn)
    gap_exact = exact.compute_gn_gap(x, y, v, w)
    assert gap_exact.gap < 1e-6, f"ReLU exact gap should be ~0, got {gap_exact.gap:.6f}"

    stoch = StochasticGNGapEstimator(model, loss_fn, n_probes=200)
    gap_stoch = stoch.estimate_gn_gap(x, y, v, w)
    assert gap_stoch.gap < 1e-3, (
        f"ReLU stochastic gap should be ~0, got {gap_stoch.gap:.6f}"
    )
    print(f"  ReLU gap exact={gap_exact.gap:.2e}, stochastic={gap_stoch.gap:.2e} - OK")


# ------------------------------------------------------------------
# 3. _assert_mean_reduction
# ------------------------------------------------------------------


def test_reduction_assertion() -> None:
    model = PlainMLP(input_dim=32, width=8, depth=3)

    bad_loss = nn.CrossEntropyLoss(reduction="sum")
    try:
        ExactGNDecomposition(model, bad_loss)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    try:
        StochasticGNGapEstimator(model, bad_loss)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # reduction='mean' - OK
    good_loss = nn.CrossEntropyLoss()
    ExactGNDecomposition(model, good_loss)
    StochasticGNGapEstimator(model, good_loss)

    print("  Reduction assertion - OK")


# ------------------------------------------------------------------
# 4. generator seam: seeded reproducibility
# ------------------------------------------------------------------


def test_generator_seam_reproducible() -> None:
    device = torch.device("cpu")

    model = PlainMLP(input_dim=32, width=8, depth=3, activation="gelu").to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v, w = names[0], names[1]

    def estimate(seed: int) -> float:
        gen = torch.Generator(device=device).manual_seed(seed)
        est = StochasticGNGapEstimator(model, loss_fn, n_probes=16, generator=gen)
        return est.estimate_gn_gap(x, y, v, w).gap

    assert estimate(0) == estimate(0), (
        "identically seeded generators must agree exactly"
    )
    assert estimate(0) != estimate(1), "differently seeded generators must not agree"
    print("  generator seam: same seed -> equal, different seed -> unequal - OK")


# ------------------------------------------------------------------
