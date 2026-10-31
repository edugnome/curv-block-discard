"""Phi's null models: what Phi would be with no structure at all."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.phi import measure_phi
from hessian.phi_baselines import n_eff, phi_null, phi_rank1, phi_sep


def test_equal_blocks_give_one_minus_inverse_nb() -> None:
    # n_b equal groups: phi_null = 1 - 1/n_b, the ceiling of phi_rank1 (Cauchy-Schwarz),
    # not of Phi, which the partition does not bound.
    assert phi_null([1] * 6) == pytest.approx(1.0 - 1.0 / 6)
    assert phi_null([7] * 4) == pytest.approx(1.0 - 1.0 / 4)


def test_a_single_block_discards_nothing() -> None:
    assert phi_null([10]) == pytest.approx(0.0)


def test_empty_and_nonpositive_groups_are_rejected() -> None:
    with pytest.raises(ValueError):
        phi_null([])
    with pytest.raises(ValueError):
        phi_null([0, 5])


def test_resnet18_null_matches_the_reported_value() -> None:
    # Measured group sizes of SegmentedResNet18(num_classes=100).get_param_groups()
    # (stem, layer1, layer2, layer3, layer4, head) - built the way
    # runner_exp8._build_model does for arch="resnet18" on CIFAR-100.
    sizes = [9_536, 147_968, 525_568, 2_099_712, 8_393_728, 51_300]
    assert phi_null(sizes) == pytest.approx(0.404, abs=0.001)


def test_phi_sep_of_equal_masses_gives_one_minus_inverse_nb() -> None:
    # Equal diagonal norms give a_v all equal too, so phi_sep collapses to the same
    # 1 - 1/n_b ceiling as phi_null's equal-groups case, whatever the common mass is.
    assert phi_sep([3.0] * 6) == pytest.approx(1.0 - 1.0 / 6)
    assert phi_sep({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}) == pytest.approx(0.75)


def test_phi_sep_matches_phi_null_when_a_is_proportional_to_p() -> None:
    # phi_null is phi_sep's special case at a proportional to p (module docstring):
    # phi_sep takes a_v = block_norms_v directly, so passing the sizes themselves makes
    # a_v equal the sizes exactly, and both sides then evaluate the same formula
    # 1 - sum(x^2) / sum(x)^2 on the same x - literally, not just approximately.
    for sizes in ([1, 2, 3, 5, 8, 13], [9_536, 147_968, 525_568, 2_099_712, 8_393_728]):
        block_norms = [float(p) for p in sizes]
        assert phi_sep(block_norms) == pytest.approx(phi_null(sizes))


def test_phi_sep_exact_on_a_rank1_matrix() -> None:
    # H = u u^T: ||H_vw||_F = ||u_v|| ||u_w|| exactly (a rank-1 outer product), so the
    # off-diagonal blocks are separable (||H_vw||_F = sqrt(a_v a_w), a_v = ||H_vv||_F)
    # and Phi = phi_sep(a) holds exactly, not just approximately (module docstring).
    torch.manual_seed(0)
    u = torch.randn(10, dtype=torch.float64)
    H = torch.outer(u, u)
    sizes = [3, 3, 4]
    starts = [0, 3, 6]
    # ||H_vv||_F = ||u_v||^2 exactly (a rank-1 outer product), the plain block norm
    # phi_sep takes.
    block_norms = [
        float(H[s : s + n, s : s + n].norm())
        for s, n in zip(starts, sizes, strict=True)
    ]
    phi_direct = 1.0 - sum(a**2 for a in block_norms) / float(H.norm() ** 2)
    assert phi_sep(block_norms) == pytest.approx(phi_direct, abs=1e-12)


def test_phi_sep_empty_and_negative_are_rejected() -> None:
    with pytest.raises(ValueError):
        phi_sep([])
    with pytest.raises(ValueError):
        phi_sep([-1.0, 2.0])


def test_n_eff_of_one_dominant_block_and_of_equal_blocks() -> None:
    assert n_eff([10.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert n_eff([2.0] * 5) == pytest.approx(5.0)


def test_n_eff_empty_and_negative_are_rejected() -> None:
    with pytest.raises(ValueError):
        n_eff([])
    with pytest.raises(ValueError):
        n_eff([-1.0, 2.0])


def test_rank1_phi_of_an_evenly_spread_vector() -> None:
    u = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert phi_rank1(u, [2, 2]) == pytest.approx(0.5)


def test_rank1_phi_of_a_vector_inside_one_group() -> None:
    u = torch.tensor([3.0, 4.0, 0.0, 0.0])
    assert phi_rank1(u, [2, 2]) == pytest.approx(0.0)


class _RankOneHessianModel(nn.Module):
    """L(theta) = 0.5 (a^T theta)^2 over two parameter groups: H = a a^T exactly.

    Each block is a bias-free ``Linear(d_i, 1)``, so ``pred = w1. x1 + w2. x2 =
    theta. x`` is linear in theta; ``0.5 * pred^2`` (target 0) then has second
    derivative ``x x^T`` in theta - constant, not merely dominant - whatever ``x`` is
    fed at measurement time.
    """

    def __init__(self, d1: int, d2: int) -> None:
        super().__init__()
        self.block1 = nn.Linear(d1, 1, bias=False)
        self.block2 = nn.Linear(d2, 1, bias=False)
        self._d1 = d1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block1(x[:, : self._d1]) + self.block2(x[:, self._d1 :])

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "block1": list(self.block1.parameters()),
            "block2": list(self.block2.parameters()),
        }


class _HalfSquaredLoss(nn.Module):
    """0.5 (pred - target)^2 summed: with target 0, exactly the rank-1 model's L(theta)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (pred - target).pow(2).sum()


def test_rank1_phi_matches_the_estimator_on_a_rank_one_hessian() -> None:
    # H = a a^T exactly (see _RankOneHessianModel), so the block estimator's `phi`
    # (Hutchinson, finite n_probes) and `phi_rank1` (closed form on the Lanczos-recovered
    # top eigenvector, exact here since H has one nonzero eigenvalue) both target the
    # same number. Measured over 10 seeds at n_probes=20000 the gap is 0.0001-0.0044;
    # 0.02 keeps an order of magnitude of margin while staying tighter than this
    # estimator's own Phi-vs-exact tolerance elsewhere (test_param_space, abs<0.04).
    torch.manual_seed(0)
    d1, d2 = 3, 5
    model = _RankOneHessianModel(d1, d2)
    model.eval()
    x = torch.randn(1, d1 + d2)
    y = torch.zeros(1, 1)

    out = measure_phi(model, x, y, _HalfSquaredLoss(), n_probes=20000, with_rank1=True)
    assert out["phi"] == pytest.approx(out["phi_rank1"], abs=0.02)


def test_with_rank1_false_leaves_the_report_unchanged() -> None:
    d1, d2 = 3, 5
    model = _RankOneHessianModel(d1, d2)
    model.eval()
    x = torch.randn(1, d1 + d2)
    y = torch.zeros(1, 1)
    loss_fn = _HalfSquaredLoss()

    torch.manual_seed(3)
    baseline = measure_phi(model, x, y, loss_fn, n_probes=50)
    torch.manual_seed(3)
    explicit = measure_phi(model, x, y, loss_fn, n_probes=50, with_rank1=False)
    assert explicit == baseline
    assert "phi_rank1" not in explicit


def test_generator_passthrough_is_independent_of_the_global_rng_position() -> None:
    # Two calls seeded identically through their own generator must agree even when
    # the ambient global RNG (torch.manual_seed) sits at a different position - the
    # property a probe-count sweep needs to isolate n_probes as the only variable.
    d1, d2 = 3, 5
    model = _RankOneHessianModel(d1, d2)
    model.eval()
    x = torch.randn(1, d1 + d2)
    y = torch.zeros(1, 1)
    loss_fn = _HalfSquaredLoss()

    torch.manual_seed(1)
    gen1 = torch.Generator()
    gen1.manual_seed(123)
    out1 = measure_phi(model, x, y, loss_fn, n_probes=20, generator=gen1)

    torch.manual_seed(999)  # a different global RNG position
    gen2 = torch.Generator()
    gen2.manual_seed(123)
    out2 = measure_phi(model, x, y, loss_fn, n_probes=20, generator=gen2)

    assert out1 == out2
