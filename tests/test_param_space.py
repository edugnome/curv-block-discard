"""Consistency test for the parameter-space Hessian estimator.

Checks the Hutchinson estimates of:
  1. cross-block Frobenius norm ||H_{theta_v, theta_w}||_F,
  2. the discarded-coupling fraction Phi,
  3. block symmetry ||H_{vw}||_F == ||H_{wv}||_F,
  4. the per-minibatch `curvature` bundle (exact gradient, exact full-parameter HVP,
     and inter-block coupling consistent with the dense Hessian),
against an exact brute-force parameter Hessian on a tiny smooth MLP.

The reference Hessian is built by a fully independent code path (differentiate
g = grad_theta L coordinate-by-coordinate), so it cross-validates the estimator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hessian.param_space import ParamBlockEstimator  # noqa: E402


class _TinyGroupedMLP(nn.Module):
    """Smooth 3-block MLP exposing parameter groups (small enough for an exact Hessian)."""

    def __init__(self, d_in: int = 5, d_h: int = 4, d_out: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_h)
        self.head = nn.Linear(d_h, d_out)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.head(h)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "fc1": list(self.fc1.parameters()),
            "fc2": list(self.fc2.parameters()),
            "head": list(self.head.parameters()),
        }


def _make_data(*, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 5, device=device)
    y = torch.randint(0, 3, (8,), device=device)
    return x, y


def _exact_param_hessian(
    model: _TinyGroupedMLP,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """Exact P x P parameter Hessian + per-group (offset, size) index ranges.

    Reference path independent of the estimator: build g = grad_theta L
    (create_graph), then differentiate each coordinate g[i] again.
    """
    groups = model.get_param_groups()
    names = list(groups.keys())
    params = [p for name in names for p in groups[name]]

    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in names:
        size = sum(p.numel() for p in groups[name])
        ranges[name] = (offset, size)
        offset += size
    total = offset

    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g = torch.cat([gi.reshape(-1) for gi in grads])

    hess = torch.zeros(total, total)
    for i in range(total):
        row = torch.autograd.grad(g[i], params, retain_graph=True, allow_unused=True)
        hess[i] = torch.cat(
            [
                (r if r is not None else torch.zeros_like(p)).reshape(-1)
                for r, p in zip(row, params, strict=True)
            ]
        )
    return hess.detach(), ranges


def _block_frob(
    hess: torch.Tensor, ranges: dict[str, tuple[int, int]], v: str, w: str
) -> float:
    ov, sv = ranges[v]
    ow, sw = ranges[w]
    block = hess[ov : ov + sv, ow : ow + sw]
    return torch.linalg.norm(block, ord="fro").item()


def _exact_phi(
    hess: torch.Tensor, ranges: dict[str, tuple[int, int]], names: list[str]
) -> float:
    full_sq = (hess**2).sum().item()
    diag_sq = sum(_block_frob(hess, ranges, n, n) ** 2 for n in names)
    return 1.0 - diag_sq / full_sq


# ------------------------------------------------------------------
# 1. Cross-block Frobenius norm vs exact
# ------------------------------------------------------------------


def test_cross_block_frob_vs_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    model = _TinyGroupedMLP().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    hess, ranges = _exact_param_hessian(model, loss_fn, x, y)
    est = ParamBlockEstimator(model, loss_fn, n_probes=6000)

    for v, w in [("fc1", "fc2"), ("fc1", "head"), ("fc2", "head")]:
        exact = _block_frob(hess, ranges, v, w)
        approx = est.estimate_block_frob_norm(x, y, v, w)
        rel = abs(exact - approx) / (exact + 1e-12)
        assert rel < 0.1, (
            f"||H_{v},{w}||_F mismatch: exact={exact:.4f} "
            f"approx={approx:.4f} rel={rel:.3f}"
        )
        print(f"  ||H_{v},{w}||_F exact={exact:.4f} approx={approx:.4f} - OK")


# ------------------------------------------------------------------
# 2. Phi vs exact
# ------------------------------------------------------------------


def test_phi_vs_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(1)

    model = _TinyGroupedMLP().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    hess, ranges = _exact_param_hessian(model, loss_fn, x, y)
    names = list(model.get_param_groups().keys())
    phi_exact = _exact_phi(hess, ranges, names)

    est = ParamBlockEstimator(model, loss_fn, n_probes=8000)
    phi_approx = est.estimate_phi(x, y)

    assert abs(phi_exact - phi_approx) < 0.04, (
        f"Phi mismatch: exact={phi_exact:.4f} approx={phi_approx:.4f}"
    )
    print(f"  Phi exact={phi_exact:.4f} approx={phi_approx:.4f} - OK")


# ------------------------------------------------------------------
# 3. Block symmetry ||H_{vw}|| == ||H_{wv}||
# ------------------------------------------------------------------


def test_block_symmetry() -> None:
    device = torch.device("cpu")
    torch.manual_seed(2)

    model = _TinyGroupedMLP().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    est = ParamBlockEstimator(model, loss_fn, n_probes=6000)
    r_vw = est.estimate_block_frob_norm(x, y, "fc1", "head")
    r_wv = est.estimate_block_frob_norm(x, y, "head", "fc1")
    rel = abs(r_vw - r_wv) / (r_vw + 1e-12)
    assert rel < 0.12, f"symmetry broken: {r_vw:.4f} vs {r_wv:.4f}"
    print(f"  ||H_fc1,head||={r_vw:.4f} ||H_head,fc1||={r_wv:.4f} - OK")


# ------------------------------------------------------------------
# 4. Per-minibatch curvature bundle (gradient + HVP + coupling)
# ------------------------------------------------------------------


def test_curvature_grad_and_hvp_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(3)

    model = _TinyGroupedMLP().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    hess, _ = _exact_param_hessian(model, loss_fn, x, y)
    total = hess.shape[0]

    groups = model.get_param_groups()
    params = [p for name in groups for p in groups[name]]
    loss = loss_fn(model(x), y)
    g_exact = torch.cat(
        [g.reshape(-1) for g in torch.autograd.grad(loss, params)]
    ).detach()

    cur = ParamBlockEstimator(model, loss_fn).curvature(x, y)

    assert not cur.flat_grad.requires_grad, "flat_grad must be detached"
    assert cur.flat_grad.numel() == total, "flat_grad must span all parameters"
    assert torch.allclose(cur.flat_grad, g_exact, atol=1e-6), "flat_grad mismatch"

    for seed in range(3):
        gen = torch.Generator().manual_seed(100 + seed)
        v = torch.randn(total, generator=gen)
        assert torch.allclose(cur.hvp(v), hess @ v, rtol=1e-3, atol=1e-4), (
            f"hvp(v) != H v (probe {seed})"
        )
    print("  curvature: flat_grad == grad_theta L, hvp == dense H action - OK")


def test_curvature_coupling_matches_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(4)

    model = _TinyGroupedMLP().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    hess, ranges = _exact_param_hessian(model, loss_fn, x, y)
    names = list(model.get_param_groups().keys())
    expected_pairs = [(v, w) for i, v in enumerate(names) for w in names[i + 1 :]]

    est = ParamBlockEstimator(model, loss_fn, n_probes=8000)
    coupling = est.curvature(x, y).coupling()

    assert list(coupling.keys()) == expected_pairs, (
        f"coupling pairs/order wrong: {list(coupling.keys())} != {expected_pairs}"
    )
    for v, w in expected_pairs:
        r_vv = _block_frob(hess, ranges, v, v)
        r_ww = _block_frob(hess, ranges, w, w)
        c_exact = _block_frob(hess, ranges, v, w) / (r_vv * r_ww) ** 0.5
        rel = abs(c_exact - coupling[(v, w)]) / (c_exact + 1e-12)
        assert rel < 0.12, (
            f"C({v},{w}) mismatch: exact={c_exact:.4f} "
            f"approx={coupling[(v, w)]:.4f} rel={rel:.3f}"
        )
        print(f"  C({v},{w}) exact={c_exact:.4f} approx={coupling[(v, w)]:.4f} - OK")


# ------------------------------------------------------------------
# 5. Phi report bundle (Phi + diagonal norms + coupling field) vs exact
# ------------------------------------------------------------------


def test_phi_report_matches_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(5)

    model = _TinyGroupedMLP().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    hess, ranges = _exact_param_hessian(model, loss_fn, x, y)
    names = list(model.get_param_groups().keys())
    expected_pairs = [(v, w) for i, v in enumerate(names) for w in names[i + 1 :]]

    est = ParamBlockEstimator(model, loss_fn, n_probes=8000)
    report = est.estimate_phi_report(x, y)

    # (a) scalar Phi consistent with the dedicated estimator and the exact Hessian
    phi_exact = _exact_phi(hess, ranges, names)
    assert abs(phi_exact - report.phi) < 0.04, (
        f"Phi mismatch: exact={phi_exact:.4f} report={report.phi:.4f}"
    )

    # (b) diagonal norms: keys in group-major order, each within tolerance
    assert list(report.diag_frob.keys()) == names, (
        f"diag_frob order wrong: {list(report.diag_frob.keys())} != {names}"
    )
    for v in names:
        exact = _block_frob(hess, ranges, v, v)
        rel = abs(exact - report.diag_frob[v]) / (exact + 1e-12)
        assert rel < 0.1, (
            f"||H_{v},{v}||_F mismatch: exact={exact:.4f} "
            f"report={report.diag_frob[v]:.4f} rel={rel:.3f}"
        )

    # (c) coupling field: same pairs/order as `curvature`, each within tolerance
    assert list(report.coupling.keys()) == expected_pairs, (
        f"coupling pairs/order wrong: {list(report.coupling.keys())} != {expected_pairs}"
    )
    for v, w in expected_pairs:
        r_vv = _block_frob(hess, ranges, v, v)
        r_ww = _block_frob(hess, ranges, w, w)
        c_exact = _block_frob(hess, ranges, v, w) / (r_vv * r_ww) ** 0.5
        rel = abs(c_exact - report.coupling[(v, w)]) / (c_exact + 1e-12)
        assert rel < 0.12, (
            f"C({v},{w}) mismatch: exact={c_exact:.4f} "
            f"report={report.coupling[(v, w)]:.4f} rel={rel:.3f}"
        )
    print(
        f"  Phi report: phi={report.phi:.4f} "
        f"({len(report.diag_frob)} blocks, {len(report.coupling)} pairs) - OK"
    )


# ------------------------------------------------------------------
