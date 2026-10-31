"""
Validates the curvlinops-backed K-FAC adapter (`CurvlinopsKFACProvider`).

The adapter is the one seam binding a real K-FAC library, so these checks guarantee we
drive that library correctly, that the direction it yields is the damped block-diagonal
Newton step, and - the reason for the curvlinops backend - that the *whole* network is
evaluated, including the skip connections and BatchNorm that a graph-tracing K-FAC
cannot follow:

  - exact: for a bias-free single linear layer + MSE - where K-FAC *is* the exact
                GGN - base_direction converges to the exact GGN Newton step -(GGN)^{-1} g
                as the factored Tikhonov damping vanishes, i.e. the library really
                computes K-FAC;
  - agnostic: for a residual MLP with skip connections and a custom forward,
                base_direction is finite and curvature-modified on every layer
                (the architecture a graph-tracing K-FAC rejects);
  - fallback: for a conv + BatchNorm + skip network, the unsupported BatchNorm affine
                weights take the identity step -g while the conv / head get K-FAC;
  - layout: block_ranges partition the flat parameter vector by group;
  - contract: base_direction before update raises;
  - end-to-end: the provider plugs into CoupleFacOptimizer and the no-harm guarantee
                holds with the real K-FAC backend;
  - amortized: both providers rebuild their curvature only on the updates with
                calls % refresh_period == 0 and count them in ``n_refreshes``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from secondorder import CoupleFacOptimizer  # noqa: E402
from secondorder.kfac_adapter import (  # noqa: E402
    CurvlinopsEKFACProvider,
    CurvlinopsKFACProvider,
)

_DTYPE = torch.float64


class _SingleLinear(nn.Module):
    """One bias-free linear layer - the case where K-FAC equals the exact GGN."""

    def __init__(self, d_in: int = 4, d_out: int = 3) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"lin": list(self.lin.parameters())}


class _ResidualMLP(nn.Module):
    """Three linear layers with residual skips and a custom forward.

    The skip connections (tensor ``+``) and the branching forward are exactly what a
    graph-tracing K-FAC cannot follow; curvlinops collects per-layer factors with local
    hooks, so every layer is still evaluated. The layers form the parameter groups.
    """

    def __init__(self, d: int = 5, d_out: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.head = nn.Linear(d, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x)) + x  # residual skip
        h = torch.tanh(self.fc2(h)) + h  # residual skip
        return self.head(h)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "fc1": list(self.fc1.parameters()),
            "fc2": list(self.fc2.parameters()),
            "head": list(self.head.parameters()),
        }


class _ConvBNNet(nn.Module):
    """Conv + BatchNorm + skip + linear head with a custom forward.

    BatchNorm affine weights are unsupported by K-FAC and must receive the identity
    step; the convolution and head are K-FAC-able. Used in eval mode so BatchNorm
    relies on running statistics during factor collection. The convolution and head
    are bias-free so each supported group is a single weight, with the BatchNorm
    affine pair isolated in its own group for the fallback assertion.
    """

    def __init__(self, c: int = 3, h: int = 4, n_cls: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.head = nn.Linear(c * h * h, n_cls, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = self.bn(z) + z  # skip across BatchNorm
        return self.head(z.flatten(1))

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "conv": list(self.conv.parameters()),  # bias-free -> [conv.weight]
            "bn": list(self.bn.parameters()),  # affine -> [bn.weight, bn.bias]
            "head": list(self.head.parameters()),  # bias-free -> [head.weight]
        }


def _params_in_order(groups: dict[str, list[nn.Parameter]]) -> list[nn.Parameter]:
    return [p for name in groups for p in groups[name]]


def _flat_grad(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    groups: dict[str, list[nn.Parameter]],
) -> torch.Tensor:
    """g = grad_theta L flattened in the group-major, parameter-major layout."""
    params = _params_in_order(groups)
    grads = torch.autograd.grad(loss_fn(model(x), y), params)
    return torch.cat([g.reshape(-1) for g in grads])


def _exact_param_hessian(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    params: list[nn.Parameter],
) -> torch.Tensor:
    """Exact P x P parameter Hessian via double backward (= GGN for linear + MSE)."""
    g = torch.autograd.grad(loss_fn(model(x), y), params, create_graph=True)
    gflat = torch.cat([gi.reshape(-1) for gi in g])
    p_dim = gflat.numel()
    h = torch.zeros(p_dim, p_dim, dtype=_DTYPE)
    for i in range(p_dim):
        row = torch.autograd.grad(gflat[i], params, retain_graph=True)
        h[i] = torch.cat([r.reshape(-1) for r in row])
    return h.detach()


def _ranges_from_groups(
    groups: dict[str, list[nn.Parameter]],
) -> dict[str, slice]:
    ranges: dict[str, slice] = {}
    offset = 0
    for name, params in groups.items():
        size = sum(p.numel() for p in params)
        ranges[name] = slice(offset, offset + size)
        offset += size
    return ranges


def _coupling_from_h(
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


def test_single_linear_kfac_is_exact_ggn() -> None:
    """Bias-free linear + MSE: as lambda -> 0, base_direction -> the exact GGN step.

    K-FAC factorises the per-layer GGN as G (x) A; for a single linear layer that
    factorisation is exact (Martens & Grosse, 2015), so the block-diagonal K-FAC step
    converges to the exact GGN Newton step -(GGN)^{-1} g as the Tikhonov damping
    vanishes. The provider damps each Kronecker factor separately,
    (G + lambda I)^{-1} (x) (A + lambda I)^{-1}, which differs from the joint
    (GGN + lambda I)^{-1} only at finite lambda (by O(lambda)); the identity is therefore
    the lambda -> 0 limit, asserted here on a well-conditioned problem (cond(GGN) ~ 10).
    """
    torch.manual_seed(0)
    lam = 1e-8
    model = _SingleLinear(d_in=4, d_out=3).double()
    loss_fn = nn.MSELoss()
    x = torch.randn(6, 4, dtype=_DTYPE)
    t = torch.randn(6, 3, dtype=_DTYPE)
    groups = model.get_param_groups()
    params = _params_in_order(groups)

    g = _flat_grad(model, loss_fn, x, t, groups)
    ggn = _exact_param_hessian(model, loss_fn, x, t, params)
    expected = torch.linalg.solve(ggn, -g)  # exact GGN Newton step (undamped)

    provider = CurvlinopsKFACProvider(model, loss_fn, damping=lam).update(x, t)
    d0 = provider.base_direction(g)

    err = (d0 - expected).abs().max().item()
    assert torch.allclose(d0, expected, atol=1e-6), (
        f"K-FAC != exact GGN in the lambda->0 limit (max err {err})"
    )


def test_residual_mlp_is_architecture_agnostic() -> None:
    """Residual MLP (skips + custom forward): every layer yields a finite direction."""
    torch.manual_seed(0)
    model = _ResidualMLP(d=5, d_out=3).double()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(8, 5, dtype=_DTYPE)
    y = torch.randint(0, 3, (8,))
    groups = model.get_param_groups()

    g = _flat_grad(model, loss_fn, x, y, groups)
    provider = CurvlinopsKFACProvider(model, loss_fn, damping=1e-1).update(x, y)
    d0 = provider.base_direction(g)

    assert d0.shape == g.shape, "direction must match the flat parameter layout"
    assert torch.isfinite(d0).all(), "every layer must yield a finite direction"
    # all groups are supported Linear layers, so K-FAC really acts: d0 != -g
    assert not torch.allclose(d0, -g), "supported layers must be curvature-modified"
    # block_ranges tile [0, P) contiguously
    ranges = provider.block_ranges()
    offset = 0
    for name, ps in groups.items():
        size = sum(p.numel() for p in ps)
        assert ranges[name] == slice(offset, offset + size)
        offset += size
    assert offset == g.numel()


def test_unsupported_params_get_identity_fallback() -> None:
    """Conv + BN + skip: BatchNorm params take the identity step -g, the rest get K-FAC."""
    torch.manual_seed(0)
    model = _ConvBNNet(c=3, h=4, n_cls=2).double().eval()
    loss_fn = nn.MSELoss()
    x = torch.randn(8, 3, 4, 4, dtype=_DTYPE)
    t = torch.randn(8, 2, dtype=_DTYPE)
    groups = model.get_param_groups()

    g = _flat_grad(model, loss_fn, x, t, groups)
    provider = CurvlinopsKFACProvider(model, loss_fn, damping=1e-2).update(x, t)
    d0 = provider.base_direction(g)
    ranges = provider.block_ranges()

    assert torch.isfinite(d0).all(), "direction must be finite"
    bn = ranges["bn"]
    assert torch.allclose(d0[bn], -g[bn], atol=1e-12), (
        "unsupported BatchNorm params must take the identity step -g"
    )
    sup = torch.cat([d0[ranges["conv"]], d0[ranges["head"]]])
    sup_id = torch.cat([-g[ranges["conv"]], -g[ranges["head"]]])
    assert not torch.allclose(sup, sup_id), (
        "supported conv / head must be curvature-modified, not identity"
    )


def test_block_ranges_partition() -> None:
    """block_ranges are contiguous, cover [0, P), and match each group's parameters."""
    model = _ResidualMLP().double()
    provider = CurvlinopsKFACProvider(model, nn.CrossEntropyLoss())
    groups = model.get_param_groups()

    ranges = provider.block_ranges()
    assert list(ranges.keys()) == list(groups.keys())
    offset = 0
    for name, params in groups.items():
        size = sum(p.numel() for p in params)
        assert ranges[name] == slice(offset, offset + size)
        offset += size
    assert offset == sum(p.numel() for p in _params_in_order(groups))


def test_base_direction_requires_update() -> None:
    """base_direction without a prior update is a usage error, not silent garbage."""
    model = _ResidualMLP().double()
    provider = CurvlinopsKFACProvider(model, nn.CrossEntropyLoss())
    raised = False
    try:
        provider.base_direction(torch.zeros(1, dtype=_DTYPE))
    except RuntimeError:
        raised = True
    assert raised, "base_direction must require update() first"


def test_optimizer_with_real_kfac_backend() -> None:
    """End-to-end: the curvlinops provider drives CoupleFacOptimizer; no-harm holds."""
    torch.manual_seed(0)
    lam = 1e-1
    model = _ResidualMLP(d=5, d_out=3).double()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(8, 5, dtype=_DTYPE)
    y = torch.randint(0, 3, (8,))
    groups = model.get_param_groups()
    params = _params_in_order(groups)

    g = _flat_grad(model, loss_fn, x, y, groups)
    h = _exact_param_hessian(model, loss_fn, x, y, params)
    coupling = _coupling_from_h(h, _ranges_from_groups(groups))

    provider = CurvlinopsKFACProvider(model, loss_fn, damping=lam).update(x, y)
    opt = CoupleFacOptimizer(provider=provider, tau=0.0)
    gen = torch.Generator().manual_seed(0)
    d = opt.compute_direction(
        g, lambda u: h @ u, coupling, refresh=True, generator=gen
    ).direction

    d0 = provider.base_direction(g)
    m0 = (g.dot(d0) + 0.5 * d0.dot(h @ d0)).item()
    md = (g.dot(d) + 0.5 * d.dot(h @ d)).item()
    assert md <= m0 + 1e-7, f"no-harm violated with real K-FAC: m(d)={md} m(d0)={m0}"


def test_single_linear_ekfac_is_exact_ggn() -> None:
    """Bias-free linear + MSE: EKFAC's eigenvalue-corrected step -> exact GGN as lambda->0.

    For a single layer the K-FAC factorisation is already exact, so EKFAC's eigenvalue
    correction leaves it exact; with exact (eigenbasis) damping the inverse converges to
    the exact GGN Newton step -(GGN)^{-1} g as lambda -> 0. Also exercises the float64
    eigenbasis path the provider runs on its private double-precision copy of the model.
    """
    torch.manual_seed(0)
    lam = 1e-8
    model = _SingleLinear(d_in=4, d_out=3).double()
    loss_fn = nn.MSELoss()
    x = torch.randn(6, 4, dtype=_DTYPE)
    t = torch.randn(6, 3, dtype=_DTYPE)
    groups = model.get_param_groups()
    params = _params_in_order(groups)

    g = _flat_grad(model, loss_fn, x, t, groups)
    ggn = _exact_param_hessian(model, loss_fn, x, t, params)
    expected = torch.linalg.solve(ggn, -g)  # exact GGN Newton step (undamped)

    provider = CurvlinopsEKFACProvider(model, loss_fn, damping=lam).update(x, t)
    d0 = provider.base_direction(g)

    err = (d0 - expected).abs().max().item()
    assert torch.allclose(d0, expected, atol=1e-6), (
        f"EKFAC != exact GGN in the lambda->0 limit (max err {err})"
    )


def test_ekfac_unsupported_params_get_identity_fallback() -> None:
    """Conv + BN + skip: EKFAC acts on conv / head, the BatchNorm affines take -g.

    Validates the EKFAC wiring on a convolutional net with an unsupported BatchNorm. The
    provider builds its eigenbasis on the private float64 copy - the precision that keeps
    the mandatory exact-damping eigendecomposition convergent on the (rank-deficient) conv
    factors of a real network - and routes the identity step -g to the norm affines.
    """
    torch.manual_seed(0)
    model = _ConvBNNet(c=3, h=4, n_cls=2).double().eval()
    loss_fn = nn.MSELoss()
    x = torch.randn(8, 3, 4, 4, dtype=_DTYPE)
    t = torch.randn(8, 2, dtype=_DTYPE)
    groups = model.get_param_groups()

    g = _flat_grad(model, loss_fn, x, t, groups)
    provider = CurvlinopsEKFACProvider(model, loss_fn, damping=1e-2).update(x, t)
    d0 = provider.base_direction(g)
    ranges = provider.block_ranges()

    assert torch.isfinite(d0).all(), "EKFAC direction must be finite on the conv net"
    bn = ranges["bn"]
    assert torch.allclose(d0[bn], -g[bn], atol=1e-12), (
        "unsupported BatchNorm params must take the identity step -g"
    )
    sup = torch.cat([d0[ranges["conv"]], d0[ranges["head"]]])
    sup_id = torch.cat([-g[ranges["conv"]], -g[ranges["head"]]])
    assert not torch.allclose(sup, sup_id), (
        "supported conv / head must be curvature-modified, not identity"
    )


def test_ekfac_base_direction_requires_update() -> None:
    """EKFAC base_direction without a prior update is a usage error, not silent garbage."""
    model = _ResidualMLP().double()
    provider = CurvlinopsEKFACProvider(model, nn.CrossEntropyLoss())
    raised = False
    try:
        provider.base_direction(torch.zeros(1, dtype=_DTYPE))
    except RuntimeError:
        raised = True
    assert raised, "EKFAC base_direction must require update() first"


def test_ekfac_eval_collection_handles_train_mode() -> None:
    """A live model in train mode is fine: factors are collected on the eval double copy.

    The provider builds its eigenbasis on a private eval-mode float64 copy, so a model left
    in train mode (BatchNorm in batch-statistic mode) does not destabilise the
    eigendecomposition; conv / head get EKFAC, the BatchNorm affines the identity step.
    """
    torch.manual_seed(0)
    model = (
        _ConvBNNet(c=3, h=4, n_cls=2).double().train()
    )  # live model left in TRAIN mode
    loss_fn = nn.MSELoss()
    x = torch.randn(8, 3, 4, 4, dtype=_DTYPE)
    t = torch.randn(8, 2, dtype=_DTYPE)
    groups = model.get_param_groups()

    g = _flat_grad(model, loss_fn, x, t, groups)
    provider = CurvlinopsEKFACProvider(model, loss_fn, damping=1e-2).update(x, t)
    d0 = provider.base_direction(g)
    ranges = provider.block_ranges()

    assert torch.isfinite(d0).all(), (
        "EKFAC must stay finite for a train-mode live model"
    )
    assert torch.allclose(d0[ranges["bn"]], -g[ranges["bn"]], atol=1e-12), (
        "BatchNorm affines take the identity step"
    )


def test_ekfac_collects_in_eval_in_the_param_dtype_and_restores_the_mode() -> None:
    """The refresh collects on the live float32 model, eval-switched, and leaves no trace.

    Collecting them instead on a float64 copy of the model would, on a Turing card
    (1/32 rate double precision), turn the type-2 passes - two per refresh, one
    backward per class each - into hours, making the run look hung. The live model is
    used in its own dtype instead - eval-switched so BatchNorm's running statistics are
    read, not updated - and the caller's train mode is restored; only the ``eigh`` runs
    in double, so the direction comes back in the parameter dtype.
    """
    torch.manual_seed(0)
    model = _ConvBNNet(c=3, h=4, n_cls=2).float().train()  # live model in TRAIN mode
    loss_fn = nn.MSELoss()
    x = torch.randn(8, 3, 4, 4)
    t = torch.randn(8, 2)
    groups = model.get_param_groups()
    running_mean = torch.as_tensor(model.bn.running_mean).clone()

    provider = CurvlinopsEKFACProvider(model, loss_fn, damping=1e-2).update(x, t)
    assert model.training, "the caller's mode is restored after the collection"
    assert torch.equal(torch.as_tensor(model.bn.running_mean), running_mean), (
        "the collection pass reads BatchNorm's running statistics, never updates them"
    )

    g = _flat_grad(model, loss_fn, x, t, groups)
    d0 = provider.base_direction(g)
    assert d0.dtype == torch.float32 and torch.isfinite(d0).all()
    ranges = provider.block_ranges()
    assert not torch.allclose(d0[ranges["conv"]], -g[ranges["conv"]]), "curved"
    assert torch.allclose(d0[ranges["bn"]], -g[ranges["bn"]]), "identity fallback"
    assert provider.n_refreshes == 1 and provider.n_refresh_failed == 0


def test_ekfac_amortizes_refresh() -> None:
    """With refresh_period > 1 the eigenbasis is rebuilt only on the scheduled updates."""
    torch.manual_seed(0)
    model = _ResidualMLP(d=5, d_out=3).double()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(8, 5, dtype=_DTYPE)
    y = torch.randint(0, 3, (8,))

    provider = CurvlinopsEKFACProvider(model, loss_fn, damping=1e-1, refresh_period=2)
    provider.update(x, y)
    first = provider._inverse  # built at call 0
    assert provider.n_refreshes == 1
    provider.update(x, y)
    assert provider._inverse is first, (
        "an update within the period reuses the eigenbasis"
    )
    assert provider.n_refreshes == 1
    provider.update(x, y)
    assert provider._inverse is not first, (
        "the scheduled update rebuilds the eigenbasis"
    )
    assert provider.n_refreshes == 2 and provider.n_refresh_failed == 0

    g = _flat_grad(model, loss_fn, x, y, model.get_param_groups())
    assert torch.isfinite(provider.base_direction(g)).all()


def test_kfac_amortizes_refresh() -> None:
    """K-FAC with refresh_period=3: 7 updates rebuild the inverse at calls 0, 3, 6."""
    torch.manual_seed(0)
    model = _ResidualMLP(d=5, d_out=3).double()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(8, 5, dtype=_DTYPE)
    y = torch.randint(0, 3, (8,))
    g = _flat_grad(model, loss_fn, x, y, model.get_param_groups())

    provider = CurvlinopsKFACProvider(model, loss_fn, damping=1e-1, refresh_period=3)
    inverses: list[object] = []
    for _ in range(7):
        provider.update(x, y)
        inverses.append(provider._inverse)
        assert torch.isfinite(provider.base_direction(g)).all()
    assert provider.n_refreshes == 3, "calls 0, 3, 6 are the scheduled refreshes"
    assert inverses[0] is inverses[1] is inverses[2], "reused within the period"
    assert inverses[3] is inverses[4] is inverses[5], "reused within the period"
    assert len({id(op) for op in inverses}) == 3, "one operator per refresh"


def test_ekfac_refresh_count_on_full_rank_mlp() -> None:
    """EKFAC with refresh_period=3 on the full-rank MLP: 7 updates, 3 refreshes, 0 failed."""
    torch.manual_seed(0)
    model = _ResidualMLP(d=5, d_out=3).double()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(8, 5, dtype=_DTYPE)
    y = torch.randint(0, 3, (8,))

    provider = CurvlinopsEKFACProvider(model, loss_fn, damping=1e-1, refresh_period=3)
    for _ in range(7):
        provider.update(x, y)
    assert provider.n_refreshes == 3 and provider.n_refresh_failed == 0
