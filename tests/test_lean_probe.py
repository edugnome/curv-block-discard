"""The lean full-parameter Hutchinson probe (`hessian.param_space._LeanFullProbe`).

The full-parameter probe is the one allocation of the Phi measurement that scales with
the whole model. The batched block probe, applied to all parameters, materialised per
probe the concatenated gradient, the probe and the returned grads - ~3.25 P bytes over
the resident set, 4.7 P on Qwen2.5-0.5B once the vmapped transient is counted - which
put Llama-3.2-1B out of reach of a 24 GB card. `_LeanFullProbe` injects each tensor's
probe slice just in time (a grad_fn pre-hook) and drains each slice of H z as it lands
(a post-accumulate-grad hook), so no P-vector exists at once. These tests pin:

  - the memory model: peak over the resident set <= 1.5 P (CPU allocation timeline;
    CUDA peak counter where available) against the batched probe's >= 3 P;
  - exactness of the mechanism: for given seeds, ||H z||^2 equals the dense Hessian's;
  - the estimate: ||H||_F^2 within Hutchinson tolerance of the exact Frobenius norm;
  - seeding, and that the Phi report's block entries are untouched by the full probe;
  - hygiene: `.grad` restored, hooks removed, constant gradients contribute nothing.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from hessian import memory
from hessian._estimators import rademacher_batch, rademacher_signs
from hessian.param_space import ParamBlockEstimator, _LeanFullProbe
from tests.test_param_space import (
    _exact_param_hessian,
    _make_data,
    _TinyGroupedMLP,
)

_CPU = torch.device("cpu")
_LOSS = nn.CrossEntropyLoss()


class _WideMLP(nn.Module):
    """A grouped MLP wide enough for P-sized tensors to dominate the activation buffers."""

    def __init__(self, d: int = 512, depth: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.head = nn.Linear(d, 10)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.head(x)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {f"fc{i}": list(m.parameters()) for i, m in enumerate(self.layers)}
        groups["head"] = list(self.head.parameters())
        return groups


def _graph(
    model: nn.Module, x: Tensor, y: Tensor
) -> tuple[list[nn.Parameter], list[Tensor]]:
    """g = grad_theta L with create_graph, as `ParamBlockEstimator._grad_graph` builds it."""
    params = list(model.parameters())
    loss = _LOSS(model(x), y)
    return params, list(torch.autograd.grad(loss, params, create_graph=True))


def _batched_full_probe(params: list[nn.Parameter], grads: list[Tensor]) -> float:
    """One probe of the former full-parameter path (`_frob_sq_batched` over all params).

    Kept verbatim as the memory baseline: the concatenated gradient, the (1, P) probe
    and the returned grads are all P-sized and all alive at once.
    """
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    probes = rademacher_batch(1, flat_grad.numel(), flat_grad.device)
    out = torch.autograd.grad(
        flat_grad,
        params,
        grad_outputs=probes,
        retain_graph=True,
        allow_unused=True,
        is_grads_batched=True,
    )
    total = torch.zeros((), device=flat_grad.device)
    for g in out:
        if g is not None:
            total = total + (g**2).sum()
    return total.item()


def _lean_full_probe(
    params: list[nn.Parameter], grads: list[Tensor], seed: int = 0
) -> float:
    with _LeanFullProbe(params, grads, params[0].device) as probe:
        gen = torch.Generator().manual_seed(seed)
        seeds = torch.randint(0, 2**62, (probe.n_slices,), generator=gen).tolist()
        return probe(seeds).item()


def _param_bytes(params: list[nn.Parameter]) -> int:
    return sum(p.numel() * p.element_size() for p in params)


# ------------------------------------------------------------------
# memory model
# ------------------------------------------------------------------


def _cpu_peak_bytes(fn: Callable[[], object]) -> int:
    """Peak bytes ``fn`` allocates over what was resident at entry, on CPU.

    CPU has no peak counter, so the torch profiler's allocation timeline (the machinery
    behind `profile.export_memory_timeline`) is replayed: every CREATE adds its bytes,
    every DESTROY subtracts them, and the running maximum is the transient peak.
    """
    from torch.profiler._memory_profiler import Action

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        fn()
    running = peak = 0
    for _, action, _, nbytes in prof._memory_profile().timeline:
        if action == Action.CREATE:
            running += nbytes
        elif action == Action.DESTROY:
            running -= nbytes
        peak = max(peak, running)
    return peak


def test_full_probe_peak_is_about_one_parameter_vector_on_cpu() -> None:
    torch.manual_seed(0)
    model = _WideMLP()
    x = torch.randn(32, 512)
    y = torch.randint(0, 10, (32,))
    params, grads = _graph(model, x, y)
    p_bytes = _param_bytes(params)

    batched = _cpu_peak_bytes(lambda: _batched_full_probe(params, grads)) / p_bytes
    lean = _cpu_peak_bytes(lambda: _lean_full_probe(params, grads)) / p_bytes
    assert batched >= 3.0, f"baseline should hold cat + probe + grads: {batched:.2f} P"
    assert lean <= 1.5, f"lean probe holds more than one P-vector: {lean:.2f} P"
    assert lean < batched / 2, f"lean {lean:.2f} P vs batched {batched:.2f} P"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_full_probe_peak_is_about_one_parameter_vector_on_cuda() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _WideMLP(d=1024, depth=6).to(device)
    x = torch.randn(64, 1024, device=device)
    y = torch.randint(0, 10, (64,), device=device)
    params, grads = _graph(model, x, y)
    p_bytes = _param_bytes(params)

    def cuda_peak(fn: Callable[[], object]) -> int:
        torch.cuda.synchronize(device)
        with memory.measure_peak(device) as peak:
            fn()
            torch.cuda.synchronize(device)
        return peak[0]

    batched = cuda_peak(lambda: _batched_full_probe(params, grads)) / p_bytes
    lean = cuda_peak(lambda: _lean_full_probe(params, grads)) / p_bytes
    assert batched >= 3.0, f"baseline should hold cat + probe + grads: {batched:.2f} P"
    assert lean <= 1.5, f"lean probe holds more than one P-vector: {lean:.2f} P"
    assert lean < batched / 2, f"lean {lean:.2f} P vs batched {batched:.2f} P"


# ------------------------------------------------------------------
# exactness of the mechanism and of the estimate
# ------------------------------------------------------------------


def test_lean_probe_equals_the_dense_hessian_action_for_given_seeds() -> None:
    torch.manual_seed(1)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    hess, _ = _exact_param_hessian(model, _LOSS, x, y)
    params, grads = _graph(model, x, y)

    seeds = list(range(11, 11 + len(grads)))
    with _LeanFullProbe(params, grads, _CPU) as probe:
        assert probe.n_slices == len(grads)
        got = probe(seeds).item()
        with pytest.raises(ValueError):
            probe(seeds[:-1])

    z = torch.cat(
        [
            rademacher_signs(
                tuple(g.shape), _CPU, torch.Generator().manual_seed(s)
            ).reshape(-1)
            for g, s in zip(grads, seeds, strict=True)
        ]
    )
    want = ((hess @ z) ** 2).sum().item()
    assert got == pytest.approx(want, rel=1e-4)


def test_full_frob_sq_matches_the_exact_frobenius_norm() -> None:
    torch.manual_seed(2)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    hess, _ = _exact_param_hessian(model, _LOSS, x, y)
    exact = (hess**2).sum().item()

    approx = ParamBlockEstimator(model, _LOSS, n_probes=3000).estimate_full_frob_sq(
        x, y
    )
    assert abs(approx - exact) / exact < 0.1, f"exact={exact:.4f} approx={approx:.4f}"


def test_full_probe_does_not_use_the_batched_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(3)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    real = torch.autograd.grad

    def spy(*args: object, **kwargs: object) -> object:
        assert not kwargs.get("is_grads_batched"), "full probe went through vmap"
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.autograd, "grad", spy)
    ParamBlockEstimator(model, _LOSS, n_probes=3).estimate_full_frob_sq(x, y)


# ------------------------------------------------------------------
# seeding and the Phi report
# ------------------------------------------------------------------


def test_a_seed_fixes_the_full_estimate() -> None:
    torch.manual_seed(4)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)

    def run(generator: torch.Generator | None) -> float:
        est = ParamBlockEstimator(model, _LOSS, n_probes=4, generator=generator)
        return est.estimate_full_frob_sq(x, y)

    a = run(torch.Generator().manual_seed(7))
    assert a == run(torch.Generator().manual_seed(7)), "same seed, same estimate"
    assert a != run(torch.Generator().manual_seed(8)), "different seed, different draw"

    torch.manual_seed(9)
    d = run(None)
    torch.manual_seed(9)
    assert d == run(None), "the global RNG path is seedable too"


def test_phi_report_block_entries_are_untouched_by_the_full_probe() -> None:
    torch.manual_seed(5)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    names = list(model.get_param_groups())
    pairs = [(v, w) for i, v in enumerate(names) for w in names[i + 1 :]]
    est = ParamBlockEstimator(model, _LOSS, n_probes=3000)

    torch.manual_seed(6)
    decomposed = est.estimate_phi_report(x, y)  # every pair: exact block decomposition
    torch.manual_seed(6)
    probed = est.estimate_phi_report(
        x, y, coupling_pairs=pairs
    )  # same pairs, full probe

    assert probed.diag_frob == decomposed.diag_frob, "diagonal norms must be unchanged"
    assert probed.coupling == decomposed.coupling, "coupling field must be unchanged"
    assert abs(probed.phi - decomposed.phi) < 0.05, (probed.phi, decomposed.phi)


def test_empty_coupling_pairs_measures_no_pair_and_still_draws_the_full_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``coupling_pairs=[]`` is a subset - the empty one - not ``None``: the block
    decomposition of ``||H||_F^2`` is then missing every off-diagonal term, so Phi must
    fall back to one full-parameter probe exactly as `estimate_phi` does, at the same
    cost (one full probe plus one group-localized probe per diagonal block, never the
    O(groups^2) coupling pairs). The test checks that the coupling field is empty and
    that the full probe is used for the denominator.
    """
    torch.manual_seed(9)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    est = ParamBlockEstimator(model, _LOSS, n_probes=8000)

    real_full_frob_sq = ParamBlockEstimator._full_frob_sq
    calls: list[None] = []

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(None)
        return real_full_frob_sq(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ParamBlockEstimator, "_full_frob_sq", spy)

    torch.manual_seed(10)
    report = est.estimate_phi_report(x, y, coupling_pairs=[])
    assert report.coupling == {}, "an empty subset must measure no coupling pair"
    assert len(calls) == 1, "an incomplete decomposition must draw one full probe"

    torch.manual_seed(10)
    phi_only = est.estimate_phi(x, y)
    assert abs(report.phi - phi_only) < 0.05, (report.phi, phi_only)


# ------------------------------------------------------------------
# hygiene
# ------------------------------------------------------------------


def test_lean_probe_leaves_grads_and_hooks_as_found() -> None:
    torch.manual_seed(7)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    params = list(model.parameters())
    stale = [
        torch.full_like(p, 3.0) if i % 2 == 0 else None for i, p in enumerate(params)
    ]
    for p, g in zip(params, stale, strict=True):
        p.grad = g

    _, grads = _graph(model, x, y)
    _lean_full_probe(params, grads)
    assert all(p.grad is g for p, g in zip(params, stale, strict=True)), (
        ".grad restored"
    )
    assert all(not p._post_accumulate_grad_hooks for p in params), "drain hooks removed"
    assert all(not g._backward_hooks for g in grads), "inject hooks removed"

    for p in params:
        p.grad = None
    _LOSS(model(x), y).backward()
    expect = torch.autograd.grad(_LOSS(model(x), y), params)
    assert all(
        p.grad is not None and torch.allclose(p.grad, e)
        for p, e in zip(params, expect, strict=True)
    ), "a later backward is not intercepted"


def test_constant_gradients_contribute_no_curvature() -> None:
    class _Linear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(4))

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.w

        def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
            return {"w": [self.w]}

    torch.manual_seed(8)
    model = _Linear()
    x = torch.randn(6, 4)
    est = ParamBlockEstimator(model, lambda out, _y: out.sum(), n_probes=3)
    assert est.estimate_full_frob_sq(x, x) == 0.0, "g is constant in w, so H = 0"
