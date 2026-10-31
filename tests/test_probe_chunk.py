"""Batched block probes: the calibrated chunk, the OOM back-off, the int8 draw.

`ParamBlockEstimator._frob_sq_batched` (the diagonal / cross-block path) runs its first
probe alone, measures that probe's transient peak (`hessian.memory.measure_peak`) and
sizes the rest of the probes from it against a fixed fraction of the device's *total*
memory - never of the free memory, which let the chunk grow until it filled a shared
card. These tests pin the arithmetic, the call pattern (one calibration call, then
chunks), the OutOfMemoryError halving, the run-level peak bookkeeping around torch's
resettable counter, and the memory-lean Rademacher draw. The estimates themselves are
validated against exact Hessians in `test_param_space`; the full-parameter probe, which
is not batched, is covered by `test_lean_probe`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hessian import memory
from hessian._estimators import _rademacher, rademacher_batch
from hessian.param_space import (
    _PROBE_ELEM_BUDGET,
    _PROBE_MEM_FRACTION,
    ParamBlockEstimator,
)
from tests.test_param_space import (
    _block_frob,
    _exact_param_hessian,
    _make_data,
    _TinyGroupedMLP,
)

_GIB = 2**30
_CPU = torch.device("cpu")


_LOSS = nn.CrossEntropyLoss()


def _estimator(
    n_probes: int,
) -> tuple[ParamBlockEstimator, _TinyGroupedMLP, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = _TinyGroupedMLP().eval()
    x, y = _make_data(device=_CPU)
    return ParamBlockEstimator(model, _LOSS, n_probes=n_probes), model, x, y


# ------------------------------------------------------------------
# chunk arithmetic
# ------------------------------------------------------------------


def test_cuda_chunk_is_the_fraction_of_total_over_the_measured_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    est, _, _, _ = _estimator(n_probes=30)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=24 * _GIB),
    )
    cuda = torch.device("cuda")
    budget = int(24 * _GIB * _PROBE_MEM_FRACTION)
    assert est._probe_chunk(1 * _GIB, 10**6, cuda) == budget // _GIB
    assert est._probe_chunk(budget + 1, 10**6, cuda) == 1, "never below one probe"
    assert est._probe_chunk(1, 10**6, cuda) == 30, "never above n_probes"
    assert est._probe_chunk(0, 10**6, cuda) == 30, "a zero measurement is harmless"


def test_cpu_chunk_uses_the_element_budget() -> None:
    est, _, _, _ = _estimator(n_probes=30)
    assert est._probe_chunk(10**12, _PROBE_ELEM_BUDGET // 4, _CPU) == 4
    assert est._probe_chunk(10**12, 2 * _PROBE_ELEM_BUDGET, _CPU) == 1
    assert est._probe_chunk(10**12, 1, _CPU) == 30


# ------------------------------------------------------------------
# call pattern: calibration probe first, then chunks; OOM halves
# ------------------------------------------------------------------


def _spy_grad(monkeypatch: pytest.MonkeyPatch, fail_above: int | None = None) -> list:
    """Record the probe count of every batched double-backward; optionally OOM above."""
    real = torch.autograd.grad
    seen: list[int] = []

    def spy(*args, **kwargs):
        if kwargs.get("is_grads_batched"):  # the probe calls, not the graph build
            m = kwargs["grad_outputs"].shape[0]
            if fail_above is not None and m > fail_above:
                raise torch.cuda.OutOfMemoryError("simulated")
            seen.append(m)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", spy)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return seen


def test_first_probe_calibrates_then_the_rest_is_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    est, _, x, y = _estimator(n_probes=8)
    seen = _spy_grad(monkeypatch)
    est.estimate_diag_frob_sq(x, y, "fc1")
    assert seen == [1, 7], "one calibration probe, the remaining probes in one chunk"


def test_oom_halves_the_chunk_and_the_estimate_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    est, model, x, y = _estimator(n_probes=8000)
    seen = _spy_grad(monkeypatch, fail_above=2048)
    torch.manual_seed(1)
    approx = est.estimate_diag_frob_sq(x, y, "fc1")
    assert seen[0] == 1 and max(seen[1:]) <= 2048 and sum(seen) == 8000
    hess, ranges = _exact_param_hessian(model, _LOSS, x, y)
    exact = _block_frob(hess, ranges, "fc1", "fc1") ** 2
    assert abs(exact - approx) / exact < 0.1


# ------------------------------------------------------------------
# run-level peak across the calibration resets
# ------------------------------------------------------------------


def test_measure_peak_is_a_no_op_off_cuda() -> None:
    with memory.measure_peak(_CPU) as peak:
        torch.zeros(1000)
    assert peak == [0]
    assert memory.run_peak_allocated(_CPU) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_measure_peak_reports_the_body_and_keeps_the_run_peak() -> None:
    device = torch.device("cuda")
    big = torch.empty(64 * 2**20, dtype=torch.uint8, device=device)  # 64 MiB resident
    torch.cuda.synchronize(device)
    before = torch.cuda.max_memory_allocated(device)
    with memory.measure_peak(device) as peak:
        torch.empty(16 * 2**20, dtype=torch.uint8, device=device)  # 16 MiB transient
    assert 16 * 2**20 <= peak[0] < 17 * 2**20, "the body's own footprint, not the base"
    assert memory.run_peak_allocated(device) >= before, "the reset lost nothing"
    del big


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_measure_peak_windows_nest() -> None:
    device = torch.device("cuda")
    with memory.measure_peak(device) as outer:
        with memory.measure_peak(device) as inner:
            torch.empty(48 * 2**20, dtype=torch.uint8, device=device)
        torch.empty(16 * 2**20, dtype=torch.uint8, device=device)
    assert 48 * 2**20 <= inner[0] < 49 * 2**20
    assert outer[0] >= inner[0], "the inner reset must not blind the outer window"


# ------------------------------------------------------------------
# the Rademacher draw
# ------------------------------------------------------------------


def test_rademacher_draw_is_float_signs_from_a_seeded_generator() -> None:
    gen = torch.Generator().manual_seed(7)
    z = rademacher_batch(4, 1000, _CPU, gen)
    assert z.dtype == torch.float32 and z.shape == (4, 1000)
    assert set(z.unique().tolist()) == {-1.0, 1.0}
    assert abs(z.mean().item()) < 0.1, "unbiased signs"
    again = rademacher_batch(4, 1000, _CPU, torch.Generator().manual_seed(7))
    assert torch.equal(z, again), "reproducible under the generator"
    single = _rademacher(1000, _CPU, torch.Generator().manual_seed(7))
    assert single.shape == (1000,) and set(single.unique().tolist()) == {-1.0, 1.0}
