"""
Run-level peak-memory bookkeeping around torch's per-device peak counter.

torch keeps one "peak bytes allocated" counter per CUDA device, and the only way to
measure the footprint of a single operation is to reset that counter around it. The
probe-chunk calibration in `param_space` needs exactly that - the transient peak of one
batched double-backward - but a bare reset would silently truncate the run-level peak
the runners report as their memory-budget line. `measure_peak` therefore folds the
counter into a process-level maximum before resetting it, and `run_peak_allocated`
returns that maximum merged with the live counter: the run's true peak, whatever was
reset in between. Both are no-ops off CUDA.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

_run_peak: dict[int, int] = {}  # device index -> peak folded in before a reset
_open: list[tuple[int, list[int]]] = []  # enclosing windows: (device index, peak cell)


def _index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


@contextmanager
def measure_peak(device: torch.device) -> Iterator[list[int]]:
    """Transient peak of the body on ``device``, in bytes, as the yielded 1-element list.

    Folds the current peak into the run-level maximum, resets the counter, and on exit
    stores ``max_memory_allocated - memory_allocated_at_entry`` - what the body needed
    on top of the resident set - in ``result[0]``. Windows nest: the counter is folded
    into every enclosing window before it is reset, so a phase that contains its own
    calibration windows still reports its true peak. Off CUDA the body runs unmeasured
    and the result stays 0.
    """
    result = [0]
    if device.type != "cuda":
        yield result
        return
    index = _index(device)
    counter = torch.cuda.max_memory_allocated(device)
    _run_peak[index] = max(_run_peak.get(index, 0), counter)
    for outer_index, cell in _open:
        if outer_index == index:
            cell[0] = max(cell[0], counter)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    cell = [base]
    _open.append((index, cell))
    try:
        yield result
    finally:
        _open.pop()
        peak = max(cell[0], torch.cuda.max_memory_allocated(device))
        result[0] = max(0, peak - base)


def run_peak_allocated(device: torch.device) -> int:
    """Peak bytes allocated on ``device`` over the run, across `measure_peak` resets."""
    if device.type != "cuda":
        return 0
    return max(
        _run_peak.get(_index(device), 0), torch.cuda.max_memory_allocated(device)
    )
