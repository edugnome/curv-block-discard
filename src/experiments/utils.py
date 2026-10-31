"""General-purpose utilities: seeding and determinism control."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def _force_deterministic() -> None:
    """Switch every backend to its deterministic kernels (no cuDNN autotuning)."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed every RNG; with ``deterministic`` also pin the kernels to deterministic ones.

    Seeding alone fixes model init and minibatch order. The deterministic block on top
    (deterministic algorithms, no cuDNN autotuning) makes the arithmetic bit-reproducible
    at a real throughput cost on the large convolutional runs, so the production profiles
    opt out and instead wrap only their measurements in `deterministic_mode`. The cuBLAS
    workspace pin is always set: it is a prerequisite for deterministic mode and harmless
    otherwise.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        _force_deterministic()


@contextmanager
def deterministic_mode() -> Iterator[None]:
    """Run the body with deterministic kernels, restoring the previous flags on exit.

    Used around the curvature *measurements* (Phi, the payoff metrics) so they are
    reproducible even when training itself runs with the faster non-deterministic
    kernels. The three flags `set_seed` pins - deterministic algorithms, cuDNN
    determinism, cuDNN autotuning - are captured on entry and restored on exit.
    """
    prev_algorithms = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    _force_deterministic()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_algorithms, warn_only=prev_warn_only)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
