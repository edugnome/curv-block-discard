"""`experiments.cli.resolve_device`: ``--gpu`` narrows ``CUDA_VISIBLE_DEVICES``, never overwrites it.

Overwriting ``CUDA_VISIBLE_DEVICES=2`` with ``--gpu 0`` instead of narrowing to it would
land the run on physical GPU 0 - a full, shared card - where K-FAC's Cholesky dies in
``cusolverDnCreate`` (no device memory left for the handle). The tests stub
``torch.cuda.is_available`` to False so they only exercise the environment logic and
never initialise the CUDA runtime under a fake device list (that would pin the runtime
for the rest of the pytest process).
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch

from experiments.cli import resolve_device


@pytest.fixture(autouse=True)
def _no_cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _args(gpu: int | None, device: str = "auto") -> argparse.Namespace:
    return argparse.Namespace(gpu=gpu, device=device)


def test_gpu_pins_the_card_when_nothing_is_restricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert resolve_device(_args(1)) == torch.device("cuda")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_gpu_narrows_a_visible_list_to_the_named_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0, 2")
    assert resolve_device(_args(2)) == torch.device("cuda")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_gpu_outside_the_visible_list_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit):
        resolve_device(_args(0))  # simultaneous index and explicit device
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"  # untouched


def test_without_gpu_flag_the_environment_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert resolve_device(_args(None)) == torch.device("cpu")
    assert resolve_device(_args(None, device="cuda")) == torch.device("cuda")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
