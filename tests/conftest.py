"""Every unit test uses synthetic inputs and a fail-closed network boundary."""

from __future__ import annotations

import socket
import sys
import urllib.request
from typing import Any

import pytest


def _block_network(event: str, args: tuple[Any, ...]) -> None:
    if event not in {"socket.connect", "socket.sendto", "urllib.Request"}:
        return
    if (
        event != "urllib.Request"
        and args
        and getattr(args[0], "family", None) == socket.AF_UNIX
    ):
        return  # Torch multiprocessing transfers tensor storage through local IPC.
    raise AssertionError("network/download access is forbidden in synthetic tests")


# Active during test-module collection, before fixtures or scientific imports.
sys.addaudithook(_block_network)


@pytest.fixture(autouse=True)
def offline_test_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Block downloader entry points, including cached Kaggle/model lookups."""
    import torch.hub
    import torchvision.datasets.utils
    import torchvision.models._api

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network/download access is forbidden in synthetic tests")

    original_connect = socket.socket.connect

    def connect(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        return blocked(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(torch.hub, "download_url_to_file", blocked)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", blocked)
    monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", blocked)
    monkeypatch.setattr(torchvision.datasets.utils, "download_url", blocked)
    monkeypatch.setattr(
        torchvision.datasets.utils, "download_file_from_google_drive", blocked
    )
    monkeypatch.setattr(torchvision.datasets.CIFAR10, "download", blocked)
    monkeypatch.setattr(torchvision.datasets.CIFAR100, "download", blocked)
    import kagglehub
    import requests

    monkeypatch.setattr(kagglehub, "dataset_download", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setenv("DAG_HESSE_DATA_ROOT", str(tmp_path / "nonexistent-datasets"))
    monkeypatch.delenv("DAG_HESSE_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("DAG_HESSE_CARS_ROOT", raising=False)
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    for name in ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "KAGGLEHUB_CACHE"):
        monkeypatch.setenv(name, str(tmp_path / "empty-cache" / name))
