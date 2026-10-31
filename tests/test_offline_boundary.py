"""Exercise download guards without attempting a real network connection."""

import os
import subprocess
import sys
from pathlib import Path

import kagglehub
import pytest
import requests
import torch.hub
import torchvision.datasets.utils
import torchvision.models._api


@pytest.mark.parametrize("event", ["socket.connect", "socket.sendto", "urllib.Request"])
def test_network_audit_boundary_is_active(event):
    with pytest.raises(AssertionError):
        sys.audit(event, None)


def test_boundary_is_active_before_fixtures_or_test_collection(tmp_path):
    conftest = Path(__file__).with_name("conftest.py")
    command = (
        "import runpy, sys; "
        f"runpy.run_path({str(conftest)!r}); "
        "sys.audit('socket.connect', None)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        (kagglehub, "dataset_download"),
        (requests.sessions.Session, "request"),
        (torch.hub, "download_url_to_file"),
        (torch.hub, "load_state_dict_from_url"),
        (torchvision.models._api, "load_state_dict_from_url"),
        (torchvision.datasets.utils, "download_url"),
        (torchvision.datasets.utils, "download_file_from_google_drive"),
        (torchvision.datasets.CIFAR10, "download"),
        (torchvision.datasets.CIFAR100, "download"),
    ],
)
def test_download_entry_points_are_replaced(owner, name):
    with pytest.raises(AssertionError):
        getattr(owner, name)("unused")


def test_dataset_and_model_caches_are_fresh_missing_paths():
    assert not Path(os.environ["DAG_HESSE_DATA_ROOT"]).exists()
    assert "DAG_HESSE_CARS_ROOT" not in os.environ
    for name in ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "KAGGLEHUB_CACHE"):
        assert not Path(os.environ[name]).exists()
