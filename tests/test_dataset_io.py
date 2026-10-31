"""Explicit local inputs; no dataset fetch is a side effect of a measurement."""

import sys
from types import SimpleNamespace

import pytest

from experiments import data


def test_cifar_uses_configured_root_and_does_not_download(monkeypatch, tmp_path):
    observed = []

    class Dataset:
        def __init__(self, root, *, train, download, transform):
            observed.append((root, train, download))

    monkeypatch.setenv("DAG_HESSE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("DAG_HESSE_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.setattr(data.torchvision.datasets, "CIFAR100", Dataset)
    monkeypatch.setattr(data, "DataLoader", lambda dataset, **kwargs: dataset)
    data.get_cifar100_loaders(2)
    assert observed == [(str(tmp_path), True, False), (str(tmp_path), False, False)]


def test_cars_missing_root_fails_without_a_network_fallback(monkeypatch):
    def forbidden_download(*args, **kwargs):
        raise AssertionError("network fallback was attempted")

    monkeypatch.setitem(
        sys.modules, "kagglehub", SimpleNamespace(dataset_download=forbidden_download)
    )
    monkeypatch.delenv("DAG_HESSE_CARS_ROOT", raising=False)
    monkeypatch.delenv("DAG_HESSE_ALLOW_DOWNLOADS", raising=False)
    with pytest.raises(FileNotFoundError):
        data._cars_root()


@pytest.mark.parametrize("existing", [False, True])
def test_missing_cifar_is_an_error_with_no_downloader(monkeypatch, tmp_path, existing):
    root = tmp_path / "empty-cifar"
    if existing:
        root.mkdir()
    with pytest.raises(RuntimeError):
        data.get_cifar100_loaders(2, data_root=str(root))


@pytest.mark.parametrize("existing", [False, True])
def test_missing_cars_is_an_error_with_no_downloader(monkeypatch, tmp_path, existing):
    root = tmp_path / "empty-cars"
    if existing:
        root.mkdir()
    monkeypatch.setenv("DAG_HESSE_CARS_ROOT", str(root))
    with pytest.raises(FileNotFoundError):
        data.get_stanford_cars_loaders(2, num_workers=0)
