"""Pretrained models use local weights unless downloads are explicitly enabled."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torchvision.models as tv_models

from experiments.models import SegmentedResNet50


@pytest.fixture
def cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(root))
    monkeypatch.delenv("DAG_HESSE_ALLOW_DOWNLOADS", raising=False)
    return root


@pytest.fixture
def backbone(monkeypatch: pytest.MonkeyPatch) -> Mock:
    builder = Mock(side_effect=lambda **kwargs: tv_models.resnet18(weights=None))
    monkeypatch.setattr(tv_models, "resnet50", builder)
    return builder


def test_pretrained_requires_local_weights_without_opt_in(
    cache_root: Path, backbone: Mock
) -> None:
    with pytest.raises(FileNotFoundError):
        SegmentedResNet50(num_classes=3)
    backbone.assert_not_called()
    assert not cache_root.exists()


def test_pretrained_uses_existing_cache_without_download_permission(
    cache_root: Path, backbone: Mock
) -> None:
    checkpoint = cache_root / "checkpoints" / "resnet50-11ad3fa6.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    model = SegmentedResNet50(num_classes=3)
    backbone.assert_called_once_with(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    assert model._base.fc.out_features == 3


def test_pretrained_allows_explicit_download_opt_in(
    monkeypatch: pytest.MonkeyPatch, cache_root: Path, backbone: Mock
) -> None:
    monkeypatch.setenv("DAG_HESSE_ALLOW_DOWNLOADS", "1")
    model = SegmentedResNet50(num_classes=3)
    backbone.assert_called_once_with(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    assert model._base.fc.out_features == 3
    assert not cache_root.exists()
