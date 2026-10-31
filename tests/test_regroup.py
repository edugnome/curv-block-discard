"""Re-partitioning a model's parameters for a Phi granularity sweep."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from experiments.models import SegmentedResNet18, SegmentedResNet50
from experiments.regroup import LEVELS, Regrouped


@pytest.fixture
def resnet18() -> SegmentedResNet18:
    return SegmentedResNet18(num_classes=10)


@pytest.fixture
def resnet50() -> SegmentedResNet50:
    return SegmentedResNet50(num_classes=10, pretrained=False)


def _params(groups: dict[str, list[nn.Parameter]]) -> list[int]:
    return sorted(id(p) for ps in groups.values() for p in ps)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_is_a_partition_of_the_trainable_parameters(
    level: str, resnet18: SegmentedResNet18
) -> None:
    groups = Regrouped(resnet18, level).get_param_groups()
    flat = _params(groups)
    assert len(flat) == len(set(flat)), "a parameter appears in two groups"
    expected = sorted(id(p) for p in resnet18.parameters() if p.requires_grad)
    assert flat == expected


def test_stage_reproduces_the_model_s_own_grouping(resnet18: SegmentedResNet18) -> None:
    assert list(Regrouped(resnet18, "stage").get_param_groups()) == list(
        resnet18.get_param_groups()
    )


def test_layer_is_the_kfac_granularity(resnet18: SegmentedResNet18) -> None:
    groups = Regrouped(resnet18, "layer").get_param_groups()
    assert len(groups) >= 20, "ResNet-18 has 20 conv/linear modules"


def test_an_unknown_level_is_rejected(resnet18: SegmentedResNet18) -> None:
    with pytest.raises(ValueError):
        Regrouped(resnet18, "nosuch")


def test_the_adapter_is_transparent_to_the_forward_pass(
    resnet18: SegmentedResNet18,
) -> None:
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(Regrouped(resnet18, "layer")(x), resnet18(x))


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_partitions_a_bottleneck_backbone_into_at_least_two_groups(
    level: str, resnet50: SegmentedResNet50
) -> None:
    """ResNet-50 is built from Bottleneck, not BasicBlock; a leaf-class list that only
    recognises BasicBlock collapses every level but "stage" into one "rest" group -
    trivially complete and disjoint, so the plain partition check above cannot catch it.
    """
    groups = Regrouped(resnet50, level).get_param_groups()
    assert len(groups) >= 2, f"{level!r} did not partition the Bottleneck backbone"
    flat = _params(groups)
    assert len(flat) == len(set(flat)), "a parameter appears in two groups"
    expected = sorted(id(p) for p in resnet50.parameters() if p.requires_grad)
    assert flat == expected


def test_block_on_resnet50_yields_one_group_per_bottleneck(
    resnet50: SegmentedResNet50,
) -> None:
    n_bottlenecks = sum(1 for m in resnet50.modules() if isinstance(m, Bottleneck))
    assert n_bottlenecks == 16, "ResNet-50 is 3+4+6+3 Bottleneck blocks"
    groups = Regrouped(resnet50, "block").get_param_groups()
    assert sum(1 for name in groups if name != "rest") == n_bottlenecks


class _NoResidualBlocks(nn.Module):
    """Two Linear layers, no BasicBlock/Bottleneck: exercises the degenerate-level guard
    without reaching for a real backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"fc1": list(self.fc1.parameters()), "fc2": list(self.fc2.parameters())}


def test_a_level_whose_leaf_class_matches_nothing_is_rejected() -> None:
    with pytest.raises(ValueError):
        Regrouped(_NoResidualBlocks(), "block")
