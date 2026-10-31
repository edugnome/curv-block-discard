"""A model's parameters re-partitioned, for reading Phi at more than one granularity.

Phi is a property of a partition as much as of a Hessian: the same network discards a
different share of its curvature mass when the blocks are stages, residual blocks or
single layers. The measured partition is the one a model's ``get_param_groups`` returns
- the stages a K-FAC provider's blocks line up with - and the granularity a K-FAC
implementation actually factors at is the layer. `Regrouped` presents the same model
under another partition, so one estimator reads Phi along that axis without touching
the models themselves.

For measurement only: a step's optimizer, its ``block_ranges`` and its preconditioner
all assume the model's own group-major layout (``ParamGroupedModel.get_param_groups``),
and this adapter changes it - never build one of those three from a `Regrouped`.
"""

from __future__ import annotations

from typing import Final, cast

import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock, Bottleneck

from hessian import ParamGroupedModel

LEVELS: Final = ("stage", "block", "layer")

_REST = "rest"

# Every residual-block class `experiments.models` builds a backbone from: BasicBlock for
# ResNet-18/34 (and the plain/width variants, which reuse BasicBlock unchanged), Bottleneck
# for ResNet-50. A backbone built from neither must fail loudly - see `_check_partition`.
_BLOCK_CLASSES: Final[tuple[type[nn.Module], ...]] = (BasicBlock, Bottleneck)

# Over this share of trainable parameters in "rest", a level is not measuring this model:
# on every backbone above, "rest" is the stem/head sitting outside every block plus the
# BatchNorm between them, under 2% of trainable parameters at "block" and "layer" on any
# of them - so 50% is a wide margin between that and a leaf class matching nothing, which
# leaves the *entire* model in "rest" (100%, the Bottleneck/ResNet-50 case this guards).
_MAX_REST_SHARE = 0.5


class Regrouped:
    """``model`` under another parameter partition (see `LEVELS`), forward pass unchanged."""

    def __init__(self, model: ParamGroupedModel, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; known levels: {list(LEVELS)}")
        groups = _param_groups(model, level)
        _check_partition(level, groups)
        self._model = model
        self._level = level
        self._groups = groups

    def __call__(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_param_groups(self) -> dict[str, list[Parameter]]:
        return self._groups


def _param_groups(model: ParamGroupedModel, level: str) -> dict[str, list[Parameter]]:
    if level == "stage":
        return model.get_param_groups()
    leaf_classes: tuple[type[nn.Module], ...] = (
        _BLOCK_CLASSES if level == "block" else (nn.Conv2d, nn.Linear)
    )
    # `named_modules` / `parameters` are `nn.Module` behaviour `ParamGroupedModel`
    # does not promise; see `experiments.param_ops.trial_forward` for the same cast.
    return _leaf_module_groups(cast(nn.Module, model), leaf_classes)


def _check_partition(level: str, groups: dict[str, list[Parameter]]) -> None:
    """Rejects a level whose leaf classes did not actually partition this model.

    Raised at construction, not at `get_param_groups()`, so a `Regrouped` never exists in
    a state whose Phi would be an artifact rather than a measurement, and a caller that
    builds one per checkpoint (as `runner_exp8` does) fails at the cheap step rather than
    after `estimate_phi_report`'s Hutchinson probes have run. Two group counts are both
    non-measurements: under 2 (Phi of a one-group partition is 0 by construction, nothing
    is off-diagonal to discard), or a "rest" holding over half of all trainable parameters
    (see `_MAX_REST_SHARE`) - the leaf classes matched too little of the architecture for
    "rest" to be the small remainder it is by design.
    """
    total = sum(p.numel() for ps in groups.values() for p in ps)
    rest = sum(p.numel() for p in groups.get(_REST, []))
    rest_share = rest / total if total else 0.0
    if len(groups) < 2 or rest_share > _MAX_REST_SHARE:
        raise ValueError(
            f"level {level!r} is not a partition of this model: {len(groups)} "
            f"group(s), {rest_share:.1%} of trainable parameters in {_REST!r}"
        )


def _leaf_module_groups(
    model: nn.Module, leaf_classes: tuple[type[nn.Module], ...]
) -> dict[str, list[Parameter]]:
    """One group per ``leaf_classes`` instance (its whole subtree), plus ``"rest"``.

    ``named_modules`` is a pre-order walk, so a leaf-class module is claimed - subtree
    and all, via its own ``.parameters()`` - before any module nested inside it is
    visited; none of the classes used here nest inside their own kind, but claiming
    defensively keeps groups disjoint even if they did. Whatever no such module claims
    (a BatchNorm outside every block, a free tensor) becomes ``"rest"`` - the
    completeness case that keeps this a partition of every trainable parameter rather
    than a silent approximation of one.
    """
    groups: dict[str, list[Parameter]] = {}
    claimed: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, leaf_classes):
            continue
        params = [
            p for p in module.parameters() if p.requires_grad and id(p) not in claimed
        ]
        if params:
            groups[name] = params
            claimed.update(id(p) for p in params)
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in claimed]
    if rest:
        groups[_REST] = rest
    return groups
