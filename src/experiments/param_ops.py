"""The group-major flat parameter layout and the step primitives keyed to it.

`group_major_params` fixes the coordinate order - `get_param_groups` values,
concatenated group by group - shared by the K-FAC provider's `block_ranges`, the
parameter-space estimator, and the second-order optimizer: a gradient, an HVP, and an
applied direction are each a flat tensor in this order, so they compose
coordinate-for-coordinate. `apply_flat` writes an update into that layout;
`trial_forward` and `trial_loss` evaluate the model and its loss at a displacement
along it without moving the live parameters, for the safeguard's trial evaluations and
the Levenberg-Marquardt actual-reduction ratio.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from hessian import ParamGroupedModel
from secondorder.safeguard import LossAt

# Step size for the damped-Newton family: a full damped-Newton step, the natural unit
# for d = -(H + lambda I)^{-1} g; the no-harm safeguard backstops it.
SECOND_ORDER_LR = 1.0


def group_major_params(model: ParamGroupedModel) -> list[Parameter]:
    """Flat parameter list in `get_param_groups` (group-major) order.

    This is the layout the K-FAC provider's `block_ranges`, the parameter-space
    estimator, and the optimizer all share, so the gradient, the HVP, and the applied
    direction compose coordinate-for-coordinate.
    """
    return [p for params in model.get_param_groups().values() for p in params]


def apply_flat(params: list[Parameter], direction: Tensor, lr: float) -> None:
    """theta <- theta + lr * direction, unflattened group-major over ``params``."""
    with torch.no_grad():
        offset = 0
        for p in params:
            n = p.numel()
            p.add_(lr * direction[offset : offset + n].view_as(p))
            offset += n


def trial_forward(
    model: ParamGroupedModel, params: list[Parameter], lr: float
) -> Callable[[Tensor, Tensor], Tensor]:
    """``forward(direction, x)``: the logits after a trial step, theta untouched.

    A functional forward on ``theta + lr * direction`` (``direction`` flat, group-major
    over ``params``) with copies of the BatchNorm buffers, in the model's current mode:
    the live parameters never move, so the retained double-backward graph of the step
    stays valid, and the buffers never update, so the trajectory is not touched by the
    trial. The module's forward hooks still fire, so a caller can observe the displaced
    network's internals (the activation pattern of a kink count) as well as its output.
    """
    # Parameter names come off the module tree, which `ParamGroupedModel` does not
    # promise; see that protocol for why the concrete type is asserted rather than asked for.
    module = cast(nn.Module, model)
    name_of = {id(p): name for name, p in module.named_parameters()}
    names = [name_of[id(p)] for p in params]  # group-major, as the flat direction

    def forward(direction: Tensor, x: Tensor) -> Tensor:
        trial: dict[str, Tensor] = {}
        offset = 0
        for name, p in zip(names, params, strict=True):
            n = p.numel()
            step = direction[offset : offset + n].view_as(p)
            trial[name] = p.detach() + lr * step
            offset += n
        trial.update({name: b.clone() for name, b in module.named_buffers()})
        return torch.func.functional_call(module, trial, (x,))

    return forward


def trial_loss(
    model: ParamGroupedModel, loss_fn: nn.Module, params: list[Parameter], lr: float
) -> Callable[[Tensor, Tensor], LossAt]:
    """``loss_at(x, y)``: the minibatch loss after a trial step, theta untouched.

    The returned closure evaluates the loss on ``theta + lr * direction`` through
    `trial_forward` (no gradient, no parameter or buffer update). It serves the
    safeguard's loss test and the actual reduction the Levenberg-Marquardt rule needs,
    in both second-order hooks.
    """
    forward = trial_forward(model, params, lr)

    def loss_at(x: Tensor, y: Tensor) -> LossAt:
        def evaluate(direction: Tensor) -> float:
            with torch.no_grad():
                return float(loss_fn(forward(direction, x), y).item())

        return evaluate

    return loss_at
