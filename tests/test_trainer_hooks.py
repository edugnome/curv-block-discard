"""Trainer OCP seam: injected optimizer + step-hook ownership of the update.

Verifies the three constructor regimes - the built-in optimizer (unchanged default),
an externally injected optimizer, and a step hook that owns the parameter update -
without disturbing the existing experiment runners.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.config import TrainingConfig  # noqa: E402
from experiments.training import StepCtx, Trainer  # noqa: E402

_DEVICE = torch.device("cpu")


def _tiny_setup(seed: int = 0) -> tuple[nn.Module, DataLoader]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    x = torch.randn(12, 4)
    y = torch.randint(0, 3, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)  # 3 minibatches / epoch
    return model, loader


def _flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _cfg() -> TrainingConfig:
    return TrainingConfig(lr=0.1, epochs=1, optimizer="sgd", scheduler="none")


def test_default_optimizer_trains() -> None:
    model, loader = _tiny_setup()
    before = _flat(model).clone()
    trainer = Trainer(model, _cfg(), _DEVICE)
    trainer.train(loader, checkpoint_epochs=[1])
    assert isinstance(trainer._optimizer, torch.optim.SGD), "default must build SGD"
    assert not torch.allclose(before, _flat(model)), "default SGD must update params"


def test_injected_optimizer_is_used() -> None:
    model, loader = _tiny_setup()
    # Inject lr=0: if this optimizer (not a config-built lr=0.1 SGD) drives the step,
    # the parameters stay frozen.
    injected = torch.optim.SGD(model.parameters(), lr=0.0)
    before = _flat(model).clone()
    trainer = Trainer(model, _cfg(), _DEVICE, optimizer=injected)
    trainer.train(loader, checkpoint_epochs=[1])
    assert trainer._optimizer is injected, "injected optimizer must be stored"
    assert torch.allclose(before, _flat(model)), "injected lr=0 must freeze params"


def test_step_hook_owns_update() -> None:
    model, loader = _tiny_setup()
    first_param = next(iter(model.parameters()))
    delta = 0.01
    seen: list[StepCtx] = []

    def hook(ctx: StepCtx) -> None:
        seen.append(ctx)
        with torch.no_grad():
            first_param.add_(delta)

    before_first = first_param.detach().clone()
    trainer = Trainer(model, _cfg(), _DEVICE, step_hook=hook)
    assert trainer._optimizer is None, "no built-in optimizer when a hook owns the step"
    trainer.train(loader, checkpoint_epochs=[1])

    # one hook call per minibatch, monotonic global step, correct context
    assert [c.step for c in seen] == [
        0,
        1,
        2,
    ], f"step sequence: {[c.step for c in seen]}"
    assert all(c.model is model for c in seen), "ctx.model must be the trained model"
    assert all(c.x.shape == (4, 4) and c.y.shape == (4,) for c in seen), (
        "ctx batch shapes wrong"
    )
    # only the hook moved params: first_param shifted by exactly 3 * delta
    assert torch.allclose(first_param.detach(), before_first + 3 * delta), (
        "hook update not applied"
    )
