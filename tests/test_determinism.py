"""Split determinism: seeding vs. deterministic kernels, and the Trainer step budget.

`set_seed` fixes the RNGs; whether it also pins every kernel to its deterministic variant
is a separate switch (the production profiles turn it off for throughput and keep
determinism only inside their measurements via `deterministic_mode`). Checked here:

  - deterministic_mode: forces the three determinism flags inside the block and
                         restores every previous value on exit;
  - set_seed: with ``deterministic=False`` leaves those flags untouched;
  - max_steps: the Trainer stops after the step budget, still finishing the
                         bookkeeping of the epoch in which it ran out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.config import TrainingConfig  # noqa: E402
from experiments.data import get_synthetic_classification_loaders  # noqa: E402
from experiments.training import StepCtx, Trainer  # noqa: E402
from experiments.utils import deterministic_mode, set_seed  # noqa: E402

_DEVICE = torch.device("cpu")


def _determinism_flags() -> tuple[bool, bool, bool]:
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def _set_determinism_flags(flags: tuple[bool, bool, bool]) -> None:
    algorithms, cudnn_deterministic, cudnn_benchmark = flags
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def test_deterministic_mode_restores_state() -> None:
    saved = _determinism_flags()
    try:
        _set_determinism_flags((False, False, True))  # the non-deterministic profile
        with deterministic_mode():
            assert _determinism_flags() == (
                True,
                True,
                False,
            ), "inside the block every flag must be deterministic"
        assert _determinism_flags() == (
            False,
            False,
            True,
        ), "exit must restore the previous flags"
    finally:
        _set_determinism_flags(saved)


def test_set_seed_without_deterministic_keeps_flags() -> None:
    saved = _determinism_flags()
    try:
        _set_determinism_flags((False, False, True))
        set_seed(0, deterministic=False)
        assert _determinism_flags() == (
            False,
            False,
            True,
        ), "deterministic=False must only seed, not pin the kernels"
        set_seed(0)
        assert _determinism_flags() == (
            True,
            True,
            False,
        ), "the default still pins every kernel"
    finally:
        _set_determinism_flags(saved)


def test_max_steps_stops_training() -> None:
    torch.manual_seed(0)
    train_loader, _ = get_synthetic_classification_loaders(
        16, in_dim=8, num_classes=3, n_train=64, n_val=16
    )  # 4 minibatches / epoch
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3))
    cfg = TrainingConfig(
        lr=0.1, epochs=3, optimizer="sgd", scheduler="none", max_steps=6
    )
    steps: list[int] = []
    epochs_ended: list[int] = []

    def hook(ctx: StepCtx) -> None:
        steps.append(ctx.step)

    trainer = Trainer(model, cfg, _DEVICE, step_hook=hook)
    trainer.train(train_loader, checkpoint_epochs=[], on_epoch_end=epochs_ended.append)

    assert steps == list(range(6)), f"budget of 6 steps, got {steps}"
    assert trainer.step_count == 6
    assert epochs_ended == [
        1,
        2,
    ], "epoch 2 (where the budget ran out) still reports; epoch 3 never starts"
