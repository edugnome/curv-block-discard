"""Training loop with checkpoint saving."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import TrainingConfig


@dataclass(frozen=True)
class StepCtx:
    """Per-optimization-step context passed to a `Trainer` step hook.

    A step hook receives the live model and the current (device-resident) minibatch and
    *owns the parameter update* for that step, replacing the built-in optimizer.step().
    This is the seam through which a second-order scheme such as COUPLE-FAC plugs in: it
    rebuilds the curvature bundle on (x, y), refreshes its K-FAC provider, and applies
    its own accepted direction. `step` is the global minibatch counter, used to amortize
    periodic work (e.g. re-measuring coupling every T steps).
    """

    model: nn.Module
    x: Tensor
    y: Tensor
    step: int
    device: torch.device


class Trainer:
    """Trains a model and saves state_dict at specified epochs."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        step_hook: Callable[[StepCtx], None] | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """loss_fn: training/eval criterion; defaults to `nn.CrossEntropyLoss()`."""
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self._step_hook = step_hook
        self._step = 0

        # Optimizer policy (open for extension): an injected optimizer wins; otherwise
        # build the configured SGD/Adam - unless a step hook owns the update, in which
        # case no built-in optimizer is needed.
        self._optimizer: torch.optim.Optimizer | None
        if optimizer is not None:
            self._optimizer = optimizer
        elif step_hook is None:
            if config.optimizer == "sgd":
                self._optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config.lr,
                    momentum=config.momentum,
                    weight_decay=config.weight_decay,
                )
            else:
                self._optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
        else:
            self._optimizer = None

        self._scheduler = None
        if self._optimizer is not None and config.scheduler == "cosine":
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=config.epochs)

    @property
    def loss_fn(self) -> nn.Module:
        return self._loss_fn

    @property
    def step_count(self) -> int:
        """Optimizer steps taken so far (the global minibatch counter)."""
        return self._step

    def train(
        self,
        train_loader: DataLoader,
        checkpoint_epochs: list[int] | None = None,
        *,
        on_epoch_end: Callable[[int], None] | None = None,
    ) -> dict[int, dict[str, Tensor]]:
        """Trains the model, returns {epoch: state_dict} for the requested epochs.

        checkpoint_epochs: list of epoch numbers at which to save state_dict.
            Default: [0 (init), epochs//2 (mid), epochs-1 (final)].
        on_epoch_end: optional callback invoked with the 1-indexed epoch number after
            each epoch (post scheduler step). Lets a caller record a per-epoch metric -
            e.g. the validation-accuracy curve behind a speed-to-target measurement -
            without materializing a weight snapshot every epoch.

        A ``max_steps`` budget in the config ends training once that many optimizer
        steps have been taken in total: the epoch in which the budget runs out still
        completes its bookkeeping (scheduler, checkpoint, ``on_epoch_end``), and no
        further epoch starts.
        """
        epochs = self._config.epochs
        if checkpoint_epochs is None:
            checkpoint_epochs = [0, epochs // 2, epochs - 1]

        checkpoints: dict[int, dict[str, Tensor]] = {}

        # Checkpoint at init (epoch 0 - before training)
        if 0 in checkpoint_epochs:
            checkpoints[0] = self._snapshot()

        for epoch in range(epochs):
            self._train_epoch(train_loader)
            if self._scheduler is not None:
                self._scheduler.step()

            ep_num = epoch + 1  # 1-indexed for compatibility
            if ep_num in checkpoint_epochs:
                checkpoints[ep_num] = self._snapshot()
            if on_epoch_end is not None:
                on_epoch_end(ep_num)
            if self._budget_exhausted():
                break

        # Final
        if epochs in checkpoint_epochs and epochs not in checkpoints:
            checkpoints[epochs] = self._snapshot()

        return checkpoints

    def restore(self, state_dict: dict[str, Tensor]) -> None:
        """Restores the model from a snapshot."""
        self._model.load_state_dict(state_dict)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Computes accuracy and loss on the given loader."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y in loader:
            x, y = x.to(self._device), y.to(self._device)
            logits = self._model(x)
            loss = self._loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        return {
            "test_acc": correct / max(n, 1),
            "test_loss": total_loss / max(n, 1),
        }

    def _budget_exhausted(self) -> bool:
        """True once the optional ``max_steps`` budget has been spent."""
        max_steps = self._config.max_steps
        return max_steps is not None and self._step >= max_steps

    def _train_epoch(self, loader: DataLoader) -> None:
        self._model.train()
        for x, y in loader:
            if self._budget_exhausted():
                return
            x, y = x.to(self._device), y.to(self._device)
            if self._step_hook is not None:
                # The hook owns this step's forward/backward and parameter update.
                self._step_hook(
                    StepCtx(
                        model=self._model,
                        x=x,
                        y=y,
                        step=self._step,
                        device=self._device,
                    )
                )
            else:
                assert self._optimizer is not None
                self._optimizer.zero_grad()
                logits = self._model(x)
                loss = self._loss_fn(logits, y)
                loss.backward()
                clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self._optimizer.step()
            self._step += 1

    def _snapshot(self) -> dict[str, Tensor]:
        return {
            k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()
        }
