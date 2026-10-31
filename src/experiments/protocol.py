"""The shared per-checkpoint measurement protocol every training runner follows.

`checkpoint_epochs` and `epoch_to_label` fix which epochs of a run are measured and
what to call them; `sample_hessian_batch` fixes the minibatch the curvature is read
off; `mean_std` is the seed reduction a runner's aggregate goes through. This module
provides:

  checkpoint_epochs    - which epochs of a run are measured: the named checkpoints
                          ``init`` / ``mid`` / ``final`` resolve to epoch 0,
                          ``total_epochs // 2`` and ``total_epochs``; a label may
                          also be a decimal integer string, resolved to that epoch
                          directly. Raises ``ValueError`` for an unknown label, an
                          epoch outside ``0..total_epochs``, or two labels that
                          resolve to the same epoch.
  epoch_to_label       - the inverse, for labelling a measured epoch in a result
                          bundle: ``0``, ``total // 2`` and ``total`` give back
                          ``init``, ``mid`` and ``final``; any other epoch is its
                          own decimal string, which keeps the labelling injective.
  sample_hessian_batch - the curvature probe's fixed minibatch: the first
                          ``batch_size`` examples of a (non-shuffling) validation
                          loader, so every architecture, method and seed of a run
                          measures curvature on the same inputs.
  mean_std             - the seed reduction a bundle's leaf values go through; a
                          non-finite input (an undamped kappa = inf, the stable rank
                          of a zero operator) propagates to the mean and leaves the
                          spread undefined (NaN) rather than raising or drowning the
                          run in RuntimeWarnings.
  save_checkpoints     - writes a run's measured checkpoints to
                          ``output_dir/checkpoints/`` and returns their paths relative
                          to ``output_dir``, so a later measurement restores a saved
                          state_dict instead of paying for another training run.
  load_checkpoints     - the other half of that promise: reads back exactly what
                          ``save_checkpoints`` wrote, ``{epoch: state_dict}`` on the
                          CPU, so a later measurement can restore a saved state_dict
                          with no training - not even a train loader - in between.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def checkpoint_epochs(total_epochs: int, labels: list[str]) -> list[int]:
    """Resolves each label to an epoch, in the order given: ``init``/``mid``/``final``
    map to 0, ``total_epochs // 2`` and ``total_epochs``; any other label must be a
    decimal integer string, resolved to that epoch. Raises ``ValueError`` for an
    unknown label, an epoch outside ``0..total_epochs``, or two labels that resolve
    to the same epoch."""
    named = {"init": 0, "mid": total_epochs // 2, "final": total_epochs}
    seen: dict[int, str] = {}
    epochs: list[int] = []
    for label in labels:
        if label in named:
            epoch = named[label]
        elif label.isdigit():
            epoch = int(label)
            if not 0 <= epoch <= total_epochs:
                raise ValueError(
                    f"checkpoint epoch {epoch} is outside 0..{total_epochs}"
                )
        else:
            raise ValueError(f"unknown checkpoint label {label!r}")
        if epoch in seen:
            raise ValueError(
                f"labels {seen[epoch]!r} and {label!r} both resolve to epoch {epoch}"
            )
        seen[epoch] = label
        epochs.append(epoch)
    return epochs


def epoch_to_label(epoch: int, total: int) -> str:
    """The checkpoint name of ``epoch`` out of ``total``: ``init``/``mid``/``final`` for
    0/``total // 2``/``total``, ``str(epoch)`` for anything else - the inverse of
    `checkpoint_epochs` that keeps its labelling injective."""
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    if epoch == total // 2:
        return "mid"
    return str(epoch)


def sample_hessian_batch(
    loader: DataLoader, batch_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """The first ``batch_size`` examples of ``loader``, on ``device`` (a fixed minibatch).

    The validation loaders never shuffle, so the batch is the same for every
    architecture, method, and seed of a run.
    """
    xs, ys = [], []
    n = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        n += x.size(0)
        if n >= batch_size:
            break
    x_cat = torch.cat(xs)[:batch_size].to(device)
    y_cat = torch.cat(ys)[:batch_size].to(device)
    return x_cat, y_cat


def mean_std(values: list[float]) -> dict[str, float]:
    """mean/std over seeds; a non-finite value (kappa = inf, D of a zero operator)
    propagates to the mean and leaves the spread undefined (NaN)."""
    arr = np.asarray(values, dtype=float)
    if np.all(np.isfinite(arr)):
        return {"mean": float(arr.mean()), "std": float(arr.std())}
    with np.errstate(invalid="ignore"):
        mean = float(arr.mean())
    return {"mean": mean, "std": math.nan}


def checkpoint_filename(arch: str, seed: int, epoch: int) -> str:
    """The on-disk name one (arch, seed, epoch) checkpoint is written/read under.

    The one place this naming is spelled out, shared by `save_checkpoints` and
    `load_checkpoints`, so a directory the first writes is always one the second can
    read back without a separate index file to keep in sync.
    """
    return f"{arch}_seed{seed}_ep{epoch}.pt"


def save_checkpoints(
    states: dict[int, dict[str, Tensor]],
    output_dir: Path,
    *,
    arch: str,
    seed: int,
) -> dict[str, str]:
    """Writes a run's checkpoints under ``output_dir/checkpoints/`` and returns the paths.

    `Trainer.train` already holds the state at every measured epoch; writing it is what
    makes a later measurement a measurement rather than a retraining run. The returned
    mapping is epoch -> path relative to ``output_dir``, so a result file stays
    relocatable. `load_checkpoints` reads back exactly what this writes.
    """
    target = output_dir / "checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for epoch, state in sorted(states.items()):
        name = checkpoint_filename(arch, seed, epoch)
        torch.save(state, target / name)
        paths[str(epoch)] = f"checkpoints/{name}"
    return paths


def load_checkpoints(
    checkpoints_dir: Path,
    *,
    arch: str,
    seed: int,
    epochs: list[int],
) -> dict[int, dict[str, Tensor]]:
    """Reads back what `save_checkpoints` wrote: ``{epoch: state_dict}``, on the CPU.

    ``checkpoints_dir`` is a run's own ``output_dir`` - the same directory
    `save_checkpoints` was given, holding ``checkpoints/<arch>_seed<seed>_ep<epoch>.pt``
    (`checkpoint_filename`), not the ``checkpoints/`` subdirectory itself. This is the
    read side of the round trip `save_checkpoints`'s own docstring promises: a later
    measurement restores a saved state_dict instead of paying for another training run,
    with no training - not even a train loader - in between.

    Every file is loaded with ``weights_only=True``: a checkpoint is data (a plain
    mapping of parameter/buffer name to `Tensor` - see `Trainer._snapshot` - never a
    pickled arbitrary object), and `save_checkpoints` never writes anything else, so
    nothing here needs the unrestricted unpickler. Raises `FileNotFoundError`, naming
    the missing path, rather than silently returning fewer checkpoints than asked for:
    a caller that iterates every one of ``epochs`` expecting it present is exactly the
    failure mode a partial, unindexed restore would hide.
    """
    target = checkpoints_dir / "checkpoints"
    states: dict[int, dict[str, Tensor]] = {}
    for epoch in epochs:
        path = target / checkpoint_filename(arch, seed, epoch)
        if not path.is_file():
            raise FileNotFoundError(
                f"no checkpoint for arch={arch!r} seed={seed} epoch={epoch} at "
                f"{path}: restore_checkpoints_dir must be a run's own output_dir - "
                "the same directory --save-checkpoints wrote checkpoints/ under"
            )
        states[epoch] = torch.load(path, map_location="cpu", weights_only=True)
    return states
