"""`experiments.protocol`: the checkpoint schedule, fixed minibatch, and seed
reduction every training runner composes its per-checkpoint measurement out of.

`checkpoint_epochs` and `epoch_to_label` are checked for named and numeric labels,
invalid cases and label/epoch round trips. `sample_hessian_batch` is
checked against a non-shuffling loader - the property every runner's "fixed
minibatch" protocol depends on - and `mean_std`, the seed reduction shared by
exp8, exp9 and exp10, against the non-finite inputs (an undamped kappa = inf, a
NaN spread, an empty run) that a seed's spectrum can produce; the case values are
the ones verified by hand against the naive ``{"mean": arr.mean(), "std":
arr.std()}`` formula. `load_checkpoints` is `save_checkpoints`'s other half - the
round trip a later, training-free measurement depends on - checked here against
what `save_checkpoints` actually writes, and against a checkpoint that is missing.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.protocol import (  # noqa: E402
    checkpoint_epochs,
    epoch_to_label,
    load_checkpoints,
    mean_std,
    sample_hessian_batch,
    save_checkpoints,
)


def test_checkpoint_epochs() -> None:
    assert checkpoint_epochs(30, ["init", "mid", "final"]) == [
        0,
        15,
        30,
    ], "init/mid/final must map to 0, total//2, total"
    assert checkpoint_epochs(0, ["init"]) == [0], (
        "epochs=0 (a2) collapses to the single init checkpoint"
    )


def test_epoch_to_label() -> None:
    assert epoch_to_label(0, 30) == "init"
    assert epoch_to_label(15, 30) == "mid"
    assert epoch_to_label(30, 30) == "final"
    assert epoch_to_label(7, 30) == "7", "any other epoch is its own decimal string"
    # a2: with total=0 the init snapshot (epoch 0) is still labelled "init"
    assert epoch_to_label(0, 0) == "init"


def test_checkpoint_epochs_numeric_labels() -> None:
    assert checkpoint_epochs(300, ["10", "150", "299"]) == [10, 150, 299]
    assert checkpoint_epochs(300, ["0", "300"]) == [0, 300], "the boundary epochs too"


def test_checkpoint_epochs_mixed_named_and_numeric() -> None:
    assert checkpoint_epochs(300, ["init", "25", "final"]) == [0, 25, 300]
    assert checkpoint_epochs(300, ["25", "mid", "final"]) == [
        25,
        150,
        300,
    ], "the order given, not sorted by epoch"


def test_checkpoint_epochs_rejects_an_unknown_label() -> None:
    with pytest.raises(ValueError):
        checkpoint_epochs(300, ["bogus"])


def test_checkpoint_epochs_rejects_an_out_of_range_epoch() -> None:
    with pytest.raises(ValueError):
        checkpoint_epochs(300, ["301"])


def test_checkpoint_epochs_rejects_two_labels_on_the_same_epoch() -> None:
    with pytest.raises(ValueError):
        checkpoint_epochs(300, ["mid", "150"])
    with pytest.raises(ValueError):
        checkpoint_epochs(300, ["0", "init"])


def test_epoch_to_label_round_trips_checkpoint_epochs() -> None:
    # Every epoch checkpoint_epochs can produce comes back as the label that produced
    # it - a named label as itself, a numeric label as str(epoch) - with one exception:
    # a numeric label that names the same epoch a named label would (e.g. "150" at
    # total=300) is indistinguishable, once resolved, from having asked for "mid", and
    # epoch_to_label has only the epoch to go on, so it returns the named form. That is
    # harmless for a single label (one was asked for, not two); asking for both is
    # exactly the duplicate-epoch ValueError above.
    total = 300
    labels = ["init", "mid", "final", "10", "25", "299"]
    epochs = checkpoint_epochs(total, labels)
    for label, epoch in zip(labels, epochs, strict=True):
        expected = label if label in ("init", "mid", "final") else str(epoch)
        assert epoch_to_label(epoch, total) == expected

    for numeric_label, named in (("0", "init"), ("150", "mid"), ("300", "final")):
        (epoch,) = checkpoint_epochs(total, [numeric_label])
        assert epoch_to_label(epoch, total) == named


def test_sample_hessian_batch_takes_first_examples_on_device() -> None:
    xs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    ys = torch.arange(10)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=4, shuffle=False)

    x_batch, y_batch = sample_hessian_batch(loader, 6, torch.device("cpu"))

    assert x_batch.shape == (6, 2) and y_batch.shape == (6,), "batch_size examples"
    assert torch.equal(x_batch, xs[:6]), "the first examples, not a shuffled sample"
    assert torch.equal(y_batch, ys[:6])
    assert x_batch.device == torch.device("cpu") and y_batch.device == torch.device(
        "cpu"
    ), "moved to the requested device"


def test_mean_std_finite() -> None:
    result = mean_std([1.0, 2.0, 3.0])
    assert abs(result["mean"] - 2.0) < 1e-12
    assert abs(result["std"] - 0.816496580927726) < 1e-9

    singleton = mean_std([1.0])
    assert singleton == {"mean": 1.0, "std": 0.0}


def test_mean_std_non_finite_propagates_mean_and_nans_the_spread() -> None:
    cases: list[tuple[list[float], float]] = [
        ([math.inf, 1.0], math.inf),
        ([math.nan, 1.0], math.nan),
        ([math.inf, -math.inf], math.nan),
        ([math.inf, math.inf], math.inf),
    ]
    for values, expected_mean in cases:
        result = mean_std(values)
        if math.isnan(expected_mean):
            assert math.isnan(result["mean"]), f"{values} must give a NaN mean"
        else:
            assert result["mean"] == expected_mean, f"{values} must propagate the mean"
        assert math.isnan(result["std"]), f"{values} must leave the spread undefined"


def test_mean_std_empty() -> None:
    result = mean_std([])
    assert math.isnan(result["mean"]) and math.isnan(result["std"])


def test_checkpoints_land_on_disk_and_are_recorded_relative_to_the_bundle(
    tmp_path: Path,
) -> None:
    states = {0: {"w": torch.ones(3)}, 30: {"w": torch.zeros(3)}}
    rel = save_checkpoints(states, tmp_path, arch="resnet18", seed=42)
    assert rel == {
        "0": "checkpoints/resnet18_seed42_ep0.pt",
        "30": "checkpoints/resnet18_seed42_ep30.pt",
    }
    loaded = torch.load(tmp_path / rel["0"])
    torch.testing.assert_close(loaded["w"], states[0]["w"])


def test_load_checkpoints_reads_back_what_save_checkpoints_wrote(
    tmp_path: Path,
) -> None:
    states = {0: {"w": torch.ones(3)}, 30: {"w": torch.zeros(3)}}
    save_checkpoints(states, tmp_path, arch="resnet18", seed=42)

    loaded = load_checkpoints(tmp_path, arch="resnet18", seed=42, epochs=[0, 30])

    assert set(loaded) == {0, 30}
    torch.testing.assert_close(loaded[0]["w"], states[0]["w"])
    torch.testing.assert_close(loaded[30]["w"], states[30]["w"])
    assert loaded[0]["w"].device == torch.device("cpu"), "restored onto the CPU"


def test_load_checkpoints_epoch_order_follows_the_epochs_argument(
    tmp_path: Path,
) -> None:
    """The returned mapping's key set, not iteration order, is what callers rely on
    (`Exp8Runner.run` does `for ep, state in checkpoints.items()`), but ``epochs`` need
    not be sorted going in - `save_checkpoints` itself sorts on write, this must not
    assume the same on read.
    """
    save_checkpoints(
        {0: {"w": torch.zeros(1)}, 5: {"w": torch.ones(1)}},
        tmp_path,
        arch="resnet18",
        seed=1,
    )
    loaded = load_checkpoints(tmp_path, arch="resnet18", seed=1, epochs=[5, 0])
    assert list(loaded) == [5, 0], "insertion order follows the requested epochs"


def test_load_checkpoints_raises_a_clear_error_for_a_missing_checkpoint(
    tmp_path: Path,
) -> None:
    save_checkpoints({0: {"w": torch.ones(3)}}, tmp_path, arch="resnet18", seed=42)
    with pytest.raises(FileNotFoundError):
        load_checkpoints(tmp_path, arch="resnet18", seed=42, epochs=[0, 30])
