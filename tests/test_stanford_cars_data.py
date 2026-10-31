"""Stanford Cars loader internals: stratified split + lazy image/label dataset.

The networked Kaggle download and `.mat` parsing are exercised at runtime; here we lock
the offline-testable logic - the deterministic per-class split and the lazy image
dataset - without touching the network or Kaggle credentials.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from experiments.data import _StanfordCarsSplit, _stratified_split  # noqa: E402


def test_stratified_split_deterministic_and_covers_classes() -> None:
    labels = [c for c in range(5) for _ in range(8)]  # 5 classes x 8 samples
    train_idx, val_idx = _stratified_split(labels, val_fraction=0.25, seed=0)

    # disjoint + exhaustive partition of all indices
    assert set(train_idx).isdisjoint(val_idx), "splits overlap"
    assert sorted(train_idx + val_idx) == list(range(len(labels))), "not a partition"

    # stratification: every class present in both splits
    train_classes = {labels[i] for i in train_idx}
    val_classes = {labels[i] for i in val_idx}
    assert train_classes == set(range(5)), f"train missing classes: {train_classes}"
    assert val_classes == set(range(5)), f"val missing classes: {val_classes}"

    # per-class val size = ceil(0.25 * 8) = 2
    for c in range(5):
        n_val_c = sum(1 for i in val_idx if labels[i] == c)
        assert n_val_c == 2, f"class {c}: expected 2 val, got {n_val_c}"

    # determinism: same seed -> identical split; different seed -> different split
    assert _stratified_split(labels, 0.25, 0) == (
        train_idx,
        val_idx,
    ), "non-deterministic"
    assert _stratified_split(labels, 0.25, 1) != (train_idx, val_idx), "seed ignored"


def test_stratified_split_singleton_class_stays_in_train() -> None:
    labels = [0, 0, 0, 0, 1]  # class 1 has a single sample
    train_idx, val_idx = _stratified_split(labels, val_fraction=0.5, seed=0)
    assert 4 in train_idx and 4 not in val_idx, "singleton class must stay in train"
    assert all(labels[i] == 0 for i in val_idx), "only class 0 can populate val here"
    assert len(val_idx) >= 1, "majority class must still contribute a val sample"


def test_stanford_cars_split_loads_image_and_label() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "00001.jpg"
        Image.new("RGB", (40, 30), color=(123, 50, 200)).save(img_path)
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])

        ds = _StanfordCarsSplit([(str(img_path), 7)], transform)
        assert len(ds) == 1, "dataset length mismatch"
        image, label = ds[0]
        assert isinstance(image, torch.Tensor), "image must be a tensor"
        assert image.shape == (3, 32, 32), f"bad image shape: {tuple(image.shape)}"
        assert label == 7, "label mismatch"


def test_cars_root_env_override(monkeypatch, tmp_path) -> None:
    """DAG_HESSE_CARS_ROOT bypasses kagglehub; a non-directory value is an error."""
    from experiments.data import _CARS_ROOT_ENV, _cars_root

    monkeypatch.setenv(_CARS_ROOT_ENV, str(tmp_path))
    assert _cars_root() == str(tmp_path)
    monkeypatch.setenv(_CARS_ROOT_ENV, str(tmp_path / "missing"))
    try:
        _cars_root()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing root must raise")
