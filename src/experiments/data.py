"""Common data loading utilities."""

from __future__ import annotations

import math
import os
import random
from typing import Any

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Class count per dataset, keyed by the ``--dataset`` CLI choice exp7 and exp8 share;
# lets a dataset override keep the classifier head consistent with the selected data.
DATASET_NUM_CLASSES = {
    "stanford_cars": 196,
    "cifar10": 10,
    "cifar100": 100,
    "imagenet32": 1001,  # HF mirror keeps the index-0 background class
    "synthetic": 10,  # low-dim synthetic task for the full-rank EKFAC control (b1_mlp)
}


def get_synthetic_classification_loaders(
    batch_size: int,
    *,
    in_dim: int = 64,
    num_classes: int = 10,
    n_train: int = 8192,
    n_val: int = 2048,
    label_noise: float = 0.15,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Deterministic low-dimensional synthetic classification (full-rank curvature).

    A fixed two-layer random teacher labels low-dimensional Gaussian inputs, with a
    fraction ``label_noise`` of labels randomised. The low input dimension keeps every
    Kronecker factor full rank at a modest batch, and the irreducible label noise gives the
    loss a positive floor so the output-gradient covariance does not collapse as the model
    fits (which would make EKFAC's eigendecomposition degenerate) - together the clean,
    well-conditioned setting for the optimizer-agnostic comparison (`Exp7Config.b1_mlp`).
    Fully reproducible and dependency-free (no download).
    """
    gen = torch.Generator().manual_seed(seed)
    hidden = 2 * in_dim
    w1 = torch.randn(in_dim, hidden, generator=gen)
    w2 = torch.randn(hidden, num_classes, generator=gen)

    def _make(n: int) -> TensorDataset:
        x = torch.randn(n, in_dim, generator=gen)
        y = (torch.tanh(x @ w1) @ w2).argmax(dim=1)
        if label_noise > 0:
            flip = torch.rand(n, generator=gen) < label_noise
            noise = torch.randint(0, num_classes, (n,), generator=gen)
            y = torch.where(flip, noise, y)
        return TensorDataset(x, y)

    train_loader = DataLoader(
        _make(n_train), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(_make(n_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_cifar10_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-10."""
    data_root = (
        os.environ.get("DAG_HESSE_DATA_ROOT", data_root)
        if data_root == "./data"
        else data_root
    )
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR10(
        data_root,
        train=True,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=transform,
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_root,
        train=False,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=transform,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def get_cifar100_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-100."""
    data_root = (
        os.environ.get("DAG_HESSE_DATA_ROOT", data_root)
        if data_root == "./data"
        else data_root
    )
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR100(
        data_root,
        train=True,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=transform,
    )
    val_ds = torchvision.datasets.CIFAR100(
        data_root,
        train=False,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=transform,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


# CIFAR-10 with proper augmentation (for convolutional models)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_image_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-10 with augmentation.

    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + per-channel normalization.
    Val: per-channel normalization only.
    """
    data_root = (
        os.environ.get("DAG_HESSE_DATA_ROOT", data_root)
        if data_root == "./data"
        else data_root
    )
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    val_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    train_ds = torchvision.datasets.CIFAR10(
        data_root,
        train=True,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=train_transform,
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_root,
        train=False,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=val_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


# CIFAR-100 with proper augmentation (for convolutional models)
_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_image_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-100 with augmentation.

    The CIFAR-100 counterpart of `get_cifar10_image_loaders` (same conv-friendly
    pipeline, CIFAR-100 normalization): train uses RandomCrop(32, padding=4) +
    RandomHorizontalFlip + per-channel normalization; val uses normalization only.
    """
    data_root = (
        os.environ.get("DAG_HESSE_DATA_ROOT", data_root)
        if data_root == "./data"
        else data_root
    )
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    val_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    train_ds = torchvision.datasets.CIFAR100(
        data_root,
        train=True,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=train_transform,
    )
    val_ds = torchvision.datasets.CIFAR100(
        data_root,
        train=False,
        download=os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") == "1",
        transform=val_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


# Stanford Cars (fine-grained, 196 classes)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CARS_KAGGLE_ID = "eduardo4jesus/stanford-cars-dataset"
# An already-downloaded copy of the Kaggle mirror (the directory holding car_devkit/
# and cars_train/) bypasses kagglehub entirely - for machines without Kaggle access.
_CARS_ROOT_ENV = "DAG_HESSE_CARS_ROOT"


def _cars_root() -> str:
    """Use an explicit local Cars mirror; network access requires an opt-in."""
    root = os.environ.get(_CARS_ROOT_ENV)
    if root:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{_CARS_ROOT_ENV}={root!r} is not a directory")
        return root
    if os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") != "1":
        raise FileNotFoundError(
            f"Set {_CARS_ROOT_ENV} to an existing Stanford Cars mirror. "
            "Automatic dataset downloads are disabled."
        )
    import kagglehub

    return str(kagglehub.dataset_download(_CARS_KAGGLE_ID))


def _cars_transforms(image_size: int, *, train: bool) -> T.Compose:
    """ImageNet-normalized transforms; fine-grained augmentation on the train split.

    Train: RandomResizedCrop(scale in [0.5, 1.0]) + horizontal flip - mild for a
    fine-grained task where over-aggressive cropping would discard the discriminative
    object. Eval: Resize(256/224 * image_size) + CenterCrop, the standard ImageNet
    protocol matching the IMAGENET1K_V2 backbone weights.
    """
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    resize = int(round(image_size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _stratified_split(
    labels: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic per-class split of sample indices into (train, val).

    Stratification keeps every class represented in both splits - essential for a
    fine-grained 196-way problem where a plain random split could starve rare classes.
    Within each class the indices are shuffled with a fixed-seed RNG and the first
    ceil(val_fraction * n_c) (clipped to leave >=1 for training) go to validation; a
    singleton class stays entirely in training.
    """
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_class):
        idxs = by_class[label][:]
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 1:
            train_idx.extend(idxs)
            continue
        n_val = min(max(1, math.ceil(val_fraction * n)), n - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


class _StanfordCarsSplit(Dataset):
    """A (path, label) list with a transform; loads RGB images lazily via PIL."""

    def __init__(self, samples: list[tuple[str, int]], transform: T.Compose) -> None:
        self._samples = samples
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self._samples[index]
        with Image.open(path) as img:
            image = self._transform(img.convert("RGB"))
        return image, label


def _load_cars_train_samples(root: str) -> list[tuple[str, int]]:
    """Parse `cars_train_annos.mat` into [(image_path, label)] with labels in [0, 195]."""
    import scipy.io as sio

    annos_path = os.path.join(root, "car_devkit", "devkit", "cars_train_annos.mat")
    images_dir = os.path.join(root, "cars_train", "cars_train")
    if not os.path.isfile(annos_path) or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Unexpected Stanford Cars layout under {root!r}; expected "
            "car_devkit/devkit/cars_train_annos.mat and cars_train/cars_train/."
        )
    annotations = sio.loadmat(annos_path)["annotations"][0]
    samples: list[tuple[str, int]] = []
    for record in annotations:
        fname = str(record["fname"][0])
        label = int(record["class"][0, 0]) - 1  # MATLAB 1-based -> 0-based
        samples.append((os.path.join(images_dir, fname), label))
    return samples


def get_stanford_cars_loaders(
    batch_size: int,
    *,
    image_size: int = 224,
    augment: bool = True,
    val_fraction: float = 0.2,
    num_workers: int = 4,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for Stanford Cars (196 classes).

    The dataset is fetched from Kaggle via `kagglehub`, which authenticates through the
    standard KAGGLE_USERNAME / KAGGLE_KEY environment variables or ~/.kaggle/kaggle.json
    (no credentials are read or stored here), unless ``DAG_HESSE_CARS_ROOT`` points at an
    already-downloaded copy (see `_cars_root`). This Kaggle mirror ships labels only for
    the 8144-image training set - the official test annotations are unlabeled - so the
    validation split is carved from the labeled training set with a deterministic,
    per-class stratified split (see `_stratified_split`).

    Images are ImageNet-normalized; the train split uses fine-grained augmentation when
    `augment` is set, matching the IMAGENET1K_V2 pretrained backbone.
    """
    samples = _load_cars_train_samples(_cars_root())
    labels = [label for _, label in samples]
    train_idx, val_idx = _stratified_split(labels, val_fraction, seed)

    train_ds = _StanfordCarsSplit(
        [samples[i] for i in train_idx],
        _cars_transforms(image_size, train=augment),
    )
    val_ds = _StanfordCarsSplit(
        [samples[i] for i in val_idx],
        _cars_transforms(image_size, train=False),
    )
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


# ======================================================================
# ImageNet-32 (downsampled ImageNet-1k, 32x32)
# ======================================================================

_IMAGENET32_HF_ID = "benjamin-paine/imagenet-1k-32x32"


def _imagenet32_transforms(*, train: bool) -> T.Compose:
    """CIFAR-style 32x32 transforms with ImageNet normalization.

    Train: RandomCrop(32, padding=4) + horizontal flip; eval: normalization only. The
    images are already 32x32, so no resize is needed.
    """
    if train:
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class _HFImageClassification(Dataset):
    """A Hugging Face image-classification split with a torchvision transform applied lazily.

    Wraps a split whose records follow the standard Hugging Face image schema
    (``{"image": PIL.Image, "label": int}``) so it plugs into a torch ``DataLoader`` like
    the other datasets here; the image is converted to RGB and transformed on access.
    """

    def __init__(self, split: Any, transform: T.Compose) -> None:
        self._split = split
        self._transform = transform

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self._split[index]
        image = self._transform(record["image"].convert("RGB"))
        return image, int(record["label"])


def get_imagenet32_loaders(
    batch_size: int,
    *,
    augment: bool = True,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for ImageNet-1k downsampled to 32x32.

    Fetched from the Hugging Face hub (`benjamin-paine/imagenet-1k-32x32`), which is gated:
    authentication uses the standard ``HF_TOKEN`` environment variable (no token is read or
    stored here). The labelled ``train`` (1.28M) and ``validation`` (50k) splits are used -
    the ``test`` split is unlabelled and ignored; labels span 1001 classes (the mirror keeps
    the index-0 background class). Images are 32x32, CIFAR-style augmented on the train split
    and ImageNet-normalized. ``datasets`` is imported lazily so the rest of the package
    imports without this heavy dependency.
    """
    try:
        from datasets import load_dataset
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only on the imagenet32 path
        raise ImportError(
            "exp8 'imagenet32' needs the Hugging Face datasets package; install it with "
            "`uv add datasets` (the optional ImageNet-32 anchor)."
        ) from exc

    dataset = load_dataset(_IMAGENET32_HF_ID)
    train_ds = _HFImageClassification(
        dataset["train"], _imagenet32_transforms(train=augment)
    )
    val_ds = _HFImageClassification(
        dataset["validation"], _imagenet32_transforms(train=False)
    )
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader
