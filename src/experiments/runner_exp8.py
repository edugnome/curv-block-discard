"""Measure discarded parameter-Hessian mass across model and weight profiles.

Phi = 1 - sum_v ||H_vv||_F^2 / ||H||_F^2 uses the configured parameter partition.
Reports also retain diagonal block norms and the directed coupling field.

The a1 profiles compare ResNet-18 and its skip-free control at configured
checkpoints; a2 measures an ImageNet-pretrained ResNet-50 backbone with a new
classifier on Stanford Cars before finetuning. Raw Phi depends on the partition,
diagonal profile and weight state; these comparisons do not isolate a causal
skip-connection or pretraining mechanism.

Models can be trained to the checkpoints or restored from saved states without
training. Each measurement uses a fixed-size validation minibatch. Results keep
raw per-seed observations alongside architecture/checkpoint means and spreads.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.cli import Experiment, add_device_args, override
from experiments.config import Exp8Config
from experiments.data import (
    DATASET_NUM_CLASSES,
    get_cifar10_image_loaders,
    get_cifar100_image_loaders,
    get_imagenet32_loaders,
    get_stanford_cars_loaders,
)
from experiments.models import (
    SegmentedPlainResNet18,
    SegmentedResNet18,
    SegmentedResNet34,
    SegmentedResNet50,
    SegmentedWideResNet18,
)
from experiments.phi import measure_phi, measure_phi_and_diag_frob
from experiments.protocol import (
    checkpoint_epochs,
    epoch_to_label,
    load_checkpoints,
    mean_std,
    sample_hessian_batch,
    save_checkpoints,
)
from experiments.provenance import META_KEY, provenance
from experiments.regroup import LEVELS, Regrouped
from experiments.training import Trainer
from experiments.utils import set_seed

logger = logging.getLogger(__name__)

# Fixed seed for the train/val split, so every architecture and seed sees the same data.
_SPLIT_SEED = 0


# ======================================================================
# Dataset / backbone construction
# ======================================================================


def _build_loaders(cfg: Exp8Config) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) for the configured dataset (fixed split).

    Phi is read off the validation loader; ``a1`` also trains on the train loader, while
    ``a2`` (epochs = 0) leaves it untouched and only samples a measurement minibatch.
    """
    batch_size = cfg.training.batch_size
    if cfg.dataset == "cifar100":
        return get_cifar100_image_loaders(batch_size)
    if cfg.dataset == "cifar10":
        return get_cifar10_image_loaders(batch_size)
    if cfg.dataset == "imagenet32":
        return get_imagenet32_loaders(batch_size, num_workers=cfg.num_workers)
    if cfg.dataset == "stanford_cars":
        return get_stanford_cars_loaders(
            batch_size,
            image_size=cfg.image_size,
            augment=False,  # diagnostic only: measure on un-augmented inputs
            num_workers=cfg.num_workers,
            seed=_SPLIT_SEED,
        )
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def _build_model(cfg: Exp8Config, arch: str) -> SegmentedResNet18:
    """Backbone for the registry key ``arch`` (all share the ResNet segmentation).

    ``resnet18_w<m>`` is ResNet-18 with every stage width scaled by ``<m>`` (the width
    axis). Only ``resnet50`` honours ``pretrained``; the from-scratch backbones reject it
    rather than silently ignoring it.
    """
    if cfg.pretrained and arch != "resnet50":
        raise ValueError(
            f"pretrained=True is only supported for 'resnet50'; arch {arch!r} builds "
            "from scratch (set pretrained=False)"
        )
    if arch == "resnet50":
        return SegmentedResNet50(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    if arch == "resnet34":
        return SegmentedResNet34(num_classes=cfg.num_classes)
    if arch == "plain_resnet18":
        return SegmentedPlainResNet18(num_classes=cfg.num_classes)
    if arch == "resnet18":
        return SegmentedResNet18(num_classes=cfg.num_classes)
    if arch.startswith("resnet18_w"):
        return SegmentedWideResNet18(
            num_classes=cfg.num_classes,
            width_mult=float(arch.removeprefix("resnet18_w")),
        )
    raise ValueError(
        f"unknown arch {arch!r}; expected one of 'resnet18', 'resnet18_w<mult>', "
        "'resnet34', 'plain_resnet18', 'resnet50'"
    )


# ======================================================================
# Aggregation
# ======================================================================


def aggregate(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregates over seeds: mean/std for each (arch, checkpoint), plus the raw reports.

    Per (arch, checkpoint): scalar Phi, per-block diagonal norm, per-pair coupling, and
    validation accuracy - each summarized over the seeds that measured it - and the
    per-seed reports themselves under ``"seeds"`` (``{"seed_42": report,...}``).
    """
    aggregated: dict[str, Any] = {}
    for arch, ckpt_map in raw.items():
        aggregated[arch] = {}
        for ckpt, seed_data in ckpt_map.items():
            payloads = list(seed_data.values())
            ref = payloads[0]
            aggregated[arch][ckpt] = {
                "phi_mean": float(np.mean([p["phi"] for p in payloads])),
                "phi_std": float(np.std([p["phi"] for p in payloads])),
                "diag_frob": {
                    block: mean_std([p["diag_frob"][block] for p in payloads])
                    for block in ref["diag_frob"]
                },
                "coupling": {
                    pair: mean_std([p["coupling"][pair] for p in payloads])
                    for pair in ref["coupling"]
                },
                "test_acc_mean": float(np.mean([p["test_acc"] for p in payloads])),
                "test_acc_std": float(np.std([p["test_acc"] for p in payloads])),
                "n_seeds": len(payloads),
                "seeds": seed_data,
            }
    return aggregated


# ======================================================================
# Per-checkpoint measurement (the seam the training and the restore path share)
# ======================================================================


def _measure_checkpoint(
    cfg: Exp8Config,
    device: torch.device,
    trainer: Trainer,
    model: SegmentedResNet18,
    state: dict[str, Tensor],
    val_loader: DataLoader,
    *,
    seed: int,
    ckpt_paths: dict[str, str],
) -> dict[str, Any]:
    """The measurement at one checkpoint: Phi, test_acc, and the optional probe sweep
    / rank-one reference / granularity axis - exactly what `Exp8Runner.run` calls
    whether ``state`` came from `Trainer.train` (a fresh run) or from
    `load_checkpoints` (a restored one, no training in between). Kept as the one place
    this logic is written, so a restored bundle is the same shape as a trained one by
    construction rather than by two call sites being kept in sync by hand.
    """
    trainer.restore(state)
    model.to(device)
    model.eval()

    x_batch, y_batch = sample_hessian_batch(val_loader, cfg.hessian_batch_size, device)
    report = measure_phi(
        model,
        x_batch,
        y_batch,
        trainer.loss_fn,
        cfg.n_probes,
        with_rank1=cfg.with_rank1,
    )
    report["test_acc"] = trainer.evaluate(val_loader)["test_acc"]
    if cfg.save_checkpoints:
        report["checkpoints"] = ckpt_paths
    if cfg.probe_sweep:
        # One generator, seeded off this seed (not the ambient global RNG's
        # position), shared by every m in the sweep: the draws differ by n_probes
        # alone, as the sweep is meant to isolate.
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        report["phi_by_probes"] = {
            str(m): measure_phi(
                model,
                x_batch,
                y_batch,
                trainer.loss_fn,
                m,
                generator=gen,
            )["phi"]
            for m in cfg.probe_sweep
        }
    if cfg.granularity:
        # "stage" is Regrouped's own name for the model's unmodified
        # get_param_groups(); measuring it again would just repeat the headline phi
        # above. Phi plus the per-group diagonal norms, not the full report: this
        # axis wants those two and nothing else, and the coupling field measure_phi
        # also builds is O(groups^2) pairs - the term that made "layer" (253 blocks
        # on ResNet-18) cost 9x a Phi-only measurement. measure_phi_and_diag_frob
        # costs exactly what a Phi-only measurement costs (one full-parameter probe
        # plus one group-localized probe per diagonal block); the norms are already
        # computed on that path and were simply discarded before.
        granularity_reports = {
            level: measure_phi_and_diag_frob(
                Regrouped(model, level),
                x_batch,
                y_batch,
                trainer.loss_fn,
                cfg.n_probes,
            )
            for level in cfg.granularity
            if level != "stage"
        }
        report["phi_by_granularity"] = {
            level: r["phi"] for level, r in granularity_reports.items()
        }
        report["diag_frob_by_granularity"] = {
            level: r["diag_frob"] for level, r in granularity_reports.items()
        }
    return report


# ======================================================================
# Runner
# ======================================================================


class Exp8Runner:
    """Compare discarded Hessian mass under the configured measurement profiles."""

    def __init__(self, config: Exp8Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {meta, arch: {checkpoint: {phi_mean/std, diag_frob, coupling, test_acc, seeds}}}."""
        _, val_loader = _build_loaders(self._cfg)
        raw: dict[str, dict[str, Any]] = {arch: {} for arch in self._cfg.archs}
        restore_dir = self._cfg.restore_checkpoints_dir

        for arch in self._cfg.archs:
            logger.info("=== Arch: %s ===", arch)

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)
                set_seed(seed, deterministic=self._cfg.training.deterministic)

                model = _build_model(self._cfg, arch)
                trainer = Trainer(model, self._cfg.training, self._device)

                epochs = self._cfg.training.epochs
                ckpt_epochs = checkpoint_epochs(epochs, self._cfg.checkpoint_epochs)

                if restore_dir is not None:
                    logger.info(
                        "  Restoring checkpoints from %s (no training)", restore_dir
                    )
                    checkpoints = load_checkpoints(
                        Path(restore_dir), arch=arch, seed=seed, epochs=ckpt_epochs
                    )
                else:
                    # Build a fresh training loader after seeding each architecture
                    # and seed to isolate persistent-worker RNG state. The shared
                    # validation loader does not shuffle or augment.
                    train_loader, _ = _build_loaders(self._cfg)
                    checkpoints = trainer.train(
                        train_loader, checkpoint_epochs=ckpt_epochs
                    )

                ckpt_paths: dict[str, str] = {}
                if self._cfg.save_checkpoints:
                    ckpt_paths = save_checkpoints(
                        checkpoints, Path(self._cfg.output_dir), arch=arch, seed=seed
                    )

                for ep, state in checkpoints.items():
                    report = _measure_checkpoint(
                        self._cfg,
                        self._device,
                        trainer,
                        model,
                        state,
                        val_loader,
                        seed=seed,
                        ckpt_paths=ckpt_paths,
                    )
                    ckpt_name = epoch_to_label(ep, epochs)
                    raw[arch].setdefault(ckpt_name, {})[f"seed_{seed}"] = report

        return {
            META_KEY: provenance(self._cfg, experiment="exp8"),
            **aggregate(raw),
        }


# ======================================================================
# CLI: exp8 (Phi diagnostic)
# ======================================================================


def add_args(parser: argparse.ArgumentParser) -> None:
    add_device_args(parser)
    parser.add_argument(
        "--profile",
        type=str,
        default="a1",
        choices=["a1", "a1_width", "a1_imagenet32", "a2"],
        help="a1: compare ResNet-18 and plain ResNet-18 on CIFAR-100; "
        "a1_width: width/depth sweep (0.5x/1x/2x ResNet-18 and ResNet-34); "
        "a1_imagenet32: the model comparison on downsampled ImageNet-32; "
        "a2: measure a pretrained ResNet-50 backbone with a new head on "
        "Stanford Cars before finetuning.",
    )
    # Optional overrides; a value of None leaves the profile default untouched.
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cifar100", "cifar10", "imagenet32", "stanford_cars"],
    )
    parser.add_argument(
        "--archs",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "resnet18",
            "resnet18_w0.5",
            "resnet18_w2.0",
            "resnet34",
            "plain_resnet18",
            "resnet50",
        ],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-probes", type=int, default=None)
    parser.add_argument("--hessian-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps in total (benchmarks and smokes)",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=str,
        nargs="+",
        default=None,
        help="which epochs to measure: init / mid / final and/or epoch numbers "
        "(default: the profile's init mid final)",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="write every measured checkpoint's state_dict under "
        "<output-dir>/checkpoints/, recorded per seed as an epoch->path map",
    )
    parser.add_argument(
        "--restore-checkpoints-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="read every requested checkpoint's state_dict back from DIR/checkpoints/ "
        "instead of training (DIR is another run's own --output-dir from an earlier "
        "--save-checkpoints run); no optimizer step runs, not even a train loader is "
        "built",
    )
    parser.add_argument(
        "--probe-sweep",
        type=int,
        nargs="+",
        default=None,
        help="additionally measure Phi at each of these probe counts (one shared "
        "seeded generator per checkpoint), recorded as phi_by_probes",
    )
    parser.add_argument(
        "--with-rank1",
        action="store_true",
        help="add phi_rank1 and lambda_max (one Lanczos run) to every measured "
        "checkpoint",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        nargs="+",
        choices=list(LEVELS),
        default=None,
        help="additionally measure Phi through Regrouped at each level beyond the "
        "model's own partition, recorded as phi_by_granularity",
    )


def build_config(args: argparse.Namespace) -> Exp8Config:
    """Selects the regime profile, then applies any explicit CLI overrides."""
    if args.profile == "a1":
        base = Exp8Config.a1_from_scratch()
    elif args.profile == "a1_width":
        base = Exp8Config.a1_width()
    elif args.profile == "a1_imagenet32":
        base = Exp8Config.a1_imagenet32()
    else:
        base = Exp8Config.a2_pretrained()

    training = override(
        base.training,
        epochs=args.epochs,
        seeds=args.seeds,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "archs": args.archs,
            "n_probes": args.n_probes,
            "hessian_batch_size": args.hessian_batch_size,
            "num_workers": args.num_workers,
            "checkpoint_epochs": args.checkpoint_epochs,
            "restore_checkpoints_dir": args.restore_checkpoints_dir,
            "probe_sweep": args.probe_sweep,
            "granularity": args.granularity,
        }.items()
        if value is not None
    }
    # A dataset switch implies its class count (keeps the head consistent).
    if "dataset" in overrides:
        overrides["num_classes"] = DATASET_NUM_CLASSES[overrides["dataset"]]

    return replace(
        base,
        training=training,
        output_dir=args.output_dir or f"results/exp8_{args.profile}",
        save_checkpoints=args.save_checkpoints,
        with_rank1=args.with_rank1,
        **overrides,
    )


SPEC = Experiment(
    name="exp8",
    help="Discarded Hessian mass across model, partition and weight-state profiles",
    add_args=add_args,
    build_config=build_config,
    run=Exp8Runner,
    default_output_dir=lambda args: f"results/exp8_{args.profile}",
)
