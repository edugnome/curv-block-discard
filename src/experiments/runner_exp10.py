"""Exp10: conditioning of the blocks (appendix "Conditioning of the blocks").

Second-order steps solve (A + lambda I) d = -g with A the full Hessian (Newton-CG), a
diagonal block (K-FAC, per block) or a reduced Hessian (the overlay); what governs them
is the spectrum of these operators. For a symmetric A this runner reports

  lambda_max(A), lambda_min(A)      extreme (algebraic) eigenvalues
  kappa_lambda(A) = (lambda_max + lambda) / (lambda_min + lambda)
                                    inf if lambda_min + lambda <= 0;  lambda in {1e-2, 1e-1}
  D(A) = ||A||_F^2 / ||A||_2^2      stable rank
  neg_frac(A)                       fraction of eigenvalues < -1e-8 max|lambda| (exact only)

in two parts (`Exp10Config.parts`):

  ``mlp`` (exact, activation space) - the Exp. 1 Plain / Residual MLPs (L = 8, width 64,
    CIFAR-10, seeds 42-44) at init and final: the activation-space block matrix
    [H^f_{v,w}] over all layer pairs (512 x 512) is assembled with `ExactBlockHessian`,
    symmetrized ((M + M^T) / 2) and its spectrum taken in float64
    (`torch.linalg.eigvalsh`) for the full matrix and for every diagonal block H^f_{v,v}.
  ``resnet18`` (matrix-free, parameter space) - `SegmentedResNet18` under the Exp. 6
    protocol (seed 42) at init and final: the six diagonal parameter blocks
    H_{theta_v, theta_v} and the full H_theta are read off the exact full-parameter HVP
    of `ParamBlockEstimator.curvature` - a block's action is z |-> (H P_v z)|_v with P_v
    the group-major block embedding - and summarized by one m-step Lanczos run per
    operator (both extremes at once, with their residual bounds and a ``converged``
    flag) plus Hutchinson (||A||_F^2: block-local probes for a block, full probes for
    H_theta); see `hessian.spectrum`.

Both parts measure on one fixed validation minibatch under `deterministic_mode` and
aggregate over seeds into per-(arch, checkpoint) mean/std of every quantity, with the
raw per-seed records kept next to the aggregates (``"seeds"``) in the exp8 bundle shape.
The bundle is JSON-serializable (an undamped kappa = inf serializes as ``Infinity``).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader

from experiments.cli import Experiment, add_device_args
from experiments.config import Exp10Config, TrainingConfig
from experiments.data import get_cifar10_image_loaders, get_cifar10_loaders
from experiments.models import PlainMLP, ResidualMLP, SegmentedResNet18
from experiments.protocol import (
    checkpoint_epochs,
    epoch_to_label,
    mean_std,
    sample_hessian_batch,
)
from experiments.training import Trainer
from experiments.utils import deterministic_mode, set_seed
from hessian import ExactBlockHessian, ParamBlockEstimator, ParamGroupedModel
from hessian.spectrum import (
    HVPClosure,
    conditioning_from_spectrum,
    matrix_free_conditioning,
)
from hessian.types import LayerID

logger = logging.getLogger(__name__)

# The exact part's architectures, keyed as in the Exp. 1 bundle.
_MLP_ARCHS: dict[str, type[PlainMLP] | type[ResidualMLP]] = {
    "plain": PlainMLP,
    "residual": ResidualMLP,
}
# The matrix-free part's single architecture key.
_RESNET_ARCH = "resnet18"

# part -> (train_loader, val_loader); the runner's injectable data seam.
LoaderFactory = Callable[[str], tuple[DataLoader, DataLoader]]
# (model, x, y, loss_fn) -> {"full": record, "blocks": {block: record}}.
Measurement = Callable[[nn.Module, Tensor, Tensor, nn.Module], dict[str, Any]]


# ======================================================================
# Exact part: the activation-space block matrix and its spectrum
# ======================================================================


def assemble_activation_hessian(
    model: PlainMLP | ResidualMLP,
    x: Tensor,
    y: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, dict[LayerID, slice]]:
    """Symmetrized full block matrix [H^f_{v,w}] (float64, CPU) and its layer slices.

    Every pair v <= w in layer order is computed once with `ExactBlockHessian`; the
    lower blocks are the transposes, and the assembled matrix is symmetrized as
    (M + M^T) / 2 to remove the float32 asymmetry of the autograd blocks.
    """
    names = model.get_layer_names()
    n = len(names)
    pairs = [(names[i], names[j]) for i in range(n) for j in range(i, n)]
    result = ExactBlockHessian(model, loss_fn).compute_all_blocks(x, y, pairs)

    ranges: dict[LayerID, slice] = {}
    offset = 0
    for name in names:
        dim = result.layer_dims[name]
        ranges[name] = slice(offset, offset + dim)
        offset += dim

    matrix = torch.zeros(offset, offset, dtype=torch.float64)
    for i, v in enumerate(names):
        for w in names[i:]:
            block = result.blocks[(v, w)].matrix.detach().to("cpu", torch.float64)
            matrix[ranges[v], ranges[w]] = block
            if w != v:
                matrix[ranges[w], ranges[v]] = block.T
    return (matrix + matrix.T) / 2, ranges


def exact_conditioning(
    model: PlainMLP | ResidualMLP,
    x: Tensor,
    y: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    lambdas: list[float],
    *,
    neg_tol: float,
) -> dict[str, Any]:
    """Exact spectra of the full [H^f_{v,w}] and of every diagonal block H^f_{v,v}.

    ``{"full": record, "blocks": {layer: record}}`` with the `conditioning_from_spectrum`
    quantities; the full record also keeps its sorted ``eigenvalues``.
    """
    with deterministic_mode():
        matrix, ranges = assemble_activation_hessian(model, x, y, loss_fn)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    full = conditioning_from_spectrum(eigenvalues, lambdas, neg_tol=neg_tol)
    full["eigenvalues"] = eigenvalues.tolist()
    blocks = {
        v: conditioning_from_spectrum(
            torch.linalg.eigvalsh(matrix[sv, sv]), lambdas, neg_tol=neg_tol
        )
        for v, sv in ranges.items()
    }
    return {"full": full, "blocks": blocks}


# ======================================================================
# Matrix-free part: parameter blocks through the exact HVP
# ======================================================================


def _block_ranges(groups: dict[LayerID, list[Parameter]]) -> dict[LayerID, slice]:
    """Group-major slices of the flat parameter vector (the estimator's layout)."""
    ranges: dict[LayerID, slice] = {}
    offset = 0
    for name, params in groups.items():
        size = sum(p.numel() for p in params)
        ranges[name] = slice(offset, offset + size)
        offset += size
    return ranges


def _block_hvp(hvp: HVPClosure, block: slice, total_dim: int) -> HVPClosure:
    """z |-> (H P_v z)|_v: embed z in block v of the flat vector, apply H, restrict."""

    def action(z: Tensor) -> Tensor:
        full = torch.zeros(total_dim, device=z.device, dtype=z.dtype)
        full[block] = z
        return hvp(full)[block]

    return action


def matrix_free_block_conditioning(
    model: ParamGroupedModel,
    x: Tensor,
    y: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    lambdas: list[float],
    *,
    n_lanczos_iter: int,
    n_probes: int,
    seed: int,
) -> dict[str, Any]:
    """HVP-only summaries of the full H_theta and of every diagonal block H_{v,v}.

    One backward graph (`ParamBlockEstimator.curvature`) serves every operator; the
    Hutchinson probes and the Krylov start vectors come from a generator seeded with
    ``seed`` so the measurement is reproducible. T + m HVPs per operator (T Lanczos
    steps, m probes), (n_blocks + 1) operators.
    """
    with deterministic_mode():
        curvature = ParamBlockEstimator(model, loss_fn).curvature(x, y)
        total_dim = curvature.flat_grad.numel()
        device = curvature.flat_grad.device
        generator = torch.Generator(device=device).manual_seed(seed)
        kwargs: dict[str, Any] = {
            "n_iter": n_lanczos_iter,
            "n_probes": n_probes,
            "device": device,
            "generator": generator,
        }
        full = matrix_free_conditioning(curvature.hvp, total_dim, lambdas, **kwargs)
        blocks = {
            v: matrix_free_conditioning(
                _block_hvp(curvature.hvp, sv, total_dim),
                sv.stop - sv.start,
                lambdas,
                **kwargs,
            )
            for v, sv in _block_ranges(model.get_param_groups()).items()
        }
    return {"full": full, "blocks": blocks}


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """mean/std of every scalar quantity of a record (kappa nested by damping).

    Non-scalar entries (the stored ``eigenvalues``) are left to the per-seed records.
    """
    aggregated: dict[str, Any] = {}
    for key, ref in records[0].items():
        if isinstance(ref, dict):
            aggregated[key] = {k: mean_std([r[key][k] for r in records]) for k in ref}
        elif isinstance(ref, (int, float)):
            aggregated[key] = mean_std([r[key] for r in records])
    return aggregated


def _aggregate(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregates over seeds: mean/std for each (arch, checkpoint), plus the raw records.

    Per (arch, checkpoint): the full-operator summary and every block's summary (each
    quantity as ``{mean, std}``), the validation accuracy, and the per-seed records
    themselves under ``"seeds"`` (``{"seed_42": {"full", "blocks", "test_acc"},...}``).
    """
    aggregated: dict[str, Any] = {}
    for arch, ckpt_map in raw.items():
        aggregated[arch] = {}
        for ckpt, seed_data in ckpt_map.items():
            payloads = list(seed_data.values())
            ref = payloads[0]
            aggregated[arch][ckpt] = {
                "full": _aggregate_record([p["full"] for p in payloads]),
                "blocks": {
                    block: _aggregate_record([p["blocks"][block] for p in payloads])
                    for block in ref["blocks"]
                },
                "test_acc_mean": float(np.mean([p["test_acc"] for p in payloads])),
                "test_acc_std": float(np.std([p["test_acc"] for p in payloads])),
                "n_seeds": len(payloads),
                "seeds": seed_data,
            }
    return aggregated


# ======================================================================
# Runner
# ======================================================================


class Exp10Runner:
    """Runner for Exp10 (conditioning of the blocks: exact MLP spectra + HVP-only ResNet)."""

    def __init__(
        self,
        config: Exp10Config,
        device: torch.device,
        *,
        loaders: LoaderFactory | None = None,
    ) -> None:
        self._cfg = config
        self._device = device
        # Injectable data seam (tests pass tiny in-memory loaders); the default is the
        # CIFAR-10 loaders of Exp. 1 (flattened MLP) and Exp. 6 (augmented images).
        self._loaders: LoaderFactory = loaders or self._cifar10_loaders

    def run(self) -> dict[str, Any]:
        """Returns {meta, part: {arch: {checkpoint: {full, blocks, test_acc, seeds}}}}."""
        cfg = self._cfg
        results: dict[str, Any] = {
            "meta": {
                "lambdas": list(cfg.lambdas),
                "n_lanczos_iter": cfg.n_lanczos_iter,
                "n_probes": cfg.n_probes,
                "neg_tol": cfg.neg_tol,
                "hessian_batch_size": cfg.hessian_batch_size,
                "depth": cfg.depth,
                "width": cfg.width,
            }
        }
        for part in cfg.parts:
            logger.info("=== Part: %s ===", part)
            if part == "mlp":
                results[part] = self._run_mlp()
            elif part == "resnet18":
                results[part] = self._run_resnet()
            else:
                raise ValueError(f"unknown part {part!r}; expected 'mlp' or 'resnet18'")
        return results

    # ------------------------------------------------------------------
    # Parts
    # ------------------------------------------------------------------

    def _run_mlp(self) -> dict[str, Any]:
        """Exact part: Plain / Residual MLP under the Exp. 1 protocol."""
        cfg = self._cfg
        train_loader, val_loader = self._loaders("mlp")
        raw: dict[str, dict[str, Any]] = {arch: {} for arch in _MLP_ARCHS}

        def measure(model: nn.Module, x: Tensor, y: Tensor, loss_fn: nn.Module) -> dict:
            assert isinstance(model, (PlainMLP, ResidualMLP))
            return exact_conditioning(
                model, x, y, loss_fn, cfg.lambdas, neg_tol=cfg.neg_tol
            )

        for seed in cfg.mlp_training.seeds:
            logger.info("  Seed %d", seed)
            for arch, model_cls in _MLP_ARCHS.items():
                logger.info("    Architecture: %s", arch)
                set_seed(seed, deterministic=cfg.mlp_training.deterministic)
                kwargs: dict[str, Any] = {"width": cfg.width, "depth": cfg.depth}
                if model_cls is PlainMLP:
                    kwargs["use_layernorm"] = cfg.use_layernorm
                model = model_cls(**kwargs)
                records = self._train_and_measure(
                    model, cfg.mlp_training, train_loader, val_loader, measure
                )
                for ckpt, record in records.items():
                    raw[arch].setdefault(ckpt, {})[f"seed_{seed}"] = record
        return _aggregate(raw)

    def _run_resnet(self) -> dict[str, Any]:
        """Matrix-free part: ResNet-18 under the Exp. 6 protocol."""
        cfg = self._cfg
        train_loader, val_loader = self._loaders("resnet18")
        raw: dict[str, dict[str, Any]] = {_RESNET_ARCH: {}}

        for seed in cfg.resnet_training.seeds:
            logger.info("  Seed %d", seed)
            set_seed(seed, deterministic=cfg.resnet_training.deterministic)
            model = SegmentedResNet18(num_classes=10)

            def measure(
                model: nn.Module, x: Tensor, y: Tensor, loss_fn: nn.Module, seed=seed
            ) -> dict:
                assert isinstance(model, SegmentedResNet18)
                return matrix_free_block_conditioning(
                    model,
                    x,
                    y,
                    loss_fn,
                    cfg.lambdas,
                    n_lanczos_iter=cfg.n_lanczos_iter,
                    n_probes=cfg.n_probes,
                    seed=seed,
                )

            records = self._train_and_measure(
                model, cfg.resnet_training, train_loader, val_loader, measure
            )
            for ckpt, record in records.items():
                raw[_RESNET_ARCH].setdefault(ckpt, {})[f"seed_{seed}"] = record
        return _aggregate(raw)

    # ------------------------------------------------------------------
    # Shared train / checkpoint / measure loop
    # ------------------------------------------------------------------

    def _train_and_measure(
        self,
        model: nn.Module,
        training: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        measure: Measurement,
    ) -> dict[str, Any]:
        """Trains, restores every requested checkpoint, measures on the fixed minibatch.

        Returns ``{checkpoint: {"full", "blocks", "test_acc"}}``.
        """
        cfg = self._cfg
        trainer = Trainer(model, training, self._device)
        ckpt_epochs = checkpoint_epochs(training.epochs, cfg.checkpoint_epochs)
        checkpoints = trainer.train(train_loader, checkpoint_epochs=ckpt_epochs)

        records: dict[str, Any] = {}
        for ep, state in checkpoints.items():
            trainer.restore(state)
            model.to(self._device)
            model.eval()
            x, y = sample_hessian_batch(
                val_loader, cfg.hessian_batch_size, self._device
            )
            record = measure(model, x, y, trainer.loss_fn)
            record["test_acc"] = trainer.evaluate(val_loader)["test_acc"]
            records[epoch_to_label(ep, training.epochs)] = record
        return records

    def _cifar10_loaders(self, part: str) -> tuple[DataLoader, DataLoader]:
        if part == "mlp":
            return get_cifar10_loaders(self._cfg.mlp_training.batch_size)
        if part == "resnet18":
            return get_cifar10_image_loaders(self._cfg.resnet_training.batch_size)
        raise ValueError(f"unknown part {part!r}")


# ======================================================================
# CLI: exp10 (conditioning of the blocks)
# ======================================================================


def add_args(parser: argparse.ArgumentParser) -> None:
    add_device_args(parser)
    parser.add_argument(
        "--part",
        type=str,
        nargs="+",
        default=None,
        choices=["mlp", "resnet18"],
        help="mlp: exact activation-space spectra of the Exp. 1 Plain / Residual MLPs; "
        "resnet18: matrix-free parameter-space blocks of ResNet-18 (Exp. 6 protocol). "
        "Default: both.",
    )
    # Optional overrides; a value of None leaves the config default untouched. The
    # training overrides apply to both parts (the MLPs' Exp. 1 schedule and ResNet-18's
    # Exp. 6 schedule alike) - meant for smoke runs, the appendix uses the defaults.
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=None)
    parser.add_argument("--n-lanczos-iter", type=int, default=None)
    parser.add_argument("--n-probes", type=int, default=None)
    parser.add_argument("--hessian-batch-size", type=int, default=None)


def build_config(args: argparse.Namespace) -> Exp10Config:
    """The appendix defaults, then any explicit CLI overrides."""
    base = Exp10Config()
    training_overrides = {
        key: value
        for key, value in {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
        }.items()
        if value is not None
    }
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "parts": args.part,
            "depth": args.depth,
            "width": args.width,
            "lambdas": args.lambdas,
            "n_lanczos_iter": args.n_lanczos_iter,
            "n_probes": args.n_probes,
            "hessian_batch_size": args.hessian_batch_size,
        }.items()
        if value is not None
    }
    return replace(
        base,
        mlp_training=replace(base.mlp_training, **training_overrides),
        resnet_training=replace(base.resnet_training, **training_overrides),
        **overrides,
    )


SPEC = Experiment(
    name="exp10",
    help="Conditioning of the blocks (exact MLP spectra + matrix-free ResNet-18)",
    add_args=add_args,
    build_config=build_config,
    run=Exp10Runner,
    default_output_dir=lambda args: "results/exp10",
)
