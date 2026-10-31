"""The CLI registry and generic driver shared by every experiment subcommand.

The rest of this module is the machinery every subcommand's `add_args` / `build_config`
is built from - device selection, the shared training/Hessian argument groups, result
serialization, the hang watchdog - kept here, separate from `main.py`, so the runner
modules can import it without reaching into a script.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import torch

from experiments.config import HessianConfig, TrainingConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)


# ======================================================================
# Common CLI arguments
# ======================================================================


def add_device_args(parser: argparse.ArgumentParser) -> None:
    """Registers device / output arguments shared by every subcommand."""
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "CUDA GPU index (nvidia-smi id, e.g. --gpu 1); under a caller's "
            "CUDA_VISIBLE_DEVICES it must be one of the visible ids. "
            "Overrides --device to cuda."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="existing CIFAR dataset root; no download by default",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="explicitly allow external dataset/model fetches",
    )
    parser.add_argument(
        "--hang-watchdog",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "dump every thread's stack to stderr every SECONDS until the run ends "
            "(faulthandler); the log then shows where a silent run is stuck"
        ),
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register common training and curvature options for diagnostic runners."""
    add_device_args(parser)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--hessian-mode",
        type=str,
        default="stochastic",
        choices=["exact", "stochastic"],
    )
    parser.add_argument("--n-probes", type=int, default=30)
    parser.add_argument("--n-power-iter", type=int, default=20)
    parser.add_argument("--hessian-batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Select the device; ``--gpu N`` pins the process to physical GPU ``N`` (as ``cuda:0``).

    ``N`` is the nvidia-smi index. When the caller has already restricted
    ``CUDA_VISIBLE_DEVICES`` (the dispatcher and the bench driver export one GPU per
    queue), ``N`` must be one of the visible ids and the list is narrowed to it - never
    overwritten: overwriting ``CUDA_VISIBLE_DEVICES=2`` with ``--gpu 0`` would instead
    land the run silently on physical GPU 0 - on the shared box, a card already fully
    occupied - where the K-FAC bench fails in ``cusolverDnCreate`` (no device memory
    left for the handle) or OOMs. The chosen card is logged with its UUID and free
    memory so the log shows *which* GPU ran.
    """
    if args.gpu is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and str(args.gpu) not in [
            s.strip() for s in visible.split(",")
        ]:
            raise SystemExit(
                f"--gpu {args.gpu} is not among CUDA_VISIBLE_DEVICES={visible!r}: "
                "pass a visible id, or drop --gpu (the caller already picked the GPU)"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        logger.info(
            "CUDA_VISIBLE_DEVICES=%s -> %s (uuid %s), %.1f of %.1f GiB free",
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            props.name,
            props.uuid,
            free / 2**30,
            total / 2**30,
        )
    return device


def arm_hang_watchdog(seconds: float | None, *, file: TextIO = sys.stderr) -> None:
    """Dump every thread's stack to ``file`` every ``seconds`` until the process exits.

    A run that stops logging is either hung or in a slow phase, and on a box without
    ptrace (``py-spy`` cannot attach) the two can only be told apart from inside the
    process: `faulthandler.dump_traceback_later` fires from its own watchdog thread, so
    it reports even while the main thread is blocked in a native call (a backward that
    never returns, a solver spinning on NaN). It costs nothing until it fires. Frames
    that move between two dumps are a slow phase; frames that never move are the hang.
    ``None`` (the CLI default) arms nothing.
    """
    if seconds is None:
        return
    if seconds <= 0:
        raise SystemExit("--hang-watchdog needs a positive number of seconds")
    faulthandler.dump_traceback_later(seconds, repeat=True, file=file)


def make_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        seeds=args.seeds,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
    )


def make_hessian_config(args: argparse.Namespace) -> HessianConfig:
    return HessianConfig(
        mode=args.hessian_mode,
        n_probes=args.n_probes,
        n_power_iter=args.n_power_iter,
        hessian_batch_size=args.hessian_batch_size,
    )


def override[T: "DataclassInstance"](base: T, **fields: Any) -> T:
    """``base`` with every non-None field replaced: the CLI's "unset leaves the profile"
    rule; the sentinel is ``None``, so a flag whose legitimate value is ``None`` cannot
    go through this helper.
    """
    return replace(base, **{k: v for k, v in fields.items() if v is not None})


def _make_serializable(obj: Any) -> Any:
    """Recursively converts dicts with int keys to str keys for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    return obj


def save_results(results: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info("Results saved to %s/results.json", out_dir)


# ======================================================================
# The experiment registry
# ======================================================================


@dataclass(frozen=True)
class Experiment:
    """One CLI subcommand: its flags, its config, its runner and where it writes."""

    name: str
    help: str
    add_args: Callable[[argparse.ArgumentParser], None]
    build_config: Callable[[argparse.Namespace], Any]
    run: Callable[[Any, torch.device], Any]  # ExpNRunner class
    default_output_dir: Callable[[argparse.Namespace], str]
    plots: Callable[[Any, argparse.Namespace, Path], None] | None = None


def build_parser(specs: list[Experiment]) -> argparse.ArgumentParser:
    """The root parser: one subcommand per entry of ``specs``, in that order."""
    parser = argparse.ArgumentParser(
        description="DAG-Hesse experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    for spec in specs:
        sub = subparsers.add_parser(spec.name, help=spec.help)
        spec.add_args(sub)
    return parser


def run_experiment(spec: Experiment, args: argparse.Namespace) -> None:
    """Resolve the device, build the config, run it, save and (optionally) plot it."""
    if getattr(args, "allow_downloads", False):
        os.environ["DAG_HESSE_ALLOW_DOWNLOADS"] = "1"
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if getattr(args, "data_root", None) is not None:
        os.environ["DAG_HESSE_DATA_ROOT"] = str(args.data_root.resolve())
    device = resolve_device(args)
    logger.info("Device: %s", device)

    config = spec.build_config(args)
    results = spec.run(config, device).run()

    out_dir = Path(args.output_dir or spec.default_output_dir(args))
    save_results(results, out_dir)

    if spec.plots is not None:
        spec.plots(results, args, out_dir)

    logger.info("%s done. Results in %s", spec.name.capitalize(), out_dir)


def main(specs: list[Experiment]) -> None:
    """Parse argv against the registry ``specs`` and run the selected subcommand."""
    parser = build_parser(specs)
    args = parser.parse_args()
    arm_hang_watchdog(args.hang_watchdog)
    by_name = {spec.name: spec for spec in specs}
    run_experiment(by_name[args.experiment], args)
