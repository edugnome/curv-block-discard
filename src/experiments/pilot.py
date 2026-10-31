"""Capture, probe, and summarize the approved new-trajectory causal pilot B.

Run with an existing environment, cached Cars data and pretrained weights. Output
paths must be new. The first probe is the smoke; pass its thresholds.json to every
subsequent probe to freeze numerical/reporting thresholds before fan-out.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch  # noqa: E402

from experiments.causal_diagnostics import (  # noqa: E402
    BudgetExceeded,
    capture_pilot,
    probe_capture,
    reference_capture,
    summarize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "capture", help="new plain-K-FAC trajectory; cached inputs only"
    )
    capture.add_argument("--output", type=Path, required=True, help="new capture root")
    capture.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    capture.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[20, 60, 200, 600],
        help="zero-based pre-update step indices",
    )
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--num-workers", type=int, default=2)
    capture.add_argument("--cars-root", type=Path, required=True)
    capture.add_argument(
        "--torch-home",
        type=Path,
        required=True,
        help="directory containing hub/checkpoints/resnet50-11ad3fa6.pth",
    )
    probe = commands.add_parser(
        "probe", help="all three diagnostics on one completed frozen-state directory"
    )
    probe.add_argument("--capture", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True, help="new probe directory")
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument(
        "--max-minutes",
        type=float,
        default=30,
        help="stop at measurement boundaries, not a hard kill",
    )
    probe.add_argument(
        "--thresholds",
        type=Path,
        help="smoke thresholds.json; required operationally for post-smoke fan-out",
    )
    reference = commands.add_parser(
        "reference",
        help="retained loss/gradient check only; not historical probe replay",
    )
    reference.add_argument("--capture", type=Path, required=True)
    reference.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    reference.add_argument("--device", default="cpu")
    summary = commands.add_parser(
        "summarize",
        help="include completed, failed, budget-limited and unprobed states",
    )
    summary.add_argument("--input-root", type=Path, required=True)
    summary.add_argument(
        "--output", type=Path, required=True, help="new strict-JSON file"
    )
    ablation = commands.add_parser(
        "residual-ablation", help="versioned saved-state residual-input measurement"
    )
    phases = ablation.add_subparsers(dest="phase", required=True)
    register = phases.add_parser("register", help="write exact new-source protocol")
    register.add_argument("--capture-root", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    inventory = phases.add_parser(
        "inventory", help="check nine historical capture hashes without tensor loading"
    )
    inventory.add_argument("--capture-root", type=Path, required=True)
    inventory.add_argument("--registration", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    preflight = phases.add_parser(
        "preflight", help="sequential all-nine CPU loading/shape check without HVP"
    )
    preflight.add_argument("--registration", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    freeze = phases.add_parser(
        "freeze", help="bind completed smoke artifacts; human acceptance is external"
    )
    freeze.add_argument("--smoke-registration", type=Path, required=True)
    freeze.add_argument("--thresholds", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    for phase in ("smoke", "final"):
        measure = phases.add_parser(phase, help=f"new separate {phase} output")
        measure.add_argument("--capture", type=Path, required=True)
        measure.add_argument("--registration", type=Path, required=True)
        measure.add_argument("--output", type=Path, required=True)
        measure.add_argument("--device", default="cuda:0")
        if phase == "final":
            measure.add_argument("--thresholds", type=Path, required=True)
    for phase_parser in (
        register,
        inventory,
        preflight,
        freeze,
        phases.choices["smoke"],
        phases.choices["final"],
    ):
        phase_parser.add_argument(
            "--protected-root",
            type=Path,
            help="optional data root to protect from output overlap",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        if args.command == "capture":
            capture_pilot(
                args.output,
                seeds=args.seeds,
                steps=args.steps,
                device=torch.device(args.device),
                num_workers=args.num_workers,
                cars_root=args.cars_root,
                torch_home=args.torch_home,
            )
        elif args.command == "probe":
            probe_capture(
                args.capture,
                args.output,
                device=torch.device(args.device),
                max_minutes=args.max_minutes,
                thresholds=args.thresholds,
            )
        elif args.command == "reference":
            reference_capture(
                args.capture, args.output, device=torch.device(args.device)
            )
        elif args.command == "residual-ablation":
            from experiments import causal_diagnostics as cd
            from experiments import residual_ablation as ablation

            if args.phase == "register":
                ablation.guard_output(
                    args.output, [args.capture_root], protected_root=args.protected_root
                )
                source_sha = cd.source_hashes()["sha256"]
                listed = ablation.inventory(
                    args.capture_root, ablation.registration_template(source_sha)
                )
                if listed["valid_bytes"] != listed["expected"]:
                    raise cd.InvalidProbe(
                        "capture inventory incomplete; registration refused"
                    )
                registered = ablation.registration_template(
                    source_sha, listed["entries"]
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as stream:
                    json.dump(
                        registered, stream, indent=2, sort_keys=True, allow_nan=False
                    )
                    stream.write("\n")
            elif args.phase == "freeze":
                ablation.freeze_final_registration(
                    args.smoke_registration,
                    args.thresholds,
                    args.output,
                    protected_root=args.protected_root,
                )
            elif args.phase == "preflight":
                registration = json.loads(args.registration.read_text())
                ablation.guard_output(
                    args.output,
                    [
                        Path(row["capture_path"])
                        for row in registration["capture_roster"]
                    ]
                    + [args.registration],
                    protected_root=args.protected_root,
                )
                result = ablation.load_preflight(args.registration)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as stream:
                    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
                if result["status"] != "complete":
                    return 1
            elif args.phase == "inventory":
                ablation.guard_output(
                    args.output,
                    [args.capture_root, args.registration],
                    protected_root=args.protected_root,
                )
                registration = json.loads(args.registration.read_text())
                if not ablation.validate_registration(
                    registration, cd.source_hashes()["sha256"]
                ):
                    raise cd.InvalidProbe("inventory registration/source mismatch")
                result = ablation.inventory(args.capture_root, registration)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as stream:
                    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
                if result["valid_bytes"] != result["expected"]:
                    return 1
            else:
                ablation.measure_capture(
                    args.capture,
                    args.output,
                    args.registration,
                    device=torch.device(args.device),
                    phase=args.phase,
                    thresholds=getattr(args, "thresholds", None),
                    protected_root=args.protected_root,
                )
        else:
            from experiments.residual_ablation import reject_legacy_summary

            reject_legacy_summary(args.input_root)
            result = summarize(args.input_root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects completed summaries from accidental overwrite.
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
    except BudgetExceeded as error:
        logging.error("Budget exhausted; partial artifacts retained: %s", error)
        return 3
    except Exception:
        logging.exception(
            "Pilot failed; consult the new output directory's status.json"
        )
        return 1
    logging.info("%s complete: %s", args.command, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
