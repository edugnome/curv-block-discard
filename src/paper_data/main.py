"""Generate the current article resources from explicit scalar result bundles."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .config import DEFAULT_RESULTS_DIR, OPTIONAL_EXPERIMENTS, DataPaths
from .crossing import crossing_summary
from .diagnostics import complete_proposal_decomposition
from .loader import load, load_optional
from .manifests import TRACKS, labels_inputs, preflight_b
from .registry import REGISTRY
from .residual_ablation import validated_ablation

log = logging.getLogger(__name__)


def _sanitise_filename(label: str) -> str:
    """Convert a LaTeX label to a safe filename."""
    return label.replace(":", "_").replace(" ", "_") + ".tex"


def generate_all(
    paths: DataPaths,
    output_dir: Path,
    only: list[str] | None = None,
    *,
    track: str | None = None,
) -> None:
    """Run generation pipeline and write.tex files."""
    labels = (
        (TRACKS[track] if track is not None else list(REGISTRY))
        if only is None
        else only
    )
    unknown = sorted(set(labels) - set(REGISTRY))
    if unknown:
        raise ValueError(f"Unknown reporting labels: {unknown}")
    if track is None and only is None:
        track = "b"
    retained = preflight_b(paths) if track == "b" else None
    if any("residual_ablation_v2" in REGISTRY[label][1] for label in labels):
        validated_ablation(load(paths, "residual_ablation_v2"))
    if retained is None and "fig:proposal-model-loss" in labels:
        complete_proposal_decomposition(load(paths, "exp7"))
    if retained is None and any(
        label in {"tab:crossing-check", "num:crossing-check", "fig:crossing-screen"}
        for label in labels
    ):
        # Validate the required crossing artifact before creating output directories.
        crossing_summary(load(paths, "exp8_crossing_check"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        fn, experiments = REGISTRY[label]
        data: list[dict[str, Any] | None] = [
            (
                retained[e]
                if retained is not None and e in retained
                else (
                    load_optional(paths, e)
                    if e in OPTIONAL_EXPERIMENTS
                    else load(paths, e)
                )
            )
            for e in experiments
        ]
        log.info("Generating %s", label)
        tex = fn(*data)
        out_path = output_dir / _sanitise_filename(label)
        out_path.write_text(tex + "\n", encoding="utf-8")
        log.info("  -> %s", out_path)


def _build_overrides(raw: list[str]) -> dict[str, Path]:
    """Parse ``key=path`` override pairs from CLI."""
    overrides: dict[str, Path] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be key=path, got: {item!r}")
        key, val = item.split("=", 1)
        overrides[key.strip()] = Path(val.strip())
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and TikZ figures from results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing per-experiment subdirectories "
        f"(default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/reporting"),
        help="Directory to write .tex files to.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Generate only these labels (e.g. tab:stage-b fig:validity).",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Generate only the labels of this paper track (see paper_data.manifests).",
    )
    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help="Print the result bundles the selection depends on, and exit.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override experiment paths: exp_name=/path/to/results.json",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_labels",
        help="List available labels and exit.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    if args.track is not None:
        if args.only is not None:
            parser.error("--track and --only select the same thing; pass one")
        try:
            args.only = TRACKS[args.track]
        except KeyError:
            parser.error(f"unknown track {args.track!r}; known: {sorted(TRACKS)}")
        if args.output_dir == Path("build/reporting"):
            args.output_dir = args.output_dir / args.track

    if args.list_inputs:
        for experiment in labels_inputs(
            args.only if args.only is not None else list(REGISTRY)
        ):
            print(experiment)
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if args.list_labels:
        for label in REGISTRY:
            _, exps = REGISTRY[label]
            print(f"  {label:30s}  ({', '.join(exps)})")
        return

    overrides = _build_overrides(args.override)
    paths = DataPaths(results_dir=args.results_dir, overrides=overrides)
    generate_all(paths, args.output_dir, args.only, track=args.track)
    log.info(
        "Done. %d items generated.",
        len(REGISTRY) if args.only is None else len(args.only),
    )


if __name__ == "__main__":
    main()
