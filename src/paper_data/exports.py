"""Export readable observation families from the retained article corpus.

Finite floats round-trip without rounding. Nonfinite values use explicit tagged
objects in strict JSONL; canonical source bytes remain untouched. No tensors or
archive members are extracted or executed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import DataPaths
from .diagnostics import complete_proposal_decomposition
from .loader import load
from .manifests import preflight_b
from .residual_ablation import residual_contrast_rows


def encode_nonfinite(value: Any) -> Any:
    """Keep missing, undefined and signed infinite observations distinguishable."""
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "nonfinite": "NaN"
            if math.isnan(value)
            else "+Infinity"
            if value > 0
            else "-Infinity"
        }
    if isinstance(value, dict):
        return {key: encode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_nonfinite(item) for item in value]
    return value


def scientific_checkpoint_epochs(observation: dict[str, Any]) -> dict[str, Any]:
    """Retain checkpoint epochs without retaining artifact paths."""
    checkpoints = observation.get("checkpoints")
    if checkpoints is None:
        return dict(observation)
    if (
        not isinstance(checkpoints, dict)
        or any(
            not isinstance(epoch, str) or not epoch.isdecimal() for epoch in checkpoints
        )
        or "checkpoint_epochs" in observation
    ):
        raise ValueError("invalid checkpoint epoch map")
    return {
        **{key: value for key, value in observation.items() if key != "checkpoints"},
        "checkpoint_epochs": list(checkpoints),
    }


def seed_rows(bundle: str, data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """One unaveraged row per explicit seed and model/checkpoint cell."""

    def walk(node: Any, cell: list[str]) -> Iterator[dict[str, Any]]:
        if not isinstance(node, dict):
            return
        if isinstance(node.get("seeds"), dict):
            for label, observation in sorted(node["seeds"].items()):
                if re.fullmatch(r"seed_\d+", label) is None:
                    raise ValueError(f"ambiguous seed label: {label}")
                yield {
                    "bundle": bundle,
                    "cell": cell,
                    "seed": int(label[5:]),
                    "observation": observation,
                }
            return
        for key, value in sorted(node.items()):
            if key != "meta":
                yield from walk(value, [*cell, key])

    yield from walk(data, [])


def validity_rows(bundle: str, data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Retain trajectory identity; the seam repeat is not an independent seed."""
    direction = {"exp11_sgd": "sgd", "exp11_adam": "adam"}.get(bundle, "kfac")
    for row in seed_rows(bundle, data):
        steps = set()
        for index, probe in enumerate(row["observation"]["probes"]):
            step = probe["step"]
            if step in steps:
                raise ValueError(
                    f"duplicate step in {bundle}/seed_{row['seed']}: {step}"
                )
            steps.add(step)
            yield {
                "bundle": bundle,
                "seed": row["seed"],
                "direction": direction,
                "step": step,
                "index": index,
                "observation": probe,
            }


def crossing_pairs(archive: Path) -> Iterator[dict[str, Any]]:
    """Read the fixed 360 paired observations directly from their exact archive."""
    pattern = re.compile(
        r"crossing/(s(42|43|44)-(resnet18|plain_resnet18)-(B0|B1))/pair-(\d{2})\.json"
    )
    seen = set()
    rows = []
    with tarfile.open(archive, "r:gz") as source:
        for member in sorted(source.getmembers(), key=lambda item: item.name):
            match = pattern.fullmatch(member.name)
            if match is None:
                continue
            cell, seed, model, batch, index = match.groups()
            identity = (cell, int(index))
            if identity in seen:
                raise ValueError(f"duplicate crossing pair: {identity}")
            if not member.isfile() or int(index) not in range(30):
                raise ValueError(f"invalid crossing member: {member.name}")
            seen.add(identity)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"missing crossing member: {member.name}")
            with stream:
                observation = json.load(stream)
            if observation["cell"] != cell or observation["replicate"] != int(index):
                raise ValueError(f"crossing identity mismatch: {member.name}")
            rows.append(
                {
                    "bundle": "exp8_crossing_check",
                    "cell": cell,
                    "seed": int(seed),
                    "model": model,
                    "batch": batch,
                    "index": int(index),
                    "observation": observation,
                }
            )
    if len(seen) != 360:
        raise ValueError(f"360 complete crossing pairs required, got {len(seen)}")
    yield from rows


def public_crossing_pairs(path: Path) -> Iterator[dict[str, Any]]:
    """Read and validate a complete public paired-component cohort."""
    expected = {
        (f"s{seed}-{model}-{batch}", index)
        for seed in (42, 43, 44)
        for model in ("resnet18", "plain_resnet18")
        for batch in ("B0", "B1")
        for index in range(30)
    }
    rows = []
    seen = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.pop("format") != "curv-scientific-crossing-pair-v1":
                    raise ValueError("invalid public crossing format")
                cell, index = row["cell"], row["index"]
                identity = (cell, index)
                if (
                    type(index) is not int
                    or identity not in expected
                    or identity in seen
                    or row["bundle"] != "exp8_crossing_check"
                    or cell != f"s{row['seed']}-{row['model']}-{row['batch']}"
                ):
                    raise ValueError("invalid public crossing identity")
                seen.add(identity)
                pair = row["observation"]
                if (
                    pair["schema"] != 1
                    or pair["cell"] != cell
                    or pair["replicate"] != index
                ):
                    raise ValueError("invalid paired observation identity")
                for partition in ("stage", "layer"):
                    component = pair[partition]
                    groups = component["groups"]
                    diag = component["diag_sq"]
                    count = (
                        6
                        if partition == "stage"
                        else (22 if row["model"] == "resnet18" else 19)
                    )
                    if (
                        component["schema"] != 1
                        or component["cell"] != cell
                        or component["replicate"] != index
                        or component["partition"] != partition
                        or component["n_probes"] != 100
                        or len(groups) != count
                        or len(diag) != count
                        or len({g["name"] for g in groups}) != count
                        or any(
                            type(g["size"]) is not int or g["size"] <= 0 for g in groups
                        )
                        or any(
                            type(v) not in (int, float) or not math.isfinite(v) or v < 0
                            for v in diag
                        )
                        or type(component["full_sq"]) not in (int, float)
                        or not math.isfinite(component["full_sq"])
                        or component["full_sq"] <= 0
                    ):
                        raise ValueError("invalid paired component")
                    norms = component["diag_frob_returned"]
                    phi = component["phi_returned"]
                    if (
                        set(norms) != {g["name"] for g in groups}
                        or type(phi) not in (int, float)
                        or not math.isfinite(phi)
                        or any(
                            not isinstance(g["parameters"], list) or not g["parameters"]
                            for g in groups
                        )
                        or any(
                            type(norms[g["name"]]) not in (int, float)
                            or not math.isfinite(norms[g["name"]])
                            or norms[g["name"]] < 0
                            or not math.isclose(
                                norms[g["name"]] ** 2, value, rel_tol=1e-5, abs_tol=1e-8
                            )
                            for g, value in zip(groups, diag, strict=True)
                        )
                        or not math.isclose(
                            phi,
                            1 - math.fsum(diag) / component["full_sq"],
                            rel_tol=1e-5,
                            abs_tol=1e-7,
                        )
                    ):
                        raise ValueError("inconsistent paired component values")
                    random_stream = component["stream"]
                    if (
                        random_stream["cell"] != cell
                        or random_stream["replicate"] != index
                        or random_stream["partition"] != partition
                        or type(random_stream["seed"]) is not int
                    ):
                        raise ValueError("invalid paired component stream")
                rows.append(row)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid public crossing records") from exc
    if seen != expected:
        raise ValueError("incomplete public crossing cohort")
    yield from rows


def reconcile_crossing_pairs(
    rows: list[dict[str, Any]], bundle: dict[str, Any]
) -> None:
    """Reconcile paired squared components with each reporting cell's pools."""
    import numpy as np

    from experiments.crossing.analysis import _arrays

    cells = bundle["source_analysis"]["analysis"]["cells"]
    grouped: dict[str, dict[int, dict[str, Any]]] = {cell["id"]: {} for cell in cells}
    try:
        for row in rows:
            cell, index = row["cell"], row["index"]
            if (
                cell not in grouped
                or type(index) is not int
                or index not in range(30)
                or index in grouped[cell]
            ):
                raise ValueError("invalid paired cohort identity")
            pair = row["observation"]
            if pair["cell"] != cell or pair["replicate"] != index:
                raise ValueError("invalid paired observation identity")
            grouped[cell][index] = pair
        for cell in cells:
            pairs = grouped[cell["id"]]
            if set(pairs) != set(range(30)):
                raise ValueError("incomplete paired cohort")
            for partition in ("stage", "layer"):
                components = [pairs[index][partition] for index in range(30)]
                diagonal, full = _arrays(components)
                pooled = cell["pooled_components"][partition]
                actual_diagonal: list[float] = np.asarray(
                    np.mean(diagonal, axis=0, dtype=np.float64)
                ).tolist()
                actual_full = float(np.mean(full, dtype=np.float64))
                expected_diagonal = pooled["diag_sq"]
                if (
                    len(actual_diagonal) != len(expected_diagonal)
                    or not all(
                        math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
                        for a, b in zip(actual_diagonal, expected_diagonal, strict=True)
                    )
                    or not math.isclose(
                        actual_full, pooled["full_sq"], rel_tol=1e-12, abs_tol=0.0
                    )
                ):
                    raise ValueError("paired components disagree with reporting pools")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid paired component records") from exc


def export_tables(data_root: Path, output: Path) -> dict[str, int]:
    """Validate reporting inputs before creating family-level files."""
    bundles = preflight_b(DataPaths(data_root / "reporting"))
    crossing = bundles["exp8_crossing_check"]
    if crossing.get("format") == "curv-scientific-crossing-v1":
        pairs = list(public_crossing_pairs(data_root / "records/crossing-pairs.jsonl"))
    else:
        pairs = list(crossing_pairs(data_root / "records/crossing-primary.tar.gz"))
    reconcile_crossing_pairs(pairs, crossing)
    tables: dict[str, list[dict[str, Any]]] = {
        "phi": [],
        "correction": [],
        "conditioning": [],
        "validity": [],
        "pilot": [],
        "crossing-summary": [],
        "crossing-pairs": pairs,
    }
    for bundle, data in sorted(bundles.items()):
        if bundle.startswith("exp11"):
            tables["validity"].extend(validity_rows(bundle, data))
        elif bundle == "exp10":
            tables["conditioning"].extend(seed_rows(bundle, data))
        elif bundle == "exp7":
            tables["correction"].extend(seed_rows(bundle, data))
        elif bundle == "causal_pilot":
            tables["pilot"] = [
                {
                    "bundle": bundle,
                    "seed": row["seed"],
                    "step": row["step"],
                    "observation": row,
                }
                for row in data["rows"]
            ]
        elif bundle == "exp8_crossing_check":
            tables["crossing-summary"] = [
                {
                    "bundle": bundle,
                    "cell": row["id"],
                    "seed": row["seed"],
                    "model": row["model"],
                    "batch": row["batch"],
                    "observation": row,
                }
                for row in data["source_analysis"]["analysis"]["cells"]
            ]
        else:
            tables["phi"].extend(seed_rows(bundle, data))
    output.mkdir(parents=True, exist_ok=False)
    for family, rows in tables.items():
        with (output / f"{family}.jsonl").open("x", encoding="ascii") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        encode_nonfinite(row), ensure_ascii=True, allow_nan=False
                    )
                    + "\n"
                )
    return {family: len(rows) for family, rows in tables.items()}


def export_proposal_rows(data_root: Path, output: Path) -> dict[str, int]:
    """Explicit flat derived export; never alters the seven raw family files."""
    analysis = complete_proposal_decomposition(
        load(DataPaths(data_root / "reporting"), "exp7")
    )
    fields = (
        "input_bundle",
        "arm",
        "seed",
        "step",
        "accepted",
        "rejected_by",
        "damping",
        "damping_source",
        "delta_m",
        "delta_penalty",
        "delta_q",
        "delta_loss",
        "binary32_sign_screen",
        "borderline_q",
    )
    raw_fields = (
        "m_kfac",
        "m_proposal",
        "norm_d0",
        "norm_d1",
        "loss_d0",
        "loss_d1",
    )
    rows = [
        {
            **{key: row[key] for key in fields},
            **{key: row["raw"][key] for key in raw_fields},
        }
        for method in ("kfac+overlay", "ekfac+overlay")
        for row in analysis["arms"][method]["records"]
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "proposal-decomposition.jsonl").open(
        "x", encoding="ascii"
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    return {"proposal-decomposition": len(rows)}


def export_residual_contrasts(data_root: Path, output: Path) -> dict[str, int]:
    """Export only retrospective scalar contrasts from the compact bundle."""
    bundle = load(DataPaths(data_root / "reporting"), "residual_ablation_v2")
    rows = list(residual_contrast_rows(bundle))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "residual-contrasts.jsonl").open("x", encoding="ascii") as stream:
        for row in rows:
            stream.write(json.dumps(encode_nonfinite(row), allow_nan=False) + "\n")
    return {
        "scale": sum(row["type"] == "scale" for row in rows),
        "matched_full_rest": sum(row["type"] == "matched_full_rest" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--derived-proposals",
        action="store_true",
        help="write only the separate flat proposal-decomposition.jsonl export",
    )
    modes.add_argument(
        "--residual-contrasts",
        action="store_true",
        help="write only the retrospective residual-contrasts.jsonl export",
    )
    args = parser.parse_args()
    export = (
        export_residual_contrasts
        if args.residual_contrasts
        else export_proposal_rows
        if args.derived_proposals
        else export_tables
    )
    print(json.dumps(export(args.data_root, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
