#!/usr/bin/env python3
"""Analyze saved residual factors, without models, HVPs or an ambient basis.

The native screen is copied from the original result, not recomputed. In double,
G = T^T C^T C T, A = V diag(eigenvalues) V^T and P = V diag(1/(|eigenvalues|+1)) V^T.
The map ratio is the spectral norm of G^(1/2) P G^(1/2) minus
(|G^(-1/2) A G^(-1/2)|+I)^(-1), divided by the norm of the first term.
This compares represented-factor maps, not float32 execution or loss errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

ROSTER = [
    (seed, step, draw)
    for seed in (42, 43, 44)
    for step in (20, 60, 200)
    for draw in ("primary", "sensitivity")
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_roster(roster: list[tuple[int, int, str]]) -> None:
    if len(roster) != len(ROSTER) or set(roster) != set(ROSTER):
        raise ValueError("geometry roster must contain exactly the 18 registered cases")


def validate_identity(
    factor: dict[str, Any],
    artifact: dict[str, Any],
    seed: int,
    step: int,
    draw: str,
    digest: str,
) -> None:
    if artifact.get("sha256") != digest:
        raise ValueError("factor hash mismatch")
    if artifact.get("filename") != f"factored_{draw}.pt":
        raise ValueError("factor identity filename mismatch")
    for key, expected in (
        ("measurement_schema", 2),
        ("seed", seed),
        ("step", step),
        ("drawset", draw),
    ):
        if factor.get(key) != expected or artifact.get(key) != expected:
            raise ValueError(f"factor identity mismatch: {key}")
    for key in (
        "seeds",
        "capture_sha256",
        "registration_sha256",
        "current_source_sha256",
    ):
        if not factor.get(key) or factor.get(key) != artifact.get(key):
            raise ValueError(f"factor identity mismatch: {key}")


def analyze_subspace(subspace: dict[str, Any]) -> dict[str, Any]:
    columns = subspace["columns"]
    transform = subspace["transform"].double()
    eigenvalues = subspace["eigenvalues"].double()
    vectors = subspace["eigenvectors"].double()
    dim = subspace["dim"]
    if not isinstance(dim, int) or dim <= 0 or not columns:
        raise ValueError("invalid column dimension")
    offsets = [0]
    for col in columns:
        start, stop, values = col["start"], col["stop"], col["values"]
        if (
            not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start < stop <= dim
            or values.ndim != 2
            or values.shape[0] != stop - start
            or values.shape[1] == 0
        ):
            raise ValueError("invalid column block")
        if not torch.isfinite(values).all():
            raise ValueError("column values must be finite")
        offsets.append(offsets[-1] + values.shape[1])
    if (
        transform.ndim != 2
        or transform.shape[0] != offsets[-1]
        or transform.shape[1] == 0
    ):
        raise ValueError("invalid transform shape")
    rank = transform.shape[1]
    if eigenvalues.shape != (rank,) or vectors.shape != (rank, rank):
        raise ValueError("invalid eigenfactor shape")
    if not all(torch.isfinite(x).all() for x in (transform, eigenvalues, vectors)):
        raise ValueError("eigenfactors and transform must be finite")
    raw = torch.zeros((offsets[-1], offsets[-1]), dtype=torch.float64)
    for i, left in enumerate(columns):
        for j in range(i, len(columns)):
            right = columns[j]
            start, stop = (
                max(left["start"], right["start"]),
                min(left["stop"], right["stop"]),
            )
            if start >= stop:
                continue
            product = torch.zeros(
                (offsets[i + 1] - offsets[i], offsets[j + 1] - offsets[j]),
                dtype=torch.float64,
            )
            for pos in range(start, stop, 65536):
                end = min(pos + 65536, stop)
                a = left["values"][pos - left["start"] : end - left["start"]].double()
                b = right["values"][
                    pos - right["start"] : end - right["start"]
                ].double()
                product += a.T @ b
            raw[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = product
            raw[offsets[j] : offsets[j + 1], offsets[i] : offsets[i + 1]] = product.T
    gram = transform.T @ raw @ transform
    gram = (gram + gram.T) / 2
    eig, u = torch.linalg.eigh(gram)
    error = gram - torch.eye(rank, dtype=torch.float64)
    out: dict[str, Any] = {
        "status": "available",
        "raw_columns": offsets[-1],
        "retained_columns": rank,
        "gram_eigen_min": eig[0].item(),
        "gram_eigen_max": eig[-1].item(),
        "gram_max_entry_error": error.abs().max().item(),
        "gram_spectral_error": (eig - 1).abs().max().item(),
        "gram_condition": None,
        "map_relative_difference": None,
    }
    if eig[0] <= 0:
        out["status"] = "nonpositive_gram"
        return out
    out["gram_condition"] = (eig[-1] / eig[0]).item()
    root = (u * eig.sqrt()) @ u.T
    invroot = (u * eig.rsqrt()) @ u.T
    a = (vectors * eigenvalues) @ vectors.T
    p = (vectors * (eigenvalues.abs() + 1).reciprocal()) @ vectors.T
    b = invroot @ a @ invroot
    b = (b + b.T) / 2
    be, bv = torch.linalg.eigh(b)
    whitened = (bv * (be.abs() + 1).reciprocal()) @ bv.T
    saved = root @ p @ root
    denominator = torch.linalg.matrix_norm(saved, ord=2).item()
    if not math.isfinite(denominator) or denominator == 0:
        out["status"] = "undefined_map_denominator"
        return out
    out["map_relative_difference"] = (
        torch.linalg.matrix_norm(saved - whitened, ord=2) / denominator
    ).item()
    return out


def build_results(run_root: Path) -> dict[str, Any]:
    state_dirs = sorted(p for p in run_root.glob("seed_*_step_*") if p.is_dir())
    expected_dirs = {f"seed_{s}_step_{t:06d}" for s, t, _ in ROSTER}
    if {p.name for p in state_dirs} != expected_dirs:
        raise ValueError("state roster mismatch")
    rows = []
    for state in state_dirs:
        result_path, status_path = state / "results.json", state / "status.json"
        result = json.loads(result_path.read_text())
        state_status = json.loads(status_path.read_text())
        seed, step = result["seed"], result["step"]
        if (
            state.name != f"seed_{seed}_step_{step:06d}"
            or result["measurement_schema"] != 2
            or result["phase"] != "final"
            or result["status"] != "complete"
            or state_status["status"] != "complete"
        ):
            raise ValueError("state identity or status mismatch")
        if set(result["drawsets"]) != {"primary", "sensitivity"}:
            raise ValueError("draw roster mismatch")
        identity = state_status["measurement_identity"]
        for key in (
            "seed",
            "step",
            "measurement_schema",
            "capture_sha256",
            "current_source_sha256",
        ):
            if identity[key] != result[key]:
                raise ValueError(f"state identity mismatch: {key}")
        if identity["registration"]["sha256"] != result["registration"]["sha256"]:
            raise ValueError("state registration identity mismatch")
        for draw in ("primary", "sensitivity"):
            d = result["drawsets"][draw]
            artifact = d["factored_geometry"]
            for key, expected in (
                ("seed", seed),
                ("step", step),
                ("drawset", draw),
                ("seeds", artifact["seeds"]),
            ):
                if d[key] != expected:
                    raise ValueError(f"draw identity mismatch: {key}")
            for key in ("capture_sha256", "current_source_sha256"):
                if artifact[key] != result[key]:
                    raise ValueError(f"result identity mismatch: {key}")
            if artifact["registration_sha256"] != result["registration"]["sha256"]:
                raise ValueError("result registration identity mismatch")
            path = state / f"factored_{draw}.pt"
            digest = sha256(path)
            factor = torch.load(path, map_location="cpu", weights_only=True)
            validate_identity(factor, artifact, seed, step, draw, digest)
            represented = analyze_subspace(factor["subspace"])
            coverage = d["coverage"]
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "drawset": draw,
                    "factor_sha256": digest,
                    "result_sha256": sha256(result_path),
                    "status_sha256": sha256(status_path),
                    "native_screen": {
                        "status": coverage["status"],
                        "max_entry_error": coverage["orthonormality_error"],
                        "eta_action": coverage["eta_action"],
                    },
                    "represented_factor": represented,
                }
            )
            del factor
    validate_roster([(r["seed"], r["step"], r["drawset"]) for r in rows])
    return {
        "schema": 1,
        "diagnostic_script": "scripts/diagnose_residual_geometry.py",
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "interpretation": "Native screen is the original float32 project/lift result. Represented-factor Gram and map ratio use saved factors in float64; they are not native arithmetic or loss error bounds.",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = build_results(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
