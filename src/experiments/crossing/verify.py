"""Frozen specification and metadata/record verification; stdlib only, no tensors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
DOMAIN = "dag-hesse:B:crossing:20261020:v1"
BATCHES = [
    {"id": "B0", "indices": list(range(32))},
    {"id": "B1", "indices": list(range(32, 64))},
]
STATISTICS = {
    "resamples": 10000,
    "seed": 20261020,
    "rng": "numpy.PCG64",
    "arithmetic": "float64 CPU",
    "resampling": "complete paired vectors; independent cells",
    "marginal": "percentile",
    "probabilities": [0.025, 0.975],
    "quantile": "linear at (n-1)*p",
    "sd_ddof": 1,
    "family_size": 36,
    "screen_level": 0.95,
    "screen": "max(abs(bootstrap-observed)/fixed bootstrap SD)",
    "jackknife": "diagnostic only; no bias correction",
    "invalid_family": "whole family unavailable",
}
ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
CAPS = {
    "cell_seconds": 900,
    "campaign_seconds": 10800,
    "termination_grace_seconds": 2,
    "kill_grace_seconds": 2,
    "one_gpu": True,
    "one_model_resident": True,
    "resume": False,
    "retry_failed_drawsets": False,
}


def roster():
    return [
        {"id": f"s{seed}-{arch}-{batch}", "seed": seed, "model": arch, "batch": batch}
        for seed in (42, 43, 44)
        for arch in ("resnet18", "plain_resnet18")
        for batch in ("B0", "B1")
    ]


def streams():
    result = []
    for c in roster():
        for r in range(30):
            for partition in ("stage", "layer"):
                name = f"{DOMAIN}/{c['id']}/r{r:02d}/{partition}"
                seed = int.from_bytes(
                    hashlib.sha256(name.encode()).digest()[:8], "big"
                ) & ((1 << 63) - 1)
                result.append(
                    {
                        "id": name,
                        "cell": c["id"],
                        "replicate": r,
                        "partition": partition,
                        "seed": seed,
                    }
                )
    return result


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path):
    def duplicates(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError(f"duplicate JSON key: {k}")
            d[k] = v
        return d

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=duplicates,
        parse_constant=lambda s: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON: {s}")
        ),
    )


def write_json(path, value):
    """Publish a complete JSON record atomically and exclusively, never replace."""
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".unpublished-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # atomic EEXIST rather than overwrite
    finally:
        os.unlink(temporary)


def check_hashes(ledger):
    errors = []
    for path, expected in ledger.items():
        try:
            if digest(path) != expected:
                errors.append(f"hash mismatch: {path}")
        except OSError as e:
            errors.append(f"missing/unreadable: {path}: {e}")
    return errors


def architecture_contract(arch):
    """Source-derived shape contract only. Does not instantiate a module or tensor."""
    if arch not in ("resnet18", "plain_resnet18"):
        raise ValueError("unknown architecture")
    tensors = []

    def add(name, shape, kind="parameter", dtype="torch.float32"):
        tensors.append({"name": name, "shape": shape, "kind": kind, "dtype": dtype})

    def bn(name, channels):
        for field in ("weight", "bias"):
            add(name + "." + field, [channels])
        for field in ("running_mean", "running_var"):
            add(name + "." + field, [channels], "buffer")
        add(name + ".num_batches_tracked", [], "buffer", "torch.int64")

    add("_base.conv1.weight", [64, 3, 7, 7])
    bn("_base.bn1", 64)
    inc = 64
    for layer, out in enumerate((64, 128, 256, 512), 1):
        for block in range(2):
            name = f"_base.layer{layer}.{block}"
            add(name + ".conv1.weight", [out, inc, 3, 3])
            bn(name + ".bn1", out)
            add(name + ".conv2.weight", [out, out, 3, 3])
            bn(name + ".bn2", out)
            if layer > 1 and block == 0 and arch == "resnet18":
                add(name + ".downsample.0.weight", [out, inc, 1, 1])
                bn(name + ".downsample.1", out)
            inc = out
    add("_base.fc.weight", [100, 512])
    add("_base.fc.bias", [100])
    params = [t for t in tensors if t["kind"] == "parameter"]
    stage_names = ["stem", "layer1", "layer2", "layer3", "layer4", "head"]

    def stage(t):
        n = t["name"]
        return (
            "head"
            if n.startswith("_base.fc.")
            else next(
                (x for x in stage_names[1:5] if n.startswith("_base." + x + ".")),
                "stem",
            )
        )

    def group(name, items):
        return {
            "name": name,
            "size": sum(math.prod(t["shape"]) for t in items),
            "parameters": [t["name"] for t in items],
        }

    stages = [group(n, [t for t in params if stage(t) == n]) for n in stage_names]
    layers = []
    claimed: set[str] = set()
    for t in params:
        name = t["name"].rsplit(".", 1)[0]
        if len(t["shape"]) in (2, 4):
            items = [x for x in params if x["name"].rsplit(".", 1)[0] == name]
            layers.append(group(name, items))
            claimed.update(x["name"] for x in items)
    layers.append(group("rest", [t for t in params if t["name"] not in claimed]))
    return {
        "tensors": tensors,
        "stage": stages,
        "layer": layers,
        "parameter_count": sum(math.prod(t["shape"]) for t in params),
    }


def manifest_contract(
    data_root: Path | str, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Construct the fixed scientific protocol with caller-supplied input paths."""
    contract = {
        "schema": 2,
        "python": sys.executable,
        "cells": roster(),
        "replicates": 30,
        "probes": 100,
        "batches": BATCHES,
        "streams": streams(),
        "statistics": STATISTICS,
        "caps": CAPS,
        "environment": ENVIRONMENT,
        "checkpoints": checkpoints,
        "partitions": {
            a: architecture_contract(a) for a in ("resnet18", "plain_resnet18")
        },
        "dataset": {
            "name": "CIFAR-100",
            "root": str(Path(data_root).resolve()),
            "train": False,
            "download": False,
            "mean": [0.5071, 0.4865, 0.4409],
            "std": [0.2673, 0.2564, 0.2762],
            "augmentation": False,
            "batch_size": 32,
            "target": "fine_labels",
        },
        "arithmetic": {
            "parameters": "torch.float32",
            "input": "torch.float32",
            "hvp": "torch.float32",
            "integer_buffers": "preserve and record torch.int64 BatchNorm counters",
            "autocast": False,
            "tf32": False,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "loss": "eval mean cross_entropy",
        },
    }
    return copy.deepcopy(contract)


def scientific_source_hashes() -> dict[str, str]:
    """Identify the actual installed scientific modules, without importing them."""
    root = Path(__file__).resolve().parents[2]
    paths = [
        p
        for package in ("hessian", "secondorder", "experiments")
        for p in (root / package).rglob("*.py")
    ]
    return {str(p): digest(p) for p in sorted(paths)}


def build_manifest(
    data_root: Path, checkpoints: list[tuple[int, str, Path]]
) -> dict[str, Any]:
    """Record actual input bytes and installed source versions; never load tensors."""
    rows = [
        {
            "arch": arch,
            "seed": seed,
            "epoch": 300,
            "path": str(Path(path).resolve()),
            "sha256": digest(path),
        }
        for seed, arch, path in checkpoints
    ]
    m = manifest_contract(data_root, sorted(rows, key=lambda r: (r["seed"], r["arch"])))
    m["source_hashes"] = scientific_source_hashes()
    inputs = [
        Path(data_root).resolve() / "cifar-100-python" / name
        for name in ("test", "meta")
    ]
    m["data_hashes"] = {str(p): digest(p) for p in inputs}
    m["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy")
    }
    m["provenance_limits"] = (
        "New observation from supplied saved weights; not a replay of historical random draws."
    )
    return validate_manifest(m)


def validate_manifest(m: dict[str, Any]) -> dict[str, Any]:
    checkpoints = m.get("checkpoints", [])
    expected_cohort = {
        (seed, arch) for seed in (42, 43, 44) for arch in ("resnet18", "plain_resnet18")
    }
    if (
        len(checkpoints) != 6
        or {(r["seed"], r["arch"]) for r in checkpoints} != expected_cohort
    ):
        raise ValueError("checkpoint cohort must contain exactly six model/seed pairs")
    for row in checkpoints:
        if (
            row["epoch"] != 300
            or not Path(row["path"]).is_absolute()
            or len(row["sha256"]) != 64
        ):
            raise ValueError("invalid saved checkpoint metadata")
    expected = manifest_contract(m["dataset"]["root"], checkpoints)
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"manifest violates frozen contract: {key}")
    if len({s["seed"] for s in m["streams"]}) != 720:
        raise ValueError("stream seed collision")
    for key in ("source_hashes", "data_hashes", "packages"):
        if not m.get(key):
            raise ValueError(f"missing manifest identity: {key}")
    if set(m["source_hashes"]) != set(scientific_source_hashes()):
        raise ValueError(
            "source identity does not cover the installed scientific modules"
        )
    required = {
        str(Path(m["dataset"]["root"]) / "cifar-100-python" / name)
        for name in ("test", "meta")
    }
    if set(m["data_hashes"]) != required:
        raise ValueError("dataset identity must cover test and meta files")
    return m


def metadata_check(m: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(m)
    ledger = dict(m["source_hashes"])
    ledger.update(m["data_hashes"])
    ledger.update({x["path"]: x["sha256"] for x in m["checkpoints"]})
    errors = check_hashes(ledger)
    if str(Path(sys.executable).absolute()) != m["python"]:
        errors.append("unexpected interpreter: " + sys.executable)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy")
    }
    if packages != m["packages"]:
        errors.append("package versions differ from manifest")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "files_hashed": len(ledger),
        "checkpoint_count": 6,
        "stream_count": 720,
        "cell_count": 12,
        "packages_metadata_only": packages,
        "tensor_or_dataset_loading": False,
        "model_construction": False,
        "cuda_calls": False,
    }


def validate_component(o, m, cell, replicate, partition, identity):
    expected_stream = next(
        s
        for s in m["streams"]
        if s["cell"] == cell["id"]
        and s["replicate"] == replicate
        and s["partition"] == partition
    )
    for k, x in (
        ("cell", cell["id"]),
        ("replicate", replicate),
        ("partition", partition),
        ("stream", expected_stream),
        ("n_probes", 100),
    ):
        if o.get(k) != x:
            raise ValueError(f"component {k} mismatch")
    groups = m["partitions"][cell["model"]][partition]
    if o["groups"] != groups:
        raise ValueError("partition contract mismatch")
    d, f, phi, norms = (
        o["diag_sq"],
        o["full_sq"],
        o["phi_returned"],
        o["diag_frob_returned"],
    )
    if len(d) != len(groups) or list(norms) != [g["name"] for g in groups]:
        raise ValueError("component dimension/order mismatch")
    if (
        not all(math.isfinite(x) and x >= 0 for x in d)
        or not math.isfinite(f)
        or f <= 0
    ):
        raise ValueError("invalid squared domain")
    if not math.isfinite(phi) or phi >= 1 or sum(d) <= 0:
        raise ValueError("unreconstructable Phi")
    for sq, norm in zip(d, norms.values(), strict=False):
        if (
            not math.isfinite(norm)
            or norm < 0
            or not math.isclose(sq, norm * norm, rel_tol=1e-10, abs_tol=0)
        ):
            raise ValueError("norm reconstruction mismatch")
    if not math.isclose(f, math.fsum(d) / (1 - phi), rel_tol=1e-10, abs_tol=0):
        raise ValueError("full reconstruction mismatch")
    for k in ("state_before", "state_after"):
        if o[k] != identity["model_digest"]:
            raise ValueError("model changed during observation")
    for k in ("input_before", "input_after"):
        if o[k] != identity["input_digest"]:
            raise ValueError("input changed during observation")
    for k in ("generator_pre", "generator_post"):
        raw = bytes.fromhex(o[k])
        if not raw:
            raise ValueError("empty generator state")
    if o["generator_pre"] == o["generator_post"]:
        raise ValueError("generator did not advance")
    if (
        o["adaptation"]["internal_retry_count"] != "UNAVAILABLE"
        or o["adaptation"]["exact_draw_replay_claimed"]
    ):
        raise ValueError("unsupported adaptation claim")
    if not math.isfinite(o["wall_seconds"]) or o["wall_seconds"] < 0:
        raise ValueError("invalid observation time")


def read_run(m, run_dir, manifest_hash):
    """Read all fixed cells, retaining partial statuses and every integrity error."""
    run_dir = Path(run_dir)
    campaign = read_json(run_dir / "campaign.json")
    initial = read_json(run_dir / "run.json")
    if (
        campaign["manifest_sha256"] != manifest_hash
        or initial["manifest_sha256"] != manifest_hash
    ):
        raise ValueError("run/manifest hash mismatch")
    if digest(run_dir / "manifest.json") != manifest_hash:
        raise ValueError("run manifest copy mismatch")
    if [x["id"] for x in campaign["cells"]] != [x["id"] for x in roster()]:
        raise ValueError("campaign cell roster/order mismatch")
    if (
        initial["cells"] != [dict(c, status="not_started") for c in roster()]
        or initial["caps"] != CAPS
    ):
        raise ValueError("initial roster/caps mismatch")
    if campaign["caps"] != CAPS:
        raise ValueError("campaign caps mismatch")
    global_errors = []
    preservation = read_json(run_dir / "preservation.json")
    if (
        preservation["status"] != "PASS"
        or campaign.get("preservation", {}).get("status") != "PASS"
    ):
        global_errors.append("campaign preservation unavailable/failed")
    if campaign["complete"] != (
        all(c["status"] == "complete" for c in campaign["cells"]) and not global_errors
    ):
        global_errors.append("campaign completeness contradicts roster")
    cells = []
    checkpoint_tensors: dict[tuple[int, str], Any] = {}
    batch_tensors: dict[str, Any] = {}
    stopped = False
    for c, status in zip(roster(), campaign["cells"], strict=False):
        d = run_dir / c["id"]
        pairs: list[dict[str, Any]] = []
        errors = list(global_errors)
        orphans: list[str] = []
        if status["status"] not in ("complete", "failed", "timed_out", "not_started"):
            errors.append("unknown cell status")
        if stopped and status["status"] != "not_started":
            errors.append("started cell after stopping boundary")
        if status["status"] != "complete":
            stopped = True
        row = dict(
            c,
            status=status["status"],
            pairs=pairs,
            integrity_errors=errors,
            orphan_records=orphans,
        )
        cells.append(row)
        try:
            identity = (
                read_json(d / "cell-start.json")
                if d.exists() and (d / "cell-start.json").exists()
                else None
            )
            files = sorted(d.glob("pair-*.json")) if d.exists() else []
            if files and identity is None:
                raise ValueError("missing cell identity")
            if identity:
                if identity["cell"] != c:
                    raise ValueError("cell-start identity mismatch")
                if (
                    identity["source_hashes"] != m["source_hashes"]
                    or identity["data_hashes"] != m["data_hashes"]
                ):
                    raise ValueError("recorded source/data identity mismatch")
                if identity["environment"] != ENVIRONMENT:
                    raise ValueError("recorded environment mismatch")
                runtime = identity["runtime"]
                for key, expected_value in {
                    "deterministic_algorithms": True,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                    "matmul_tf32": False,
                    "cudnn_tf32": False,
                    "autocast": False,
                    "floating_dtype": "torch.float32",
                }.items():
                    if runtime.get(key) != expected_value:
                        raise ValueError("recorded arithmetic flag mismatch: " + key)
                if identity["groups"] != {
                    p: m["partitions"][c["model"]][p] for p in ("stage", "layer")
                }:
                    raise ValueError("recorded group ownership mismatch")
                checkpoint = next(
                    x
                    for x in m["checkpoints"]
                    if x["arch"] == c["model"] and x["seed"] == c["seed"]
                )
                if identity["checkpoint"] != checkpoint:
                    raise ValueError("loaded checkpoint identity mismatch")
                if identity["model_digest"] != object_digest(identity["model_tensors"]):
                    raise ValueError("model tensor digest mismatch")
                if identity["loaded_tensors"] != identity["model_tensors"]:
                    raise ValueError("loaded/model tensors differ")
                if identity["input_digest"] != object_digest(identity["input_tensors"]):
                    raise ValueError("input digest mismatch")
                if identity["indices"] != next(
                    b["indices"] for b in BATCHES if b["id"] == c["batch"]
                ):
                    raise ValueError("input indices mismatch")
                expected = m["partitions"][c["model"]]["tensors"]
                observed = [
                    {k: t[k] for k in ("name", "shape", "kind", "dtype")}
                    for t in identity["model_tensors"]
                ]
                if sorted(observed, key=lambda x: x["name"]) != sorted(
                    expected, key=lambda x: x["name"]
                ):
                    raise ValueError("model tensor schema mismatch")
                targets = identity["ordered_targets"]
                if len(targets) != 32 or any(
                    type(v) is not int or not 0 <= v < 100 for v in targets
                ):
                    raise ValueError("invalid ordered targets")
                inputs = identity["input_tensors"]
                schemas = [
                    {
                        "name": "X",
                        "shape": [32, 3, 32, 32],
                        "dtype": "torch.float32",
                        "kind": "input",
                    },
                    {
                        "name": "y",
                        "shape": [32],
                        "dtype": "torch.int64",
                        "kind": "target",
                    },
                ]
                if [
                    {k: t[k] for k in ("name", "shape", "dtype", "kind")}
                    for t in inputs
                ] != schemas:
                    raise ValueError("input dtype/shape/order mismatch")
                if (
                    hashlib.sha256(struct.pack("<32q", *targets)).hexdigest()
                    != inputs[1]["sha256"]
                ):
                    raise ValueError("ordered targets differ from y content hash")
                for tensor in identity["model_tensors"] + inputs:
                    if (
                        len(tensor["sha256"]) != 64
                        or len(bytes.fromhex(tensor["sha256"])) != 32
                    ):
                        raise ValueError("invalid tensor content hash")
                checkpoint_key = (c["seed"], c["model"])
                if (
                    checkpoint_key in checkpoint_tensors
                    and checkpoint_tensors[checkpoint_key] != identity["model_tensors"]
                ):
                    raise ValueError("same checkpoint differs across batches")
                checkpoint_tensors[checkpoint_key] = identity["model_tensors"]
                if c["batch"] in batch_tensors and batch_tensors[c["batch"]] != inputs:
                    raise ValueError("same fixed batch differs across models/seeds")
                batch_tensors[c["batch"]] = inputs
            for index, p in enumerate(files):
                if p.name != f"pair-{index:02d}.json":
                    raise ValueError("noncontiguous complete pairs")
                pair = read_json(p)
                if pair["replicate"] != index or pair["cell"] != c["id"]:
                    raise ValueError("pair identity mismatch")
                for part in ("stage", "layer"):
                    name = d / f"{part}-{index:02d}.json"
                    if digest(name) != pair["component_hashes"][part]:
                        raise ValueError("pair component hash mismatch")
                    if read_json(name) != pair[part]:
                        raise ValueError("pair component differs from stage record")
                    validate_component(pair[part], m, c, index, part, identity)
                pairs.append(pair)
            for part in ("stage", "layer"):
                for p in sorted(d.glob(part + "-*.json")) if d.exists() else []:
                    if p.name not in {
                        f"{part}-{i:02d}.json" for i in range(len(pairs))
                    }:
                        orphans.append(str(p))
            if len(pairs) > 30:
                raise ValueError("excess pairs")
            if status["status"] == "complete":
                end = read_json(d / "cell-complete.json")
                if (
                    end["cell"] != c
                    or end["complete_pairs"] != 30
                    or end["preservation"]["status"] != "PASS"
                ):
                    raise ValueError("invalid cell-complete attestation")
                if (
                    identity is None
                    or len(pairs) != 30
                    or end["model_tensors"] != identity["model_tensors"]
                    or end["input_tensors"] != identity["input_tensors"]
                ):
                    raise ValueError("incomplete/changed cell marked complete")
                if orphans:
                    raise ValueError("orphan in complete cell")
            if status["complete_pairs"] != len(pairs):
                raise ValueError("campaign/record pair-count mismatch")
            if errors:
                row["status"] = "invalid"
            if status["status"] == "not_started" and d.exists():
                raise ValueError("not-started cell has output directory")
        except (OSError, ValueError, KeyError, TypeError) as e:
            errors.append(str(e))
            row["status"] = "invalid"
    return cells, campaign


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument(
        "--prepare",
        action="store_true",
        help="write a manifest from explicit inputs, without loading tensors",
    )
    p.add_argument("--data-root", type=Path)
    p.add_argument(
        "--checkpoint", action="append", default=[], metavar="SEED:MODEL=PATH"
    )
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--analysis", type=Path)
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.prepare:
        if (
            args.manifest
            or args.metadata_only
            or args.run_dir
            or args.analysis
            or not args.data_root
            or not args.output
        ):
            raise ValueError(
                "--prepare requires --data-root, six --checkpoint values and --output only"
            )
        rows = []
        for raw in args.checkpoint:
            identity, path = raw.split("=", 1)
            seed, model = identity.split(":", 1)
            rows.append((int(seed), model, Path(path)))
        write_json(args.output, build_manifest(args.data_root, rows))
        return 0
    if not args.manifest:
        raise ValueError("--manifest is required unless --prepare is selected")
    if args.metadata_only and (args.run_dir or args.analysis):
        raise ValueError("metadata-only does not inspect records")
    if not args.metadata_only and not args.run_dir:
        raise ValueError("provide --metadata-only or --run-dir")
    m = read_json(args.manifest)
    result = metadata_check(m)
    if args.run_dir:
        cells, campaign = read_run(m, args.run_dir, digest(args.manifest))
        result["cells"] = [
            {k: v for k, v in c.items() if k != "pairs"}
            | {"complete_pairs": len(c["pairs"])}
            for c in cells
        ]
        result["complete"] = all(
            c["status"] == "complete" and len(c["pairs"]) == 30 for c in cells
        )
        result["errors"] += [
            f"{c['id']}: {e}" for c in cells for e in c["integrity_errors"]
        ]
        result["N02"] = (
            "OPEN: requires uncertainty analysis and independent result review"
            if result["complete"]
            else "OPEN: incomplete run"
        )
        if args.analysis:
            from experiments.crossing.analysis import analyze

            observed = read_json(args.analysis)
            recomputed = analyze(cells)
            if (
                observed.get("analysis") != recomputed
                or observed.get("manifest_sha256") != digest(args.manifest)
                or observed.get("campaign_sha256")
                != digest(args.run_dir / "campaign.json")
            ):
                result["errors"].append(
                    "analysis is not reproducible from fixed records"
                )
            result["family_screen_status"] = recomputed["family"]["status"]
            result["N02"] = recomputed["N02"]
        if result["errors"]:
            result["status"] = "FAIL"
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
