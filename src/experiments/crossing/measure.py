"""Subprocess-capped saved-weight check. No training or resume path.

Importing this module and requesting --help are metadata-only. Heavy imports live
only behind the explicit execution guard in the private cell subprocess.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from experiments.crossing.verify import (
    CAPS,
    ENVIRONMENT,
    PYTHON,
    digest,
    metadata_check,
    object_digest,
    read_json,
    roster,
    validate_manifest,
    write_json,
)

ADAPTATION = {
    "enabled": True,
    "internal_retry_count": "UNAVAILABLE",
    "internal_chunk_history": "UNAVAILABLE",
    "exact_draw_replay_claimed": False,
    "retained_budget": "successful observations, not all attempted kernels; internal retry work is inside wall cap",
}


def launch_guard(reviewed, visible):
    if not reviewed:
        raise ValueError(
            "scientific execution requires explicit --execute-reviewed acknowledgement"
        )
    if not visible or "," in visible or visible.strip() != visible or visible == "-1":
        raise ValueError(
            "operator must assign exactly one available GPU in CUDA_VISIBLE_DEVICES"
        )


def create_output(path):
    Path(path).mkdir(exist_ok=False)


def communicate_capped(process, timeout, grace, kill_group=None):
    """communicate uses pipe-selector events. No sleep or polling loops."""
    if kill_group is None:
        kill_group = os.killpg
    timed_out, reaped = False, True
    try:
        stdout, _ = process.communicate(timeout=max(0, timeout))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            kill_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, _ = process.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                kill_group(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                stdout, _ = process.communicate(timeout=grace)
            except subprocess.TimeoutExpired as e:
                # Never start another worker if a killed process cannot be reaped.
                reaped = False
                stdout = e.output or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
    return {
        "timed_out": timed_out,
        "returncode": process.returncode,
        "stdout": stdout or "",
        "reaped": reaped,
    }


def capped_subprocess(command, timeout, grace=2, env=None):
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        result = communicate_capped(
            process, max(0, timeout - (time.monotonic() - started)), grace
        )
    except BaseException as e:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        result = communicate_capped(process, grace, grace)
        result["supervisor_interruption"] = repr(e)
        result["returncode"] = result["returncode"] if result["returncode"] else -1
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def supervise(
    root,
    manifest_hash,
    worker,
    cells,
    campaign_cap,
    cell_cap,
    clock=time.monotonic,
    finalizer=None,
    started=None,
):
    started = clock() if started is None else started
    rows = [dict(c, status="not_started", complete_pairs=0) for c in cells]
    result = {
        "schema": 1,
        "manifest_sha256": manifest_hash,
        "cells": rows,
        "complete": False,
        "caps": dict(CAPS),
        "stop_reason": None,
        "adaptation": dict(ADAPTATION),
    }
    for i, cell in enumerate(cells):
        remaining = campaign_cap - (clock() - started)
        if remaining <= 0:
            result["stop_reason"] = "campaign_cap"
            break
        path = Path(root) / cell["id"]
        try:
            path.mkdir(exist_ok=False)
            outcome = worker(cell, path, min(cell_cap, remaining))
            write_json(path / "subprocess.json", outcome)
            rows[i]["subprocess"] = {k: v for k, v in outcome.items() if k != "stdout"}
            # The pipe captures both stdout and surfaced stderr, including tracebacks.
            with open(path / "stdout-stderr.txt", "x") as f:
                f.write(outcome.get("stdout", ""))
            rows[i]["complete_pairs"] = len(list(path.glob("pair-*.json")))
            if outcome["timed_out"] or outcome["elapsed_seconds"] > min(
                cell_cap, remaining
            ):
                rows[i]["status"] = "timed_out"
                result["stop_reason"] = "cell_or_campaign_cap"
                break
            if (
                not outcome.get("reaped", True)
                or outcome["returncode"] != 0
                or not (path / "cell-complete.json").is_file()
                or rows[i]["complete_pairs"] != 30
            ):
                rows[i]["status"] = "failed"
                result["stop_reason"] = "surfaced_failure_or_incomplete_cell"
                break
            rows[i]["status"] = "complete"
        except BaseException as e:
            rows[i]["status"] = "failed"
            rows[i]["error"] = repr(e)
            rows[i]["complete_pairs"] = len(list(path.glob("pair-*.json")))
            result["stop_reason"] = "supervisor_failure"
            break
    if finalizer is not None:
        remaining = campaign_cap - (clock() - started)
        try:
            result["preservation"] = finalizer(max(0, remaining))
        except BaseException as e:
            result["preservation"] = {"status": "FAIL", "errors": [repr(e)]}
    result["elapsed_seconds"] = clock() - started
    result["complete"] = all(c["status"] == "complete" for c in rows)
    if result.get("preservation", {}).get("status", "PASS") != "PASS":
        result["complete"] = False
        result["stop_reason"] = (
            result["stop_reason"] or "preservation_unavailable_or_failed"
        )
    if result["elapsed_seconds"] > campaign_cap:
        result["complete"] = False
        result["stop_reason"] = "campaign_cap"
    if result["complete"]:
        result["stop_reason"] = "fixed_budget_complete"
    write_json(Path(root) / "campaign.json", result)
    return result


def reconstruct(raw, names):
    phi, norms = float(raw["phi"]), raw["diag_frob"]
    if list(norms) != names:
        raise ValueError("returned diagonal group names/order mismatch")
    if not math.isfinite(phi) or phi >= 1:
        raise ValueError("nonfinite/unreconstructable Phi")
    values = [float(norms[n]) for n in names]
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("invalid diagonal norm")
    d = [x * x for x in values]
    total = math.fsum(d)
    f = total / (1 - phi)
    if (
        not all(math.isfinite(x) for x in d)
        or not math.isfinite(f)
        or f <= 0
        or total <= 0
    ):
        raise ValueError("invalid reconstructed squared domain")
    # Negative noisy Phi is retained; no exact-operator bound clipping.
    return {
        "diag_sq": d,
        "full_sq": f,
        "phi_returned": phi,
        "diag_frob_returned": {n: x for n, x in zip(names, values, strict=False)},
    }


def partition_contract(groups, named_parameters, expected):
    trainable = {id(p): n for n, p in named_parameters.items() if p.requires_grad}
    claimed: set[int] = set()
    actual = []
    for name, params in groups.items():
        ids = [id(p) for p in params]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or set(ids) & claimed
            or not set(ids) <= set(trainable)
        ):
            raise ValueError("empty, overlapping or foreign parameters in partition")
        claimed.update(ids)
        actual.append(
            {
                "name": name,
                "size": sum(p.numel() for p in params),
                "parameters": [trainable[x] for x in ids],
            }
        )
    if claimed != set(trainable) or actual != expected:
        raise ValueError("partition coverage/order/size/ownership mismatch")
    return actual


def tensor_identity(name, tensor, kind):
    cpu = tensor.detach().cpu().contiguous()
    import hashlib

    return {
        "name": name,
        "kind": kind,
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
    }


def state_identity(state, parameter_names):
    return [
        tensor_identity(n, t, "parameter" if n in parameter_names else "buffer")
        for n, t in sorted(state.items())
    ]


def assert_tensor_schema(identities, expected):
    actual = [{k: t[k] for k in ("name", "shape", "kind", "dtype")} for t in identities]
    if sorted(actual, key=lambda t: t["name"]) != sorted(
        expected, key=lambda t: t["name"]
    ):
        raise ValueError(
            "checkpoint parameter/buffer name, shape or dtype mismatch (float32 required; int64 counters preserved)"
        )


def make_test_batch(m, cell, torch, torchvision):
    # This dedicated dataset object never constructs a training dataset or enables download.
    from torchvision import transforms as T

    root = Path(m["dataset"]["root"])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("data root must be verified absolute path")
    ds = torchvision.datasets.CIFAR100(
        str(root),
        train=False,
        download=False,
        transform=T.Compose(
            [T.ToTensor(), T.Normalize(m["dataset"]["mean"], m["dataset"]["std"])]
        ),
    )
    indices = next(b["indices"] for b in m["batches"] if b["id"] == cell["batch"])
    examples = [ds[i] for i in indices]
    x = torch.stack([v[0] for v in examples])
    y = torch.tensor([v[1] for v in examples], dtype=torch.int64)
    if (
        x.dtype != torch.float32
        or list(x.shape) != [32, 3, 32, 32]
        or not bool(torch.isfinite(x).all())
    ):
        raise ValueError("unexpected normalized input shape/dtype/domain")
    return x, y, indices


def cell_worker(m, cell, output):
    # Verify bytes BEFORE any heavy import, model construction or checkpoint load.
    metadata = metadata_check(m)
    if metadata["status"] != "PASS":
        raise ValueError(metadata["errors"])
    for k, v in ENVIRONMENT.items():
        if os.environ.get(k) != v:
            raise ValueError("worker environment mismatch: " + k)
    import torch
    import torchvision

    from experiments.models import SegmentedPlainResNet18, SegmentedResNet18
    from experiments.phi import measure_phi_and_diag_frob
    from experiments.regroup import Regrouped

    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled():
        raise ValueError("autocast must be disabled")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one assigned CUDA device required")
    checkpoint = next(
        p
        for p in m["checkpoints"]
        if p["arch"] == cell["model"] and p["seed"] == cell["seed"]
    )
    # Hash and deserialize the SAME open, read-only descriptor; check it again afterwards.
    with open(checkpoint["path"], "rb") as f:
        import hashlib

        h = hashlib.file_digest(f, "sha256").hexdigest()
        if h != checkpoint["sha256"]:
            raise ValueError("checkpoint changed before load")
        f.seek(0)
        state = torch.load(f, map_location="cpu", weights_only=True)
        f.seek(0)
        if hashlib.file_digest(f, "sha256").hexdigest() != h:
            raise ValueError("checkpoint changed during load")
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(t, torch.Tensor) for k, t in state.items()
    ):
        raise ValueError("expected plain tensor state_dict only")
    build = SegmentedResNet18 if cell["model"] == "resnet18" else SegmentedPlainResNet18
    model = build(num_classes=100, activation="relu")
    parameters = dict(model.named_parameters())
    expected = m["partitions"][cell["model"]]
    for name, tensor in state.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("nonfinite saved tensor: " + name)
    loaded = state_identity(state, set(parameters))
    assert_tensor_schema(loaded, expected["tensors"])
    # Strict restores both parameters and persistent buffers; no key remapping/casting repair.
    restored = model.load_state_dict(state, strict=True)
    if restored.missing_keys or restored.unexpected_keys:
        raise ValueError("strict restore mismatch")
    model.eval()
    model.to(device="cuda:0", dtype=torch.float32)
    parameters = dict(model.named_parameters())
    del state
    before = state_identity(model.state_dict(), set(parameters))
    if loaded != before:
        raise ValueError("loaded tensors changed on restore/device transfer")
    adapters: dict[str, Any] = {"stage": model, "layer": Regrouped(model, "layer")}
    groups = {
        part: partition_contract(adapter.get_param_groups(), parameters, expected[part])
        for part, adapter in adapters.items()
    }
    x, y, indices = make_test_batch(m, cell, torch, torchvision)
    targets = y.tolist()
    x, y = x.to("cuda:0"), y.to("cuda:0")
    inputs = [tensor_identity("X", x, "input"), tensor_identity("y", y, "target")]
    md, xd = object_digest(before), object_digest(inputs)
    identity = {
        "schema": 1,
        "cell": cell,
        "checkpoint": checkpoint,
        "loaded_tensors": loaded,
        "model_tensors": before,
        "model_digest": md,
        "input_tensors": inputs,
        "input_digest": xd,
        "indices": indices,
        "ordered_targets": targets,
        "groups": groups,
        "source_hashes": m["source_hashes"],
        "data_hashes": m["data_hashes"],
        "environment": {k: os.environ.get(k) for k in ENVIRONMENT},
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "autocast": torch.is_autocast_enabled(),
            "floating_dtype": str(torch.get_default_dtype()),
        },
        "provenance_limits": m["provenance_limits"],
        "adaptation": dict(ADAPTATION),
    }
    write_json(output / "cell-start.json", identity)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    for replicate in range(30):
        pair = {
            "schema": 1,
            "cell": cell["id"],
            "replicate": replicate,
            "component_hashes": {},
        }
        for part in ("stage", "layer"):
            stream = next(
                s
                for s in m["streams"]
                if s["cell"] == cell["id"]
                and s["replicate"] == replicate
                and s["partition"] == part
            )
            generator = torch.Generator(device="cuda:0")
            generator.manual_seed(stream["seed"])
            pre = generator.get_state().cpu().numpy().tobytes().hex()
            t0 = time.monotonic()
            raw = None
            try:
                state_before = object_digest(
                    state_identity(model.state_dict(), set(parameters))
                )
                input_before = object_digest(
                    [
                        tensor_identity("X", x, "input"),
                        tensor_identity("y", y, "target"),
                    ]
                )
                if state_before != md or input_before != xd:
                    raise ValueError("state/input changed before observation")
                raw = measure_phi_and_diag_frob(
                    adapters[part], x, y, loss_fn, 100, generator=generator
                )
                observation = reconstruct(raw, [g["name"] for g in groups[part]])
                state_after = object_digest(
                    state_identity(model.state_dict(), set(parameters))
                )
                input_after = object_digest(
                    [
                        tensor_identity("X", x, "input"),
                        tensor_identity("y", y, "target"),
                    ]
                )
                if state_after != md or input_after != xd:
                    raise ValueError("state/input changed during observation")
                observation.update(
                    schema=1,
                    cell=cell["id"],
                    replicate=replicate,
                    partition=part,
                    stream=stream,
                    n_probes=100,
                    groups=groups[part],
                    generator_pre=pre,
                    generator_post=generator.get_state().cpu().numpy().tobytes().hex(),
                    state_before=state_before,
                    state_after=state_after,
                    input_before=input_before,
                    input_after=input_after,
                    wall_seconds=time.monotonic() - t0,
                    adaptation=dict(ADAPTATION),
                )
                name = output / f"{part}-{replicate:02d}.json"
                write_json(name, observation)
                pair[part] = observation
                pair["component_hashes"][part] = digest(name)
                print(
                    f"{cell['id']} replicate={replicate} partition={part} saved",
                    flush=True,
                )
            except BaseException as e:
                write_json(
                    output / f"failure-{part}-{replicate:02d}.json",
                    {
                        "cell": cell["id"],
                        "replicate": replicate,
                        "partition": part,
                        "error": repr(e),
                        "raw_return_repr": repr(raw),
                        "stream": stream,
                        "generator_pre": pre,
                        "generator_post": generator.get_state()
                        .cpu()
                        .numpy()
                        .tobytes()
                        .hex(),
                        "wall_seconds": time.monotonic() - t0,
                        "adaptation": dict(ADAPTATION),
                    },
                )
                raise
        write_json(output / f"pair-{replicate:02d}.json", pair)
    after = state_identity(model.state_dict(), set(parameters))
    final_inputs = [tensor_identity("X", x, "input"), tensor_identity("y", y, "target")]
    if after != before or final_inputs != inputs:
        raise ValueError("end-of-cell identity changed")
    preservation = metadata_check(m)
    if preservation["status"] != "PASS":
        raise ValueError(preservation["errors"])
    write_json(
        output / "cell-complete.json",
        {
            "cell": cell,
            "complete_pairs": 30,
            "model_tensors": after,
            "input_tensors": final_inputs,
            "preservation": preservation,
        },
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--execute-reviewed",
        action="store_true",
        help="acknowledge responsibility for executing the measurement",
    )
    p.add_argument(
        "--worker-cell", choices=[c["id"] for c in roster()], help=argparse.SUPPRESS
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    launch_guard(args.execute_reviewed, os.environ.get("CUDA_VISIBLE_DEVICES"))
    campaign_started = time.monotonic()
    m = validate_manifest(read_json(args.manifest))
    if args.worker_cell:
        cell = next(c for c in roster() if c["id"] == args.worker_cell)
        if not (args.output.parent / "run.json").is_file():
            raise ValueError("worker cannot run without supervisor roster")
        try:
            cell_worker(m, cell, args.output)
        except BaseException as e:
            write_json(
                args.output / "worker-error.json",
                {
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "adaptation": dict(ADAPTATION),
                },
            )
            raise
        return 0
    metadata = metadata_check(m)
    if metadata["status"] != "PASS":
        raise ValueError(metadata["errors"])
    create_output(args.output)
    with (
        open(args.manifest, "rb") as source,
        open(args.output / "manifest.json", "xb") as target,
    ):
        shutil.copyfileobj(source, target)
    mh = digest(args.manifest)
    write_json(
        args.output / "run.json",
        {
            "schema": 1,
            "manifest_sha256": mh,
            "cells": [dict(c, status="not_started") for c in roster()],
            "caps": dict(CAPS),
            "adaptation": dict(ADAPTATION),
            "launch_guard": "operator explicitly supplied --execute-reviewed",
        },
    )
    environment = dict(os.environ)
    environment.update(ENVIRONMENT)

    def worker(cell, path, timeout):
        # Reserve the bounded TERM/KILL grace INSIDE both hard clock budgets.
        grace = CAPS["termination_grace_seconds"]
        if timeout <= 2 * grace:
            return {
                "timed_out": True,
                "returncode": None,
                "stdout": "insufficient remaining termination reserve",
                "elapsed_seconds": 0.0,
                "reaped": True,
            }
        command = [
            PYTHON,
            "-I",
            "-m",
            "experiments.crossing.measure",
            "--manifest",
            str(args.manifest),
            "--output",
            str(path),
            "--execute-reviewed",
            "--worker-cell",
            cell["id"],
        ]
        return capped_subprocess(
            command, timeout - 2 * grace, grace=grace, env=environment
        )

    def finalizer(remaining):
        # Even the final metadata-only hashing runs in a deadline-bounded subprocess.
        grace = CAPS["termination_grace_seconds"]
        if remaining <= 2 * grace:
            outcome = {
                "status": "UNAVAILABLE",
                "errors": ["campaign cap leaves no final hashing budget"],
            }
            write_json(args.output / "preservation.json", outcome)
            return outcome
        command = [
            PYTHON,
            "-I",
            "-m",
            "experiments.crossing.verify",
            "--manifest",
            str(args.manifest),
            "--metadata-only",
            "--output",
            str(args.output / "preservation.json"),
        ]
        execution = capped_subprocess(
            command, remaining - 2 * grace, grace=grace, env=environment
        )
        write_json(args.output / "preservation-subprocess.json", execution)
        if (
            execution["timed_out"]
            or execution["returncode"] != 0
            or not (args.output / "preservation.json").is_file()
        ):
            failure = {
                "status": "FAIL",
                "errors": ["final hash subprocess failed or hit campaign cap"],
            }
            if not (args.output / "preservation.json").exists():
                write_json(args.output / "preservation.json", failure)
            return failure
        return read_json(args.output / "preservation.json")

    # Handle graceful external termination using the same bounded worker cleanup.
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: (_ for _ in ()).throw(
            SystemExit("supervisor terminated")
        ),
    )
    result = supervise(
        args.output,
        mh,
        worker,
        roster(),
        CAPS["campaign_seconds"],
        CAPS["cell_seconds"],
        finalizer=finalizer,
        started=campaign_started,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "stop_reason": result["stop_reason"],
                "preservation": result["preservation"]["status"],
            }
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
