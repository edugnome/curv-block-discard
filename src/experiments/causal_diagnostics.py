"""Frozen-state causal diagnostics for pilot B, not historical trajectory replay.

The trajectory applies only plain K-FAC's d0. Probes own a hook-free diagnostic
copy, use the saved batch/d0, and retain at most one curvature graph. Artifacts
contain tensors and primitive containers only; neither graphs nor optimizers are
pickled. No data/model downloads, dispatching, or remote execution live here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from experiments.config import Exp7Config, Exp11Config, TrainingConfig
from experiments.param_ops import (
    apply_flat,
    group_major_params,
    trial_forward,
    trial_loss,
)
from experiments.provenance import config_to_dict, environment_block
from experiments.training import StepCtx, Trainer
from experiments.utils import set_seed
from hessian import ParamBlockEstimator
from hessian.memory import run_peak_allocated
from hessian.param_space import ParamCurvature
from secondorder.overlay import CoupleFacOverlay, OverlaySubspace
from secondorder.safeguard import TrustRegionSafeguard, model_value

SCHEMA = 1
SCALE_GRID = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
RAY_GRID = tuple(Exp11Config().t_grid)
PHASE = "after_provider_refresh_before_gradient_forward"


@dataclass(frozen=True)
class Tolerances:
    grad_rtol: float = 1e-4
    hvp_rtol: float = 1e-3
    atol: float = 1e-6
    gain_floor: float = 1e-4
    correction_relative_floor: float = 1e-3


_DEFAULT_TOLERANCES = Tolerances()


class BudgetExceeded(RuntimeError):
    """A measurement boundary reached the declared wall-time budget."""


class InvalidProbe(RuntimeError):
    """Technical invalidity, never a scientific negative result."""


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write strict JSON; NaN/Infinity are errors, not scientific data."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def save_tensors(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(_cpu(payload), temporary)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tree_hash(root: Path, files: list[Path] | None = None) -> dict[str, Any]:
    paths = sorted(
        files if files is not None else (p for p in root.rglob("*") if p.is_file())
    )
    entries = {
        str(path.relative_to(root)): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "n_files": len(entries), "files": entries}


def source_hashes() -> dict[str, Any]:
    """Hash the actual installed first-party Python sources used by the pilot."""
    root = Path(__file__).resolve().parents[1]
    files = [
        p
        for package in ("hessian", "secondorder", "experiments", "paper_data")
        for p in (root / package).rglob("*.py")
    ]
    return tree_hash(root, files)


class ArtifactDirectory:
    """Exclusive output directory with persistent success/failure/budget status.

    Refusing *any* existing directory is deliberately stronger than protecting only
    completed scientific artifacts. A retry always gets a new destination.
    """

    def __init__(self, path: Path, *, kind: str, max_minutes: float | None = None):
        self.path = Path(path)
        if max_minutes is not None and (
            not math.isfinite(max_minutes) or max_minutes <= 0
        ):
            raise ValueError("max_minutes must be finite and positive")
        self.deadline_seconds = None if max_minutes is None else 60 * max_minutes
        self.start = time.monotonic()
        self.status: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": kind,
            "status": "running",
            "started_at": _utc(),
            "stage": "initializing",
            "stages": {},
        }

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        self.flush()
        return self

    def flush(self) -> None:
        self.status["elapsed_seconds"] = time.monotonic() - self.start
        write_json(self.path / "status.json", self.status)

    def _check_budget(self) -> None:
        if (
            self.deadline_seconds is not None
            and time.monotonic() - self.start >= self.deadline_seconds
        ):
            raise BudgetExceeded("wall-time budget exhausted at a measurement boundary")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.status["stage"] = name
        self.flush()
        self._check_budget()
        start = time.monotonic()
        try:
            yield
        finally:
            self.status["stages"][name] = {"seconds": time.monotonic() - start}
            self.flush()
        self._check_budget()

    def __exit__(self, exc_type, exc, tb):
        self.status["status"] = (
            "complete"
            if exc is None
            else (
                "budget_exhausted"
                if isinstance(exc, BudgetExceeded)
                else "technical_failure"
            )
        )
        self.status["ended_at"] = _utc()
        if exc is not None:
            self.status["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            }
        self.flush()
        return False


def execution_state(device: torch.device | None = None) -> dict[str, Any]:
    """Primitive/CPU-tensor RNG and arithmetic flags, safe for weights_only=True.

    Only the working CUDA device is touched. A probe may map that RNG state to a
    different logical device; masking GPUs for one-process-per-GPU is supported.
    """
    numpy_state: Any = np.random.get_state(legacy=True)
    cuda_device = None
    if device is not None and device.type == "cuda":
        cuda_device = (
            torch.cuda.current_device() if device.index is None else device.index
        )
    elif device is None and torch.cuda.is_initialized():
        cuda_device = torch.cuda.current_device()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            torch.cuda.get_rng_state(cuda_device).cpu().clone()
            if cuda_device is not None
            else None
        ),
        "cuda_device": cuda_device,
        "backend": {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def _restore_execution(
    state: Mapping[str, Any], device: torch.device | None = None
) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None and (device is None or device.type == "cuda"):
        target = state["cuda_device"] if device is None else device
        torch.cuda.set_rng_state(cuda_state.cpu(), device=target)
    b = state["backend"]
    torch.use_deterministic_algorithms(b["deterministic"], warn_only=b["warn_only"])
    torch.backends.cudnn.deterministic = b["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = b["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = b["cudnn_tf32"]
    torch.set_float32_matmul_precision(b["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = b["matmul_tf32"]
    workspace = b["cublas_workspace_config"]
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


@contextmanager
def isolated_execution(
    state: Mapping[str, Any], *, device: torch.device | None = None
) -> Iterator[None]:
    previous = execution_state(device)
    try:
        _restore_execution(state, device)
        yield
    finally:
        _restore_execution(previous, device)


def parameter_layout(model: Any) -> list[dict[str, Any]]:
    names = {id(p): name for name, p in model.named_parameters()}
    layout = []
    offset = 0
    for group, params in model.get_param_groups().items():
        size = sum(p.numel() for p in params)
        layout.append(
            {
                "group": group,
                "start": offset,
                "stop": offset + size,
                "parameters": [
                    {
                        "name": names[id(p)],
                        "shape": list(p.shape),
                        "dtype": str(p.dtype),
                    }
                    for p in params
                ],
            }
        )
        offset += size
    return layout


def restore_model(model: Any, payload: Mapping[str, Any]) -> None:
    model.load_state_dict(payload["state_dict"], strict=True)
    modules = dict(model.named_modules())
    if set(modules) != set(payload["modes"]):
        raise InvalidProbe("module paths differ from capture")
    for name, training in payload["modes"].items():
        modules[name].training = training
    if parameter_layout(model) != payload["layout"]:
        raise InvalidProbe("group-major parameter layout differs from capture")


def build_capture_hook(
    model: Any,
    protocol: Exp7Config,
    *,
    seed: int,
    steps: list[int],
    sink: Callable[[dict[str, Any]], None],
    provider: Any = None,
) -> Callable[[StepCtx], None]:
    """provider refresh -> pre-gradient snapshot -> g/d0 -> write -> apply d0.

    Snapshotting is tensor copying/serialization only, with no extra live forward.
    The saved buffers precede the original gradient-forward, not its BN update.
    """
    if provider is None:
        from secondorder.kfac_adapter import CurvlinopsKFACProvider

        provider = CurvlinopsKFACProvider(
            model,
            nn.CrossEntropyLoss(),
            damping=protocol.damping,
            fisher_type=protocol.fisher_type,
            refresh_period=protocol.kfac_refresh_period,
        )
    estimator = ParamBlockEstimator(model, nn.CrossEntropyLoss())
    params = group_major_params(model)
    selected = set(steps)

    def hook(ctx: StepCtx) -> None:
        provider.update(ctx.x, ctx.y)
        payload = None
        if ctx.step in selected:
            payload = {
                "schema": SCHEMA,
                "seed": seed,
                "step": ctx.step,
                "phase": PHASE,
                "trajectory": "new_plain_kfac_d0_only",
                "config": config_to_dict(protocol),
                "state_dict": _cpu(model.state_dict()),
                "modes": {
                    name: module.training for name, module in model.named_modules()
                },
                "layout": parameter_layout(model),
                "x": _cpu(ctx.x),
                "y": _cpu(ctx.y),
                "runtime": execution_state(ctx.device),
            }
        cur = estimator.curvature(ctx.x, ctx.y)
        d0 = provider.base_direction(cur.flat_grad)
        _finite(cur.loss, cur.flat_grad, d0)
        if payload is not None:
            payload.update(
                {"loss0": cur.loss, "g0": _cpu(cur.flat_grad), "d0": _cpu(d0)}
            )
        del cur
        if payload is not None:
            # A sink cannot perturb the subsequent training RNG/backend flags.
            with isolated_execution(execution_state(ctx.device), device=ctx.device):
                sink(payload)
        apply_flat(params, d0, 1.0)

    return hook


def write_capture(
    path: Path, payload: dict[str, Any], *, provenance: dict[str, Any]
) -> None:
    with ArtifactDirectory(path, kind="frozen_state") as run:
        with run.stage("serialize"):
            save_tensors(run.path / "state.pt", payload)
            state_hashes = {
                name: tensor_sha256(value)
                for name, value in payload["state_dict"].items()
            }
            manifest = {
                "schema": SCHEMA,
                "seed": payload["seed"],
                "step": payload["step"],
                "phase": payload["phase"],
                "trajectory": payload["trajectory"],
                "sha256": file_sha256(run.path / "state.pt"),
                "state_hashes": state_hashes,
                "batch_hashes": {
                    key: tensor_sha256(payload[key]) for key in ("x", "y")
                },
                "config": payload["config"],
                "provenance": provenance,
            }
            write_json(run.path / "manifest.json", manifest)


def load_capture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    status = json.loads((path / "status.json").read_text())
    if status["status"] != "complete" or status["kind"] != "frozen_state":
        raise InvalidProbe("capture is not a completed frozen-state artifact")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["sha256"] != file_sha256(path / "state.pt"):
        raise ValueError("capture hash mismatch")
    payload = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if payload["schema"] != SCHEMA or payload["phase"] != PHASE:
        raise InvalidProbe("unsupported capture schema or snapshot phase")
    for key in ("x", "y"):
        if tensor_sha256(payload[key]) != manifest["batch_hashes"][key]:
            raise InvalidProbe("batch hash mismatch")
    return payload, manifest


class _BufferedModel:
    """Estimator seam: live differentiable parameters, private per-forward buffers."""

    def __init__(self, model: Any):
        self.model = model

    def get_param_groups(self):
        return self.model.get_param_groups()

    def __call__(self, x: Tensor) -> Tensor:
        buffers = {name: value.clone() for name, value in self.model.named_buffers()}
        return torch.func.functional_call(self.model, buffers, (x,))


def _finite(*values: Any) -> None:
    for value in values:
        valid = (
            bool(torch.isfinite(value).all())
            if isinstance(value, Tensor)
            else math.isfinite(float(value))
        )
        if not valid:
            raise InvalidProbe("nonfinite primary quantity")


def agreement(
    actual: Tensor | float, expected: Tensor | float, *, rtol: float, atol: float
) -> dict[str, Any]:
    a = (
        actual.detach().cpu().double()
        if isinstance(actual, Tensor)
        else torch.tensor(actual, dtype=torch.float64)
    )
    b = (
        expected.detach().cpu().double()
        if isinstance(expected, Tensor)
        else torch.tensor(expected, dtype=torch.float64)
    )
    _finite(a, b)
    error = float((a - b).norm())
    reference = float(b.norm())
    return {
        "passed": error <= atol + rtol * reference,
        "error_norm": error,
        "reference_norm": reference,
        "rtol": rtol,
        "atol": atol,
    }


def replay_curvature(
    model: Any,
    payload: dict[str, Any],
    *,
    n_probes: int,
    generator: torch.Generator,
    tolerances: Tolerances = _DEFAULT_TOLERANCES,
) -> tuple[ParamCurvature, dict[str, Any]]:
    device = next(model.parameters()).device
    x, y = payload["x"].to(device), payload["y"].to(device)
    estimator = ParamBlockEstimator(
        _BufferedModel(model),
        nn.CrossEntropyLoss(),
        n_probes=n_probes,
        generator=generator,
    )
    with isolated_execution(payload["runtime"], device=device):
        cur = estimator.curvature(x, y)
    checks = {
        "loss": agreement(
            cur.loss, payload["loss0"], rtol=tolerances.grad_rtol, atol=tolerances.atol
        ),
        "gradient": agreement(
            cur.flat_grad,
            payload["g0"],
            rtol=tolerances.grad_rtol,
            atol=tolerances.atol,
        ),
    }
    if not all(check["passed"] for check in checks.values()):
        raise InvalidProbe(f"capture replay mismatch: {checks}")
    return cur, checks


def reference_capture(
    capture: Path,
    output: Path,
    *,
    device: torch.device,
    model_factory: Callable[[Exp7Config], Any] | None = None,
    tolerances: Tolerances = _DEFAULT_TOLERANCES,
) -> dict[str, Any]:
    """Check retained loss/gradient only, not the source-locked historical probe.

    Uses embedded inputs and saved modes/runtime. The K-FAC direction is retained,
    not recomputed; no HVPs, overlay interventions, optimizer or training run here.
    A source mismatch is exposed, never treated as historical probe compatibility.
    """
    with (
        ArtifactDirectory(output, kind="retained_state_reference") as run,
        isolated_execution(execution_state(device), device=device),
    ):
        with run.stage("retained_reference"):
            payload, manifest = load_capture(capture)
            source = manifest["provenance"].get("source")
            current_source = source_hashes()
            protocol = _config_from_payload(payload)
            if model_factory is None:
                from experiments.runner_exp7 import build_model

                model_factory = build_model
            with isolated_execution(payload["runtime"], device=device):
                model = model_factory(replace(protocol, pretrained=False)).to(
                    device=device, dtype=payload["d0"].dtype
                )
            restore_model(model, payload)
            d0 = payload["d0"].to(device)
            _finite(d0)
            if d0.shape != payload["g0"].shape:
                raise InvalidProbe("stored direction/gradient shape mismatch")
            cur, checks = replay_curvature(
                model,
                payload,
                n_probes=1,
                generator=torch.Generator(device=device).manual_seed(0),
                tolerances=tolerances,
            )
            result = {
                "schema": SCHEMA,
                "scope": "retained_state_loss_gradient_with_stored_d0",
                "historical_probe_replay": False,
                "d0_recomputed": False,
                "source_fingerprint_match": (
                    source["sha256"] == current_source["sha256"]
                    if source is not None
                    else None
                ),
                "capture_sha256": manifest["sha256"],
                "seed": payload["seed"],
                "step": payload["step"],
                "device": str(device),
                "dtype": str(d0.dtype),
                "checks": checks,
                "loss": cur.loss,
                "stored_d0_norm": float(d0.norm()),
                "gradient_dot_stored_d0": float(cur.flat_grad @ d0),
                "environment": environment_block(),
            }
            del cur
            write_json(run.path / "results.json", result)
        return result


def displacement_curvature(
    forward: Callable[[Tensor, Tensor], Tensor],
    loss_fn: nn.Module,
    x: Tensor,
    y: Tensor,
    displacement: Tensor,
) -> tuple[float, Tensor, Callable[[Tensor], Tensor]]:
    """Differentiate the flat displacement, never the detached live Parameters."""
    point = displacement.detach().clone().requires_grad_(True)
    loss = loss_fn(forward(point, x), y)
    grad = torch.autograd.grad(loss, point, create_graph=True)[0]

    def hvp(vector: Tensor) -> Tensor:
        if not grad.requires_grad:
            return torch.zeros_like(point)
        result = torch.autograd.grad(
            grad, point, grad_outputs=vector, retain_graph=True, allow_unused=True
        )[0]
        return torch.zeros_like(point) if result is None else result.detach()

    return float(loss.detach()), grad.detach(), hvp


def reduce_original(
    grad: Tensor, d0: Tensor, hd0: Tensor, subspace: OverlaySubspace, *, damping: float
) -> dict[str, Any]:
    return {
        "g_d0": float(grad.dot(d0)),
        "d0_h_d0": float(d0.dot(hd0)),
        "norm_d0_sq": float(d0.dot(d0)),
        "damping": damping,
        "m0": model_value(grad, d0, hd0, damping=damping),
        "u_g": subspace.project(grad).detach(),
        "u_hd0": subspace.project(hd0).detach(),
        "u_d0": subspace.project(d0).detach(),
        "h": subspace.reduced.detach(),
    }


def original_model_values(
    reduced: Mapping[str, Any],
    c: Tensor,
    *,
    norm_sq: float,
    damped_model: float | None = None,
) -> dict[str, float]:
    """Initial-pass reduced decomposition, optionally with a full replayed H0 value.

    Full/reduced arithmetic and separate H0 replays need not be bitwise equal.
    full_minus_reduced records their combined discrepancy, not just roundoff;
    strict gates compare full values with the baseline from the same replay.
    """
    correction = float(
        (reduced["u_g"] + reduced["u_hd0"]).dot(c) + 0.5 * c.dot(reduced["h"] @ c)
    )
    undamped = reduced["g_d0"] + 0.5 * reduced["d0_h_d0"] + correction
    damping = 0.5 * reduced["damping"] * norm_sq
    if damped_model is None:
        model_delta = correction + 0.5 * reduced["damping"] * (
            norm_sq - reduced["norm_d0_sq"]
        )
        damped_model = reduced["m0"] + model_delta
    return {
        "undamped": undamped,
        "damping": damping,
        "damped": damped_model,
        "full_minus_reduced": damped_model - (undamped + damping),
    }


def _sf_solve(h: Tensor, residual: Tensor, damping: float) -> Tensor:
    if not residual.numel():
        return residual.clone()
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (h + h.t()))
    return -(
        eigenvectors @ ((eigenvectors.t() @ residual) / (eigenvalues.abs() + damping))
    )


def refresh_coordinates(
    reduced: dict[str, Any], endpoint_gradient: Tensor, endpoint_h: Tensor
) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    """Four solves in the same U and lambda, including lambda*d0 after refresh."""
    damping = reduced["damping"]
    old_r = reduced["u_g"] + reduced["u_hd0"] + damping * reduced["u_d0"]
    new_r = endpoint_gradient + damping * reduced["u_d0"]
    systems = {
        "original": {"h": reduced["h"], "residual": old_r},
        "gradient_refresh": {"h": reduced["h"], "residual": new_r},
        "curvature_refresh": {"h": endpoint_h, "residual": old_r},
        "both_refresh": {"h": endpoint_h, "residual": new_r},
    }
    return {
        name: _sf_solve(system["h"], system["residual"], damping)
        for name, system in systems.items()
    }, systems


def add_controls(
    coordinates: dict[str, Tensor], subspace: OverlaySubspace
) -> tuple[dict[str, Tensor], dict[str, dict[str, str]]]:
    candidates = dict(coordinates)
    original = coordinates["original"]
    for i, alpha in enumerate(SCALE_GRID):
        candidates[f"scale_{i}"] = alpha * original
    pairs = {}
    norm0 = float(subspace.lift(original).norm())
    for name, c in coordinates.items():
        if name == "original":
            continue
        norm1 = float(subspace.lift(c).norm())
        target = min(norm0, norm1)
        left, right = f"matched_{name}_original", f"matched_{name}_refreshed"
        candidates[left] = original * (target / norm0 if norm0 else 0.0)
        candidates[right] = c * (target / norm1 if norm1 else 0.0)
        pairs[name] = {"original": left, "refreshed": right}
    return candidates, pairs


def preference(gain: float, floor: float) -> str:
    _finite(gain)
    return "tie" if abs(gain) <= floor else ("correction" if gain > 0 else "base")


def _gates(
    norm: float,
    norm0: float,
    model_delta: float,
    loss: float,
    loss0: float,
    tr_ratio: float,
) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "ratio": norm <= tr_ratio * norm0,
        "model": model_delta <= TrustRegionSafeguard().tol,
        "loss": loss <= loss0,
    }
    gates["accepted"] = all(gates.values())
    gates["rejected_by"] = next(
        (key for key in ("ratio", "model", "loss") if not gates[key]), None
    )
    return gates


def measure_candidates(
    d0: Tensor,
    subspace: OverlaySubspace,
    candidates: dict[str, Tensor],
    reduced: dict[str, Any],
    systems: dict[str, dict[str, Tensor]],
    loss_at: Callable[[Tensor], float],
    *,
    tr_ratio: float,
    gain_floor: float,
    correction_relative_floor: float = 1e-3,
    original_models: Mapping[str, float] | None = None,
    original_model_baseline: float | None = None,
) -> dict[str, Any]:
    """All gates/losses without short-circuiting; full H0 values share one baseline.

    Numerical replay tolerances never relax these gates. Without original_models
    this is an initial-pass anchored reduced-model diagnostic only.
    """
    if (original_models is None) != (original_model_baseline is None):
        raise ValueError(
            "original_models and their same-replay baseline are required together"
        )
    baseline = (
        reduced["m0"] if original_model_baseline is None else original_model_baseline
    )
    _finite(baseline)
    loss0 = loss_at(d0)
    norm0 = float(d0.norm())
    rows = {}
    for name, c in candidates.items():
        delta = subspace.lift(c)
        direction = d0 + delta
        norm, correction_norm = float(direction.norm()), float(delta.norm())
        values = original_model_values(
            reduced,
            c,
            norm_sq=float(direction.dot(direction)),
            damped_model=None if original_models is None else original_models[name],
        )
        loss = loss_at(direction)
        _finite(loss0, loss, norm, correction_norm, *values.values())
        gain = loss0 - loss
        model_delta = values["damped"] - baseline
        row = {
            "loss": loss,
            "loss_d0": loss0,
            "actual_gain": gain,
            "preference": preference(gain, gain_floor),
            "step_norm": norm,
            "correction_norm": correction_norm,
            "near_zero_correction": correction_norm < correction_relative_floor * norm0
            or correction_norm == 0,
            "model": values,
            "model_evaluation": {
                "source": (
                    "replayed_model_value_H0"
                    if original_models is not None
                    else "anchored_reduced_only"
                ),
                "tensor_dtype": str(d0.dtype),
                "baseline": baseline,
                "model_delta": model_delta,
            },
            "original_gates": _gates(norm, norm0, model_delta, loss, loss0, tr_ratio),
        }
        family = name
        if name.startswith("scale_") or (
            name.startswith("matched_") and name.endswith("_original")
        ):
            family = "original"
        elif name.startswith("matched_") and name.endswith("_refreshed"):
            family = name[len("matched_") : -len("_refreshed")]
        system = systems.get(family)
        if system is not None:
            updated_delta = float(
                system["residual"].dot(c)
                + 0.5 * c.dot(system["h"] @ c)
                + 0.5 * reduced["damping"] * delta.dot(delta)
            )
            row["updated_model_delta"] = updated_delta
            row["updated_gates"] = _gates(
                norm, norm0, updated_delta, loss, loss0, tr_ratio
            )
        else:
            row["updated_gates"] = {"status": "not_applicable"}
        rows[name] = row
    return rows


def comparison_decomposition(
    reduced: dict[str, Any],
    c: Tensor,
    d0: Tensor,
    delta: Tensor,
    *,
    loss_d0: float,
    loss_d1: float,
    endpoint_slope: float,
    gain_floor: float,
) -> dict[str, Any]:
    predicted_slope = float((reduced["u_g"] + reduced["u_hd0"]).dot(c))
    quadratic = 0.5 * float(c.dot(reduced["h"] @ c))
    model_difference = predicted_slope + quadratic
    actual_difference = loss_d1 - loss_d0
    damping_difference = (
        0.5
        * reduced["damping"]
        * float((d0 + delta).square().sum() - d0.square().sum())
    )
    transport = endpoint_slope - predicted_slope
    remainder = actual_difference - endpoint_slope - quadratic
    actual_gain, model_gain = -actual_difference, -model_difference
    return {
        "actual_gain": actual_gain,
        "undamped_gain": model_gain,
        "damping_gain": -damping_difference,
        "damped_gain": model_gain - damping_difference,
        "predicted_endpoint_slope": predicted_slope,
        "predicted_endpoint_slope_damped": predicted_slope
        + reduced["damping"] * float(d0.dot(delta)),
        "endpoint_slope": endpoint_slope,
        "transport_error": transport,
        "finite_step_remainder": remainder,
        "actual_minus_undamped_model": actual_difference - model_difference,
        "actual_preference": preference(actual_gain, gain_floor),
        "undamped_preference": preference(model_gain, gain_floor),
        "damped_preference": preference(model_gain - damping_difference, gain_floor),
        "ranking_error": (
            None
            if "tie"
            in (
                preference(actual_gain, gain_floor),
                preference(model_gain - damping_difference, gain_floor),
            )
            else preference(actual_gain, gain_floor)
            != preference(model_gain - damping_difference, gain_floor)
        ),
    }


def summarize(input_root: Path) -> dict[str, Any]:
    """Keep every selected capture and technical failure; do not infer unrun effects."""
    input_root = Path(input_root)
    captures, probes, capture_runs = [], [], []
    expected_states: list[dict[str, Any]] = []
    statuses = []
    for status_path in sorted(input_root.rglob("status.json")):
        status = json.loads(status_path.read_text())
        statuses.append(status["status"])
        folder = status_path.parent
        if status["kind"] == "capture":
            capture_runs.append(
                {
                    "path": str(folder.resolve()),
                    "status": status["status"],
                    "error": status.get("error"),
                }
            )
            provenance_path = folder / "provenance.json"
            registered = (
                json.loads(provenance_path.read_text())
                if provenance_path.exists()
                else status.get("requested", {})
            )
            expected_states.extend(
                {
                    "seed": seed,
                    "step": step,
                    "capture_root": str(folder.resolve()),
                    "capture_status": "not_run",
                    "probe_status": "not_run",
                }
                for seed in registered.get("seeds", [])
                for step in registered.get("steps", [])
            )
        elif status["kind"] == "frozen_state":
            manifest_path = folder / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            )
            captures.append(
                {
                    "path": str(folder.resolve()),
                    "status": status["status"],
                    "error": status.get("error"),
                    "seed": manifest.get("seed"),
                    "step": manifest.get("step"),
                    "sha256": manifest.get("sha256"),
                    "probe_status": "not_run",
                }
            )
        elif status["kind"] == "probe":
            result_path = folder / "results.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            probes.append(
                {
                    "path": str(folder.resolve()),
                    "status": status["status"],
                    "error": status.get("error"),
                    "result": result,
                }
            )
    for capture in captures:
        matched = [
            p
            for p in probes
            if capture["sha256"] is not None
            and p["result"].get("capture_sha256") == capture["sha256"]
        ]
        if matched:
            capture["probe_status"] = [p["status"] for p in matched]
    for state in expected_states:
        expected_path = (
            Path(state["capture_root"])
            / f"seed_{state['seed']}"
            / f"step_{state['step']:06d}"
        )
        matching_captures = [c for c in captures if Path(c["path"]) == expected_path]
        if matching_captures:
            state["capture_status"] = matching_captures[0]["status"]
            state["probe_status"] = matching_captures[0]["probe_status"]
    return {
        "schema": SCHEMA,
        "created_at": _utc(),
        "expected_states": expected_states,
        "captures": captures,
        "capture_runs": capture_runs,
        "probes": probes,
        "counts": {
            "captured": sum(c["status"] == "complete" for c in captures),
            "complete_probes": sum(p["status"] == "complete" for p in probes),
            "technical_failure": statuses.count("technical_failure"),
            "budget_exhausted": statuses.count("budget_exhausted"),
        },
        "scientific_conclusion": "undetermined",
        "independent_verification": "not_performed_by_this_cli",
    }


def _config_from_payload(payload: dict[str, Any]) -> Exp7Config:
    values = dict(payload["config"])
    values["training"] = TrainingConfig(**values["training"])
    return Exp7Config(**values)


def _subspace_payload(subspace: OverlaySubspace) -> dict[str, Any]:
    return {
        "pairs": list(subspace.pairs),
        "dim": subspace.dim,
        "columns": [
            {"start": sl.start, "stop": sl.stop, "values": values}
            for sl, values in subspace.columns
        ],
        "transform": subspace.transform,
        "eigenvalues": subspace.eigenvalues,
        "eigenvectors": subspace.eigenvectors,
    }


def _subspace_to(subspace: OverlaySubspace, device: torch.device) -> OverlaySubspace:
    """Transfer factors only, never materialize U or change its arithmetic dtype."""
    return OverlaySubspace(
        subspace.pairs,
        tuple((sl, values.to(device)) for sl, values in subspace.columns),
        subspace.transform.to(device),
        subspace.dim,
        subspace.eigenvalues.to(device),
        subspace.eigenvectors.to(device),
    )


def _empty_subspace(d0: Tensor) -> OverlaySubspace:
    # One empty block keeps the existing factored project/lift API well-defined.
    empty = d0.new_empty((0, 0))
    return OverlaySubspace(
        (), ((slice(0, 0), empty),), empty, d0.numel(), d0.new_empty(0), empty
    )


def _geometry(
    model: Any, payload: dict[str, Any], protocol: Exp7Config, tolerances: Tolerances
) -> dict[str, Any]:
    device = next(model.parameters()).device
    seed = int(payload["seed"])
    seeds = {
        "coupling": seed + 100_003,
        "overlay": seed + 200_003,
        "jet": seed + 300_007,
    }
    cur, checks = replay_curvature(
        model,
        payload,
        n_probes=protocol.coupling_n_probes_online,
        generator=torch.Generator(device=device).manual_seed(seeds["coupling"]),
        tolerances=tolerances,
    )
    d0 = payload["d0"].to(device)
    overlay = CoupleFacOverlay(
        rank=protocol.overlay_rank,
        n_power_iter=protocol.overlay_n_power_iter,
        damping=protocol.damping,
        max_pairs=protocol.overlay_max_pairs,
    )
    coupling = cur.coupling()
    _finite(*coupling.values())
    selected = overlay.select(coupling, protocol.tau)
    ranges = {
        row["group"]: slice(row["start"], row["stop"]) for row in payload["layout"]
    }
    subspace = overlay.build(
        d0,
        cur.hvp,
        ranges,
        selected,
        generator=torch.Generator(device=device).manual_seed(seeds["overlay"]),
    )
    if subspace is None:
        subspace = _empty_subspace(d0)
    hd0 = cur.hvp(d0)
    reduced = reduce_original(
        cur.flat_grad, d0, hd0, subspace, damping=protocol.damping
    )
    raw_d1 = overlay.correct(cur.flat_grad, d0, hd0, subspace)
    delta = raw_d1 - d0
    # Match correct()'s full residual projection and reciprocal multiplication:
    # projecting its terms separately can change even the raw float32 proposal.
    residual = cur.flat_grad + hd0 + protocol.damping * d0
    inv_curvature = 1.0 / (subspace.eigenvalues.abs() + protocol.damping)
    original_c = -(
        subspace.eigenvectors
        @ (inv_curvature * (subspace.eigenvectors.t() @ subspace.project(residual)))
    )
    lifted = subspace.lift(original_c)
    lift_check = agreement(
        lifted, delta, rtol=tolerances.grad_rtol, atol=tolerances.atol
    )
    if not lift_check["passed"] or not torch.equal(d0 + lifted, raw_d1):
        raise InvalidProbe(f"raw proposal/reduced-coordinate mismatch: {lift_check}")
    random_probe = torch.randn(
        d0.shape,
        device=device,
        dtype=d0.dtype,
        generator=torch.Generator(device=device).manual_seed(seeds["jet"]),
    )
    random_probe *= float(d0.norm()) / max(float(random_probe.norm()), 1e-30)
    probes = {"d0": d0, "correction": delta, "random": random_probe}
    reference = {
        "loss": cur.loss,
        "gradient": _cpu(cur.flat_grad),
        "hvps": {
            name: _cpu(hd0 if name == "d0" else cur.hvp(vector))
            for name, vector in probes.items()
        },
    }
    loss_fn = nn.CrossEntropyLoss()
    raw_loss_at = trial_loss(model, loss_fn, group_major_params(model), 1.0)(
        payload["x"].to(device), payload["y"].to(device)
    )

    def loss_at(d: Tensor) -> float:
        with isolated_execution(payload["runtime"], device=device):
            return raw_loss_at(d)

    decision = TrustRegionSafeguard().accept(
        cur.flat_grad,
        d0,
        raw_d1,
        cur.hvp,
        hd0=hd0,
        tr_ratio=protocol.tr_ratio,
        damping=protocol.damping,
        loss_at=loss_at,
    )
    production = {
        "accepted": decision.accepted,
        "rejected_by": decision.rejected_by,
        "raw_proposal_preserved": True,
        "applied_candidate": "raw_d1" if decision.accepted else "d0",
        "m0": decision.m0,
        "m1": decision.m1 if math.isfinite(decision.m1) else None,
        "loss_d0": decision.loss_d0 if math.isfinite(decision.loss_d0) else None,
        "loss_d1": decision.loss_d1 if math.isfinite(decision.loss_d1) else None,
    }
    _finite(d0, raw_d1, delta, hd0, original_c, subspace.eigenvalues)
    # No curvature bundle/closure escapes: the H0 graph dies at this return.
    return {
        "d0": d0,
        "raw_d1": raw_d1.detach(),
        "delta": delta.detach(),
        "subspace": subspace,
        "reduced": reduced,
        "original_c": original_c,
        "replay": checks,
        "reference": reference,
        "probes": probes,
        "coupling": [
            {"pair": list(pair), "value": value} for pair, value in coupling.items()
        ],
        "selected_pairs": [list(pair) for pair in selected],
        "seeds": seeds,
        "production_decision": production,
        "lift_check": lift_check,
    }


def _projected_hessian(
    hvp: Callable[[Tensor], Tensor], subspace: OverlaySubspace
) -> Tensor:
    k = subspace.transform.shape[1]
    if not k:
        return subspace.transform.new_empty((0, 0))
    eye = torch.eye(k, dtype=subspace.transform.dtype, device=subspace.transform.device)
    reduced = torch.stack(
        [subspace.project(hvp(subspace.lift(eye[:, j]))) for j in range(k)], dim=1
    )
    return 0.5 * (reduced + reduced.t())


def _endpoint_geometry(
    forward: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    loss, grad, hvp = displacement_curvature(
        forward, nn.CrossEntropyLoss(), x, y, geometry["d0"]
    )
    subspace = geometry["subspace"]
    projected_h = _projected_hessian(hvp, subspace)
    _finite(loss, grad, projected_h)
    return {
        "loss": loss,
        "gradient": _cpu(grad),
        "u_gradient": subspace.project(grad),
        "h": projected_h,
        "slope": float(grad.dot(geometry["delta"])),
    }


def _original_candidate_models(
    model: Any,
    payload: dict[str, Any],
    geometry: dict[str, Any],
    candidates: dict[str, Tensor],
    tolerances: Tolerances,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Replay H0 after H_end dies; check repeatability, not bitwise equivalence.

    Return scalar candidate values, their same-pass m0, and initial/replay checks.
    AD tolerances apply only to repeatability; the downstream model gate is zero.
    """
    device = next(model.parameters()).device
    cur, capture_checks = replay_curvature(
        model,
        payload,
        n_probes=1,
        generator=torch.Generator(device=device).manual_seed(
            geometry["seeds"]["coupling"]
        ),
        tolerances=tolerances,
    )
    d0, subspace = geometry["d0"], geometry["subspace"]
    damping = geometry["reduced"]["damping"]
    hd0 = cur.hvp(d0)
    m0 = model_value(cur.flat_grad, d0, hd0, damping=damping)
    _finite(m0)
    reference = geometry["reference"]
    checks: dict[str, Any] = {
        "capture_replay": capture_checks,
        "loss": agreement(
            cur.loss, reference["loss"], rtol=tolerances.grad_rtol, atol=tolerances.atol
        ),
        "gradient": agreement(
            cur.flat_grad,
            reference["gradient"],
            rtol=tolerances.grad_rtol,
            atol=tolerances.atol,
        ),
        "hvp_d0": agreement(
            hd0, reference["hvps"]["d0"], rtol=tolerances.hvp_rtol, atol=tolerances.atol
        ),
    }
    values = {}
    for name, c in candidates.items():
        direction = d0 + subspace.lift(c)
        values[name] = (
            m0
            if torch.equal(direction, d0)
            else model_value(
                cur.flat_grad, direction, cur.hvp(direction), damping=damping
            )
        )
        _finite(values[name])
    for name, initial, replayed in (
        ("m0", geometry["reduced"]["m0"], m0),
        ("m1", geometry["production_decision"]["m1"], values["original"]),
    ):
        checks[name] = {
            "initial": initial,
            "replayed": replayed,
            "difference": None if initial is None else replayed - initial,
            **(
                {"status": "not_evaluated_in_initial_decision"}
                if initial is None
                else agreement(
                    replayed, initial, rtol=tolerances.hvp_rtol, atol=tolerances.atol
                )
            ),
        }
    checks["passed"] = all(
        checks[key]["passed"] for key in ("loss", "gradient", "hvp_d0", "m0")
    ) and checks["m1"].get("passed", True)
    if not checks["passed"]:
        raise InvalidProbe(f"replayed H0 numerical repeatability mismatch: {checks}")
    return values, m0, checks


def _jet(
    forward: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    geometry: dict[str, Any],
    tolerances: Tolerances,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loss, grad, hvp = displacement_curvature(
        forward, nn.CrossEntropyLoss(), x, y, torch.zeros_like(geometry["d0"])
    )
    reference = geometry["reference"]
    checks: dict[str, Any] = {
        "loss": agreement(
            loss, reference["loss"], rtol=tolerances.grad_rtol, atol=tolerances.atol
        ),
        "gradient": agreement(
            grad,
            reference["gradient"],
            rtol=tolerances.grad_rtol,
            atol=tolerances.atol,
        ),
        "hvps": {},
    }
    tensors: dict[str, Any] = {"loss": loss, "gradient": _cpu(grad), "hvps": {}}
    for name, vector in geometry["probes"].items():
        product = hvp(vector.to(x.device))
        checks["hvps"][name] = agreement(
            product,
            reference["hvps"][name],
            rtol=tolerances.hvp_rtol,
            atol=tolerances.atol,
        )
        tensors["hvps"][name] = _cpu(product)
        del product
    checks["passed"] = all(
        checks[key]["passed"] for key in ("loss", "gradient")
    ) and all(value["passed"] for value in checks["hvps"].values())
    return checks, tensors


def _endpoint_slope(
    forward: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    d0: Tensor,
    delta: Tensor,
) -> float:
    point = d0.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(forward(point, x), y)
    grad = torch.autograd.grad(loss, point)[0]
    slope = float(grad.dot(delta))
    _finite(slope)
    return slope


def _rays(
    loss_at: Callable[[Tensor], float], geometry: dict[str, Any]
) -> dict[str, Any]:
    d0, d1 = geometry["d0"], geometry["raw_d1"]
    r, c = geometry["reduced"], geometry["original_c"]
    loss0 = loss_at(torch.zeros_like(d0))
    coefficients = {
        "d0": (r["g_d0"], r["d0_h_d0"]),
        "d1": (
            r["g_d0"] + float(r["u_g"].dot(c)),
            r["d0_h_d0"] + 2 * float(r["u_hd0"].dot(c)) + float(c.dot(r["h"] @ c)),
        ),
    }
    rays = {}
    for name, direction in (("d0", d0), ("d1", d1)):
        gd, dhd = coefficients[name]
        rows = []
        for t in RAY_GRID:
            loss = loss_at(t * direction)
            undamped = t * gd + 0.5 * t * t * dhd
            damped = undamped + 0.5 * r["damping"] * t * t * float(
                direction.dot(direction)
            )
            _finite(loss, undamped, damped)
            rows.append(
                {
                    "t": t,
                    "loss": loss,
                    "delta_loss": loss - loss0,
                    "undamped_model": undamped,
                    "damped_model": damped,
                    "actual_minus_undamped_model": loss - loss0 - undamped,
                    "actual_minus_damped_model": loss - loss0 - damped,
                }
            )
        rays[name] = rows
    return rays


def _measure(
    loss_at: Callable[[Tensor], float],
    geometry: dict[str, Any],
    candidates: dict[str, Tensor],
    systems: dict[str, Any],
    pairs: dict[str, Any],
    *,
    protocol: Exp7Config,
    tolerances: Tolerances,
    gain_floor: float,
    endpoint_slope: float,
    rays: bool,
) -> dict[str, Any]:
    rows = measure_candidates(
        geometry["d0"],
        geometry["subspace"],
        candidates,
        geometry["reduced"],
        systems,
        loss_at,
        tr_ratio=protocol.tr_ratio,
        gain_floor=gain_floor,
        correction_relative_floor=tolerances.correction_relative_floor,
        original_models=geometry["original_models"],
        original_model_baseline=geometry["original_model_baseline"],
    )
    scale = [
        {
            "alpha": alpha,
            "candidate": f"scale_{i}",
            "gain": rows[f"scale_{i}"]["actual_gain"],
        }
        for i, alpha in enumerate(SCALE_GRID)
    ]
    best_scale_gain = max(point["gain"] for point in scale)
    primary = {
        name: {
            "gain_over_best_scale_control": rows[name]["actual_gain"] - best_scale_gain,
            "preference_over_scale_control": preference(
                rows[name]["actual_gain"] - best_scale_gain, gain_floor
            ),
            "near_zero_correction": rows[name]["near_zero_correction"],
            "substantive": not rows[name]["near_zero_correction"]
            and rows[name]["actual_gain"] > gain_floor
            and rows[name]["actual_gain"] - best_scale_gain > gain_floor,
        }
        for name in ("gradient_refresh", "curvature_refresh", "both_refresh")
    }
    matched = {
        name: {
            **pair,
            "refreshed_minus_original_gain": rows[pair["refreshed"]]["actual_gain"]
            - rows[pair["original"]]["actual_gain"],
        }
        for name, pair in pairs.items()
    }
    comparison = comparison_decomposition(
        geometry["reduced"],
        geometry["original_c"],
        geometry["d0"],
        geometry["delta"],
        loss_d0=rows["original"]["loss_d0"],
        loss_d1=rows["original"]["loss"],
        endpoint_slope=endpoint_slope,
        gain_floor=gain_floor,
    )
    gate_gain = -rows["original"]["model_evaluation"]["model_delta"]
    comparison.update(
        {
            "gate_damped_gain": gate_gain,
            "gate_minus_reduced_damped_gain": gate_gain - comparison["damped_gain"],
            "model_evaluation": rows["original"]["model_evaluation"],
        }
    )
    result = {
        "candidates": rows,
        "scale_control": scale,
        "best_scale_gain": best_scale_gain,
        "primary_refresh_comparison": primary,
        "norm_matched": matched,
        "comparison": comparison,
    }
    if rays:
        result["rays"] = _rays(loss_at, geometry)
    return result


def _calibrate(
    loss_at: Callable[[Tensor], float],
    geometry: dict[str, Any],
    tolerances: Tolerances,
    *,
    capture_sha256: str,
    supplied: Path | None,
) -> dict[str, Any]:
    repeated = {
        name: [loss_at(direction) for _ in range(3)]
        for name, direction in (
            ("zero", torch.zeros_like(geometry["d0"])),
            ("d0", geometry["d0"]),
            ("raw_d1", geometry["raw_d1"]),
        )
    }
    for values in repeated.values():
        _finite(*values)
    error = max(max(values) - min(values) for values in repeated.values())
    if supplied is None:
        return {
            "schema": SCHEMA,
            "tolerances": asdict(tolerances),
            "gain_floor": max(tolerances.gain_floor, 10 * error),
            "repeatability_error": error,
            "repeats": repeated,
            "calibration_capture_sha256": capture_sha256,
            "role": "smoke_calibration_freeze_before_fanout",
        }
    fixed = json.loads(supplied.read_text())
    if fixed["tolerances"] != asdict(tolerances):
        raise InvalidProbe("registered numerical tolerances differ from this probe")
    floor = fixed["gain_floor"]
    if not math.isfinite(floor) or floor < max(tolerances.gain_floor, 10 * error):
        raise InvalidProbe("repeatability exceeds the frozen gain/noise threshold")
    return {
        **fixed,
        "role": "fixed_from_smoke",
        "registration_file": str(supplied.resolve()),
        "this_repeatability_error": error,
        "this_repeats": repeated,
    }


def probe_capture(
    capture: Path,
    output: Path,
    *,
    device: torch.device,
    max_minutes: float = 30,
    thresholds: Path | None = None,
    model_factory: Callable[[Exp7Config], Any] | None = None,
    tolerances: Tolerances = _DEFAULT_TOLERANCES,
) -> dict[str, Any]:
    """One frozen state, one graph at a time, all four fixed-geometry interventions."""
    with (
        ArtifactDirectory(output, kind="probe", max_minutes=max_minutes) as run,
        isolated_execution(execution_state(device), device=device),
    ):
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "capture": str(Path(capture).resolve()),
            "tests": {
                "model_error": "not_run",
                "refresh": "not_run",
                "architecture_intervention": "not_run",
            },
            "scientific_conclusion": "undetermined",
            "independent_verification": "not_performed",
        }
        with run.stage("load_and_validate_capture"):
            payload, manifest = load_capture(capture)
            _restore_execution(payload["runtime"], device)
            result.update(
                {
                    "capture_sha256": manifest["sha256"],
                    "seed": payload["seed"],
                    "step": payload["step"],
                }
            )
            write_json(run.path / "results.json", result)
            source = manifest["provenance"].get("source")
            current_source = source_hashes()
            if source is not None and source["sha256"] != current_source["sha256"]:
                raise InvalidProbe("probe source differs from captured source snapshot")
            write_json(
                run.path / "provenance.json",
                {
                    "environment": environment_block(),
                    "source": current_source,
                    "capture_manifest": manifest,
                    "device": str(device),
                },
            )
            protocol = _config_from_payload(payload)
            if model_factory is None:
                from experiments.runner_exp7 import build_model

                model_factory = build_model
            # Construction never requests pretrained weights or creates an optimizer.
            with isolated_execution(execution_state(device), device=device):
                model = model_factory(replace(protocol, pretrained=False)).to(
                    device=device, dtype=payload["d0"].dtype
                )
            restore_model(model, payload)
            x, y = payload["x"].to(device), payload["y"].to(device)
            raw_forward = trial_forward(model, group_major_params(model), 1.0)

            def forward(d: Tensor, batch: Tensor) -> Tensor:
                with isolated_execution(payload["runtime"], device=device):
                    return raw_forward(d, batch)

            def loss_at(d: Tensor) -> float:
                with torch.no_grad():
                    value = float(nn.CrossEntropyLoss()(forward(d, x), y))
                _finite(value)
                return value

        with run.stage("native_geometry"):
            geometry = _geometry(model, payload, protocol, tolerances)
            result.update(
                {
                    key: geometry[key]
                    for key in (
                        "replay",
                        "coupling",
                        "selected_pairs",
                        "seeds",
                        "production_decision",
                        "lift_check",
                    )
                }
            )
            save_tensors(
                run.path / "base_geometry.pt",
                {
                    "d0": geometry["d0"],
                    "raw_d1": geometry["raw_d1"],
                    "g0": payload["g0"],
                    "reduced": geometry["reduced"],
                    "subspace": _subspace_payload(geometry["subspace"]),
                    "reference_jet": geometry["reference"],
                    "probes": geometry["probes"],
                },
            )
            write_json(run.path / "results.json", result)
        with run.stage("repeatability"):
            calibration = _calibrate(
                loss_at,
                geometry,
                tolerances,
                capture_sha256=manifest["sha256"],
                supplied=thresholds,
            )
            write_json(run.path / "thresholds.json", calibration)
            result["thresholds"] = calibration
        with run.stage("native_flat_ad_jet"):
            jet, tensors = _jet(forward, x, y, geometry, tolerances)
            save_tensors(run.path / "jet_native.pt", tensors)
            result["native_jet"] = jet
            write_json(run.path / "results.json", result)
            del tensors
            if not jet["passed"]:
                raise InvalidProbe("native displacement AD-jet mismatch")
        with run.stage("endpoint_refresh"):
            endpoint = _endpoint_geometry(forward, x, y, geometry)
            coordinates, systems = refresh_coordinates(
                geometry["reduced"], endpoint["u_gradient"], endpoint["h"]
            )
            coordinates["original"] = geometry["original_c"]
            candidates, pairs = add_controls(coordinates, geometry["subspace"])
        with run.stage("original_candidate_models"):
            (
                geometry["original_models"],
                geometry["original_model_baseline"],
                geometry["original_model_replay"],
            ) = _original_candidate_models(
                model, payload, geometry, candidates, tolerances
            )
            result["original_model_replay"] = geometry["original_model_replay"]
            write_json(run.path / "results.json", result)
            save_tensors(
                run.path / "decision_bundle.pt",
                {
                    "capture_sha256": manifest["sha256"],
                    "d0": geometry["d0"],
                    "g0": payload["g0"],
                    "raw_d1": geometry["raw_d1"],
                    "subspace": _subspace_payload(geometry["subspace"]),
                    "reduced": geometry["reduced"],
                    "original_models": geometry["original_models"],
                    "original_model_baseline": geometry["original_model_baseline"],
                    "original_model_replay": geometry["original_model_replay"],
                    "production_decision": geometry["production_decision"],
                    "endpoint": endpoint,
                    "coordinates": coordinates,
                    "candidates": candidates,
                    "systems": systems,
                    "matched_pairs": pairs,
                },
            )
        with run.stage("native_measurements"):
            native = _measure(
                loss_at,
                geometry,
                candidates,
                systems,
                pairs,
                protocol=protocol,
                tolerances=tolerances,
                gain_floor=calibration["gain_floor"],
                endpoint_slope=endpoint["slope"],
                rays=True,
            )
            result["native"] = native
            result["tests"].update({"model_error": "complete", "refresh": "complete"})
            # A raw proposal stays raw even when the production path falls back.
            initial_accepted = geometry["production_decision"]["accepted"]
            replayed_accepted = native["candidates"]["original"]["original_gates"][
                "accepted"
            ]
            result["production_acceptance_check"] = {
                "initial_accepted": initial_accepted,
                "replayed_accepted": replayed_accepted,
                "passed": replayed_accepted == initial_accepted,
            }
            # Preserve both decisions and the measurements even on a strict-gate flip.
            write_json(run.path / "results.json", result)
            if not result["production_acceptance_check"]["passed"]:
                raise InvalidProbe(
                    "independent original gates disagree with production safeguard: "
                    f"{result['production_acceptance_check']}; "
                    f"model replay: {geometry['original_model_replay']}"
                )
        # The full native H0 graph is already gone. Architectural AD checks do
        # not need U: retain only CPU factors, and reload them for no-grad loss
        # measurements so candidate arithmetic stays on its original device.
        cpu_subspace = _subspace_to(geometry["subspace"], torch.device("cpu"))
        geometry["subspace"] = cpu_subspace
        geometry["probes"] = _cpu(geometry["probes"])
        result["memory_policy"] = {
            "one_retained_graph": True,
            "architecture_jets_factored_U": "cpu",
            "branch_tape_storage": "cpu",
            "candidate_arithmetic_device": str(device),
        }
        from experiments.frozen_branches import (
            BranchTape,
            branch_intervention,
            record_branches,
        )

        with run.stage("record_reference_branches"):
            with torch.no_grad():
                tape = record_branches(
                    model, lambda: forward(torch.zeros_like(geometry["d0"]), x)
                )
            tape_payload = tape.to_payload()
            save_tensors(run.path / "branches.pt", tape_payload)
            tape = BranchTape.from_payload(tape_payload)
            del tape_payload
            result["branch_tape"] = tape.summary()
            result["interventions"] = {}
            write_json(run.path / "results.json", result)
        for mode in ("sham", "relu", "pool", "both"):
            with run.stage(f"intervention_{mode}_jet"):
                with branch_intervention(model, tape, mode):
                    jet, tensors = _jet(forward, x, y, geometry, tolerances)
                save_tensors(run.path / f"jet_{mode}.pt", tensors)
                del tensors
                result["interventions"][mode] = {
                    "jet": jet,
                    "measurements": {"status": "not_run"},
                }
                write_json(run.path / "results.json", result)
                if not jet["passed"]:
                    raise InvalidProbe(f"{mode} reference AD-jet mismatch")
            with run.stage(f"intervention_{mode}_measurements"):
                with branch_intervention(model, tape, mode):
                    slope = _endpoint_slope(
                        forward, x, y, geometry["d0"], geometry["delta"]
                    )
                    geometry["subspace"] = _subspace_to(cpu_subspace, device)
                    try:
                        measurement = _measure(
                            loss_at,
                            geometry,
                            candidates,
                            systems,
                            pairs,
                            protocol=protocol,
                            tolerances=tolerances,
                            gain_floor=calibration["gain_floor"],
                            endpoint_slope=slope,
                            rays=mode == "sham",
                        )
                    finally:
                        geometry["subspace"] = cpu_subspace
                result["interventions"][mode]["measurements"] = measurement
                if mode == "sham":
                    errors = [
                        abs(row["loss"] - native["candidates"][name]["loss"])
                        for name, row in measurement["candidates"].items()
                    ]
                    errors += [
                        abs(a["loss"] - b["loss"])
                        for ray in ("d0", "d1")
                        for a, b in zip(
                            measurement["rays"][ray], native["rays"][ray], strict=True
                        )
                    ]
                    max_error = max(errors, default=0.0)
                    result["sham_finite_losses"] = {
                        "max_absolute_error": max_error,
                        "atol": tolerances.atol,
                        "passed": max_error <= tolerances.atol,
                    }
                    if not result["sham_finite_losses"]["passed"]:
                        write_json(run.path / "results.json", result)
                        raise InvalidProbe("sham changes finite candidate losses")
                write_json(run.path / "results.json", result)
        result["tests"]["architecture_intervention"] = "complete"
        result["scientific_conclusion"] = (
            "measured_exploratory_requires_independent_verification"
        )
        result["resources"] = {
            "gpu_peak_allocated_bytes": run_peak_allocated(device),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "elapsed_seconds": time.monotonic() - run.start,
        }
        write_json(run.path / "results.json", result)
        return result


def capture_pilot(
    output: Path,
    *,
    seeds: list[int],
    steps: list[int],
    device: torch.device,
    num_workers: int,
    cars_root: Path,
    torch_home: Path,
) -> dict[str, Any]:
    """B1 Cars with cached weights/data only, fresh loaders per training seed."""
    from experiments.data import get_stanford_cars_loaders
    from experiments.runner_exp7 import build_model

    if (
        not seeds
        or not steps
        or len(seeds) != len(set(seeds))
        or len(steps) != len(set(steps))
        or any(step < 0 for step in steps)
    ):
        raise ValueError(
            "seeds/steps must be nonempty, unique; steps must be nonnegative"
        )
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    with ArtifactDirectory(output, kind="capture") as run:
        run.status["requested"] = {
            "seeds": seeds,
            "steps": sorted(steps),
            "device": str(device),
            "num_workers": num_workers,
            "cars_root": str(cars_root),
            "torch_home": str(torch_home),
        }
        run.flush()
        with run.stage("cached_inputs_and_provenance"):
            cars_root, torch_home = (
                Path(cars_root).resolve(),
                Path(torch_home).resolve(),
            )
            weight = torch_home / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
            if not cars_root.is_dir() or not weight.is_file():
                raise FileNotFoundError(
                    "explicit Cars root and cached resnet50-11ad3fa6.pth are required; downloads are disabled"
                )
            if not (
                cars_root / "car_devkit" / "devkit" / "cars_train_annos.mat"
            ).is_file():
                raise FileNotFoundError(
                    "Cars annotations are missing from the explicit local root"
                )
            os.environ["DAG_HESSE_CARS_ROOT"] = str(cars_root)
            os.environ["TORCH_HOME"] = str(torch_home)
            protocol = Exp7Config.b1_cars()
            protocol = replace(
                protocol, training=replace(protocol.training, seeds=list(seeds))
            )
            provenance = {
                "environment": environment_block(),
                "source": source_hashes(),
                "data_root": str(cars_root),
                "data": tree_hash(cars_root),
                "weight": {"path": str(weight), "sha256": file_sha256(weight)},
                "config": config_to_dict(protocol),
                "num_workers": num_workers,
                "steps": sorted(steps),
                "seeds": seeds,
                "step_indexing": "zero_based_pre_update",
                "trajectory": "new_plain_kfac_d0_only",
            }
            write_json(run.path / "provenance.json", provenance)
            records: dict[str, Any] = {}
        for seed in seeds:
            with run.stage(f"capture_seed_{seed}"):
                set_seed(seed, deterministic=protocol.training.deterministic)
                train_loader, _ = get_stanford_cars_loaders(
                    protocol.training.batch_size,
                    image_size=protocol.image_size,
                    augment=protocol.augment,
                    seed=0,
                    num_workers=num_workers,
                )
                n_steps = max(steps) + 1
                epochs = max(1, math.ceil(n_steps / max(len(train_loader), 1)))
                training = replace(protocol.training, epochs=epochs, max_steps=n_steps)
                model = build_model(protocol).to(device)
                seed_records: list[str] = []

                def sink(
                    payload: dict[str, Any],
                    seed: int = seed,
                    seed_records: list[str] = seed_records,
                ) -> None:
                    path = run.path / f"seed_{seed}" / f"step_{payload['step']:06d}"
                    write_capture(path, payload, provenance=provenance)
                    seed_records.append(str(path.resolve()))

                hook = build_capture_hook(
                    model, protocol, seed=seed, steps=steps, sink=sink
                )
                trainer = Trainer(model, training, device, step_hook=hook)
                trainer.train(train_loader, checkpoint_epochs=[])
                if trainer.step_count != n_steps or len(seed_records) != len(steps):
                    raise InvalidProbe(
                        "capture ended before every preregistered state was saved"
                    )
                records[str(seed)] = {
                    "captures": seed_records,
                    "n_steps": trainer.step_count,
                }
                write_json(run.path / "captures.json", records)
                del trainer, hook, model, train_loader
        return records
