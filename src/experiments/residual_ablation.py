"""Version 2 saved-state residual-input ablation with fixed six-group geometry.

This is a new measurement, not a replay of historical pilot probes. Inventory
checks capture bytes without loading tensors; measurement requires explicit launch.
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from experiments import causal_diagnostics as cd
from experiments.param_ops import group_major_params, trial_forward, trial_loss
from experiments.provenance import environment_block
from secondorder.overlay import CoupleFacOverlay, OverlaySubspace
from secondorder.safeguard import model_value

SCHEMA = 2
GROUPS = ("stem", "layer1", "layer2", "layer3", "layer4", "head")
SCALE_GRID = cd.SCALE_GRID
DRAWSETS = {"primary": (100_003, 200_003), "sensitivity": (300_003, 400_003)}
SEEDS = (42, 43, 44)
STEPS = (20, 60, 200)
TOLERANCES = cd.Tolerances()
ARMS = ("full", "off", "rest")
PAIRS = (("full", "off"), ("full", "rest"), ("off", "rest"))


def environment_identity() -> dict[str, Any]:
    """Small installed-runtime identity, retaining the actual virtual environment."""
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "numpy": version("numpy"),
        "torchvision": version("torchvision"),
    }


def guard_output(
    output: Path, inputs: list[Path], *, protected_root: Path | None = None
) -> None:
    """Reject resolved output overlap with inputs or an explicitly protected root."""
    target = output.resolve()
    for protected in [
        *inputs,
        *([protected_root] if protected_root is not None else []),
    ]:
        root = protected.resolve()
        if target == root or target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError(f"output/input overlap: {target} and {root}")


def registration_template(
    source_sha256: str, capture_roster: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Fixed protocol plus nine historical capture/source identities."""
    return {
        "capture_roster": capture_roster if capture_roster is not None else [],
        "measurement_schema": SCHEMA,
        "registration_phase": "smoke",
        "current_source_sha256": source_sha256,
        "installed_source_root": str(Path(__file__).resolve().parents[1]),
        "environment_identity": environment_identity(),
        "historical_probe_replay": False,
        "d0_recomputed": False,
        "partition": list(GROUPS),
        "seeds": list(SEEDS),
        "steps": list(STEPS),
        "drawsets": {k: list(v) for k, v in DRAWSETS.items()},
        "alpha_grid": list(SCALE_GRID),
        "lambda": 1.0,
        "coupling_probes": 10,
        "rank": 4,
        "power_iterations": 8,
        "tau": 0.1,
        "max_pairs": 3,
        "norm_ratio": 2.0,
        "tolerances": asdict(TOLERANCES),
        "precision": "saved_runtime_float32_no_autocast_change",
        "cohorts": "smoke_and_final_separate;final_remeasures_all_nine_states",
    }


def validate_registration(registration: Mapping[str, Any], current_sha: str) -> bool:
    roster = registration.get("capture_roster", [])
    if (
        not isinstance(roster, list)
        or len(roster) != len(SEEDS) * len(STEPS)
        or not all(isinstance(row, dict) for row in roster)
        or [(row.get("seed"), row.get("step")) for row in roster]
        != [(seed, step) for seed in SEEDS for step in STEPS]
        or not all(
            row.get("status") == "bytes_checked_not_loaded"
            and isinstance(row.get("capture_sha256"), str)
            and isinstance(row.get("historical_source_sha256"), str)
            and isinstance(row.get("capture_path"), str)
            and Path(row["capture_path"]).is_absolute()
            for row in roster
        )
        or len({row["capture_path"] for row in roster}) != len(roster)
    ):
        return False
    expected = registration_template(current_sha, roster)
    if registration.get("registration_phase") == "final":
        binding = registration.get("smoke_calibration")
        if not isinstance(binding, dict) or set(binding) != {
            "thresholds",
            "smoke_registration",
            "smoke_results",
            "smoke_status",
        }:
            return False
        if not all(
            isinstance(entry, dict)
            and set(entry) == {"path", "sha256"}
            and isinstance(entry["path"], str)
            and isinstance(entry["sha256"], str)
            for entry in binding.values()
        ):
            return False
        expected.update(registration_phase="final", smoke_calibration=binding)
    return dict(registration) == expected


def _registered_capture(
    registration: Mapping[str, Any], seed: int, step: int
) -> dict[str, Any]:
    return next(
        row
        for row in registration["capture_roster"]
        if (row["seed"], row["step"]) == (seed, step)
    )


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": cd.file_sha256(path)}


def _validate_thresholds(
    fixed: Mapping[str, Any], registration: Mapping[str, Any], registration_sha: str
) -> None:
    calibration = _registered_capture(registration, 42, 60)
    if fixed.get("capture_sha256") != calibration["capture_sha256"]:
        raise cd.InvalidProbe(
            "thresholds calibration capture differs from registered seed 42 step 60"
        )
    if (
        fixed.get("measurement_schema") != SCHEMA
        or fixed.get("source_sha256") != registration["current_source_sha256"]
        or fixed.get("capture_seed") != 42
        or fixed.get("capture_step") != 60
        or fixed.get("tolerances") != asdict(TOLERANCES)
        or fixed.get("role") != "technical_smoke_only_not_final_data"
        or fixed.get("smoke_registration_sha256") != registration_sha
        or fixed.get("environment_identity") != registration["environment_identity"]
    ):
        raise cd.InvalidProbe(
            "thresholds role/source/registration/environment differs from smoke"
        )
    spread, floor = (
        fixed.get("same_candidate_repeatability_spread"),
        fixed.get("gain_floor"),
    )
    if (
        not isinstance(spread, (float, int))
        or isinstance(spread, bool)
        or not isinstance(floor, (float, int))
        or isinstance(floor, bool)
        or not math.isfinite(spread)
        or spread < 0
        or not math.isfinite(floor)
        or floor != max(1e-4, 10 * spread)
    ):
        raise cd.InvalidProbe(
            "threshold gain floor does not match finite smoke repeatability formula"
        )


def freeze_final_registration(
    smoke_registration: Path,
    thresholds: Path,
    output: Path,
    *,
    protected_root: Path | None = None,
) -> dict[str, Any]:
    """Bind completed smoke artifacts for final registration."""
    smoke = json.loads(smoke_registration.read_text())
    if (
        not validate_registration(smoke, cd.source_hashes()["sha256"])
        or smoke["registration_phase"] != "smoke"
    ):
        raise cd.InvalidProbe(
            "freeze requires the matching installed smoke registration"
        )
    guard_output(
        output,
        [Path(row["capture_path"]) for row in smoke["capture_roster"]]
        + [
            smoke_registration,
            thresholds,
            thresholds.parent / "results.json",
            thresholds.parent / "status.json",
        ],
        protected_root=protected_root,
    )
    fixed = json.loads(thresholds.read_text())
    _validate_thresholds(fixed, smoke, cd.file_sha256(smoke_registration))
    results_path, status_path = (
        thresholds.parent / "results.json",
        thresholds.parent / "status.json",
    )
    results, status = (
        json.loads(results_path.read_text()),
        json.loads(status_path.read_text()),
    )
    if (
        status.get("status") != "complete"
        or status.get("kind") != f"residual_ablation_v{SCHEMA}_smoke"
        or results.get("phase") != "smoke"
        or results.get("status") != "complete"
        or results.get("thresholds") != fixed
        or results.get("registration", {}).get("sha256")
        != cd.file_sha256(smoke_registration)
        or results.get("loading_preflight", {}).get("status") != "complete"
        or results["loading_preflight"].get("registration_sha256")
        != cd.file_sha256(smoke_registration)
        or results["loading_preflight"].get("capture_roster") != smoke["capture_roster"]
        or results["loading_preflight"].get("current_source_sha256")
        != smoke["current_source_sha256"]
        or results["loading_preflight"].get("environment_identity")
        != smoke["environment_identity"]
        or results["loading_preflight"].get("valid") != 9
        or results.get("capture_sha256") != fixed["capture_sha256"]
        or results.get("current_source_sha256") != smoke["current_source_sha256"]
        or results.get("environment_identity") != smoke["environment_identity"]
        or results.get("drawset_counts", {}).get("valid") != 2
    ):
        raise cd.InvalidProbe(
            "freeze requires a completed compatible smoke and all-nine load preflight"
        )
    final = {
        **smoke,
        "registration_phase": "final",
        "smoke_calibration": {
            "thresholds": _artifact(thresholds),
            "smoke_registration": _artifact(smoke_registration),
            "smoke_results": _artifact(results_path),
            "smoke_status": _artifact(status_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(final, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return final


def _final_thresholds(
    registration: Mapping[str, Any], supplied: Path
) -> dict[str, Any]:
    if registration["registration_phase"] != "final":
        raise cd.InvalidProbe(
            "final measurement requires a post-smoke frozen final registration"
        )
    binding = registration["smoke_calibration"]
    if str(supplied.resolve()) != binding["thresholds"]["path"]:
        raise cd.InvalidProbe("threshold path differs from frozen artifact")
    for name, entry in binding.items():
        if cd.file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise cd.InvalidProbe(f"frozen smoke artifact changed: {name}")
    smoke_path = Path(binding["smoke_registration"]["path"])
    smoke = json.loads(smoke_path.read_text())
    if (
        not validate_registration(smoke, registration["current_source_sha256"])
        or smoke["registration_phase"] != "smoke"
        or smoke["capture_roster"] != registration["capture_roster"]
    ):
        raise cd.InvalidProbe(
            "frozen smoke registration differs from final capture/source/environment"
        )
    fixed = json.loads(supplied.read_text())
    _validate_thresholds(fixed, smoke, binding["smoke_registration"]["sha256"])
    return fixed


def _check(
    error: Tensor,
    components: tuple[Tensor, ...],
    rtol: float,
    *,
    scale_override: float | None = None,
) -> dict[str, Any]:
    absolute = float(error.double().norm())
    scale = (
        sum(float(x.double().norm()) for x in components)
        if scale_override is None
        else scale_override
    )
    threshold = TOLERANCES.atol + rtol * scale
    return {
        "absolute_error": absolute,
        "component_norm_sum": scale,
        "scaled_error": absolute / max(scale, TOLERANCES.atol),
        "threshold": threshold,
        "passed": math.isfinite(absolute) and absolute <= threshold,
    }


def split_residual(
    g: Tensor,
    d0: Tensor,
    hd0: Tensor,
    hvp: Callable[[Tensor], Tensor],
    ranges: Mapping[str, slice],
    damping: float,
) -> dict[str, Any]:
    """Six masked-source HVPs, diagonal receivers and off-receiver action."""
    if (
        tuple(ranges) != GROUPS
        or any(
            ranges[name].start != (0 if i == 0 else ranges[GROUPS[i - 1]].stop)
            or ranges[name].stop <= ranges[name].start
            for i, name in enumerate(GROUPS)
        )
        or ranges[GROUPS[-1]].stop != d0.numel()
    ):
        raise cd.InvalidProbe("capture must have six contiguous saved groups")
    source_sum = torch.zeros_like(d0)
    source_norm_sum = 0.0
    diagonal = torch.zeros_like(d0)
    off_receivers = torch.zeros_like(d0)
    source_checks = []
    for name in GROUPS:
        sl = ranges[name]
        masked = torch.zeros_like(d0)
        masked[sl] = d0[sl]
        action = hvp(masked)
        cd._finite(action)
        action_norm = float(action.double().norm())
        source_norm_sum += action_norm
        source_sum += action
        diagonal[sl] = action[sl]
        off_receivers += action
        off_receivers[sl] -= action[sl]
        source_checks.append({"group": name, "action_norm": action_norm})
    full = g + hd0 + damping * d0  # keep native float32 arithmetic
    off = hd0 - diagonal
    rest = g + diagonal + damping * d0
    checks: dict[str, Any] = {
        "masked_sum": _check(
            hd0 - source_sum,
            (hd0,),
            TOLERANCES.hvp_rtol,
            scale_override=float(hd0.double().norm()) + source_norm_sum,
        ),
        "off_receivers": _check(
            off - off_receivers, (hd0, diagonal, off_receivers), TOLERANCES.hvp_rtol
        ),
        "residual": _check(
            full - (off + rest), (full, off, rest), TOLERANCES.grad_rtol
        ),
    }
    checks["passed"] = all(row["passed"] for row in checks.values())
    if not checks["passed"]:
        raise cd.InvalidProbe(f"six-group residual decomposition failed: {checks}")
    # HVP/residual resolution is in vector-norm units, never derived from loss floor.
    resolution = max(
        TOLERANCES.atol,
        checks["masked_sum"]["absolute_error"]
        + checks["off_receivers"]["absolute_error"]
        + TOLERANCES.hvp_rtol * (float(hd0.double().norm()) + source_norm_sum)
        + TOLERANCES.grad_rtol * checks["residual"]["component_norm_sum"],
    )
    return {
        "full": full,
        "off": off,
        "rest": rest,
        "checks": checks,
        "source_checks": source_checks,
        "hvp_residual_resolution": resolution,
    }


def action_coverage(
    subspace: OverlaySubspace | None, off: Tensor, *, resolution: float
) -> dict[str, Any]:
    """Energy of the summed off action projected onto checked factored U."""
    denominator = float(off.double().dot(off.double()))
    row: dict[str, Any] = {
        "numerator": None,
        "denominator": denominator,
        "resolution_norm": resolution,
        "eta_action": None,
        "status": "unavailable",
        "orthonormality_error": None,
    }
    if (
        not math.isfinite(denominator)
        or not math.isfinite(resolution)
        or resolution < 0
    ):
        row["status"] = "nonfinite_action_or_resolution"
    elif denominator == 0:
        row["status"] = "zero_denominator"
    elif math.sqrt(denominator) <= resolution:
        row["status"] = "unresolved_denominator"
    elif subspace is None or subspace.transform.shape[1] == 0:
        row.update({"numerator": 0.0, "eta_action": 0.0, "status": "resolved"})
    else:
        k = subspace.transform.shape[1]
        eye = torch.eye(k, dtype=off.dtype, device=off.device)
        gram = torch.stack(
            [subspace.project(subspace.lift(eye[:, j])) for j in range(k)], dim=1
        )
        error = float((gram.double() - eye.double()).abs().max())
        row["orthonormality_error"] = error
        projection = subspace.project(off).double()
        numerator = float(projection.dot(projection))
        row["numerator"] = numerator
        if not math.isfinite(error) or error > TOLERANCES.grad_rtol:
            row["status"] = "nonorthogonal_basis"
        else:
            ratio = numerator / denominator
            if not math.isfinite(ratio) or ratio > 1 + TOLERANCES.grad_rtol:
                row["status"] = "projection_contract_unresolved"
            else:
                row.update({"eta_action": ratio, "status": "resolved"})
    return row


def corrections(
    split: Mapping[str, Any],
    subspace: OverlaySubspace | None,
    overlay: CoupleFacOverlay,
    d0: Tensor,
    hd0: Tensor,
    g: Tensor,
) -> dict[str, Any]:
    """Solve all arms with one eigensystem; keep native full direction arithmetic."""
    result: dict[str, Any] = {}
    for name in ("full", "off", "rest"):
        delta = (
            torch.zeros_like(d0)
            if subspace is None
            else overlay._reduced_newton(subspace, split[name])
        )
        direction = (
            overlay.correct(g, d0, hd0, subspace) if name == "full" else d0 + delta
        )
        result[name] = {"delta": delta, "direction": direction}
    native_direction = result["full"]["direction"]
    expected_direction = d0 + result["full"]["delta"]
    native_check = _check(
        native_direction - expected_direction,
        (native_direction, expected_direction),
        TOLERANCES.grad_rtol,
    )
    native_check["comparison_space"] = "direction"
    native_check["bitwise_equal"] = torch.equal(native_direction, expected_direction)
    native_check["passed"] = native_check["passed"] and native_check["bitwise_equal"]
    recovered = native_direction - d0
    recovery_check = _check(
        recovered - result["full"]["delta"],
        (recovered, result["full"]["delta"]),
        TOLERANCES.grad_rtol,
    )
    recovery_check["diagnostic_only"] = True
    add = _check(
        result["full"]["delta"] - result["off"]["delta"] - result["rest"]["delta"],
        tuple(result[name]["delta"] for name in ("full", "off", "rest")),
        TOLERANCES.grad_rtol,
    )
    add["native_full_check"] = native_check
    add["recovered_correction_check"] = recovery_check
    result["additivity"] = add
    result["off_rest_dot"] = float(result["off"]["delta"].dot(result["rest"]["delta"]))
    if not native_check["passed"]:
        raise cd.InvalidProbe(
            f"native full direction equivalence failed: {native_check}"
        )
    if not add["passed"]:
        raise cd.InvalidProbe(f"projected correction additivity failed: {add}")
    return result


def norm_matched(corrections: Mapping[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """Pairwise down-only control, including zero-length pairs."""
    result = {}
    for left, right in (("full", "off"), ("full", "rest"), ("off", "rest")):
        a, b = corrections[left], corrections[right]
        na, nb = float(a.norm()), float(b.norm())
        target = min(na, nb)
        result[f"{left}__{right}"] = {
            left: a * (target / na if na else 0.0),
            right: b * (target / nb if nb else 0.0),
        }
    return result


def _candidate_labels() -> dict[str, dict[str, float | None]]:
    labels: dict[str, dict[str, float | None]] = {
        arm: {f"alpha_{i}": alpha for i, alpha in enumerate(SCALE_GRID)} for arm in ARMS
    }
    for left, right in PAIRS:
        for arm in (left, right):
            labels[arm][f"matched_{left}__{right}"] = None
    return labels


def _row_counts(
    rows: Mapping[str, Any], status_field: str = "status"
) -> dict[str, int]:
    statuses = [row.get(status_field, "not_attempted") for row in rows.values()]
    return {
        "expected": len(rows),
        "attempted": sum(value != "not_attempted" for value in statuses),
        "valid": statuses.count("finite"),
        "technical_failure": statuses.count("technical_failure"),
        "not_attempted": statuses.count("not_attempted"),
        "unresolved": sum(
            row.get(status_field) == "finite"
            and row.get(
                "applied_interpretation"
                if status_field == "applied_status"
                else "interpretation"
            )
            == "unresolved_preference"
            for row in rows.values()
        ),
    }


def _measurement_counts(measurements: dict[str, Any]) -> None:
    for field, target in (
        ("status", "counts"),
        ("raw_status", "raw_counts"),
        ("applied_status", "applied_counts"),
    ):
        measurements[target] = {
            arm: _row_counts(rows, field) for arm, rows in measurements["arms"].items()
        }
    measurements["unresolved_is_subset_of_valid"] = True


def _empty_measurements() -> dict[str, Any]:
    result = {
        "arms": {
            arm: {
                label: {
                    "status": "not_attempted",
                    "raw_status": "not_attempted",
                    "applied_status": "not_attempted",
                    "alpha": scale if label.startswith("alpha_") else None,
                    "kind": "scale"
                    if label.startswith("alpha_")
                    else "matched_down_only",
                }
                for label, scale in labels.items()
            }
            for arm, labels in _candidate_labels().items()
        }
    }
    _measurement_counts(result)
    return result


def _number(value: float, field: str, failures: list[str]) -> float | None:
    if not math.isfinite(value):
        failures.append(f"{field}: nonfinite {value}")
        return None
    return value


def _repeat_losses(
    loss_at: Callable[[Tensor], float],
    direction: Tensor,
    field: str,
    failures: list[str],
) -> list[float | None]:
    values: list[float | None] = []
    for i in range(3):
        try:
            values.append(_number(float(loss_at(direction)), f"{field}[{i}]", failures))
        except (RuntimeError, ValueError, OverflowError, TypeError) as error:
            failures.append(f"{field}[{i}]: {type(error).__name__}: {error}")
            values.append(None)
    return values


def measure_candidates(
    d0: Tensor,
    g: Tensor,
    hd0: Tensor,
    hvp: Callable[[Tensor], Tensor],
    solved: Mapping[str, Any],
    loss_at: Callable[[Tensor], float],
    *,
    floor: float,
    norm_ratio: float = 2.0,
    reverse_order_check: bool = False,
) -> dict[str, Any]:
    """Full row accounting, including finite loss evidence on model/norm failures."""
    baseline_failures: list[str] = []
    origin = _repeat_losses(
        loss_at, torch.zeros_like(d0), "loss_theta", baseline_failures
    )
    base = _repeat_losses(loss_at, d0, "loss_d0", baseline_failures)
    m0 = _number(model_value(g, d0, hd0, damping=1.0), "model_d0", baseline_failures)
    norm0 = _number(float(d0.norm()), "norm_d0", baseline_failures)
    if not all(bool(torch.isfinite(vector).all()) for vector in (d0, g, hd0)):
        baseline_failures.append("nonfinite baseline direction/gradient/HVP")
    candidates = _candidate_labels()
    correction_norms = {arm: float(solved[arm]["delta"].norm()) for arm in ARMS}
    for left, right in PAIRS:
        na, nb = correction_norms[left], correction_norms[right]
        for arm in (left, right):
            arm_norm = correction_norms[arm]
            candidates[arm][f"matched_{left}__{right}"] = (
                (min(na, nb) / arm_norm if arm_norm else 0.0)
                if math.isfinite(na) and math.isfinite(nb)
                else None
            )

    def direction_at(arm: str, label: str, scale: float | None) -> Tensor | None:
        if label == "alpha_0":
            return d0
        if label == "alpha_6":
            return solved[arm]["direction"]
        return None if scale is None else d0 + scale * solved[arm]["delta"]

    result = _empty_measurements()
    rows = result["arms"]
    for arm, entries in candidates.items():
        for label, scale in entries.items():
            row = rows[arm][label]
            failures = list(baseline_failures)
            row.update(
                status="technical_failure",
                raw_status="not_attempted",
                applied_status="technical_failure",
                loss_theta_repeats=origin,
                loss_d0_repeats=base,
                loss_theta=origin[0],
                loss_d0=base[0],
                model_baseline=m0,
                raw_delta_loss=None,
                actual_gain=None,
                applied_gain=None,
                applied_direction=None,
            )
            direction = direction_at(arm, label, scale)
            if direction is None or not bool(torch.isfinite(direction).all()):
                failures.append("nonfinite direction or unresolved norm-matching scale")
                row["numerical_failures"] = failures
                continue
            norm = _number(float(direction.norm()), "step_norm", failures)
            correction_norm = _number(
                float((direction - d0).norm()), "correction_norm", failures
            )
            row.update(
                step_norm=norm,
                correction_norm=correction_norm,
                near_zero_correction=(
                    correction_norm == 0
                    or correction_norm < TOLERANCES.correction_relative_floor * norm0
                )
                if correction_norm is not None and norm0 is not None
                else None,
                norm_match_scale=scale if label.startswith("matched_") else None,
            )
            undamped = damped = damping_value = None
            try:
                hd = hd0 if label == "alpha_0" else hvp(direction)
                if not bool(torch.isfinite(hd).all()):
                    failures.append("nonfinite candidate HVP")
                undamped = _number(
                    model_value(g, direction, hd, damping=0.0),
                    "model.undamped",
                    failures,
                )
                damping_value = _number(
                    float(0.5 * direction.dot(direction)), "model.damping", failures
                )
                damped = _number(
                    model_value(g, direction, hd, damping=1.0), "model.damped", failures
                )
            except (RuntimeError, ValueError, OverflowError, TypeError) as error:
                failures.append(f"candidate HVP/model: {type(error).__name__}: {error}")
            model_delta = (
                _number(damped - m0, "model.model_delta", failures)
                if damped is not None and m0 is not None
                else None
            )
            row["model"] = {
                "undamped": undamped,
                "damping": damping_value,
                "damped": damped,
                "model_delta": model_delta,
            }
            loss_failures: list[str] = []
            repetitions = _repeat_losses(loss_at, direction, "loss", loss_failures)
            failures.extend(loss_failures)
            row.update(
                loss_repeats=repetitions,
                loss=repetitions[0],
                raw_status="technical_failure",
            )
            if repetitions[0] is not None and base[0] is not None:
                delta_l = _number(repetitions[0] - base[0], "raw_delta_loss", failures)
                row["raw_delta_loss"] = delta_l
                if delta_l is not None:
                    row.update(
                        actual_gain=-delta_l,
                        raw_status="finite"
                        if not loss_failures
                        and all(value is not None for value in base)
                        else "technical_failure",
                        strict_sign="improvement"
                        if delta_l < 0
                        else "harm"
                        if delta_l > 0
                        else "exact_tie",
                        interpretation="unresolved_preference"
                        if abs(delta_l) <= floor
                        else "improvement"
                        if delta_l < 0
                        else "harm",
                    )
            if not failures:
                assert (
                    norm is not None
                    and norm0 is not None
                    and model_delta is not None
                    and repetitions[0] is not None
                    and base[0] is not None
                )
                gates = cd._gates(
                    norm, norm0, model_delta, repetitions[0], base[0], norm_ratio
                )
                row.update(
                    status="finite",
                    gates=gates,
                    applied_status="finite",
                    applied_direction="raw" if gates["accepted"] else "d0",
                    applied_gain=row["actual_gain"] if gates["accepted"] else 0.0,
                )
                applied = row["applied_gain"]
                row["applied_strict_sign"] = (
                    "improvement"
                    if applied > 0
                    else "harm"
                    if applied < 0
                    else "exact_tie"
                )
                row["applied_interpretation"] = (
                    "unresolved_preference"
                    if abs(applied) <= floor
                    else "improvement"
                    if applied > 0
                    else "harm"
                )
            else:
                row["gates_status"] = "unavailable_technical_failure"
                row["numerical_failures"] = failures
                row["technical_error"] = "; ".join(failures)
    if reverse_order_check:
        for arm in reversed(ARMS):
            for label, scale in reversed(tuple(candidates[arm].items())):
                row = rows[arm][label]
                if row["status"] == "finite":
                    try:
                        direction = direction_at(arm, label, scale)
                        assert direction is not None
                        value = float(loss_at(direction))
                        if not math.isfinite(value):
                            raise cd.InvalidProbe("nonfinite reverse-order loss")
                        row["reverse_order_loss"] = value
                    except (
                        RuntimeError,
                        ValueError,
                        OverflowError,
                        TypeError,
                    ) as error:
                        row.update(
                            status="technical_failure",
                            applied_status="technical_failure",
                            applied_gain=None,
                            applied_direction=None,
                            technical_error=f"reverse order: {error}",
                        )
    result["loss_interaction"] = {}
    for i in range(len(SCALE_GRID)):
        label = f"alpha_{i}"
        values = [rows[arm][label] for arm in ARMS]
        if all(row["raw_status"] == "finite" for row in values):
            interaction = (
                values[0]["raw_delta_loss"]
                - values[1]["raw_delta_loss"]
                - values[2]["raw_delta_loss"]
            )
            result["loss_interaction"][label] = {
                "status": "finite" if math.isfinite(interaction) else "nonfinite",
                "delta_loss_full_minus_components": interaction
                if math.isfinite(interaction)
                else None,
            }
        else:
            result["loss_interaction"][label] = {"status": "technical_unavailable"}
    result["baselines"] = {
        "theta": origin,
        "d0": base,
        "model_d0": m0,
        "norm_d0": norm0,
        "status": "technical_failure" if baseline_failures else "finite",
        "numerical_failures": baseline_failures,
    }
    result["norm_matched_pairs"] = [f"{left}__{right}" for left, right in PAIRS]
    _measurement_counts(result)
    return result


def _capture_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    status = json.loads((path / "status.json").read_text())
    if status["kind"] != "frozen_state" or status["status"] != "complete":
        raise cd.InvalidProbe("capture not completed")
    if manifest["sha256"] != cd.file_sha256(path / "state.pt"):
        raise cd.InvalidProbe("capture bytes differ from manifest")
    if manifest["schema"] != cd.SCHEMA or manifest["phase"] != cd.PHASE:
        raise cd.InvalidProbe("capture schema or phase mismatch")
    return manifest


def inventory(root: Path, registration: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory nine capture bytes and source identities without tensor deserialization."""
    entries = []
    for seed in SEEDS:
        for step in STEPS:
            path = root / f"seed_{seed}" / f"step_{step:06d}"
            try:
                manifest = _capture_manifest(path)
                if (manifest["seed"], manifest["step"]) != (seed, step):
                    raise cd.InvalidProbe("capture seed/step identity mismatch")
                source = manifest["provenance"].get("source")
                if not source or "sha256" not in source:
                    raise cd.InvalidProbe("historical source identity missing")
                entries.append(
                    {
                        "seed": seed,
                        "step": step,
                        "status": "bytes_checked_not_loaded",
                        "capture_sha256": manifest["sha256"],
                        "capture_path": str(path.resolve()),
                        "historical_source_sha256": source["sha256"],
                    }
                )
            except (KeyError, ValueError, OSError, TypeError, cd.InvalidProbe) as error:
                entries.append(
                    {
                        "seed": seed,
                        "step": step,
                        "status": "technical_failure",
                        "reason": str(error),
                    }
                )
    if registration.get("capture_roster") and entries != registration["capture_roster"]:
        raise cd.InvalidProbe(
            "capture bytes or historical source drifted since registration"
        )
    return {
        "measurement_schema": SCHEMA,
        "current_source_sha256": registration["current_source_sha256"],
        "expected": 9,
        "valid_bytes": sum(e["status"] == "bytes_checked_not_loaded" for e in entries),
        "entries": entries,
        "tensor_deserialized": False,
        "model_loaded": False,
        "hvp_run": False,
    }


def _load_registered_capture(
    registered: Mapping[str, Any], model_factory: Callable[[Any], Any] | None
) -> dict[str, Any]:
    path = Path(registered["capture_path"])
    manifest = _capture_manifest(path)
    if (
        manifest["sha256"] != registered["capture_sha256"]
        or manifest["provenance"]["source"]["sha256"]
        != registered["historical_source_sha256"]
    ):
        raise cd.InvalidProbe(
            "loading preflight capture bytes/source differ from registration"
        )
    payload, loaded_manifest = cd.load_capture(path)
    if loaded_manifest != manifest or (payload["seed"], payload["step"]) != (
        registered["seed"],
        registered["step"],
    ):
        raise cd.InvalidProbe("loading preflight capture identity mismatch")
    d0, g0, x, y = (payload[key] for key in ("d0", "g0", "x", "y"))
    if (
        d0.ndim != 1
        or g0.shape != d0.shape
        or d0.dtype != torch.float32
        or g0.dtype != d0.dtype
        or x.dtype != d0.dtype
        or x.ndim < 2
        or y.dtype != torch.int64
        or y.ndim != 1
        or x.shape[0] != y.shape[0]
        or y.numel() == 0
    ):
        raise cd.InvalidProbe(
            "loading preflight d0/g0/prepared batch dimensions or dtypes invalid"
        )
    cd._finite(d0, g0, x, y)
    protocol = cd._config_from_payload(payload)
    if int(y.min()) < 0 or int(y.max()) >= protocol.num_classes:
        raise cd.InvalidProbe(
            "loading preflight class target outside registered class range"
        )
    if model_factory is None:
        from experiments.runner_exp7 import build_model

        model_factory = build_model
    with cd.isolated_execution(payload["runtime"], device=torch.device("cpu")):
        model = model_factory(replace(protocol, pretrained=False)).to(
            device="cpu", dtype=d0.dtype
        )
        expected = model.state_dict()
        stored = payload["state_dict"]
        if set(expected) != set(stored) or any(
            expected[name].shape != stored[name].shape
            or expected[name].dtype != stored[name].dtype
            for name in expected
        ):
            raise cd.InvalidProbe(
                "loading preflight weights/buffers shape or dtype mismatch"
            )
        cd._finite(*stored.values())
        if not all(type(mode) is bool for mode in payload["modes"].values()):
            raise cd.InvalidProbe("loading preflight invalid saved module modes")
        cd.restore_model(model, payload)
        if tuple(row["group"] for row in payload["layout"]) != GROUPS:
            raise cd.InvalidProbe(
                "loading preflight partition is not the six saved groups"
            )
        if d0.numel() != sum(p.numel() for p in group_major_params(model)):
            raise cd.InvalidProbe(
                "loading preflight stored vector length does not match layout"
            )
        with torch.no_grad():
            logits = cd._BufferedModel(model)(x)
        cd._finite(logits)
        if logits.shape != (y.numel(), protocol.num_classes):
            raise cd.InvalidProbe("loading preflight output/class shape mismatch")
        if any(
            not torch.equal(value, stored[name])
            for name, value in model.state_dict().items()
        ):
            raise cd.InvalidProbe("loading preflight mutated model state")
    return {
        **registered,
        "status": "compatible_loaded",
        "batch_shape": list(x.shape),
        "target_shape": list(y.shape),
        "dtype": str(d0.dtype),
        "parameter_count": d0.numel(),
        "num_classes": protocol.num_classes,
        "state_preserved": True,
    }


def load_preflight(
    registration_path: Path, *, model_factory: Callable[[Any], Any] | None = None
) -> dict[str, Any]:
    """Sequential CPU loading of all nine inputs; no gradient, HVP or acquisition."""
    registration = json.loads(registration_path.read_text())
    if not validate_registration(registration, cd.source_hashes()["sha256"]):
        raise cd.InvalidProbe(
            "loading preflight registration/source/environment mismatch"
        )
    entries = []
    with cd.isolated_execution(
        cd.execution_state(torch.device("cpu")), device=torch.device("cpu")
    ):
        for registered in registration["capture_roster"]:
            try:
                # The helper releases model, payload and forward outputs before the next state.
                entries.append(_load_registered_capture(registered, model_factory))
            except Exception as error:
                entries.append(
                    {
                        **registered,
                        "status": "technical_failure",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
    valid = sum(entry["status"] == "compatible_loaded" for entry in entries)
    return {
        "measurement_schema": SCHEMA,
        "status": "complete" if valid == 9 else "technical_failure",
        "registration_sha256": cd.file_sha256(registration_path),
        "current_source_sha256": registration["current_source_sha256"],
        "environment_identity": environment_identity(),
        "capture_roster": registration["capture_roster"],
        "expected": 9,
        "attempted": 9,
        "valid": valid,
        "technical_failure": 9 - valid,
        "not_attempted": 0,
        "entries": entries,
        "device": "cpu",
        "hvp_run": False,
        "gradient_run": False,
        "training_run": False,
        "pretrained_requested": False,
    }


def _geometry(
    model: Any, payload: dict[str, Any], protocol: Any, name: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    device = next(model.parameters()).device
    coupling_seed, basis_seed = (
        int(payload["seed"]) + offset for offset in DRAWSETS[name]
    )
    cur, checks = cd.replay_curvature(
        model,
        payload,
        n_probes=10,
        generator=torch.Generator(device=device).manual_seed(coupling_seed),
    )
    d0 = payload["d0"].to(device)
    hd0 = cur.hvp(d0)
    ranges = {
        row["group"]: slice(row["start"], row["stop"]) for row in payload["layout"]
    }
    split = split_residual(cur.flat_grad, d0, hd0, cur.hvp, ranges, 1.0)
    coupling = cur.coupling()
    cd._finite(*coupling.values())
    overlay = CoupleFacOverlay(rank=4, n_power_iter=8, damping=1.0, max_pairs=3)
    selected = overlay.select(coupling, 0.1)
    subspace = overlay.build(
        d0,
        cur.hvp,
        ranges,
        selected,
        generator=torch.Generator(device=device).manual_seed(basis_seed),
    )
    solved = corrections(split, subspace, overlay, d0, hd0, cur.flat_grad)
    coverage = action_coverage(
        subspace, split["off"], resolution=split["hvp_residual_resolution"]
    )
    geometry = {
        "drawset": name,
        "seeds": {"coupling": coupling_seed, "basis": basis_seed},
        "replay_checks": checks,
        "split_checks": split["checks"],
        "source_checks": split["source_checks"],
        "hvp_residual_resolution": split["hvp_residual_resolution"],
        "residual_norms": {
            arm: float(split[arm].norm()) for arm in ("full", "off", "rest")
        },
        "projected_residual_norms": {
            arm: float(subspace.project(split[arm]).norm()) if subspace else 0.0
            for arm in ("full", "off", "rest")
        },
        "coverage": coverage,
        "selected_pairs": [list(pair) for pair in selected],
        "coupling": [
            {"pair": list(pair), "value": value} for pair, value in coupling.items()
        ],
        "correction_additivity": solved["additivity"],
        "correction_norms": {
            arm: float(solved[arm]["delta"].norm()) for arm in ("full", "off", "rest")
        },
        "off_rest_dot": solved["off_rest_dot"],
    }
    return (
        cur,
        {"split": split, "subspace": subspace, "solved": solved, "hd0": hd0},
        geometry,
    )


def _result_counts(result: dict[str, Any]) -> None:
    draws = result["drawsets"]
    for draw in draws.values():
        _measurement_counts(draw["measurements"])
    result["drawset_counts"] = {
        "expected": len(DRAWSETS),
        "attempted": sum(draw["status"] != "not_attempted" for draw in draws.values()),
        "valid": sum(draw["status"] == "complete" for draw in draws.values()),
        "technical_failure": sum(
            draw["status"] == "technical_failure" for draw in draws.values()
        ),
        "not_attempted": sum(
            draw["status"] == "not_attempted" for draw in draws.values()
        ),
    }
    for field in ("counts", "raw_counts", "applied_counts"):
        result[field] = {
            arm: {
                key: sum(
                    draw["measurements"][field][arm][key] for draw in draws.values()
                )
                for key in (
                    "expected",
                    "attempted",
                    "valid",
                    "technical_failure",
                    "not_attempted",
                    "unresolved",
                )
            }
            for arm in ARMS
        }
    result["count_semantics"] = (
        "unresolved is a subset of valid, not an additional outcome; steps/draws are not independent training seeds"
    )


def _write_result(path: Path, result: dict[str, Any]) -> None:
    """Keep scalar failure evidence JSON-safe without replacing nonfinite values by zero."""
    problems: list[str] = []

    def clean(value: Any, field: str) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            problems.append(f"{field}: nonfinite {value}")
            return None
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = clean(item, f"{field}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = clean(item, f"{field}[{i}]")
        elif isinstance(value, tuple):
            return [clean(item, f"{field}[{i}]") for i, item in enumerate(value)]
        return value

    clean(result, "result")
    if problems:
        result.setdefault("nonfinite_fields", []).extend(problems)
        result["status"] = "technical_failure"
    _result_counts(result)
    cd.write_json(path, result)
    if problems:
        raise cd.InvalidProbe(
            "nonfinite scalar diagnostics retained as null with explicit field reasons"
        )


def measure_capture(
    capture: Path,
    output: Path,
    registration_path: Path,
    *,
    device: torch.device,
    phase: str,
    thresholds: Path | None = None,
    model_factory: Callable[[Any], Any] | None = None,
    protected_root: Path | None = None,
) -> dict[str, Any]:
    """One identified state; complete failure roster before loading or any HVP."""
    if phase not in ("smoke", "final") or (phase == "final") != (
        thresholds is not None
    ):
        raise ValueError(
            "final requires frozen thresholds; smoke must calibrate its own"
        )
    registration = json.loads(registration_path.read_text())
    current = cd.source_hashes()
    if not validate_registration(registration, current["sha256"]):
        raise cd.InvalidProbe(
            "registered current source/config/environment differs from installed source"
        )
    if phase == "smoke" and registration["registration_phase"] != "smoke":
        raise cd.InvalidProbe("smoke requires a smoke registration")
    fixed = (
        _final_thresholds(registration, thresholds) if thresholds is not None else None
    )
    registered = next(
        (
            row
            for row in registration["capture_roster"]
            if Path(row["capture_path"]).resolve() == capture.resolve()
        ),
        None,
    )
    if registered is None:
        raise cd.InvalidProbe("requested capture path is not registered")
    seed, step = registered["seed"], registered["step"]
    if phase == "smoke" and (seed, step) != (42, 60):
        raise cd.InvalidProbe("technical smoke is seed 42 step 60 only")
    guard_output(
        output,
        [Path(row["capture_path"]) for row in registration["capture_roster"]]
        + [capture, registration_path]
        + ([thresholds] if thresholds is not None else [])
        + (
            [
                Path(entry["path"])
                for entry in registration["smoke_calibration"].values()
            ]
            if thresholds is not None
            else []
        ),
        protected_root=protected_root,
    )
    registration_sha = cd.file_sha256(registration_path)
    with (
        cd.ArtifactDirectory(
            output, kind=f"residual_ablation_v{SCHEMA}_{phase}"
        ) as run,
        cd.isolated_execution(cd.execution_state(device), device=device),
    ):
        result: dict[str, Any] = {
            "measurement_schema": SCHEMA,
            "phase": phase,
            "status": "running",
            "requested_capture": str(capture.resolve()),
            "seed": seed,
            "step": step,
            "capture_sha256": registered["capture_sha256"],
            "historical_source_sha256": registered["historical_source_sha256"],
            "current_source_sha256": current["sha256"],
            "registration": {
                "path": str(registration_path.resolve()),
                "sha256": registration_sha,
                "phase": registration["registration_phase"],
            },
            "historical_probe_replay": False,
            "d0_recomputed": False,
            "device": str(device),
            "dtype": None,
            "runtime_flags": None,
            "environment": environment_block(),
            "environment_identity": environment_identity(),
            "capture_status": "not_attempted",
            "technical_validation": "not_run",
            "loading_preflight": {"status": "not_attempted", "expected": 9},
            "drawsets": {
                name: {
                    "drawset": name,
                    "status": "not_attempted",
                    "seed": seed,
                    "step": step,
                    "seeds": {
                        "coupling": seed + offsets[0],
                        "basis": seed + offsets[1],
                    },
                    "measurements": _empty_measurements(),
                }
                for name, offsets in DRAWSETS.items()
            },
        }
        _write_result(run.path / "results.json", result)
        run.status["measurement_identity"] = {
            key: result[key]
            for key in (
                "measurement_schema",
                "phase",
                "seed",
                "step",
                "requested_capture",
                "capture_sha256",
                "current_source_sha256",
                "registration",
            )
        }
        run.flush()
        try:
            if phase == "smoke":
                with run.stage("all_nine_cpu_loading_preflight"):
                    result["loading_preflight"] = load_preflight(
                        registration_path, model_factory=model_factory
                    )
                    cd.write_json(
                        run.path / "loading_preflight.json", result["loading_preflight"]
                    )
                    _write_result(run.path / "results.json", result)
                    if result["loading_preflight"]["status"] != "complete":
                        raise cd.InvalidProbe(
                            "all-nine CPU loading/compatibility preflight failed before any HVP"
                        )
            else:
                result["loading_preflight"] = {
                    "status": "bound_to_frozen_smoke",
                    "smoke_results": registration["smoke_calibration"]["smoke_results"],
                }
            with run.stage("load_and_validate"):
                result["capture_status"] = "running"
                _write_result(run.path / "results.json", result)
                manifest = _capture_manifest(capture)
                payload, loaded_manifest = cd.load_capture(capture)
                if (
                    manifest != loaded_manifest
                    or (payload["seed"], payload["step"]) != (seed, step)
                    or registered["capture_sha256"] != manifest["sha256"]
                    or registered["historical_source_sha256"]
                    != manifest["provenance"]["source"]["sha256"]
                ):
                    raise cd.InvalidProbe(
                        "capture bytes/identity/historical source differ from registration"
                    )
                if payload["d0"].dtype != torch.float32:
                    raise cd.InvalidProbe("saved direction is not float32")
                protocol = cd._config_from_payload(payload)
                for field, expected in (
                    ("damping", 1.0),
                    ("tau", 0.1),
                    ("coupling_n_probes_online", 10),
                    ("overlay_rank", 4),
                    ("overlay_n_power_iter", 8),
                    ("overlay_max_pairs", 3),
                    ("tr_ratio", 2.0),
                ):
                    if getattr(protocol, field) != expected:
                        raise cd.InvalidProbe(f"saved protocol differs: {field}")
                if model_factory is None:
                    from experiments.runner_exp7 import build_model

                    model_factory = build_model
                with cd.isolated_execution(payload["runtime"], device=device):
                    model = model_factory(replace(protocol, pretrained=False)).to(
                        device=device, dtype=payload["d0"].dtype
                    )
                cd.restore_model(model, payload)
                params = group_major_params(model)
                d0 = payload["d0"].to(device)
                if (
                    d0.numel() != sum(p.numel() for p in params)
                    or d0.shape != payload["g0"].shape
                ):
                    raise cd.InvalidProbe(
                        "stored direction does not match parameter layout"
                    )
                x, y = payload["x"].to(device), payload["y"].to(device)
                raw_loss = trial_loss(model, nn.CrossEntropyLoss(), params, 1.0)(x, y)

                def loss_at(direction: Tensor) -> float:
                    with cd.isolated_execution(payload["runtime"], device=device):
                        return raw_loss(direction)

                result.update(
                    dtype=str(d0.dtype),
                    runtime_flags=payload["runtime"]["backend"],
                    capture_status="complete",
                )
                _write_result(run.path / "results.json", result)
            repeats: list[float] = []
            for name in DRAWSETS:
                result["drawsets"][name]["status"] = "running"
                _write_result(run.path / "results.json", result)
                with (
                    run.stage(f"geometry_and_measure_{name}"),
                    cd.isolated_execution(payload["runtime"], device=device),
                ):
                    forward = trial_forward(model, params, 1.0)
                    flat_loss, flat_g, flat_hvp = cd.displacement_curvature(
                        forward, nn.CrossEntropyLoss(), x, y, torch.zeros_like(d0)
                    )
                    flat_hd0 = flat_hvp(d0)
                    del flat_hvp
                    cur, bundle, geometry = _geometry(model, payload, protocol, name)
                    geometry.update(
                        status="running",
                        seed=seed,
                        step=step,
                        measurements=result["drawsets"][name]["measurements"],
                    )
                    result["drawsets"][name] = geometry
                    subspace = bundle["subspace"]
                    metadata = {
                        "measurement_schema": SCHEMA,
                        "capture_sha256": manifest["sha256"],
                        "current_source_sha256": current["sha256"],
                        "registration_sha256": registration_sha,
                        "seed": seed,
                        "step": step,
                        "drawset": name,
                        "seeds": geometry["seeds"],
                    }
                    factor_file = run.path / f"factored_{name}.pt"
                    cd.save_tensors(
                        factor_file,
                        {
                            **metadata,
                            "subspace": cd._subspace_payload(
                                subspace
                                if subspace is not None
                                else cd._empty_subspace(d0)
                            ),
                        },
                    )
                    geometry["factored_geometry"] = {
                        **metadata,
                        "filename": factor_file.name,
                        "sha256": cd.file_sha256(factor_file),
                    }
                    flat_check = cd.agreement(
                        flat_g,
                        cur.flat_grad,
                        rtol=TOLERANCES.grad_rtol,
                        atol=TOLERANCES.atol,
                    )
                    hvp_check = cd.agreement(
                        flat_hd0,
                        bundle["hd0"],
                        rtol=TOLERANCES.hvp_rtol,
                        atol=TOLERANCES.atol,
                    )
                    loss_check = cd.agreement(
                        flat_loss,
                        cur.loss,
                        rtol=TOLERANCES.grad_rtol,
                        atol=TOLERANCES.atol,
                    )
                    geometry["independent_flat_ad"] = {
                        "loss": loss_check,
                        "gradient": flat_check,
                        "hvp_d0": hvp_check,
                    }
                    _write_result(run.path / "results.json", result)
                    if not all(
                        check["passed"] for check in (loss_check, flat_check, hvp_check)
                    ):
                        raise cd.InvalidProbe(
                            "independent flat-AD loss/gradient/HVP mismatch"
                        )
                    geometry["measurements"] = measure_candidates(
                        d0,
                        cur.flat_grad,
                        bundle["hd0"],
                        cur.hvp,
                        bundle["solved"],
                        loss_at,
                        floor=fixed["gain_floor"] if fixed else TOLERANCES.gain_floor,
                        reverse_order_check=phase == "smoke",
                    )
                    measurements = geometry["measurements"]
                    _write_result(run.path / "results.json", result)
                    if any(
                        counts["technical_failure"]
                        for counts in measurements["counts"].values()
                    ):
                        raise cd.InvalidProbe(
                            "candidate technical failures retained; measurement stopped"
                        )
                    for arm in measurements["arms"].values():
                        for row in arm.values():
                            repeats.append(
                                max(row["loss_repeats"]) - min(row["loss_repeats"])
                            )
                            if "reverse_order_loss" in row:
                                repeats.append(
                                    abs(
                                        row["reverse_order_loss"]
                                        - row["loss_repeats"][0]
                                    )
                                )
                    for key in ("theta", "d0"):
                        values = measurements["baselines"][key]
                        repeats.append(max(values) - min(values))
                    geometry["status"] = "complete"
                    _write_result(run.path / "results.json", result)
                    del cur, bundle
            error = max(repeats, default=0.0)
            if fixed is None:
                fixed = {
                    "measurement_schema": SCHEMA,
                    "source_sha256": current["sha256"],
                    "capture_seed": 42,
                    "capture_step": 60,
                    "capture_sha256": manifest["sha256"],
                    "tolerances": asdict(TOLERANCES),
                    "gain_floor": max(1e-4, 10 * error),
                    "same_candidate_repeatability_spread": error,
                    "role": "technical_smoke_only_not_final_data",
                    "smoke_registration_sha256": registration_sha,
                    "environment_identity": environment_identity(),
                }
                _validate_thresholds(fixed, registration, registration_sha)
                cd.write_json(run.path / "thresholds.json", fixed)
            elif error > fixed["gain_floor"] / 10:
                raise cd.InvalidProbe(
                    "final same-candidate repeatability exceeds frozen gain floor"
                )
            for geometry in result["drawsets"].values():
                for arm in geometry["measurements"]["arms"].values():
                    for row in arm.values():
                        if row["raw_status"] == "finite":
                            raw = row["raw_delta_loss"]
                            row["interpretation"] = (
                                "unresolved_preference"
                                if abs(raw) <= fixed["gain_floor"]
                                else "improvement"
                                if raw < 0
                                else "harm"
                            )
                        if row["applied_status"] == "finite":
                            gain = row["applied_gain"]
                            row["applied_interpretation"] = (
                                "unresolved_preference"
                                if abs(gain) <= fixed["gain_floor"]
                                else "improvement"
                                if gain > 0
                                else "harm"
                            )
            preserved = all(
                torch.equal(value.detach().cpu(), payload["state_dict"][key])
                for key, value in model.state_dict().items()
            )
            result["state_preservation_check"] = {"passed": preserved}
            if not preserved:
                raise cd.InvalidProbe("measurement mutated restored model state")
            result.update(
                thresholds=fixed,
                status="complete",
                technical_validation="measured_requires_external_smoke_or_result_review",
                resources={
                    "elapsed_seconds": time.monotonic() - run.start,
                    "process_peak_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                },
            )
            _write_result(run.path / "results.json", result)
            return result
        except Exception as error:
            result["status"] = "technical_failure"
            result["technical_error"] = f"{type(error).__name__}: {error}"
            if result["capture_status"] == "running":
                result["capture_status"] = "technical_failure"
            for draw in result["drawsets"].values():
                if draw["status"] == "running":
                    draw.update(status="technical_failure", technical_error=str(error))
                elif draw["status"] == "not_attempted":
                    draw["not_run_reason"] = "blocked_by_prior_technical_failure"
            _write_result(run.path / "results.json", result)
            raise


def reject_legacy_summary(root: Path) -> None:
    """Do not let the historical summarizer silently drop schema-2 cohorts."""
    for path in root.rglob("status.json"):
        if (
            json.loads(path.read_text())
            .get("kind", "")
            .startswith("residual_ablation_v")
        ):
            raise cd.InvalidProbe(
                "legacy summarize does not support residual-ablation records; use their explicit results/accounting schema"
            )
