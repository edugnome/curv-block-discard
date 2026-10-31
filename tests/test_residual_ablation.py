"""Synthetic CPU contracts for the saved-state residual-input intervention."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from experiments import causal_diagnostics as cd
from experiments import residual_ablation as ra
from experiments.config import Exp7Config
from experiments.training import StepCtx
from secondorder.overlay import CoupleFacOverlay, OverlaySubspace


def six_group_oracle(block_diagonal: bool = False):
    h = torch.diag(torch.tensor([2.0, -0.5, 1.0, 0.7, 3.0, 1.5]))
    if not block_diagonal:
        h[0, 4] = h[4, 0] = 0.6
        h[1, 5] = h[5, 1] = -0.4
    ranges = {name: slice(i, i + 1) for i, name in enumerate(ra.GROUPS)}
    return h, ranges


def factored_identity(h: torch.Tensor) -> OverlaySubspace:
    dim = h.shape[0]
    eye = torch.eye(dim)
    values, vectors = torch.linalg.eigh(h)
    return OverlaySubspace((), ((slice(0, dim), eye),), eye, dim, values, vectors)


def test_six_masked_sources_split_and_native_float32_full():
    h, ranges = six_group_oracle()
    d0 = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
    g = -h @ d0 - d0
    split = ra.split_residual(g, d0, h @ d0, lambda v: h @ v, ranges, 1.0)
    assert len(split["source_checks"]) == 6
    assert torch.allclose(split["off"], (h - torch.diag(h.diagonal())) @ d0)
    assert torch.allclose(split["full"], split["off"] + split["rest"], atol=1e-6)
    assert torch.equal(split["full"], g + h @ d0 + d0)
    assert split["checks"]["passed"]
    u = factored_identity(h)
    overlay = CoupleFacOverlay(damping=1.0)
    corrections = ra.corrections(split, u, overlay, d0, h @ d0, g)
    assert torch.equal(
        corrections["full"]["direction"], overlay.correct(g, d0, h @ d0, u)
    )
    assert corrections["additivity"]["passed"]
    values, vectors = torch.linalg.eigh(h)
    matrix_absolute = (vectors * values.abs()) @ vectors.T
    assert torch.allclose(
        corrections["off"]["delta"],
        -torch.linalg.solve(matrix_absolute + torch.eye(6), split["off"]),
        atol=1e-6,
    )
    assert torch.allclose(
        corrections["full"]["delta"],
        corrections["off"]["delta"] + corrections["rest"]["delta"],
        atol=1e-6,
    )


def test_native_direction_survives_rounded_correction_recovery():
    width = 3_000
    d0 = torch.ones(6 * width)
    ranges = {
        name: slice(i * width, (i + 1) * width) for i, name in enumerate(ra.GROUPS)
    }
    stem, head = ranges["stem"], ranges["head"]

    def hvp(vector):
        action = 2 * vector
        action[stem] += 0.5 * vector[head]
        action[head] += 0.5 * vector[stem]
        return action

    hd0 = hvp(d0)
    g = -hd0 - d0 + 2.0**-21
    split = ra.split_residual(g, d0, hd0, hvp, ranges, 1.0)
    overlay = CoupleFacOverlay(rank=4, n_power_iter=8, damping=1, max_pairs=3)
    subspace = overlay.build(
        d0,
        hvp,
        ranges,
        [("stem", "head")],
        generator=torch.Generator().manual_seed(42),
    )
    assert subspace is not None
    delta = overlay._reduced_newton(subspace, split["full"])
    native_direction = overlay.correct(g, d0, hd0, subspace)
    assert torch.equal(native_direction, d0 + delta)
    recovered_check = ra._check(
        (native_direction - d0) - delta,
        (native_direction - d0, delta),
        ra.TOLERANCES.grad_rtol,
    )
    assert recovered_check["absolute_error"] > recovered_check["threshold"]
    solved = ra.corrections(split, subspace, overlay, d0, hd0, g)
    assert torch.equal(solved["full"]["direction"], native_direction)
    assert solved["additivity"]["passed"]
    native_check = solved["additivity"]["native_full_check"]
    assert native_check["passed"] and native_check["bitwise_equal"]
    assert native_check["comparison_space"] == "direction"
    assert native_check["absolute_error"] == 0
    assert not solved["additivity"]["recovered_correction_check"]["passed"]
    assert solved["additivity"]["recovered_correction_check"]["diagnostic_only"]


def test_calibration_capture_mismatch_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    roster = [
        {
            "seed": seed,
            "step": step,
            "status": "bytes_checked_not_loaded",
            "capture_sha256": f"registered-{seed}-{step}",
            "capture_path": str(
                tmp_path / "inputs" / f"seed_{seed}" / f"step_{step:06d}"
            ),
            "historical_source_sha256": "historical",
        }
        for seed in ra.SEEDS
        for step in ra.STEPS
    ]
    source_sha = cd.source_hashes()["sha256"]
    registration = ra.registration_template(source_sha, roster)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration))
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "measurement_schema": ra.SCHEMA,
                "source_sha256": source_sha,
                "capture_seed": 42,
                "capture_step": 60,
                "capture_sha256": "different-calibration-capture",
                "tolerances": registration["tolerances"],
                "gain_floor": 1e-4,
            }
        )
    )

    def forbidden_loading(_):
        pytest.fail("mismatched calibration reached capture loading")

    monkeypatch.setattr(ra, "_capture_manifest", forbidden_loading)
    with pytest.raises(cd.InvalidProbe):
        ra.freeze_final_registration(
            registration_path, thresholds_path, tmp_path / "new-final"
        )
    assert not (tmp_path / "new-final").exists()


def test_zero_off_and_zero_rest_baselines():
    h, ranges = six_group_oracle(block_diagonal=True)
    d0 = torch.arange(1, 7, dtype=torch.float32) / 10
    g = -(h @ d0 + d0)
    split = ra.split_residual(g, d0, h @ d0, lambda v: h @ v, ranges, 1.0)
    assert torch.equal(split["off"], torch.zeros_like(d0))
    assert torch.allclose(split["rest"], torch.zeros_like(d0), atol=1e-6)
    assert (
        ra.action_coverage(None, split["off"], resolution=1e-6)["status"]
        == "zero_denominator"
    )


def test_action_coverage_empty_unresolved_nonorthogonal_and_factored():
    h, _ = six_group_oracle()
    v = torch.ones(6)
    assert ra.action_coverage(None, v, resolution=1e-6)["eta_action"] == 0.0
    assert ra.action_coverage(factored_identity(h), v, resolution=1e-6)[
        "eta_action"
    ] == pytest.approx(1.0, abs=1e-5)
    assert (
        ra.action_coverage(None, v * 1e-8, resolution=1e-6)["status"]
        == "unresolved_denominator"
    )
    bad = OverlaySubspace(
        (),
        ((slice(0, 6), torch.eye(6)),),
        2 * torch.eye(6),
        6,
        torch.zeros(6),
        torch.eye(6),
    )
    assert (
        ra.action_coverage(bad, v, resolution=1e-6)["status"] == "nonorthogonal_basis"
    )


def test_nonorthogonal_action_coverage_retains_raw_double_dot_energy():
    off = torch.ones(6, dtype=torch.float32)
    subspace = OverlaySubspace(
        (),
        ((slice(0, 6), torch.eye(6)),),
        2 * torch.eye(6),
        6,
        torch.zeros(6),
        torch.eye(6),
    )
    coordinates = subspace.project(off).double()
    numerator = float(coordinates.dot(coordinates))
    denominator = float(off.double().dot(off.double()))
    assert numerator != float(subspace.project(off).norm()) ** 2

    coverage = ra.action_coverage(subspace, off, resolution=1e-6)

    assert coverage["numerator"] == numerator
    assert coverage["denominator"] == denominator
    assert coverage["eta_action"] is None
    assert coverage["status"] == "nonorthogonal_basis"
    assert coverage["orthonormality_error"] > ra.TOLERANCES.grad_rtol


def test_model_harm_allowed_for_component_and_raw_gates_preserved():
    h, ranges = six_group_oracle()
    d0 = torch.ones(6) / 10
    g = -h @ d0 - d0
    split = ra.split_residual(g, d0, h @ d0, lambda v: h @ v, ranges, 1.0)
    subspace = factored_identity(h)
    correction = ra.corrections(
        split, subspace, CoupleFacOverlay(damping=1), d0, h @ d0, g
    )

    def loss(d):
        return float(5.0 + g @ d + 0.5 * d @ h @ d)

    measured = ra.measure_candidates(
        d0, g, h @ d0, lambda v: h @ v, correction, loss, floor=1e-4, norm_ratio=0.5
    )
    rows = measured["arms"]
    assert measured["applied_counts"]["off"]["unresolved"] == 9
    assert measured["unresolved_is_subset_of_valid"]
    full = rows["full"]["alpha_6"]
    assert full["status"] == "finite" and full["raw_delta_loss"] is not None
    assert full["gates"]["ratio"] is False
    assert full["applied_gain"] == 0 and full["applied_direction"] == "d0"
    assert full["applied_strict_sign"] == "exact_tie"
    assert full["applied_interpretation"] == "unresolved_preference"
    assert rows["full"]["alpha_0"]["raw_delta_loss"] == 0
    assert rows["full"]["alpha_0"]["strict_sign"] == "exact_tie"
    assert rows["full"]["alpha_0"]["interpretation"] == "unresolved_preference"
    assert (
        rows["off"]["alpha_6"]["model"]["damped"]
        > rows["off"]["alpha_0"]["model"]["damped"]
    )
    assert len(rows["rest"]) == len(ra.SCALE_GRID) + 2


def test_no_selected_pairs_give_exact_base_across_grid():
    h, ranges = six_group_oracle()
    d0 = torch.ones(6) / 10
    g = torch.ones(6) / 5
    split = ra.split_residual(g, d0, h @ d0, lambda v: h @ v, ranges, 1.0)
    solved = ra.corrections(split, None, CoupleFacOverlay(damping=1), d0, h @ d0, g)
    assert solved["additivity"]["passed"]
    assert all(
        torch.equal(solved[name]["direction"], d0) for name in ("full", "off", "rest")
    )
    measured = ra.measure_candidates(
        d0, g, h @ d0, lambda v: h @ v, solved, lambda _: 2.0, floor=1e-4
    )
    for arm in measured["arms"].values():
        assert all(row["strict_sign"] == "exact_tie" for row in arm.values())
        assert all(row["raw_delta_loss"] == 0 for row in arm.values())


def test_pairwise_norm_matching_only_shrinks():
    corrections = {
        "full": torch.tensor([4.0, 0.0]),
        "off": torch.tensor([0.0, 2.0]),
        "rest": torch.zeros(2),
    }
    matched = ra.norm_matched(corrections)
    for key, pair in matched.items():
        left, right = key.split("__")
        assert pair[left].norm() <= corrections[left].norm()
        assert pair[right].norm() <= corrections[right].norm()
        assert pair[left].norm() == pytest.approx(pair[right].norm(), abs=1e-6)


def test_execution_isolation_on_failure_and_legacy_source_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    before = cd.execution_state()
    with pytest.raises(RuntimeError):
        with cd.isolated_execution(before):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("injected")
    after = cd.execution_state()
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])
    assert (
        ra.validate_registration(
            {"measurement_schema": ra.SCHEMA, "current_source_sha256": "other"},
            "actual",
        )
        is False
    )


def test_cli_inventory_uses_distinct_versioned_cohorts():
    from experiments.pilot import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "residual-ablation",
            "inventory",
            "--capture-root",
            "/abs/captures",
            "--output",
            "/abs/new",
            "--registration",
            "/abs/registration.json",
        ]
    )
    assert args.command == "residual-ablation"
    assert args.phase == "inventory"
    assert json.dumps(ra.registration_template("abc"), allow_nan=False)


def test_capture_registration_inventory_never_deserializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "retained"
    for seed in ra.SEEDS:
        for step in ra.STEPS:
            folder = root / f"seed_{seed}" / f"step_{step:06d}"
            folder.mkdir(parents=True)
            state = folder / "state.pt"
            state.write_bytes(f"static synthetic bytes {seed} {step}".encode())
            manifest = {
                "schema": cd.SCHEMA,
                "phase": cd.PHASE,
                "seed": seed,
                "step": step,
                "sha256": cd.file_sha256(state),
                "provenance": {"source": {"sha256": "historical"}},
            }
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (folder / "status.json").write_text(
                json.dumps({"kind": "frozen_state", "status": "complete"})
            )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("inventory loaded a tensor")
    )
    entries = ra.inventory(root, ra.registration_template("new"))
    assert entries["valid_bytes"] == entries["expected"] == 9
    registration = ra.registration_template("new", entries["entries"])
    assert ra.validate_registration(registration, "new")
    assert not ra.validate_registration(registration, "changed")
    assert ra.inventory(root, registration)["tensor_deserialized"] is False
    (root / "seed_42" / "step_000060" / "state.pt").write_bytes(b"corrupt")
    with pytest.raises(cd.InvalidProbe):
        ra.inventory(root, registration)


def test_failed_candidate_is_technical_not_scientific_negative():
    h, ranges = six_group_oracle()
    d0 = torch.ones(6) / 10
    g = -(h @ d0 + d0)
    split = ra.split_residual(g, d0, h @ d0, lambda x: h @ x, ranges, 1.0)
    solved = ra.corrections(
        split, factored_identity(h), CoupleFacOverlay(damping=1), d0, h @ d0, g
    )

    def fail_nonzero(direction):
        if not torch.equal(direction, d0) and not torch.equal(
            direction, torch.zeros_like(d0)
        ):
            raise ValueError("synthetic forward failed")
        return 1.0

    results = ra.measure_candidates(
        d0, g, h @ d0, lambda x: h @ x, solved, fail_nonzero, floor=1e-4
    )
    failed = results["arms"]["off"]["alpha_6"]
    assert failed["status"] == "technical_failure"
    assert "gates" not in failed
    assert failed["raw_delta_loss"] is None
    assert failed["raw_status"] == "technical_failure"
    assert results["counts"]["off"]["technical_failure"] >= 1
    assert (
        sum(results["counts"]["off"][key] for key in ("valid", "technical_failure"))
        == results["counts"]["off"]["expected"]
    )


class SixGroupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.groups = nn.ModuleDict(
            {name: nn.Linear(2, 2, bias=False) for name in ra.GROUPS}
        )
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(sum(layer(x) for layer in self.groups.values()))

    def get_param_groups(self):
        return {name: list(self.groups[name].parameters()) for name in ra.GROUPS}


class SyntheticProvider:
    def update(self, x, y):
        return None

    def base_direction(self, grad):
        return -0.05 * grad


def test_synthetic_saved_state_smoke_and_separate_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from experiments.pilot import main

    torch.manual_seed(11)
    model = SixGroupModel().train()
    protocol = replace(Exp7Config.b1_cars(), pretrained=False, num_classes=2)
    records: list[dict[str, Any]] = []
    hook = cd.build_capture_hook(
        model,
        protocol,
        seed=42,
        steps=[60],
        sink=records.append,
        provider=SyntheticProvider(),
    )
    x = torch.tensor([[1.0, 0.3], [0.2, 0.4], [-0.3, 0.7], [-0.8, 1.4]])
    y = torch.tensor([0, 1, 0, 1])
    hook(StepCtx(model, x, y, 60, torch.device("cpu")))
    root = tmp_path / "synthetic_captures"
    for seed in ra.SEEDS:
        for step in ra.STEPS:
            payload = {**records[0], "seed": seed, "step": step}
            (root / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
            cd.write_capture(
                root / f"seed_{seed}" / f"step_{step:06d}",
                payload,
                provenance={"source": {"sha256": "historical-version"}},
            )
    registration = tmp_path / "registration.json"
    assert (
        main(
            [
                "residual-ablation",
                "register",
                "--capture-root",
                str(root),
                "--output",
                str(registration),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "residual-ablation",
                "inventory",
                "--capture-root",
                str(root),
                "--registration",
                str(registration),
                "--output",
                str(tmp_path / "inventory.json"),
            ]
        )
        == 0
    )
    original = cd.replay_curvature

    def with_fixed_synthetic_coupling(*args, **kwargs):
        cur, checks = original(*args, **kwargs)
        return replace(cur, coupling=lambda: {("stem", "head"): 1.0}), checks

    monkeypatch.setattr(cd, "replay_curvature", with_fixed_synthetic_coupling)

    def factory(protocol):
        assert protocol.pretrained is False
        return SixGroupModel()

    before = cd.execution_state()
    with monkeypatch.context() as isolated:

        def no_gradient(*args, **kwargs):
            pytest.fail("CPU loading preflight attempted a gradient/HVP")

        isolated.setattr(torch.autograd, "grad", no_gradient)
        loading = ra.load_preflight(registration, model_factory=factory)
    assert loading["status"] == "complete"
    assert loading["valid"] == loading["attempted"] == loading["expected"] == 9
    smoke = ra.measure_capture(
        root / "seed_42" / "step_000060",
        tmp_path / "smoke",
        registration,
        device=torch.device("cpu"),
        phase="smoke",
        model_factory=factory,
    )
    final_registration = tmp_path / "final-registration.json"
    assert (
        main(
            [
                "residual-ablation",
                "freeze",
                "--smoke-registration",
                str(registration),
                "--thresholds",
                str(tmp_path / "smoke" / "thresholds.json"),
                "--output",
                str(final_registration),
            ]
        )
        == 0
    )
    final = ra.measure_capture(
        root / "seed_42" / "step_000060",
        tmp_path / "final",
        final_registration,
        device=torch.device("cpu"),
        phase="final",
        thresholds=tmp_path / "smoke" / "thresholds.json",
        model_factory=factory,
    )
    assert smoke["phase"] == "smoke" and final["phase"] == "final"
    assert smoke["historical_source_sha256"] != smoke["current_source_sha256"]
    for result in (smoke, final):
        assert set(result["drawsets"]) == set(ra.DRAWSETS)
        for draw in result["drawsets"].values():
            assert draw["split_checks"]["passed"]
            assert draw["independent_flat_ad"]["hvp_d0"]["passed"]
            assert draw["measurements"]["counts"]["full"]["expected"] == 9
            assert draw["measurements"]["arms"]["full"]["alpha_6"]["status"] == "finite"
            parent = tmp_path / result["phase"]
            artifact = draw["factored_geometry"]
            assert artifact["filename"] == f"factored_{draw['drawset']}.pt"
            assert artifact["sha256"] == cd.file_sha256(parent / artifact["filename"])
            factors = torch.load(parent / artifact["filename"], weights_only=True)
            for key in (
                "capture_sha256",
                "current_source_sha256",
                "drawset",
                "seed",
                "step",
                "registration_sha256",
            ):
                assert factors[key] == artifact[key]
            assert "subspace" in factors
    assert len(list((tmp_path / "smoke").glob("factored_*.pt"))) == 2
    assert before["python"] == cd.execution_state()["python"]
    assert torch.equal(before["torch_cpu"], cd.execution_state()["torch_cpu"])
    with pytest.raises(FileExistsError):
        ra.measure_capture(
            root / "seed_42" / "step_000060",
            tmp_path / "final",
            final_registration,
            device=torch.device("cpu"),
            phase="final",
            thresholds=tmp_path / "smoke" / "thresholds.json",
            model_factory=factory,
        )

    threshold_path = tmp_path / "smoke" / "thresholds.json"
    accepted_bytes = threshold_path.read_bytes()
    for field, value in (
        ("capture_sha256", "0" * 64),
        ("gain_floor", 42.0),
        ("role", "unrelated"),
        ("same_candidate_repeatability_spread", float("nan")),
    ):
        altered = json.loads(accepted_bytes)
        altered[field] = value
        threshold_path.write_text(json.dumps(altered))
        try:
            with pytest.raises(cd.InvalidProbe):
                ra.freeze_final_registration(
                    registration,
                    threshold_path,
                    tmp_path / f"invalid-freeze-{field}.json",
                )
            with pytest.raises(cd.InvalidProbe):
                ra.measure_capture(
                    root / "seed_42" / "step_000060",
                    tmp_path / f"blocked-final-{field}",
                    final_registration,
                    device=torch.device("cpu"),
                    phase="final",
                    thresholds=threshold_path,
                    model_factory=factory,
                )
            assert not (tmp_path / f"blocked-final-{field}").exists()
        finally:
            threshold_path.write_bytes(accepted_bytes)
    assert (
        main(
            [
                "summarize",
                "--input-root",
                str(tmp_path / "smoke"),
                "--output",
                str(tmp_path / "wrong-legacy-summary.json"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "wrong-legacy-summary.json").exists()

    with monkeypatch.context() as isolated:

        def broken_primary(*args, **kwargs):
            raise cd.InvalidProbe("injected primary geometry failure")

        isolated.setattr(ra, "_geometry", broken_primary)
        with pytest.raises(cd.InvalidProbe):
            ra.measure_capture(
                root / "seed_42" / "step_000060",
                tmp_path / "primary-failed",
                registration,
                device=torch.device("cpu"),
                phase="smoke",
                model_factory=factory,
            )
    partial = json.loads((tmp_path / "primary-failed" / "results.json").read_text())
    assert partial["drawsets"]["primary"]["status"] == "technical_failure"
    assert partial["drawsets"]["sensitivity"]["status"] == "not_attempted"
    assert (
        partial["drawset_counts"]["attempted"]
        == partial["drawset_counts"]["technical_failure"]
        == 1
    )
    assert partial["drawset_counts"]["not_attempted"] == 1
    assert partial["counts"]["full"]["not_attempted"] == 18

    # A byte-valid, loadable capture with incompatible class targets blocks smoke.
    bad_capture = root / "seed_44" / "step_000200"
    bad_payload = torch.load(bad_capture / "state.pt", weights_only=True)
    bad_payload["y"] = torch.full_like(bad_payload["y"], 999)
    torch.save(bad_payload, bad_capture / "state.pt")
    bad_manifest = json.loads((bad_capture / "manifest.json").read_text())
    bad_manifest["sha256"] = cd.file_sha256(bad_capture / "state.pt")
    bad_manifest["batch_hashes"]["y"] = cd.tensor_sha256(bad_payload["y"])
    (bad_capture / "manifest.json").write_text(json.dumps(bad_manifest))
    bad_registration = tmp_path / "incompatible-registration.json"
    assert (
        main(
            [
                "residual-ablation",
                "register",
                "--capture-root",
                str(root),
                "--output",
                str(bad_registration),
            ]
        )
        == 0
    )
    with monkeypatch.context() as isolated:
        isolated.setattr(torch.autograd, "grad", no_gradient)
        with pytest.raises(cd.InvalidProbe):
            ra.measure_capture(
                root / "seed_42" / "step_000060",
                tmp_path / "blocked-smoke",
                bad_registration,
                device=torch.device("cpu"),
                phase="smoke",
                model_factory=factory,
            )
    blocked = json.loads((tmp_path / "blocked-smoke" / "results.json").read_text())
    assert blocked["loading_preflight"]["valid"] == 8
    assert blocked["loading_preflight"]["technical_failure"] == 1
    assert blocked["drawset_counts"]["attempted"] == 0
    assert before["python"] == cd.execution_state()["python"]
    assert torch.equal(before["torch_cpu"], cd.execution_state()["torch_cpu"])


def test_environment_identity_keeps_prefix_and_package_builds():
    import sys

    identity = ra.registration_template("source")["environment_identity"]
    assert identity["sys_prefix"] == sys.prefix
    assert identity["python_executable"] == sys.executable
    assert all(identity[key] for key in ("python", "torch", "numpy", "torchvision"))


def test_overflow_baseline_keeps_rows_and_strict_json():
    d0 = torch.full((6,), 1e20)
    zero = torch.zeros_like(d0)
    solved = {
        name: {"delta": zero, "direction": d0} for name in ("full", "off", "rest")
    }
    result = ra.measure_candidates(
        d0, zero, zero, lambda _: zero, solved, lambda _: 1.0, floor=1e-4
    )
    assert json.dumps(result, allow_nan=False)
    assert result["baselines"]["model_d0"] is None
    assert result["baselines"]["status"] == "technical_failure"
    for arm in result["arms"].values():
        for row in arm.values():
            assert row["status"] == "technical_failure"
            assert row["raw_delta_loss"] == 0.0
            assert row["applied_gain"] is None
            assert row["numerical_failures"]
    assert result["counts"]["full"]["technical_failure"] == 9


def registered_synthetic_roster(tmp_path: Path):
    roster = [
        {
            "seed": seed,
            "step": step,
            "status": "bytes_checked_not_loaded",
            "capture_path": str(
                tmp_path / "inputs" / f"seed_{seed}" / f"step_{step:06d}"
            ),
            "capture_sha256": f"registered-{seed}-{step}",
            "historical_source_sha256": "historical",
        }
        for seed in ra.SEEDS
        for step in ra.STEPS
    ]
    path = tmp_path / "registration.json"
    path.write_text(
        json.dumps(ra.registration_template(cd.source_hashes()["sha256"], roster))
    )
    return path, roster


def test_failure_record_predeclares_both_drawsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registration, roster = registered_synthetic_roster(tmp_path)
    requested = Path(roster[1]["capture_path"])

    def corrupt(_):
        raise cd.InvalidProbe("synthetic corruption before loading")

    monkeypatch.setattr(ra, "_capture_manifest", corrupt)
    output = tmp_path / "new-smoke"
    with pytest.raises(cd.InvalidProbe):
        ra.measure_capture(
            requested, output, registration, device=torch.device("cpu"), phase="smoke"
        )
    result = json.loads((output / "results.json").read_text())
    assert result["requested_capture"] == str(requested)
    assert (result["seed"], result["step"]) == (42, 60)
    assert result["registration"]["sha256"] == cd.file_sha256(registration)
    assert set(result["drawsets"]) == set(ra.DRAWSETS)
    for draw in result["drawsets"].values():
        assert draw["status"] == "not_attempted"
        assert draw["measurements"]["counts"]["full"]["not_attempted"] == 9
    assert result["status"] == "technical_failure"


def test_freeze_cli_and_output_overlap_guard(tmp_path: Path):
    from experiments.pilot import build_parser

    args = build_parser().parse_args(
        [
            "residual-ablation",
            "freeze",
            "--smoke-registration",
            "/new/reg.json",
            "--thresholds",
            "/new/smoke/thresholds.json",
            "--output",
            "/new/final-reg.json",
        ]
    )
    assert args.phase == "freeze"
    with pytest.raises(ValueError):
        ra.guard_output(tmp_path / "capture" / "output", [tmp_path / "capture"])
