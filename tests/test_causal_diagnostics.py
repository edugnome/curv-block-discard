"""CPU contracts for the new (not historical replay) frozen-state pilot B."""

from __future__ import annotations

import gc
import importlib
import json
import random
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from experiments.config import Exp7Config  # noqa: E402
from experiments.param_ops import group_major_params, trial_forward  # noqa: E402
from experiments.training import StepCtx  # noqa: E402
from secondorder.overlay import OverlaySubspace  # noqa: E402
from secondorder.safeguard import TrustRegionSafeguard, model_value  # noqa: E402


def api():
    return importlib.import_module("experiments.causal_diagnostics")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 3)
        self.bn = nn.BatchNorm1d(3)
        self.act = nn.Tanh()
        self.last = nn.Linear(3, 2)
        self.double()

    def forward(self, x):
        return self.last(self.act(self.bn(self.first(x))))

    def get_param_groups(self):
        return {
            "first": [*self.first.parameters(), *self.bn.parameters()],
            "last": list(self.last.parameters()),
        }


class TinyBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(2, 2)
        self.double()

    def forward(self, x):
        value = self.relu(self.bn(self.conv(x)))
        value = self.relu(value + 0.1)
        return self.head(self.pool(value).mean((2, 3)))

    def get_param_groups(self):
        return {
            "first": [*self.conv.parameters(), *self.bn.parameters()],
            "last": list(self.head.parameters()),
        }


class TinyProvider:
    def __init__(self, model):
        self.model = model

    def update(self, x, y):
        with torch.no_grad():
            self.model(x)
        random.random()
        np.random.rand()
        torch.rand(())

    def base_direction(self, grad):
        return -0.02 * grad


def sample():
    x = torch.tensor(
        [[0.4, -1.0], [1.1, 0.2], [-0.7, 0.5], [0.9, 1.3]], dtype=torch.float64
    )
    return x, torch.tensor([0, 1, 0, 1])


def config():
    base = Exp7Config.b1_cars()
    return replace(
        base,
        model="tiny",
        pretrained=False,
        overlay_rank=1,
        overlay_n_power_iter=1,
        overlay_max_pairs=1,
        coupling_n_probes_online=2,
        tau=-1.0,
    )


def captured(branches=False):
    torch.manual_seed(63)
    model = (TinyBranchModel() if branches else TinyModel()).train()
    saved: list[dict[str, Any]] = []
    hook = api().build_capture_hook(
        model,
        config(),
        seed=42,
        steps=[0],
        sink=saved.append,
        provider=TinyProvider(model),
    )
    x, y = sample()
    if branches:
        x = torch.linspace(-1, 1, 64, dtype=torch.float64).reshape(4, 1, 4, 4)
    hook(StepCtx(model, x, y, 0, torch.device("cpu")))
    return saved[0]


def test_capture_on_off_preserves_trajectory_rng_and_pre_gradient_buffers():
    cd = api()

    def run(enabled):
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        model = TinyModel().train()
        records: list[dict[str, Any]] = []
        hook = cd.build_capture_hook(
            model,
            config(),
            seed=42,
            steps=[0, 1] if enabled else [],
            sink=records.append,
            provider=TinyProvider(model),
        )
        x, y = sample()
        for step in range(2):
            hook(StepCtx(model, x, y, step, torch.device("cpu")))
        return model, records, cd.execution_state()

    plain, _, plain_rng = run(False)
    model, records, rng = run(True)
    for key, tensor in plain.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[key]), key
    assert torch.equal(rng["torch_cpu"], plain_rng["torch_cpu"])
    assert rng["python"] == plain_rng["python"]
    assert rng["numpy"] == plain_rng["numpy"]
    assert records[0]["state_dict"]["bn.num_batches_tracked"].item() == 1
    assert records[1]["state_dict"]["bn.num_batches_tracked"].item() == 3
    assert records[0]["phase"] == "after_provider_refresh_before_gradient_forward"
    assert records[0]["step"] == 0
    assert records[0]["modes"]["bn"] is True
    assert torch.equal(records[0]["x"], sample()[0])


def test_capture_roundtrip_weights_only_replay_and_no_live_buffer_updates(tmp_path):
    cd = api()
    payload = captured()
    destination = tmp_path / "state"
    cd.write_capture(destination, payload, provenance={"test": True})
    loaded, manifest = cd.load_capture(destination)
    assert manifest["sha256"] == cd.file_sha256(destination / "state.pt")
    safe = torch.load(destination / "state.pt", weights_only=True)
    assert safe["step"] == 0
    model = TinyModel()
    cd.restore_model(model, loaded)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    cur, checks = cd.replay_curvature(
        model,
        loaded,
        n_probes=2,
        generator=torch.Generator().manual_seed(1),
        tolerances=cd.Tolerances(grad_rtol=1e-10, hvp_rtol=1e-9, atol=1e-12),
    )
    assert checks["loss"]["passed"] and checks["gradient"]["passed"]
    assert torch.allclose(cur.flat_grad, payload["g0"], atol=1e-12, rtol=1e-10)
    assert torch.isfinite(cur.hvp(payload["d0"])).all()
    for key, tensor in before.items():
        assert torch.equal(tensor, model.state_dict()[key]), key
    with pytest.raises(FileExistsError):
        cd.write_capture(destination, payload, provenance={})
    assert cd.load_capture(destination)[1]["sha256"] == manifest["sha256"]


def test_capture_corruption_is_not_silently_replayed(tmp_path):
    cd = api()
    folder = tmp_path / "state"
    cd.write_capture(folder, captured(), provenance={})
    with (folder / "state.pt").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError):
        cd.load_capture(folder)


def test_execution_context_restores_rng_backend_and_failure():
    cd = api()
    before = cd.execution_state()
    with pytest.raises(RuntimeError):
        with cd.isolated_execution(before):
            random.random()
            np.random.rand()
            torch.rand(3)
            torch.use_deterministic_algorithms(not before["backend"]["deterministic"])
            raise RuntimeError("injected")
    after = cd.execution_state()
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert before["backend"] == after["backend"]


def quadratic_bundle():
    h = torch.tensor([[1.5, 0.7], [0.7, -0.4]], dtype=torch.float64)
    g = torch.tensor([0.3, -0.2], dtype=torch.float64)
    d0 = torch.tensor([-0.4, 0.2], dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(h)
    u = OverlaySubspace(
        pairs=(("a", "b"),),
        columns=((slice(0, 2), torch.eye(2, dtype=h.dtype)),),
        transform=torch.eye(2, dtype=h.dtype),
        dim=2,
        eigenvalues=vals,
        eigenvectors=vecs,
    )
    return h, g, d0, u


def test_reduced_original_model_matches_independent_dense_values_without_basis(
    monkeypatch,
):
    cd = api()
    h, g, d0, u = quadratic_bundle()
    monkeypatch.setattr(
        OverlaySubspace, "basis", property(lambda _: pytest.fail("dense U forbidden"))
    )
    reduced = cd.reduce_original(g, d0, h @ d0, u, damping=0.8)
    for c in [torch.zeros_like(g), torch.tensor([0.1, -0.3], dtype=g.dtype)]:
        d = d0 + u.lift(c)
        values = cd.original_model_values(reduced, c, norm_sq=d.dot(d).item())
        undamped = (g.dot(d) + 0.5 * d.dot(h @ d)).item()
        assert values["undamped"] == pytest.approx(undamped, abs=1e-13)
        assert values["damped"] == pytest.approx(
            undamped + 0.4 * d.dot(d).item(), abs=1e-13
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_exact_noop_model_is_anchored_to_production_without_slack(dtype, empty):
    cd = api()
    h, g, d0, subspace = quadratic_bundle()
    h, g, d0 = h.to(dtype), g.to(dtype), (0.5 * d0).to(dtype)
    if empty:
        subspace = cd._empty_subspace(d0)
    else:
        vals, vecs = torch.linalg.eigh(h)
        eye = torch.eye(2, dtype=dtype)
        subspace = OverlaySubspace(
            (("a", "b"),), ((slice(0, 2), eye),), eye, 2, vals, vecs
        )
    c = d0.new_zeros(subspace.transform.shape[1])
    reduced = cd.reduce_original(g, d0, h @ d0, subspace, damping=1.0)
    row = cd.measure_candidates(
        d0,
        subspace,
        {"original": c},
        reduced,
        {},
        lambda _: 1.0,
        tr_ratio=2.0,
        gain_floor=1e-4,
    )["original"]
    decision = TrustRegionSafeguard().accept(
        g,
        d0,
        d0 + subspace.lift(c),
        lambda d: h @ d,
        hd0=h @ d0,
        tr_ratio=2.0,
        damping=1.0,
        loss_at=lambda _: 1.0,
    )
    assert decision.accepted
    assert row["model"]["damped"] == decision.m0 == decision.m1
    assert row["model"]["damped"] - reduced["m0"] == 0.0
    assert row["original_gates"] == {
        "ratio": True,
        "model": True,
        "loss": True,
        "accepted": True,
        "rejected_by": None,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonzero_near_tie_gates_use_full_original_model_not_reduced_sign(dtype):
    cd = api()
    h, g, d0, _ = quadratic_bundle()
    h, g, d0 = h.to(dtype), g.to(dtype), (0.5 * d0).to(dtype)
    vals, vecs = torch.linalg.eigh(h)
    eye = torch.eye(2, dtype=dtype)
    subspace = OverlaySubspace((("a", "b"),), ((slice(0, 2), eye),), eye, 2, vals, vecs)
    candidates = {
        str(alpha): torch.tensor([alpha, -alpha], dtype=dtype)
        for alpha in (-1e-5, -1e-8, 0.0, 1e-8, 1e-5)
    }
    exact = {
        name: model_value(g, d0 + c, h @ (d0 + c), damping=1.0)
        for name, c in candidates.items()
    }
    reduced = cd.reduce_original(g, d0, h @ d0, subspace, damping=1.0)
    rows = cd.measure_candidates(
        d0,
        subspace,
        candidates,
        reduced,
        {},
        lambda _: 1.0,
        tr_ratio=2.0,
        gain_floor=1e-4,
        original_models=exact,
        original_model_baseline=reduced["m0"],
    )
    for name, c in candidates.items():
        row = rows[name]
        decision = TrustRegionSafeguard().accept(
            g,
            d0,
            d0 + c,
            lambda d: h @ d,
            hd0=h @ d0,
            tr_ratio=2.0,
            damping=1.0,
            loss_at=lambda _: 1.0,
        )
        assert row["model"]["damped"] == decision.m1
        assert row["original_gates"]["accepted"] == decision.accepted
        assert row["model"]["full_minus_reduced"] == pytest.approx(
            row["model"]["damped"]
            - (row["model"]["undamped"] + row["model"]["damping"]),
            abs=0.0,
        )
        assert row["preference"] == "tie"
    if dtype == torch.float32:
        # The anchored reduced delta is negative, but production rounds upward.
        assert not torch.equal(d0 + candidates["-1e-08"], d0)
        assert exact["-1e-08"] > reduced["m0"]
        assert rows["-1e-08"]["original_gates"]["model"] is False
        assert rows["-1e-05"]["original_gates"]["model"] is True


def replay_model_case(monkeypatch, *, dtype=torch.float64, drift=1e-5):
    cd = api()
    h, g, d0, _ = quadratic_bundle()
    h, g, d0 = h.to(dtype), g.to(dtype), (0.5 * d0).to(dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(h)
    eye = torch.eye(2, dtype=dtype)
    subspace = OverlaySubspace(
        (("a", "b"),), ((slice(0, 2), eye),), eye, 2, eigenvalues, eigenvectors
    )
    candidates = {
        "original": torch.tensor([0.03, -0.01], dtype=dtype),
        "noop": torch.zeros(2, dtype=dtype),
        "near_down": torch.tensor([-1e-7, 0.0], dtype=dtype),
        "near_up": torch.tensor([1e-7, 0.0], dtype=dtype),
    }
    reduced = cd.reduce_original(g, d0, h @ d0, subspace, damping=1.0)
    decision = TrustRegionSafeguard().accept(
        g,
        d0,
        d0 + candidates["original"],
        lambda d: h @ d,
        hd0=h @ d0,
        tr_ratio=2.0,
        damping=1.0,
        loss_at=lambda _: 1.0,
    )
    geometry = {
        "d0": d0,
        "subspace": subspace,
        "reduced": reduced,
        "seeds": {"coupling": 1},
        "reference": {"loss": 1.0, "gradient": g, "hvps": {"d0": h @ d0}},
        "production_decision": {"m0": decision.m0, "m1": decision.m1},
    }
    replay_g, replay_h = g + 0.1 * drift, h + drift * eye
    cur = cd.ParamCurvature(1.0, replay_g, lambda d: replay_h @ d, lambda: {})
    monkeypatch.setattr(cd, "replay_curvature", lambda *args, **kwargs: (cur, {}))
    return nn.Linear(2, 2), geometry, candidates, cur


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_H0_replay_drift_uses_same_pass_baseline_and_strict_near_ties(
    monkeypatch, dtype
):
    cd = api()
    model, geometry, candidates, cur = replay_model_case(monkeypatch, dtype=dtype)
    models, baseline, checks = cd._original_candidate_models(
        model, {}, geometry, candidates, cd.Tolerances()
    )
    initial = geometry["reduced"]["m0"]
    assert baseline != initial
    assert checks["passed"]
    assert checks["m0"]["initial"] == initial
    assert checks["m0"]["replayed"] == baseline
    assert checks["m0"]["difference"] == baseline - initial
    assert checks["m1"]["difference"] != 0.0
    assert checks["gradient"]["rtol"] == cd.Tolerances().grad_rtol
    assert checks["hvp_d0"]["rtol"] == cd.Tolerances().hvp_rtol
    assert geometry["reduced"]["m0"] == initial  # The initial pass is not rewritten.
    rows = cd.measure_candidates(
        geometry["d0"],
        geometry["subspace"],
        candidates,
        geometry["reduced"],
        {},
        lambda _: 1.0,
        tr_ratio=2.0,
        gain_floor=1e-4,
        original_models=models,
        original_model_baseline=baseline,
    )
    assert rows["noop"]["model_evaluation"]["model_delta"] == 0.0
    assert rows["noop"]["original_gates"]["model"]
    for name, c in candidates.items():
        direction = geometry["d0"] + c
        expected = model_value(
            cur.flat_grad, direction, cur.hvp(direction), damping=1.0
        )
        assert models[name] == expected
        assert rows[name]["original_gates"]["model"] == (expected <= baseline)
        assert rows[name]["model_evaluation"]["baseline"] == baseline
        assert rows[name]["model_evaluation"]["source"] == "replayed_model_value_H0"
    assert 0.0 < models["near_down"] - baseline < cd.Tolerances().atol
    assert not rows["near_down"]["original_gates"]["model"]
    assert rows["near_up"]["original_gates"]["model"]


@pytest.mark.parametrize("quantity", ["operator", "m0", "m1"])
def test_large_H0_replay_drift_is_rejected_with_numerical_evidence(
    monkeypatch, quantity
):
    cd = api()
    model, geometry, candidates, _ = replay_model_case(
        monkeypatch, drift=0.1 if quantity == "operator" else 1e-5
    )
    if quantity == "m0":
        geometry["reduced"]["m0"] += 1.0
    elif quantity == "m1":
        geometry["production_decision"]["m1"] += 1.0
    with pytest.raises(cd.InvalidProbe):
        cd._original_candidate_models(model, {}, geometry, candidates, cd.Tolerances())


def test_full_models_require_an_explicit_matching_baseline():
    cd = api()
    h, g, d0, subspace = quadratic_bundle()
    reduced = cd.reduce_original(g, d0, h @ d0, subspace, damping=1.0)
    with pytest.raises(ValueError):
        cd.measure_candidates(
            d0,
            subspace,
            {"original": torch.zeros_like(d0)},
            reduced,
            {},
            lambda _: 1.0,
            tr_ratio=2.0,
            gain_floor=1e-4,
            original_models={"original": reduced["m0"]},
        )


def test_refresh_solves_keep_lambda_d0_and_fixed_subspace():
    cd = api()
    h, g, d0, u = quadratic_bundle()
    damping = 0.8
    reduced = cd.reduce_original(g, d0, h @ d0, u, damping=damping)
    endpoint_g = torch.tensor([-0.2, 0.6], dtype=g.dtype)
    endpoint_h = torch.tensor([[0.3, -0.2], [-0.2, 0.9]], dtype=g.dtype)
    coordinates, systems = cd.refresh_coordinates(
        reduced, u.project(endpoint_g), endpoint_h
    )
    for name, matrix, residual in [
        ("original", h, g + h @ d0 + damping * d0),
        ("gradient_refresh", h, endpoint_g + damping * d0),
        ("curvature_refresh", endpoint_h, g + h @ d0 + damping * d0),
        ("both_refresh", endpoint_h, endpoint_g + damping * d0),
    ]:
        vals, vecs = torch.linalg.eigh(matrix)
        expected = -vecs @ ((vecs.T @ residual) / (vals.abs() + damping))
        assert torch.allclose(coordinates[name], expected, atol=1e-13, rtol=1e-12)
        assert torch.allclose(systems[name]["residual"], residual)
    assert not torch.allclose(
        coordinates["gradient_refresh"],
        -torch.linalg.solve(h.abs() + damping * torch.eye(2), endpoint_g),
    )


def test_all_candidate_gates_and_actual_loss_are_measured_after_norm_rejection():
    cd = api()
    h, g, d0, u = quadratic_bundle()
    reduced = cd.reduce_original(g, d0, h @ d0, u, damping=0.8)
    c = torch.tensor([2.0, -3.0], dtype=g.dtype)
    calls = []

    def loss_at(d):
        calls.append(d.clone())
        return -d.norm().item()

    rows = cd.measure_candidates(
        d0, u, {"large": c}, reduced, {}, loss_at, tr_ratio=1.0, gain_floor=1e-4
    )
    row = rows["large"]
    assert len(calls) == 2
    assert row["original_gates"]["ratio"] is False
    assert isinstance(row["original_gates"]["model"], bool)
    assert row["original_gates"]["loss"] is True
    assert row["original_gates"]["accepted"] is False
    assert row["loss"] == pytest.approx(-float((d0 + c).norm()))
    assert row["correction_norm"] == pytest.approx(float(c.norm()))


def test_scale_and_norm_matched_controls_never_enlarge_corrections():
    cd = api()
    h, g, d0, u = quadratic_bundle()
    base = torch.tensor([1.0, 0.0], dtype=g.dtype)
    refreshed = torch.tensor([0.0, 0.1], dtype=g.dtype)
    coordinates = {"original": base, "gradient_refresh": refreshed}
    candidates, pairs = cd.add_controls(coordinates, u)
    assert len([name for name in candidates if name.startswith("scale_")]) == 7
    assert set(cd.SCALE_GRID) == {0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1}
    pair = pairs["gradient_refresh"]
    left, right = (
        u.lift(candidates[pair[key]]).norm().item() for key in ("original", "refreshed")
    )
    assert left == pytest.approx(0.1)
    assert right == pytest.approx(0.1)
    assert torch.equal(candidates["original"], base)


def test_comparison_decomposition_on_smooth_cubic():
    cd = api()
    h, g, d0, u = quadratic_bundle()
    c = torch.tensor([0.2, -0.1], dtype=g.dtype)
    delta = u.lift(c)
    beta = 0.7

    def f(d):
        return g.dot(d) + 0.5 * d.dot(h @ d) + beta * d[0] ** 3

    endpoint_g = g + h @ d0 + torch.tensor([3 * beta * d0[0] ** 2, 0.0], dtype=g.dtype)
    reduced = cd.reduce_original(g, d0, h @ d0, u, damping=0.8)
    record = cd.comparison_decomposition(
        reduced,
        c,
        d0,
        delta,
        loss_d0=f(d0).item(),
        loss_d1=f(d0 + delta).item(),
        endpoint_slope=endpoint_g.dot(delta).item(),
        gain_floor=1e-4,
    )
    actual = (f(d0 + delta) - f(d0)).item()
    assert record["actual_gain"] == pytest.approx(-actual)
    assert record["undamped_gain"] + record["damping_gain"] == pytest.approx(
        record["damped_gain"]
    )
    assert record["transport_error"] + record["finite_step_remainder"] == pytest.approx(
        record["actual_minus_undamped_model"]
    )
    assert record["endpoint_slope"] == pytest.approx(endpoint_g.dot(delta).item())


def test_displacement_curvature_differentiates_flat_offset_not_live_parameters():
    cd = api()
    torch.manual_seed(23)
    model = TinyModel().train()
    x, y = sample()
    params = group_major_params(model)
    forward = trial_forward(model, params, 1.0)
    d = torch.zeros(sum(p.numel() for p in params), dtype=x.dtype)
    loss, grad, hvp = cd.displacement_curvature(forward, nn.CrossEntropyLoss(), x, y, d)
    assert loss > 0 and grad.norm() > 0
    v = torch.linspace(-0.1, 0.1, d.numel(), dtype=d.dtype)
    eps = 1e-5
    _, plus, _ = cd.displacement_curvature(
        forward, nn.CrossEntropyLoss(), x, y, d + eps * v
    )
    _, minus, _ = cd.displacement_curvature(
        forward, nn.CrossEntropyLoss(), x, y, d - eps * v
    )
    assert torch.allclose(hvp(v), (plus - minus) / (2 * eps), atol=1e-8, rtol=1e-6)
    assert all(p.grad is None for p in params)
    assert model.bn.num_batches_tracked is not None
    assert model.bn.num_batches_tracked.item() == 0


def test_artifact_errors_budget_and_no_overwrite(tmp_path):
    cd = api()
    folder = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        with cd.ArtifactDirectory(folder, kind="probe") as run:
            with run.stage("first"):
                raise RuntimeError("injected")
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "technical_failure"
    assert status["error"]["type"] == "RuntimeError"
    assert status["stage"] == "first"
    with pytest.raises(FileExistsError):
        with cd.ArtifactDirectory(folder, kind="probe"):
            pass
    budget = tmp_path / "budget"
    with pytest.raises(cd.BudgetExceeded):
        with cd.ArtifactDirectory(budget, kind="probe", max_minutes=1e-20) as run:
            with run.stage("never_started"):
                pytest.fail("budget boundary ignored")
    assert (
        json.loads((budget / "status.json").read_text())["status"] == "budget_exhausted"
    )


def test_summary_includes_failed_missing_and_unprobed_states(tmp_path):
    cd = api()
    root = tmp_path / "input"
    cd.write_capture(
        root / "captures" / "seed_42" / "step_000000", captured(), provenance={}
    )
    with pytest.raises(ValueError):
        with cd.ArtifactDirectory(root / "probes" / "bad", kind="probe"):
            raise ValueError("invalid jet")
    result = cd.summarize(root)
    assert result["counts"]["captured"] == 1
    assert result["counts"]["complete_probes"] == 0
    assert result["counts"]["technical_failure"] == 1
    assert result["captures"][0]["probe_status"] == "not_run"
    assert result["scientific_conclusion"] == "undetermined"


@pytest.mark.parametrize("branches", [False, True])
def test_probe_full_bundle_safe_serialization_fixed_candidates_and_jets(
    tmp_path, monkeypatch, branches
):
    cd = api()
    folder = tmp_path / "capture"
    cd.write_capture(folder, captured(branches=branches), provenance={})
    output = tmp_path / "probe"
    monkeypatch.setattr(
        OverlaySubspace, "basis", property(lambda _: pytest.fail("dense U forbidden"))
    )
    result = cd.probe_capture(
        folder,
        output,
        device=torch.device("cpu"),
        max_minutes=2,
        model_factory=lambda _: TinyBranchModel() if branches else TinyModel(),
        tolerances=cd.Tolerances(grad_rtol=1e-10, hvp_rtol=1e-8, atol=1e-11),
    )
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    assert result["replay"]["gradient"]["passed"]
    assert result["sham_finite_losses"]["passed"]
    assert set(result["interventions"]) == {"sham", "relu", "pool", "both"}
    assert all(value["jet"]["passed"] for value in result["interventions"].values())
    assert len(result["native"]["rays"]["d0"]) == len(cd.RAY_GRID)
    assert len(result["native"]["rays"]["d1"]) == len(cd.RAY_GRID)
    assert len(result["native"]["scale_control"]) == 7
    decision = torch.load(output / "decision_bundle.pt", weights_only=True)
    assert "raw_d1" in decision and "subspace" in decision
    assert "original" in decision["coordinates"]
    assert decision["subspace"]["dim"] == decision["d0"].numel()
    assert set(decision["systems"]) == {
        "original",
        "gradient_refresh",
        "curvature_refresh",
        "both_refresh",
    }
    assert result["capture_sha256"] == cd.file_sha256(folder / "state.pt")
    assert result["tests"]["architecture_intervention"] == "complete"
    with pytest.raises(FileExistsError):
        cd.probe_capture(
            folder,
            output,
            device=torch.device("cpu"),
            max_minutes=2,
            model_factory=lambda _: TinyModel(),
        )


def test_probe_saves_initial_and_replayed_models_without_mixing_baselines(
    tmp_path, monkeypatch
):
    cd = api()
    payload = captured()
    payload["config"]["tau"] = 1e20  # Exact no-op makes an old-baseline gate wrong.
    folder, output = tmp_path / "capture", tmp_path / "probe"
    cd.write_capture(folder, payload, provenance={})
    original = cd.replay_curvature
    calls = 0

    def drifting_replay(*args, **kwargs):
        nonlocal calls
        cur, checks = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            cur = replace(cur, flat_grad=(1.0 - 1e-5) * cur.flat_grad)
        return cur, checks

    monkeypatch.setattr(cd, "replay_curvature", drifting_replay)
    result = cd.probe_capture(
        folder, output, device=torch.device("cpu"), model_factory=lambda _: TinyModel()
    )
    replay = result["original_model_replay"]
    assert replay["passed"] and replay["m0"]["difference"] > 0.0
    assert replay["m0"]["initial"] == result["production_decision"]["m0"]
    assert replay["m1"]["initial"] == result["production_decision"]["m1"]
    row = result["native"]["candidates"]["original"]
    assert row["model"]["damped"] == replay["m0"]["replayed"]
    assert row["model"]["damped"] > result["production_decision"]["m0"]
    assert row["model_evaluation"]["model_delta"] == 0.0
    assert row["original_gates"]["accepted"]
    assert result["native"]["comparison"]["gate_damped_gain"] == 0.0
    assert result["production_acceptance_check"]["passed"]
    bundle = torch.load(output / "decision_bundle.pt", weights_only=True)
    assert bundle["original_model_baseline"] == replay["m0"]["replayed"]
    assert bundle["original_model_replay"] == replay
    assert bundle["reduced"]["m0"] == result["production_decision"]["m0"]
    assert (
        json.loads((output / "results.json").read_text())["original_model_replay"]
        == replay
    )


def test_probe_acceptance_flip_is_saved_and_rejected(tmp_path, monkeypatch):
    cd = api()
    folder, output = tmp_path / "capture", tmp_path / "probe"
    cd.write_capture(folder, captured(), provenance={})
    original = cd._measure

    def flipped_measurement(*args, **kwargs):
        measured = original(*args, **kwargs)
        gates = measured["candidates"]["original"]["original_gates"]
        gates["accepted"] = not gates["accepted"]
        return measured

    monkeypatch.setattr(cd, "_measure", flipped_measurement)
    with pytest.raises(cd.InvalidProbe):
        cd.probe_capture(
            folder,
            output,
            device=torch.device("cpu"),
            model_factory=lambda _: TinyModel(),
        )
    partial = json.loads((output / "results.json").read_text())
    check = partial["production_acceptance_check"]
    assert check["passed"] is False
    assert check["initial_accepted"] == partial["production_decision"]["accepted"]
    assert check["replayed_accepted"] != check["initial_accepted"]
    assert "native" in partial and "original_model_replay" in partial
    assert (
        json.loads((output / "status.json").read_text())["status"]
        == "technical_failure"
    )


def test_probe_failure_status_and_partial_identity(tmp_path):
    cd = api()
    folder = tmp_path / "capture"
    cd.write_capture(folder, captured(), provenance={})
    output = tmp_path / "bad_probe"

    def bad_factory(_):
        raise RuntimeError("factory failed")

    with pytest.raises(RuntimeError):
        cd.probe_capture(
            folder, output, device=torch.device("cpu"), model_factory=bad_factory
        )
    assert (
        json.loads((output / "status.json").read_text())["status"]
        == "technical_failure"
    )
    partial = json.loads((output / "results.json").read_text())
    assert partial["capture_sha256"] == cd.file_sha256(folder / "state.pt")
    assert partial["tests"]["architecture_intervention"] == "not_run"


def test_probe_rejects_changed_source_before_interpreting_results(tmp_path):
    cd = api()
    folder = tmp_path / "capture"
    cd.write_capture(folder, captured(), provenance={"source": {"sha256": "wrong"}})
    with pytest.raises(cd.InvalidProbe):
        cd.probe_capture(
            folder,
            tmp_path / "probe",
            device=torch.device("cpu"),
            model_factory=lambda _: TinyModel(),
        )


def test_scalar_agreement_does_not_round_float64_losses_to_float32():
    check = api().agreement(1.0 + 1e-9, 1.0, rtol=0.0, atol=1e-12)
    assert check["passed"] is False
    assert check["error_norm"] == pytest.approx(1e-9, abs=1e-15)


def test_summary_preserves_selected_but_uncaptured_states(tmp_path):
    cd = api()
    root = tmp_path / "pilot"
    with cd.ArtifactDirectory(root / "captures", kind="capture"):
        cd.write_json(
            root / "captures" / "provenance.json",
            {"seeds": [42, 43], "steps": [20, 60]},
        )
        payload = captured()
        payload["step"] = 20
        cd.write_capture(
            root / "captures" / "seed_42" / "step_000020", payload, provenance={}
        )
    summary = cd.summarize(root)
    assert len(summary["expected_states"]) == 4
    assert (
        sum(
            state["capture_status"] == "not_run" for state in summary["expected_states"]
        )
        == 3
    )
    assert summary["counts"]["captured"] == 1


@pytest.mark.parametrize("seed", [0, 1, 5, 6])
@pytest.mark.parametrize("empty", [False, True])
def test_float32_probe_exact_H0_gates_and_sequential_graphs(
    tmp_path, monkeypatch, seed, empty
):
    cd = api()
    torch.manual_seed(seed)
    model = TinyModel().float().train()
    protocol = replace(config(), tau=1e20 if empty else -1.0)
    records: list[dict[str, Any]] = []
    hook = cd.build_capture_hook(
        model,
        protocol,
        seed=seed,
        steps=[0],
        sink=records.append,
        provider=TinyProvider(model),
    )
    x, y = sample()
    hook(StepCtx(model, x.float(), y, 0, torch.device("cpu")))
    payload = records[0]
    folder, output = tmp_path / "capture", tmp_path / "probe"
    cd.write_capture(folder, payload, provenance={})
    closures: list[weakref.ReferenceType[Any]] = []
    original_curvature = cd.ParamBlockEstimator.curvature
    original_displacement = cd.displacement_curvature

    def check_graphs_released():
        gc.collect()
        assert all(ref() is None for ref in closures), "curvature graphs overlap"

    def tracked_curvature(self, *args, **kwargs):
        check_graphs_released()
        cur = original_curvature(self, *args, **kwargs)
        closures.extend([weakref.ref(cur.hvp), weakref.ref(cur.coupling)])
        return cur

    def tracked_displacement(*args, **kwargs):
        check_graphs_released()
        result = original_displacement(*args, **kwargs)
        closures.append(weakref.ref(result[2]))
        return result

    monkeypatch.setattr(cd.ParamBlockEstimator, "curvature", tracked_curvature)
    monkeypatch.setattr(cd, "displacement_curvature", tracked_displacement)
    monkeypatch.setattr(
        OverlaySubspace, "basis", property(lambda _: pytest.fail("dense U forbidden"))
    )
    result = cd.probe_capture(
        folder,
        output,
        device=torch.device("cpu"),
        model_factory=lambda _: TinyModel(),
    )
    check_graphs_released()
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    assert bool(result["selected_pairs"]) is not empty
    rows = result["native"]["candidates"]
    assert (
        rows["original"]["original_gates"]["accepted"]
        == result["production_decision"]["accepted"]
    )
    assert rows["scale_0"]["original_gates"]["accepted"]
    assert rows["scale_0"]["model"]["damped"] == result["production_decision"]["m0"]
    bundle = torch.load(output / "decision_bundle.pt", weights_only=True)
    saved = bundle["subspace"]
    subspace = OverlaySubspace(
        tuple(tuple(pair) for pair in saved["pairs"]),
        tuple(
            (slice(row["start"], row["stop"]), row["values"])
            for row in saved["columns"]
        ),
        saved["transform"],
        saved["dim"],
        saved["eigenvalues"],
        saved["eigenvectors"],
    )
    d0 = bundle["d0"]
    assert torch.equal(
        d0 + subspace.lift(bundle["candidates"]["original"]), bundle["raw_d1"]
    )
    diagnostic = TinyModel().float()
    cd.restore_model(diagnostic, payload)
    cur, _ = cd.replay_curvature(
        diagnostic, payload, n_probes=1, generator=torch.Generator().manual_seed(9)
    )
    m0 = model_value(cur.flat_grad, d0, cur.hvp(d0), damping=protocol.damping)
    if not empty:
        assert (bundle["endpoint"]["h"] - bundle["reduced"]["h"]).norm() > 1e-8
    for name, c in bundle["candidates"].items():
        direction = d0 + subspace.lift(c)
        expected = model_value(
            cur.flat_grad, direction, cur.hvp(direction), damping=protocol.damping
        )
        assert rows[name]["model"]["damped"] == expected
        assert rows[name]["original_gates"]["model"] == (expected <= m0)
        assert rows[name]["original_gates"]["ratio"] == (
            float(direction.norm()) <= protocol.tr_ratio * float(d0.norm())
        )
        for mode in result["interventions"].values():
            assert (
                mode["measurements"]["candidates"][name]["model"] == rows[name]["model"]
            )


def test_probe_no_pairs_is_a_near_zero_control_not_a_substantive_success(tmp_path):
    cd = api()
    payload = captured()
    payload["config"]["tau"] = 1e20
    folder = tmp_path / "capture"
    cd.write_capture(folder, payload, provenance={})
    result = cd.probe_capture(
        folder,
        tmp_path / "probe",
        device=torch.device("cpu"),
        model_factory=lambda _: TinyModel(),
    )
    assert result["selected_pairs"] == []
    for name in ("original", "gradient_refresh", "curvature_refresh", "both_refresh"):
        row = result["native"]["candidates"][name]
        assert row["near_zero_correction"]
        assert row["preference"] == "tie"
    assert not any(
        value["substantive"]
        for value in result["native"]["primary_refresh_comparison"].values()
    )


def test_raw_proposal_survives_production_fallback_and_all_gates_are_saved(tmp_path):
    cd = api()
    payload = captured()
    payload["config"]["tr_ratio"] = 1e-4
    folder = tmp_path / "capture"
    cd.write_capture(folder, payload, provenance={})
    output = tmp_path / "probe"
    result = cd.probe_capture(
        folder, output, device=torch.device("cpu"), model_factory=lambda _: TinyModel()
    )
    assert result["production_decision"]["applied_candidate"] == "d0"
    assert result["production_decision"]["rejected_by"] == "ratio"
    row = result["native"]["candidates"]["original"]
    assert isinstance(row["loss"], float)
    assert isinstance(row["original_gates"]["loss"], bool)
    decision = torch.load(output / "decision_bundle.pt", weights_only=True)
    assert not torch.equal(decision["raw_d1"], decision["d0"])


def test_registered_thresholds_are_not_recalibrated_after_effects():
    cd = api()
    h, g, d0, u = quadratic_bundle()
    geometry = {"d0": d0, "raw_d1": d0 + g}
    values = iter([0.0, 0.01, 0.0] * 3)
    # The fixed threshold must reject worse noise, not silently raise the floor.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "thresholds.json"
        cd.write_json(
            path, {"tolerances": cd.asdict(cd.Tolerances()), "gain_floor": 1e-4}
        )
        with pytest.raises(cd.InvalidProbe):
            cd._calibrate(
                lambda _: next(values),
                geometry,
                cd.Tolerances(),
                capture_sha256="test",
                supplied=path,
            )


def test_real_kfac_provider_capture_replays_without_recreating_optimizer():
    cd = api()
    torch.manual_seed(13)
    model = TinyModel().train()
    records: list[dict[str, Any]] = []
    hook = cd.build_capture_hook(
        model, config(), seed=42, steps=[0], sink=records.append
    )
    x, y = sample()
    hook(StepCtx(model, x, y, 0, torch.device("cpu")))
    diagnostic = TinyModel()
    cd.restore_model(diagnostic, records[0])
    cur, checks = cd.replay_curvature(
        diagnostic, records[0], n_probes=1, generator=torch.Generator().manual_seed(31)
    )
    assert checks["gradient"]["passed"]
    assert cur.loss == pytest.approx(records[0]["loss0"])
    assert torch.isfinite(records[0]["d0"]).all()


def test_cli_has_required_capture_probe_and_summary_controls():
    from experiments import pilot

    module = {"build_parser": pilot.build_parser}
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "capture",
            "--output",
            "/tmp/captures",
            "--seeds",
            "42",
            "43",
            "--steps",
            "20",
            "60",
            "200",
            "600",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--cars-root",
            "/tmp/cars",
            "--torch-home",
            "/tmp/weights",
        ]
    )
    assert args.steps == [20, 60, 200, 600] and args.num_workers == 0
    probe = parser.parse_args(
        [
            "probe",
            "--capture",
            "/tmp/state",
            "--output",
            "/tmp/probe",
            "--device",
            "cpu",
            "--max-minutes",
            "30",
        ]
    )
    assert probe.max_minutes == 30
    summary = parser.parse_args(
        ["summarize", "--input-root", "/tmp/pilot", "--output", "/tmp/summary.json"]
    )
    assert summary.command == "summarize"


def test_missing_cached_inputs_preserve_registration_and_failure_in_summary(tmp_path):
    cd = api()
    root = tmp_path / "pilot"
    with pytest.raises(FileNotFoundError):
        cd.capture_pilot(
            root / "captures",
            seeds=[42],
            steps=[20, 60, 200, 600],
            device=torch.device("cpu"),
            num_workers=0,
            cars_root=tmp_path / "missing_data",
            torch_home=tmp_path / "missing_weights",
        )
    summary = cd.summarize(root)
    assert summary["counts"]["technical_failure"] == 1
    assert len(summary["expected_states"]) == 4
    assert all(
        state["capture_status"] == "not_run" for state in summary["expected_states"]
    )
    assert summary["capture_runs"][0]["status"] == "technical_failure"


def test_primary_success_requires_absolute_gain_above_frozen_floor_too():
    cd = api()
    h, g, d0, subspace = quadratic_bundle()
    c = torch.tensor([0.2, -0.1], dtype=g.dtype)
    geometry = {
        "d0": d0,
        "delta": c,
        "raw_d1": d0 + c,
        "subspace": subspace,
        "original_c": c,
        "reduced": cd.reduce_original(g, d0, h @ d0, subspace, damping=0.8),
    }
    names = ["original", "gradient_refresh", "curvature_refresh", "both_refresh"]
    candidates = {name: c for name in names}
    candidates.update(
        {f"scale_{i}": alpha * c for i, alpha in enumerate(cd.SCALE_GRID)}
    )
    geometry["original_model_baseline"] = geometry["reduced"]["m0"]
    geometry["original_models"] = {
        name: model_value(g, d0 + vector, h @ (d0 + vector), damping=0.8)
        for name, vector in candidates.items()
    }
    floor = 1e-4
    losses = iter([1.0, 1.1, 1.0 - 0.95 * floor, 1.1, 1.1] + [1.0 + 0.1 * floor] * 7)
    measured = cd._measure(
        lambda _: next(losses),
        geometry,
        candidates,
        {},
        {},
        protocol=config(),
        tolerances=cd.Tolerances(),
        gain_floor=floor,
        endpoint_slope=0.0,
        rays=False,
    )
    primary = measured["primary_refresh_comparison"]["gradient_refresh"]
    assert primary["gain_over_best_scale_control"] > floor
    assert measured["candidates"]["gradient_refresh"]["actual_gain"] < floor
    assert primary["substantive"] is False


def test_factored_subspace_offload_preserves_coordinates_without_dense_basis(
    monkeypatch,
):
    cd = api()
    _, _, _, subspace = quadratic_bundle()
    monkeypatch.setattr(
        OverlaySubspace, "basis", property(lambda _: pytest.fail("dense U forbidden"))
    )
    offloaded = cd._subspace_to(subspace, torch.device("cpu"))
    vector = torch.tensor([0.4, -0.7], dtype=torch.float64)
    assert torch.equal(offloaded.project(vector), subspace.project(vector))
    assert torch.equal(offloaded.lift(vector), subspace.lift(vector))
    assert offloaded.transform.device.type == "cpu"
    assert all(values.device.type == "cpu" for _, values in offloaded.columns)


def test_probe_reads_rng_only_for_requested_working_device(tmp_path, monkeypatch):
    cd = api()
    folder = tmp_path / "capture"
    cd.write_capture(folder, captured(), provenance={})
    observed = []
    original = cd.execution_state

    def tracked(device=None):
        observed.append(device)
        return original(device)

    monkeypatch.setattr(cd, "execution_state", tracked)
    cd.probe_capture(
        folder,
        tmp_path / "probe",
        device=torch.device("cpu"),
        model_factory=lambda _: TinyModel(),
    )
    assert observed and all(device == torch.device("cpu") for device in observed)


def test_reference_only_replay_labels_changed_source_and_preserves_probe_guard(
    tmp_path, monkeypatch
):
    cd = api()
    capture = tmp_path / "capture"
    cd.write_capture(capture, captured(), provenance={"source": {"sha256": "old"}})
    monkeypatch.setattr(cd, "source_hashes", lambda: {"sha256": "current"})
    before = cd.execution_state()
    result = cd.reference_capture(
        capture,
        tmp_path / "reference",
        device=torch.device("cpu"),
        model_factory=lambda _: TinyModel(),
    )
    assert result["scope"] == "retained_state_loss_gradient_with_stored_d0"
    assert result["source_fingerprint_match"] is False
    assert result["historical_probe_replay"] is False
    assert result["d0_recomputed"] is False
    assert all(check["passed"] for check in result["checks"].values())
    assert torch.equal(before["torch_cpu"], cd.execution_state()["torch_cpu"])
    assert (
        json.loads((tmp_path / "reference/status.json").read_text())["status"]
        == "complete"
    )
    with pytest.raises(cd.InvalidProbe):
        cd.probe_capture(capture, tmp_path / "full-probe", device=torch.device("cpu"))
    with pytest.raises(FileExistsError):
        cd.reference_capture(
            capture,
            tmp_path / "reference",
            device=torch.device("cpu"),
            model_factory=lambda _: TinyModel(),
        )


def test_reference_only_replay_rejects_changed_reference(tmp_path):
    cd = api()
    payload = captured()
    payload["g0"] += 1.0
    capture = tmp_path / "capture"
    cd.write_capture(capture, payload, provenance={})
    with pytest.raises(cd.InvalidProbe):
        cd.reference_capture(
            capture,
            tmp_path / "reference",
            device=torch.device("cpu"),
            model_factory=lambda _: TinyModel(),
        )
    assert (
        json.loads((tmp_path / "reference/status.json").read_text())["status"]
        == "technical_failure"
    )
