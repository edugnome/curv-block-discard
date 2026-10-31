"""Exp10 runner: the exact and matrix-free conditioning records and the bundle shape.

A tiny CPU run (depth 3, width 8, 2 epochs cut to a few steps, 1 seed, 2 power steps
and 2 probes) on in-memory CIFAR-shaped loaders checks that the runner composes
training, the checkpoint schedule, the exact block-matrix assembly and the HVP-only
summaries into the documented JSON bundle; every record is checked against the kappa =
inf rule and the exact records against their own stored spectrum (stable rank, sigma).
The block-matrix assembly is checked against the per-block `ExactBlockHessian` blocks,
the block HVP and the matrix-free summaries against a dense parameter Hessian of a tiny
`SegmentedMLP`, the seed aggregation on a hand-built record, and the CLI overrides on a
parsed namespace. `test_spectrum` covers the numerical kernels themselves.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.config import Exp10Config  # noqa: E402
from experiments.models import PlainMLP, SegmentedMLP  # noqa: E402
from experiments.runner_exp10 import (  # noqa: E402
    Exp10Runner,
    _aggregate,
    _block_hvp,
    _block_ranges,
    assemble_activation_hessian,
    build_config,
    exact_conditioning,
    matrix_free_block_conditioning,
)
from hessian import ExactBlockHessian, ParamBlockEstimator  # noqa: E402
from hessian.spectrum import conditioning_from_spectrum  # noqa: E402

_DEVICE = torch.device("cpu")
_LAMBDAS = [1e-2, 1e-1]
_RECORD_KEYS = {"lambda_max", "lambda_min", "sigma", "frob_sq", "kappa", "stable_rank"}
# What the matrix-free (Lanczos) route adds: its own convergence diagnostics.
_LANCZOS_KEYS = {"n_iter_used", "residual_max", "residual_min", "converged"}


# ======================================================================
# Helpers
# ======================================================================


def _tiny_cifar_loaders(part: str) -> tuple[DataLoader, DataLoader]:
    """In-memory CIFAR-shaped loaders (3x32x32, 10 classes) for both parts."""
    gen = torch.Generator().manual_seed(0 if part == "mlp" else 1)
    x = torch.randn(64, 3, 32, 32, generator=gen)
    y = torch.randint(0, 10, (64,), generator=gen)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    val = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=16, shuffle=False)
    return train, val


def _tiny_cfg() -> Exp10Config:
    base = Exp10Config()
    return replace(
        base,
        depth=3,
        width=8,
        n_lanczos_iter=2,
        n_probes=2,
        hessian_batch_size=8,
        mlp_training=replace(
            base.mlp_training, epochs=2, max_steps=3, seeds=[42], deterministic=False
        ),
        resnet_training=replace(
            base.resnet_training, epochs=2, max_steps=2, seeds=[42], deterministic=False
        ),
    )


def _check_kappa_rule(record: dict[str, Any]) -> None:
    """kappa_lambda = (lambda_max + lambda) / (lambda_min + lambda), inf where the
    denominator is not positive."""
    for key, lam in (("0.01", 1e-2), ("0.1", 1e-1)):
        denom = record["lambda_min"] + lam
        kappa = record["kappa"][key]
        if denom <= 0:
            assert math.isinf(kappa), f"kappa_{key} must be inf at denominator {denom}"
        else:
            expected = (record["lambda_max"] + lam) / denom
            assert math.isclose(kappa, expected, rel_tol=1e-9), (kappa, expected)


def _dense_param_hessian(model: SegmentedMLP, x: Tensor, y: Tensor) -> Tensor:
    """The full parameter Hessian of a tiny model from unit-vector HVPs."""
    curvature = ParamBlockEstimator(model, nn.CrossEntropyLoss()).curvature(x, y)
    n = curvature.flat_grad.numel()
    columns = [curvature.hvp(torch.eye(n)[i]) for i in range(n)]
    return torch.stack(columns, dim=1)


# ======================================================================
# Exact part
# ======================================================================


def test_assemble_activation_hessian_matches_blocks() -> None:
    torch.manual_seed(0)
    model = PlainMLP(input_dim=48, width=8, depth=3).eval()
    x, y = torch.randn(4, 48), torch.randint(0, 10, (4,))
    loss_fn = nn.CrossEntropyLoss()

    matrix, ranges = assemble_activation_hessian(model, x, y, loss_fn)
    assert matrix.dtype == torch.float64 and matrix.shape == (24, 24)
    assert torch.equal(matrix, matrix.T), "symmetrized"
    assert list(ranges) == model.get_layer_names()
    assert [s.stop - s.start for s in ranges.values()] == [8, 8, 8]

    exact = ExactBlockHessian(model, loss_fn)
    diag = exact.compute_block(x, y, "block_1", "block_1").matrix.double()
    assert torch.allclose(matrix[ranges["block_1"], ranges["block_1"]], diag, atol=1e-5)
    cross = exact.compute_block(x, y, "block_0", "block_2").matrix.double()
    assert torch.allclose(
        matrix[ranges["block_0"], ranges["block_2"]], cross, atol=1e-4
    ), "the (0, 2) block is H_{0,2} up to the float32 asymmetry removed by symmetrizing"


def test_exact_conditioning_record() -> None:
    torch.manual_seed(1)
    model = PlainMLP(input_dim=48, width=8, depth=3).eval()
    x, y = torch.randn(4, 48), torch.randint(0, 10, (4,))
    rec = exact_conditioning(model, x, y, nn.CrossEntropyLoss(), _LAMBDAS, neg_tol=1e-8)

    assert set(rec) == {"full", "blocks"}
    assert set(rec["full"]) == _RECORD_KEYS | {"neg_frac", "eigenvalues"}
    assert list(rec["blocks"]) == model.get_layer_names()
    assert all(set(b) == _RECORD_KEYS | {"neg_frac"} for b in rec["blocks"].values())

    eig = torch.tensor(rec["full"]["eigenvalues"], dtype=torch.float64)
    assert eig.numel() == 24 and torch.all(eig[1:] >= eig[:-1]), "sorted spectrum"
    assert math.isclose(rec["full"]["lambda_max"], eig.max().item())
    assert math.isclose(rec["full"]["sigma"], eig.abs().max().item())
    assert math.isclose(
        rec["full"]["stable_rank"], (eig**2).sum().item() / eig.abs().max().item() ** 2
    ), "D = ||M||_F^2 / ||M||_2^2 from the eigenvalues"
    _check_kappa_rule(rec["full"])
    for block in rec["blocks"].values():
        _check_kappa_rule(block)
        assert 0.0 <= block["neg_frac"] <= 1.0


# ======================================================================
# Matrix-free part
# ======================================================================


def test_block_hvp_and_matrix_free_summaries_match_dense() -> None:
    torch.manual_seed(2)
    model = SegmentedMLP(in_dim=4, hidden=4, depth=2, num_classes=3).eval()
    x, y = torch.randn(6, 4), torch.randint(0, 3, (6,))
    h = _dense_param_hessian(model, x, y)
    n = h.shape[0]
    assert torch.allclose(h, h.T, atol=1e-5), "the exact HVP is symmetric"

    ranges = _block_ranges(model.get_param_groups())
    assert list(ranges) == ["fc0", "fc1", "head"] and ranges["head"].stop == n

    curvature = ParamBlockEstimator(model, nn.CrossEntropyLoss()).curvature(x, y)
    gen = torch.Generator().manual_seed(0)
    for sv in ranges.values():
        z = torch.randn(sv.stop - sv.start, generator=gen)
        expected = h[sv, sv] @ z
        assert torch.allclose(
            _block_hvp(curvature.hvp, sv, n)(z), expected, atol=1e-5
        ), "(H P_v z)|_v is the diagonal block's action"

    rec = matrix_free_block_conditioning(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        _LAMBDAS,
        n_lanczos_iter=400,
        n_probes=2000,
        seed=0,
    )
    assert set(rec["full"]) == _RECORD_KEYS | _LANCZOS_KEYS, "no neg_frac, diagnostics"
    assert list(rec["blocks"]) == list(ranges)
    operators = {"full": h, **{v: h[sv, sv] for v, sv in ranges.items()}}
    for name, a in operators.items():
        est = rec["full"] if name == "full" else rec["blocks"][name]
        exact = conditioning_from_spectrum(torch.linalg.eigvalsh(a.double()), _LAMBDAS)
        tol = 2e-2 * exact["sigma"]  # a tiny MLP's spectrum may cluster near the top
        for key in ("lambda_max", "lambda_min", "sigma"):
            assert abs(est[key] - exact[key]) <= tol, f"{name} {key}: {est} vs {exact}"
        assert abs(est["stable_rank"] - exact["stable_rank"]) <= 0.15 * max(
            exact["stable_rank"], 1.0
        ), f"{name} D: {est['stable_rank']} vs {exact['stable_rank']}"
        _check_kappa_rule(est)


# ======================================================================
# Aggregation
# ======================================================================


def test_aggregate_keeps_seeds_and_propagates_inf() -> None:
    def record(lam_max: float, lam_min: float, kappa: float) -> dict[str, Any]:
        return {
            "lambda_max": lam_max,
            "lambda_min": lam_min,
            "sigma": lam_max,
            "frob_sq": 4.0,
            "kappa": {"0.01": kappa, "0.1": 2.0},
            "stable_rank": 4.0 / lam_max**2,
            "neg_frac": 0.25,
            "eigenvalues": [lam_min, lam_max],
        }

    raw = {
        "plain": {
            "init": {
                "seed_42": {
                    "full": record(2.0, -1.0, math.inf),
                    "blocks": {"block_0": record(1.0, 0.5, 3.0)},
                    "test_acc": 0.1,
                },
                "seed_43": {
                    "full": record(4.0, -1.0, 10.0),
                    "blocks": {"block_0": record(3.0, 0.5, 5.0)},
                    "test_acc": 0.3,
                },
            }
        }
    }
    agg = _aggregate(raw)["plain"]["init"]
    assert agg["seeds"] == raw["plain"]["init"], "raw per-seed records are kept"
    assert agg["n_seeds"] == 2
    assert math.isclose(agg["test_acc_mean"], 0.2) and math.isclose(
        agg["test_acc_std"], 0.1
    )
    assert math.isclose(agg["full"]["lambda_max"]["mean"], 3.0)
    assert math.isclose(agg["full"]["lambda_max"]["std"], 1.0)
    assert math.isinf(agg["full"]["kappa"]["0.01"]["mean"]), "inf propagates"
    assert math.isnan(agg["full"]["kappa"]["0.01"]["std"]), "spread undefined"
    assert math.isclose(agg["full"]["kappa"]["0.1"]["mean"], 2.0)
    assert math.isclose(agg["blocks"]["block_0"]["kappa"]["0.01"]["mean"], 4.0)
    assert math.isclose(agg["full"]["neg_frac"]["mean"], 0.25)
    assert "eigenvalues" not in agg["full"], "lists stay in the per-seed records"


# ======================================================================
# Runner smoke + CLI seam
# ======================================================================


def test_exp10_smoke_tiny() -> None:
    """Depth 3, width 8, a few steps, one seed, both parts, on tiny in-memory data."""
    cfg = _tiny_cfg()
    results = Exp10Runner(cfg, _DEVICE, loaders=_tiny_cifar_loaders).run()
    json.dumps(results)  # JSON-serializable (inf / NaN allowed as floats)

    assert set(results) == {"meta", "mlp", "resnet18"}
    assert results["meta"]["lambdas"] == _LAMBDAS and results["meta"]["depth"] == 3

    # --- exact part ---
    assert set(results["mlp"]) == {"plain", "residual"}
    for arch in ("plain", "residual"):
        assert set(results["mlp"][arch]) == {"init", "final"}, arch
        for ckpt, entry in results["mlp"][arch].items():
            assert set(entry) == {
                "full",
                "blocks",
                "test_acc_mean",
                "test_acc_std",
                "n_seeds",
                "seeds",
            }, (arch, ckpt)
            assert entry["n_seeds"] == 1 and list(entry["seeds"]) == ["seed_42"]
            seed = entry["seeds"]["seed_42"]
            assert set(seed) == {"full", "blocks", "test_acc"}
            assert 0.0 <= seed["test_acc"] <= 1.0
            assert set(seed["full"]) == _RECORD_KEYS | {"neg_frac", "eigenvalues"}
            assert list(seed["blocks"]) == ["block_0", "block_1", "block_2"]
            eig = torch.tensor(seed["full"]["eigenvalues"])
            assert eig.numel() == 3 * 8, "depth x width spectrum"
            assert math.isclose(
                seed["full"]["stable_rank"],
                (eig**2).sum().item() / eig.abs().max().item() ** 2,
                rel_tol=1e-6,
            ), "the full-matrix stable rank is ||M||_F^2 / ||M||_2^2"
            _check_kappa_rule(seed["full"])
            for block in seed["blocks"].values():
                _check_kappa_rule(block)
                assert set(block) == _RECORD_KEYS | {"neg_frac"}
            # aggregates over the single seed reproduce it
            assert entry["full"]["lambda_max"]["mean"] == seed["full"]["lambda_max"]
            assert entry["full"]["lambda_max"]["std"] == 0.0
            assert set(entry["full"]["kappa"]) == {"0.01", "0.1"}
            assert "eigenvalues" not in entry["full"]

    # --- matrix-free part ---
    assert set(results["resnet18"]) == {"resnet18"}
    for ckpt in ("init", "final"):
        entry = results["resnet18"]["resnet18"][ckpt]
        assert entry["n_seeds"] == 1
        seed = entry["seeds"]["seed_42"]
        assert set(seed["full"]) == _RECORD_KEYS | _LANCZOS_KEYS, "matrix-free record"
        assert list(seed["blocks"]) == [
            "stem",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "head",
        ]
        _check_kappa_rule(seed["full"])
        for block in seed["blocks"].values():
            assert set(block) == _RECORD_KEYS | _LANCZOS_KEYS
            _check_kappa_rule(block)
            assert block["sigma"] >= 0.0 and block["frob_sq"] >= 0.0
        assert "neg_frac" not in entry["full"]


def test_run_rejects_unknown_part() -> None:
    cfg = replace(_tiny_cfg(), parts=["vit"])
    raised = False
    try:
        Exp10Runner(cfg, _DEVICE, loaders=_tiny_cifar_loaders).run()
    except ValueError:
        raised = True
    assert raised, "an unknown part must raise ValueError"


def test_make_exp10_config_overrides() -> None:
    def ns(**overrides: Any) -> argparse.Namespace:
        fields = {
            "part": None,
            "seeds": None,
            "epochs": None,
            "max_steps": None,
            "depth": None,
            "width": None,
            "lambdas": None,
            "n_lanczos_iter": None,
            "n_probes": None,
            "hessian_batch_size": None,
        }
        fields.update(overrides)
        return argparse.Namespace(**fields)

    default = build_config(ns())
    assert default == Exp10Config(), "no overrides -> the appendix defaults"
    assert default.mlp_training.seeds == [42, 43, 44]
    assert default.resnet_training.seeds == [42]
    assert default.mlp_training.epochs == 50 and default.resnet_training.epochs == 100

    cfg = build_config(
        ns(part=["mlp"], seeds=[1, 2], epochs=3, max_steps=5, depth=4, n_probes=7)
    )
    assert cfg.parts == ["mlp"] and cfg.depth == 4 and cfg.n_probes == 7
    for training in (cfg.mlp_training, cfg.resnet_training):
        assert training.seeds == [1, 2]
        assert training.epochs == 3 and training.max_steps == 5
    assert cfg.width == 64 and cfg.n_lanczos_iter == 60, "untouched defaults"
    assert cfg.mlp_training.lr == 0.01 and cfg.resnet_training.lr == 0.1
