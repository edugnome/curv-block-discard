"""Conditioning aggregation over synthetic spectra."""

from __future__ import annotations

from typing import Any

import torch

from experiments.runner_exp10 import _aggregate  # noqa: E402
from hessian.spectrum import (  # noqa: E402
    conditioning_from_extremes,
    conditioning_from_spectrum,
)

_LAMBDAS = [1e-2, 1e-1]
_META = {
    "lambdas": _LAMBDAS,
    "n_lanczos_iter": 60,
    "n_probes": 30,
    "neg_tol": 1e-8,
    "hessian_batch_size": 32,
    "depth": 8,
    "width": 64,
}


def _exact(eigs: list[float]) -> dict[str, Any]:
    return conditioning_from_spectrum(torch.tensor(eigs), _LAMBDAS)


def _mlp_seed(scale: float) -> dict[str, Any]:
    """full: spectrum {-0.5, 0, 1, 4} * scale (kappa inf); blocks: block_0 kappa_1e-2 =
    2.01 / 0.11, block_1 kappa = 1 (the smallest), block_2 kappa = inf (the largest)."""
    return {
        "full": {
            **_exact([-0.5 * scale, 0.0, 1.0 * scale, 4.0 * scale]),
            "eigenvalues": [-0.5 * scale, 0.0, 1.0 * scale, 4.0 * scale],
        },
        "blocks": {
            "block_0": _exact([0.1, 2.0]),
            "block_1": _exact([1.0, 1.0]),
            "block_2": _exact([-1.0, 3.0]),
        },
        "test_acc": 0.5,
    }


def _resnet_seed() -> dict[str, Any]:
    return {
        "full": conditioning_from_extremes(1500.0, -2.0, 1500.0, 9.0e6, _LAMBDAS),
        "blocks": {
            "stem": conditioning_from_extremes(3.0, 0.5, 3.0, 36.0, _LAMBDAS),
            "layer1": conditioning_from_extremes(20.0, -0.001, 20.0, 800.0, _LAMBDAS),
            "head": conditioning_from_extremes(0.002, 0.0, 0.002, 1.6e-5, _LAMBDAS),
        },
        "test_acc": 0.9,
    }


def _bundle(*, mlp: bool = True, resnet: bool = True) -> dict[str, Any]:
    bundle: dict[str, Any] = {"meta": dict(_META)}
    if mlp:
        raw_mlp = {
            arch: {
                ckpt: {"seed_42": _mlp_seed(1.0), "seed_43": _mlp_seed(3.0)}
                for ckpt in ("init", "final")
            }
            for arch in ("plain", "residual")
        }
        bundle["mlp"] = _aggregate(raw_mlp)
    if resnet:
        bundle["resnet18"] = _aggregate(
            {
                "resnet18": {
                    ckpt: {"seed_42": _resnet_seed()} for ckpt in ("init", "final")
                }
            }
        )
    return bundle


# ---------------------------------------------------------------------------
# tab:conditioning-resnet
# ---------------------------------------------------------------------------


def test_conditioning_aggregation_uses_per_seed_spectra() -> None:
    bundle = _bundle()
    entry = bundle["mlp"]["plain"]["init"]
    assert entry["n_seeds"] == 2
    assert entry["full"]["lambda_max"]["mean"] == 8.0
    assert entry["full"]["lambda_max"]["std"] == 4.0
    assert entry["full"]["neg_frac"]["mean"] == 0.25
    assert entry["seeds"]["seed_42"]["full"]["lambda_max"] == 4.0
    assert entry["seeds"]["seed_43"]["full"]["lambda_max"] == 12.0
