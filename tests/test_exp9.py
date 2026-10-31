"""Exp9 runner wiring: backend dispatch, token batching, and Phi measurement.

White-box, dependency-free checks that the Stage-A "A3" ladder runner composes the toy
transformer, the parameter-space estimator, and the seed aggregation into a JSON-safe Phi
report - the integration seam. The ``hf`` backend (frozen Hugging Face LMs) needs network
weights and the optional ``transformers`` dependency, so it is exercised on demand, not
here; numerical fidelity of Phi is covered in `test_param_space`.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from experiments.config import Exp9Config  # noqa: E402
from experiments.llm import CausalLMCrossEntropy, SegmentedTransformer  # noqa: E402
from experiments.phi import group_sizes_of, measure_phi  # noqa: E402
from experiments.runner_exp9 import (  # noqa: E402
    _build_model,
    _count_params,
    _coupling_pairs,
    _layer_key,
    _make_token_batch,
    _measure_phi,
    aggregate,
)

_DEVICE = torch.device("cpu")
# Small toy measurement config: keeps the white-box runner checks fast.
_CFG = dataclasses.replace(Exp9Config(), hessian_batch_size=2, seq_len=8, n_probes=2)


def test_build_model_toy_dispatch() -> None:
    model, tokenizer = _build_model(_CFG, "tiny", _DEVICE, torch.float32)
    assert isinstance(model, SegmentedTransformer), (
        "toy backend must build a transformer"
    )
    assert tokenizer is None, "toy backend has no tokenizer"


def test_build_model_unknown_backend_raises() -> None:
    bad = dataclasses.replace(_CFG, backend="jax")
    raised = False
    try:
        _build_model(bad, "tiny", _DEVICE, torch.float32)
    except ValueError:
        raised = True
    assert raised, "unknown backend must raise ValueError"


def test_make_token_batch_toy() -> None:
    model, tokenizer = _build_model(_CFG, "tiny", _DEVICE, torch.float32)
    assert isinstance(model, SegmentedTransformer)
    ids = _make_token_batch(_CFG, model, tokenizer, _DEVICE)
    assert ids.shape == (_CFG.hessian_batch_size, _CFG.seq_len), "batch shape"
    assert ids.dtype == torch.long, "token ids must be long"
    assert int(ids.min()) >= 0 and int(ids.max()) < model.vocab_size, "ids in vocab"


def test_count_params_matches_model() -> None:
    model, _ = _build_model(_CFG, "tiny", _DEVICE, torch.float32)
    assert isinstance(model, SegmentedTransformer)
    total = sum(p.numel() for p in model.parameters())
    assert _count_params(model) == total, "untied toy params: count == sum over groups"


def test_measure_phi_structure() -> None:
    model, tokenizer = _build_model(_CFG, "tiny", _DEVICE, torch.float32)
    ids = _make_token_batch(_CFG, model, tokenizer, _DEVICE)
    loss_fn = CausalLMCrossEntropy()

    torch.manual_seed(42)
    report = _measure_phi(
        model, ids, loss_fn, n_probes=_CFG.n_probes, coupling_band=None
    )

    assert set(report) == {
        "phi",
        "diag_frob",
        "coupling",
        "group_sizes",
    }, "report keys"
    assert math.isfinite(report["phi"]), "phi must be finite"

    names = list(model.get_param_groups().keys())
    assert list(report["diag_frob"].keys()) == names, "diag_frob must be group-major"
    expected_pairs = {f"{v}->{w}" for i, v in enumerate(names) for w in names[i + 1 :]}
    assert set(report["coupling"]) == expected_pairs, "coupling covers all v<w pairs"
    assert report["group_sizes"] == group_sizes_of(model), "group_sizes via the helper"

    # Identical seeds and coupling pairs give identical CPU estimates.
    torch.manual_seed(42)
    direct = measure_phi(
        model,
        ids,
        ids,
        loss_fn,
        _CFG.n_probes,
        coupling_pairs=_coupling_pairs(names, None),
    )
    assert direct == report, "_measure_phi must match measure_phi bit-for-bit"


def test_layer_key() -> None:
    assert _layer_key("L0.attn_q") == "L0"
    assert _layer_key("L12.mlp") == "L12"
    assert _layer_key("embed") is None
    assert _layer_key("head") is None


def test_coupling_pairs_full_is_none() -> None:
    names = ["embed", "L0.attn_q", "L0.mlp", "head"]
    assert _coupling_pairs(names, None) is None, "full scope defers to the estimator"


def test_coupling_pairs_band_keeps_intra_layer_and_band() -> None:
    names = [
        "embed",
        "L0.attn_q",
        "L0.attn_k",
        "L0.attn_v",
        "L0.attn_o",
        "L0.mlp",
        "L1.attn_q",
        "L1.attn_k",
        "L1.attn_v",
        "L1.attn_o",
        "L1.mlp",
        "head",
    ]
    pairs = _coupling_pairs(names, band=1)
    assert pairs is not None
    pairset = set(pairs)
    # Far-apart pairs in the SAME layer survive the band (the attention hotspots).
    assert ("L0.attn_q", "L0.mlp") in pairset
    assert ("L1.attn_v", "L1.attn_o") in pairset
    # A cross-layer pair beyond the band is dropped...
    assert ("L0.attn_q", "L1.mlp") not in pairset
    # ...but an adjacent cross-layer pair (distance 1) is kept.
    assert ("L0.mlp", "L1.attn_q") in pairset
    # embed / head have no layer, so they couple only within the band.
    assert ("embed", "L0.attn_q") in pairset
    assert ("embed", "L0.attn_k") not in pairset
    # Every pair is upper-triangular in group order.
    idx = {n: i for i, n in enumerate(names)}
    assert all(idx[v] < idx[w] for v, w in pairs), "pairs must be v-before-w"


def test_aggregate() -> None:
    raw = {
        "tiny": {
            "seed_42": {
                "phi": 0.4,
                "diag_frob": {"embed": 1.0},
                "coupling": {"embed->head": 0.2},
            },
            "seed_43": {
                "phi": 0.6,
                "diag_frob": {"embed": 3.0},
                "coupling": {"embed->head": 0.4},
            },
        }
    }
    agg = aggregate(raw, {"tiny": 1234})

    assert agg["tiny"]["seeds"] == raw["tiny"], "raw per-seed reports are kept"
    assert math.isclose(agg["tiny"]["phi"]["mean"], 0.5), "phi mean over seeds"
    assert math.isclose(agg["tiny"]["diag_frob"]["embed"]["mean"], 2.0), (
        "diagonal-norm mean"
    )
    assert math.isclose(agg["tiny"]["coupling"]["embed->head"]["mean"], 0.3), (
        "coupling mean"
    )
    assert agg["tiny"]["n_params"] == 1234, "parameter count carried through"
    assert agg["tiny"]["n_seeds"] == 2, "seed count recorded"
