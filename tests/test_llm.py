"""Toy SegmentedTransformer: attention-visible partition, LM loss, and Phi conformance.

White-box checks that the exp9 toy decoder exposes `get_param_groups` as a clean partition
with the query / key / value / output projections kept as separate (visible) blocks, that
the causal LM loss shifts correctly, and that the model plugs into `ParamBlockEstimator`
as a `ParamGroupedModel` (the seam exp9 measures). Numerical fidelity of Phi itself is
covered in `test_param_space`; here we verify structure and the integration seam.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.llm import (  # noqa: E402
    CausalLMCrossEntropy,
    HFSegmentedCausalLM,
    SegmentedTransformer,
    random_token_batch,
    sample_token_batch,
)
from hessian.param_space import ParamBlockEstimator  # noqa: E402
from tests.test_group_sizes import _N_LAYERS, _fake_model  # noqa: E402


def _assert_partition(model: SegmentedTransformer) -> None:
    groups = model.get_param_groups()
    grouped = [p for ps in groups.values() for p in ps]
    grouped_ids = {id(p) for p in grouped}
    all_ids = {id(p) for p in model.parameters()}
    assert len(grouped_ids) == len(grouped), "a parameter appears in two groups"
    assert grouped_ids == all_ids, "groups must partition the model parameters exactly"


def test_param_groups_visible_attention() -> None:
    model = SegmentedTransformer.from_preset("tiny")  # n_layers = 2
    _assert_partition(model)

    names = list(model.get_param_groups().keys())
    expected = [
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
    assert names == expected, f"group order / visibility wrong: {names}"

    # The pre-attention norm rides with the query block (BN-with-segment convention).
    groups = model.get_param_groups()
    ln1_ids = {
        id(p) for n, p in model.named_parameters() if n.startswith("blocks.0.ln1.")
    }
    q_ids = {id(p) for p in groups["L0.attn_q"]}
    assert ln1_ids <= q_ids, "pre-attention LayerNorm must ride with the query block"


def test_toy_decoder_partition_composition() -> None:
    """The toy decoder groups two MLP projections, both embeddings and norms.

    The MLP comprises fc1 and fc2; embed combines token and position tables;
    the pre-MLP norm belongs to mlp, and head combines final norm and output.
    """
    model = SegmentedTransformer.from_preset("small")
    block = model.blocks[0]

    # Exactly two Linear children make up the MLP -- fc1 and fc2 -- once the
    # four attention projections are excluded from the block's own children.
    linear_children = {
        n for n, m in block.named_children() if isinstance(m, torch.nn.Linear)
    }
    mlp_linear_names = linear_children - {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert mlp_linear_names == {"fc1", "fc2"}, (
        "the toy MLP must be exactly fc1, fc2 (two projections), not a "
        "three-projection gate/up/down MLP"
    )

    groups = model.get_param_groups()

    # embed fuses two Embedding tables (token, position), unlike the HF
    # backend's single embed_tokens table.
    tok_ids = {id(p) for p in model.tok_emb.parameters()}
    pos_ids = {id(p) for p in model.pos_emb.parameters()}
    assert tok_ids and pos_ids and tok_ids.isdisjoint(pos_ids)
    assert {id(p) for p in groups["embed"]} == tok_ids | pos_ids, (
        "embed must be the token table fused with the position table"
    )

    # The pre-MLP norm rides with the mlp group.
    ln2_ids = {
        id(p) for n, p in model.named_parameters() if n.startswith("blocks.0.ln2.")
    }
    mlp_group_ids = {id(p) for p in groups["L0.mlp"]}
    assert ln2_ids <= mlp_group_ids, "pre-MLP LayerNorm must ride with the mlp block"

    # head fuses the final norm with the output projection.
    lnf_ids = {id(p) for p in model.ln_f.parameters()}
    lmhead_ids = {id(p) for p in model.lm_head.parameters()}
    assert {id(p) for p in groups["head"]} == lnf_ids | lmhead_ids, (
        "head must be the final norm fused with the output projection"
    )


def test_pretrained_partition_composition_tied_embeddings() -> None:
    """Tied output weights belong to embed, leaving the final norm in head.

    A fake pretrained-shaped model exercises deduplication without a download.
    """
    model = _fake_model()
    model.lm_head.weight = model.model.embed_tokens.weight  # tie, in place

    groups = HFSegmentedCausalLM(model).get_param_groups()

    norm_ids = {id(p) for p in model.model.norm.parameters()}
    lm_head_ids = {id(p) for p in model.lm_head.parameters()}
    embed_ids = {id(p) for p in groups["embed"]}
    head_ids = {id(p) for p in groups["head"]}

    assert lm_head_ids <= embed_ids, (
        "a tied output projection must be folded into embed by _dedupe_groups, "
        "not left behind in head"
    )
    assert head_ids == norm_ids, (
        "with tied embeddings, head must reduce to the final normalisation "
        "alone, not the norm paired with the output projection"
    )

    grouped = [p for ps in groups.values() for p in ps]
    grouped_ids = {id(p) for p in grouped}
    all_ids = {id(p) for p in model.parameters()}
    assert len(grouped_ids) == len(grouped), "a parameter appears in two groups"
    assert grouped_ids == all_ids, "groups must partition the model parameters exactly"
    assert len(groups) == 1 + _N_LAYERS * 5 + 1, (
        "group count must be embed, five groups per layer, and head"
    )


def test_pretrained_partition_composition_untied_embeddings() -> None:
    """The paired, untied case for the test above: without weight tying, ``head``
    keeps both the final normalisation and the (independent) output projection --
    two modules, matching the toy ladder's ``head`` -- so the tied-embeddings test
    above is exercising the tying, not a property of `HFSegmentedCausalLM` that
    holds regardless of it.
    """
    model = _fake_model()  # untied by construction (test_group_sizes.py's docstring)

    groups = HFSegmentedCausalLM(model).get_param_groups()

    norm_ids = {id(p) for p in model.model.norm.parameters()}
    lm_head_ids = {id(p) for p in model.lm_head.parameters()}
    head_ids = {id(p) for p in groups["head"]}

    assert norm_ids and lm_head_ids and norm_ids.isdisjoint(lm_head_ids)
    assert head_ids == norm_ids | lm_head_ids, (
        "without tying, head must fuse the final norm with the output projection"
    )


def test_param_groups_block_granularity() -> None:
    model = SegmentedTransformer.from_preset("tiny", attn_granularity="block")
    _assert_partition(model)
    names = list(model.get_param_groups().keys())
    assert names == [
        "embed",
        "L0.attn",
        "L0.mlp",
        "L1.attn",
        "L1.mlp",
        "head",
    ], f"coarse attention grouping wrong: {names}"


def test_forward_shape() -> None:
    model = SegmentedTransformer.from_preset("tiny")
    ids = torch.randint(0, model.vocab_size, (2, 8))
    out = model(ids)
    assert out.shape == (2, 8, model.vocab_size), "logits shape (B, S, V)"


def test_from_preset_unknown_raises() -> None:
    raised = False
    try:
        SegmentedTransformer.from_preset("gpt5")
    except ValueError:
        raised = True
    assert raised, "unknown preset must raise ValueError"


def test_invalid_granularity_raises() -> None:
    raised = False
    try:
        SegmentedTransformer.from_preset("tiny", attn_granularity="heads")
    except ValueError:
        raised = True
    assert raised, "invalid attn_granularity must raise ValueError"


def test_causal_lm_cross_entropy_shift() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.randint(0, 7, (2, 5))
    loss = CausalLMCrossEntropy()(logits, targets)
    manual = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 7), targets[:, 1:].reshape(-1)
    )
    assert torch.allclose(loss, manual), "loss must score logits_t against token_{t+1}"


def test_random_token_batch_deterministic() -> None:
    a = random_token_batch(50, 4, 6, generator=torch.Generator().manual_seed(3))
    b = random_token_batch(50, 4, 6, generator=torch.Generator().manual_seed(3))
    assert a.shape == (4, 6) and a.dtype == torch.long, "shape / dtype"
    assert torch.equal(a, b), "same seed must give the same batch"
    assert int(a.min()) >= 0 and int(a.max()) < 50, "ids must lie in the vocabulary"


def test_sample_token_batch_is_padding_free() -> None:
    class _FakeTok:
        def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
            ids = torch.arange(len(text.split())).unsqueeze(0)
            return SimpleNamespace(input_ids=ids)

    batch = sample_token_batch(_FakeTok(), batch_size=3, seq_len=4)
    assert batch.shape == (3, 4), "concatenate-and-chunk must fill (batch, seq_len)"


def test_estimate_phi_report_respects_coupling_pairs() -> None:
    torch.manual_seed(0)
    model = SegmentedTransformer.from_preset("tiny")
    model.eval()
    ids = random_token_batch(
        model.vocab_size, 2, 8, generator=torch.Generator().manual_seed(0)
    )
    est = ParamBlockEstimator(model, CausalLMCrossEntropy(), n_probes=2)
    names = list(model.get_param_groups().keys())
    subset = [(names[0], names[1]), (names[1], names[2])]
    report = est.estimate_phi_report(ids, ids, coupling_pairs=subset)
    assert set(report.coupling) == set(subset), "coupling restricted to the given pairs"
    assert list(report.diag_frob.keys()) == names, "diagonals still cover every block"
    assert math.isfinite(report.phi), "phi is unaffected by the coupling subset"


def test_is_param_grouped_model_for_phi() -> None:
    torch.manual_seed(0)
    model = SegmentedTransformer.from_preset("tiny")
    model.eval()
    ids = random_token_batch(
        model.vocab_size, 2, 8, generator=torch.Generator().manual_seed(0)
    )
    report = ParamBlockEstimator(
        model, CausalLMCrossEntropy(), n_probes=2
    ).estimate_phi_report(ids, ids)

    names = list(model.get_param_groups().keys())
    assert list(report.diag_frob.keys()) == names, "diag_frob must be group-major"
    assert math.isfinite(report.phi), "phi must be finite"

    expected_pairs = {(v, w) for i, v in enumerate(names) for w in names[i + 1 :]}
    assert set(report.coupling) == expected_pairs, "coupling must cover all v<w pairs"
