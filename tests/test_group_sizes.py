"""``experiments.group_sizes``: Phi_null's partitions rebuilt without weights or data.

White-box checks for the CPU-only, no-download rebuild `paper_data.tables.exp789`
falls back on when a bundle predates `experiments.phi.group_sizes_of`. The exp9
on-demand LLM branch (``exp9_a3_llm``) is exercised in two layers: `_hf_group_sizes`
(the pure module-tree -> sizes step) on a FAKE Llama/Qwen2-shaped module built on the
``meta`` device - no network, no real weights - and `group_sizes_for` itself, which
must degrade to None rather than raise when ``transformers`` cannot be imported at all
(this repo runs without it; see `experiments.llm`'s module docstring). The exp8-family
and toy-transformer branches are exercised end-to-end already, through the Phi-map
table's own tests (`test_paper_data_exp789.py`).
"""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

from experiments.group_sizes import _hf_group_sizes, group_sizes_for  # noqa: E402

# ======================================================================
# A FAKE Llama/Qwen2-shaped module tree, on the meta device
# ======================================================================


class _FakeRMSNorm(nn.Module):
    """A weight-only norm, standing in for Llama/Qwen2's RMSNorm (no bias term)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, device="meta"))


class _FakeAttn(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device="meta")
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device="meta")
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device="meta")
        self.o_proj = nn.Linear(d_model, d_model, bias=False, device="meta")


class _FakeMLP(nn.Module):
    """A 3-Linear SwiGLU-shaped MLP (gate/up/down projections), as Llama/Qwen2 use."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, device="meta")
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, device="meta")
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, device="meta")


class _FakeLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttn(d_model)
        self.mlp = _FakeMLP(d_model, d_ff)
        self.input_layernorm = _FakeRMSNorm(d_model)
        self.post_attention_layernorm = _FakeRMSNorm(d_model)


class _FakeBackbone(nn.Module):
    def __init__(self, vocab: int, d_model: int, d_ff: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d_model, device="meta")
        self.layers = nn.ModuleList(_FakeLayer(d_model, d_ff) for _ in range(n_layers))
        self.norm = _FakeRMSNorm(d_model)


class _FakeCausalLM(nn.Module):
    """A Llama/Qwen2-shaped causal LM with untied input/output embeddings.

    Untied on purpose (a fresh ``lm_head`` Linear, not a view of ``embed_tokens``):
    `experiments.llm.HFSegmentedCausalLM._dedupe_groups` only matters for the tied
    case, which is covered on the real toy/HF models elsewhere (``test_llm.py``); here
    every parameter belongs to exactly one group from the start, so
    ``_assert_partition_complete`` holding is a direct check of the segmentation
    itself, not of the dedup step.
    """

    def __init__(self, vocab: int, d_model: int, d_ff: int, n_layers: int) -> None:
        super().__init__()
        self.model = _FakeBackbone(vocab, d_model, d_ff, n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False, device="meta")


_VOCAB, _D_MODEL, _D_FF, _N_LAYERS = 6, 4, 8, 2


def _fake_model() -> _FakeCausalLM:
    return _FakeCausalLM(_VOCAB, _D_MODEL, _D_FF, _N_LAYERS)


def _expected_sizes() -> list[int]:
    """Hand-derived group sizes for `_fake_model`, in `get_param_groups`'s own order.

    Mirrors ``"qkv"`` attention granularity directly (`experiments.llm._layer_groups`):
    ``attn_q`` carries the pre-attention norm, ``mlp`` the pre-mlp norm - not obtained
    by calling that helper, so this is an independent check of what it produces.
    """
    d, ff, v = _D_MODEL, _D_FF, _VOCAB
    per_layer = [d * d + d, d * d, d * d, d * d, 3 * d * ff + d]  # q,k,v,o,mlp
    return [v * d] + per_layer * _N_LAYERS + [d + v * d]  # embed, ...layers..., head


# ======================================================================
# _hf_group_sizes: the pure module-tree -> sizes step
# ======================================================================


def test_hf_group_sizes_matches_hand_derived_sizes_and_order() -> None:
    sizes = _hf_group_sizes(_fake_model(), "qkv")
    assert sizes == _expected_sizes()


def test_hf_group_sizes_totals_the_fake_models_parameter_count() -> None:
    # A meta tensor's numel is exact (it is the shape's property, not the storage's),
    # so this holds even though not one real weight was allocated.
    fake = _fake_model()
    sizes = _hf_group_sizes(fake, "qkv")
    assert sum(sizes) == sum(p.numel() for p in fake.parameters())


# ======================================================================
# group_sizes_for("exp9_a3_llm", ...): degrades to None, never raises
# ======================================================================


def test_group_sizes_for_llm_returns_none_when_transformers_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This repo runs without `transformers` installed (experiments.llm's module
    # docstring); forcing the same ImportError here - rather than relying on whichever
    # way the environment happens to be set up - makes the test meaningful whether or
    # not the optional 'llm' extra is installed on the machine running it.
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert group_sizes_for("exp9_a3_llm", "Qwen/Qwen2.5-0.5B", None) is None
