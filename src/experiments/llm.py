"""Decoder-only transformers for the Phi ladder (exp9).

Exp9 measures, on frozen decoder-only transformers, the same parameter-space blind-spot
fraction exp8 reports on convolutional DAGs:

  Phi = 1 - sum_v ||H_{theta_v, theta_v}||_F^2 / ||H||_F^2  (exact nonzero Hessian: [0, 1])

- the curvature mass a block-diagonal preconditioner (K-FAC / EKFAC) discards - together
with the coupling field C(v, w) that locates it. The ladder asks
how Phi scales with model size and whether pretraining (source (P)) inflates it.

This module supplies the models the runner measures, in two backends that share *one*
parameter-grouping convention so the diagnostic reads identically across them:

  - ``SegmentedTransformer``: a self-contained pre-norm decoder (pure PyTorch, no Hugging
    Face dependency) - the local ladder and the test-suite. Attention is kept *visible*:
    each layer exposes its query / key / value / output projections as separate parameter
    blocks, so the coupling the paper isolates in attention (H^T_{Q,K} != 0, exp5) is
    directly measurable.
  - ``load_hf_segmented``: a thin wrapper that segments a pretrained Hugging Face causal
    LM (Qwen2.5, Llama-3.1,...) under the *same* block convention. ``transformers`` is
    imported lazily, so this module - and the toy ladder - import without it.

Parameter-space Phi differentiates the loss twice w.r.t. the weights, so the measured
model must hold differentiable parameters: the wrapper loads in bf16 / fp32 with gradients
enabled. 4-bit quantized weights are *not* differentiable and are therefore out of scope
here (deferred).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Built-in measurement text for the ``hf`` backend: concatenated and chunked into a
# padding-free token minibatch (original neutral prose, no external corpus dependency).
_SAMPLE_TEXT: tuple[str, ...] = (
    "The curvature of a deep network describes how its loss bends as the weights move.",
    "A block-diagonal preconditioner keeps each layer's own curvature and discards the "
    "coupling between layers.",
    "Attention ties queries, keys, and values together, so the curvature shared across "
    "those projections is exactly what a layer-wise approximation throws away.",
    "Measuring this discarded fraction turns an architectural intuition into a number "
    "one can track as a model grows.",
)


def resolve_dtype(name: str) -> torch.dtype:
    """Maps a config dtype name to its ``torch.dtype`` (float32 / bfloat16)."""
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


# ======================================================================
# Shared attention-visible block convention
# ======================================================================


def _layer_groups(
    prefix: str,
    *,
    q: list[nn.Parameter],
    k: list[nn.Parameter],
    v: list[nn.Parameter],
    o: list[nn.Parameter],
    mlp: list[nn.Parameter],
    pre_attn_norm: list[nn.Parameter],
    pre_mlp_norm: list[nn.Parameter],
    attn_granularity: str,
) -> dict[str, list[nn.Parameter]]:
    """One layer's parameter blocks under the shared attention-visible convention.

    With ``attn_granularity == "qkv"`` the query / key / value / output projections are
    four separate blocks, so the attention coupling the paper isolates (H^T_{Q,K} != 0)
    is measurable; ``"block"`` merges them into a single attention block for the large
    models where four blocks per layer is too many. The two LayerNorms ride with a
    neighbour - the pre-attention norm with the query block, the pre-MLP norm with the
    MLP - mirroring the BatchNorm-with-segment convention of `SegmentedResNet18` (a K-FAC
    backend that does not factor a norm falls back to an identity block).
    """
    if attn_granularity == "qkv":
        return {
            f"{prefix}.attn_q": q + pre_attn_norm,
            f"{prefix}.attn_k": k,
            f"{prefix}.attn_v": v,
            f"{prefix}.attn_o": o,
            f"{prefix}.mlp": mlp + pre_mlp_norm,
        }
    return {
        f"{prefix}.attn": q + k + v + o + pre_attn_norm,
        f"{prefix}.mlp": mlp + pre_mlp_norm,
    }


def _check_granularity(attn_granularity: str) -> None:
    if attn_granularity not in ("qkv", "block"):
        raise ValueError(
            f"attn_granularity {attn_granularity!r} not in ('qkv', 'block')"
        )


# ======================================================================
# Causal language-model loss
# ======================================================================


class CausalLMCrossEntropy(nn.Module):
    """Next-token cross-entropy with an in-module causal shift.

    Matches the estimator's ``loss_fn(model(x), y)`` contract with ``x = y = input_ids``:
    the logits at position t are scored against the token at t + 1.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_targets = targets[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_targets)


# ======================================================================
# Toy decoder-only transformer (attention kept visible)
# ======================================================================


@dataclass(frozen=True)
class TransformerSpec:
    """Shape of a toy decoder-only transformer."""

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = 512
    max_seq_len: int = 512


_TOY_PRESETS: dict[str, TransformerSpec] = {
    "tiny": TransformerSpec(d_model=64, n_layers=2, n_heads=2, d_ff=256),
    "small": TransformerSpec(d_model=128, n_layers=4, n_heads=4, d_ff=512),
    "base": TransformerSpec(d_model=256, n_layers=6, n_heads=8, d_ff=1024),
}


class _DecoderBlock(nn.Module):
    """Pre-norm decoder block with separate Q/K/V/O projections (manual attention).

    Attention is computed by hand (not via fused SDPA) so the second backward the
    parameter-space Hessian needs is always available.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.ln1(x))
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x

    def _attn(self, h: Tensor) -> Tensor:
        b, s, _ = h.shape
        q = self._split_heads(self.q_proj(h), b, s)
        k = self._split_heads(self.k_proj(h), b, s)
        v = self._split_heads(self.v_proj(h), b, s)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=h.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (b, n_heads, s, head_dim)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def _split_heads(self, t: Tensor, b: int, s: int) -> Tensor:
        return t.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)


class SegmentedTransformer(nn.Module):
    """Pre-norm decoder-only transformer with an attention-visible block partition.

    Conforms to ``ParamGroupedModel``: ``forward(input_ids) -> logits`` and
    ``get_param_groups`` returns the parameter-space blocks the Phi estimator consumes,
    in group-major order ``embed`` -> per-layer attention / MLP blocks -> ``head``.
    """

    def __init__(self, spec: TransformerSpec, *, attn_granularity: str = "qkv") -> None:
        super().__init__()
        _check_granularity(attn_granularity)
        self._spec = spec
        self._attn_granularity = attn_granularity
        self.tok_emb = nn.Embedding(spec.vocab_size, spec.d_model)
        self.pos_emb = nn.Embedding(spec.max_seq_len, spec.d_model)
        self.blocks = nn.ModuleList(
            _DecoderBlock(spec.d_model, spec.n_heads, spec.d_ff)
            for _ in range(spec.n_layers)
        )
        self.ln_f = nn.LayerNorm(spec.d_model)
        self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)

    @classmethod
    def from_preset(
        cls, name: str, *, attn_granularity: str = "qkv"
    ) -> SegmentedTransformer:
        """Builds a toy model from a named size in ``_TOY_PRESETS``."""
        if name not in _TOY_PRESETS:
            raise ValueError(
                f"unknown toy preset {name!r}; expected one of {sorted(_TOY_PRESETS)}"
            )
        return cls(_TOY_PRESETS[name], attn_granularity=attn_granularity)

    @property
    def vocab_size(self) -> int:
        return self._spec.vocab_size

    def forward(self, input_ids: Tensor) -> Tensor:
        _, s = input_ids.shape
        if s > self._spec.max_seq_len:
            raise ValueError(
                f"sequence length {s} exceeds max_seq_len {self._spec.max_seq_len}"
            )
        pos = torch.arange(s, device=input_ids.device)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.ln_f(h))

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {
            "embed": list(self.tok_emb.parameters()) + list(self.pos_emb.parameters()),
        }
        for i, raw_block in enumerate(self.blocks):
            block = cast(_DecoderBlock, raw_block)
            groups.update(
                _layer_groups(
                    f"L{i}",
                    q=list(block.q_proj.parameters()),
                    k=list(block.k_proj.parameters()),
                    v=list(block.v_proj.parameters()),
                    o=list(block.o_proj.parameters()),
                    mlp=list(block.fc1.parameters()) + list(block.fc2.parameters()),
                    pre_attn_norm=list(block.ln1.parameters()),
                    pre_mlp_norm=list(block.ln2.parameters()),
                    attn_granularity=self._attn_granularity,
                )
            )
        groups["head"] = list(self.ln_f.parameters()) + list(self.lm_head.parameters())
        return groups


# ======================================================================
# Frozen Hugging Face causal LM (on-demand backend)
# ======================================================================


def _dedupe_groups(
    groups: dict[str, list[nn.Parameter]],
) -> dict[str, list[nn.Parameter]]:
    """Drops parameters already claimed by an earlier block, keeping a true partition.

    Tied input/output embeddings (the same Parameter behind ``embed_tokens`` and
    ``lm_head``, common on the smaller Qwen2.5 rungs) would otherwise be double-counted;
    a block left empty by de-duplication is dropped.
    """
    seen: set[int] = set()
    deduped: dict[str, list[nn.Parameter]] = {}
    for name, params in groups.items():
        kept = [p for p in params if id(p) not in seen]
        seen.update(id(p) for p in kept)
        if kept:
            deduped[name] = kept
    return deduped


def _assert_partition_complete(
    groups: dict[str, list[nn.Parameter]],
    model: Any,
) -> None:
    """Fails fast if the segmentation missed a trainable parameter of ``model``.

    Parameter-space Phi is measured over exactly the grouped parameters, so a tensor left
    out of every block would be silently dropped from both ||H||_F^2 and the diagonals -
    a quietly wrong Phi. On the on-demand HF backend, where the module tree is *assumed*
    to follow the Llama / Qwen2 layout, this turns a layout mismatch into a clear error
    before an expensive run rather than a silent bias after it.
    """
    grouped = {id(p) for ps in groups.values() for p in ps}
    expected = {id(p) for p in model.parameters()}
    missing = expected - grouped
    if missing:
        raise ValueError(
            f"segmentation missed {len(missing)} of {len(expected)} parameter tensors; "
            "the model layout differs from the expected Llama / Qwen2 module tree"
        )


class HFSegmentedCausalLM(nn.Module):
    """Segments a frozen Hugging Face causal LM under the toy model's block convention.

    Targets the Llama / Qwen2 module layout (shared across both families):
    ``model.embed_tokens``, ``model.layers[i].self_attn.{q,k,v,o}_proj`` with
    ``input_layernorm`` / ``post_attention_layernorm``, ``model.layers[i].mlp``, and the
    (tied or untied) ``lm_head`` over ``model.norm``. ``forward(input_ids) -> logits`` and
    ``get_param_groups`` make it a ``ParamGroupedModel``.
    """

    def __init__(self, model: nn.Module, *, attn_granularity: str = "qkv") -> None:
        super().__init__()
        _check_granularity(attn_granularity)
        # Typed Any: attribute access below targets the HF module tree, not nn.Module.
        self._model: Any = model
        self._attn_granularity = attn_granularity

    def forward(self, input_ids: Tensor) -> Tensor:
        return self._model(input_ids=input_ids).logits

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone = self._model.model
        groups: dict[str, list[nn.Parameter]] = {
            "embed": list(backbone.embed_tokens.parameters()),
        }
        for i, layer in enumerate(backbone.layers):
            attn = layer.self_attn
            groups.update(
                _layer_groups(
                    f"L{i}",
                    q=list(attn.q_proj.parameters()),
                    k=list(attn.k_proj.parameters()),
                    v=list(attn.v_proj.parameters()),
                    o=list(attn.o_proj.parameters()),
                    mlp=list(layer.mlp.parameters()),
                    pre_attn_norm=list(layer.input_layernorm.parameters()),
                    pre_mlp_norm=list(layer.post_attention_layernorm.parameters()),
                    attn_granularity=self._attn_granularity,
                )
            )
        groups["head"] = list(backbone.norm.parameters()) + list(
            self._model.lm_head.parameters()
        )
        deduped = _dedupe_groups(groups)
        _assert_partition_complete(deduped, self._model)
        return deduped


def load_hf_segmented(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    attn_granularity: str = "qkv",
) -> tuple[HFSegmentedCausalLM, Any]:
    """Loads a frozen HF causal LM in ``dtype`` and segments it; returns (model, tokenizer).

    ``transformers`` is imported lazily so the package imports without it; gradients are
    enabled on every parameter because parameter-space Phi differentiates twice w.r.t. the
    weights. 4-bit quantized weights are not differentiable and so are not supported here.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only on the hf backend
        raise ImportError(
            "exp9 'hf' backend needs transformers; install it with "
            "`uv pip install transformers accelerate` (the optional 'llm' extra)."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=resolve_dtype(dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for param in model.parameters():
        param.requires_grad_(True)
    return HFSegmentedCausalLM(model, attn_granularity=attn_granularity), tokenizer


# ======================================================================
# Token minibatches
# ======================================================================


def random_token_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """A deterministic ``(batch_size, seq_len)`` block of random token ids (toy backend)."""
    return torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long
    )


def sample_token_batch(tokenizer: Any, batch_size: int, seq_len: int) -> Tensor:
    """A padding-free ``(batch_size, seq_len)`` token block from the built-in sample text.

    The sample is concatenated into one stream and chunked into blocks - the standard
    language-model batching - so every position is a real token and the next-token loss
    carries no padding.
    """
    ids = tokenizer(" ".join(_SAMPLE_TEXT), return_tensors="pt").input_ids[0]
    need = batch_size * seq_len
    if ids.numel() < need:
        ids = ids.repeat((need // ids.numel()) + 1)
    return ids[:need].view(batch_size, seq_len)
