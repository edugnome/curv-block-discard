"""Group sizes for a Phi-map row, without downloading weights or touching data.

phi_null (`hessian.phi_baselines.phi_null`) needs only the parameter partition's block
sizes p_v, which are fixed by the architecture - and, for a classification head, by the
number of classes - never by the trained weights or the data a model was measured on.
Old bundles do not record this partition (a later measurement writes it to every seed
record, see `experiments.phi.measure_phi`'s and `experiments.runner_exp9._measure_phi`'s
own ``group_sizes``, both built on `experiments.phi.group_sizes_of`), so this module
rebuilds it on CPU, the same way the runner built the model in the first place, without
loading a dataset or downloading a checkpoint. The exp9 on-demand LLM ladder
(``exp9_a3_llm``) rebuilds its config and module tree the same cache-only way: its
``AutoConfig`` comes from the local Hugging Face cache alone (never the network), and
the module tree it segments is built on the ``meta`` device, so no real weight is ever
allocated for a statistic that only needs shapes.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
import torch.nn as nn

from experiments.config import Exp8Config, Exp9Config
from experiments.llm import HFSegmentedCausalLM, SegmentedTransformer
from experiments.phi import group_sizes_of
from experiments.runner_exp8 import _build_model

log = logging.getLogger(__name__)

# Profile fallback for exp8 bundles without metadata. The profile fixes model
# construction fields except num_classes, which the bundle may override.
_EXP8_PROFILE = {
    "exp8": Exp8Config.a1_from_scratch,
    "exp8_a1_width": Exp8Config.a1_width,
    "exp8_a1_imagenet32": Exp8Config.a1_imagenet32,
    "exp8_a2": Exp8Config.a2_pretrained,
}


def _hf_group_sizes(hf_module: nn.Module, attn_granularity: str) -> list[int]:
    """Segments an HF causal-LM module tree and sums each group's parameter count.

    The module-tree -> sizes step, kept separate from the network/cache lookup in
    `group_sizes_for` so a test can hand it a small FAKE Llama-shaped module built on
    the ``meta`` device - no download, no real weights (``Tensor.numel`` is exact on a
    meta tensor: it reads the shape, not the absent storage) - instead of a real
    multi-hundred-million-parameter model. Raises ``ValueError`` exactly when
    `experiments.llm.HFSegmentedCausalLM.get_param_groups` does: the module tree does
    not match the expected Llama / Qwen2 layout (`group_sizes_for` turns that into
    None, same as an unrecognised exp8 arch or toy preset).
    """
    wrapped = HFSegmentedCausalLM(hf_module, attn_granularity=attn_granularity)
    return group_sizes_of(wrapped)


def group_sizes_for(
    experiment: str, model_key: str, num_classes: int | None
) -> list[int] | None:
    """Parameter-group sizes of one Phi-map row, or None when they cannot be had safely.

    ``experiment`` is the row's registry key (``exp8``, ``exp8_a1_width``,
    ``exp8_a1_imagenet32``, ``exp8_a2``, ``exp9``, ``exp9_a3_llm``); ``model_key`` is
    the row's own key inside that bundle - an arch name for the exp8 family, a model id
    for exp9. ``num_classes`` is the bundle's own recorded ``meta.config.num_classes``
    when its provenance block has one; None falls back to the profile's default (the
    build touches neither data nor weights, so the two agree unless a run overrode
    ``--num-classes``, which changes only the head - a small group next to the trunk's).

    CPU-only and never downloads: an exp8 ``resnet50`` row is built with
    ``pretrained=False`` regardless of the profile (sizes do not depend on the
    weights); an exp9 Hugging Face model id (``exp9_a3_llm``) rebuilds only its
    ``AutoConfig`` from the local Hugging Face cache (``local_files_only=True`` - a
    machine without that model's config already downloaded gets None rather than this
    function reaching out to the network) and builds the module tree it segments on the
    ``meta`` device, so no real weight is ever allocated; ``transformers`` is imported
    lazily, so not having it installed at all (this module ships without it) also
    degrades to None rather than raising. An arch/preset name neither builder
    recognises, or any other experiment key this module does not recognise, also
    returns None (logged) rather than raising: this is a reference statistic next to
    the measured Phi, not the measurement itself, so a row a future arch rename - or an
    offline machine, for the LLM ladder - orphans should read as unavailable, not fail
    the whole table.
    """
    try:
        if experiment in _EXP8_PROFILE:
            cfg = _EXP8_PROFILE[experiment]()
            if num_classes is not None:
                cfg = replace(cfg, num_classes=num_classes)
            cfg = replace(cfg, pretrained=False)
            model = _build_model(cfg, model_key)
            return group_sizes_of(model)
        if experiment == "exp9":
            toy = SegmentedTransformer.from_preset(
                model_key, attn_granularity=Exp9Config.a3_toy().attn_granularity
            )
            return group_sizes_of(toy)
        if experiment == "exp9_a3_llm":
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(model_key, local_files_only=True)
            with torch.device("meta"):
                hf_model = AutoModelForCausalLM.from_config(config)
            return _hf_group_sizes(hf_model, Exp9Config.a3_llm().attn_granularity)
    except (ValueError, ImportError, OSError):
        log.info(
            "group_sizes_for(%r, %r): could not rebuild the model -- Phi_null n/a",
            experiment,
            model_key,
        )
        return None
    return None  # an experiment key this module does not recognise
