"""Exp9: frozen-transformer Phi ladder.

Exp9 carries exp8's parameter-space blind-spot diagnostic onto decoder-only transformers:
with no training, for each frozen model it reports the discarded-coupling fraction

  Phi = 1 - sum_v ||H_{theta_v, theta_v}||_F^2 / ||H||_F^2  (exact nonzero Hessian: [0, 1])

- the curvature mass a block-diagonal preconditioner (K-FAC / EKFAC) throws away on a
transformer - together with the coupling field C(v, w) that locates it. Both
come from one parameter-space backward graph via `ParamBlockEstimator.estimate_phi_report`.

Attention is kept *visible*: each layer's query / key / value / output projections are
separate blocks, so the coupling the paper isolates in attention (H^T_{Q,K} != 0, exp5)
is read off directly. The question is how Phi climbs the model-size ladder
and whether pretraining (source (P)) inflates it.

Two backends share the runner (`Exp9Config`):

  - ``toy`` (the local, dependency-free ladder): `SegmentedTransformer` at growing sizes,
    measured on a random token minibatch. Self-contained - this is what the test-suite and
    the default sweep run.
  - ``hf`` (the on-demand ladder): frozen Hugging Face causal LMs (Qwen2.5-0.5B / 1.5B
    and the Llama-3.2-1B cross-family check) measured on a fixed text minibatch, loaded
    in fp32 with gradients enabled - the rungs whose full-parameter Hutchinson graph fits
    a 24 GB accelerator; ``transformers`` is imported lazily.

For each model the network is built (toy) or loaded (hf) once under a fixed build seed and
a fixed-size token minibatch is drawn; ``seeds`` then vary only the Hutchinson probes, so
Phi / the diagonal norms / the coupling field are measured once per seed and aggregated
into per-model mean/std plus the parameter count that indexes the ladder, with the raw
per-seed reports kept next to the aggregates (``"seeds"``) so shards split by seed can be
merged and re-aggregated. The bundle is JSON-serializable.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, cast

import torch
from torch import Tensor

from experiments.cli import Experiment, add_device_args, override
from experiments.config import Exp9Config
from experiments.llm import (
    CausalLMCrossEntropy,
    SegmentedTransformer,
    load_hf_segmented,
    random_token_batch,
    resolve_dtype,
    sample_token_batch,
)
from experiments.phi import measure_phi
from experiments.protocol import mean_std
from experiments.provenance import META_KEY, provenance
from experiments.utils import set_seed
from hessian import ParamGroupedModel
from hessian.memory import run_peak_allocated

logger = logging.getLogger(__name__)

# Fixed seed for model construction and the measurement batch, so a model's identity and
# its inputs are held constant while ``seeds`` vary only the Hutchinson probes.
_BUILD_SEED = 0


# ======================================================================
# Model / token-batch construction
# ======================================================================


def _build_model(
    cfg: Exp9Config,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ParamGroupedModel, Any]:
    """(model, tokenizer) for ``model_id``; tokenizer is None on the toy backend.

    Gradients stay enabled (parameter-space Phi differentiates twice w.r.t. the weights);
    ``eval`` only disables stochastic layers - neither backend trains here.
    """
    if cfg.backend == "toy":
        toy = SegmentedTransformer.from_preset(
            model_id, attn_granularity=cfg.attn_granularity
        )
        toy.to(device=device, dtype=dtype)
        toy.eval()
        return toy, None
    if cfg.backend == "hf":
        hf, tokenizer = load_hf_segmented(
            model_id, dtype=cfg.dtype, attn_granularity=cfg.attn_granularity
        )
        hf.to(device)
        hf.eval()
        return hf, tokenizer
    raise ValueError(f"unknown backend {cfg.backend!r}; expected 'toy' or 'hf'")


def _make_token_batch(
    cfg: Exp9Config,
    model: ParamGroupedModel,
    tokenizer: Any,
    device: torch.device,
) -> Tensor:
    """A fixed ``(hessian_batch_size, seq_len)`` token minibatch for the measurement.

    Toy: a deterministic block of random ids over the model's vocabulary. HF: a
    padding-free chunk of the built-in sample text tokenized by the model's tokenizer.
    """
    if cfg.backend == "toy":
        gen = torch.Generator().manual_seed(_BUILD_SEED)
        vocab = cast(SegmentedTransformer, model).vocab_size
        ids = random_token_batch(
            vocab, cfg.hessian_batch_size, cfg.seq_len, generator=gen
        )
    else:
        ids = sample_token_batch(tokenizer, cfg.hessian_batch_size, cfg.seq_len)
    return ids.to(device)


def _count_params(model: ParamGroupedModel) -> int:
    """Total parameters across the measured blocks - the ladder's size axis."""
    return sum(
        p.numel() for params in model.get_param_groups().values() for p in params
    )


# ======================================================================
# Phi measurement
# ======================================================================


def _layer_key(name: str) -> str | None:
    """The layer tag of a group name (``"L3.attn_q" -> "L3"``); None for embed / head."""
    head = name.split(".", 1)[0]
    if head.startswith("L") and head[1:].isdigit():
        return head
    return None


def _coupling_pairs(names: list[str], band: int | None) -> list[tuple[str, str]] | None:
    """Block pairs whose coupling to measure: all (None) or a depth band + intra-layer.

    ``band is None`` returns None - the estimator then measures every v-before-w pair (the
    O(L^2) toy default). An int ``band`` keeps a pair (v before w in group order) when the
    two blocks are at most ``band`` apart in that order *or* share a transformer layer, so
    every intra-layer q/k/v/o/mlp coupling (the attention hotspots) survives while the
    cross-layer field is limited to a depth band - O(L) pairs for a many-block LLM.
    """
    if band is None:
        return None
    keys = [_layer_key(n) for n in names]
    pairs: list[tuple[str, str]] = []
    for i, v in enumerate(names):
        for j in range(i + 1, len(names)):
            same_layer = keys[i] is not None and keys[i] == keys[j]
            if same_layer or (j - i) <= band:
                pairs.append((v, names[j]))
    return pairs


def _measure_phi(
    model: ParamGroupedModel,
    input_ids: Tensor,
    loss_fn: torch.nn.Module,
    n_probes: int,
    coupling_band: int | None,
) -> dict[str, Any]:
    """JSON-safe Phi report on the token minibatch (x = y = input_ids; loss shifts).

    ``coupling_band`` scopes the coupling field (None = every pair; see `_coupling_pairs`);
    Phi and the diagonal norms always cover every block. A thin wrapper around
    `experiments.phi.measure_phi`: exp9 measures through the same shared function as
    exp7/exp8, so it inherits `deterministic_mode` and the ``group_sizes`` bookkeeping
    (`experiments.phi.group_sizes_of`) for free, rather than re-deriving either here.
    """
    pairs = _coupling_pairs(list(model.get_param_groups()), coupling_band)
    return measure_phi(
        model, input_ids, input_ids, loss_fn, n_probes, coupling_pairs=pairs
    )


# ======================================================================
# Aggregation
# ======================================================================


def aggregate(
    raw: dict[str, dict[str, Any]],
    n_params: dict[str, int],
) -> dict[str, Any]:
    """Aggregates over seeds: per-model mean/std of Phi, diagonal norms, and couplings.

    Each model keeps its own block names (different depths give different keys), so models
    are summarized independently; the cross-model comparison is Phi vs ``n_params``. The
    per-seed reports are kept under ``"seeds"`` (``{"seed_42": report,...}``).
    """
    out: dict[str, Any] = {}
    for model_id, per_seed in raw.items():
        reports = list(per_seed.values())
        diag_keys = reports[0]["diag_frob"].keys()
        coupling_keys = reports[0]["coupling"].keys()
        out[model_id] = {
            "phi": mean_std([r["phi"] for r in reports]),
            "diag_frob": {
                k: mean_std([r["diag_frob"][k] for r in reports]) for k in diag_keys
            },
            "coupling": {
                k: mean_std([r["coupling"][k] for r in reports]) for k in coupling_keys
            },
            "n_params": n_params[model_id],
            "n_seeds": len(reports),
            "seeds": per_seed,
        }
    return out


# ======================================================================
# Runner
# ======================================================================


class Exp9Runner:
    """Runs the frozen-transformer Phi ladder for one `Exp9Config`."""

    def __init__(self, config: Exp9Config, device: torch.device) -> None:
        self._config = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {meta, model_id: {phi, diag_frob, coupling, n_params, n_seeds, seeds}}."""
        cfg = self._config
        dtype = resolve_dtype(cfg.dtype)
        loss_fn = CausalLMCrossEntropy()

        raw: dict[str, dict[str, Any]] = {}
        n_params: dict[str, int] = {}
        for model_id in cfg.models:
            logger.info("exp9: %s (backend=%s)", model_id, cfg.backend)
            set_seed(_BUILD_SEED)
            model, tokenizer = _build_model(cfg, model_id, self._device, dtype)
            input_ids = _make_token_batch(cfg, model, tokenizer, self._device)
            n_params[model_id] = _count_params(model)

            per_seed: dict[str, Any] = {}
            for seed in cfg.seeds:
                set_seed(seed)
                report = _measure_phi(
                    model, input_ids, loss_fn, cfg.n_probes, cfg.coupling_band
                )
                per_seed[f"seed_{seed}"] = report
                logger.info("  seed %d: phi=%.4f", seed, report["phi"])
            raw[model_id] = per_seed
            if self._device.type == "cuda":  # the ladder's budget figure per rung
                logger.info(
                    "  peak GPU memory so far: %.2f GiB allocated, %.2f GiB reserved",
                    run_peak_allocated(self._device) / 2**30,
                    torch.cuda.max_memory_reserved(self._device) / 2**30,
                )

        return {
            META_KEY: provenance(cfg, experiment="exp9"),
            **aggregate(raw, n_params),
        }


# ======================================================================
# CLI: exp9 (Phi ladder on frozen transformers)
# ======================================================================


def add_args(parser: argparse.ArgumentParser) -> None:
    add_device_args(parser)
    parser.add_argument(
        "--profile",
        type=str,
        default="a3_toy",
        choices=["a3_toy", "a3_llm"],
        help="a3_toy: local toy-decoder size ladder (default sweep, no extra deps); "
        "a3_llm: on-demand frozen Hugging Face LM ladder sized for a 24 GB accelerator "
        f"({', '.join(Exp9Config.a3_llm().models)}; fp32, needs the optional "
        "transformers dependency).",
    )
    # Optional overrides; a value of None leaves the profile default untouched.
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="toy presets (tiny/small/base) or Hugging Face model ids",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-probes", type=int, default=None)
    parser.add_argument("--hessian-batch-size", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--dtype", type=str, default=None, choices=["float32", "bfloat16"]
    )
    parser.add_argument(
        "--attn-granularity", type=str, default=None, choices=["qkv", "block"]
    )
    parser.add_argument(
        "--coupling-band",
        type=int,
        default=None,
        help="limit the coupling field to block pairs within this many blocks (plus all "
        "intra-layer pairs); omit for the profile default (full for toy, banded for the "
        "on-demand LLM ladder).",
    )


def build_config(args: argparse.Namespace) -> Exp9Config:
    """Selects the regime profile, then applies any explicit CLI overrides."""
    if args.profile == "a3_toy":
        base = Exp9Config.a3_toy()
    else:
        base = Exp9Config.a3_llm()

    return override(
        base,
        models=args.models,
        seq_len=args.seq_len,
        n_probes=args.n_probes,
        hessian_batch_size=args.hessian_batch_size,
        seeds=args.seeds,
        dtype=args.dtype,
        attn_granularity=args.attn_granularity,
        coupling_band=args.coupling_band,
    )


SPEC = Experiment(
    name="exp9",
    help="Phi ladder on frozen transformers (toy sizes / on-demand LLMs)",
    add_args=add_args,
    build_config=build_config,
    run=Exp9Runner,
    default_output_dir=lambda args: f"results/exp9_{args.profile}",
)
