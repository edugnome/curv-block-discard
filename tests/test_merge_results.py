"""Shard merging (`scripts/merge_results.py`): a split run re-merges to the unsplit bundle.

The dispatcher splits an experiment over GPUs by method and/or seed; merging must give
exactly what one unsplit run writes. A tiny exp7 run on the synthetic MLP task (as
`test_exp7_smoke_synthetic`) is split by method, by seed, and both ways, and every merge
is compared with the unsplit bundle (NaN-aware, after the JSON round trip the files take);
exp8 and exp9 are covered the same way on hand-built raw reports pushed through the
runners' aggregators. A seed present in two shards and shard constants that disagree are
errors. The synthetic and the merged bundles are shared with the paper_data tests.

Every shard also carries the ``meta`` provenance block (`experiments.provenance`), which
is not data: it is merged into one block for the merged bundle, its split axes (the seeds,
the methods / architectures / models the job was restricted to) unioned, and a shard that
disagrees on a protocol constant - batch size, probe counts, damping, widths - is refused.

exp11 is covered like exp8 / exp9 (hand-built per-seed probe records re-aggregated
through `runner_exp11._aggregate`), but its ``meta`` is not a provenance block - it is
exp11's own flat protocol record - so `merge_exp11` checks it with its own rule: every
field must agree except ``seeds``, which the merge unions instead.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.config import Exp7Config, Exp8Config, Exp9Config  # noqa: E402
from experiments.merge_results import (  # noqa: E402
    main as merge_main,
)
from experiments.merge_results import (  # noqa: E402
    merge_exp7,
    merge_exp8,
    merge_exp9,
    merge_exp11,
)
from experiments.provenance import META_KEY, data_keys, provenance  # noqa: E402
from experiments.runner_exp7 import Exp7Runner, aggregate_method  # noqa: E402
from experiments.runner_exp8 import aggregate as _aggregate_exp8  # noqa: E402
from experiments.runner_exp9 import aggregate as _aggregate_exp9  # noqa: E402
from experiments.runner_exp11 import _aggregate as _aggregate_exp11  # noqa: E402

_DEVICE = torch.device("cpu")


# ======================================================================
# Helpers shared with the paper_data tests
# ======================================================================


def nan_equal(a: Any, b: Any) -> bool:
    """Structural equality with NaN == NaN (the bundles carry NaN accept rates)."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(nan_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            nan_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return bool(a == b)


def roundtrip(bundle: Any) -> Any:
    """The bundle as read back from results.json (tuples -> lists, NaN preserved)."""
    return json.loads(json.dumps(bundle))


# The protocols the hand-built exp8 / exp9 shards below claim to have been measured
# under; each shard narrows their split axes to its own job.
_EXP8_CONFIG = replace(Exp8Config.a1_from_scratch(), n_probes=4, hessian_batch_size=8)
_EXP9_CONFIG = replace(Exp9Config.a3_toy(), models=["tiny", "small"], n_probes=4)


def _seed_numbers(seeds: Sequence[str]) -> list[int]:
    """``["seed_43", "seed_42"] -> [42, 43]``: the seed axis as a config records it."""
    return sorted(int(name.partition("_")[2]) for name in seeds)


def _exp7_meta(
    bundle: dict[str, Any], methods: Sequence[str], seeds: Sequence[str]
) -> dict[str, Any]:
    """The run's provenance narrowed to one job's methods and seeds."""
    meta = copy.deepcopy(bundle[META_KEY])
    meta["config"]["methods"] = list(methods)
    meta["config"]["training"]["seeds"] = _seed_numbers(seeds)
    return meta


def _exp8_meta(archs: Sequence[str], seeds: Sequence[str]) -> dict[str, Any]:
    meta = provenance(_EXP8_CONFIG, experiment="exp8")
    meta["config"]["archs"] = list(archs)
    meta["config"]["training"]["seeds"] = _seed_numbers(seeds)
    return meta


def _exp9_meta(models: Sequence[str], seeds: Sequence[str]) -> dict[str, Any]:
    meta = provenance(_EXP9_CONFIG, experiment="exp9")
    meta["config"]["models"] = list(models)
    meta["config"]["seeds"] = _seed_numbers(seeds)
    return meta


def _without_meta(shard: dict[str, Any]) -> dict[str, Any]:
    """A shard with no provenance block."""
    return {key: value for key, value in shard.items() if key != META_KEY}


def _mutated(shard: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """A copy of ``shard`` whose provenance config field at ``path`` is ``value``."""
    out = copy.deepcopy(shard)
    node = out[META_KEY]["config"]
    *parents, leaf = path.split(".")
    for name in parents:
        node = node[name]
    node[leaf] = value
    return out


@functools.cache
def synthetic_exp7_bundle() -> dict[str, Any]:
    """A 12-step exp7 on the synthetic MLP task: two methods, two seeds, 3 records each.

    Cached: the one runner call the merge and the paper_data tests share.
    """
    base = Exp7Config.b1_mlp()
    cfg = replace(
        base,
        methods=["kfac", "kfac+overlay"],
        measure_period=4,
        coupling_n_probes_online=2,
        coupling_n_probes=1,
        overlay_rank=1,
        overlay_n_power_iter=1,
        hessian_batch_size=8,
        training=replace(base.training, epochs=1, max_steps=12, seeds=[42, 43]),
    )
    return roundtrip(Exp7Runner(cfg, _DEVICE).run())


def exp7_shard(
    bundle: dict[str, Any], methods: list[str], seeds: list[str]
) -> dict[str, Any]:
    """What the runner writes for a job restricted to ``methods`` x ``seeds``."""
    return roundtrip(
        {
            META_KEY: _exp7_meta(bundle, methods, seeds),
            **{
                m: aggregate_method(
                    {s: bundle[m]["seeds"][s] for s in seeds}, bundle[m]["target_acc"]
                )
                for m in methods
            },
        }
    )


@functools.cache
def merged_synthetic_exp7() -> dict[str, Any]:
    """A synthetic bundle split by seed and merged back."""
    bundle = synthetic_exp7_bundle()
    methods = data_keys(bundle)
    seeds = list(bundle[methods[0]]["seeds"])
    return merge_exp7([exp7_shard(bundle, methods, [s]) for s in seeds])


def _raises_value_error(fn: Any, *args: Any) -> None:
    with pytest.raises(ValueError):
        fn(*args)


# ======================================================================
# exp7
# ======================================================================


def test_merge_exp7_split_by_method() -> None:
    bundle = synthetic_exp7_bundle()
    seeds = list(bundle["kfac"]["seeds"])
    shards = [exp7_shard(bundle, [m], seeds) for m in data_keys(bundle)]
    assert nan_equal(merge_exp7(shards), bundle), "method shards must re-merge exactly"
    assert nan_equal(merge_exp7(shards[::-1]), bundle), "shard order is irrelevant"


def test_merge_exp7_split_by_seed() -> None:
    bundle = synthetic_exp7_bundle()
    assert nan_equal(merged_synthetic_exp7(), bundle), "seed shards must re-merge"
    methods = data_keys(bundle)
    seeds = list(bundle[methods[0]]["seeds"])
    reversed_shards = [exp7_shard(bundle, methods, [s]) for s in seeds[::-1]]
    assert nan_equal(merge_exp7(reversed_shards), bundle), "seeds are re-ordered"


def test_merge_exp7_split_by_method_and_seed() -> None:
    bundle = synthetic_exp7_bundle()
    seeds = list(bundle["kfac"]["seeds"])
    shards = [
        exp7_shard(bundle, ["kfac"], seeds[:1]),
        exp7_shard(bundle, ["kfac", "kfac+overlay"], seeds[1:]),
        exp7_shard(bundle, ["kfac+overlay"], seeds[:1]),
    ]
    assert nan_equal(merge_exp7(shards), bundle)


def test_merge_exp7_rejects_duplicate_seed() -> None:
    bundle = synthetic_exp7_bundle()
    seeds = list(bundle["kfac"]["seeds"])
    whole = exp7_shard(bundle, data_keys(bundle), seeds)
    again = exp7_shard(bundle, ["kfac"], seeds[:1])
    _raises_value_error(merge_exp7, [whole, again])


def test_merge_exp7_rejects_target_mismatch() -> None:
    bundle = synthetic_exp7_bundle()
    seeds = list(bundle["kfac"]["seeds"])
    first = exp7_shard(bundle, ["kfac"], seeds[:1])
    second = exp7_shard(bundle, ["kfac"], seeds[1:])
    second["kfac"]["target_acc"] += 0.1
    _raises_value_error(merge_exp7, [first, second])


# ======================================================================
# exp8
# ======================================================================


def _exp8_raw(seeds: tuple[int, ...] = (42, 43, 44)) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for a, arch in enumerate(["resnet18", "plain_resnet18"]):
        raw[arch] = {}
        for c, ckpt in enumerate(["init", "mid", "final"]):
            raw[arch][ckpt] = {
                f"seed_{s}": {
                    "phi": 0.2 * (a + 1) + 0.01 * c + 0.001 * s,
                    "diag_frob": {"stem": 1.0 + s, "head": 2.0 + c},
                    "coupling": {"stem->head": 0.3 + 0.01 * s},
                    "test_acc": 0.5 + 0.01 * s,
                }
                for s in seeds
            }
    return raw


def _exp8_shard(
    raw: dict[str, Any], archs: list[str], seeds: list[str]
) -> dict[str, Any]:
    return roundtrip(
        {
            META_KEY: _exp8_meta(archs, seeds),
            **_aggregate_exp8(
                {
                    a: {c: {s: raw[a][c][s] for s in seeds} for c in raw[a]}
                    for a in archs
                }
            ),
        }
    )


def _exp8_unsplit(
    raw: dict[str, Any], archs: list[str] | None = None
) -> dict[str, Any]:
    """What one unsplit run over ``raw`` writes (``archs`` orders the axis it records)."""
    keys = list(raw) if archs is None else archs
    return _exp8_shard(raw, keys, list(raw[keys[0]]["init"]))


def test_merge_exp8_matches_unsplit() -> None:
    raw = _exp8_raw()
    unsplit = _exp8_unsplit(raw)
    archs = list(raw)
    seeds = list(raw[archs[0]]["init"])

    by_seed = [_exp8_shard(raw, archs, [s]) for s in seeds]
    assert nan_equal(merge_exp8(by_seed), unsplit), "seed shards"
    assert nan_equal(merge_exp8(by_seed[::-1]), unsplit), "seed shards, any order"

    # A merged split axis is the sorted union (it cannot depend on the shard order), so
    # the arch-sharded merge records ``archs`` sorted where the unsplit run declared it
    # ResNet-first; the measurements are the unsplit ones either way.
    sorted_axis = _exp8_unsplit(raw, sorted(archs))
    by_arch = [_exp8_shard(raw, [a], seeds) for a in archs]
    assert nan_equal(merge_exp8(by_arch), sorted_axis), "arch shards"

    mixed = [
        _exp8_shard(raw, ["resnet18"], seeds[:2]),
        _exp8_shard(raw, ["resnet18"], seeds[2:]),
        _exp8_shard(raw, ["plain_resnet18"], seeds),
    ]
    assert nan_equal(merge_exp8(mixed), sorted_axis), "arch x seed shards"


def test_merge_exp8_rejects_duplicate_seed() -> None:
    raw = _exp8_raw()
    whole = _exp8_shard(raw, list(raw), ["seed_42", "seed_43", "seed_44"])
    again = _exp8_shard(raw, ["plain_resnet18"], ["seed_43"])
    _raises_value_error(merge_exp8, [whole, again])


# ======================================================================
# exp9
# ======================================================================


def _exp9_raw(
    seeds: tuple[int, ...] = (42, 43, 44),
) -> tuple[dict[str, Any], dict[str, int]]:
    raw = {
        model: {
            f"seed_{s}": {
                "phi": 0.1 * (i + 1) + 0.001 * s,
                "diag_frob": {"embed": 1.0 + s, "head": 2.0},
                "coupling": {"embed->head": 0.2 + 0.01 * s},
            }
            for s in seeds
        }
        for i, model in enumerate(["tiny", "small"])
    }
    return raw, {"tiny": 1000, "small": 4000}


def _exp9_shard(
    raw: dict[str, Any], n_params: dict[str, int], models: list[str], seeds: list[str]
) -> dict[str, Any]:
    return roundtrip(
        {
            META_KEY: _exp9_meta(models, seeds),
            **_aggregate_exp9(
                {m: {s: raw[m][s] for s in seeds} for m in models},
                {m: n_params[m] for m in models},
            ),
        }
    )


def _exp9_unsplit(
    raw: dict[str, Any], n_params: dict[str, int], models: list[str] | None = None
) -> dict[str, Any]:
    """What one unsplit run over ``raw`` writes (``models`` orders the axis it records)."""
    keys = list(raw) if models is None else models
    return _exp9_shard(raw, n_params, keys, list(raw[keys[0]]))


def test_merge_exp9_matches_unsplit() -> None:
    raw, n_params = _exp9_raw()
    unsplit = _exp9_unsplit(raw, n_params)
    models = list(raw)
    seeds = list(raw[models[0]])

    by_seed = [_exp9_shard(raw, n_params, models, [s]) for s in seeds]
    assert nan_equal(merge_exp9(by_seed), unsplit), "seed shards"
    assert nan_equal(merge_exp9(by_seed[::-1]), unsplit), "seed shards, any order"

    # As for exp8: the merged model axis is the sorted union of the shards'.
    by_model = [_exp9_shard(raw, n_params, [m], seeds) for m in models]
    assert nan_equal(
        merge_exp9(by_model), _exp9_unsplit(raw, n_params, sorted(models))
    ), "model shards"


def test_merge_exp9_rejects_duplicate_seed_and_n_params_mismatch() -> None:
    raw, n_params = _exp9_raw()
    seeds = list(raw["tiny"])
    whole = _exp9_shard(raw, n_params, list(raw), seeds)
    again = _exp9_shard(raw, n_params, ["tiny"], seeds[:1])
    _raises_value_error(merge_exp9, [whole, again])

    first = _exp9_shard(raw, n_params, ["tiny"], seeds[:1])
    second = _exp9_shard(raw, n_params, ["tiny"], seeds[1:])
    second["tiny"]["n_params"] += 1
    _raises_value_error(merge_exp9, [first, second])


# ======================================================================
# exp11
# ======================================================================

# The flat protocol record `Exp11Runner._meta` writes: exp11's ``meta`` is not built by
# `experiments.provenance` (no ``config`` nesting), so the hand-built shards below stand
# it up directly rather than through `provenance()`, field order matching `_meta`'s.
_EXP11_META: dict[str, Any] = {
    "model": "mlp",
    "dataset": "synthetic",
    "pretrained": False,
    "batch_size": 256,
    "damping": 1.0,
    "lr": 1.0,
    "fisher_type": "mc",
    "kfac_refresh_period": 20,
    "probe_steps": [0, 20],
    "t_grid": [0.01, 0.1, 1.0],
    "kink_t_grid": [0.1, 1.0],
    "tolerance": 0.1,
    "n_lanczos_iter": 3,
}


def _exp11_meta(seeds: Sequence[str]) -> dict[str, Any]:
    """The run's protocol block narrowed to one job's seeds (exp11's only split axis)."""
    return {**_EXP11_META, "seeds": _seed_numbers(seeds)}


def _exp11_probe(step: int, seed: int) -> dict[str, Any]:
    """A minimal hand-built probe record (`_aggregate_tree` walks whatever it is given)."""
    return {
        "step": step,
        "t_star": 0.01 * step + 0.001 * seed,
        "decomposition": {"norm_d0": 1.0 + 0.1 * step + seed},
    }


def _exp11_raw(seeds: tuple[int, ...] = (42, 43, 44)) -> dict[str, Any]:
    return {
        f"seed_{s}": {
            "probes": [_exp11_probe(0, s), _exp11_probe(20, s)],
            "loss_trajectory": [1.0, 0.5],
            "val_acc": [0.5],
            "n_steps": 21,
        }
        for s in seeds
    }


def _exp11_shard(raw: dict[str, Any], seeds: list[str]) -> dict[str, Any]:
    chosen = {s: raw[s] for s in seeds}
    return roundtrip(
        {
            "meta": _exp11_meta(seeds),
            "seeds": chosen,
            "aggregate": _aggregate_exp11(chosen),
        }
    )


def _exp11_unsplit(raw: dict[str, Any]) -> dict[str, Any]:
    """What one unsplit run over ``raw`` writes."""
    return _exp11_shard(raw, list(raw))


def test_merge_exp11_matches_unsplit() -> None:
    raw = _exp11_raw()
    unsplit = _exp11_unsplit(raw)
    seeds = list(raw)

    by_seed = [_exp11_shard(raw, [s]) for s in seeds]
    assert nan_equal(merge_exp11(by_seed), unsplit), "seed shards"
    assert nan_equal(merge_exp11(by_seed[::-1]), unsplit), "seed shards, any order"


def test_merge_exp11_rejects_duplicate_seed() -> None:
    raw = _exp11_raw()
    whole = _exp11_shard(raw, list(raw))
    again = _exp11_shard(raw, ["seed_43"])
    _raises_value_error(merge_exp11, [whole, again])


def test_merge_exp11_rejects_a_protocol_field_mismatch() -> None:
    """A disagreement anywhere but ``seeds`` is refused, naming the field."""
    raw = _exp11_raw()
    seeds = list(raw)
    first = _exp11_shard(raw, seeds[:1])
    second = _exp11_shard(raw, seeds[1:])
    second["meta"]["damping"] = 10.0
    _raises_value_error(merge_exp11, [first, second])


def test_merge_exp11_meta_seeds_is_the_sorted_union() -> None:
    raw = _exp11_raw()
    shards = [_exp11_shard(raw, [s]) for s in list(raw)[::-1]]
    assert merge_exp11(shards)["meta"]["seeds"] == [42, 43, 44]


def test_merge_exp11_environment_key_disagreement_becomes_a_list() -> None:
    """`git_dirty` (a `SORTED_DISTINCT_FIELDS` field, not a protocol constant) merges by
    `_merge_block`'s rule - the sorted distinct values - instead of being refused.
    """
    raw = _exp11_raw()
    seeds = list(raw)
    first = _exp11_shard(raw, seeds[:1])
    second = _exp11_shard(raw, seeds[1:])
    first["meta"]["git_dirty"] = True
    second["meta"]["git_dirty"] = False
    assert merge_exp11([first, second])["meta"]["git_dirty"] == [False, True]


def test_merge_exp11_environment_keys_may_be_absent_on_older_shards() -> None:
    """Exp11 shards may omit sorted-distinct environment fields present on others."""
    raw = _exp11_raw()
    seeds = list(raw)
    old = _exp11_shard(raw, seeds[:1])  # no environment keys
    new = _exp11_shard(raw, seeds[1:])
    new["meta"]["torch"] = "2.11.0+cu130"
    new["meta"]["git_commit"] = "a" * 40
    new["meta"]["git_tree"] = "checkout"
    merged = merge_exp11([old, new])["meta"]
    assert merged["torch"] == "2.11.0+cu130", "the single shard that has it, unlisted"
    assert merged["git_commit"] == "a" * 40
    assert merged["git_tree"] == "checkout"
    assert merged["seeds"] == [42, 43, 44]


# ======================================================================
# Provenance
# ======================================================================


def test_merge_keeps_one_provenance_block_with_the_axes_unioned() -> None:
    """The merged bundle carries one block: shared protocol, re-united split axes."""
    bundle = synthetic_exp7_bundle()
    merged = merged_synthetic_exp7()
    assert list(merged)[0] == META_KEY, "provenance stays the bundle's first key"
    assert merged[META_KEY] == bundle[META_KEY], "seed shards re-unite the run's block"

    raw = _exp8_raw()
    seeds = list(raw["resnet18"]["init"])
    config = merge_exp8([_exp8_shard(raw, [a], seeds) for a in raw])[META_KEY]["config"]
    assert config["archs"] == ["plain_resnet18", "resnet18"], "sorted union of the axis"
    assert config["training"]["seeds"] == [42, 43, 44]
    assert config["n_probes"] == _EXP8_CONFIG.n_probes, "a protocol constant is shared"
    assert config["hessian_batch_size"] == _EXP8_CONFIG.hessian_batch_size


def test_merge_rejects_a_protocol_constant_mismatch() -> None:
    """Probe counts, batch sizes, widths, the learning rate: all must agree."""
    raw = _exp8_raw()
    seeds = list(raw["resnet18"]["init"])
    first = _exp8_shard(raw, ["resnet18"], seeds[:1])
    rest = _exp8_shard(raw, ["resnet18"], seeds[1:])
    for path, value in (
        ("n_probes", 100),
        ("hessian_batch_size", 64),
        ("training.batch_size", 256),
        ("training.lr", 0.5),
        ("training.epochs", 10),
        ("image_size", 224),
        ("num_classes", 10),
    ):
        _raises_value_error(merge_exp8, [first, _mutated(rest, path, value)])


def test_merge_exp7_rejects_a_damping_mismatch() -> None:
    """The same rule on the exp7 axis: one damping per run, whatever the shards."""
    bundle = synthetic_exp7_bundle()
    seeds = list(bundle["kfac"]["seeds"])
    first = exp7_shard(bundle, ["kfac"], seeds[:1])
    second = _mutated(exp7_shard(bundle, ["kfac"], seeds[1:]), "damping", 10.0)
    _raises_value_error(merge_exp7, [first, second])


def test_merge_rejects_a_foreign_experiment() -> None:
    raw = _exp8_raw()
    seeds = list(raw["resnet18"]["init"])
    first = _exp8_shard(raw, ["resnet18"], seeds[:1])
    second = _exp8_shard(raw, ["resnet18"], seeds[1:])
    second[META_KEY]["experiment"] = "exp9"
    _raises_value_error(merge_exp8, [first, second])


def test_merge_records_a_differing_environment_instead_of_refusing_it() -> None:
    """A shard rerun on another torch build (or commit) is merged, and says so."""
    raw = _exp8_raw()
    seeds = list(raw["resnet18"]["init"])
    first = _exp8_shard(raw, ["resnet18"], seeds[:1])
    second = _exp8_shard(raw, ["resnet18"], seeds[1:])
    second[META_KEY]["torch"] = "1.13.0"
    if "git_commit" in first[META_KEY]:
        second[META_KEY]["git_commit"] = "0" * 40
    merged = merge_exp8([first, second])[META_KEY]
    assert merged["torch"] == sorted(["1.13.0", first[META_KEY]["torch"]])
    if "git_commit" in first[META_KEY]:
        assert merged["git_commit"] == sorted(["0" * 40, first[META_KEY]["git_commit"]])


def test_merge_rejects_a_shard_without_provenance() -> None:
    """Reject mixed shards with and without provenance metadata."""
    raw = _exp8_raw()
    seeds = list(raw["resnet18"]["init"])
    new = _exp8_shard(raw, ["resnet18"], seeds[:1])
    old = _without_meta(_exp8_shard(raw, ["resnet18"], seeds[1:]))
    _raises_value_error(merge_exp8, [new, old])


def test_merge_of_shards_without_provenance_adds_no_block() -> None:
    """Shards without provenance merge without a provenance block."""
    raw = _exp8_raw()
    seeds = list(raw["resnet18"]["init"])
    shards = [_without_meta(_exp8_shard(raw, list(raw), [s])) for s in seeds]
    merged = merge_exp8(shards)
    assert META_KEY not in merged
    assert nan_equal(merged, _without_meta(_exp8_unsplit(raw)))


# ======================================================================
# CLI
# ======================================================================


def test_merge_cli_writes_merged_bundle() -> None:
    raw, n_params = _exp9_raw()
    seeds = list(raw["tiny"])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shard_paths = []
        for s in seeds:
            path = root / f"exp9_{s}" / "results.json"
            path.parent.mkdir()
            path.write_text(json.dumps(_exp9_shard(raw, n_params, list(raw), [s])))
            shard_paths.append(str(path))
        out = root / "merged" / "results.json"
        merge_main([str(out), *shard_paths, "--kind", "exp9"])
        assert out.is_file()
