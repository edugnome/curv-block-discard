"""Merge exp7 / exp8 / exp9 / exp11 result shards and re-aggregate through the runners.

Each input shard contains a runner result for a subset of methods or seeds. Merging
unions the raw per-seed payloads and applies the corresponding runner aggregator.
The aggregators are imported from the runners - one source of truth:

  exp7   {method: {seeds: {seed_N: payload}, target_acc,...}}
         -> union of seeds per method, `runner_exp7.aggregate_method(seeds, target_acc)`
  exp8   {arch: {checkpoint: {seeds: {seed_N: report},...}}}
         -> union per (arch, checkpoint), `runner_exp8.aggregate(raw)`
  exp9   {model_id: {seeds: {seed_N: report}, n_params,...}}
         -> union per model, `runner_exp9.aggregate(raw, n_params)`
  exp11  {meta: {...protocol...}, seeds: {seed_N: payload}, aggregate: {...}}
         -> union of seeds, `runner_exp11._aggregate(seeds)`

Seeds are ordered numerically before aggregation, so the result does not depend on the
order in which the shards are listed. The same seed for the same method / arch /
checkpoint / model in two shards is an error (a rerun replaces its shard, it does not sit
next to it), and so is a shard-level constant (``target_acc``, ``n_params``) that
disagrees between shards.

The ``meta`` block of exp7 / exp8 / exp9 is `experiments.provenance` and is not data: it
is stripped before the union, merged separately by `merge_provenance` - one block for the
merged bundle, an error when the shards disagree on a protocol constant - and written
back as the merged bundle's first key. Inputs may all omit this block, but mixing
inputs with and without it is rejected. Exp11 uses a flat protocol record instead of
nested ``config`` metadata. Its ``seeds`` are unioned, sorted-distinct fields may be
missing or differ, and other fields must match.

Usage
-----
    curv-merge OUT.json SHARD.json [SHARD.json...] --kind exp7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from experiments.provenance import (  # noqa: E402
    META_KEY,
    SORTED_DISTINCT_FIELDS,
    _distinct,
    data_items,
    merge_provenance,
)
from experiments.runner_exp7 import aggregate_method  # noqa: E402
from experiments.runner_exp8 import aggregate as _aggregate_exp8  # noqa: E402
from experiments.runner_exp9 import aggregate as _aggregate_exp9  # noqa: E402
from experiments.runner_exp11 import _aggregate as _aggregate_exp11  # noqa: E402

Bundle = dict[str, Any]


# ======================================================================
# Helpers
# ======================================================================


def _seed_key(name: str) -> tuple[int, str]:
    """Sort key for ``seed_N`` keys: numeric N first, non-numeric names last."""
    _, _, number = name.partition("_")
    return (int(number), name) if number.isdigit() else (sys.maxsize, name)


def _sorted_seeds(seeds: dict[str, Any]) -> dict[str, Any]:
    return dict(sorted(seeds.items(), key=lambda item: _seed_key(item[0])))


def _add_seeds(target: dict[str, Any], seeds: dict[str, Any], where: str) -> None:
    """Union of per-seed payloads; the same seed twice is an error."""
    for seed, payload in seeds.items():
        if seed in target:
            raise ValueError(f"{where}: seed {seed!r} appears in two shards")
        target[seed] = payload


def _same_constant(
    seen: dict[str, Any], key: str, value: Any, name: str, where: str
) -> None:
    """Records the shard-level constant ``name`` and rejects a shard that disagrees."""
    if key in seen and seen[key] != value:
        raise ValueError(
            f"{where}: {name} differs between shards ({seen[key]} vs {value})"
        )
    seen[key] = value


def _with_meta(meta: Bundle | None, merged: Bundle) -> Bundle:
    """The merged data with the merged provenance block first, as the runners write it.

    When ``meta`` is None, the result has no provenance block.
    """
    return merged if meta is None else {META_KEY: meta, **merged}


# ======================================================================
# Merge functions (pure)
# ======================================================================


def merge_exp7(shards: Sequence[Bundle]) -> Bundle:
    """Union of ``seeds`` per method, re-aggregated with the shard's ``target_acc``."""
    meta = merge_provenance(shards)
    seeds_by_method: dict[str, dict[str, Any]] = {}
    target_by_method: dict[str, Any] = {}
    for shard in shards:
        for method, entry in data_items(shard):
            where = f"exp7 method {method!r}"
            _add_seeds(seeds_by_method.setdefault(method, {}), entry["seeds"], where)
            _same_constant(
                target_by_method, method, entry["target_acc"], "target_acc", where
            )
    return _with_meta(
        meta,
        {
            method: aggregate_method(_sorted_seeds(seeds), target_by_method[method])
            for method, seeds in seeds_by_method.items()
        },
    )


def merge_exp8(shards: Sequence[Bundle]) -> Bundle:
    """Union of ``seeds`` per (arch, checkpoint), re-aggregated by the exp8 runner."""
    meta = merge_provenance(shards)
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    for shard in shards:
        for arch, checkpoints in data_items(shard):
            for checkpoint, entry in checkpoints.items():
                target = raw.setdefault(arch, {}).setdefault(checkpoint, {})
                _add_seeds(target, entry["seeds"], f"exp8 {arch}/{checkpoint}")
    return _with_meta(
        meta,
        _aggregate_exp8(
            {
                arch: {
                    ckpt: _sorted_seeds(seeds) for ckpt, seeds in checkpoints.items()
                }
                for arch, checkpoints in raw.items()
            }
        ),
    )


def merge_exp9(shards: Sequence[Bundle]) -> Bundle:
    """Union of ``seeds`` per model, re-aggregated with the shard's ``n_params``."""
    meta = merge_provenance(shards)
    raw: dict[str, dict[str, Any]] = {}
    n_params: dict[str, Any] = {}
    for shard in shards:
        for model_id, entry in data_items(shard):
            where = f"exp9 model {model_id!r}"
            _add_seeds(raw.setdefault(model_id, {}), entry["seeds"], where)
            _same_constant(n_params, model_id, entry["n_params"], "n_params", where)
    return _with_meta(
        meta,
        _aggregate_exp9(
            {model_id: _sorted_seeds(seeds) for model_id, seeds in raw.items()},
            n_params,
        ),
    )


def _merge_exp11_meta(shards: Sequence[Bundle]) -> Bundle:
    """Merge exp11 metadata, requiring matching ordered protocol fields and values.

    ``seeds`` is a sorted union. Fields in ``SORTED_DISTINCT_FIELDS`` may be absent
    on some shards; present values are retained as one value or sorted distinct values.
    """
    metas = [shard["meta"] for shard in shards]
    fields = [
        [key for key in meta if key not in SORTED_DISTINCT_FIELDS] for meta in metas
    ]
    if any(other != fields[0] for other in fields[1:]):
        raise ValueError(
            "exp11 meta: shards record different protocol fields "
            f"({fields[0]} vs {next(o for o in fields[1:] if o != fields[0])})"
        )
    seen: dict[str, Any] = {}
    seed_union: set[int] = set()
    environment: dict[str, list[Any]] = {}
    for meta in metas:
        for key, value in meta.items():
            if key == "seeds":
                seed_union.update(value)
            elif key in SORTED_DISTINCT_FIELDS:
                environment.setdefault(key, []).append(value)
            else:
                _same_constant(seen, key, value, key, "exp11 meta")
    seen["seeds"] = sorted(seed_union)
    for key, values in environment.items():
        seen[key] = _distinct(values)
    return seen


def merge_exp11(shards: Sequence[Bundle]) -> Bundle:
    """Union of ``seeds``, re-aggregated by the exp11 runner; ``meta`` by
    `_merge_exp11_meta` (exp11's own protocol block, not `experiments.provenance`)."""
    seeds: dict[str, Any] = {}
    for shard in shards:
        _add_seeds(seeds, shard["seeds"], "exp11")
    seeds = _sorted_seeds(seeds)
    return {
        "meta": _merge_exp11_meta(shards),
        "seeds": seeds,
        "aggregate": _aggregate_exp11(seeds),
    }


MERGERS: dict[str, Callable[[Sequence[Bundle]], Bundle]] = {
    "exp7": merge_exp7,
    "exp8": merge_exp8,
    "exp9": merge_exp9,
    "exp11": merge_exp11,
}


# ======================================================================
# CLI
# ======================================================================


def _describe(kind: str, merged: Bundle) -> str:
    """One line per data key: how many seeds the merged entry holds (``meta`` skipped).

    exp11 is not partitioned by method / arch / model - its ``seeds`` is already the
    bundle's one data key of interest - so it gets its own one-line report.
    """
    if kind == "exp11":
        return f"  seeds: {len(merged['seeds'])}"
    lines = []
    for key, entry in data_items(merged):
        if kind == "exp8":
            counts = ", ".join(
                f"{ckpt}: {len(ck['seeds'])}" for ckpt, ck in entry.items()
            )
            lines.append(f"  {key}: {counts}")
        else:
            lines.append(f"  {key}: {len(entry['seeds'])} seeds")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge result shards (split by method and/or seed) into one "
        "results.json, re-aggregating through the experiment runners."
    )
    parser.add_argument("out", type=Path, help="merged results.json to write")
    parser.add_argument("shards", type=Path, nargs="+", help="shard results.json files")
    parser.add_argument("--kind", required=True, choices=sorted(MERGERS))
    args = parser.parse_args(argv)

    shards: list[Bundle] = []
    for path in args.shards:
        with open(path) as f:
            shards.append(json.load(f))
    merged = MERGERS[args.kind](shards)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"{args.kind}: merged {len(shards)} shards -> {args.out}")
    print(_describe(args.kind, merged))


if __name__ == "__main__":
    main()
