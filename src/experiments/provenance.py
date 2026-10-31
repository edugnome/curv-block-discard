"""Protocol metadata, legacy record reading, and strict result-shard merging.

Run locations marked provenance=False are omitted. Protocol mismatches fail
merging; split axes are unioned, and environment differences remain explicit."""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

log = logging.getLogger(__name__)

# Reserved top-level key of every bundle: the provenance block, never data.
META_KEY: Final = "meta"

# Shard axes are unioned in sorted order, independently of input order.
SPLIT_AXIS_FIELDS: Final = frozenset({"seeds", "methods", "archs", "models", "parts"})

# Provenance fields that may legitimately differ between shards without indexing the
# measurement the way a split axis (`SPLIT_AXIS_FIELDS`) does: shards that disagree on
# one of these keep the sorted list of distinct values instead of being rejected. Two
# kinds of member share this one rule, not one: the machine/checkout facts - `torch`
# and the code-provenance keys `git_commit`, `git_dirty`, `git_tree`, `synced_at`, all
# top-level block keys - and `restore_checkpoints_dir` (an `Exp8Config` field nested
# under `config`, not top-level), which is not an environment fact at all but a path
# that legitimately differs per shard (each seed restored from its own earlier
# --output-dir) and is merged by the identical rule rather than a third pattern.
# `_merge_field` matches every name here by the field itself, at whatever nesting
# depth it occurs, so a nested key is caught exactly like a top-level one.
SORTED_DISTINCT_FIELDS: Final = frozenset(
    {
        "torch",
        "git_commit",
        "git_dirty",
        "git_tree",
        "synced_at",
        "restore_checkpoints_dir",
    }
)

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_GIT_TIMEOUT: Final = 5.0

# Check measured-code inputs, not generated outputs or datasets.
_PROVENANCE_PATHS: Final = ("src", "scripts", "pyproject.toml", "requirements.lock")

# Legacy input only; this package does not create private synchronization notes.
SYNC_SOURCE_FILE: Final = ".provenance-source"


# ======================================================================
# Building the block
# ======================================================================


def git_commit() -> str | None:
    """Return this project's HEAD, or None outside its own Git checkout."""
    try:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "--git-dir",
                str(_REPO_ROOT / ".git"),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else None


def git_dirty() -> bool | None:
    """Return whether measured-code paths differ from HEAD, or None if unknown."""
    try:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "--git-dir",
                str(_REPO_ROOT / ".git"),
                "status",
                "--porcelain",
                "--",
            ]
            + list(_PROVENANCE_PATHS),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(done.stdout.strip()) if done.returncode == 0 else None


def sync_source() -> dict[str, Any] | None:
    """Read a legacy synced-source record when present; malformed records are ignored."""
    try:
        raw = (_REPO_ROOT / SYNC_SOURCE_FILE).read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.info("%s is not valid JSON; ignoring", SYNC_SOURCE_FILE)
        return None
    if not _is_valid_sync_source(data):
        log.info("%s does not match the expected schema; ignoring", SYNC_SOURCE_FILE)
        return None
    return {
        "git_commit": data["git_commit"],
        "git_dirty": data["git_dirty"],
        "synced_at": data["synced_at"],
    }


def _is_valid_sync_source(data: Any) -> bool:
    """Validate the legacy synced-source record without accepting partial identities."""
    if not isinstance(data, dict):
        return False
    commit, dirty, synced_at = (
        data.get("git_commit"),
        data.get("git_dirty"),
        data.get("synced_at"),
    )
    return (
        isinstance(commit, str)
        and len(commit) == 40
        and set(commit) <= set("0123456789abcdef")
        and isinstance(dirty, bool)
        and isinstance(synced_at, str)
        and bool(synced_at)
    )


def code_provenance() -> dict[str, Any]:
    """Return available checkout metadata, falling back to a legacy synced-source record."""
    block: dict[str, Any] = {}
    commit = git_commit()
    if commit is not None:
        block["git_commit"] = commit
        dirty = git_dirty()
        if dirty is not None:
            block["git_dirty"] = dirty
        block["git_tree"] = "checkout"
    else:
        source = sync_source()
        if source is not None:
            block["git_commit"] = source["git_commit"]
            block["git_dirty"] = source["git_dirty"]
            block["git_tree"] = "synced"
            block["synced_at"] = source["synced_at"]
    return block


def environment_block() -> dict[str, Any]:
    """Return the Torch version and available code metadata for runner records."""
    return {"torch": str(torch.__version__), **code_provenance()}


def config_to_dict(config: Any) -> dict[str, Any]:
    """Every field of a config dataclass, recursively, as a JSON-safe mapping.

    Field declaration order is kept (so the block reads like `experiments.config`),
    nested dataclasses - `TrainingConfig`, `HessianConfig`, the `Exp7Config` inside
    `Exp11Config` - become nested mappings, and tuples become lists so the block
    survives the JSON round trip unchanged.
    """
    if not _is_dataclass_instance(config):
        raise TypeError(
            "config_to_dict() needs a config dataclass instance, got "
            f"{type(config).__name__}"
        )
    converted = _jsonify(config)
    assert isinstance(converted, dict)  # _is_dataclass_instance guarantees it
    return converted


def provenance(config: Any, *, experiment: str) -> dict[str, Any]:
    """The `META_KEY` block of a bundle that ``experiment`` produced under ``config``.

    ``{"experiment", "config"}`` plus `environment_block()` (`torch`, plus code
    provenance when it can be told - see there for exactly which keys and when).
    """
    return {
        "experiment": experiment,
        "config": config_to_dict(config),
        **environment_block(),
    }


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _in_provenance(f: dataclasses.Field[Any]) -> bool:
    """False only when ``f`` is marked ``metadata={"provenance": False}``: a run location
    (a path, a hostname) rather than a protocol constant, kept out of the block.
    """
    return f.metadata.get("provenance", True) is not False


def _jsonify(value: Any) -> Any:
    """`value` as JSON-safe data: dataclasses and mappings to dicts, sequences to lists.

    A dataclass field is dropped when `_in_provenance` says so, at every nesting level -
    the recursion calls `dataclasses.fields` (and so `_in_provenance`) afresh on every
    nested dataclass it descends into, not just the top one. A field type that is
    neither dataclass, mapping, sequence nor scalar degrades to
    its ``repr`` rather than breaking the run that was about to write the bundle.
    """
    if _is_dataclass_instance(value):
        return {
            field.name: _jsonify(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if _in_provenance(field)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


# ======================================================================
# Reading a bundle
# ======================================================================


def data_keys(bundle: Mapping[str, Any]) -> list[str]:
    """The bundle's data keys in their own order: the top level without `META_KEY`."""
    return [key for key in bundle if key != META_KEY]


def data_items(bundle: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """``bundle.items()`` without the `META_KEY` entry (the data keys' order kept)."""
    return [(key, bundle[key]) for key in data_keys(bundle)]


# ======================================================================
# Merging shards
# ======================================================================


def merge_provenance(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """One provenance block for merged shards, or None when none of them carries one.

    The shards must agree on every field except the split axes (`SPLIT_AXIS_FIELDS`,
    reported as the sorted union) and the sorted-distinct fields (`SORTED_DISTINCT_FIELDS`,
    reported as the sorted distinct values when they differ). Any other disagreement -
    batch size, probe counts, damping, widths, epochs, the dataset, the experiment
    itself - raises `ValueError`: those shards did not measure the same thing, and
    merging them would describe incompatible protocols. If all shards lack the block,
    return None; mixing shards with and without it raises `ValueError`.
    """
    blocks = [shard[META_KEY] for shard in shards if META_KEY in shard]
    if not blocks:
        return None
    if len(blocks) != len(shards):
        raise ValueError(
            f"{len(shards) - len(blocks)} of {len(shards)} shards carry no "
            f"{META_KEY!r} block: merge the shards of one run, or strip the block "
            "from all of them"
        )
    return _merge_block(blocks)


def _merge_block(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = [list(block) for block in blocks]
    if any(other != fields[0] for other in fields[1:]):
        raise ValueError(
            f"{META_KEY}: shards record different provenance fields "
            f"({fields[0]} vs {next(o for o in fields[1:] if o != fields[0])})"
        )
    return {
        key: _merge_field(
            key, [block[key] for block in blocks], path=f"{META_KEY}.{key}"
        )
        for key in fields[0]
    }


def _merge_field(field: str, values: Sequence[Any], *, path: str) -> Any:
    """The one value the shards agree on at ``path``, the sorted union of a split axis,
    or the sorted distinct values of a sorted-distinct field - see `SPLIT_AXIS_FIELDS`
    and `SORTED_DISTINCT_FIELDS`. Both sets are matched by the field's own name alone,
    so a key nested inside a mapping (`config.restore_checkpoints_dir`) gets the same
    treatment as a top-level one (`git_commit`), at whatever depth the recursion below
    finds it.
    """
    first, rest = values[0], values[1:]
    if isinstance(first, Mapping):
        if any(
            not isinstance(other, Mapping) or list(other) != list(first)
            for other in rest
        ):
            raise ValueError(f"{path}: shards record different fields here")
        return {
            key: _merge_field(
                key, [value[key] for value in values], path=f"{path}.{key}"
            )
            for key in first
        }
    if all(other == first for other in rest):
        return first
    if field in SPLIT_AXIS_FIELDS and all(isinstance(v, list) for v in values):
        return sorted({item for value in values for item in value})
    if field in SORTED_DISTINCT_FIELDS:
        return _distinct(values)
    other = next(o for o in rest if o != first)
    raise ValueError(f"{path}: differs between shards ({first!r} vs {other!r})")


def _distinct(values: Sequence[Any]) -> Any:
    """The shards' distinct values at a sorted-distinct field: the scalar, or a sorted
    list."""
    unique = sorted(
        {
            item
            for value in values
            for item in (value if isinstance(value, list) else [value])
        }
    )
    return unique[0] if len(unique) == 1 else unique
