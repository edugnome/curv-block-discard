"""Protocol metadata, JSON conversion, shard rules, and runner integration."""

from __future__ import annotations

import dataclasses
import json
import socket
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import experiments.provenance as provenance_mod
import experiments.runner_exp8 as runner_exp8
from experiments.config import (
    Exp7Config,
    Exp8Config,
    Exp9Config,
    Exp10Config,
    Exp11Config,
    HessianConfig,
    TrainingConfig,
)
from experiments.provenance import (
    META_KEY,
    SYNC_SOURCE_FILE,
    code_provenance,
    config_to_dict,
    data_items,
    data_keys,
    environment_block,
    git_commit,
    git_dirty,
    merge_provenance,
    provenance,
    sync_source,
)
from experiments.runner_exp9 import Exp9Runner
from tests.test_merge_results import synthetic_exp7_bundle

_DEVICE = torch.device("cpu")

# One instance of every experiment config, including the production profiles, so the
# recursion is exercised on the nested dataclasses (`TrainingConfig`, `HessianConfig`,
# the `Exp7Config` inside `Exp11Config`) and on the tuple field `damping_bounds`.
_CONFIGS: list[tuple[str, Any]] = [
    ("exp7", Exp7Config.b1_cars()),
    ("exp7", Exp7Config.b1_mlp()),
    ("exp8", Exp8Config.a1_from_scratch()),
    ("exp8", Exp8Config.a2_pretrained()),
    ("exp9", Exp9Config.a3_toy()),
    ("exp9", Exp9Config.a3_llm()),
    ("exp10", Exp10Config()),
    ("exp11", Exp11Config.b1_cars()),
]

# Field names that would mean a credential slipped into a config.
_SECRET_HINTS = ("token", "secret", "password", "credential", "api_key")


def _listify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listify(item) for item in value]
    return value


def _reference(config: Any) -> Any:
    """The config as data, converted independently of `experiments.provenance`.

    `dataclasses.asdict`, but skipping a field marked ``metadata={"provenance": False}``
    - written against `dataclasses.fields` directly, not by calling into the module
    under test, so the cross-check stays independent of it.
    """
    return _listify(_without_marked_fields(config))


def _without_marked_fields(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _without_marked_fields(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.metadata.get("provenance", True) is not False
        }
    if isinstance(value, (list, tuple)):
        return [_without_marked_fields(item) for item in value]
    if isinstance(value, dict):
        return {k: _without_marked_fields(v) for k, v in value.items()}
    return value


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every scalar of a nested block with the dotted path that reaches it."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


# ======================================================================
# The block
# ======================================================================


def test_provenance_records_every_config_field() -> None:
    """Field for field, recursively, for every experiment config there is."""
    for name, config in _CONFIGS:
        block = provenance(config, experiment=name)
        assert block["experiment"] == name
        assert block["config"] == _reference(config), name
        assert block["torch"] == str(torch.__version__), name


def test_provenance_tuples_become_lists_and_nesting_is_kept() -> None:
    """`damping_bounds` is a tuple and `Exp11Config.protocol` a dataclass."""
    block = provenance(Exp11Config.b1_cars(), experiment="exp11")
    protocol = block["config"]["protocol"]
    assert protocol["damping_bounds"] == [1e-3, 1e3]
    assert protocol["training"]["batch_size"] == 64
    assert protocol["training"]["seeds"] == [42], (
        "exp11 narrows the protocol to seed 42"
    )


def test_provenance_survives_the_json_round_trip() -> None:
    for name, config in _CONFIGS:
        block = provenance(config, experiment=name)
        assert json.loads(json.dumps(block)) == block, name


def _assert_leaks_nothing(block: dict[str, Any], host: str, home: str) -> None:
    for path, value in _leaves(block):
        assert not any(hint in path.lower() for hint in _SECRET_HINTS), path
        if not isinstance(value, str):
            continue
        assert not value.startswith(("/", "~", "\\")), (path, value)
        assert home not in value, (path, value)
        if len(host) >= 4:
            assert host not in value, (path, value)


def test_provenance_leaks_no_machine_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """No absolute path, no home directory, no hostname, no credential-shaped field.

    Checked on both code-provenance shapes: a real checkout, and a synced copy whose
    `git_commit` / `git_dirty` / `synced_at` come from `sync_source()` - the path that
    file was read from must not be among them either.
    """
    host = socket.gethostname()
    home = str(Path.home())
    for name, config in _CONFIGS:
        _assert_leaks_nothing(provenance(config, experiment=name), host, home)

    monkeypatch.setattr(provenance_mod, "git_commit", lambda: None)
    monkeypatch.setattr(
        provenance_mod,
        "sync_source",
        lambda: {
            "git_commit": "c" * 40,
            "git_dirty": True,
            "synced_at": "2026-09-22T00:00:00Z",
        },
    )
    synced_block = provenance(Exp8Config(), experiment="exp8")
    _assert_leaks_nothing(synced_block, host, home)
    source_path = str(provenance_mod._REPO_ROOT / SYNC_SOURCE_FILE)
    for _, value in _leaves(synced_block):
        if isinstance(value, str):
            assert source_path not in value


def test_provenance_keys_are_exactly_the_documented_ones() -> None:
    block = provenance(Exp8Config(), experiment="exp8")
    assert set(block) <= {
        "experiment",
        "config",
        "torch",
        "git_commit",
        "git_dirty",
        "git_tree",
        "synced_at",
    }
    assert {"experiment", "config", "torch"} <= set(block)


def test_config_to_dict_rejects_anything_but_a_config_instance() -> None:
    with pytest.raises(TypeError):
        config_to_dict({"width": 64})
    with pytest.raises(TypeError):
        config_to_dict(Exp8Config)  # the class, not an instance


def test_git_commit_is_a_hash_or_none() -> None:
    commit = git_commit()
    assert commit is None or (
        len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    )


def test_git_commit_survives_a_machine_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent binary, absent repository, and a hung call all give None, never an error."""

    def missing(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance_mod.subprocess, "run", missing)
    assert git_commit() is None

    def timed_out(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)

    monkeypatch.setattr(provenance_mod.subprocess, "run", timed_out)
    assert git_commit() is None

    def not_a_repository(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["git"], returncode=128, stdout="")

    monkeypatch.setattr(provenance_mod.subprocess, "run", not_a_repository)
    assert git_commit() is None


def test_git_dirty_is_a_bool_or_none() -> None:
    dirty = git_dirty()
    assert dirty is None or isinstance(dirty, bool)


def test_git_dirty_reads_the_porcelain_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-empty `git status --porcelain` output is dirty; empty output is clean."""

    def dirty(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout=" M src/experiments/provenance.py\n"
        )

    monkeypatch.setattr(provenance_mod.subprocess, "run", dirty)
    assert git_dirty() is True

    def clean(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="")

    monkeypatch.setattr(provenance_mod.subprocess, "run", clean)
    assert git_dirty() is False


def test_git_dirty_survives_a_machine_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent binary, absent repository, and a hung call all give None, never an error."""

    def missing(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance_mod.subprocess, "run", missing)
    assert git_dirty() is None

    def timed_out(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)

    monkeypatch.setattr(provenance_mod.subprocess, "run", timed_out)
    assert git_dirty() is None

    def not_a_repository(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["git"], returncode=128, stdout="")

    monkeypatch.setattr(provenance_mod.subprocess, "run", not_a_repository)
    assert git_dirty() is None


# ======================================================================
# The synced-copy fallback
# ======================================================================

_VALID_SYNC_SOURCE = {
    "git_commit": "a" * 40,
    "git_dirty": True,
    "synced_at": "2026-09-22T00:00:00Z",
}


def test_sync_source_reads_a_valid_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(provenance_mod, "_REPO_ROOT", tmp_path)
    (tmp_path / SYNC_SOURCE_FILE).write_text(json.dumps(_VALID_SYNC_SOURCE))
    assert sync_source() == _VALID_SYNC_SOURCE


def test_sync_source_is_none_when_malformed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent, not JSON, a hash that is not 40 lowercase hex, or a non-bool dirty."""
    monkeypatch.setattr(provenance_mod, "_REPO_ROOT", tmp_path)
    assert sync_source() is None, "no file at all"

    source_file = tmp_path / SYNC_SOURCE_FILE
    source_file.write_text("not json{")
    assert sync_source() is None, "not JSON"

    source_file.write_text(json.dumps({**_VALID_SYNC_SOURCE, "git_commit": "ABC123"}))
    assert sync_source() is None, "not a 40-char lowercase-hex commit"

    source_file.write_text(json.dumps({**_VALID_SYNC_SOURCE, "git_dirty": "yes"}))
    assert sync_source() is None, "git_dirty is not a bool"


# ======================================================================
# code_provenance() / environment_block()
# ======================================================================


def test_code_provenance_is_empty_when_there_is_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No checkout, no sync note: `{}`, not a block of fabricated nulls."""
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: None)
    monkeypatch.setattr(provenance_mod, "sync_source", lambda: None)
    assert code_provenance() == {}


def test_code_provenance_on_the_checkout_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: "a" * 40)
    monkeypatch.setattr(provenance_mod, "git_dirty", lambda: True)
    assert code_provenance() == {
        "git_commit": "a" * 40,
        "git_dirty": True,
        "git_tree": "checkout",
    }


def test_code_provenance_on_the_synced_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: None)
    monkeypatch.setattr(provenance_mod, "sync_source", lambda: _VALID_SYNC_SOURCE)
    assert code_provenance() == {
        "git_commit": _VALID_SYNC_SOURCE["git_commit"],
        "git_dirty": _VALID_SYNC_SOURCE["git_dirty"],
        "git_tree": "synced",
        "synced_at": _VALID_SYNC_SOURCE["synced_at"],
    }


def test_environment_block_is_torch_plus_code_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one definition of the key set: whatever `code_provenance` reports, plus
    `torch` - `provenance()` and exp11's flat meta (`runner_exp11._meta`) both build on
    this instead of each keeping its own copy of the checkout-or-synced branching.
    """
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: "b" * 40)
    monkeypatch.setattr(provenance_mod, "git_dirty", lambda: False)
    assert environment_block() == {"torch": str(torch.__version__), **code_provenance()}


def test_provenance_delegates_to_environment_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provenance()` keeps no logic of its own for the git/sync branching: it is
    exactly ``{"experiment", "config"}`` plus whatever `environment_block()` reports.
    """
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: "c" * 40)
    monkeypatch.setattr(provenance_mod, "git_dirty", lambda: True)
    config = Exp8Config()
    assert provenance(config, experiment="exp8") == {
        "experiment": "exp8",
        "config": config_to_dict(config),
        **environment_block(),
    }


# ======================================================================
# provenance(): checkout, synced copy, or neither
# ======================================================================


def test_provenance_omits_git_commit_when_there_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The keys are absent, not null: a reader must not see fabricated code provenance."""
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: None)
    monkeypatch.setattr(provenance_mod, "sync_source", lambda: None)
    block = provenance(Exp8Config(), experiment="exp8")
    assert not {"git_commit", "git_dirty", "git_tree", "synced_at"} & set(block)


def test_provenance_uses_the_checkout_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`git_tree` says "checkout"; `git_dirty` is there only when it could be told."""
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: "a" * 40)
    monkeypatch.setattr(provenance_mod, "git_dirty", lambda: True)
    block = provenance(Exp8Config(), experiment="exp8")
    assert block["git_commit"] == "a" * 40
    assert block["git_dirty"] is True
    assert block["git_tree"] == "checkout"
    assert "synced_at" not in block

    monkeypatch.setattr(provenance_mod, "git_dirty", lambda: None)
    block = provenance(Exp8Config(), experiment="exp8")
    assert block["git_tree"] == "checkout"
    assert "git_dirty" not in block


def test_provenance_falls_back_to_the_sync_source_when_there_is_no_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance_mod, "git_commit", lambda: None)
    monkeypatch.setattr(provenance_mod, "sync_source", lambda: _VALID_SYNC_SOURCE)
    block = provenance(Exp8Config(), experiment="exp8")
    assert block["git_commit"] == _VALID_SYNC_SOURCE["git_commit"]
    assert block["git_dirty"] == _VALID_SYNC_SOURCE["git_dirty"]
    assert block["git_tree"] == "synced"
    assert block["synced_at"] == _VALID_SYNC_SOURCE["synced_at"]


def test_merge_provenance_reports_differing_git_dirty_without_raising() -> None:
    """Two checkouts that agree on the commit but differ on cleanliness still merge."""
    meta = {
        "experiment": "exp8",
        "config": {"width": 4},
        "torch": str(torch.__version__),
        "git_commit": "a" * 40,
        "git_dirty": True,
        "git_tree": "checkout",
    }
    first = {META_KEY: meta}
    second = {META_KEY: {**meta, "git_dirty": False}}
    merged = merge_provenance([first, second])
    assert merged is not None
    assert merged["git_dirty"] == [False, True], "sorted distinct values, per _distinct"
    assert merged["git_tree"] == "checkout"
    assert merged["git_commit"] == "a" * 40


def test_merge_provenance_restore_checkpoints_dir_becomes_sorted_distinct() -> None:
    """`config.restore_checkpoints_dir` is nested under ``config``, not a top-level
    meta key, but is treated like `SORTED_DISTINCT_FIELDS`: a
    provenance fact that legitimately differs per shard (each seed restored from its
    own earlier --output-dir) without indexing the measurement the way a split axis
    (`SPLIT_AXIS_FIELDS`) does. `_merge_field` matches a field by name at any nesting
    depth, so three shards restored from three different directories merge to the
    sorted list of distinct directories instead of raising.
    """
    meta: dict[str, Any] = {
        "experiment": "exp8",
        "config": {"width": 4, "restore_checkpoints_dir": "results/exp8_a1_long_s42"},
        "torch": str(torch.__version__),
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_tree": "checkout",
    }
    shards = [
        {
            META_KEY: {
                **meta,
                "config": {**meta["config"], "restore_checkpoints_dir": d},
            }
        }
        for d in (
            "results/exp8_a1_long_s42",
            "results/exp8_a1_long_s43",
            "results/exp8_a1_long_s44",
        )
    ]
    merged = merge_provenance(shards)
    assert merged is not None
    assert merged["config"]["restore_checkpoints_dir"] == [
        "results/exp8_a1_long_s42",
        "results/exp8_a1_long_s43",
        "results/exp8_a1_long_s44",
    ]
    assert merged["config"]["width"] == 4, "an ordinary agreeing field is untouched"


def test_merge_provenance_still_rejects_an_uncovered_field_mismatch() -> None:
    """`width` is neither a split axis nor a sorted-distinct field: shards that disagree on
    it must still raise. The new `restore_checkpoints_dir` rule must not have widened
    `_merge_field`'s tolerance past the two documented exceptions - this refusal is
    the reason the un-mergeable bundle was caught in the first place.
    """
    meta: dict[str, Any] = {
        "experiment": "exp8",
        "config": {"width": 4, "restore_checkpoints_dir": "results/exp8_a1_long_s42"},
        "torch": str(torch.__version__),
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_tree": "checkout",
    }
    first = {META_KEY: meta}
    second = {META_KEY: {**meta, "config": {**meta["config"], "width": 8}}}
    with pytest.raises(ValueError):
        merge_provenance([first, second])


# ======================================================================
# Excluding a run location from the block
# ======================================================================


@dataclasses.dataclass
class _WithARunLocation:
    kept: int = 1
    hidden: str = dataclasses.field(
        default="/should/not/appear", metadata={"provenance": False}
    )


@dataclasses.dataclass
class _OuterWithNestedRunLocation:
    top: int = 9
    inner: _WithARunLocation = dataclasses.field(default_factory=_WithARunLocation)


def test_config_to_dict_skips_a_field_marked_provenance_false() -> None:
    """The marked field is gone; every other field of the same config is untouched."""
    assert config_to_dict(_WithARunLocation()) == {"kept": 1}


def test_config_to_dict_skips_a_marked_field_on_a_nested_dataclass() -> None:
    assert config_to_dict(_OuterWithNestedRunLocation()) == {
        "top": 9,
        "inner": {"kept": 1},
    }


def test_exp8_output_dir_never_reaches_the_block_even_when_absolute() -> None:
    """`Exp8Config.output_dir` names where a run wrote its checkpoints, not a protocol
    constant: an absolute path (a session's scratchpad, say) must not survive into a
    block that ships in a git-versioned, Zenodo-published bundle.
    """
    config = replace(
        Exp8Config.a1_from_scratch(),
        output_dir="/tmp/claude-1000/some-session/scratchpad/exp8_smoke",
    )
    block = provenance(config, experiment="exp8")
    assert "output_dir" not in block["config"]
    for path, value in _leaves(block):
        if isinstance(value, str):
            assert "/" not in value, (path, value)


def test_exp8_restore_checkpoints_dir_is_recorded_unlike_output_dir() -> None:
    """`Exp8Config.restore_checkpoints_dir` is a path too, but - unlike `output_dir`,
    which never affects a measured number - it names the weights a restored bundle was
    measured from: leaving it out would let a restored bundle be mistaken for a fresh
    training run. It stays in the block while `output_dir` stays excluded.
    """
    restored = provenance(
        replace(
            Exp8Config.a1_from_scratch(),
            restore_checkpoints_dir="results/exp8_a1_long_s42",
        ),
        experiment="exp8",
    )
    assert restored["config"]["restore_checkpoints_dir"] == "results/exp8_a1_long_s42"

    trained = provenance(Exp8Config.a1_from_scratch(), experiment="exp8")
    assert trained["config"]["restore_checkpoints_dir"] is None, (
        "a fresh training run must record the absence of a restore source, not omit "
        "the field"
    )


# ======================================================================
# The reserved key
# ======================================================================


def test_data_keys_and_items_hide_the_reserved_key() -> None:
    bundle = {META_KEY: {"experiment": "exp8"}, "8": {"a": 1}, "12": {"a": 2}}
    assert data_keys(bundle) == ["8", "12"], "data order is the bundle's own"
    assert data_items(bundle) == [("8", {"a": 1}), ("12", {"a": 2})]
    assert data_keys({"8": {}}) == ["8"], "a bundle from before the block is untouched"
    assert data_keys({META_KEY: {}}) == []


# ======================================================================
# Runner wiring
# ======================================================================

_TRAINING = TrainingConfig(epochs=1, seeds=[42], batch_size=2)
_HESSIAN = HessianConfig(
    mode="stochastic", n_probes=1, n_power_iter=1, hessian_batch_size=2
)


def _flat_loaders(num_classes: int = 10) -> tuple[DataLoader, DataLoader]:
    """Tiny CIFAR-shaped loaders, flattened, for the MLP runners (exp1-exp4)."""
    torch.manual_seed(0)
    data = TensorDataset(
        torch.randn(4, 3 * 32 * 32), torch.randint(0, num_classes, (4,))
    )
    return DataLoader(data, batch_size=2), DataLoader(data, batch_size=2)


def _image_loaders(size: int = 16) -> tuple[DataLoader, DataLoader]:
    """Tiny image loaders for the convolutional runners (exp6, exp8)."""
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(4, 3, size, size), torch.randint(0, 10, (4,)))
    return DataLoader(data, batch_size=2), DataLoader(data, batch_size=2)


def _check(
    bundle: dict[str, Any], config: Any, experiment: str, keys: list[str]
) -> None:
    """The bundle is ``{meta,...unchanged data}`` and the block is ``config``'s."""
    assert list(bundle)[0] == META_KEY, f"{experiment}: provenance comes first"
    assert data_keys(bundle) == keys, experiment
    assert bundle[META_KEY]["experiment"] == experiment
    assert bundle[META_KEY]["config"] == _reference(config), experiment
    json.dumps(bundle)  # JSON-serializable (NaN allowed as a float)


def test_exp7_bundle_carries_its_provenance() -> None:
    """The synthetic 12-step run the merge tests share (one runner call for both)."""
    bundle = synthetic_exp7_bundle()
    assert list(bundle)[0] == META_KEY
    assert data_keys(bundle) == ["kfac", "kfac+overlay"]
    meta = bundle[META_KEY]
    assert meta["experiment"] == "exp7"
    assert meta["config"]["methods"] == ["kfac", "kfac+overlay"]
    assert meta["config"]["training"]["seeds"] == [42, 43]
    assert meta["config"]["training"]["max_steps"] == 12
    assert meta["config"]["hessian_batch_size"] == 8
    assert meta["config"]["coupling_n_probes"] == 1


def test_exp8_bundle_carries_its_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_exp8, "get_cifar100_image_loaders", lambda *_a, **_k: _image_loaders()
    )
    config = Exp8Config(
        archs=["resnet18"],
        num_classes=10,
        n_probes=1,
        hessian_batch_size=2,
        checkpoint_epochs=["init"],
        training=replace(_TRAINING, epochs=0),
    )
    _check(runner_exp8.Exp8Runner(config, _DEVICE).run(), config, "exp8", ["resnet18"])


def test_exp9_bundle_carries_its_provenance() -> None:
    """The toy backend needs no weights and no data."""
    config = Exp9Config(
        models=["tiny"], seq_len=8, hessian_batch_size=2, n_probes=1, seeds=[42]
    )
    _check(Exp9Runner(config, _DEVICE).run(), config, "exp9", ["tiny"])


# ======================================================================
# The plots main.py draws from the same bundle
# ======================================================================
