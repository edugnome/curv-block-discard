"""Load and cache experiment results.json files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DataPaths

log = logging.getLogger(__name__)

# Module-level cache: path -> parsed dict.
_cache: dict[Path, dict[str, Any]] = {}


def load(paths: DataPaths, experiment: str) -> dict[str, Any]:
    """Load *experiment* results, caching for repeated access."""
    p = paths.get(experiment).resolve()
    if p in _cache:
        return _cache[p]
    log.info("Loading %s from %s", experiment, p)
    with open(p) as f:
        data = json.load(f)
    _cache[p] = data
    return data


def load_optional(paths: DataPaths, experiment: str) -> dict[str, Any] | None:
    """`load`, or None (logged) when the experiment's results file does not exist.

    For the on-demand experiments (`config.OPTIONAL_EXPERIMENTS`): a generator marks the
    corresponding cells instead of failing.
    """
    p = paths.get(experiment).resolve()
    if not p.exists():
        log.warning("Optional experiment %s: %s not found -- skipping", experiment, p)
        return None
    return load(paths, experiment)


def clear_cache() -> None:
    """Drop all cached data (useful in tests)."""
    _cache.clear()
