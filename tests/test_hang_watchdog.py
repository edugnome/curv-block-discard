"""Watchdog scheduling and CLI contracts."""

from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path

import pytest

from experiments.cli import add_device_args, arm_hang_watchdog


def test_watchdog_arms_repeated_dumps(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        faulthandler, "dump_traceback_later", lambda *a, **kw: calls.append((a, kw))
    )
    with (tmp_path / "stacks.txt").open("w") as output:
        arm_hang_watchdog(0.2, file=output)
        assert calls == [((0.2,), {"repeat": True, "file": output})]


def test_watchdog_is_off_by_default(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        faulthandler, "dump_traceback_later", lambda *a, **kw: calls.append((a, kw))
    )
    arm_hang_watchdog(None)
    assert calls == []


def test_watchdog_rejects_a_nonpositive_period() -> None:
    with pytest.raises(SystemExit):
        arm_hang_watchdog(0.0)


def test_cli_flag_is_registered_with_the_device_args() -> None:
    parser = argparse.ArgumentParser()
    add_device_args(parser)
    assert parser.parse_args([]).hang_watchdog is None
    assert parser.parse_args(["--hang-watchdog", "300"]).hang_watchdog == 300.0
