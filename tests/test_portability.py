"""Portable reporting and CLI behavior without measured inputs."""

from pathlib import Path

import pytest

from paper_data.config import DEFAULT_RESULTS_DIR, DataPaths
from paper_data.main import generate_all


def test_reporting_default_is_relative_to_callers_data_directory():
    assert DEFAULT_RESULTS_DIR == Path("data/reporting")


def test_missing_inputs_leave_no_partial_output(tmp_path):
    output = tmp_path / "generated"
    with pytest.raises((FileNotFoundError, ValueError)):
        generate_all(DataPaths(tmp_path / "missing"), output)
    assert not output.exists()


def test_unknown_label_is_an_error_not_silent_success(tmp_path):
    with pytest.raises(ValueError):
        generate_all(DataPaths(tmp_path), tmp_path / "out", only=["tab:typo"])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "module",
    [
        "experiments.main",
        "experiments.pilot",
        "paper_data.main",
        "experiments.crossing.measure",
        "experiments.crossing.verify",
    ],
)
def test_installed_cli_help_is_offline_outside_project(tmp_path, module):
    import os
    import subprocess
    import sys

    command = (
        "import runpy, sys; "
        "sys.addaudithook(lambda event, args: (_ for _ in ()).throw(AssertionError('network access')) "
        "if event == 'socket.connect' else None); "
        f"sys.argv = [{module!r}, '--help']; runpy.run_module({module!r}, run_name='__main__')"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
