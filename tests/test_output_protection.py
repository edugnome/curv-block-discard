"""Synthetic coverage for residual-ablation output path protection."""

import pytest

from experiments import causal_diagnostics as cd
from experiments import pilot
from experiments import residual_ablation as ablation


def test_disjoint_output_and_optional_root(tmp_path):
    capture = tmp_path / "input" / "capture"
    root = tmp_path / "protected"
    output = tmp_path / "output" / "result.json"
    ablation.guard_output(output, [capture])
    ablation.guard_output(output, [capture], protected_root=root)


@pytest.mark.parametrize("relation", ["same", "child", "ancestor", "symlink"])
@pytest.mark.parametrize("protect_as_root", [False, True])
def test_resolved_overlap_is_rejected(tmp_path, relation, protect_as_root):
    protected = tmp_path / "data" / "capture"
    protected.mkdir(parents=True)
    if relation == "same":
        output = protected
    elif relation == "child":
        output = protected / "output"
    elif relation == "ancestor":
        output = protected.parent
    else:
        link = tmp_path / "alias"
        link.symlink_to(protected, target_is_directory=True)
        output = link / "output"
    kwargs = {"protected_root": protected} if protect_as_root else {}
    inputs = [] if protect_as_root else [protected]
    with pytest.raises(ValueError):
        ablation.guard_output(output, inputs, **kwargs)


@pytest.mark.parametrize(
    ("phase", "extra"),
    [
        ("register", ["--capture-root", "capture"]),
        ("inventory", ["--capture-root", "capture", "--registration", "registration"]),
        ("preflight", ["--registration", "registration"]),
        (
            "freeze",
            ["--smoke-registration", "registration", "--thresholds", "thresholds"],
        ),
        ("smoke", ["--capture", "capture", "--registration", "registration"]),
        (
            "final",
            [
                "--capture",
                "capture",
                "--registration",
                "registration",
                "--thresholds",
                "thresholds",
            ],
        ),
    ],
)
def test_cli_forwards_protected_root(tmp_path, monkeypatch, phase, extra):
    registration = tmp_path / "registration.json"
    registration.write_text('{"capture_roster": []}')
    output = tmp_path / "output"
    protected = tmp_path / "protected"
    replacements = {
        "registration": str(registration),
        "capture": str(tmp_path / "capture"),
        "thresholds": str(tmp_path / "thresholds"),
    }
    extra = [replacements.get(value, value) for value in extra]
    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs.get("protected_root"))
        raise RuntimeError("stop after forwarding")

    monkeypatch.setattr(ablation, "guard_output", record)
    monkeypatch.setattr(ablation, "freeze_final_registration", record)
    monkeypatch.setattr(ablation, "measure_capture", record)
    assert (
        pilot.main(
            [
                "residual-ablation",
                phase,
                *extra,
                "--output",
                str(output),
                "--protected-root",
                str(protected),
            ]
        )
        == 1
    )
    assert seen == [protected]


def test_register_refuses_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original")
    monkeypatch.setattr(cd, "source_hashes", lambda: {"sha256": "test"})
    monkeypatch.setattr(ablation, "registration_template", lambda *args: {})
    monkeypatch.setattr(
        ablation,
        "inventory",
        lambda *args: {
            "valid_bytes": 1,
            "expected": 1,
            "entries": [],
        },
    )
    assert (
        pilot.main(
            [
                "residual-ablation",
                "register",
                "--capture-root",
                str(tmp_path / "capture"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert output.read_text() == "original"
