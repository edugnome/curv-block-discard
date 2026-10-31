"""Functional protection of inputs and generated build outputs."""

import pytest

from scripts import project


def _release_tree(monkeypatch, root):
    monkeypatch.setattr(project, "ROOT", root)
    for name in project.CODE_PATHS:
        path = root / name
        if name in {"src", "tests", "scripts", "data"}:
            path.mkdir()
            filename = "measurements.json" if name == "data" else "example.py"
            (path / filename).write_text("{}\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
    return root / "build" / "release"


def test_clean_requires_owned_build_directory(tmp_path):
    output = tmp_path / "unowned"
    (output / "reporting").mkdir(parents=True)
    sentinel = output / "reporting" / "keep.bin"
    sentinel.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError):
        project.clean(output)
    assert sentinel.read_bytes() == b"\x01\x02"


def test_clean_preserves_data_and_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "ROOT", tmp_path)
    build = tmp_path / "build"
    project.prepare(build)
    (build / "reporting").mkdir()
    (tmp_path / "data").mkdir()
    sentinel = tmp_path / "data" / "keep.bin"
    sentinel.write_bytes(b"\x01\x02")
    project.clean(build)
    assert not (build / "reporting").exists()
    assert sentinel.read_bytes() == b"\x01\x02"
    with pytest.raises(ValueError):
        project.clean(tmp_path / "data")


def test_release_refuses_existing_archive(monkeypatch, tmp_path):
    output = _release_tree(monkeypatch, tmp_path)
    project.release(output)
    with pytest.raises(FileExistsError):
        project.release(output)


def test_release_refuses_orphaned_checksum(monkeypatch, tmp_path):
    output = _release_tree(monkeypatch, tmp_path)
    archive = project.release(output)
    archive.unlink()
    with pytest.raises(FileExistsError):
        project.release(output)


@pytest.mark.parametrize("target_kind", ["archive", "checksum"])
def test_release_refuses_dangling_output_symlinks(monkeypatch, tmp_path, target_kind):
    output = _release_tree(monkeypatch, tmp_path)
    archive = project.release(output)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    archive.unlink()
    checksum.unlink()
    target = archive if target_kind == "archive" else checksum
    target.symlink_to(tmp_path / "missing-output")
    with pytest.raises(FileExistsError):
        project.release(output)


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".ckpt"])
def test_release_rejects_checkpoint_inputs(monkeypatch, tmp_path, suffix):
    output = _release_tree(monkeypatch, tmp_path)
    (tmp_path / "data" / f"weights{suffix}").write_bytes(b"\x01\x02")
    with pytest.raises(ValueError):
        project.release(output)


def test_clean_refuses_symlinked_generated_output(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "ROOT", tmp_path)
    build = tmp_path / "build"
    project.prepare(build)
    data = tmp_path / "data"
    data.mkdir()
    (build / "reporting").symlink_to(data, target_is_directory=True)
    with pytest.raises(ValueError):
        project.clean(build)
    assert data.is_dir()


def test_clean_does_not_adopt_foreign_generated_names(tmp_path):
    (tmp_path / "reporting").mkdir()
    with pytest.raises(ValueError):
        project.clean(tmp_path)
    assert (tmp_path / "reporting").is_dir()


def test_prepare_does_not_adopt_nonempty_foreign_directory(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError):
        project.prepare(tmp_path)
