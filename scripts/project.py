"""Allowlisted source packaging and cleanup of this project's generated outputs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = (
    "src",
    "tests",
    "scripts",
    "README.md",
    "REPRODUCING.md",
    "data",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "pyproject.toml",
    "requirements.lock",
    "Makefile",
    ".python-version",
    ".gitignore",
)
GENERATED_PATHS = ("reporting", "tables", "release", "mypy", "pytest", "ruff")
MARKER = ".curv-block-discard-build"


def _check_build_path(build_dir: Path) -> Path:
    build_dir = build_dir.resolve()
    if build_dir == ROOT or ROOT.is_relative_to(build_dir):
        raise ValueError("build directory must not contain the source project")
    working = ROOT / "working"
    private_build = working / "build"
    if build_dir.is_relative_to(working) and not build_dir.is_relative_to(
        private_build
    ):
        raise ValueError("private build outputs belong under working/build")
    protected = [
        (ROOT / name).resolve()
        for name in ("src", "tests", "scripts", "docs", "data", "paper", ".venv")
    ]
    if any(build_dir == path or build_dir.is_relative_to(path) for path in protected):
        raise ValueError("build directory is a source/data/environment path")
    return build_dir


def prepare(build_dir: Path) -> None:
    """Mark an empty build root before writing caches or generated resources."""
    build_dir = _check_build_path(build_dir)
    marker = build_dir / MARKER
    if marker.is_symlink():
        raise ValueError("build marker must not be a symlink")
    if marker.is_file():
        if marker.read_text(encoding="ascii").strip() != str(ROOT):
            raise ValueError("build directory belongs to a different source project")
        return
    if build_dir.exists() and any(build_dir.iterdir()):
        raise ValueError("refusing to adopt a nonempty, unmarked build directory")
    build_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(ROOT) + "\n", encoding="ascii")


def release(output: Path) -> Path:
    """Write a source-and-data archive, excluding private author materials."""
    output = output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("release output must not contain the source project")
    archive = output / "curv-block-discard-0.1.0.tar.gz"
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    for target in (archive, checksum_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace {target}")
    inputs: list[tuple[Path, str]] = []
    data_bytes = 0
    for name in CODE_PATHS:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"required code input is absent: {path}")
        for source in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if source.is_symlink():
                raise ValueError(f"symlinks are not release inputs: {source}")
            if (
                source.is_file()
                and not any(
                    part.startswith(".")
                    or part.endswith(".egg-info")
                    or part == "__pycache__"
                    for part in source.relative_to(
                        path.parent if path.is_file() else path
                    ).parts
                )
                and source.suffix not in {".pyc", ".pyo"}
            ):
                if name == "data":
                    if source.suffix not in {
                        ".json",
                        ".jsonl",
                        ".csv",
                        ".parquet",
                    } and source.name not in {"README.md", "LICENSE"}:
                        raise ValueError(f"unsupported public data input: {source}")
                    data_bytes += source.stat().st_size
                inputs.append((source, str(source.relative_to(ROOT))))
            elif source == path and source.is_file():
                inputs.append((source, name))
    if data_bytes > 100_000_000:
        raise ValueError("public data exceeds 100 MB")
    prepare(output.parent)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz", format=tarfile.PAX_FORMAT) as tar:
        for source, name in inputs:
            tar.add(source, arcname=f"curv-block-discard/{name}", recursive=False)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    with checksum_path.open("x", encoding="ascii") as stream:
        stream.write(f"{checksum}  {archive.name}\n")
    return archive


def clean(build_dir: Path) -> None:
    """Remove only named generated directories, not a caller's entire build root."""
    build_dir = _check_build_path(build_dir)
    if not build_dir.exists():
        return
    marker = build_dir / MARKER
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(
            "refusing to clean an unowned build directory without its marker"
        )
    if marker.read_text(encoding="ascii").strip() != str(ROOT):
        raise ValueError("clean requires this project's build ownership marker")
    protected = [
        (ROOT / name).resolve()
        for name in ("src", "tests", "scripts", "docs", "data", "paper", ".venv")
    ]
    for name in GENERATED_PATHS:
        target = build_dir / name
        if target.is_symlink():
            raise ValueError(f"refusing to clean symlink: {target}")
        resolved = target.resolve()
        if any(
            resolved == path
            or path.is_relative_to(resolved)
            or resolved.is_relative_to(path)
            for path in protected
        ):
            raise ValueError(
                f"refusing to clean source/data/environment path: {target}"
            )
    for name in GENERATED_PATHS:
        target = build_dir / name
        if target.is_dir():
            shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("release", "clean", "prepare"))
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if args.action == "release":
        prepare(args.build_dir)
        print(release(args.build_dir / "release"))
        print("Source and bundled data; private author materials are excluded.")
    elif args.action == "clean":
        clean(args.build_dir)
    else:
        prepare(args.build_dir)


if __name__ == "__main__":
    main()
