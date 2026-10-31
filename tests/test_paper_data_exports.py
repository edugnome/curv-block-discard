"""Readable exports retain raw identity, precision and undefined outcomes."""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path

import pytest

from paper_data.exports import (
    crossing_pairs,
    encode_nonfinite,
    export_tables,
    seed_rows,
    validity_rows,
)


def test_seed_rows_keep_model_checkpoint_and_exact_payload():
    value = 0.12345678901234567
    payload = {"phi": value, "test_acc": None, "diag_frob": {"a": 2.0}}
    rows = list(seed_rows("exp8", {"net": {"final": {"seeds": {"seed_42": payload}}}}))
    assert rows == [
        {"bundle": "exp8", "cell": ["net", "final"], "seed": 42, "observation": payload}
    ]
    assert json.loads(json.dumps(rows))[0]["observation"]["phi"] == value


def test_nonfinite_encoding_is_explicit_strict_json_and_distinct_from_missing():
    actual = encode_nonfinite({"values": [math.nan, math.inf, -math.inf, None, 1.25]})
    assert json.loads(json.dumps(actual, allow_nan=False)) == {
        "values": [
            {"nonfinite": "NaN"},
            {"nonfinite": "+Infinity"},
            {"nonfinite": "-Infinity"},
            None,
            1.25,
        ]
    }


def test_validity_repeat_keeps_cohort_and_step_without_inventing_seed():
    data = {"seeds": {"seed_42": {"probes": [{"step": 1500, "t_star": math.nan}]}}}
    (row,) = validity_rows("exp11_kfac_seam", data)
    assert (row["bundle"], row["seed"], row["step"], row["direction"]) == (
        "exp11_kfac_seam",
        42,
        1500,
        "kfac",
    )
    assert math.isnan(row["observation"]["t_star"])
    data["seeds"]["seed_42"]["probes"].append({"step": 1500})
    with pytest.raises(ValueError):
        list(validity_rows("exp11_kfac_seam", data))


def test_seed_rows_reject_ambiguous_seed_labels():
    with pytest.raises(ValueError):
        list(seed_rows("exp8", {"seeds": {"42-repeat": {}}}))


def write_pairs(path: Path, *, duplicate: bool = False, missing: bool = False):
    with tarfile.open(path, "w:gz") as archive:
        for seed in (42, 43, 44):
            for model in ("resnet18", "plain_resnet18"):
                for batch in ("B0", "B1"):
                    cell = f"s{seed}-{model}-{batch}"
                    for index in range(30):
                        if (
                            missing
                            and seed == 44
                            and model == "plain_resnet18"
                            and batch == "B1"
                            and index == 29
                        ):
                            continue
                        value = {
                            "cell": cell,
                            "replicate": index,
                            "stage": {"diag_sq": [1.2], "full_sq": 2.3},
                            "layer": {"diag_sq": [0.8], "full_sq": 1.9},
                        }
                        blob = json.dumps(value).encode()
                        member = tarfile.TarInfo(
                            f"crossing/{cell}/pair-{index:02d}.json"
                        )
                        member.size = len(blob)
                        archive.addfile(member, io.BytesIO(blob))
                        if (
                            duplicate
                            and seed == 42
                            and model == "resnet18"
                            and batch == "B0"
                            and index == 0
                        ):
                            archive.addfile(member, io.BytesIO(blob))
        blob = b"not executed or extracted"
        member = tarfile.TarInfo("../../planted.py")
        member.size = len(blob)
        archive.addfile(member, io.BytesIO(blob))


def test_crossing_pairs_retain_all_observations_without_extracting(tmp_path):
    path = tmp_path / "records.tar.gz"
    write_pairs(path)
    rows = list(crossing_pairs(path))
    assert len(rows) == 360
    assert (rows[0]["seed"], rows[0]["model"], rows[0]["batch"], rows[0]["index"]) == (
        42,
        "plain_resnet18",
        "B0",
        0,
    )
    assert rows[0]["observation"]["stage"]["full_sq"] == 2.3
    assert not (tmp_path / "crossing").exists()


@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_crossing_pairs_fail_on_duplicate_or_incomplete_family(tmp_path, mode):
    path = tmp_path / "records.tar.gz"
    write_pairs(path, **{mode: True})
    with pytest.raises(ValueError):
        list(crossing_pairs(path))


def test_export_missing_input_does_not_create_output(tmp_path):
    with pytest.raises(ValueError):
        export_tables(tmp_path / "absent", tmp_path / "tables")
    assert not (tmp_path / "tables").exists()


def test_flat_derived_export_is_separate_and_retains_all_pairs(tmp_path):
    from paper_data.exports import export_proposal_rows
    from tests.test_paper_data_diagnostics import _full_proposals

    root = tmp_path / "source"
    path = root / "reporting" / "exp7" / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_full_proposals()))
    output = tmp_path / "derived"
    assert export_proposal_rows(root, output) == {"proposal-decomposition": 900}


def test_residual_contrast_export_is_separate_and_requires_new_directory(
    monkeypatch, tmp_path
):
    from paper_data import exports

    sample = [{"type": "scale", "alpha": 1}, {"type": "matched_full_rest", "y": -1}]
    monkeypatch.setattr(exports, "load", lambda *_: {})
    monkeypatch.setattr(exports, "residual_contrast_rows", lambda _: iter(sample))
    destination = tmp_path / "out"
    assert exports.export_residual_contrasts(tmp_path, destination) == {
        "scale": 1,
        "matched_full_rest": 1,
    }
    with pytest.raises(FileExistsError):
        exports.export_residual_contrasts(tmp_path, destination)
