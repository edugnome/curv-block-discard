"""Focused contracts for the saved-factor geometry diagnostic."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from paper_data.residual_geometry import geometry_numbers, geometry_table

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_residual_geometry.py"
spec = importlib.util.spec_from_file_location("geometry_diagnostic", SCRIPT)
assert spec and spec.loader
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def factors(columns=None, transform=None, vectors=None):
    return {
        "dim": 2,
        "columns": columns or [{"start": 0, "stop": 2, "values": torch.eye(2)}],
        "transform": torch.eye(2) if transform is None else transform,
        "eigenvalues": torch.tensor([1.0, -2.0]),
        "eigenvectors": torch.eye(2) if vectors is None else vectors,
    }


def test_orthonormal_and_perturbed_saved_eigenfactor_map():
    result = diagnostic.analyze_subspace(factors())
    assert result["status"] == "available"
    assert result["gram_eigen_min"] == pytest.approx(1)
    assert result["map_relative_difference"] == pytest.approx(0, abs=1e-14)
    # Non-orthogonal V distinguishes the saved-factor P_s from inverse(|A_s|+I).
    v = torch.tensor([[1.0, 0.02], [0.0, 1.0]], dtype=torch.float64)
    t = torch.diag(torch.tensor([1.01, 0.99], dtype=torch.float64))
    result = diagnostic.analyze_subspace(factors(transform=t, vectors=v))
    g = t.T @ t
    gs = torch.diag(torch.sqrt(torch.diag(g)))
    gi = torch.linalg.inv(gs)
    a = v @ torch.diag(torch.tensor([1.0, -2.0], dtype=torch.float64)) @ v.T
    eig, vec = torch.linalg.eigh(gi @ a @ gi)
    other = (vec * (1 / (eig.abs() + 1))) @ vec.T
    ps = v @ torch.diag(torch.tensor([0.5, 1 / 3], dtype=torch.float64)) @ v.T
    saved = gs @ ps @ gs
    expected = torch.linalg.matrix_norm(
        saved - other, ord=2
    ) / torch.linalg.matrix_norm(saved, ord=2)
    assert result["map_relative_difference"] == pytest.approx(
        expected.item(), abs=1e-12
    )


def test_overlapping_and_rank_deficient_columns_are_not_truncated():
    cols = [
        {"start": 0, "stop": 2, "values": torch.tensor([[1.0], [0.0]])},
        {"start": 0, "stop": 2, "values": torch.tensor([[1.0], [0.0]])},
    ]
    result = diagnostic.analyze_subspace(factors(columns=cols))
    assert result["status"] == "nonpositive_gram"
    assert result["map_relative_difference"] is None
    assert result["gram_condition"] is None


def test_zero_saved_map_is_explicitly_unavailable():
    result = diagnostic.analyze_subspace(factors(vectors=torch.zeros((2, 2))))
    assert result["status"] == "undefined_map_denominator"
    assert result["map_relative_difference"] is None
    assert result["gram_condition"] == pytest.approx(1)


def test_block_overlap_matches_dense_synthetic_gram():
    cols = [
        {"start": 0, "stop": 2, "values": torch.tensor([[1.0], [2.0]])},
        {"start": 1, "stop": 2, "values": torch.tensor([[3.0]])},
    ]
    result = diagnostic.analyze_subspace(factors(columns=cols))
    dense = torch.tensor([[1.0, 0.0], [2.0, 3.0]], dtype=torch.float64)
    spectrum = torch.linalg.eigvalsh(dense.T @ dense)
    assert result["gram_eigen_min"] == pytest.approx(spectrum[0].item())
    assert result["gram_eigen_max"] == pytest.approx(spectrum[-1].item())


def test_nonfinite_and_invalid_shapes_rejected():
    sub = factors()
    sub["transform"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        diagnostic.analyze_subspace(sub)
    sub = factors()
    sub["columns"][0]["stop"] = 1
    with pytest.raises(ValueError):
        diagnostic.analyze_subspace(sub)


def test_identity_and_roster_validation():
    factor = {
        "measurement_schema": 2,
        "seed": 42,
        "step": 20,
        "drawset": "primary",
        "seeds": {"basis": 1},
        "capture_sha256": "a",
        "registration_sha256": "b",
        "current_source_sha256": "c",
    }
    result = {**factor, "filename": "factored_primary.pt", "sha256": "hash"}
    diagnostic.validate_identity(factor, result, 42, 20, "primary", "hash")
    with pytest.raises(ValueError):
        diagnostic.validate_identity(factor, result, 43, 20, "primary", "hash")
    result["sha256"] = "different"
    with pytest.raises(ValueError):
        diagnostic.validate_identity(factor, result, 42, 20, "primary", "hash")
    with pytest.raises(ValueError):
        diagnostic.validate_roster([(42, 20, "primary")])


def bundle():
    rows = []
    for seed, step, draw in diagnostic.ROSTER:
        rows.append(
            {
                "seed": seed,
                "step": step,
                "drawset": draw,
                "native_screen": {
                    "status": "resolved",
                    "max_entry_error": 1e-5,
                    "eta_action": 0.4,
                },
                "represented_factor": {
                    "status": "available",
                    "raw_columns": 30,
                    "retained_columns": 28,
                    "gram_spectral_error": 0.0002,
                    "gram_eigen_min": 0.9999,
                    "gram_eigen_max": 1.0002,
                    "gram_condition": 1.0003,
                    "map_relative_difference": 0.00025,
                },
            }
        )
    return {"rows": rows}


def test_scalar_renderer_and_validation():
    data = bundle()
    invalid_rank = copy.deepcopy(data)
    invalid_rank["rows"][0]["represented_factor"]["retained_columns"] = 31
    with pytest.raises(ValueError):
        geometry_table(invalid_rank)
    invalid_error = copy.deepcopy(data)
    invalid_error["rows"][0]["represented_factor"]["gram_spectral_error"] = float("nan")
    with pytest.raises(ValueError):
        geometry_table(invalid_error)
    bad = copy.deepcopy(data)
    bad["rows"].pop()
    with pytest.raises(ValueError):
        geometry_table(bad)
    bad = copy.deepcopy(data)
    bad["rows"][0]["represented_factor"]["gram_condition"] = math.nan
    with pytest.raises(ValueError):
        geometry_numbers(bad)
