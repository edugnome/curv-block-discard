"""Inline-number primitive contracts; full resources are checked by reproduction."""

from copy import deepcopy

import pytest

from experiments.protocol import mean_std
from hessian import phi_null
from paper_data.formatting import exact_str
from paper_data.numbers import _merge, _phi_cells_by_weight_state


def _phi_map_bundles():
    def cell(phi=0.4, *, norms=None, sizes=None):
        norms = norms or [{"a": 1.0, "b": 1.0}] * 3
        sizes = sizes or [4, 1]
        return {
            "phi_mean": phi,
            "seeds": {
                f"seed_{seed}": {
                    "phi": phi,
                    "diag_frob": norm,
                    "group_sizes": sizes,
                }
                for seed, norm in zip((42, 43, 44), norms, strict=True)
            },
        }

    def scratch(rows):
        return {
            row: {"init": cell(0.2), "mid": cell(), "final": cell()} for row in rows
        }

    return (
        scratch(("resnet18", "plain_resnet18")),
        scratch(("resnet18_w0.5", "resnet18", "resnet18_w2.0", "resnet34")),
        scratch(("resnet18", "plain_resnet18")),
        {"resnet50": {"init": cell(0.8)}},
        {"toy": {"phi": {"mean": 0.2}, "seeds": cell()["seeds"]}},
        {"frozen": {"phi": {"mean": 0.4}, "seeds": cell()["seeds"]}},
    )


def test_cnn_mid_final_count_and_exclusions():
    counts = _phi_cells_by_weight_state(*_phi_map_bundles())
    assert counts.cnn_mid_final == 16
    assert counts.cnn_between_references == 16
    assert counts.trained == 17  # plus the synthetic frozen model
    assert counts.trained_below_sep == 17
    assert counts.random_init == 9  # eight convolutional initial states plus the toy
    assert counts.pretrained_body == 1


def test_cnn_cell_uses_seed_means_not_seed_majority_or_norm_mean():
    bundles = deepcopy(_phi_map_bundles())
    entry = bundles[2]["plain_resnet18"]["mid"]
    entry["seeds"]["seed_43"]["phi"] = 0.31  # below the size reference 0.32
    entry["phi_mean"] = (0.4 + 0.31 + 0.4) / 3
    assert _phi_cells_by_weight_state(*bundles).cnn_between_references == 16

    # Individual Phi_sep values are 0.5, about 0.0196, 0.5; their mean
    # is about 0.3399, whereas Phi_sep of the mean norms is about 0.057.
    entry["phi_mean"] = 0.25
    entry["seeds"] = {
        f"seed_{seed}": {"phi": 0.25, "diag_frob": norm, "group_sizes": [100, 1]}
        for seed, norm in zip(
            (42, 43, 44),
            ({"a": 1.0, "b": 1.0}, {"a": 100.0, "b": 1.0}, {"a": 1.0, "b": 1.0}),
            strict=True,
        )
    }
    assert _phi_cells_by_weight_state(*bundles).cnn_between_references == 16


@pytest.mark.parametrize("reference", ["lower", "upper"])
def test_cnn_equality_is_not_strictly_between(reference):
    bundles = deepcopy(_phi_map_bundles())
    entry = bundles[0]["plain_resnet18"]["mid"]
    entry["phi_mean"] = phi_null([4, 1]) if reference == "lower" else 0.5
    for payload in entry["seeds"].values():
        payload["phi"] = entry["phi_mean"]
    if reference == "upper":
        with pytest.raises(ValueError):
            _phi_cells_by_weight_state(*bundles)
    else:
        assert _phi_cells_by_weight_state(*bundles).cnn_between_references == 15


def test_cnn_unequal_seed_equality_uses_published_mean():
    bundles = deepcopy(_phi_map_bundles())
    entry = bundles[1]["resnet34"]["final"]
    sizes = [9536, 221952, 1116416, 6822400, 13114368, 51300]
    phis = [0.15, 0.41, 0.9912826382197968]
    published = mean_std(phis)
    entry["phi_mean"] = published["mean"]
    entry["phi_std"] = published["std"]
    for payload, value in zip(entry["seeds"].values(), phis, strict=True):
        payload.update(
            phi=value,
            group_sizes=sizes,
            diag_frob={str(i): 1.0 for i in range(len(sizes))},
        )
    # Python's recomputed mean can be one ULP higher than the producer's mean.
    assert entry["phi_mean"] == phi_null(sizes)
    assert _phi_cells_by_weight_state(*bundles).cnn_between_references == 15


@pytest.mark.parametrize(
    "failure",
    [
        "missing_checkpoint",
        "missing_seed",
        "nonfinite_phi",
        "nonfinite_norm",
        "missing_sizes",
    ],
)
def test_cnn_cohort_rejects_incomplete_or_nonfinite_cells(failure):
    bundles = deepcopy(_phi_map_bundles())
    entry = bundles[1]["resnet34"]["mid"]
    if failure == "missing_checkpoint":
        del bundles[1]["resnet34"]["mid"]
    elif failure == "missing_seed":
        del entry["seeds"]["seed_43"]
    elif failure == "nonfinite_phi":
        entry["phi_mean"] = float("nan")
    elif failure == "nonfinite_norm":
        entry["seeds"]["seed_43"]["diag_frob"]["a"] = float("inf")
    else:
        for payload in entry["seeds"].values():
            del payload["group_sizes"]
    with pytest.raises(ValueError):
        _phi_cells_by_weight_state(*bundles)


def test_merge_raises_on_duplicate_key() -> None:
    with pytest.raises(ValueError):
        _merge({"PhiDup": "1"}, {"Other": "2"}, {"PhiDup": "3"})


def test_merge_unions_disjoint_groups() -> None:
    assert _merge({"A": "1"}, {"B": "2"}, {"C": "3"}) == {
        "A": "1",
        "B": "2",
        "C": "3",
    }


def test_exact_str_rejects_scientific_notation() -> None:
    """A value fine enough to need scientific notation under `%g` is not silently
    printed as `1e-06` -- it is a caller mistake (this grid's own values never need
    it), so `exact_str` raises, naming the offending value."""
    with pytest.raises(ValueError):
        exact_str(1e-6)
