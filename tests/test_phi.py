"""``experiments.phi``'s shared helpers: group sizes and the coupling-pairs passthrough.

`group_sizes_of` is the architecture-only building block `measure_phi` uses to record
``group_sizes`` (Phi_null's ``{p_v}``, `hessian.phi_baselines.phi_null`) and
`experiments.group_sizes` reuses to rebuild it on a model instantiated purely for that
purpose. White-box check that it returns the partition in `get_param_groups`'s own
order (the order ``diag_frob`` keys and `phi_rank1`'s cumulative-size indexing both
rely on), and that `measure_phi` records exactly what it returns.

Also checks that `measure_phi`'s ``coupling_pairs`` reaches
`hessian.param_space.ParamBlockEstimator.estimate_phi_report` unchanged: a subset
restricts the reported ``coupling`` dict to exactly that subset (the seam exp9's
depth-banded ladder needs, `experiments.runner_exp9._coupling_pairs`), and ``None``
still covers every v-before-w pair. Numerical fidelity of Phi itself is covered in
`test_param_space` / `test_phi_baselines`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from experiments.phi import group_sizes_of, measure_phi, measure_phi_and_diag_frob


class _TwoGroupModel(nn.Module):
    """Two named parameter groups of different sizes, declared in a fixed order.

    ``first`` (a bias-free 3->2 Linear, 6 scalars) is declared before ``second`` (a
    2->1 Linear with bias, 3 scalars): `get_param_groups` returns them in that
    insertion order, so a helper that sorted by name (``first`` < ``second``, no
    change) or by size (ascending: ``second`` before ``first``) would still pass a
    weaker test - the sizes below are picked so only the declared order matches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 2, bias=False)  # 3*2 = 6
        self.second = nn.Linear(2, 1, bias=True)  # 2*1 + 1 = 3

    def forward(self, x: Tensor) -> Tensor:
        return self.second(self.first(x))

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "first": list(self.first.parameters()),
            "second": list(self.second.parameters()),
        }


class _ThreeGroupModel(nn.Module):
    """Three named parameter groups - enough pairs that a coupling_pairs subset (one
    pair) is a strictly smaller set than "every v-before-w pair" (three pairs), so the
    two `measure_phi` coupling-field tests below are actually distinguishing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 3)
        self.head = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.fc2(self.fc1(x)))

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "fc1": list(self.fc1.parameters()),
            "fc2": list(self.fc2.parameters()),
            "head": list(self.head.parameters()),
        }


def test_group_sizes_of_lists_numel_sums_in_get_param_groups_order() -> None:
    model = _TwoGroupModel()
    assert group_sizes_of(model) == [6, 3]


def test_measure_phi_records_exactly_group_sizes_of() -> None:
    model = _TwoGroupModel()
    model.eval()
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)
    out = measure_phi(model, x, y, nn.MSELoss(), n_probes=4)
    assert out["group_sizes"] == group_sizes_of(model) == [6, 3]


def test_measure_phi_with_coupling_pairs_subset_reports_only_that_pair() -> None:
    """A caller-scoped subset (exp9's depth band) must restrict ``coupling`` to it -
    and, per `estimate_phi_report`'s fallback for an incomplete pair set, still leave
    ``phi`` a valid (finite, in-range) fraction, just from a full-parameter Hutchinson
    probe instead of the exact block decomposition.
    """
    model = _ThreeGroupModel()
    model.eval()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    torch.manual_seed(0)
    out = measure_phi(
        model, x, y, nn.MSELoss(), n_probes=8, coupling_pairs=[("fc1", "head")]
    )
    assert set(out["coupling"]) == {"fc1->head"}, "coupling_pairs restricts the report"
    assert math.isfinite(out["phi"]), "phi must be finite"
    assert 0.0 <= out["phi"] <= 1.0, "phi must land in [0, 1]"


def test_measure_phi_without_coupling_pairs_covers_every_pair() -> None:
    model = _ThreeGroupModel()
    model.eval()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    out = measure_phi(model, x, y, nn.MSELoss(), n_probes=4)
    names = list(model.get_param_groups().keys())
    expected = {f"{v}->{w}" for i, v in enumerate(names) for w in names[i + 1 :]}
    assert set(out["coupling"]) == expected, "None must cover every v-before-w pair"


def test_measure_phi_and_diag_frob_carries_no_coupling_or_group_sizes() -> None:
    """The cheap sibling of `measure_phi_only`: Phi plus the per-group diagonal norms
    and nothing else - no ``coupling`` (which would cost O(groups^2) pairs) and none of
    `measure_phi`'s optional extras (``group_sizes``, ``phi_rank1``, ``lambda_max``).
    """
    model = _ThreeGroupModel()
    model.eval()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    out = measure_phi_and_diag_frob(model, x, y, nn.MSELoss(), n_probes=4)

    assert set(out) == {"phi", "diag_frob"}, "phi and the norms, nothing past them"
    assert isinstance(out["phi"], float)
    assert list(out["diag_frob"]) == list(model.get_param_groups()), (
        "diag_frob keys follow get_param_groups order (group-major)"
    )
    assert all(isinstance(v, float) for v in out["diag_frob"].values())
