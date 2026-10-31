"""Label -> generator registry: what builds each table and figure, and from what.

A label is a LaTeX `\ref`-able name. Its entry says which function builds it and
which result bundles feed that function, in the order the function's parameters
take them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .causal_pilot import causal_pilot_numbers, causal_pilot_table
from .figures.crossing import fig_crossing_screen
from .figures.exp11 import fig_validity
from .figures.exp789 import fig_eps_vs_C, fig_phi_map, fig_proposal_model_loss
from .figures.residual import fig_residual_contrast
from .numbers import num_b
from .residual_ablation import ablation_controls, ablation_numbers, ablation_table
from .residual_geometry import geometry_numbers, geometry_table
from .tables.crossing import num_crossing_check, tab_crossing_check
from .tables.diagnostics import tab_proposal_decomposition, tab_validity_resolution
from .tables.exp10 import tab_conditioning_resnet
from .tables.exp11 import tab_validity, tab_validity_arms
from .tables.exp789 import (
    tab_coupling_sensitivity,
    tab_gates,
    tab_payoff_budget,
    tab_phi_long,
    tab_phi_map,
    tab_stage_b,
    tab_two_partitions,
)


def _residual_numbers(
    residual_ablation_v2: dict[str, Any], residual_basis_geometry: dict[str, Any]
) -> str:
    return (
        ablation_numbers(residual_ablation_v2)
        + "\n"
        + geometry_numbers(residual_basis_geometry)
    )


GeneratorEntry = tuple[Callable[..., str], list[str]]
REGISTRY: dict[str, GeneratorEntry] = {
    "num:residual-ablation": (
        _residual_numbers,
        ["residual_ablation_v2", "residual_basis_geometry"],
    ),
    "tab:residual-geometry": (geometry_table, ["residual_basis_geometry"]),
    "fig:residual-contrast": (fig_residual_contrast, ["residual_ablation_v2"]),
    "tab:residual-ablation": (ablation_table, ["residual_ablation_v2"]),
    "tab:residual-controls": (ablation_controls, ["residual_ablation_v2"]),
    "num:b": (
        num_b,
        [
            "exp7",
            "exp8",
            "exp8_a1_width",
            "exp8_a1_imagenet32",
            "exp8_a2",
            "exp9",
            "exp9_a3_llm",
            "exp8_a1_v2",
            "exp8_a1_long",
            "exp11",
            "exp11_sgd",
            "exp11_adam",
            "exp8_a2_gran_v3",
            "exp8_a1_long_gran_m3000",
            "exp11_kfac_seam",
        ],
    ),
    "num:crossing-check": (num_crossing_check, ["exp8_crossing_check"]),
    "causal_pilot_numbers": (causal_pilot_numbers, ["causal_pilot"]),
    "causal_pilot_table": (causal_pilot_table, ["causal_pilot"]),
    "tab:phi-map": (
        tab_phi_map,
        [
            "exp8",
            "exp8_a1_width",
            "exp8_a1_imagenet32",
            "exp8_a2",
            "exp9",
            "exp9_a3_llm",
            "exp8_a1_v2",
        ],
    ),
    "tab:phi-long": (tab_phi_long, ["exp8_a1_long", "exp8", "exp8_a1_v2"]),
    "tab:two-partitions": (tab_two_partitions, ["exp8_a1_long_gran_m3000"]),
    "tab:crossing-check": (tab_crossing_check, ["exp8_crossing_check"]),
    "tab:stage-b": (tab_stage_b, ["exp7"]),
    "tab:proposal-decomposition": (tab_proposal_decomposition, ["exp7"]),
    "tab:coupling-sensitivity": (tab_coupling_sensitivity, ["exp7"]),
    "tab:payoff-budget": (tab_payoff_budget, ["exp7"]),
    "tab:gates": (tab_gates, ["exp7", "exp8", "exp9", "exp9_a3_llm", "exp8_a1_v2"]),
    "tab:conditioning-resnet": (tab_conditioning_resnet, ["exp10"]),
    "tab:validity": (tab_validity, ["exp11"]),
    "tab:validity-resolution": (tab_validity_resolution, ["exp11", "exp11_kfac_seam"]),
    "tab:validity-arms": (tab_validity_arms, ["exp11", "exp11_sgd", "exp11_adam"]),
    "fig:phi-map": (
        fig_phi_map,
        [
            "exp8",
            "exp8_a1_width",
            "exp8_a1_imagenet32",
            "exp8_a2",
            "exp9",
            "exp9_a3_llm",
        ],
    ),
    "fig:proposal-model-loss": (fig_proposal_model_loss, ["exp7"]),
    "fig:crossing-screen": (fig_crossing_screen, ["exp8_crossing_check"]),
    "fig:eps-vs-C": (fig_eps_vs_C, ["exp7"]),
    "fig:validity": (fig_validity, ["exp11"]),
}
