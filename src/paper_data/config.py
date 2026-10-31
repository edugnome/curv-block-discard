"""Data path configuration and experiment registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Default input root is relative to the invoking working directory.
DEFAULT_RESULTS_DIR: Path = Path("data/reporting")

# Bundle subpaths below the explicit results directory.
EXPERIMENT_FILES: dict[str, str] = {
    "exp7": "exp7/results.json",
    "exp8": "exp8/results.json",
    "exp8_a1_v2": "exp8_a1_v2/results.json",
    "exp8_a1_long": "exp8_a1_long/results.json",
    "exp8_a1_width": "exp8_a1_width/results.json",
    "exp8_a1_imagenet32": "exp8_a1_imagenet32/results.json",
    "exp8_a2": "exp8_a2/results.json",
    "exp8_a2_gran_v3": "exp8_a2_gran_v3/results.json",
    "exp8_a1_long_gran_m3000": "exp8_a1_long_gran_m3000/results.json",
    "exp9": "exp9/results.json",
    "exp9_a3_llm": "exp9_a3_llm/results.json",
    "exp10": "exp10/results.json",
    "exp11": "exp11/results.json",
    "exp11_sgd": "exp11_sgd/results.json",
    "exp11_adam": "exp11_adam/results.json",
    "exp11_kfac_seam": "exp11_kfac_seam/results.json",
    "causal_pilot": "causal_pilot/results.json",
    "exp8_crossing_check": "exp8_crossing_check/results.json",
    "residual_ablation_v2": "residual_ablation_v2/results.json",
    "residual_basis_geometry": "residual_basis_geometry/results.json",
}

# Individual renderers can accept absent optional bundles as None and display `---`.
# Other missing inputs fail; complete tracks enforce their cohort preflight.
#
# exp11_sgd / exp11_adam (the SGD/Adam arms of tab:validity-arms) are deliberately
# left out of this set even though exp11 (their K-FAC arm) is in it: tab_validity_arms
# has no placeholder form, so a missing arm bundle should fail loudly at `load()`
# (FileNotFoundError) rather than reach the generator as None. exp11 itself stays
# optional for tab:validity's own on-demand rendering; tab_validity_arms handles the
# resulting possibility of a None `exp11` itself (see its docstring).
OPTIONAL_EXPERIMENTS: frozenset[str] = frozenset(
    {
        "exp8_a1_v2",
        "exp8_a1_long",
        "exp8_a1_width",
        "exp8_a1_imagenet32",
        "exp8_a2",
        "exp8_a2_gran_v3",
        "exp8_a1_long_gran_m3000",
        "exp9_a3_llm",
        "exp10",
        "exp11",
        "exp11_kfac_seam",
    }
)


@dataclass
class DataPaths:
    """Resolve experiment data file paths.

    Parameters
    ----------
    results_dir: Path
        Root directory containing per-experiment subdirectories.
    overrides: dict[str, Path]
        Override specific experiment paths (e.g. when data is scattered).
    """

    results_dir: Path
    overrides: dict[str, Path] = field(default_factory=dict)

    def get(self, experiment: str) -> Path:
        """Return the resolved path for *experiment*."""
        if experiment in self.overrides:
            return self.overrides[experiment]
        if experiment not in EXPERIMENT_FILES:
            raise KeyError(f"Unknown experiment {experiment!r}")
        return self.results_dir / EXPERIMENT_FILES[experiment]
