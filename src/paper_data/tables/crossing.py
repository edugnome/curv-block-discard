"""The twelve retained cells and their approximate joint Monte Carlo screen."""

from __future__ import annotations

from typing import Any

from ..crossing import crossing_summary


def tab_crossing_check(exp8_crossing_check: dict[str, Any]) -> str:
    """Render the reviewed analysis only; marginal intervals are not family endpoints."""
    summary = crossing_summary(exp8_crossing_check)
    rows = []
    classifications = {
        "resolved_crossing": "Resolved",
        "resolved_reverse_crossing": "Reverse",
        "unresolved": "Unresolved",
    }
    for cell in summary["cells"]:
        columns = [
            "ResNet-18" if cell["model"] == "resnet18" else "Plain-18",
            str(cell["seed"]),
            cell["batch"],
        ]
        for partition, metric in (("stage", "S_stage"), ("layer", "S_layer")):
            estimate = cell["partitions"][partition]["S"]["estimate"]
            lo, hi = cell["screen_95"][metric]
            columns.extend([f"${estimate:.4f}$", rf"$[{lo:.4f},\,{hi:.4f}]$"])
        columns.append(classifications[cell["crossing"]])
        rows.append("    " + " & ".join(columns) + r" \\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Saved-state crossing check on the two fixed disjoint CIFAR-100
    test inputs (B0: indices 0--31; B1: 32--63), with all twelve cells retained.
    Endpoints are the approximate 95 per cent conditional family screen over all 36
    prespecified $S_{{\mathrm{{stage}}}}$, $S_{{\mathrm{{layer}}}}$ and $\Delta S$
    metrics, not pointwise percentile intervals or across-seed intervals.
    Resolved means unrounded stage lower endpoint $>1$ and layer upper endpoint
    $<1$; Reverse means the opposite exclusions; otherwise Unresolved.
    Positive $\Delta S$ alone does not resolve a crossing.}}
  \label{{tab:crossing-check}}
  \scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{@{{}}lrc c c c c l@{{}}}}
    \toprule
    Model & Seed & Input & $S_{{\mathrm{{stage}}}}$ & Family endpoints & $S_{{\mathrm{{layer}}}}$ & Family endpoints & Screen \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def num_crossing_check(exp8_crossing_check: dict[str, Any]) -> str:
    """Generate crossing-check macros separately from the num:b output."""
    summary = crossing_summary(exp8_crossing_check)
    cells = summary["cells"]
    narrow = next(c for c in cells if c["id"] == "s44-resnet18-B1")["screen_95"][
        "S_layer"
    ][1]
    values = {
        "CrossResnetResolved": str(summary["resolved"]["resnet18"]),
        "CrossPlainResolved": str(summary["resolved"]["plain_resnet18"]),
        "CrossCellsPerModel": str(sum(c["model"] == "resnet18" for c in cells)),
        "CrossPlainUnresolved": str(
            sum(
                c["model"] == "plain_resnet18" and c["crossing"] == "unresolved"
                for c in cells
            )
        ),
        "CrossSGoalsUnmet": str(len(summary["unmet_S"])),
        "CrossFamilyGoalsUnmet": str(len(summary["unmet_family"])),
        "CrossSHalfWidthLo": f"{summary['S_half_width_range'][0]:.4f}",
        "CrossSHalfWidthHi": f"{summary['S_half_width_range'][1]:.4f}",
        "CrossCriticalValue": f"{summary['critical_value']:.6f}",
        "CrossNarrowLayerUpper": f"{narrow:.4f}",
    }
    return "\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()
    )
