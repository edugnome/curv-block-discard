"""Conditional crossing from the retained, jointly screened 36-metric family."""

from __future__ import annotations

from typing import Any

from ..crossing import crossing_summary


def fig_crossing_screen(exp8_crossing_check: dict[str, Any]) -> str:
    """Show all twelve stage/layer pairs and their stored joint screen intervals."""
    summary = crossing_summary(exp8_crossing_check)
    cells = summary["cells"]
    endpoints = [
        endpoint
        for cell in cells
        for partition in ("stage", "layer")
        for endpoint in cell["screen_95"][f"S_{partition}"]
    ]
    xmin, xmax = min(1.0, *endpoints), max(1.0, *endpoints)
    pad = max(0.02, 0.04 * (xmax - xmin))
    plots = []
    statuses = {
        "resolved_crossing": "R",
        "resolved_reverse_crossing": "X",
        "unresolved": "U",
    }
    for model, title in (("resnet18", "ResNet-18"), ("plain_resnet18", "Plain-18")):
        rows = [cell for cell in cells if cell["model"] == model]
        labels = [
            f"{cell['seed']}/{cell['batch']} {statuses[cell['crossing']]}"
            for cell in rows
        ]
        lines = [
            f"    \\nextgroupplot[title={{{title}}}, yticklabels={{{','.join(labels)}}}]",
            "    \\addplot[gray!65, thin, no marks, forget plot] coordinates {(1,0.5) (1,6.5)};",
            "    \\addlegendimage{only marks, mark=square*, screenstage} \\addlegendentry{stage}",
            "    \\addlegendimage{only marks, mark=*, screenlayer} \\addlegendentry{layer}",
        ]
        for y, cell in enumerate(rows, 1):
            stage = cell["partitions"]["stage"]["S"]["estimate"]
            layer = cell["partitions"]["layer"]["S"]["estimate"]
            lines.append(
                "    \\addplot+[no marks, gray!65, line width=0.7pt, forget plot] "
                f"coordinates {{({stage:.12g},{y}) ({layer:.12g},{y})}};"
            )
            for partition, color, mark in (
                ("stage", "screenstage", "square*"),
                ("layer", "screenlayer", "*"),
            ):
                point = cell["partitions"][partition]["S"]["estimate"]
                lo, hi = cell["screen_95"][f"S_{partition}"]
                lines.append(
                    "    \\addplot+[only marks, "
                    f"{color}, mark={mark}, mark size=1.7pt, "
                    f"mark options={{fill={color}, draw={color}}}, forget plot, "
                    "error bars/.cd, x dir=both, x explicit, error mark=none] coordinates "
                    f"{{({point:.12g},{y}) += ({hi - point:.12g},0) "
                    f"-= ({point - lo:.12g},0)}}; "
                    f"% {cell['id']} {partition} [{lo:.6f},{hi:.6f}]"
                )
        plots.append("\n".join(lines))
    body = "\n".join(plots)
    return rf"""\begin{{figure*}}[t]
  \centering
  \begin{{tikzpicture}}
  \definecolor{{screenstage}}{{HTML}}{{2A78D6}}
  \definecolor{{screenlayer}}{{HTML}}{{C45638}}

    \begin{{groupplot}}[
      group style={{group size=2 by 1, horizontal sep=1.8cm}},
      width=0.43\textwidth, height=7.0cm,
      xmin={xmin - pad:.8g}, xmax={xmax + pad:.8g},
      ymin=0.5, ymax=6.5, ytick={{1,2,3,4,5,6}},
      y dir=reverse,
      xlabel={{Separability $S$}},
      xmajorgrids=true, ymajorgrids=false, grid style={{gray!20}},
      tick label style={{font=\scriptsize}},
      title style={{font=\small}},
      legend style={{font=\scriptsize, at={{(0.5,-0.22)}}, anchor=north}},
      legend columns=2,
    ]
{body}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Same-state partition crossing of the profile envelope, six seed/input cells for each model.
    Each row connects the stage (blue square) and layer (red circle) pooled-component
    $S$ estimates; horizontal bars are the approximate 95\% joint screen
    intervals, and the vertical reference is $S=1$. The screen is computed over
    all 36 prespecified metrics (including 12 paired $\Delta S$), not just the
    24 plotted $S$ values. Row codes are R: Resolved (stage lower endpoint
    $>1$ and layer upper endpoint $<1$); X: Reverse (opposite exclusions);
    U: Unresolved (all remaining outcomes, including an endpoint exactly at 1).
    These are not across-seed $t$ intervals.}}
  \label{{fig:crossing-screen}}
\end{{figure*}}"""
