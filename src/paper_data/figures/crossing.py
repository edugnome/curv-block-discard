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
    for model, title in (
        ("resnet18", "(a) ResNet-18"),
        ("plain_resnet18", "(b) Plain-18"),
    ):
        rows = [cell for cell in cells if cell["model"] == model]
        labels = [
            f"{cell['seed']}/{cell['batch']} {statuses[cell['crossing']]}"
            for cell in rows
        ]
        lines = [
            f"    \\nextgroupplot[title={{{title}}}, yticklabels={{{','.join(labels)}}}]",
            "    \\addplot[black!50, line width=0.4pt, no marks, forget plot] coordinates {(1,0.5) (1,6.5)};",
            "    \\addlegendimage{only marks, mark=square*, screenstage} \\addlegendentry{stage}",
            "    \\addlegendimage{only marks, mark=*, screenlayer} \\addlegendentry{layer}",
        ]
        for y, cell in enumerate(rows, 1):
            stage = cell["partitions"]["stage"]["S"]["estimate"]
            layer = cell["partitions"]["layer"]["S"]["estimate"]
            lines.append(
                "    \\addplot+[no marks, black!50, line width=0.7pt, forget plot] "
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
                    f"{color}, mark={mark}, mark size=2.2pt, "
                    f"mark options={{fill={color}, draw={color}}}, forget plot, "
                    "error bars/.cd, x dir=both, x explicit, error mark=none, error bar style={line width=0.6pt}] coordinates "
                    f"{{({point:.12g},{y}) += ({hi - point:.12g},0) "
                    f"-= ({point - lo:.12g},0)}}; "
                    f"% {cell['id']} {partition} [{lo:.6f},{hi:.6f}]"
                )
        plots.append("\n".join(lines))
    body = "\n".join(plots)
    return rf"""\begin{{figure*}}[t]
  \centering
  \begin{{tikzpicture}}[font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, line width=0.4pt]
  \definecolor{{screenstage}}{{HTML}}{{2A78D6}}
  \definecolor{{screenlayer}}{{HTML}}{{C45638}}

    \begin{{groupplot}}[
      axis line style={{line width=0.4pt}}, tick style={{line width=0.4pt}},
      every axis plot/.append style={{line width=0.6pt}},
      error bars/error bar style={{line width=0.6pt}},
      error bars/error mark options={{line width=0.6pt}},
      group style={{group size=1 by 2, vertical sep=80pt}},
      scale only axis, width=300pt, height=100pt,
      xtick={{0.8,1,1.2}},
      xmin={xmin - pad:.8g}, xmax={xmax + pad:.8g},
      ymin=0.5, ymax=6.5, ytick={{1,2,3,4,5,6}},
      y dir=reverse,
      xlabel={{Profile ratio $S$}},
      xmajorgrids=true, ymajorgrids=false, grid style={{gray!20, line width=0.35pt}},
      tick label style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont}},
      label style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont}},
      title style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, at={{(0,1)}}, anchor=south west, yshift=26pt}},
      legend style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, at={{(0.5,1.03)}}, anchor=south, draw=none, /tikz/column 2/.style={{column sep=5pt}}, /tikz/column 4/.style={{column sep=5pt}}}},
      legend columns=2,
    ]
{body}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{At the same trained weights, changing the grouping can reverse the profile comparison.
    Each row joins stage (square) and layer (circle) estimates for one
    seed/input cell; there are six cells per model: (a) ResNet-18 and (b) Plain-18. Labels 42--44 denote
    training seeds; B0 and B1 are two fixed, disjoint 32-image CIFAR-100 input
    batches (test indices 0--31 and 32--63, respectively). The reference $S=1$
    separates mass below ($S>1$) and above ($S<1$) the PSD profile envelope.
    Bars show approximate 95\% joint conditional intervals over 36 metrics,
    not across-seed intervals. R marks a resolved stage-to-layer crossing,
    X the reverse and U an unresolved comparison. The screen and endpoint
    rules are specified in \ref{{app:crossing-check}}}}
  \label{{fig:crossing-screen}}
\end{{figure*}}"""
