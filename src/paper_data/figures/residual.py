"""Retrospective standalone and combined-candidate scalar contrasts."""

from __future__ import annotations

from typing import Any

from ..residual_ablation import residual_contrast_rows
from .exp789 import _proposal_scale, _proposal_ticks


def fig_residual_contrast(residual_ablation_v2: dict[str, Any]) -> str:
    """One point per available state/draw at the established unit scale."""
    rows = [
        row
        for row in residual_contrast_rows(residual_ablation_v2)
        if row["type"] == "scale" and row["alpha"] == 1
    ]
    finite = [row for row in rows if row["status"] == "finite"]
    if not finite:
        raise ValueError(
            "residual contrast figure requires finite unit-scale contrasts"
        )
    values = [v for row in finite for v in (row["x"], row["y"])]
    lo, hi = min(0.0, *values), max(0.0, *values)
    lower, upper = _proposal_scale(lo), _proposal_scale(hi)
    margin = max(0.2, 0.04 * (upper - lower))
    ticks = _proposal_ticks(values, "x") + "\n      " + _proposal_ticks(values, "y")
    colors = {42: "seedA", 43: "seedB", 44: "seedC"}
    marks = {42: "*", 43: "square*", 44: "triangle*"}
    lines = []
    for seed in (42, 43, 44):
        for step in (20, 60, 200):
            pair = [r for r in finite if r["seed"] == seed and r["step"] == step]
            if len(pair) == 2:
                points = " ".join(
                    f"({_proposal_scale(r['x']):.12g},{_proposal_scale(r['y']):.12g})"
                    for r in pair
                )
                lines.append(
                    r"    \addplot[black!50, line width=0.4pt, no marks, forget plot] coordinates {"
                    + points
                    + "};"
                )
    for row in finite:
        color = colors[row["seed"]]
        mark = marks[row["seed"]]
        fill = color if row["drawset"] == "primary" else "white"
        lines.append(
            rf"    \addplot[only marks, mark={mark}, mark size=2.6pt, {color}, "
            rf"mark options={{fill={fill}, draw={color}!75!black, line width=0.6pt}}, forget plot] "
            f"coordinates {{({_proposal_scale(row['x']):.12g},{_proposal_scale(row['y']):.12g})}}; "
            f"% seed {row['seed']} step {row['step']} {row['drawset']}"
        )
    body = "\n".join(lines)
    missing = len(rows) - len(finite)
    missing_note = (
        f". {missing} unavailable observations are not plotted" if missing else ""
    )
    return rf"""\begin{{figure}}[t]
  \centering
  \begin{{tikzpicture}}[font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, line width=0.4pt]
  \definecolor{{seedA}}{{HTML}}{{2A78D6}}
  \definecolor{{seedB}}{{HTML}}{{EB6834}}
  \definecolor{{seedC}}{{HTML}}{{1BAF7A}}
  \begin{{axis}}[
      axis line style={{line width=0.4pt}}, tick style={{line width=0.4pt}},
      every axis plot/.append style={{line width=0.6pt}},
      error bars/error bar style={{line width=0.6pt}},
      error bars/error mark options={{line width=0.6pt}},
    scale only axis, width=318pt, height=220pt,
    xmin={lower - margin:.12g}, xmax={upper + margin:.12g},
    ymin={lower - margin:.12g}, ymax={upper + margin:.12g},
    {ticks}
    xlabel={{Off alone $\Delta L_{{\rm off}}$}},
    ylabel={{Combined contrast $\Delta L_{{\rm full}}-\Delta L_{{\rm rest}}$}},
    tick label style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont}}, label style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont}},
    grid=major, grid style={{gray!20, line width=0.35pt}},
    legend style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, at={{(0.5,-0.22)}}, anchor=north, draw=none, /tikz/column 2/.style={{column sep=5pt}}, /tikz/column 4/.style={{column sep=5pt}}}},
    legend columns=3,
  ]
    \addplot[black!50, line width=0.4pt, no marks, forget plot] coordinates {{({lower:.12g},0) ({upper:.12g},0)}};
    \addplot[black!50, line width=0.4pt, no marks, forget plot] coordinates {{(0,{lower:.12g}) (0,{upper:.12g})}};
    \addplot[black!55, dashed, line width=0.4pt, no marks, forget plot] coordinates {{({lower:.12g},{lower:.12g}) ({upper:.12g},{upper:.12g})}};
    \addlegendimage{{only marks, mark=*, seedA}} \addlegendentry{{seed 42}}
    \addlegendimage{{only marks, mark=square*, seedB}} \addlegendentry{{seed 43}}
    \addlegendimage{{only marks, mark=triangle*, seedC}} \addlegendentry{{seed 44}}
    \addlegendimage{{only marks, mark=*, black}} \addlegendentry{{primary draw}}
    \addlegendimage{{only marks, mark=*, mark options={{fill=white, draw=black}}, black}} \addlegendentry{{sensitivity draw}}
{body}
  \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Standalone and combined corrections answer different loss questions.
    At unit scale, $x$ compares off alone with the base candidate and $y$
    compares full with rest, using the same correction map.
    The lower-right quadrant ($x>0$, $y<0$) means off alone harms the base
    while full beats rest; full need not beat the base.
    Each point is one state and draw; segments join paired geometry draws. Circles, squares and triangles
    identify training seeds 42, 43 and 44, respectively; filled and hollow
    marks identify the primary and sensitivity geometry draws. Checkpoints within a
    trajectory are dependent. The dashed line $y=x$ marks equal standalone
    and combined loss contrasts. A quadratic loss can also give unequal contrasts.
    Both axes use $T(z)=\operatorname{{sign}}(z)\log_{{10}}(1+|z|/10^{{-4}})$,
    with ticks in raw loss units. The comparison is retrospective{missing_note}}}
  \label{{fig:residual-contrast}}
\end{{figure}}"""
