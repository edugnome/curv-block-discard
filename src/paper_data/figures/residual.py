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
    marks = {"primary": "*", "sensitivity": "square*"}
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
                    r"    \addplot[gray!55, thin, no marks, forget plot] coordinates {"
                    + points
                    + "};"
                )
    for row in finite:
        color = colors[row["seed"]]
        mark = marks[row["drawset"]]
        lines.append(
            rf"    \addplot[only marks, mark={mark}, mark size=2.3pt, {color}, "
            rf"mark options={{fill={color}, draw=white, line width=0.3pt}}, forget plot] "
            f"coordinates {{({_proposal_scale(row['x']):.12g},{_proposal_scale(row['y']):.12g})}}; "
            f"% seed {row['seed']} step {row['step']} {row['drawset']}"
        )
    body = "\n".join(lines)
    missing = len(rows) - len(finite)
    missing_note = (
        f" {missing} unavailable observations are not plotted." if missing else ""
    )
    return rf"""\begin{{figure}}[t]
  \centering
  \begin{{tikzpicture}}
  \definecolor{{seedA}}{{HTML}}{{2A78D6}}
  \definecolor{{seedB}}{{HTML}}{{EB6834}}
  \definecolor{{seedC}}{{HTML}}{{1BAF7A}}
  \begin{{axis}}[
    width=0.94\linewidth, height=6.5cm,
    xmin={lower - margin:.12g}, xmax={upper + margin:.12g},
    ymin={lower - margin:.12g}, ymax={upper + margin:.12g},
    {ticks}
    xlabel={{Off alone $\Delta L_{{\rm off}}$}},
    ylabel={{Combined contrast $\Delta L_{{\rm full}}-\Delta L_{{\rm rest}}$}},
    tick label style={{font=\scriptsize}}, label style={{font=\small}},
    grid=major, grid style={{gray!15}},
    legend style={{font=\scriptsize, at={{(0.5,-0.24)}}, anchor=north, draw=none}},
    legend columns=3,
  ]
    \addplot[black!50, thin, no marks, forget plot] coordinates {{({lower:.12g},0) ({upper:.12g},0)}};
    \addplot[black!50, thin, no marks, forget plot] coordinates {{(0,{lower:.12g}) (0,{upper:.12g})}};
    \addplot[black!55, dashed, thin, no marks, forget plot] coordinates {{({lower:.12g},{lower:.12g}) ({upper:.12g},{upper:.12g})}};
    \addlegendimage{{only marks, mark=*, seedA}} \addlegendentry{{seed 42}}
    \addlegendimage{{only marks, mark=*, seedB}} \addlegendentry{{seed 43}}
    \addlegendimage{{only marks, mark=*, seedC}} \addlegendentry{{seed 44}}
    \addlegendimage{{only marks, mark=*, black}} \addlegendentry{{primary}}
    \addlegendimage{{only marks, mark=square*, black}} \addlegendentry{{sensitivity}}
{body}
  \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Retrospective unit-scale standalone and combined-candidate contrasts.
    Each point is one state and draw; thin segments join the two draws of the
    same seed and step. The lower-right quadrant ($x>0$, $y<0$) means off
    alone harms the base while full has lower loss than rest; this does not imply that full improves on the base.
    The dashed line is $y=x$, where the loss interaction $J=0$; zero axes are solid.
    Both axes use $T(z)=\operatorname{{sign}}(z)\log_{{10}}(1+|z|/10^{{-4}})$,
    with ticks in raw loss units. Colours identify training seeds and shapes
    identify draws. Checkpoints within a training trajectory are dependent, and
    the two draws at each checkpoint are paired.{missing_note}}}
  \label{{fig:residual-contrast}}
\end{{figure}}"""
