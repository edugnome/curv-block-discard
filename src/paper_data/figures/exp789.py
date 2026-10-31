"""TikZ figures for Experiments 7-9.

Generated figures
-----------------
- fig:phi-map  - Phi by run: (a) the convolutional runs at init / final (bars, +-1 sigma),
                 (b) the transformer ladders against the parameter count (log x, one
                 tick per decade labelled 100k / 1M / 10M /... / 1B)
- fig:eps-vs-C - the cross-term ratio eps(v, w) (log axis) against the coupling C(v, w)
                 (linear axis), one marker per ordered block pair at its median over
                 the refresh records and seeds of an overlay method with the
                 interquartile range as a bar; the two orientations of a pair are
                 joined by a grey segment (they share C), the rows v in {stem, head}
                 have their own marker, the largest ratios are labelled, and rho_s over
                 all records and ordered pairs goes to the caption

The on-demand inputs may be None (results.json absent); they are then left out.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from experiments.provenance import data_items

from ..analysis import coupling_eps_by_pair, spearman_coupling_eps
from ..formatting import tex
from ..tables.exp789 import method_label, method_order

log = logging.getLogger(__name__)

# (registry experiment, suffix of the bar label) for the convolutional runs.
_CONV_RUNS = [
    ("exp8", ""),
    ("exp8_a1_width", "/W"),
    ("exp8_a1_imagenet32", "/IN32"),
    ("exp8_a2", "/pre"),
]
_SHORT = {
    "resnet18": "R18",
    "plain_resnet18": "Plain",
    "resnet18_w0.5": "R18x0.5",
    "resnet18_w2.0": "R18x2",
    "resnet34": "R34",
    "resnet50": "R50",
}
_BAR_STYLE = {"init": "fill=blue!40", "final": "fill=red!60"}
_LADDER_STYLE = [
    ("exp9", "blue, thick, mark=*", "toy decoders"),
    ("exp9_a3_llm", "red, only marks, mark=square*", "frozen LLMs"),
]
# Ordered pairs whose first block is one of these (the rows where the block-diagonal
# step is worst on the ResNet profile) get their own marker series in fig:eps-vs-C.
_EDGE_BLOCKS = ("stem", "head")
# (edge-row style, other-row style) per overlay method: filled marks for the first
# method of the paper's order, hollow ones for the second.
_PAIR_STYLE = {
    "kfac+overlay": ("red!80!black, mark=square*", "blue, mark=*"),
    "ekfac+overlay": ("red!80!black, mark=square", "blue, mark=o"),
}
_PAIR_STYLE_DEFAULT = ("black, mark=triangle*", "black!60, mark=triangle")
# Ordered pairs labelled by name (the largest median ratios of the first method).
_N_LABELS = 4
# Padding of the linear coupling axis before its limits are rounded outward to 0.1.
_C_PAD = 0.03


def _bar_label(arch: str, suffix: str) -> str:
    return _SHORT.get(arch, arch.replace("_", "-")) + suffix


def _decade_label(exponent: int) -> str:
    """``100k`` / ``1M`` / ``10B`` for the parameter count 10**exponent."""
    for unit_exponent, unit in ((9, "B"), (6, "M"), (3, "k")):
        if exponent >= unit_exponent:
            return f"{10 ** (exponent - unit_exponent)}{unit}"
    return str(10**exponent)


def _decade_ticks(n_params: list[int]) -> str:
    """Axis options of the log parameter axis: limits at the decades enclosing the
    counts and one tick per decade, labelled as a plain number with a unit (pgfplots
    would otherwise label the half decades ``10^{5.5}``)."""
    low = math.floor(math.log10(min(n_params)))
    high = max(math.ceil(math.log10(max(n_params))), low + 1)
    ticks = ",".join(str(10**e) for e in range(low, high + 1))
    labels = ",".join(_decade_label(e) for e in range(low, high + 1))
    return (
        f"xmin={10**low}, xmax={10**high},\n"
        f"      xtick={{{ticks}}}, xticklabels={{{labels}}},"
    )


def fig_phi_map(
    exp8: dict[str, Any],
    exp8_a1_width: dict[str, Any] | None = None,
    exp8_a1_imagenet32: dict[str, Any] | None = None,
    exp8_a2: dict[str, Any] | None = None,
    exp9: dict[str, Any] | None = None,
    exp9_a3_llm: dict[str, Any] | None = None,
) -> str:
    """Phi by run: bars for the convolutional runs, a ladder for the transformers."""
    bars: list[tuple[str, dict[str, Any]]] = []
    for (key, suffix), bundle in zip(
        _CONV_RUNS, [exp8, exp8_a1_width, exp8_a1_imagenet32, exp8_a2], strict=True
    ):
        if bundle is None:
            log.info("fig:phi-map: %s absent -- left out", key)
            continue
        bars.extend(
            (_bar_label(arch, suffix), ckpts) for arch, ckpts in data_items(bundle)
        )

    panels: list[str] = []
    if bars:
        series: list[str] = []
        for ckpt, style in _BAR_STYLE.items():
            coords = [
                f"({label},{c[ckpt]['phi_mean']:.4f}) +- (0,{c[ckpt]['phi_std']:.4f})"
                for label, c in bars
                if ckpt in c
            ]
            if coords:
                series.append(
                    f"    \\addplot[ybar, {style}, draw=black!60,\n"
                    f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
                    f"        {chr(10).join(coords)}\n"
                    f"    }}; \\addlegendentry{{{ckpt}}}"
                )
        labels = ",".join(label for label, _ in bars)
        panels.append(
            f"    \\nextgroupplot[title={{(a) convolutional runs}}, ybar,\n"
            f"      bar width=5pt, symbolic x coords={{{labels}}}, xtick=data,\n"
            f"      x tick label style={{rotate=45, anchor=east, font=\\footnotesize}},\n"
            f"      enlarge x limits=0.1, ymin=0, ymax=1]\n" + "\n".join(series)
        )

    ladder: list[str] = []
    n_params: list[int] = []
    for (key, style, legend), bundle in zip(
        _LADDER_STYLE, [exp9, exp9_a3_llm], strict=True
    ):
        rungs = data_items(bundle) if bundle else []
        if not rungs:
            log.info("fig:phi-map: %s absent -- left out", key)
            continue
        points = sorted(
            (e["n_params"], e["phi"]["mean"], e["phi"]["std"]) for _, e in rungs
        )
        n_params.extend(int(n) for n, _, _ in points)
        coords = [f"({n},{m:.4f}) +- (0,{s:.4f})" for n, m, s in points]
        ladder.append(
            f"    \\addplot[{style},\n"
            f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
            f"        {chr(10).join(coords)}\n"
            f"    }}; \\addlegendentry{{{legend}}}"
        )
    if ladder:
        panels.append(
            "    \\nextgroupplot[title={(b) transformer ladder}, xmode=log,\n"
            f"      {_decade_ticks(n_params)}\n"
            "      xlabel={parameters}, ymin=0, ymax=1]\n" + "\n".join(ladder)
        )

    if not panels:
        plots = "    % no Phi data"
        group = "1 by 1"
    else:
        plots = "\n".join(panels)
        group = f"{len(panels)} by 1"

    return rf"""\begin{{figure*}}[t]
  \centering
  \begin{{tikzpicture}}
    \begin{{groupplot}}[
      group style={{
        group size={group},
        horizontal sep=1.8cm,
      }},
      width=0.48\textwidth,
      height=5.5cm,
      ylabel={{$\Phi$}},
      grid=major,
      grid style={{gray!30}},
      tick label style={{font=\footnotesize}},
      label style={{font=\small}},
      title style={{yshift=1.7em}},
      legend style={{font=\footnotesize, at={{(0.5,1.03)}}, anchor=south}},
      legend columns=2, legend cell align=left,
    ]
{plots}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Stage A: the discarded-coupling fraction $\Phi$ by run
    ($\pm 1\sigma$ over seeds). Read these raw values against the size and
    diagonal-profile references in Table~\ref{{tab:phi-map}}.
    \textbf{{(a)}}~Convolutional runs at init and final: ResNet-18 vs.\ its
    skip-free control (CIFAR-100; /W the width/depth axis, /IN32 ImageNet-32,
    /pre the ResNet-50 before finetuning (ImageNet weights, new random head)).
    \textbf{{(b)}}~Decoders against parameter count: the untrained toy ladder and
    the frozen, pretrained LLMs.}}
  \label{{fig:phi-map}}
\end{{figure*}}"""


def _pair_stats(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Per ordered pair ``v->w``: median coupling, median ratio and its quartiles over
    all records and seeds (pairs in order of first appearance; records with a zero
    ratio are left out, the axis is logarithmic)."""
    stats: list[dict[str, Any]] = []
    for pair, (c, e) in coupling_eps_by_pair(entry).items():
        v, _, w = pair.partition("->")
        c, e = c[e > 0], e[e > 0]
        if e.size == 0:
            continue
        low, median, high = np.percentile(e, [25, 50, 75])
        stats.append(
            {
                "pair": pair,
                "v": v,
                "w": w,
                "c": float(np.median(c)),
                "e": float(median),
                "e_low": float(low),
                "e_high": float(high),
            }
        )
    return stats


def _pair_name(stat: dict[str, Any]) -> str:
    return f"{tex(stat['v'])}$\\to${tex(stat['w'])}"


def _segments(stats: list[dict[str, Any]]) -> str | None:
    """One grey segment per unordered pair, joining its two orientations (a blank line
    between the segments makes pgfplots lift the pen)."""
    by_pair = {stat["pair"]: stat for stat in stats}
    seen: set[frozenset[str]] = set()
    pieces: list[str] = []
    for stat in stats:
        reverse = by_pair.get(f"{stat['w']}->{stat['v']}")
        key = frozenset((stat["v"], stat["w"]))
        if reverse is None or key in seen:
            continue
        seen.add(key)
        pieces.append(
            f"({stat['c']:.4f},{stat['e']:.4e}) ({reverse['c']:.4f},{reverse['e']:.4e})"
        )
    if not pieces:
        return None
    return (
        "    \\addplot[gray!60, thin, no marks, forget plot] coordinates {\n        "
        + "\n\n        ".join(pieces)
        + "\n    };"
    )


def _marker_series(stats: list[dict[str, Any]], style: str, legend: str) -> str:
    coords = "\n        ".join(
        f"({s['c']:.4f},{s['e']:.4e})"
        f" += (0,{s['e_high'] - s['e']:.4e}) -= (0,{s['e'] - s['e_low']:.4e})"
        for s in stats
    )
    return (
        f"    \\addplot[only marks, {style}, mark size=1.8pt,\n"
        f"      error bars/.cd, y dir=both, y explicit, error mark=none,\n"
        f"      error bar style={{thin}}] coordinates {{\n        {coords}\n"
        f"    }}; \\addlegendentry{{{legend}}}"
    )


def _labels(stats: list[dict[str, Any]], x_mid: float) -> list[str]:
    """Text nodes at the ``_N_LABELS`` largest median ratios: to the right of a point in
    the left half of the axis, to the left otherwise; above and below alternate down
    the ranking (neighbouring extremes sit at the same coupling)."""
    nodes: list[str] = []
    ranked = sorted(stats, key=lambda s: -s["e"])[:_N_LABELS]
    for rank, stat in enumerate(ranked):
        vertical = "south" if rank % 2 == 0 else "north"
        horizontal, shift = ("west", 2) if stat["c"] <= x_mid else ("east", -2)
        nodes.append(
            f"    \\node[font=\\scriptsize, anchor={vertical} {horizontal},"
            f" inner sep=1pt, xshift={shift}pt]\n"
            f"      at (axis cs:{stat['c']:.4f},{stat['e']:.4e}) {{{_pair_name(stat)}}};"
        )
    return nodes


def _decade(value: float, side: str) -> float:
    """The power of ten strictly below (``"low"``) / above (``"high"``) ``value``."""
    exponent = math.log10(value)
    power = math.ceil(exponent) - 1 if side == "low" else math.floor(exponent) + 1
    return 10.0**power


def _rho_sentence(rhos: list[tuple[str, dict[str, Any]]]) -> str:
    """Descriptive Spearman coefficient and dependent-record count, per method."""

    def tail(spearman: dict[str, Any]) -> str:
        return f"($n={spearman['n_pairs']}$ dependent ordered-pair records)"

    if len(rhos) == 1:
        spearman = rhos[0][1]
        return f"Spearman $\\rho_s={spearman['rho']:.2f}$ {tail(spearman)}"
    parts = [f"{label} ${sp['rho']:.2f}$ {tail(sp)}" for label, sp in rhos]
    return "Spearman $\\rho_s$: " + "; ".join(parts)


def fig_eps_vs_C(exp7: dict[str, Any]) -> str:
    """eps(v, w) against C(v, w), one marker per ordered pair (median, IQR bar).

    Every overlay method of the bundle is drawn (its own marker fill); the first one in
    the paper's order sets the axis limits, carries the labels and is named in the
    caption, which also holds rho_s of every method.
    """
    edge_legend = (
        "$v\\in\\{" + ",".join(f"\\mathrm{{{b}}}" for b in _EDGE_BLOCKS) + "\\}$"
    )
    plots: list[str] = []
    rhos: list[tuple[str, dict[str, Any]]] = []
    primary: list[dict[str, Any]] = []
    drawn: list[dict[str, Any]] = []  # every method's points: the axis frames them all
    n_seeds = 0
    for method in method_order(exp7):
        if not method.endswith("+overlay"):
            continue
        stats = _pair_stats(exp7[method])
        if not stats:
            log.info("fig:eps-vs-C: %s has no cross-term records -- left out", method)
            continue
        prefix = f"{method_label(method)}: " if primary else ""
        edge_style, other_style = _PAIR_STYLE.get(method, _PAIR_STYLE_DEFAULT)
        edge = [s for s in stats if s["v"] in _EDGE_BLOCKS]
        other = [s for s in stats if s["v"] not in _EDGE_BLOCKS]
        if segments := _segments(stats):
            plots.append(segments)
        if edge:
            plots.append(_marker_series(edge, edge_style, prefix + edge_legend))
        if other:
            legend = prefix + ("other $v$" if edge else "all ordered pairs")
            plots.append(_marker_series(other, other_style, legend))
        if not primary:
            primary, n_seeds = stats, exp7[method]["n_seeds"]
        drawn.extend(stats)
        rhos.append((method_label(method), spearman_coupling_eps(exp7[method])))

    if primary:
        c_values = [s["c"] for s in drawn]
        xmin = math.floor((min(c_values) - _C_PAD) * 10) / 10
        xmax = math.ceil((max(c_values) + _C_PAD) * 10) / 10
        ymin = _decade(min(s["e_low"] for s in drawn), "low")
        ymax = _decade(max(s["e_high"] for s in drawn), "high")
        reference = (
            f"    \\draw[gray, dashed, thin] (axis cs:{xmin},1) -- (axis cs:{xmax},1);"
        )
        body = "\n".join([reference, *plots, *_labels(primary, (xmin + xmax) / 2)])
        limits = f"      xmin={xmin}, xmax={xmax}, ymin={ymin:g}, ymax={ymax:g},\n"
        label, spearman = rhos[0]
        what = (
            f"one marker per ordered block pair $(v,w)$ at its median over refresh "
            f"records and seeds for the corresponding method; vertical bars show "
            f"the interquartile range of $\\varepsilon$. The {label} runs have "
            f"{spearman['n_records']} refresh records ({n_seeds} seeds). "
            f"Only records with positive $\\varepsilon$ are used for the logarithmic display"
        )
        if {"COUPLE-FAC (K-FAC)", "COUPLE-FAC (EKFAC)"} <= {name for name, _ in rhos}:
            what += "; filled markers denote COUPLE-FAC (K-FAC) and hollow markers denote COUPLE-FAC (EKFAC)"
        edge_note = (
            f"; squares mark the rows {edge_legend}"
            if any(s["v"] in _EDGE_BLOCKS for s in primary)
            else ""
        )
        n_labelled = min(_N_LABELS, len(primary))
        rho_note = _rho_sentence(rhos)
    else:
        body = "    % no overlay records"
        limits = "      xmin=0, xmax=1.2, ymin=1e-2, ymax=1e2,\n"
        what = "one marker per ordered block pair (no records of the correction)"
        edge_note = ""
        n_labelled = 0
        rho_note = "Spearman $\\rho_s$: no records of the correction"

    return rf"""\begin{{figure}}[t]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
      width=0.90\linewidth,
      height=6.5cm,
      ymode=log,
{limits}      xlabel={{Coupling $\mathcal{{C}}(v,w)$ (median over records)}},
      ylabel={{Cross-term ratio $\varepsilon(v,w)$}},
      grid=major,
      grid style={{gray!30}},
      tick label style={{font=\footnotesize}},
      label style={{font=\small}},
      legend style={{font=\footnotesize, at={{(0.5,-0.2)}}, anchor=north}},
      legend columns=2, legend cell align=left,
      clip=false,
    ]
{body}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{The cross-term ratio $\varepsilon(v,w)$
    of~\eqref{{eq:cross-term}} against the coupling $\mathcal{{C}}(v,w)$ by which the
    correction ranks pairs for selection: {what}. The two orientations of a pair share
    $\mathcal{{C}}(v,w)=\mathcal{{C}}(w,v)$ and are joined by a grey segment{edge_note};
    the {n_labelled} largest ratios are labelled; the dashed line is
    $\varepsilon=1$. Descriptive {rho_note}; the cross-term ratio is ordered.
    This raw ordered comparison does not test symmetric or scale-adjusted targets;
    retrospective per-seed associations are in Table~\ref{{tab:coupling-sensitivity}}.}}
  \label{{fig:eps-vs-C}}
\end{{figure}}"""


def _proposal_scale(value: float) -> float:
    """A continuous signed log1p display transform, with zero unchanged."""
    return math.copysign(math.log1p(abs(value) / 1e-4) / math.log(10), value)


def _proposal_ticks(values: list[float], axis: str) -> str:
    """Display original units on the explicitly transformed signed axes."""
    low, high = min(0.0, *values), max(0.0, *values)
    ticks = [0.0]
    for exponent in (-3, -1, 1, 3):
        magnitude = 10.0**exponent
        ticks.extend(v for v in (-magnitude, magnitude) if low <= v <= high)
    ticks.sort()
    positions = ",".join(f"{_proposal_scale(v):.12g}" for v in ticks)
    labels = ",".join(
        "$0$"
        if v == 0
        else rf"${'-' if v < 0 else ''}10^{{{round(math.log10(abs(v)))}}}$"
        for v in ticks
    )
    return f"{axis}tick={{{positions}}}, {axis}ticklabels={{{labels}}},"


def fig_proposal_model_loss(exp7: dict[str, Any]) -> str:
    """All raw signed proposals on explicitly transformed, zero-preserving axes."""
    from ..diagnostics import complete_proposal_decomposition

    result = complete_proposal_decomposition(exp7)
    all_rows = [row for arm in result["arms"].values() for row in arm["records"]]
    q_values = [row["delta_q"] for row in all_rows]
    l_values = [row["delta_loss"] for row in all_rows]
    x = [_proposal_scale(v) for v in (*q_values, 0.0)]
    y = [_proposal_scale(v) for v in (*l_values, 0.0)]
    xmin, xmax = min(x) - 0.2, max(x) + 0.2
    ymin, ymax = min(y) - 0.2, max(y) + 0.2
    methods = (("kfac+overlay", "K-FAC"), ("ekfac+overlay", "EKFAC"))
    marks = {"seed_42": "*", "seed_43": "square*", "seed_44": "triangle*"}
    panels = []
    for method, name in methods:
        arm = result["arms"][method]
        lines = [
            f"    \\nextgroupplot[title={{{name}}}]",
            f"    \\addplot[gray!60, thin, no marks, forget plot] coordinates {{({xmin:.12g},0) ({xmax:.12g},0)}};",
            f"    \\addplot[gray!60, thin, no marks, forget plot] coordinates {{(0,{ymin:.12g}) (0,{ymax:.12g})}};",
        ]
        for seed in marks:
            rows = [row for row in arm["records"] if row["seed"] == seed]
            for status in (False, True):
                selected = [row for row in rows if row["accepted"] == status]
                if not selected:
                    continue
                fill = "screenstage" if status else "white"
                coords = "\n".join(
                    f"      ({_proposal_scale(row['delta_q']):.17g},"
                    f"{_proposal_scale(row['delta_loss']):.17g}) "
                    f"% record {method}/{seed}/{row['step']} "
                    f"raw=({row['delta_q']:.17g},{row['delta_loss']:.17g}) "
                    f"{'accepted' if status else row['rejected_by']}"
                    for row in selected
                )
                lines.append(
                    f"    \\addplot+[only marks, screenstage, mark={marks[seed]}, "
                    f"mark size=1.6pt, mark options={{fill={fill}}}, "
                    "forget plot] coordinates {\n"
                    f"{coords}\n    }};"
                )
            lines.append(
                f"    \\addlegendimage{{only marks, mark={marks[seed]}, screenstage}} "
                f"\\addlegendentry{{{seed.removeprefix('seed_')}}}"
            )
        lines.extend(
            [
                "    \\addlegendimage{only marks, mark=*, mark options={fill=white}, screenstage} \\addlegendentry{rejected}",
                "    \\addlegendimage{only marks, mark=*, screenstage} \\addlegendentry{accepted}",
            ]
        )
        panels.append("\n".join(lines))
    body = "\n".join(panels)
    return rf"""\begin{{figure*}}[t]
  \centering
  \begin{{tikzpicture}}
    \definecolor{{screenstage}}{{HTML}}{{2A78D6}}
    \begin{{groupplot}}[
      group style={{group size=2 by 1, horizontal sep=1.8cm}},
      width=0.43\textwidth, height=6.5cm,
      xmin={xmin:.12g}, xmax={xmax:.12g},
      ymin={ymin:.12g}, ymax={ymax:.12g},
      {_proposal_ticks(q_values, "x")}
      {_proposal_ticks(l_values, "y")}
      xlabel={{Signed-log $\Delta q$ (ticks: raw units)}},
      ylabel={{Signed-log $\Delta L$ (ticks: raw units)}},
      grid=major, grid style={{gray!20}},
      tick label style={{font=\scriptsize}},
      legend style={{font=\scriptsize, at={{(0.5,-0.25)}}, anchor=north}},
      legend columns=3,
    ]
{body}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{All 450 raw proposals per method (150 per training seed):
    $\Delta q=m_{{\rm proposal}}-m_{{\rm kfac}}-
    \lambda(\|d_1\|^2-\|d_0\|^2)/2$ versus
    $\Delta L=L(\theta+d_1)-L(\theta+d_0)$; negative is improvement.
    Both axes use $T(z)=\operatorname{{sign}}(z)\log_{{10}}(1+|z|/10^{{-4}})$,
    with ticks in raw units: zeros, opposing signs and every observation remain.
    Shapes identify seeds; filled/hollow marks show native acceptance/rejection,
    not applied losses. No gamma filtering is used. Refreshes within each training
    trajectory are dependent; summaries are descriptive by training seed.}}
  \label{{fig:proposal-model-loss}}
\end{{figure*}}"""
