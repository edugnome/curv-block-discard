"""TikZ figure for Exp11: the model validity radius along the K-FAC trajectory.

Generated figures
-----------------
- fig:validity - (a) |Delta L| (solid) and |m(t d0)| (dashed) against the fraction t of
                 the K-FAC step, log-log, one marker shape per probe step. Absolute
                 values of seed means are descriptive curves, not the signed
                 per-seed grid-prefix validity test;
                 (b) lambda_max and the curvature along d0 - pointwise
                 d0^T H d0 / ||d0||^2 against the secant the loss has over the step -
                 against the training step (log axes; the step axis is shifted by one
                 so step 0 has a place).

The bundle is on demand: ``None`` leaves an empty axis.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from experiments.protocol import mean_std

from ..formatting import NA, sci
from ..tables.exp11 import _kappa_sec, _seed_values, mean_of

log = logging.getLogger(__name__)

_MARKS = ("*", "square*", "triangle*", "diamond*", "pentagon*", "x")
_COLOURS = ("blue", "red", "black!70", "green!55!black", "orange", "violet")
# (record path, style, legend) of the curvature panel; the secant is the t = 1 point.
# No stability-threshold series: for a preconditioned or adaptive method such as
# K-FAC, that threshold is stated for the largest eigenvalue of the *preconditioned*
# Hessian, not the (bare) H plotted here, so the comparison is not drawn.
_SHARPNESS_SERIES = (
    ("lambda_max", "black, thick, mark=*", r"$\lambda_{\max}(\mathbf{H})$"),
    (
        "curv_along_d0",
        "red, thick, mark=square*",
        r"$\mathbf{d}_0^{\top}\mathbf{H}\mathbf{d}_0/\|\mathbf{d}_0\|^2$",
    ),
    ("secant", "blue, thick, mark=triangle*", r"$\kappa_{\mathrm{sec}}$ (secant)"),
)


def _coords(points: list[tuple[float, float]]) -> str:
    """Positive points only (log axes), one per line."""
    return "\n        ".join(f"({x:.6g},{y:.6g})" for x, y in points if y > 0)


def _panel_curves(probes: list[dict[str, Any]]) -> str:
    """(a): |Delta L| and |m| against t along d0, one marker shape per probe step."""
    plots: list[str] = []
    for i, probe in enumerate(probes):
        colour = _COLOURS[i % len(_COLOURS)]
        mark = _MARKS[i % len(_MARKS)]
        curve = probe["curve_d0"]
        actual = [(float(p["t"]), abs(mean_of(p["dL"]))) for p in curve]
        model = [(float(p["t"]), abs(mean_of(p["m"]))) for p in curve]
        plots.append(
            f"    \\addplot[{colour}, thick, mark={mark}, mark size=2pt, mark options={{draw=black, line width=0.4pt}}] coordinates {{\n"
            f"        {_coords(actual)}\n"
            f"    }}; \\addlegendentry{{step {probe['step']}}}\n"
            f"    \\addplot[{colour}, thick, dashed, mark={mark}, mark size=2pt, mark options={{fill=white, draw=black, line width=0.4pt}}, forget plot] coordinates {{\n"
            f"        {_coords(model)}\n"
            f"    }};"
        )
    return (
        "    \\nextgroupplot[title={(a) loss vs.\\ model along $\\mathbf{d}_0$},\n"
        "      legend columns=3,\n"
        "      xtick={0.0001,0.001,0.01,0.1,1}, ytick={0.00001,0.001,0.1,10},\n"
        "      xmode=log, ymode=log,\n"
        "      xlabel={$t$ (fraction of the K-FAC step)},\n"
        "      ylabel={$|\\overline{\\Delta L}|$, $|\\overline{m(t \\mathbf{d}_0)}|$}]\n"
        + "\n".join(plots)
    )


def _sharpness_value(
    probe: dict[str, Any], key: str, seeds: dict[str, Any], i: int
) -> float:
    """Return aggregate pointwise curvature or mean per-seed normalised secant.

    Secants use each seed's own squared step norm before averaging.
    """
    if key == "secant":
        return float(mean_std(_seed_values(seeds, i, _kappa_sec))["mean"])
    return mean_of(probe["sharpness"][key])


def _factor(values: list[float]) -> float:
    """Return max/min over positive finite values, or NaN for an empty selection."""
    positive = [v for v in values if math.isfinite(v) and v > 0]
    return max(positive) / min(positive) if positive else float("nan")


def _one_sig_fig(factor: float) -> str:
    r"""*factor* to one significant figure, the way the caption reads best:
    scientific notation ($2{\cdot}10^{2}$, `formatting.sci`) once it reaches ten
    or more, a bare rounded integer below that (so a factor of 3.74 prints "4",
    not "4{\cdot}10^{0}"); ``NA`` when *factor* is not finite (the bundle-less
    placeholder figure has no series to compute it from)."""
    if not math.isfinite(factor):
        return NA
    exp = math.floor(math.log10(factor))
    return str(round(factor)) if exp == 0 else sci(factor, 0)


# The late-step summary requires every sampled seed extent to meet this bound.
_FAR_SHORTER = 0.1
_FAR_SHORTER_FROM_STEP = 200


def _far_shorter_from_step(exp11: dict[str, Any]) -> bool:
    """Whether the bundle supports the caption's "far shorter than the step" from step
    `_FAR_SHORTER_FROM_STEP` on: at least one probe at or after that step, and every
    seed's defined t* at every such probe is at most `_FAR_SHORTER`.
    An undefined/empty-prefix t* does not support a finite extent claim."""
    radii = [
        float(probe["t_star"])
        for payload in exp11["seeds"].values()
        for probe in payload["probes"]
        if int(probe["step"]) >= _FAR_SHORTER_FROM_STEP
    ]
    return bool(radii) and all(math.isfinite(t) and t <= _FAR_SHORTER for t in radii)


def _panel_sharpness(
    probes: list[dict[str, Any]], seeds: dict[str, Any]
) -> tuple[str, dict[str, list[float]]]:
    """(b): curvatures against the training step (step + 1 on the log axis).

    Returns the panel's TikZ source together with every series' plotted y-values,
    keyed by its `_SHARPNESS_SERIES` name -- the caption's two growth factors are
    computed from these same numbers (`fig_validity`), not a second pass over the
    bundle that could silently disagree with what the panel actually draws.
    """
    steps = [int(p["step"]) for p in probes]
    ticks = ",".join(str(s + 1) for s in steps)
    labels = ",".join(str(s) for s in steps)
    plots: list[str] = []
    series_values: dict[str, list[float]] = {}
    for key, style, legend in _SHARPNESS_SERIES:
        values = [_sharpness_value(p, key, seeds, i) for i, p in enumerate(probes)]
        series_values[key] = values
        coords = _coords(list(zip((s + 1.0 for s in steps), values, strict=True)))
        if not coords:
            log.info("fig:validity: %s has no positive value -- left out", key)
            continue
        plots.append(
            f"    \\addplot[{style}] coordinates {{\n        {coords}\n"
            f"    }}; \\addlegendentry{{{legend}}}"
        )
    panel = (
        "    \\nextgroupplot[title={(b) curvature along training},\n"
        "      xmode=log, ymode=log,\n"
        f"      xtick={{{ticks}}}, xticklabels={{{labels}}},\n"
        "      ytick={0.1,1,10,100}, legend columns=3,\n"
        "      xlabel={training step}, ylabel={curvature}]\n" + "\n".join(plots)
    )
    return panel, series_values


def fig_validity(exp11: dict[str, Any] | None) -> str:
    """The validity radius figure: the loss against the model, and the sharpening."""
    if exp11 is None:
        log.warning("exp11 results absent -- fig:validity rendered empty")
        plots = (
            "    \\nextgroupplot[title={(a) loss vs.\\ model along $\\mathbf{d}_0$}]\n"
        )
        plots += "    % no validity data"
        group = "1 by 1"
        curv_factor = secant_factor = float("nan")
    else:
        probes = exp11["aggregate"]["probes"]
        seeds = exp11["seeds"]
        panel_b, sharpness_series = _panel_sharpness(probes, seeds)
        plots = "\n".join([_panel_curves(probes), panel_b])
        group = "1 by 2"
        curv_factor = _factor(sharpness_series["curv_along_d0"])
        secant_factor = _factor(sharpness_series["secant"])
    meta = (exp11 or {}).get("meta", {})
    tolerance = meta.get("tolerance", 0.1)
    curv_factor_tex = _one_sig_fig(curv_factor)
    secant_factor_tex = _one_sig_fig(secant_factor)
    if exp11 is not None and _far_shorter_from_step(exp11):
        scale_claim = (
            "the damped model's extent is short relative to "
            rf"the step at sampled probe steps $\ge{_FAR_SHORTER_FROM_STEP}$ "
            f"($t^*\\le{_FAR_SHORTER:g}$ on every seed)"
        )
    else:  # the bundle does not show it, so the caption does not claim it
        scale_claim = (
            "the signed per-seed extents are reported in "
            r"Table~\ref{tab:validity}, probe step by probe step"
        )

    return rf"""\begin{{figure*}}[t]
  \centering
  \begin{{tikzpicture}}[font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, line width=0.4pt]
    \begin{{groupplot}}[
      axis line style={{line width=0.4pt}}, tick style={{line width=0.4pt}},
      every axis plot/.append style={{line width=0.6pt}},
      error bars/error bar style={{line width=0.6pt}},
      error bars/error mark options={{line width=0.6pt}},
      group style={{
        group size={group},
        vertical sep=85pt,
      }},
      scale only axis, width=318pt,
      height=110pt,
      grid=major,
      grid style={{gray!20, line width=0.35pt}},
      tick label style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont}},
      label style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont}},
      title style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, at={{(0,1)}}, anchor=south west, yshift=28pt}},
      legend style={{font=\fontfamily{{phv}}\fontsize{{9}}{{11}}\selectfont, at={{(0.5,1.03)}}, anchor=south, draw=none, /tikz/column 2/.style={{column sep=5pt}}, /tikz/column 4/.style={{column sep=5pt}}}},
      legend cell align=left,
    ]
{plots}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Validity of the damped quadratic model along the K-FAC trajectory
    (seed means).
    \textbf{{(a)}}~The absolute values of seed means $|\overline{{\Delta L}}|$
    (solid) and $|\overline{{m(t \mathbf{{d}}_0)}}|$ (dashed), against the fraction $t$ of the
    step. Marker shapes identify probe steps on both curves; line style
    distinguishes loss from model. Absolute curves do not show signs or establish
    validity: the sampled validity extent $t^*$ uses the signed per-seed inequality
    $|\Delta L-m|\le{tolerance:g}|\Delta L|$ at every sampled grid point in the initial prefix.
    An empty prefix is reported as ${{<}}\,t_{{\min}}$.
    \textbf{{(b)}}~$\lambda_{{\max}}(\mathbf{{H}})$ and two measures of curvature along~$\mathbf{{d}}_0$:
    the pointwise value $\mathbf{{d}}_0^{{\top}}\mathbf{{H}}\mathbf{{d}}_0/\|\mathbf{{d}}_0\|^2$ and the secant value
    $\kappa_{{\mathrm{{sec}}}}
    =2\,(\Delta L(1)-\mathbf{{g}}^{{\top}}\mathbf{{d}}_0)/\|\mathbf{{d}}_0\|^2$ realised by the loss over the whole
    step. The pointwise curvature grows
    by a factor of ${curv_factor_tex}$ along the
    trajectory while the secant value stays within a factor of~${secant_factor_tex}$: {scale_claim}}}
  \label{{fig:validity}}
\end{{figure*}}"""
