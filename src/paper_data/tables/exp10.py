"""Table for Exp10: conditioning of the blocks (appendix "Conditioning of the blocks").

Generated tables
----------------
- tab:conditioning - lambda_max, lambda_min, kappa_lambda (lambda = 1e-2, 1e-1), stable
  rank and (exact spectra only) the negative-eigenvalue fraction, per block and for the
  full matrix, at init and final; the Exp. 1 MLPs (exact) and ResNet-18 (matrix-free).

- tab:conditioning-resnet - the same lambda_max, lambda_min and kappa_lambda,
  restricted to the matrix-free ResNet-18 part of tab:conditioning's bundle (the full
  H_theta and its six parameter groups, at init and final). No neg.-frac. column: that
  quantity is only ever available from an exact spectrum, which this part of the
  bundle never carries, so it is dropped rather than printed as ``---`` throughout.
  No stable-rank column either: the appendix text does not use it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..formatting import NA, round_str, sci

log = logging.getLogger(__name__)

# The bundle's damping keys (`hessian.spectrum.lambda_key`: f"{lam:g}") and their headers.
_KAPPA_KEYS = ("0.01", "0.1")
_CHECKPOINTS = ("init", "final")
_MLP_MODELS = (("plain", "Plain MLP"), ("residual", "Residual MLP"))
_RESNET_MODEL = ("resnet18", "ResNet-18")


# ---------------------------------------------------------------------------
# cell formatting
# ---------------------------------------------------------------------------


def _num(val: float, decimals: int = 3) -> str:
    """A signed quantity: ``$\\infty$`` / ``---`` for non-finite, scientific outside
    [1e-2, 1e3), fixed decimals inside."""
    if math.isnan(val):
        return NA
    if math.isinf(val):
        return r"$\infty$" if val > 0 else r"$-\infty$"
    if val == 0:
        return "$0$"
    if abs(val) >= 1e3 or abs(val) < 1e-2:
        return f"${sci(val, 2)}$"
    return f"${round_str(val, decimals)}$"


def _summary_cells(summary: dict[str, Any]) -> list[str]:
    """lambda_max, lambda_min, kappa (per damping), D, neg. frac of one aggregated record."""
    neg = summary.get("neg_frac")
    return [
        _num(summary["lambda_max"]["mean"]),
        _num(summary["lambda_min"]["mean"]),
        *[_num(summary["kappa"][k]["mean"], 1) for k in _KAPPA_KEYS],
        _num(summary["stable_rank"]["mean"], 1),
        NA if neg is None else f"${round_str(neg['mean'], 3)}$",
    ]


def _row(model: str, block: str, cells: list[str]) -> str:
    return "    " + " & ".join([model, block, *cells]) + r" \\"


def _placeholder_row(model: str, block: str = "full", n_cells: int = 6) -> str:
    return _row(model, block, [NA] * n_cells)


def _extreme_blocks(blocks: dict[str, Any]) -> tuple[str, str]:
    """(block with the largest kappa_{1e-2}, block with the smallest); ties by order."""
    kappas = {b: s["kappa"][_KAPPA_KEYS[0]]["mean"] for b, s in blocks.items()}
    ordered = list(kappas)
    largest = max(ordered, key=lambda b: (kappas[b], -ordered.index(b)))
    smallest = min(ordered, key=lambda b: (kappas[b], ordered.index(b)))
    return largest, smallest


def _mlp_block_label(block: str, role: str) -> str:
    """``block_3`` -> ``block 3 (max $\\kappa$)``."""
    return f"{block.replace('_', ' ')} ({role} $\\kappa$)"


# ---------------------------------------------------------------------------
# tab:conditioning
# ---------------------------------------------------------------------------


def tab_conditioning(exp10: dict[str, Any] | None) -> str:
    """Generate the appendix Table: conditioning of the blocks.

    MLP rows (exact spectra): per model and checkpoint the full block matrix and the
    diagonal blocks with the largest and the smallest kappa_{1e-2}. ResNet-18 rows
    (matrix-free): the full H_theta and every parameter group; their neg. frac is
    ``---``. Every cell is ``---`` for a part (or the whole bundle) that is absent.
    """
    if exp10 is None:
        log.warning("exp10 results absent -- tab:conditioning rendered as placeholders")
    meta = (exp10 or {}).get("meta", {})
    depth = meta.get("depth", 8)
    width = meta.get("width", 64)
    n_lanczos_iter = meta.get("n_lanczos_iter", 60)
    n_probes = meta.get("n_probes", 30)

    rows: list[str] = []

    # --- exact part: the Exp. 1 MLPs ---
    mlp = (exp10 or {}).get("mlp")
    n_seeds = 0
    for arch, label in _MLP_MODELS:
        for ckpt in _CHECKPOINTS:
            entry = (mlp or {}).get(arch, {}).get(ckpt)
            model = f"{label} ({ckpt})"
            if entry is None:
                rows.append(_placeholder_row(model))
                continue
            n_seeds = max(n_seeds, int(entry.get("n_seeds", 0)))
            rows.append(_row(model, "full", _summary_cells(entry["full"])))
            largest, smallest = _extreme_blocks(entry["blocks"])
            rows.append(
                _row(
                    "",
                    _mlp_block_label(largest, "max"),
                    _summary_cells(entry["blocks"][largest]),
                )
            )
            if smallest != largest:
                rows.append(
                    _row(
                        "",
                        _mlp_block_label(smallest, "min"),
                        _summary_cells(entry["blocks"][smallest]),
                    )
                )
        rows.append(r"    \midrule")

    # --- matrix-free part: ResNet-18 ---
    arch, label = _RESNET_MODEL
    resnet = (exp10 or {}).get("resnet18", {}).get(arch)
    for ckpt in _CHECKPOINTS:
        entry = (resnet or {}).get(ckpt)
        model = f"{label} ({ckpt})"
        if entry is None:
            rows.append(_placeholder_row(model))
            continue
        rows.append(_row(model, "full", _summary_cells(entry["full"])))
        for block, summary in entry["blocks"].items():
            rows.append(_row("", block.replace("_", r"\_"), _summary_cells(summary)))

    body = "\n".join(rows)
    seeds_note = f"mean over {n_seeds}~seeds" if n_seeds else "mean over seeds"
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Conditioning~\eqref{{eq:cond}} of the activation-space block matrix
    $[H^f_{{v,w}}]$ of the Exp.\,1 MLPs ($L\!=\!{depth}$, width$\,={width}$,
    CIFAR-10; exact ${depth * width}\times{depth * width}$ spectra, {seeds_note};
    rows: the full matrix and the diagonal blocks $H^f_{{v,v}}$ with the largest
    and the smallest $\kappa_{{10^{{-2}}}}$) and of the parameter-space blocks
    $H_{{\theta_v,\theta_v}}$ and the full $H_\theta$ of ResNet-18 (Exp.\,6
    protocol, seed~42; $\lambda_{{\max}}$, $\lambda_{{\min}}$ from one Lanczos run
    of $m\!=\!{n_lanczos_iter}$ steps and the stable rank from $\|A\|_F^2$ estimated by
    Hutchinson with {n_probes}~probes; the spectrum is not formed there, so the fraction
    of negative eigenvalues is not available).
    $\infty$: $\lambda_{{\min}}+\lambda\le0$, a non-PD diagnostic convention,
    not the general condition number of an invertible indefinite matrix.
    Neg.\ frac.\ counts the eigenvalues below $-10^{{-8}}\lambda_{{\max}}$ of the same
    matrix; where $\lambda_{{\min}}$ itself is of order $10^{{-9}}$ that threshold sits at
    roundoff and the count carries no curvature.}}
  \label{{tab:conditioning}}
  \footnotesize
  \begin{{tabular}}{{@{{}}llcccccc@{{}}}}
    \toprule
    Model & Block & $\lambda_{{\max}}$ & $\lambda_{{\min}}$ & $\kappa_{{10^{{-2}}}}$ & $\kappa_{{10^{{-1}}}}$ & $\mathcal{{D}}$ & neg.\ frac. \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:conditioning-resnet
# ---------------------------------------------------------------------------


def tab_conditioning_resnet(exp10: dict[str, Any] | None) -> str:
    """Generate the appendix Table: conditioning of the ResNet-18 blocks.

    The matrix-free part of `tab_conditioning`'s bundle only: the full H_theta and
    every one of its six parameter groups, at init and final. No neg. frac. column -
    that quantity is available only from an exact spectrum, which this part of the
    bundle never carries, so every one of its cells would read ``---`` throughout;
    dropping the column says so once instead. No stable-rank column either (the
    last two cells of `_summary_cells` are left out). A part (or the whole bundle)
    that is absent renders as ``---`` cells, as in `tab_conditioning`.
    """
    if exp10 is None:
        log.warning(
            "exp10 results absent -- tab:conditioning-resnet rendered as placeholders"
        )

    rows: list[str] = []
    arch, label = _RESNET_MODEL
    resnet = (exp10 or {}).get("resnet18", {}).get(arch)
    for ckpt in _CHECKPOINTS:
        entry = (resnet or {}).get(ckpt)
        model = f"{label} ({ckpt})"
        if entry is None:
            rows.append(_placeholder_row(model, n_cells=4))
            continue
        rows.append(_row(model, "full", _summary_cells(entry["full"])[:-2]))
        for block, summary in entry["blocks"].items():
            rows.append(
                _row("", block.replace("_", r"\_"), _summary_cells(summary)[:-2])
            )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Conditioning~\eqref{{eq:cond}} of the parameter-space blocks
    $H_{{\theta_v,\theta_v}}$ and the full $H_\theta$ of ResNet-18 (seed~42, at
    initialisation and after training). $\infty$ means the damped matrix
    $A+\lambda I$ is not positive definite
    ($\lambda_{{\min}}+\lambda\le0$): a non-PD diagnostic convention,
    not the general condition number of an invertible indefinite matrix.}}
  \label{{tab:conditioning-resnet}}
  \footnotesize
  \begin{{tabular}}{{@{{}}llcccc@{{}}}}
    \toprule
    Model & Block & $\lambda_{{\max}}$ & $\lambda_{{\min}}$ & $\kappa_{{10^{{-2}}}}$ & $\kappa_{{10^{{-1}}}}$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""
