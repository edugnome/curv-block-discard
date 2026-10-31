"""Compact, retrospective tables from retained same-record scalar diagnostics."""

from __future__ import annotations

from typing import Any

from ..diagnostics import proposal_decomposition, validity_resolution
from ..formatting import NA, sci, tex


def _scalar(value: float | None) -> str:
    return NA if value is None else f"${sci(value, 2)}$"


def tab_proposal_decomposition(exp7: dict[str, Any]) -> str:
    """All refresh denominators; medians over loss-rejected same-record pairs."""
    analysis = proposal_decomposition(exp7)
    rows = []
    for arm, label in (("kfac+overlay", "K-FAC"), ("ekfac+overlay", "EKFAC")):
        result = analysis["arms"][arm]
        for seed, counts in (*result["per_seed"].items(), ("pooled", result["pooled"])):
            cells = [
                label,
                seed.removeprefix("seed_"),
                str(counts["n_candidates"]),
                str(counts["n_rejected"]),
                f"{counts['n_loss_rejected_lower_q']}/{counts['n_loss_rejected']}",
                *[
                    _scalar(counts["loss_rejected_medians"][key])
                    for key in ("delta_m", "delta_penalty", "delta_q", "delta_loss")
                ],
            ]
            rows.append("    " + " & ".join(cells) + r" \\")
        if arm == "kfac+overlay":
            rows.append(r"    \midrule")
    legacy = any(
        row["damping_source"] == "historical_fixed_protocol_not_recorded"
        for result in analysis["arms"].values()
        for row in result["records"]
    )
    damping_note = (
        r" K-FAC's legacy records omit $\lambda$: its stated fixed historical protocol supplies $\lambda=1$, not current software defaults; EKFAC uses each record's damping."
        if legacy
        else r" Damping is read separately from each refresh record."
    )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Retrospective same-record proposal decomposition on the
    corrected K-FAC and EKFAC trajectories (ResNet-50, Stanford Cars).
    $N$ counts every retained raw refresh candidate, without gamma-eligibility
    filtering; $R$ counts all rejected refresh candidates, not all training steps.
    $Q^-/R_L$ counts strict $\Delta q<0$ among the $R_L$ loss-rejected candidates.
    The last four columns are medians over those $R_L$ pairs, using the same
    pre-update weights and minibatch within each pair:
    $\Delta m=m(d_1)-m(d_0)$ for the damped model,
    $\Delta p=\lambda(\|d_1\|^2-\|d_0\|^2)/2$,
    $\Delta q=\Delta m-\Delta p$ for the undamped local quadratic, and
    $\Delta L=L(\theta+d_1)-L(\theta+d_0)$ for the native data loss, not a model value.
    Here $d_1$ is the proposal even when rejected.{damping_note}
    Subtracting the penalty does not remove damping from candidate construction.
    Counts use unrounded logged-scalar signs; rounded medians do not define them.}}
  \label{{tab:proposal-decomposition}}
  \scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{@{{}}llrrccccc@{{}}}}
    \toprule
    Arm & Seed & $N$ & $R$ & $Q^-/R_L$ & median $\Delta m$ & median $\Delta p$ & median $\Delta q$ & median $\Delta L$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def _range(values: list[float | None], *, grid: bool = False) -> str:
    finite = [value for value in values if value is not None]
    if not finite:
        return r"$\mathrm{none}$"
    lo, hi = min(finite), max(finite)
    lower = f"{lo:g}" if grid else sci(lo, 2)
    upper = f"{hi:g}" if grid else sci(hi, 2)
    # Stack scientific range endpoints, as the existing tables stack long
    # headers: four horizontal scientific ranges would make this panel too wide.
    if not grid and lo != hi:
        cell = (
            rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}${lower}$\\${upper}$\end{{tabular}}"
        )
        if len(finite) != len(values):
            cell += f" ({len(finite)}/{len(values)})"
        return cell
    fragment = lower if lo == hi else rf"[{lower},\,{upper}]"
    if len(finite) != len(values):
        fragment += rf"\;({len(finite)}/{len(values)})"
    return f"${fragment}$"


def _prefix_range(cells: list[dict[str, Any]]) -> str:
    defined = [
        cell["prefix_extent"] for cell in cells if cell["prefix_extent"] is not None
    ]
    undefined = len(cells) - len(defined)
    if not defined:
        return r"$\varnothing$"
    fragment = _range(defined, grid=True)
    return fragment + (rf"; $\varnothing$: {undefined}" if undefined else "")


def _validity_row(
    cohort: str, steps: str, kind: str, cells: list[dict[str, Any]]
) -> str:
    failed = [cell["first_failure"] for cell in cells]
    columns = [cohort, steps, kind, _prefix_range(cells)]
    for key, grid in (
        ("t", True),
        ("delta_loss", False),
        ("model", False),
        ("signed_margin", False),
        ("absolute_margin_over_binary32_loss_spacing", False),
    ):
        columns.append(
            _range(
                [point[key] if point is not None else None for point in failed],
                grid=grid,
            )
        )
    return "    " + " & ".join(columns) + r" \\"


def tab_validity_resolution(
    exp11: dict[str, Any] | None, exp11_kfac_seam: dict[str, Any] | None
) -> str:
    """Ranges for the nine original late cells; repeat checkpoints stay separate."""
    analysis = validity_resolution(exp11, exp11_kfac_seam)
    cells = analysis["cells"]
    rows = []
    for kind in ("damped", "undamped"):
        original = [
            cell
            for cell in cells
            if cell["input_bundle"] == "exp11" and cell["model_kind"] == kind
        ]
        rows.append(_validity_row("Original 42--44", "200/600/1500", kind, original))
    rows.append(r"    \midrule")
    for step in (200, 600, 1500):
        for kind in ("damped", "undamped"):
            repeat = [
                cell
                for cell in cells
                if cell["input_bundle"] == "exp11_kfac_seam"
                and cell["step"] == step
                and cell["model_kind"] == kind
            ]
            rows.append(_validity_row("Repeat 42", str(step), kind, repeat))
    body = "\n".join(rows)
    meta = cells[0]["input"]
    grid = ", ".join(f"{t:g}" for t in meta["t_grid"])
    model = {"resnet50": "ResNet-50"}.get(meta["model"], tex(meta["model"]))
    dataset = {"stanford_cars": "Stanford Cars"}.get(
        meta["dataset"], tex(meta["dataset"])
    )
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Retrospective validity-resolution diagnostic on K-FAC
    ({model}, {dataset}, batch {meta["batch_size"]},
    $\lambda={meta["damping"]:g}$), using each record's own pre-update training minibatch
    (sample identities not retained). Original: exactly nine cells, seeds
    42--44 crossed with steps 200, 600, 1500 from the original input bundle;
    repeat: the separate seed-42 K-FAC repeat, never an extra independent seed.
    Original rows give componentwise min--max ranges over those nine cells
    (scientific endpoints stacked: minimum above maximum), not paired extrema;
    repeat rows show each checkpoint separately.
    Grid: $\{{{grid}\}}$. Sampled validity extent is the last successful point before the
    first failure $t_f$ of $|\Delta L-m|\le0.1|\Delta L|$.
    At $t_f$ the table shows native $\Delta L=L(\theta+t_fd_0)-L(\theta)$, the
    model value and signed margin $M=|\Delta L-m|-0.1|\Delta L|$ (positive fails).
    Damped uses the logged model; undamped subtracts
    $t^2\lambda\|d_0\|^2/2$ from it, not from native loss.
    $\varnothing$ means an undefined prefix (the first point failed); ``none''
    in the failure columns means no sampled failure, not an undefined prefix.
    Zero/zero succeeds; nonzero model with zero loss change fails.
    $s_{{32}}=|\operatorname{{spacing}}(\operatorname{{binary32}}(L(\theta)))|$ is
    an illustrative subtraction-resolution scale at the rounded logged loss, not an
    error bound; a missing scale or overflowing ratio is shown as ``none''.
    Failure ends the prefix even if later points pass.}}
  \label{{tab:validity-resolution}}
  \scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{@{{}}lllcccccc@{{}}}}
    \toprule
    Cohort / seed & Step & Model & Prefix & $t_f$ & $\Delta L(t_f)$ & $m(t_f)$ & $M(t_f)$ & $|M|/s_{{32}}$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""
