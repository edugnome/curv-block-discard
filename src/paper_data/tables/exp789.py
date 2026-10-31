"""Tables for Experiments 7-9: the Stage-A Phi map, the Stage-B payoff, and the gates.

Generated tables
----------------
- tab:phi-map       - Phi per run / architecture / checkpoint (exp8 family, exp9 ladders)
                      against Phi_null (per architecture) and, per checkpoint, Phi_sep
                      (the matched separable reference), with the per-seed inversion
                      Delta_s of the A1 pair and, when a replication bundle is given,
                      its own Delta_s and the two runs' pooled estimate; when
                      exp8_a1_imagenet32 is given, that bundle's own per-seed Delta_s
                      at the middle checkpoint as well
- tab:phi-long      - the A1 pair on the long (300-epoch) schedule of `exp8_a1_long`:
                      one row per checkpoint (accuracy and Phi, 300-probe estimate, of
                      each architecture, and their same-label Delta Phi), with the
                      pre-registered accuracy-matched result as a
                      footer - see `paper_data.analysis`'s module docstring for the
                      criterion; this is a replication with its own pre-registration,
                      not a re-take of criterion (a); the caption compares the final Phi of
                      the long schedule with the individual seeds of the 100-epoch run
                      and its replication (`exp8`, `exp8_a1_v2`), computed from those
                      bundles
- tab:two-partitions - the A1 pair at the six-group partition and at the layer
                      partition, on `exp8_a1_long_gran_m3000`'s final checkpoints:
                      group count, Phi, Phi_sep, S and S's 95% CI, read through the
                      same `paper_data.numbers` granularity helpers the Final*/Gran*
                      macros use, so the table and
                      the macros can never disagree
- tab:stage-b       - exp7 per method: final accuracy, epochs to target, gap closed
                      (median, 95% CI), share of records with m(d_0) > 0 (the base step
                      outside the model's validity), safeguard accept rate, the
                      safeguard's verdicts by test (overlay methods), rho_s
- tab:payoff-budget - exp7 per method: exact HVPs and seconds per step, and the
                      Newton-CG oracle's HVPs per solve and non-positive-curvature stops
                      on the method's trajectory
- tab:gates         - the pre-specified criteria (a)-(d): verdict and the numbers
                      behind it

The statistics come from `paper_data.analysis` (the pre-specified definitions); the
parameter-group sizes Phi_null needs are rebuilt from the architecture, not read from
the bundle (`experiments.group_sizes.group_sizes_for`), since old bundles never
recorded them. The on-demand inputs (exp8_a1_width, exp8_a1_imagenet32, exp8_a2,
exp9_a3_llm, exp8_a1_v2) may be None when their results.json is absent; their rows and
cells then read ``---``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

from experiments.group_sizes import group_sizes_for
from experiments.protocol import checkpoint_epochs, mean_std
from experiments.provenance import data_items, data_keys
from hessian import phi_null, phi_sep

from ..analysis import (
    SAFEGUARD_TESTS,
    accuracy_matched_delta_phi,
    coupling_sensitivity,
    gap_closed,
    gate_a,
    gate_a_replicated,
    gate_b,
    gate_c,
    gate_d,
    negative_curvature_share,
    reject_breakdown,
    same_label_delta_phi,
    spearman_coupling_eps,
    validity_share,
)
from ..formatting import NA, p_str, pm, round_str, tex

# tab:two-partitions reads Phi/Phi_sep/S and the S interval through the same three
# helpers `paper_data.numbers.num_b` uses for the Final*/Gran* macros of the same
# bundle  -- private to that module (as `test_paper_data_numbers.
# py` already imports them directly for the same reason), rather than a fourth
# definition of any of the three living here.
from ..numbers import _gran_triplet, _granularity_stats, _t_interval

log = logging.getLogger(__name__)

_METHOD_ORDER = [
    "sgd",
    "adam",
    "kfac",
    "kfac+overlay",
    "ekfac",
    "ekfac+overlay",
    "newton_cg",
]
_METHOD_LABEL = {
    "sgd": "SGD",
    "adam": "Adam",
    "kfac": "K-FAC",
    "kfac+overlay": "COUPLE-FAC (K-FAC)",
    "ekfac": "EKFAC",
    "ekfac+overlay": "COUPLE-FAC (EKFAC)",
    "newton_cg": "Newton-CG",
}
_ARCH_LABEL = {
    "resnet18": "ResNet-18",
    "plain_resnet18": "Plain-18",
    "resnet18_w0.5": r"ResNet-18 ($0.5\times$)",
    "resnet18_w2.0": r"ResNet-18 ($2\times$)",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
}
_CKPT_ORDER = ["init", "mid", "final"]
# (registry experiment, row label) of the Phi map, in row order.
_PHI_RUNS = [
    ("exp8", "A1 CIFAR-100"),
    ("exp8_a1_width", "A1 width/depth"),
    ("exp8_a1_imagenet32", "A1 ImageNet-32"),
    ("exp8_a2", "A2 pretrained"),
    ("exp9", "A3 toy ladder"),
    ("exp9_a3_llm", "A3 LLM ladder"),
]
_ROLE_TEX = {
    "v->o": r"$v\to o$",
    "q<->k": r"$q\leftrightarrow k$",
    "o->mlp": r"$o\to\mathrm{mlp}$",
    "mlp->next.attn_q": r"$\mathrm{mlp}\to q'$",
}


# ---------------------------------------------------------------------------
# Shared helpers (also used by figures/exp789.py)
# ---------------------------------------------------------------------------


def method_order(exp7: dict[str, Any]) -> list[str]:
    """Methods of an exp7 bundle in the paper's order (unknown ones last).

    Reads the bundle's data keys, so the ``meta`` provenance block never becomes a row.
    """
    methods = data_keys(exp7)
    known = [m for m in _METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in _METHOD_ORDER)


def method_label(method: str) -> str:
    return _METHOD_LABEL.get(method, tex(method))


def arch_label(arch: str) -> str:
    return _ARCH_LABEL.get(arch, tex(arch))


def fmt_params(n: int) -> str:
    """Parameter count as ``0.5B`` / ``12.3M`` / ``85k`` (billions from 100M up, as
    the model names do: Qwen2.5-0.5B has 494M parameters)."""
    if n >= 1e8:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}k"


def role_tex(role: str) -> str:
    if role in _ROLE_TEX:
        return _ROLE_TEX[role]
    block = tex(role.removeprefix("adjacent."))
    return rf"$\mathrm{{{block}}}\to\mathrm{{{block}}}'$"


def _mean_finite(values: Iterable[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _num(value: float, decimals: int) -> str:
    """A math-mode number, ``---`` when not finite."""
    return f"${value:.{decimals}f}$" if math.isfinite(value) else NA


def _epochs(value: float) -> str:
    if math.isinf(value):
        return r"\infty"
    return f"{value:.1f}" if math.isfinite(value) else r"\mathrm{NaN}"


def _gamma_cell(gamma: dict[str, Any]) -> str:
    if not math.isfinite(gamma["median"]):
        return NA
    return (
        f"${gamma['median']:.2f}\\,[{gamma['ci_low']:.2f},\\,{gamma['ci_high']:.2f}]$"
    )


def _rho_cell(spearman: dict[str, Any]) -> str:
    if not math.isfinite(spearman["rho"]):
        return NA
    return f"${spearman['rho']:.2f}$ (${p_str(spearman['p'])}$)"


def _pct(value: float, decimals: int) -> str:
    """A math-mode percentage of a share, ``---`` when not finite."""
    return f"${100 * value:.{decimals}f}\\,\\%$" if math.isfinite(value) else NA


def _verdicts_cell(verdicts: dict[str, Any]) -> str:
    r"""``$41\,\%$; $2/10/47$``: proposals accepted; rejected by ratio / model / loss.

    The column header names the three tests in this order. (The spelled-out form
    ``$41\,\%$ acc. (rej.\ ratio $2$ / model $10$ / loss $47$)`` is 133\,pt too wide for
    the 468\,pt text width of ``elsarticle[3p]`` even at ``\scriptsize``.)
    """
    if verdicts["n_proposed"] == 0:
        return "none proposed"
    by_test = verdicts["n_reject_by"]
    rejected = "/".join(str(by_test.get(test, 0)) for test in SAFEGUARD_TESTS)
    return f"{_pct(verdicts['accept_share'], 0)}; ${rejected}$"


def _row(cells: list[str]) -> str:
    return "    " + " & ".join(cells) + r" \\"


# ---------------------------------------------------------------------------
# tab:stage-b
# ---------------------------------------------------------------------------


def tab_stage_b(exp7: dict[str, Any]) -> str:
    """Exp7 per method: accuracy, epochs to target, gap closed, validity share, verdicts.

    The gap closed, the verdicts and rho_s exist only for the overlay methods (``---``
    otherwise); the share of records with m(d_0) > 0 is reported for every method that
    has payoff records with ``m_kfac`` (the K-FAC / EKFAC bases included).
    """
    rows: list[str] = []
    entries = [entry for _, entry in data_items(exp7)]
    n_seeds = max((entry["n_seeds"] for entry in entries), default=0)
    target = entries[0]["target_acc"] if entries else math.nan
    eligibility = []
    for method in method_order(exp7):
        entry = exp7[method]
        acc = f"${pm(100 * entry['final_acc_mean'], 100 * entry['final_acc_std'], 1)}$"
        reached = f"({entry['n_reached_target']}/{entry['n_seeds']})"
        e2t = entry["epochs_to_target_mean"]
        e2t_cell = f"${e2t:.1f}$ {reached}" if e2t is not None else f"{NA} {reached}"
        if method.endswith("+overlay"):
            gamma = gap_closed(entry)
            gamma_cell = _gamma_cell(gamma)
            eligibility.append(
                f"{method_label(method)} ${gamma['n_records']}/"
                f"{gamma['n_records'] + gamma['n_skipped']}$"
            )
            rho_cell = _rho_cell(spearman_coupling_eps(entry))
            verdicts_cell = _verdicts_cell(reject_breakdown(entry))
        else:
            gamma_cell = rho_cell = verdicts_cell = NA
        validity_cell = _pct(validity_share(entry)["pooled"], 1)
        accept = _mean_finite(
            rate
            for payload in entry["seeds"].values()
            for rate in payload["safeguard"]["accept_rate_per_epoch"]
        )
        rows.append(
            _row(
                [
                    method_label(method),
                    acc,
                    e2t_cell,
                    gamma_cell,
                    validity_cell,
                    _num(accept, 2),
                    verdicts_cell,
                    rho_cell,
                ]
            )
        )

    body = "\n".join(rows)
    eligibility_note = "; ".join(eligibility)
    return rf"""\begin{{table*}}[t]
  \centering
  \caption{{Stage B: per method, final validation accuracy
    (mean$\,{{\pm}}\,1\sigma$ over {n_seeds}~seeds), epochs to the target
    accuracy of ${100 * target:.0f}\,\%$ (mean over the seeds that reached it;
    reached/total), gap closed $\gamma$ (applied step; median over eligible refresh records,
    95\,\% stratified record-bootstrap CI; eligible/total: {eligibility_note}), safeguard accept rate (mean over epochs and
    seeds), and Spearman $\rho_s$ between $\mathcal{{C}}(v,w)$ and
    $\varepsilon(v,w)$ over all finite ordered pairs in the refresh records,
    without the gamma eligibility filter. Eligibility requires
    $m(d_0)-m(d_N)>10^{{-8}}\max(1,|m(d_N)|)$; the truncated oracle is not an optimum.
    The historical $p$-values are nominal and the record bootstrap does not account
    fully for serial dependence or new trajectories. Accuracy means describe separate
    trajectories, not paired causal effects. Proposal model and native-loss comparisons
    within each refresh are same-state comparisons. $m(d_0)>0$ is the share of
    refresh records, pooled over seeds, on which the damped model predicts an
    increase along the base step $d_0$. Verdicts are the safeguard's decisions pooled over seeds:
    the share of proposals accepted, then the numbers rejected by the ratio, the
    model and the loss test, in that order.}}
  \label{{tab:stage-b}}
  \scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{@{{}}lccccccc@{{}}}}
    \toprule
    Method & Final acc.\,(\%) & Ep.\ to target & $\gamma$ [95\,\% CI]
    & $m(d_0)>0$ & Accept & \begin{{tabular}}[c]{{@{{}}c@{{}}}}Verdicts\\acc.; rej.\ by\\ratio/model/loss\end{{tabular}} & $\rho_s$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}"""


# ---------------------------------------------------------------------------
# tab:coupling-sensitivity (retrospective; Appendix A)
# ---------------------------------------------------------------------------


def tab_coupling_sensitivity(exp7: dict[str, Any]) -> str:
    """One compact per-seed table, without new p-values or gate changes."""
    rows = []
    for method in method_order(exp7):
        if not method.endswith("+overlay"):
            continue
        for seed, row in coupling_sensitivity(exp7[method]).items():
            domains = [row[d] for d in ("raw", "symmetric", "adjusted")]
            rows.append(
                _row(
                    [
                        method_label(method),
                        tex(seed.removeprefix("seed_")),
                        str(len(exp7[method]["seeds"][seed]["cross_term"])),
                        *[_num(domain["rho"], 3) for domain in domains],
                        "/".join(str(domain["n_pairs"]) for domain in domains),
                        "/".join(str(domain["n_skipped"]) for domain in domains),
                    ]
                )
            )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Retrospective coupling sensitivity, per seed: Spearman associations
    pooled within seed.
    Raw: finite ordered $(\mathcal{{C}},\varepsilon)$, including zeros.
    Symmetric: one observation per unordered pair at each refresh, comparing $\mathcal{{C}}$ with
    $\sqrt{{\varepsilon(v,w)\varepsilon(w,v)}}$; both orientations must be present
    with finite non-negative values and matching $\mathcal{{C}}$.
    Adjusted: finite strictly positive $(\mathcal{{C}},\varepsilon)$; both log variables
    are fitted by OLS on intercept + receiver + source per refresh, followed by pooled residual ranks
    within seed. No gamma eligibility filter is used. $R$ counts refresh records;
    $N$ and excluded counts are raw/symmetric/adjusted observations.
    Undefined correlations (fewer than three observations or constant values) read {NA}.}}
  \label{{tab:coupling-sensitivity}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{@{{}}llrcccll@{{}}}}
    \toprule
    Method & Seed & $R$ & Raw & Symmetric & Adjusted & $N$ & Excluded \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:payoff-budget
# ---------------------------------------------------------------------------


def _per_step(payload: dict[str, Any], key: str) -> float:
    n_steps = payload["n_steps"]
    return sum(payload[key]) / n_steps if n_steps > 0 else math.nan


def _hvp_per_solve(entry: dict[str, Any]) -> float:
    """Mean ``n_hvp_newton`` over the payoff records, pooled over seeds (NaN without)."""
    counts = [
        record["n_hvp_newton"]
        for payload in entry["seeds"].values()
        for record in payload["model_decrease"]
        if "n_hvp_newton" in record
    ]
    return sum(counts) / len(counts) if counts else math.nan


def tab_payoff_budget(exp7: dict[str, Any]) -> str:
    """Exp7 per method: exact HVPs per step and seconds per step (mean over seeds).

    HVP/step counts the exact HVPs issued through the method's counter -- the
    optimiser's, the overlay's, the safeguard's and those of the recorded diagnostics
    (the oracle's solve, ``m_newton``, the cross-term ratios epsilon) -- and not the
    Hutchinson probes of the coupling field, which run through the estimator's own
    block-wise closures (`MethodTrace`'s ``n_hvp``); the caption says so, in one clause.

    HVP/solve and non-pos. curv. describe the truncated Newton-CG oracle's solve on each
    method's own trajectory (for the ``newton_cg`` method: its own step): the mean
    ``n_hvp_newton`` (CG iterations) over the payoff records and the share of the
    ``newton_cg`` solves that stopped on non-positive curvature, both pooled over seeds;
    ``---`` where there are no records / no solves. When a Newton-CG run became
    non-finite (`nonfinite_record_steps`: a NaN ``loss_theta`` in its records), both
    columns of that row also include the solves after that point, where no stop can be
    marked; the caption explicitly retains these non-finite-state solves.
    No column of constants: the steps and the Hessian minibatch are the same on every
    row, and Appendix A's table of configurations has them.
    """
    rows: list[str] = []
    for method in method_order(exp7):
        entry = exp7[method]
        payloads = list(entry["seeds"].values())
        hvp = _mean_finite(_per_step(p, "hvp_per_epoch") for p in payloads)
        sec = _mean_finite(_per_step(p, "wall_sec_per_epoch") for p in payloads)
        solve_cell = _num(_hvp_per_solve(entry), 1)
        negative_cell = _pct(negative_curvature_share(entry)["share"], 0)
        rows.append(
            _row(
                [
                    method_label(method),
                    _num(hvp, 1),
                    solve_cell,
                    negative_cell,
                    _num(sec, 3),
                ]
            )
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Cost per optimiser step by method: exact Hessian-vector products (HVPs) per
    step (the diagnostics included, the Hutchinson probes of the coupling field not)
    and training wall-clock seconds per step, means over seeds. HVP/solve and
    non-pos.\ curv.\ describe the truncated Newton-CG oracle's solve on each method's own
    trajectory (for the Newton-CG method, its own step): the number of
    conjugate-gradient iterations (one exact HVP each) per solve and the share of
    solves that stopped on non-positive curvature (including zero), a non-PD diagnostic.
    Counts retain solves on non-finite states,
    where a non-positive-curvature stop need not be flagged.}}
  \label{{tab:payoff-budget}}
  \small
  \begin{{tabular}}{{@{{}}lcccc@{{}}}}
    \toprule
    Method & HVP/step & HVP/solve & Non-pos.\ curv. & s/step \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:phi-map
# ---------------------------------------------------------------------------


def _phi_with_sep(phi_mean: float, phi_std: float, seeds: dict[str, Any]) -> str:
    """``$mean \\pm std$``, with that measurement's matched separable reference
    Phi_sep in parenthesis: the mean over seeds of `phi_sep` on the seed's own
    diagonal block norms (`hessian.phi_sep`) - a reference that moves with the
    weights, not a null.
    """
    # diag_frob is already the plain norm ||H_vv||_F, the unit phi_sep takes.
    sep = _mean_finite(phi_sep(payload["diag_frob"]) for payload in seeds.values())
    sep_cell = f" (${sep:.3f}$)" if math.isfinite(sep) else ""
    return f"${pm(phi_mean, phi_std, 3)}${sep_cell}"


def _phi_cells(checkpoints: dict[str, Any]) -> list[str]:
    """One Phi cell per checkpoint column (init, mid, final); NA where not measured."""
    cells = []
    for ckpt in _CKPT_ORDER:
        entry = checkpoints.get(ckpt)
        cells.append(
            NA
            if entry is None
            else _phi_with_sep(entry["phi_mean"], entry["phi_std"], entry["seeds"])
        )
    return cells


def _n_seeds(checkpoints: dict[str, Any]) -> str:
    counts = sorted({int(entry["n_seeds"]) for entry in checkpoints.values()})
    return "/".join(str(n) for n in counts)


def _recorded_sizes_exp8(checkpoints: dict[str, Any]) -> list[int] | None:
    """A ``group_sizes`` already recorded on some seed of some checkpoint, or None.

    The partition is a property of the architecture, not of the seed or checkpoint
    it was measured at, so the first recorded value is sufficient (see
    experiments.phi.measure_phi).
    """
    for entry in checkpoints.values():
        for payload in entry["seeds"].values():
            sizes = payload.get("group_sizes")
            if sizes is not None:
                return sizes
    return None


def _recorded_sizes_exp9(entry: dict[str, Any]) -> list[int] | None:
    """As `_recorded_sizes_exp8`, for an exp9 entry (one measurement, seeds at top)."""
    for payload in entry["seeds"].values():
        sizes = payload.get("group_sizes")
        if sizes is not None:
            return sizes
    return None


def _phi_null_cell(
    bundle: dict[str, Any], experiment: str, model_key: str, recorded: list[int] | None
) -> str:
    """Phi_null for one row: a recorded ``group_sizes`` if any seed has one, else
    rebuilt on CPU by `experiments.group_sizes.group_sizes_for`; NA when neither works
    (frozen Hugging Face models - Phi_null would require downloading them).
    """
    sizes = recorded
    if sizes is None:
        num_classes = bundle.get("meta", {}).get("config", {}).get("num_classes")
        sizes = group_sizes_for(experiment, model_key, num_classes)
    return _num(phi_null(sizes), 3) if sizes is not None else NA


# The A2 run's network is the ImageNet-pretrained ResNet-50 under a NEW randomly
# initialised head (`experiments.models.SegmentedResNet50`: ``base.fc = nn.Linear(...)``,
# drawn per seed), so its Model cell says so. The note sits in the Model cell and not in the run label: that column is already
# as wide as the language models' names, which are longer, so the table does not widen.
_NEW_HEAD_NOTE = ", new random head"
_NEW_HEAD_RUNS = frozenset({("exp8_a2", "resnet50")})


def _phi_rows_exp8(key: str, label: str, bundle: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for arch, checkpoints in data_items(bundle):
        null_cell = _phi_null_cell(bundle, key, arch, _recorded_sizes_exp8(checkpoints))
        model = arch_label(arch)
        if (key, arch) in _NEW_HEAD_RUNS:
            model += _NEW_HEAD_NOTE
        cells = [label if not rows else "", model, null_cell]
        rows.append(_row(cells + _phi_cells(checkpoints) + [_n_seeds(checkpoints)]))
    return rows


def _phi_rows_exp9(key: str, label: str, bundle: dict[str, Any]) -> list[str]:
    """Frozen transformers: one measurement, reported in the init column."""
    rows: list[str] = []
    for model_id in sorted(data_keys(bundle), key=lambda m: bundle[m]["n_params"]):
        entry = bundle[model_id]
        null_cell = _phi_null_cell(bundle, key, model_id, _recorded_sizes_exp9(entry))
        cells = [
            label if not rows else "",
            f"{tex(model_id)} ({fmt_params(entry['n_params'])})",
            null_cell,
            _phi_with_sep(entry["phi"]["mean"], entry["phi"]["std"], entry["seeds"]),
            NA,
            NA,
            str(entry["n_seeds"]),
        ]
        rows.append(_row(cells))
    return rows


def _inversion_note(
    exp8: dict[str, Any],
    exp8_a1_v2: dict[str, Any] | None = None,
    exp8_a1_imagenet32: dict[str, Any] | None = None,
) -> str:
    """The A1 paired difference per seed at the pre-registered (final) and the mid
    checkpoint; with a replication bundle also present, its own per-seed delta, the
    two runs' pooled mean and 95% t-interval, and the conservative t-interval over the
    per-seed means (`gate_a_replicated`).

    With ``exp8_a1_imagenet32`` also present, a trailing sentence gives that
    bundle's own per-seed $\\Delta_s$ at the middle checkpoint.
    """
    parts = []
    for ckpt in ("final", "mid"):
        gate = gate_a(exp8, checkpoint=ckpt)
        if gate["missing"]:
            continue
        per_seed = ", ".join(f"${delta:.3f}$" for delta in gate["delta"].values())
        text = f"{ckpt}: {per_seed} (mean ${gate['mean_delta']:.3f}$)"
        if exp8_a1_v2 is not None:
            pooled = gate_a_replicated([exp8, exp8_a1_v2], checkpoint=ckpt)
            if not pooled["missing"]:
                rep_delta = ", ".join(
                    f"${d:.3f}$" for d in pooled["per_bundle"][1]["delta"].values()
                )
                text += (
                    f"; replication: {rep_delta}; pooled $n={pooled['n']}$: "
                    f"mean ${pooled['mean']:.3f}$, "
                    f"95\\,\\% CI $[{pooled['ci_low']:.3f},\\,{pooled['ci_high']:.3f}]$"
                    f"; per-seed means $n={pooled['n_seeds']}$: CI "
                    f"$[{pooled['ci_low_seeds']:.3f},\\,{pooled['ci_high_seeds']:.3f}]$"
                )
        parts.append(text)
    if not parts:
        return "A1 pair absent: no $\\Delta_s$."
    note = (
        r"A1 $\Delta_s=\Phi_{\mathrm{ResNet}}-\Phi_{\mathrm{Plain}}$ per seed "
        + "; ".join(parts)
        + "."
    )
    if exp8_a1_imagenet32 is not None:
        inet_gate = gate_a(exp8_a1_imagenet32, checkpoint="mid")
        if not inet_gate["missing"]:
            inet_per_seed = ", ".join(
                f"${delta:.3f}$" for delta in inet_gate["delta"].values()
            )
            note += (
                r" ImageNet-32 A1 $\Delta_s$ per seed, mid: "
                f"{inet_per_seed} (mean ${inet_gate['mean_delta']:.3f}$)."
            )
    return note


def tab_phi_map(
    exp8: dict[str, Any],
    exp8_a1_width: dict[str, Any] | None = None,
    exp8_a1_imagenet32: dict[str, Any] | None = None,
    exp8_a2: dict[str, Any] | None = None,
    exp9: dict[str, Any] | None = None,
    exp9_a3_llm: dict[str, Any] | None = None,
    exp8_a1_v2: dict[str, Any] | None = None,
) -> str:
    """Phi per run and architecture, one column per checkpoint, plus the A1 inversion.

    ``exp8_a1_v2`` is the optional A1 replication bundle (same configuration, seeds
    42-44): present, it extends the footnote with its own per-seed Delta_s and the
    pooled estimate over both runs; it is not itself a row of the table.
    """
    bundles = dict(
        zip(
            [key for key, _ in _PHI_RUNS],
            [exp8, exp8_a1_width, exp8_a1_imagenet32, exp8_a2, exp9, exp9_a3_llm],
            strict=True,
        )
    )
    rows: list[str] = []
    for i, (key, label) in enumerate(_PHI_RUNS):
        if i:
            rows.append(r"    \midrule")
        bundle = bundles[key]
        if bundle is None:
            log.info("tab:phi-map: %s absent -- row marked %s", key, NA)
            rows.append(_row([label] + [NA] * 6))
        elif key.startswith("exp9"):
            rows.extend(_phi_rows_exp9(key, label, bundle))
        else:
            rows.extend(_phi_rows_exp8(key, label, bundle))

    body = "\n".join(rows)
    note = _inversion_note(exp8, exp8_a1_v2, exp8_a1_imagenet32)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Stage A: discarded-coupling fraction $\Phi$ per run and architecture at the
    three checkpoints (mean$\,{{\pm}}\,1\sigma$ over $n$~seeds), against
    $\Phi_{{\mathrm{{null}}}}$ - the group-size reference, fixed per architecture -
    and, in parenthesis next to each measured $\Phi$,
    $\Phi_{{\mathrm{{sep}}}}$: the matched separable reference computed from that
    checkpoint's own diagonal block norms, which moves with the weights and is not a
    null. Both are computed per seed before averaging. For the band-estimated
    decoder rows, the full norm is estimated separately. Frozen transformers are
    measured once, with the seeds varying the Hutchinson probes. Checkpoints not applicable to a run read {NA}.}}
  \label{{tab:phi-map}}
  \footnotesize
  \setlength{{\tabcolsep}}{{2pt}}
  \begin{{tabular}}{{@{{}}llccccc@{{}}}}
    \toprule
    Run & Model & $\Phi_{{\mathrm{{null}}}}$ & $\Phi_{{\mathrm{{init}}}}$ & $\Phi_{{\mathrm{{mid}}}}$ & $\Phi_{{\mathrm{{final}}}}$ & $n$ \\
    \midrule
{body}
    \bottomrule
    \multicolumn{{7}}{{@{{}}p{{0.95\linewidth}}}}{{\footnotesize {note}}}
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:phi-long
# ---------------------------------------------------------------------------

# The long-schedule checkpoint labels used for placeholder rows, in schedule order.
_LONG_LABELS = (
    "init",
    "25",
    "50",
    "75",
    "100",
    "125",
    "mid",
    "175",
    "200",
    "225",
    "250",
    "275",
    "final",
)
# The schedule tab:phi-long is specific to; used for the placeholder epoch column and
# as the fallback when a bundle predates the `meta` provenance block.
_LONG_TOTAL_EPOCHS = 300
_LONG_TOL = 0.02
_LONG_PROBES = "300"


def _phi_at(payload: dict[str, Any], probes: str) -> float:
    """The Phi estimate at ``probes`` Hutchinson probes off one seed record, falling
    back to the headline ``phi`` when the sweep, or that count in it, is absent - the
    same estimate choice `paper_data.analysis.accuracy_matched_delta_phi` and
    `same_label_delta_phi` make (their own module-private ``_phi_estimate``)."""
    by_probes = payload.get("phi_by_probes")
    if by_probes and probes in by_probes:
        return float(by_probes[probes])
    return float(payload["phi"])


def _phi_at_stat(entry: dict[str, Any], probes: str) -> dict[str, float]:
    """Mean/std over seeds of `_phi_at` for one (arch, checkpoint) entry."""
    return mean_std([_phi_at(p, probes) for p in entry["seeds"].values()])


def _pm_cell(mean: float, sd: float, decimals: int) -> str:
    """``$mean \\pm sd$``, a bare ``$mean$`` when ``sd`` is not finite, ``---`` when
    ``mean`` itself is not finite (no measurement at all)."""
    if not math.isfinite(mean):
        return NA
    if not math.isfinite(sd):
        return f"${mean:.{decimals}f}$"
    return f"${pm(mean, sd, decimals)}$"


def _hundred_epoch_seed_phis(
    exp8: dict[str, Any], exp8_a1_v2: dict[str, Any] | None
) -> list[tuple[str, str, float]]:
    """(bundle, seed, Phi) of the ResNet-18 at ``final`` for each individual seed of the
    100-epoch A1 run (``exp8``) and of its replication (``exp8_a1_v2``, when present):
    the headline ``phi`` of the seed's record, the value `tab:phi-map` averages."""
    return [
        (name, seed, float(payload["phi"]))
        for name, bundle in (("exp8", exp8), ("exp8_a1_v2", exp8_a1_v2))
        if bundle is not None
        for seed, payload in bundle["resnet18"]["final"]["seeds"].items()
    ]


def _schedule_note(
    long_final: float, seed_phis: list[tuple[str, str, float]], replicated: bool
) -> str:
    """The caption's comparison of the schedules, from the bundles: the long schedule's mean Phi_final(ResNet) against the range of
    the individual seeds of the 100-epoch run and its replication.

    Raises ``ValueError`` unless ``long_final`` exceeds every individual seed.
    """
    name, seed, top = max(seed_phis, key=lambda item: item[2])
    if not long_final > top:
        raise ValueError(
            "tab:phi-long (LongPhiResnetFinal): the manuscript says the residual network "
            "ends the long schedule 'above every individual seed of the hundred-epoch "
            "run and of its replication', but the long schedule's mean "
            f"Phi_final(ResNet), {long_final!r}, is not above {seed} of {name} "
            f"({top!r}); the sentence and the caption's comparison must change with it"
        )
    low = min(phi for _, _, phi in seed_phis)
    return (
        rf" $\Phi$ at a given accuracy depends on the schedule: here "
        rf"$\Phi_{{\mathrm{{final}}}}(\mathrm{{ResNet}})={long_final:.2f}$ "
        rf"(mean over 3~seeds) against ${low:.2f}$--${top:.2f}$ across the individual "
        f"seeds of the 100-epoch run{' and its replication' if replicated else ''} "
        r"(Table~\ref{tab:phi-map})."
    )


def tab_phi_long(
    exp8_a1_long: dict[str, Any] | None,
    exp8: dict[str, Any],
    exp8_a1_v2: dict[str, Any] | None = None,
) -> str:
    """The A1 pair (ResNet-18 vs plain) on the long, 300-epoch cosine schedule.

    One row per checkpoint: epoch number, test accuracy and Phi (300-probe estimate)
    of each architecture (mean over seeds), and their same-label Delta Phi
    (`same_label_delta_phi`, mean +/- std). The footer carries the pre-registered
    accuracy-matched result (`accuracy_matched_delta_phi`): the
    module docstring of `paper_data.analysis` has the criterion, and why it is a
    replication with its own pre-registration rather than a re-take of criterion (a).
    Without
    a bundle the default checkpoints of the long schedule are rendered as placeholder
    rows, as `tables.exp11.tab_validity` does for exp11. With one, the rows are the
    checkpoints the two architectures actually share, in the bundle's own order (not
    assumed to be the 13 labels above) - so a smaller hand-built bundle exercises the
    same code path as the real, 13-checkpoint one.

    The caption's last sentence compares the long schedule's Phi at ``final`` with the
    individual seeds of the 100-epoch run (``exp8``) and its replication (``exp8_a1_v2``,
    optional: without it the range is the 100-epoch run's alone), `_schedule_note`;
    without the long bundle there is nothing to compare and the sentence is left out.
    """
    if exp8_a1_long is None:
        log.warning("exp8_a1_long absent -- tab:phi-long rendered as placeholders")
        labels = list(_LONG_LABELS)
        ckpt_epochs = checkpoint_epochs(_LONG_TOTAL_EPOCHS, labels)
        rows = [
            _row([label, str(epoch)] + [NA] * 5)
            for label, epoch in zip(labels, ckpt_epochs, strict=True)
        ]
        footer = (
            rf"Accuracy-matched pairs ($|\Delta\mathrm{{acc}}|\le{_LONG_TOL:g}$): {NA}."
        )
        schedule_note = ""
    else:
        resnet, plain = exp8_a1_long["resnet18"], exp8_a1_long["plain_resnet18"]
        meta_epochs = (
            exp8_a1_long.get("meta", {})
            .get("config", {})
            .get("training", {})
            .get("epochs")
        )
        total_epochs = _LONG_TOTAL_EPOCHS if meta_epochs is None else int(meta_epochs)
        labels = [label for label in plain if label in resnet]
        ckpt_epochs = checkpoint_epochs(total_epochs, labels)
        same_label = same_label_delta_phi(exp8_a1_long, probes=_LONG_PROBES)
        rows = []
        for label, epoch in zip(labels, ckpt_epochs, strict=True):
            r_entry, p_entry = resnet[label], plain[label]
            phi_r = _phi_at_stat(r_entry, _LONG_PROBES)
            phi_p = _phi_at_stat(p_entry, _LONG_PROBES)
            delta = same_label[label]
            rows.append(
                _row(
                    [
                        label,
                        str(epoch),
                        _num(r_entry["test_acc_mean"], 3),
                        _num(p_entry["test_acc_mean"], 3),
                        _pm_cell(phi_r["mean"], phi_r["std"], 3),
                        _pm_cell(phi_p["mean"], phi_p["std"], 3),
                        _pm_cell(delta["mean"], delta["sd"], 3),
                    ]
                )
            )
        matched = accuracy_matched_delta_phi(
            exp8_a1_long, tol=_LONG_TOL, probes=_LONG_PROBES
        )
        footer = (
            rf"Accuracy-matched pairs ($|\Delta\mathrm{{acc}}|\le{_LONG_TOL:g}$): "
            rf"${matched['n_pairs']}$ pairs, mean $\Delta\Phi={matched['mean']:.3f}$, "
            rf"95\,\% $t$-CI $[{matched['ci_low']:.3f},\,{matched['ci_high']:.3f}]$ "
            rf"over {matched['n_seeds']}~seeds."
        )
        schedule_note = _schedule_note(
            _phi_at_stat(resnet["final"], _LONG_PROBES)["mean"],
            _hundred_epoch_seed_phis(exp8, exp8_a1_v2),
            replicated=exp8_a1_v2 is not None,
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{The A1 inversion (ResNet-18 vs plain ResNet-18, CIFAR-100) on one
    300-epoch cosine schedule, both architectures, checkpoints \texttt{{init}}, every
    25~epochs, and \texttt{{final}} = epoch~300 with the learning rate annealed to 0;
    accuracy and $\Phi$ (300 Hutchinson probes) are the mean over 3~seeds, $\Phi$ also
    with its standard deviation. $\Delta\Phi$ is the same-label difference
    $\Phi_{{\mathrm{{ResNet}}}}-\Phi_{{\mathrm{{Plain}}}}$ (mean$\,{{\pm}}\,1\sigma$
    over seeds). This is a matched-accuracy replication of the inversion: the
    footer pairs each plain checkpoint after initialisation with the resnet checkpoint
    of the \emph{{same seed}} nearest in accuracy, kept when
    $|\Delta\mathrm{{acc}}|\le0.02$, rather than comparing at the same label.{schedule_note}}}
  \label{{tab:phi-long}}
  \small
  \begin{{tabular}}{{@{{}}lrccccc@{{}}}}
    \toprule
    Checkpoint & Epoch & Acc.\ ResNet & Acc.\ Plain & $\Phi_{{\mathrm{{ResNet}}}}$ & $\Phi_{{\mathrm{{Plain}}}}$ & $\Delta\Phi$ \\
    \midrule
{body}
    \bottomrule
    \multicolumn{{7}}{{@{{}}p{{0.95\linewidth}}}}{{\footnotesize {footer}}}
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:two-partitions
# ---------------------------------------------------------------------------

# "stage" is Regrouped's name for the model's own six-group partition, "layer" the
# finer, per-module one (paper_data.numbers._gran_diag_frob/_gran_phi); these are
# this table's row labels, not the bundle's own vocabulary.
_TWO_PARTITION_LEVEL_LABEL = {"stage": "six groups", "layer": "layer"}
_TWO_PARTITION_ARCHES = (
    ("resnet18", _ARCH_LABEL["resnet18"]),
    ("plain_resnet18", _ARCH_LABEL["plain_resnet18"]),
)
_TWO_PARTITION_LEVELS = (
    ("stage", _TWO_PARTITION_LEVEL_LABEL["stage"]),
    ("layer", _TWO_PARTITION_LEVEL_LABEL["layer"]),
)


def tab_two_partitions(
    exp8_a1_long_gran_m3000: dict[str, Any] | None,
) -> str:
    """The A1 pair (ResNet-18, Plain-18) at the six-group partition and at the layer
    partition, on `exp8_a1_long_gran_m3000`'s final checkpoints.

    Four rows, in this order: resnet18 at six groups, resnet18 at the layer
    partition, plain_resnet18 at six groups, plain_resnet18 at the layer partition.
    Group count, Phi, Phi_sep and S are the seed mean from `paper_data.numbers.
    _granularity_stats`; S's 95% interval is the t-interval (`paper_data.numbers.
    _t_interval`) over the three per-seed S values (`paper_data.numbers.
    _gran_triplet`) -- the same helpers `paper_data.numbers.num_b` uses for its
    Final*/Gran* macros. Without a bundle, every cell reads ``---``.
    """
    if exp8_a1_long_gran_m3000 is None:
        log.warning(
            "exp8_a1_long_gran_m3000 absent -- tab:two-partitions rendered as "
            "placeholders"
        )
        rows = [
            _row([f"{arch_label} ({level_label})", *([NA] * 5)])
            for _, arch_label in _TWO_PARTITION_ARCHES
            for _, level_label in _TWO_PARTITION_LEVELS
        ]
        n_probes = 3000
        total_epochs = _LONG_TOTAL_EPOCHS
    else:
        meta_config = exp8_a1_long_gran_m3000.get("meta", {}).get("config", {})
        n_probes = meta_config.get("n_probes", 3000)
        total_epochs = meta_config.get("training", {}).get("epochs", _LONG_TOTAL_EPOCHS)

        rows = []
        for arch, arch_label in _TWO_PARTITION_ARCHES:
            seeds = list(exp8_a1_long_gran_m3000[arch]["final"]["seeds"].values())
            for level, level_label in _TWO_PARTITION_LEVELS:
                point = _granularity_stats(seeds, level)
                s_values = [_gran_triplet(p, level)[2] for p in seeds]
                interval = _t_interval(s_values)
                rows.append(
                    _row(
                        [
                            f"{arch_label} ({level_label})",
                            f"${round_str(point['n_groups'], 0)}$",
                            f"${round_str(point['phi'], 3)}$",
                            f"${round_str(point['sep'], 3)}$",
                            f"${round_str(point['s'], 3)}$",
                            f"$[{round_str(interval['ci_low'], 3)},"
                            f"\\,{round_str(interval['ci_high'], 3)}]$",
                        ]
                    )
                )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{The A1 pair (ResNet-18 and its skip-free twin, Plain-18, CIFAR-100) at
    the six-group partition and at the layer partition: group count, $\Phi$,
    $\Phi_{{\mathrm{{sep}}}}$ and $S$ of~\eqref{{eq:s-identity}} (mean over
    3~seeds), with $S$'s 95\,\% $t$-interval over the three per-seed values.
    Measured on the saved checkpoints of the {total_epochs}-epoch schedule, not
    retrained. {n_probes}~Hutchinson probes.}}
  \label{{tab:two-partitions}}
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{@{{}}lccccc@{{}}}}
    \toprule
    Model & Groups & $\Phi$ & $\Phi_{{\mathrm{{sep}}}}$ & $S$ & $S$ 95\,\% CI \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:gates
# ---------------------------------------------------------------------------


def _verdict(gate: dict[str, Any]) -> str:
    if gate["missing"]:
        return "n/a"
    return r"\textbf{pass}" if gate["pass"] else r"\textbf{fail}"


def _missing(gate: dict[str, Any]) -> str:
    """The Numbers cell of a gate whose inputs are absent (its verdict is ``n/a``)."""
    return f"missing: {tex(', '.join(gate['missing']))}"


def _gate_a_numbers(
    gate: dict[str, Any], replicated: dict[str, Any] | None = None
) -> str:
    if gate["missing"] or not gate["n_seeds"]:
        return NA
    per_seed = ", ".join(f"${d:.3f}$" for d in gate["delta"].values())
    text = f"mean $\\Delta={gate['mean_delta']:.3f}$ ($\\Delta_s$: {per_seed})"
    if math.isfinite(gate["ci_low"]):
        text += f"; 95\\,\\% CI $[{gate['ci_low']:.3f},\\,{gate['ci_high']:.3f}]$"
    if replicated is not None and not replicated["missing"]:
        rep_delta = ", ".join(
            f"${d:.3f}$" for d in replicated["per_bundle"][1]["delta"].values()
        )
        text += (
            "; replication (same config, seeds 42--44): "
            f"$\\Delta_s$: {rep_delta}; "
            f"pooled $n={replicated['n']}$: mean ${replicated['mean']:.3f}$, "
            f"95\\,\\% CI $[{replicated['ci_low']:.3f},\\,{replicated['ci_high']:.3f}]$"
            f"; per-seed means $n={replicated['n_seeds']}$: CI "
            f"$[{replicated['ci_low_seeds']:.3f},\\,{replicated['ci_high_seeds']:.3f}]$"
        )
    return text


def _rho_by_arm(spearman_by_arm: dict[str, dict[str, Any]]) -> str:
    """``K-FAC 0.01 (p=0.105), EKFAC 0.04 (p<1e-3)``, then a shared or per-arm ``n``."""
    parts = [
        f"{method_label(arm)} ${rho['rho']:.2f}$ (${p_str(rho['p'])}$)"
        for arm, rho in spearman_by_arm.items()
    ]
    n_values = {rho["n_pairs"] for rho in spearman_by_arm.values()}
    n_note = (
        f"$n={n_values.pop()}$ each"
        if len(n_values) == 1
        else "; ".join(
            f"{method_label(arm)} $n={rho['n_pairs']}$"
            for arm, rho in spearman_by_arm.items()
        )
    )
    return f"$\\rho_s$: {', '.join(parts)}, {n_note}"


def _gate_b_numbers(gate: dict[str, Any]) -> str:
    if gate["missing"]:
        return NA
    gamma = gate["gap_closed"]
    base, overlay = gate["epochs_to_target_median"].values()
    # Omit the skipped count when it is zero.
    skipped = f"{gamma['n_skipped']} skipped, " if gamma["n_skipped"] else ""
    return (
        f"$\\gamma={gamma['median']:.2f}\\,"
        f"[{gamma['ci_low']:.2f},\\,{gamma['ci_high']:.2f}]$ "
        f"({gamma['n_records']} records, {skipped}"
        f"${100 * gamma['frac_above_one']:.2f}\\,\\%$ above 1); "
        f"epochs to target ${_epochs(overlay)}$ vs ${_epochs(base)}$; "
        f"$\\min_s\\Delta\\mathrm{{acc}}="
        f"{100 * gate['min_final_acc_delta']:.1f}$\\,pp; "
        f"{_rho_by_arm(gate['spearman_by_arm'])}"
    )


def _gate_c_numbers(gate: dict[str, Any]) -> str:
    if gate["missing"]:
        return NA
    return (
        f"max failed-refresh share ${gate['max_failed_ratio']:.3f}$; "
        f"NaN in val.\\ loss: {'yes' if gate['any_nan_val_loss'] else 'no'}"
    )


def _gate_d_numbers(gate: dict[str, Any]) -> str:
    r"""``toy base: $v\to o$ $\times3$; Qwen/Qwen2.5-0.5B: $J=3/3$ ($v\to o$);...``: the
    roles of the largest toy model's top three, each with its pair count when above one,
    then per language model the J of the registered criterion and the roles it matched.
    """
    if gate["missing"]:
        return NA
    toy_roles = (
        ", ".join(
            role_tex(role) + (rf" $\times{n}$" if n > 1 else "")
            for role, n in gate["toy_role_counts"].items()
        )
        or "none"
    )
    parts = [f"toy {tex(gate['toy_model'])}: {toy_roles}"]
    for model_id, entry in gate["llm"].items():
        matched = ", ".join(role_tex(role) for role in entry["matched_roles"]) or "none"
        parts.append(f"{tex(model_id)}: $J={entry['J']}$ ({matched})")
    return "; ".join(parts)


def tab_gates(
    exp7: dict[str, Any],
    exp8: dict[str, Any],
    exp9: dict[str, Any],
    exp9_a3_llm: dict[str, Any] | None = None,
    exp8_a1_v2: dict[str, Any] | None = None,
) -> str:
    """The pre-specified criteria (a)-(d) with their verdicts and the numbers behind them.

    The paper calls them criteria (a)-(d); the first column prints that and the column
    of the rules is headed "Requirement". ``exp8_a1_v2`` is the optional A1 replication bundle
    (see `tab_phi_map`): present, criterion (a)'s numbers cell appends its own per-seed
    Delta_s and the pooled estimate over both runs, without changing criterion (a)'s
    pre-registered verdict or numbers.
    """
    replicated = (
        gate_a_replicated([exp8, exp8_a1_v2], checkpoint="final")
        if exp8_a1_v2 is not None
        else None
    )
    gates = [
        (
            "(a)",
            "Inversion (Stage A)",
            r"$\Delta_s>0$ all seeds, mean $\ge0.05$",
            gate_a(exp8),
            lambda g: _gate_a_numbers(g, replicated),
        ),
        (
            "(b)",
            "Payoff (Stage B)",
            r"$\gamma\ge0.2$, CI $>0$; epochs to target $\le$ K-FAC "
            r"(median); $\Delta\mathrm{acc}\ge-0.5$\,pp every seed; "
            r"$\rho_s>0$, $p<0.01$",
            gate_b(exp7),
            _gate_b_numbers,
        ),
        (
            "(c)",
            "EKFAC (Stage B)",
            r"failed eigenbasis refreshes $<5\,\%$; no NaN val.\ loss",
            gate_c(exp7),
            _gate_c_numbers,
        ),
        (
            "(d)",
            "Transformers (Stage A)",
            r"$J\ge2/3$, top-3 pairs matched by role",
            gate_d(exp9, exp9_a3_llm),
            _gate_d_numbers,
        ),
    ]
    rows = [
        _row(
            [
                letter,
                name,
                criterion,
                _missing(gate) if gate["missing"] else numbers(gate),
                _verdict(gate),
            ]
        )
        for letter, name, criterion, gate, numbers in gates
    ]

    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
  \centering
  \caption{{Pre-specified acceptance criteria (a)--(d) and verdicts
    (Section~\ref{{sec:results-b}}). Criterion~(b) is evaluated on the K-FAC pair;
    criterion~(d) compares the frozen language models with the largest toy
    decoder.}}
  \label{{tab:gates}}
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{@{{}}llp{{3.6cm}}p{{6.8cm}}l@{{}}}}
    \toprule
    Criterion & Name & Requirement & Numbers & Verdict \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}"""
