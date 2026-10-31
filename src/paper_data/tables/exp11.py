"""Table for Exp11: the model validity radius along the K-FAC trajectory.

Generated tables
----------------
- tab:validity - one row per probe step of the b1 K-FAC run: ||d0||, g^T d0, the
  pointwise curvature along the step d0^T H d0 / ||d0||^2 against the secant curvature
  the loss has over the whole step (both per unit ||d0||^2), lambda_max, lambda_min
  (both from the same Lanczos run), the validity radius t* and the fraction of ReLU
  kinks the step crosses. With more than one seed, every column but t* is mean +/- std
  over seeds (computed from each seed's own raw record - a secant or a kink fraction
  is never averaged before it is combined with that seed's own ||d0||, see
  `_kappa_sec`); t* lists the per-seed values, sorted ascending.

- tab:validity-arms - t* per (arm, seed, step) across three training trajectories
  sharing the same protocol and the same three seeds (K-FAC, GD, Adam - the runner's
  ``--step-source``), one column per arm, one row per probe step, each cell the
  per-seed t* as in tab:validity. A cell with no nonempty valid grid prefix (a NaN t*) shows as
  ``< t_min`` and brings its legend into the caption, and only then. The table has no
  footer.

The bundle is on demand: ``None`` renders the rows of the default probe steps as
``---`` placeholders. (tab:validity-arms has no such placeholder form: all three arm
bundles are required.)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from experiments.protocol import mean_std

from ..formatting import NA, pm, sci

log = logging.getLogger(__name__)

# The probe steps of `Exp11Config` (the rows of the placeholder table).
_DEFAULT_STEPS = (0, 20, 60, 200, 600, 1500)
_N_CELLS = 8


# ---------------------------------------------------------------------------
# bundle access (aggregate records: every number is {mean, std})
# ---------------------------------------------------------------------------


def mean_of(node: Any) -> float:
    """The seed mean of an aggregate entry (a plain number passes through)."""
    return float(node["mean"]) if isinstance(node, dict) else float(node)


def point_at(points: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    """The grid point of a curve at ``t`` (None when the grid has no such t)."""
    for point in points:
        if math.isclose(float(point["t"]), t):
            return point
    return None


def t_min(exp11: dict[str, Any]) -> float:
    """The smallest grid t of the bundle (below it a NaN t* is reported)."""
    return min(exp11["meta"]["t_grid"])


# ---------------------------------------------------------------------------
# per-seed quantities (each takes one seed's raw probe record)
# ---------------------------------------------------------------------------


def _norm_d0(probe: dict[str, Any]) -> float:
    return float(probe["decomposition"]["norm_d0"])


def _g_dot_d0(probe: dict[str, Any]) -> float:
    return float(probe["decomposition"]["g_dot_d0"])


def _curv_along_d0(probe: dict[str, Any]) -> float:
    return float(probe["sharpness"]["curv_along_d0"])


def _kappa_sec(probe: dict[str, Any]) -> float:
    """The secant curvature at t = 1 per unit ||d0||^2, for one seed's own record
    (do not divide a mean secant by a mean ||d0||^2 - see module docstring)."""
    at_one = point_at(probe["curve_d0"], 1.0)
    if at_one is None:
        return math.nan
    return float(at_one["secant"]) / _norm_d0(probe) ** 2


def _lambda_max(probe: dict[str, Any]) -> float:
    return float(probe["sharpness"]["lambda_max"])


def _lambda_min(probe: dict[str, Any]) -> float:
    return float(probe["sharpness"]["lambda_min"])


def _kink_frac(probe: dict[str, Any]) -> float:
    kink = point_at(probe["kinks"]["rows"], 1.0)
    return float(kink["relu_frac"]) if kink else math.nan


# ---------------------------------------------------------------------------
# cell formatting
# ---------------------------------------------------------------------------


def _num(val: float, decimals: int) -> str:
    """A math-mode number: ``---`` when not finite, scientific outside [1e-2, 1e4)."""
    if not math.isfinite(val):
        return NA
    if val != 0 and (abs(val) >= 1e4 or abs(val) < 1e-2):
        return f"${sci(val, 2)}$"
    return f"${val:.{decimals}f}$"


def _t_star(val: float, floor: float) -> str:
    """The validity radius: a grid t, or ``< t_min`` when no grid point was valid."""
    return f"${val:g}$" if math.isfinite(val) else rf"${{<}}\,{floor:g}$"


def _percent(val: float) -> str:
    return rf"${100 * val:.1f}\,\%$" if math.isfinite(val) else NA


def _row(step: int | str, cells: list[str]) -> str:
    return "    " + " & ".join([str(step), *cells]) + r" \\"


def _seed_values(
    seeds: dict[str, Any], i: int, getter: Callable[[dict[str, Any]], float]
) -> list[float]:
    """``getter`` applied to probe index ``i`` of every seed's raw record, in the
    bundle's seed order (irrelevant to every cell below: the multi-seed cells reduce
    by mean/std or by sorting, neither of which reads seed identity or order)."""
    return [getter(payload["probes"][i]) for payload in seeds.values()]


def _seed_cell(values: Sequence[float], decimals: int) -> str:
    """mean +/- std over seeds (`experiments.protocol.mean_std`); the bare `_num` for
    a single seed."""
    if len(values) == 1:
        return _num(values[0], decimals)
    stats = mean_std(list(values))
    if not math.isfinite(stats["mean"]):
        return NA
    return f"${pm(stats['mean'], stats['std'], decimals)}$"


def _kink_cell(values: Sequence[float]) -> str:
    """As `_seed_cell`, but as a percentage with one decimal on both mean and std."""
    if len(values) == 1:
        return _percent(values[0])
    stats = mean_std(list(values))
    if not math.isfinite(stats["mean"]):
        return NA
    return f"${pm(100 * stats['mean'], 100 * stats['std'], 1)}\\,\\%$"


def _t_star_cell(values: Sequence[float], floor: float) -> str:
    """The per-seed t*, ascending, ``/``-joined; a NaN seed (no nonempty valid grid prefix) as
    ``< t_min``, sorted first (it stands for a validity radius below every finite
    value shown). A single seed prints its bare value."""
    if len(values) == 1:
        return _t_star(values[0], floor)
    ordered = sorted(values, key=lambda v: v if math.isfinite(v) else -math.inf)
    parts = [f"{v:g}" if math.isfinite(v) else rf"{{<}}\,{floor:g}" for v in ordered]
    return "$" + "/".join(parts) + "$"


def _cells(seeds: dict[str, Any], i: int, floor: float) -> list[str]:
    """The eight cells of one probe step, over every seed's raw record at index ``i``."""
    return [
        _seed_cell(_seed_values(seeds, i, _norm_d0), 3),
        _seed_cell(_seed_values(seeds, i, _g_dot_d0), 3),
        _seed_cell(_seed_values(seeds, i, _curv_along_d0), 1),
        _seed_cell(_seed_values(seeds, i, _kappa_sec), 1),
        _seed_cell(_seed_values(seeds, i, _lambda_max), 1),
        _seed_cell(_seed_values(seeds, i, _lambda_min), 1),
        _t_star_cell(_seed_values(seeds, i, lambda p: float(p["t_star"])), floor),
        _kink_cell(_seed_values(seeds, i, _kink_frac)),
    ]


# ---------------------------------------------------------------------------
# tab:validity
# ---------------------------------------------------------------------------


def tab_validity(exp11: dict[str, Any] | None) -> str:
    """Generate Table: the validity radius of the damped model along the K-FAC run.

    One row per probe step, every column but t* mean +/- std over seeds (the bare
    value for a single seed); t* lists the per-seed values sorted ascending. Without a
    bundle the default probe steps are rendered as placeholder rows.
    """
    if exp11 is None:
        log.warning("exp11 results absent -- tab:validity rendered as placeholders")
        rows = [_row(step, [NA] * _N_CELLS) for step in _DEFAULT_STEPS]
        meta: dict[str, Any] = {}
        n_seeds = 0
    else:
        floor = t_min(exp11)
        seeds = exp11["seeds"]
        rows = [
            _row(probe["step"], _cells(seeds, i, floor))
            for i, probe in enumerate(exp11["aggregate"]["probes"])
        ]
        meta = exp11["meta"]
        n_seeds = int(exp11["aggregate"]["n_seeds"])

    body = "\n".join(rows)
    batch = meta.get("batch_size", 64)
    damping = meta.get("damping", 1.0)
    tolerance = meta.get("tolerance", 0.1)
    seeds_note = (
        f"{n_seeds}~seeds"
        if n_seeds > 1
        else "a single seed (42)"
        if n_seeds == 1
        else "seeds"
    )
    stat_note = (
        r"; mean $\pm\,1\sigma$ over seeds, $t^*$ listed in ascending order"
        if n_seeds > 1
        else ""
    )
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Validity of the damped quadratic model
    $m(t d_0)=t\,g^{{\top}}d_0+\tfrac{{t^2}}{{2}}\,d_0^{{\top}}(H+\lambda I)d_0$ along the
    K-FAC step $d_0$ (ResNet-50 on Stanford Cars, batch~{batch},
    $\lambda={damping:g}$, $\eta=1$, MC~Fisher;
    {seeds_note}{stat_note}), on the step's own minibatch before the step is applied.
    Curvatures per unit $\|d_0\|^2$: the pointwise $d_0^{{\top}}Hd_0/\|d_0\|^2$ and the
    secant $\kappa_{{\mathrm{{sec}}}}=2\,(\Delta L(1)-g^{{\top}}d_0)/\|d_0\|^2$ realised by the
    loss over the whole step; $\lambda_{{\max}}$ and $\lambda_{{\min}}$ from the same
    Lanczos run. Sampled validity extent $t^*$: the largest fraction of the
    step up to which $|\Delta L-m|\le{tolerance:g}\,|\Delta L|$ holds at every
    sampled grid point in the initial prefix, separately per seed; an empty prefix
    reads ${{<}}\,t_{{\min}}$.
    This does not assert failure at every later grid point. Kinks: the fraction of ReLU units whose pre-activation changes sign between
    $\theta$ and $\theta+d_0$.}}
  \label{{tab:validity}}
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{@{{}}rcccccccc@{{}}}}
    \toprule
    Step & $\|d_0\|$ & $g^{{\top}}d_0$ & $d_0^{{\top}}Hd_0/\|d_0\|^2$ & $\kappa_{{\mathrm{{sec}}}}$ & $\lambda_{{\max}}$ & $\lambda_{{\min}}$ & $t^*$ & kinks \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:validity-arms
# ---------------------------------------------------------------------------

# The registry's bundle names for this label, in column order (`registry.REGISTRY`);
# `tab_validity_arms`'s own parameters must spell these same names, in this same
# order (`test_every_generator_takes_its_registry_bundles_in_order`) - the LaTeX
# header below spells out the arm instead ("K-FAC", "GD", "Adam"). The second arm is "GD"
# and not "SGD" although its bundle is ``exp11_sgd`` (the runner's ``--step-source sgd``):
# it steps by d0 = -g, which is not Stage B's SGD.
_ARMS: tuple[str, str, str] = ("exp11", "exp11_sgd", "exp11_adam")

# The step source each arm's bundle records in ``meta.step_source``; the caption names the
# arms by it. The K-FAC bundle of the three-seed design predates the key.
_ARM_STEP_SOURCES: dict[str, str] = dict(
    zip(_ARMS, ("kfac", "sgd", "adam"), strict=True)
)


def _arm_steps(bundle: dict[str, Any]) -> list[int]:
    """The probe steps of one arm's aggregate, in bundle order."""
    return [int(probe["step"]) for probe in bundle["aggregate"]["probes"]]


def _check_steps(arms: Mapping[str, dict[str, Any]]) -> list[int]:
    """The probe steps shared by every arm (the rows of the table); a `ValueError`
    naming the mismatch when they disagree - a row is one probe step, so there is no
    layout to fall back on once the arms no longer agree on what that step is."""
    per_arm = {name: _arm_steps(bundle) for name, bundle in arms.items()}
    reference = next(iter(per_arm.values()))
    if any(steps != reference for steps in per_arm.values()):
        raise ValueError(f"arms disagree on probe steps: {per_arm}")
    return reference


def _check_seeds(arms: Mapping[str, dict[str, Any]]) -> None:
    """Every arm must carry the same seeds; a `ValueError` names the arm and the
    seed(s) it lacks relative to the others. The caption says each arm is its own run on
    the same seeds, so a ragged arm is rejected rather than rendered."""
    seeds_by_arm = {name: set(bundle["seeds"]) for name, bundle in arms.items()}
    union: set[str] = set().union(*seeds_by_arm.values())
    for name, seeds in seeds_by_arm.items():
        missing = union - seeds
        if missing:
            nums = ", ".join(sorted(s.removeprefix("seed_") for s in missing))
            raise ValueError(f"arm {name!r} is missing seed(s) {nums}")


def _check_arm_protocol(arms: Mapping[str, dict[str, Any]]) -> None:
    """The columns are headed K-FAC, GD and Adam, and the caption says that the model's
    damping, the update's eta and the batch are the same "in every arm", so a bundle that does not bear it out is
    a `ValueError` naming the arm or the field, not a table with a false header or caption.

    Every bundle's ``meta.step_source`` (the runner's ``--step-source``) must be its
    column's (`_ARM_STEP_SOURCES`; the K-FAC bundle may leave the key out, as the one of
    the three-seed design does), and ``damping``, ``lr`` (eta) and ``batch_size`` must be
    equal across the arms.
    """
    for name, bundle in arms.items():
        recorded = bundle["meta"].get("step_source")
        expected = _ARM_STEP_SOURCES[name]
        if name == "exp11" and recorded is None:
            continue
        if recorded != expected:
            raise ValueError(
                f"arm {name!r} records step source {recorded!r}, but its column is "
                f"{expected!r}"
            )
    for field in ("damping", "lr", "batch_size"):
        per_arm = {name: bundle["meta"].get(field) for name, bundle in arms.items()}
        if len(set(per_arm.values())) != 1:
            raise ValueError(f"arms disagree on {field}: {per_arm}")


def tab_validity_arms(
    exp11: dict[str, Any] | None,
    exp11_sgd: dict[str, Any] | None,
    exp11_adam: dict[str, Any] | None,
) -> str:
    """Generate Table: the validity radius t* across three training trajectories.

    One row per probe step, one column per arm - K-FAC (the b1 step of
    tab:validity, bundle ``exp11``), GD (``exp11_sgd``: d0 = -g, no state,
    `runner_exp11.SGDStep`) and Adam (``exp11_adam``: Adam's direction at learning rate
    1e-3, `runner_exp11.AdamStep`): each its own run on the same seeds, differing from the
    K-FAC arm only in the step source the runner's ``--step-source`` selects, every d0
    applied as theta <- theta + d0 (eta = 1) with no weight decay, clipping or schedule
    (the hook owns the update, `experiments.training.Trainer`). The column is GD, not SGD:
    Stage B's SGD is momentum SGD at 1e-2, which this is not;
    and `_check_arm_protocol` refuses a bundle that is not the arm its column is headed
    with. Each cell is the per-seed t* of that arm and step (`_t_star_cell`,
    ascending); a cell that shows as a ``< t_min`` mark (no nonempty valid grid prefix at all)
    brings its legend into the caption, and only then. There is no footer: the counts
    of (arm, seed, step) cells at or below a threshold from a given step onward, which
    it carried, are the text's.

    Unlike tab:validity, this table has no placeholder form: every arm is required,
    and a missing one is a `ValueError` naming it rather than a row of ``---``.
    ``exp11`` stays in `config.OPTIONAL_EXPERIMENTS` for tab:validity's own on-demand
    rendering, so it alone can still arrive here as ``None`` (a missing
    ``exp11_sgd``/``exp11_adam`` bundle is not optional and fails earlier, at
    `paper_data.loader.load`). `ValueError` is also raised if the three arms disagree
    on seeds or on probe steps, since neither the row layout nor the caption's "the
    same seeds" would then mean what it says.
    """
    bundles = {"exp11": exp11, "exp11_sgd": exp11_sgd, "exp11_adam": exp11_adam}
    absent = [name for name, bundle in bundles.items() if bundle is None]
    if absent:
        raise ValueError(
            "tab:validity-arms requires every arm bundle; missing: " + ", ".join(absent)
        )
    arms: dict[str, dict[str, Any]] = cast(dict[str, dict[str, Any]], bundles)
    _check_seeds(arms)
    steps = _check_steps(arms)
    _check_arm_protocol(arms)

    rows = [
        _row(
            step,
            [
                _t_star_cell(
                    _seed_values(arms[name]["seeds"], i, lambda p: float(p["t_star"])),
                    t_min(arms[name]),
                )
                for name in _ARMS
            ],
        )
        for i, step in enumerate(steps)
    ]
    body = "\n".join(rows)
    marked = any(
        math.isnan(value)
        for name in _ARMS
        for i in range(len(steps))
        for value in _seed_values(arms[name]["seeds"], i, lambda p: float(p["t_star"]))
    )
    legend = (
        r" ${<}\,t_{\min}$ marks a cell where the first grid point failed, so there is no nonempty valid grid prefix."
        if marked
        else ""
    )

    meta = arms["exp11"]["meta"]
    batch = meta.get("batch_size", 64)
    damping = meta.get("damping", 1.0)
    lr = meta.get("lr", 1.0)
    tolerance = meta.get("tolerance", 0.1)
    n_seeds = len(arms["exp11"]["seeds"])

    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{The sampled validity extent $t^*$ of the damped quadratic model of
    Table~\ref{{tab:validity}} ($|\Delta L-m|\le{tolerance:g}\,|\Delta L|$ at every
    sampled grid point in the initial prefix, separately per seed)
    along three training trajectories (ResNet-50 on Stanford Cars, batch~{batch};
    $\lambda={damping:g}$ in the model; outer displacement multiplier $\eta={lr:g}$ in every arm;
    K-FAC/GD learning rate one, Adam learning rate $10^{{-3}}$ already included in $d_0$),
    each its own run on the same {n_seeds}~seeds. $t^*$ is listed in ascending
    order.{legend}}}
  \label{{tab:validity-arms}}
  \footnotesize
  \begin{{tabular}}{{@{{}}rccc@{{}}}}
    \toprule
    Step & K-FAC & GD & Adam \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""
