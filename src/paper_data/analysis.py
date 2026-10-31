"""Historical pre-specified statistics and retrospective reporting (exp7-9).

The definitions were fixed before the runs - that is what makes the gates below a test
rather than a description - and this module is the single place the tables, the figures
and the gates read them from. `coupling_sensitivity` is explicitly retrospective,
not a replacement endpoint for any gate. Input JSON bundles reserve ``meta`` for
provenance rather than measurements. Enumerate their data with
``experiments.provenance.data_keys`` / ``data_items``.

gap closed (exp7, a method with the overlay), per payoff record s = (seed, refresh step):

  gamma_s = (m_kfac - m_overlay) / (m_kfac - m_newton),

computed only where m_kfac - m_newton > 1e-8 * max(1, |m_newton|); the other records are
skipped and counted (n_skipped). gamma_s >= 0 by the no-harm safeguard, but gamma_s <= 1
is not guaranteed (d_Newton is a truncated damped CG solve, not the minimizer of m), so
gamma is NOT clipped and the fraction of records with gamma_s > 1 is reported separately.
Point estimate: the median over all (seed, record). 95% CI: percentile bootstrap,
B = 10000, resampling records with replacement within each seed (stratified), rng seed 0.
The per-seed medians are reported alongside.

spearman(C, eps): the pairs (C(v, w), eps(v, w)) over all records and all ordered block
pairs v -> w of the method, NaNs dropped; rho and p from scipy.stats.spearmanr; 95% CI by
percentile bootstrap over records (B = 10000; a record carries all of its pairs). The
same pairs grouped by ordered block pair (coupling_eps_by_pair) feed the per-pair
medians and quartiles of fig:eps-vs-C.

coupling field (exp7): every finite value of C(v, w) = ||H_vw||_F / sqrt(a_v a_w) itself
(coupling_field_values), pooled over all records, seeds and ordered pairs of a method --
cross_term[*].pairs[*].coupling, non-finite dropped. Unlike the two statistics above,
eps is not consulted, so a pair with a finite coupling and a non-finite eps is still
counted. Median and maximum over this pool are the evidence that the aggregate
inequality Phi < Phi_sep (S > 1) is a statement about a sum of squared pair norms, not about any single
pair.

validity share (exp7): the share of payoff records with m_kfac > 0, where the damped
model predicts an increase along the base step d_0. This sign alone does not establish
prediction error, and gamma_s compares model values rather than loss. Per seed and pooled
over all records; records without a finite m_kfac (the newton_cg method's) are skipped. Reported
next to gamma, which stays as pre-registered but is read together with this share.
safeguard verdicts (exp7): pooled over seeds from `safeguard`, the proposals accepted
(n_proposed - n_reject) and rejected by the ratio / model / loss test, as counts and as
shares of n_proposed (NaN when nothing was proposed).
negative curvature (exp7): pooled from `newton_cg`, n_negative_curvature / n_solves of
the truncated-CG solves (the oracle's on the payoff records plus the newton_cg method's).

gate A (exp8): delta_s = Phi_resnet(final) - Phi_plain(final) per seed; pass iff every
delta_s > 0 and mean delta >= 0.05; with >= 5 seeds also a percentile-bootstrap CI on
the mean. gate_a_replicated pools delta_s over several same-configuration bundles (the
pre-registered run and a same-config replication): not a gate - no pass/fail, no 0.05
threshold - just the pooled mean, sd and a 95% t-interval (scipy.stats.t.ppf, n - 1
degrees of freedom) over every bundle's seeds together, which gate_a itself does not
compute. Alongside it, the conservative t-interval over the per-seed means (n = seeds,
not runs * seeds), since same-seed runs are pairs, not independent draws (see
gate_a_replicated's own docstring).
accuracy-matched delta Phi (exp8): a matched-accuracy replication
of the inversion with its own pre-registration - not a re-take of gate A, whose verdict
on the 100-epoch schedule it leaves untouched. Gate A compares Phi at the *same*
checkpoint label, where the two architectures sit at very different accuracies; this
criterion instead asks whether the inversion survives once accuracy is held fixed. On
the long (300-epoch) schedule, for each seed and each plain checkpoint other than
init, the plain checkpoint is paired with the resnet checkpoint of the *same seed*
whose test_acc is nearest, and the pair is kept only when |Delta acc| <= tol (default
0.02) - so a comparison is made only where training has brought both architectures to
a comparable point, wherever that falls in the schedule. Delta Phi:= Phi_resnet -
Phi_plain at a chosen probe estimate (default phi_by_probes["300"], falling back to
the headline phi when the sweep is absent), averaged over the kept pairs within a
seed, then averaged over seeds: mean, sample sd (ddof=1) and a 95% t-interval
(scipy.stats.t.ppf, n_seeds - 1 degrees of freedom), computed by
accuracy_matched_delta_phi. same_label_delta_phi reports the companion, simpler
number for context: Delta Phi at equal checkpoint labels (same nominal schedule
position, not accuracy-matched), mean and sample sd over seeds at every label the two
architectures share.
gate B (exp7): median gamma >= 0.2 and CI lower bound > 0; epochs_to_target(overlay) <=
epochs_to_target(base) in median over seeds (not reached = +inf); final_acc(overlay) -
final_acc(base) >= -0.005 on every seed; spearman rho > 0 with p < 0.01. The rho_s
sub-criterion names no method; it is evaluated on the primary overlay only (as
pre-registered), but spearman_by_arm also reports it for every other arm present in
the bundle (default: the EKFAC pair), so a reader can see both without the verdict
depending on which arm is shown.
gate C (exp7): n_refresh_failed / n_refreshes < 0.05 for the EKFAC methods, with
n_refreshes = n_steps / ekfac_refresh_period (the period is not stored in the bundle;
10 per `Exp7Config`), and no NaN in val_loss of those methods.
gate D (exp9):
  J:= |Top3(C_toy) intersection Top3(C_llm)| / 3 >= 2/3, where pairs are matched by structural role.
The top-3 coupling pairs (by mean C) of the toy ladder's largest model and of every LLM
are matched one to one by role - within a layer: v->o, q<->k, o->mlp; across layers: mlp
-> next attn_q, or the same block in adjacent layers - and at least two of the three must
be matched on every LLM. Pairs are matched, not distinct roles: three v->o pairs on each
side are J = 3/3.

Every gate returns {"pass": bool, "missing": [...],...numbers...}; ``missing`` names the
methods / architectures / models the bundle lacks, in which case ``pass`` is False.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
from scipy import stats

from experiments.provenance import data_items, data_keys

N_BOOT = 10000
RNG_SEED = 0
# Bootstrap replicates per vectorized batch (bounds the memory of the rank matrices).
_CHUNK = 200
# Relative tolerance below which m_kfac - m_newton is treated as zero (record skipped).
_GAMMA_TOL = 1e-8


# ======================================================================
# Helpers
# ======================================================================


def seed_number(name: str) -> tuple[int, str]:
    """Sort key for ``seed_N`` keys: numeric N first, non-numeric names last."""
    _, _, number = name.partition("_")
    return (int(number), name) if number.isdigit() else (2**63 - 1, name)


def _sorted_seeds(seeds: dict[str, Any]) -> list[tuple[str, Any]]:
    return sorted(seeds.items(), key=lambda item: seed_number(item[0]))


def _chunks(n: int, size: int) -> Iterator[tuple[int, int]]:
    """(start, length) of consecutive batches covering range(n)."""
    for start in range(0, n, size):
        yield start, min(size, n - start)


def _percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    tail = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(finite, [tail, 100.0 - tail])
    return float(lo), float(hi)


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else math.nan


# ======================================================================
# Gap closed
# ======================================================================


def gamma_records(payload: dict[str, Any]) -> tuple[list[float], int]:
    """(gamma_s over the payload's records, n_skipped) for one seed."""
    gammas: list[float] = []
    n_skipped = 0
    for record in payload["model_decrease"]:
        m_kfac, m_overlay, m_newton = (
            record["m_kfac"],
            record["m_overlay"],
            record["m_newton"],
        )
        denominator = m_kfac - m_newton
        if denominator > _GAMMA_TOL * max(1.0, abs(m_newton)):
            gammas.append((m_kfac - m_overlay) / denominator)
        else:
            n_skipped += 1
    return gammas, n_skipped


def gap_closed(
    method: dict[str, Any], *, n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> dict[str, Any]:
    """Median gap closed with its stratified percentile-bootstrap CI (see module doc).

    ``method`` is one exp7 method entry (``{"seeds": {seed_N: payload},...}``).
    """
    per_seed: dict[str, list[float]] = {}
    n_skipped = 0
    for name, payload in _sorted_seeds(method["seeds"]):
        gammas, skipped = gamma_records(payload)
        per_seed[name] = gammas
        n_skipped += skipped
    pooled = np.array([g for gammas in per_seed.values() for g in gammas], dtype=float)

    result: dict[str, Any] = {
        "median": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "per_seed_median": {name: _median(g) for name, g in per_seed.items()},
        "n_records": int(pooled.size),
        "n_skipped": n_skipped,
        "frac_above_one": float(np.mean(pooled > 1.0)) if pooled.size else math.nan,
        "n_boot": n_boot,
    }
    if pooled.size == 0:
        return result

    result["median"] = float(np.median(pooled))
    strata = [np.asarray(g, dtype=float) for g in per_seed.values() if g]
    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot)
    for start, size in _chunks(n_boot, _CHUNK):
        columns = [s[rng.integers(0, s.size, size=(size, s.size))] for s in strata]
        medians[start : start + size] = np.median(
            np.concatenate(columns, axis=1), axis=1
        )
    result["ci_low"], result["ci_high"] = _percentile_ci(medians)
    return result


# ======================================================================
# Validity share, safeguard verdicts, negative curvature
# ======================================================================

# Safeguard checks are applied in this order.
SAFEGUARD_TESTS = ("ratio", "model", "loss")


def validity_share(method: dict[str, Any]) -> dict[str, Any]:
    """Share of payoff records where the damped model predicts an increase.

    ``method`` is one exp7 method entry (``{"seeds": {seed_N: payload},...}``). A record
    with ``m_kfac > 0`` is one on which the damped model predicts an *increase* along the
    base step. This sign alone does not establish prediction error. Records without a
    finite ``m_kfac`` (the ``newton_cg`` method's) are skipped. Returns
    ``{"per_seed": {seed: share}, "pooled": share, "n_records": n}`` with the seeds in
    numeric order; a share is NaN where no record was counted.

    ``loss_not_falling`` lists, as (seed, step), the counted records (``m_kfac > 0``) on
    which the minibatch loss does not demonstrably fall after the base step: ``loss_d0``
    not below ``loss_theta``, or either missing or not finite. The manuscript says the
    damped model predicts an increase on these records "while the minibatch loss
    decreases", and the generator
    checks that this list is empty. A record with ``m_kfac <= 0`` is not counted, so a
    rising loss on it is not listed.
    """
    per_seed: dict[str, float] = {}
    n_positive = 0
    n_records = 0
    loss_not_falling: list[tuple[str, int]] = []
    for name, payload in _sorted_seeds(method["seeds"]):
        values = [
            m
            for record in payload["model_decrease"]
            if (m := record.get("m_kfac")) is not None and math.isfinite(m)
        ]
        positive = sum(1 for m in values if m > 0.0)
        per_seed[name] = positive / len(values) if values else math.nan
        n_positive += positive
        n_records += len(values)
        for record in payload["model_decrease"]:
            m = record.get("m_kfac")
            if m is None or not math.isfinite(m) or m <= 0.0:
                continue
            before, after = record.get("loss_theta"), record.get("loss_d0")
            falls = (
                before is not None
                and after is not None
                and math.isfinite(before)
                and math.isfinite(after)
                and after < before
            )
            if not falls:
                loss_not_falling.append((name, int(record.get("step", -1))))
    return {
        "per_seed": per_seed,
        "pooled": n_positive / n_records if n_records else math.nan,
        "n_records": n_records,
        "loss_not_falling": loss_not_falling,
    }


def nonfinite_record_steps(method: dict[str, Any]) -> dict[str, list[int]]:
    """Per seed of ``method``, the steps of the payoff records whose ``loss_theta`` is
    NaN or infinite: the states a run reached after it diverged. Seeds
    with none are left out, steps in record order; a record without a ``loss_theta``
    (a method that does not write one) is not counted. Seeds in numeric order.
    """
    steps: dict[str, list[int]] = {}
    for name, payload in _sorted_seeds(method["seeds"]):
        bad = [
            int(record["step"])
            for record in payload["model_decrease"]
            if "loss_theta" in record and not math.isfinite(float(record["loss_theta"]))
        ]
        if bad:
            steps[name] = bad
    return steps


def reject_breakdown(method: dict[str, Any]) -> dict[str, Any]:
    """The safeguard's verdicts pooled over seeds: accepted / rejected by ratio, model, loss.

    Read from each payload's ``safeguard`` (``n_proposed``, ``n_reject``, ``n_reject_by``);
    ``n_accepted = n_proposed - n_reject``. The shares are of ``n_proposed`` and NaN when
    nothing was proposed. A payload without ``n_reject_by`` keeps its
    ``n_reject`` but attributes it to no test (the by-test counts stay 0).
    """
    n_proposed = 0
    n_reject = 0
    by_test = dict.fromkeys(SAFEGUARD_TESTS, 0)
    for _, payload in _sorted_seeds(method["seeds"]):
        safeguard = payload["safeguard"]
        n_proposed += int(safeguard["n_proposed"])
        n_reject += int(safeguard["n_reject"])
        for test, count in safeguard.get("n_reject_by", {}).items():
            by_test[test] = by_test.get(test, 0) + int(count)

    def share(count: int) -> float:
        return count / n_proposed if n_proposed else math.nan

    return {
        "n_proposed": n_proposed,
        "n_accepted": n_proposed - n_reject,
        "n_reject": n_reject,
        "n_reject_by": by_test,
        "accept_share": share(n_proposed - n_reject),
        "reject_share_by": {test: share(count) for test, count in by_test.items()},
    }


def negative_curvature_share(method: dict[str, Any]) -> dict[str, Any]:
    """Pooled share of truncated-CG solves that stopped on non-positive curvature.

    The historical ``n_negative_curvature`` field includes zero: p^T A p <= 0.
    It diagnoses a non-PD damped operator, not necessarily strict negative curvature.

    From each payload's ``newton_cg`` (``n_solves``, ``n_negative_curvature``): the
    oracle's solves on the payoff records plus every step of the ``newton_cg`` method.
    Payloads without the block contribute nothing; ``share``
    is NaN when there was no solve.
    """
    n_solves = 0
    n_negative = 0
    for _, payload in _sorted_seeds(method["seeds"]):
        block = payload.get("newton_cg") or {}
        n_solves += int(block.get("n_solves", 0))
        n_negative += int(block.get("n_negative_curvature", 0))
    return {
        "n_solves": n_solves,
        "n_negative_curvature": n_negative,
        "share": n_negative / n_solves if n_solves else math.nan,
    }


# ======================================================================
# Spearman(C, eps)
# ======================================================================


def _cross_term_records(
    method: dict[str, Any],
) -> Iterator[dict[str, dict[str, float]]]:
    """Every cross-term record's ``pairs`` mapping (ordered pair -> ``{"coupling",
    "eps"}``) of ``method``, over all seeds in seed order, each record's own pair
    order preserved.

    The one traversal of ``cross_term[*].pairs``: `coupling_eps_records` reduces each
    yielded mapping to that one record's arrays, `coupling_eps_by_pair` pools the same
    mappings by pair key, and `coupling_field_values` pools every ``coupling`` flat
    regardless of pair or record -- three different reductions of the one walk,
    written once here rather than three times.
    """
    for _, payload in _sorted_seeds(method["seeds"]):
        for record in payload["cross_term"]:
            yield record["pairs"]


def coupling_eps_records(method: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per cross-term record: (couplings, epsilons) over its ordered pairs, NaNs dropped.

    Records left with no finite pair are dropped.
    """
    records: list[tuple[np.ndarray, np.ndarray]] = []
    for pairs in _cross_term_records(method):
        couplings: list[float] = []
        epsilons: list[float] = []
        for pair in pairs.values():
            c, e = pair["coupling"], pair["eps"]
            if math.isfinite(c) and math.isfinite(e):
                couplings.append(c)
                epsilons.append(e)
        if couplings:
            records.append((np.asarray(couplings), np.asarray(epsilons)))
    return records


def coupling_eps_by_pair(
    method: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per ordered pair ``"v->w"``: (couplings, epsilons) over all records and seeds.

    Non-finite values are dropped; the pairs keep their order of first appearance, and
    a pair left without a finite record is absent.
    """
    couplings: dict[str, list[float]] = {}
    epsilons: dict[str, list[float]] = {}
    for pairs in _cross_term_records(method):
        for pair, values in pairs.items():
            c, e = values["coupling"], values["eps"]
            if math.isfinite(c) and math.isfinite(e):
                couplings.setdefault(pair, []).append(c)
                epsilons.setdefault(pair, []).append(e)
    return {
        pair: (np.asarray(couplings[pair]), np.asarray(epsilons[pair]))
        for pair in couplings
    }


def coupling_field_values(method: dict[str, Any]) -> list[float]:
    r"""Every finite value of the coupling field C(v, w) = ||H_vw||_F / sqrt(a_v a_w),
    pooled over all records, seeds and ordered pairs of ``method`` --
    ``cross_term[*].pairs[*].coupling``, non-finite dropped.

    Unlike `coupling_eps_records` / `coupling_eps_by_pair`, ``eps`` is not consulted:
    a pair with a finite coupling and a non-finite eps is still counted here, so this
    is not a filter of either of those two functions' output.
    """
    return [
        float(values["coupling"])
        for pairs in _cross_term_records(method)
        for values in pairs.values()
        if math.isfinite(values["coupling"])
    ]


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Average ranks along axis 1 (ties share their mean rank), 1-based.

    Row-wise `scipy.stats.rankdata(method="average")`: runs of equal values in the sorted
    row get the mean of their positions, found as (first position + last position) / 2.
    """
    n = x.shape[1]
    order = np.argsort(x, axis=1, kind="stable")
    sorted_x = np.take_along_axis(x, order, axis=1)
    pos = np.arange(n)
    run_start = np.ones(sorted_x.shape, dtype=bool)
    run_start[:, 1:] = sorted_x[:, 1:] != sorted_x[:, :-1]
    run_end = np.ones(sorted_x.shape, dtype=bool)
    run_end[:, :-1] = run_start[:, 1:]
    first = np.maximum.accumulate(np.where(run_start, pos, 0), axis=1)
    last = np.minimum.accumulate(np.where(run_end, pos, n - 1)[:, ::-1], axis=1)
    average = (first + last[:, ::-1]) / 2.0 + 1.0
    ranks = np.empty_like(average)
    np.put_along_axis(ranks, order, average, axis=1)
    return ranks


def _spearman_rows(c: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Spearman rho of each row pair (Pearson on the average ranks); NaN if constant."""
    rc = _rank_rows(c)
    re = _rank_rows(e)
    rc -= rc.mean(axis=1, keepdims=True)
    re -= re.mean(axis=1, keepdims=True)
    numerator = (rc * re).sum(axis=1)
    denominator = np.sqrt((rc * rc).sum(axis=1) * (re * re).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, math.nan)


def spearman_coupling_eps(
    method: dict[str, Any], *, n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> dict[str, Any]:
    """Spearman rho between C(v, w) and eps(v, w) with a record-bootstrap CI."""
    records = coupling_eps_records(method)
    result: dict[str, Any] = {
        "rho": math.nan,
        "p": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "n_pairs": int(sum(c.size for c, _ in records)),
        "n_records": len(records),
        "n_boot": n_boot,
    }
    if result["n_pairs"] < 3:
        return result

    c_all = np.concatenate([c for c, _ in records])
    e_all = np.concatenate([e for _, e in records])
    rho, p = stats.spearmanr(c_all, e_all)
    result["rho"], result["p"] = float(rho), float(p)

    rng = np.random.default_rng(seed)
    n_records = len(records)
    rhos = np.empty(n_boot)
    lengths = {c.size for c, _ in records}
    if len(lengths) == 1:
        # Equal-length records: gather whole replicates as (batch, n_records * L) rows.
        c_mat = np.stack([c for c, _ in records])
        e_mat = np.stack([e for _, e in records])
        for start, size in _chunks(n_boot, _CHUNK):
            idx = rng.integers(0, n_records, size=(size, n_records))
            rhos[start : start + size] = _spearman_rows(
                c_mat[idx].reshape(size, -1), e_mat[idx].reshape(size, -1)
            )
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n_records, size=n_records)
            c_row = np.concatenate([records[i][0] for i in idx])[None, :]
            e_row = np.concatenate([records[i][1] for i in idx])[None, :]
            rhos[b] = _spearman_rows(c_row, e_row)[0]
    result["ci_low"], result["ci_high"] = _percentile_ci(rhos)
    return result


# ======================================================================
# Retrospective coupling/epsilon sensitivity (not a gate)
# ======================================================================


def _descriptive_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Pooled rank association only; no p-value or inference for dependent records."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def _refresh_log_residuals(
    pairs: dict[str, dict[str, float]],
) -> tuple[list[float], list[float]]:
    """Project both logs on intercept + receiver + source on the same positive cells."""
    keys = [
        key
        for key, values in pairs.items()
        if all(math.isfinite(values[f]) and values[f] > 0 for f in ("coupling", "eps"))
    ]
    if not keys:
        return [], []
    groups = sorted({group for key in keys for group in key.split("->")})
    design = np.asarray(
        [
            [1]
            + [int(key.split("->")[0] == g) for g in groups[1:]]
            + [int(key.split("->")[1] == g) for g in groups[1:]]
            for key in keys
        ],
        dtype=float,
    )
    residuals = []
    for field in ("coupling", "eps"):
        y = np.asarray([math.log(pairs[key][field]) for key in keys])
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        # Do not rank floating-point projection noise from a constant/exact fit.
        if np.max(np.abs(residual)) <= 64 * np.finfo(float).eps * max(
            1.0, np.max(np.abs(y))
        ):
            residual[:] = 0.0
        residuals.append(residual.tolist())
    return residuals[0], residuals[1]


def coupling_sensitivity(method: dict[str, Any]) -> dict[str, Any]:
    r"""Per-seed retrospective associations on cross_term, independent of gamma.

    raw: finite ordered (C, eps), exactly the historical raw observation domain;
    zeros/non-positive values remain. symmetric: one lexically canonical unordered
    pair per refresh, requiring both orientations with finite non-negative C and eps;
    compare C to sqrt(eps_vw eps_wv), retaining defined zero targets. Reverse C must
    agree (relative tolerance 1e-12); inconsistent finite couplings raise ValueError.
    adjusted: finite strictly positive C AND eps on the same ordered cells; fit
    BOTH logs separately on intercept + receiver + source within EACH refresh,
    then pool residual ranks within seed, never across seeds. Exact-fit roundoff
    at <= 64 machine eps times max(1, max|log values|) is zero, not rank signal.

    Each domain reports n_pairs, n_candidates, n_skipped and n_records with at
    least one retained observation. Symmetric candidates are the union of observed
    unordered keys; n_missing_reverse is a subset of n_skipped. No missing reverse
    is imputed. Fewer than three observations or a constant variable gives NaN rho.
    No p-values, bootstrap or criterion changes; association is not selection utility.
    """
    result: dict[str, Any] = {}
    for seed, payload in _sorted_seeds(method["seeds"]):
        values: dict[str, tuple[list[float], list[float]]] = {
            domain: ([], []) for domain in ("raw", "symmetric", "adjusted")
        }
        counts = {domain: {"n_candidates": 0, "n_records": 0} for domain in values}
        missing_reverse = 0
        for record in payload["cross_term"]:
            pairs = record["pairs"]
            unordered = set()
            for key in pairs:
                blocks = key.split("->")
                if len(blocks) != 2 or not all(blocks) or blocks[0] == blocks[1]:
                    raise ValueError(f"invalid off-diagonal ordered pair {key!r}")
                unordered.add(tuple(sorted(blocks)))
            before = {domain: len(x) for domain, (x, _) in values.items()}
            counts["raw"]["n_candidates"] += len(pairs)
            counts["adjusted"]["n_candidates"] += len(pairs)
            counts["symmetric"]["n_candidates"] += len(unordered)
            for row in pairs.values():
                c, e = row["coupling"], row["eps"]
                if math.isfinite(c) and math.isfinite(e):
                    values["raw"][0].append(c)
                    values["raw"][1].append(e)
            for v, w in sorted(unordered):
                forward, reverse = pairs.get(f"{v}->{w}"), pairs.get(f"{w}->{v}")
                if forward is None or reverse is None:
                    missing_reverse += 1
                    continue
                c, cr = forward["coupling"], reverse["coupling"]
                if (
                    math.isfinite(c)
                    and math.isfinite(cr)
                    and not math.isclose(c, cr, rel_tol=1e-12)
                ):
                    raise ValueError(
                        f"{seed}: inconsistent symmetric coupling for {v!r}, {w!r}"
                    )
                e, er = forward["eps"], reverse["eps"]
                if all(math.isfinite(x) and x >= 0 for x in (c, cr, e, er)):
                    values["symmetric"][0].append(c)
                    values["symmetric"][1].append(math.sqrt(e) * math.sqrt(er))
            c_residuals, e_residuals = _refresh_log_residuals(pairs)
            values["adjusted"][0].extend(c_residuals)
            values["adjusted"][1].extend(e_residuals)
            for domain, (x, _) in values.items():
                counts[domain]["n_records"] += int(len(x) > before[domain])
        result[seed] = {}
        for domain, (x, y) in values.items():
            result[seed][domain] = {
                **counts[domain],
                "n_pairs": len(x),
                "n_skipped": counts[domain]["n_candidates"] - len(x),
                "rho": _descriptive_rho(x, y),
            }
        result[seed]["symmetric"]["n_missing_reverse"] = missing_reverse
    return result


# ======================================================================
# Eps variance decomposition
# ======================================================================


def _eps_additive_fit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The traversal of ``records`` and the least-squares fit behind
    `eps_variance_decomposition`: the (row, col, log eps) rows and the mu + a_v + b_w
    design, built once.

    Returns everything `eps_variance_decomposition` reports (``residual_share``,
    ``additive_share``, ``row_effect_sd``, ``col_effect_sd``, ``n_used``,
    ``n_skipped`` -- see that function's own docstring for the model, the gauge
    freedom and what each one means). Both shares and both effect standard
    deviations are NaN under the same ``n_used == 0`` / zero-spread conditions that
    function already documents.
    """
    rows: list[str] = []
    cols: list[str] = []
    logs: list[float] = []
    n_skipped = 0
    for record in records:
        for pair, values in record["pairs"].items():
            eps = values["eps"]
            if not (math.isfinite(eps) and eps > 0.0):
                n_skipped += 1
                continue
            v, _, w = pair.partition("->")
            rows.append(v)
            cols.append(w)
            logs.append(math.log(eps))

    n_used = len(logs)
    result: dict[str, Any] = {
        "residual_share": math.nan,
        "additive_share": math.nan,
        "row_effect_sd": math.nan,
        "col_effect_sd": math.nan,
        "n_used": n_used,
        "n_skipped": n_skipped,
    }
    if n_used == 0:
        return result

    y = np.asarray(logs, dtype=float)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    if ss_total <= 0.0:
        return result

    row_names = sorted(set(rows))
    col_names = sorted(set(cols))
    row_index = {name: i for i, name in enumerate(row_names)}
    col_index = {name: i for i, name in enumerate(col_names)}
    n_row, n_col = len(row_names), len(col_names)

    design = np.zeros((n_used, 1 + n_row + n_col))
    design[:, 0] = 1.0
    for i, v in enumerate(rows):
        design[i, 1 + row_index[v]] = 1.0
    for i, w in enumerate(cols):
        design[i, 1 + n_row + col_index[w]] = 1.0

    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    ss_resid = float(np.sum(residuals**2))

    result["residual_share"] = ss_resid / ss_total
    result["additive_share"] = 1.0 - result["residual_share"]
    result["row_effect_sd"] = float(np.std(coef[1 : 1 + n_row]))
    result["col_effect_sd"] = float(np.std(coef[1 + n_row :]))
    return result


def eps_variance_decomposition(records: list[dict[str, Any]]) -> dict[str, float]:
    """Historical pooled epsilon-only fit, NOT the retrospective two-log adjustment.

    How much of the spread of log eps(v, w) the additive model mu + a_v + b_w
    explains, and how much is left over for pair-specific (v, w) structure.

    a_v, b_w and mu are fit by ordinary least squares on the indicator design (one
    column per row level, one per column level, plus the intercept): this is the
    orthogonal projection of log eps onto the additive model, so residual_share is the
    same number no matter the order rows and columns are considered in, or which of
    the two is called "row". (The fit is rank-deficient by exactly the mu/a_v/b_w gauge
    freedom - shifting a constant between mu and every a_v, or between mu and every
    b_w, leaves the fitted values unchanged - so `numpy.linalg.lstsq` returns one of
    several equally-valid (a_v, b_w); `row_effect_sd` and `col_effect_sd` are invariant
    to that choice because every gauge direction shifts a whole group by one constant,
    which does not change its spread.)

    The pairs are the diagonal-free grid: v -> w is recorded for every v != w,
    and v == v never is. With at least three groups and all off-diagonal cells,
    the receiver/source design is connected and its effects are identifiable up
    to the intercept gauges above. Missing cells may add non-identifiability, but
    fitted values and residuals remain the unique orthogonal projection.
    residual_share = SSres / SStot; additive_share = 1 - residual_share describes
    additive block effects rather than departures specific to an observed pair.

    eps(v, w) = ||H_vw d_w|| / ||H_vv d_v||, so -log||H_vv d_v||, a pure row effect, is
    part of a_v - but a_v also absorbs any systematic scale of row v's numerators
    ||H_vw d_w|| across w (e.g. v having systematically larger or smaller neighbors,
    not just a small ||H_vv d_v||). A small residual share describes departures
    from the additive fit; it does not bound all information in cross-block structure,
    since systematic numerator variation is also absorbed by the additive effects.

    Entries with non-finite or non-positive ``eps`` are dropped (undefined in log) and
    counted in ``n_skipped``. Both shares and both effect standard deviations are NaN
    when there is nothing to explain (``n_used == 0``, or every retained ``log eps`` is
    identical).
    """
    fit = _eps_additive_fit(records)
    return {
        "residual_share": fit["residual_share"],
        "additive_share": fit["additive_share"],
        "row_effect_sd": fit["row_effect_sd"],
        "col_effect_sd": fit["col_effect_sd"],
        "n_used": fit["n_used"],
        "n_skipped": fit["n_skipped"],
    }


# ======================================================================
# Gates
# ======================================================================


def gate_a(
    exp8: dict[str, Any],
    *,
    resnet: str = "resnet18",
    plain: str = "plain_resnet18",
    checkpoint: str = "final",
    min_mean_delta: float = 0.05,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Gate A (inversion): Phi_resnet - Phi_plain > 0 on every seed, mean >= 0.05."""
    missing = [
        f"{arch}/{checkpoint}"
        for arch in (resnet, plain)
        if arch not in exp8 or checkpoint not in exp8[arch]
    ]
    result: dict[str, Any] = {
        "pass": False,
        "missing": missing,
        "delta": {},
        "mean_delta": math.nan,
        "n_seeds": 0,
        "ci_low": math.nan,
        "ci_high": math.nan,
    }
    if missing:
        return result

    seeds_r = exp8[resnet][checkpoint]["seeds"]
    seeds_p = exp8[plain][checkpoint]["seeds"]
    common = sorted(set(seeds_r) & set(seeds_p), key=seed_number)
    delta = {s: seeds_r[s]["phi"] - seeds_p[s]["phi"] for s in common}
    values = np.array(list(delta.values()), dtype=float)
    result.update({"delta": delta, "n_seeds": int(values.size)})
    if values.size == 0:
        return result

    mean = float(values.mean())
    result["mean_delta"] = mean
    if values.size >= 5:
        rng = np.random.default_rng(seed)
        means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(
            axis=1
        )
        result["ci_low"], result["ci_high"] = _percentile_ci(means)
    result["pass"] = bool(np.all(values > 0.0)) and mean >= min_mean_delta
    return result


def gate_a_replicated(
    bundles: Sequence[dict[str, Any]],
    *,
    resnet: str = "resnet18",
    plain: str = "plain_resnet18",
    checkpoint: str = "final",
) -> dict[str, Any]:
    """Gate A's delta_s pooled over several same-configuration runs (e.g. the
    pre-registered run and a same-config replication).

    Not a gate: no pass/fail and no minimum-delta threshold, just the pooled mean, sd
    and a 95% t-interval `gate_a` itself does not compute, over the n = sum of every
    bundle's seeds (run, seed) samples. Each bundle contributes its own `gate_a` (its
    pre-registered semantics are unchanged; its own bootstrap CI is skipped here with
    n_boot=0, since this function computes its own interval instead). ``missing`` is
    the union of every bundle's ``missing``; the interval uses the sample standard
    deviation (ddof=1) and ``scipy.stats.t.ppf`` with n - 1 degrees of freedom, and is
    NaN below 2 pooled samples.

    The runs being pooled share seeds (e.g. 42, 43, 44 for the A1 pair, run only once
    each with different GPU nondeterminism): the n = runs * seeds values are pairs
    sharing initialization and data order, not that many independent draws, so the
    pooled interval above assumes more independence than the design has and can
    understate its width. ``per_seed_mean`` (mean delta per seed, across runs),
    ``n_seeds`` and the ``ci_low_seeds`` / ``ci_high_seeds`` 95% t-interval over those
    n_seeds means are the conservative alternative: never more independent samples
    than there are seeds, at the cost of throwing away the run-to-run replication. A
    bundle missing a seed some other bundle has would already show up in ``missing``
    above (checked per bundle before any of this), so pairing the bundles' deltas by
    seed here is well defined.
    """
    per_bundle = [
        gate_a(bundle, resnet=resnet, plain=plain, checkpoint=checkpoint, n_boot=0)
        for bundle in bundles
    ]
    missing = [m for gate in per_bundle for m in gate["missing"]]
    result: dict[str, Any] = {
        "missing": missing,
        "per_bundle": per_bundle,
        "mean": math.nan,
        "sd": math.nan,
        "n": 0,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "per_seed_mean": {},
        "n_seeds": 0,
        "ci_low_seeds": math.nan,
        "ci_high_seeds": math.nan,
    }
    if missing:
        return result

    pooled = np.array(
        [delta for gate in per_bundle for delta in gate["delta"].values()],
        dtype=float,
    )
    n = int(pooled.size)
    result["n"] = n
    if n > 0:
        mean = float(pooled.mean())
        result["mean"] = mean
        if n >= 2:
            sd = float(pooled.std(ddof=1))
            result["sd"] = sd
            se = sd / math.sqrt(n)
            if se > 0.0:
                t_crit = float(stats.t.ppf(0.975, df=n - 1))
                result["ci_low"] = mean - t_crit * se
                result["ci_high"] = mean + t_crit * se
            else:
                result["ci_low"] = result["ci_high"] = mean

    # Pair by seed: the seeds every bundle's delta has in common (see the docstring -
    # a seed absent from one bundle but not another would already be in `missing`).
    seed_keys = set(per_bundle[0]["delta"]) if per_bundle else set()
    for gate in per_bundle[1:]:
        seed_keys &= set(gate["delta"])
    ordered_seeds = sorted(seed_keys, key=seed_number)
    n_seeds = len(ordered_seeds)
    result["n_seeds"] = n_seeds
    if n_seeds == 0:
        return result

    per_seed_values = {
        s: np.array([gate["delta"][s] for gate in per_bundle], dtype=float)
        for s in ordered_seeds
    }
    per_seed_mean = {s: float(values.mean()) for s, values in per_seed_values.items()}
    result["per_seed_mean"] = per_seed_mean

    seed_means = np.array(list(per_seed_mean.values()), dtype=float)
    if n_seeds >= 2:
        mean_seeds = float(seed_means.mean())
        sd_seeds = float(seed_means.std(ddof=1))
        se_seeds = sd_seeds / math.sqrt(n_seeds)
        if se_seeds > 0.0:
            t_crit_seeds = float(stats.t.ppf(0.975, df=n_seeds - 1))
            result["ci_low_seeds"] = mean_seeds - t_crit_seeds * se_seeds
            result["ci_high_seeds"] = mean_seeds + t_crit_seeds * se_seeds
        else:
            result["ci_low_seeds"] = result["ci_high_seeds"] = mean_seeds

    return result


def _phi_estimate(payload: dict[str, Any], probes: str) -> float:
    """The chosen Phi estimate off one seed record: ``phi_by_probes[probes]``, falling
    back to the headline ``phi`` when the probe sweep, or that probe count in it, is
    absent (including when ``probes`` is not among the counts swept)."""
    by_probes = payload.get("phi_by_probes")
    if by_probes and probes in by_probes:
        return float(by_probes[probes])
    return float(payload["phi"])


def accuracy_matched_delta(
    exp8: dict[str, Any],
    *,
    value: Callable[[dict[str, Any]], float],
    resnet: str = "resnet18",
    plain: str = "plain_resnet18",
    tol: float = 0.02,
) -> dict[str, Any]:
    """Accuracy-matched checkpoint pairing (see the module
    docstring), differencing ``value`` -- a scalar read off one seed's checkpoint
    record -- rather than Phi specifically. `accuracy_matched_delta_phi` is this
    function's Phi-reading specialization (``value=lambda payload:
    _phi_estimate(payload, probes)``); a second quantity that must be compared over
    the *same* accuracy-matched pairs  calls this function directly with its own
    ``value`` instead of re-deriving the pairing.

    For each seed and each ``plain`` checkpoint other than ``init``, pairs it with the
    ``resnet`` checkpoint of the *same seed* whose ``test_acc`` is nearest (over every
    resnet checkpoint that carries this seed, ``init`` included; an exact tie in
    ``|test_acc|`` distance goes to the checkpoint listed first in the bundle); the pair
    is kept when ``|test_acc_resnet - test_acc_plain| <= tol``, dropped otherwise. The
    differenced value at that pair is ``value(resnet_seed_record) -
    value(plain_seed_record)``; pairs are averaged within a seed, and those per-seed
    means are then averaged across seeds for ``mean``, the sample standard deviation
    (``ddof=1``) and a 95% t-interval (``scipy.stats.t.ppf``, ``n_seeds - 1`` degrees
    of freedom). A seed that keeps no pair contributes to neither ``per_seed_mean``
    nor ``n_seeds``; a seed one architecture's checkpoint lacks - a shard the merge
    never completed - is skipped for that checkpoint rather than raising, on either
    side.

    Returns ``{"pairs": {seed: [(plain_label, resnet_label, acc_plain, acc_resnet,
    delta),...]}, "per_seed_mean": {seed: mean}, "mean", "sd", "n_seeds", "ci_low",
    "ci_high", "n_pairs"}`` (``n_pairs`` is the total kept over every seed).
    """
    resnet_ckpts = exp8[resnet]
    plain_ckpts = exp8[plain]
    seeds = sorted(
        {seed for entry in plain_ckpts.values() for seed in entry["seeds"]},
        key=seed_number,
    )

    pairs: dict[str, list[tuple[str, str, float, float, float]]] = {}
    per_seed_mean: dict[str, float] = {}
    for seed in seeds:
        seed_pairs: list[tuple[str, str, float, float, float]] = []
        for plain_label, entry in plain_ckpts.items():
            if plain_label == "init" or seed not in entry["seeds"]:
                continue
            acc_plain = entry["seeds"][seed]["test_acc"]
            candidates = [
                label for label in resnet_ckpts if seed in resnet_ckpts[label]["seeds"]
            ]
            if not candidates:
                continue
            resnet_label = min(
                candidates,
                key=lambda label: abs(
                    resnet_ckpts[label]["seeds"][seed]["test_acc"] - acc_plain
                ),
            )
            acc_resnet = resnet_ckpts[resnet_label]["seeds"][seed]["test_acc"]
            if abs(acc_resnet - acc_plain) > tol:
                continue
            delta = value(resnet_ckpts[resnet_label]["seeds"][seed]) - value(
                entry["seeds"][seed]
            )
            seed_pairs.append((plain_label, resnet_label, acc_plain, acc_resnet, delta))
        pairs[seed] = seed_pairs
        if seed_pairs:
            per_seed_mean[seed] = sum(p[4] for p in seed_pairs) / len(seed_pairs)

    values = np.array(list(per_seed_mean.values()), dtype=float)
    n_seeds = int(values.size)
    result: dict[str, Any] = {
        "pairs": pairs,
        "per_seed_mean": per_seed_mean,
        "mean": math.nan,
        "sd": math.nan,
        "n_seeds": n_seeds,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "n_pairs": sum(len(p) for p in pairs.values()),
    }
    if n_seeds == 0:
        return result

    mean = float(values.mean())
    result["mean"] = mean
    if n_seeds >= 2:
        sd = float(values.std(ddof=1))
        result["sd"] = sd
        se = sd / math.sqrt(n_seeds)
        if se > 0.0:
            t_crit = float(stats.t.ppf(0.975, df=n_seeds - 1))
            result["ci_low"] = mean - t_crit * se
            result["ci_high"] = mean + t_crit * se
        else:
            result["ci_low"] = result["ci_high"] = mean
    return result


def accuracy_matched_delta_phi(
    exp8: dict[str, Any],
    *,
    resnet: str = "resnet18",
    plain: str = "plain_resnet18",
    tol: float = 0.02,
    probes: str = "300",
) -> dict[str, Any]:
    """Delta Phi at accuracy-matched checkpoint pairs: `accuracy_matched_delta`
    passing Phi (`_phi_estimate` at ``probes``) as the differenced ``value`` - the
    accuracy-matched replication of the inversion, not gate A.

    Delta Phi is ``phi_resnet - phi_plain`` at the chosen pair, read by
    `_phi_estimate` at ``probes``. See `accuracy_matched_delta` for the pairing rule,
    the seed-then-across-seed averaging, and every returned field.
    """
    return accuracy_matched_delta(
        exp8,
        value=lambda payload: _phi_estimate(payload, probes),
        resnet=resnet,
        plain=plain,
        tol=tol,
    )


def same_label_delta_phi(
    exp8: dict[str, Any],
    *,
    resnet: str = "resnet18",
    plain: str = "plain_resnet18",
    probes: str = "300",
) -> dict[str, dict[str, Any]]:
    """Delta Phi = Phi_resnet - Phi_plain at *equal* checkpoint labels (same nominal
    schedule position, not accuracy-matched - the companion `accuracy_matched_delta_phi`
    reports next to this one).

    Every label the two architectures share; per label, mean and sample standard
    deviation (``ddof=1``) of ``phi_resnet - phi_plain`` over the seeds they share
    (read at ``probes`` by `_phi_estimate`), in numeric seed order. Returns
    ``{label: {"mean", "sd", "n", "deltas"}}``; ``sd`` is NaN below 2 seeds.
    """
    resnet_ckpts = exp8[resnet]
    plain_ckpts = exp8[plain]
    result: dict[str, dict[str, Any]] = {}
    for label, plain_entry in plain_ckpts.items():
        if label not in resnet_ckpts:
            continue
        resnet_entry = resnet_ckpts[label]
        seeds = sorted(
            set(resnet_entry["seeds"]) & set(plain_entry["seeds"]), key=seed_number
        )
        deltas = [
            _phi_estimate(resnet_entry["seeds"][s], probes)
            - _phi_estimate(plain_entry["seeds"][s], probes)
            for s in seeds
        ]
        values = np.array(deltas, dtype=float)
        n = int(values.size)
        result[label] = {
            "mean": float(values.mean()) if n else math.nan,
            "sd": float(values.std(ddof=1)) if n >= 2 else math.nan,
            "n": n,
            "deltas": deltas,
        }
    return result


def gate_b(
    exp7: dict[str, Any],
    *,
    base: str = "kfac",
    overlay: str = "kfac+overlay",
    other_arms: tuple[str, ...] = ("ekfac+overlay",),
    min_gamma: float = 0.2,
    acc_tolerance: float = 0.005,
    max_p: float = 0.01,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Gate B (payoff): gap closed, speed to target, no accuracy regression, rho_s.

    The pre-registered rho_s sub-criterion names no method; it is evaluated on
    ``overlay`` alone (unchanged pass/fail), but ``spearman_by_arm`` also reports
    `spearman_coupling_eps` for every arm of ``other_arms`` present in ``exp7``
    (including ``overlay`` itself), so a caller can show every arm's rho_s.
    """
    missing = [m for m in (base, overlay) if m not in exp7]
    result: dict[str, Any] = {
        "pass": False,
        "missing": missing,
        "checks": {},
        "gap_closed": None,
        "spearman": None,
        "spearman_by_arm": {},
        "epochs_to_target_median": {},
        "final_acc_delta": {},
        "min_final_acc_delta": math.nan,
        "n_seeds": 0,
    }
    if missing:
        return result

    gamma = gap_closed(exp7[overlay], n_boot=n_boot, seed=seed)
    rho = spearman_coupling_eps(exp7[overlay], n_boot=n_boot, seed=seed)
    spearman_by_arm = {overlay: rho}
    for arm in other_arms:
        if arm != overlay and arm in exp7:
            spearman_by_arm[arm] = spearman_coupling_eps(
                exp7[arm], n_boot=n_boot, seed=seed
            )
    seeds_b = exp7[base]["seeds"]
    seeds_o = exp7[overlay]["seeds"]
    common = sorted(set(seeds_b) & set(seeds_o), key=seed_number)

    def epochs(payload: dict[str, Any]) -> float:
        value = payload["epochs_to_target"]
        return math.inf if value is None else float(value)

    median_base = _median([epochs(seeds_b[s]) for s in common])
    median_overlay = _median([epochs(seeds_o[s]) for s in common])
    acc_delta = {s: seeds_o[s]["final_acc"] - seeds_b[s]["final_acc"] for s in common}
    min_delta = min(acc_delta.values()) if acc_delta else math.nan

    checks = {
        "gap_closed": gamma["median"] >= min_gamma and gamma["ci_low"] > 0.0,
        "epochs_to_target": bool(common) and median_overlay <= median_base,
        "final_acc": bool(common) and min_delta >= -acc_tolerance,
        "spearman": rho["rho"] > 0.0 and rho["p"] < max_p,
    }
    result.update(
        {
            "pass": all(checks.values()),
            "checks": checks,
            "gap_closed": gamma,
            "spearman": rho,
            "spearman_by_arm": spearman_by_arm,
            "epochs_to_target_median": {base: median_base, overlay: median_overlay},
            "final_acc_delta": acc_delta,
            "min_final_acc_delta": min_delta,
            "n_seeds": len(common),
        }
    )
    return result


def gate_c(
    exp7: dict[str, Any],
    *,
    methods: Sequence[str] | None = None,
    refresh_period: int = 10,
    max_failed_ratio: float = 0.05,
) -> dict[str, Any]:
    """Gate C (EKFAC): failed eigenbasis refreshes < 5% and no NaN validation loss."""
    if methods is None:
        methods = [m for m in data_keys(exp7) if m.startswith("ekfac")]
    missing = [m for m in methods if m not in exp7] if methods else ["ekfac"]
    result: dict[str, Any] = {
        "pass": False,
        "missing": missing,
        "methods": {},
        "max_failed_ratio": math.nan,
        "any_nan_val_loss": False,
        "refresh_period": refresh_period,
    }
    if missing:
        return result

    per_method: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    any_nan = False
    for method in methods:
        per_method[method] = {}
        for name, payload in _sorted_seeds(exp7[method]["seeds"]):
            n_refreshes = payload["n_steps"] / refresh_period
            failed = payload["safeguard"]["n_refresh_failed"]
            ratio = failed / n_refreshes if n_refreshes > 0 else math.nan
            nan_loss = any(math.isnan(v) for v in payload["val_loss"])
            per_method[method][name] = {
                "failed_ratio": ratio,
                "n_refresh_failed": failed,
                "nan_in_val_loss": nan_loss,
            }
            ratios.append(ratio)
            any_nan = any_nan or nan_loss
    max_ratio = max(ratios) if ratios else math.nan
    result.update(
        {
            "pass": max_ratio < max_failed_ratio and not any_nan,
            "methods": per_method,
            "max_failed_ratio": max_ratio,
            "any_nan_val_loss": any_nan,
        }
    )
    return result


def _split_block(name: str) -> tuple[int | None, str]:
    """``"L3.attn_v" -> (3, "attn_v")``; blocks outside a layer (embed, head) -> None."""
    layer, dot, block = name.partition(".")
    if dot and layer.startswith("L") and layer[1:].isdigit():
        return int(layer[1:]), block
    return None, name


def pair_role(pair: str) -> str | None:
    """Structural role of a coupling pair ``"Li.<block>->Lj.<block>"`` (None if none).

    Within a layer: ``v->o`` (attn_v, attn_o), ``q<->k`` (attn_q, attn_k), ``o->mlp``
    (attn_o, mlp). Across adjacent layers: ``mlp->next.attn_q`` (mlp of layer i, attn_q of
    layer i + 1) or ``adjacent.<block>`` (the same block in layers i and i + 1). Either
    key order is accepted; the bundles key pairs v-before-w in group order.
    """
    v, arrow, w = pair.partition("->")
    if not arrow:
        return None
    layer_v, block_v = _split_block(v)
    layer_w, block_w = _split_block(w)
    if layer_v is None or layer_w is None:
        return None
    if layer_v == layer_w:
        blocks = {block_v, block_w}
        if blocks == {"attn_v", "attn_o"}:
            return "v->o"
        if blocks == {"attn_q", "attn_k"}:
            return "q<->k"
        if blocks == {"attn_o", "mlp"}:
            return "o->mlp"
        return None
    if abs(layer_v - layer_w) != 1:
        return None
    if layer_v > layer_w:  # orient as (earlier layer, later layer)
        block_v, block_w = block_w, block_v
    if (block_v, block_w) == ("mlp", "attn_q"):
        return "mlp->next.attn_q"
    if block_v == block_w:
        return f"adjacent.{block_v}"
    return None


def top_pairs(entry: dict[str, Any], k: int = 3) -> list[tuple[str, float]]:
    """The ``k`` pairs with the largest mean coupling of an exp9 model entry."""
    items = [
        (pair, float(c["mean"]))
        for pair, c in entry["coupling"].items()
        if math.isfinite(c["mean"])
    ]
    items.sort(key=lambda item: -item[1])
    return items[:k]


def _role_counts(pairs: Sequence[tuple[str, float]]) -> dict[str, int]:
    """How many of ``pairs`` (``(pair, mean coupling)``, strongest first) have each
    structural role, the roles in the order they first appear; a pair without a role
    (`pair_role` is None) counts for none."""
    counts: dict[str, int] = {}
    for pair, _ in pairs:
        role = pair_role(pair)
        if role is not None:
            counts[role] = counts.get(role, 0) + 1
    return counts


def gate_d(
    exp9_toy: dict[str, Any] | None,
    exp9_llm: dict[str, Any] | None,
    *,
    k: int = 3,
    min_matched: int = 2,
) -> dict[str, Any]:
    """Gate D (LLM ladder):

        J:= |Top3(C_toy) intersection Top3(C_llm)| / 3 >= 2/3, where pairs are matched by structural role

    The roles are matched pair by pair, one to one, not as a set. The pairs matched are the
    sum over roles of min(the number of pairs of that role among the top ``k`` of the toy
    ladder's largest model, the number among the LLM's), so three ``v->o`` pairs on each
    side are three matched pairs, J = 3/3, and not one shared role; a pair without a role
    (`pair_role` is None) is matched with none. Per LLM:
    ``role_counts`` (pairs per role in its top ``k``), ``matched_roles`` (pairs matched per
    role), ``n_matched``, ``J`` (the string ``"n_matched/k"``) and ``pass``
    (``n_matched >= min_matched``, i.e. J >= 2/3 for ``k = 3``); the gate passes when every
    LLM does. ``toy_role_counts`` is the same count for the toy model.
    """
    missing = []
    if not (exp9_toy and data_keys(exp9_toy)):
        missing.append("exp9 (toy ladder)")
    if not (exp9_llm and data_keys(exp9_llm)):
        missing.append("exp9_a3_llm")
    result: dict[str, Any] = {
        "pass": False,
        "missing": missing,
        "toy_model": None,
        "toy_top_pairs": [],
        "toy_role_counts": {},
        "llm": {},
    }
    if missing:
        return result
    assert exp9_toy is not None and exp9_llm is not None  # for the type checker

    largest = max(data_keys(exp9_toy), key=lambda m: exp9_toy[m]["n_params"])
    toy_top = top_pairs(exp9_toy[largest], k)
    toy_counts = _role_counts(toy_top)
    per_llm: dict[str, dict[str, Any]] = {}
    for model_id, entry in data_items(exp9_llm):
        top = top_pairs(entry, k)
        counts = _role_counts(top)
        matched = {
            role: min(n, toy_counts[role])
            for role, n in counts.items()
            if role in toy_counts
        }
        n_matched = sum(matched.values())
        per_llm[model_id] = {
            "top_pairs": top,
            "role_counts": counts,
            "matched_roles": matched,
            "n_matched": n_matched,
            "J": f"{n_matched}/{k}",
            "pass": n_matched >= min_matched,
        }
    result.update(
        {
            "pass": bool(per_llm) and all(v["pass"] for v in per_llm.values()),
            "toy_model": largest,
            "toy_top_pairs": toy_top,
            "toy_role_counts": toy_counts,
            "llm": per_llm,
        }
    )
    return result
