"""CPU float64 paired-vector uncertainty for the fixed 12-cell, 36-metric family."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from experiments.crossing.verify import (
    STATISTICS,
    digest,
    metadata_check,
    read_json,
    read_run,
    roster,
    write_json,
)


def quantile(values, probability):
    x = sorted(float(z) for z in values)
    if not x or not 0 <= probability <= 1 or not all(math.isfinite(z) for z in x):
        raise ValueError("invalid quantile domain")
    position = (len(x) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return x[low] + (position - low) * (x[high] - x[low])


def sample_sd(values):
    xs = [float(x) for x in values]
    if len(xs) < 2 or not all(math.isfinite(x) for x in xs):
        raise ValueError("invalid SD domain")
    mean = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def _arrays(components):
    import numpy as np

    if not components:
        raise ValueError("empty components")
    try:
        d = np.asarray([c["diag_sq"] for c in components], dtype=np.float64)
        f = np.asarray([c["full_sq"] for c in components], dtype=np.float64)
    except (TypeError, ValueError, KeyError) as e:
        raise ValueError("malformed components") from e
    if d.ndim != 2 or d.shape[1] < 1 or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("invalid diagonal squared components")
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("invalid full squared components")
    return d, f


def _metrics(d, f):
    import numpy as np

    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("nonregular/invalid pooled diagonal squared component")
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("invalid pooled full denominator")
    total = np.sum(d, axis=-1, dtype=np.float64)
    separable_sq = np.sum(np.sqrt(d), axis=-1, dtype=np.float64) ** 2
    result = np.stack(
        (separable_sq / f, 1 - total / f, 1 - total / separable_sq), axis=-1
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite nonlinear metric")
    return result


def pool(components):
    import numpy as np

    d, f = _arrays(components)
    result = _metrics(
        np.mean(d, axis=0, dtype=np.float64), np.mean(f, dtype=np.float64)
    )
    return {k: float(x) for k, x in zip(("S", "Phi", "Phi_sep"), result, strict=False)}


def resample_indices(rng, cells, resamples=10000):
    # A fresh independent index array for each cell; within a cell share it for all components.
    return [rng.integers(0, 30, size=(resamples, 30)) for _ in range(cells)]


def bootstrap_cell(stage, layer, indices):
    import numpy as np

    result = {}
    for name, components in (("stage", stage), ("layer", layer)):
        d, f = _arrays(components)
        if len(f) != 30:
            raise ValueError("30 complete pairs required")
        result[name] = _metrics(
            np.mean(d[indices], axis=1, dtype=np.float64),
            np.mean(f[indices], axis=1, dtype=np.float64),
        )
    result["delta"] = result["stage"][:, 0] - result["layer"][:, 0]
    return result


def jackknife(components):
    import numpy as np

    d, f = _arrays(components)
    r = len(f)
    if r < 2:
        raise ValueError("jackknife needs at least two vectors")
    point = _metrics(d.mean(axis=0, dtype=np.float64), f.mean(dtype=np.float64))
    loo = np.stack(
        [
            _metrics(
                np.delete(d, i, axis=0).mean(axis=0, dtype=np.float64),
                np.delete(f, i).mean(dtype=np.float64),
            )
            for i in range(r)
        ]
    )
    mean = loo.mean(axis=0, dtype=np.float64)
    return {
        k: {
            "bias": float((r - 1) * (mean[j] - point[j])),
            "se": float(
                np.sqrt(
                    (r - 1) / r * np.sum((loo[:, j] - mean[j]) ** 2, dtype=np.float64)
                )
            ),
        }
        for j, k in enumerate(("S", "Phi", "Phi_sep"))
    }


def paired_delta_jackknife(stage, layer):
    import numpy as np

    values = [
        pool(stage[:i] + stage[i + 1 :])["S"] - pool(layer[:i] + layer[i + 1 :])["S"]
        for i in range(30)
    ]
    point = pool(stage)["S"] - pool(layer)["S"]
    mean = float(np.mean(values, dtype=np.float64))
    return {
        "bias": 29 * (mean - point),
        "se": math.sqrt(29 / 30 * math.fsum((x - mean) ** 2 for x in values)),
    }


def max_screen(observed, boot):
    import numpy as np

    observed, boot = (
        np.asarray(observed, dtype=np.float64),
        np.asarray(boot, dtype=np.float64),
    )
    if observed.shape != (36,) or boot.ndim != 2 or boot.shape[1] != 36:
        raise ValueError("fixed 36-metric family required")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(boot)):
        raise ValueError("nonfinite family metric")
    sd = np.std(boot, axis=0, ddof=1, dtype=np.float64)
    # Exact constant empirical draws are zero-scale even if a mean reduction leaves roundoff residue.
    if (
        np.any(np.ptp(boot, axis=0) == 0)
        or np.any(sd <= 0)
        or not np.all(np.isfinite(sd))
    ):
        raise ValueError("zero/nonfinite family bootstrap SD")
    maxima = np.max(np.abs(boot - observed) / sd, axis=1)
    critical = quantile(maxima, 0.95)
    return {
        "critical_value": critical,
        "fixed_sd": sd.tolist(),
        "intervals": np.stack(
            (observed - critical * sd, observed + critical * sd), axis=1
        ).tolist(),
    }


def classify(stage_interval, layer_interval):
    if stage_interval[0] > 1 and layer_interval[1] < 1:
        return "resolved_crossing"
    if stage_interval[1] < 1 and layer_interval[0] > 1:
        return "resolved_reverse_crossing"
    return "unresolved"


def _marginal(point, values):
    interval = [quantile(values, p) for p in (0.025, 0.975)]
    return {
        "estimate": point,
        "bootstrap_mc_se": sample_sd(values),
        "percentile_95": interval,
        "half_width": (interval[1] - interval[0]) / 2,
        "descriptive_half_width_target": 0.03,
        "target_met": (interval[1] - interval[0]) / 2 <= 0.03,
    }


def analyze(cells: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    order = roster()
    registry = [
        f"{c['id']}:{k}" for c in order for k in ("S_stage", "S_layer", "DeltaS")
    ]
    result: dict[str, Any] = {
        "statistics": dict(STATISTICS),
        "cells": [],
        "family": {"metrics": registry},
        "input_comparisons": [],
        "interpretation": [
            "NEW estimates, not historical Table 2 error bars.",
            "Conditional Monte Carlo precision does not bound total float32 forward/HVP/reduction error.",
            "One fixed disjoint input comparison; no iid batch or training-seed population coverage.",
            "The max-bootstrap family screen is approximate, not rigorous FWER or population coverage.",
            "Jackknife diagnostics do not bias-correct point estimates; probes never extend for width/sign.",
        ],
    }
    rng = np.random.default_rng(20261020)
    indices = resample_indices(
        rng, 12
    )  # generate even for incomplete cells, do not outcome-select streams
    lookup = {c["id"]: c for c in cells}
    reasons = []
    if len(lookup) != len(cells) or set(lookup) != {c["id"] for c in order}:
        reasons.append("input cell roster differs from fixed 12")
    observed, boot_columns = [], []
    for index, fixed in enumerate(order):
        c = lookup.get(fixed["id"], {"status": "not_started", "pairs": []})
        pairs = c.get("pairs", [])
        out = dict(
            fixed,
            status=c["status"],
            complete_pairs=len(pairs),
            reasons=list(c.get("integrity_errors", [])),
            crossing="screen_unavailable",
        )
        result["cells"].append(out)
        try:
            if out["reasons"]:
                raise ValueError("record integrity errors")
            if c["status"] != "complete" or len(pairs) != 30:
                raise ValueError("incomplete cell: 30 complete pairs required")
            if [p["replicate"] for p in pairs] != list(range(30)):
                raise ValueError("replicate identity/order mismatch")
            stage, layer = [p["stage"] for p in pairs], [p["layer"] for p in pairs]
            points = {k: pool(v) for k, v in (("stage", stage), ("layer", layer))}
            b = bootstrap_cell(stage, layer, indices[index])
            # Jackknife may fail on a leave-one-out nonregular domain without invalidating a valid bootstrap.
            diagnostics = {}
            for name, components in (("stage", stage), ("layer", layer)):
                try:
                    diagnostics[name] = jackknife(components)
                except ValueError as e:
                    diagnostics[name] = {"unavailable": str(e)}
            try:
                diagnostics["DeltaS"] = paired_delta_jackknife(stage, layer)
            except ValueError as e:
                diagnostics["DeltaS"] = {"unavailable": str(e)}
            out["partitions"] = {
                name: {
                    k: _marginal(points[name][k], b[name][:, j])
                    for j, k in enumerate(("S", "Phi", "Phi_sep"))
                }
                for name in ("stage", "layer")
            }
            out["pooled_components"] = {}
            for name, components in (("stage", stage), ("layer", layer)):
                d, f = _arrays(components)
                out["pooled_components"][name] = {
                    "diag_sq": d.mean(axis=0, dtype=np.float64).tolist(),
                    "full_sq": float(f.mean(dtype=np.float64)),
                }
            delta = points["stage"]["S"] - points["layer"]["S"]
            out["DeltaS"] = _marginal(delta, b["delta"])
            out["jackknife"] = diagnostics
            out["marginal_domain"] = (
                "Complete cell; all original and bootstrap pooled domains finite and regular. Approximate conditional MC intervals."
            )
            observed.extend([points["stage"]["S"], points["layer"]["S"], delta])
            boot_columns.extend([b["stage"][:, 0], b["layer"][:, 0], b["delta"]])
        except (ValueError, TypeError, KeyError) as e:
            out["reasons"].append(str(e))
            reasons.append(f"{fixed['id']}: {e}")
            out["marginal_domain"] = (
                "Unavailable; no draws deleted, clipped or replaced."
            )
    if not reasons:
        try:
            screen = max_screen(observed, np.stack(boot_columns, axis=1))
            result["family"].update(status="available", **screen)
            for i, out in enumerate(result["cells"]):
                intervals = screen["intervals"][3 * i : 3 * i + 3]
                out["screen_95"] = dict(
                    zip(("S_stage", "S_layer", "DeltaS"), intervals, strict=False)
                )
                out["crossing"] = classify(intervals[0], intervals[1])
        except ValueError as e:
            reasons.append(str(e))
    if reasons:
        result["family"].update(
            status="unavailable", reasons=reasons, critical_value=None
        )
    result["complete"] = all(
        c["status"] == "complete" and c["complete_pairs"] == 30 and not c["reasons"]
        for c in result["cells"]
    )
    for seed in (42, 43, 44):
        for model in ("resnet18", "plain_resnet18"):
            a, b = [
                next(
                    c
                    for c in result["cells"]
                    if c["seed"] == seed and c["model"] == model and c["batch"] == batch
                )
                for batch in ("B0", "B1")
            ]
            row: dict[str, Any] = {
                "seed": seed,
                "model": model,
                "interpretation": "One fixed disjoint-input replication, same saved checkpoint; not an iid sample.",
            }
            if "partitions" in a and "partitions" in b:
                row["B1_minus_B0_S"] = {
                    p: b["partitions"][p]["S"]["estimate"]
                    - a["partitions"][p]["S"]["estimate"]
                    for p in ("stage", "layer")
                }
                row["B1_minus_B0_DeltaS"] = (
                    b["DeltaS"]["estimate"] - a["DeltaS"]["estimate"]
                )
            else:
                row["unavailable"] = "one or both cells incomplete/invalid"
            result["input_comparisons"].append(row)
    result["N02"] = (
        "awaits independent result review"
        if result["complete"] and result["family"]["status"] == "available"
        else "OPEN: incomplete/invalid family"
    )
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    m = read_json(args.manifest)
    metadata = metadata_check(m)
    if metadata["status"] != "PASS":
        raise ValueError(metadata["errors"])
    cells, campaign = read_run(m, args.run_dir, digest(args.manifest))
    result = {
        "manifest_sha256": digest(args.manifest),
        "campaign_sha256": digest(args.run_dir / "campaign.json"),
        "metadata_verification": metadata,
        "analysis": analyze(cells),
    }
    write_json(args.output, result)
    print(
        f"Analysis saved exclusively to {args.output}; complete={result['analysis']['complete']}; family={result['analysis']['family']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
