"""Which generated items belong to which paper.

`REGISTRY` in `registry` says how every table and figure is built; this module says who
reads it. One paper is one track: a list of labels, in the order they appear in the
manuscript. The split is the seam between papers - the measurement code and the result
bundles are shared, the manuscripts and their label sets are not - so a paper
regenerates exactly its own items, into its own output directory, and a label that
outlives the section that cited it is caught here rather than in a stale \\input.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .config import DataPaths
from .registry import REGISTRY

# Track -> the labels its manuscript \input's, in manuscript order.
TRACKS: dict[str, list[str]] = {
    "b-ablation": [
        "num:residual-ablation",
        "tab:residual-ablation",
        "tab:residual-controls",
        "tab:residual-geometry",
        "fig:residual-contrast",
    ],
    "b": [
        "num:b",  # inline prose numbers -- regenerated before anything quotes them
        "num:crossing-check",  # conditional crossing diagnostic
        "tab:phi-map",  # Phi against Phi_null across architectures (Stage A)
        "tab:phi-long",  # the A1 inversion on the long, 300-epoch schedule
        "fig:phi-map",
        "tab:two-partitions",  # comparison at two partitions
        "tab:stage-b",  # the overlay as a negative result
        "tab:proposal-decomposition",  # same-record damped/undamped/native contrasts
        "fig:proposal-model-loss",  # all signed model/loss changes by method
        "tab:gates",
        "tab:payoff-budget",
        "tab:coupling-sensitivity",  # retrospective per-seed sensitivity, Appendix A
        "fig:eps-vs-C",
        "tab:validity",  # the mechanism: t* along the trajectory
        "tab:validity-resolution",  # late original cells and separate seam repeat
        "tab:validity-arms",  # t* off the K-FAC trajectory too: SGD, Adam
        "fig:validity",
        "causal_pilot_numbers",  # Appendix D, separate from the trajectory macros
        "causal_pilot_table",
        "tab:conditioning-resnet",  # appendix O
        "tab:crossing-check",  # Appendix E, conditional MC evidence
        "fig:crossing-screen",  # stage/layer joint screen for all twelve cells
    ],
}


def labels_inputs(labels: Iterable[str]) -> list[str]:
    """The result bundles these labels are generated from, sorted and deduplicated."""
    return sorted({exp for label in labels for exp in REGISTRY[label][1]})


def track_inputs(track: str) -> list[str]:
    """The result bundles a track's labels are generated from, sorted and deduplicated."""
    if track not in TRACKS:
        raise KeyError(f"unknown track {track!r}; known tracks: {sorted(TRACKS)}")
    return labels_inputs(TRACKS[track])


# Canonical B requires these retained cohorts and cells.
# Legacy/on-demand generation without --track b keeps OPTIONAL_EXPERIMENTS semantics.
_B_SEEDS = ("seed_42", "seed_43", "seed_44")
_B_PAIR = ("resnet18", "plain_resnet18")
_B_SHORT = ("init", "mid", "final")
_B_LONG = (
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
_B_PHI_CELLS = {
    "exp8": (_B_PAIR, _B_SHORT, _B_SEEDS),
    "exp8_a1_v2": (_B_PAIR, _B_SHORT, _B_SEEDS),
    "exp8_a1_width": (
        ("resnet18_w0.5", "resnet18", "resnet18_w2.0", "resnet34"),
        _B_SHORT,
        _B_SEEDS,
    ),
    "exp8_a1_imagenet32": (_B_PAIR, _B_SHORT, _B_SEEDS),
    "exp8_a1_long": (_B_PAIR, _B_LONG, _B_SEEDS),
    "exp8_a2": (("resnet50",), ("init",), (*_B_SEEDS, "seed_45", "seed_46")),
    "exp8_a2_gran_v3": (("resnet50",), ("init",), _B_SEEDS),
    "exp8_a1_long_gran_m3000": (_B_PAIR, ("final",), _B_SEEDS),
    "exp9": (("tiny", "small", "base"), (), _B_SEEDS),
    "exp9_a3_llm": (("Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"), (), _B_SEEDS),
}


def _b_require(condition: bool, context: str) -> None:
    if not condition:
        raise ValueError(f"track b retained evidence: {context}")


def _b_phi_cell(cell: dict[str, Any], seeds: tuple[str, ...], context: str) -> None:
    _b_require(set(cell["seeds"]) == set(seeds), f"{context}: seed cohort")
    for seed, payload in cell["seeds"].items():
        sizes = payload.get("group_sizes")
        norms = payload["diag_frob"]
        _b_require(
            isinstance(sizes, list)
            and len(sizes) == len(norms)
            and len(sizes) > 1
            and all(type(size) is int and size > 0 for size in sizes),
            f"{context}/{seed}: recorded group_sizes reference missing or invalid",
        )
        _b_require(
            math.isfinite(payload["phi"])
            and all(math.isfinite(v) and v >= 0 for v in norms.values()),
            f"{context}/{seed}: Phi/profile",
        )
    aggregate = (
        cell["phi"]
        if "phi" in cell
        else {"mean": cell["phi_mean"], "std": cell["phi_std"]}
    )
    for key in ("mean", "std"):
        _b_require(math.isfinite(aggregate[key]), f"{context}: Phi {key}")


def _b_validity_probe(
    probe: dict[str, Any],
    grid: list[float],
    context: str,
    *,
    aggregate: bool = False,
    undefined_rho: bool = False,
    infinite_rel_err: set[float] | None = None,
) -> None:
    """The nested measurements read by B's validity reporting, not a full schema."""

    def measurement(value: Any, field: str, *, allow_nan: bool = False) -> None:
        values = (value["mean"], value["std"]) if aggregate else (value,)
        _b_require(
            all(math.isfinite(v) or (allow_nan and math.isnan(v)) for v in values),
            f"{context}: invalid {field}",
        )

    measurement(probe["t_star"], "t_star", allow_nan=True)
    for section, fields in (
        ("decomposition", ("norm_d0", "g_dot_d0", "d0_H_d0", "lambda_norm_d0_sq")),
        ("sharpness", ("curv_along_d0", "lambda_max", "lambda_min")),
        ("payoff", ("m_kfac", "loss_theta", "loss_d0", "norm_d0")),
    ):
        for field in fields:
            measurement(probe[section][field], f"{section}/{field}")
    # The runner records an undefined payoff ratio only for a zero model value.
    # Its aggregate then carries the same NaN (and an undefined spread).
    measurement(
        probe["payoff"]["rho"],
        "payoff/rho",
        allow_nan=undefined_rho or (not aggregate and probe["payoff"]["m_kfac"] == 0),
    )
    _b_require(
        [point["t"] for point in probe["curve_d0"]] == grid,
        f"{context}: incomplete curve_d0",
    )
    for point in probe["curve_d0"]:
        for field in ("dL", "m", "secant"):
            measurement(point[field], f"curve_d0/t={point['t']}/{field}")
        # Infinity is legitimate for zero/nonzero, or overflow of the producer's
        # relative-error quotient on finite scalars. It fails the prefix, but is
        # not missing/invalid evidence. Zero/zero instead has relative error zero.
        if not aggregate and point["rel_err"] == math.inf:
            actual, predicted = point["dL"], point["m"]
            expected = (
                (0.0 if predicted == 0 else math.inf)
                if actual == 0
                else abs(actual - predicted) / abs(actual)
            )
            if expected == math.inf:
                continue
        # Only already validated raw infinite-error cases justify this aggregate.
        if aggregate and infinite_rel_err and point["t"] in infinite_rel_err:
            _b_require(
                point["rel_err"]["mean"] == math.inf
                and math.isnan(point["rel_err"]["std"]),
                f"{context}: invalid curve_d0/t={point['t']}/rel_err",
            )
            continue
        measurement(point["rel_err"], f"curve_d0/t={point['t']}/rel_err")
    at_one = [row for row in probe["kinks"]["rows"] if row["t"] == 1.0]
    _b_require(len(at_one) == 1, f"{context}: kinks must contain one t=1 point")
    measurement(at_one[0]["relu_frac"], "kinks/t=1/relu_frac")


def _b_zero_rank_recorded(records: list[dict[str, Any]]) -> bool:
    """An undefined aggregate rank needs a producer-defined raw zero case."""
    from hessian.spectrum import conditioning_from_extremes

    undefined = False
    for record in records:
        keys = ("lambda_max", "lambda_min", "sigma", "frob_sq")
        if not all(math.isfinite(record[key]) for key in keys):
            return False
        if record["sigma"] < 0 or record["frob_sq"] < 0:
            return False
        expected = conditioning_from_extremes(
            record["lambda_max"],
            record["lambda_min"],
            record["sigma"],
            record["frob_sq"],
            (),
        )["stable_rank"]
        if math.isnan(expected):
            if not math.isnan(record["stable_rank"]):
                return False
            undefined = True
        elif not math.isfinite(record["stable_rank"]):
            return False
    return undefined


def _validate_pilot(pilot: dict[str, Any]) -> None:
    """Validate the scientific cohort and the receipts of author bundles."""
    from .causal_pilot import _validated_rows

    _validated_rows(pilot)
    if pilot.get("format") == "curv-scientific-pilot-v1":
        _b_require("provenance" not in pilot, "causal_pilot: public envelope")
        return
    provenance = pilot["provenance"]
    _b_require(
        len(provenance["inputs"]) == 9
        and len(provenance["source_sha256"]) == 64
        and len(provenance["summary"]["sha256"]) == 64,
        "causal_pilot: missing scalar provenance",
    )
    for record in provenance["inputs"]:
        _b_require(
            set(record["sha256"])
            == {"results.json", "status.json", "provenance.json", "thresholds.json"},
            "causal_pilot: incomplete primary sidecar manifest",
        )


def _validate_granularity_restore(gran: dict[str, Any]) -> None:
    config = gran["meta"]["config"]
    if gran["meta"].get("format") == "curv-scientific-granularity-v1":
        _b_require(
            "restore_checkpoints_dir" not in config
            and config["restored_from"]
            == [
                {"experiment": "exp8_a1_long", "seed": seed, "epoch": 300}
                for seed in (42, 43, 44)
            ],
            "exp8_a1_long_gran_m3000: public checkpoint identities",
        )
    else:
        _b_require(
            "restored_from" not in config
            and set(config["restore_checkpoints_dir"])
            == {f"results/exp8_a1_long_s{s}" for s in (42, 43, 44)},
            "exp8_a1_long_gran_m3000: restored checkpoint origins",
        )


def validate_b_inputs(bundles: dict[str, dict[str, Any]]) -> None:
    """Require B's retained cohorts/cells/references; permit scientific undefineds.

    In particular, a NaN t_star is a recorded grid-prefix outcome, an infinite
    indefinite condition number is valid, and exact-spectrum-only fields need not
    exist for the matrix-free conditioning panel. No model is rebuilt for a missing
    group-size reference and no historical correction/verdict is redefined here.
    """
    from .crossing import crossing_summary
    from .diagnostics import complete_proposal_decomposition, validity_resolution
    from .tables.exp11 import _check_arm_protocol, _check_seeds, _check_steps

    for experiment in track_inputs("b"):
        _b_require(
            isinstance(bundles.get(experiment), dict), f"{experiment}: missing bundle"
        )
    # Validate the crossing family before rendering or creating output.
    crossing_summary(bundles["exp8_crossing_check"])
    for experiment, (models, checkpoints, seeds) in _B_PHI_CELLS.items():
        try:
            for model in models:
                entry = bundles[experiment][model]
                for checkpoint in checkpoints or (None,):
                    cell = entry if checkpoint is None else entry[checkpoint]
                    _b_phi_cell(cell, seeds, f"{experiment}/{model}/{checkpoint}")
                    for payload in cell["seeds"].values():
                        if experiment == "exp8_a1_long":
                            _b_require(
                                "300" in payload["phi_by_probes"],
                                f"{experiment}: 300-probe estimate",
                            )
                        levels = (
                            ("block", "layer")
                            if experiment == "exp8_a2_gran_v3"
                            else (
                                ("layer",)
                                if experiment == "exp8_a1_long_gran_m3000"
                                else ()
                            )
                        )
                        for level in levels:
                            _b_require(
                                bool(payload["diag_frob_by_granularity"][level]),
                                f"{experiment}: {level} profile",
                            )
                            _b_require(
                                math.isfinite(payload["phi_by_granularity"][level])
                                and all(
                                    math.isfinite(v) and v >= 0
                                    for v in payload["diag_frob_by_granularity"][
                                        level
                                    ].values()
                                ),
                                f"{experiment}: {level} Phi/profile",
                            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"track b retained evidence: {experiment}: missing/invalid cell or reference {exc}"
            ) from exc
    try:
        # Validate consumed scalars and their derived domains before writing output.
        # Scientific undefined prefixes and legitimate infinite rel_err survive.
        complete_proposal_decomposition(bundles["exp7"])
        validity_resolution(bundles["exp11"], bundles["exp11_kfac_seam"])
        gran = bundles["exp8_a1_long_gran_m3000"]
        config = gran["meta"]["config"]
        _b_require(
            config["n_probes"] == 3000
            and config["training"]["epochs"] == 300
            and config["dataset"] == "cifar100",
            "exp8_a1_long_gran_m3000: protocol",
        )
        _b_require(
            config["hessian_batch_size"] == 32,
            "exp8_a1_long_gran_m3000: hessian_batch_size must be 32",
        )
        _validate_granularity_restore(gran)
        for model in _B_PAIR:
            for seed in _B_SEEDS:
                _b_require(
                    gran[model]["final"]["seeds"][seed]["test_acc"]
                    == bundles["exp8_a1_long"][model]["final"]["seeds"][seed][
                        "test_acc"
                    ],
                    f"exp8_a1_long_gran_m3000/{model}/{seed}: same-weights accuracy fingerprint",
                )
        arms = {name: bundles[name] for name in ("exp11", "exp11_sgd", "exp11_adam")}
        _check_arm_protocol(arms)
        _b_require(
            "step_source" not in arms["exp11"]["meta"]
            or arms["exp11"]["meta"]["step_source"] == "kfac",
            "exp11: recorded step_source must be kfac when present",
        )
        _b_require(
            bundles["exp11_kfac_seam"]["meta"].get("step_source") == "kfac",
            "exp11_kfac_seam: recorded step_source must be kfac",
        )
        _check_seeds(arms)
        steps = [0, 20, 60, 200, 600, 1500]
        _b_require(_check_steps(arms) == steps, "exp11 arms: probe steps")
        for name in (*arms, "exp11_kfac_seam"):
            arm = bundles[name]
            expected_seeds = ("seed_42",) if name == "exp11_kfac_seam" else _B_SEEDS
            _b_require(set(arm["seeds"]) == set(expected_seeds), f"{name}: seed cohort")
            for key in (
                "t_grid",
                "probe_steps",
                "tolerance",
                "damping",
                "lr",
                "batch_size",
            ):
                _b_require(
                    arm["meta"][key] == arms["exp11"]["meta"][key],
                    f"{name}: {key} protocol",
                )
            _b_require(
                [p["step"] for p in arm["aggregate"]["probes"]] == steps,
                f"{name}: incomplete aggregate probe steps",
            )
            for seed, payload in arm["seeds"].items():
                _b_require(
                    [p["step"] for p in payload["probes"]] == steps,
                    f"{name}/{seed}: missing/duplicate probe cell",
                )
                for probe in payload["probes"]:
                    _b_validity_probe(
                        probe,
                        arm["meta"]["t_grid"],
                        f"{name}/{seed}/step={probe['step']}",
                    )
            for i, probe in enumerate(arm["aggregate"]["probes"]):
                _b_validity_probe(
                    probe,
                    arm["meta"]["t_grid"],
                    f"{name}/aggregate/step={probe['step']}",
                    aggregate=True,
                    infinite_rel_err={
                        point["t"]
                        for payload in arm["seeds"].values()
                        for point in payload["probes"][i]["curve_d0"]
                        if point["rel_err"] == math.inf
                    },
                    undefined_rho=any(
                        math.isnan(payload["probes"][i]["payoff"]["rho"])
                        for payload in arm["seeds"].values()
                    ),
                )
        exp7 = bundles["exp7"]
        for method in (
            "sgd",
            "adam",
            "kfac",
            "kfac+overlay",
            "ekfac",
            "ekfac+overlay",
            "newton_cg",
        ):
            _b_require(
                set(exp7[method]["seeds"]) == set(_B_SEEDS),
                f"exp7/{method}: seed cohort",
            )
            for seed, payload in exp7[method]["seeds"].items():
                for key in (
                    "val_acc",
                    "val_loss",
                    "wall_sec_per_epoch",
                    "hvp_per_epoch",
                ):
                    _b_require(
                        len(payload[key]) == 30,
                        f"exp7/{method}/{seed}: incomplete {key}",
                    )
                if method not in ("sgd", "adam"):
                    _b_require(
                        [r["step"] for r in payload["model_decrease"]]
                        == list(range(0, 3000, 20)),
                        f"exp7/{method}/{seed}: incomplete refresh measurements",
                    )
                if method in ("kfac+overlay", "ekfac+overlay"):
                    _b_require(
                        [r["step"] for r in payload["cross_term"]]
                        == list(range(0, 3000, 20)),
                        f"exp7/{method}/{seed}: incomplete cross-term measurements",
                    )
                    groups = ("stem", "layer1", "layer2", "layer3", "layer4", "head")
                    expected_pairs = {
                        f"{v}->{w}" for v in groups for w in groups if v != w
                    }
                    for record in payload["cross_term"]:
                        _b_require(
                            set(record["pairs"]) == expected_pairs,
                            f"exp7/{method}/{seed}/step={record['step']}: cross-term pair grid",
                        )
                        for pair, values in record["pairs"].items():
                            _b_require(
                                all(
                                    math.isfinite(values.get(field, math.nan))
                                    and values[field] >= 0
                                    for field in ("coupling", "eps")
                                ),
                                f"exp7/{method}/{seed}/step={record['step']}: cross-term {pair} values",
                            )
                            v, w = pair.split("->")
                            _b_require(
                                math.isclose(
                                    values["coupling"],
                                    record["pairs"][f"{w}->{v}"]["coupling"],
                                    rel_tol=1e-12,
                                ),
                                f"exp7/{method}/{seed}/step={record['step']}: cross-term symmetric coupling",
                            )
        conditioning = bundles["exp10"]["resnet18"]["resnet18"]
        for checkpoint in ("init", "final"):
            cell = conditioning[checkpoint]
            _b_require(
                set(cell["blocks"])
                == {"stem", "layer1", "layer2", "layer3", "layer4", "head"},
                f"exp10/{checkpoint}: missing block cell",
            )
            for block, record in (("full", cell["full"]), *cell["blocks"].items()):
                for key in ("lambda_max", "lambda_min"):
                    for statistic in ("mean", "std"):
                        _b_require(
                            math.isfinite(record[key][statistic]),
                            f"exp10/{checkpoint}: invalid {key}/{statistic}",
                        )
                rank = record["stable_rank"]
                if not all(math.isfinite(rank[key]) for key in ("mean", "std")):
                    raw = [
                        payload["full"] if block == "full" else payload["blocks"][block]
                        for payload in cell["seeds"].values()
                    ]
                    _b_require(
                        math.isnan(rank["mean"])
                        and math.isnan(rank["std"])
                        and _b_zero_rank_recorded(raw),
                        f"exp10/{checkpoint}: invalid stable_rank",
                    )
                for damping in ("0.01", "0.1"):
                    kappa = record["kappa"][damping]
                    indefinite = record["lambda_min"]["mean"] + float(damping) <= 0
                    _b_require(
                        math.isfinite(kappa["mean"])
                        or (indefinite and kappa["mean"] == math.inf),
                        f"exp10/{checkpoint}: invalid kappa/{damping}/mean",
                    )
                    _b_require(
                        math.isfinite(kappa["std"])
                        or (kappa["mean"] == math.inf and math.isnan(kappa["std"])),
                        f"exp10/{checkpoint}: invalid kappa/{damping}/std",
                    )
        _validate_pilot(bundles["causal_pilot"])
    except (KeyError, TypeError, ValueError) as exc:
        if str(exc).startswith("track b retained evidence:"):
            raise
        raise ValueError(f"track b retained evidence: {exc}") from exc


def preflight_b(paths: DataPaths) -> dict[str, dict[str, Any]]:
    """Load/validate all retained B evidence before the first output write."""
    from .loader import load

    bundles = {}
    for experiment in track_inputs("b"):
        path = paths.get(experiment)
        _b_require(path.is_file(), f"{experiment}: missing retained bundle {path}")
        try:
            bundles[experiment] = load(paths, experiment)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"track b retained evidence: {experiment}: {path}: {exc}"
            ) from exc
    validate_b_inputs(bundles)
    return bundles
