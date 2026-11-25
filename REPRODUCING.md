# Reproducing the results

**When does off-block Hessian curvature help an optimization step?** — repository version **3.0.0**.

Use Python 3.12 and install from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
make install PYTHON=python
make reproduce PYTHON=python
make export-data PYTHON=python
```

The bundled scalar measurements produce 27 LaTeX resources in `build/reporting/`
and numerical observation tables in `build/tables/`. Reporting needs no training,
model checkpoints, source datasets or GPU. Export destinations must be new;
choose another with `EXPORT_DIR=/path/to/new-tables`.

## Results and inputs

Inputs are under `data/reporting/<bundle>/results.json`.

| Results | Input families |
| --- | --- |
| Phi/profile comparisons and partition dependence | `exp8*`, `exp9*` |
| Correction, payoff and model-versus-loss comparisons | `exp7` |
| Validity curves and sampled agreement extents | `exp11*` |
| Conditioning | `exp10` |
| Captured-state pilot | `causal_pilot` |
| Stage/layer crossing | `exp8_crossing_check` and paired records |
| Residual candidates and scale/norm contrasts | `residual_ablation_v2` |
| Basis and correction-map diagnostics | `residual_basis_geometry` |

List the available resources or generate one:

```sh
curv-report --list
curv-report --list-inputs
curv-report --only tab:stage-b --results-dir data/reporting \
  --output-dir build/single-resource
```

`--track b` selects the 22 structural, correction, validity and pilot resources.
`--track b-ablation` selects the five residual-input and geometry resources.
`make reproduce` generates both. To use another corpus, set
`DATA_ROOT=/path/to/data` or `DATA_DIR=/path/to/reporting`.

## Observation exports

The proposal decomposition and residual contrasts have separate exports:

```sh
python -I -m paper_data.exports --data-root data \
  --output-dir build/proposals --derived-proposals
python -I -m paper_data.exports --data-root data \
  --output-dir build/contrasts --residual-contrasts
```

These calculate 900 proposal-decomposition rows and 144 contrast rows from
recorded measurements. They perform no model evaluations. The contrast export
contains 126 scale comparisons and 18 norm-matched full/rest comparisons.
Field conventions and available observations are described in [data/README.md](data/README.md).

## Checking the main comparisons

You can inspect these bundled files directly; no export or model run is needed.

| Question | File and selection | What to read |
| --- | --- | --- |
| Which input helped at seed 42, step 60? | [Residual candidates](data/tables/residual-ablation.jsonl): select `seed=42`, `step=60`, `kind=scale`, `alpha=1`; group by `drawset`. | Within each draw, compare `raw_delta_loss` for `arm=off`, `rest` and `full`, then `gates.accepted`. Only `full` lowers loss and is accepted. |
| Does matching correction lengths change the choice? | [Residual contrasts](data/tables/residual-contrasts.jsonl): compare `type=scale`, `alpha=1` with `type=matched_full_rest` at the same `(seed, step, drawset)`. | `y` is full loss minus rest loss. At 42/60 it goes from negative to positive in both draws. `norm_match_scales` identifies which correction was shortened. |
| Did the model predict the direction of change in loss at the endpoint? | [Frozen-state pilot](data/reporting/causal_pilot/results.json): use `rows` and each row's `native.comparison`. | Compare the signs of `predicted_endpoint_slope` and `endpoint_slope` along the original correction. They disagree at `(42,200)`, `(43,60)`, `(43,200)` and `(44,200)`. |
| Does the grouping cross the profile reference at fixed weights and inputs? | [Crossing summary](data/tables/crossing-summary.jsonl): one row per `(model, seed, batch)`. | For a resolved stage-to-layer crossing, the joint conditional intervals in `observation.screen_95` put `S_stage` above one and `S_layer` below one. Read the recorded classification in `observation.crossing`; four Plain-18 cells remain unresolved. |

## Interpretation and fresh measurements

Phi uses the declared parameter partition; the six measurement groups are not
K-FAC's per-module blocks. Raw proposal loss and the applied step are separate
outcomes. The seed-42 validity repeat remains a distinct run. Missing values,
nonfinite estimates and unresolved comparisons retain their recorded meanings.

Repeating measurements requires the corresponding datasets, weights and captured
inputs. Inspect `curv-experiment`, `curv-pilot` and `curv-crossing` with `--help`
before selecting a protocol. Basis-factor diagnostics additionally require the
saved factor files; their bundled scalar results suffice for table generation.
Retraining is not a promise of bitwise identity across software and hardware.

Run the numerical and functional tests with `make check PYTHON=python THREADS=2`.
