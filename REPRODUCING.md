# Reproducing the results

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
