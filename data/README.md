# Experimental data

Scientific records for *Profile-aware diagnostics of discarded
parameter-Hessian mass*, by Maxim Bolshim and Alexander Kugaevskikh
([arXiv:2611.01335](https://arxiv.org/abs/2611.01335)).
The authors' records are licensed under [MIT](LICENSE).
The archived software and data are available on
[Zenodo](https://doi.org/10.5281/zenodo.23914948).

## Contents

| Path | Observations |
| --- | --- |
| `reporting/exp8*/`, `reporting/exp9*/` | Hessian fractions, diagonal-block norms, group sizes and couplings |
| `reporting/exp7/` | K-FAC/EKFAC proposals, model values, finite losses and applied outcomes |
| `reporting/exp11*/` | Loss/model curves and sampled validity extents |
| `reporting/exp10/` | Conditioning measurements |
| `reporting/causal_pilot/` | Nine captured-state comparisons |
| `reporting/exp8_crossing_check/` | Twelve stage/layer cells and their conditional precision analysis |
| `reporting/residual_ablation_v2/` | Nine states, two geometry draws and full/off/rest correction candidates |
| `reporting/residual_basis_geometry/` | Eighteen basis and correction-map diagnostics |
| `records/crossing-pairs.jsonl` | 360 paired stage/layer component measurements |
| `tables/` | Observation exports, including 900 proposals, 486 residual candidate slots and 144 contrasts |

Each reporting directory contains `results.json`. JSONL files contain one record
per line. Fields such as `bundle`, `seed`, `step`, `cell`, `drawset` and `arm`
identify the protocol and observation. The `observation` field in family tables
contains the corresponding scientific record; loss/proposal exports expose
individual quantities as columns.

## Conventions

- Phi is the off-block squared Frobenius fraction for the stated partition.
  Group sizes and diagonal-block norms define its size/profile references.
- `raw_delta_loss` is candidate loss minus base-step loss; negative improves on
  the base. Applied outcomes also account for the recorded gates.
- In residual contrasts, `x` is off-minus-base, `y` is full-minus-rest and
  `j = y - x`. Scale and norm-matched comparisons are distinct row types.
- Status/reason fields distinguish valid, unavailable, rejected and unresolved
  outcomes. Missing values are not zero. JSONL encodes nonfinite observations as
  `{"nonfinite":"NaN"}`, `{"nonfinite":"+Infinity"}` or
  `{"nonfinite":"-Infinity"}`; JSON null remains missing/not applicable.
  Some reporting JSON retains NaN/Infinity tokens accepted by the project reader.
- Steps within a training seed are dependent. The two geometry draws are paired
  sensitivity measurements. `exp11_kfac_seam` is a separate seed-42 repeat,
  not a fourth independent seed.

The bundle contains scalar measurements, not model weights, images or basis
factors. Third-party inputs and models retain their own licences. Reproducing
tables from these records is distinct from rerunning the measurements.

## Use

From the installed repository root:

```sh
make reproduce PYTHON=python
make export-data PYTHON=python
```

See [REPRODUCING.md](../REPRODUCING.md) for individual resources and derived
exports. Export destinations must be new; the bundled inputs remain unchanged.
