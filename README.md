# When does off-block Hessian curvature help an optimization step?

[![arXiv](https://img.shields.io/badge/arXiv-2611.01335-b31b1b.svg)](https://arxiv.org/abs/2611.01335)
[![DOI](https://zenodo.org/badge/1441269179.svg)](https://doi.org/10.5281/zenodo.23914947)

Research code and experimental records for **When does off-block Hessian curvature help an optimization step?**

**Maxim Bolshim, Alexander Kugaevskikh** · Software version **3.0.0**

**Journal target:** [Optimization Methods and Software](https://www.tandfonline.com/journals/goms20).

Should an off-block Hessian correction be discarded if testing it alone raises
minibatch loss? At the proposed scale, that decision can discard a combination
that improves on the base step. We measure both the missed improvement and the
gain from rescaling the remaining correction.

[Preprint](https://arxiv.org/abs/2611.01335) |
[Preprint PDF](https://arxiv.org/pdf/2611.01335) |
[Archived code and data](https://doi.org/10.5281/zenodo.23914947) |
[Reproduce results](#reproduce-the-paper-results) |
[Experimental data](data/README.md) |
[Reproduction guide](REPRODUCING.md) |
[Citation](#citation)

## Overview

- **What the grouping omits.** The squared Frobenius fraction `Phi` measures
  off-block mass. The diagonal-block norms give a sharp upper bound for that
  fraction in positive semidefinite matrices with the same norm profile.
  Comparing mass with this reference distinguishes a change in the diagonal
  profile from a change in mass relative to that profile. Regrouping fixed
  ResNet-18 weights from stages to layers moves the estimate across its own
  partition-specific reference in all six CIFAR-100 checkpoint/input comparisons
  under the conditional probe screen.
- **What the omitted component does to a step.** On Stanford Cars, nine saved
  ResNet-50 checkpoints from three K-FAC trajectories each have two paired
  subspace draws. At unit scale, the off-block correction alone raises loss in
  17 of 18 comparisons. The combination beats the remainder in ten, including
  nine of those 17. At **seed 42, step 60**, only the combination improves on
  the base step and passes the safeguard in both draws. Matching the correction
  norms leaves the combination unchanged there, but shortening the remainder
  gives a lower loss still. Norm matching reverses seven of the ten preferences.
  The full training runs did not demonstrate a training gain.

A separate diagnostic on those nine checkpoints compares the predicted and
measured slopes along the original correction at the **base-step endpoint**.
Their signs differ at four checkpoints; seed 42, step 60 is not one of them.
See [the record-level checks](REPRODUCING.md#checking-the-main-comparisons)
for the corresponding data and field names.

## About this revision

This is a revision of the same manuscript as the [earlier preprint](https://arxiv.org/abs/2611.01335v3).
After the earlier submissions, we reread the paper and found that its main
result was hard to identify. The text did not clearly distinguish a combined
correction that beats the remainder from one that also improves on the base
step. We revised the abstract, results, discussion and captions to make those
comparisons explicit and to show separately what changes when the remainder
is shortened. This editorial revision did not change the measurements.

## Installation

Use Python 3.12, GNU Make and a Linux environment. From the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
make install PYTHON=python
```

`requirements.lock` pins the tested dependency versions. Optional frozen
Hugging Face model experiments also require:

```sh
python -m pip install -c requirements.lock '.[llm]'
```

Reporting and the CPU tests use the bundled data and synthetic examples.
Large-network measurements require suitable hardware and the relevant datasets
and model weights. Dataset/model acquisition is separate from installation.

A small example of the profile references:

```python
from hessian.phi_baselines import phi_null, phi_sep

print(phi_null([2, 2]))     # 0.5: size-only reference
print(phi_sep([1.0, 1.0]))  # 0.5: PSD ceiling for equal diagonal-block norms
```

## Reproduce the paper results

After installation:

```sh
make reproduce PYTHON=python
make export-data PYTHON=python
```

`make reproduce` generates the article's 27 LaTeX tables, figures and numerical
macro files in `build/reporting/`. `make export-data` writes observation tables
to `build/tables/`; the export directory must be new.

Both commands use the bundled `data/` directory. They require no training,
checkpoints, source datasets or GPU. See [REPRODUCING.md](REPRODUCING.md) for
individual resources and the proposal/residual contrast exports.

### Experimental records

The [data directory](data/README.md) contains the scientific inputs for reporting
and readable observation tables. It preserves individual seeds and states,
failed or unresolved outcomes, and the distinction between raw proposals and
applied steps. The exp11 seed-42 repeat is a separate run, rather than an
additional independent seed.

Rebuilding results from recorded measurements is distinct from repeating the
measurements. Fresh measurements require compatible model states, inputs and
software. The bundled geometry diagnostics are scalar records; recalculating
them from basis factors requires those factors separately.

## Run experiments

Use the installed commands to inspect the available protocols:

```sh
curv-experiment exp7 --help
curv-experiment exp8 --help
curv-experiment exp9 --help
curv-experiment exp10 --help
curv-experiment exp11 --help
curv-pilot --help
curv-crossing --help
```

| Family | Entry point | Inputs |
| --- | --- | --- |
| K-FAC/EKFAC corrections | `curv-experiment exp7 --profile b1` | Stanford Cars and pretrained ResNet-50 weights |
| Convolutional block mass | `curv-experiment exp8 --profile a1` | CIFAR-100; saved checkpoints for remeasurement |
| Frozen transformers | `curv-experiment exp9 --profile a3_llm` | Optional dependencies and cached model weights |
| Conditioning | `curv-experiment exp10` | Synthetic examples or the selected model/data |
| Finite-step validity | `curv-experiment exp11 --profile b1` | Dataset/model inputs; trains the selected trajectory |
| Captured-state comparisons | `curv-pilot` | Captured states and embedded input batches |
| Stage/layer crossing | `curv-crossing` | Saved states, CIFAR-100 and a prepared manifest |

For example, a convolutional measurement run is:

```sh
curv-experiment exp8 --profile a1 --data-root /path/to/cifar \
  --output-dir /path/to/new-run
```

CIFAR input paths use `--data-root` or `DAG_HESSE_DATA_ROOT`; Stanford Cars uses
`DAG_HESSE_CARS_ROOT`. Downloads require explicit `--allow-downloads`. Follow the
dataset and model licences when obtaining inputs. Measurement jobs are separate
from reporting, tests and packaging. For residual-ablation measurement and
collection commands, `--protected-root /path/to/corpus` additionally prevents
outputs within or above that data tree; supplied input paths are always protected.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/hessian/` | Parameter-space Hessian estimators, spectra and block diagnostics |
| `src/secondorder/` | K-FAC/EKFAC adapters, reduced corrections and safeguards |
| `src/experiments/` | Models, training and measurement interfaces |
| `src/paper_data/` | Numerical analysis, reporting and observation exports |
| `data/` | Bundled scientific records and readable tables |
| `tests/` | Numerical and functional tests |
| `scripts/` | Packaging and factor-based geometry diagnostics |

## Development

```sh
make test PYTHON=python          # Functional CPU tests
make lint PYTHON=python          # Ruff
make format-check PYTHON=python  # Formatting
make typecheck PYTHON=python     # mypy
make check PYTHON=python         # All four checks
```

GPU-only tests are skipped in CPU runs. `THREADS` controls CPU threading and
defaults to 2. `DATA_ROOT`, `DATA_DIR`, `BUILD_DIR` and `EXPORT_DIR` select input
and output locations.

`make clean` removes named generated outputs under `BUILD_DIR`, preserving
source, data and environments. `make release` creates a source-and-data archive
under `working/build/release/`; it performs no upload.

## Citation

If you use the code or experimental records, please cite the [arXiv preprint](https://arxiv.org/abs/2611.01335):

```bibtex
@misc{bolshim2026profile,
  title         = {When does off-block {Hessian} curvature help an optimization step?},
  author        = {Bolshim, Maxim and Kugaevskikh, Alexander},
  year          = {2026},
  eprint        = {2611.01335},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.01335}
}
```

For the software and experimental records, the following DOI identifies the
project across all releases. To cite a specific release, use its version and
version-specific DOI from Zenodo (for example, [release 2.0.0](https://doi.org/10.5281/zenodo.24343776)).

```bibtex
@software{bolshim2026curvblockdiscard,
  title  = {When does off-block {Hessian} curvature help an optimization step?},
  author = {Bolshim, Maxim and Kugaevskikh, Alexander},
  year   = {2026},
  doi    = {10.5281/zenodo.23914947},
  url    = {https://doi.org/10.5281/zenodo.23914947}
}
```

Software citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

The source code and authors' experimental records are licensed under the
[MIT License](LICENSE); the data notice is in [data/LICENSE](data/LICENSE).
Scientific citation is requested, not an additional licence condition.
Third-party datasets, model weights and dependencies retain their own terms.
