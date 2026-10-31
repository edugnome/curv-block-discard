# Profile-aware diagnostics of discarded parameter-Hessian mass

Research code and experimental records for **Profile-aware diagnostics of
discarded parameter-Hessian mass**.

**Maxim Bolshim, Alexander Kugaevskikh**

Unpublished manuscript.

[Reproduce results](#reproduce-the-paper-results) |
[Experimental data](data/README.md) |
[Reproduction guide](REPRODUCING.md) |
[Citation](#citation)

## Overview

The project measures the parameter-Hessian mass discarded by a block-diagonal
approximation. It compares the off-block squared Frobenius fraction, Phi, with
references determined by group sizes and measured diagonal-block norms. Separate
experiments examine corrections to K-FAC/EKFAC steps and the agreement between
quadratic models and finite-step loss.

- **Structural diagnostics:** parameter-Hessian block norms, coupling, spectra,
  and size/profile references for an explicit parameter partition.
- **Experiments:** convolutional networks, frozen transformers, K-FAC/EKFAC
  corrections, and finite-step validity along optimisation trajectories.
- **Saved-state comparisons:** stage/layer partition crossings and full/off/rest
  residual-input corrections with a common reduced geometry.
- **Recorded data:** scalar measurements, loss curves, proposal outcomes, paired
  components and geometry diagnostics used to generate the article results.

On the saved trained ResNet-18 states, changing from stage groups to layers moves
estimated discarded mass across the profile envelope. In the residual-input
comparison, off-group input alone increases loss in 17 of 18 state/geometry
cases. The full candidate has lower loss than the remainder-only candidate in
nine of those cases; seven of the ten full-over-remainder preferences reverse
after correction-norm matching. These comparisons concern the reported states,
partitions and correction scales.

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

If you use the code or experimental records, please cite the manuscript:

```bibtex
@misc{bolshim2026profile,
  title  = {Profile-aware diagnostics of discarded parameter-Hessian mass},
  author = {Bolshim, Maxim and Kugaevskikh, Alexander},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```

Software citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

The source code and authors' experimental records are licensed under the
[MIT License](LICENSE); the data notice is in [data/LICENSE](data/LICENSE).
Scientific citation is requested, not an additional licence condition.
Third-party datasets, model weights and dependencies retain their own terms.
