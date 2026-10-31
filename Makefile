.DEFAULT_GOAL := help
PYTHON ?= python3
DATA_ROOT ?= data
DATA_DIR ?= $(DATA_ROOT)/reporting
BUILD_DIR ?= build
EXPORT_DIR ?= $(BUILD_DIR)/tables
RELEASE_BUILD_DIR ?= working/build
THREADS ?= 2
export OMP_NUM_THREADS := $(THREADS)
export MKL_NUM_THREADS := $(THREADS)
export OPENBLAS_NUM_THREADS := $(THREADS)

.PHONY: prepare help install lint format-check typecheck test check reproduce export-data release clean
help:
	@printf '%s\n' \
	  'install       Install pinned runtime and development dependencies' \
	  'lint          Run Ruff lint' \
	  'format-check  Check Ruff formatting' \
	  'typecheck     Check source, scripts and tests with mypy' \
	  'test          Run functional CPU tests without model/dataset downloads' \
	  'check         Run all four quality gates' \
	  'reproduce     Generate article tables and figures from bundled data' \
	  'export-data   Export numerical observation tables into a new EXPORT_DIR' \
	  'release       Package source and bundled data under RELEASE_BUILD_DIR' \
	  'clean         Remove named generated outputs under BUILD_DIR' \
	  'Variables: PYTHON, DATA_ROOT, DATA_DIR, BUILD_DIR, EXPORT_DIR, RELEASE_BUILD_DIR, THREADS'
install:
	$(PYTHON) -m pip install -c requirements.lock '.[dev]'
prepare:
	$(PYTHON) scripts/project.py prepare --build-dir "$(BUILD_DIR)"
lint: prepare
	$(PYTHON) -m ruff check --cache-dir "$(BUILD_DIR)/ruff" src tests scripts
format-check: prepare
	$(PYTHON) -m ruff format --check --cache-dir "$(BUILD_DIR)/ruff" src tests scripts
typecheck: prepare
	$(PYTHON) -m mypy --cache-dir "$(BUILD_DIR)/mypy" src tests scripts
test: prepare
	@test "$(THREADS)" -ge 1 || { echo 'THREADS must be a positive integer'; exit 2; }
	CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m pytest -q -o cache_dir="$(BUILD_DIR)/pytest" tests
check: lint format-check typecheck test
reproduce: prepare
	HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m paper_data.main --track b --results-dir "$(DATA_DIR)" --output-dir "$(abspath $(BUILD_DIR)/reporting)"
	HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m paper_data.main --track b-ablation --results-dir "$(DATA_DIR)" --output-dir "$(abspath $(BUILD_DIR)/reporting)"
export-data: prepare
	$(PYTHON) -I -m paper_data.exports --data-root "$(DATA_ROOT)" --output-dir "$(abspath $(EXPORT_DIR))"
release:
	$(PYTHON) scripts/project.py release --build-dir "$(RELEASE_BUILD_DIR)"
clean:
	$(PYTHON) scripts/project.py clean --build-dir "$(BUILD_DIR)"
