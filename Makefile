PYTHON ?= python3.12
VENV   ?= .venv
PY     := $(VENV)/bin/python
CLI    := $(PY) -m app.cli

CONFIG  ?= config/model.yaml
DATASET ?= data/golden_dataset.jsonl
POLICY  ?= config/gates.yaml

.PHONY: install test validate baseline evaluate gate demo demo-pass demo-fail demo-fixed clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e ".[dev]"

test:
	$(PY) -m pytest

validate:
	$(CLI) validate-dataset --dataset $(DATASET) --expect-per-category 15
	$(CLI) verify-dataset-hash --dataset $(DATASET) --baseline baselines

# Deliberate, human-only action. Run as: ALLOW_BASELINE_UPDATE=true make baseline
baseline:
	$(CLI) evaluate --config $(CONFIG) --dataset $(DATASET) --output reports/baseline-candidate
	$(CLI) approve-baseline --source reports/baseline-candidate --dataset $(DATASET) --baseline-dir baselines

evaluate:
	$(CLI) evaluate --config $(CONFIG) --dataset $(DATASET) --output reports/candidate

gate:
	$(CLI) gate --baseline baselines --candidate reports/candidate --policy $(POLICY) --output reports/comparison

# Each demo evaluates one mock against the unchanged golden dataset and baseline.
# demo-fail exits nonzero after writing its full report.
define run_demo
	@$(CLI) evaluate --config $(CONFIG) --model $(1) --dataset $(DATASET) --output reports/$(2)/candidate
	@$(CLI) gate --baseline baselines --candidate reports/$(2)/candidate --policy $(POLICY) --output reports/$(2)/comparison; \
	status=$$?; echo ""; echo ">>> $(1): exit code $$status, report at reports/$(2)/comparison/comparison.md"; exit $$status
endef

demo-pass:
	$(call run_demo,mock-v1-baseline,demo-pass)

demo-fail:
	$(call run_demo,mock-v2-regression,demo-fail)

demo-fixed:
	$(call run_demo,mock-v3-fixed,demo-fixed)

demo:
	-@$(MAKE) --no-print-directory demo-pass
	-@$(MAKE) --no-print-directory demo-fail
	-@$(MAKE) --no-print-directory demo-fixed

clean:
	find reports -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
