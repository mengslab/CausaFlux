# CausaFlux v1.4.0 User Guide

## 1. Installation

The recommended entry point on macOS is:

```bash
sh run.sh
```

The launcher creates `.causaflux_env`, selects Python 3.10–3.12, installs CausaFlux in editable mode, runs the integrated reference workflow, validates all outputs, and opens the report.

Manual setup:

```bash
python3.11 -m venv .causaflux_env
.causaflux_env/bin/python -m pip install --upgrade pip setuptools wheel
.causaflux_env/bin/python -m pip install -e '.[dev,app]'
```

## 2. Check readiness

```bash
.causaflux_env/bin/causaflux doctor
.causaflux_env/bin/causaflux version
```

## 3. Run the integrated platform

```bash
.causaflux_env/bin/causaflux run \
  --config configs/cancer_closed_loop_v1.4.0.yaml \
  --output causaflux_v1.4.0_output
```

The output contains cancer, multimodal, spatial, causal, therapeutic, biomarker, active-learning, and neurobiology subdirectories.

## 4. Use packaged demos

```bash
.causaflux_env/bin/causaflux demo-list
.causaflux_env/bin/causaflux demo-run cancer_quickstart
.causaflux_env/bin/causaflux demo-run neurobiology_quickstart
```

## 5. Validate an output

```bash
.causaflux_env/bin/causaflux platform-validate \
  --input causaflux_v1.4.0_output
```

Use `--refresh` after adding approved artifacts to regenerate dataset cards, provenance, hashes, and the platform report.

## 6. Replace synthetic data

Cancer studies should supply one observation per cell or tracked lineage/time point with donor, sample, cell type, time, treatment, outcome, and molecular features. Neurobiology studies should supply one neural or glial observation per time point with donor, lineage, cell type, time, genotype/context, modality availability, and a prospectively defined future outcome.

Never split cells from the same donor between training and testing. Missing assays must be represented as missing modality availability, not as biological zeros.

## 7. Interpret results

Review the evidence ladder, calibration, donor-bootstrap intervals, extrapolation warnings, normal-cell penalties, and causal assumptions before interpreting any ranking. Association, temporal precedence, causal proximity, perturbational support, and clinical assayability are distinct evidence dimensions.

## Run the biological-validation release

```bash
sh run.sh
```

The default v1.4.0 workflow analyzes the bundled public SEA-AD metadata with frozen ACT discovery and ADRC replication cohorts. For a faster demonstration:

```bash
CAUSAFLUX_VALIDATION_BOOTSTRAP=100 sh run.sh demo_validation
```

The report is written to `causaflux_v1.4.0_validation/reports/index.html` by default.

## v1.4 multimodal dynamic benchmark

The default `sh run.sh` command now runs the multimodal dynamic-state benchmark.

```bash
causaflux multimodal-dynamic-run \
  --output causaflux_v1.4.0_multimodal_dynamic \
  --require-gate

causaflux multimodal-dynamic-validate \
  --input causaflux_v1.4.0_multimodal_dynamic
```

To inspect or adapt the external schema:

```bash
causaflux multimodal-dynamic-export-fixture \
  --output multimodal_dynamic_fixture_v1.4.0.npz
```

A conforming real dataset must retain per-modality tensors, per-time observation masks, donor and cohort IDs, perturbation-history IDs, times, baseline covariates, and the later destructive outcome. Do not replace missing observations with population means before export; missingness must remain explicit for modality-dropout and sensitivity analyses.
