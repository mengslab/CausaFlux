# CausaFlux v2.0.0 — Prospectively Validated Virtual Cell

CausaFlux is a Python research platform for dynamic, multimodal and perturbable virtual-cell modeling. Version 2.0.0 introduces a hard evidence gate: **the software can be a stable v2 release without claiming that the virtual cell has already been prospectively validated.** The claim is authorized only after the required real experimental evidence is supplied and all gates pass.

## One-command Mac run

```bash
cd ~/Downloads/CausaFlux_v2.0.0
sh run.sh
```

The installer automatically creates/reuses an isolated Python 3.10–3.12 environment. The Intel-macOS compatibility path retains PyTorch 2.2.2 and NumPy 1.26.x.

The default output is:

```text
causaflux_v2.0.0_release/
```

Open:

```text
causaflux_v2.0.0_release/report/index.html
```

The standard `sh run.sh` command validates software readiness and builds the evidence report. It does **not** fail merely because real prospective evidence is absent. To demand the biological claim:

```bash
sh validate_prospective_v2.sh causaflux_v2.0.0_release
```

or:

```bash
causaflux v2-validate --input causaflux_v2.0.0_release --require-prospectively-validated
```

## v2 release requirements

The prospectively validated claim requires all of the following:

- real Phase 1 dynamic superiority;
- real multimodal forecasting validation;
- real unseen-intervention generalization;
- real spatial-context predictive benefit;
- at least two real prospectively locked experiment cycles;
- an external laboratory or independent-cohort replication;
- calibrated uncertainty maintained under a prespecified distribution shift;
- all claims linked to the evidence ledger;
- explicit failed-assay and negative-result reporting;
- at least one actual longitudinal perturbation dataset connected to model training and the prospective experimental loop.

Synthetic software fixtures do not satisfy these requirements.

## Real longitudinal perturbation data

Export the dataset registry and input contract:

```bash
causaflux longitudinal-contract --output real_longitudinal_contract
```

The bundled registry includes GEO GSE8057, LINCS GSE70138 and LINCS GSE101406 as public starting points. Large repository data are not redistributed in the package.

Convert your processed longitudinal experiment:

```bash
causaflux longitudinal-convert \
  --input my_longitudinal_experiment.csv \
  --output my_longitudinal_experiment.npz \
  --manifest my_longitudinal_manifest.json
```

Run the real held-out-history benchmark:

```bash
causaflux longitudinal-benchmark \
  --input my_longitudinal_experiment.csv \
  --output my_real_dynamic_benchmark
```

See `docs/v2/REAL_LONGITUDINAL_DATA.md`.

## Distribution-shift uncertainty

Provide a table with `observed`, `predicted_mean`, `predicted_sd`, and `shift_group`:

```bash
causaflux shift-calibration \
  --input external_predictions.csv \
  --output external_shift_calibration \
  --require-gate
```

## Evidence ingestion

Start from:

```text
templates/v2_evidence/evidence_ledger.csv
```

Then rebuild the release bundle using the locked real evidence:

```bash
causaflux v2-run \
  --external-evidence-dir /path/to/locked_evidence \
  --output causaflux_v2_real
```

External evidence can support, fail, falsify or qualify a claim. A real PASS row is eligible only when its `source` file exists and its SHA-256 matches the ledger. Negative and failed experiments remain visible.

## Interactive interface

```bash
sh ui.sh causaflux_v2.0.0_release
```

The UI separates software readiness from the real prospective-validation claim and provides the virtual-cell explorer, AI router, real longitudinal dataset registry, evidence ledger, prospective status and publication figures.

## Publication-grade reporting

The v2 report generates:

- `Figure1_v2_release_evidence_ladder`;
- `Figure2_claim_evidence_matrix`;
- `Figure3_prospective_cycle_evidence`;
- `GraphicalAbstract_v2_prospectively_validated_virtual_cell`.

Each is exported as SVG, PDF, 600-dpi PNG and 600-dpi TIFF with source-data CSV and a hashed figure manifest.

## Important interpretation boundary

Do not describe a CausaFlux installation as a **prospectively validated virtual cell** unless:

```json
"prospectively_validated_virtual_cell": true
```

appears in the v2 release gate and the strict validator passes. Version 2.0.0 is the software release; prospective biological validation is an evidence status.

## Documentation

- `docs/v2/V2_PROSPECTIVE_VALIDATION_STANDARD.md`
- `docs/v2/VIRTUAL_CELL_2_0.md`
- `docs/v2/REAL_LONGITUDINAL_DATA.md`
- `docs/v2/EVIDENCE_LEDGER.md`
- `docs/v2/FAILURE_NEGATIVE_RESULTS.md`
- `docs/v2/STABLE_RELEASE_CHECKLIST.md`
- retained v1.x architecture, validation and third-party-data documentation

## License

MIT. Third-party datasets, pretrained models and checkpoints remain subject to their original licenses and access terms.
