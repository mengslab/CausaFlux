# CausaFlux v2.0.0 verification results

## Release status

**PASS — stable software release with a locked prospective-validation claim.**

## Automated validation

- 111 packaged automated tests passed across 27 test modules, run in subsystem batches to avoid command-duration limits.
- Retained dynamic forecasting benchmark: PASS.
- Retained multimodal dynamic benchmark: PASS.
- Retained unseen-intervention generalization benchmark: PASS.
- Retained spatiotemporal tissue benchmark: PASS.
- Retained foundation-pretraining benchmark: PASS.
- Retained publication graphics suite: PASS.
- Retained real-data, biological-validation, prospective-loop, virtual-cell, causal, spatial, biomarker, therapeutic, active-learning and platform tests: PASS.
- New v2 evidence-ledger and longitudinal-data tests: PASS.
- Exact `sh run.sh` smoke path: PASS.
- Strict validation without real evidence: correctly FAILS.
- Copying the distributed PENDING evidence template cannot unlock the claim: PASS.
- Qualifying evidence requires an existing local source artifact with a matching SHA-256: enforced.

## Reference release output

The bundled `v2_release_reference/` reports:

- `software_release_ready = true`
- `prospectively_validated_virtual_cell = false`
- `release_claim_status = NOT_YET_ELIGIBLE`
- real required criteria passed = 0/10

This is intentional. Synthetic and historical software fixtures validate implementation and release integrity but do not constitute prospective biological validation.

## Real longitudinal perturbation bridge

The v2 public registry includes authoritative starting points for:

- GSE8057 — platinum-drug time-course and dose-response expression data;
- GSE70138 — LINCS L1000 perturbagen/cell/dose/time profiles;
- GSE101406 — matched P100, GCP and L1000 perturbation readouts.

Repository data are not silently redistributed. The package provides contracts and conversion/training commands for locally downloaded or laboratory-generated real data.

## Figure validation

The v2 reference report includes five figure families. Each is exported as SVG, PDF, 600-dpi PNG and 600-dpi TIFF with panel source data and hashed figure manifests.
