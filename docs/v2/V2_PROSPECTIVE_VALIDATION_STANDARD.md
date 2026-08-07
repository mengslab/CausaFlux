# CausaFlux v2.0.0 prospective-validation standard

CausaFlux v2.0.0 is a release standard, not a claim produced by a version number.

The label **Prospectively Validated Virtual Cell** is authorized only when all of the following are supported by qualifying non-synthetic evidence in the locked evidence ledger:

1. Phase 1 dynamic superiority is established on real longitudinal perturbation data using a prespecified held-out-history design and static baselines.
2. Multimodal forecasting is validated on real multimodal longitudinal or temporally ordered perturbation data.
3. Unseen intervention generalization is validated on interventions, doses, combinations, or sequences excluded from model fitting.
4. Spatial context produces reproducible predictive benefit on held-out real tissue, donor, or section data beyond the matched cell-intrinsic model.
5. At least two prospective experimental cycles are completed with prediction/model locks before outcome access.
6. At least one independent cohort or external laboratory replication passes its prespecified endpoint.
7. Predictive uncertainty remains calibrated under a prespecified distribution shift.
8. Every release-level claim maps to one or more evidence-ledger entries.
9. Failed assays and negative biological results are explicitly retained and reported.
10. At least one actual longitudinal perturbation dataset has been connected to CausaFlux model training and the prospective recommendation loop.

Synthetic fixtures can establish software correctness, regression stability, split logic, calibration code, reporting, and prospective locking. They cannot satisfy any real biological criterion above.

## Evidence classes

Qualifying real evidence kinds include `real_longitudinal_perturbation`, `real_multimodal_perturbation`, `real_spatial_perturbation`, `prospective_cycle`, `external_lab_replication`, `independent_cohort_replication`, `distribution_shift_calibration`, `real_negative_result`, and `real_failed_assay`.

## Strict validation

After real evidence has been ingested:

```bash
causaflux v2-run --external-evidence-dir /path/to/locked_evidence --output causaflux_v2_real
causaflux v2-validate --input causaflux_v2_real --require-prospectively-validated
```

The strict command exits non-zero unless every real gate passes.
