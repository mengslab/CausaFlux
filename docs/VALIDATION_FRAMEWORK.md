# Validation Framework

CausaFlux v1.4.0 uses four validation layers.

## 1. Data-contract validation

Checks identifiers, modality alignment, required metadata, time units, treatment fields, missingness, graph endpoints, probability normalization, and output schemas.

## 2. Statistical validation

Donors define held-out folds. Bootstrap procedures resample donors rather than cells. Calibration and ensemble uncertainty are generated without donor leakage. Transition, therapeutic, biomarker, and experiment intervals are checked for ordering and valid probability ranges.

## 3. Scientific-workflow validation

The release verifier checks all supported domains: multimodal data, spatial graph, causal effects, counterfactual therapeutics, biomarkers, active learning, and neurobiology. It confirms category coverage, budget constraints, posterior normalization, exclusion of completed experiments, and imaging/electrophysiology outputs.

## 4. Platform validation

The platform gate confirms:

- framework and version consistency;
- workflow completion;
- cancer and neurobiology domain coverage;
- donor separation;
- integrated, neurobiology, and platform reports;
- dataset and model cards;
- environment snapshot;
- SHA-256 artifact integrity;
- synthetic-data disclosure;
- packaged demo registry.

“Validated research platform” means these software, reproducibility, and analysis-contract checks pass. It does not mean clinical or biological validation.

## 5. Publication and visual-regression validation

The graphics gate checks all registered panels for:

- profile dimensions and final-size typography;
- valid PNG and 600-dpi LZW TIFF raster files;
- editable SVG and PDF exports;
- panel-level source-data tables;
- figure manifests and synthetic-only disclosure;
- deterministic graph-layout coordinates;
- perceptual-hash visual baselines;
- complete inventory coverage across cancer and neurobiology workflows.

Visual regression is a software-stability check. It does not establish biological validity of the synthetic demonstration.

## Biological-evidence gates in v1.4.0

Biological claims are governed separately from software correctness. A hypothesis must be preregistered, supported in discovery, and replicated in non-overlapping donors before CausaFlux permits the label `replicated_association`. External-dataset replication, perturbational support, causal interpretation, biomarker utility, and clinical utility are separate gates. Missing gates cannot be inferred from model performance, graph centrality, or statistical significance.

## Dynamic-model gate in v1.4.0

The primary benchmark withholds complete target–dose–sequence histories. The software performance gate requires one dynamic model to achieve lower validation-calibrated Gaussian NLL and lower fate log loss than both `LatestStateMLP` and `HistorySummaryMLP`, while also reducing trajectory RMSE relative to `LatestStateMLP`.

The bundled synthetic reference can pass the software performance gate, but it cannot authorize foundation pretraining. `foundation_pretraining_allowed` remains false until a real external longitudinal dataset passes the same criteria and receives human review.

## Multimodal dynamic-state gate in v1.4.0

The v1.4 gate is intentionally separate from the v1.3 dynamic-history gate.

The primary test set contains perturbation histories absent from training. Donor overlap is permitted and explicitly reported because this gate asks whether a new perturbation history can be forecast in potentially observed donors.

Prespecified baselines are:

1. baseline covariates only;
2. latest RNA plus baseline covariates;
3. static latest-snapshot multimodal fusion.

A qualifying v1.4 model must use the temporal history of imaging and reporters, have lower calibrated log loss and Brier score than all three baselines, and maintain AUC within 0.02 of the strongest baseline. The package also reports a PoE imaging/reporter ablation but does not use synthetic ablation superiority as authorization for foundation pretraining.

Missingness is evaluated under observed, MCAR, destructive-state-dependent imaging/reporting, and assay-quality-dependent omics scenarios. These are sensitivity analyses rather than identified causal missingness models.

## v1.6 spatiotemporal tissue gate

The primary tissue gate is evaluated independently under atomic held-out-section and complete held-out-donor splits. For both regimes, the full neighborhood-conditioned model must improve multivariate future-state RMSE and tissue-outcome RMSE over the matched cell-intrinsic model, and the section-cluster bootstrap lower bound for state-RMSE improvement must be positive. Communication-edge gate AUC must also be predictive in the synthetic software fixture. Secondary scalar endpoints remain visible even when they do not improve.
