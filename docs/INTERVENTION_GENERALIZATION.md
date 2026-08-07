# CausaFlux v1.5.0 Intervention Generalization

CausaFlux v1.5.0 evaluates whether an intervention-aware model generalizes beyond observed treatments rather than memorizing perturbation labels.

## Held-out axes

The benchmark contains four explicit test strata: unseen perturbations, unseen continuous doses, unseen combinations, and unseen treatment sequences. Training, validation, and test rows are frozen before fitting.

## Model inputs

The native CausaFlux model uses gene and compound embeddings, continuous dose features, analytic PK/PD exposure summaries, pairwise interaction features, sequence/order features, and biological context covariates.

## Baselines and adapters

Native software baselines include additive prediction and nearest-neighbor retrieval. CPA, GEARS, TxPert, and scGPT are exposed through row-aligned external prediction contracts. The bundled `*AdapterProxy` models are lightweight regression fixtures used only to verify adapter/evaluation plumbing; they are not executions or reproductions of the published methods.

Actual established-model comparison therefore remains blocked until predictions produced in the corresponding external environments are imported for the identical frozen test rows.

## Longitudinal causal comparators

Sequential g-computation and a stabilized-weight marginal structural model are reported as transparent comparators. Their outputs do not become causal estimates unless consistency, exchangeability, positivity, and model-specification assumptions are substantively justified.

## Uncertainty and support

Split-conformal intervals are calibrated on validation residuals. Positivity diagnostics report nearest-support distance, intervention support, pair support, dose-range support, and sequence support. A warning identifies extrapolation; it does not make an unsupported counterfactual identifiable.

## Software exit gate

The deterministic software fixture passes only when `CausaFluxInterventionGeneralizer` has lower RMSE than additive, nearest-neighbor, and all bundled adapter proxies overall and on unseen perturbation, dose, and combination strata. Sequence generalization must beat additive and nearest-neighbor baselines.

The real established-model gate is separate and requires actual CPA, GEARS, TxPert, and scGPT predictions.
