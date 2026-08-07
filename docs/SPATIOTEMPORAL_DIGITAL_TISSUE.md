# CausaFlux v1.7.0 — Spatiotemporal Digital Tissue

## Scientific objective

CausaFlux v1.7.0 tests whether time-varying tissue neighborhood context provides reproducible predictive information beyond a cell's intrinsic molecular/state measurements. The primary software gate is evaluated under two atomic split regimes:

1. **held-out tissue sections** — no section is split across train/test;
2. **held-out donors** — no donor is represented in train and test.

The bundled reference is synthetic software-validation data, not biological evidence.

## Architecture

The v1.6 tissue layer contains:

- persistent cell identities across irregular tissue snapshots;
- time-varying heterogeneous cell–cell graphs;
- typed sender/receiver relations;
- learned communication-edge gates trained on training edges only;
- graph-conditioned continuous state-rate prediction;
- within-cell regulatory graph tables;
- within-cell organelle graph tables;
- cell-intrinsic versus neighborhood prediction decomposition;
- standardized spatial-interference estimands;
- a tissue-level outcome head;
- an external Nicheformer embedding adapter contract.

### Continuous dynamics

For each cell, the model predicts a state derivative and integrates it over the observed interval:

`z(t + Δt) = z(t) + Δt · f(z(t), neighborhood(t), regulatory(t), organelle(t))`

The benchmark compares the full model against:

- `CellIntrinsicContinuous`;
- `UngatedNeighborhoodContinuous`;
- `NicheformerAdapterProxy`;
- `CausaFluxSpatiotemporalGNN`.

The Nicheformer proxy is an interface-regression fixture only. It is not an execution of the published Nicheformer model.

## Learned communication-edge gates

The gate receives relation type, sender ligand-like activity, receiver receptor-like activity and distance. In the bundled synthetic fixture, continuous edge activity is available as software ground truth. Real datasets must define a prespecified edge-learning target or use externally supplied communication scores.

## Spatial interference

The bundled estimator reports a standardized contrast obtained by setting learned neighborhood-damage exposure to training-set Q75 versus Q25 while holding intrinsic covariates fixed. This is a software demonstration of the estimand interface, not a biological causal claim.

## Regulatory and organelle graph layers

Each cell snapshot exports explicit within-cell edges. The software fixture includes representative regulatory edges such as XBP1→proteostasis and organelle edges such as ER→mitochondria. Real projects should substitute experimentally supported graph definitions.

## Nicheformer adapter

CausaFlux does not redistribute Nicheformer weights or environments. The adapter is file-contract based so embeddings generated with the official implementation can be aligned to CausaFlux `row_id` values. The official Nicheformer publication is Tejada-Lapuerta et al., *Nature Methods* 22, 2525–2538 (2025), DOI 10.1038/s41592-025-02814-z.

Expected embedding table:

```text
row_id,niche_embedding_0,niche_embedding_1,...
```

## Exit criterion

The software gate passes only when `CausaFluxSpatiotemporalGNN`:

- improves multivariate future-state RMSE over `CellIntrinsicContinuous`;
- has a positive section-cluster bootstrap lower confidence bound for that improvement;
- improves tissue-level outcome RMSE over `CellIntrinsicContinuous`;
- learns predictive communication gates;

under both held-out-section and held-out-donor evaluation.

The destructive-state scalar remains a secondary diagnostic and is not substituted for the prespecified multivariate state endpoint.

## Real-data boundary

The real-tissue validation gate remains blocked until these same criteria are tested on real longitudinal or pseudo-longitudinal spatial tissue data with predeclared sections, donors and outcomes. Foundation pretraining is not authorized by the synthetic software gate.
