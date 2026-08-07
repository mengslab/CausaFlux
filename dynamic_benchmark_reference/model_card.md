# CausaFlux v1.7.0 dynamic benchmark model card

## Models

- LatestStateLinear
- LatestStateMLP
- HistorySummaryMLP
- GRUDynamic
- CausaFluxFactorizedGRU
- IrregularTimeTransformer
- NeuralCDE
- PRESCIENTComparator

## Primary evaluation

Complete target × dose × sequence histories are held out. Validation data calibrate predictive intervals; test data are never used for training, hyperparameter selection, or calibration.

## Foundation-pretraining gate

- Status: **PASS**
- Winning dynamic model: **CausaFluxFactorizedGRU**
- Passing dynamic models: CausaFluxFactorizedGRU
- Foundation pretraining allowed: **False**

## Limitations

The Neural CDE is a dependency-light piecewise-linear Euler implementation. The PRESCIENT comparator is a lightweight latent-drift reference and not the upstream PRESCIENT software. scVI/scGPT support is provided as an interface for precomputed embeddings; no third-party checkpoints are bundled.
