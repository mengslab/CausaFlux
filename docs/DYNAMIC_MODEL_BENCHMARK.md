# CausaFlux v1.4.0 Dynamic Model Benchmark

## Scientific question

The release evaluates one narrowly defined claim:

> Can a dynamic model forecast future molecular states and terminal fate for completely unseen perturbation histories better than static models that observe only the latest state or engineered history summaries?

The release does not treat random cell holdout or donor holdout as sufficient evidence of perturbation generalization. The primary split unit is the complete intervention history:

```text
target × dose × sequence × schedule × recovery interval
```

Every trajectory sharing a held-out history remains outside training and validation.

## Model registry

### Static baselines

- `LatestStateLinear`: one linear future-state and fate model using the latest observation and known future intervention schedule.
- `LatestStateMLP`: nonlinear static model using the same latest-state information.
- `HistorySummaryMLP`: static model using last, mean, standard deviation, slope, cumulative intervention, and future schedule.

### Dynamic models

- `GRUDynamic`: conventional recurrent encoder with an autoregressive future decoder.
- `CausaFluxFactorizedGRU`: separates stable identity, context, reversible adaptation, and monotonically accumulating commitment.
- `IrregularTimeTransformer`: attention model with explicit continuous-time features.
- `NeuralCDE`: piecewise-linear Euler neural controlled differential equation implemented without requiring `torchcde`.
- `PRESCIENTComparator`: dependency-light latent-drift comparator inspired by potential-driven population dynamics. It is not the upstream PRESCIENT package.

### Optional static embeddings

The release defines interfaces for precomputed scVI and scGPT embeddings. It does not download or redistribute third-party checkpoints. Embeddings can replace or augment the observation tensor in an external benchmark.

## Input contract

The external NPZ contract requires:

```text
observations       float32 [trajectory, time, feature]
interventions      float32 [trajectory, time, intervention]
times              float32 [trajectory, time]
fates              int64   [trajectory]
trajectory_ids     string  [trajectory]
donor_ids          string  [trajectory]
history_ids        string  [trajectory]
targets             string  [trajectory]
doses               float32 [trajectory]
sequences           string  [trajectory]
feature_names       string  [feature]
intervention_names  string  [intervention]
fate_names          string  [fate]
```

`history_id` must encode every intervention attribute that could leak the answer: target, perturbation identity, dose, order, pulse shape, timing, and recovery interval.

## Split policies

### Perturbation-history split

The primary release gate. Complete histories are assigned to train, validation, or test using a deterministic SHA-256 rank. Train/test history overlap must be zero.

### Dose holdout

The reference fixture withholds the highest configured dose. External datasets may specify another dose interval.

### Sequence holdout

The reference fixture withholds a complete stress–rescue–stress sequence.

### Temporal extrapolation

Uses a held-out-history split and evaluates only observations after the context window.

### Donor holdout

Complete donors are separated across train, validation, and test. Donor holdout supplements but does not replace history holdout.

## Forecast task

The reference fixture contains eight irregular observations. The first five form the context and the final three form the forecast horizon. Every model receives the future intervention schedule because it is a planned input, not a post-outcome measurement.

Outputs are:

- three-step future molecular trajectory;
- Gaussian predictive uncertainty;
- terminal fate probabilities for recovery, persistent dysfunction, and death.

## Metrics

### Trajectory

- standardized RMSE;
- standardized MAE;
- trajectory correlation;
- validation-calibrated Gaussian negative log likelihood.

### Fate

- accuracy;
- macro F1;
- multiclass log loss;
- Brier score;
- expected calibration error.

### Uncertainty

- empirical 50%, 80%, 90%, and 95% interval coverage;
- corresponding interval widths;
- validation-only scale calibration.

No test observation is used for model selection or uncertainty calibration.

## Exit gate

The software performance gate is blocked unless one dynamic model simultaneously achieves:

1. lower trajectory RMSE than `LatestStateMLP`;
2. lower calibrated Gaussian NLL than both baselines;
3. lower fate log loss than both baselines;
4. zero train/test history overlap.

The machine-readable result is written to:

```text
foundation_pretraining_gate.json
```

A blocked gate is a valid benchmark outcome. It must not be overridden by changing the report text. Passing the bundled synthetic performance gate does not authorize foundation pretraining; the machine-readable guard remains `BLOCKED_REAL_LONGITUDINAL_GATE_REQUIRED` until an external real dataset passes the same criteria and receives human review.

## Reference fixture

The bundled fixture models IRE1–XBP1, PERK–ATF4, and ATF6 stress programs under continuous, pulse-recovery, delayed-rescue, pulsatile, and stress–rescue–stress sequences. Hidden damage, commitment, repair reserve, and inflammatory memory create genuine historical dependence that is not fully recoverable from the latest observation.

The fixture is used only for:

- model API verification;
- leakage-policy validation;
- metric and uncertainty testing;
- release-gate regression;
- publication figure generation.

It is not biological evidence.

## Commands

```bash
causaflux dynamic-benchmark-run \
  --output causaflux_v1.4.0_dynamic_benchmark \
  --require-gate

causaflux dynamic-benchmark-validate \
  --input causaflux_v1.4.0_dynamic_benchmark

causaflux dynamic-benchmark-export-fixture \
  --output dynamic_benchmark_fixture_v1.4.0.npz
```

A smaller developer run can reduce epochs or select models explicitly:

```bash
causaflux dynamic-benchmark-run \
  --models LatestStateMLP HistorySummaryMLP CausaFluxFactorizedGRU NeuralCDE \
  --epochs 8 \
  --replicates-per-history 3 \
  --output developer_benchmark
```

## Output structure

```text
dynamic_benchmark/
├── benchmark_config.json
├── trajectory_metadata.csv
├── split_manifest.csv
├── split_audit.json
├── split_diagnostics.csv
├── external_benchmark_contract.json
├── optional_embedding_adapters.csv
├── model_comparison.csv
├── metric_intervals.csv
├── foundation_pretraining_gate.json
├── dynamic_benchmark_status.json
├── dynamic_benchmark_validation.json
├── models/
│   └── <model>/
│       ├── checkpoint.pt
│       ├── metrics.json
│       ├── training_history.csv
│       └── test_predictions.csv
├── figures/
│   ├── SVG/PDF/PNG/TIFF exports
│   ├── source_data/
│   └── figure_manifests/
└── report/index.html
```

## Real-data progression

Passing the synthetic release gate proves only that the software can detect a history-dependent signal under a known generative system. The next evidence step is execution on a real longitudinal UPR perturbation dataset with locked target, dose, sequence, recovery, donor, and outcome definitions.
