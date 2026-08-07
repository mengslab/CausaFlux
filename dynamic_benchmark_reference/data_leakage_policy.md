# Dynamic benchmark leakage policy

1. Complete perturbation histories are indivisible split groups.
2. A history includes target, intervention identity, dose, order, schedule, pulse shape, and recovery interval.
3. Donor holdout is evaluated separately and does not substitute for history holdout.
4. Future intervention schedules are permitted inputs only when specified before the forecasted outcome.
5. External test cohorts cannot be used for feature selection, hyperparameter tuning, early stopping, variance calibration, or model selection.
6. scVI/scGPT embeddings must be generated without fitting on held-out benchmark outcomes or leaking test-cohort labels.
