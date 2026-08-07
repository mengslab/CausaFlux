# Baselines, calibration, and uncertainty

## Evaluation unit

CausaFlux treats the donor as the independent evaluation and resampling unit.
Cells and longitudinal observations from a held-out donor are never used to fit
that fold's state model.

## Linear baseline panel

The panel intentionally emphasizes transparent models. It includes a prior-only
control, regularized logistic regressions, an elastic-net stochastic linear
classifier, and shrinkage linear discriminant analysis. The ensemble averages the
best donor-cross-fitted probability variant from each non-dummy model family.

## Calibration

Calibration methods consume only out-of-fold base probabilities. For each held-out
donor, the calibrator is fit using predictions and labels from all other donors.
This prevents a donor's labels from calibrating predictions for that same donor.

The selected variant minimizes donor-held-out multiclass log loss. Calibration
should also be inspected using Brier score, ECE, classwise ECE, and reliability
curves. A reduction in one metric does not guarantee improvement in all metrics.

## Donor-cluster bootstrap

Metric bootstrap intervals resample donors with replacement and include every row
from each sampled donor. Duplicate donor draws are retained as duplicate clusters.
This approximates cohort-composition uncertainty while preserving within-donor
dependence.

Row-level bootstrap probabilities are stricter: the predicted donor is excluded,
other donors are sampled with replacement, the configured L2 logistic reference model is refit,
and probabilities are generated for the held-out donor. Exported intervals are
percentile summaries across successful fits.

## Ensemble uncertainty

Let each model family produce a categorical probability vector. CausaFlux reports:

- Predictive entropy of the mean probability vector
- Mean entropy of member probability vectors
- Mutual information proxy: predictive entropy minus mean member entropy
- Variation ratio: one minus the fraction of members voting for the modal class
- Per-class standard deviation, minimum, and maximum across members

High mutual information or variation ratio indicates model-family disagreement.
Low disagreement does not imply that all models are correct, especially under
shared misspecification or distribution shift.

## Transition uncertainty

The transition matrix is recomputed after resampling complete donors and all their
longitudinal lineages. The resulting percentile intervals quantify sensitivity to
donor composition, not uncertainty from unmeasured states or incorrect Markov
assumptions.
