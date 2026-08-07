# CausaFlux v1.8.0 Prospective Experimental Loop

## Purpose

The v1.8 layer separates **model development** from **prospective evidence generation**. Its core rule is simple: no outcome can influence a recommendation that is later scored as prospective. Each cycle therefore has an immutable model freeze and prediction lock created before QC/outcome access.

## State machine

```text
MODEL_FROZEN
    ↓
PREDICTIONS_EXPORTED_AND_LOCKED
    ↓
EXPERIMENT_CONTRACT_RELEASED
    ↓
QC_AND_OUTCOMES_INGESTED
    ↓
LOCKED_EVALUATION
    ↓
POSTERIOR_UPDATED
    ↓
NEXT_MODEL_FREEZE
```

A cycle cannot skip the prediction lock. External outcome ingestion rejects experiment IDs not present in the locked contract.

## LIMS/ELN contract

The schema is intentionally generic enough for CSV export/import into common LIMS and ELN systems. It contains study/cycle/experiment/sample identifiers, plate/well/run metadata, assay and perturbation fields, dose/time fields, batch/replicate/randomization metadata, protocol/sample-manifest URIs, expected cost, model-freeze ID, preregistration ID, and experiment status.

CausaFlux does not assume a particular vendor. Institutions can map the columns to Benchling, LabVantage, Sapio, Signals Notebook, custom REDCap-style tables, or internal databases without changing the model logic.

## Prediction preregistration

`preregistered_predictions.csv` contains the complete candidate-level forecast table, not just the selected experiments. The lock stores its SHA-256 digest and the digest of `selected_experiments.csv`. This preserves:

- the ranking actually available at decision time;
- predictions for candidates not selected;
- uncertainty and expected information gain available before outcome access;
- the ability to audit whether the recommendation changed post hoc.

## Failed assays

A technical failure does **not** disappear from the analysis. The v1.8 rule is:

1. failed or primary-endpoint-unusable assays are excluded from posterior updating;
2. failed assays are excluded from locked prediction-error metrics because the primary outcome is unavailable;
3. attempted cost remains fully charged;
4. failure reason and QC fields are preserved;
5. replacement, when used in a real protocol, must be chosen from the next preregistered eligible rank without inspecting the failed biological outcome.

This prevents AI-guided strategies from appearing artificially efficient by ignoring failed experiments.

## Posterior model update

The reference implementation performs a Bayesian update over competing mechanism hypotheses. Neural/foundation parameters are frozen during each cycle. A real implementation may replace this posterior state with a retrained or fine-tuned CausaFlux model, but the next cycle must receive a new freeze manifest and must never inherit outcomes without recording them in the update provenance.

## Adaptive stopping

The reference rules are all prespecified before Cycle 1:

- complete at least three cycles for the v1.8 prospective demonstration;
- stop after the maximum number of cycles;
- after the minimum-cycle requirement, stop if posterior confidence exceeds the threshold;
- stop if all remaining expected information gains fall below the threshold;
- stop if the total prespecified budget is exhausted.

A real study can change numerical thresholds only before the relevant prediction lock or by creating a documented protocol amendment; it should not silently edit prior locks.

## Calibration

For each cycle the system records:

- prediction RMSE;
- 90% prediction-interval coverage;
- Brier score for discovery probability;
- mean predicted discovery probability;
- observed discovery fraction;
- recovery trajectories identified.

`cycle_calibration.csv` enables detection of calibration drift as the adaptive loop moves away from the original training distribution.

## Non-AI comparator

The reference comparator is a fixed-order, budget-constrained strategy defined before outcomes are sampled. It cannot use model scores, posterior probabilities, or observed outcomes to reorder experiments. Both policies receive the same deterministic potential outcome for a given experiment ID, ensuring a fair software-level comparison.

For a real study, the comparator should be selected before Cycle 1 and could be:

- expert-prioritized fixed order;
- uniform/random design with a locked seed;
- standard dose-escalation schedule;
- conventional factorial design;
- current laboratory SOP.

## Exit criterion

At least one prespecified metric must improve relative to the non-AI comparator:

- outcome discovery;
- uncertainty reduction;
- experiment efficiency;
- recovery-trajectory identification.

For real prospective validation, report all four metrics regardless of which one passes.

## Evidence language

Passing the bundled synthetic gate permits the statement:

> “The v1.8 software correctly executes and audits a three-cycle prospectively locked experimental loop in a synthetic validation fixture.”

It does **not** permit:

> “CausaFlux prospectively improves biological discovery.”

The latter requires real experiments with the same locking and comparison rules plus independent Cycle 3 confirmation/falsification.
