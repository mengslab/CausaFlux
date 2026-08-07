# CausaFlux v1.4.0 Causal Biomarker Engine

## Purpose

The engine prioritizes measurements that can warn of a future disease transition and
that are mechanistically close to a configured causal outcome node. It does not claim
that a biomarker is itself a manipulable cause.

## Candidate-level calculations

For every candidate and every measurement time before the terminal outcome:

1. calculate a direction-oriented binary AUC;
2. calculate a standardized group effect;
3. evaluate direction and AUC separately within donors;
4. quantify association of change from baseline with the outcome;
5. select the earliest time meeting warning AUC and donor-stability thresholds;
6. calculate lead time to the terminal outcome;
7. calculate evidence-weighted graph distance to the target node;
8. determine whether an intervention node is upstream;
9. calculate assayability and redundancy components;
10. combine the normalized components using explicit weights.

## Default score weights

```text
association                 0.19
donor stability             0.14
causal proximity            0.17
perturbational support      0.10
lead-time fraction          0.14
temporal delta association  0.08
assayability                0.11
uniqueness                  0.07
```

The score weights are stored in `BiomarkerConfig.score_weights` and can be replaced.

## Causal proximity

The engine finds the shortest directed path from the candidate node to the configured
target. Edge evidence is mapped to a weight and averaged along the path. The proximity
score combines inverse path length and path evidence.

A missing path produces a proximity score of zero. This is conservative with respect to
the configured graph, but it also means that incomplete graphs can under-rank useful
candidates.

## Bootstrap uncertainty

The donor is the resampling unit. Donors are sampled with replacement; duplicated donors
receive unique bootstrap identifiers. The complete time-course ranking is recomputed in
every replicate.

Reported quantities include:

- score percentile interval;
- lead-time percentile interval;
- median bootstrap rank;
- probability of ranking in the top three.

## Compact panels

Candidates are considered in uncertainty-adjusted rank order. Highly redundant
candidates are skipped. Panels are evaluated with leave-one-donor-out scaling and
orientation. The exported panel predictions contain one row per lineage and panel size.

## Required columns

```text
row_id
donor_id
lineage_id
time_hours
cell_type
future_resistant
<candidate features>
```

The outcome column, target cell type, and causal target node are configurable.

## Limitations

- Univariate AUC is not a causal effect.
- Graph distance depends on the assumed graph.
- Perturbational support depends on intervention nodes and edges supplied by the user.
- Synthetic assayability values are placeholders.
- Bootstrap intervals do not account for all model, assay, cohort, and causal uncertainty.
- External validation remains mandatory.
