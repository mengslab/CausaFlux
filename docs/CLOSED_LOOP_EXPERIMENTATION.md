# CausaFlux v1.4.0 Closed-Loop Experimentation

## Purpose

The closed-loop engine selects experiments that are expected to distinguish among explicitly stated causal mechanisms. It does not infer that one mechanism is true merely because it has the highest prior or because one software-simulated result favors it.

## Hypothesis model

Each hypothesis has:

- a stable identifier;
- a biological interpretation;
- a linked mechanism;
- a prior probability.

Priors must sum to one. They are assumptions supplied by the research team and should be documented before outcome inspection.

## Candidate experiment model

Every candidate records:

- experiment type: CRISPR, drug, imaging, or sampling time;
- target and mechanism;
- planned measurement time;
- primary readout;
- relative cost and duration;
- technical risk;
- measurement noise;
- model uncertainty;
- expected standardized readout under every hypothesis.

The expected readout vector is the core discriminative object. Candidates with nearly identical predictions under all hypotheses have low information value even if they may have therapeutic value.

## Expected information gain

For hypothesis variable `H` and a candidate result `Y`, CausaFlux estimates:

```text
I(H;Y) = H(H) - E_Y[H(H | Y)]
```

The observation model is a hypothesis-specific Gaussian distribution. Monte Carlo integration estimates the expected posterior entropy. The output is reported in nats and as a fraction of the maximum entropy of the configured hypothesis set.

## Composite priority

The nominal score combines:

- expected information gain;
- therapeutic value;
- biomarker value;
- temporal value;
- feasibility.

Cost, duration, and technical risk contribute to feasibility but remain visible as separate columns. The score is a planning aid, not a biological truth metric.

## Batch selection

The first batch is selected under:

- a total relative budget;
- a maximum batch size;
- a maximum number per experiment type;
- a diversity penalty for repeated mechanisms or methods;
- optional type coverage across CRISPR, drug, imaging, and sampling time.

When all four experiment types can fit within the first-round budget and capacity, CausaFlux chooses a feasible one-per-type portfolio before filling remaining capacity.

## Uncertainty

The release perturbs:

- hypothesis priors through a Dirichlet distribution;
- predicted mechanism-conditioned readouts through configured model uncertainty.

It then recomputes expected information gain and rank. The framework reports information-gain intervals and the probability that a candidate falls within the nominal batch size.

This uncertainty model does not include every source of biological, assay, causal, or operational uncertainty.

## Updating from completed experiments

Completed results require:

```text
experiment_id
observed_standardized_readout
standard_error_or_posterior_sd   optional
```

The engine applies the configured likelihood model sequentially, normalizes the posterior after each result, excludes completed experiment IDs, and recomputes the next ranking and batch.

## Outcome templates

The exported template includes:

- primary and mechanistic readouts;
- required controls;
- minimum biological-replicate guidance;
- quality-control fields;
- result and uncertainty field names;
- a decision rule stating that no single experiment should establish a mechanism.

These are general software defaults and must be replaced by domain-appropriate experimental protocols.

## Limitations

- Expected readout models can be wrong.
- Competing hypotheses may be incomplete or nonexclusive.
- Observation distributions may be non-Gaussian.
- Candidate outcomes may be correlated.
- Batch effects, interference, adaptive stopping, and failed assays require explicit handling.
- Information gain does not guarantee therapeutic relevance.
- High predicted utility does not establish safety or efficacy.
