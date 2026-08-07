# Counterfactual therapeutics model

## Scope

CausaFlux v1.4.0 ranks five therapeutic object types:

1. gene perturbations;
2. individual drugs;
3. simultaneous two-agent combinations;
4. ordered drug sequences;
5. intervention-time alternatives.

The bundled implementation is an interpretable reference baseline. It is designed to
make assumptions and failure modes visible before more expressive pharmacology,
causal sequence models, or graph neural networks are added.

## Intervention schema

Every intervention has:

- a stable identifier and name;
- type: `gene` or `drug`;
- target and direction;
- mechanism class;
- potency and tumor selectivity;
- normal-cell toxicity prior;
- onset, half-life, and default duration;
- an optimal-time center and timing-window width;
- explicit effects on named state variables.

The default state variables are treatment stress, IRE1-XBP1 activity, proteostasis,
enhancer plasticity, mitochondrial reserve, antigen presentation, immune exclusion,
inflammatory signaling, viability, and apoptosis.

## Regimen schema

A regimen contains one or more events. Each event records:

```text
intervention_id
start_hour
dose
duration_hours
sequence_position
```

Simultaneous combinations share a start time. Sequence candidates assign different
start times and preserve event order. Timing candidates scan one intervention across a
configured time grid.

## Counterfactual calculation

For each donor-balanced reference state:

1. estimate a donor-audited baseline probability of eventual resistance;
2. calculate dose saturation, timing compatibility, onset, and persistence;
3. alter named state variables with bounded pathway-dependent effects;
4. apply mechanism-pair synergy and directional sequence modifiers;
5. predict counterfactual resistance from the modified state;
6. calculate tumor viability, apoptosis, antigen-presentation, and exclusion changes;
7. estimate normal-cell toxicity from non-tumor pathway vulnerability;
8. calculate extrapolation and multi-objective utility.

The resistance surrogate is trained with donor-grouped cross-validation. The model
never uses regimen names as predictive features.

## Utility

The default utility combines:

- resistance-risk reduction;
- tumor-viability reduction;
- apoptosis gain;
- antigen-presentation gain;
- immune-exclusion reduction;
- normal-cell toxicity;
- regimen complexity;
- extrapolation outside observed state support.

Weights are configuration choices, not biological constants. Real projects should
pre-register or sensitivity-test them.

## Uncertainty

Donors are sampled with replacement. For each bootstrap replicate, CausaFlux refits
the resistance surrogate and predicts all regimens. The release reports percentile
intervals for resistance probability, risk reduction, toxicity, and utility.

These intervals do not capture all uncertainty. They omit, among other sources:

- intervention-effect uncertainty when only prior effects are supplied;
- pharmacokinetic and pharmacodynamic model misspecification;
- off-target effects;
- unmeasured confounding;
- spatial interference;
- drug–drug metabolism and transport interactions;
- between-species translation;
- clinical dose and safety uncertainty.

## Extrapolation

A regimen is flagged when its counterfactual features move outside the 1st–99th
percentile support of the training data. This is a simple diagnostic and is not a
formal positivity guarantee.

## Pareto analysis

A regimen is Pareto optimal when no other candidate has both greater resistance-risk
reduction and lower predicted normal-cell toxicity. Pareto status does not imply that a
regimen is effective, safe, or experimentally feasible.

## Real-data requirements

A credible real-data analysis should include:

- donor- or patient-level identifiers;
- longitudinal cell-state measurements;
- perturbation identity, dose, start, duration, and sequence;
- genetic and pharmacological controls;
- tumor and relevant normal-cell readouts;
- measured response and toxicity outcomes;
- adequate overlap among treatment alternatives;
- prospective or external validation.

## Future extensions

Planned extensions can replace the reference baseline with:

- structural causal models with time-varying treatments;
- g-computation and marginal structural models;
- neural controlled differential equations;
- pharmacokinetic/pharmacodynamic compartments;
- graph-conditioned response models;
- chemical and protein foundation-model representations;
- constrained Bayesian optimization for experimental selection.


## Production stage isolation

Therapeutic fitting runs in a fresh process after spatial and causal outputs have been
serialized. Model tables and bootstrap intervals are written before figure rendering.
A separate finalization process then creates plots and the HTML report. This separation
does not change the counterfactual equations; it makes execution reproducible across
macOS and Linux scientific Python stacks.
