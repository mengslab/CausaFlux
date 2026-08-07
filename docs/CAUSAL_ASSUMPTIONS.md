# Causal interpretation checklist

CausaFlux separates software computation from scientific identification. Before
interpreting an effect causally, document the following.

## Estimand

Define the intervention, comparator, biological unit, outcome, and time horizon.
The default demonstration estimates the average effect of adding a pathway-directed
intervention to standard therapy on final binary resistance.

## Exchangeability

The configured baseline covariates must block relevant noncausal paths between
treatment and outcome. Unmeasured confounding can invalidate the estimate.

## Positivity

Biological units with each relevant covariate profile must have a nonzero chance of
receiving both compared treatments. The output reports the observed propensity
range, but full diagnostics remain the investigator's responsibility.

## Consistency

The encoded treatment label must correspond to a sufficiently well-defined
intervention. Dose, timing, formulation, and sequence can violate consistency when
collapsed into a single label.

## Interference

The current AIPW estimator assumes one lineage's assigned treatment does not alter
another lineage's outcome. This assumption is especially problematic in spatial,
immune, and shared-culture experiments and should be modeled explicitly in future
multicellular graph releases.

## Temporal order

Adjustment variables should precede treatment. Biomarkers should precede the
transition or outcome they are claimed to predict.

## Sensitivity and validation

Use negative controls, alternate graph specifications, different adjustment sets,
leave-donor/model-out validation, measurement-error analyses, and prospective
perturbation experiments. A narrow confidence interval does not protect against a
wrong graph or systematic bias.
