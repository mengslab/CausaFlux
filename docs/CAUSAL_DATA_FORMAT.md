# Causal longitudinal CSV format

CausaFlux v1.4.0 expects one row per measured cell or aggregate biological unit at
one time point.

## Required identity and design columns

```text
row_id
donor_id
sample_id
lineage_id
time_hours
cell_type
state
therapy
future_resistant
```

`lineage_id` must identify the same evolving tumor lineage across time. Cells that
cannot be longitudinally linked may use population or pseudobulk identifiers, but
the interpretation of transition probabilities must then be changed accordingly.

## Bundled molecular features

```text
ire1_xbp1
proteostasis_capacity
enhancer_plasticity
mitochondrial_reserve
antigen_presentation
immune_exclusion
inflammatory_signaling
viability
apoptosis_signal
```

The default configuration additionally uses:

```text
mutation_burden
treatment_stress
resistance_score
ire1_inhibition
mitochondrial_inhibition
ifng_support
```

## Permitted tumor states

```text
treatment_sensitive
early_stress
reversible_tolerance
stable_resistance
```

The bundled non-tumor demonstration uses `responsive_niche` and `supportive_niche`.
Custom state vocabularies require corresponding code or configuration changes in
this release.

## Important design rules

- `donor_id` represents biological independence for model evaluation and bootstrap
  resampling.
- `row_id` must be unique.
- Times must be nondecreasing within each tumor lineage.
- Treatment assignment, timing, dose, and comparator must be encoded explicitly.
- Confounders used for adjustment must be measured before treatment or otherwise
  justified by the causal graph.
- Do not use post-treatment mediators as ordinary confounders.
