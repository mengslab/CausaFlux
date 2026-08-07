# Neurobiology configuration

CausaFlux v1.4.0 provides an interpretable reference configuration for integrating molecular state, neural and glial identity, live imaging, electrophysiology, genotype, and longitudinal sampling.

## Design principles

1. **Donors are the validation unit.** Cells from one donor never appear in both training and testing.
2. **Missing modalities are explicit.** Electrophysiology is unavailable for non-neuronal cells; it is not encoded as zero activity.
3. **State and function are separate.** A molecular state estimate does not replace measured electrical function.
4. **APOE is context, not destiny.** Genotype modifies predictions but does not define a state by itself.
5. **Trajectories are probabilistic.** State probabilities and transition intervals accompany labels.
6. **Synthetic results are not biological evidence.** The bundled cohort validates code paths only.

## State vocabulary

```text
homeostatic
compensated_stress
adaptive_glial_response
maladaptive_inflammation
synaptic_dysfunction
irreversible_degeneration
```

## Reference modalities

### RNA-like pathway features

- proteostasis capacity;
- APOE-dependent lipid exchange;
- inflammatory program;
- synaptic program;
- myelination program;
- calcium homeostasis;
- mitochondrial program.

### Live imaging

- protein-aggregate burden;
- neurite integrity;
- mitochondrial membrane potential;
- calcium-event rate and amplitude;
- microglial motility;
- astrocyte process complexity.

### Electrophysiology

- resting membrane potential;
- action-potential amplitude;
- firing rate;
- input resistance;
- spontaneous excitatory postsynaptic-current rate;
- burst synchrony.

## Main algorithms

- donor-held-out multinomial logistic state model;
- donor-held-out binary degeneration-risk model;
- donor-bootstrap transition probabilities;
- Spearman imaging–electrophysiology alignment;
- cross-modal cell-type driver score;
- APOE-stratified longitudinal aggregation.

## Real-data extension points

The same interface can accept features extracted from calcium imaging, high-content morphology, patch clamp, multielectrode arrays, single-cell or spatial RNA/ATAC, proteomics, lipidomics, and perturbation experiments. For real datasets, labels, outcome definitions, lineage matching, and missingness mechanisms must be supplied by the study team.
