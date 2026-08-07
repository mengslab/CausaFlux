# Biological validation framework

CausaFlux v1.4.0 separates preregistration, discovery, replication, perturbational support, causal interpretation, and clinical interpretation.

## Evidence gates

1. Hypothesis and analysis plan are frozen with SHA-256 before results are computed.
2. Discovery and replication use non-overlapping donors.
3. Primary endpoints and directions are prespecified.
4. Established statistical methods are reported beside CausaFlux estimates.
5. Cross-endpoint sensitivity analyses are identified as secondary.
6. Perturbational support is reported only when an intervention dataset has actually been executed.
7. External-dataset replication is distinct from replication in a second source cohort.
8. Causal and clinical language are disabled unless their separate gates pass.

## Executed public benchmark

The bundled public SEA-AD validation uses ACT donors for discovery and ADRC Clinical Core donors for replication. Three hypotheses are tested against Gabitto 2024 CPS and examined across Travaglini 2026 and Kana 2026 CPS definitions:

- APOE epsilon-4 carriers have higher pathological-progression scores.
- Donors with dementia have higher pathological-progression scores.
- Increasing AD neuropathological-change stage is associated with higher progression scores.

These results establish replicated observational associations within two independent SEA-AD source cohorts. They do not establish causality, biomarker utility, treatment response, or clinical guidance. AMP-AD external replication and molecular/perturbational validation remain pending.

## Commands

```bash
causaflux validation-list
causaflux validation-preregister --output validation/preregistration
causaflux validation-run --output validation
causaflux validation-validate --input validation
```

## Manuscript package

Each validation run writes editable SVG/PDF, 600-dpi PNG/TIFF, panel-level CSV source data, figure manifests, statistical methods, result tables, and a claims ledger under `manuscript_package/`.
