# Dataset card — HTAN spatial breast-cancer benchmark

- **Benchmark ID:** `htan_spatial_cancer`
- **Domain:** cancer-spatial
- **Status:** accession-ready
- **Estimated storage:** 100-5000 depending on imaging level GB
- **Primary question:** Can CausaFlux recover reproducible tumor–immune–stromal niches and communication circuits associated with treatment state?
- **Discovery cohort:** OMS metastatic breast atlas
- **Validation cohorts:** Breast Pre-Cancer Atlas, Cellular Geography of Therapeutic Resistance
- **Outer split unit:** patient/case

## Sources

- **discovery:** htan_oms_breast — `HTAN center hta9; file Synapse IDs resolved from portal query` (mixed open and controlled; NIH/HTAN data-use terms; file-level access requirements)
- **independent_validation:** htan_breast_precancer — `HTAN center hta6; file Synapse IDs resolved from portal query` (mixed open and controlled; NIH/HTAN data-use terms; file-level access requirements)
- **secondary_validation:** htan_cellular_geography_resistance — `HTAN center hta5; file Synapse IDs resolved from portal query` (mixed open and controlled; NIH/HTAN data-use terms)

## Leakage controls

- multiple sections from one patient across folds

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
