# Dataset card — AMP-AD molecular validation benchmark

- **Benchmark ID:** `amp_ad_molecular`
- **Domain:** neurobiology-bulk-multiomics
- **Status:** accession-ready-controlled-components
- **Estimated storage:** 5-1000 GB
- **Primary question:** Do candidate neural–glial and molecular programs replicate across independent human Alzheimer cohorts?
- **Discovery cohort:** ROSMAP or harmonized discovery partition
- **Validation cohorts:** MSBB, MayoRNAseq, AMP-AD Diverse Cohorts
- **Outer split unit:** participant/donor

## Sources

- **harmonized_discovery:** ampad_rnaseq_harmonization — `syn21241740` (mixed open/controlled; Synapse account and conditions may be required; Study-specific Synapse access requirements and data-use terms)
- **discovery_cohort:** rosmap — `syn3219045` (controlled for many individual-level files; ROSMAP/AD Knowledge Portal data-use terms)
- **independent_validation:** msbb — `syn3159438` (mixed/controlled; MSBB/AD Knowledge Portal terms)
- **independent_validation:** mayo_rnaseq — `syn5550404` (mixed/controlled; Mayo/AD Knowledge Portal terms)
- **generalization_validation:** ampad_diverse_cohorts — `syn51732482` (mixed/controlled; Study-specific terms)

## Leakage controls

- same donor or duplicated specimen across harmonized and source cohorts

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
