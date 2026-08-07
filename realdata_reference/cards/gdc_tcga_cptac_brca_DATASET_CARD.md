# Dataset card — GDC TCGA/CPTAC breast molecular benchmark

- **Benchmark ID:** `gdc_tcga_cptac_brca`
- **Domain:** cancer-molecular
- **Status:** accession-ready
- **Estimated storage:** 20-500 depending on raw versus harmonized level GB
- **Primary question:** Do multimodal molecular state representations replicate across TCGA genomics and CPTAC proteogenomics?
- **Discovery cohort:** TCGA-BRCA
- **Validation cohorts:** CPTAC-2/PDC000120, PDC000582
- **Outer split unit:** patient

## Sources

- **discovery:** tcga_brca — `TCGA-BRCA` (open harmonized derived files; controlled raw/germline files; NCI GDC data-use and dbGaP rules by file access class)
- **independent_validation:** cptac2_breast_genomics — `CPTAC-2` (mixed open and controlled; NCI GDC/dbGaP terms)
- **independent_validation:** cptac_brca_proteome — `PDC000120` (open processed proteomic data; file-specific terms; NCI PDC data-use terms)
- **therapy_response_validation:** calgb40601_proteogenomics — `PDC000582` (open/controlled by file; NCI PDC data-use terms)

## Leakage controls

- aliquots or omics from one patient across folds

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
