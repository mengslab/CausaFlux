# Dataset card — SEA-AD neural–glial benchmark

- **Benchmark ID:** `sea_ad_neural_glial`
- **Domain:** neurobiology-single-cell-spatial
- **Status:** includes-real-metadata-snapshot
- **Estimated storage:** 3-5000 by selected regions and image levels GB
- **Primary question:** Can neural–glial state trajectories and spatial vulnerability patterns reproduce across regions and modalities?
- **Discovery cohort:** MTG multimodal atlas
- **Validation cohorts:** region-held-out SEA-AD, AMP-AD external cohorts via amp_ad_molecular benchmark
- **Outer split unit:** donor

## Sources

- **discovery:** sea_ad_processed_single_cell — `AWS s3://sea-ad-single-cell-profiling/; Brain Knowledge project UMSVXTDIAZTAFKGE43T` (open processed data; Allen Institute Terms of Use and Citation Policy)
- **spatial_discovery:** sea_ad_spatial — `AWS s3://sea-ad-spatial-transcriptomics/` (open; Allen Institute Terms of Use and Citation Policy)
- **phenotype_anchor:** sea_ad_neuropathology — `AWS s3://sea-ad-quantitative-neuropathology/` (open; Allen Institute Terms of Use)
- **within-program_validation:** sea_ad_multiregion — `SEA-AD processed 11-region release (June 2026)` (open processed; Allen Institute Terms of Use)

## Leakage controls

- cells or regions from same donor across donor-held-out folds

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
