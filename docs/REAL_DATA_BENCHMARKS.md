# Real-data benchmark framework

CausaFlux v1.4.0 replaces synthetic headline findings with accession-pinned benchmark definitions. It does **not** redistribute controlled, licensed, or multi-terabyte source data.

## Benchmark families

1. HTAN spatial breast cancer: hta9 discovery, hta6 and hta5 validation.
2. GDC/TCGA/CPTAC: TCGA-BRCA discovery, CPTAC-2, PDC000120 and PDC000582 validation.
3. DepMap–PRISM–LINCS: DepMap Public 26Q1 and PRISM discovery, GSE92742/GSE70138/GSE106127 validation.
4. SEA-AD: open processed single-cell, spatial and neuropathology AWS resources.
5. AMP-AD: syn21241740, ROSMAP syn3219045, MSBB syn3159438, Mayo syn5550404 and diverse cohorts syn51732482.
6. DANDI: DANDI:000048 discovery, DANDI:000039 imaging validation and DANDI:000568 electrophysiology validation.

## Commands

```bash
causaflux benchmark-list
causaflux benchmark-show --id sea_ad_neural_glial
causaflux benchmark-preflight --output preflight
causaflux benchmark-plan --id all --output realdata_benchmarks/data
causaflux benchmark-report --output realdata_benchmarks --project-root .
causaflux benchmark-validate --input realdata_benchmarks
```

`benchmark-plan` never downloads by itself. DepMap is manual by design because the portal explicitly asks users not to scrape it. Synapse and controlled GDC data require the user's own account and approvals.

## Locking policy

Every executed benchmark must produce exact file identifiers, versions, checksums, download timestamps, access class and license acceptance. `latest` labels are resolved to immutable versions before analysis.

## Validation policy

Participants or donors—not cells, sections, aliquots or repeated recordings—define the outer validation unit. Independent cohorts are never used for feature selection or calibration. Cross-program mappings are labeled reference-mapped rather than paired.

## Bundled real data

Only two small, public SEA-AD metadata workbooks are bundled: donor metadata and continuous pseudoprogression scores. They are retained unchanged with SHA-256 hashes and are used for descriptive report verification. No molecular matrix, image, genotype, or controlled subject-level file is redistributed.

## Adapter contract

Repository adapters are implemented in `causaflux.realdata_adapters`. They emit auditable commands or API payloads and accession locks; they never bypass authentication or license acceptance.
