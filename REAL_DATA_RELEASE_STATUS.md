# Real-data release status

CausaFlux v1.4.0 changes the default demonstration from synthetic biological conclusions to a real-data registry and public metadata benchmark.

## Executed in the packaged release

- Validation of all accession manifests, citations, licenses, access classes, and discovery/validation definitions.
- Repository-specific adapter planning and accession-lock generation.
- Descriptive analysis of unchanged public SEA-AD donor metadata and continuous pseudoprogression scores.
- Validation of publication reporting, provenance, and artifact hashes.

## Accession-ready but not fully executed in the package build

- HTAN spatial assay matrices and images.
- GDC/TCGA/CPTAC/PDC molecular files.
- DepMap/PRISM and LINCS perturbation matrices.
- SEA-AD single-cell, spatial, and quantitative-neuropathology matrices.
- AMP-AD controlled or account-gated molecular files.
- DANDI NWB electrophysiology and imaging assets.

These datasets remain at their authoritative repositories because they are large, versioned, licensed, or controlled. Run `causaflux benchmark-plan` after reviewing source-specific terms.
