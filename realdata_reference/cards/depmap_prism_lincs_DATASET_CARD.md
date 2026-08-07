# Dataset card — DepMap–PRISM–LINCS perturbation benchmark

- **Benchmark ID:** `depmap_prism_lincs`
- **Domain:** cancer-perturbation
- **Status:** accession-ready
- **Estimated storage:** 30-200 GB
- **Primary question:** Can gene dependencies, drug sensitivity, and perturbational signatures jointly prioritize reproducible state-specific interventions?
- **Discovery cohort:** DepMap 26Q1 + PRISM
- **Validation cohorts:** LINCS Phase I, LINCS Phase II, LINCS genetic perturbations
- **Outer split unit:** cell line/model

## Sources

- **discovery:** depmap_26q1 — `DepMap Public 26Q1` (public bulk download; acceptance of file-level terms required; Broad-generated DepMap release files generally CC BY 4.0; verify each file)
- **drug_response_discovery:** prism_repurposing — `PRISM Repurposing Secondary Screen` (public bulk download with file-level terms; Verify file-specific terms in DepMap download panel)
- **independent_signature_validation:** lincs_phase1 — `GSE92742` (open; NCBI GEO terms and submitter-provided usage; citation required)
- **independent_signature_validation:** lincs_phase2 — `GSE70138` (open; NCBI GEO terms and submitter-provided usage)
- **genetic_signature_validation:** lincs_genetic_perturbations — `GSE106127` (open; NCBI GEO terms)

## Leakage controls

- same model or compound analogs across folds without scaffold grouping

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
