# Dataset card — DANDI electrophysiology and imaging benchmark

- **Benchmark ID:** `dandi_neurophysiology`
- **Domain:** neurophysiology
- **Status:** accession-ready
- **Estimated storage:** 1-1000 by selected assets GB
- **Primary question:** Can CausaFlux extract standardized functional state features from NWB electrophysiology and optical physiology and transfer them across experiments?
- **Discovery cohort:** DANDI:000048
- **Validation cohorts:** DANDI:000039, DANDI:000568
- **Outer split unit:** animal/session/cell as appropriate, with subject grouping

## Sources

- **multimodal_discovery:** dandi_000048 — `DANDI:000048` (open; Resolve exact CC0 or CC-BY license from immutable Dandiset metadata)
- **independent_imaging_validation:** dandi_000039 — `DANDI:000039` (open; Resolve from immutable Dandiset metadata)
- **independent_ephys_validation:** dandi_000568 — `DANDI:000568` (open; Resolve from immutable Dandiset metadata)

## Leakage controls

- recordings from same subject or session across folds

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
