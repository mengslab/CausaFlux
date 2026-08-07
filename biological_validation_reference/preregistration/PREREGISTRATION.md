# CausaFlux v1.7.0 preregistration

Hypotheses and primary analysis choices were frozen before results were computed.

## BV-SEA-AD-001 — APOE epsilon-4 and pathological progression
APOE epsilon-4 carriers have higher continuous pseudo-progression scores than non-carriers.
- Status: executed
- Primary endpoint: Gabitto 2024 CPS
- Discovery: SEA-AD ACT donors
- Replication: SEA-AD ADRC Clinical Core donors
- Analysis: one-sided Mann-Whitney U plus age/sex-adjusted OLS with donor bootstrap
- SHA-256: `91beddadce392580569487c15fb030fbaabbef4aa1f71d778cceee2c0dfd9ca5`

## BV-SEA-AD-002 — Dementia status and pathological progression
Donors with dementia have higher continuous pseudo-progression scores than donors without dementia.
- Status: executed
- Primary endpoint: Gabitto 2024 CPS
- Discovery: SEA-AD ACT donors
- Replication: SEA-AD ADRC Clinical Core donors
- Analysis: one-sided Mann-Whitney U plus age/sex-adjusted OLS with donor bootstrap
- SHA-256: `1822b5163aea0108d86d2ae0f6ef9799d365aed3339630d846e5d8225f6a896e`

## BV-SEA-AD-003 — Neuropathological stage and pathological progression
Increasing AD neuropathological-change stage is monotonically associated with higher continuous pseudo-progression score.
- Status: executed
- Primary endpoint: Gabitto 2024 CPS
- Discovery: SEA-AD ACT donors
- Replication: SEA-AD ADRC Clinical Core donors
- Analysis: one-sided Spearman rank correlation plus age/sex-adjusted OLS with donor bootstrap
- SHA-256: `eedf9b2eb123bf7ce28aa2d6c3262fe9b12f2b6bf3f068d348738d8bf80f01e6`

## BV-HTAN-001 — Macrophage barrier and treatment resistance
Spatial macrophage barriers are associated with lower antigen-presentation connectivity and greater residual-disease burden.
- Status: preregistered_pending_data
- Primary endpoint: residual disease burden
- Discovery: HTAN metastatic breast cancer atlas
- Replication: HTAN cellular geography of therapeutic resistance
- Analysis: donor-blocked spatial permutation and mixed-effects regression
- SHA-256: `97a5d1c55e9c313be4dfafa9d7d33da371bfcbcb3b45bc496da48faaae4f797e`

## BV-DEPMAP-001 — Proteostasis dependence and drug response
IRE1-XBP1 proteostasis dependence predicts sensitivity to proteostasis-disrupting combinations after lineage and growth-rate adjustment.
- Status: preregistered_pending_data
- Primary endpoint: PRISM viability response
- Discovery: DepMap Public 26Q1 and PRISM
- Replication: held-out lineages and independent LINCS accessions
- Analysis: nested lineage-held-out regression against elastic net and random forest
- SHA-256: `15ef979aa16fd4b034d8b795d942c9a1c4adc35517ee1c8d6d8841c510edcc41`

## BV-SEA-AD-004 — Glial stress programs and pathological progression
Microglial inflammatory and astrocytic proteostasis programs increase before neuronal dysfunction across pathological progression.
- Status: preregistered_pending_molecular_data
- Primary endpoint: cell-state program score versus CPS
- Discovery: SEA-AD MTG single-cell atlas
- Replication: AMP-AD ROSMAP and MSBB
- Analysis: donor-blocked pseudobulk regression and region-held-out replication
- SHA-256: `ff2a87d14a44199a80ef3a4b6c7c2c64b1a16703827e88b05237823bb10319f9`
