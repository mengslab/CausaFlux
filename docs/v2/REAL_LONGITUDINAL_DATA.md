# Real longitudinal perturbation data in CausaFlux v2.0.0

CausaFlux v2.0.0 introduces a direct bridge from real longitudinal perturbation tables into the same held-out-history dynamic benchmark used by the virtual-cell model.

## Public starting datasets

The release registry includes:

- **GSE8057** — A2780 ovarian cancer cells exposed to cisplatin or oxaliplatin. The GEO study includes time-course samples before treatment and at 0, 2, 6, 16, and 24 hours around a two-hour drug exposure, together with concentration-response samples. Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8057
- **GSE70138** — Broad LINCS Phase II L1000 perturbational profiles. Metadata encode perturbagen, cell line, dose, and time point and therefore support large-scale intervention/time/dose holdout designs. Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138
- **GSE101406** — matched P100 phosphoproteomic, GCP chromatin-PTM, and L1000 transcriptional perturbation profiles for 90 small molecules in six cell lines, measured at 3, 24, and 6 hours respectively. Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE101406

CausaFlux does not silently redistribute the large repository files. Download them from the authoritative repository and retain source metadata/checksums.

## Table contract

A local real-data table must contain:

`trajectory_id, donor_id, time, history_id, target, dose, sequence, fate`

Molecular/state features use the prefix `feature__`; intervention channels use `int__`.

Example:

```text
trajectory_id,donor_id,time,history_id,target,dose,sequence,fate,int__cisplatin,feature__XBP1,feature__ATF4
T001,D001,0,cis_25uM,cisplatin,25,continuous,recovery,0,0.20,0.15
T001,D001,2,cis_25uM,cisplatin,25,continuous,recovery,1,0.62,0.41
...
```

## Convert and benchmark

```bash
causaflux longitudinal-convert \
  --input real_longitudinal.csv \
  --output real_longitudinal.npz \
  --manifest real_longitudinal_manifest.json

causaflux longitudinal-benchmark \
  --input real_longitudinal.csv \
  --output real_longitudinal_benchmark
```

The default benchmark compares latest-state and history-summary static models with the factorized CausaFlux dynamic model while holding out complete intervention histories.

## Prospective-loop connection

A real longitudinal benchmark result becomes release evidence only after it is entered into the v2 evidence ledger. A dataset being downloaded, registered, or successfully parsed is not by itself evidence of model superiority.
