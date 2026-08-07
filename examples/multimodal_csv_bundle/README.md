# Multimodal CSV bundle template

Create six files in this directory:

- `obs.csv`
- `rna.csv`
- `atac.csv`
- `protein.csv`
- `mutation.csv`
- `drug_response.csv`

All files must contain `row_id`. `obs.csv` also contains donor, sample, lineage,
time, cell type, state, therapy, and outcome metadata. Modality files contain only
`row_id` plus numeric measurements.

Generate a complete example bundle with:

```bash
sh run.sh
```

Then inspect `causaflux_v1.4.0_output/multimodal/csv_bundle/`.
