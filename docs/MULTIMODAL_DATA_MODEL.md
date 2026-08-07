# CausaFlux Multimodal Data Model

## Design objective

CausaFlux uses one observation axis across RNA, ATAC, protein, mutation, and
drug-response measurements. This makes donor-aware splitting, longitudinal linkage,
causal adjustment, assay-availability auditing, and cross-modal fusion explicit.

## Required modalities

| Key | Rows | Columns | Typical storage |
|---|---|---|---|
| `rna` | cells/samples | genes or RNA features | dense or sparse |
| `atac` | same observation IDs | peaks or regulatory features | sparse |
| `protein` | same observation IDs | proteins/phosphoproteins | dense |
| `mutation` | same observation IDs | variants/CNV features | sparse or dense |
| `drug_response` | same observation IDs | viability, IC50, AUC, resistance scores | dense |

Every modality must have the same ordered `obs_names`. Feature names are local to a
modality and are converted to fused names such as `rna__XBP1` or
`protein__XBP1s`.

## Shared observation metadata

Required fields:

```text
donor_id
sample_id
lineage_id
time_hours
cell_type
state
therapy
future_resistant
```

The observation index is `row_id`. Recommended additional fields include genotype,
batch, tissue, spatial region, dose, treatment sequence, clinical outcome, and
perturbation target.

## Missing modalities

CausaFlux stores per-observation masks:

```text
has_rna
has_atac
has_protein
has_mutation
has_drug_response
```

A completely unavailable assay row can be encoded as missing values. During model
assessment, median imputation and missingness indicators are learned only from the
training donors in each fold.

## CausaFlux metadata

The MuData `uns` mapping contains:

```python
mdata.uns["causaflux_schema"] = {
    "framework": "CausaFlux",
    "version": "1.4.0",
    "schema_version": "1.0",
    "modalities": ["rna", "atac", "protein", "mutation", "drug_response"],
    "observation_key": "row_id",
    "synthetic": False,
}
```

Provenance should record data sources, transformations, software versions, and the
responsible analysis configuration.

## H5MU interoperability

Normal installations use the official `anndata` and `mudata` packages. The project
also includes a deliberately narrow compatibility backend for constrained test
environments. Its writer uses the MuData v0.1.0 root hierarchy, AnnData modality
groups, one-based `obsmap`/`varmap` arrays, and public AnnData element encodings. It
is not intended to replace the scverse object APIs.

## CSV bundle format

`obs.csv` contains metadata and one row per observation. Every modality file contains
`row_id` followed by numeric assay columns. Row order can differ because import aligns
by `row_id`; duplicated or missing IDs are invalid.

## Integration in v1.4.0

The default integration is transparent early fusion:

1. Prefix every feature with its modality.
2. Restrict model fitting to tumor states for the bundled state task.
3. Split by donor.
4. Fit imputation and scaling within each training fold.
5. Compare individual modalities with the fused representation.
6. Refit leave-one-modality-out models to quantify incremental information.
7. Run the complete calibrated baseline and uncertainty suite on fusion features.

Future releases can add modality-specific encoders, variational integration, graph
fusion, contrastive learning, and foundation-model adapters without changing the
external MuData contract.


## Spatial coordinates in v1.4.0

Two-dimensional tissue coordinates are stored in `mdata.obsm["spatial"]` and mirrored
as `spatial_x` and `spatial_y` in shared `mdata.obs`. This keeps the five molecular and
phenotypic modalities unchanged while allowing every modality to share one spatial axis.
The CSV-bundle representation preserves the coordinate columns in `obs.csv`; the H5MU
representation preserves the numeric matrix in `obsm/spatial`.
