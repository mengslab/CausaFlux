# CausaFlux v1.4.0 Multicellular Spatial Graph

## Purpose

CausaFlux v1.4.0 represents a diseased tissue as a typed, multiplex graph. Each measured
cell or aggregate biological unit is a node. Spatial proximity and directed molecular
communication are distinct edge families so that physical colocalization is never
mistaken for signaling.

The bundled example is synthetic software-validation data. Its coordinates, niches,
ligand–receptor activities, and circuits are not biological findings.

## Node model

The core node types are:

- `tumor`
- `macrophage`
- `dendritic_cell`
- `t_cell`
- `fibroblast`
- `vascular`

Every node retains its `row_id`, donor, sample, treatment, time, state, multimodal
features, spatial coordinates, inferred niche, and neighborhood-composition features.
The broader compartments are tumor, immune, stromal, and vascular.

Spatial coordinates are stored in both:

```text
MuData.obsm["spatial"]
MuData.obs[["spatial_x", "spatial_y"]]
```

## Edge model

### Spatial proximity

A sample-specific k-nearest-neighbor graph is constructed from the two-dimensional
coordinates. Edges beyond `max_distance` are discarded. The exported undirected contact
is represented as two directed edges in GraphML for compatibility with directed graph
libraries.

Key fields are:

```text
source, target, sample_id, donor_id, therapy, time_hours,
source_cell_type, target_cell_type, distance, spatial_weight,
edge_type = spatial_proximity
```

### Ligand–receptor communication

Communication edges are directed from a sender cell to a receiver cell. A rule must:

1. match the sender and receiver cell types;
2. occur within the configured communication radius;
3. have measurable sender and receiver activities;
4. retain its ligand, receptor, family, proposed effect, distance, and score.

The demonstration score combines sender activity, receiver activity, spatial decay,
and a small niche-context adjustment. It is a prioritization score, not a causal effect
or biochemical binding probability.

The included synthetic catalog covers examples of:

- tumor checkpoint signaling to T cells;
- tumor recruitment of macrophages;
- macrophage immune suppression and stromal remodeling;
- fibroblast immune exclusion and tumor support;
- dendritic antigen presentation and T-cell recruitment;
- T-cell IFNG feedback to tumor cells;
- tumor–vascular angiogenic support.

The catalog is exported as `ligand_receptor_catalog.csv` and can be replaced or expanded.

## Spatial niches

CausaFlux calculates local cell-type fractions around each node and assigns one of five
interpretable niche labels:

- `tumor_core`
- `immune_infiltrated`
- `macrophage_barrier`
- `stromal_perivascular`
- `mixed_interface`

These deterministic labels provide an auditable baseline. Future releases can add
learned niche models while retaining this rule-based comparator.

## Communication circuits

Cell-level ligand–receptor edges are aggregated by donor and circuit. A circuit is
identified by sender cell type, ligand, receiver cell type, receptor, family, and
proposed effect. The summary reports:

- mean communication score;
- total score and supporting cell-level edges;
- donor support and support fraction;
- mean distance;
- donor-bootstrap 95% interval;
- composite circuit score used for ranking.

Donors are the resampling unit. Cells from the same donor are not treated as independent
replicates.

## Contact enrichment

For each unordered cell-type pair, CausaFlux compares its observed share of spatial
edges with the expected share under random mixing based on global node frequencies.
Values above one indicate overrepresentation relative to that simple null model.
This is a descriptive spatial statistic, not evidence for attraction or causation.

## Heterogeneous-graph exports

The complete graph is available as:

```text
spatial_graph/graph_nodes.csv
spatial_graph/spatial_edges.csv
spatial_graph/communication_edges.csv
spatial_graph/spatial_heterograph.graphml
spatial_graph/pyg_metadata.json
```

The CSV tables are the authoritative, loss-minimizing exchange format. The metadata file
lists typed node and edge relations that can be used to construct a PyTorch Geometric
`HeteroData` object without requiring `torch-geometric` in the core installation.

## Real spatial data

For real projects, provide `spatial_x` and `spatial_y` in the causal observation table
and set:

```yaml
spatial_graph:
  coordinate_mode: existing
```

The causal table and MuData object must contain identical `row_id` values. Coordinates
must be in a consistent unit within each sample. Spatial graphs are always built within
`sample_id`; CausaFlux never connects cells from different tissue sections.

## Limitations

- A ligand and receptor can be coexpressed without functional signaling.
- Spatial proximity does not establish direct interaction.
- The bundled rule catalog is deliberately small and illustrative.
- Tissue segmentation, deconvolution, registration, and image-feature extraction are
  outside the v1.4.0 core.
- Interference between neighboring units complicates conventional causal estimands.
- Graph neural networks are not used in this release; transparent graph construction
  and baseline circuit inference come first.
