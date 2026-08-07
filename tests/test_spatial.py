from pathlib import Path

import numpy as np
import pandas as pd

from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo
from causaflux.multimodal import MultimodalDemoConfig, generate_multimodal_mudata, read_multimodal, write_multimodal
from causaflux.spatial import (
    NICHE_ORDER,
    SpatialGraphConfig,
    attach_spatial_to_mudata,
    build_spatial_heterograph,
    generate_spatial_coordinates,
    validate_spatial_graph,
    write_spatial_graph_outputs,
)


def _small_frame():
    return generate_cancer_demo(
        CancerDemoConfig(n_donors=3, clones_per_donor=7, non_tumor_cells_per_type=2, seed=57)
    )


def test_spatial_coordinates_are_deterministic_and_bounded():
    frame = _small_frame()
    config = SpatialGraphConfig(seed=57)
    first = generate_spatial_coordinates(frame, config).sort_values("row_id")
    second = generate_spatial_coordinates(frame, config).sort_values("row_id")
    assert np.allclose(first[["spatial_x", "spatial_y"]], second[["spatial_x", "spatial_y"]])
    assert first["spatial_x"].between(0, config.width).all()
    assert first["spatial_y"].between(0, config.height).all()


def test_spatial_heterograph_has_typed_edges_niches_and_circuits():
    result = build_spatial_heterograph(_small_frame(), SpatialGraphConfig(seed=57, bootstrap=5))
    report = validate_spatial_graph(result.nodes, result.spatial_edges, result.communication_edges, SpatialGraphConfig())
    assert report["valid"]
    assert set(result.nodes["cell_type"]) == {
        "tumor", "macrophage", "dendritic_cell", "t_cell", "fibroblast", "vascular"
    }
    assert set(result.nodes["spatial_niche"]).issubset(set(NICHE_ORDER))
    assert set(result.spatial_edges["edge_type"]) == {"spatial_proximity"}
    assert set(result.communication_edges["edge_type"]) == {"ligand_receptor"}
    assert not result.circuits.empty
    assert result.circuits["ci_low"].le(result.circuits["mean_communication_score"]).all()
    assert result.circuits["ci_high"].ge(result.circuits["mean_communication_score"]).all()


def test_spatial_outputs_are_reconstructible(tmp_path: Path):
    result = build_spatial_heterograph(
        _small_frame(), SpatialGraphConfig(seed=57, bootstrap=3, export_graphml=True)
    )
    paths = write_spatial_graph_outputs(result, tmp_path, export_graphml=True)
    assert paths["graphml"].exists()
    nodes = pd.read_csv(paths["nodes"])
    edges = pd.read_csv(paths["spatial_edges"])
    communication = pd.read_csv(paths["communication_edges"])
    assert set(edges["source"]).issubset(set(nodes["row_id"]))
    assert set(communication["target"]).issubset(set(nodes["row_id"]))


def test_spatial_coordinates_roundtrip_in_mudata(tmp_path: Path):
    frame = generate_spatial_coordinates(_small_frame(), SpatialGraphConfig(seed=57))
    mdata = generate_multimodal_mudata(frame, MultimodalDemoConfig(seed=57))
    attach_spatial_to_mudata(mdata, frame)
    path = write_multimodal(mdata, tmp_path / "spatial.h5mu")
    restored = read_multimodal(path)
    assert "spatial" in restored.obsm
    assert restored.obsm["spatial"].shape == (len(frame), 2)
    assert np.isfinite(restored.obsm["spatial"]).all()
