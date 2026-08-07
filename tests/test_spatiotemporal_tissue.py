from pathlib import Path

import numpy as np
import pandas as pd

from causaflux.spatiotemporal_tissue import (
    MODEL_ORDER,
    SPLIT_REGIMES,
    SpatiotemporalTissueConfig,
    generate_spatiotemporal_tissue_data,
    load_external_spatiotemporal_npz,
    nicheformer_adapter_spec,
    run_spatiotemporal_tissue_benchmark,
    save_external_spatiotemporal_npz,
    validate_spatiotemporal_tissue,
)


def test_spatiotemporal_fixture_has_time_varying_graph_layers():
    cfg = SpatiotemporalTissueConfig(n_donors=4, cells_per_section=12, bootstrap_replicates=3)
    data = generate_spatiotemporal_tissue_data(cfg)
    assert data.nodes.section_id.nunique() == 8
    assert data.nodes.time.nunique() == 4
    assert set(data.edges.relation)
    assert set(data.regulatory_edges.graph_type) == {"regulatory"}
    assert set(data.organelle_edges.graph_type) == {"organelle"}
    assert data.edges.time.nunique() == 4


def test_spatiotemporal_external_contract_roundtrip(tmp_path: Path):
    cfg = SpatiotemporalTissueConfig(n_donors=4, cells_per_section=10, bootstrap_replicates=2)
    data = generate_spatiotemporal_tissue_data(cfg)
    path = save_external_spatiotemporal_npz(data, tmp_path / "tissue.npz")
    restored = load_external_spatiotemporal_npz(path)
    assert len(restored.nodes) == len(data.nodes)
    assert len(restored.edges) == len(data.edges)
    assert restored.nodes.row_id.is_unique


def test_nicheformer_adapter_is_external_and_citation_pinned():
    spec = nicheformer_adapter_spec()
    assert spec["name"] == "Nicheformer"
    assert "theislab/nicheformer" in spec["upstream_project"]
    assert "10.1038/s41592-025-02814-z" in spec["citation"]
    assert "does not redistribute" in spec["execution_boundary"]


def test_spatiotemporal_production_gate_and_split_audit(tmp_path: Path):
    cfg = SpatiotemporalTissueConfig(cells_per_section=30, bootstrap_replicates=12)
    status = run_spatiotemporal_tissue_benchmark(tmp_path, cfg, require_gate=True)
    assert status["valid"]
    audit = pd.read_csv(tmp_path / "split_audit.csv")
    assert (audit.section_overlap_train_test == 0).all()
    donor = audit[audit.split_regime.eq("heldout_donor")].iloc[0]
    assert donor.donor_overlap_train_test == 0
    comparison = pd.read_csv(tmp_path / "model_comparison.csv")
    assert set(MODEL_ORDER).issubset(set(comparison.model))
    assert set(comparison.split_regime) == set(SPLIT_REGIMES)
    result = validate_spatiotemporal_tissue(tmp_path)
    assert result["valid"], result["errors"]
