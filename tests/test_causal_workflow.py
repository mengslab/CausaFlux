from pathlib import Path

import yaml

from causaflux.causal_workflow import run_causal_experiment


def test_small_causal_workflow(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "configs" / "cancer_closed_loop_v1.7.0.yaml").read_text())
    config["experiment"]["seed"] = 19
    config["data"].update(
        {
            "n_donors": 4,
            "clones_per_donor": 8,
            "non_tumor_cells_per_type": 1,
        }
    )
    config["causal_estimation"]["bootstrap"] = 3
    config["baseline_uncertainty"].update(
        {
            "metric_bootstrap": 3,
            "prediction_bootstrap": 1,
            "transition_bootstrap": 3,
        }
    )
    config["spatial_graph"].update({"bootstrap": 3, "export_graphml": False})
    config["therapeutics"].update({"bootstrap": 2, "timing_grid": [0, 24, 72], "max_reference_rows_per_donor": 8})
    config["biomarkers"].update({"bootstrap": 2, "top_panel_size": 2})
    config_path = tmp_path / "causal.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "output"
    result = run_causal_experiment(config_path, output)
    assert result["report"].exists()
    assert result["mudata"].exists()
    assert result["modality_metrics"].exists()
    assert result["spatial_nodes"].exists()
    assert result["spatial_edges"].exists()
    assert result["communication_edges"].exists()
    assert result["spatial_circuits"].exists()
    assert result["baseline_metrics"].exists()
    assert result["uncertainty"].exists()
    assert result["effects"].exists()
    assert result["biomarkers"].exists()
    assert result["recommendations"].exists()
    assert result["therapeutic_predictions"].exists()
    assert result["therapeutic_recommendations"].exists()
    assert (output / "therapeutics" / "therapeutic_qc.json").exists()
    assert (output / "model_card.md").exists()
