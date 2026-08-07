from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from causaflux.biomarkers import (
    BiomarkerConfig,
    run_causal_biomarkers,
    validate_biomarker_outputs,
    write_biomarker_outputs,
)
from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo
from causaflux.causal_models import build_causal_graph


@pytest.fixture(scope="module")
def biomarker_bundle():
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "configs" / "cancer_closed_loop_v1.7.0.yaml").read_text())
    frame = generate_cancer_demo(
        CancerDemoConfig(n_donors=6, clones_per_donor=12, non_tumor_cells_per_type=2, seed=73)
    )
    graph = build_causal_graph(config["causal_graph"]["nodes"], config["causal_graph"]["edges"])
    payload = config["biomarkers"]
    result = run_causal_biomarkers(
        frame,
        graph,
        payload["features"],
        BiomarkerConfig(bootstrap=4, top_panel_size=3, seed=73),
        assayability=payload["assayability"],
        metadata_overrides=payload["metadata_overrides"],
    )
    return frame, result


def test_early_warning_excludes_terminal_time(biomarker_bundle):
    frame, result = biomarker_bundle
    terminal = frame.loc[frame["cell_type"] == "tumor", "time_hours"].max()
    assert result.timecourse["time_hours"].max() < terminal
    assert result.ranking["selected_time_hours"].max() < terminal
    assert result.ranking["early_warning_lead_hours"].gt(0).all()


def test_causal_ranking_has_separate_evidence_axes(biomarker_bundle):
    _, result = biomarker_bundle
    required = {
        "association_auc",
        "donor_stability",
        "causal_proximity",
        "perturbational_support",
        "assayability",
        "uniqueness",
        "causal_biomarker_score",
        "uncertainty_adjusted_score",
    }
    assert required.issubset(result.ranking.columns)
    assert result.ranking["causal_biomarker_score"].between(0, 1).all()
    indexed = result.ranking.set_index("biomarker")
    assert indexed.loc["immune_exclusion", "causal_proximity"] > indexed.loc["viability", "causal_proximity"]


def test_donor_bootstrap_intervals_and_rank_probabilities(biomarker_bundle):
    _, result = biomarker_bundle
    assert result.qc["bootstrap_completed"] == 4
    assert result.ranking["score_ci_low"].le(result.ranking["score_ci_high"]).all()
    assert result.ranking["rank_probability_top3"].between(0, 1).all()
    assert result.bootstrap["bootstrap_replicate"].nunique() == 4


def test_compact_panels_are_donor_held_out(biomarker_bundle):
    _, result = biomarker_bundle
    assert set(result.panels["panel_size"]) == {1, 2, 3}
    assert result.panels["donor_held_out_auc"].between(0.5, 1.0).all()
    assert result.panel_predictions["held_out_donor"].astype(str).equals(
        result.panel_predictions["donor_id"].astype(str)
    )
    assert result.panel_predictions.groupby(["panel_size", "lineage_id"]).size().max() == 1


def test_biomarker_outputs_round_trip_and_validate(biomarker_bundle, tmp_path: Path):
    _, result = biomarker_bundle
    paths = write_biomarker_outputs(result, tmp_path, write_plots=True)
    assert paths["ranking"].exists()
    assert paths["heatmap_plot"].exists()
    report = validate_biomarker_outputs(tmp_path)
    assert report["cli_validation"]
    loaded = pd.read_csv(paths["ranking"])
    assert len(loaded) == len(result.ranking)
