#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from causaflux.multimodal import MODALITY_ORDER, read_multimodal, validate_multimodal
from causaflux.neurobiology import NEURAL_CELL_TYPES, NEURO_STATES, validate_neurobiology_outputs
from causaflux.platform import validate_research_platform
from causaflux.visualization.publication import validate_publication_bundle


def fail(message: str) -> None:
    raise SystemExit(f"VERIFICATION FAILED: {message}")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else project / "reference_demo"
    if not output.is_absolute():
        output = (project / output).resolve()
    required = [
        "run_manifest.json",
        "report/index.html",
        "multimodal/causaflux_multimodal.h5mu",
        "multimodal/validation_report.json",
        "multimodal/modality_inventory.csv",
        "multimodal/feature_manifest.csv",
        "multimodal/modality_ablation_metrics.csv",
        "multimodal/modality_contributions.csv",
        "multimodal/modality_ablation.png",
        "multimodal/cross_modal_correlation.png",
        "spatial_graph/spatial_graph_qc.json",
        "spatial_graph/graph_nodes.csv",
        "spatial_graph/spatial_edges.csv",
        "spatial_graph/communication_edges.csv",
        "spatial_graph/communication_circuit_summary.csv",
        "spatial_graph/spatial_niche_summary.csv",
        "spatial_graph/contact_enrichment.csv",
        "spatial_graph/ligand_receptor_catalog.csv",
        "spatial_graph/pyg_metadata.json",
        "spatial_graph/spatial_heterograph.graphml",
        "spatial_graph/spatial_atlas.png",
        "spatial_graph/contact_enrichment_heatmap.png",
        "spatial_graph/communication_circuits.png",
        "spatial_graph/heterograph_summary.png",
        "spatial_graph/spatial_niche_composition.png",
        "baselines/donor_split_manifest.csv",
        "baselines/linear_baseline_metrics.csv",
        "calibration/reliability_diagram.png",
        "uncertainty/metric_bootstrap_intervals.csv",
        "uncertainty/donor_bootstrap_state_probabilities.csv",
        "uncertainty/ensemble_uncertainty.csv",
        "transitions/transition_bootstrap_intervals.csv",
        "causal/causal_effects.csv",
        "therapeutics/intervention_catalog.csv",
        "therapeutics/regimen_catalog.csv",
        "therapeutics/all_regimen_predictions.csv",
        "therapeutics/gene_predictions.csv",
        "therapeutics/drug_predictions.csv",
        "therapeutics/combination_predictions.csv",
        "therapeutics/sequence_predictions.csv",
        "therapeutics/timing_predictions.csv",
        "therapeutics/top_therapeutic_recommendations.csv",
        "therapeutics/donor_bootstrap_intervals.csv",
        "therapeutics/mechanistic_state_changes.csv",
        "therapeutics/therapeutic_model_metrics.json",
        "therapeutics/therapeutic_qc.json",
        "therapeutics/therapeutic_ranking.png",
        "therapeutics/timing_heatmap.png",
        "therapeutics/sequence_comparison.png",
        "therapeutics/benefit_toxicity_pareto.png",
        "therapeutics/counterfactual_waterfall.png",
        "biomarkers/causal_biomarker_ranking.csv",
        "biomarkers/biomarker_ranking.csv",
        "biomarkers/early_warning_timecourse.csv",
        "biomarkers/biomarker_bootstrap_distributions.csv",
        "biomarkers/biomarker_panel_metrics.csv",
        "biomarkers/biomarker_panel_oof_predictions.csv",
        "biomarkers/assay_manifest.csv",
        "biomarkers/biomarker_qc.json",
        "biomarkers/biomarker_ranking.png",
        "biomarkers/early_warning_heatmap.png",
        "biomarkers/causal_lead_map.png",
        "biomarkers/biomarker_panel_performance.png",
        "active_learning/hypothesis_priors.csv",
        "active_learning/experiment_catalog.csv",
        "active_learning/round1_experiment_recommendations.csv",
        "active_learning/round1_selected_batch.csv",
        "active_learning/synthetic_round1_observations.csv",
        "active_learning/hypothesis_posterior_history.csv",
        "active_learning/round2_experiment_recommendations.csv",
        "active_learning/round2_selected_batch.csv",
        "active_learning/experiment_outcome_templates.csv",
        "active_learning/experiment_bootstrap_distributions.csv",
        "active_learning/closed_loop_qc.json",
        "active_learning/experiment_priority_ranking.png",
        "active_learning/information_gain_by_type.png",
        "active_learning/hypothesis_posterior_update.png",
        "active_learning/batch_portfolio.png",
        "active_learning/sampling_time_recommendations.png",
        "report/neurobiology.html",
        "neurobiology/neural_glial_observations.csv",
        "neurobiology/neural_glial_state_probabilities.csv",
        "neurobiology/neural_glial_trajectory_summary.csv",
        "neurobiology/neural_glial_transition_matrix.csv",
        "neurobiology/neural_glial_transition_intervals.csv",
        "neurobiology/degeneration_risk_predictions.csv",
        "neurobiology/neuro_model_metrics.csv",
        "neurobiology/imaging_ephys_alignment.csv",
        "neurobiology/cell_type_driver_scores.csv",
        "neurobiology/apoe_stratified_risk.csv",
        "neurobiology/neuro_modality_inventory.csv",
        "neurobiology/neurobiology_qc.json",
        "neurobiology/neural_glial_trajectories.png",
        "neurobiology/imaging_ephys_alignment.png",
        "neurobiology/apoe_neural_risk.png",
        "neurobiology/cell_type_drivers.png",
        "neurobiology/neural_glial_transition_matrix.png",
        "report/platform.html",
        "cards/cancer_demo_dataset_card.md",
        "cards/neurobiology_demo_dataset_card.md",
        "cards/platform_model_card.md",
        "demo_registry.csv",
        "provenance/environment.json",
        "provenance/artifact_manifest.csv",
        "provenance/provenance_summary.json",
        "provenance/platform_validation.json",
        "provenance/platform_validation.csv",
        "publication_graphics/figure_inventory.csv",
        "publication_graphics/visual_regression_baselines.csv",
        "publication_graphics/publication_graphics_qc.json",
    ]
    for relative in required:
        path = output / relative
        if not path.exists() or path.stat().st_size == 0:
            fail(f"missing or empty file: {relative}")

    manifest = json.loads((output / "run_manifest.json").read_text())
    if manifest.get("framework") != "CausaFlux":
        fail("run manifest framework is not CausaFlux")
    if manifest.get("version") != "1.7.0":
        fail("run manifest version is not 1.7.0")
    if manifest.get("modalities") != list(MODALITY_ORDER):
        fail("run manifest modality order is incomplete or incorrect")
    if manifest.get("platform_profile") != "validated_research_platform":
        fail("run manifest platform profile is missing")
    if set(manifest.get("domains", [])) != {"cancer", "neurobiology"}:
        fail("run manifest domain coverage is incomplete")
    if not manifest.get("synthetic_demonstration"):
        fail("run manifest does not disclose synthetic demonstration status")

    h5mu_path = output / "multimodal/causaflux_multimodal.h5mu"
    if h5mu_path.read_bytes()[:6] != b"MuData":
        fail("H5MU user-block marker is missing")
    with h5py.File(h5mu_path, "r") as handle:
        if handle.attrs.get("encoding-type") != "MuData":
            fail("H5MU root encoding-type is not MuData")
        if handle.attrs.get("encoding-version") != "0.1.0":
            fail("unexpected H5MU encoding version")
        required_groups = {"obs", "var", "obsm", "varm", "obsp", "varp", "obsmap", "varmap", "uns", "mod"}
        if not required_groups.issubset(handle.keys()):
            fail("H5MU standardized root hierarchy is incomplete")
        if "spatial" not in handle["obsm"]:
            fail("H5MU does not contain obsm/spatial coordinates")
        if handle["obsm"]["spatial"].shape[1] != 2:
            fail("H5MU spatial coordinates are not two-dimensional")
        for modality in MODALITY_ORDER:
            group = handle["mod"][modality]
            if group.attrs.get("encoding-type") != "anndata":
                fail(f"{modality} is not stored as an AnnData modality")
            if not {"X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns"}.issubset(group.keys()):
                fail(f"{modality} AnnData hierarchy is incomplete")

    mdata = read_multimodal(h5mu_path)
    report = validate_multimodal(mdata)
    if not report["valid"]:
        fail("MuData validation failed")

    inventory = pd.read_csv(output / "multimodal/modality_inventory.csv")
    if set(inventory["modality"]) != set(MODALITY_ORDER):
        fail("modality inventory is incomplete")
    feature_manifest = pd.read_csv(output / "multimodal/feature_manifest.csv")
    if len(feature_manifest) < 50 or feature_manifest["fused_column"].duplicated().any():
        fail("multimodal feature manifest is incomplete or duplicated")
    modality_metrics = pd.read_csv(output / "multimodal/modality_ablation_metrics.csv")
    if set(modality_metrics["feature_set"]) != set(MODALITY_ORDER) | {"fusion"}:
        fail("modality benchmark is incomplete")
    if not np.isfinite(modality_metrics["log_loss"]).all():
        fail("non-finite modality benchmark values")

    spatial_qc = json.loads((output / "spatial_graph/spatial_graph_qc.json").read_text())
    if not spatial_qc.get("valid"):
        fail("spatial graph QC did not pass")
    nodes = pd.read_csv(output / "spatial_graph/graph_nodes.csv")
    spatial_edges = pd.read_csv(output / "spatial_graph/spatial_edges.csv")
    communication = pd.read_csv(output / "spatial_graph/communication_edges.csv")
    circuits = pd.read_csv(output / "spatial_graph/communication_circuit_summary.csv")
    node_ids = set(nodes["row_id"].astype(str))
    if not set(spatial_edges["source"].astype(str)).issubset(node_ids):
        fail("spatial edges reference unknown source nodes")
    if not set(spatial_edges["target"].astype(str)).issubset(node_ids):
        fail("spatial edges reference unknown target nodes")
    if not set(communication["source"].astype(str)).issubset(node_ids):
        fail("communication edges reference unknown source nodes")
    if set(nodes["cell_type"]) != {"tumor", "macrophage", "dendritic_cell", "t_cell", "fibroblast", "vascular"}:
        fail("spatial graph node types are incomplete")
    if set(spatial_edges["edge_type"]) != {"spatial_proximity"}:
        fail("spatial edge typing is incorrect")
    if set(communication["edge_type"]) != {"ligand_receptor"}:
        fail("communication edge typing is incorrect")
    if circuits.empty or not circuits["ci_low"].le(circuits["mean_communication_score"]).all():
        fail("communication circuit lower intervals are invalid")
    if not circuits["ci_high"].ge(circuits["mean_communication_score"]).all():
        fail("communication circuit upper intervals are invalid")
    pyg = json.loads((output / "spatial_graph/pyg_metadata.json").read_text())
    if len(pyg.get("node_types", [])) != 6 or not pyg.get("edge_types"):
        fail("PyG-compatible graph metadata is incomplete")

    splits = pd.read_csv(output / "baselines/donor_split_manifest.csv").fillna("")
    if splits["donor_overlap"].astype(str).str.len().gt(0).any():
        fail("donor leakage detected")

    probabilities = pd.read_csv(output / "states/state_probabilities.csv")
    probability_columns = [name for name in probabilities if name.startswith("probability_")]
    if not np.allclose(probabilities[probability_columns].sum(axis=1), 1.0, atol=1e-6):
        fail("selected state probabilities do not sum to one")

    bootstrap = pd.read_csv(output / "uncertainty/donor_bootstrap_state_probabilities.csv")
    if bootstrap["bootstrap_successful_replicates"].min() < 1:
        fail("one or more rows have no successful bootstrap fits")

    transition = pd.read_csv(output / "transitions/transition_bootstrap_intervals.csv")
    if not transition["ci_low"].le(transition["bootstrap_mean"]).all():
        fail("transition means fall below lower intervals")
    if not transition["ci_high"].ge(transition["bootstrap_mean"]).all():
        fail("transition means exceed upper intervals")

    ensemble = pd.read_csv(output / "uncertainty/ensemble_uncertainty.csv")
    if ensemble["mutual_information"].lt(-1e-10).any():
        fail("negative ensemble mutual information")

    therapeutic_qc = json.loads((output / "therapeutics/therapeutic_qc.json").read_text())
    if not therapeutic_qc.get("valid"):
        fail("counterfactual therapeutic QC did not pass")
    therapeutic = pd.read_csv(output / "therapeutics/all_regimen_predictions.csv")
    required_categories = {"gene", "drug", "combination", "sequence", "timing"}
    if not required_categories.issubset(set(therapeutic["regimen_category"])):
        fail("therapeutic regimen categories are incomplete")
    if not therapeutic["counterfactual_resistance_probability"].between(0, 1).all():
        fail("therapeutic resistance probabilities are outside [0, 1]")
    if not therapeutic["normal_cell_toxicity"].between(0, 1).all():
        fail("therapeutic normal-cell toxicity is outside [0, 1]")
    if not therapeutic["utility_ci_low"].le(therapeutic["utility_ci_high"]).all():
        fail("therapeutic utility intervals are invalid")
    if therapeutic["bootstrap_successful_replicates"].min() < 1:
        fail("therapeutic donor bootstrap has no successful replicate")
    if therapeutic["rank"].duplicated().any() or therapeutic["rank"].min() != 1:
        fail("therapeutic rankings are invalid")
    regimen_catalog = pd.read_csv(output / "therapeutics/regimen_catalog.csv")
    if regimen_catalog["regimen_id"].duplicated().any():
        fail("therapeutic regimen IDs are duplicated")
    therapeutic_splits = pd.read_csv(output / "therapeutics/therapeutic_donor_split_manifest.csv").fillna("")
    if therapeutic_splits["donor_overlap"].astype(str).str.len().gt(0).any():
        fail("donor leakage detected in therapeutic surrogate")

    biomarker_qc = json.loads((output / "biomarkers/biomarker_qc.json").read_text())
    if not biomarker_qc.get("valid"):
        fail("causal biomarker QC did not pass")
    biomarkers = pd.read_csv(output / "biomarkers/causal_biomarker_ranking.csv")
    required_biomarker_columns = {
        "rank", "biomarker", "causal_biomarker_score", "uncertainty_adjusted_score",
        "score_ci_low", "score_ci_high", "early_warning_lead_hours",
        "association_auc", "donor_stability", "causal_proximity", "assayability",
    }
    if not required_biomarker_columns.issubset(biomarkers.columns):
        fail("causal biomarker ranking columns are incomplete")
    if not biomarkers["causal_biomarker_score"].between(0, 1).all():
        fail("causal biomarker scores are outside [0, 1]")
    if not biomarkers["score_ci_low"].le(biomarkers["score_ci_high"]).all():
        fail("causal biomarker intervals are invalid")
    if biomarkers["rank"].duplicated().any() or biomarkers["rank"].min() != 1:
        fail("causal biomarker ranks are invalid")
    timecourse = pd.read_csv(output / "biomarkers/early_warning_timecourse.csv")
    if not timecourse["association_auc"].between(0.5, 1.0).all():
        fail("early-warning AUC values are outside [0.5, 1]")
    if timecourse["time_hours"].max() >= pd.read_csv(output / "data/cancer_longitudinal.csv")["time_hours"].max():
        fail("early-warning timecourse includes the terminal outcome time")
    panels = pd.read_csv(output / "biomarkers/biomarker_panel_metrics.csv")
    if not panels["donor_held_out_auc"].between(0.5, 1.0).all():
        fail("biomarker panel AUC values are invalid")
    panel_oof = pd.read_csv(output / "biomarkers/biomarker_panel_oof_predictions.csv")
    if panel_oof["held_out_donor"].astype(str).ne(panel_oof["donor_id"].astype(str)).any():
        fail("biomarker panel held-out donor labels are inconsistent")
    bootstrap_biomarkers = pd.read_csv(output / "biomarkers/biomarker_bootstrap_distributions.csv")
    if bootstrap_biomarkers["bootstrap_replicate"].nunique() < 1:
        fail("biomarker donor bootstrap has no successful replicate")

    active_qc = json.loads((output / "active_learning/closed_loop_qc.json").read_text())
    if active_qc.get("version") != "1.7.0":
        fail("closed-loop QC version is not 1.7.0")
    catalog = pd.read_csv(output / "active_learning/experiment_catalog.csv")
    if set(catalog["experiment_type"]) != {"crispr", "drug", "imaging", "sampling_time"}:
        fail("closed-loop experiment types are incomplete")
    if catalog["experiment_id"].duplicated().any():
        fail("closed-loop experiment IDs are duplicated")
    ranking = pd.read_csv(output / "active_learning/round1_experiment_recommendations.csv")
    required_experiment_columns = {
        "rank", "experiment_id", "experiment_type", "experiment_name",
        "expected_information_gain_nats", "priority_score",
        "eig_ci_low", "eig_ci_high", "bootstrap_batch_selection_probability",
    }
    if not required_experiment_columns.issubset(ranking.columns):
        fail("closed-loop recommendation columns are incomplete")
    if ranking["rank"].duplicated().any() or ranking["rank"].min() != 1:
        fail("closed-loop ranking is invalid")
    if (ranking["expected_information_gain_nats"] < -1e-10).any():
        fail("negative expected information gain")
    if not ranking["eig_ci_low"].le(ranking["eig_ci_high"]).all():
        fail("closed-loop information-gain intervals are invalid")
    if not ranking["bootstrap_batch_selection_probability"].between(0, 1).all():
        fail("closed-loop selection probabilities are outside [0, 1]")
    batch = pd.read_csv(output / "active_learning/round1_selected_batch.csv")
    if batch["relative_cost"].sum() > float(active_qc["round1_budget"]) + 1e-8:
        fail("closed-loop round 1 batch exceeds budget")
    if len(batch) > int(active_qc["round1_batch_size"]):
        fail("closed-loop batch size is inconsistent")
    posterior = pd.read_csv(output / "active_learning/hypothesis_posterior_history.csv")
    hypotheses = pd.read_csv(output / "active_learning/hypothesis_priors.csv")
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    if not np.allclose(posterior[hypothesis_ids].sum(axis=1), 1.0, atol=1e-6):
        fail("closed-loop posterior probabilities do not sum to one")
    if posterior.iloc[-1]["entropy_nats"] > posterior.iloc[0]["entropy_nats"] + 1e-8:
        fail("demonstration posterior entropy increased")
    round2 = pd.read_csv(output / "active_learning/round2_experiment_recommendations.csv")
    if set(batch["experiment_id"]).intersection(set(round2["experiment_id"])):
        fail("completed round 1 experiments were not excluded from round 2")
    templates = pd.read_csv(output / "active_learning/experiment_outcome_templates.csv")
    if not {"experiment_id", "result_field", "required_control", "decision_rule"}.issubset(templates.columns):
        fail("experiment outcome templates are incomplete")

    neuro_qc = validate_neurobiology_outputs(output / "neurobiology")
    if neuro_qc.get("n_donors") != 8:
        fail("neurobiology donor count is incorrect")
    neuro_obs = pd.read_csv(output / "neurobiology/neural_glial_observations.csv")
    if set(neuro_obs["cell_type"]) != set(NEURAL_CELL_TYPES):
        fail("neurobiology cell types are incomplete")
    if not set(NEURO_STATES).issubset(set(neuro_obs["state"])):
        fail("neurobiology states are incomplete")
    neuro_probs = pd.read_csv(output / "neurobiology/neural_glial_state_probabilities.csv")
    neuro_probability_columns = [f"probability_{state}" for state in NEURO_STATES]
    if not np.allclose(neuro_probs[neuro_probability_columns].sum(axis=1), 1.0, atol=1e-6):
        fail("neurobiology state probabilities do not sum to one")
    if neuro_probs["donor_id"].astype(str).ne(neuro_probs["held_out_donor"].astype(str)).any():
        fail("neurobiology donor leakage detected")
    neuro_risk = pd.read_csv(output / "neurobiology/degeneration_risk_predictions.csv")
    if not neuro_risk["predicted_degeneration_probability"].between(0, 1).all():
        fail("neurobiology degeneration risk is outside [0, 1]")
    neuro_intervals = pd.read_csv(output / "neurobiology/neural_glial_transition_intervals.csv")
    if not neuro_intervals["ci_low"].le(neuro_intervals["bootstrap_mean"]).all():
        fail("neurobiology transition lower intervals are invalid")
    if not neuro_intervals["ci_high"].ge(neuro_intervals["bootstrap_mean"]).all():
        fail("neurobiology transition upper intervals are invalid")
    if manifest.get("neurobiology_observations") != int(neuro_qc["n_observations"]):
        fail("run manifest neurobiology observation count is inconsistent")

    publication_report = validate_publication_bundle(output, check_hashes=True)
    if not publication_report["valid"]:
        fail(f"publication graphics validation failed: {publication_report['errors']}")

    platform_report = validate_research_platform(output, verify_hashes=True)
    if not platform_report.valid:
        failed = [check.check_id for check in platform_report.checks if check.status == "fail"]
        fail(f"platform validation failed: {failed}")
    registry = pd.read_csv(output / "demo_registry.csv")
    if set(registry["demo_id"]) != {"foundation_pretraining", "spatiotemporal_digital_tissue", "intervention_generalization", "multimodal_dynamic_state", "dynamic_model_benchmark", "cancer_quickstart", "neurobiology_quickstart", "integrated_reference", "realdata_registry", "biological_validation"}:
        fail("packaged demo registry is incomplete")

    html = (output / "report/index.html").read_text(encoding="utf-8")
    for fragment in [
        "../spatial_graph/spatial_atlas.png",
        "../spatial_graph/contact_enrichment_heatmap.png",
        "../spatial_graph/communication_circuits.png",
        "../spatial_graph/heterograph_summary.png",
        "../spatial_graph/spatial_niche_composition.png",
        "../multimodal/modality_ablation.png",
        "../multimodal/cross_modal_correlation.png",
        "../baselines/linear_baseline_benchmark.png",
        "../calibration/reliability_diagram.png",
        "../transitions/transition_heatmap.png",
        "../graph/causal_graph.png",
        "../biomarkers/biomarker_ranking.png",
        "../biomarkers/early_warning_heatmap.png",
        "../biomarkers/causal_lead_map.png",
        "../biomarkers/biomarker_panel_performance.png",
        "../therapeutics/therapeutic_ranking.png",
        "../therapeutics/timing_heatmap.png",
        "../therapeutics/sequence_comparison.png",
        "../therapeutics/benefit_toxicity_pareto.png",
        "../therapeutics/counterfactual_waterfall.png",
        "../active_learning/experiment_priority_ranking.png",
        "../active_learning/information_gain_by_type.png",
        "../active_learning/hypothesis_posterior_update.png",
        "../active_learning/batch_portfolio.png",
        "../active_learning/sampling_time_recommendations.png",
        "../neurobiology/neural_glial_trajectories.png",
    ]:
        if fragment not in html:
            fail(f"report does not reference {fragment}")
        asset = (output / "report" / fragment).resolve()
        if not asset.exists():
            fail(f"report asset not found: {asset}")

    if "platform.html" not in html or "CAUSAFLUX_PLATFORM_V1" not in html:
        fail("integrated report does not link the v1 platform validation page")

    print(f"CausaFlux v1.7.0 verification passed: {output}")


if __name__ == "__main__":
    main()
