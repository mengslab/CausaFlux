from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CausaFlux v1.5.0", layout="wide")
st.title("CausaFlux v1.5.0")
st.caption("Multimodal causal disease evolution, neural–glial trajectories, counterfactual therapeutics, causal biomarkers, and closed-loop experiment design")

root = Path(os.environ.get("CAUSAFLUX_OUTPUT", "causaflux_v1.5.0_output"))
root = Path(st.sidebar.text_input("Output directory", str(root))).expanduser()

if not root.exists():
    st.error(f"Output directory not found: {root}")
    st.code("sh run.sh")
    st.stop()

manifest_path = root / "run_manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    st.sidebar.success(f"{manifest.get('framework', 'CausaFlux')} {manifest.get('version', '')}")
    st.sidebar.caption("Modalities: " + ", ".join(manifest.get("modalities", [])))


dynamic_status_path = root / "dynamic_benchmark_status.json"
if dynamic_status_path.exists():
    status = json.loads(dynamic_status_path.read_text())
    gate = status.get("gate", {})
    st.header("Dynamic model benchmark")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", len(status.get("models_evaluated", [])))
    c2.metric("Histories", status.get("n_histories", 0))
    c3.metric("Donors", status.get("n_donors", 0))
    c4.metric("Performance gate", gate.get("status", "unknown"))
    st.write(f"Winning dynamic model: **{gate.get('winning_dynamic_model', 'none')}**")
    if gate.get("foundation_pretraining_allowed"):
        st.success("Foundation-pretraining gate is eligible after human review.")
    else:
        st.warning(gate.get("foundation_pretraining_status", "Foundation pretraining remains blocked."))
    comparison = root / "model_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), use_container_width=True, hide_index=True)
    for name, title in [
        ("trajectory_forecast_benchmark.png", "Future-trajectory forecasting"),
        ("fate_prediction_benchmark.png", "Fate prediction"),
        ("uncertainty_coverage.png", "Uncertainty coverage"),
        ("history_split_design.png", "Perturbation-history split"),
    ]:
        path = root / "figures" / name
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
    st.info("The packaged dynamic reference is synthetic and validates software behavior only. Real longitudinal perturbation data are required before foundation pretraining.")
    st.stop()

platform_validation = root / "provenance" / "platform_validation.csv"
artifact_manifest = root / "provenance" / "artifact_manifest.csv"
demo_registry = root / "demo_registry.csv"
if platform_validation.exists():
    st.header("Research-platform validation")
    validation_table = pd.read_csv(platform_validation)
    passed = int((validation_table["status"] == "pass").sum())
    failed = int((validation_table["status"] == "fail").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Passed gates", passed)
    c2.metric("Failed gates", failed)
    if artifact_manifest.exists():
        c3.metric("Hashed artifacts", len(pd.read_csv(artifact_manifest)))
    st.dataframe(validation_table, use_container_width=True, hide_index=True)
if demo_registry.exists():
    st.header("Packaged demonstrations")
    st.dataframe(pd.read_csv(demo_registry), use_container_width=True, hide_index=True)

sections = {
    "Neural–glial trajectories": (
        root / "neurobiology" / "neural_glial_trajectories.png",
        root / "neurobiology" / "neural_glial_trajectory_summary.csv",
    ),
    "APOE-stratified neural risk": (
        root / "neurobiology" / "apoe_neural_risk.png",
        root / "neurobiology" / "apoe_stratified_risk.csv",
    ),
    "Imaging–electrophysiology integration": (
        root / "neurobiology" / "imaging_ephys_alignment.png",
        root / "neurobiology" / "imaging_ephys_alignment.csv",
    ),
    "Neural–glial state transitions": (
        root / "neurobiology" / "neural_glial_transition_matrix.png",
        root / "neurobiology" / "neural_glial_transition_intervals.csv",
    ),
    "Neurobiology cell-type drivers": (
        root / "neurobiology" / "cell_type_drivers.png",
        root / "neurobiology" / "cell_type_driver_scores.csv",
    ),
    "Spatial atlas": (
        root / "spatial_graph" / "spatial_atlas.png",
        root / "spatial_graph" / "graph_nodes.csv",
    ),
    "Heterogeneous graph summary": (
        root / "spatial_graph" / "heterograph_summary.png",
        root / "spatial_graph" / "communication_circuit_summary.csv",
    ),
    "Communication circuits": (
        root / "spatial_graph" / "communication_circuits.png",
        root / "spatial_graph" / "communication_circuit_summary.csv",
    ),
    "Spatial niches": (
        root / "spatial_graph" / "spatial_niche_composition.png",
        root / "spatial_graph" / "spatial_niche_summary.csv",
    ),
    "Contact enrichment": (
        root / "spatial_graph" / "contact_enrichment_heatmap.png",
        root / "spatial_graph" / "contact_enrichment.csv",
    ),
    "Multimodal inventory": (None, root / "multimodal" / "modality_inventory.csv"),
    "Modality benchmark": (
        root / "multimodal" / "modality_ablation.png",
        root / "multimodal" / "modality_ablation_metrics.csv",
    ),
    "Modality contributions": (None, root / "multimodal" / "modality_contributions.csv"),
    "Cross-modal structure": (
        root / "multimodal" / "cross_modal_correlation.png",
        root / "multimodal" / "cross_modal_summary_correlations.csv",
    ),
    "Linear baseline benchmark": (
        root / "baselines" / "linear_baseline_benchmark.png",
        root / "baselines" / "linear_baseline_metrics.csv",
    ),
    "Donor split audit": (None, root / "baselines" / "donor_split_manifest.csv"),
    "Probability calibration": (
        root / "calibration" / "reliability_diagram.png",
        root / "calibration" / "calibration_comparison.csv",
    ),
    "Bootstrap metric intervals": (None, root / "uncertainty" / "metric_bootstrap_intervals.csv"),
    "Ensemble uncertainty": (None, root / "uncertainty" / "ensemble_uncertainty.csv"),
    "Disease transitions": (
        root / "transitions" / "transition_heatmap.png",
        root / "transitions" / "transition_matrix.csv",
    ),
    "Transition uncertainty": (None, root / "transitions" / "transition_bootstrap_intervals.csv"),
    "Causal graph": (
        root / "graph" / "causal_graph.png",
        root / "graph" / "causal_edges.csv",
    ),
    "Therapeutic ranking": (
        root / "therapeutics" / "therapeutic_ranking.png",
        root / "therapeutics" / "top_therapeutic_recommendations.csv",
    ),
    "Benefit–toxicity frontier": (
        root / "therapeutics" / "benefit_toxicity_pareto.png",
        root / "therapeutics" / "all_regimen_predictions.csv",
    ),
    "Timing predictions": (
        root / "therapeutics" / "timing_heatmap.png",
        root / "therapeutics" / "timing_predictions.csv",
    ),
    "Sequence predictions": (
        root / "therapeutics" / "sequence_comparison.png",
        root / "therapeutics" / "sequence_predictions.csv",
    ),
    "Counterfactual intervals": (
        root / "therapeutics" / "counterfactual_waterfall.png",
        root / "therapeutics" / "donor_bootstrap_intervals.csv",
    ),
    "Intervention catalog": (None, root / "therapeutics" / "intervention_catalog.csv"),
    "Intervention effects": (None, root / "causal" / "causal_effects.csv"),
    "Evidence ladder": (None, root / "causal" / "evidence_ladder.csv"),
    "Causal biomarker ranking": (
        root / "biomarkers" / "biomarker_ranking.png",
        root / "biomarkers" / "causal_biomarker_ranking.csv",
    ),
    "Early-warning timecourse": (
        root / "biomarkers" / "early_warning_heatmap.png",
        root / "biomarkers" / "early_warning_timecourse.csv",
    ),
    "Lead time and causal proximity": (
        root / "biomarkers" / "causal_lead_map.png",
        root / "biomarkers" / "assay_manifest.csv",
    ),
    "Compact biomarker panels": (
        root / "biomarkers" / "biomarker_panel_performance.png",
        root / "biomarkers" / "biomarker_panel_metrics.csv",
    ),
    "Round 1 experiment ranking": (
        root / "active_learning" / "experiment_priority_ranking.png",
        root / "active_learning" / "round1_experiment_recommendations.csv",
    ),
    "Information gain by experiment type": (
        root / "active_learning" / "information_gain_by_type.png",
        root / "active_learning" / "experiment_catalog.csv",
    ),
    "Selected experiment batch": (
        root / "active_learning" / "batch_portfolio.png",
        root / "active_learning" / "round1_selected_batch.csv",
    ),
    "Hypothesis posterior update": (
        root / "active_learning" / "hypothesis_posterior_update.png",
        root / "active_learning" / "hypothesis_posterior_history.csv",
    ),
    "Sampling-time recommendations": (
        root / "active_learning" / "sampling_time_recommendations.png",
        root / "active_learning" / "round2_experiment_recommendations.csv",
    ),
    "Experiment outcome templates": (None, root / "active_learning" / "experiment_outcome_templates.csv"),
}

for title, (image_path, table_path) in sections.items():
    st.header(title)
    if image_path is not None and image_path.exists():
        st.image(str(image_path), use_container_width=True)
    if table_path.exists():
        table = pd.read_csv(table_path)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Missing output: {table_path}")

st.info(
    "Bundled outputs are synthetic software demonstrations. Spatial proximity, ligand–receptor "
    "scores, inferred niches, multimodal integration, counterfactual regimens, sequence and timing rankings, biomarker rankings, experiment recommendations, posterior updates, neural–glial trajectory estimates, imaging–electrophysiology relationships, calibration, and uncertainty diagnostics "
    "do not establish biological causality or clinical validity."
)
