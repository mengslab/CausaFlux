"""Multi-process production runner for CausaFlux v1.7.0.

Each numerical phase consumes and emits explicit files.  The separation avoids
native-library shutdown interactions and makes every phase independently
reproducible and auditable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .causal_models import (
    build_causal_graph,
    build_evidence_ladder,
    causal_effects_table,
    estimate_binary_treatment_effect,
)
from .causal_report import generate_causal_report
from .active_learning import (
    ClosedLoopConfig,
    run_closed_loop_experimentation,
    validate_closed_loop_outputs,
    write_closed_loop_outputs,
)
from .biomarkers import (
    BiomarkerConfig,
    run_causal_biomarkers,
    validate_biomarker_outputs,
    write_biomarker_outputs,
)
from .causal_workflow import _model_card, load_causal_config
from .multimodal import MODALITY_ORDER
from .neurobiology import (
    NeurobiologyConfig,
    generate_neurobiology_report,
    run_neurobiology_configuration,
    validate_neurobiology_outputs,
    write_neurobiology_outputs,
)
from .therapeutics import (
    TherapeuticConfig,
    plot_counterfactual_waterfall,
    plot_sequence_comparison,
    plot_therapeutic_pareto,
    plot_therapeutic_ranking,
    plot_timing_heatmap,
    run_counterfactual_therapeutics,
    write_therapeutic_outputs,
)
from .utils import ensure_dir, json_dump


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def run_causal_effect_stage(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_causal_config(config_path)
    seed = int(config.get("experiment", {}).get("seed", 31))
    frame = pd.read_csv(output_dir / "data" / "cancer_longitudinal.csv")
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    final_time = tumor["time_hours"].max()
    final_tumor = tumor.loc[tumor["time_hours"] == final_time].copy()
    baseline_names = [
        "mutation_burden", "enhancer_plasticity", "mitochondrial_reserve",
        "antigen_presentation", "ire1_xbp1",
    ]
    baseline = (
        tumor.sort_values("time_hours")
        .groupby("lineage_id", as_index=False)
        .first()[["lineage_id", *baseline_names]]
        .rename(columns={name: f"baseline_{name}" for name in baseline_names})
    )
    final_tumor = final_tumor.merge(baseline, on="lineage_id", how="left")
    estimation = config.get("causal_estimation", {})
    covariates = estimation.get(
        "covariates",
        [
            "donor_id", "baseline_mutation_burden", "baseline_enhancer_plasticity",
            "baseline_mitochondrial_reserve", "baseline_antigen_presentation",
            "baseline_ire1_xbp1",
        ],
    )
    comparator = str(estimation.get("comparator", "standard_therapy"))
    treatment_arms = estimation.get(
        "treatments", ["standard_plus_ire1i", "standard_plus_mitoi", "standard_plus_ifng"]
    )
    bootstrap = int(estimation.get("bootstrap", 120))
    effect_results = []
    counterfactual_frames = []
    for treatment in treatment_arms:
        result, counterfactuals = estimate_binary_treatment_effect(
            final_tumor, treatment_arm=str(treatment), comparator=comparator,
            outcome_column="future_resistant", covariates=covariates,
            n_bootstrap=bootstrap, seed=seed,
        )
        effect_results.append(result)
        counterfactual_frames.append(counterfactuals)
    causal_dir = ensure_dir(output_dir / "causal")
    effects = causal_effects_table(effect_results)
    effects.to_csv(causal_dir / "causal_effects.csv", index=False)
    pd.concat(counterfactual_frames, ignore_index=True).to_csv(
        causal_dir / "counterfactual_predictions.csv", index=False
    )
    evidence = build_evidence_ladder(effects)
    evidence.to_csv(causal_dir / "evidence_ladder.csv", index=False)
    json_dump(
        {"stage": "causal_effects_complete", "framework": "CausaFlux",
         "version": "1.7.0", "n_effects": int(len(effects))},
        output_dir / "stage_status.json",
    )
    return {"effects": causal_dir / "causal_effects.csv", "evidence": causal_dir / "evidence_ladder.csv"}

def run_therapeutic_stage(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_causal_config(config_path)
    seed = int(config.get("experiment", {}).get("seed", 31))
    frame = pd.read_csv(output_dir / "data" / "cancer_longitudinal.csv")
    comparator = str(config.get("causal_estimation", {}).get("comparator", "standard_therapy"))
    final_time = float(frame.loc[frame["cell_type"] == "tumor", "time_hours"].max())
    payload = config.get("therapeutics", {})
    therapeutic_config = TherapeuticConfig(
        comparator=str(payload.get("comparator", comparator)),
        decision_time_hours=float(payload.get("decision_time_hours", 24.0)),
        horizon_hours=float(payload.get("horizon_hours", final_time)),
        timing_grid=tuple(float(value) for value in payload.get("timing_grid", [0, 24, 48, 72, 120])),
        default_start_hour=float(payload.get("default_start_hour", 24.0)),
        sequence_delay_hours=float(payload.get("sequence_delay_hours", 24.0)),
        bootstrap=int(payload.get("bootstrap", 30)),
        max_reference_rows_per_donor=int(payload.get("max_reference_rows_per_donor", 30)),
        seed=seed,
        uncertainty_penalty=float(payload.get("uncertainty_penalty", 0.12)),
        normal_toxicity_weight=float(payload.get("normal_toxicity_weight", 0.30)),
    )
    result = run_counterfactual_therapeutics(
        frame,
        therapeutic_config,
        intervention_overrides=payload.get("interventions", []),
    )
    paths = write_therapeutic_outputs(
        result,
        output_dir / "therapeutics",
        therapeutic_config,
        write_plots=False,
    )
    json_dump(
        {
            "stage": "therapeutics_complete",
            "framework": "CausaFlux",
            "version": "1.7.0",
            "n_regimens": int(result.qc["n_regimens"]),
            "top_regimen": result.qc["top_regimen"],
        },
        output_dir / "stage_status.json",
    )
    return paths


def run_biomarker_stage(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_causal_config(config_path)
    seed = int(config.get("experiment", {}).get("seed", 31))
    frame = pd.read_csv(output_dir / "data" / "cancer_longitudinal.csv")
    graph_config = config.get("causal_graph", {})
    graph = build_causal_graph(graph_config.get("nodes", []), graph_config.get("edges", []))
    payload = config.get("biomarkers", {})
    biomarker_config = BiomarkerConfig(
        outcome_column=str(payload.get("outcome_column", "future_resistant")),
        cell_type=str(payload.get("cell_type", "tumor")),
        target_node=str(payload.get("target_node", "stable_resistance")),
        warning_auc_threshold=float(payload.get("warning_auc_threshold", 0.65)),
        warning_stability_threshold=float(payload.get("warning_stability_threshold", 0.60)),
        bootstrap=int(payload.get("bootstrap", 80)),
        top_panel_size=int(payload.get("top_panel_size", 3)),
        seed=seed,
    )
    result = run_causal_biomarkers(
        frame,
        graph,
        features=payload.get("features", []),
        config=biomarker_config,
        assayability=payload.get("assayability", {}),
        metadata_overrides=payload.get("metadata_overrides", {}),
    )
    paths = write_biomarker_outputs(result, output_dir / "biomarkers", write_plots=True)
    validate_biomarker_outputs(output_dir / "biomarkers")
    json_dump(
        {
            "stage": "biomarkers_complete",
            "framework": "CausaFlux",
            "version": "1.7.0",
            "n_candidates": int(result.qc["n_candidates"]),
            "top_biomarker": result.qc["top_biomarker"],
            "top_panel": result.qc["top_panel"],
        },
        output_dir / "stage_status.json",
    )
    return paths



def run_active_learning_stage(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_causal_config(config_path)
    seed = int(config.get("experiment", {}).get("seed", 31))
    payload = config.get("closed_loop", {})
    closed_loop_config = ClosedLoopConfig(
        budget=float(payload.get("budget", 2.4)),
        batch_size=int(payload.get("batch_size", 4)),
        round2_budget=float(payload.get("round2_budget", 2.0)),
        round2_batch_size=int(payload.get("round2_batch_size", 3)),
        max_per_type=int(payload.get("max_per_type", 2)),
        require_type_coverage=bool(payload.get("require_type_coverage", True)),
        diversity_penalty=float(payload.get("diversity_penalty", 0.10)),
        information_gain_weight=float(payload.get("information_gain_weight", 0.40)),
        therapeutic_value_weight=float(payload.get("therapeutic_value_weight", 0.22)),
        biomarker_value_weight=float(payload.get("biomarker_value_weight", 0.13)),
        temporal_value_weight=float(payload.get("temporal_value_weight", 0.10)),
        feasibility_weight=float(payload.get("feasibility_weight", 0.15)),
        bootstrap=int(payload.get("bootstrap", 60)),
        eig_samples=int(payload.get("eig_samples", 1200)),
        seed=seed,
        simulate_demonstration_round=bool(payload.get("simulate_demonstration_round", True)),
        true_hypothesis=payload.get("true_hypothesis"),
    )
    therapeutic_predictions = pd.read_csv(output_dir / "therapeutics" / "all_regimen_predictions.csv")
    biomarkers = pd.read_csv(output_dir / "biomarkers" / "causal_biomarker_ranking.csv")
    biomarker_timecourse = pd.read_csv(output_dir / "biomarkers" / "early_warning_timecourse.csv")
    transition_uncertainty = pd.read_csv(output_dir / "transitions" / "transition_bootstrap_intervals.csv")
    candidates = payload.get("candidates") or None
    hypotheses = payload.get("hypotheses") or None
    result = run_closed_loop_experimentation(
        hypotheses_payload=hypotheses,
        candidates_payload=candidates,
        config=closed_loop_config,
        therapeutic_predictions=therapeutic_predictions,
        biomarkers=biomarkers,
        biomarker_timecourse=biomarker_timecourse,
        transition_uncertainty=transition_uncertainty,
    )
    paths = write_closed_loop_outputs(result, output_dir / "active_learning", write_plots=True)
    validate_closed_loop_outputs(output_dir / "active_learning")
    json_dump(
        {
            "stage": "closed_loop_complete",
            "framework": "CausaFlux",
            "version": "1.7.0",
            "n_candidates": int(result.qc["n_candidates"]),
            "round1_batch_size": int(result.qc["round1_batch_size"]),
            "top_round1_experiment": result.qc["top_round1_experiment"],
        },
        output_dir / "stage_status.json",
    )
    return paths


def run_neurobiology_stage(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_causal_config(config_path)
    seed = int(config.get("experiment", {}).get("seed", 47))
    payload = config.get("neurobiology", {})
    neuro_config = NeurobiologyConfig(
        n_donors=int(payload.get("n_donors", 8)),
        cells_per_type=int(payload.get("cells_per_type", 16)),
        times_days=tuple(float(value) for value in payload.get("times_days", [0, 7, 21, 42])),
        apoe4_fraction=float(payload.get("apoe4_fraction", 0.5)),
        bootstrap=int(payload.get("bootstrap", 50)),
        seed=int(payload.get("seed", seed + 16)),
        warning_time_days=float(payload.get("warning_time_days", 21.0)),
        terminal_time_days=float(payload.get("terminal_time_days", 42.0)),
    )
    result = run_neurobiology_configuration(neuro_config)
    paths = write_neurobiology_outputs(result, output_dir / "neurobiology", write_plots=True)
    validate_neurobiology_outputs(output_dir / "neurobiology")
    json_dump(
        {
            "stage": "neurobiology_complete",
            "framework": "CausaFlux",
            "version": "1.7.0",
            "n_observations": int(result.qc["n_observations"]),
            "n_cell_types": int(result.qc["n_cell_types"]),
            "degeneration_risk_oof_auc": float(result.qc["degeneration_risk_oof_auc"]),
        },
        output_dir / "stage_status.json",
    )
    return paths

def finalize_report_stage(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_causal_config(config_path)
    experiment_name = str(config.get("experiment", {}).get("name", "CausaFlux v1.7.0"))

    data_dir = output_dir / "data"
    multimodal_dir = output_dir / "multimodal"
    spatial_dir = output_dir / "spatial_graph"
    states_dir = output_dir / "states"
    baselines_dir = output_dir / "baselines"
    calibration_dir = output_dir / "calibration"
    uncertainty_dir = output_dir / "uncertainty"
    transitions_dir = output_dir / "transitions"
    graph_dir = output_dir / "graph"
    causal_dir = output_dir / "causal"
    biomarker_dir = ensure_dir(output_dir / "biomarkers")
    active_dir = ensure_dir(output_dir / "active_learning")
    therapeutics_dir = output_dir / "therapeutics"
    report_dir = ensure_dir(output_dir / "report")
    neuro_dir = ensure_dir(output_dir / "neurobiology")

    frame = pd.read_csv(data_dir / "cancer_longitudinal.csv")
    validation = _load_json(data_dir / "validation_report.json")
    multimodal_validation = _load_json(multimodal_dir / "validation_report.json")
    inventory = pd.read_csv(multimodal_dir / "modality_inventory.csv")
    modality_metrics = pd.read_csv(multimodal_dir / "modality_ablation_metrics.csv")
    modality_contributions = pd.read_csv(multimodal_dir / "modality_contributions.csv")
    state_metrics = _load_json(states_dir / "state_metrics.json")
    baseline_metrics = pd.read_csv(baselines_dir / "linear_baseline_metrics.csv")
    bootstrap_metrics = pd.read_csv(uncertainty_dir / "metric_bootstrap_intervals.csv")

    transition_matrix = pd.read_csv(transitions_dir / "transition_matrix.csv", index_col=0)
    transition_uncertainty = pd.read_csv(transitions_dir / "transition_bootstrap_intervals.csv")
    effects = pd.read_csv(causal_dir / "causal_effects.csv")
    evidence = pd.read_csv(causal_dir / "evidence_ladder.csv")

    spatial_qc = _load_json(spatial_dir / "spatial_graph_qc.json")
    spatial_nodes = pd.read_csv(spatial_dir / "graph_nodes.csv")
    spatial_circuits = pd.read_csv(spatial_dir / "communication_circuit_summary.csv")
    niche_summary = pd.read_csv(spatial_dir / "spatial_niche_summary.csv")
    contact_enrichment = pd.read_csv(spatial_dir / "contact_enrichment.csv")

    therapeutic_qc = _load_json(therapeutics_dir / "therapeutic_qc.json")
    therapeutic_metrics = _load_json(therapeutics_dir / "therapeutic_model_metrics.json")
    therapeutic_predictions = pd.read_csv(therapeutics_dir / "all_regimen_predictions.csv")
    therapeutic_paths = {
        "ranking_plot": plot_therapeutic_ranking(therapeutic_predictions, therapeutics_dir / "therapeutic_ranking.png"),
        "timing_plot": plot_timing_heatmap(therapeutic_predictions, therapeutics_dir / "timing_heatmap.png"),
        "sequence_plot": plot_sequence_comparison(therapeutic_predictions, therapeutics_dir / "sequence_comparison.png"),
        "pareto_plot": plot_therapeutic_pareto(therapeutic_predictions, therapeutics_dir / "benefit_toxicity_pareto.png"),
        "waterfall_plot": plot_counterfactual_waterfall(therapeutic_predictions, therapeutics_dir / "counterfactual_waterfall.png"),
    }

    biomarkers = pd.read_csv(biomarker_dir / "causal_biomarker_ranking.csv")
    biomarker_timecourse = pd.read_csv(biomarker_dir / "early_warning_timecourse.csv")
    biomarker_panels = pd.read_csv(biomarker_dir / "biomarker_panel_metrics.csv")
    biomarker_assays = pd.read_csv(biomarker_dir / "assay_manifest.csv")
    biomarker_qc = _load_json(biomarker_dir / "biomarker_qc.json")
    biomarker_plot = biomarker_dir / "biomarker_ranking.png"
    recommendations = pd.read_csv(active_dir / "round1_experiment_recommendations.csv")
    round1_batch = pd.read_csv(active_dir / "round1_selected_batch.csv")
    round2_recommendations = pd.read_csv(active_dir / "round2_experiment_recommendations.csv")
    posterior_history = pd.read_csv(active_dir / "hypothesis_posterior_history.csv")
    active_qc = _load_json(active_dir / "closed_loop_qc.json")
    neuro_qc = _load_json(neuro_dir / "neurobiology_qc.json")

    report = generate_causal_report(
        report_dir / "index.html",
        experiment_name=experiment_name,
        validation=validation,
        multimodal_validation=multimodal_validation,
        modality_inventory=inventory,
        modality_metrics=modality_metrics,
        modality_contributions=modality_contributions,
        modality_plot=multimodal_dir / "modality_ablation.png",
        correlation_plot=multimodal_dir / "cross_modal_correlation.png",
        spatial_validation=spatial_qc,
        spatial_nodes=spatial_nodes,
        spatial_circuits=spatial_circuits,
        niche_summary=niche_summary,
        contact_enrichment=contact_enrichment,
        spatial_atlas_plot=spatial_dir / "spatial_atlas.png",
        contact_plot=spatial_dir / "contact_enrichment_heatmap.png",
        circuit_plot=spatial_dir / "communication_circuits.png",
        heterograph_plot=spatial_dir / "heterograph_summary.png",
        niche_plot=spatial_dir / "spatial_niche_composition.png",
        therapeutic_qc=therapeutic_qc,
        therapeutic_predictions=therapeutic_predictions,
        therapeutic_model_metrics=therapeutic_metrics,
        therapeutic_ranking_plot=therapeutic_paths["ranking_plot"],
        therapeutic_timing_plot=therapeutic_paths["timing_plot"],
        therapeutic_sequence_plot=therapeutic_paths["sequence_plot"],
        therapeutic_pareto_plot=therapeutic_paths["pareto_plot"],
        therapeutic_waterfall_plot=therapeutic_paths["waterfall_plot"],
        state_metrics=state_metrics,
        baseline_metrics=baseline_metrics,
        bootstrap_metrics=bootstrap_metrics,
        transition_matrix=transition_matrix,
        transition_uncertainty=transition_uncertainty,
        effects=effects,
        evidence=evidence,
        biomarkers=biomarkers,
        recommendations=recommendations,
        transition_plot=transitions_dir / "transition_heatmap.png",
        graph_plot=graph_dir / "causal_graph.png",
        biomarker_plot=biomarker_plot,
        biomarker_qc=biomarker_qc,
        biomarker_timecourse=biomarker_timecourse,
        biomarker_panels=biomarker_panels,
        biomarker_assays=biomarker_assays,
        biomarker_heatmap_plot=biomarker_dir / "early_warning_heatmap.png",
        biomarker_causal_lead_plot=biomarker_dir / "causal_lead_map.png",
        biomarker_panel_plot=biomarker_dir / "biomarker_panel_performance.png",
        active_learning_qc=active_qc,
        round1_batch=round1_batch,
        round2_recommendations=round2_recommendations,
        posterior_history=posterior_history,
        experiment_ranking_plot=active_dir / "experiment_priority_ranking.png",
        information_gain_plot=active_dir / "information_gain_by_type.png",
        posterior_update_plot=active_dir / "hypothesis_posterior_update.png",
        batch_portfolio_plot=active_dir / "batch_portfolio.png",
        sampling_time_plot=active_dir / "sampling_time_recommendations.png",
        benchmark_plot=baselines_dir / "linear_baseline_benchmark.png",
        reliability_plot=calibration_dir / "reliability_diagram.png",
    )
    neuro_report = generate_neurobiology_report(neuro_dir, report_dir / "neurobiology.html")
    report_html = Path(report).read_text(encoding="utf-8")
    neuro_section = f"""
<section class='section'><h2>Neurobiology configuration</h2>
<p>CausaFlux v1.7.0 adds donor-aware neural–glial trajectories with RNA-like pathway states, live imaging, electrophysiology, APOE context and degeneration-risk prediction.</p>
<div class='metrics'><div class='metric'><b>{neuro_qc['n_observations']:,}</b><span>neural–glial observations</span></div><div class='metric'><b>{neuro_qc['n_cell_types']}</b><span>cell types</span></div><div class='metric'><b>{neuro_qc['degeneration_risk_oof_auc']:.3f}</b><span>held-out degeneration AUC</span></div></div>
<p><a href='neurobiology.html'>Open the complete neurobiology report →</a></p>
<img src='../neurobiology/neural_glial_trajectories.png' alt='Neural glial trajectories' style='max-width:100%;'>
</section>
"""
    if "</body>" in report_html:
        report_html = report_html.replace("</body>", neuro_section + "</body>")
    else:
        report_html += neuro_section
    Path(report).write_text(report_html, encoding="utf-8")

    _model_card(
        output_dir / "model_card.md",
        config,
        validation,
        multimodal_validation,
        spatial_qc,
        state_metrics,
        therapeutic_qc,
        biomarker_qc,
    )
    with (output_dir / "model_card.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## Closed-loop experimentation\n"
            "Competing mechanisms are represented by explicit prior probabilities. CRISPR, drug, imaging, and sampling-time candidates are scored by expected information gain, therapeutic value, biomarker value, temporal value, and feasibility. Batch selection obeys budget, capacity, and diversity constraints.\n"
            f"- Candidate experiments: {active_qc['n_candidates']}\n"
            f"- Round 1 selected experiments: {active_qc['round1_batch_size']}\n"
            f"- Round 1 relative cost: {active_qc['round1_cost']:.3f} / {active_qc['round1_budget']:.3f}\n"
            f"- Demonstration posterior entropy: {active_qc['posterior_entropy_nats']:.3f} nats\n"
            "Synthetic observations are included only to verify posterior updating and must not be interpreted as completed experiments.\n"
        )
    with (output_dir / "model_card.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## Neurobiology configuration\n"
            "Neural and glial states are integrated across RNA-like pathway features, live-imaging measurements, electrophysiology, cell identity, longitudinal time and APOE context. All predictions use donor-held-out evaluation.\n"
            f"- Neural–glial observations: {neuro_qc['n_observations']}\n"
            f"- Cell types: {neuro_qc['n_cell_types']}\n"
            f"- State vocabulary: {neuro_qc['n_states']} states\n"
            f"- Donor-held-out degeneration AUC: {neuro_qc['degeneration_risk_oof_auc']:.3f}\n"
            f"- Top cross-modal driver cell type: {neuro_qc['top_driver_cell_type']}\n"
            "The bundled neurobiology cohort is synthetic and is not evidence of disease mechanism or clinical validity.\n"
        )
    manifest = pd.read_csv(multimodal_dir / "feature_manifest.csv")
    split_mode = str(config.get("baseline_uncertainty", {}).get("split_mode", "leave_one_donor_out"))
    json_dump(
        {
            "framework": "CausaFlux",
            "version": "1.7.0",
            "experiment": experiment_name,
            "config": "experiment_config.yaml",
            "output": ".",
            "data_rows": int(len(frame)),
            "mudata": "multimodal/causaflux_multimodal.h5mu",
            "modalities": list(MODALITY_ORDER),
            "multimodal_features": int(len(manifest)),
            "spatial_nodes": int(spatial_qc["n_nodes"]),
            "spatial_edges": int(spatial_qc["n_spatial_edges"]),
            "communication_edges": int(spatial_qc["n_communication_edges"]),
            "spatial_niches": int(spatial_qc["n_niches"]),
            "top_spatial_circuit": spatial_circuits.iloc[0]["circuit"] if not spatial_circuits.empty else None,
            "therapeutic_interventions": int(therapeutic_qc["n_interventions"]),
            "therapeutic_regimens": int(therapeutic_qc["n_regimens"]),
            "top_therapeutic_regimen": therapeutic_qc["top_regimen"],
            "top_therapeutic_category": therapeutic_qc["top_category"],
            "selected_state_model": state_metrics["selected_model"],
            "selected_calibration": state_metrics["selected_variant"],
            "split_mode": split_mode,
            "top_biomarker": biomarkers.iloc[0]["biomarker"] if not biomarkers.empty else None,
            "top_biomarker_score": float(biomarkers.iloc[0]["causal_biomarker_score"]) if not biomarkers.empty else None,
            "biomarker_candidates": int(biomarker_qc["n_candidates"]),
            "biomarker_bootstraps": int(biomarker_qc["bootstrap_completed"]),
            "top_biomarker_panel": biomarker_qc.get("top_panel"),
            "top_biomarker_panel_auc": biomarker_qc.get("top_panel_auc"),
            "closed_loop_candidates": int(active_qc["n_candidates"]),
            "closed_loop_round1_batch_size": int(active_qc["round1_batch_size"]),
            "closed_loop_round1_cost": float(active_qc["round1_cost"]),
            "closed_loop_posterior_entropy_nats": float(active_qc["posterior_entropy_nats"]),
            "top_experiment": recommendations.iloc[0]["experiment_name"] if not recommendations.empty else None,
            "top_next_round_experiment": round2_recommendations.iloc[0]["experiment_name"] if not round2_recommendations.empty else None,
            "neurobiology_report": "report/neurobiology.html",
            "neurobiology_observations": int(neuro_qc["n_observations"]),
            "neurobiology_cell_types": int(neuro_qc["n_cell_types"]),
            "neurobiology_states": int(neuro_qc["n_states"]),
            "neurobiology_degeneration_oof_auc": float(neuro_qc["degeneration_risk_oof_auc"]),
            "neurobiology_top_driver_cell_type": neuro_qc["top_driver_cell_type"],
        },
        output_dir / "run_manifest.json",
    )
    json_dump(
        {"stage": "complete", "framework": "CausaFlux", "version": "1.7.0"},
        output_dir / "stage_status.json",
    )
    return {
        "output_dir": output_dir,
        "report": report,
        "therapeutic_predictions": therapeutics_dir / "all_regimen_predictions.csv",
        "biomarkers": biomarker_dir / "biomarker_ranking.csv",
        "neurobiology_report": neuro_report,
    }
