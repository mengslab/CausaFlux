from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    return frame.head(max_rows).to_html(
        index=False,
        border=0,
        classes="dataframe",
        float_format=lambda x: f"{x:.3f}",
    )


def generate_causal_report(
    output_path: str | Path,
    experiment_name: str,
    validation: dict[str, Any],
    state_metrics: dict[str, Any],
    baseline_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    transition_uncertainty: pd.DataFrame,
    effects: pd.DataFrame,
    evidence: pd.DataFrame,
    biomarkers: pd.DataFrame,
    recommendations: pd.DataFrame,
    transition_plot: str | Path,
    graph_plot: str | Path,
    biomarker_plot: str | Path,
    benchmark_plot: str | Path,
    reliability_plot: str | Path,
    multimodal_validation: dict[str, Any] | None = None,
    modality_inventory: pd.DataFrame | None = None,
    modality_metrics: pd.DataFrame | None = None,
    modality_contributions: pd.DataFrame | None = None,
    modality_plot: str | Path | None = None,
    correlation_plot: str | Path | None = None,
    spatial_validation: dict[str, Any] | None = None,
    spatial_nodes: pd.DataFrame | None = None,
    spatial_circuits: pd.DataFrame | None = None,
    niche_summary: pd.DataFrame | None = None,
    contact_enrichment: pd.DataFrame | None = None,
    spatial_atlas_plot: str | Path | None = None,
    contact_plot: str | Path | None = None,
    circuit_plot: str | Path | None = None,
    heterograph_plot: str | Path | None = None,
    niche_plot: str | Path | None = None,
    therapeutic_qc: dict[str, Any] | None = None,
    therapeutic_predictions: pd.DataFrame | None = None,
    therapeutic_model_metrics: dict[str, Any] | None = None,
    therapeutic_ranking_plot: str | Path | None = None,
    therapeutic_timing_plot: str | Path | None = None,
    therapeutic_sequence_plot: str | Path | None = None,
    therapeutic_pareto_plot: str | Path | None = None,
    therapeutic_waterfall_plot: str | Path | None = None,
    biomarker_qc: dict[str, Any] | None = None,
    biomarker_timecourse: pd.DataFrame | None = None,
    biomarker_panels: pd.DataFrame | None = None,
    biomarker_assays: pd.DataFrame | None = None,
    biomarker_heatmap_plot: str | Path | None = None,
    biomarker_causal_lead_plot: str | Path | None = None,
    biomarker_panel_plot: str | Path | None = None,
    active_learning_qc: dict[str, Any] | None = None,
    round1_batch: pd.DataFrame | None = None,
    round2_recommendations: pd.DataFrame | None = None,
    posterior_history: pd.DataFrame | None = None,
    experiment_ranking_plot: str | Path | None = None,
    information_gain_plot: str | Path | None = None,
    posterior_update_plot: str | Path | None = None,
    batch_portfolio_plot: str | Path | None = None,
    sampling_time_plot: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = html.escape(
        json.dumps(
            {
                "validation": validation,
                "multimodal_validation": multimodal_validation or {},
                "spatial_validation": spatial_validation or {},
                "top_spatial_circuit": spatial_circuits.iloc[0].to_dict() if spatial_circuits is not None and not spatial_circuits.empty else {},
                "state_metrics": state_metrics,
                "top_biomarker": biomarkers.iloc[0].to_dict() if not biomarkers.empty else {},
                "biomarker_qc": biomarker_qc or {},
                "top_biomarker_panel": biomarker_panels.iloc[-1].to_dict() if biomarker_panels is not None and not biomarker_panels.empty else {},
                "top_experiment": recommendations.iloc[0].to_dict() if not recommendations.empty else {},
                "closed_loop_qc": active_learning_qc or {},
                "round1_batch": round1_batch.to_dict(orient="records") if round1_batch is not None else [],
                "therapeutic_qc": therapeutic_qc or {},
                "top_therapeutic_regimen": therapeutic_predictions.iloc[0].to_dict() if therapeutic_predictions is not None and not therapeutic_predictions.empty else {},
            },
            indent=2,
            default=str,
        )
    )
    modality_inventory = modality_inventory if modality_inventory is not None else pd.DataFrame()
    modality_metrics = modality_metrics if modality_metrics is not None else pd.DataFrame()
    modality_contributions = modality_contributions if modality_contributions is not None else pd.DataFrame()
    multimodal_validation = multimodal_validation or {}
    spatial_validation = spatial_validation or {}
    spatial_nodes = spatial_nodes if spatial_nodes is not None else pd.DataFrame()
    spatial_circuits = spatial_circuits if spatial_circuits is not None else pd.DataFrame()
    niche_summary = niche_summary if niche_summary is not None else pd.DataFrame()
    contact_enrichment = contact_enrichment if contact_enrichment is not None else pd.DataFrame()
    therapeutic_qc = therapeutic_qc or {}
    therapeutic_predictions = therapeutic_predictions if therapeutic_predictions is not None else pd.DataFrame()
    therapeutic_model_metrics = therapeutic_model_metrics or {}
    biomarker_qc = biomarker_qc or {}
    biomarker_timecourse = biomarker_timecourse if biomarker_timecourse is not None else pd.DataFrame()
    biomarker_panels = biomarker_panels if biomarker_panels is not None else pd.DataFrame()
    biomarker_assays = biomarker_assays if biomarker_assays is not None else pd.DataFrame()
    active_learning_qc = active_learning_qc or {}
    round1_batch = round1_batch if round1_batch is not None else pd.DataFrame()
    round2_recommendations = round2_recommendations if round2_recommendations is not None else pd.DataFrame()
    posterior_history = posterior_history if posterior_history is not None else pd.DataFrame()
    modality_plot_html = f'<img src="../multimodal/{Path(modality_plot).name}" alt="Modality ablation benchmark">' if modality_plot else ""
    correlation_plot_html = f'<img src="../multimodal/{Path(correlation_plot).name}" alt="Cross-modal correlation">' if correlation_plot else ""
    spatial_atlas_html = f'<img src="../spatial_graph/{Path(spatial_atlas_plot).name}" alt="Spatial atlas">' if spatial_atlas_plot else ""
    contact_plot_html = f'<img src="../spatial_graph/{Path(contact_plot).name}" alt="Spatial contact enrichment">' if contact_plot else ""
    circuit_plot_html = f'<img src="../spatial_graph/{Path(circuit_plot).name}" alt="Communication circuits">' if circuit_plot else ""
    heterograph_plot_html = f'<img src="../spatial_graph/{Path(heterograph_plot).name}" alt="Heterogeneous graph summary">' if heterograph_plot else ""
    niche_plot_html = f'<img src="../spatial_graph/{Path(niche_plot).name}" alt="Spatial niche composition">' if niche_plot else ""
    therapeutic_ranking_html = f'<img src="../therapeutics/{Path(therapeutic_ranking_plot).name}" alt="Therapeutic ranking">' if therapeutic_ranking_plot else ""
    therapeutic_timing_html = f'<img src="../therapeutics/{Path(therapeutic_timing_plot).name}" alt="Timing heatmap">' if therapeutic_timing_plot else ""
    therapeutic_sequence_html = f'<img src="../therapeutics/{Path(therapeutic_sequence_plot).name}" alt="Sequence comparison">' if therapeutic_sequence_plot else ""
    therapeutic_pareto_html = f'<img src="../therapeutics/{Path(therapeutic_pareto_plot).name}" alt="Benefit toxicity frontier">' if therapeutic_pareto_plot else ""
    therapeutic_waterfall_html = f'<img src="../therapeutics/{Path(therapeutic_waterfall_plot).name}" alt="Counterfactual effect intervals">' if therapeutic_waterfall_plot else ""
    biomarker_heatmap_html = f'<img src="../biomarkers/{Path(biomarker_heatmap_plot).name}" alt="Early warning heatmap">' if biomarker_heatmap_plot else ""
    biomarker_causal_lead_html = f'<img src="../biomarkers/{Path(biomarker_causal_lead_plot).name}" alt="Causal proximity and lead time">' if biomarker_causal_lead_plot else ""
    biomarker_panel_html = f'<img src="../biomarkers/{Path(biomarker_panel_plot).name}" alt="Biomarker panel performance">' if biomarker_panel_plot else ""
    experiment_ranking_html = f'<img src="../active_learning/{Path(experiment_ranking_plot).name}" alt="Experiment priority ranking">' if experiment_ranking_plot else ""
    information_gain_html = f'<img src="../active_learning/{Path(information_gain_plot).name}" alt="Information gain by experiment type">' if information_gain_plot else ""
    posterior_update_html = f'<img src="../active_learning/{Path(posterior_update_plot).name}" alt="Hypothesis posterior update">' if posterior_update_plot else ""
    batch_portfolio_html = f'<img src="../active_learning/{Path(batch_portfolio_plot).name}" alt="Selected experiment batch portfolio">' if batch_portfolio_plot else ""
    sampling_time_html = f'<img src="../active_learning/{Path(sampling_time_plot).name}" alt="Sampling-time recommendations">' if sampling_time_plot else ""
    key_bootstrap = bootstrap_metrics.loc[
        bootstrap_metrics["metric"].isin(["log_loss", "expected_calibration_error"])
    ] if not bootstrap_metrics.empty else bootstrap_metrics
    transition_key = transition_uncertainty.loc[
        transition_uncertainty["current_state"] != transition_uncertainty["next_state"]
    ].sort_values("bootstrap_mean", ascending=False) if not transition_uncertainty.empty else transition_uncertainty
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(experiment_name)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; line-height: 1.5; background: #f5f6f8; color: #18202a; }}
main {{ max-width: 1220px; margin: auto; padding: 30px; }}
.hero {{ background: white; border-radius: 16px; padding: 28px; box-shadow: 0 3px 16px #00000012; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 18px; margin-top: 18px; }}
.card {{ background: white; border-radius: 14px; padding: 20px; margin-top: 18px; box-shadow: 0 3px 14px #00000010; overflow-x: auto; }}
.grid .card {{ margin-top: 0; }}
img {{ max-width: 100%; height: auto; border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border-bottom: 1px solid #d8dde5; padding: 7px; text-align: left; }}
th {{ position: sticky; top: 0; background: #eef1f5; }}
.note {{ padding: 12px 15px; border-left: 4px solid #7b8794; background: #f0f2f5; }}
.good {{ padding: 12px 15px; border-left: 4px solid #37815b; background: #edf7f1; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
pre {{ white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 14px; border-radius: 10px; }}
</style>
</head>
<body><main>
<section class="hero">
<h1>{html.escape(experiment_name)}</h1>
<p><strong>CausaFlux v1.7.0 closed-loop experimentation report.</strong> The release integrates RNA, ATAC, protein, mutation, drug response, spatial tumor–immune–stromal graphs, counterfactual therapeutics, causal biomarkers, and an uncertainty-aware engine that recommends CRISPR, drug, imaging, and sampling-time experiments.</p>
<p class="note">All bundled data are synthetic and intended only to verify software behavior. Results are not biological evidence, clinical predictions, or treatment recommendations.</p>
<p class="good"><strong>Selected state model:</strong> {html.escape(str(state_metrics['selected_model']))} / {html.escape(str(state_metrics['selected_variant']))}. <strong>Split:</strong> {html.escape(str(state_metrics['split_mode']))}. Donor-held-out log loss: {state_metrics['donor_grouped_log_loss']:.3f}; ECE: {state_metrics['expected_calibration_error']:.3f}.</p>
</section>
<div class="grid">
<section class="card"><h2>Data validation</h2>{_table(pd.DataFrame([validation]))}</section>
<section class="card"><h2>MuData validation</h2>{_table(pd.DataFrame([{k: v for k, v in multimodal_validation.items() if k != "inventory"}]))}</section>
</div>
<section class="card"><h2>Multimodal data model</h2><p>All modalities share the same observation axis and are stored in <code>multimodal/causaflux_multimodal.h5mu</code>. ATAC and mutation matrices may be sparse; assay availability is represented by per-modality masks.</p>{_table(modality_inventory)}</section>
<div class="grid">
<section class="card"><h2>Modality benchmark</h2>{modality_plot_html}{_table(modality_metrics, 12)}</section>
<section class="card"><h2>Cross-modal structure</h2>{correlation_plot_html}{_table(modality_contributions, 10)}</section>
</div>
<section class="card"><h2>Spatial graph validation</h2><p>The full graph is reconstructible from typed node and edge tables and is also exported as GraphML. Spatial proximity and ligand–receptor communication are represented as distinct edge types.</p>{_table(pd.DataFrame([spatial_validation]))}</section>
<div class="grid">
<section class="card"><h2>Multicellular spatial atlas</h2>{spatial_atlas_html}<p>Representative synthetic sample; coordinates are generated only to verify software behavior.</p></section>
<section class="card"><h2>Heterogeneous graph</h2>{heterograph_plot_html}<p>Node size represents cell abundance and directed edge width represents aggregate communication strength.</p></section>
</div>
<div class="grid">
<section class="card"><h2>Spatial contact structure</h2>{contact_plot_html}{_table(contact_enrichment, 14)}</section>
<section class="card"><h2>Inferred spatial niches</h2>{niche_plot_html}{_table(niche_summary, 14)}</section>
</div>
<section class="card"><h2>Ligand–receptor communication circuits</h2>{circuit_plot_html}<p>Scores combine sender activity, receiver activity, spatial proximity, donor support, and donor-bootstrap uncertainty. They are mechanistic hypotheses, not experimentally established signaling events.</p>{_table(spatial_circuits, 16)}</section>
<section class="card"><h2>Selected fused state model</h2>{_table(pd.DataFrame([state_metrics]))}</section>
<div class="grid">
<section class="card"><h2>Linear baseline benchmark</h2><img src="../baselines/linear_baseline_benchmark.png" alt="Linear baseline benchmark">{_table(baseline_metrics, 16)}</section>
<section class="card"><h2>Probability calibration</h2><img src="../calibration/reliability_diagram.png" alt="Reliability diagram"><p>Calibration is fit and evaluated across held-out donors rather than cells randomly split from the same donor.</p></section>
</div>
<section class="card"><h2>Donor-bootstrap metric intervals</h2><p>Donors, not individual cells, are the resampling unit. This preserves within-donor dependence.</p>{_table(key_bootstrap, 24)}</section>
<div class="grid">
<section class="card"><h2>Disease transitions</h2><img src="../transitions/transition_heatmap.png" alt="Transition matrix">{_table(transition_matrix.reset_index(names="current_state"))}</section>
<section class="card"><h2>Transition uncertainty</h2><p>Percentile intervals are obtained by donor-cluster bootstrap of complete longitudinal lineages.</p>{_table(transition_key, 16)}</section>
</div>
<section class="card"><h2>Causal graph</h2><img src="../graph/causal_graph.png" alt="Causal graph"></section>
<section class="card"><h2>Causal intervention effects</h2>{_table(effects)}</section>
<section class="card"><h2>Evidence ladder</h2>{_table(evidence)}</section>
<section class="card"><h2>Counterfactual therapeutics validation</h2><p>Regimens are generated from an explicit intervention catalog. Each event changes named biological state variables before a donor-audited resistance surrogate estimates the counterfactual outcome. Normal-cell toxicity is scored separately from non-tumor pathway vulnerability.</p>{_table(pd.DataFrame([therapeutic_qc]))}</section>
<div class="grid">
<section class="card"><h2>Therapeutic ranking</h2>{therapeutic_ranking_html}{_table(therapeutic_predictions.nsmallest(16, "rank") if not therapeutic_predictions.empty else therapeutic_predictions, 16)}</section>
<section class="card"><h2>Benefit–toxicity frontier</h2>{therapeutic_pareto_html}<p>Pareto-optimal regimens are not dominated simultaneously on predicted resistance reduction and normal-cell toxicity.</p></section>
</div>
<div class="grid">
<section class="card"><h2>Intervention timing</h2>{therapeutic_timing_html}<p>Timing windows are mechanism-specific hypotheses and should be prospectively tested.</p></section>
<section class="card"><h2>Treatment sequence</h2>{therapeutic_sequence_html}<p>Order effects arise from configured directional state dependencies and are reported separately from simultaneous combinations.</p></section>
</div>
<section class="card"><h2>Counterfactual effect intervals</h2>{therapeutic_waterfall_html}<p>Intervals refit the resistance model after donor-cluster bootstrap resampling. They do not represent clinical confidence intervals.</p>{_table(pd.DataFrame([therapeutic_model_metrics]))}</section>

<section class="card"><h2>Causal biomarker validation</h2><p>Candidate scores keep prediction, timing, causal-graph proximity, perturbational support, donor stability, assayability, redundancy, and bootstrap uncertainty as separate evidence components. Causal proximity does not by itself establish causality.</p>{_table(pd.DataFrame([biomarker_qc]))}</section>
<div class="grid">
<section class="card"><h2>Early-warning and causal-proximity ranking</h2><img src="../biomarkers/biomarker_ranking.png" alt="Biomarker ranking">{_table(biomarkers, 16)}</section>
<section class="card"><h2>Temporal warning map</h2>{biomarker_heatmap_html}<p>Association is evaluated before the terminal resistance measurement; donors remain the validation unit.</p>{_table(biomarker_timecourse, 16)}</section>
</div>
<div class="grid">
<section class="card"><h2>Lead time versus causal proximity</h2>{biomarker_causal_lead_html}<p>Point size reflects assayability. The two axes are deliberately not collapsed until the final ranking step.</p></section>
<section class="card"><h2>Compact biomarker panels</h2>{biomarker_panel_html}<p>Panel performance uses leave-one-donor-out predictions and fold-specific scaling.</p>{_table(biomarker_panels, 8)}</section>
</div>
<section class="card"><h2>Assay translation manifest</h2>{_table(biomarker_assays, 16)}</section>
<section class="card"><h2>Closed-loop experiment-design validation</h2><p>Competing mechanisms retain explicit prior probabilities. Expected information gain is calculated from a transparent hypothesis-conditioned observation model. Information value, therapeutic value, biomarker value, timing value, feasibility, cost, and risk remain separate before batch selection.</p>{_table(pd.DataFrame([active_learning_qc]))}</section>
<div class="grid">
<section class="card"><h2>Round 1 experiment ranking</h2>{experiment_ranking_html}{_table(recommendations, 16)}</section>
<section class="card"><h2>Information value by experiment type</h2>{information_gain_html}<p>The catalog contains CRISPR, drug, imaging, and sampling-time candidates.</p></section>
</div>
<div class="grid">
<section class="card"><h2>Budget-constrained batch</h2>{batch_portfolio_html}{_table(round1_batch, 10)}</section>
<section class="card"><h2>Hypothesis posterior update</h2>{posterior_update_html}{_table(posterior_history, 10)}<p>Posterior changes in the bundled demonstration use synthetic outcomes solely to test the closed-loop machinery.</p></section>
</div>
<div class="grid">
<section class="card"><h2>Sampling-time design</h2>{sampling_time_html}<p>Sampling recommendations target gaps or transition windows with high expected information value.</p></section>
<section class="card"><h2>Next-round recommendations</h2>{_table(round2_recommendations, 12)}<p>The second ranking is recomputed after the demonstration posterior update and excludes completed round 1 experiments.</p></section>
</div>
<section class="card"><h2>Uncertainty interpretation</h2>
<ul>
<li><strong>Metric bootstrap intervals</strong> quantify sensitivity to which donors are represented.</li>
<li><strong>Row-level donor bootstrap intervals</strong> refit the configured L2 logistic reference model without the predicted donor.</li>
<li><strong>Ensemble mutual information and variation ratio</strong> quantify disagreement among linear model families.</li>
<li><strong>Spatial-circuit intervals</strong> resample donors after aggregating edge evidence within donor.</li>
<li><strong>Therapeutic intervals</strong> refit the counterfactual resistance surrogate after donor-cluster bootstrap resampling.</li>
<li><strong>Biomarker intervals</strong> resample complete donors, then recompute warning time, association, stability, and the composite causal-biomarker score.</li>
<li><strong>Experiment-design intervals</strong> perturb hypothesis priors and mechanism-conditioned readout predictions, then recompute expected information gain and batch-selection rank.</li>
<li>These quantities do not account for every source of biological, technical, causal, spatial, or distributional uncertainty.</li>
</ul>
</section>
<section class="card"><h2>Machine-readable highlights</h2><pre>{payload}</pre></section>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
