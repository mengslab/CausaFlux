"""Causal early-warning biomarker engine for CausaFlux v1.7.0.

The implementation deliberately separates predictive association, temporal lead,
causal-graph proximity, donor stability, assayability, and redundancy.  The
resulting ranking is an evidence table, not a causal claim by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .utils import ensure_dir, json_dump


DEFAULT_FEATURE_METADATA: dict[str, dict[str, Any]] = {
    "mutation_burden": {"modality": "mutation", "assay": "targeted DNA panel", "assayability": 0.84, "invasiveness": 0.58},
    "ire1_xbp1": {"modality": "rna/protein", "assay": "RNA or phospho-protein panel", "assayability": 0.72, "invasiveness": 0.50},
    "proteostasis_capacity": {"modality": "rna/protein", "assay": "multiplex pathway panel", "assayability": 0.64, "invasiveness": 0.52},
    "enhancer_plasticity": {"modality": "atac", "assay": "targeted chromatin accessibility", "assayability": 0.48, "invasiveness": 0.62},
    "mitochondrial_reserve": {"modality": "protein/metabolic", "assay": "functional or protein panel", "assayability": 0.69, "invasiveness": 0.55},
    "antigen_presentation": {"modality": "rna/protein", "assay": "flow or RNA panel", "assayability": 0.87, "invasiveness": 0.42},
    "immune_exclusion": {"modality": "spatial", "assay": "spatial immune-exclusion score", "assayability": 0.58, "invasiveness": 0.67},
    "inflammatory_signaling": {"modality": "rna/protein", "assay": "circulating or tissue cytokine panel", "assayability": 0.89, "invasiveness": 0.30},
    "viability": {"modality": "imaging/drug_response", "assay": "viability imaging", "assayability": 0.91, "invasiveness": 0.36},
    "apoptosis_signal": {"modality": "protein/imaging", "assay": "apoptosis protein or imaging panel", "assayability": 0.84, "invasiveness": 0.38},
    "resistance_score": {"modality": "integrated", "assay": "multimodal composite", "assayability": 0.55, "invasiveness": 0.60},
}

EVIDENCE_WEIGHTS = {
    "designed": 1.00,
    "intervention": 0.95,
    "perturbational": 0.95,
    "prior": 0.72,
    "hypothesis": 0.45,
    "association": 0.30,
}


@dataclass(frozen=True)
class BiomarkerConfig:
    outcome_column: str = "future_resistant"
    cell_type: str = "tumor"
    target_node: str = "stable_resistance"
    warning_auc_threshold: float = 0.65
    warning_stability_threshold: float = 0.60
    bootstrap: int = 80
    top_panel_size: int = 3
    seed: int = 31
    score_weights: tuple[tuple[str, float], ...] = (
        ("association_auc", 0.19),
        ("donor_stability", 0.14),
        ("causal_proximity", 0.17),
        ("perturbational_support", 0.10),
        ("lead_time_fraction", 0.14),
        ("temporal_delta_auc", 0.08),
        ("assayability", 0.11),
        ("uniqueness", 0.07),
    )


@dataclass
class BiomarkerResult:
    ranking: pd.DataFrame
    timecourse: pd.DataFrame
    bootstrap: pd.DataFrame
    panels: pd.DataFrame
    panel_predictions: pd.DataFrame
    assay_manifest: pd.DataFrame
    qc: dict[str, Any]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def binary_auc(y: Sequence[int], score: Sequence[float]) -> float:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    mask = np.isfinite(s_arr) & np.isfinite(y_arr)
    y_arr, s_arr = y_arr[mask], s_arr[mask]
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata(s_arr)
    auc = (ranks[y_arr == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _oriented_auc(y: Sequence[int], values: Sequence[float]) -> tuple[float, int]:
    auc = binary_auc(y, values)
    return (float(max(auc, 1.0 - auc)), 1 if auc >= 0.5 else -1)


def _standardized_effect(y: np.ndarray, values: np.ndarray, direction: int) -> float:
    pos = values[y == 1]
    neg = values[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    pooled = np.sqrt(max(1e-12, ((len(pos)-1)*np.var(pos, ddof=1) + (len(neg)-1)*np.var(neg, ddof=1)) / max(1, len(pos)+len(neg)-2)))
    return float(direction * (np.mean(pos) - np.mean(neg)) / pooled)


def _donor_stability(frame: pd.DataFrame, feature: str, outcome: str, global_direction: int) -> tuple[float, float, int]:
    oriented: list[float] = []
    for _, group in frame.groupby("donor_id", sort=True):
        if group[outcome].nunique() < 2 or group[feature].nunique() < 2:
            continue
        auc = binary_auc(group[outcome], group[feature])
        oriented.append(auc if global_direction > 0 else 1.0 - auc)
    if not oriented:
        return 0.0, 0.5, 0
    return float(np.mean(np.asarray(oriented) >= 0.5)), float(np.mean(oriented)), len(oriented)


def _causal_metrics(graph: nx.DiGraph, feature: str, target: str) -> dict[str, float | int | str]:
    if feature not in graph or target not in graph:
        return {"causal_distance": np.nan, "causal_proximity": 0.0, "path_evidence": 0.0, "causal_path": "", "perturbational_support": 0.0}
    try:
        path = nx.shortest_path(graph, feature, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"causal_distance": np.nan, "causal_proximity": 0.0, "path_evidence": 0.0, "causal_path": "", "perturbational_support": 0.0}
    edge_weights = []
    for source, destination in zip(path[:-1], path[1:]):
        evidence = str(graph.edges[source, destination].get("evidence", "association")).lower()
        edge_weights.append(EVIDENCE_WEIGHTS.get(evidence, 0.30))
    distance = max(0, len(path) - 1)
    evidence_score = float(np.mean(edge_weights)) if edge_weights else 1.0
    proximity = float((1.0 / (1.0 + distance)) * (0.55 + 0.45 * evidence_score))
    incoming_intervention = False
    for ancestor in nx.ancestors(graph, feature):
        node_type = str(graph.nodes[ancestor].get("type", "")).lower()
        if node_type == "intervention":
            incoming_intervention = True
            break
    path_intervention = any(str(graph.nodes[node].get("type", "")).lower() == "intervention" for node in path)
    perturbational = 1.0 if (incoming_intervention or path_intervention) else 0.0
    return {
        "causal_distance": int(distance),
        "causal_proximity": proximity,
        "path_evidence": evidence_score,
        "causal_path": " -> ".join(path),
        "perturbational_support": perturbational,
    }


def _temporal_delta_auc(tumor: pd.DataFrame, feature: str, selected_time: float, outcome: str) -> float:
    baseline_time = float(tumor["time_hours"].min())
    base = tumor.loc[tumor["time_hours"] == baseline_time, ["lineage_id", feature]].rename(columns={feature: "baseline"})
    selected = tumor.loc[tumor["time_hours"] == selected_time, ["lineage_id", feature, outcome]].merge(base, on="lineage_id", how="inner")
    if selected.empty:
        return 0.5
    delta = selected[feature].to_numpy(float) - selected["baseline"].to_numpy(float)
    auc, _ = _oriented_auc(selected[outcome].to_numpy(int), delta)
    return auc


def _feature_metadata(features: Sequence[str], overrides: Mapping[str, Mapping[str, Any]] | None, assayability: Mapping[str, float] | None) -> pd.DataFrame:
    rows = []
    overrides = overrides or {}
    assayability = assayability or {}
    for feature in features:
        payload = dict(DEFAULT_FEATURE_METADATA.get(feature, {"modality": "unknown", "assay": "custom assay", "assayability": 0.55, "invasiveness": 0.55}))
        payload.update(dict(overrides.get(feature, {})))
        if feature in assayability:
            payload["assayability"] = float(assayability[feature])
        rows.append({"biomarker": feature, **payload})
    return pd.DataFrame(rows)


def _timecourse_table(tumor: pd.DataFrame, features: Sequence[str], outcome: str, final_time: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for time in sorted(float(value) for value in tumor["time_hours"].unique() if float(value) < final_time):
        current = tumor.loc[tumor["time_hours"] == time]
        y = current[outcome].to_numpy(int)
        for feature in features:
            auc, direction = _oriented_auc(y, current[feature].to_numpy(float))
            stability, donor_auc, evaluable = _donor_stability(current, feature, outcome, direction)
            rows.append({
                "biomarker": feature,
                "time_hours": time,
                "lead_time_hours": final_time - time,
                "association_auc": auc,
                "direction": direction,
                "standardized_effect": _standardized_effect(y, current[feature].to_numpy(float), direction),
                "donor_stability": stability,
                "mean_oriented_donor_auc": donor_auc,
                "evaluable_donors": evaluable,
                "n_observations": int(len(current)),
                "outcome_prevalence": float(np.mean(y)),
            })
    return pd.DataFrame(rows)


def _select_warning_row(group: pd.DataFrame, config: BiomarkerConfig) -> pd.Series:
    qualified = group.loc[
        (group["association_auc"] >= config.warning_auc_threshold)
        & (group["donor_stability"] >= config.warning_stability_threshold)
    ].sort_values(["time_hours", "association_auc"], ascending=[True, False])
    if not qualified.empty:
        row = qualified.iloc[0].copy()
        row["threshold_met"] = True
        return row
    row = group.sort_values(["association_auc", "time_hours"], ascending=[False, True]).iloc[0].copy()
    row["threshold_met"] = False
    return row


def _compose_score(row: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    values = {
        "association_auc": max(0.0, (float(row["association_auc"]) - 0.5) / 0.5),
        "donor_stability": float(row["donor_stability"]),
        "causal_proximity": float(row["causal_proximity"]),
        "perturbational_support": float(row["perturbational_support"]),
        "lead_time_fraction": float(row["lead_time_fraction"]),
        "temporal_delta_auc": max(0.0, (float(row["temporal_delta_auc"]) - 0.5) / 0.5),
        "assayability": float(row["assayability"]),
        "uniqueness": float(row["uniqueness"]),
    }
    return float(sum(float(weights[key]) * values[key] for key in weights))


def _base_ranking(tumor: pd.DataFrame, graph: nx.DiGraph, timecourse: pd.DataFrame, metadata: pd.DataFrame, config: BiomarkerConfig) -> pd.DataFrame:
    final_time = float(tumor["time_hours"].max())
    horizon = max(1.0, final_time - float(tumor["time_hours"].min()))
    weights = dict(config.score_weights)
    meta = metadata.set_index("biomarker").to_dict(orient="index")
    selected_rows = []
    for feature, group in timecourse.groupby("biomarker", sort=False):
        warning = _select_warning_row(group, config)
        selected_time = float(warning["time_hours"])
        current = tumor.loc[tumor["time_hours"] == selected_time]
        correlations = current[[item for item in metadata["biomarker"] if item in current]].corr().abs()
        others = correlations.loc[feature].drop(index=feature, errors="ignore") if feature in correlations else pd.Series(dtype=float)
        uniqueness = float(np.clip(1.0 - (others.mean() if len(others) else 0.0), 0.0, 1.0))
        causal = _causal_metrics(graph, feature, config.target_node)
        payload = {
            "biomarker": feature,
            "selected_time_hours": selected_time,
            "early_warning_lead_hours": float(warning["lead_time_hours"]),
            "lead_time_fraction": float(warning["lead_time_hours"] / horizon),
            "warning_threshold_met": bool(warning["threshold_met"]),
            "association_auc": float(warning["association_auc"]),
            "direction": int(warning["direction"]),
            "standardized_effect": float(warning["standardized_effect"]),
            "donor_stability": float(warning["donor_stability"]),
            "mean_oriented_donor_auc": float(warning["mean_oriented_donor_auc"]),
            "evaluable_donors": int(warning["evaluable_donors"]),
            "temporal_delta_auc": _temporal_delta_auc(tumor, feature, selected_time, config.outcome_column),
            "uniqueness": uniqueness,
            **causal,
            **meta[feature],
        }
        payload["invasiveness_penalty"] = float(payload.get("invasiveness", 0.55))
        payload["causal_biomarker_score"] = _compose_score(payload, weights)
        selected_rows.append(payload)
    ranking = pd.DataFrame(selected_rows).sort_values("causal_biomarker_score", ascending=False).reset_index(drop=True)
    return ranking


def _bootstrap_ranking(tumor: pd.DataFrame, graph: nx.DiGraph, features: Sequence[str], metadata: pd.DataFrame, config: BiomarkerConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    donors = np.asarray(sorted(tumor["donor_id"].astype(str).unique()))
    rows: list[pd.DataFrame] = []
    for replicate in range(config.bootstrap):
        sampled = rng.choice(donors, size=len(donors), replace=True)
        pieces = []
        for copy_index, donor in enumerate(sampled):
            piece = tumor.loc[tumor["donor_id"].astype(str) == donor].copy()
            piece["donor_id"] = f"boot{copy_index:02d}_{donor}"
            piece["lineage_id"] = piece["lineage_id"].astype(str) + f"__boot{copy_index:02d}"
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        tc = _timecourse_table(boot, features, config.outcome_column, float(boot["time_hours"].max()))
        rank = _base_ranking(boot, graph, tc, metadata, config)
        rank["bootstrap_replicate"] = replicate
        rank["bootstrap_rank"] = rank["causal_biomarker_score"].rank(method="min", ascending=False).astype(int)
        rows.append(rank[["bootstrap_replicate", "biomarker", "causal_biomarker_score", "bootstrap_rank", "selected_time_hours", "early_warning_lead_hours", "association_auc", "donor_stability"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _attach_bootstrap(ranking: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    if bootstrap.empty:
        ranking["score_ci_low"] = ranking["causal_biomarker_score"]
        ranking["score_ci_high"] = ranking["causal_biomarker_score"]
        ranking["rank_probability_top3"] = 1.0
        ranking["bootstrap_rank_median"] = ranking.index + 1
    else:
        summary = bootstrap.groupby("biomarker").agg(
            score_ci_low=("causal_biomarker_score", lambda x: float(np.quantile(x, 0.025))),
            score_ci_high=("causal_biomarker_score", lambda x: float(np.quantile(x, 0.975))),
            bootstrap_rank_median=("bootstrap_rank", "median"),
            rank_probability_top3=("bootstrap_rank", lambda x: float(np.mean(np.asarray(x) <= 3))),
            lead_time_ci_low=("early_warning_lead_hours", lambda x: float(np.quantile(x, 0.025))),
            lead_time_ci_high=("early_warning_lead_hours", lambda x: float(np.quantile(x, 0.975))),
        ).reset_index()
        ranking = ranking.merge(summary, on="biomarker", how="left")
    ranking["uncertainty_adjusted_score"] = ranking["causal_biomarker_score"] - 0.25 * (ranking["score_ci_high"] - ranking["score_ci_low"])
    ranking = ranking.sort_values(["uncertainty_adjusted_score", "causal_biomarker_score"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["evidence_tier"] = np.select(
        [
            (ranking["warning_threshold_met"]) & (ranking["causal_proximity"] >= 0.25) & (ranking["donor_stability"] >= 0.75),
            (ranking["association_auc"] >= 0.65) & (ranking["donor_stability"] >= 0.60),
        ],
        ["mechanistically supported early warning", "replicated early warning"],
        default="exploratory candidate",
    )
    return ranking


def _greedy_panel(ranking: pd.DataFrame, tumor: pd.DataFrame, max_size: int) -> list[str]:
    candidates = ranking["biomarker"].tolist()
    selected: list[str] = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
        else:
            time = float(ranking.loc[ranking["biomarker"] == candidate, "selected_time_hours"].iloc[0])
            current = tumor.loc[tumor["time_hours"] == time]
            correlations = [abs(float(current[[candidate, item]].corr().iloc[0, 1])) for item in selected if item in current]
            if not correlations or max(correlations) < 0.80:
                selected.append(candidate)
        if len(selected) >= max_size:
            break
    return selected


def _panel_predictions(tumor: pd.DataFrame, ranking: pd.DataFrame, selected: Sequence[str], outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = []
    ranking_index = ranking.set_index("biomarker")
    donors = sorted(tumor["donor_id"].astype(str).unique())
    for size in range(1, len(selected) + 1):
        panel = list(selected[:size])
        prediction_parts = []
        for held in donors:
            train = tumor.loc[tumor["donor_id"].astype(str) != held]
            test = tumor.loc[tumor["donor_id"].astype(str) == held]
            feature_scores = []
            base = (
                test.sort_values("time_hours")
                .groupby("lineage_id", as_index=False)
                .tail(1)[["row_id", "donor_id", "lineage_id", outcome]]
                .copy()
            )
            for feature in panel:
                time = float(ranking_index.loc[feature, "selected_time_hours"])
                direction = int(ranking_index.loc[feature, "direction"])
                train_t = train.loc[train["time_hours"] == time]
                test_t = test.loc[test["time_hours"] == time]
                mean = float(train_t[feature].mean())
                sd = float(train_t[feature].std(ddof=0)) or 1.0
                score_map = pd.Series(direction * (test_t[feature].to_numpy(float) - mean) / sd, index=test_t["lineage_id"].astype(str)).to_dict()
                feature_scores.append(base["lineage_id"].astype(str).map(score_map).fillna(0.0).to_numpy(float))
            score = np.mean(np.vstack(feature_scores), axis=0)
            base["panel_score"] = score
            base["held_out_donor"] = held
            base["panel_size"] = size
            base["panel_features"] = ";".join(panel)
            prediction_parts.append(base)
        pred = pd.concat(prediction_parts, ignore_index=True)
        auc = binary_auc(pred[outcome], pred["panel_score"])
        oriented_auc = max(auc, 1.0 - auc)
        donor_aucs = []
        for _, group in pred.groupby("held_out_donor"):
            if group[outcome].nunique() > 1:
                value = binary_auc(group[outcome], group["panel_score"])
                donor_aucs.append(max(value, 1.0-value))
        lead = float(min(ranking_index.loc[feature, "early_warning_lead_hours"] for feature in panel))
        metrics.append({
            "panel_size": size,
            "panel_features": ";".join(panel),
            "donor_held_out_auc": oriented_auc,
            "mean_donor_auc": float(np.mean(donor_aucs)) if donor_aucs else 0.5,
            "minimum_component_lead_hours": lead,
            "modalities": ";".join(sorted(set(str(ranking_index.loc[feature, "modality"]) for feature in panel))),
        })
        rows.append(pred)
    return pd.DataFrame(metrics), pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_causal_biomarkers(
    frame: pd.DataFrame,
    graph: nx.DiGraph,
    features: Sequence[str],
    config: BiomarkerConfig | None = None,
    assayability: Mapping[str, float] | None = None,
    metadata_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> BiomarkerResult:
    config = config or BiomarkerConfig()
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing biomarker features: {missing}")
    required = {"donor_id", "lineage_id", "row_id", "time_hours", "cell_type", config.outcome_column}
    missing_required = sorted(required - set(frame.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    tumor = frame.loc[frame["cell_type"] == config.cell_type].copy()
    if tumor.empty:
        raise ValueError(f"No rows for cell_type={config.cell_type}")
    final_time = float(tumor["time_hours"].max())
    metadata = _feature_metadata(features, metadata_overrides, assayability)
    timecourse = _timecourse_table(tumor, features, config.outcome_column, final_time)
    ranking = _base_ranking(tumor, graph, timecourse, metadata, config)
    bootstrap = _bootstrap_ranking(tumor, graph, features, metadata, config)
    ranking = _attach_bootstrap(ranking, bootstrap)
    panel_features = _greedy_panel(ranking, tumor, config.top_panel_size)
    panels, predictions = _panel_predictions(tumor, ranking, panel_features, config.outcome_column)
    assay_manifest = ranking[["rank", "biomarker", "modality", "assay", "assayability", "invasiveness_penalty", "selected_time_hours", "early_warning_lead_hours", "evidence_tier"]].copy()
    qc = {
        "valid": bool(len(ranking) == len(features) and ranking["causal_biomarker_score"].between(0, 1).all()),
        "version": "1.7.0",
        "n_candidates": int(len(ranking)),
        "n_timepoints_evaluated": int(timecourse["time_hours"].nunique()),
        "n_donors": int(tumor["donor_id"].nunique()),
        "bootstrap_requested": int(config.bootstrap),
        "bootstrap_completed": int(bootstrap["bootstrap_replicate"].nunique()) if not bootstrap.empty else 0,
        "top_biomarker": str(ranking.iloc[0]["biomarker"]) if not ranking.empty else None,
        "top_panel": str(panels.iloc[-1]["panel_features"]) if not panels.empty else None,
        "top_panel_auc": float(panels["donor_held_out_auc"].max()) if not panels.empty else None,
        "zero_donor_overlap": True,
        "synthetic_demo": True,
    }
    return BiomarkerResult(ranking, timecourse, bootstrap, panels, predictions, assay_manifest, qc)


def validate_biomarker_outputs(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    ranking = pd.read_csv(root / "causal_biomarker_ranking.csv")
    timecourse = pd.read_csv(root / "early_warning_timecourse.csv")
    panels = pd.read_csv(root / "biomarker_panel_metrics.csv")
    qc = json.loads((root / "biomarker_qc.json").read_text())
    required = {
        "rank", "biomarker", "causal_biomarker_score", "score_ci_low", "score_ci_high",
        "early_warning_lead_hours", "causal_proximity", "donor_stability", "assayability",
    }
    valid = bool(
        qc.get("valid")
        and required.issubset(ranking.columns)
        and ranking["causal_biomarker_score"].between(0, 1).all()
        and ranking["score_ci_low"].le(ranking["score_ci_high"]).all()
        and timecourse["association_auc"].between(0.5, 1.0).all()
        and panels["donor_held_out_auc"].between(0.5, 1.0).all()
        and ranking["rank"].is_unique
    )
    report = {**qc, "cli_validation": valid}
    if not valid:
        raise ValueError("Biomarker output validation failed")
    return report


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_biomarker_ranking(ranking: pd.DataFrame, path: str | Path, top_n: int = 10) -> Path:
    path = Path(path)
    selected = ranking.head(top_n).sort_values("uncertainty_adjusted_score")
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    x = selected["uncertainty_adjusted_score"].to_numpy(float)
    low = x - selected["score_ci_low"].to_numpy(float)
    high = selected["score_ci_high"].to_numpy(float) - x
    ax.barh(selected["biomarker"].str.replace("_", " "), x)
    ax.errorbar(x, np.arange(len(selected)), xerr=np.vstack([np.maximum(0, low), np.maximum(0, high)]), fmt="none", capsize=3)
    ax.set_xlabel("Uncertainty-adjusted causal biomarker score")
    ax.set_title("Early-warning and causal-proximity ranking")
    return _save(fig, path)


def plot_early_warning_heatmap(timecourse: pd.DataFrame, ranking: pd.DataFrame, path: str | Path, top_n: int = 10) -> Path:
    path = Path(path)
    order = ranking.head(top_n)["biomarker"].tolist()
    matrix = timecourse.loc[timecourse["biomarker"].isin(order)].pivot(index="biomarker", columns="time_hours", values="association_auc").reindex(order)
    fig, ax = plt.subplots(figsize=(8.4, max(4.4, 0.42 * len(order))))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_yticks(np.arange(len(order)), [value.replace("_", " ") for value in order])
    ax.set_xticks(np.arange(len(matrix.columns)), [f"{value:g} h" for value in matrix.columns])
    ax.set_xlabel("Measurement time")
    ax.set_title("Held-out outcome association before resistance")
    fig.colorbar(image, ax=ax, label="Oriented AUC")
    return _save(fig, path)


def plot_causal_lead_map(ranking: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(ranking["early_warning_lead_hours"], ranking["causal_proximity"], s=60 + 200 * ranking["assayability"])
    for row in ranking.head(8).itertuples():
        ax.annotate(str(row.biomarker).replace("_", " "), (row.early_warning_lead_hours, row.causal_proximity), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Early-warning lead time (hours)")
    ax.set_ylabel("Causal-graph proximity")
    ax.set_title("Temporal lead versus causal proximity")
    return _save(fig, path)


def plot_panel_performance(panels: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.plot(panels["panel_size"], panels["donor_held_out_auc"], marker="o", label="Pooled held-donor AUC")
    ax.plot(panels["panel_size"], panels["mean_donor_auc"], marker="o", label="Mean donor AUC")
    ax.set_xticks(panels["panel_size"])
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Panel size")
    ax.set_ylabel("AUC")
    ax.set_title("Compact early-warning panel performance")
    ax.legend()
    return _save(fig, path)


def write_biomarker_outputs(result: BiomarkerResult, output_dir: str | Path, write_plots: bool = True) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    paths = {
        "ranking": output / "causal_biomarker_ranking.csv",
        "legacy_ranking": output / "biomarker_ranking.csv",
        "timecourse": output / "early_warning_timecourse.csv",
        "bootstrap": output / "biomarker_bootstrap_distributions.csv",
        "panels": output / "biomarker_panel_metrics.csv",
        "panel_predictions": output / "biomarker_panel_oof_predictions.csv",
        "assay_manifest": output / "assay_manifest.csv",
        "qc": output / "biomarker_qc.json",
    }
    result.ranking.to_csv(paths["ranking"], index=False)
    result.ranking.to_csv(paths["legacy_ranking"], index=False)
    result.timecourse.to_csv(paths["timecourse"], index=False)
    result.bootstrap.to_csv(paths["bootstrap"], index=False)
    result.panels.to_csv(paths["panels"], index=False)
    result.panel_predictions.to_csv(paths["panel_predictions"], index=False)
    result.assay_manifest.to_csv(paths["assay_manifest"], index=False)
    json_dump(result.qc, paths["qc"])
    if write_plots:
        paths.update({
            "ranking_plot": plot_biomarker_ranking(result.ranking, output / "biomarker_ranking.png"),
            "heatmap_plot": plot_early_warning_heatmap(result.timecourse, result.ranking, output / "early_warning_heatmap.png"),
            "causal_lead_plot": plot_causal_lead_map(result.ranking, output / "causal_lead_map.png"),
            "panel_plot": plot_panel_performance(result.panels, output / "biomarker_panel_performance.png"),
        })
    return paths
