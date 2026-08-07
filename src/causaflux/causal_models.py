from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .causal_data import BIOMARKER_FEATURES, STATE_ORDER


@dataclass(frozen=True)
class AIPWResult:
    treatment: str
    comparator: str
    outcome: str
    n: int
    treated_n: int
    comparator_n: int
    ate: float
    ci_low: float
    ci_high: float
    bootstrap_sd: float
    sign_consistency: float
    donor_effect_sd: float
    positivity_min: float
    positivity_max: float
    evidence_level: str


def _build_preprocessor(frame: pd.DataFrame, covariates: Sequence[str]) -> ColumnTransformer:
    categorical = [name for name in covariates if frame[name].dtype == object or str(frame[name].dtype).startswith("category")]
    numeric = [name for name in covariates if name not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_state_classifier(
    frame: pd.DataFrame,
    features: Sequence[str] | None = None,
    group_column: str = "donor_id",
    seed: int = 31,
) -> tuple[pd.DataFrame, dict[str, float], Pipeline]:
    features = list(features or BIOMARKER_FEATURES)
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    tumor = tumor.loc[tumor["state"].isin(STATE_ORDER)].reset_index(drop=True)
    X = tumor[features]
    y = tumor["state"].astype(str)
    groups = tumor[group_column].astype(str)
    n_splits = min(4, groups.nunique())
    if n_splits < 2:
        raise ValueError("At least two donor groups are required for donor-aware evaluation")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    splitter = GroupKFold(n_splits=n_splits)
    probabilities = cross_val_predict(model, X, y, groups=groups, cv=splitter, method="predict_proba")
    classes = np.unique(y)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = {
        "donor_grouped_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "donor_grouped_log_loss": float(log_loss(y, probabilities, labels=classes)),
        "n_tumor_rows": int(len(tumor)),
        "n_donors": int(groups.nunique()),
    }
    model.fit(X, y)
    output = tumor[["row_id", "donor_id", "lineage_id", "time_hours", "state"]].copy()
    output["predicted_state"] = predictions
    for index, state in enumerate(classes):
        output[f"probability_{state}"] = probabilities[:, index]
    return output, metrics, model


def estimate_transition_matrix(
    frame: pd.DataFrame,
    states: Sequence[str] = STATE_ORDER,
    alpha: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    counts = pd.DataFrame(alpha, index=states, columns=states, dtype=float)
    prediction_rows: list[dict[str, object]] = []
    for lineage_id, group in tumor.groupby("lineage_id"):
        group = group.sort_values("time_hours")
        values = group["state"].astype(str).tolist()
        times = group["time_hours"].astype(float).tolist()
        for index in range(len(values) - 1):
            current, next_state = values[index], values[index + 1]
            if current not in states or next_state not in states:
                continue
            counts.loc[current, next_state] += 1.0
            prediction_rows.append(
                {
                    "lineage_id": lineage_id,
                    "time_hours": times[index],
                    "next_time_hours": times[index + 1],
                    "current_state": current,
                    "observed_next_state": next_state,
                }
            )
    probabilities = counts.div(counts.sum(axis=1), axis=0)
    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        for state in states:
            predictions[f"probability_{state}"] = predictions["current_state"].map(probabilities[state])
        predictions["predicted_next_state"] = predictions["current_state"].map(probabilities.idxmax(axis=1))
        predictions["correct"] = predictions["predicted_next_state"] == predictions["observed_next_state"]
    return probabilities, predictions


def plot_transition_matrix(matrix: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=max(0.5, float(matrix.to_numpy().max())))
    ax.set_xticks(range(len(matrix.columns)), [name.replace("_", "\n") for name in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [name.replace("_", "\n") for name in matrix.index])
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("Tumor-state transition probabilities")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_causal_graph(nodes: Iterable[dict[str, object]], edges: Iterable[dict[str, object]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in nodes:
        payload = dict(node)
        name = str(payload.pop("name"))
        graph.add_node(name, **payload)
    for edge in edges:
        payload = dict(edge)
        source = str(payload.pop("source"))
        target = str(payload.pop("target"))
        graph.add_edge(source, target, **payload)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Configured causal graph must be a directed acyclic graph")
    return graph


def graph_tables(graph: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame([{"name": name, **attributes} for name, attributes in graph.nodes(data=True)])
    edges = pd.DataFrame(
        [{"source": source, "target": target, **attributes} for source, target, attributes in graph.edges(data=True)]
    )
    return nodes, edges


def plot_causal_graph(
    graph: nx.DiGraph,
    output_path: str | Path,
    title: str = "CausaFlux editable causal graph",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        generations = list(nx.topological_generations(graph))
        position = {}
        for x_index, generation in enumerate(generations):
            for y_index, node in enumerate(generation):
                position[node] = (x_index, -y_index)
    except Exception:
        position = nx.spring_layout(graph, seed=7)
    labels = {node: node.replace("_", "\n") for node in graph.nodes}
    nx.draw_networkx_nodes(graph, position, node_size=1700, ax=ax)
    nx.draw_networkx_edges(graph, position, arrows=True, arrowsize=18, width=1.4, ax=ax)
    nx.draw_networkx_labels(graph, position, labels=labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _fit_outcome_models(
    data: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    covariates: Sequence[str],
    seed: int,
):
    preprocessor = _build_preprocessor(data, covariates)
    propensity = Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    outcome_covariates = list(covariates) + [treatment_column]
    outcome_preprocessor = _build_preprocessor(data, outcome_covariates)
    outcome = Pipeline(
        [
            ("prep", outcome_preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    propensity.fit(data[list(covariates)], data[treatment_column].astype(int))
    outcome.fit(data[outcome_covariates], data[outcome_column].astype(int))
    return propensity, outcome


def _aipw_once(
    data: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    covariates: Sequence[str],
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    propensity_model, outcome_model = _fit_outcome_models(
        data, treatment_column, outcome_column, covariates, seed
    )
    treatment = data[treatment_column].to_numpy(dtype=float)
    outcome = data[outcome_column].to_numpy(dtype=float)
    propensity = np.clip(propensity_model.predict_proba(data[list(covariates)])[:, 1], 0.03, 0.97)
    treated = data[list(covariates) + [treatment_column]].copy()
    untreated = treated.copy()
    treated[treatment_column] = 1
    untreated[treatment_column] = 0
    mu1 = outcome_model.predict_proba(treated)[:, 1]
    mu0 = outcome_model.predict_proba(untreated)[:, 1]
    pseudo = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / propensity
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - propensity)
    )
    return float(np.mean(pseudo)), propensity, mu1, mu0


def estimate_binary_treatment_effect(
    final_tumor: pd.DataFrame,
    treatment_arm: str,
    comparator: str,
    outcome_column: str,
    covariates: Sequence[str],
    n_bootstrap: int = 120,
    seed: int = 31,
) -> tuple[AIPWResult, pd.DataFrame]:
    data = final_tumor.loc[final_tumor["therapy"].isin([treatment_arm, comparator])].copy()
    treatment_column = "_treatment"
    data[treatment_column] = (data["therapy"] == treatment_arm).astype(int)
    if data[treatment_column].nunique() != 2:
        raise ValueError(f"Both {treatment_arm} and {comparator} are required")
    ate, propensity, mu1, mu0 = _aipw_once(data, treatment_column, outcome_column, covariates, seed)

    rng = np.random.default_rng(seed)
    donors = data["donor_id"].astype(str).unique()
    bootstrap_values: list[float] = []
    for bootstrap_index in range(n_bootstrap):
        sampled_donors = rng.choice(donors, size=len(donors), replace=True)
        parts = []
        for copy_index, donor in enumerate(sampled_donors):
            part = data.loc[data["donor_id"].astype(str) == donor].copy()
            part["_bootstrap_donor"] = f"{donor}_{copy_index}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        try:
            value, _, _, _ = _aipw_once(
                sample, treatment_column, outcome_column, covariates, seed + bootstrap_index + 1
            )
            bootstrap_values.append(value)
        except Exception:
            continue
    if bootstrap_values:
        ci_low, ci_high = np.quantile(bootstrap_values, [0.025, 0.975])
        bootstrap_sd = float(np.std(bootstrap_values, ddof=1)) if len(bootstrap_values) > 1 else 0.0
    else:
        ci_low = ci_high = ate
        bootstrap_sd = 0.0

    donor_effects = []
    for _, group in data.groupby("donor_id"):
        treated_outcome = group.loc[group[treatment_column] == 1, outcome_column]
        comparator_outcome = group.loc[group[treatment_column] == 0, outcome_column]
        if len(treated_outcome) and len(comparator_outcome):
            donor_effects.append(float(treated_outcome.mean() - comparator_outcome.mean()))
    sign_consistency = (
        float(np.mean(np.sign(donor_effects) == np.sign(ate))) if donor_effects and ate != 0 else 0.0
    )
    donor_effect_sd = float(np.std(donor_effects, ddof=1)) if len(donor_effects) > 1 else 0.0
    evidence_level = "perturbational_support"
    if sign_consistency >= 0.75 and not (ci_low <= 0 <= ci_high):
        evidence_level = "causal_convergence"
    elif ci_low <= 0 <= ci_high:
        evidence_level = "association_only"

    result = AIPWResult(
        treatment=treatment_arm,
        comparator=comparator,
        outcome=outcome_column,
        n=int(len(data)),
        treated_n=int(data[treatment_column].sum()),
        comparator_n=int((1 - data[treatment_column]).sum()),
        ate=ate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        bootstrap_sd=bootstrap_sd,
        sign_consistency=sign_consistency,
        donor_effect_sd=donor_effect_sd,
        positivity_min=float(propensity.min()),
        positivity_max=float(propensity.max()),
        evidence_level=evidence_level,
    )
    counterfactuals = data[["row_id", "donor_id", "lineage_id", "therapy", outcome_column]].copy()
    counterfactuals["predicted_outcome_if_treated"] = mu1
    counterfactuals["predicted_outcome_if_comparator"] = mu0
    counterfactuals["individual_effect"] = mu1 - mu0
    counterfactuals["comparison"] = f"{treatment_arm}_vs_{comparator}"
    return result, counterfactuals


def causal_effects_table(results: Sequence[AIPWResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])


def build_evidence_ladder(effect_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mechanism_map = {
        "standard_plus_ire1i": "IRE1-XBP1 proteostasis",
        "standard_plus_mitoi": "Mitochondrial reserve",
        "standard_plus_ifng": "Antigen presentation",
    }
    for row in effect_table.to_dict(orient="records"):
        rows.append(
            {
                "mechanism": mechanism_map.get(str(row["treatment"]), str(row["treatment"])),
                "association": True,
                "temporal_precedence": True,
                "cross_context_invariance": bool(row["sign_consistency"] >= 0.75),
                "perturbational_support": bool(not (row["ci_low"] <= 0 <= row["ci_high"])),
                "causal_convergence": bool(row["evidence_level"] == "causal_convergence"),
                "estimated_effect_on_resistance": float(row["ate"]),
                "evidence_level": row["evidence_level"],
            }
        )
    return pd.DataFrame(rows)


def rank_biomarkers(
    frame: pd.DataFrame,
    graph: nx.DiGraph,
    features: Sequence[str] | None = None,
    assayability: dict[str, float] | None = None,
) -> pd.DataFrame:
    features = list(features or BIOMARKER_FEATURES)
    assayability = assayability or {}
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    early_time = float(sorted(tumor["time_hours"].unique())[1]) if tumor["time_hours"].nunique() > 1 else float(tumor["time_hours"].min())
    early = tumor.loc[tumor["time_hours"] <= early_time].copy()
    early = early.sort_values("time_hours").groupby("lineage_id", as_index=False).tail(1)
    y = early["future_resistant"].astype(int).to_numpy()
    correlation = early[features].corr().abs()
    rows = []
    for feature in features:
        values = early[feature].to_numpy(dtype=float)
        try:
            auc = float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else 0.5
            association = max(auc, 1.0 - auc)
        except ValueError:
            association = 0.5
        donor_correlations = []
        for _, group in early.groupby("donor_id"):
            if group["future_resistant"].nunique() > 1 and group[feature].std() > 1e-8:
                donor_correlations.append(float(group[[feature, "future_resistant"]].corr().iloc[0, 1]))
        if donor_correlations:
            direction = np.sign(np.nanmean(donor_correlations))
            stability = float(np.mean(np.sign(donor_correlations) == direction))
        else:
            stability = 0.0
        if feature in graph and "stable_resistance" in graph:
            try:
                distance = nx.shortest_path_length(graph, feature, "stable_resistance")
                proximity = 1.0 / (1.0 + float(distance))
            except nx.NetworkXNoPath:
                proximity = 0.0
        else:
            proximity = 0.0
        other = correlation.loc[feature].drop(index=feature, errors="ignore")
        uniqueness = float(1.0 - other.mean()) if len(other) else 1.0
        assay = float(assayability.get(feature, 0.65))
        score = (
            0.31 * association
            + 0.24 * stability
            + 0.23 * proximity
            + 0.14 * assay
            + 0.08 * max(0.0, uniqueness)
        )
        rows.append(
            {
                "biomarker": feature,
                "early_time_hours": early_time,
                "association_auc": association,
                "donor_stability": stability,
                "causal_proximity": proximity,
                "assayability": assay,
                "uniqueness": uniqueness,
                "biomarker_score": score,
            }
        )
    ranking = pd.DataFrame(rows).sort_values("biomarker_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def plot_biomarkers(ranking: pd.DataFrame, output_path: str | Path, top_n: int = 9) -> None:
    selected = ranking.head(top_n).sort_values("biomarker_score")
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(selected["biomarker"].str.replace("_", " "), selected["biomarker_score"])
    ax.set_xlabel("Mechanistic biomarker score")
    ax.set_title("Early-warning biomarker ranking")
    ax.set_xlim(0, max(1.0, float(selected["biomarker_score"].max()) * 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def recommend_experiments(
    candidates: Sequence[dict[str, object]],
    effects: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    effect_lookup = effects.set_index("treatment").to_dict(orient="index") if not effects.empty else {}
    evidence_lookup = evidence.set_index("mechanism").to_dict(orient="index") if not evidence.empty else {}
    rows = []
    for candidate in candidates:
        name = str(candidate["name"])
        intervention = str(candidate.get("linked_treatment", ""))
        mechanism = str(candidate.get("mechanism", ""))
        cost = float(candidate.get("relative_cost", 0.5))
        duration = float(candidate.get("relative_duration", 0.5))
        effect = effect_lookup.get(intervention, {})
        estimated_effect = float(effect.get("ate", candidate.get("prior_effect", -0.05)))
        uncertainty = float(effect.get("ci_high", 0.15)) - float(effect.get("ci_low", -0.15))
        evidence_row = evidence_lookup.get(mechanism, {})
        mechanism_certainty = 1.0 if evidence_row.get("causal_convergence") else 0.55
        information_gain_proxy = min(1.0, 0.35 + uncertainty + (1.0 - mechanism_certainty) * 0.35)
        therapeutic_value = min(1.0, max(0.0, -estimated_effect) * 3.5)
        feasibility = max(0.0, 1.0 - 0.55 * cost - 0.30 * duration)
        score = 0.43 * information_gain_proxy + 0.37 * therapeutic_value + 0.20 * feasibility
        rows.append(
            {
                "experiment": name,
                "mechanism": mechanism,
                "linked_treatment": intervention,
                "estimated_effect_on_resistance": estimated_effect,
                "effect_uncertainty_width": uncertainty,
                "expected_information_gain_proxy": information_gain_proxy,
                "therapeutic_value_proxy": therapeutic_value,
                "feasibility": feasibility,
                "priority_score": score,
                "rationale": candidate.get("rationale", "Discriminate among candidate causal mechanisms."),
            }
        )
    output = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    return output
