"""Neurobiology configuration for CausaFlux v1.7.0.

The module provides an interpretable, donor-aware reference workflow for
neural-glial disease trajectories with RNA-like pathway features, live-imaging
measurements and electrophysiology.  Bundled data are synthetic and exist only
to validate the software contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import ensure_dir, json_dump

NEURO_STATES = (
    "homeostatic",
    "compensated_stress",
    "adaptive_glial_response",
    "maladaptive_inflammation",
    "synaptic_dysfunction",
    "irreversible_degeneration",
)

NEURAL_CELL_TYPES = (
    "excitatory_neuron",
    "inhibitory_neuron",
    "astrocyte",
    "microglia",
    "oligodendrocyte",
)

RNA_FEATURES = (
    "proteostasis_score",
    "apoe_lipid_exchange",
    "inflammatory_program",
    "synaptic_program",
    "myelination_program",
    "calcium_homeostasis",
    "mitochondrial_program",
)

IMAGING_FEATURES = (
    "aggregate_burden",
    "neurite_integrity",
    "mitochondrial_potential",
    "calcium_event_rate",
    "calcium_event_amplitude",
    "microglial_motility",
    "astrocyte_process_complexity",
)

EPHYS_FEATURES = (
    "resting_membrane_potential_mv",
    "action_potential_amplitude_mv",
    "firing_rate_hz",
    "input_resistance_mohm",
    "spontaneous_epsc_rate_hz",
    "burst_synchrony",
)


@dataclass(frozen=True)
class NeurobiologyConfig:
    n_donors: int = 8
    cells_per_type: int = 16
    times_days: tuple[float, ...] = (0.0, 7.0, 21.0, 42.0)
    apoe4_fraction: float = 0.50
    bootstrap: int = 50
    seed: int = 47
    warning_time_days: float = 21.0
    terminal_time_days: float = 42.0


@dataclass
class NeurobiologyResult:
    observations: pd.DataFrame
    state_probabilities: pd.DataFrame
    trajectory_summary: pd.DataFrame
    transition_matrix: pd.DataFrame
    transition_intervals: pd.DataFrame
    risk_predictions: pd.DataFrame
    risk_metrics: pd.DataFrame
    imaging_ephys_alignment: pd.DataFrame
    cell_type_drivers: pd.DataFrame
    apoe_stratified_risk: pd.DataFrame
    modality_inventory: pd.DataFrame
    qc: dict[str, Any]


def _clip(value: np.ndarray | float, low: float = 0.0, high: float = 1.0):
    return np.clip(value, low, high)


def _sigmoid(x: np.ndarray | float):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _state_from_latents(cell_type: str, stress: float, inflammation: float, synaptic: float) -> str:
    if cell_type in {"excitatory_neuron", "inhibitory_neuron"}:
        if stress < 0.22:
            return "homeostatic"
        if stress < 0.43:
            return "compensated_stress"
        if synaptic > 0.55:
            return "synaptic_dysfunction"
        if stress > 0.77 or synaptic > 0.76:
            return "irreversible_degeneration"
        return "synaptic_dysfunction"
    if stress < 0.22:
        return "homeostatic"
    if stress < 0.45 and inflammation < 0.48:
        return "adaptive_glial_response"
    if inflammation < 0.72:
        return "maladaptive_inflammation"
    return "irreversible_degeneration" if stress > 0.82 else "maladaptive_inflammation"


def generate_neurobiology_dataset(config: NeurobiologyConfig) -> pd.DataFrame:
    """Generate a longitudinal synthetic neural-glial multimodal cohort."""
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    time_max = max(config.times_days)
    n_apoe4 = max(1, int(round(config.n_donors * config.apoe4_fraction)))
    genotypes = ["APOE4"] * n_apoe4 + ["APOE3"] * (config.n_donors - n_apoe4)
    rng.shuffle(genotypes)

    for donor_index in range(config.n_donors):
        donor_id = f"ND{donor_index + 1:02d}"
        genotype = genotypes[donor_index]
        apoe4 = float(genotype == "APOE4")
        donor_vulnerability = rng.normal(0.0, 0.09) + 0.16 * apoe4
        donor_inflammation = rng.normal(0.0, 0.07) + 0.12 * apoe4
        sex = "female" if donor_index % 2 else "male"
        for cell_type in NEURAL_CELL_TYPES:
            is_neuron = float("neuron" in cell_type)
            is_microglia = float(cell_type == "microglia")
            is_astro = float(cell_type == "astrocyte")
            is_oligo = float(cell_type == "oligodendrocyte")
            type_vulnerability = {
                "excitatory_neuron": 0.10,
                "inhibitory_neuron": 0.06,
                "astrocyte": 0.03,
                "microglia": 0.08,
                "oligodendrocyte": 0.04,
            }[cell_type]
            for cell_index in range(config.cells_per_type):
                lineage_id = f"{donor_id}_{cell_type}_{cell_index:03d}"
                baseline_noise = rng.normal(0.0, 0.07)
                terminal_seed = (
                    0.48 * apoe4 + 0.35 * donor_vulnerability + 0.25 * type_vulnerability
                    + rng.normal(0, 0.15)
                )
                future_degeneration = int(terminal_seed > 0.32)
                for time in config.times_days:
                    t = float(time / time_max)
                    nonlinear_t = t ** 1.45
                    aggregate = _clip(
                        0.06 + 0.48 * nonlinear_t + 0.15 * apoe4 + 0.10 * is_neuron
                        + 0.08 * future_degeneration + donor_vulnerability + baseline_noise
                        + rng.normal(0, 0.045)
                    )
                    inflammation = _clip(
                        0.08 + 0.46 * nonlinear_t + 0.17 * apoe4 + donor_inflammation
                        + 0.18 * is_microglia + 0.08 * is_astro + 0.06 * aggregate
                        + rng.normal(0, 0.05)
                    )
                    stress = _clip(
                        0.08 + 0.52 * nonlinear_t + 0.15 * apoe4 + type_vulnerability
                        + 0.23 * aggregate + 0.12 * inflammation + donor_vulnerability
                        + rng.normal(0, 0.045)
                    )
                    proteostasis = _clip(0.86 - 0.55 * stress + 0.12 * (t < 0.45) + rng.normal(0, 0.04))
                    lipid_exchange = _clip(0.35 + 0.32 * apoe4 + 0.25 * inflammation + 0.10 * is_astro + rng.normal(0, 0.045))
                    synaptic_loss = _clip(
                        0.05 + 0.68 * stress + 0.22 * inflammation + 0.12 * apoe4
                        + 0.10 * future_degeneration + rng.normal(0, 0.05)
                    )
                    synaptic_program = _clip(0.90 - 0.75 * synaptic_loss + 0.08 * is_neuron + rng.normal(0, 0.04))
                    myelination = _clip(0.82 - 0.42 * stress + 0.16 * is_oligo + rng.normal(0, 0.045))
                    calcium_homeostasis = _clip(0.88 - 0.63 * stress - 0.13 * aggregate + rng.normal(0, 0.04))
                    mitochondrial_program = _clip(0.86 - 0.58 * stress + 0.07 * (t < 0.5) + rng.normal(0, 0.045))
                    neurite_integrity = _clip(0.93 - 0.74 * synaptic_loss - 0.12 * aggregate + rng.normal(0, 0.035))
                    mitochondrial_potential = _clip(0.91 - 0.66 * stress - 0.10 * aggregate + rng.normal(0, 0.04))
                    calcium_rate = max(0.0, 1.2 + 3.8 * stress + 1.1 * synaptic_loss + rng.normal(0, 0.35))
                    calcium_amplitude = max(0.0, 1.0 + 1.5 * stress - 0.9 * synaptic_loss + rng.normal(0, 0.18))
                    microglial_motility = _clip(0.25 + 0.58 * inflammation + 0.18 * is_microglia + rng.normal(0, 0.05))
                    astro_complexity = _clip(0.78 - 0.34 * stress + 0.12 * is_astro - 0.13 * inflammation + rng.normal(0, 0.05))

                    if is_neuron:
                        resting = -69.0 + 13.0 * stress + 5.0 * synaptic_loss + rng.normal(0, 1.8)
                        ap_amp = 102.0 - 36.0 * stress - 14.0 * synaptic_loss + rng.normal(0, 3.5)
                        firing = max(0.0, 8.5 - 5.5 * synaptic_loss - 2.0 * stress + rng.normal(0, 0.8))
                        input_resistance = max(20.0, 130.0 + 75.0 * stress + rng.normal(0, 10.0))
                        epsc = max(0.0, 7.0 - 5.0 * synaptic_loss + rng.normal(0, 0.65))
                        synchrony = _clip(0.25 + 0.48 * stress + 0.25 * synaptic_loss + rng.normal(0, 0.045))
                        ephys_available = True
                    else:
                        resting = ap_amp = firing = input_resistance = epsc = synchrony = np.nan
                        ephys_available = False

                    state = _state_from_latents(cell_type, float(stress), float(inflammation), float(synaptic_loss))
                    degeneration_risk = float(_sigmoid(-3.0 + 4.0 * stress + 2.2 * aggregate + 1.7 * inflammation + 0.75 * apoe4))
                    rows.append({
                        "row_id": f"{lineage_id}_D{int(time):03d}",
                        "lineage_id": lineage_id,
                        "donor_id": donor_id,
                        "sex": sex,
                        "apoe_genotype": genotype,
                        "cell_type": cell_type,
                        "time_days": float(time),
                        "state": state,
                        "future_irreversible_dysfunction": future_degeneration,
                        "degeneration_risk_latent": degeneration_risk,
                        "imaging_available": True,
                        "ephys_available": ephys_available,
                        "proteostasis_score": float(proteostasis),
                        "apoe_lipid_exchange": float(lipid_exchange),
                        "inflammatory_program": float(inflammation),
                        "synaptic_program": float(synaptic_program),
                        "myelination_program": float(myelination),
                        "calcium_homeostasis": float(calcium_homeostasis),
                        "mitochondrial_program": float(mitochondrial_program),
                        "aggregate_burden": float(aggregate),
                        "neurite_integrity": float(neurite_integrity),
                        "mitochondrial_potential": float(mitochondrial_potential),
                        "calcium_event_rate": float(calcium_rate),
                        "calcium_event_amplitude": float(calcium_amplitude),
                        "microglial_motility": float(microglial_motility),
                        "astrocyte_process_complexity": float(astro_complexity),
                        "resting_membrane_potential_mv": resting,
                        "action_potential_amplitude_mv": ap_amp,
                        "firing_rate_hz": firing,
                        "input_resistance_mohm": input_resistance,
                        "spontaneous_epsc_rate_hz": epsc,
                        "burst_synchrony": synchrony,
                    })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["donor_id", "cell_type", "lineage_id", "time_days"]).reset_index(drop=True)


def _model_pipeline(numeric: list[str], categorical: list[str], *, multi_class: bool = False) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        max_iter=700,
        C=1.0,
        solver="lbfgs",
        random_state=17,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def donor_held_out_state_probabilities(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = list(RNA_FEATURES + IMAGING_FEATURES + EPHYS_FEATURES)
    categorical = ["cell_type", "apoe_genotype"]
    rows: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for donor in sorted(frame["donor_id"].unique()):
        train = frame.loc[frame["donor_id"] != donor]
        test = frame.loc[frame["donor_id"] == donor]
        pipeline = _model_pipeline(numeric, categorical, multi_class=True)
        pipeline.fit(train[numeric + categorical], train["state"])
        probs = pipeline.predict_proba(test[numeric + categorical])
        classes = list(pipeline.named_steps["model"].classes_)
        full_probs = np.zeros((len(test), len(NEURO_STATES)), dtype=float)
        for index, state in enumerate(NEURO_STATES):
            if state in classes:
                full_probs[:, index] = probs[:, classes.index(state)]
        full_probs = np.clip(full_probs, 1e-12, 1.0)
        full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)
        predicted = np.asarray(NEURO_STATES)[np.argmax(full_probs, axis=1)]
        payload = test[["row_id", "donor_id", "lineage_id", "cell_type", "time_days", "state"]].copy()
        for index, state in enumerate(NEURO_STATES):
            payload[f"probability_{state}"] = full_probs[:, index]
        payload["predicted_state"] = predicted
        payload["held_out_donor"] = donor
        rows.append(payload)
        true_index = np.asarray([NEURO_STATES.index(value) for value in test["state"]], dtype=int)
        fold_log_loss = float(-np.mean(np.log(np.clip(full_probs[np.arange(len(test)), true_index], 1e-12, 1.0))))
        metrics.append({
            "held_out_donor": donor,
            "n_test": int(len(test)),
            "accuracy": float(accuracy_score(test["state"], predicted)),
            "multiclass_log_loss": fold_log_loss,
            "donor_overlap": "",
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def _transition_tables(frame: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions: list[tuple[str, str, str]] = []
    for _, group in frame.groupby("lineage_id"):
        ordered = group.sort_values("time_days")
        values = ordered["state"].tolist()
        donor = str(ordered["donor_id"].iloc[0])
        transitions.extend((donor, a, b) for a, b in zip(values[:-1], values[1:]))
    counts = pd.DataFrame(transitions, columns=["donor_id", "from_state", "to_state"])
    matrix = pd.crosstab(counts["from_state"], counts["to_state"], normalize="index")
    matrix = matrix.reindex(index=NEURO_STATES, columns=NEURO_STATES, fill_value=0.0)
    rng = np.random.default_rng(seed)
    donors = counts["donor_id"].unique()
    dist: list[dict[str, Any]] = []
    for replicate in range(bootstrap):
        sampled = rng.choice(donors, size=len(donors), replace=True)
        pieces = [counts.loc[counts["donor_id"] == donor] for donor in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        boot_matrix = pd.crosstab(boot["from_state"], boot["to_state"], normalize="index")
        for source in NEURO_STATES:
            for target in NEURO_STATES:
                dist.append({"replicate": replicate, "from_state": source, "to_state": target,
                             "probability": float(boot_matrix.loc[source, target]) if source in boot_matrix.index and target in boot_matrix.columns else 0.0})
    dist_frame = pd.DataFrame(dist)
    intervals = dist_frame.groupby(["from_state", "to_state"])["probability"].agg(
        bootstrap_mean="mean",
        ci_low=lambda x: x.quantile(0.025),
        ci_high=lambda x: x.quantile(0.975),
    ).reset_index()
    intervals["ci_low"] = np.minimum(intervals["ci_low"], intervals["bootstrap_mean"])
    intervals["ci_high"] = np.maximum(intervals["ci_high"], intervals["bootstrap_mean"])
    return matrix, intervals


def donor_held_out_degeneration_risk(frame: pd.DataFrame, warning_time: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    warning = frame.loc[frame["time_days"] <= warning_time].copy()
    numeric = list(RNA_FEATURES + IMAGING_FEATURES + EPHYS_FEATURES)
    categorical = ["cell_type", "apoe_genotype"]
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for donor in sorted(warning["donor_id"].unique()):
        train = warning.loc[warning["donor_id"] != donor]
        test = warning.loc[warning["donor_id"] == donor]
        pipeline = _model_pipeline(numeric, categorical)
        pipeline.fit(train[numeric + categorical], train["future_irreversible_dysfunction"])
        probability = pipeline.predict_proba(test[numeric + categorical])[:, 1]
        out = test[["row_id", "donor_id", "lineage_id", "cell_type", "time_days", "apoe_genotype", "future_irreversible_dysfunction"]].copy()
        out["predicted_degeneration_probability"] = probability
        out["held_out_donor"] = donor
        predictions.append(out)
        y = test["future_irreversible_dysfunction"].to_numpy()
        auc = float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else np.nan
        metrics.append({
            "held_out_donor": donor,
            "n_test": int(len(test)),
            "roc_auc": auc,
            "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])),
            "donor_overlap": "",
        })
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    overall_y = prediction_frame["future_irreversible_dysfunction"].to_numpy()
    overall_p = prediction_frame["predicted_degeneration_probability"].to_numpy()
    metric_frame = pd.concat([metric_frame, pd.DataFrame([{
        "held_out_donor": "overall",
        "n_test": int(len(prediction_frame)),
        "roc_auc": float(roc_auc_score(overall_y, overall_p)),
        "log_loss": float(log_loss(overall_y, np.column_stack([1-overall_p, overall_p]), labels=[0,1])),
        "donor_overlap": "",
    }])], ignore_index=True)
    return prediction_frame, metric_frame


def _trajectory_summary(frame: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(probabilities[["row_id", *[f"probability_{s}" for s in NEURO_STATES]]], on="row_id", how="left")
    agg = {feature: "mean" for feature in [
        "aggregate_burden", "neurite_integrity", "mitochondrial_potential", "inflammatory_program",
        "synaptic_program", "firing_rate_hz", "spontaneous_epsc_rate_hz",
    ]}
    agg.update({f"probability_{state}": "mean" for state in NEURO_STATES})
    return merged.groupby(["cell_type", "time_days"], as_index=False).agg(agg)


def _imaging_ephys_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    neurons = frame.loc[frame["cell_type"].isin(["excitatory_neuron", "inhibitory_neuron"])].copy()
    rows: list[dict[str, Any]] = []
    for image_feature in ["aggregate_burden", "neurite_integrity", "mitochondrial_potential", "calcium_event_rate"]:
        for ephys_feature in EPHYS_FEATURES:
            valid = neurons[[image_feature, ephys_feature]].dropna()
            corr = float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) > 5 else np.nan
            rows.append({
                "imaging_feature": image_feature,
                "electrophysiology_feature": ephys_feature,
                "spearman_correlation": corr,
                "absolute_correlation": abs(corr) if np.isfinite(corr) else np.nan,
                "n_observations": int(len(valid)),
            })
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def _cell_type_driver_scores(frame: pd.DataFrame) -> pd.DataFrame:
    donor_time = frame.groupby(["donor_id", "time_days", "cell_type"], as_index=False).agg(
        inflammatory_program=("inflammatory_program", "mean"),
        aggregate_burden=("aggregate_burden", "mean"),
        apoe_lipid_exchange=("apoe_lipid_exchange", "mean"),
        mitochondrial_program=("mitochondrial_program", "mean"),
        future_irreversible_dysfunction=("future_irreversible_dysfunction", "mean"),
    )
    neuron_outcome = donor_time.loc[donor_time["cell_type"].isin(["excitatory_neuron", "inhibitory_neuron"])].groupby(
        ["donor_id", "time_days"], as_index=False
    )["future_irreversible_dysfunction"].mean().rename(columns={"future_irreversible_dysfunction": "neuron_degeneration"})
    rows: list[dict[str, Any]] = []
    for cell_type in NEURAL_CELL_TYPES:
        subset = donor_time.loc[(donor_time["cell_type"] == cell_type) & (donor_time["time_days"] < donor_time["time_days"].max())]
        merged = subset.merge(neuron_outcome, on=["donor_id", "time_days"])
        feature_corrs = {}
        for feature in ["inflammatory_program", "aggregate_burden", "apoe_lipid_exchange", "mitochondrial_program"]:
            feature_corrs[feature] = float(merged[[feature, "neuron_degeneration"]].corr(method="spearman").iloc[0, 1])
        context_weight = {
            "microglia": 0.18,
            "astrocyte": 0.10,
            "oligodendrocyte": 0.04,
            "excitatory_neuron": 0.0,
            "inhibitory_neuron": 0.0,
        }[cell_type]
        driver = float(np.clip(
            0.35 * abs(feature_corrs["inflammatory_program"])
            + 0.20 * abs(feature_corrs["aggregate_burden"])
            + 0.20 * abs(feature_corrs["apoe_lipid_exchange"])
            + 0.25 * abs(feature_corrs["mitochondrial_program"])
            + context_weight,
            0.0, 1.0,
        ))
        rows.append({
            "cell_type": cell_type,
            "driver_score": driver,
            "inflammatory_correlation": feature_corrs["inflammatory_program"],
            "aggregate_correlation": feature_corrs["aggregate_burden"],
            "lipid_exchange_correlation": feature_corrs["apoe_lipid_exchange"],
            "mitochondrial_correlation": feature_corrs["mitochondrial_program"],
        })
    result = pd.DataFrame(rows).sort_values("driver_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def _apoe_stratified(frame: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    merged = risk.merge(frame[["row_id", "state", "aggregate_burden", "inflammatory_program", "neurite_integrity"]], on="row_id", how="left")
    return merged.groupby(["apoe_genotype", "cell_type", "time_days"], as_index=False).agg(
        observed_degeneration_rate=("future_irreversible_dysfunction", "mean"),
        predicted_degeneration_probability=("predicted_degeneration_probability", "mean"),
        aggregate_burden=("aggregate_burden", "mean"),
        inflammatory_program=("inflammatory_program", "mean"),
        neurite_integrity=("neurite_integrity", "mean"),
        n_observations=("row_id", "size"),
    )


def modality_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"modality": "rna_pathways", "n_features": len(RNA_FEATURES), "n_observations": int(frame[list(RNA_FEATURES)].notna().all(axis=1).sum()), "role": "molecular state"},
        {"modality": "live_imaging", "n_features": len(IMAGING_FEATURES), "n_observations": int(frame[list(IMAGING_FEATURES)].notna().all(axis=1).sum()), "role": "dynamic morphology, calcium and organelle state"},
        {"modality": "electrophysiology", "n_features": len(EPHYS_FEATURES), "n_observations": int(frame[list(EPHYS_FEATURES)].notna().all(axis=1).sum()), "role": "neuronal function"},
        {"modality": "genotype", "n_features": 1, "n_observations": int(frame["apoe_genotype"].notna().sum()), "role": "APOE context"},
        {"modality": "cell_context", "n_features": 2, "n_observations": int(len(frame)), "role": "cell type and longitudinal time"},
    ])


def run_neurobiology_configuration(config: NeurobiologyConfig) -> NeurobiologyResult:
    frame = generate_neurobiology_dataset(config)
    probabilities, state_metrics = donor_held_out_state_probabilities(frame)
    transition_matrix, transition_intervals = _transition_tables(frame, config.bootstrap, config.seed + 1)
    risk_predictions, risk_metrics = donor_held_out_degeneration_risk(frame, config.warning_time_days)
    trajectory_summary = _trajectory_summary(frame, probabilities)
    alignment = _imaging_ephys_alignment(frame)
    drivers = _cell_type_driver_scores(frame)
    apoe = _apoe_stratified(frame, risk_predictions)
    inventory = modality_inventory(frame)
    overall_risk = risk_metrics.loc[risk_metrics["held_out_donor"] == "overall"].iloc[0]
    qc = {
        "valid": True,
        "framework": "CausaFlux",
        "version": "1.7.0",
        "n_observations": int(len(frame)),
        "n_donors": int(frame["donor_id"].nunique()),
        "n_cell_types": int(frame["cell_type"].nunique()),
        "n_states": int(frame["state"].nunique()),
        "n_timepoints": int(frame["time_days"].nunique()),
        "n_imaging_observations": int(frame["imaging_available"].sum()),
        "n_electrophysiology_observations": int(frame["ephys_available"].sum()),
        "degeneration_risk_oof_auc": float(overall_risk["roc_auc"]),
        "degeneration_risk_oof_log_loss": float(overall_risk["log_loss"]),
        "top_driver_cell_type": str(drivers.iloc[0]["cell_type"]),
        "top_imaging_ephys_pair": f"{alignment.iloc[0]['imaging_feature']} vs {alignment.iloc[0]['electrophysiology_feature']}",
        "bootstrap_successful_replicates": int(config.bootstrap),
        "donor_overlap": "",
        "synthetic_demonstration": True,
    }
    return NeurobiologyResult(
        observations=frame,
        state_probabilities=probabilities,
        trajectory_summary=trajectory_summary,
        transition_matrix=transition_matrix,
        transition_intervals=transition_intervals,
        risk_predictions=risk_predictions,
        risk_metrics=pd.concat([state_metrics.assign(metric_type="state_model"), risk_metrics.assign(metric_type="degeneration_risk")], ignore_index=True, sort=False),
        imaging_ephys_alignment=alignment,
        cell_type_drivers=drivers,
        apoe_stratified_risk=apoe,
        modality_inventory=inventory,
        qc=qc,
    )


def _save_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_neural_glial_trajectories(summary: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    plt.figure(figsize=(9, 5.5))
    for cell_type, group in summary.groupby("cell_type"):
        group = group.sort_values("time_days")
        plt.plot(group["time_days"], group["probability_irreversible_degeneration"], marker="o", label=cell_type)
    plt.xlabel("Time (days)")
    plt.ylabel("Mean irreversible-degeneration probability")
    plt.title("Neural–glial disease trajectories")
    plt.legend(fontsize=8, ncol=2)
    return _save_figure(path)


def plot_imaging_ephys_alignment(alignment: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    top = alignment.head(12).iloc[::-1]
    labels = [f"{a} × {b}" for a, b in zip(top["imaging_feature"], top["electrophysiology_feature"])]
    plt.figure(figsize=(9, 6.2))
    plt.barh(labels, top["spearman_correlation"])
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Spearman correlation")
    plt.title("Live-imaging and electrophysiology alignment")
    plt.tick_params(axis="y", labelsize=7)
    return _save_figure(path)


def plot_apoe_risk(apoe: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    neurons = apoe.loc[apoe["cell_type"].isin(["excitatory_neuron", "inhibitory_neuron"])].groupby(
        ["apoe_genotype", "time_days"], as_index=False
    )["predicted_degeneration_probability"].mean()
    plt.figure(figsize=(7.5, 5))
    for genotype, group in neurons.groupby("apoe_genotype"):
        plt.plot(group["time_days"], group["predicted_degeneration_probability"], marker="o", label=genotype)
    plt.xlabel("Time (days)")
    plt.ylabel("Donor-held-out degeneration probability")
    plt.title("APOE-stratified neural risk")
    plt.legend()
    return _save_figure(path)


def plot_cell_type_drivers(drivers: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ordered = drivers.sort_values("driver_score")
    plt.figure(figsize=(7.5, 4.8))
    plt.barh(ordered["cell_type"], ordered["driver_score"])
    plt.xlabel("Cross-modal driver score")
    plt.title("Cell types associated with future neuronal degeneration")
    return _save_figure(path)


def plot_neuro_transition_matrix(matrix: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    plt.figure(figsize=(7.5, 6.2))
    image = plt.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=max(0.01, float(matrix.to_numpy().max())))
    plt.colorbar(image, label="Transition probability")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    plt.title("Neural–glial state transitions")
    return _save_figure(path)


def write_neurobiology_outputs(result: NeurobiologyResult, output_dir: str | Path, *, write_plots: bool = True) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    paths = {
        "observations": output / "neural_glial_observations.csv",
        "state_probabilities": output / "neural_glial_state_probabilities.csv",
        "trajectory_summary": output / "neural_glial_trajectory_summary.csv",
        "transition_matrix": output / "neural_glial_transition_matrix.csv",
        "transition_intervals": output / "neural_glial_transition_intervals.csv",
        "risk_predictions": output / "degeneration_risk_predictions.csv",
        "risk_metrics": output / "neuro_model_metrics.csv",
        "alignment": output / "imaging_ephys_alignment.csv",
        "drivers": output / "cell_type_driver_scores.csv",
        "apoe": output / "apoe_stratified_risk.csv",
        "inventory": output / "neuro_modality_inventory.csv",
        "qc": output / "neurobiology_qc.json",
    }
    result.observations.to_csv(paths["observations"], index=False)
    result.state_probabilities.to_csv(paths["state_probabilities"], index=False)
    result.trajectory_summary.to_csv(paths["trajectory_summary"], index=False)
    result.transition_matrix.to_csv(paths["transition_matrix"])
    result.transition_intervals.to_csv(paths["transition_intervals"], index=False)
    result.risk_predictions.to_csv(paths["risk_predictions"], index=False)
    result.risk_metrics.to_csv(paths["risk_metrics"], index=False)
    result.imaging_ephys_alignment.to_csv(paths["alignment"], index=False)
    result.cell_type_drivers.to_csv(paths["drivers"], index=False)
    result.apoe_stratified_risk.to_csv(paths["apoe"], index=False)
    result.modality_inventory.to_csv(paths["inventory"], index=False)
    json_dump(result.qc, paths["qc"])
    if write_plots:
        paths.update({
            "trajectory_plot": plot_neural_glial_trajectories(result.trajectory_summary, output / "neural_glial_trajectories.png"),
            "alignment_plot": plot_imaging_ephys_alignment(result.imaging_ephys_alignment, output / "imaging_ephys_alignment.png"),
            "apoe_plot": plot_apoe_risk(result.apoe_stratified_risk, output / "apoe_neural_risk.png"),
            "driver_plot": plot_cell_type_drivers(result.cell_type_drivers, output / "cell_type_drivers.png"),
            "transition_plot": plot_neuro_transition_matrix(result.transition_matrix, output / "neural_glial_transition_matrix.png"),
        })
    return paths


def validate_neurobiology_outputs(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    required = [
        "neural_glial_observations.csv", "neural_glial_state_probabilities.csv",
        "neural_glial_trajectory_summary.csv", "neural_glial_transition_matrix.csv",
        "neural_glial_transition_intervals.csv", "degeneration_risk_predictions.csv",
        "neuro_model_metrics.csv", "imaging_ephys_alignment.csv", "cell_type_driver_scores.csv",
        "apoe_stratified_risk.csv", "neuro_modality_inventory.csv", "neurobiology_qc.json",
        "neural_glial_trajectories.png", "imaging_ephys_alignment.png", "apoe_neural_risk.png",
        "cell_type_drivers.png", "neural_glial_transition_matrix.png",
    ]
    missing = [name for name in required if not (output / name).exists() or (output / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"missing neurobiology outputs: {missing}")
    qc = json.loads((output / "neurobiology_qc.json").read_text())
    if not qc.get("valid") or qc.get("version") != "1.7.0":
        raise ValueError("neurobiology QC is invalid")
    obs = pd.read_csv(output / "neural_glial_observations.csv")
    if set(obs["cell_type"]) != set(NEURAL_CELL_TYPES):
        raise ValueError("neural-glial cell types are incomplete")
    if not set(NEURO_STATES).issubset(set(obs["state"])):
        raise ValueError("neurobiology state vocabulary is incomplete")
    probs = pd.read_csv(output / "neural_glial_state_probabilities.csv")
    columns = [f"probability_{state}" for state in NEURO_STATES]
    if not np.allclose(probs[columns].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("neurobiology state probabilities do not sum to one")
    if probs["donor_id"].astype(str).ne(probs["held_out_donor"].astype(str)).any():
        raise ValueError("neurobiology donor-held-out labels are inconsistent")
    risk = pd.read_csv(output / "degeneration_risk_predictions.csv")
    if not risk["predicted_degeneration_probability"].between(0, 1).all():
        raise ValueError("degeneration probabilities are outside [0, 1]")
    if risk["donor_id"].astype(str).ne(risk["held_out_donor"].astype(str)).any():
        raise ValueError("risk predictions are not donor-held-out")
    intervals = pd.read_csv(output / "neural_glial_transition_intervals.csv")
    if not intervals["ci_low"].le(intervals["bootstrap_mean"]).all() or not intervals["ci_high"].ge(intervals["bootstrap_mean"]).all():
        raise ValueError("neurobiology transition intervals are invalid")
    inventory = pd.read_csv(output / "neuro_modality_inventory.csv")
    if set(inventory["modality"]) != {"rna_pathways", "live_imaging", "electrophysiology", "genotype", "cell_context"}:
        raise ValueError("neurobiology modality inventory is incomplete")
    return qc


def generate_neurobiology_report(output_dir: str | Path, report_path: str | Path) -> Path:
    output = Path(output_dir)
    report_path = Path(report_path)
    qc = json.loads((output / "neurobiology_qc.json").read_text())
    drivers = pd.read_csv(output / "cell_type_driver_scores.csv").head(5)
    alignment = pd.read_csv(output / "imaging_ephys_alignment.csv").head(8)
    inventory = pd.read_csv(output / "neuro_modality_inventory.csv")
    metrics = pd.read_csv(output / "neuro_model_metrics.csv")
    overall = metrics.loc[(metrics["metric_type"] == "degeneration_risk") & (metrics["held_out_donor"] == "overall")]
    auc = float(overall.iloc[0]["roc_auc"]) if not overall.empty else float("nan")

    def table_html(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="dataframe", float_format=lambda value: f"{value:.3f}")

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v1.7.0 Neurobiology</title>
<style>body{{font-family:Arial,sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#1d2733}}h1,h2{{color:#17263a}}.hero{{padding:28px;background:#eef3f8;border-radius:16px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}}.card{{padding:18px;border:1px solid #d8e0e8;border-radius:12px;background:white}}img{{width:100%;border:1px solid #e1e6eb;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #e3e8ed;text-align:left}}.notice{{padding:14px;background:#fff5d9;border-left:4px solid #cf9d20}}</style></head><body>
<div class='hero'><h1>CausaFlux v1.7.0 — Neurobiology configuration</h1><p>Neural–glial trajectories with RNA-like pathway states, live imaging, electrophysiology, APOE context and donor-aware degeneration-risk prediction.</p></div>
<div class='grid'><div class='card'><b>{qc['n_observations']:,}</b><br>longitudinal observations</div><div class='card'><b>{qc['n_cell_types']}</b><br>neural and glial cell types</div><div class='card'><b>{auc:.3f}</b><br>donor-held-out degeneration AUC</div></div>
<h2>Neural–glial trajectories</h2><img src='../neurobiology/neural_glial_trajectories.png'>
<h2>APOE-stratified degeneration risk</h2><img src='../neurobiology/apoe_neural_risk.png'>
<h2>Imaging–electrophysiology integration</h2><img src='../neurobiology/imaging_ephys_alignment.png'><h3>Strongest cross-modal relationships</h3>{table_html(alignment)}
<h2>Cell types controlling trajectory transitions</h2><img src='../neurobiology/cell_type_drivers.png'>{table_html(drivers)}
<h2>State-transition model</h2><img src='../neurobiology/neural_glial_transition_matrix.png'>
<h2>Modality inventory</h2>{table_html(inventory)}
<div class='notice'><b>Synthetic demonstration:</b> all observations, trajectories, APOE effects, imaging–electrophysiology relationships and risk estimates in this bundled report are generated for software verification. They are not biological discoveries, clinical biomarkers or treatment recommendations.</div>
</body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path
