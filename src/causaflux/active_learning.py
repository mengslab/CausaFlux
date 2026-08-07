"""Closed-loop experimental design for CausaFlux v1.7.0.

The engine ranks CRISPR, drug, imaging and sampling-time experiments against
explicit competing mechanistic hypotheses. It uses a transparent Bayesian
expected-information-gain calculation, keeps utility components separate,
selects a diverse budget-constrained batch, and demonstrates posterior updating
without claiming that synthetic outcomes are biological evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, json_dump


EXPERIMENT_TYPES = ("crispr", "drug", "imaging", "sampling_time")


@dataclass(frozen=True)
class ClosedLoopConfig:
    budget: float = 2.4
    batch_size: int = 4
    round2_budget: float = 2.0
    round2_batch_size: int = 3
    max_per_type: int = 2
    require_type_coverage: bool = True
    diversity_penalty: float = 0.10
    information_gain_weight: float = 0.40
    therapeutic_value_weight: float = 0.22
    biomarker_value_weight: float = 0.13
    temporal_value_weight: float = 0.10
    feasibility_weight: float = 0.15
    bootstrap: int = 60
    eig_samples: int = 1200
    seed: int = 31
    simulate_demonstration_round: bool = True
    true_hypothesis: str | None = None


@dataclass
class ClosedLoopResult:
    hypotheses: pd.DataFrame
    catalog: pd.DataFrame
    round1_ranking: pd.DataFrame
    round1_batch: pd.DataFrame
    simulated_observations: pd.DataFrame
    posterior_history: pd.DataFrame
    round2_ranking: pd.DataFrame
    round2_batch: pd.DataFrame
    outcome_templates: pd.DataFrame
    bootstrap_distributions: pd.DataFrame
    qc: dict[str, Any]


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


def _normal_logpdf(values: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return -0.5 * ((values[..., None] - means[None, ...]) / sigma) ** 2 - math.log(
        sigma * math.sqrt(2.0 * math.pi)
    )


def _expected_information_gain(
    prior: np.ndarray,
    means: np.ndarray,
    sigma: float,
    n_samples: int,
    seed: int,
) -> float:
    """Monte-Carlo mutual information I(H;Y) for a Gaussian observation model."""
    prior = np.asarray(prior, dtype=float)
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()
    means = np.asarray(means, dtype=float)
    if means.size != prior.size:
        raise ValueError("hypothesis means must match the prior length")
    rng = np.random.default_rng(seed)
    hypotheses = rng.choice(prior.size, size=max(int(n_samples), 100), p=prior)
    observations = rng.normal(means[hypotheses], max(float(sigma), 1e-6))
    log_likelihood = _normal_logpdf(observations, means, sigma)
    log_weights = log_likelihood + np.log(prior)[None, :]
    log_weights -= log_weights.max(axis=1, keepdims=True)
    posterior = np.exp(log_weights)
    posterior /= posterior.sum(axis=1, keepdims=True)
    posterior_entropy = np.mean([-np.sum(row[row > 0] * np.log(row[row > 0])) for row in posterior])
    return max(0.0, _entropy(prior) - float(posterior_entropy))


def _posterior_update(prior: np.ndarray, observation: float, means: np.ndarray, sigma: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    prior = np.clip(prior, 1e-12, None)
    prior /= prior.sum()
    log_weight = np.log(prior) + _normal_logpdf(np.asarray([observation]), means, sigma)[0]
    log_weight -= log_weight.max()
    posterior = np.exp(log_weight)
    posterior /= posterior.sum()
    return posterior


def _normalize(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    lo, hi = float(numeric.min()), float(numeric.max())
    if math.isclose(lo, hi):
        result = pd.Series(np.full(len(numeric), 0.5), index=numeric.index)
    else:
        result = (numeric - lo) / (hi - lo)
    return result if higher_is_better else 1.0 - result


def _default_hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": "H1_PROTEOSTASIS_UPSTREAM",
            "hypothesis": "IRE1-XBP1 proteostasis is upstream of persistent tolerance",
            "mechanism": "IRE1-XBP1 proteostasis",
            "prior_probability": 0.30,
        },
        {
            "hypothesis_id": "H2_MITOCHONDRIAL_PARALLEL",
            "hypothesis": "Mitochondrial reserve is a parallel driver of persistent tolerance",
            "mechanism": "Mitochondrial reserve",
            "prior_probability": 0.25,
        },
        {
            "hypothesis_id": "H3_ANTIGEN_EXCLUSION",
            "hypothesis": "Antigen-presentation failure drives immune-protected resistance",
            "mechanism": "Antigen presentation",
            "prior_probability": 0.25,
        },
        {
            "hypothesis_id": "H4_ENHANCER_COMMITMENT",
            "hypothesis": "Enhancer plasticity controls commitment to stable resistance",
            "mechanism": "Enhancer plasticity",
            "prior_probability": 0.20,
        },
    ]


def _default_candidates(hypothesis_ids: Sequence[str]) -> list[dict[str, Any]]:
    # Expected standardized readout under each competing hypothesis. Separation,
    # rather than absolute magnitude, determines information value.
    h1, h2, h3, h4 = hypothesis_ids
    return [
        {
            "experiment_id": "CRISPR_XBP1_24H",
            "experiment_name": "CRISPRi XBP1 at 24 h",
            "experiment_type": "crispr",
            "mechanism": "IRE1-XBP1 proteostasis",
            "target": "XBP1",
            "readout": "resistance probability and proteostasis-state shift",
            "sample_time_hours": 72,
            "relative_cost": 0.62,
            "relative_duration": 0.58,
            "technical_risk": 0.24,
            "measurement_noise": 0.32,
            "model_uncertainty": 0.10,
            "hypothesis_effects": {h1: -1.15, h2: -0.35, h3: -0.12, h4: -0.45},
            "therapeutic_value": 0.88,
            "rationale": "Tests whether the IRE1-XBP1 program is causally upstream of persistent tolerance.",
        },
        {
            "experiment_id": "CRISPR_NDUFS1_24H",
            "experiment_name": "CRISPRi NDUFS1 at 24 h",
            "experiment_type": "crispr",
            "mechanism": "Mitochondrial reserve",
            "target": "NDUFS1",
            "readout": "mitochondrial reserve and resistant-state probability",
            "sample_time_hours": 72,
            "relative_cost": 0.64,
            "relative_duration": 0.60,
            "technical_risk": 0.28,
            "measurement_noise": 0.34,
            "model_uncertainty": 0.11,
            "hypothesis_effects": {h1: -0.25, h2: -1.10, h3: -0.10, h4: -0.28},
            "therapeutic_value": 0.78,
            "rationale": "Discriminates a parallel energetic-reserve mechanism from proteostasis dependence.",
        },
        {
            "experiment_id": "CRISPR_B2M_TAP1_RESCUE",
            "experiment_name": "CRISPRa B2M/TAP1 antigen-presentation rescue",
            "experiment_type": "crispr",
            "mechanism": "Antigen presentation",
            "target": "B2M;TAP1",
            "readout": "antigen presentation, T-cell contact and resistance",
            "sample_time_hours": 96,
            "relative_cost": 0.70,
            "relative_duration": 0.66,
            "technical_risk": 0.34,
            "measurement_noise": 0.36,
            "model_uncertainty": 0.12,
            "hypothesis_effects": {h1: -0.10, h2: -0.08, h3: -1.18, h4: -0.18},
            "therapeutic_value": 0.74,
            "rationale": "Tests whether restoring antigen processing breaks an immune-protected resistant niche.",
        },
        {
            "experiment_id": "CRISPR_EP300_24H",
            "experiment_name": "CRISPRi EP300 enhancer-plasticity perturbation",
            "experiment_type": "crispr",
            "mechanism": "Enhancer plasticity",
            "target": "EP300",
            "readout": "enhancer-state commitment and resistant-state probability",
            "sample_time_hours": 72,
            "relative_cost": 0.66,
            "relative_duration": 0.62,
            "technical_risk": 0.31,
            "measurement_noise": 0.35,
            "model_uncertainty": 0.12,
            "hypothesis_effects": {h1: -0.32, h2: -0.20, h3: -0.10, h4: -1.08},
            "therapeutic_value": 0.64,
            "rationale": "Tests whether enhancer plasticity controls the irreversible commitment step.",
        },
        {
            "experiment_id": "DRUG_IRE1I_TIMECOURSE",
            "experiment_name": "IRE1 inhibitor dose-time course",
            "experiment_type": "drug",
            "mechanism": "IRE1-XBP1 proteostasis",
            "target": "IRE1 inhibitor",
            "readout": "dose-dependent resistance and XBP1 program suppression",
            "sample_time_hours": 72,
            "relative_cost": 0.42,
            "relative_duration": 0.43,
            "technical_risk": 0.18,
            "measurement_noise": 0.30,
            "model_uncertainty": 0.09,
            "hypothesis_effects": {h1: -1.00, h2: -0.30, h3: -0.12, h4: -0.38},
            "therapeutic_value": 0.92,
            "rationale": "Estimates the intervention window and separates target dependence from generic toxicity.",
        },
        {
            "experiment_id": "DRUG_MITOI_TIMECOURSE",
            "experiment_name": "Mitochondrial-reserve inhibitor dose-time course",
            "experiment_type": "drug",
            "mechanism": "Mitochondrial reserve",
            "target": "mitochondrial inhibitor",
            "readout": "reserve capacity, viability and resistance",
            "sample_time_hours": 72,
            "relative_cost": 0.44,
            "relative_duration": 0.45,
            "technical_risk": 0.23,
            "measurement_noise": 0.31,
            "model_uncertainty": 0.10,
            "hypothesis_effects": {h1: -0.25, h2: -1.02, h3: -0.10, h4: -0.24},
            "therapeutic_value": 0.80,
            "rationale": "Tests energetic reserve as an independently actionable resistance mechanism.",
        },
        {
            "experiment_id": "DRUG_IFNG_RESCUE",
            "experiment_name": "IFNG antigen-presentation rescue",
            "experiment_type": "drug",
            "mechanism": "Antigen presentation",
            "target": "IFNG support",
            "readout": "antigen presentation, immune exclusion and resistance",
            "sample_time_hours": 96,
            "relative_cost": 0.40,
            "relative_duration": 0.38,
            "technical_risk": 0.20,
            "measurement_noise": 0.33,
            "model_uncertainty": 0.10,
            "hypothesis_effects": {h1: -0.08, h2: -0.08, h3: -0.98, h4: -0.14},
            "therapeutic_value": 0.76,
            "rationale": "Tests whether restored antigen presentation reduces immune-protected resistance.",
        },
        {
            "experiment_id": "DRUG_DUAL_IRE1_MITO",
            "experiment_name": "Dual IRE1 and mitochondrial perturbation",
            "experiment_type": "drug",
            "mechanism": "Proteostasis-mitochondrial interaction",
            "target": "IRE1 inhibitor + mitochondrial inhibitor",
            "readout": "synergy, toxicity and resistance",
            "sample_time_hours": 72,
            "relative_cost": 0.78,
            "relative_duration": 0.58,
            "technical_risk": 0.32,
            "measurement_noise": 0.34,
            "model_uncertainty": 0.14,
            "hypothesis_effects": {h1: -1.10, h2: -1.06, h3: -0.18, h4: -0.50},
            "therapeutic_value": 0.95,
            "rationale": "Tests redundancy or synergy between proteostasis and energetic adaptation.",
        },
        {
            "experiment_id": "IMG_ER_REPORTER_24_72",
            "experiment_name": "Live ER-stress reporter imaging from 24–72 h",
            "experiment_type": "imaging",
            "mechanism": "IRE1-XBP1 proteostasis",
            "target": "XBP1/ER-stress reporter",
            "readout": "single-cell activation timing, persistence and reversibility",
            "sample_time_hours": 48,
            "relative_cost": 0.48,
            "relative_duration": 0.50,
            "technical_risk": 0.26,
            "measurement_noise": 0.28,
            "model_uncertainty": 0.08,
            "hypothesis_effects": {h1: 1.05, h2: 0.25, h3: 0.12, h4: 0.55},
            "therapeutic_value": 0.38,
            "rationale": "Resolves whether proteostasis activation precedes or follows tolerance commitment.",
        },
        {
            "experiment_id": "IMG_MITO_POTENTIAL_24_72",
            "experiment_name": "Live mitochondrial-potential imaging from 24–72 h",
            "experiment_type": "imaging",
            "mechanism": "Mitochondrial reserve",
            "target": "mitochondrial membrane potential",
            "readout": "reserve-state persistence and lineage fate",
            "sample_time_hours": 48,
            "relative_cost": 0.45,
            "relative_duration": 0.48,
            "technical_risk": 0.22,
            "measurement_noise": 0.29,
            "model_uncertainty": 0.09,
            "hypothesis_effects": {h1: 0.28, h2: 1.06, h3: 0.08, h4: 0.24},
            "therapeutic_value": 0.34,
            "rationale": "Determines whether mitochondrial reserve rises before persistent tolerance.",
        },
        {
            "experiment_id": "IMG_SPATIAL_AP_TCELL_96",
            "experiment_name": "Spatial antigen-presentation/T-cell imaging at 96 h",
            "experiment_type": "imaging",
            "mechanism": "Antigen presentation",
            "target": "HLA-I/TCR spatial circuit",
            "readout": "immune contact, antigen presentation and exclusion niche",
            "sample_time_hours": 96,
            "relative_cost": 0.60,
            "relative_duration": 0.62,
            "technical_risk": 0.28,
            "measurement_noise": 0.31,
            "model_uncertainty": 0.10,
            "hypothesis_effects": {h1: 0.10, h2: 0.08, h3: 1.12, h4: 0.18},
            "therapeutic_value": 0.42,
            "rationale": "Tests whether spatial immune exclusion is a driver rather than a late correlate.",
        },
        {
            "experiment_id": "SAMPLE_MULTIOME_48H",
            "experiment_name": "Add a 48 h single-cell multiome sample",
            "experiment_type": "sampling_time",
            "mechanism": "Enhancer plasticity",
            "target": "48 h RNA/ATAC state",
            "readout": "transition-state and enhancer commitment",
            "sample_time_hours": 48,
            "relative_cost": 0.50,
            "relative_duration": 0.40,
            "technical_risk": 0.18,
            "measurement_noise": 0.27,
            "model_uncertainty": 0.08,
            "hypothesis_effects": {h1: 0.62, h2: 0.36, h3: 0.18, h4: 1.02},
            "therapeutic_value": 0.25,
            "rationale": "Improves temporal resolution near the predicted reversible-to-persistent transition.",
        },
        {
            "experiment_id": "SAMPLE_PHOSPHO_36H",
            "experiment_name": "Add a 36 h phosphoproteomic sample",
            "experiment_type": "sampling_time",
            "mechanism": "IRE1-XBP1 proteostasis",
            "target": "36 h phospho-IRE1/ATF4 signaling",
            "readout": "early pathway ordering before transcriptional commitment",
            "sample_time_hours": 36,
            "relative_cost": 0.46,
            "relative_duration": 0.36,
            "technical_risk": 0.20,
            "measurement_noise": 0.30,
            "model_uncertainty": 0.09,
            "hypothesis_effects": {h1: 1.00, h2: 0.30, h3: 0.10, h4: 0.60},
            "therapeutic_value": 0.24,
            "rationale": "Orders early signaling events before state commitment.",
        },
        {
            "experiment_id": "SAMPLE_SPATIAL_120H",
            "experiment_name": "Add a 120 h spatial-omics sample",
            "experiment_type": "sampling_time",
            "mechanism": "Antigen presentation",
            "target": "120 h tumor–immune niche",
            "readout": "emergence of immune exclusion before terminal resistance",
            "sample_time_hours": 120,
            "relative_cost": 0.58,
            "relative_duration": 0.54,
            "technical_risk": 0.24,
            "measurement_noise": 0.32,
            "model_uncertainty": 0.10,
            "hypothesis_effects": {h1: 0.16, h2: 0.12, h3: 1.04, h4: 0.28},
            "therapeutic_value": 0.22,
            "rationale": "Captures spatial niche assembly before the terminal resistance endpoint.",
        },
    ]


def _hypothesis_frame(payload: Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    rows = [dict(item) for item in (payload or _default_hypotheses())]
    frame = pd.DataFrame(rows)
    required = {"hypothesis_id", "hypothesis", "mechanism", "prior_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hypothesis catalog missing columns: {sorted(missing)}")
    frame["prior_probability"] = pd.to_numeric(frame["prior_probability"], errors="raise")
    if (frame["prior_probability"] < 0).any() or frame["prior_probability"].sum() <= 0:
        raise ValueError("hypothesis prior probabilities must be nonnegative and sum to >0")
    frame["prior_probability"] /= frame["prior_probability"].sum()
    if frame["hypothesis_id"].duplicated().any():
        raise ValueError("hypothesis IDs must be unique")
    return frame.reset_index(drop=True)


def _catalog_frame(
    hypotheses: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]] | None,
) -> pd.DataFrame:
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    rows = [dict(item) for item in (candidates or _default_candidates(hypothesis_ids))]
    output: list[dict[str, Any]] = []
    for row in rows:
        experiment_type = str(row.get("experiment_type", "")).lower()
        if experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"unsupported experiment type: {experiment_type}")
        effects = row.get("hypothesis_effects", {})
        if isinstance(effects, str):
            effects = json.loads(effects)
        missing = set(hypothesis_ids) - set(effects)
        if missing:
            raise ValueError(f"{row.get('experiment_id')} missing effects for {sorted(missing)}")
        item = {
            "experiment_id": str(row["experiment_id"]),
            "experiment_name": str(row["experiment_name"]),
            "experiment_type": experiment_type,
            "mechanism": str(row.get("mechanism", "unspecified")),
            "target": str(row.get("target", "")),
            "readout": str(row.get("readout", "")),
            "sample_time_hours": float(row.get("sample_time_hours", 72.0)),
            "relative_cost": float(row.get("relative_cost", 0.5)),
            "relative_duration": float(row.get("relative_duration", 0.5)),
            "technical_risk": float(row.get("technical_risk", 0.25)),
            "measurement_noise": max(float(row.get("measurement_noise", 0.35)), 0.05),
            "model_uncertainty": max(float(row.get("model_uncertainty", 0.10)), 0.0),
            "therapeutic_value_prior": float(row.get("therapeutic_value", 0.3)),
            "rationale": str(row.get("rationale", "Discriminate among candidate mechanisms.")),
            "hypothesis_effects_json": json.dumps({key: float(effects[key]) for key in hypothesis_ids}, sort_keys=True),
        }
        for hypothesis_id in hypothesis_ids:
            item[f"expected_readout__{hypothesis_id}"] = float(effects[hypothesis_id])
        output.append(item)
    frame = pd.DataFrame(output)
    if frame["experiment_id"].duplicated().any():
        raise ValueError("experiment IDs must be unique")
    return frame


def _mechanism_biomarker_values(biomarkers: pd.DataFrame) -> dict[str, float]:
    if biomarkers.empty:
        return {}
    score_col = "uncertainty_adjusted_score" if "uncertainty_adjusted_score" in biomarkers else "causal_biomarker_score"
    mapping: dict[str, float] = {}
    for _, row in biomarkers.iterrows():
        marker = str(row.get("biomarker", "")).lower()
        mechanism = ""
        if "ire1" in marker or "proteostasis" in marker:
            mechanism = "IRE1-XBP1 proteostasis"
        elif "mitochond" in marker:
            mechanism = "Mitochondrial reserve"
        elif "antigen" in marker or "immune_exclusion" in marker or "inflammatory" in marker:
            mechanism = "Antigen presentation"
        elif "enhancer" in marker:
            mechanism = "Enhancer plasticity"
        if mechanism:
            mapping[mechanism] = max(mapping.get(mechanism, 0.0), float(row.get(score_col, 0.0)))
    if mapping:
        values = np.asarray(list(mapping.values()), dtype=float)
        lo, hi = float(values.min()), float(values.max())
        if hi > lo:
            mapping = {key: (value - lo) / (hi - lo) for key, value in mapping.items()}
        else:
            mapping = {key: 0.5 for key in mapping}
    return mapping


def _therapeutic_lookup(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {}
    value_col = "uncertainty_adjusted_utility" if "uncertainty_adjusted_utility" in predictions else "resistance_risk_reduction"
    output: dict[str, float] = {}
    for mechanism, tokens in {
        "IRE1-XBP1 proteostasis": ("ire1", "xbp1"),
        "Mitochondrial reserve": ("mito", "ndufs"),
        "Antigen presentation": ("ifng", "antigen", "b2m", "tap1"),
        "Enhancer plasticity": ("ep300", "enhancer"),
        "Proteostasis-mitochondrial interaction": ("ire1", "mito"),
    }.items():
        matches = predictions.loc[
            predictions["regimen_name"].astype(str).str.lower().apply(lambda text: any(token in text for token in tokens))
        ]
        if not matches.empty:
            output[mechanism] = float(pd.to_numeric(matches[value_col], errors="coerce").max())
    if output:
        vals = np.asarray(list(output.values()), dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        output = {k: (v - lo) / (hi - lo) if hi > lo else 0.5 for k, v in output.items()}
    return output


def _temporal_value(candidate: pd.Series, transition_uncertainty: pd.DataFrame, biomarker_timecourse: pd.DataFrame) -> float:
    value = 0.35
    experiment_type = str(candidate["experiment_type"])
    sample_time = float(candidate["sample_time_hours"])
    if experiment_type == "sampling_time":
        # Reward samples falling in gaps between observed times and near uncertain transitions.
        observed_times = sorted(pd.to_numeric(biomarker_timecourse.get("time_hours", pd.Series(dtype=float)), errors="coerce").dropna().unique())
        if observed_times:
            nearest = min(abs(sample_time - float(t)) for t in observed_times)
            value += min(0.35, nearest / max(max(observed_times), 1.0))
        if not transition_uncertainty.empty and "ci_high" in transition_uncertainty and "ci_low" in transition_uncertainty:
            width = pd.to_numeric(transition_uncertainty["ci_high"], errors="coerce") - pd.to_numeric(transition_uncertainty["ci_low"], errors="coerce")
            value += min(0.30, float(width.fillna(0).mean()))
    elif experiment_type == "imaging":
        value += 0.20
    return float(np.clip(value, 0.0, 1.0))


def score_experiments(
    catalog: pd.DataFrame,
    hypotheses: pd.DataFrame,
    prior: np.ndarray,
    config: ClosedLoopConfig,
    therapeutic_predictions: pd.DataFrame | None = None,
    biomarkers: pd.DataFrame | None = None,
    biomarker_timecourse: pd.DataFrame | None = None,
    transition_uncertainty: pd.DataFrame | None = None,
    excluded_ids: Iterable[str] = (),
    round_number: int = 1,
) -> pd.DataFrame:
    therapeutic_predictions = therapeutic_predictions if therapeutic_predictions is not None else pd.DataFrame()
    biomarkers = biomarkers if biomarkers is not None else pd.DataFrame()
    biomarker_timecourse = biomarker_timecourse if biomarker_timecourse is not None else pd.DataFrame()
    transition_uncertainty = transition_uncertainty if transition_uncertainty is not None else pd.DataFrame()
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    max_entropy = max(_entropy(np.full(len(prior), 1.0 / len(prior))), 1e-9)
    biomarker_values = _mechanism_biomarker_values(biomarkers)
    therapeutic_values = _therapeutic_lookup(therapeutic_predictions)
    excluded = set(excluded_ids)
    rows: list[dict[str, Any]] = []
    for _, candidate in catalog.iterrows():
        if str(candidate["experiment_id"]) in excluded:
            continue
        means = candidate[[f"expected_readout__{hid}" for hid in hypothesis_ids]].to_numpy(dtype=float)
        eig = _expected_information_gain(
            prior,
            means,
            float(candidate["measurement_noise"]),
            config.eig_samples,
            _stable_seed(str(candidate["experiment_id"]) + f":round{round_number}", config.seed),
        )
        eig_fraction = float(np.clip(eig / max_entropy, 0.0, 1.0))
        mechanism = str(candidate["mechanism"])
        therapeutic_value = therapeutic_values.get(mechanism, float(candidate["therapeutic_value_prior"]))
        biomarker_value = biomarker_values.get(mechanism, 0.35)
        temporal_value = _temporal_value(candidate, transition_uncertainty, biomarker_timecourse)
        feasibility = float(np.clip(
            1.0
            - 0.42 * float(candidate["relative_cost"])
            - 0.26 * float(candidate["relative_duration"])
            - 0.32 * float(candidate["technical_risk"]),
            0.0,
            1.0,
        ))
        priority = (
            config.information_gain_weight * eig_fraction
            + config.therapeutic_value_weight * therapeutic_value
            + config.biomarker_value_weight * biomarker_value
            + config.temporal_value_weight * temporal_value
            + config.feasibility_weight * feasibility
        )
        dominant_index = int(np.argmax(np.abs(means - np.average(means, weights=prior))))
        row = candidate.to_dict()
        row.update(
            {
                "round": round_number,
                "prior_entropy_nats": _entropy(prior),
                "expected_information_gain_nats": eig,
                "expected_information_gain_fraction": eig_fraction,
                "therapeutic_value": therapeutic_value,
                "biomarker_value": biomarker_value,
                "temporal_value": temporal_value,
                "feasibility": feasibility,
                "priority_score": float(priority),
                "most_discriminated_hypothesis": hypothesis_ids[dominant_index],
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["priority_score", "expected_information_gain_nats"], ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def select_batch(
    ranking: pd.DataFrame,
    budget: float,
    batch_size: int,
    max_per_type: int,
    diversity_penalty: float,
    require_type_coverage: bool = True,
) -> pd.DataFrame:
    if ranking.empty or batch_size <= 0 or budget <= 0:
        return ranking.head(0).copy()
    selected_rows: list[pd.Series] = []
    selected_ids: set[str] = set()
    spent = 0.0
    type_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}

    # When a batch can contain all supported experiment classes, choose the best
    # feasible one-per-type portfolio first. This prevents a high-scoring single
    # mechanism from consuming the entire first round and makes the loop genuinely
    # multimethod rather than merely having a diverse catalog.
    available_types = [kind for kind in EXPERIMENT_TYPES if kind in set(ranking["experiment_type"])]
    if require_type_coverage and batch_size >= len(available_types) and available_types:
        pools = [list(ranking.loc[ranking["experiment_type"] == kind].iterrows()) for kind in available_types]
        best_combo: tuple[float, tuple[tuple[int, pd.Series], ...]] | None = None
        for combo in itertools.product(*pools):
            rows = [item[1] for item in combo]
            cost = sum(float(row["relative_cost"]) for row in rows)
            if cost > budget + 1e-12:
                continue
            mechanism_counts_combo: dict[str, int] = {}
            score = 0.0
            for row in rows:
                mechanism = str(row["mechanism"])
                score += float(row["priority_score"]) / max(float(row["relative_cost"]), 0.12)
                score -= diversity_penalty * mechanism_counts_combo.get(mechanism, 0)
                mechanism_counts_combo[mechanism] = mechanism_counts_combo.get(mechanism, 0) + 1
            if best_combo is None or score > best_combo[0]:
                best_combo = (score, combo)
        if best_combo is not None:
            for _, row in best_combo[1]:
                row = row.copy()
                row["marginal_priority_per_cost"] = float(row["priority_score"]) / max(float(row["relative_cost"]), 0.12)
                spent += float(row["relative_cost"])
                row["cumulative_cost"] = spent
                selected_rows.append(row)
                selected_ids.add(str(row["experiment_id"]))
                experiment_type = str(row["experiment_type"])
                mechanism = str(row["mechanism"])
                type_counts[experiment_type] = type_counts.get(experiment_type, 0) + 1
                mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1

    while len(selected_rows) < batch_size:
        best: tuple[float, pd.Series] | None = None
        for _, row in ranking.iterrows():
            experiment_id = str(row["experiment_id"])
            if experiment_id in selected_ids:
                continue
            cost = float(row["relative_cost"])
            experiment_type = str(row["experiment_type"])
            mechanism = str(row["mechanism"])
            if spent + cost > budget + 1e-12:
                continue
            if type_counts.get(experiment_type, 0) >= max_per_type:
                continue
            marginal = float(row["priority_score"])
            marginal -= diversity_penalty * mechanism_counts.get(mechanism, 0)
            marginal -= 0.5 * diversity_penalty * type_counts.get(experiment_type, 0)
            marginal /= max(cost, 0.12)
            if best is None or marginal > best[0]:
                best = (marginal, row)
        if best is None:
            break
        row = best[1].copy()
        row["marginal_priority_per_cost"] = best[0]
        row["cumulative_cost"] = spent + float(row["relative_cost"])
        selected_rows.append(row)
        selected_ids.add(str(row["experiment_id"]))
        spent += float(row["relative_cost"])
        type_counts[str(row["experiment_type"])] = type_counts.get(str(row["experiment_type"]), 0) + 1
        mechanism_counts[str(row["mechanism"])] = mechanism_counts.get(str(row["mechanism"]), 0) + 1
    if not selected_rows:
        return ranking.head(0).copy()
    output = pd.DataFrame(selected_rows).reset_index(drop=True)
    output.insert(0, "batch_position", np.arange(1, len(output) + 1))
    return output


def _simulate_and_update(
    batch: pd.DataFrame,
    hypotheses: pd.DataFrame,
    prior: np.ndarray,
    true_hypothesis: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    if true_hypothesis not in hypothesis_ids:
        raise ValueError(f"true_hypothesis {true_hypothesis!r} is not in the hypothesis catalog")
    true_index = hypothesis_ids.index(true_hypothesis)
    posterior = np.asarray(prior, dtype=float).copy()
    history = [{"update_step": 0, "experiment_id": "PRIOR", **{hid: posterior[i] for i, hid in enumerate(hypothesis_ids)}, "entropy_nats": _entropy(posterior)}]
    observations: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for step, (_, row) in enumerate(batch.iterrows(), 1):
        means = row[[f"expected_readout__{hid}" for hid in hypothesis_ids]].to_numpy(dtype=float)
        sigma = float(row["measurement_noise"])
        observed = float(rng.normal(means[true_index], sigma))
        prior_before = posterior.copy()
        posterior = _posterior_update(posterior, observed, means, sigma)
        observations.append(
            {
                "update_step": step,
                "experiment_id": row["experiment_id"],
                "experiment_name": row["experiment_name"],
                "experiment_type": row["experiment_type"],
                "mechanism": row["mechanism"],
                "observed_standardized_readout": observed,
                "measurement_noise": sigma,
                "synthetic_truth_hypothesis": true_hypothesis,
                "prior_entropy_nats": _entropy(prior_before),
                "posterior_entropy_nats": _entropy(posterior),
                "realized_information_gain_nats": _entropy(prior_before) - _entropy(posterior),
                "synthetic_demonstration": True,
            }
        )
        history.append(
            {
                "update_step": step,
                "experiment_id": row["experiment_id"],
                **{hid: posterior[i] for i, hid in enumerate(hypothesis_ids)},
                "entropy_nats": _entropy(posterior),
            }
        )
    return pd.DataFrame(observations), pd.DataFrame(history), posterior


def _bootstrap_scores(
    catalog: pd.DataFrame,
    hypotheses: pd.DataFrame,
    prior: np.ndarray,
    config: ClosedLoopConfig,
    base_ranking: pd.DataFrame,
) -> pd.DataFrame:
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    max_entropy = max(_entropy(np.full(len(prior), 1.0 / len(prior))), 1e-9)
    base_lookup = base_ranking.set_index("experiment_id")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 811)
    for replicate in range(config.bootstrap):
        perturbed_prior = rng.dirichlet(np.clip(prior, 1e-4, None) * 80.0)
        for _, candidate in catalog.iterrows():
            means = candidate[[f"expected_readout__{hid}" for hid in hypothesis_ids]].to_numpy(dtype=float)
            means = means + rng.normal(0.0, float(candidate["model_uncertainty"]), size=len(means))
            eig = _expected_information_gain(
                perturbed_prior,
                means,
                float(candidate["measurement_noise"]),
                max(300, config.eig_samples // 3),
                int(rng.integers(0, 2**31 - 1)),
            )
            eig_fraction = float(np.clip(eig / max_entropy, 0.0, 1.0))
            base = base_lookup.loc[str(candidate["experiment_id"])]
            priority = (
                config.information_gain_weight * eig_fraction
                + config.therapeutic_value_weight * float(base["therapeutic_value"])
                + config.biomarker_value_weight * float(base["biomarker_value"])
                + config.temporal_value_weight * float(base["temporal_value"])
                + config.feasibility_weight * float(base["feasibility"])
            )
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "experiment_id": candidate["experiment_id"],
                    "expected_information_gain_nats": eig,
                    "priority_score": priority,
                }
            )
    distributions = pd.DataFrame(rows)
    if distributions.empty:
        return distributions
    ranks = distributions.groupby("bootstrap_replicate")["priority_score"].rank(method="min", ascending=False)
    distributions["bootstrap_rank"] = ranks
    return distributions


def _attach_bootstrap_intervals(ranking: pd.DataFrame, distributions: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    if ranking.empty or distributions.empty:
        return ranking
    summaries = []
    for experiment_id, group in distributions.groupby("experiment_id", sort=False):
        summaries.append(
            {
                "experiment_id": experiment_id,
                "eig_bootstrap_mean": float(group["expected_information_gain_nats"].mean()),
                "eig_ci_low": float(group["expected_information_gain_nats"].quantile(0.025)),
                "eig_ci_high": float(group["expected_information_gain_nats"].quantile(0.975)),
                "priority_ci_low": float(group["priority_score"].quantile(0.025)),
                "priority_ci_high": float(group["priority_score"].quantile(0.975)),
                "bootstrap_batch_selection_probability": float((group["bootstrap_rank"] <= batch_size).mean()),
            }
        )
    return ranking.merge(pd.DataFrame(summaries), on="experiment_id", how="left")


def _outcome_templates(batch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in batch.iterrows():
        experiment_type = str(row["experiment_type"])
        if experiment_type == "crispr":
            control = "non-targeting guide; matched editing/QC control"
            primary = "target engagement plus resistant-state probability"
            minimum_replicates = 3
        elif experiment_type == "drug":
            control = "vehicle and standard-therapy comparator"
            primary = "dose/time response, resistance reduction and normal-cell toxicity"
            minimum_replicates = 4
        elif experiment_type == "imaging":
            control = "matched untreated field and reporter-negative control"
            primary = "single-cell trajectory or spatial-circuit readout"
            minimum_replicates = 3
        else:
            control = "adjacent measured timepoints and batch-matched reference"
            primary = "transition uncertainty and mechanism-specific state readout"
            minimum_replicates = 3
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "experiment_name": row["experiment_name"],
                "experiment_type": experiment_type,
                "target": row["target"],
                "planned_sample_time_hours": row["sample_time_hours"],
                "primary_readout": primary,
                "mechanistic_readout": row["readout"],
                "required_control": control,
                "minimum_biological_replicates": minimum_replicates,
                "result_field": "observed_standardized_readout",
                "uncertainty_field": "standard_error_or_posterior_sd",
                "quality_fields": "target_engagement;batch_qc;missingness;cell_count",
                "decision_rule": "update all hypothesis probabilities; do not accept a mechanism from a single experiment",
            }
        )
    return pd.DataFrame(rows)


def run_closed_loop_experimentation(
    hypotheses_payload: Sequence[Mapping[str, Any]] | None = None,
    candidates_payload: Sequence[Mapping[str, Any]] | None = None,
    config: ClosedLoopConfig | None = None,
    therapeutic_predictions: pd.DataFrame | None = None,
    biomarkers: pd.DataFrame | None = None,
    biomarker_timecourse: pd.DataFrame | None = None,
    transition_uncertainty: pd.DataFrame | None = None,
) -> ClosedLoopResult:
    config = config or ClosedLoopConfig()
    hypotheses = _hypothesis_frame(hypotheses_payload)
    catalog = _catalog_frame(hypotheses, candidates_payload)
    prior = hypotheses["prior_probability"].to_numpy(dtype=float)
    round1 = score_experiments(
        catalog,
        hypotheses,
        prior,
        config,
        therapeutic_predictions=therapeutic_predictions,
        biomarkers=biomarkers,
        biomarker_timecourse=biomarker_timecourse,
        transition_uncertainty=transition_uncertainty,
        round_number=1,
    )
    bootstrap = _bootstrap_scores(catalog, hypotheses, prior, config, round1)
    round1 = _attach_bootstrap_intervals(round1, bootstrap, config.batch_size)
    batch1 = select_batch(round1, config.budget, config.batch_size, config.max_per_type, config.diversity_penalty, config.require_type_coverage)

    true_hypothesis = config.true_hypothesis or str(hypotheses.iloc[0]["hypothesis_id"])
    if config.simulate_demonstration_round and not batch1.empty:
        observations, posterior_history, posterior = _simulate_and_update(
            batch1, hypotheses, prior, true_hypothesis, config.seed + 99
        )
    else:
        observations = pd.DataFrame(columns=["experiment_id", "observed_standardized_readout"])
        posterior_history = pd.DataFrame([
            {"update_step": 0, "experiment_id": "PRIOR", **{
                hid: prior[i] for i, hid in enumerate(hypotheses["hypothesis_id"].astype(str))
            }, "entropy_nats": _entropy(prior)}
        ])
        posterior = prior.copy()

    round2 = score_experiments(
        catalog,
        hypotheses,
        posterior,
        config,
        therapeutic_predictions=therapeutic_predictions,
        biomarkers=biomarkers,
        biomarker_timecourse=biomarker_timecourse,
        transition_uncertainty=transition_uncertainty,
        excluded_ids=batch1["experiment_id"].astype(str).tolist() if not batch1.empty else (),
        round_number=2,
    )
    batch2 = select_batch(round2, config.round2_budget, config.round2_batch_size, config.max_per_type, config.diversity_penalty, config.require_type_coverage)
    templates = _outcome_templates(pd.concat([batch1, batch2], ignore_index=True).drop_duplicates("experiment_id"))
    final_entropy = _entropy(posterior)
    qc = {
        "framework": "CausaFlux",
        "version": "1.7.0",
        "n_hypotheses": int(len(hypotheses)),
        "n_candidates": int(len(catalog)),
        "candidate_types": sorted(catalog["experiment_type"].unique().tolist()),
        "round1_batch_size": int(len(batch1)),
        "round1_cost": float(batch1["relative_cost"].sum()) if not batch1.empty else 0.0,
        "round1_budget": float(config.budget),
        "round2_batch_size": int(len(batch2)),
        "round2_cost": float(batch2["relative_cost"].sum()) if not batch2.empty else 0.0,
        "round2_budget": float(config.round2_budget),
        "bootstrap_requested": int(config.bootstrap),
        "bootstrap_completed": int(bootstrap["bootstrap_replicate"].nunique()) if not bootstrap.empty else 0,
        "initial_entropy_nats": _entropy(prior),
        "posterior_entropy_nats": final_entropy,
        "demonstration_information_gain_nats": _entropy(prior) - final_entropy,
        "top_round1_experiment": str(round1.iloc[0]["experiment_name"]) if not round1.empty else None,
        "top_round2_experiment": str(round2.iloc[0]["experiment_name"]) if not round2.empty else None,
        "selected_round1_experiments": batch1["experiment_id"].astype(str).tolist() if not batch1.empty else [],
        "selected_round2_experiments": batch2["experiment_id"].astype(str).tolist() if not batch2.empty else [],
        "synthetic_demonstration": bool(config.simulate_demonstration_round),
        "synthetic_truth_hypothesis": true_hypothesis if config.simulate_demonstration_round else None,
    }
    return ClosedLoopResult(
        hypotheses=hypotheses,
        catalog=catalog,
        round1_ranking=round1,
        round1_batch=batch1,
        simulated_observations=observations,
        posterior_history=posterior_history,
        round2_ranking=round2,
        round2_batch=batch2,
        outcome_templates=templates,
        bootstrap_distributions=bootstrap,
        qc=qc,
    )


def plot_experiment_ranking(ranking: pd.DataFrame, output_path: str | Path, top_n: int = 12) -> Path:
    output_path = Path(output_path)
    selected = ranking.nsmallest(top_n, "rank").sort_values("priority_score") if not ranking.empty else ranking
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    if not selected.empty:
        ax.barh(selected["experiment_name"], selected["priority_score"])
    ax.set_xlabel("Closed-loop priority score")
    ax.set_title("Round 1 experiment ranking")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_information_gain_by_type(ranking: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    summary = ranking.groupby("experiment_type", as_index=False)["expected_information_gain_nats"].mean() if not ranking.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    if not summary.empty:
        ax.bar(summary["experiment_type"], summary["expected_information_gain_nats"])
    ax.set_ylabel("Mean expected information gain (nats)")
    ax.set_title("Information value by experiment type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_posterior_update(history: pd.DataFrame, hypotheses: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for _, hypothesis in hypotheses.iterrows():
        hypothesis_id = str(hypothesis["hypothesis_id"])
        if hypothesis_id in history:
            ax.plot(history["update_step"], history[hypothesis_id], marker="o", label=hypothesis_id)
    ax.set_xlabel("Sequential experiment update")
    ax.set_ylabel("Hypothesis probability")
    ax.set_ylim(0, 1)
    ax.set_title("Closed-loop posterior update")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_batch_portfolio(batch: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    summary = batch.groupby("experiment_type", as_index=False).agg(
        experiments=("experiment_id", "count"), cost=("relative_cost", "sum")
    ) if not batch.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if not summary.empty:
        ax.bar(summary["experiment_type"], summary["cost"])
        for index, row in summary.iterrows():
            ax.text(index, row["cost"], f"n={int(row['experiments'])}", ha="center", va="bottom")
    ax.set_ylabel("Relative batch cost")
    ax.set_title("Selected round 1 portfolio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_sampling_time_recommendations(ranking: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    sampling = ranking.loc[ranking["experiment_type"] == "sampling_time"].sort_values("sample_time_hours") if not ranking.empty else ranking
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if not sampling.empty:
        ax.scatter(sampling["sample_time_hours"], sampling["expected_information_gain_nats"], s=90)
        for _, row in sampling.iterrows():
            ax.annotate(str(row["experiment_name"]).replace("Add a ", ""), (row["sample_time_hours"], row["expected_information_gain_nats"]), fontsize=8, xytext=(3, 5), textcoords="offset points")
    ax.set_xlabel("Recommended sampling time (hours)")
    ax.set_ylabel("Expected information gain (nats)")
    ax.set_title("Sampling-time design")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_closed_loop_outputs(result: ClosedLoopResult, output_dir: str | Path, write_plots: bool = True) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    paths = {
        "hypotheses": output / "hypothesis_priors.csv",
        "catalog": output / "experiment_catalog.csv",
        "round1_ranking": output / "round1_experiment_recommendations.csv",
        "round1_batch": output / "round1_selected_batch.csv",
        "observations": output / "synthetic_round1_observations.csv",
        "posterior": output / "hypothesis_posterior_history.csv",
        "round2_ranking": output / "round2_experiment_recommendations.csv",
        "round2_batch": output / "round2_selected_batch.csv",
        "templates": output / "experiment_outcome_templates.csv",
        "bootstrap": output / "experiment_bootstrap_distributions.csv",
        "qc": output / "closed_loop_qc.json",
    }
    result.hypotheses.to_csv(paths["hypotheses"], index=False)
    result.catalog.to_csv(paths["catalog"], index=False)
    result.round1_ranking.to_csv(paths["round1_ranking"], index=False)
    # Compatibility alias for earlier active-learning output.
    result.round1_ranking.to_csv(output / "experiment_recommendations.csv", index=False)
    result.round1_batch.to_csv(paths["round1_batch"], index=False)
    result.simulated_observations.to_csv(paths["observations"], index=False)
    result.posterior_history.to_csv(paths["posterior"], index=False)
    result.round2_ranking.to_csv(paths["round2_ranking"], index=False)
    result.round2_batch.to_csv(paths["round2_batch"], index=False)
    result.outcome_templates.to_csv(paths["templates"], index=False)
    result.bootstrap_distributions.to_csv(paths["bootstrap"], index=False)
    json_dump(result.qc, paths["qc"])
    if write_plots:
        paths.update(
            {
                "ranking_plot": plot_experiment_ranking(result.round1_ranking, output / "experiment_priority_ranking.png"),
                "type_plot": plot_information_gain_by_type(result.round1_ranking, output / "information_gain_by_type.png"),
                "posterior_plot": plot_posterior_update(result.posterior_history, result.hypotheses, output / "hypothesis_posterior_update.png"),
                "batch_plot": plot_batch_portfolio(result.round1_batch, output / "batch_portfolio.png"),
                "sampling_plot": plot_sampling_time_recommendations(result.round1_ranking, output / "sampling_time_recommendations.png"),
            }
        )
    return paths


def validate_closed_loop_outputs(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    required = [
        "hypothesis_priors.csv",
        "experiment_catalog.csv",
        "round1_experiment_recommendations.csv",
        "round1_selected_batch.csv",
        "hypothesis_posterior_history.csv",
        "round2_experiment_recommendations.csv",
        "experiment_outcome_templates.csv",
        "closed_loop_qc.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise ValueError(f"missing closed-loop outputs: {missing}")
    hypotheses = pd.read_csv(output / "hypothesis_priors.csv")
    catalog = pd.read_csv(output / "experiment_catalog.csv")
    ranking = pd.read_csv(output / "round1_experiment_recommendations.csv")
    batch = pd.read_csv(output / "round1_selected_batch.csv")
    posterior = pd.read_csv(output / "hypothesis_posterior_history.csv")
    qc = json.loads((output / "closed_loop_qc.json").read_text(encoding="utf-8"))
    if set(catalog["experiment_type"]) != set(EXPERIMENT_TYPES):
        raise ValueError("experiment catalog must contain CRISPR, drug, imaging and sampling-time candidates")
    if ranking["priority_score"].isna().any() or ranking["expected_information_gain_nats"].isna().any():
        raise ValueError("closed-loop ranking contains missing scores")
    if (batch["relative_cost"].sum() > float(qc["round1_budget"]) + 1e-8):
        raise ValueError("selected batch exceeds the configured budget")
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    sums = posterior[hypothesis_ids].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("posterior probabilities do not sum to one")
    if (ranking["eig_ci_low"] > ranking["eig_ci_high"]).any():
        raise ValueError("information-gain intervals are not ordered")
    return {
        "valid": True,
        "n_hypotheses": int(len(hypotheses)),
        "n_candidates": int(len(catalog)),
        "n_selected_round1": int(len(batch)),
        "candidate_types": sorted(catalog["experiment_type"].unique().tolist()),
        "posterior_updates": int(len(posterior) - 1),
        "round1_cost": float(batch["relative_cost"].sum()),
    }


def update_closed_loop_from_observations(
    hypotheses: pd.DataFrame,
    catalog: pd.DataFrame,
    observations: pd.DataFrame,
    config: ClosedLoopConfig | None = None,
    therapeutic_predictions: pd.DataFrame | None = None,
    biomarkers: pd.DataFrame | None = None,
    biomarker_timecourse: pd.DataFrame | None = None,
    transition_uncertainty: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Update hypothesis probabilities from completed experiments and rank the next batch.

    ``observations`` must contain ``experiment_id`` and
    ``observed_standardized_readout``. An optional ``standard_error_or_posterior_sd``
    column overrides the catalog measurement noise for the corresponding result.
    """
    config = config or ClosedLoopConfig(simulate_demonstration_round=False)
    hypothesis_ids = hypotheses["hypothesis_id"].astype(str).tolist()
    prior = pd.to_numeric(hypotheses["prior_probability"], errors="raise").to_numpy(dtype=float)
    prior = prior / prior.sum()
    required = {"experiment_id", "observed_standardized_readout"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observation table missing columns: {sorted(missing)}")
    lookup = catalog.set_index("experiment_id", drop=False)
    posterior = prior.copy()
    history = [{"update_step": 0, "experiment_id": "PRIOR", **{hid: posterior[i] for i, hid in enumerate(hypothesis_ids)}, "entropy_nats": _entropy(posterior)}]
    completed: list[str] = []
    for step, (_, observation) in enumerate(observations.iterrows(), 1):
        experiment_id = str(observation["experiment_id"])
        if experiment_id not in lookup.index:
            raise ValueError(f"observation references unknown experiment: {experiment_id}")
        row = lookup.loc[experiment_id]
        means = row[[f"expected_readout__{hid}" for hid in hypothesis_ids]].to_numpy(dtype=float)
        sigma_value = observation.get("standard_error_or_posterior_sd", np.nan)
        sigma = float(sigma_value) if pd.notna(sigma_value) else float(row["measurement_noise"])
        posterior = _posterior_update(
            posterior,
            float(observation["observed_standardized_readout"]),
            means,
            sigma,
        )
        completed.append(experiment_id)
        history.append({"update_step": step, "experiment_id": experiment_id, **{hid: posterior[i] for i, hid in enumerate(hypothesis_ids)}, "entropy_nats": _entropy(posterior)})
    ranking = score_experiments(
        catalog,
        hypotheses,
        posterior,
        config,
        therapeutic_predictions=therapeutic_predictions,
        biomarkers=biomarkers,
        biomarker_timecourse=biomarker_timecourse,
        transition_uncertainty=transition_uncertainty,
        excluded_ids=completed,
        round_number=len(observations) + 1,
    )
    batch = select_batch(ranking, config.round2_budget, config.round2_batch_size, config.max_per_type, config.diversity_penalty, config.require_type_coverage)
    return pd.DataFrame(history), ranking, batch
