from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, json_dump


THERAPEUTIC_FEATURES: tuple[str, ...] = (
    "time_hours",
    "mutation_burden",
    "treatment_stress",
    "ire1_xbp1",
    "proteostasis_capacity",
    "enhancer_plasticity",
    "mitochondrial_reserve",
    "antigen_presentation",
    "immune_exclusion",
    "inflammatory_signaling",
    "viability",
    "apoptosis_signal",
)

STATE_EFFECT_FEATURES: tuple[str, ...] = (
    "treatment_stress",
    "ire1_xbp1",
    "proteostasis_capacity",
    "enhancer_plasticity",
    "mitochondrial_reserve",
    "antigen_presentation",
    "immune_exclusion",
    "inflammatory_signaling",
    "viability",
    "apoptosis_signal",
)

EFFECT_COLUMNS: tuple[str, ...] = tuple(f"delta_{name}" for name in STATE_EFFECT_FEATURES)


@dataclass(frozen=True)
class TherapeuticConfig:
    comparator: str = "standard_therapy"
    decision_time_hours: float = 24.0
    horizon_hours: float = 168.0
    timing_grid: tuple[float, ...] = (0.0, 24.0, 48.0, 72.0, 120.0)
    default_start_hour: float = 24.0
    sequence_delay_hours: float = 24.0
    bootstrap: int = 30
    max_reference_rows_per_donor: int = 30
    seed: int = 31
    uncertainty_penalty: float = 0.12
    normal_toxicity_weight: float = 0.30


@dataclass
class TherapeuticModel:
    pipeline: Pipeline
    feature_names: list[str]
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    metrics: dict[str, Any]

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, self.feature_names].astype(float)
        return self.pipeline.predict_proba(matrix)[:, 1]


@dataclass
class TherapeuticResult:
    intervention_catalog: pd.DataFrame
    regimen_catalog: pd.DataFrame
    predictions: pd.DataFrame
    bootstrap_intervals: pd.DataFrame
    state_changes: pd.DataFrame
    model_metrics: dict[str, Any]
    qc: dict[str, Any]


def _default_interventions() -> list[dict[str, Any]]:
    zero = {name: 0.0 for name in EFFECT_COLUMNS}

    def row(
        intervention_id: str,
        name: str,
        intervention_type: str,
        target: str,
        direction: str,
        mechanism: str,
        potency: float,
        tumor_selectivity: float,
        normal_toxicity: float,
        onset_hours: float,
        half_life_hours: float,
        default_duration_hours: float,
        optimal_time_hours: float,
        timing_width_hours: float,
        effects: dict[str, float],
        rationale: str,
    ) -> dict[str, Any]:
        payload = {
            "intervention_id": intervention_id,
            "intervention_name": name,
            "intervention_type": intervention_type,
            "target": target,
            "direction": direction,
            "mechanism": mechanism,
            "potency": potency,
            "tumor_selectivity": tumor_selectivity,
            "normal_toxicity": normal_toxicity,
            "onset_hours": onset_hours,
            "half_life_hours": half_life_hours,
            "default_duration_hours": default_duration_hours,
            "optimal_time_hours": optimal_time_hours,
            "timing_width_hours": timing_width_hours,
            "rationale": rationale,
            **zero,
        }
        payload.update({f"delta_{key}": value for key, value in effects.items()})
        return payload

    return [
        row(
            "XBP1_CRISPRI",
            "XBP1 CRISPRi",
            "gene",
            "XBP1",
            "inhibit",
            "IRE1-XBP1 proteostasis",
            0.92,
            0.74,
            0.20,
            18.0,
            144.0,
            168.0,
            24.0,
            42.0,
            {
                "ire1_xbp1": -0.46,
                "proteostasis_capacity": -0.28,
                "enhancer_plasticity": -0.10,
                "antigen_presentation": 0.08,
                "viability": -0.07,
                "apoptosis_signal": 0.10,
            },
            "Tests whether the IRE1-XBP1 adaptive state is required for persistent tolerance.",
        ),
        row(
            "ATF4_CRISPRI",
            "ATF4 CRISPRi",
            "gene",
            "ATF4",
            "inhibit",
            "Integrated stress response",
            0.84,
            0.66,
            0.29,
            20.0,
            132.0,
            168.0,
            36.0,
            48.0,
            {
                "proteostasis_capacity": -0.18,
                "enhancer_plasticity": -0.24,
                "inflammatory_signaling": -0.07,
                "viability": -0.09,
                "apoptosis_signal": 0.12,
            },
            "Perturbs a parallel stress-adaptation route that can compensate for IRE1-XBP1 loss.",
        ),
        row(
            "HLA1_CRISPRA",
            "HLA-I CRISPRa",
            "gene",
            "HLA-I antigen-processing module",
            "activate",
            "Antigen presentation",
            0.86,
            0.81,
            0.08,
            16.0,
            120.0,
            168.0,
            72.0,
            50.0,
            {
                "antigen_presentation": 0.42,
                "immune_exclusion": -0.24,
                "inflammatory_signaling": 0.04,
                "apoptosis_signal": 0.04,
            },
            "Restores tumor visibility and tests whether immune exclusion is causally reversible.",
        ),
        row(
            "PGC1A_CRISPRI",
            "PGC1A CRISPRi",
            "gene",
            "PPARGC1A",
            "inhibit",
            "Mitochondrial reserve",
            0.82,
            0.63,
            0.27,
            20.0,
            132.0,
            168.0,
            48.0,
            48.0,
            {
                "mitochondrial_reserve": -0.38,
                "viability": -0.10,
                "apoptosis_signal": 0.13,
                "inflammatory_signaling": -0.03,
            },
            "Tests whether energetic reserve supports survival after initial treatment stress.",
        ),
        row(
            "IRE1I",
            "IRE1 RNase inhibitor",
            "drug",
            "IRE1alpha RNase",
            "inhibit",
            "IRE1-XBP1 proteostasis",
            0.88,
            0.79,
            0.14,
            3.0,
            20.0,
            24.0,
            24.0,
            36.0,
            {
                "ire1_xbp1": -0.34,
                "proteostasis_capacity": -0.20,
                "antigen_presentation": 0.08,
                "viability": -0.05,
                "apoptosis_signal": 0.09,
            },
            "Pharmacologically suppresses the IRE1-XBP1 adaptive branch during tolerance formation.",
        ),
        row(
            "MITORESERVEI",
            "Mitochondrial-reserve inhibitor",
            "drug",
            "Mitochondrial reserve",
            "inhibit",
            "Mitochondrial reserve",
            0.83,
            0.67,
            0.25,
            2.0,
            14.0,
            18.0,
            48.0,
            42.0,
            {
                "mitochondrial_reserve": -0.34,
                "viability": -0.12,
                "apoptosis_signal": 0.15,
                "treatment_stress": 0.04,
            },
            "Removes the energetic buffer predicted to protect drug-tolerant cells.",
        ),
        row(
            "IFNG_SUPPORT",
            "IFNG antigen-presentation support",
            "drug",
            "IFNG-JAK-STAT axis",
            "activate",
            "Antigen presentation",
            0.80,
            0.77,
            0.12,
            5.0,
            28.0,
            36.0,
            72.0,
            48.0,
            {
                "antigen_presentation": 0.34,
                "immune_exclusion": -0.20,
                "inflammatory_signaling": 0.06,
                "apoptosis_signal": 0.04,
            },
            "Tests whether restoring antigen presentation disrupts an immune-protected niche.",
        ),
        row(
            "EPILOCKI",
            "Enhancer-plasticity inhibitor",
            "drug",
            "Stress-responsive enhancer remodeling",
            "inhibit",
            "Enhancer plasticity",
            0.78,
            0.70,
            0.18,
            4.0,
            24.0,
            30.0,
            36.0,
            40.0,
            {
                "enhancer_plasticity": -0.31,
                "inflammatory_signaling": -0.06,
                "viability": -0.05,
                "apoptosis_signal": 0.07,
            },
            "Constrains the regulatory plasticity that stabilizes tolerant and resistant states.",
        ),
        row(
            "KRASI",
            "KRAS-pathway inhibitor",
            "drug",
            "KRAS-MAPK",
            "inhibit",
            "Oncogenic survival",
            0.90,
            0.82,
            0.16,
            2.0,
            22.0,
            24.0,
            24.0,
            48.0,
            {
                "treatment_stress": 0.09,
                "enhancer_plasticity": -0.08,
                "mitochondrial_reserve": -0.07,
                "viability": -0.18,
                "apoptosis_signal": 0.20,
            },
            "Represents the disease-directed backbone whose residual cells require adaptive support.",
        ),
    ]


def intervention_catalog(overrides: Iterable[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Return the auditable gene and drug intervention catalog.

    Overrides may replace any default intervention by ``intervention_id`` or append new
    interventions. Every intervention must specify all state-effect columns, although
    omitted effects are interpreted as zero.
    """

    rows = _default_interventions()
    by_id = {row["intervention_id"]: row for row in rows}
    for override in overrides or []:
        item = dict(override)
        intervention_id = str(item["intervention_id"])
        base = dict(by_id.get(intervention_id, {}))
        base.update(item)
        for column in EFFECT_COLUMNS:
            base.setdefault(column, 0.0)
        by_id[intervention_id] = base
    frame = pd.DataFrame(list(by_id.values())).sort_values(
        ["intervention_type", "intervention_id"]
    ).reset_index(drop=True)
    required = {
        "intervention_id",
        "intervention_name",
        "intervention_type",
        "target",
        "direction",
        "mechanism",
        "potency",
        "tumor_selectivity",
        "normal_toxicity",
        "onset_hours",
        "half_life_hours",
        "default_duration_hours",
        "optimal_time_hours",
        "timing_width_hours",
        *EFFECT_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Intervention catalog is missing columns: {missing}")
    if frame["intervention_id"].duplicated().any():
        raise ValueError("Intervention IDs must be unique")
    if not set(frame["intervention_type"]).issubset({"gene", "drug"}):
        raise ValueError("intervention_type must be gene or drug")
    return frame


def _event_payload(
    spec: pd.Series,
    start_hour: float,
    dose: float,
    position: int,
) -> dict[str, Any]:
    return {
        "intervention_id": str(spec["intervention_id"]),
        "intervention_name": str(spec["intervention_name"]),
        "intervention_type": str(spec["intervention_type"]),
        "mechanism": str(spec["mechanism"]),
        "start_hour": float(start_hour),
        "dose": float(dose),
        "duration_hours": float(spec["default_duration_hours"]),
        "sequence_position": int(position),
    }


def build_regimen_catalog(
    catalog: pd.DataFrame,
    config: TherapeuticConfig = TherapeuticConfig(),
) -> pd.DataFrame:
    """Enumerate gene, drug, combination, sequence, and timing counterfactuals."""

    records: list[dict[str, Any]] = []
    lookup = catalog.set_index("intervention_id", drop=False)

    def add(regimen_id: str, name: str, category: str, events: list[dict[str, Any]]) -> None:
        mechanisms = sorted({str(event["mechanism"]) for event in events})
        records.append(
            {
                "regimen_id": regimen_id,
                "regimen_name": name,
                "regimen_category": category,
                "n_events": len(events),
                "n_mechanisms": len(mechanisms),
                "mechanisms": ";".join(mechanisms),
                "first_start_hour": min(float(event["start_hour"]) for event in events),
                "last_start_hour": max(float(event["start_hour"]) for event in events),
                "complexity_penalty": max(0.0, 0.04 * (len(events) - 1)),
                "events_json": json.dumps(events, sort_keys=True),
            }
        )

    default_start = config.default_start_hour
    for _, spec in catalog.iterrows():
        category = str(spec["intervention_type"])
        event = _event_payload(spec, default_start, 1.0, 1)
        add(
            f"{category.upper()}__{spec['intervention_id']}",
            f"{spec['intervention_name']} at {default_start:g} h",
            category,
            [event],
        )

    # Simultaneous two-agent combinations include gene-drug and drug-drug pairs.
    for first_id, second_id in combinations(catalog["intervention_id"].tolist(), 2):
        first, second = lookup.loc[first_id], lookup.loc[second_id]
        events = [
            _event_payload(first, default_start, 1.0, 1),
            _event_payload(second, default_start, 1.0, 1),
        ]
        add(
            f"COMBO__{first_id}__{second_id}",
            f"{first['intervention_name']} + {second['intervention_name']}",
            "combination",
            events,
        )

    # Ordered sequences are limited to drugs because sequence is directly actionable.
    drug_ids = catalog.loc[catalog["intervention_type"] == "drug", "intervention_id"].tolist()
    for first_id, second_id in permutations(drug_ids, 2):
        first, second = lookup.loc[first_id], lookup.loc[second_id]
        events = [
            _event_payload(first, 0.0, 1.0, 1),
            _event_payload(second, config.sequence_delay_hours, 1.0, 2),
        ]
        add(
            f"SEQUENCE__{first_id}__THEN__{second_id}",
            f"{first['intervention_name']} then {second['intervention_name']}",
            "sequence",
            events,
        )

    # Timing scans are separate from default single-agent predictions.
    for _, spec in catalog.iterrows():
        for start in config.timing_grid:
            event = _event_payload(spec, float(start), 1.0, 1)
            add(
                f"TIMING__{spec['intervention_id']}__T{float(start):g}",
                f"{spec['intervention_name']} at {float(start):g} h",
                "timing",
                [event],
            )

    frame = pd.DataFrame(records)
    if frame["regimen_id"].duplicated().any():
        raise RuntimeError("Generated regimen IDs are not unique")
    return frame.sort_values(["regimen_category", "regimen_id"]).reset_index(drop=True)


def fit_therapeutic_model(
    frame: pd.DataFrame,
    seed: int = 31,
    n_splits: int = 4,
    audit: bool = True,
) -> TherapeuticModel:
    """Fit a donor-audited logistic surrogate for eventual resistance."""

    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    missing = sorted(set(THERAPEUTIC_FEATURES) - set(tumor.columns))
    if missing:
        raise ValueError(f"Therapeutic model input is missing features: {missing}")
    if tumor["future_resistant"].nunique() < 2:
        raise ValueError("Therapeutic model requires both resistant and non-resistant outcomes")
    features = list(THERAPEUTIC_FEATURES)
    groups = tumor["donor_id"].astype(str).to_numpy()
    labels = tumor["future_resistant"].astype(int).to_numpy()
    matrix = tumor[features].astype(float)
    unique_donors = np.unique(groups)
    split_count = min(max(2, n_splits), len(unique_donors))
    oof = np.full(len(tumor), np.nan, dtype=float)
    split_rows: list[dict[str, Any]] = []
    if audit:
        splitter = GroupKFold(n_splits=split_count)
        for fold, (train_index, test_index) in enumerate(splitter.split(matrix, labels, groups)):
            fold_pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            solver="liblinear",
                            max_iter=300,
                            random_state=seed + fold,
                        ),
                    ),
                ]
            )
            fold_pipeline.fit(matrix.iloc[train_index], labels[train_index])
            oof[test_index] = fold_pipeline.predict_proba(matrix.iloc[test_index])[:, 1]
            train_donors = sorted(set(groups[train_index]))
            test_donors = sorted(set(groups[test_index]))
            split_rows.append(
                {
                    "fold": fold,
                    "train_donors": ";".join(train_donors),
                    "test_donors": ";".join(test_donors),
                    "donor_overlap": ";".join(sorted(set(train_donors) & set(test_donors))),
                }
            )
        if np.isnan(oof).any():
            raise RuntimeError("Therapeutic donor-held-out predictions are incomplete")
        predicted = (oof >= 0.5).astype(int)
        metrics: dict[str, Any] = {
            "n_rows": int(len(tumor)),
            "n_donors": int(len(unique_donors)),
            "split_mode": "group_k_fold_by_donor",
            "n_splits": int(split_count),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "log_loss": float(log_loss(labels, np.column_stack([1.0 - oof, oof]), labels=[0, 1])),
            "brier_score": float(brier_score_loss(labels, oof)),
            "roc_auc": float(roc_auc_score(labels, oof)),
            "donor_overlap_detected": bool(any(row["donor_overlap"] for row in split_rows)),
            "split_manifest": split_rows,
        }
    else:
        metrics = {
            "n_rows": int(len(tumor)),
            "n_donors": int(len(unique_donors)),
            "split_mode": "bootstrap_refit_without_cross_validation",
            "n_splits": 0,
            "donor_overlap_detected": False,
            "split_manifest": [],
        }
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=300,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(matrix, labels)
    return TherapeuticModel(
        pipeline=pipeline,
        feature_names=features,
        lower_bounds=matrix.quantile(0.01),
        upper_bounds=matrix.quantile(0.99),
        metrics=metrics,
    )


def _reference_cohort(
    frame: pd.DataFrame,
    config: TherapeuticConfig,
    donors: Iterable[str] | None = None,
) -> pd.DataFrame:
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    if donors is not None:
        donor_list = [str(value) for value in donors]
        pieces = [tumor.loc[tumor["donor_id"].astype(str) == donor].copy() for donor in donor_list]
        tumor = pd.concat(pieces, ignore_index=True) if pieces else tumor.iloc[0:0]
    available_times = np.asarray(sorted(tumor["time_hours"].astype(float).unique()))
    selected_time = float(available_times[np.argmin(np.abs(available_times - config.decision_time_hours))])
    reference = tumor.loc[tumor["time_hours"].astype(float) == selected_time].copy()
    comparator = reference.loc[reference["therapy"].astype(str) == config.comparator].copy()
    if not comparator.empty:
        reference = comparator
    if reference.empty:
        raise ValueError("No tumor rows are available for the therapeutic reference cohort")
    # Balance the reference cohort across donors so large donors do not dominate rankings.
    selected = []
    for _, group in reference.groupby("donor_id", sort=True):
        selected.append(group.head(config.max_reference_rows_per_donor))
    reference = pd.concat(selected, ignore_index=True)
    reference.loc[:, "time_hours"] = config.horizon_hours
    return reference


def _dose_response(dose: float) -> float:
    dose = max(float(dose), 0.0)
    return dose / (0.45 + dose) if dose > 0 else 0.0


def _timing_factor(spec: pd.Series, start_hour: float) -> float:
    optimum = float(spec["optimal_time_hours"])
    width = max(float(spec["timing_width_hours"]), 1.0)
    gaussian = np.exp(-0.5 * ((float(start_hour) - optimum) / width) ** 2)
    return float(0.28 + 0.72 * gaussian)


def _persistence_factor(spec: pd.Series, event: dict[str, Any], horizon: float) -> float:
    start = float(event["start_hour"])
    if horizon <= start:
        return 0.0
    duration = max(float(event["duration_hours"]), 0.0)
    onset = max(float(spec["onset_hours"]), 0.2)
    half_life = max(float(spec["half_life_hours"]), 0.2)
    active_time = min(horizon - start, duration)
    onset_fraction = 1.0 - np.exp(-max(active_time, 0.0) / onset)
    time_after = max(horizon - (start + duration), 0.0)
    decay = 2.0 ** (-time_after / half_life)
    acute = onset_fraction * decay
    # A completed pulse can leave a durable disease-state consequence even after the
    # compound itself is cleared. This memory term is deliberately explicit and is
    # weaker than ongoing exposure; real projects should replace it with measured PK/PD.
    memory = 0.35 * onset_fraction * (0.55 + 0.45 * np.exp(-time_after / (6.0 * half_life)))
    return float(max(acute, memory))


def _pair_synergy(first: pd.Series, second: pd.Series) -> float:
    pair = frozenset({str(first["mechanism"]), str(second["mechanism"])})
    special = {
        frozenset({"IRE1-XBP1 proteostasis", "Mitochondrial reserve"}): 0.20,
        frozenset({"IRE1-XBP1 proteostasis", "Antigen presentation"}): 0.14,
        frozenset({"Enhancer plasticity", "Oncogenic survival"}): 0.14,
        frozenset({"Mitochondrial reserve", "Oncogenic survival"}): 0.12,
        frozenset({"Antigen presentation", "Oncogenic survival"}): 0.10,
        frozenset({"Integrated stress response", "IRE1-XBP1 proteostasis"}): 0.08,
    }
    if str(first["target"]) == str(second["target"]):
        return -0.12
    if str(first["mechanism"]) == str(second["mechanism"]):
        return -0.06
    return special.get(pair, 0.06)


def _sequence_modifier(first: pd.Series, second: pd.Series) -> float:
    first_mechanism = str(first["mechanism"])
    second_mechanism = str(second["mechanism"])
    directional = {
        ("Enhancer plasticity", "Oncogenic survival"): 0.13,
        ("IRE1-XBP1 proteostasis", "Mitochondrial reserve"): 0.12,
        ("Oncogenic survival", "Antigen presentation"): 0.14,
        ("Mitochondrial reserve", "IRE1-XBP1 proteostasis"): 0.06,
        ("Oncogenic survival", "IRE1-XBP1 proteostasis"): 0.10,
        ("Antigen presentation", "Oncogenic survival"): -0.03,
    }
    return directional.get((first_mechanism, second_mechanism), 0.02)


def _apply_regimen(
    reference: pd.DataFrame,
    regimen: pd.Series,
    catalog_lookup: pd.DataFrame,
    horizon: float,
) -> tuple[pd.DataFrame, dict[str, float], float, float]:
    modified = reference.copy()
    events = sorted(json.loads(str(regimen["events_json"])), key=lambda item: item["start_hour"])
    specs = [catalog_lookup.loc[event["intervention_id"]] for event in events]
    pair_bonus = 0.0
    if len(specs) >= 2:
        for first, second in combinations(specs, 2):
            pair_bonus += _pair_synergy(first, second)
        pair_bonus /= max(1, len(list(combinations(specs, 2))))
    state_deltas = {name: 0.0 for name in STATE_EFFECT_FEATURES}
    intensities: list[float] = []
    for index, (event, spec) in enumerate(zip(events, specs)):
        intensity = (
            float(spec["potency"])
            * _dose_response(float(event["dose"]))
            * _timing_factor(spec, float(event["start_hour"]))
            * _persistence_factor(spec, event, horizon)
        )
        if len(events) > 1:
            intensity *= 1.0 + max(-0.25, min(pair_bonus, 0.30))
        if index > 0 and float(event["start_hour"]) > float(events[index - 1]["start_hour"]):
            intensity *= 1.0 + _sequence_modifier(specs[index - 1], spec)
        intensities.append(float(intensity))
        for feature in STATE_EFFECT_FEATURES:
            delta = float(spec[f"delta_{feature}"])
            if delta == 0.0:
                continue
            current = modified[feature].to_numpy(dtype=float)
            # Inhibitory effects scale with available pathway activity; activating effects
            # scale with remaining headroom. This keeps state values bounded and auditable.
            susceptibility = 0.35 + 0.65 * (current if delta < 0 else 1.0 - current)
            change = delta * intensity * susceptibility
            modified.loc[:, feature] = np.clip(current + change, 0.0, 1.0)
            state_deltas[feature] += float(np.mean(change))
    synergy = float(pair_bonus)
    total_intensity = float(np.mean(intensities)) if intensities else 0.0
    return modified, state_deltas, synergy, total_intensity


def _normal_toxicity(
    frame: pd.DataFrame,
    regimen: pd.Series,
    catalog_lookup: pd.DataFrame,
    config: TherapeuticConfig,
) -> float:
    normal = frame.loc[frame["cell_type"] != "tumor"].copy()
    if normal.empty:
        vulnerability = {feature: 0.5 for feature in STATE_EFFECT_FEATURES}
    else:
        vulnerability = normal[list(STATE_EFFECT_FEATURES)].mean().to_dict()
    events = json.loads(str(regimen["events_json"]))
    survival = 1.0
    for event in events:
        spec = catalog_lookup.loc[event["intervention_id"]]
        effect_features = [
            feature for feature in STATE_EFFECT_FEATURES if abs(float(spec[f"delta_{feature}"])) > 0
        ]
        pathway_vulnerability = (
            float(np.mean([vulnerability[feature] for feature in effect_features]))
            if effect_features
            else 0.5
        )
        intensity = (
            float(spec["potency"])
            * _dose_response(float(event["dose"]))
            * _timing_factor(spec, float(event["start_hour"]))
        )
        toxicity = (
            float(spec["normal_toxicity"])
            * intensity
            * (1.10 - 0.45 * float(spec["tumor_selectivity"]))
            * (0.70 + 0.60 * pathway_vulnerability)
        )
        survival *= 1.0 - np.clip(toxicity, 0.0, 0.85)
    combined = 1.0 - survival
    if len(events) > 1:
        combined = min(1.0, combined + 0.025 * (len(events) - 1))
    return float(combined)


def _extrapolation_score(model: TherapeuticModel, modified: pd.DataFrame) -> float:
    matrix = modified[model.feature_names].astype(float)
    below = matrix.lt(model.lower_bounds, axis="columns")
    above = matrix.gt(model.upper_bounds, axis="columns")
    return float((below | above).to_numpy().mean())


def predict_regimens(
    frame: pd.DataFrame,
    model: TherapeuticModel,
    catalog: pd.DataFrame,
    regimens: pd.DataFrame,
    config: TherapeuticConfig = TherapeuticConfig(),
    reference_donors: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict all therapeutic counterfactuals for a donor-balanced reference cohort."""

    reference = _reference_cohort(frame, config, donors=reference_donors)
    baseline = reference.copy()
    baseline.loc[:, "time_hours"] = config.horizon_hours
    baseline_probability = model.predict_probability(baseline)
    baseline_resistance = float(np.mean(baseline_probability))
    baseline_viability = float(reference["viability"].mean())
    baseline_apoptosis = float(reference["apoptosis_signal"].mean())
    baseline_antigen = float(reference["antigen_presentation"].mean())
    baseline_exclusion = float(reference["immune_exclusion"].mean())
    lookup = catalog.set_index("intervention_id", drop=False)
    predictions: list[dict[str, Any]] = []
    state_change_rows: list[dict[str, Any]] = []
    for _, regimen in regimens.iterrows():
        modified, state_deltas, synergy, intensity = _apply_regimen(
            reference, regimen, lookup, config.horizon_hours
        )
        modified.loc[:, "time_hours"] = config.horizon_hours
        resistance = float(np.mean(model.predict_probability(modified)))
        risk_reduction = baseline_resistance - resistance
        viability = float(modified["viability"].mean())
        apoptosis = float(modified["apoptosis_signal"].mean())
        antigen = float(modified["antigen_presentation"].mean())
        exclusion = float(modified["immune_exclusion"].mean())
        normal_toxicity = _normal_toxicity(frame, regimen, lookup, config)
        extrapolation = _extrapolation_score(model, modified)
        viability_reduction = baseline_viability - viability
        apoptosis_gain = apoptosis - baseline_apoptosis
        antigen_gain = antigen - baseline_antigen
        exclusion_reduction = baseline_exclusion - exclusion
        complexity = float(regimen["complexity_penalty"])
        utility = (
            0.45 * risk_reduction
            + 0.15 * viability_reduction
            + 0.12 * apoptosis_gain
            + 0.10 * antigen_gain
            + 0.08 * exclusion_reduction
            - config.normal_toxicity_weight * normal_toxicity
            - complexity
            - 0.08 * extrapolation
        )
        predictions.append(
            {
                **regimen.to_dict(),
                "baseline_resistance_probability": baseline_resistance,
                "counterfactual_resistance_probability": resistance,
                "resistance_risk_reduction": risk_reduction,
                "baseline_tumor_viability": baseline_viability,
                "counterfactual_tumor_viability": viability,
                "tumor_viability_reduction": viability_reduction,
                "apoptosis_gain": apoptosis_gain,
                "antigen_presentation_gain": antigen_gain,
                "immune_exclusion_reduction": exclusion_reduction,
                "normal_cell_toxicity": normal_toxicity,
                "therapeutic_index": risk_reduction / max(normal_toxicity + 0.03, 0.03),
                "mechanistic_synergy": synergy,
                "mean_effective_intensity": intensity,
                "extrapolation_score": extrapolation,
                "extrapolation_flag": bool(extrapolation > 0.10),
                "utility": utility,
                "n_reference_rows": int(len(reference)),
                "n_reference_donors": int(reference["donor_id"].nunique()),
            }
        )
        for feature, delta in state_deltas.items():
            state_change_rows.append(
                {
                    "regimen_id": regimen["regimen_id"],
                    "regimen_name": regimen["regimen_name"],
                    "regimen_category": regimen["regimen_category"],
                    "state_feature": feature,
                    "mean_counterfactual_change": delta,
                }
            )
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame["pareto_optimal"] = _pareto_mask(
        prediction_frame["resistance_risk_reduction"].to_numpy(),
        prediction_frame["normal_cell_toxicity"].to_numpy(),
    )
    return prediction_frame, pd.DataFrame(state_change_rows)


def _pareto_mask(benefit: np.ndarray, cost: np.ndarray) -> np.ndarray:
    mask = np.ones(len(benefit), dtype=bool)
    for i in range(len(benefit)):
        dominated = (
            (benefit >= benefit[i])
            & (cost <= cost[i])
            & ((benefit > benefit[i]) | (cost < cost[i]))
        )
        dominated[i] = False
        if np.any(dominated):
            mask[i] = False
    return mask


def bootstrap_therapeutic_predictions(
    frame: pd.DataFrame,
    catalog: pd.DataFrame,
    regimens: pd.DataFrame,
    config: TherapeuticConfig,
) -> pd.DataFrame:
    """Refit the therapeutic model after donor-cluster bootstrap resampling."""

    donors = sorted(frame["donor_id"].astype(str).unique())
    rng = np.random.default_rng(config.seed + 701)
    records: list[pd.DataFrame] = []
    for replicate in range(config.bootstrap):
        sampled = rng.choice(donors, size=len(donors), replace=True)
        pieces = []
        for draw, donor in enumerate(sampled):
            piece = frame.loc[frame["donor_id"].astype(str) == donor].copy()
            piece.loc[:, "donor_id"] = f"{donor}__draw{draw}"
            pieces.append(piece)
        bootstrap_frame = pd.concat(pieces, ignore_index=True)
        try:
            model = fit_therapeutic_model(
                bootstrap_frame,
                seed=config.seed + replicate + 1,
                n_splits=min(4, len(donors)),
                audit=False,
            )
            predictions, _ = predict_regimens(
                bootstrap_frame,
                model,
                catalog,
                regimens,
                config,
            )
        except (ValueError, RuntimeError):
            continue
        subset = predictions[
            [
                "regimen_id",
                "counterfactual_resistance_probability",
                "resistance_risk_reduction",
                "normal_cell_toxicity",
                "utility",
            ]
        ].copy()
        subset["bootstrap_replicate"] = replicate
        records.append(subset)
    if not records:
        raise RuntimeError("No therapeutic donor-bootstrap replicate completed successfully")
    draws = pd.concat(records, ignore_index=True)
    summaries = []
    for regimen_id, group in draws.groupby("regimen_id", sort=False):
        row: dict[str, Any] = {
            "regimen_id": regimen_id,
            "bootstrap_successful_replicates": int(group["bootstrap_replicate"].nunique()),
        }
        for metric in [
            "counterfactual_resistance_probability",
            "resistance_risk_reduction",
            "normal_cell_toxicity",
            "utility",
        ]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_bootstrap_mean"] = float(np.mean(values))
            row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
        summaries.append(row)
    return pd.DataFrame(summaries)


def run_counterfactual_therapeutics(
    frame: pd.DataFrame,
    config: TherapeuticConfig = TherapeuticConfig(),
    intervention_overrides: Iterable[dict[str, Any]] | None = None,
) -> TherapeuticResult:
    catalog = intervention_catalog(intervention_overrides)
    regimens = build_regimen_catalog(catalog, config)
    model = fit_therapeutic_model(frame, seed=config.seed)
    predictions, state_changes = predict_regimens(frame, model, catalog, regimens, config)
    intervals = bootstrap_therapeutic_predictions(frame, catalog, regimens, config)
    predictions = predictions.merge(intervals, on="regimen_id", how="left", validate="one_to_one")
    predictions["utility_interval_width"] = (
        predictions["utility_ci_high"] - predictions["utility_ci_low"]
    )
    predictions["uncertainty_adjusted_utility"] = (
        predictions["utility"]
        - config.uncertainty_penalty * predictions["utility_interval_width"]
    )
    predictions["rank"] = predictions["uncertainty_adjusted_utility"].rank(
        method="first", ascending=False
    ).astype(int)
    predictions = predictions.sort_values("rank").reset_index(drop=True)
    qc = validate_therapeutic_predictions(catalog, regimens, predictions, config)
    model_metrics = {key: value for key, value in model.metrics.items() if key != "split_manifest"}
    model_metrics["split_manifest"] = model.metrics["split_manifest"]
    return TherapeuticResult(
        intervention_catalog=catalog,
        regimen_catalog=regimens,
        predictions=predictions,
        bootstrap_intervals=intervals,
        state_changes=state_changes,
        model_metrics=model_metrics,
        qc=qc,
    )


def validate_therapeutic_predictions(
    catalog: pd.DataFrame,
    regimens: pd.DataFrame,
    predictions: pd.DataFrame,
    config: TherapeuticConfig,
) -> dict[str, Any]:
    categories = {"gene", "drug", "combination", "sequence", "timing"}
    observed = set(predictions["regimen_category"].astype(str))
    probabilities = predictions["counterfactual_resistance_probability"]
    valid = bool(
        not catalog.empty
        and not regimens.empty
        and not predictions.empty
        and categories.issubset(observed)
        and probabilities.between(0.0, 1.0).all()
        and predictions["normal_cell_toxicity"].between(0.0, 1.0).all()
        and predictions["utility_ci_low"].le(predictions["utility_ci_high"]).all()
        and predictions["bootstrap_successful_replicates"].min() >= 1
        and not regimens["regimen_id"].duplicated().any()
    )
    return {
        "valid": valid,
        "n_interventions": int(len(catalog)),
        "n_gene_interventions": int((catalog["intervention_type"] == "gene").sum()),
        "n_drug_interventions": int((catalog["intervention_type"] == "drug").sum()),
        "n_regimens": int(len(regimens)),
        "n_gene_predictions": int((predictions["regimen_category"] == "gene").sum()),
        "n_drug_predictions": int((predictions["regimen_category"] == "drug").sum()),
        "n_combination_predictions": int((predictions["regimen_category"] == "combination").sum()),
        "n_sequence_predictions": int((predictions["regimen_category"] == "sequence").sum()),
        "n_timing_predictions": int((predictions["regimen_category"] == "timing").sum()),
        "n_pareto_optimal": int(predictions["pareto_optimal"].sum()),
        "bootstrap_requested": int(config.bootstrap),
        "bootstrap_min_successful": int(predictions["bootstrap_successful_replicates"].min()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "top_regimen": str(predictions.iloc[0]["regimen_name"]),
        "top_category": str(predictions.iloc[0]["regimen_category"]),
        "top_uncertainty_adjusted_utility": float(
            predictions.iloc[0]["uncertainty_adjusted_utility"]
        ),
        "extrapolation_flagged": int(predictions["extrapolation_flag"].sum()),
    }


def plot_therapeutic_ranking(
    predictions: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 16,
) -> Path:
    output_path = Path(output_path)
    top = predictions.nsmallest(top_n, "rank").sort_values(
        "uncertainty_adjusted_utility", ascending=True
    )
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(top["regimen_name"], top["uncertainty_adjusted_utility"])
    ax.set_xlabel("Uncertainty-adjusted utility")
    ax.set_title("Counterfactual therapeutic ranking")
    ax.axvline(0.0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_timing_heatmap(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    timing = predictions.loc[predictions["regimen_category"] == "timing"].copy()
    rows = []
    for _, row in timing.iterrows():
        event = json.loads(row["events_json"])[0]
        rows.append(
            {
                "intervention": event["intervention_name"],
                "start_hour": float(event["start_hour"]),
                "utility": float(row["uncertainty_adjusted_utility"]),
            }
        )
    pivot = pd.DataFrame(rows).pivot(index="intervention", columns="start_hour", values="utility")
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:g} h" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Intervention start")
    ax.set_title("Timing-dependent therapeutic utility")
    fig.colorbar(image, ax=ax, label="Uncertainty-adjusted utility")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sequence_comparison(
    predictions: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 14,
) -> Path:
    output_path = Path(output_path)
    sequences = predictions.loc[predictions["regimen_category"] == "sequence"].nsmallest(
        top_n, "rank"
    ).sort_values("uncertainty_adjusted_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.barh(sequences["regimen_name"], sequences["uncertainty_adjusted_utility"])
    ax.set_xlabel("Uncertainty-adjusted utility")
    ax.set_title("Ordered treatment-sequence comparison")
    ax.axvline(0.0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_therapeutic_pareto(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for category, group in predictions.groupby("regimen_category", sort=True):
        ax.scatter(
            group["normal_cell_toxicity"],
            group["resistance_risk_reduction"],
            label=category,
            alpha=0.72,
        )
    pareto = predictions.loc[predictions["pareto_optimal"]]
    ax.scatter(
        pareto["normal_cell_toxicity"],
        pareto["resistance_risk_reduction"],
        marker="x",
        s=65,
        label="Pareto optimal",
    )
    ax.set_xlabel("Predicted normal-cell toxicity")
    ax.set_ylabel("Resistance-risk reduction")
    ax.set_title("Therapeutic benefit–toxicity frontier")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_counterfactual_waterfall(
    predictions: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 14,
) -> Path:
    output_path = Path(output_path)
    top = predictions.nsmallest(top_n, "rank").sort_values("resistance_risk_reduction")
    lower = top["resistance_risk_reduction"] - top["resistance_risk_reduction_ci_low"]
    upper = top["resistance_risk_reduction_ci_high"] - top["resistance_risk_reduction"]
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.barh(top["regimen_name"], top["resistance_risk_reduction"])
    ax.errorbar(
        top["resistance_risk_reduction"],
        np.arange(len(top)),
        xerr=np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)]),
        fmt="none",
        capsize=2,
    )
    ax.set_xlabel("Predicted reduction in resistance probability")
    ax.set_title("Donor-bootstrap counterfactual effects")
    ax.axvline(0.0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _model_card(result: TherapeuticResult, config: TherapeuticConfig) -> str:
    top = result.predictions.iloc[0]
    return "\n".join(
        [
            "# CausaFlux v1.7.0 counterfactual therapeutics model card",
            "",
            "## Intended use",
            "Research-method development for ranking gene perturbations, drugs, combinations, ordered sequences, and intervention times against a configured disease-state model.",
            "",
            "## Prediction mechanism",
            "An interpretable intervention catalog changes explicit state variables. A donor-audited logistic surrogate then maps the counterfactual state to eventual resistance probability. Normal-cell toxicity is estimated separately from non-tumor pathway vulnerability.",
            "",
            "## Uncertainty",
            "The release refits the resistance surrogate after donor-cluster bootstrap resampling. Intervals therefore quantify donor composition and model-refit variability, but not all experimental, pharmacokinetic, structural, causal, or clinical uncertainty.",
            "",
            "## Counterfactual assumptions",
            "State effects, timing windows, pair interactions, and sequence modifiers are configured mechanistic hypotheses. Predictions require consistency, positivity, correct state measurement, no unmodeled interference, and adequate support near the counterfactual state.",
            "",
            "## Demonstration summary",
            f"- Interventions: {result.qc['n_interventions']}",
            f"- Regimens: {result.qc['n_regimens']}",
            f"- Donor bootstrap requested: {config.bootstrap}",
            f"- Top synthetic regimen: {top['regimen_name']}",
            f"- Top synthetic category: {top['regimen_category']}",
            f"- Predicted resistance-risk reduction: {top['resistance_risk_reduction']:.3f}",
            f"- Predicted normal-cell toxicity: {top['normal_cell_toxicity']:.3f}",
            "",
            "## Prohibited interpretation",
            "Bundled rankings are synthetic software outputs. They are not treatment recommendations, evidence of efficacy, dosing guidance, safety assessments, or substitutes for pharmacology and prospective validation.",
        ]
    )


def write_therapeutic_outputs(
    result: TherapeuticResult,
    output_dir: str | Path,
    config: TherapeuticConfig,
    *,
    write_plots: bool = True,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    result.intervention_catalog.to_csv(output_dir / "intervention_catalog.csv", index=False)
    result.regimen_catalog.to_csv(output_dir / "regimen_catalog.csv", index=False)
    result.predictions.to_csv(output_dir / "all_regimen_predictions.csv", index=False)
    result.bootstrap_intervals.to_csv(output_dir / "donor_bootstrap_intervals.csv", index=False)
    result.state_changes.to_csv(output_dir / "mechanistic_state_changes.csv", index=False)
    for category, filename in [
        ("gene", "gene_predictions.csv"),
        ("drug", "drug_predictions.csv"),
        ("combination", "combination_predictions.csv"),
        ("sequence", "sequence_predictions.csv"),
        ("timing", "timing_predictions.csv"),
    ]:
        result.predictions.loc[result.predictions["regimen_category"] == category].to_csv(
            output_dir / filename, index=False
        )
    result.predictions.nsmallest(20, "rank").to_csv(
        output_dir / "top_therapeutic_recommendations.csv", index=False
    )
    pd.DataFrame(result.model_metrics.get("split_manifest", [])).to_csv(
        output_dir / "therapeutic_donor_split_manifest.csv", index=False
    )
    metrics = {key: value for key, value in result.model_metrics.items() if key != "split_manifest"}
    json_dump(metrics, output_dir / "therapeutic_model_metrics.json")
    json_dump(result.qc, output_dir / "therapeutic_qc.json")
    (output_dir / "therapeutic_model_card.md").write_text(
        _model_card(result, config), encoding="utf-8"
    )
    if write_plots:
        paths["ranking_plot"] = plot_therapeutic_ranking(
            result.predictions, output_dir / "therapeutic_ranking.png"
        )
        paths["timing_plot"] = plot_timing_heatmap(
            result.predictions, output_dir / "timing_heatmap.png"
        )
        paths["sequence_plot"] = plot_sequence_comparison(
            result.predictions, output_dir / "sequence_comparison.png"
        )
        paths["pareto_plot"] = plot_therapeutic_pareto(
            result.predictions, output_dir / "benefit_toxicity_pareto.png"
        )
        paths["waterfall_plot"] = plot_counterfactual_waterfall(
            result.predictions, output_dir / "counterfactual_waterfall.png"
        )
    paths.update(
        {
            "predictions": output_dir / "all_regimen_predictions.csv",
            "top": output_dir / "top_therapeutic_recommendations.csv",
            "qc": output_dir / "therapeutic_qc.json",
        }
    )
    return paths
