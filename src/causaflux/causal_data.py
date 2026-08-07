from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

STATE_ORDER = [
    "treatment_sensitive",
    "early_stress",
    "reversible_tolerance",
    "stable_resistance",
]

CELL_TYPES = [
    "tumor",
    "macrophage",
    "dendritic_cell",
    "t_cell",
    "fibroblast",
    "vascular",
]

BIOMARKER_FEATURES = [
    "ire1_xbp1",
    "proteostasis_capacity",
    "enhancer_plasticity",
    "mitochondrial_reserve",
    "antigen_presentation",
    "immune_exclusion",
    "inflammatory_signaling",
    "viability",
    "apoptosis_signal",
]

REQUIRED_COLUMNS = {
    "row_id",
    "donor_id",
    "sample_id",
    "lineage_id",
    "time_hours",
    "cell_type",
    "state",
    "therapy",
    "future_resistant",
    *BIOMARKER_FEATURES,
}


@dataclass(frozen=True)
class CancerDemoConfig:
    n_donors: int = 8
    clones_per_donor: int = 36
    non_tumor_cells_per_type: int = 12
    times: tuple[float, ...] = (0.0, 24.0, 72.0, 168.0)
    seed: int = 31


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _state_from_score(score: float) -> str:
    if score < 0.30:
        return STATE_ORDER[0]
    if score < 0.50:
        return STATE_ORDER[1]
    if score < 0.78:
        return STATE_ORDER[2]
    return STATE_ORDER[3]


def generate_cancer_demo(config: CancerDemoConfig | None = None) -> pd.DataFrame:
    """Generate a structured cancer-evolution dataset for software demonstrations.

    The generator contains a known, deliberately simplified mechanism. Values are
    synthetic and must never be interpreted as experimental or clinical evidence.
    """

    config = config or CancerDemoConfig()
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    therapies = np.asarray(
        [
            "standard_therapy",
            "standard_plus_ire1i",
            "standard_plus_mitoi",
            "standard_plus_ifng",
        ]
    )
    therapy_probabilities = np.asarray([0.38, 0.24, 0.20, 0.18])

    tumor_records_by_lineage: dict[str, list[dict[str, object]]] = {}
    for donor_index in range(config.n_donors):
        donor_id = f"D{donor_index + 1:02d}"
        donor_stress = rng.normal(0.0, 0.22)
        donor_immunity = rng.normal(0.0, 0.25)
        for clone_index in range(config.clones_per_donor):
            lineage_id = f"{donor_id}_T{clone_index + 1:03d}"
            therapy = str(rng.choice(therapies, p=therapy_probabilities))
            mutation_burden = float(np.clip(rng.beta(2.2, 5.0), 0.0, 1.0))
            baseline_plasticity = float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0))
            baseline_mito = float(np.clip(rng.normal(0.58, 0.14), 0.05, 0.98))
            baseline_antigen = float(np.clip(rng.normal(0.63 + donor_immunity * 0.08, 0.13), 0.02, 0.98))
            ire1i = float(therapy == "standard_plus_ire1i")
            mitoi = float(therapy == "standard_plus_mitoi")
            ifng = float(therapy == "standard_plus_ifng")
            lineage_rows: list[dict[str, object]] = []

            for time_index, time_hours in enumerate(config.times):
                progress = float(time_index / max(len(config.times) - 1, 1))
                treatment_stress = float(np.clip(0.08 + 0.86 * progress + rng.normal(0, 0.04), 0, 1))
                ire1_xbp1 = float(
                    np.clip(
                        0.18
                        + 0.55 * treatment_stress
                        + 0.20 * baseline_plasticity
                        + 0.08 * donor_stress
                        - 0.43 * ire1i * progress
                        + rng.normal(0, 0.055),
                        0,
                        1,
                    )
                )
                proteostasis = float(
                    np.clip(0.28 + 0.60 * ire1_xbp1 - 0.10 * treatment_stress + rng.normal(0, 0.05), 0, 1)
                )
                enhancer = float(
                    np.clip(
                        0.12
                        + 0.50 * baseline_plasticity
                        + 0.34 * treatment_stress
                        - 0.08 * ire1i * progress
                        + rng.normal(0, 0.05),
                        0,
                        1,
                    )
                )
                mito = float(
                    np.clip(
                        baseline_mito
                        + 0.23 * treatment_stress
                        - 0.40 * mitoi * progress
                        + rng.normal(0, 0.05),
                        0,
                        1,
                    )
                )
                antigen = float(
                    np.clip(
                        baseline_antigen
                        - 0.26 * treatment_stress
                        - 0.14 * ire1_xbp1
                        + 0.40 * ifng * progress
                        + rng.normal(0, 0.055),
                        0,
                        1,
                    )
                )
                immune_exclusion = float(
                    np.clip(
                        0.12
                        + 0.43 * treatment_stress
                        + 0.25 * ire1_xbp1
                        - 0.42 * antigen
                        - 0.20 * ifng * progress
                        + rng.normal(0, 0.05),
                        0,
                        1,
                    )
                )
                inflammation = float(
                    np.clip(0.12 + 0.36 * treatment_stress + 0.25 * immune_exclusion + rng.normal(0, 0.06), 0, 1)
                )
                latent_resistance = float(
                    _sigmoid(
                        -2.15
                        + 1.25 * mutation_burden
                        + 1.00 * enhancer
                        + 0.95 * proteostasis
                        + 0.90 * mito
                        + 0.75 * immune_exclusion
                        + 0.75 * progress
                        + donor_stress
                        - 0.30 * ire1i * progress
                        - 0.26 * mitoi * progress
                        - 0.20 * ifng * progress
                        + rng.normal(0, 0.12)
                    )
                )
                state = _state_from_score(latent_resistance)
                apoptosis = float(
                    np.clip(0.10 + 0.58 * treatment_stress - 0.32 * proteostasis - 0.24 * mito + rng.normal(0, 0.05), 0, 1)
                )
                viability = float(
                    np.clip(0.96 - 0.52 * apoptosis + 0.20 * latent_resistance + rng.normal(0, 0.04), 0, 1)
                )
                record = {
                    "row_id": f"{lineage_id}_{int(time_hours):03d}",
                    "donor_id": donor_id,
                    "sample_id": f"{donor_id}_{therapy}_{int(time_hours):03d}",
                    "lineage_id": lineage_id,
                    "time_hours": float(time_hours),
                    "cell_type": "tumor",
                    "state": state,
                    "therapy": therapy,
                    "ire1_inhibition": ire1i,
                    "mitochondrial_inhibition": mitoi,
                    "ifng_support": ifng,
                    "mutation_burden": mutation_burden,
                    "treatment_stress": treatment_stress,
                    "ire1_xbp1": ire1_xbp1,
                    "proteostasis_capacity": proteostasis,
                    "enhancer_plasticity": enhancer,
                    "mitochondrial_reserve": mito,
                    "antigen_presentation": antigen,
                    "immune_exclusion": immune_exclusion,
                    "inflammatory_signaling": inflammation,
                    "viability": viability,
                    "apoptosis_signal": apoptosis,
                    "resistance_score": latent_resistance,
                }
                lineage_rows.append(record)
            tumor_records_by_lineage[lineage_id] = lineage_rows

    for lineage_rows in tumor_records_by_lineage.values():
        final_resistant = int(lineage_rows[-1]["state"] == "stable_resistance")
        for record in lineage_rows:
            record["future_resistant"] = final_resistant
            rows.append(record)

    # Non-tumor populations provide a multicellular context. They are generated at
    # the donor/time/therapy level and are not used as lineage-transition units.
    tumor_frame = pd.DataFrame(rows)
    context_rows: list[dict[str, object]] = []
    for (donor_id, therapy, time_hours), group in tumor_frame.groupby(
        ["donor_id", "therapy", "time_hours"], sort=False
    ):
        tumor_context = group[BIOMARKER_FEATURES + ["resistance_score"]].mean(numeric_only=True)
        progress = float(time_hours / max(config.times))
        for cell_type in CELL_TYPES[1:]:
            for cell_index in range(config.non_tumor_cells_per_type):
                noise = rng.normal(0, 0.07, size=len(BIOMARKER_FEATURES))
                base = np.asarray(
                    [
                        0.18 + 0.20 * tumor_context["ire1_xbp1"],
                        0.35 + 0.22 * tumor_context["proteostasis_capacity"],
                        0.20 + 0.18 * tumor_context["enhancer_plasticity"],
                        0.45 + 0.18 * tumor_context["mitochondrial_reserve"],
                        0.58 - 0.28 * tumor_context["immune_exclusion"],
                        0.18 + 0.45 * tumor_context["immune_exclusion"],
                        0.20 + 0.46 * tumor_context["inflammatory_signaling"],
                        0.80 - 0.20 * progress,
                        0.15 + 0.16 * progress,
                    ]
                )
                if cell_type == "macrophage":
                    base[5] += 0.14
                    base[6] += 0.14
                elif cell_type == "dendritic_cell":
                    base[4] += 0.18
                elif cell_type == "t_cell":
                    base[4] += 0.08
                    base[7] -= 0.08 * tumor_context["immune_exclusion"]
                elif cell_type == "fibroblast":
                    base[5] += 0.10
                elif cell_type == "vascular":
                    base[3] += 0.06
                values = np.clip(base + noise, 0, 1)
                state = "supportive_niche" if values[5] > 0.52 else "responsive_niche"
                context_rows.append(
                    {
                        "row_id": f"{donor_id}_{therapy}_{int(time_hours):03d}_{cell_type}_{cell_index:03d}",
                        "donor_id": donor_id,
                        "sample_id": f"{donor_id}_{therapy}_{int(time_hours):03d}",
                        "lineage_id": f"{donor_id}_{therapy}_{cell_type}_{cell_index:03d}",
                        "time_hours": float(time_hours),
                        "cell_type": cell_type,
                        "state": state,
                        "therapy": therapy,
                        "ire1_inhibition": float(therapy == "standard_plus_ire1i"),
                        "mitochondrial_inhibition": float(therapy == "standard_plus_mitoi"),
                        "ifng_support": float(therapy == "standard_plus_ifng"),
                        "mutation_burden": 0.0,
                        "treatment_stress": float(np.clip(0.12 + 0.70 * progress + rng.normal(0, 0.04), 0, 1)),
                        **{name: float(value) for name, value in zip(BIOMARKER_FEATURES, values)},
                        "resistance_score": float(tumor_context["resistance_score"]),
                        "future_resistant": int(tumor_context["resistance_score"] > 0.70),
                    }
                )

    frame = pd.concat([tumor_frame, pd.DataFrame(context_rows)], ignore_index=True)
    return frame.sort_values(["donor_id", "lineage_id", "time_hours"], kind="stable").reset_index(drop=True)


def validate_causal_frame(frame: pd.DataFrame, required_columns: Iterable[str] | None = None) -> dict[str, object]:
    required = set(required_columns or REQUIRED_COLUMNS)
    missing = sorted(required - set(frame.columns))
    duplicate_rows = int(frame["row_id"].duplicated().sum()) if "row_id" in frame else -1
    invalid_states = []
    if "state" in frame:
        allowed = set(STATE_ORDER) | {"supportive_niche", "responsive_niche"}
        invalid_states = sorted(set(frame["state"].dropna().astype(str)) - allowed)
    donor_count = int(frame["donor_id"].nunique()) if "donor_id" in frame else 0
    lineage_count = int(frame["lineage_id"].nunique()) if "lineage_id" in frame else 0
    time_count = int(frame["time_hours"].nunique()) if "time_hours" in frame else 0
    tumor = frame.loc[frame.get("cell_type", pd.Series(index=frame.index, dtype=str)) == "tumor"]
    non_monotonic = 0
    if not tumor.empty and {"lineage_id", "time_hours"}.issubset(tumor.columns):
        for _, group in tumor.groupby("lineage_id"):
            values = group.sort_values("time_hours")["time_hours"].to_numpy()
            non_monotonic += int(np.any(np.diff(values) < 0))
    report = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_donors": donor_count,
        "n_lineages": lineage_count,
        "n_timepoints": time_count,
        "missing_required_columns": missing,
        "duplicate_row_ids": duplicate_rows,
        "invalid_states": invalid_states,
        "non_monotonic_tumor_lineages": non_monotonic,
        "missing_value_fraction": float(frame.isna().mean().mean()),
        "valid": not missing and duplicate_rows == 0 and not invalid_states and non_monotonic == 0,
    }
    if not report["valid"]:
        raise ValueError(f"Causal dataset validation failed: {report}")
    return report


def load_or_generate_cancer_data(
    mode: str,
    path: str | Path | None,
    config: CancerDemoConfig,
) -> pd.DataFrame:
    mode = mode.lower()
    if mode == "synthetic":
        return generate_cancer_demo(config)
    if mode != "csv":
        raise ValueError("causal data.mode must be 'synthetic' or 'csv'")
    if path is None:
        raise ValueError("causal data.path is required when mode='csv'")
    return pd.read_csv(path)
