from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utils import json_dump, set_seed
from .visualization.publication import COLORS, apply_publication_style, export_figure


MODALITY_ORDER = (
    "rna",
    "imaging",
    "reporter",
    "phosphoprotein",
    "metabolome",
    "lipidome",
)
LATE_OMICS_MODALITIES = ("rna", "phosphoprotein", "metabolome", "lipidome")
MODEL_ORDER = (
    "BaselineCovariatesMLP",
    "LatestRNAMLP",
    "StaticMultimodalFusion",
    "EarlyImagingReporterGRU",
    "CausaFluxPoEDynamic",
    "CausaFluxMoEDynamic",
    "CausaFluxPoE_NoImagingReporter",
)
DYNAMIC_IMAGING_MODELS = {
    "EarlyImagingReporterGRU",
    "CausaFluxPoEDynamic",
    "CausaFluxMoEDynamic",
}
BASELINE_MODELS = {
    "BaselineCovariatesMLP",
    "LatestRNAMLP",
    "StaticMultimodalFusion",
}


@dataclass
class MultimodalDynamicConfig:
    seed: int = 140
    n_donors: int = 15
    n_cohorts: int = 3
    replicates_per_history: int = 5
    steps: int = 7
    context_steps: int = 4
    latent_dim: int = 24
    hidden_dim: int = 48
    epochs: int = 30
    patience: int = 7
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    modality_dropout: float = 0.25
    bootstrap_replicates: int = 100
    device: str = "cpu"

    @property
    def horizon(self) -> int:
        return self.steps - self.context_steps


@dataclass
class MultimodalDynamicData:
    modalities: dict[str, np.ndarray]
    observed_masks: dict[str, np.ndarray]
    times: np.ndarray
    baseline_covariates: np.ndarray
    donor_ids: np.ndarray
    cohort_ids: np.ndarray
    trajectory_ids: np.ndarray
    history_ids: np.ndarray
    targets: np.ndarray
    doses: np.ndarray
    sequences: np.ndarray
    destructive_score: np.ndarray
    destructive_label: np.ndarray
    feature_names: dict[str, list[str]]
    baseline_names: list[str]
    quality_scores: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.trajectory_ids)


FEATURE_NAMES: dict[str, list[str]] = {
    "rna": [
        "XBP1", "ATF4", "ATF6", "HSPA5", "DDIT3", "NQO1", "IL6",
        "PPARGC1A", "BCL2", "CASP3",
    ],
    "imaging": [
        "mitochondrial_fragmentation", "calcium_burst", "er_expansion",
        "nuclear_condensation", "organelle_contact", "lysosomal_burden",
        "cell_motility", "morphologic_stress_index",
    ],
    "reporter": ["XBP1_reporter", "ATF4_reporter", "ATF6_reporter"],
    "phosphoprotein": [
        "pIRE1a", "pPERK", "pEIF2a", "pJNK", "pAKT", "pAMPK", "pSTAT3", "pSRC3",
    ],
    "metabolome": ["ATP", "NADH", "lactate", "glutathione", "succinate", "acetyl_CoA"],
    "lipidome": ["TG", "DG", "ceramide", "PC", "PE", "cholesteryl_ester"],
}
BASELINE_NAMES = ["age_scaled", "sex_binary", "baseline_viability", "stress_susceptibility", "baseline_reserve"]


def _schedule(target_index: int, dose: float, sequence: str, steps: int) -> np.ndarray:
    x = np.zeros((steps, 4), dtype=np.float32)
    if sequence == "continuous":
        x[1:, target_index] = dose
    elif sequence == "pulse_recovery":
        x[1:3, target_index] = dose
        x[4:, 3] = 0.55
    elif sequence == "delayed_rescue":
        x[1:4, target_index] = dose
        x[5:, 3] = 0.9
    elif sequence == "stress_rescue_stress":
        x[1:3, target_index] = dose
        x[3:5, 3] = 0.9
        x[5:, target_index] = dose * 0.8
    else:
        raise ValueError(f"unknown sequence {sequence}")
    return x


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate y over x using the trapezoidal rule on NumPy 1.26+.

    The Intel-macOS compatibility environment intentionally pins NumPy 1.26.4,
    which does not expose ``np.trapezoid``. Implementing the elementary rule
    directly keeps the numerical calculation identical across supported NumPy
    versions without relying on a version-specific alias.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y and x must be one-dimensional arrays of equal length")
    if y.shape[0] < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def generate_multimodal_dynamic_data(
    config: MultimodalDynamicConfig | None = None,
) -> MultimodalDynamicData:
    """Generate a deterministic software fixture for multimodal dynamic benchmarking.

    The fixture intentionally contains an early imaging/reporting signal that partially
    resolves later destructive commitment after the latest RNA snapshot has adapted.
    It exists only to verify model plumbing and gate behavior, not as biological evidence.
    """

    cfg = config or MultimodalDynamicConfig()
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)
    targets = ["IRE1_XBP1", "PERK_ATF4", "ATF6"]
    doses = [0.55, 1.0, 1.65]
    sequences = ["continuous", "pulse_recovery", "delayed_rescue", "stress_rescue_stress"]

    donor_sus = rng.normal(1.0, 0.13, size=cfg.n_donors)
    donor_reserve = np.clip(rng.normal(0.95, 0.09, size=cfg.n_donors), 0.68, 1.18)
    donor_age = np.clip(rng.normal(0.52, 0.18, size=cfg.n_donors), 0.05, 0.95)
    donor_sex = rng.integers(0, 2, size=cfg.n_donors)
    cohort_shifts = np.linspace(-0.12, 0.16, cfg.n_cohorts)

    rows: dict[str, list[np.ndarray]] = {m: [] for m in MODALITY_ORDER}
    masks: dict[str, list[np.ndarray]] = {m: [] for m in MODALITY_ORDER}
    qualities: dict[str, list[float]] = {m: [] for m in MODALITY_ORDER}
    times_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    donor_rows: list[str] = []
    cohort_rows: list[str] = []
    trajectory_rows: list[str] = []
    history_rows: list[str] = []
    target_rows: list[str] = []
    dose_rows: list[float] = []
    sequence_rows: list[str] = []
    score_rows: list[float] = []

    history_counter = 0
    trajectory_counter = 0
    for target_idx, target in enumerate(targets):
        for dose in doses:
            for sequence in sequences:
                history_id = f"MMH{history_counter:03d}_{target}_{dose:g}_{sequence}"
                history_counter += 1
                sched = _schedule(target_idx, dose, sequence, cfg.steps)
                for rep in range(cfg.replicates_per_history):
                    donor_idx = (history_counter * 5 + rep * 7) % cfg.n_donors
                    cohort_idx = donor_idx % cfg.n_cohorts
                    susceptibility = float(donor_sus[donor_idx])
                    reserve0 = float(donor_reserve[donor_idx])
                    baseline_viability = float(np.clip(rng.normal(0.94, 0.025), 0.82, 1.0))
                    baseline = np.asarray(
                        [
                            donor_age[donor_idx],
                            donor_sex[donor_idx],
                            baseline_viability,
                            susceptibility,
                            reserve0,
                        ],
                        dtype=np.float32,
                    )
                    dt = rng.uniform(7.5, 16.5, size=cfg.steps - 1)
                    times = np.concatenate([[0.0], np.cumsum(dt)]).astype(np.float32)

                    adaptation = np.asarray(rng.normal(0.09, 0.02, size=3), dtype=float)
                    damage = float(max(0.0, rng.normal(0.025, 0.01)))
                    commitment = float(max(0.0, rng.normal(0.01, 0.005)))
                    reserve = reserve0
                    inflammatory_memory = 0.02
                    pulse_memory = 0.0
                    order_memory = 0.0
                    prior_stress = 0.0
                    prior_rescue = 0.0
                    early_imaging_vulnerability = float(np.clip(rng.normal(0.32, 0.15) * susceptibility, 0.02, 0.82))

                    modality_arrays = {
                        m: np.zeros((cfg.steps, len(FEATURE_NAMES[m])), dtype=np.float32)
                        for m in MODALITY_ORDER
                    }

                    for step in range(cfg.steps):
                        stress_vec = sched[step, :3].astype(float)
                        stress = float(stress_vec.sum()) * susceptibility
                        rescue = float(sched[step, 3])
                        if step > 0:
                            delta = float(times[step] - times[step - 1]) / 12.0
                            stress_change = stress - prior_stress
                            pulse_memory = 0.72 * pulse_memory + max(stress_change, 0.0) * (0.75 + 0.25 * susceptibility)
                            order_memory *= 0.84
                            if rescue > 0 and prior_stress > 0:
                                order_memory -= 0.32 * rescue * prior_stress
                            if stress > 0 and prior_rescue > 0:
                                order_memory += 0.58 * stress * prior_rescue
                            adaptation += delta * (
                                0.31 * stress_vec * susceptibility
                                - 0.24 * adaptation
                                - 0.07 * damage
                                + 0.12 * rescue * (0.30 - adaptation)
                            )
                            overload = max(0.0, stress - (0.72 + 0.26 * adaptation.mean()))
                            damage += delta * (
                                0.19 * overload
                                + 0.025 * stress**2
                                + 0.11 * max(pulse_memory - 0.45, 0.0)
                                + 0.09 * max(order_memory, 0.0)
                                - 0.11 * rescue * reserve0
                                - 0.035 * reserve
                                + (0.085 * early_imaging_vulnerability if step >= cfg.context_steps else 0.0)
                            )
                            damage = float(max(0.0, damage))
                            commitment += delta * (
                                0.075 * max(damage - 0.22, 0.0)
                                + 0.065 * max(pulse_memory - 0.55, 0.0)
                                + 0.105 * max(order_memory, 0.0)
                                - 0.055 * rescue
                            )
                            commitment = float(max(0.0, commitment))
                            reserve += delta * (0.045 * rescue - 0.095 * damage - 0.035 * stress)
                            reserve = float(np.clip(reserve, 0.05, 1.15))
                            inflammatory_memory = max(
                                0.0,
                                0.86 * inflammatory_memory + 0.14 * damage + 0.10 * max(order_memory, 0.0),
                            )
                        prior_stress = stress
                        prior_rescue = rescue

                        early_weight = 1.0 if step <= 2 else 0.35
                        # Reporters retain rapid pathway dynamics.
                        reporter = np.asarray(
                            [
                                0.12 + 0.70 * adaptation[0] + 0.43 * stress_vec[0] + 0.10 * pulse_memory + early_weight * 0.10 * early_imaging_vulnerability,
                                0.11 + 0.74 * adaptation[1] + 0.42 * stress_vec[1] + 0.17 * damage + early_weight * 0.08 * early_imaging_vulnerability,
                                0.10 + 0.70 * adaptation[2] + 0.40 * stress_vec[2] - 0.05 * damage,
                            ]
                        )
                        # Imaging exposes transient history that can normalize by the latest RNA snapshot.
                        imaging = np.asarray(
                            [
                                0.12 + 0.57 * damage + early_weight * (0.42 * pulse_memory + 0.58 * early_imaging_vulnerability),
                                0.08 + early_weight * (0.62 * pulse_memory + 0.72 * early_imaging_vulnerability) + 0.26 * max(order_memory, 0.0),
                                0.16 + 0.46 * adaptation.mean() + 0.18 * stress,
                                0.05 + 0.52 * commitment + 0.30 * damage,
                                0.83 - 0.48 * damage - early_weight * 0.16 * pulse_memory,
                                0.10 + 0.50 * damage + 0.20 * inflammatory_memory,
                                0.82 - 0.43 * damage - 0.22 * commitment,
                                0.10 + early_weight * (0.45 * pulse_memory + 0.52 * early_imaging_vulnerability) + 0.31 * damage,
                            ]
                        )
                        # RNA reflects current adaptive state more strongly than transient pulse history.
                        rna = np.asarray(
                            [
                                0.17 + 0.73 * adaptation[0] + 0.15 * damage,
                                0.14 + 0.75 * adaptation[1] + 0.18 * damage,
                                0.13 + 0.71 * adaptation[2] - 0.06 * damage,
                                0.78 + 0.18 * adaptation.mean() - 0.48 * damage,
                                0.08 + 0.55 * commitment + 0.42 * damage,
                                0.58 + 0.30 * reserve - 0.28 * damage,
                                0.08 + 0.50 * inflammatory_memory,
                                0.64 * reserve - 0.23 * damage,
                                0.58 - 0.29 * commitment + 0.10 * rescue,
                                0.05 + 0.67 * commitment + 0.38 * damage,
                            ]
                        )
                        phospho = np.asarray(
                            [
                                reporter[0] * 0.86,
                                reporter[1] * 0.84,
                                reporter[1] * 0.76 + 0.16 * damage,
                                0.10 + 0.50 * inflammatory_memory + 0.25 * damage,
                                0.62 * reserve,
                                0.55 * reserve - 0.18 * stress,
                                0.08 + 0.48 * inflammatory_memory,
                                0.22 + 0.35 * adaptation.mean() + 0.18 * order_memory,
                            ]
                        )
                        metabolome = np.asarray(
                            [
                                0.70 * reserve - 0.30 * damage,
                                0.42 + 0.22 * stress + 0.15 * damage,
                                0.12 + 0.52 * damage + 0.22 * stress,
                                0.60 * reserve - 0.32 * inflammatory_memory,
                                0.12 + 0.43 * damage,
                                0.52 * reserve - 0.15 * stress,
                            ]
                        )
                        lipidome = np.asarray(
                            [
                                0.12 + 0.48 * damage + 0.14 * stress,
                                0.10 + 0.37 * damage,
                                0.08 + 0.58 * commitment + 0.26 * inflammatory_memory,
                                0.56 * reserve - 0.19 * damage,
                                0.50 * reserve - 0.12 * damage,
                                0.10 + 0.33 * damage + 0.16 * cohort_shifts[cohort_idx],
                            ]
                        )

                        for modality, values, scale in [
                            ("rna", rna, 0.035),
                            ("imaging", imaging, 0.030),
                            ("reporter", reporter, 0.025),
                            ("phosphoprotein", phospho, 0.040),
                            ("metabolome", metabolome, 0.040),
                            ("lipidome", lipidome, 0.040),
                        ]:
                            noisy = values + cohort_shifts[cohort_idx] * 0.08 + rng.normal(0.0, scale, size=len(values))
                            modality_arrays[modality][step] = np.clip(noisy, -0.20, 1.65).astype(np.float32)

                    # Later destructive commitment depends strongly on transient early dynamics.
                    early_img = modality_arrays["imaging"][: cfg.context_steps]
                    early_rep = modality_arrays["reporter"][: cfg.context_steps]
                    early_calcium_auc = float(_trapezoid_integral(early_img[:, 1], times[: cfg.context_steps]) / max(times[cfg.context_steps - 1], 1.0))
                    early_fragment_peak = float(np.max(early_img[:, 0]))
                    reporter_dispersion = float(np.std(early_rep[:, 0] - early_rep[:, 1]))
                    score = float(
                        _sigmoid(
                            -2.0
                            + 2.0 * damage
                            + 2.2 * commitment
                            + 1.75 * early_calcium_auc
                            + 1.35 * early_fragment_peak
                            + 1.35 * early_imaging_vulnerability
                            + 0.95 * reporter_dispersion
                            + 0.55 * (susceptibility - 1.0)
                            + 0.75 * cohort_shifts[cohort_idx]
                            - 0.85 * reserve
                        )
                    )
                    score = float(np.clip(score + rng.normal(0.0, 0.025), 0.01, 0.99))

                    # Informative baseline missingness. Imaging/reporters are mostly present early;
                    # destructive late omics become more likely to be missing as viability falls.
                    for modality in MODALITY_ORDER:
                        q = float(np.clip(rng.normal(0.92, 0.05), 0.68, 1.0))
                        qualities[modality].append(q)
                        base_missing = {
                            "rna": 0.03,
                            "imaging": 0.05,
                            "reporter": 0.04,
                            "phosphoprotein": 0.10,
                            "metabolome": 0.12,
                            "lipidome": 0.12,
                        }[modality]
                        m = np.ones(cfg.steps, dtype=np.float32)
                        for step in range(cfg.steps):
                            late_penalty = 0.0
                            if step >= cfg.context_steps and modality in LATE_OMICS_MODALITIES:
                                late_penalty = 0.18 * score
                            prob_missing = np.clip(base_missing + late_penalty + 0.10 * (1.0 - q), 0.0, 0.65)
                            if rng.random() < prob_missing:
                                m[step] = 0.0
                        # Preserve at least one observed context time per required modality.
                        if m[: cfg.context_steps].sum() == 0:
                            m[cfg.context_steps - 1] = 1.0
                        masks[modality].append(m)
                        rows[modality].append(modality_arrays[modality])

                    times_rows.append(times)
                    baseline_rows.append(baseline)
                    donor_rows.append(f"D{donor_idx:02d}")
                    cohort_rows.append(f"C{cohort_idx}")
                    trajectory_rows.append(f"MMT{trajectory_counter:04d}")
                    history_rows.append(history_id)
                    target_rows.append(target)
                    dose_rows.append(float(dose))
                    sequence_rows.append(sequence)
                    score_rows.append(score)
                    trajectory_counter += 1

    score_array = np.asarray(score_rows, dtype=np.float32)
    # Fixed biological-style threshold keeps label meaning stable across splits.
    labels = (score_array >= 0.50).astype(np.int64)
    return MultimodalDynamicData(
        modalities={m: np.stack(rows[m]).astype(np.float32) for m in MODALITY_ORDER},
        observed_masks={m: np.stack(masks[m]).astype(np.float32) for m in MODALITY_ORDER},
        times=np.stack(times_rows).astype(np.float32),
        baseline_covariates=np.stack(baseline_rows).astype(np.float32),
        donor_ids=np.asarray(donor_rows, dtype=str),
        cohort_ids=np.asarray(cohort_rows, dtype=str),
        trajectory_ids=np.asarray(trajectory_rows, dtype=str),
        history_ids=np.asarray(history_rows, dtype=str),
        targets=np.asarray(target_rows, dtype=str),
        doses=np.asarray(dose_rows, dtype=np.float32),
        sequences=np.asarray(sequence_rows, dtype=str),
        destructive_score=score_array,
        destructive_label=labels,
        feature_names={k: list(v) for k, v in FEATURE_NAMES.items()},
        baseline_names=list(BASELINE_NAMES),
        quality_scores={m: np.asarray(qualities[m], dtype=np.float32) for m in MODALITY_ORDER},
    )


def save_external_multimodal_npz(data: MultimodalDynamicData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "times": data.times,
        "baseline_covariates": data.baseline_covariates,
        "donor_ids": data.donor_ids,
        "cohort_ids": data.cohort_ids,
        "trajectory_ids": data.trajectory_ids,
        "history_ids": data.history_ids,
        "targets": data.targets,
        "doses": data.doses,
        "sequences": data.sequences,
        "destructive_score": data.destructive_score,
        "destructive_label": data.destructive_label,
        "baseline_names": np.asarray(data.baseline_names, dtype=str),
    }
    for modality in MODALITY_ORDER:
        payload[f"modality__{modality}"] = data.modalities[modality]
        payload[f"mask__{modality}"] = data.observed_masks[modality]
        payload[f"features__{modality}"] = np.asarray(data.feature_names[modality], dtype=str)
        payload[f"quality__{modality}"] = data.quality_scores[modality]
    np.savez_compressed(path, **payload)
    return path


def load_external_multimodal_npz(path: str | Path) -> MultimodalDynamicData:
    src = np.load(Path(path), allow_pickle=False)
    required = [
        "times", "baseline_covariates", "donor_ids", "cohort_ids", "trajectory_ids",
        "history_ids", "targets", "doses", "sequences", "destructive_score", "destructive_label",
    ]
    missing = [key for key in required if key not in src]
    for modality in MODALITY_ORDER:
        for prefix in ["modality__", "mask__", "features__"]:
            key = f"{prefix}{modality}"
            if key not in src:
                missing.append(key)
    if missing:
        raise ValueError(f"multimodal benchmark NPZ is missing keys: {sorted(set(missing))}")
    modalities = {m: src[f"modality__{m}"].astype(np.float32) for m in MODALITY_ORDER}
    masks = {m: src[f"mask__{m}"].astype(np.float32) for m in MODALITY_ORDER}
    n = len(src["trajectory_ids"])
    steps = src["times"].shape[1]
    for m in MODALITY_ORDER:
        if modalities[m].shape[:2] != (n, steps):
            raise ValueError(f"{m} shape does not align with trajectory/time axes")
        if masks[m].shape != (n, steps):
            raise ValueError(f"{m} mask shape does not align with trajectory/time axes")
    return MultimodalDynamicData(
        modalities=modalities,
        observed_masks=masks,
        times=src["times"].astype(np.float32),
        baseline_covariates=src["baseline_covariates"].astype(np.float32),
        donor_ids=src["donor_ids"].astype(str),
        cohort_ids=src["cohort_ids"].astype(str),
        trajectory_ids=src["trajectory_ids"].astype(str),
        history_ids=src["history_ids"].astype(str),
        targets=src["targets"].astype(str),
        doses=src["doses"].astype(np.float32),
        sequences=src["sequences"].astype(str),
        destructive_score=src["destructive_score"].astype(np.float32),
        destructive_label=src["destructive_label"].astype(np.int64),
        feature_names={m: src[f"features__{m}"].astype(str).tolist() for m in MODALITY_ORDER},
        baseline_names=src["baseline_names"].astype(str).tolist() if "baseline_names" in src else list(BASELINE_NAMES),
        quality_scores={m: src[f"quality__{m}"].astype(np.float32) if f"quality__{m}" in src else np.ones(n, dtype=np.float32) for m in MODALITY_ORDER},
    )


def history_split(data: MultimodalDynamicData, seed: int = 140) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    histories = np.unique(data.history_ids)
    histories = histories.copy()
    rng.shuffle(histories)
    n_test = max(1, int(round(0.20 * len(histories))))
    n_val = max(1, int(round(0.17 * len(histories))))
    test_h = set(histories[:n_test].tolist())
    val_h = set(histories[n_test:n_test + n_val].tolist())
    train_h = set(histories[n_test + n_val:].tolist())
    return {
        "train": np.where(np.isin(data.history_ids, list(train_h)))[0],
        "validation": np.where(np.isin(data.history_ids, list(val_h)))[0],
        "test": np.where(np.isin(data.history_ids, list(test_h)))[0],
    }


def split_audit(data: MultimodalDynamicData, split: Mapping[str, np.ndarray]) -> dict[str, Any]:
    train, val, test = (np.asarray(split[k], dtype=int) for k in ["train", "validation", "test"])
    htrain, hval, htest = (set(data.history_ids[idx].tolist()) for idx in [train, val, test])
    dtrain, dval, dtest = (set(data.donor_ids[idx].tolist()) for idx in [train, val, test])
    return {
        "mode": "perturbation_history",
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_test": int(len(test)),
        "train_histories": int(len(htrain)),
        "validation_histories": int(len(hval)),
        "test_histories": int(len(htest)),
        "history_overlap_train_validation": sorted(htrain & hval),
        "history_overlap_train_test": sorted(htrain & htest),
        "history_overlap_validation_test": sorted(hval & htest),
        "history_split_valid": not bool((htrain & hval) or (htrain & htest) or (hval & htest)),
        "donor_overlap_train_validation": sorted(dtrain & dval),
        "donor_overlap_train_test": sorted(dtrain & dtest),
        "donor_holdout_enforced": False,
        "donor_overlap_expected": True,
        "primary_generalization_target": "multimodal prediction on unseen perturbation histories in potentially observed donors",
        "history_leakage": bool((htrain & hval) or (htrain & htest) or (hval & htest)),
    }


def _late_target(data: MultimodalDynamicData, indices: np.ndarray) -> tuple[np.ndarray, dict[str, slice]]:
    parts: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    start = 0
    for modality in LATE_OMICS_MODALITIES:
        values = data.modalities[modality][indices, -1]
        parts.append(values)
        stop = start + values.shape[1]
        slices[modality] = slice(start, stop)
        start = stop
    return np.concatenate(parts, axis=1).astype(np.float32), slices


def _fit_scalers(data: MultimodalDynamicData, train_idx: np.ndarray, context_steps: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for modality in MODALITY_ORDER:
        x = data.modalities[modality][train_idx, :context_steps]
        m = data.observed_masks[modality][train_idx, :context_steps][..., None]
        denom = np.maximum(m.sum(axis=(0, 1)), 1.0)
        mean = (x * m).sum(axis=(0, 1)) / denom
        var = (((x - mean) ** 2) * m).sum(axis=(0, 1)) / denom
        scalers[modality] = (mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-5)).astype(np.float32))
    b = data.baseline_covariates[train_idx]
    scalers["baseline"] = (b.mean(axis=0).astype(np.float32), np.maximum(b.std(axis=0), 1e-4).astype(np.float32))
    late, _ = _late_target(data, train_idx)
    scalers["late"] = (late.mean(axis=0).astype(np.float32), np.maximum(late.std(axis=0), 1e-4).astype(np.float32))
    return scalers


class MultimodalTrajectoryDataset(Dataset):
    def __init__(
        self,
        data: MultimodalDynamicData,
        indices: np.ndarray,
        scalers: Mapping[str, tuple[np.ndarray, np.ndarray]],
        context_steps: int,
        donor_map: Mapping[str, int],
        cohort_map: Mapping[str, int],
        override_masks: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=int)
        self.scalers = scalers
        self.context_steps = context_steps
        self.donor_map = dict(donor_map)
        self.cohort_map = dict(cohort_map)
        self.override_masks = override_masks
        late, _ = _late_target(data, self.indices)
        mean, std = scalers["late"]
        self.late_targets = ((late - mean) / std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        output: dict[str, torch.Tensor] = {}
        for modality in MODALITY_ORDER:
            mean, std = self.scalers[modality]
            x = (self.data.modalities[modality][idx, : self.context_steps] - mean) / std
            if self.override_masks is None:
                mask = self.data.observed_masks[modality][idx, : self.context_steps]
            else:
                mask = self.override_masks[modality][idx, : self.context_steps]
            x = x * mask[:, None]
            output[f"x__{modality}"] = torch.tensor(x, dtype=torch.float32)
            output[f"mask__{modality}"] = torch.tensor(mask, dtype=torch.float32)
        bmean, bstd = self.scalers["baseline"]
        output["baseline"] = torch.tensor((self.data.baseline_covariates[idx] - bmean) / bstd, dtype=torch.float32)
        t = self.data.times[idx, : self.context_steps]
        tscale = max(float(t[-1]), 1.0)
        output["times"] = torch.tensor(t / tscale, dtype=torch.float32)
        output["label"] = torch.tensor(float(self.data.destructive_label[idx]), dtype=torch.float32)
        output["score"] = torch.tensor(float(self.data.destructive_score[idx]), dtype=torch.float32)
        output["late_omics"] = torch.tensor(self.late_targets[item], dtype=torch.float32)
        output["donor"] = torch.tensor(self.donor_map.get(str(self.data.donor_ids[idx]), len(self.donor_map)), dtype=torch.long)
        output["cohort"] = torch.tensor(self.cohort_map.get(str(self.data.cohort_ids[idx]), len(self.cohort_map)), dtype=torch.long)
        output["row_index"] = torch.tensor(idx, dtype=torch.long)
        return output


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, probabilistic: bool = True) -> None:
        super().__init__()
        self.probabilistic = probabilistic
        out = latent_dim * 2 if probabilistic else latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, max(latent_dim * 2, 16)),
            nn.GELU(),
            nn.LayerNorm(max(latent_dim * 2, 16)),
            nn.Linear(max(latent_dim * 2, 16), out),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        y = self.net(x)
        if not self.probabilistic:
            return y
        mu, logvar = y.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -5.0, 4.0)


class ProductOfExpertsFusion(nn.Module):
    def forward(
        self,
        experts: list[tuple[torch.Tensor, torch.Tensor]],
        masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = experts[0][0].shape
        total_precision = torch.ones(shape, device=experts[0][0].device)
        weighted_mu = torch.zeros(shape, device=experts[0][0].device)
        for (mu, logvar), mask in zip(experts, masks):
            precision = torch.exp(-logvar) * mask.unsqueeze(-1)
            total_precision = total_precision + precision
            weighted_mu = weighted_mu + mu * precision
        fused_mu = weighted_mu / total_precision
        fused_logvar = -torch.log(total_precision)
        return fused_mu, fused_logvar


class MixtureOfExpertsFusion(nn.Module):
    def __init__(self, latent_dim: int, n_modalities: int) -> None:
        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(n_modalities)])

    def forward(self, embeddings: list[torch.Tensor], masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = []
        for emb, gate, mask in zip(embeddings, self.gates, masks):
            score = gate(emb).squeeze(-1)
            score = score.masked_fill(mask <= 0, -1e4)
            scores.append(score)
        stacked = torch.stack(scores, dim=-1)
        weights = torch.softmax(stacked, dim=-1)
        fused = sum(emb * weights[..., i : i + 1] for i, emb in enumerate(embeddings))
        return fused, weights


class StaticPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim // 2, 1)
        self.score = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {"logit": self.logit(h).squeeze(-1), "score": torch.sigmoid(self.score(h).squeeze(-1)), "latent": h}


class EarlyImagingReporterGRU(nn.Module):
    def __init__(self, imaging_dim: int, reporter_dim: int, baseline_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(imaging_dim + reporter_dim + 1, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + baseline_dim, hidden_dim), nn.GELU())
        self.logit = nn.Linear(hidden_dim, 1)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat([batch["x__imaging"], batch["x__reporter"], batch["times"].unsqueeze(-1)], dim=-1)
        _, h = self.gru(x)
        z = self.head(torch.cat([h[-1], batch["baseline"]], dim=-1))
        return {"logit": self.logit(z).squeeze(-1), "score": torch.sigmoid(self.score(z).squeeze(-1)), "latent": z}


class CausaFluxMultimodalDynamic(nn.Module):
    def __init__(
        self,
        feature_dims: Mapping[str, int],
        baseline_dim: int,
        late_dim: int,
        n_donors: int,
        n_cohorts: int,
        latent_dim: int = 24,
        hidden_dim: int = 48,
        fusion: str = "poe",
        modality_dropout: float = 0.25,
        exclude_modalities: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.modalities = [m for m in MODALITY_ORDER if m not in set(exclude_modalities)]
        self.fusion_name = fusion
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict()
        if fusion == "poe":
            for m in self.modalities:
                self.encoders[m] = ModalityEncoder(feature_dims[m], latent_dim, probabilistic=True)
            self.fusion = ProductOfExpertsFusion()
        elif fusion == "moe":
            for m in self.modalities:
                self.encoders[m] = ModalityEncoder(feature_dims[m], latent_dim, probabilistic=False)
            self.fusion = MixtureOfExpertsFusion(latent_dim, len(self.modalities))
        else:
            raise ValueError("fusion must be 'poe' or 'moe'")
        self.temporal = nn.GRU(latent_dim + 1, hidden_dim, batch_first=True)
        self.donor_embedding = nn.Embedding(n_donors + 1, 6)
        self.cohort_embedding = nn.Embedding(n_cohorts + 1, 4)
        combined = hidden_dim + baseline_dim + 6 + 4
        self.context = nn.Sequential(nn.Linear(combined, hidden_dim), nn.GELU(), nn.Dropout(0.10))
        self.logit = nn.Linear(hidden_dim, 1)
        self.score = nn.Linear(hidden_dim, 1)
        self.cross_modal_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, late_dim))

    def _drop_masks(self, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return masks
        result = []
        for mask in masks:
            keep = (torch.rand(mask.shape[0], 1, device=mask.device) > self.modality_dropout).float()
            result.append(mask * keep)
        stacked = torch.stack(result, dim=0).sum(dim=0)
        none = stacked <= 0
        if none.any():
            result[0] = torch.where(none, masks[0], result[0])
        return result

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        masks = [batch[f"mask__{m}"] for m in self.modalities]
        masks = self._drop_masks(masks)
        if self.fusion_name == "poe":
            experts = [self.encoders[m](batch[f"x__{m}"]) for m in self.modalities]
            fused, aux = self.fusion(experts, masks)
        else:
            embeddings = [self.encoders[m](batch[f"x__{m}"]) for m in self.modalities]
            fused, aux = self.fusion(embeddings, masks)
        temporal_input = torch.cat([fused, batch["times"].unsqueeze(-1)], dim=-1)
        _, h = self.temporal(temporal_input)
        donor = self.donor_embedding(batch["donor"])
        cohort = self.cohort_embedding(batch["cohort"])
        z = self.context(torch.cat([h[-1], batch["baseline"], donor, cohort], dim=-1))
        return {
            "logit": self.logit(z).squeeze(-1),
            "score": torch.sigmoid(self.score(z).squeeze(-1)),
            "late_omics": self.cross_modal_decoder(z),
            "latent": z,
            "fusion_aux": aux,
        }


def _static_features(batch: Mapping[str, torch.Tensor], model_name: str) -> torch.Tensor:
    baseline = batch["baseline"]
    if model_name == "BaselineCovariatesMLP":
        return baseline
    if model_name == "LatestRNAMLP":
        return torch.cat([baseline, batch["x__rna"][:, -1], batch["mask__rna"][:, -1:].float()], dim=-1)
    if model_name == "StaticMultimodalFusion":
        parts = [baseline]
        for modality in MODALITY_ORDER:
            parts += [batch[f"x__{modality}"][:, -1], batch[f"mask__{modality}"][:, -1:].float()]
        return torch.cat(parts, dim=-1)
    raise KeyError(model_name)


def _model_for_name(
    name: str,
    data: MultimodalDynamicData,
    cfg: MultimodalDynamicConfig,
    late_dim: int,
    donor_map: Mapping[str, int],
    cohort_map: Mapping[str, int],
) -> nn.Module:
    feature_dims = {m: data.modalities[m].shape[-1] for m in MODALITY_ORDER}
    if name == "BaselineCovariatesMLP":
        return StaticPredictor(data.baseline_covariates.shape[1], cfg.hidden_dim)
    if name == "LatestRNAMLP":
        return StaticPredictor(data.baseline_covariates.shape[1] + feature_dims["rna"] + 1, cfg.hidden_dim)
    if name == "StaticMultimodalFusion":
        dim = data.baseline_covariates.shape[1] + sum(feature_dims[m] + 1 for m in MODALITY_ORDER)
        return StaticPredictor(dim, cfg.hidden_dim)
    if name == "EarlyImagingReporterGRU":
        return EarlyImagingReporterGRU(feature_dims["imaging"], feature_dims["reporter"], data.baseline_covariates.shape[1], cfg.hidden_dim)
    if name == "CausaFluxPoEDynamic":
        return CausaFluxMultimodalDynamic(feature_dims, data.baseline_covariates.shape[1], late_dim, len(donor_map), len(cohort_map), cfg.latent_dim, cfg.hidden_dim, "poe", cfg.modality_dropout)
    if name == "CausaFluxMoEDynamic":
        return CausaFluxMultimodalDynamic(feature_dims, data.baseline_covariates.shape[1], late_dim, len(donor_map), len(cohort_map), cfg.latent_dim, cfg.hidden_dim, "moe", cfg.modality_dropout)
    if name == "CausaFluxPoE_NoImagingReporter":
        return CausaFluxMultimodalDynamic(feature_dims, data.baseline_covariates.shape[1], late_dim, len(donor_map), len(cohort_map), cfg.latent_dim, cfg.hidden_dim, "poe", cfg.modality_dropout, exclude_modalities=["imaging", "reporter"])
    raise KeyError(name)


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model: nn.Module, batch: Mapping[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    if name in BASELINE_MODELS:
        return model(_static_features(batch, name))
    return model(batch)


def _train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: MultimodalDynamicConfig,
    output_dir: Path,
) -> tuple[nn.Module, pd.DataFrame]:
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_n = 0
        for raw in train_loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            out = _forward(model, batch, name)
            loss = bce(out["logit"], batch["label"]) + 0.35 * mse(out["score"], batch["score"])
            if "late_omics" in out:
                loss = loss + 0.20 * mse(out["late_omics"], batch["late_omics"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch["label"])
            train_n += len(batch["label"])
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for raw in val_loader:
                batch = _move(raw, device)
                out = _forward(model, batch, name)
                loss = bce(out["logit"], batch["label"]) + 0.35 * mse(out["score"], batch["score"])
                if "late_omics" in out:
                    loss = loss + 0.20 * mse(out["late_omics"], batch["late_omics"])
                val_loss += float(loss) * len(batch["label"])
                val_n += len(batch["label"])
        row = {"epoch": epoch + 1, "train_loss": train_loss / max(train_n, 1), "validation_loss": val_loss / max(val_n, 1)}
        history.append(row)
        if row["validation_loss"] < best - 1e-5:
            best = row["validation_loss"]
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": name, "state_dict": model.state_dict(), "config": asdict(cfg)}, output_dir / f"{name}.pt")
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / f"{name}_training_history.csv", index=False)
    return model, frame


def _collect_predictions(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    device_name: str,
) -> dict[str, np.ndarray]:
    device = torch.device(device_name)
    model.eval()
    store: dict[str, list[np.ndarray]] = {"logit": [], "score": [], "label": [], "row_index": [], "late_omics": [], "latent": []}
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            out = _forward(model, batch, name)
            store["logit"].append(out["logit"].cpu().numpy())
            store["score"].append(out["score"].cpu().numpy())
            store["label"].append(batch["label"].cpu().numpy())
            store["row_index"].append(batch["row_index"].cpu().numpy())
            if "late_omics" in out:
                store["late_omics"].append(out["late_omics"].cpu().numpy())
            store["latent"].append(out["latent"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) if v else np.empty((0,)) for k, v in store.items()}


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.exp(np.linspace(math.log(0.35), math.log(3.0), 80))
    best_t, best = 1.0, float("inf")
    for t in candidates:
        p = _sigmoid(logits / t)
        value = log_loss(labels, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        if value < best:
            best, best_t = value, float(t)
    return best_t


def _metrics(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict[str, float]:
    p = np.asarray(_sigmoid(logits / temperature), dtype=float)
    pred = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(labels, p)) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "log_loss": float(log_loss(labels, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, p)),
        "auc": auc,
        "accuracy": float(accuracy_score(labels, pred)),
        "temperature": float(temperature),
    }


def _bootstrap_metrics(
    data: MultimodalDynamicData,
    row_indices: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    donors = np.unique(data.donor_ids[row_indices])
    rows: list[dict[str, float]] = []
    for b in range(replicates):
        sampled = rng.choice(donors, size=len(donors), replace=True)
        picks: list[int] = []
        for donor in sampled:
            local = np.where(data.donor_ids[row_indices] == donor)[0]
            picks.extend(local.tolist())
        if not picks:
            continue
        y = labels[picks]
        if len(np.unique(y)) < 2:
            continue
        m = _metrics(logits[picks], y, temperature)
        rows.append({"bootstrap": b, **{k: m[k] for k in ["log_loss", "brier", "auc", "accuracy"]}})
    return pd.DataFrame(rows)


def _residual_coverage(
    val_pred: np.ndarray,
    val_true: np.ndarray,
    test_pred: np.ndarray,
    test_true: np.ndarray,
    slices: Mapping[str, slice],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for modality, slc in slices.items():
        residual = val_true[:, slc] - val_pred[:, slc]
        scale = np.maximum(np.std(residual, axis=0), 1e-4)
        test_residual = test_true[:, slc] - test_pred[:, slc]
        for nominal, z in [(0.50, 0.67449), (0.80, 1.28155), (0.90, 1.64485), (0.95, 1.95996)]:
            covered = np.abs(test_residual) <= z * scale
            rows.append({
                "modality": modality,
                "nominal_coverage": nominal,
                "observed_coverage": float(np.mean(covered)),
                "mean_interval_half_width_z": float(np.mean(z * scale)),
            })
    return pd.DataFrame(rows)


def _mnar_masks(
    data: MultimodalDynamicData,
    scenario: str,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {m: data.observed_masks[m].copy() for m in MODALITY_ORDER}
    n = len(data)
    c = 4 if data.times.shape[1] >= 4 else data.times.shape[1]
    if scenario == "observed":
        return result
    if scenario == "MCAR_20":
        for m in MODALITY_ORDER:
            extra = rng.random((n, c)) < 0.20
            result[m][:, :c][extra] = 0.0
    elif scenario == "MNAR_destructive_imaging":
        prob = np.clip(0.10 + 0.45 * data.destructive_score, 0.0, 0.70)
        for i in range(n):
            for m in ["imaging", "reporter"]:
                if rng.random() < prob[i]:
                    step = rng.integers(0, c)
                    result[m][i, step] = 0.0
    elif scenario == "MNAR_low_quality_omics":
        for m in ["phosphoprotein", "metabolome", "lipidome"]:
            q = data.quality_scores[m]
            for i in range(n):
                if rng.random() < np.clip(0.10 + 0.65 * (1.0 - q[i]), 0.0, 0.75):
                    result[m][i, :c] = 0.0
    else:
        raise ValueError(scenario)
    # Guarantee one modality is observed at each context time.
    for i in range(n):
        for t in range(c):
            if sum(result[m][i, t] > 0 for m in MODALITY_ORDER) == 0:
                result["rna"][i, t] = 1.0
    return result


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_report(output: Path, comparison: pd.DataFrame, gate: dict[str, Any], crossmodal: pd.DataFrame, mnar: pd.DataFrame, audit: dict[str, Any]) -> Path:
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v1.7.0 multimodal dynamic benchmark</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:28px auto;padding:0 22px;color:#202124}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px;text-align:left}}th{{background:#f4f4f4}}.ok{{border-left:4px solid #2D7F78;padding:12px;background:#f2fbf8}}.warn{{border-left:4px solid #B64C4C;padding:12px;background:#fff5f3}}code{{background:#f3f3f3;padding:2px 4px}}</style></head><body>
<h1>CausaFlux v1.7.0 — Multimodal Dynamic State Model</h1>
<div class='{'ok' if gate['software_exit_gate_passed'] else 'warn'}'><strong>Software exit gate: {'PASS' if gate['software_exit_gate_passed'] else 'BLOCKED'}.</strong> {gate['interpretation']}</div>
<p>This bundled benchmark is a deterministic synthetic software fixture. It validates the multimodal architecture, missingness handling, cross-modal decoding, and gate logic; it is not biological evidence.</p>
<h2>Split audit</h2><pre>{json.dumps(audit, indent=2)}</pre>
<h2>Destructive-state prediction</h2>{comparison.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Cross-modal forecasting</h2>{crossmodal.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Missing-not-at-random sensitivity</h2>{mnar.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Exit criterion</h2><p>At least one dynamic model that uses early imaging and reporter history must improve later destructive-state prediction beyond baseline covariates, latest RNA, and static multimodal fusion. The real-data authorization remains blocked until this gate is repeated on a locked real longitudinal dataset.</p>
</body></html>"""
    path = report_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _write_figures(output: Path, comparison: pd.DataFrame, crossmodal: pd.DataFrame, mnar: pd.DataFrame) -> list[str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    exports: list[str] = []

    apply_publication_style("nature_double")
    order = comparison.sort_values("test_log_loss")["model"].tolist()
    plot = comparison.set_index("model").loc[order]
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(plot)), plot["test_log_loss"].to_numpy(), color=COLORS["blue"])
    ax.set_yticks(np.arange(len(plot)), [x.replace("CausaFlux", "CF-") for x in plot.index])
    ax.invert_yaxis(); ax.set_xlabel("Calibrated log loss ↓"); ax.set_title("Later destructive-state prediction")
    ax.spines[["top", "right"]].set_visible(False)
    exp = export_figure(fig, figure_dir / "destructive_state_prediction.png", figure_id="destructive_state_prediction", source_data=comparison, metadata={"benchmark":"synthetic_multimodal_dynamic"}, synthetic_only=True)
    plt.close(fig); exports.append(exp.manifest)

    fig, ax = plt.subplots()
    p = crossmodal[crossmodal["metric"] == "rmse"].copy()
    ax.bar(np.arange(len(p)), p["value"].to_numpy(), color=COLORS["teal"])
    ax.set_xticks(np.arange(len(p)), p["modality"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Normalized RMSE ↓"); ax.set_title("Early-state to late-omics forecasting")
    ax.spines[["top", "right"]].set_visible(False)
    exp = export_figure(fig, figure_dir / "cross_modal_forecasting.png", figure_id="cross_modal_forecasting", source_data=crossmodal, metadata={"benchmark":"synthetic_multimodal_dynamic"}, synthetic_only=True)
    plt.close(fig); exports.append(exp.manifest)

    fig, ax = plt.subplots()
    p = mnar[mnar["model"] == "CausaFluxPoEDynamic"].copy()
    ax.plot(np.arange(len(p)), p["log_loss"].to_numpy(), marker="o", color=COLORS["purple"])
    ax.set_xticks(np.arange(len(p)), p["scenario"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Calibrated log loss ↓"); ax.set_title("Missing-not-at-random sensitivity")
    ax.spines[["top", "right"]].set_visible(False)
    exp = export_figure(fig, figure_dir / "mnar_sensitivity.png", figure_id="mnar_sensitivity", source_data=mnar, metadata={"benchmark":"synthetic_multimodal_dynamic"}, synthetic_only=True)
    plt.close(fig); exports.append(exp.manifest)
    return exports


def run_multimodal_dynamic_benchmark(
    output: str | Path,
    config: MultimodalDynamicConfig | None = None,
    data: MultimodalDynamicData | None = None,
    models: Iterable[str] | None = None,
    require_gate: bool = False,
) -> dict[str, Any]:
    cfg = config or MultimodalDynamicConfig()
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = data or generate_multimodal_dynamic_data(cfg)
    split = history_split(data, cfg.seed)
    audit = split_audit(data, split)
    if not audit["history_split_valid"]:
        raise RuntimeError("history leakage detected")
    scalers = _fit_scalers(data, split["train"], cfg.context_steps)
    donor_map = {d: i for i, d in enumerate(sorted(np.unique(data.donor_ids[split["train"]]).tolist()))}
    cohort_map = {d: i for i, d in enumerate(sorted(np.unique(data.cohort_ids[split["train"]]).tolist()))}
    late_train, slices = _late_target(data, split["train"])
    late_dim = late_train.shape[1]
    datasets = {
        part: MultimodalTrajectoryDataset(data, split[part], scalers, cfg.context_steps, donor_map, cohort_map)
        for part in ["train", "validation", "test"]
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True),
        "validation": DataLoader(datasets["validation"], batch_size=max(cfg.batch_size, 64), shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=max(cfg.batch_size, 64), shuffle=False),
    }
    names = list(models) if models else list(MODEL_ORDER)
    unknown = sorted(set(names) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")

    model_dir = output / "models"
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    trained: dict[str, nn.Module] = {}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for model_index, name in enumerate(names):
        set_seed(cfg.seed + model_index * 13)
        model = _model_for_name(name, data, cfg, late_dim, donor_map, cohort_map)
        model, history = _train_one(name, model, loaders["train"], loaders["validation"], cfg, model_dir)
        trained[name] = model
        pred_val = _collect_predictions(name, model, loaders["validation"], cfg.device)
        pred_test = _collect_predictions(name, model, loaders["test"], cfg.device)
        predictions[name] = {"validation": pred_val, "test": pred_test}
        temp = _temperature(pred_val["logit"], pred_val["label"].astype(int))
        metrics = _metrics(pred_test["logit"], pred_test["label"].astype(int), temp)
        score_rmse = float(np.sqrt(np.mean((pred_test["score"] - data.destructive_score[pred_test["row_index"].astype(int)]) ** 2)))
        comparison_rows.append({
            "model": name,
            "uses_temporal_history": name not in BASELINE_MODELS,
            "uses_imaging_reporter_history": name in DYNAMIC_IMAGING_MODELS,
            "test_log_loss": metrics["log_loss"],
            "test_brier": metrics["brier"],
            "test_auc": metrics["auc"],
            "test_accuracy": metrics["accuracy"],
            "destructive_score_rmse": score_rmse,
            "temperature": metrics["temperature"],
            "epochs_completed": int(len(history)),
        })
        boot = _bootstrap_metrics(data, pred_test["row_index"].astype(int), pred_test["logit"], pred_test["label"].astype(int), temp, cfg.bootstrap_replicates, cfg.seed + 900 + model_index)
        if not boot.empty:
            boot.insert(0, "model", name)
            bootstrap_frames.append(boot)

    comparison = pd.DataFrame(comparison_rows).sort_values("test_log_loss").reset_index(drop=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(output / "donor_bootstrap_metrics.csv", index=False) if bootstrap_frames else pd.DataFrame().to_csv(output / "donor_bootstrap_metrics.csv", index=False)

    # Cross-modal decoder evaluation for PoE/MoE models.
    late_test_raw, _ = _late_target(data, split["test"])
    late_val_raw, _ = _late_target(data, split["validation"])
    lmean, lstd = scalers["late"]
    late_test_z = (late_test_raw - lmean) / lstd
    late_val_z = (late_val_raw - lmean) / lstd
    cross_rows: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []
    for name in [m for m in ["CausaFluxPoEDynamic", "CausaFluxMoEDynamic", "CausaFluxPoE_NoImagingReporter"] if m in predictions]:
        val_pred = predictions[name]["validation"]["late_omics"]
        test_pred = predictions[name]["test"]["late_omics"]
        if val_pred.size == 0 or test_pred.size == 0:
            continue
        for modality, slc in slices.items():
            rmse = float(np.sqrt(np.mean((test_pred[:, slc] - late_test_z[:, slc]) ** 2)))
            corr_values = []
            for j in range(slc.start, slc.stop):
                if np.std(test_pred[:, j]) > 1e-6 and np.std(late_test_z[:, j]) > 1e-6:
                    corr_values.append(float(np.corrcoef(test_pred[:, j], late_test_z[:, j])[0, 1]))
            cross_rows += [
                {"model": name, "modality": modality, "metric": "rmse", "value": rmse},
                {"model": name, "modality": modality, "metric": "mean_feature_correlation", "value": float(np.nanmean(corr_values)) if corr_values else float("nan")},
            ]
        cov = _residual_coverage(val_pred, late_val_z, test_pred, late_test_z, slices)
        cov.insert(0, "model", name)
        coverage_frames.append(cov)
    crossmodal = pd.DataFrame(cross_rows)
    crossmodal.to_csv(output / "cross_modal_forecasting.csv", index=False)
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    coverage.to_csv(output / "cross_modal_uncertainty_coverage.csv", index=False)

    # Explicit missing-not-at-random sensitivity.
    mnar_rows: list[dict[str, Any]] = []
    for name in [m for m in ["EarlyImagingReporterGRU", "CausaFluxPoEDynamic", "CausaFluxMoEDynamic"] if m in trained]:
        temp = float(comparison.loc[comparison.model == name, "temperature"].iloc[0])
        for scenario_index, scenario in enumerate(["observed", "MCAR_20", "MNAR_destructive_imaging", "MNAR_low_quality_omics"]):
            override = _mnar_masks(data, scenario, cfg.seed + 2000 + scenario_index)
            ds = MultimodalTrajectoryDataset(data, split["test"], scalers, cfg.context_steps, donor_map, cohort_map, override_masks=override)
            loader = DataLoader(ds, batch_size=max(cfg.batch_size, 64), shuffle=False)
            pred = _collect_predictions(name, trained[name], loader, cfg.device)
            mm = _metrics(pred["logit"], pred["label"].astype(int), temp)
            mnar_rows.append({"model": name, "scenario": scenario, **{k: mm[k] for k in ["log_loss", "brier", "auc", "accuracy"]}})
    mnar = pd.DataFrame(mnar_rows)
    if not mnar.empty:
        ref = mnar[mnar.scenario == "observed"].set_index("model")["log_loss"]
        mnar["delta_log_loss_vs_observed"] = mnar.apply(lambda r: r.log_loss - ref.get(r.model, np.nan), axis=1)
    mnar.to_csv(output / "missingness_sensitivity.csv", index=False)

    # Gate: at least one model using early imaging/reporting history beats all three baselines
    # on calibrated log loss and Brier score, with AUC no worse than the strongest baseline.
    baseline = comparison[comparison.model.isin(BASELINE_MODELS)]
    baseline_logloss = float(baseline.test_log_loss.min())
    baseline_brier = float(baseline.test_brier.min())
    baseline_auc = float(baseline.test_auc.max())
    qualifiers: list[str] = []
    gate_details: list[dict[str, Any]] = []
    for name in DYNAMIC_IMAGING_MODELS:
        row = comparison[comparison.model == name]
        if row.empty:
            continue
        r = row.iloc[0]
        passed = bool(r.test_log_loss < baseline_logloss and r.test_brier < baseline_brier and r.test_auc >= baseline_auc - 0.02)
        gate_details.append({
            "model": name,
            "passed": passed,
            "log_loss_improvement": baseline_logloss - float(r.test_log_loss),
            "brier_improvement": baseline_brier - float(r.test_brier),
            "auc_delta_vs_best_baseline": float(r.test_auc) - baseline_auc,
        })
        if passed:
            qualifiers.append(name)
    ablation_delta = None
    if "CausaFluxPoEDynamic" in comparison.model.values and "CausaFluxPoE_NoImagingReporter" in comparison.model.values:
        full = float(comparison.loc[comparison.model == "CausaFluxPoEDynamic", "test_log_loss"].iloc[0])
        ablated = float(comparison.loc[comparison.model == "CausaFluxPoE_NoImagingReporter", "test_log_loss"].iloc[0])
        ablation_delta = ablated - full
    gate = {
        "framework": "CausaFlux",
        "version": "1.7.0",
        "software_exit_gate_passed": bool(qualifiers),
        "qualifying_models": sorted(qualifiers),
        "baseline_best_log_loss": baseline_logloss,
        "baseline_best_brier": baseline_brier,
        "baseline_best_auc": baseline_auc,
        "gate_details": gate_details,
        "poe_imaging_reporter_ablation_log_loss_delta": ablation_delta,
        "foundation_pretraining_authorization": "BLOCKED_REAL_MULTIMODAL_LONGITUDINAL_GATE_REQUIRED",
        "interpretation": "At least one dynamic early-imaging/reporter model beats baseline covariates, latest RNA, and static multimodal fusion on the synthetic software fixture." if qualifiers else "No dynamic early-imaging/reporter model beats all prespecified baselines; the multimodal gate is blocked.",
        "synthetic_software_fixture": True,
    }
    json_dump(gate, output / "multimodal_exit_gate.json")
    json_dump(audit, output / "split_audit.json")
    json_dump(asdict(cfg), output / "benchmark_config.json")
    json_dump({
        "modalities": {m: {"n_features": len(data.feature_names[m]), "features": data.feature_names[m]} for m in MODALITY_ORDER},
        "late_omics_modalities": list(LATE_OMICS_MODALITIES),
        "fusion": ["product_of_experts", "mixture_of_experts"],
        "modality_dropout_training": cfg.modality_dropout,
        "donor_cohort_latent_context": True,
        "external_contract": "NPZ with explicit per-modality tensors, masks, donors, cohorts, histories, and outcomes",
    }, output / "modality_schema.json")

    # Save predictions and latent state summaries.
    pred_frames = []
    for name, blocks in predictions.items():
        testp = blocks["test"]
        temp = float(comparison.loc[comparison.model == name, "temperature"].iloc[0])
        prob = _sigmoid(testp["logit"] / temp)
        for j, row_idx in enumerate(testp["row_index"].astype(int)):
            pred_frames.append({
                "model": name,
                "trajectory_id": data.trajectory_ids[row_idx],
                "donor_id": data.donor_ids[row_idx],
                "cohort_id": data.cohort_ids[row_idx],
                "history_id": data.history_ids[row_idx],
                "target": data.targets[row_idx],
                "dose": data.doses[row_idx],
                "sequence": data.sequences[row_idx],
                "observed_destructive": int(data.destructive_label[row_idx]),
                "observed_score": float(data.destructive_score[row_idx]),
                "predicted_probability": float(prob[j]),
                "predicted_score": float(testp["score"][j]),
            })
    pd.DataFrame(pred_frames).to_csv(output / "test_predictions.csv", index=False)

    _write_figures(output, comparison, crossmodal, mnar)
    _write_report(output, comparison, gate, crossmodal, mnar, audit)

    dataset_card = f"""# CausaFlux v1.7.0 multimodal dynamic benchmark dataset card

This bundled dataset is a deterministic **synthetic software fixture** containing {len(data)} longitudinal trajectories, {len(np.unique(data.history_ids))} perturbation histories, {len(np.unique(data.donor_ids))} donors, and {len(np.unique(data.cohort_ids))} cohorts.

Modalities: {', '.join(MODALITY_ORDER)}.

Primary split: complete perturbation-history holdout. Donor overlap is intentional and is not described as donor holdout. Synthetic observations are not biological evidence.
"""
    (output / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")
    model_card = f"""# CausaFlux v1.7.0 multimodal dynamic model card

The release compares baseline covariates, latest RNA, static multimodal fusion, early imaging/reporter temporal modeling, and CausaFlux PoE/MoE dynamic models. Modality-specific encoders, modality dropout, donor/cohort latent context, cross-modal decoders, and MNAR sensitivity analyses are included.

Software exit gate: **{'PASS' if gate['software_exit_gate_passed'] else 'BLOCKED'}**.

Foundation pretraining remains **blocked** until the same gate passes on a locked real longitudinal multimodal perturbation dataset.
"""
    (output / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    # Artifact hashes are generated last and exclude the manifest itself.
    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.csv", "artifact_manifest.json"}:
            manifest_rows.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": _hash(path)})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output / "artifact_manifest.csv", index=False)
    json_dump({"framework":"CausaFlux", "version":"1.7.0", "artifacts": manifest_rows}, output / "artifact_manifest.json")

    qc = validate_multimodal_dynamic_benchmark(output, verify_hashes=True)
    if require_gate and not gate["software_exit_gate_passed"]:
        raise RuntimeError("CausaFlux v1.7.0 multimodal exit gate did not pass")
    return {"output": output, "gate": gate, "comparison": comparison, "qc": qc}


def validate_multimodal_dynamic_benchmark(output: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    output = Path(output)
    required = [
        "model_comparison.csv", "donor_bootstrap_metrics.csv", "cross_modal_forecasting.csv",
        "cross_modal_uncertainty_coverage.csv", "missingness_sensitivity.csv", "multimodal_exit_gate.json",
        "split_audit.json", "benchmark_config.json", "modality_schema.json", "test_predictions.csv",
        "DATASET_CARD.md", "MODEL_CARD.md", "report/index.html", "artifact_manifest.csv",
        "figures/destructive_state_prediction.svg", "figures/destructive_state_prediction.pdf",
        "figures/destructive_state_prediction.tiff", "figures/cross_modal_forecasting.svg",
        "figures/mnar_sensitivity.svg",
    ]
    missing = [p for p in required if not (output / p).exists()]
    checks: dict[str, Any] = {"missing_files": missing}
    valid = not missing
    if not missing:
        comparison = pd.read_csv(output / "model_comparison.csv")
        gate = json.loads((output / "multimodal_exit_gate.json").read_text())
        audit = json.loads((output / "split_audit.json").read_text())
        schema = json.loads((output / "modality_schema.json").read_text())
        checks.update({
            "models_complete": set(MODEL_ORDER).issubset(set(comparison.model)),
            "history_split_valid": bool(audit.get("history_split_valid")) and not bool(audit.get("history_leakage")),
            "modalities_complete": set(MODALITY_ORDER).issubset(set(schema.get("modalities", {}))),
            "poe_and_moe_present": set(schema.get("fusion", [])) == {"product_of_experts", "mixture_of_experts"},
            "modality_dropout_positive": float(schema.get("modality_dropout_training", 0)) > 0,
            "software_exit_gate_passed": bool(gate.get("software_exit_gate_passed")),
            "real_foundation_pretraining_blocked": str(gate.get("foundation_pretraining_authorization", "")).startswith("BLOCKED"),
        })
        valid = valid and all(bool(v) for k, v in checks.items() if k not in {"missing_files"})
    hash_errors: list[str] = []
    if verify_hashes and (output / "artifact_manifest.csv").exists():
        manifest = pd.read_csv(output / "artifact_manifest.csv")
        for row in manifest.itertuples(index=False):
            path = output / str(row.path)
            if not path.exists() or _hash(path) != str(row.sha256):
                hash_errors.append(str(row.path))
        valid = valid and not hash_errors
    checks["hash_errors"] = hash_errors
    checks["valid"] = bool(valid)
    return checks
