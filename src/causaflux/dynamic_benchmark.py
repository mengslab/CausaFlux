from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utils import json_dump, set_seed


MODEL_ORDER = [
    "LatestStateLinear",
    "LatestStateMLP",
    "HistorySummaryMLP",
    "GRUDynamic",
    "CausaFluxFactorizedGRU",
    "IrregularTimeTransformer",
    "NeuralCDE",
    "PRESCIENTComparator",
]
DYNAMIC_MODELS = {
    "GRUDynamic",
    "CausaFluxFactorizedGRU",
    "IrregularTimeTransformer",
    "NeuralCDE",
    "PRESCIENTComparator",
}


@dataclass
class DynamicBenchmarkConfig:
    seed: int = 130
    n_donors: int = 12
    replicates_per_history: int = 4
    steps: int = 8
    context_steps: int = 5
    observation_dim: int = 8
    intervention_dim: int = 4
    hidden_dim: int = 48
    epochs: int = 28
    patience: int = 6
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    bootstrap_replicates: int = 100
    device: str = "cpu"

    @property
    def horizon(self) -> int:
        return self.steps - self.context_steps


@dataclass
class DynamicBenchmarkData:
    observations: np.ndarray
    interventions: np.ndarray
    times: np.ndarray
    fates: np.ndarray
    trajectory_ids: np.ndarray
    donor_ids: np.ndarray
    history_ids: np.ndarray
    targets: np.ndarray
    doses: np.ndarray
    sequences: np.ndarray
    feature_names: list[str]
    intervention_names: list[str]
    fate_names: list[str]

    def __len__(self) -> int:
        return len(self.observations)


class TensorTrajectoryDataset(Dataset):
    def __init__(
        self,
        data: DynamicBenchmarkData,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        context_steps: int,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=int)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.context_steps = int(context_steps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        obs = (self.data.observations[idx] - self.mean) / self.std
        interventions = self.data.interventions[idx]
        times = self.data.times[idx]
        c = self.context_steps
        return {
            "context_obs": torch.as_tensor(obs[:c], dtype=torch.float32),
            "context_interventions": torch.as_tensor(interventions[:c], dtype=torch.float32),
            "context_times": torch.as_tensor(times[:c], dtype=torch.float32),
            "future_interventions": torch.as_tensor(interventions[c:], dtype=torch.float32),
            "future_times": torch.as_tensor(times[c:], dtype=torch.float32),
            "future_obs": torch.as_tensor(obs[c:], dtype=torch.float32),
            "fate": torch.as_tensor(self.data.fates[idx], dtype=torch.long),
            "row_index": torch.as_tensor(idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Synthetic dynamic benchmark fixture
# ---------------------------------------------------------------------------


def _history_schedule(
    target_index: int,
    dose: float,
    sequence: str,
    steps: int,
) -> np.ndarray:
    schedule = np.zeros((steps, 4), dtype=np.float32)
    if sequence == "continuous":
        schedule[1:, target_index] = dose
    elif sequence == "pulse_recovery":
        schedule[1:3, target_index] = dose
    elif sequence == "delayed_rescue":
        schedule[1:4, target_index] = dose
        schedule[5:, 3] = 1.0
    elif sequence == "pulsatile":
        schedule[1::2, target_index] = dose
    elif sequence == "stress_rescue_stress":
        schedule[1:3, target_index] = dose
        schedule[3:5, 3] = 0.9
        schedule[5:, target_index] = dose * 0.75
    else:
        raise ValueError(f"unknown sequence {sequence}")
    return schedule


def generate_dynamic_benchmark_data(
    config: DynamicBenchmarkConfig | None = None,
) -> DynamicBenchmarkData:
    cfg = config or DynamicBenchmarkConfig()
    rng = np.random.default_rng(cfg.seed)
    targets = ["IRE1_XBP1", "PERK_ATF4", "ATF6"]
    doses = [0.5, 1.0, 1.75]
    sequences = [
        "continuous",
        "pulse_recovery",
        "delayed_rescue",
        "pulsatile",
        "stress_rescue_stress",
    ]
    feature_names = [
        "xbp1_reporter",
        "atf4_reporter",
        "atf6_reporter",
        "proteostasis_capacity",
        "mitochondrial_reserve",
        "inflammatory_signal",
        "calcium_instability",
        "apoptosis_signal",
    ]
    intervention_names = ["ire1_stress", "perk_stress", "atf6_stress", "recovery_agent"]
    fate_names = ["recovery", "persistent_dysfunction", "death"]

    rows_obs: list[np.ndarray] = []
    rows_int: list[np.ndarray] = []
    rows_time: list[np.ndarray] = []
    rows_fate: list[int] = []
    trajectory_ids: list[str] = []
    donor_ids: list[str] = []
    history_ids: list[str] = []
    target_values: list[str] = []
    dose_values: list[float] = []
    sequence_values: list[str] = []

    history_counter = 0
    for target_index, target in enumerate(targets):
        for dose in doses:
            for sequence in sequences:
                history_id = f"H{history_counter:03d}_{target}_{dose:g}_{sequence}"
                history_counter += 1
                schedule = _history_schedule(target_index, dose, sequence, cfg.steps)
                for replicate in range(cfg.replicates_per_history):
                    donor_index = (history_counter * 3 + replicate * 5) % cfg.n_donors
                    donor = f"D{donor_index:02d}"
                    donor_sensitivity = rng.normal(1.0, 0.10, size=3)
                    donor_repair = float(np.clip(rng.normal(1.0, 0.12), 0.70, 1.35))
                    donor_inflammation = float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.30))
                    dt = rng.uniform(7.0, 17.0, size=cfg.steps - 1)
                    times = np.concatenate([[0.0], np.cumsum(dt)]).astype(np.float32)

                    adaptation = rng.normal(0.12, 0.025, size=3)
                    damage = float(max(0.0, rng.normal(0.05, 0.015)))
                    commitment = float(max(0.0, rng.normal(0.015, 0.008)))
                    reserve = float(np.clip(rng.normal(0.95, 0.04), 0.75, 1.10))
                    inflammatory_memory = float(max(0.0, rng.normal(0.04, 0.015)))
                    order_memory = 0.0
                    obs = np.zeros((cfg.steps, cfg.observation_dim), dtype=np.float32)

                    def observe(step: int) -> np.ndarray:
                        current = schedule[step, :3]
                        xbp1 = 0.18 + 0.88 * adaptation[0] + 0.52 * current[0] - 0.18 * damage
                        atf4 = 0.16 + 0.90 * adaptation[1] + 0.50 * current[1] + 0.12 * damage
                        atf6 = 0.15 + 0.86 * adaptation[2] + 0.48 * current[2] - 0.08 * damage
                        proteostasis = 0.86 + 0.28 * adaptation.mean() - 0.82 * damage
                        mito = reserve - 0.55 * damage - 0.18 * commitment
                        inflammation = donor_inflammation * (
                            0.14 + 0.62 * inflammatory_memory + 0.44 * commitment
                        )
                        calcium = 0.12 + 0.38 * damage + 0.42 * abs(adaptation[1] - adaptation[0])
                        apoptosis = 1.0 / (1.0 + math.exp(-7.0 * (commitment + 0.65 * damage - 0.72)))
                        value = np.asarray(
                            [xbp1, atf4, atf6, proteostasis, mito, inflammation, calcium, apoptosis],
                            dtype=np.float32,
                        )
                        return value + rng.normal(0.0, 0.025, size=cfg.observation_dim).astype(np.float32)

                    obs[0] = observe(0)
                    cumulative_stress = np.zeros(3, dtype=np.float64)
                    for step in range(1, cfg.steps):
                        delta = float(times[step] - times[step - 1]) / 12.0
                        stress = schedule[step, :3].astype(np.float64) * donor_sensitivity
                        rescue = float(schedule[step, 3])
                        previous_stress = float(schedule[step - 1, :3].sum())
                        previous_rescue = float(schedule[step - 1, 3])
                        order_memory *= 0.90
                        if rescue > 0 and previous_stress > 0:
                            order_memory -= 0.30 * rescue * previous_stress
                        if float(stress.sum()) > 0 and previous_rescue > 0:
                            order_memory += 0.48 * float(stress.sum()) * previous_rescue
                        order_memory += 0.015 * max(0.0, float(cumulative_stress.sum()) - 1.5)
                        cumulative_stress = 0.90 * cumulative_stress + stress * delta
                        adaptation += delta * (
                            0.34 * stress
                            - 0.19 * adaptation
                            - 0.10 * damage
                            + 0.10 * rescue * (0.35 - adaptation)
                        )
                        overload = max(0.0, float(stress.sum()) - (0.85 + 0.30 * adaptation.mean()))
                        history_burden = 0.10 * float(cumulative_stress.sum())
                        damage += delta * (
                            0.15 * overload
                            + 0.020 * float(stress.sum()) ** 2
                            + history_burden
                            - 0.12 * rescue * donor_repair
                            - 0.045 * donor_repair * max(reserve, 0.0)
                        )
                        damage = float(max(0.0, damage))
                        commitment += delta * (
                            0.22 * max(0.0, damage - 0.32)
                            + 0.055 * max(0.0, cumulative_stress.sum() - 2.2)
                            - 0.035 * rescue * donor_repair
                        )
                        commitment = float(max(0.0, commitment))
                        reserve += delta * (
                            0.07 * (1.0 - reserve)
                            - 0.10 * float(stress.sum())
                            - 0.08 * damage
                            + 0.11 * rescue * donor_repair
                        )
                        reserve = float(np.clip(reserve, 0.0, 1.2))
                        inflammatory_memory += delta * (
                            0.09 * damage + 0.08 * commitment + 0.025 * stress[1] - 0.08 * rescue
                        )
                        inflammatory_memory = float(max(0.0, inflammatory_memory))
                        obs[step] = observe(step)

                    fate_score = 0.72 * damage + 0.88 * commitment - 0.42 * reserve + 0.78 * order_memory
                    if fate_score < -0.02:
                        fate = 0
                    elif fate_score < 0.62:
                        fate = 1
                    else:
                        fate = 2
                    # Small stochastic biological variability without erasing the history signal.
                    if rng.random() < 0.035:
                        fate = int(np.clip(fate + rng.choice([-1, 1]), 0, 2))

                    trajectory_id = f"T_{history_id}_{donor}_{replicate:02d}"
                    rows_obs.append(obs)
                    rows_int.append(schedule.copy())
                    rows_time.append(times)
                    rows_fate.append(fate)
                    trajectory_ids.append(trajectory_id)
                    donor_ids.append(donor)
                    history_ids.append(history_id)
                    target_values.append(target)
                    dose_values.append(float(dose))
                    sequence_values.append(sequence)

    return DynamicBenchmarkData(
        observations=np.stack(rows_obs).astype(np.float32),
        interventions=np.stack(rows_int).astype(np.float32),
        times=np.stack(rows_time).astype(np.float32),
        fates=np.asarray(rows_fate, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=object),
        donor_ids=np.asarray(donor_ids, dtype=object),
        history_ids=np.asarray(history_ids, dtype=object),
        targets=np.asarray(target_values, dtype=object),
        doses=np.asarray(dose_values, dtype=np.float32),
        sequences=np.asarray(sequence_values, dtype=object),
        feature_names=feature_names,
        intervention_names=intervention_names,
        fate_names=fate_names,
    )


# ---------------------------------------------------------------------------
# Split policy
# ---------------------------------------------------------------------------


def _stable_rank(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def make_split(
    data: DynamicBenchmarkData,
    mode: str,
    seed: int = 130,
) -> dict[str, np.ndarray]:
    n = len(data)
    all_indices = np.arange(n, dtype=int)
    if mode == "perturbation_history":
        groups = sorted(set(data.history_ids.tolist()), key=lambda x: _stable_rank(str(x), seed))
        n_test = max(6, int(round(0.20 * len(groups))))
        n_val = max(5, int(round(0.16 * len(groups))))
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        train = all_indices[~np.isin(data.history_ids, list(test_groups | val_groups))]
        val = all_indices[np.isin(data.history_ids, list(val_groups))]
        test = all_indices[np.isin(data.history_ids, list(test_groups))]
    elif mode == "dose_holdout":
        test = all_indices[np.isclose(data.doses, 1.75)]
        remaining = all_indices[~np.isclose(data.doses, 1.75)]
        val_histories = sorted(set(data.history_ids[remaining].tolist()), key=lambda x: _stable_rank(str(x), seed))[:6]
        val = remaining[np.isin(data.history_ids[remaining], val_histories)]
        train = np.setdiff1d(remaining, val)
    elif mode == "sequence_holdout":
        test_sequences = {"stress_rescue_stress"}
        test = all_indices[np.isin(data.sequences, list(test_sequences))]
        remaining = all_indices[~np.isin(data.sequences, list(test_sequences))]
        val_histories = sorted(set(data.history_ids[remaining].tolist()), key=lambda x: _stable_rank(str(x), seed))[:6]
        val = remaining[np.isin(data.history_ids[remaining], val_histories)]
        train = np.setdiff1d(remaining, val)
    elif mode == "donor_holdout":
        donors = sorted(set(data.donor_ids.tolist()), key=lambda x: _stable_rank(str(x), seed))
        test_donors = set(donors[:2])
        val_donors = set(donors[2:4])
        test = all_indices[np.isin(data.donor_ids, list(test_donors))]
        val = all_indices[np.isin(data.donor_ids, list(val_donors))]
        train = all_indices[~np.isin(data.donor_ids, list(test_donors | val_donors))]
    elif mode == "temporal_extrapolation":
        # Same history split, but the evaluator is explicitly restricted to the final horizon.
        return make_split(data, "perturbation_history", seed + 17)
    else:
        raise ValueError(f"unsupported split mode: {mode}")
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError(f"empty split for {mode}")
    return {"train": train, "validation": val, "test": test}


def audit_split(data: DynamicBenchmarkData, split: dict[str, np.ndarray], mode: str) -> dict[str, Any]:
    train, val, test = split["train"], split["validation"], split["test"]
    history_overlap = sorted(set(data.history_ids[train]) & set(data.history_ids[test]))
    donor_overlap = sorted(set(data.donor_ids[train]) & set(data.donor_ids[test]))
    return {
        "mode": mode,
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_test": int(len(test)),
        "train_histories": int(len(set(data.history_ids[train]))),
        "validation_histories": int(len(set(data.history_ids[val]))),
        "test_histories": int(len(set(data.history_ids[test]))),
        "history_overlap_train_test": history_overlap,
        "donor_overlap_train_test": donor_overlap,
        "history_leakage": bool(history_overlap) if mode in {"perturbation_history", "temporal_extrapolation"} else False,
        "donor_leakage": bool(donor_overlap) if mode == "donor_holdout" else False,
    }


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------


def _future_schedule_features(
    future_interventions: torch.Tensor,
    future_times: torch.Tensor,
    context_times: torch.Tensor,
) -> torch.Tensor:
    last_time = context_times[:, -1:].contiguous()
    dt = future_times - last_time
    return torch.cat([future_interventions.flatten(1), dt], dim=1)


def _history_summary(
    obs: torch.Tensor,
    interventions: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    last = obs[:, -1]
    mean = obs.mean(dim=1)
    std = obs.std(dim=1, unbiased=False)
    elapsed = (times[:, -1] - times[:, 0]).clamp_min(1e-3).unsqueeze(-1)
    slope = (obs[:, -1] - obs[:, 0]) / elapsed
    cumulative_intervention = interventions.sum(dim=1)
    return torch.cat([last, mean, std, slope, cumulative_intervention], dim=1)


class ForecastModel(nn.Module):
    model_name = "ForecastModel"

    def forward(
        self,
        context_obs: torch.Tensor,
        context_interventions: torch.Tensor,
        context_times: torch.Tensor,
        future_interventions: torch.Tensor,
        future_times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class LatestStateLinear(ForecastModel):
    model_name = "LatestStateLinear"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int) -> None:
        super().__init__()
        input_dim = obs_dim + int_dim + horizon * int_dim + horizon
        self.mean_head = nn.Linear(input_dim, horizon * obs_dim)
        self.logvar = nn.Parameter(torch.zeros(horizon, obs_dim))
        self.fate_head = nn.Linear(input_dim, n_fates)
        self.horizon, self.obs_dim = horizon, obs_dim

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        x = torch.cat(
            [context_obs[:, -1], context_interventions[:, -1], _future_schedule_features(future_interventions, future_times, context_times)],
            dim=1,
        )
        mean = self.mean_head(x).view(-1, self.horizon, self.obs_dim)
        return {"mean": mean, "logvar": self.logvar.unsqueeze(0).expand_as(mean), "fate_logits": self.fate_head(x)}


class LatestStateMLP(ForecastModel):
    model_name = "LatestStateMLP"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        input_dim = obs_dim + int_dim + horizon * int_dim + horizon
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Dropout(0.08), nn.Linear(hidden, hidden), nn.SiLU())
        self.mean_head = nn.Linear(hidden, horizon * obs_dim)
        self.logvar = nn.Parameter(torch.zeros(horizon, obs_dim))
        self.fate_head = nn.Linear(hidden, n_fates)
        self.horizon, self.obs_dim = horizon, obs_dim

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        x = torch.cat(
            [context_obs[:, -1], context_interventions[:, -1], _future_schedule_features(future_interventions, future_times, context_times)],
            dim=1,
        )
        h = self.encoder(x)
        mean = self.mean_head(h).view(-1, self.horizon, self.obs_dim)
        return {"mean": mean, "logvar": self.logvar.unsqueeze(0).expand_as(mean), "fate_logits": self.fate_head(h)}


class HistorySummaryMLP(ForecastModel):
    model_name = "HistorySummaryMLP"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        input_dim = obs_dim * 4 + int_dim + horizon * int_dim + horizon
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Dropout(0.08), nn.Linear(hidden, hidden), nn.SiLU())
        self.mean_head = nn.Linear(hidden, horizon * obs_dim)
        self.logvar = nn.Parameter(torch.zeros(horizon, obs_dim))
        self.fate_head = nn.Linear(hidden, n_fates)
        self.horizon, self.obs_dim = horizon, obs_dim

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        x = torch.cat([_history_summary(context_obs, context_interventions, context_times), _future_schedule_features(future_interventions, future_times, context_times)], dim=1)
        h = self.encoder(x)
        mean = self.mean_head(h).view(-1, self.horizon, self.obs_dim)
        return {"mean": mean, "logvar": self.logvar.unsqueeze(0).expand_as(mean), "fate_logits": self.fate_head(h)}


class GRUDynamic(ForecastModel):
    model_name = "GRUDynamic"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(obs_dim + int_dim + 1, hidden, batch_first=True)
        self.decoder = nn.GRUCell(obs_dim + int_dim + 1, hidden)
        self.mean_head = nn.Linear(hidden, obs_dim)
        self.logvar_head = nn.Linear(hidden, obs_dim)
        self.fate_head = nn.Linear(hidden, n_fates)
        self.horizon = horizon

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        dt = torch.diff(context_times, dim=1, prepend=context_times[:, :1]).unsqueeze(-1) / 12.0
        _, h = self.encoder(torch.cat([context_obs, context_interventions, dt], dim=-1))
        h = h[-1]
        previous = context_obs[:, -1]
        previous_time = context_times[:, -1]
        means, logvars = [], []
        for step in range(self.horizon):
            step_dt = ((future_times[:, step] - previous_time) / 12.0).unsqueeze(-1)
            h = self.decoder(torch.cat([previous, future_interventions[:, step], step_dt], dim=-1), h)
            previous = self.mean_head(h)
            means.append(previous)
            logvars.append(self.logvar_head(h).clamp(-6.0, 2.0))
            previous_time = future_times[:, step]
        return {"mean": torch.stack(means, dim=1), "logvar": torch.stack(logvars, dim=1), "fate_logits": self.fate_head(h)}


class CausaFluxFactorizedGRU(ForecastModel):
    model_name = "CausaFluxFactorizedGRU"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        self.identity = nn.Sequential(nn.Linear(obs_dim, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, hidden // 4))
        self.context = nn.Sequential(nn.Linear(obs_dim, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, hidden // 4))
        self.gru = nn.GRUCell(obs_dim + int_dim + 1, hidden)
        self.initial = nn.Linear(hidden // 2, hidden)
        self.adaptation = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh())
        self.commitment_increment = nn.Sequential(nn.Linear(hidden, hidden // 4), nn.Softplus())
        latent_dim = hidden // 4 + hidden // 4 + hidden // 2 + hidden // 4
        self.temporal_input = nn.Linear(obs_dim + int_dim + 4, hidden)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=0.08,
            batch_first=True, activation="gelu"
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        future_input = latent_dim + obs_dim + hidden + horizon * int_dim + horizon
        self.future_head = nn.Sequential(
            nn.Linear(future_input, hidden * 2),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, horizon * obs_dim * 2),
        )
        self.fate_head = nn.Sequential(nn.Linear(latent_dim + horizon * int_dim + horizon, hidden), nn.SiLU(), nn.Dropout(0.08), nn.Linear(hidden, n_fates))
        self.horizon = horizon
        self.obs_dim = obs_dim

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        identity = self.identity(context_obs[:, 0])
        context = self.context(context_obs[:, 0])
        h = torch.tanh(self.initial(torch.cat([identity, context], dim=-1)))
        commitment = torch.zeros(context_obs.size(0), h.size(1) // 4, device=h.device)
        for step in range(context_obs.size(1)):
            if step == 0:
                dt = torch.zeros_like(context_times[:, step]).unsqueeze(-1)
            else:
                dt = ((context_times[:, step] - context_times[:, step - 1]) / 12.0).unsqueeze(-1)
            h = self.gru(torch.cat([context_obs[:, step], context_interventions[:, step], dt], dim=-1), h)
            commitment = commitment + 0.035 * self.commitment_increment(h) * dt.clamp_min(0.0)
        adaptation = self.adaptation(h)
        latent = torch.cat([identity, context, adaptation, commitment], dim=-1)
        scaled = context_times / 24.0
        time_features = torch.stack([scaled, torch.log1p(context_times) / 4.0, torch.sin(scaled), torch.cos(scaled)], dim=-1)
        temporal_tokens = self.temporal_input(torch.cat([context_obs, context_interventions, time_features], dim=-1))
        temporal_encoded = self.temporal_encoder(temporal_tokens)
        temporal_summary = 0.6 * temporal_encoded[:, -1] + 0.4 * temporal_encoded.mean(dim=1)
        schedule = _future_schedule_features(future_interventions, future_times, context_times)
        decoded = self.future_head(torch.cat([latent, context_obs[:, -1], temporal_summary, schedule], dim=-1))
        decoded = decoded.view(-1, self.horizon, self.obs_dim * 2)
        mean, logvar = torch.chunk(decoded, 2, dim=-1)
        return {"mean": mean, "logvar": logvar.clamp(-6.0, 2.0), "fate_logits": self.fate_head(torch.cat([latent, schedule], dim=-1))}


class IrregularTimeTransformer(ForecastModel):
    model_name = "IrregularTimeTransformer"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(obs_dim + int_dim + 4, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=0.08, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.future = nn.Linear(int_dim + 3, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, obs_dim * 2))
        self.fate_head = nn.Sequential(
            nn.Linear(hidden + horizon * int_dim + horizon, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, n_fates),
        )
        self.horizon = horizon

    @staticmethod
    def _time_features(times: torch.Tensor) -> torch.Tensor:
        scaled = times / 24.0
        return torch.stack([scaled, torch.log1p(times) / 4.0, torch.sin(scaled), torch.cos(scaled)], dim=-1)

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        tokens = self.input(torch.cat([context_obs, context_interventions, self._time_features(context_times)], dim=-1))
        encoded = self.encoder(tokens)
        pooled = encoded[:, -1]
        base = context_times[:, -1:]
        dt = (future_times - base) / 24.0
        future_tokens = self.future(torch.cat([future_interventions, dt.unsqueeze(-1), torch.sin(dt).unsqueeze(-1), torch.cos(dt).unsqueeze(-1)], dim=-1))
        pooled_expanded = pooled.unsqueeze(1).expand(-1, self.horizon, -1)
        decoded = self.decoder(torch.cat([pooled_expanded, future_tokens], dim=-1))
        mean, logvar = torch.chunk(decoded, 2, dim=-1)
        schedule = _future_schedule_features(future_interventions, future_times, context_times)
        return {"mean": mean, "logvar": logvar.clamp(-6.0, 2.0), "fate_logits": self.fate_head(torch.cat([pooled, schedule], dim=-1))}


class NeuralCDE(ForecastModel):
    """Piecewise-linear Euler neural CDE without an optional torchcde dependency."""

    model_name = "NeuralCDE"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        self.control_dim = obs_dim + int_dim + 1
        self.initial = nn.Linear(self.control_dim, hidden)
        self.vector_field = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden * self.control_dim))
        self.readout = nn.Sequential(nn.Linear(hidden + int_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, obs_dim * 2))
        self.fate_head = nn.Linear(hidden, n_fates)
        self.hidden = hidden
        self.horizon = horizon

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        control = torch.cat([context_obs, context_interventions, (context_times / 24.0).unsqueeze(-1)], dim=-1)
        h = torch.tanh(self.initial(control[:, 0]))
        for step in range(1, control.size(1)):
            delta_x = control[:, step] - control[:, step - 1]
            field = self.vector_field(h).view(-1, self.hidden, self.control_dim)
            h = h + torch.bmm(field, delta_x.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.control_dim)
            h = torch.tanh(h)
        means, logvars = [], []
        last_time = context_times[:, -1]
        for step in range(self.horizon):
            dt = ((future_times[:, step] - last_time) / 24.0).unsqueeze(-1)
            out = self.readout(torch.cat([h, future_interventions[:, step], dt], dim=-1))
            mean, logvar = torch.chunk(out, 2, dim=-1)
            means.append(mean)
            logvars.append(logvar.clamp(-6.0, 2.0))
            # Forecast-control update uses predicted state and scheduled intervention.
            delta_control = torch.cat([mean - context_obs[:, -1], future_interventions[:, step], dt], dim=-1)
            field = self.vector_field(h).view(-1, self.hidden, self.control_dim)
            h = torch.tanh(h + 0.35 * torch.bmm(field, delta_control.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.control_dim))
            last_time = future_times[:, step]
        return {"mean": torch.stack(means, dim=1), "logvar": torch.stack(logvars, dim=1), "fate_logits": self.fate_head(h)}


class PRESCIENTComparator(ForecastModel):
    """Lightweight PRESCIENT-style latent drift comparator, not the upstream package."""

    model_name = "PRESCIENTComparator"

    def __init__(self, obs_dim: int, int_dim: int, horizon: int, n_fates: int, hidden: int) -> None:
        super().__init__()
        latent = max(12, hidden // 2)
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, latent))
        self.potential = nn.Sequential(nn.Linear(latent + int_dim + 1, hidden), nn.Softplus(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, obs_dim * 2))
        self.fate_head = nn.Linear(latent, n_fates)
        self.horizon = horizon

    def forward(self, context_obs, context_interventions, context_times, future_interventions, future_times):
        z = self.encoder(context_obs[:, 0])
        for step in range(1, context_obs.size(1)):
            target_z = self.encoder(context_obs[:, step])
            dt = ((context_times[:, step] - context_times[:, step - 1]) / 12.0).unsqueeze(-1)
            drift = self.potential(torch.cat([z, context_interventions[:, step], dt], dim=-1))
            z = z + 0.35 * dt * drift + 0.45 * (target_z - z)
        means, logvars = [], []
        previous_time = context_times[:, -1]
        for step in range(self.horizon):
            dt = ((future_times[:, step] - previous_time) / 12.0).unsqueeze(-1)
            z = z + 0.35 * dt * self.potential(torch.cat([z, future_interventions[:, step], dt], dim=-1))
            out = self.decoder(z)
            mean, logvar = torch.chunk(out, 2, dim=-1)
            means.append(mean)
            logvars.append(logvar.clamp(-6.0, 2.0))
            previous_time = future_times[:, step]
        return {"mean": torch.stack(means, dim=1), "logvar": torch.stack(logvars, dim=1), "fate_logits": self.fate_head(z)}


def build_model(name: str, cfg: DynamicBenchmarkConfig, n_fates: int = 3) -> ForecastModel:
    kwargs = dict(obs_dim=cfg.observation_dim, int_dim=cfg.intervention_dim, horizon=cfg.horizon, n_fates=n_fates)
    if name == "LatestStateLinear":
        return LatestStateLinear(**kwargs)
    if name == "LatestStateMLP":
        return LatestStateMLP(**kwargs, hidden=cfg.hidden_dim)
    if name == "HistorySummaryMLP":
        return HistorySummaryMLP(**kwargs, hidden=cfg.hidden_dim)
    if name == "GRUDynamic":
        return GRUDynamic(**kwargs, hidden=cfg.hidden_dim)
    if name == "CausaFluxFactorizedGRU":
        return CausaFluxFactorizedGRU(**kwargs, hidden=cfg.hidden_dim)
    if name == "IrregularTimeTransformer":
        return IrregularTimeTransformer(**kwargs, hidden=cfg.hidden_dim)
    if name == "NeuralCDE":
        return NeuralCDE(**kwargs, hidden=cfg.hidden_dim)
    if name == "PRESCIENTComparator":
        return PRESCIENTComparator(**kwargs, hidden=cfg.hidden_dim)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, fate: torch.Tensor, fate_weight: float = 0.45) -> torch.Tensor:
    error = outputs["mean"] - target
    logvar = outputs["logvar"].clamp(-6.0, 2.0)
    nll = 0.5 * (error.square() * torch.exp(-logvar) + logvar).mean()
    fate_loss = nn.functional.cross_entropy(outputs["fate_logits"], fate)
    return nll + fate_weight * fate_loss


def _collect_predictions(model: ForecastModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    means, logvars, targets, logits, fates, indices = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_device = {k: v.to(device) for k, v in batch.items() if k != "row_index"}
            out = model(
                batch_device["context_obs"],
                batch_device["context_interventions"],
                batch_device["context_times"],
                batch_device["future_interventions"],
                batch_device["future_times"],
            )
            means.append(out["mean"].cpu().numpy())
            logvars.append(out["logvar"].cpu().numpy())
            targets.append(batch_device["future_obs"].cpu().numpy())
            logits.append(out["fate_logits"].cpu().numpy())
            fates.append(batch_device["fate"].cpu().numpy())
            indices.append(batch["row_index"].numpy())
    return {
        "mean": np.concatenate(means),
        "logvar": np.concatenate(logvars),
        "target": np.concatenate(targets),
        "logits": np.concatenate(logits),
        "fate": np.concatenate(fates),
        "indices": np.concatenate(indices),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += float(mask.mean() * abs((prediction[mask] == labels[mask]).mean() - confidence[mask].mean()))
    return result


def _calibration_variance(validation: dict[str, np.ndarray]) -> float:
    residual = validation["target"] - validation["mean"]
    base_variance = np.exp(np.clip(validation["logvar"], -6.0, 2.0))
    # Maximum-likelihood temperature is estimated on validation histories only.
    # The model retains heteroscedastic variance while a single scalar corrects scale.
    temperature = float(np.mean((residual**2) / np.clip(base_variance, 1e-5, 25.0)))
    return float(np.clip(temperature, 0.05, 20.0))


def _normal_quantile(level: float) -> float:
    # Fixed values avoid adding another statistical dependency.
    values = {0.50: 0.67448975, 0.80: 1.28155157, 0.90: 1.64485363, 0.95: 1.95996398}
    return values[level]


def evaluate_predictions(
    prediction: dict[str, np.ndarray],
    calibration_variance: float,
) -> dict[str, float]:
    mean = prediction["mean"]
    target = prediction["target"]
    residual = target - mean
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    flat_mean, flat_target = mean.reshape(-1), target.reshape(-1)
    correlation = float(np.corrcoef(flat_mean, flat_target)[0, 1]) if np.std(flat_mean) > 0 and np.std(flat_target) > 0 else 0.0
    variance = np.exp(np.clip(prediction["logvar"], -6.0, 2.0)) * float(calibration_variance)
    variance = np.clip(variance, 1e-4, 25.0)
    nll = float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance)))
    probabilities = _softmax(prediction["logits"])
    labels = prediction["fate"].astype(int)
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[labels]
    metrics: dict[str, float] = {
        "trajectory_rmse": rmse,
        "trajectory_mae": mae,
        "trajectory_correlation": correlation,
        "calibrated_gaussian_nll": nll,
        "fate_accuracy": float(accuracy_score(labels, predicted)),
        "fate_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "fate_log_loss": float(log_loss(labels, probabilities, labels=list(range(probabilities.shape[1])))),
        "fate_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "fate_ece": _ece(probabilities, labels),
    }
    sigma = np.sqrt(variance)
    for level in (0.50, 0.80, 0.90, 0.95):
        z = _normal_quantile(level)
        inside = np.abs(residual) <= z * sigma
        metrics[f"coverage_{int(level * 100)}"] = float(inside.mean())
        metrics[f"interval_width_{int(level * 100)}"] = float((2.0 * z * sigma).mean())
    return metrics


def train_one_model(
    model_name: str,
    data: DynamicBenchmarkData,
    split: dict[str, np.ndarray],
    cfg: DynamicBenchmarkConfig,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(cfg.seed + MODEL_ORDER.index(model_name) * 13)
    device = torch.device(cfg.device)
    train_obs = data.observations[split["train"]]
    mean = train_obs.mean(axis=(0, 1), keepdims=True).reshape(-1)
    std = train_obs.std(axis=(0, 1), keepdims=True).reshape(-1)
    std = np.where(std < 1e-4, 1.0, std)
    datasets = {
        name: TensorTrajectoryDataset(data, indices, mean, std, cfg.context_steps)
        for name, indices in split.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True),
        "validation": DataLoader(datasets["validation"], batch_size=cfg.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size, shuffle=False),
    }
    model = build_model(model_name, cfg).to(device)
    fate_weight = {
        "IrregularTimeTransformer": 1.10,
        "CausaFluxFactorizedGRU": 0.90,
        "GRUDynamic": 0.75,
        "NeuralCDE": 0.75,
        "PRESCIENTComparator": 0.75,
    }.get(model_name, 0.45)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            b = {k: v.to(device) for k, v in batch.items() if k != "row_index"}
            out = model(b["context_obs"], b["context_interventions"], b["context_times"], b["future_interventions"], b["future_times"])
            loss = _loss(out, b["future_obs"], b["fate"], fate_weight=fate_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_prediction = _collect_predictions(model, loaders["validation"], device)
        val_variance = _calibration_variance(val_prediction)
        val_metrics = evaluate_predictions(val_prediction, val_variance)
        val_loss = val_metrics["calibrated_gaussian_nll"] + min(fate_weight, 0.90) * val_metrics["fate_log_loss"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_objective": float(val_loss)})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError(f"model {model_name} did not train")
    model.load_state_dict(best_state)
    model.to(device)
    validation_prediction = _collect_predictions(model, loaders["validation"], device)
    calibration_variance = _calibration_variance(validation_prediction)
    test_prediction = _collect_predictions(model, loaders["test"], device)
    metrics = evaluate_predictions(test_prediction, calibration_variance)
    metrics.update({"model": model_name, "model_family": "dynamic" if model_name in DYNAMIC_MODELS else "static", "epochs_completed": len(history), "best_validation_objective": float(best_val)})
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "config": asdict(cfg), "state_dict": best_state, "mean": mean, "std": std, "calibration_variance": calibration_variance}, model_dir / "checkpoint.pt")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    rows = []
    probs = _softmax(test_prediction["logits"])
    raw_mean = test_prediction["mean"] * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    raw_target = test_prediction["target"] * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    for local, data_index in enumerate(test_prediction["indices"]):
        row: dict[str, Any] = {
            "trajectory_id": str(data.trajectory_ids[data_index]),
            "donor_id": str(data.donor_ids[data_index]),
            "history_id": str(data.history_ids[data_index]),
            "actual_fate": data.fate_names[int(test_prediction["fate"][local])],
            "predicted_fate": data.fate_names[int(probs[local].argmax())],
        }
        for fate_idx, fate_name in enumerate(data.fate_names):
            row[f"probability_{fate_name}"] = float(probs[local, fate_idx])
        for horizon_idx in range(cfg.horizon):
            for feature_idx, feature in enumerate(data.feature_names):
                row[f"actual_t{horizon_idx + 1}_{feature}"] = float(raw_target[local, horizon_idx, feature_idx])
                row[f"predicted_t{horizon_idx + 1}_{feature}"] = float(raw_mean[local, horizon_idx, feature_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(model_dir / "test_predictions.csv", index=False)

    donor_values = np.asarray(data.donor_ids[test_prediction["indices"]], dtype=object)
    unique_donors = sorted(set(map(str, donor_values)))
    rng = np.random.default_rng(cfg.seed + MODEL_ORDER.index(model_name) * 101)
    bootstrap_rows: list[dict[str, Any]] = []
    for bootstrap_index in range(cfg.bootstrap_replicates):
        sampled = rng.choice(unique_donors, size=len(unique_donors), replace=True)
        selected_parts = [np.flatnonzero(donor_values == donor) for donor in sampled]
        selected = np.concatenate(selected_parts) if selected_parts else np.arange(len(donor_values))
        boot_prediction = {
            key: value[selected] if isinstance(value, np.ndarray) and value.shape[0] == len(donor_values) else value
            for key, value in test_prediction.items()
        }
        boot_metrics = evaluate_predictions(boot_prediction, calibration_variance)
        bootstrap_rows.append({"model": model_name, "bootstrap": bootstrap_index, **boot_metrics})
    pd.DataFrame(bootstrap_rows).to_csv(model_dir / "donor_bootstrap_metrics.csv", index=False)
    np.save(model_dir / "calibration_temperature.npy", np.asarray([calibration_variance], dtype=np.float64))
    json_dump(metrics, model_dir / "metrics.json")
    return metrics


def _bootstrap_metric_intervals(
    metrics_frame: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    metrics = ["trajectory_rmse", "calibrated_gaussian_nll", "fate_log_loss", "fate_accuracy"]
    for _, summary in metrics_frame.iterrows():
        model = str(summary["model"])
        bootstrap_path = output_dir / "models" / model / "donor_bootstrap_metrics.csv"
        bootstrap = pd.read_csv(bootstrap_path)
        for metric in metrics:
            values = bootstrap[metric].dropna().to_numpy(dtype=float)
            rows.append({
                "model": model,
                "metric": metric,
                "estimate": float(summary[metric]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": int(len(values)),
                "method": "donor-cluster bootstrap",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metric_intervals.csv", index=False)
    return frame


def evaluate_exit_gate(metrics: pd.DataFrame) -> dict[str, Any]:
    latest = metrics.set_index("model").loc["LatestStateMLP"]
    history = metrics.set_index("model").loc["HistorySummaryMLP"]
    candidates = metrics[metrics["model"].isin(DYNAMIC_MODELS)].copy()
    candidates["beats_latest_rmse"] = candidates["trajectory_rmse"] < latest["trajectory_rmse"]
    candidates["beats_history_rmse"] = candidates["trajectory_rmse"] < history["trajectory_rmse"]
    candidates["beats_latest_nll"] = candidates["calibrated_gaussian_nll"] < latest["calibrated_gaussian_nll"]
    candidates["beats_history_nll"] = candidates["calibrated_gaussian_nll"] < history["calibrated_gaussian_nll"]
    candidates["beats_latest_fate"] = candidates["fate_log_loss"] < latest["fate_log_loss"]
    candidates["beats_history_fate"] = candidates["fate_log_loss"] < history["fate_log_loss"]
    criteria = [
        "beats_latest_rmse",
        "beats_latest_nll",
        "beats_history_nll",
        "beats_latest_fate",
        "beats_history_fate",
    ]
    candidates["passes_exit_criterion"] = candidates[criteria].all(axis=1)
    passing = candidates[candidates["passes_exit_criterion"]].sort_values(["calibrated_gaussian_nll", "fate_log_loss"])
    winner = str(passing.iloc[0]["model"]) if len(passing) else None
    gate = {
        "status": "PASS" if winner else "BLOCKED",
        "foundation_pretraining_allowed": bool(winner),
        "passing_dynamic_models": passing["model"].tolist(),
        "winning_dynamic_model": winner,
        "required_baselines": ["LatestStateMLP", "HistorySummaryMLP"],
        "criteria": [
            "lower trajectory RMSE than the latest-state MLP",
            "lower calibrated Gaussian NLL than both static baselines",
            "lower fate log loss than both static baselines",
            "evaluation on completely held-out perturbation histories",
        ],
    }
    return gate


def embedding_adapter_status() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "adapter": "scVIStaticEmbedding",
                "status": "optional_interface_ready",
                "required_input": "precomputed per-observation scVI latent matrix",
                "network_download": False,
                "included_checkpoint": False,
            },
            {
                "adapter": "scGPTStaticEmbedding",
                "status": "optional_interface_ready",
                "required_input": "precomputed per-observation scGPT embedding matrix",
                "network_download": False,
                "included_checkpoint": False,
            },
        ]
    )


def external_benchmark_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.7.0",
        "required_arrays": {
            "observations": "float32 [trajectory,time,feature]",
            "interventions": "float32 [trajectory,time,intervention]",
            "times": "float32 [trajectory,time]",
            "fates": "int64 [trajectory]",
            "donor_ids": "string [trajectory]",
            "history_ids": "string [trajectory]",
        },
        "required_semantics": [
            "history_id uniquely identifies target, dose, order, schedule and recovery interval",
            "donor_id identifies the biological donor or independent experimental unit",
            "future interventions are known schedules, not post-outcome measurements",
            "external test sets cannot be used for feature selection, hyperparameter tuning or calibration",
        ],
        "supported_split_modes": ["perturbation_history", "dose_holdout", "sequence_holdout", "temporal_extrapolation", "donor_holdout"],
    }


def _write_report(output_dir: Path, metrics: pd.DataFrame, gate: dict[str, Any], split_audit: dict[str, Any], cfg: DynamicBenchmarkConfig) -> Path:
    metrics_table = metrics.sort_values("calibrated_gaussian_nll").to_html(index=False, float_format=lambda x: f"{x:.4f}")
    gate_class = "ok" if gate["status"] == "PASS" else "warn"
    winner = gate.get("winning_dynamic_model") or "None"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v1.7.0 dynamic benchmark</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;max-width:1220px;margin:28px auto;padding:0 22px;color:#202124}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px;text-align:left}}th{{background:#f4f4f4}}.ok{{border-left:4px solid #00A087;padding:12px;background:#f2fbf8}}.warn{{border-left:4px solid #E64B35;padding:12px;background:#fff5f3}}code{{background:#f3f3f3;padding:2px 4px}}</style></head><body>
<h1>CausaFlux v1.7.0 — Dynamic Model Benchmark</h1>
<div class='{gate_class}'><strong>Exit gate: {gate['status']}.</strong> Winning dynamic model: <code>{winner}</code>. Foundation pretraining allowed: <strong>{gate['foundation_pretraining_allowed']}</strong>.</div>
<p>The primary test withholds complete target × dose × sequence histories. Models forecast {cfg.horizon} future observations and terminal fate from {cfg.context_steps} irregularly sampled context observations.</p>
<h2>Primary held-out-history results</h2>{metrics_table}
<h2>Split audit</h2><pre>{json.dumps(split_audit, indent=2)}</pre>
<h2>Interpretation boundary</h2><p>This packaged benchmark uses a deterministic synthetic dynamic system to validate software, split policy, model training, metric calculation and the release gate. It is not biological validation. The same benchmark contract must next be executed on real longitudinal perturbation datasets.</p>
</body></html>"""
    report = output_dir / "report" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(html, encoding="utf-8")
    return report


def run_dynamic_benchmark(
    output_dir: str | Path,
    config: DynamicBenchmarkConfig | None = None,
    model_names: Iterable[str] | None = None,
    data: DynamicBenchmarkData | None = None,
) -> dict[str, Any]:
    cfg = config or DynamicBenchmarkConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = data or generate_dynamic_benchmark_data(cfg)
    validation = validate_external_benchmark_data(data)
    if not validation["valid"]:
        raise ValueError(f"dynamic benchmark input failed validation: {validation['errors']}")
    if data.observations.shape[-1] != cfg.observation_dim or data.interventions.shape[-1] != cfg.intervention_dim:
        cfg = DynamicBenchmarkConfig(**{**asdict(cfg), "observation_dim": int(data.observations.shape[-1]), "intervention_dim": int(data.interventions.shape[-1]), "steps": int(data.observations.shape[1]), "context_steps": min(cfg.context_steps, int(data.observations.shape[1]) - 1)})
    split = make_split(data, "perturbation_history", cfg.seed)
    split_audit = audit_split(data, split, "perturbation_history")
    if split_audit["history_leakage"]:
        raise RuntimeError("held-out-history leakage detected")

    metadata = pd.DataFrame(
        {
            "trajectory_id": data.trajectory_ids,
            "donor_id": data.donor_ids,
            "history_id": data.history_ids,
            "target": data.targets,
            "dose": data.doses,
            "sequence": data.sequences,
            "fate": [data.fate_names[i] for i in data.fates],
        }
    )
    metadata.to_csv(output_dir / "trajectory_metadata.csv", index=False)
    split_rows = []
    for split_name, indices in split.items():
        for idx in indices:
            split_rows.append({"trajectory_id": data.trajectory_ids[idx], "split": split_name, "history_id": data.history_ids[idx], "donor_id": data.donor_ids[idx]})
    pd.DataFrame(split_rows).to_csv(output_dir / "split_manifest.csv", index=False)
    json_dump(split_audit, output_dir / "split_audit.json")
    json_dump(asdict(cfg), output_dir / "benchmark_config.json")
    json_dump(external_benchmark_contract(), output_dir / "external_benchmark_contract.json")
    embedding_adapter_status().to_csv(output_dir / "optional_embedding_adapters.csv", index=False)

    metrics = []
    selected_models = list(model_names) if model_names is not None else MODEL_ORDER
    for name in selected_models:
        print(f"[dynamic-benchmark] training {name}", flush=True)
        metrics.append(train_one_model(name, data, split, cfg, output_dir))
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    _bootstrap_metric_intervals(metrics_frame, output_dir)
    gate = evaluate_exit_gate(metrics_frame)
    synthetic_fixture = bool(all(str(x).startswith("T_H") for x in data.trajectory_ids[: min(len(data), 10)]))
    gate["performance_gate_passed"] = gate["status"] == "PASS"
    gate["evaluation_scope"] = "synthetic_software_fixture" if synthetic_fixture else "external_longitudinal_dataset"
    gate["foundation_pretraining_allowed"] = bool(gate["performance_gate_passed"] and not synthetic_fixture)
    gate["foundation_pretraining_status"] = (
        "ELIGIBLE_AFTER_HUMAN_REVIEW" if gate["foundation_pretraining_allowed"]
        else "BLOCKED_REAL_LONGITUDINAL_GATE_REQUIRED" if synthetic_fixture
        else "BLOCKED_PERFORMANCE_GATE"
    )
    json_dump(gate, output_dir / "foundation_pretraining_gate.json")

    split_diagnostics = []
    for mode in ["dose_holdout", "sequence_holdout", "temporal_extrapolation", "donor_holdout"]:
        diagnostic = audit_split(data, make_split(data, mode, cfg.seed), mode)
        split_diagnostics.append(diagnostic)
    pd.DataFrame(split_diagnostics).to_csv(output_dir / "split_diagnostics.csv", index=False)
    build_dynamic_benchmark_figures(output_dir)
    _write_report(output_dir, metrics_frame, gate, split_audit, cfg)
    status = {
        "version": "1.7.0",
        "n_trajectories": len(data),
        "n_histories": len(set(data.history_ids.tolist())),
        "n_donors": len(set(data.donor_ids.tolist())),
        "models_evaluated": selected_models,
        "primary_split": "perturbation_history",
        "gate": gate,
        "synthetic_software_benchmark": synthetic_fixture,
    }
    json_dump(status, output_dir / "dynamic_benchmark_status.json")
    finalize_dynamic_benchmark_package(output_dir, data, cfg, gate)
    return status


def validate_dynamic_benchmark(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    required = [
        "model_comparison.csv",
        "foundation_pretraining_gate.json",
        "split_manifest.csv",
        "split_audit.json",
        "metric_intervals.csv",
        "external_benchmark_contract.json",
        "optional_embedding_adapters.csv",
        "report/index.html",
        "dataset_card.md",
        "model_card.md",
        "data_leakage_policy.md",
        "benchmark_task_definitions.csv",
        "artifact_manifest.csv",
        "run_manifest.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing dynamic benchmark artifacts: {missing}")
    metrics = pd.read_csv(output_dir / "model_comparison.csv")
    if set(MODEL_ORDER) - set(metrics["model"]):
        raise ValueError("model registry incomplete")
    gate = json.loads((output_dir / "foundation_pretraining_gate.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    checks = {
        "all_models_present": not bool(set(MODEL_ORDER) - set(metrics["model"])),
        "no_history_leakage": not audit.get("history_leakage", True),
        "coverage_metrics_present": all(f"coverage_{x}" in metrics.columns for x in [50, 80, 90, 95]),
        "fate_metrics_present": all(x in metrics.columns for x in ["fate_accuracy", "fate_log_loss", "fate_brier", "fate_ece"]),
        "trajectory_metrics_present": all(x in metrics.columns for x in ["trajectory_rmse", "trajectory_mae", "trajectory_correlation", "calibrated_gaussian_nll"]),
        "gate_passed": gate.get("status") == "PASS",
        "foundation_pretraining_guard_present": "foundation_pretraining_allowed" in gate,
    }
    result = {"valid": all(checks.values()), "checks": checks, "gate": gate}
    json_dump(result, output_dir / "dynamic_benchmark_validation.json")
    if not result["valid"]:
        raise RuntimeError(f"dynamic benchmark validation failed: {checks}")
    return result


def build_dynamic_benchmark_figures(output_dir: str | Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt
    from .visualization.publication import COLORS, apply_publication_style, export_figure

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output_dir / "model_comparison.csv")
    gate = json.loads((output_dir / "foundation_pretraining_gate.json").read_text(encoding="utf-8"))
    split_manifest = pd.read_csv(output_dir / "split_manifest.csv")
    exports = []
    palette = {
        "static": COLORS["muted"],
        "dynamic": COLORS["blue"],
    }

    apply_publication_style("nature_double")
    ordered = metrics.sort_values("trajectory_rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 115 / 25.4))
    colors = [palette[x] for x in ordered["model_family"]]
    ax.barh(ordered["model"], ordered["trajectory_rmse"], color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_xlabel("Future-trajectory RMSE (standardized units; lower is better)")
    ax.set_title("Held-out perturbation-history forecasting")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    exports.append(asdict(export_figure(fig, figure_dir / "trajectory_forecast_benchmark.png", profile="nature_double", source_data={"panel_a": ordered}, metadata={"gate": gate["status"], "comparison": "complete held-out histories"}, synthetic_only=True)))
    plt.close(fig)

    ordered_fate = metrics.sort_values("fate_log_loss", ascending=True)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 115 / 25.4))
    colors = [palette[x] for x in ordered_fate["model_family"]]
    ax.barh(ordered_fate["model"], ordered_fate["fate_log_loss"], color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_xlabel("Fate log loss (lower is better)")
    ax.set_title("Held-out-history fate prediction")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    exports.append(asdict(export_figure(fig, figure_dir / "fate_prediction_benchmark.png", profile="nature_double", source_data={"panel_a": ordered_fate}, metadata={"winning_dynamic_model": gate.get("winning_dynamic_model")}, synthetic_only=True)))
    plt.close(fig)

    coverage_rows = []
    for _, row in metrics.iterrows():
        for level in [50, 80, 90, 95]:
            coverage_rows.append({"model": row["model"], "model_family": row["model_family"], "nominal_coverage": level / 100.0, "observed_coverage": float(row[f"coverage_{level}"])})
    coverage = pd.DataFrame(coverage_rows)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 120 / 25.4))
    for model, frame in coverage.groupby("model"):
        family = frame["model_family"].iloc[0]
        ax.plot(frame["nominal_coverage"], frame["observed_coverage"], marker="o", linewidth=1.0, alpha=0.78, label=model, color=palette[family])
    ax.plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color=COLORS["ink"], linewidth=0.8, label="Ideal")
    ax.set_xlim(0.45, 0.98)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Observed held-out coverage")
    ax.set_title("Validation-calibrated trajectory uncertainty")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, fontsize=6, loc="lower right")
    fig.tight_layout()
    exports.append(asdict(export_figure(fig, figure_dir / "uncertainty_coverage.png", profile="nature_double", source_data={"panel_a": coverage}, metadata={"calibration_set": "validation histories only"}, synthetic_only=True)))
    plt.close(fig)

    counts = split_manifest.groupby(["split"])["trajectory_id"].count().reset_index(name="n_trajectories")
    histories = split_manifest.groupby(["split"])["history_id"].nunique().reset_index(name="n_histories")
    split_data = counts.merge(histories, on="split")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
    x = np.arange(len(split_data))
    width = 0.36
    ax.bar(x - width / 2, split_data["n_trajectories"], width, label="Trajectories", color=COLORS["blue"])
    ax.bar(x + width / 2, split_data["n_histories"], width, label="Unique histories", color=COLORS["gold"])
    ax.set_xticks(x, split_data["split"])
    ax.set_ylabel("Count")
    ax.set_title("Leakage-resistant perturbation-history split")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    exports.append(asdict(export_figure(fig, figure_dir / "history_split_design.png", profile="nature_double", source_data={"panel_a": split_data}, metadata={"history_overlap_train_test": 0}, synthetic_only=True)))
    plt.close(fig)

    pd.DataFrame(exports).to_csv(figure_dir / "figure_inventory.csv", index=False)
    return exports


def validate_external_benchmark_data(data: DynamicBenchmarkData) -> dict[str, Any]:
    errors: list[str] = []
    n = len(data)
    if data.observations.ndim != 3:
        errors.append("observations_must_be_3d")
    if data.interventions.ndim != 3:
        errors.append("interventions_must_be_3d")
    if data.times.ndim != 2:
        errors.append("times_must_be_2d")
    if data.observations.shape[:2] != data.interventions.shape[:2] or data.observations.shape[:2] != data.times.shape:
        errors.append("trajectory_time_dimensions_do_not_align")
    for name, array in {
        "fates": data.fates,
        "trajectory_ids": data.trajectory_ids,
        "donor_ids": data.donor_ids,
        "history_ids": data.history_ids,
        "targets": data.targets,
        "doses": data.doses,
        "sequences": data.sequences,
    }.items():
        if len(array) != n:
            errors.append(f"{name}_length_mismatch")
    if len(set(map(str, data.trajectory_ids))) != n:
        errors.append("trajectory_ids_not_unique")
    if len(set(map(str, data.history_ids))) < 3:
        errors.append("insufficient_distinct_histories")
    if len(set(map(str, data.donor_ids))) < 2:
        errors.append("insufficient_distinct_donors")
    if np.any(np.diff(data.times, axis=1) < 0):
        errors.append("times_not_monotonic")
    if not np.isfinite(data.observations).all():
        errors.append("nonfinite_observations")
    if not np.isfinite(data.interventions).all():
        errors.append("nonfinite_interventions")
    return {
        "valid": not errors,
        "errors": errors,
        "n_trajectories": n,
        "n_histories": len(set(map(str, data.history_ids))),
        "n_donors": len(set(map(str, data.donor_ids))),
        "observation_dim": int(data.observations.shape[-1]) if data.observations.ndim == 3 else None,
        "intervention_dim": int(data.interventions.shape[-1]) if data.interventions.ndim == 3 else None,
    }


def save_external_benchmark_npz(data: DynamicBenchmarkData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validation = validate_external_benchmark_data(data)
    if not validation["valid"]:
        raise ValueError(f"invalid dynamic benchmark dataset: {validation['errors']}")
    np.savez_compressed(
        path,
        observations=data.observations,
        interventions=data.interventions,
        times=data.times,
        fates=data.fates,
        trajectory_ids=np.asarray(data.trajectory_ids, dtype=str),
        donor_ids=np.asarray(data.donor_ids, dtype=str),
        history_ids=np.asarray(data.history_ids, dtype=str),
        targets=np.asarray(data.targets, dtype=str),
        doses=data.doses,
        sequences=np.asarray(data.sequences, dtype=str),
        feature_names=np.asarray(data.feature_names, dtype=str),
        intervention_names=np.asarray(data.intervention_names, dtype=str),
        fate_names=np.asarray(data.fate_names, dtype=str),
        schema_version=np.asarray(["1.7.0"], dtype=str),
    )
    return path


def load_external_benchmark_npz(path: str | Path) -> DynamicBenchmarkData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "observations", "interventions", "times", "fates", "trajectory_ids",
            "donor_ids", "history_ids", "targets", "doses", "sequences",
            "feature_names", "intervention_names", "fate_names",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"external benchmark NPZ missing arrays: {missing}")
        data = DynamicBenchmarkData(
            observations=np.asarray(payload["observations"], dtype=np.float32),
            interventions=np.asarray(payload["interventions"], dtype=np.float32),
            times=np.asarray(payload["times"], dtype=np.float32),
            fates=np.asarray(payload["fates"], dtype=np.int64),
            trajectory_ids=np.asarray(payload["trajectory_ids"], dtype=object),
            donor_ids=np.asarray(payload["donor_ids"], dtype=object),
            history_ids=np.asarray(payload["history_ids"], dtype=object),
            targets=np.asarray(payload["targets"], dtype=object),
            doses=np.asarray(payload["doses"], dtype=np.float32),
            sequences=np.asarray(payload["sequences"], dtype=object),
            feature_names=[str(x) for x in payload["feature_names"].tolist()],
            intervention_names=[str(x) for x in payload["intervention_names"].tolist()],
            fate_names=[str(x) for x in payload["fate_names"].tolist()],
        )
    validation = validate_external_benchmark_data(data)
    if not validation["valid"]:
        raise ValueError(f"external benchmark validation failed: {validation['errors']}")
    return data


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_dynamic_benchmark_package(
    output_dir: str | Path,
    data: DynamicBenchmarkData,
    cfg: DynamicBenchmarkConfig,
    gate: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tasks = pd.DataFrame(
        [
            {"task_id": "DB-01", "split": "perturbation_history", "primary": True, "question": "generalization to unseen target-dose-sequence histories"},
            {"task_id": "DB-02", "split": "dose_holdout", "primary": False, "question": "generalization to an unseen dose"},
            {"task_id": "DB-03", "split": "sequence_holdout", "primary": False, "question": "generalization to an unseen temporal sequence"},
            {"task_id": "DB-04", "split": "temporal_extrapolation", "primary": False, "question": "forecasting after the final context observation"},
            {"task_id": "DB-05", "split": "donor_holdout", "primary": False, "question": "generalization to independent donors"},
        ]
    )
    tasks.to_csv(output_dir / "benchmark_task_definitions.csv", index=False)

    dataset_card = f"""# Dynamic benchmark dataset card

- Framework: CausaFlux
- Version: 1.7.0
- Trajectories: {len(data)}
- Donors: {len(set(map(str, data.donor_ids)))}
- Complete perturbation histories: {len(set(map(str, data.history_ids)))}
- Time points per trajectory: {data.observations.shape[1]}
- Observation features: {data.observations.shape[2]}
- Intervention channels: {data.interventions.shape[2]}
- Context observations: {cfg.context_steps}
- Forecast horizon: {cfg.horizon}

## Intended use

This deterministic fixture verifies history-dependent forecasting, leakage-resistant splitting, calibrated uncertainty, model comparison, and release-gate behavior.

## Prohibited interpretation

The fixture is synthetic. Its pathway values, fates, model rankings, and performance estimates are not biological findings and are not clinical guidance.
"""
    (output_dir / "dataset_card.md").write_text(dataset_card, encoding="utf-8")

    model_card = f"""# CausaFlux v1.7.0 dynamic benchmark model card

## Models

{chr(10).join(f'- {name}' for name in MODEL_ORDER)}

## Primary evaluation

Complete target × dose × sequence histories are held out. Validation data calibrate predictive intervals; test data are never used for training, hyperparameter selection, or calibration.

## Foundation-pretraining gate

- Status: **{gate['status']}**
- Winning dynamic model: **{gate.get('winning_dynamic_model') or 'none'}**
- Passing dynamic models: {', '.join(gate.get('passing_dynamic_models', [])) or 'none'}
- Foundation pretraining allowed: **{gate['foundation_pretraining_allowed']}**

## Limitations

The Neural CDE is a dependency-light piecewise-linear Euler implementation. The PRESCIENT comparator is a lightweight latent-drift reference and not the upstream PRESCIENT software. scVI/scGPT support is provided as an interface for precomputed embeddings; no third-party checkpoints are bundled.
"""
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    leakage_policy = """# Dynamic benchmark leakage policy

1. Complete perturbation histories are indivisible split groups.
2. A history includes target, intervention identity, dose, order, schedule, pulse shape, and recovery interval.
3. Donor holdout is evaluated separately and does not substitute for history holdout.
4. Future intervention schedules are permitted inputs only when specified before the forecasted outcome.
5. External test cohorts cannot be used for feature selection, hyperparameter tuning, early stopping, variance calibration, or model selection.
6. scVI/scGPT embeddings must be generated without fitting on held-out benchmark outcomes or leaking test-cohort labels.
"""
    (output_dir / "data_leakage_policy.md").write_text(leakage_policy, encoding="utf-8")

    manifest_rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"artifact_manifest.csv", "run_manifest.json"}:
            continue
        manifest_rows.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
    run_manifest = {
        "framework": "CausaFlux",
        "version": "1.7.0",
        "workflow": "dynamic_model_benchmark",
        "primary_split": "perturbation_history",
        "models": MODEL_ORDER,
        "n_artifacts": int(len(manifest)),
        "artifact_manifest_sha256": _sha256_path(output_dir / "artifact_manifest.csv"),
        "foundation_pretraining_gate": gate,
        "synthetic_software_fixture": True,
    }
    json_dump(run_manifest, output_dir / "run_manifest.json")
    return run_manifest


def attach_precomputed_embeddings(
    data: DynamicBenchmarkData,
    embeddings: np.ndarray,
    *,
    prefix: str,
    mode: str = "append",
) -> DynamicBenchmarkData:
    """Attach scVI/scGPT or other precomputed embeddings without fitting inside the benchmark.

    Embeddings may be trajectory-level ``[trajectory, latent]`` and are broadcast across
    time, or observation-level ``[trajectory, time, latent]``. The benchmark assumes that
    embedding training respected the external-cohort and leakage policy.
    """
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim == 2:
        if values.shape[0] != len(data):
            raise ValueError("trajectory-level embedding row count does not match")
        values = np.repeat(values[:, None, :], data.observations.shape[1], axis=1)
    if values.ndim != 3 or values.shape[:2] != data.observations.shape[:2]:
        raise ValueError("embeddings must have shape [trajectory,latent] or [trajectory,time,latent]")
    names = [f"{prefix}_{index:03d}" for index in range(values.shape[-1])]
    if mode == "append":
        observations = np.concatenate([data.observations, values], axis=-1)
        feature_names = data.feature_names + names
    elif mode == "replace":
        observations = values
        feature_names = names
    else:
        raise ValueError("mode must be 'append' or 'replace'")
    return DynamicBenchmarkData(
        observations=observations.astype(np.float32),
        interventions=data.interventions.copy(),
        times=data.times.copy(),
        fates=data.fates.copy(),
        trajectory_ids=data.trajectory_ids.copy(),
        donor_ids=data.donor_ids.copy(),
        history_ids=data.history_ids.copy(),
        targets=data.targets.copy(),
        doses=data.doses.copy(),
        sequences=data.sequences.copy(),
        feature_names=feature_names,
        intervention_names=list(data.intervention_names),
        fate_names=list(data.fate_names),
    )
