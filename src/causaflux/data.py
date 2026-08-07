from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_NAMES = [
    "XBP1_activity",
    "ATF4_activity",
    "ATF6_activity",
    "calcium_homeostasis",
    "redox_balance",
    "mitochondrial_function",
    "protein_aggregation",
    "inflammatory_signaling",
    "metabolic_capacity",
    "viability",
    "adaptive_reserve",
    "commitment_signal",
]

INTERVENTION_NAMES = [
    "ER_stress",
    "IRE1_inhibition",
    "PERK_inhibition",
    "ATF6_activation",
]

FATE_NAMES = ["recovery", "persistent_dysfunction", "death"]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        trajectory_mask: np.ndarray,
        observation_mask: np.ndarray | None = None,
    ) -> "Standardizer":
        values = np.asarray(values, dtype=np.float32)
        trajectory_mask = np.asarray(trajectory_mask, dtype=bool)
        if observation_mask is None:
            observation_mask = np.ones_like(values, dtype=bool)
        else:
            observation_mask = np.asarray(observation_mask, dtype=bool)
        valid = observation_mask & trajectory_mask[..., None] & np.isfinite(values)
        mean = np.zeros(values.shape[-1], dtype=np.float32)
        std = np.ones(values.shape[-1], dtype=np.float32)
        for index in range(values.shape[-1]):
            selected = values[..., index][valid[..., index]]
            if selected.size:
                mean[index] = float(selected.mean())
                feature_std = float(selected.std())
                std[index] = feature_std if feature_std >= 1e-6 else 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.std + self.mean).astype(np.float32)


class ChronoDataset(Dataset):
    """Padded irregular-time trajectories with feature-level missingness masks."""

    def __init__(
        self,
        times: np.ndarray,
        observations: np.ndarray,
        interventions: np.ndarray,
        mask: np.ndarray,
        fates: np.ndarray,
        trajectory_ids: np.ndarray | None = None,
        observation_mask: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        intervention_names: Sequence[str] | None = None,
        fate_names: Sequence[str] | None = None,
        standardizer: Standardizer | None = None,
        fit_standardizer: bool = False,
    ) -> None:
        self.times = np.asarray(times, dtype=np.float32)
        self.raw_observations = np.asarray(observations, dtype=np.float32)
        self.interventions = np.asarray(interventions, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=np.float32)
        self.fates = np.asarray(fates, dtype=np.int64)
        n = len(self.times)
        self.trajectory_ids = (
            np.arange(n).astype(str)
            if trajectory_ids is None
            else np.asarray(trajectory_ids).astype(str)
        )
        self.group_ids = (
            self.trajectory_ids.copy()
            if group_ids is None
            else np.asarray(group_ids).astype(str)
        )
        if observation_mask is None:
            observation_mask = np.isfinite(self.raw_observations).astype(np.float32)
        self.observation_mask = np.asarray(observation_mask, dtype=np.float32)
        self.raw_observations = np.nan_to_num(self.raw_observations, nan=0.0)

        self.feature_names = list(feature_names or _default_names(FEATURE_NAMES, self.raw_observations.shape[-1], "feature"))
        self.intervention_names = list(
            intervention_names
            or _default_names(INTERVENTION_NAMES, self.interventions.shape[-1], "intervention")
        )
        self.fate_names = list(fate_names or FATE_NAMES)

        self._validate()
        if fit_standardizer:
            standardizer = Standardizer.fit(
                self.raw_observations,
                self.mask,
                self.observation_mask,
            )
        self.standardizer = standardizer
        transformed = (
            self.raw_observations
            if standardizer is None
            else standardizer.transform(self.raw_observations)
        )
        self.observations = np.where(self.observation_mask > 0, transformed, 0.0).astype(np.float32)

    def _validate(self) -> None:
        if self.times.ndim != 2:
            raise ValueError("times must have shape [N, T]")
        n, t = self.times.shape
        if self.raw_observations.ndim != 3 or self.raw_observations.shape[:2] != (n, t):
            raise ValueError("observations must have shape [N, T, D]")
        if self.interventions.ndim != 3 or self.interventions.shape[:2] != (n, t):
            raise ValueError("interventions must have shape [N, T, U]")
        if self.observation_mask.shape != self.raw_observations.shape:
            raise ValueError("observation_mask must have the same shape as observations")
        if self.mask.shape != (n, t):
            raise ValueError("mask must have shape [N, T]")
        if self.fates.shape != (n,):
            raise ValueError("fates must have shape [N]")
        if self.trajectory_ids.shape != (n,) or self.group_ids.shape != (n,):
            raise ValueError("trajectory_ids and group_ids must have shape [N]")
        if len(self.feature_names) != self.raw_observations.shape[-1]:
            raise ValueError("feature_names length does not match observation dimension")
        if len(self.intervention_names) != self.interventions.shape[-1]:
            raise ValueError("intervention_names length does not match intervention dimension")
        if np.any((self.mask != 0) & (self.mask != 1)):
            raise ValueError("mask must contain only 0 and 1")
        if np.any((self.observation_mask != 0) & (self.observation_mask != 1)):
            raise ValueError("observation_mask must contain only 0 and 1")
        for index in range(n):
            valid_times = self.times[index, self.mask[index].astype(bool)]
            if valid_times.size > 1 and np.any(np.diff(valid_times) < -1e-6):
                raise ValueError(f"times must be nondecreasing in trajectory {index}")
        if self.fates.size and (self.fates.min() < 0 or self.fates.max() >= len(self.fate_names)):
            raise ValueError("fate labels are outside the configured fate_names range")

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        standardizer: Standardizer | None = None,
        fit_standardizer: bool = False,
    ) -> "ChronoDataset":
        payload = np.load(path, allow_pickle=True)
        keys = set(payload.files)
        observations = payload["observations"]
        return cls(
            times=payload["times"],
            observations=observations,
            interventions=payload["interventions"],
            mask=payload["mask"],
            fates=payload["fates"],
            trajectory_ids=payload["trajectory_ids"] if "trajectory_ids" in keys else None,
            observation_mask=(
                payload["observation_mask"]
                if "observation_mask" in keys
                else np.isfinite(observations).astype(np.float32)
            ),
            group_ids=payload["group_ids"] if "group_ids" in keys else None,
            feature_names=payload["feature_names"].tolist() if "feature_names" in keys else None,
            intervention_names=(
                payload["intervention_names"].tolist() if "intervention_names" in keys else None
            ),
            fate_names=payload["fate_names"].tolist() if "fate_names" in keys else None,
            standardizer=standardizer,
            fit_standardizer=fit_standardizer,
        )

    @classmethod
    def from_long_csv(
        cls,
        path: str | Path,
        feature_names: Sequence[str] | None = None,
        intervention_names: Sequence[str] | None = None,
        fate_names: Sequence[str] | None = None,
        trajectory_column: str = "trajectory_id",
        time_column: str = "time",
        fate_column: str = "fate",
        group_column: str = "group_id",
    ) -> "ChronoDataset":
        frame = pd.read_csv(path)
        required = {trajectory_column, time_column, fate_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        fate_names = list(fate_names or FATE_NAMES)
        feature_names = list(feature_names or [name for name in FEATURE_NAMES if name in frame.columns])
        intervention_names = list(
            intervention_names or [name for name in INTERVENTION_NAMES if name in frame.columns]
        )
        if not feature_names:
            raise ValueError("No feature columns were found; provide feature_names explicitly")
        if not intervention_names:
            raise ValueError("No intervention columns were found; provide intervention_names explicitly")
        absent = [name for name in feature_names + intervention_names if name not in frame.columns]
        if absent:
            raise ValueError(f"CSV is missing configured data columns: {absent}")

        frame = frame.sort_values([trajectory_column, time_column], kind="stable")
        grouped = list(frame.groupby(trajectory_column, sort=False))
        n = len(grouped)
        max_steps = max(len(group) for _, group in grouped)
        times = np.zeros((n, max_steps), dtype=np.float32)
        observations = np.zeros((n, max_steps, len(feature_names)), dtype=np.float32)
        observation_mask = np.zeros_like(observations)
        interventions = np.zeros((n, max_steps, len(intervention_names)), dtype=np.float32)
        mask = np.zeros((n, max_steps), dtype=np.float32)
        fates = np.zeros(n, dtype=np.int64)
        trajectory_ids = np.empty(n, dtype=object)
        group_ids = np.empty(n, dtype=object)

        fate_lookup = {name: index for index, name in enumerate(fate_names)}
        for index, (trajectory_id, group) in enumerate(grouped):
            steps = len(group)
            trajectory_ids[index] = str(trajectory_id)
            group_ids[index] = str(group[group_column].iloc[0]) if group_column in group else str(trajectory_id)
            times[index, :steps] = group[time_column].to_numpy(dtype=np.float32)
            mask[index, :steps] = 1.0
            feature_values = group[feature_names].to_numpy(dtype=np.float32)
            observed = np.isfinite(feature_values)
            observations[index, :steps] = np.nan_to_num(feature_values, nan=0.0)
            observation_mask[index, :steps] = observed.astype(np.float32)
            interventions[index, :steps] = np.nan_to_num(
                group[intervention_names].to_numpy(dtype=np.float32), nan=0.0
            )
            unique_fates = group[fate_column].dropna().astype(str).unique()
            if len(unique_fates) != 1:
                raise ValueError(f"trajectory {trajectory_id} must have exactly one fate")
            fate_text = unique_fates[0]
            if fate_text in fate_lookup:
                fates[index] = fate_lookup[fate_text]
            else:
                try:
                    fates[index] = int(float(fate_text))
                except ValueError as exc:
                    raise ValueError(f"unknown fate '{fate_text}' in trajectory {trajectory_id}") from exc

        return cls(
            times=times,
            observations=observations,
            interventions=interventions,
            mask=mask,
            fates=fates,
            trajectory_ids=trajectory_ids,
            observation_mask=observation_mask,
            group_ids=group_ids,
            feature_names=feature_names,
            intervention_names=intervention_names,
            fate_names=fate_names,
        )

    def to_npz(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=self.times,
            observations=self.raw_observations,
            interventions=self.interventions,
            mask=self.mask,
            observation_mask=self.observation_mask,
            fates=self.fates,
            trajectory_ids=self.trajectory_ids,
            group_ids=self.group_ids,
            feature_names=np.asarray(self.feature_names),
            intervention_names=np.asarray(self.intervention_names),
            fate_names=np.asarray(self.fate_names),
        )

    def to_long_csv(self, path: str | Path) -> None:
        rows: list[dict[str, object]] = []
        for index, trajectory_id in enumerate(self.trajectory_ids):
            fate_name = self.fate_names[int(self.fates[index])]
            for step in range(self.times.shape[1]):
                if self.mask[index, step] == 0:
                    continue
                row: dict[str, object] = {
                    "trajectory_id": trajectory_id,
                    "group_id": self.group_ids[index],
                    "time": float(self.times[index, step]),
                    "fate": fate_name,
                }
                for feature_index, name in enumerate(self.feature_names):
                    row[name] = (
                        float(self.raw_observations[index, step, feature_index])
                        if self.observation_mask[index, step, feature_index] > 0
                        else np.nan
                    )
                for intervention_index, name in enumerate(self.intervention_names):
                    row[name] = float(self.interventions[index, step, intervention_index])
                rows.append(row)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    def subset(self, indices: np.ndarray, standardizer: Standardizer | None = None) -> "ChronoDataset":
        indices = np.asarray(indices, dtype=int)
        return ChronoDataset(
            self.times[indices],
            self.raw_observations[indices],
            self.interventions[indices],
            self.mask[indices],
            self.fates[indices],
            trajectory_ids=self.trajectory_ids[indices],
            observation_mask=self.observation_mask[indices],
            group_ids=self.group_ids[indices],
            feature_names=self.feature_names,
            intervention_names=self.intervention_names,
            fate_names=self.fate_names,
            standardizer=standardizer,
        )

    def summary(self) -> dict[str, object]:
        valid_steps = self.mask.sum(axis=1)
        observed_fraction = float(
            (self.observation_mask * self.mask[..., None]).sum()
            / max(1.0, self.mask.sum() * self.raw_observations.shape[-1])
        )
        counts = np.bincount(self.fates, minlength=len(self.fate_names))
        return {
            "n_trajectories": int(len(self)),
            "n_groups": int(len(np.unique(self.group_ids))),
            "max_steps": int(self.times.shape[1]),
            "mean_valid_steps": float(valid_steps.mean()),
            "observation_dim": int(self.raw_observations.shape[-1]),
            "intervention_dim": int(self.interventions.shape[-1]),
            "observed_feature_fraction": observed_fraction,
            "fate_counts": {name: int(counts[index]) for index, name in enumerate(self.fate_names)},
            "feature_names": self.feature_names,
            "intervention_names": self.intervention_names,
        }

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "times": torch.from_numpy(self.times[index]),
            "observations": torch.from_numpy(self.observations[index]),
            "observation_mask": torch.from_numpy(self.observation_mask[index]),
            "interventions": torch.from_numpy(self.interventions[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "fate": torch.tensor(self.fates[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def _default_names(defaults: Sequence[str], dimension: int, prefix: str) -> list[str]:
    if len(defaults) == dimension:
        return list(defaults)
    return [f"{prefix}_{index}" for index in range(dimension)]


def load_dataset(path: str | Path, **kwargs) -> ChronoDataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return ChronoDataset.from_npz(path, **kwargs)
    if suffix == ".csv":
        return ChronoDataset.from_long_csv(path, **kwargs)
    raise ValueError(f"unsupported dataset format: {path}; expected .npz or .csv")


def split_indices(
    n: int,
    seed: int = 7,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError("at least three trajectories are required for train/val/test splitting")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def grouped_split_indices(
    group_ids: Sequence[str],
    seed: int = 7,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids).astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return split_indices(len(groups), seed, train_fraction, val_fraction)
    train_groups, val_groups, test_groups = split_indices(
        len(unique_groups), seed, train_fraction, val_fraction
    )
    train_set = set(unique_groups[train_groups])
    val_set = set(unique_groups[val_groups])
    test_set = set(unique_groups[test_groups])
    train = np.flatnonzero(np.isin(groups, list(train_set)))
    val = np.flatnonzero(np.isin(groups, list(val_set)))
    test = np.flatnonzero(np.isin(groups, list(test_set)))
    if min(len(train), len(val), len(test)) == 0:
        return split_indices(len(groups), seed, train_fraction, val_fraction)
    return train, val, test


def iter_valid_rows(
    dataset: ChronoDataset,
) -> Iterator[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]]:
    for i, trajectory_id in enumerate(dataset.trajectory_ids):
        for t in range(dataset.times.shape[1]):
            if dataset.mask[i, t] > 0:
                yield (
                    trajectory_id,
                    float(dataset.times[i, t]),
                    dataset.raw_observations[i, t],
                    dataset.observation_mask[i, t],
                    dataset.interventions[i, t],
                )
