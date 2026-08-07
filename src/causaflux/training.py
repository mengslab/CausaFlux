from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    ChronoDataset,
    Standardizer,
    grouped_split_indices,
    load_dataset,
    split_indices,
)
from .model import CausaFlux, CausaFluxConfig
from .utils import ensure_dir, json_dump, select_device, set_seed


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 7
    seed: int = 7
    num_workers: int = 0
    device: str = "auto"
    split_mode: str = "group"
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    observation_dropout: float = 0.05
    gradient_clip: float = 5.0


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    fate_class_weights=None,
    observation_dropout: float = 0.0,
    gradient_clip: float = 5.0,
):
    model.train(train)
    totals = {
        "loss": 0.0,
        "state_nll": 0.0,
        "state_mse": 0.0,
        "fate_loss": 0.0,
        "smoothness_loss": 0.0,
        "variance_penalty": 0.0,
    }
    n_batches = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        interventions = batch["interventions"].to(device)
        times = batch["times"].to(device)
        mask = batch["mask"].to(device)
        target_observation_mask = batch["observation_mask"].to(device)
        input_observation_mask = target_observation_mask
        if train and observation_dropout > 0:
            keep = (
                torch.rand_like(input_observation_mask) >= observation_dropout
            ).to(input_observation_mask.dtype)
            keep[:, 0] = 1.0
            input_observation_mask = input_observation_mask * keep
        fates = batch["fate"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(
                observations,
                interventions,
                times,
                mask,
                observation_mask=input_observation_mask,
                target_observation_mask=target_observation_mask,
            )
            losses = model.loss(
                outputs,
                observations,
                fates,
                fate_class_weights=fate_class_weights,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


def _resolve_split(base: ChronoDataset, config: TrainingConfig):
    if config.split_mode == "group":
        return grouped_split_indices(
            base.group_ids,
            seed=config.seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
        )
    if config.split_mode == "random":
        return split_indices(
            len(base),
            seed=config.seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
        )
    raise ValueError("split_mode must be 'group' or 'random'")


def train_model(
    data_path: str | Path,
    output_dir: str | Path,
    model_config: CausaFluxConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict:
    training_config = training_config or TrainingConfig()
    set_seed(training_config.seed)
    output_dir = ensure_dir(output_dir)
    device = select_device(training_config.device)

    base = load_dataset(data_path)
    if model_config is None:
        model_config = CausaFluxConfig(
            observation_dim=base.raw_observations.shape[-1],
            intervention_dim=base.interventions.shape[-1],
            n_fates=len(base.fate_names),
        )
    if model_config.observation_dim != base.raw_observations.shape[-1]:
        raise ValueError("model observation_dim does not match dataset")
    if model_config.intervention_dim != base.interventions.shape[-1]:
        raise ValueError("model intervention_dim does not match dataset")
    if model_config.n_fates != len(base.fate_names):
        raise ValueError("model n_fates does not match dataset")

    train_idx, val_idx, test_idx = _resolve_split(base, training_config)
    train_raw = base.subset(train_idx)
    standardizer = Standardizer.fit(
        train_raw.raw_observations,
        train_raw.mask,
        train_raw.observation_mask,
    )
    train_ds = base.subset(train_idx, standardizer=standardizer)
    val_ds = base.subset(val_idx, standardizer=standardizer)
    test_ds = base.subset(test_idx, standardizer=standardizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    model = CausaFlux(model_config).to(device)
    fate_counts = np.bincount(
        train_ds.fates, minlength=model_config.n_fates
    ).astype(np.float32)
    fate_class_weights = np.sqrt(fate_counts.sum() / np.maximum(fate_counts, 1.0))
    fate_class_weights = fate_class_weights / fate_class_weights.mean()
    fate_class_weights_tensor = torch.from_numpy(fate_class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, training_config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            fate_class_weights=fate_class_weights_tensor,
            observation_dropout=training_config.observation_dropout,
            gradient_clip=training_config.gradient_clip,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            fate_class_weights=fate_class_weights_tensor,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} state_rmse={np.sqrt(max(val_metrics['state_mse'], 0)):.4f} "
            f"device={device}"
        )
        if val_metrics["loss"] < best_val - 1e-5:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": "0.2",
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "training_config": asdict(training_config),
                    "standardizer_mean": standardizer.mean,
                    "standardizer_std": standardizer.std,
                    "train_indices": train_idx,
                    "val_indices": val_idx,
                    "test_indices": test_idx,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "fate_class_weights": fate_class_weights,
                    "feature_names": base.feature_names,
                    "intervention_names": base.intervention_names,
                    "fate_names": base.fate_names,
                    "dataset_summary": base.summary(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                print(f"early stopping at epoch {epoch}")
                break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    split_manifest = {
        "split_mode": training_config.split_mode,
        "train_trajectory_ids": base.trajectory_ids[train_idx].tolist(),
        "validation_trajectory_ids": base.trajectory_ids[val_idx].tolist(),
        "test_trajectory_ids": base.trajectory_ids[test_idx].tolist(),
        "train_group_ids": sorted(set(base.group_ids[train_idx].tolist())),
        "validation_group_ids": sorted(set(base.group_ids[val_idx].tolist())),
        "test_group_ids": sorted(set(base.group_ids[test_idx].tolist())),
    }
    json_dump(split_manifest, output_dir / "split_manifest.json")
    json_dump(
        {
            "device": str(device),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "checkpoint": str(checkpoint_path),
            "dataset": base.summary(),
        },
        output_dir / "training_summary.json",
    )
    return {
        "checkpoint": checkpoint_path,
        "history": history,
        "test_dataset": test_ds,
        "device": device,
    }


def load_checkpoint(path: str | Path, device: str = "auto"):
    resolved_device = select_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    config = CausaFluxConfig(**checkpoint["model_config"])
    model = CausaFlux(config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    standardizer = Standardizer(
        mean=np.asarray(checkpoint["standardizer_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["standardizer_std"], dtype=np.float32),
    )
    return model, standardizer, checkpoint, resolved_device
