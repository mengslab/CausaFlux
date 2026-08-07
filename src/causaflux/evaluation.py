from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import load_dataset
from .training import load_checkpoint
from .utils import json_dump


def _expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    if len(labels) == 0:
        return 0.0
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: int = 128,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, standardizer, checkpoint, resolved_device = load_checkpoint(checkpoint_path, device)
    base = load_dataset(data_path)
    test_indices = np.asarray(
        checkpoint.get("test_indices", np.arange(len(base))), dtype=int
    )
    dataset = base.subset(test_indices, standardizer=standardizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    squared_error_sum = 0.0
    nll_sum = 0.0
    valid_values = 0.0
    coverage_count = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    feature_squared_error = np.zeros(len(base.feature_names), dtype=np.float64)
    feature_counts = np.zeros(len(base.feature_names), dtype=np.float64)
    prediction_rows: list[dict[str, object]] = []
    trajectory_cursor = 0

    with torch.no_grad():
        for batch in loader:
            obs = batch["observations"].to(resolved_device)
            observation_mask = batch["observation_mask"].to(resolved_device)
            interventions = batch["interventions"].to(resolved_device)
            times = batch["times"].to(resolved_device)
            mask = batch["mask"].to(resolved_device)
            fate = batch["fate"].to(resolved_device)
            outputs = model(
                obs,
                interventions,
                times,
                mask,
                observation_mask=observation_mask,
                target_observation_mask=observation_mask,
            )
            target = obs[:, 1:]
            mean = outputs["next_observation_mean"]
            log_variance = outputs["next_observation_log_variance"]
            target_mask = outputs["target_observation_mask"]
            error = mean - target
            variance = torch.exp(log_variance)
            gaussian_nll = 0.5 * (error.pow(2) / variance + log_variance)
            squared_error_sum += float((error.pow(2) * target_mask).sum().cpu())
            nll_sum += float((gaussian_nll * target_mask).sum().cpu())
            valid_values += float(target_mask.sum().cpu())
            lower = mean - 1.645 * torch.sqrt(variance)
            upper = mean + 1.645 * torch.sqrt(variance)
            coverage_count += float(
                (((target >= lower) & (target <= upper)).float() * target_mask).sum().cpu()
            )

            raw_mean = standardizer.inverse_transform(mean.cpu().numpy())
            raw_target = standardizer.inverse_transform(target.cpu().numpy())
            raw_mask = target_mask.cpu().numpy()
            raw_error_squared = (raw_mean - raw_target) ** 2
            feature_squared_error += (raw_error_squared * raw_mask).sum(axis=(0, 1))
            feature_counts += raw_mask.sum(axis=(0, 1))

            prob = torch.softmax(outputs["fate_logits"], dim=-1)
            pred = prob.argmax(dim=-1)
            batch_labels = fate.cpu().numpy()
            batch_predictions = pred.cpu().numpy()
            batch_probabilities = prob.cpu().numpy()
            labels.extend(batch_labels.tolist())
            predictions.extend(batch_predictions.tolist())
            probabilities.extend(batch_probabilities.tolist())
            for row_index in range(len(batch_labels)):
                dataset_index = trajectory_cursor + row_index
                row: dict[str, object] = {
                    "trajectory_id": dataset.trajectory_ids[dataset_index],
                    "actual_fate": dataset.fate_names[int(batch_labels[row_index])],
                    "predicted_fate": dataset.fate_names[int(batch_predictions[row_index])],
                }
                for fate_index, fate_name in enumerate(dataset.fate_names):
                    row[f"probability_{fate_name}"] = float(
                        batch_probabilities[row_index, fate_index]
                    )
                prediction_rows.append(row)
            trajectory_cursor += len(batch_labels)

    labels_array = np.asarray(labels, dtype=int)
    pred_array = np.asarray(predictions, dtype=int)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    n_fates = len(dataset.fate_names)
    confusion = np.zeros((n_fates, n_fates), dtype=int)
    for actual, predicted in zip(labels_array, pred_array):
        confusion[int(actual), int(predicted)] += 1
    accuracy = float((labels_array == pred_array).mean()) if len(labels_array) else 0.0
    rmse_standardized = float(np.sqrt(squared_error_sum / max(valid_values, 1.0)))
    mean_nll = float(nll_sum / max(valid_values, 1.0))
    brier = (
        float(
            np.mean(
                np.sum(
                    (probability_array - np.eye(n_fates)[labels_array]) ** 2,
                    axis=1,
                )
            )
        )
        if len(labels_array)
        else 0.0
    )
    per_feature_rmse = np.sqrt(
        feature_squared_error / np.maximum(feature_counts, 1.0)
    )
    feature_frame = pd.DataFrame(
        {
            "feature": dataset.feature_names,
            "rmse_raw_units": per_feature_rmse,
            "n_observed_targets": feature_counts.astype(int),
        }
    )
    feature_frame.to_csv(output_dir / "per_feature_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "fate_predictions.csv", index=False)
    confusion_frame = pd.DataFrame(
        confusion,
        index=[f"actual_{name}" for name in dataset.fate_names],
        columns=[f"predicted_{name}" for name in dataset.fate_names],
    )
    confusion_frame.to_csv(output_dir / "confusion_matrix.csv")

    metrics = {
        "n_test": int(len(dataset)),
        "next_state_rmse_standardized": rmse_standardized,
        "next_state_mean_gaussian_nll": mean_nll,
        "predictive_interval_90_coverage": float(coverage_count / max(valid_values, 1.0)),
        "fate_accuracy": accuracy,
        "fate_brier_score": brier,
        "fate_expected_calibration_error": _expected_calibration_error(
            probability_array, labels_array
        ),
        "confusion_matrix": confusion.tolist(),
        "fate_order": dataset.fate_names,
        "per_feature_rmse_raw_units": {
            name: float(value)
            for name, value in zip(dataset.feature_names, per_feature_rmse)
        },
        "device": str(resolved_device),
    }
    json_dump(metrics, output_dir / "metrics.json")
    return metrics
