from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .causal_data import BIOMARKER_FEATURES, STATE_ORDER

_EPS = 1e-8


@dataclass(frozen=True)
class BenchmarkResult:
    predictions: pd.DataFrame
    all_predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration_metrics: pd.DataFrame
    split_manifest: pd.DataFrame
    bootstrap_metrics: pd.DataFrame
    ensemble_uncertainty: pd.DataFrame
    bootstrap_predictions: pd.DataFrame
    selected_model: str
    selected_variant: str


def _model_factories(seed: int) -> dict[str, Callable[[], object]]:
    """Transparent probability-producing baselines used in every release benchmark."""

    return {
        "dummy_prior": lambda: DummyClassifier(strategy="prior"),
        "logistic_l2": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=0.0005,
                        max_iter=5000,
                        tol=1e-4,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logistic_sparse": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l1",
                        alpha=0.0008,
                        max_iter=5000,
                        tol=1e-4,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "sgd_elasticnet": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="elasticnet",
                        alpha=0.001,
                        l1_ratio=0.20,
                        max_iter=4000,
                        tol=1e-4,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "lda_shrinkage": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        ),
    }


def _splitter(groups: pd.Series, mode: str, n_splits: int):
    unique_groups = int(groups.nunique())
    if unique_groups < 2:
        raise ValueError("At least two donor groups are required")
    if mode == "leave_one_donor_out":
        return LeaveOneGroupOut()
    if mode == "group_kfold":
        return GroupKFold(n_splits=min(max(2, n_splits), unique_groups))
    raise ValueError("split mode must be 'leave_one_donor_out' or 'group_kfold'")


def _aligned_probabilities(model, X: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = [str(value) for value in model.classes_]
    output = np.full((len(X), len(classes)), _EPS, dtype=float)
    for source_index, label in enumerate(model_classes):
        if label in classes:
            output[:, classes.index(label)] = raw[:, source_index]
    output = np.clip(output, _EPS, None)
    return output / output.sum(axis=1, keepdims=True)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    recalls: list[float] = []
    for class_index in range(n_classes):
        selected = labels == class_index
        if selected.any():
            recalls.append(float(np.mean(predictions[selected] == class_index)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    selected = np.clip(probabilities[np.arange(len(labels)), labels], _EPS, 1.0)
    return float(-np.mean(np.log(selected)))


def _multiclass_brier(probabilities: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    target = np.eye(n_classes, dtype=float)[labels]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if selected.any():
            value += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(value)


def classwise_ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    values: list[float] = []
    for class_index in range(probabilities.shape[1]):
        binary = (labels == class_index).astype(float)
        score = probabilities[:, class_index]
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            selected = (score >= lower) & (score < upper if index < bins - 1 else score <= upper)
            if selected.any():
                ece += float(selected.mean()) * abs(float(binary[selected].mean()) - float(score[selected].mean()))
        values.append(ece)
    return float(np.mean(values))


def _metric_record(
    model_name: str,
    variant: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    return {
        "model": model_name,
        "variant": variant,
        "balanced_accuracy": _balanced_accuracy(labels, predictions, len(classes)),
        "log_loss": _multiclass_log_loss(labels, probabilities),
        "brier_score": _multiclass_brier(probabilities, labels, len(classes)),
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
        "classwise_ece": classwise_ece(probabilities, labels),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "n": int(len(labels)),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_platt_numpy(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a regularized sigmoid calibrator with deterministic gradient descent."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=float)
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if scale < 1e-8:
        scale = 1.0
    z = (x - mean) / scale
    prevalence = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
    a = 1.0
    b = float(np.log(prevalence / (1.0 - prevalence)))
    learning_rate = 0.08
    regularization = 0.01
    for iteration in range(500):
        prediction = _sigmoid(a * z + b)
        error = prediction - y
        grad_a = float(np.mean(error * z) + regularization * a)
        grad_b = float(np.mean(error))
        step = learning_rate / np.sqrt(1.0 + iteration / 50.0)
        a -= step * grad_a
        b -= step * grad_b
        if abs(grad_a) + abs(grad_b) < 1e-7:
            break
    return a, b, mean, scale


def _crossfit_sigmoid(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    seed: int,
) -> np.ndarray:
    del seed  # deterministic NumPy optimizer
    output = np.zeros_like(probabilities)
    for held_group in np.unique(groups):
        train = groups != held_group
        test = ~train
        calibrated = np.zeros((int(test.sum()), probabilities.shape[1]), dtype=float)
        for class_index in range(probabilities.shape[1]):
            binary = (labels[train] == class_index).astype(float)
            if len(np.unique(binary)) < 2:
                calibrated[:, class_index] = probabilities[test, class_index]
                continue
            train_score = np.log(
                np.clip(probabilities[train, class_index], _EPS, 1.0 - _EPS)
                / np.clip(1.0 - probabilities[train, class_index], _EPS, 1.0)
            )
            test_score = np.log(
                np.clip(probabilities[test, class_index], _EPS, 1.0 - _EPS)
                / np.clip(1.0 - probabilities[test, class_index], _EPS, 1.0)
            )
            a, b, mean, scale = _fit_platt_numpy(train_score, binary)
            calibrated[:, class_index] = _sigmoid(a * ((test_score - mean) / scale) + b)
        calibrated = np.clip(calibrated, _EPS, None)
        output[test] = calibrated / calibrated.sum(axis=1, keepdims=True)
    return output


def _pava_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Pure NumPy pooled-adjacent-violators calibration with interpolation."""
    order = np.argsort(x_train, kind="mergesort")
    x = np.asarray(x_train, dtype=float)[order]
    y = np.asarray(y_train, dtype=float)[order]
    blocks: list[list[float]] = []  # x_sum, y_sum, weight, x_min, x_max
    for xi, yi in zip(x, y):
        blocks.append([float(xi), float(yi), 1.0, float(xi), float(xi)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[1] / left[2] <= right[1] / right[2] + 1e-15:
                break
            merged = [
                left[0] + right[0],
                left[1] + right[1],
                left[2] + right[2],
                left[3],
                right[4],
            ]
            blocks[-2:] = [merged]
    centers = np.array([block[0] / block[2] for block in blocks], dtype=float)
    values = np.array([block[1] / block[2] for block in blocks], dtype=float)
    if len(centers) == 1:
        return np.full(len(x_test), values[0], dtype=float)
    return np.interp(np.asarray(x_test, dtype=float), centers, values, left=values[0], right=values[-1])


def _crossfit_isotonic(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(probabilities)
    for held_group in np.unique(groups):
        train = groups != held_group
        test = ~train
        calibrated = np.zeros((int(test.sum()), probabilities.shape[1]), dtype=float)
        for class_index in range(probabilities.shape[1]):
            binary = (labels[train] == class_index).astype(float)
            if len(np.unique(binary)) < 2:
                calibrated[:, class_index] = probabilities[test, class_index]
                continue
            calibrated[:, class_index] = _pava_predict(
                probabilities[train, class_index], binary, probabilities[test, class_index]
            )
        calibrated = np.clip(calibrated, _EPS, None)
        output[test] = calibrated / calibrated.sum(axis=1, keepdims=True)
    return output

def _donor_bootstrap_metrics(
    donor_ids: np.ndarray,
    labels: np.ndarray,
    probability_sets: dict[tuple[str, str], np.ndarray],
    classes: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    donors = np.unique(donor_ids)
    metric_names = [
        "balanced_accuracy",
        "log_loss",
        "brier_score",
        "expected_calibration_error",
        "classwise_ece",
    ]
    values: dict[tuple[str, str, str], list[float]] = {}
    for _ in range(n_bootstrap):
        sampled = rng.choice(donors, size=len(donors), replace=True)
        indices = np.concatenate([np.flatnonzero(donor_ids == donor) for donor in sampled])
        for (model_name, variant), probabilities in probability_sets.items():
            record = _metric_record(model_name, variant, probabilities[indices], labels[indices], classes)
            for metric in metric_names:
                values.setdefault((model_name, variant, metric), []).append(float(record[metric]))
    rows: list[dict[str, object]] = []
    for (model_name, variant, metric), samples in values.items():
        array = np.asarray(samples, dtype=float)
        rows.append(
            {
                "model": model_name,
                "variant": variant,
                "metric": metric,
                "bootstrap_mean": float(array.mean()),
                "bootstrap_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "n_bootstrap": int(len(array)),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "variant", "metric"]).reset_index(drop=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, _EPS, 1.0)
    return -np.sum(values * np.log(values), axis=-1)


def _ensemble_uncertainty(
    member_probabilities: dict[str, np.ndarray],
    row_metadata: pd.DataFrame,
    classes: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    names = list(member_probabilities)
    stack = np.stack([member_probabilities[name] for name in names], axis=0)
    mean = stack.mean(axis=0)
    member_entropy = _entropy(stack)
    predictive_entropy = _entropy(mean)
    mutual_information = np.maximum(0.0, predictive_entropy - member_entropy.mean(axis=0))
    selected = stack.argmax(axis=2)
    variation_ratio = np.asarray(
        [1.0 - np.bincount(selected[:, row], minlength=len(classes)).max() / len(names) for row in range(stack.shape[1])]
    )
    output = row_metadata.copy()
    output["ensemble_predicted_state"] = np.asarray(classes)[mean.argmax(axis=1)]
    output["ensemble_confidence"] = mean.max(axis=1)
    output["predictive_entropy"] = predictive_entropy
    output["expected_member_entropy"] = member_entropy.mean(axis=0)
    output["mutual_information"] = mutual_information
    output["variation_ratio"] = variation_ratio
    output["n_ensemble_members"] = len(names)
    output["ensemble_members"] = ";".join(names)
    for class_index, state in enumerate(classes):
        output[f"ensemble_probability_{state}"] = mean[:, class_index]
        output[f"model_sd_{state}"] = stack[:, :, class_index].std(axis=0, ddof=1) if len(names) > 1 else 0.0
        output[f"model_min_{state}"] = stack[:, :, class_index].min(axis=0)
        output[f"model_max_{state}"] = stack[:, :, class_index].max(axis=0)
    return mean, output


def _cluster_bootstrap_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    row_metadata: pd.DataFrame,
    classes: Sequence[str],
    factory: Callable[[], object],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    group_values = groups.astype(str).to_numpy()
    arrays = np.full((n_bootstrap, len(X), len(classes)), np.nan, dtype=float)
    donors = np.unique(group_values)
    for donor_index, held_donor in enumerate(donors):
        test_indices = np.flatnonzero(group_values == held_donor)
        train_donors = donors[donors != held_donor]
        for bootstrap_index in range(n_bootstrap):
            sampled_donors = rng.choice(train_donors, size=len(train_donors), replace=True)
            train_indices = np.concatenate([np.flatnonzero(group_values == donor) for donor in sampled_donors])
            model = factory()
            try:
                model.fit(X.iloc[train_indices], y.iloc[train_indices])
                arrays[bootstrap_index, test_indices] = _aligned_probabilities(
                    model, X.iloc[test_indices], list(classes)
                )
            except Exception:
                continue
    output = row_metadata.copy()
    for class_index, state in enumerate(classes):
        values = arrays[:, :, class_index]
        output[f"bootstrap_mean_{state}"] = np.nanmean(values, axis=0)
        successful = np.sum(np.isfinite(values), axis=0)
        output[f"bootstrap_sd_{state}"] = np.nanstd(values, axis=0, ddof=0)
        output.loc[successful == 0, f"bootstrap_sd_{state}"] = np.nan
        output[f"bootstrap_ci_low_{state}"] = np.nanquantile(values, 0.025, axis=0)
        output[f"bootstrap_ci_high_{state}"] = np.nanquantile(values, 0.975, axis=0)
    output["bootstrap_successful_replicates"] = np.sum(np.isfinite(arrays[:, :, 0]), axis=0)
    return output


def benchmark_linear_baselines(
    frame: pd.DataFrame,
    features: Sequence[str] | None = None,
    group_column: str = "donor_id",
    split_mode: str = "leave_one_donor_out",
    n_splits: int = 4,
    calibration_methods: Sequence[str] = ("sigmoid", "isotonic"),
    n_metric_bootstrap: int = 200,
    n_prediction_bootstrap: int = 40,
    bootstrap_model: str = "logistic_l2",
    seed: int = 31,
) -> BenchmarkResult:
    features = list(features or BIOMARKER_FEATURES)
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    tumor = tumor.loc[tumor["state"].isin(STATE_ORDER)].reset_index(drop=True)
    classes = list(STATE_ORDER)
    X = tumor[features]
    y_text = tumor["state"].astype(str)
    labels = y_text.map({name: index for index, name in enumerate(classes)}).to_numpy(dtype=int)
    groups = tumor[group_column].astype(str)
    splitter = _splitter(groups, split_mode, n_splits)
    splits = list(splitter.split(X, y_text, groups))
    split_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits, start=1):
        train_donors = sorted(groups.iloc[train].unique())
        test_donors = sorted(groups.iloc[test].unique())
        split_rows.append(
            {
                "fold": fold,
                "split_mode": split_mode,
                "n_train_rows": len(train),
                "n_test_rows": len(test),
                "n_train_donors": len(train_donors),
                "n_test_donors": len(test_donors),
                "train_donors": ";".join(train_donors),
                "test_donors": ";".join(test_donors),
                "donor_overlap": ";".join(sorted(set(train_donors) & set(test_donors))),
            }
        )
    split_manifest = pd.DataFrame(split_rows)
    if split_manifest["donor_overlap"].astype(bool).any():
        raise RuntimeError("Donor leakage detected in split manifest")

    factories = _model_factories(seed)
    probability_sets: dict[tuple[str, str], np.ndarray] = {}
    metrics: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    raw_by_model: dict[str, np.ndarray] = {}
    selected_by_model: dict[str, np.ndarray] = {}
    selected_variant_by_model: dict[str, str] = {}

    # Fit every base family before any calibration step. Some scientific
    # Python stacks combine BLAS, HDF5, and optimization libraries with native
    # thread state; grouping estimator refits first makes execution robust.
    for model_name, factory in factories.items():
        print(f"[CausaFlux]   fitting baseline family: {model_name}", flush=True)
        oof = np.zeros((len(tumor), len(classes)), dtype=float)
        for fold, (train, test) in enumerate(splits):
            model = factory()
            model.fit(X.iloc[train], y_text.iloc[train])
            oof[test] = _aligned_probabilities(model, X.iloc[test], classes)
        raw_by_model[model_name] = oof
        probability_sets[(model_name, "raw")] = oof

    metadata = tumor[["row_id", "donor_id", "lineage_id", "time_hours", "state"]].copy()
    if bootstrap_model not in factories or bootstrap_model == "dummy_prior":
        raise ValueError(f"Unsupported bootstrap_model: {bootstrap_model}")
    print("[CausaFlux]   donor-bootstrap probability intervals", flush=True)
    bootstrap_predictions = _cluster_bootstrap_predictions(
        X,
        y_text,
        groups,
        metadata,
        classes,
        factories[bootstrap_model],
        n_prediction_bootstrap,
        seed + 202,
    )
    if not bootstrap_predictions.empty:
        bootstrap_predictions["bootstrap_reference_model"] = bootstrap_model

    print("[CausaFlux]   cross-fitting probability calibration", flush=True)
    for model_name, oof in raw_by_model.items():
        print(f"[CausaFlux]     calibrating: {model_name}", flush=True)
        raw_metric = _metric_record(model_name, "raw", oof, labels, classes)
        metrics.append(raw_metric)
        variants = {"raw": oof}
        if "sigmoid" in calibration_methods:
            print(f"[CausaFlux]       sigmoid: {model_name}", flush=True)
            variants["sigmoid"] = _crossfit_sigmoid(oof, labels, groups.to_numpy(), classes, seed)
        if "isotonic" in calibration_methods:
            print(f"[CausaFlux]       isotonic: {model_name}", flush=True)
            variants["isotonic"] = _crossfit_isotonic(oof, labels, groups.to_numpy())
        for variant, probabilities in variants.items():
            if variant != "raw":
                probability_sets[(model_name, variant)] = probabilities
                record = _metric_record(model_name, variant, probabilities, labels, classes)
                metrics.append(record)
                calibration_rows.append(
                    {
                        "model": model_name,
                        "calibration": variant,
                        "delta_log_loss_vs_raw": record["log_loss"] - raw_metric["log_loss"],
                        "delta_brier_vs_raw": record["brier_score"] - raw_metric["brier_score"],
                        "delta_ece_vs_raw": record["expected_calibration_error"] - raw_metric["expected_calibration_error"],
                    }
                )
        candidates = [
            (variant, probs, _metric_record(model_name, variant, probs, labels, classes)["log_loss"])
            for variant, probs in variants.items()
        ]
        best_variant, best_probs, _ = min(candidates, key=lambda item: item[2])
        selected_by_model[model_name] = best_probs
        selected_variant_by_model[model_name] = best_variant

    print("[CausaFlux]   assembling calibrated ensemble", flush=True)
    member_probabilities = {
        f"{name}:{selected_variant_by_model[name]}": probabilities
        for name, probabilities in selected_by_model.items()
        if name != "dummy_prior"
    }
    ensemble_probabilities, uncertainty = _ensemble_uncertainty(member_probabilities, metadata, classes)
    probability_sets[("linear_ensemble", "member_calibrated_mean")] = ensemble_probabilities
    ensemble_metric = _metric_record(
        "linear_ensemble", "member_calibrated_mean", ensemble_probabilities, labels, classes
    )
    metrics.append(ensemble_metric)

    metric_frame = pd.DataFrame(metrics).sort_values(
        ["log_loss", "expected_calibration_error", "model", "variant"]
    ).reset_index(drop=True)
    metric_frame.insert(0, "rank_by_log_loss", np.arange(1, len(metric_frame) + 1))
    selected_row = metric_frame.loc[
        ~metric_frame["model"].isin(["dummy_prior", "linear_ensemble"])
    ].iloc[0]
    selected_model = str(selected_row["model"])
    selected_variant = str(selected_row["variant"])

    predictions = metadata.copy()
    predictions["selected_model"] = selected_model
    predictions["selected_variant"] = selected_variant
    selected_probabilities = probability_sets[(selected_model, selected_variant)]
    predictions["predicted_state"] = np.asarray(classes)[selected_probabilities.argmax(axis=1)]
    predictions["prediction_confidence"] = selected_probabilities.max(axis=1)
    for class_index, state in enumerate(classes):
        predictions[f"probability_{state}"] = selected_probabilities[:, class_index]

    all_prediction_frames: list[pd.DataFrame] = []
    for (model_name, variant), probabilities in probability_sets.items():
        part = metadata.copy()
        part["model"] = model_name
        part["variant"] = variant
        part["predicted_state"] = np.asarray(classes)[probabilities.argmax(axis=1)]
        part["confidence"] = probabilities.max(axis=1)
        for class_index, state in enumerate(classes):
            part[f"probability_{state}"] = probabilities[:, class_index]
        all_prediction_frames.append(part)
    all_predictions = pd.concat(all_prediction_frames, ignore_index=True)

    print("[CausaFlux]   donor-bootstrap metric intervals", flush=True)
    bootstrap_metrics = _donor_bootstrap_metrics(
        groups.to_numpy(), labels, probability_sets, classes, n_metric_bootstrap, seed + 101
    )
    return BenchmarkResult(
        predictions=predictions,
        all_predictions=all_predictions,
        metrics=metric_frame,
        calibration_metrics=pd.DataFrame(calibration_rows),
        split_manifest=split_manifest,
        bootstrap_metrics=bootstrap_metrics,
        ensemble_uncertainty=uncertainty,
        bootstrap_predictions=bootstrap_predictions,
        selected_model=selected_model,
        selected_variant=selected_variant,
    )


def transition_bootstrap_uncertainty(
    frame: pd.DataFrame,
    states: Sequence[str] = STATE_ORDER,
    n_bootstrap: int = 200,
    alpha: float = 0.5,
    seed: int = 31,
) -> pd.DataFrame:
    tumor = frame.loc[frame["cell_type"] == "tumor"].copy()
    donors = tumor["donor_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    for bootstrap_index in range(n_bootstrap):
        sampled = rng.choice(donors, size=len(donors), replace=True)
        parts = []
        for copy_index, donor in enumerate(sampled):
            part = tumor.loc[tumor["donor_id"].astype(str) == donor].copy()
            suffix = f"__b{bootstrap_index}_{copy_index}"
            part["lineage_id"] = part["lineage_id"].astype(str) + suffix
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        counts = pd.DataFrame(alpha, index=states, columns=states, dtype=float)
        for _, group in sample.groupby("lineage_id"):
            values = group.sort_values("time_hours")["state"].astype(str).tolist()
            for current, next_state in zip(values[:-1], values[1:]):
                if current in states and next_state in states:
                    counts.loc[current, next_state] += 1.0
        matrices.append(counts.div(counts.sum(axis=1), axis=0).to_numpy())
    stack = np.stack(matrices, axis=0)
    rows: list[dict[str, object]] = []
    for row_index, current in enumerate(states):
        for column_index, next_state in enumerate(states):
            values = stack[:, row_index, column_index]
            rows.append(
                {
                    "current_state": current,
                    "next_state": next_state,
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "n_bootstrap": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def calibration_curve_table(
    predictions: pd.DataFrame,
    probability_prefix: str,
    classes: Sequence[str] = STATE_ORDER,
    bins: int = 10,
) -> pd.DataFrame:
    labels = predictions["state"].map({name: index for index, name in enumerate(classes)}).to_numpy()
    probabilities = predictions[[f"{probability_prefix}{state}" for state in classes]].to_numpy(dtype=float)
    confidence = probabilities.max(axis=1)
    correctness = (probabilities.argmax(axis=1) == labels).astype(int)
    fraction, mean = calibration_curve(correctness, confidence, n_bins=bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted_confidence": mean, "observed_accuracy": fraction})


def plot_baseline_benchmark(metrics: pd.DataFrame, output_path: str | Path) -> None:
    selected = metrics.sort_values("log_loss").head(12).copy()
    selected["display_name"] = (
        selected["model"].str.replace("_", " ")
        + " | "
        + selected["variant"].str.replace("_", " ")
    )
    selected = selected.sort_values("log_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    ax.barh(selected["display_name"], selected["log_loss"])
    ax.set_xlabel("Donor-held-out multiclass log loss (lower is better)")
    ax.set_title("Linear baseline and calibration benchmark")
    lower = max(0.0, float(selected["log_loss"].min()) - 0.08)
    upper = float(selected["log_loss"].max()) + 0.04
    ax.set_xlim(lower, upper)
    for row_index, value in enumerate(selected["log_loss"]):
        ax.text(float(value) + 0.006, row_index, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_reliability(
    raw_curve: pd.DataFrame,
    calibrated_curve: pd.DataFrame,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.plot(
        raw_curve["mean_predicted_confidence"],
        raw_curve["observed_accuracy"],
        marker="o",
        label="Raw selected baseline",
    )
    ax.plot(
        calibrated_curve["mean_predicted_confidence"],
        calibrated_curve["observed_accuracy"],
        marker="o",
        label="Calibrated linear ensemble",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Donor-cross-fitted reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
