from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history_csv: str | Path, output_path: str | Path) -> None:
    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("CausaFlux v0.2 training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_simulation(
    times,
    mean,
    lower,
    upper,
    output_path: str | Path,
    feature_names: Sequence[str],
    selected: Sequence[int] | None = None,
    title: str = "CausaFlux counterfactual forecast",
) -> None:
    if selected is None:
        selected = list(range(min(6, len(feature_names))))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for feature_index in selected:
        ax.plot(times, mean[:, feature_index], label=feature_names[feature_index])
        ax.fill_between(
            times,
            lower[:, feature_index],
            upper[:, feature_index],
            alpha=0.12,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Simulated feature value")
    ax.set_title(f"{title} (90% interval)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scenario_comparison(
    scenario_frames: dict[str, pd.DataFrame],
    fate_frame: pd.DataFrame,
    output_path: str | Path,
    feature_names: Sequence[str],
    selected_features: Sequence[str] | None = None,
) -> None:
    if selected_features is None:
        preferred = [
            "XBP1_activity",
            "protein_aggregation",
            "inflammatory_signaling",
            "viability",
            "commitment_signal",
        ]
        selected_features = [name for name in preferred if name in feature_names]
        if not selected_features:
            selected_features = list(feature_names[: min(5, len(feature_names))])
    fig, axes = plt.subplots(
        len(selected_features) + 1,
        1,
        figsize=(9.5, 2.5 * (len(selected_features) + 1)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for axis, feature in zip(axes[:-1], selected_features):
        for scenario, frame in scenario_frames.items():
            axis.plot(frame["time"], frame[f"{feature}_mean"], label=scenario)
        axis.set_ylabel(feature.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fate_pivot = fate_frame.pivot(index="scenario", columns="fate", values="mean_probability")
    fate_pivot.plot(kind="bar", ax=axes[-1])
    axes[-1].set_ylabel("Fate probability")
    axes[-1].set_xlabel("Scenario")
    axes[-1].tick_params(axis="x", rotation=25)
    axes[-1].legend(fontsize=8)
    fig.suptitle("CausaFlux intervention scenario comparison", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
