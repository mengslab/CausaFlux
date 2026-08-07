from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .data import Standardizer
from .model import CausaFlux


@dataclass(frozen=True)
class InterventionEvent:
    channel: int | str
    value: float
    start: float
    stop: float
    shape: str = "constant"
    end_value: float | None = None
    period: float | None = None
    duty_cycle: float = 0.5


def _channel_index(channel: int | str, intervention_names: Sequence[str] | None) -> int:
    if isinstance(channel, (int, np.integer)):
        return int(channel)
    if intervention_names is None:
        raise ValueError("string intervention channels require intervention_names")
    try:
        return list(intervention_names).index(str(channel))
    except ValueError as exc:
        raise ValueError(f"unknown intervention channel: {channel}") from exc


def build_intervention_schedule(
    forecast_times: np.ndarray,
    intervention_dim: int,
    events: Sequence[InterventionEvent],
    intervention_names: Sequence[str] | None = None,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """Build a dose-by-time matrix from constant, ramp, or pulse events."""
    forecast_times = np.asarray(forecast_times, dtype=np.float32)
    if forecast_times.ndim != 1 or len(forecast_times) < 2:
        raise ValueError("forecast_times must be a one-dimensional array with at least two values")
    if np.any(np.diff(forecast_times) < 0):
        raise ValueError("forecast_times must be nondecreasing")
    schedule = np.zeros((len(forecast_times), intervention_dim), dtype=np.float32)
    for event in events:
        index = _channel_index(event.channel, intervention_names)
        if not 0 <= index < intervention_dim:
            raise ValueError(f"invalid intervention channel: {event.channel}")
        if event.stop < event.start:
            raise ValueError("event stop must be greater than or equal to start")
        active = (forecast_times >= event.start) & (forecast_times <= event.stop)
        values = np.zeros(len(forecast_times), dtype=np.float32)
        if event.shape == "constant":
            values[active] = event.value
        elif event.shape == "linear":
            stop_value = event.value if event.end_value is None else event.end_value
            duration = max(event.stop - event.start, 1e-8)
            fraction = np.clip((forecast_times - event.start) / duration, 0.0, 1.0)
            values[active] = event.value + (stop_value - event.value) * fraction[active]
        elif event.shape == "pulse":
            if event.period is None or event.period <= 0:
                raise ValueError("pulse events require a positive period")
            if not 0 < event.duty_cycle <= 1:
                raise ValueError("pulse duty_cycle must be in (0, 1]")
            phase = np.mod(forecast_times - event.start, event.period)
            pulse_active = active & (phase <= event.period * event.duty_cycle)
            values[pulse_active] = event.value
        else:
            raise ValueError(f"unknown event shape: {event.shape}")
        schedule[:, index] += values
    if clip_min is not None or clip_max is not None:
        lower = -np.inf if clip_min is None else clip_min
        upper = np.inf if clip_max is None else clip_max
        schedule = np.clip(schedule, lower, upper)
    return schedule.astype(np.float32)


def events_from_csv(path: str | Path) -> list[InterventionEvent]:
    frame = pd.read_csv(path)
    required = {"channel", "value", "start", "stop"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"schedule CSV is missing columns: {sorted(missing)}")
    events: list[InterventionEvent] = []
    for row in frame.to_dict(orient="records"):
        channel_text = row["channel"]
        channel: int | str
        try:
            channel = int(channel_text)
        except (TypeError, ValueError):
            channel = str(channel_text)
        events.append(
            InterventionEvent(
                channel=channel,
                value=float(row["value"]),
                start=float(row["start"]),
                stop=float(row["stop"]),
                shape=str(row.get("shape", "constant")),
                end_value=(
                    None
                    if pd.isna(row.get("end_value"))
                    else float(row.get("end_value"))
                ),
                period=None if pd.isna(row.get("period")) else float(row.get("period")),
                duty_cycle=float(row.get("duty_cycle", 0.5)),
            )
        )
    return events


def _single_rollout(
    model: CausaFlux,
    initial_observation: torch.Tensor,
    initial_observation_mask: torch.Tensor,
    forecast_times: torch.Tensor,
    schedule: torch.Tensor,
    sample_process_noise: bool,
):
    identity, context, hidden, commitment = model._initial_state(
        initial_observation, initial_observation_mask
    )
    current = initial_observation
    current_mask = initial_observation_mask
    predictions = [current]
    latents = []
    adaptations = []
    commitments = []
    predicted_std = [torch.zeros_like(current)]
    for index in range(len(forecast_times) - 1):
        dt = (forecast_times[index + 1] - forecast_times[index]).view(1)
        (
            mean,
            log_variance,
            latent,
            hidden,
            commitment,
            adaptation,
        ) = model.step(
            current,
            current_mask,
            schedule[index].view(1, -1),
            dt,
            identity,
            context,
            hidden,
            commitment,
        )
        std = torch.exp(0.5 * log_variance)
        current = mean + torch.randn_like(mean) * std if sample_process_noise else mean
        current_mask = torch.ones_like(current)
        predictions.append(current)
        predicted_std.append(std)
        latents.append(latent)
        adaptations.append(adaptation)
        commitments.append(commitment)
    if latents:
        final_latent = latents[-1]
    else:
        final_latent = torch.cat(
            [
                identity,
                torch.zeros(
                    (1, model.config.adaptation_dim),
                    device=current.device,
                    dtype=current.dtype,
                ),
                commitment,
                context,
            ],
            dim=-1,
        )
    fate = torch.softmax(model.fate_head(final_latent), dim=-1)
    return (
        torch.cat(predictions, dim=0),
        torch.cat(predicted_std, dim=0),
        fate.squeeze(0),
        torch.cat(adaptations, dim=0) if adaptations else None,
        torch.cat(commitments, dim=0) if commitments else None,
    )


def simulate_with_uncertainty(
    model: CausaFlux,
    standardizer: Standardizer,
    initial_observation: np.ndarray,
    forecast_times: np.ndarray,
    schedule: np.ndarray,
    device: torch.device,
    mc_samples: int = 30,
    initial_observation_mask: np.ndarray | None = None,
    sample_process_noise: bool = True,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    raw_initial = np.asarray(initial_observation, dtype=np.float32)
    if initial_observation_mask is None:
        initial_observation_mask = np.isfinite(raw_initial).astype(np.float32)
    initial_observation_mask = np.asarray(initial_observation_mask, dtype=np.float32)
    filled_initial = np.nan_to_num(raw_initial, nan=standardizer.mean)
    initial = standardizer.transform(filled_initial[None, :])
    initial_tensor = torch.from_numpy(initial).to(device)
    initial_mask_tensor = torch.from_numpy(initial_observation_mask[None, :]).to(device)
    time_tensor = torch.from_numpy(np.asarray(forecast_times, dtype=np.float32)).to(device)
    schedule_tensor = torch.from_numpy(np.asarray(schedule, dtype=np.float32)).to(device)

    trajectories = []
    predictive_stds = []
    fates = []
    adaptations = []
    commitments = []
    previous_mode = model.training
    model.train()  # Monte Carlo dropout
    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(mc_samples):
            trajectory, predicted_std, fate, adaptation, commitment = _single_rollout(
                model,
                initial_tensor,
                initial_mask_tensor,
                time_tensor,
                schedule_tensor,
                sample_process_noise,
            )
            trajectories.append(trajectory.cpu().numpy())
            predictive_stds.append(predicted_std.cpu().numpy())
            fates.append(fate.cpu().numpy())
            if adaptation is not None:
                adaptations.append(adaptation.cpu().numpy())
            if commitment is not None:
                commitments.append(commitment.cpu().numpy())
    model.train(previous_mode)

    trajectories_array = np.stack(trajectories, axis=0)
    raw = standardizer.inverse_transform(trajectories_array)
    predictive_std_array = np.stack(predictive_stds, axis=0) * standardizer.std
    fate_array = np.stack(fates, axis=0)
    result = {
        "trajectory_mean": raw.mean(axis=0),
        "trajectory_std": raw.std(axis=0),
        "trajectory_lower": np.quantile(raw, 0.05, axis=0),
        "trajectory_upper": np.quantile(raw, 0.95, axis=0),
        "decoder_std_mean": predictive_std_array.mean(axis=0),
        "fate_probability_mean": fate_array.mean(axis=0),
        "fate_probability_std": fate_array.std(axis=0),
    }
    if adaptations:
        result["adaptation_mean"] = np.stack(adaptations, axis=0).mean(axis=0)
    if commitments:
        result["commitment_mean"] = np.stack(commitments, axis=0).mean(axis=0)
    return result
