from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import ChronoDataset


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_upr(
    n_trajectories: int = 512,
    min_steps: int = 8,
    max_steps: int = 16,
    seed: int = 7,
    missing_feature_rate: float = 0.08,
    replicate_size: int = 4,
) -> ChronoDataset:
    """Generate an irregular-time UPR testbed with feature-level missingness.

    The simulator exists for software verification. Values are not experimental
    measurements and must not be interpreted as biological evidence.
    """
    if not 0 <= missing_feature_rate < 0.8:
        raise ValueError("missing_feature_rate must be in [0, 0.8)")
    rng = np.random.default_rng(seed)
    feature_dim = 12
    intervention_dim = 4

    times = np.zeros((n_trajectories, max_steps), dtype=np.float32)
    obs = np.zeros((n_trajectories, max_steps, feature_dim), dtype=np.float32)
    observation_mask = np.zeros_like(obs)
    interventions = np.zeros((n_trajectories, max_steps, intervention_dim), dtype=np.float32)
    mask = np.zeros((n_trajectories, max_steps), dtype=np.float32)
    fates = np.zeros(n_trajectories, dtype=np.int64)
    group_ids = np.empty(n_trajectories, dtype=object)
    scenario_ids = np.empty(n_trajectories, dtype=object)

    scenario_choices = [
        "continuous",
        "recovery",
        "pulsatile",
        "ire1_inhibition",
        "perk_inhibition",
        "atf6_support",
    ]

    for i in range(n_trajectories):
        steps = int(rng.integers(min_steps, max_steps + 1))
        dt = rng.uniform(0.35, 1.3, size=steps - 1)
        t = np.concatenate([[0.0], np.cumsum(dt)]).astype(np.float32)
        times[i, :steps] = t
        mask[i, :steps] = 1.0
        group_ids[i] = f"replicate_{i // max(1, replicate_size):04d}"

        scenario = scenario_choices[i % len(scenario_choices)]
        scenario_ids[i] = scenario
        stress_intensity = float(rng.uniform(0.4, 1.35))
        stress_start = float(rng.uniform(0.0, max(0.2, t[-1] * 0.15)))
        if scenario == "continuous":
            stress_duration = float(t[-1] - stress_start + 0.5)
        elif scenario == "recovery":
            stress_duration = float(rng.uniform(t[-1] * 0.25, t[-1] * 0.45))
        else:
            stress_duration = float(rng.uniform(t[-1] * 0.35, t[-1] * 0.75))
        pulsatile = scenario == "pulsatile"
        ire1_inhib = float(rng.uniform(0.35, 0.7)) if scenario == "ire1_inhibition" else 0.0
        perk_inhib = float(rng.uniform(0.30, 0.65)) if scenario == "perk_inhibition" else 0.0
        atf6_act = float(rng.uniform(0.25, 0.6)) if scenario == "atf6_support" else 0.0

        state = np.array(
            [
                0.12,
                0.10,
                0.10,
                0.85,
                0.85,
                0.82,
                0.08,
                0.08,
                0.85,
                0.95,
                0.90,
                0.04,
            ],
            dtype=np.float64,
        )
        state += rng.normal(0, 0.025, size=feature_dim)
        obs[i, 0] = state + rng.normal(0, 0.015, size=feature_dim)

        cumulative_burden = 0.0
        for k in range(steps - 1):
            current_t = t[k]
            delta = float(t[k + 1] - t[k])
            active = stress_start <= current_t <= stress_start + stress_duration
            if pulsatile and active:
                active = (int((current_t - stress_start) / 0.8) % 2) == 0
            stress = stress_intensity if active else 0.0
            u = np.array([stress, ire1_inhib, perk_inhib, atf6_act], dtype=np.float64)
            interventions[i, k] = u

            (
                xbp1,
                atf4,
                atf6,
                ca,
                redox,
                mito,
                aggregate,
                inflam,
                metabolism,
                viability,
                reserve,
                commit,
            ) = state
            xbp1_drive = stress * (1.0 - 0.75 * ire1_inhib)
            atf4_drive = stress * (1.0 - 0.75 * perk_inhib)
            atf6_drive = stress * 0.65 + atf6_act
            adaptive_signal = 0.48 * xbp1 + 0.42 * atf4 + 0.35 * atf6
            unresolved = max(
                0.0,
                stress + aggregate + 0.3 * inflam - adaptive_signal - 0.4 * reserve,
            )
            cumulative_burden += unresolved * delta

            derivative = np.zeros(feature_dim, dtype=np.float64)
            derivative[0] = 1.3 * xbp1_drive - 0.75 * xbp1
            derivative[1] = 1.2 * atf4_drive - 0.72 * atf4
            derivative[2] = 1.0 * atf6_drive - 0.62 * atf6
            derivative[3] = 0.5 * adaptive_signal - 0.9 * stress - 0.35 * (ca - 0.8)
            derivative[4] = 0.45 * adaptive_signal - 0.8 * stress - 0.32 * (redox - 0.8)
            derivative[5] = 0.36 * xbp1 + 0.18 * atf4 - 0.72 * stress - 0.3 * (mito - 0.8)
            derivative[6] = 0.9 * stress - 0.65 * adaptive_signal - 0.45 * aggregate
            derivative[7] = 0.58 * unresolved + 0.15 * commit - 0.35 * inflam
            derivative[8] = 0.34 * xbp1 + 0.16 * atf6 - 0.55 * stress - 0.28 * (metabolism - 0.8)
            derivative[9] = (
                0.20 * adaptive_signal
                - 0.50 * unresolved
                - 0.45 * commit
                - 0.12 * (viability - 0.95)
            )
            derivative[10] = 0.28 * adaptive_signal - 0.35 * stress - 0.25 * reserve
            derivative[11] = 0.42 * unresolved + 0.12 * inflam - 0.08 * commit

            state = state + delta * derivative + rng.normal(
                0, 0.012 * np.sqrt(delta), feature_dim
            )
            state[:3] = np.clip(state[:3], 0.0, 2.5)
            state[3:6] = np.clip(state[3:6], 0.0, 1.4)
            state[6:9] = np.clip(state[6:9], 0.0, 2.2)
            state[9:11] = np.clip(state[9:11], 0.0, 1.2)
            state[11] = np.clip(state[11], 0.0, 3.0)
            obs[i, k + 1] = state + rng.normal(0, 0.02, size=feature_dim)

        interventions[i, steps - 1] = interventions[i, steps - 2] if steps > 1 else 0.0
        death_score = (
            2.6 * (0.45 - state[9])
            + 1.0 * state[11]
            + 0.4 * cumulative_burden
            - 2.0
        )
        dysfunction_score = (
            1.5 * state[7]
            + 1.1 * state[6]
            + 0.8 * state[11]
            - 1.2 * state[10]
            - 0.8
        )
        death_probability = float(_sigmoid(death_score))
        dysfunction_probability = float(_sigmoid(dysfunction_score)) * (
            1.0 - death_probability
        )
        draw = rng.random()
        if draw < death_probability:
            fate = 2
        elif draw < death_probability + dysfunction_probability:
            fate = 1
        else:
            fate = 0
        fates[i] = fate

        valid_mask = rng.random((steps, feature_dim)) >= missing_feature_rate
        valid_mask[0] = True
        # Keep core viability and commitment measurements at the terminal point.
        valid_mask[-1, [9, 11]] = True
        observation_mask[i, :steps] = valid_mask.astype(np.float32)

    dataset = ChronoDataset(
        times=times,
        observations=obs,
        interventions=interventions,
        mask=mask,
        fates=fates,
        trajectory_ids=np.asarray([f"synthetic_{i:05d}" for i in range(n_trajectories)]),
        observation_mask=observation_mask,
        group_ids=group_ids,
    )
    dataset.scenario_ids = scenario_ids
    return dataset


def save_synthetic_upr(path: str | Path, **kwargs) -> ChronoDataset:
    dataset = generate_synthetic_upr(**kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_npz(path)
    return dataset
