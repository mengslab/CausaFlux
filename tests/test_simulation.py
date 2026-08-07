import numpy as np
import torch

from causaflux.data import INTERVENTION_NAMES, Standardizer
from causaflux.model import CausaFlux, CausaFluxConfig
from causaflux.simulation import InterventionEvent, build_intervention_schedule, simulate_with_uncertainty


def test_pulse_and_ramp_schedule():
    times = np.linspace(0, 6, 13, dtype=np.float32)
    schedule = build_intervention_schedule(
        times,
        4,
        [
            InterventionEvent("ER_stress", 0.2, 0, 2, shape="linear", end_value=1.0),
            InterventionEvent("ATF6_activation", 0.5, 0, 6, shape="pulse", period=2, duty_cycle=0.5),
        ],
        intervention_names=INTERVENTION_NAMES,
    )
    assert schedule.shape == (13, 4)
    assert schedule[:, 0].max() == 1.0
    assert 0 < np.count_nonzero(schedule[:, 3]) < len(times)


def test_simulation_output_shapes():
    model = CausaFlux(CausaFluxConfig(dropout=0.1))
    standardizer = Standardizer(
        mean=np.zeros(12, dtype=np.float32),
        std=np.ones(12, dtype=np.float32),
    )
    times = np.linspace(0, 5, 7, dtype=np.float32)
    schedule = build_intervention_schedule(
        times,
        4,
        [InterventionEvent(channel=0, value=1.0, start=0, stop=2.5)],
    )
    result = simulate_with_uncertainty(
        model,
        standardizer,
        np.zeros(12, dtype=np.float32),
        times,
        schedule,
        torch.device("cpu"),
        mc_samples=3,
    )
    assert result["trajectory_mean"].shape == (7, 12)
    assert result["decoder_std_mean"].shape == (7, 12)
    assert result["fate_probability_mean"].shape == (3,)
