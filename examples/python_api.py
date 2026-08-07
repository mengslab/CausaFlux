from pathlib import Path

import numpy as np

from causaflux import InterventionEvent, build_intervention_schedule, run_experiment
from causaflux.simulation import simulate_with_uncertainty
from causaflux.training import load_checkpoint

# Run the full experiment manifest.
result = run_experiment(
    "configs/demo_v0.2.yaml",
    output_dir="python_api_output",
    device_override="cpu",
)
print("Report:", result["report"])

# Load the trained model and construct a custom pulsed intervention.
model, standardizer, checkpoint, device = load_checkpoint(result["checkpoint"], "cpu")
times = np.linspace(0, 10, 21, dtype=np.float32)
events = [
    InterventionEvent(
        channel="ER_stress",
        value=1.0,
        start=0,
        stop=7,
        shape="pulse",
        period=2.0,
        duty_cycle=0.4,
    ),
    InterventionEvent(
        channel="ATF6_activation",
        value=0.4,
        start=1,
        stop=7,
    ),
]
schedule = build_intervention_schedule(
    times,
    model.config.intervention_dim,
    events,
    intervention_names=checkpoint["intervention_names"],
)
initial = standardizer.mean.copy()
simulation = simulate_with_uncertainty(
    model,
    standardizer,
    initial,
    times,
    schedule,
    device,
    mc_samples=20,
)
print("Mean fate probabilities:", simulation["fate_probability_mean"])
