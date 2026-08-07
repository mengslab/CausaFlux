from __future__ import annotations

import json
import numpy as np
import pandas as pd

from causaflux.neurobiology import (
    EPHYS_FEATURES,
    NEURAL_CELL_TYPES,
    NEURO_STATES,
    NeurobiologyConfig,
    generate_neurobiology_dataset,
    run_neurobiology_configuration,
    validate_neurobiology_outputs,
    write_neurobiology_outputs,
)


def small_config() -> NeurobiologyConfig:
    return NeurobiologyConfig(n_donors=4, cells_per_type=4, bootstrap=6, seed=13)


def test_neural_glial_dataset_contains_required_context():
    frame = generate_neurobiology_dataset(small_config())
    assert set(frame["cell_type"]) == set(NEURAL_CELL_TYPES)
    assert set(NEURO_STATES).issubset(set(frame["state"]))
    assert set(frame["apoe_genotype"]) == {"APOE3", "APOE4"}
    assert frame["donor_id"].nunique() == 4
    assert frame["time_days"].nunique() == 4


def test_electrophysiology_is_neuron_specific():
    frame = generate_neurobiology_dataset(small_config())
    neurons = frame["cell_type"].str.contains("neuron")
    assert frame.loc[neurons, list(EPHYS_FEATURES)].notna().all().all()
    assert frame.loc[~neurons, list(EPHYS_FEATURES)].isna().all().all()


def test_neuro_probabilities_are_donor_held_out():
    result = run_neurobiology_configuration(small_config())
    columns = [f"probability_{state}" for state in NEURO_STATES]
    assert np.allclose(result.state_probabilities[columns].sum(axis=1), 1.0)
    assert result.state_probabilities["donor_id"].astype(str).equals(
        result.state_probabilities["held_out_donor"].astype(str)
    )
    assert result.risk_predictions["predicted_degeneration_probability"].between(0, 1).all()


def test_neuro_transition_intervals_are_ordered():
    result = run_neurobiology_configuration(small_config())
    intervals = result.transition_intervals
    assert intervals["ci_low"].le(intervals["bootstrap_mean"]).all()
    assert intervals["ci_high"].ge(intervals["bootstrap_mean"]).all()
    assert result.transition_matrix.shape == (len(NEURO_STATES), len(NEURO_STATES))


def test_neuro_output_round_trip(tmp_path):
    result = run_neurobiology_configuration(small_config())
    write_neurobiology_outputs(result, tmp_path, write_plots=True)
    qc = validate_neurobiology_outputs(tmp_path)
    assert qc["valid"]
    assert qc["version"] == "1.7.0"
    assert (tmp_path / "imaging_ephys_alignment.csv").exists()
    assert (tmp_path / "neural_glial_trajectories.png").exists()
