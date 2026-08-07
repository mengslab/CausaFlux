from __future__ import annotations

import numpy as np
import pandas as pd

from causaflux.active_learning import (
    ClosedLoopConfig,
    run_closed_loop_experimentation,
    update_closed_loop_from_observations,
    validate_closed_loop_outputs,
    write_closed_loop_outputs,
)


def _result():
    return run_closed_loop_experimentation(
        config=ClosedLoopConfig(
            budget=2.0,
            batch_size=4,
            round2_budget=1.6,
            round2_batch_size=3,
            bootstrap=4,
            eig_samples=180,
            seed=9,
            true_hypothesis="H1_PROTEOSTASIS_UPSTREAM",
        )
    )


def test_catalog_contains_all_four_experiment_types():
    result = _result()
    assert set(result.catalog["experiment_type"]) == {"crispr", "drug", "imaging", "sampling_time"}
    assert result.catalog["experiment_id"].is_unique


def test_batch_respects_budget_capacity_and_type_limits():
    result = _result()
    batch = result.round1_batch
    assert len(batch) <= 4
    assert batch["relative_cost"].sum() <= 2.0 + 1e-9
    assert batch.groupby("experiment_type").size().max() <= 2
    assert batch["experiment_id"].is_unique


def test_closed_loop_posterior_is_normalized_and_reduces_entropy():
    result = _result()
    ids = result.hypotheses["hypothesis_id"].tolist()
    probabilities = result.posterior_history[ids]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.posterior_history.iloc[-1]["entropy_nats"] <= result.posterior_history.iloc[0]["entropy_nats"] + 1e-9
    assert result.qc["demonstration_information_gain_nats"] >= -1e-9


def test_output_writer_and_validator(tmp_path):
    result = _result()
    write_closed_loop_outputs(result, tmp_path, write_plots=True)
    report = validate_closed_loop_outputs(tmp_path)
    assert report["valid"]
    assert report["candidate_types"] == ["crispr", "drug", "imaging", "sampling_time"]
    assert (tmp_path / "hypothesis_posterior_update.png").exists()


def test_observation_update_excludes_completed_experiment():
    result = _result()
    completed = result.round1_batch.iloc[0]
    observations = pd.DataFrame([
        {
            "experiment_id": completed["experiment_id"],
            "observed_standardized_readout": completed["expected_readout__H1_PROTEOSTASIS_UPSTREAM"],
            "standard_error_or_posterior_sd": completed["measurement_noise"],
        }
    ])
    history, ranking, batch = update_closed_loop_from_observations(
        result.hypotheses,
        result.catalog,
        observations,
        config=ClosedLoopConfig(bootstrap=2, eig_samples=120, simulate_demonstration_round=False),
    )
    assert len(history) == 2
    assert completed["experiment_id"] not in set(ranking["experiment_id"])
    assert completed["experiment_id"] not in set(batch["experiment_id"])
