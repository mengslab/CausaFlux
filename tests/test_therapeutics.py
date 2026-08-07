from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo
from causaflux.therapeutics import (
    TherapeuticConfig,
    build_regimen_catalog,
    intervention_catalog,
    run_counterfactual_therapeutics,
    validate_therapeutic_predictions,
    write_therapeutic_outputs,
)


@pytest.fixture(scope="module")
def therapeutic_bundle():
    frame = generate_cancer_demo(
        CancerDemoConfig(
            n_donors=6,
            clones_per_donor=10,
            non_tumor_cells_per_type=2,
            seed=43,
        )
    )
    config = TherapeuticConfig(
        bootstrap=2,
        timing_grid=(0.0, 24.0, 72.0),
        max_reference_rows_per_donor=8,
        seed=43,
    )
    result = run_counterfactual_therapeutics(frame, config)
    return frame, config, result


def test_intervention_and_regimen_catalogs_are_complete():
    catalog = intervention_catalog()
    regimens = build_regimen_catalog(catalog, TherapeuticConfig(timing_grid=(0, 24, 72)))
    assert set(catalog["intervention_type"]) == {"gene", "drug"}
    assert catalog["intervention_id"].is_unique
    assert {"gene", "drug", "combination", "sequence", "timing"}.issubset(
        set(regimens["regimen_category"])
    )
    assert regimens["regimen_id"].is_unique


def test_counterfactual_predictions_are_bounded_and_ranked(therapeutic_bundle):
    _, _, result = therapeutic_bundle
    predictions = result.predictions
    assert predictions["counterfactual_resistance_probability"].between(0, 1).all()
    assert predictions["normal_cell_toxicity"].between(0, 1).all()
    assert predictions["rank"].is_unique
    assert predictions["rank"].min() == 1
    assert result.qc["valid"]


def test_sequence_and_timing_predictions_are_directional(therapeutic_bundle):
    _, _, result = therapeutic_bundle
    predictions = result.predictions
    sequences = predictions.loc[predictions["regimen_category"] == "sequence"]
    forward = sequences.loc[
        sequences["regimen_id"] == "SEQUENCE__IRE1I__THEN__MITORESERVEI",
        "uncertainty_adjusted_utility",
    ].iloc[0]
    reverse = sequences.loc[
        sequences["regimen_id"] == "SEQUENCE__MITORESERVEI__THEN__IRE1I",
        "uncertainty_adjusted_utility",
    ].iloc[0]
    assert forward != reverse
    timing = predictions.loc[
        (predictions["regimen_category"] == "timing")
        & predictions["regimen_id"].str.contains("IRE1I")
    ]
    assert timing["uncertainty_adjusted_utility"].nunique() > 1


def test_donor_bootstrap_intervals_are_ordered(therapeutic_bundle):
    _, config, result = therapeutic_bundle
    predictions = result.predictions
    assert predictions["utility_ci_low"].le(predictions["utility_ci_high"]).all()
    assert predictions["resistance_risk_reduction_ci_low"].le(
        predictions["resistance_risk_reduction_ci_high"]
    ).all()
    assert predictions["bootstrap_successful_replicates"].min() >= 1
    report = validate_therapeutic_predictions(
        result.intervention_catalog,
        result.regimen_catalog,
        predictions,
        config,
    )
    assert report["valid"]


def test_therapeutic_outputs_round_trip(therapeutic_bundle, tmp_path: Path):
    _, config, result = therapeutic_bundle
    paths = write_therapeutic_outputs(result, tmp_path, config)
    assert paths["predictions"].exists()
    assert paths["ranking_plot"].exists()
    loaded = pd.read_csv(paths["predictions"])
    assert len(loaded) == len(result.predictions)
    assert (tmp_path / "gene_predictions.csv").exists()
    assert (tmp_path / "drug_predictions.csv").exists()
    assert (tmp_path / "combination_predictions.csv").exists()
    assert (tmp_path / "sequence_predictions.csv").exists()
    assert (tmp_path / "timing_predictions.csv").exists()
