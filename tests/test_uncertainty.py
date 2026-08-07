import numpy as np

from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo
from causaflux.uncertainty import benchmark_linear_baselines, transition_bootstrap_uncertainty


def test_donor_aware_baseline_and_uncertainty_outputs():
    frame = generate_cancer_demo(
        CancerDemoConfig(
            n_donors=5,
            clones_per_donor=12,
            non_tumor_cells_per_type=1,
            seed=17,
        )
    )
    result = benchmark_linear_baselines(
        frame,
        split_mode="leave_one_donor_out",
        n_metric_bootstrap=4,
        n_prediction_bootstrap=2,
        seed=17,
    )
    assert not result.metrics.empty
    assert not result.calibration_metrics.empty
    assert (result.split_manifest["donor_overlap"] == "").all()
    assert result.ensemble_uncertainty["mutual_information"].ge(0).all()
    probability_columns = [
        column for column in result.predictions if column.startswith("probability_")
    ]
    assert np.allclose(result.predictions[probability_columns].sum(axis=1), 1.0)
    assert result.bootstrap_predictions["bootstrap_successful_replicates"].min() >= 1


def test_transition_cluster_bootstrap():
    frame = generate_cancer_demo(
        CancerDemoConfig(n_donors=4, clones_per_donor=8, non_tumor_cells_per_type=1, seed=23)
    )
    uncertainty = transition_bootstrap_uncertainty(frame, n_bootstrap=5, seed=23)
    assert len(uncertainty) == 16
    assert uncertainty["ci_low"].le(uncertainty["bootstrap_mean"]).all()
    assert uncertainty["ci_high"].ge(uncertainty["bootstrap_mean"]).all()
