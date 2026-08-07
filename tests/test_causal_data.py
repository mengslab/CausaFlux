from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo, validate_causal_frame


def test_cancer_demo_is_valid_and_longitudinal():
    frame = generate_cancer_demo(
        CancerDemoConfig(
            n_donors=4,
            clones_per_donor=12,
            non_tumor_cells_per_type=2,
            seed=11,
        )
    )
    report = validate_causal_frame(frame)
    assert report["valid"]
    assert report["n_donors"] == 4
    assert set(frame["cell_type"]) >= {"tumor", "macrophage", "t_cell"}
    assert frame.loc[frame["cell_type"] == "tumor", "lineage_id"].nunique() == 48
