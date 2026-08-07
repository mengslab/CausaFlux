from pathlib import Path

import numpy as np

from causaflux.data import ChronoDataset, grouped_split_indices
from causaflux.synthetic import generate_synthetic_upr


def test_synthetic_shapes_masks_and_groups():
    dataset = generate_synthetic_upr(
        n_trajectories=12,
        min_steps=5,
        max_steps=8,
        missing_feature_rate=0.2,
        seed=1,
    )
    assert dataset.times.shape == (12, 8)
    assert dataset.observations.shape == (12, 8, 12)
    assert dataset.observation_mask.shape == dataset.observations.shape
    assert dataset.interventions.shape == (12, 8, 4)
    assert len(np.unique(dataset.group_ids)) >= 3
    for index in range(len(dataset)):
        valid = dataset.mask[index].astype(bool)
        assert np.all(np.diff(dataset.times[index, valid]) >= 0)


def test_csv_roundtrip(tmp_path: Path):
    dataset = generate_synthetic_upr(n_trajectories=10, min_steps=5, max_steps=7, seed=2)
    csv_path = tmp_path / "data.csv"
    npz_path = tmp_path / "data.npz"
    dataset.to_long_csv(csv_path)
    loaded = ChronoDataset.from_long_csv(csv_path)
    loaded.to_npz(npz_path)
    reloaded = ChronoDataset.from_npz(npz_path)
    assert len(reloaded) == len(dataset)
    assert reloaded.feature_names == dataset.feature_names
    assert reloaded.intervention_names == dataset.intervention_names
    assert np.array_equal(reloaded.observation_mask, loaded.observation_mask)


def test_grouped_split_has_disjoint_groups():
    groups = np.array([f"g{i // 2}" for i in range(18)])
    train, val, test = grouped_split_indices(groups, seed=3)
    train_groups = set(groups[train])
    val_groups = set(groups[val])
    test_groups = set(groups[test])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
