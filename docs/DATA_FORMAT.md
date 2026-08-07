# Preserved v0.2 dynamic data format

## NPZ representation

CausaFlux's internal NPZ format contains:

| Key | Shape | Meaning |
|---|---:|---|
| `times` | `[N,T]` | Irregular observation times |
| `observations` | `[N,T,D]` | Raw feature values |
| `observation_mask` | `[N,T,D]` | 1 when a feature is measured |
| `interventions` | `[N,T,U]` | Intervention dose/activity |
| `mask` | `[N,T]` | 1 for valid trajectory positions |
| `fates` | `[N]` | Integer terminal outcome |
| `trajectory_ids` | `[N]` | Unique trajectory identifiers |
| `group_ids` | `[N]` | Biological replicate/donor/batch grouping |
| `feature_names` | `[D]` | Ordered feature names |
| `intervention_names` | `[U]` | Ordered intervention names |
| `fate_names` | `[K]` | Ordered outcome names |

## Long-format CSV

Each row is a measured trajectory time point. Required columns:

- `trajectory_id`
- `time`
- `fate`

`group_id` is strongly recommended. All rows from one trajectory must have the same
fate and group. Feature columns may contain blanks/NaN. Intervention blanks are
converted to zero.

The default UPR names are listed in the main README. A different dimensional schema
can be used through the Python API by passing explicit feature and intervention
names to `ChronoDataset.from_long_csv`.

## Leakage prevention

With `training.split_mode: group`, all trajectories sharing a `group_id` are placed
in the same train, validation, or test partition. The exact assignments are written
to `run/split_manifest.json`.

Use `group_id` for the highest-level unit that must not leak, such as donor,
biological replicate, organoid line, plate, animal, or experimental batch.
