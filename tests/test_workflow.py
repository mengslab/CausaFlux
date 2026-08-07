from pathlib import Path

import yaml

from causaflux.workflow import run_experiment


def test_small_workflow(tmp_path: Path):
    config = {
        "experiment": {"name": "test", "seed": 3},
        "data": {
            "mode": "synthetic",
            "n_trajectories": 36,
            "min_steps": 5,
            "max_steps": 7,
            "missing_feature_rate": 0.05,
            "replicate_size": 2,
            "export_long_csv": False,
        },
        "model": {"hidden_dim": 32, "adaptation_dim": 8},
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "patience": 1,
            "split_mode": "group",
            "device": "cpu",
        },
        "simulation": {
            "final_time": 3,
            "steps": 5,
            "mc_samples": 2,
            "scenarios": [
                {
                    "name": "recovery",
                    "events": [
                        {"channel": "ER_stress", "value": 1, "start": 0, "stop": 1}
                    ],
                }
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    result = run_experiment(config_path, output, "cpu")
    assert result["checkpoint"].exists()
    assert result["report"].exists()
