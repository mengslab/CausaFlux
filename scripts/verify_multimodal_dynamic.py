#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from causaflux.multimodal_dynamic import MODEL_ORDER, MODALITY_ORDER, validate_multimodal_dynamic_benchmark


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    root = Path(args.input).resolve()
    result = validate_multimodal_dynamic_benchmark(root, verify_hashes=True)
    if not result["valid"]:
        raise SystemExit(json.dumps(result, indent=2))
    metrics = pd.read_csv(root / "model_comparison.csv")
    if set(metrics["model"]) != set(MODEL_ORDER):
        raise SystemExit("multimodal model registry mismatch")
    schema = json.loads((root / "modality_schema.json").read_text())
    if set(schema["modalities"]) != set(MODALITY_ORDER):
        raise SystemExit("multimodal modality registry mismatch")
    gate = json.loads((root / "multimodal_exit_gate.json").read_text())
    if not gate.get("software_exit_gate_passed"):
        raise SystemExit("multimodal software exit gate blocked")
    if not str(gate.get("foundation_pretraining_authorization", "")).startswith("BLOCKED"):
        raise SystemExit("real-data foundation-pretraining block was removed")
    cross = pd.read_csv(root / "cross_modal_forecasting.csv")
    if set(cross["modality"]) != {"rna", "phosphoprotein", "metabolome", "lipidome"}:
        raise SystemExit("cross-modal decoder coverage incomplete")
    mnar = pd.read_csv(root / "missingness_sensitivity.csv")
    required_scenarios = {"observed", "MCAR_20", "MNAR_destructive_imaging", "MNAR_low_quality_omics"}
    if not required_scenarios.issubset(set(mnar["scenario"])):
        raise SystemExit("MNAR sensitivity scenarios incomplete")
    manifest = pd.read_csv(root / "artifact_manifest.csv")
    failures = []
    for row in manifest.itertuples(index=False):
        path = root / row.path
        if not path.exists() or sha256(path) != row.sha256:
            failures.append(str(row.path))
    if failures:
        raise SystemExit(f"artifact hash failures: {failures[:10]}")
    print(json.dumps({
        "valid": True,
        "version": "1.7.0",
        "models": len(metrics),
        "modalities": len(schema["modalities"]),
        "qualifying_models": gate.get("qualifying_models", []),
        "poe_imaging_reporter_ablation_log_loss_delta": gate.get("poe_imaging_reporter_ablation_log_loss_delta"),
        "artifact_hashes_verified": len(manifest),
        "foundation_pretraining_authorization": gate.get("foundation_pretraining_authorization"),
    }, indent=2))


if __name__ == "__main__":
    main()
