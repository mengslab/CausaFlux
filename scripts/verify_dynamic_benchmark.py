#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from causaflux.dynamic_benchmark import MODEL_ORDER, validate_dynamic_benchmark


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
    result = validate_dynamic_benchmark(root)
    metrics = pd.read_csv(root / "model_comparison.csv")
    if set(metrics["model"]) != set(MODEL_ORDER):
        raise SystemExit("model registry mismatch")
    manifest = pd.read_csv(root / "artifact_manifest.csv")
    failures = []
    for row in manifest.itertuples(index=False):
        path = root / row.path
        if not path.exists() or sha256(path) != row.sha256:
            failures.append(str(row.path))
    if failures:
        raise SystemExit(f"artifact hash failures: {failures[:10]}")
    gate = json.loads((root / "foundation_pretraining_gate.json").read_text())
    payload = {
        "valid": result["valid"],
        "version": "1.7.0",
        "models": len(metrics),
        "software_performance_gate": gate["status"],
        "winning_dynamic_model": gate.get("winning_dynamic_model"),
        "evaluation_scope": gate.get("evaluation_scope"),
        "foundation_pretraining_status": gate.get("foundation_pretraining_status"),
        "artifact_hashes_verified": len(manifest),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
