#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from causaflux.biological_validation import validate_biological_validation

out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "biological_validation_reference"
result = validate_biological_validation(out)
if not result["valid"]:
    raise SystemExit("BIOLOGICAL VALIDATION FAILED: " + "; ".join(result["errors"]))
print(json.dumps(result, indent=2))
