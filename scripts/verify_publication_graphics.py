#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from causaflux.visualization.publication import validate_publication_bundle

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    report = validate_publication_bundle(args.input, check_hashes=not args.skip_hashes)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
