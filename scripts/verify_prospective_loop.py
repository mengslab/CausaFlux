from __future__ import annotations
import argparse
import json
from causaflux.prospective_loop import validate_prospective_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--require-gate", action="store_true", default=True)
    args = parser.parse_args()
    report = validate_prospective_loop(args.input, require_gate=args.require_gate)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
