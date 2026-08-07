#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from causaflux.platform import finalize_research_platform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    report = finalize_research_platform(args.output, project_root=project)
    print(
        f"[CausaFlux] Stage 10/10 complete: platform validation "
        f"{'passed' if report.valid else 'failed'} ({len(report.checks)} checks)",
        flush=True,
    )
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
