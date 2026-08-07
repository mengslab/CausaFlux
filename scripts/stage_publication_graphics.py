#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

GROUPS = ("core", "spatial", "therapeutics", "biomarkers", "active_learning", "neurobiology")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-group", choices=GROUPS)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        from causaflux.visualization.publication import finalize_publication_inventory
        inventory = finalize_publication_inventory(args.output)
        print(f"[CausaFlux] Stage 9/10 complete: {len(inventory)} publication figure bundles", flush=True)
        return
    if args.worker_group:
        from causaflux.visualization.publication import rebuild_reference_figure_group
        inventory = rebuild_reference_figure_group(args.output, args.worker_group)
        print(f"[CausaFlux] publication group {args.worker_group}: {len(inventory)} figures", flush=True)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    script = Path(__file__).resolve()
    for group in GROUPS:
        subprocess.run([sys.executable, str(script), "--output", args.output, "--worker-group", group], check=True)
    from causaflux.visualization.publication import finalize_publication_inventory
    inventory = finalize_publication_inventory(args.output)
    print(f"[CausaFlux] Stage 9/10 complete: {len(inventory)} publication figure bundles", flush=True)


if __name__ == "__main__":
    main()
