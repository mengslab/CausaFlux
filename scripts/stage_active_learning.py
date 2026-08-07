#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from causaflux.staged_workflow import run_active_learning_stage

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_active_learning_stage(args.config, args.output)
    print(f"[CausaFlux] Stage 6/9 complete: {result['round1_ranking']}", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

if __name__ == '__main__':
    main()
