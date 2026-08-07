#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from causaflux.staged_workflow import run_neurobiology_stage

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    result=run_neurobiology_stage(a.config,a.output)
    print(f"[CausaFlux] Stage 7/9 complete: {result['qc']}",flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
if __name__=='__main__': main()
