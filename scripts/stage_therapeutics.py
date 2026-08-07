#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from causaflux.staged_workflow import run_therapeutic_stage

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    run_therapeutic_stage(a.config,a.output)
    print('[CausaFlux] Stage 4/9 complete: counterfactual therapeutic predictions serialized',flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
if __name__=='__main__': main()
