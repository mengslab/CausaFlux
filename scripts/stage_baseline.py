#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from causaflux.causal_workflow import run_causal_experiment

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    run_causal_experiment(a.config,a.output,stop_after_baseline=True)
    print('[CausaFlux] Stage 1/9 complete: donor-aware baseline artifacts serialized',flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
if __name__=='__main__': main()
