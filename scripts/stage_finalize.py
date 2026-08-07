#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from causaflux.staged_workflow import finalize_report_stage

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    result=finalize_report_stage(a.config,a.output)
    print(f"[CausaFlux] Stage 8/10 complete: {result['report']}",flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
if __name__=='__main__': main()
