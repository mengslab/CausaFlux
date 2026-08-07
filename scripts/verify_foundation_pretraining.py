from __future__ import annotations
import json, sys
from pathlib import Path
from causaflux.foundation_pretraining import validate_foundation_pretraining

def main():
    path=Path(sys.argv[1] if len(sys.argv)>1 else 'foundation_pretraining_reference')
    result=validate_foundation_pretraining(path,verify_hashes=True)
    print(json.dumps(result,indent=2))
    if not result['valid']: raise SystemExit(1)
if __name__=='__main__': main()
