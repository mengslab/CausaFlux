#!/usr/bin/env python3
import json, sys
from pathlib import Path
from causaflux.spatiotemporal_tissue import validate_spatiotemporal_tissue
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('spatiotemporal_tissue_reference')
r=validate_spatiotemporal_tissue(p)
print(json.dumps(r,indent=2))
raise SystemExit(0 if r['valid'] else 1)
