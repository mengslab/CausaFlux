#!/usr/bin/env python3
import json, sys
from pathlib import Path
from causaflux.intervention_generalization import validate_intervention_generalization
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('intervention_generalization_reference')
r=validate_intervention_generalization(p)
print(json.dumps(r,indent=2))
raise SystemExit(0 if r['valid'] else 1)
