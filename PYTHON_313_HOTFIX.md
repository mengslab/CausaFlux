# Python 3.13 launcher hotfix

CausaFlux v1.4.0 supports Python 3.10, 3.11, and 3.12. The original top-level
launcher accidentally used the active `python3` command to create `.causaflux_env`,
which could select Python 3.13 from an activated Conda base environment.

The corrected launcher:

1. re-enters Bash when started with `sh run.sh`;
2. rejects and removes only an incompatible project-local `.causaflux_env`;
3. selects Python 3.10-3.12 when installed;
4. otherwise creates a project-local Conda environment;
5. prefers Python 3.11 on Intel macOS for PyTorch wheel compatibility;
6. uses `.causaflux_env` consistently across setup, execution, diagnostics, and the app.

## Repair an existing extracted directory

From the CausaFlux directory:

```bash
rm -rf .causaflux_env
sh run.sh
```

With the corrected launcher, no change to the active Conda base environment is required.
