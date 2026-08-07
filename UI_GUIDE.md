# CausaFlux v1.9.0 — UI Guide

Run the reference once:

```bash
sh run.sh
```

Then launch the local interface:

```bash
sh ui.sh
```

The UI runs locally with Streamlit and reads the v1.9 output directory. It does not upload your datasets to a CausaFlux server.

You can change the output directory:

```bash
sh ui.sh my_virtual_cell_output
```

## Suggested workflow

1. Review **Overview** to confirm the evidence gates.
2. Explore interventions in **Virtual Cell**.
3. Inspect model contributions and reliability in **AI Models**.
4. Review or preview datasets in **Real-world Data**.
5. Confirm locked-cycle calibration in **Prospective Validation**.
6. Export/view publication assets from **Figures & Reports**.

The interactive sliders operate the reference simulator unless you replace it with a trained dataset-specific model. The interface labels this boundary explicitly.
