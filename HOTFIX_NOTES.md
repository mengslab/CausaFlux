# CausaFlux v1.4.0 Intel macOS compatibility hotfix

## Issue

The v1.4.0 macOS launcher intentionally pins NumPy 1.26.x on Intel Macs for the supported PyTorch 2.2.2 stack. The multimodal synthetic data generator called `numpy.trapezoid`, which exists in NumPy 2.x but not NumPy 1.26.4. Installation therefore completed successfully and the benchmark failed at data generation.

## Fix

The early-imaging calcium AUC now uses a small internal implementation of the trapezoidal integration rule based only on NumPy operations available in NumPy 1.26+. The scientific calculation is unchanged.

A regression test disables `np.trapezoid` and verifies the compatibility path. The exact production benchmark settings (30 epochs, 5 replicates per history, 100 bootstrap replicates) reproduce the original v1.4.0 qualifying models and exit-gate metrics.

## Supported environment

- Python 3.10–3.12
- Intel macOS: NumPy 1.26.x + PyTorch 2.2.2
- `sh run.sh` remains the supported one-command launcher.
