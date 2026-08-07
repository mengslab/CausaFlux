"""CausaFlux v1.4.0 multimodal dynamic-state benchmark example."""
from causaflux import MultimodalDynamicConfig, run_multimodal_dynamic_benchmark

result = run_multimodal_dynamic_benchmark(
    "causaflux_v1.4.0_multimodal_dynamic",
    MultimodalDynamicConfig(),
    require_gate=True,
)
print(result["comparison"])
print(result["gate"])
