"""CausaFlux v1.4.0 held-out perturbation-history benchmark example."""

from causaflux import DynamicBenchmarkConfig, run_dynamic_benchmark

status = run_dynamic_benchmark(
    "dynamic_benchmark_example",
    DynamicBenchmarkConfig(
        epochs=8,
        patience=3,
        replicates_per_history=3,
        hidden_dim=32,
        bootstrap_replicates=20,
    ),
)

print(status["gate"])
