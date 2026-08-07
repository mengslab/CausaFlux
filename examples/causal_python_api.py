from causaflux import run_causal_experiment

result = run_causal_experiment(
    "configs/cancer_closed_loop_v1.4.0.yaml",
    "causaflux_v1.4.0_output",
)
print(result["report"])
