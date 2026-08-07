from causaflux import SpatialGraphConfig, build_spatial_heterograph
from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo

frame = generate_cancer_demo(CancerDemoConfig(n_donors=4, seed=31))
result = build_spatial_heterograph(frame, SpatialGraphConfig(seed=31, bootstrap=50))

print(result.qc)
print(result.circuits.head(10).to_string(index=False))
