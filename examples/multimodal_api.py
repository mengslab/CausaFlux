from causaflux import read_multimodal, validate_multimodal
from causaflux.multimodal import modality_feature_frame

mdata = read_multimodal("causaflux_v1.4.0_output/multimodal/causaflux_multimodal.h5mu")
print(validate_multimodal(mdata))
features = modality_feature_frame(mdata)
print(features.shape)
print(features.filter(regex=r"^(rna|atac|protein|mutation|drug_response)__").columns[:10])
