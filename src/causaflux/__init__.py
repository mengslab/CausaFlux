"""CausaFlux: dynamic, intervention-aware virtual-cell modeling."""

from .data import (
    ChronoDataset,
    FATE_NAMES,
    FEATURE_NAMES,
    INTERVENTION_NAMES,
    Standardizer,
    load_dataset,
)
from .model import CausaFlux, CausaFluxConfig
from .simulation import (
    InterventionEvent,
    build_intervention_schedule,
    events_from_csv,
    simulate_with_uncertainty,
)
from .workflow import run_experiment
from .causal_workflow import run_causal_experiment
from .uncertainty import benchmark_linear_baselines, transition_bootstrap_uncertainty
from .biomarkers import (
    BiomarkerConfig,
    BiomarkerResult,
    binary_auc,
    run_causal_biomarkers,
    validate_biomarker_outputs,
    write_biomarker_outputs,
)
from .active_learning import (
    ClosedLoopConfig,
    ClosedLoopResult,
    run_closed_loop_experimentation,
    score_experiments,
    select_batch,
    validate_closed_loop_outputs,
    update_closed_loop_from_observations,
    write_closed_loop_outputs,
)
from .therapeutics import (
    TherapeuticConfig,
    TherapeuticResult,
    build_regimen_catalog,
    fit_therapeutic_model,
    intervention_catalog,
    predict_regimens,
    run_counterfactual_therapeutics,
    validate_therapeutic_predictions,
    write_therapeutic_outputs,
)
from .spatial import (
    SpatialGraphConfig,
    SpatialGraphResult,
    attach_spatial_to_mudata,
    build_spatial_heterograph,
    generate_spatial_coordinates,
    ligand_receptor_catalog,
    validate_spatial_graph,
    write_spatial_graph_outputs,
)
from .neurobiology import (
    NEURAL_CELL_TYPES,
    NEURO_STATES,
    NeurobiologyConfig,
    NeurobiologyResult,
    generate_neurobiology_dataset,
    generate_neurobiology_report,
    run_neurobiology_configuration,
    validate_neurobiology_outputs,
    write_neurobiology_outputs,
)
from .multimodal import (
    MODALITY_ORDER,
    MultimodalDemoConfig,
    generate_multimodal_mudata,
    modality_feature_frame,
    read_multimodal,
    validate_multimodal,
    write_multimodal,
)

from .intervention_generalization import (
    InterventionGeneralizationConfig,
    InterventionGeneralizationData,
    adapter_registry_frame,
    generate_intervention_generalization_data,
    run_intervention_generalization_benchmark,
    validate_intervention_generalization,
    save_external_intervention_npz,
    load_external_intervention_npz,
)
from .spatiotemporal_tissue import (
    SpatiotemporalTissueConfig,
    SpatiotemporalTissueData,
    generate_spatiotemporal_tissue_data,
    run_spatiotemporal_tissue_benchmark,
    validate_spatiotemporal_tissue,
    save_external_spatiotemporal_npz,
    load_external_spatiotemporal_npz,
    nicheformer_adapter_spec,
)

CausaFluxDataset = ChronoDataset

__all__ = [
    "CausaFlux",
    "CausaFluxConfig",
    "CausaFluxDataset",
    "ChronoDataset",
    "Standardizer",
    "InterventionEvent",
    "build_intervention_schedule",
    "events_from_csv",
    "simulate_with_uncertainty",
    "load_dataset",
    "run_experiment",
    "run_causal_experiment",
    "benchmark_linear_baselines",
    "transition_bootstrap_uncertainty",
    "MODALITY_ORDER",
    "MultimodalDemoConfig",
    "generate_multimodal_mudata",
    "modality_feature_frame",
    "read_multimodal",
    "validate_multimodal",
    "write_multimodal",
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "run_closed_loop_experimentation",
    "score_experiments",
    "select_batch",
    "validate_closed_loop_outputs",
    "update_closed_loop_from_observations",
    "write_closed_loop_outputs",
    "BiomarkerConfig",
    "BiomarkerResult",
    "binary_auc",
    "run_causal_biomarkers",
    "validate_biomarker_outputs",
    "write_biomarker_outputs",
    "TherapeuticConfig",
    "TherapeuticResult",
    "intervention_catalog",
    "build_regimen_catalog",
    "fit_therapeutic_model",
    "predict_regimens",
    "run_counterfactual_therapeutics",
    "validate_therapeutic_predictions",
    "write_therapeutic_outputs",
    "SpatialGraphConfig",
    "SpatialGraphResult",
    "attach_spatial_to_mudata",
    "build_spatial_heterograph",
    "generate_spatial_coordinates",
    "ligand_receptor_catalog",
    "validate_spatial_graph",
    "write_spatial_graph_outputs",
    "NEURAL_CELL_TYPES",
    "NEURO_STATES",
    "NeurobiologyConfig",
    "NeurobiologyResult",
    "generate_neurobiology_dataset",
    "generate_neurobiology_report",
    "run_neurobiology_configuration",
    "validate_neurobiology_outputs",
    "write_neurobiology_outputs",
    "InterventionGeneralizationConfig",
    "InterventionGeneralizationData",
    "adapter_registry_frame",
    "generate_intervention_generalization_data",
    "run_intervention_generalization_benchmark",
    "validate_intervention_generalization",
    "save_external_intervention_npz",
    "load_external_intervention_npz",
    "SpatiotemporalTissueConfig",
    "SpatiotemporalTissueData",
    "generate_spatiotemporal_tissue_data",
    "run_spatiotemporal_tissue_benchmark",
    "validate_spatiotemporal_tissue",
    "save_external_spatiotemporal_npz",
    "load_external_spatiotemporal_npz",
    "nicheformer_adapter_spec",
    "FEATURE_NAMES",
    "INTERVENTION_NAMES",
    "FATE_NAMES",
]

__version__ = "2.0.0"

from .platform import (
    PLATFORM_VERSION,
    DemoSpec,
    PlatformValidationReport,
    ValidationCheck,
    build_artifact_manifest,
    demo_registry_frame,
    environment_snapshot,
    finalize_research_platform,
    get_demo_registry,
    platform_doctor,
    sha256_file,
    validate_research_platform,
)

__all__.extend([
    "PLATFORM_VERSION",
    "DemoSpec",
    "PlatformValidationReport",
    "ValidationCheck",
    "build_artifact_manifest",
    "demo_registry_frame",
    "environment_snapshot",
    "finalize_research_platform",
    "get_demo_registry",
    "platform_doctor",
    "sha256_file",
    "validate_research_platform",
])

from .visualization import (
    EXPORT_PROFILES,
    FigureExport,
    apply_publication_style,
    compare_visual_baseline,
    export_figure,
    rebuild_reference_figures,
    validate_publication_bundle,
)

__all__.extend([
    "EXPORT_PROFILES",
    "FigureExport",
    "apply_publication_style",
    "compare_visual_baseline",
    "export_figure",
    "rebuild_reference_figures",
    "validate_publication_bundle",
])

from .realdata import (
    REALDATA_VERSION,
    BenchmarkSpec,
    SourceSpec,
    accession_manifest_frame,
    benchmark_registry_frame,
    build_download_plan,
    generate_realdata_reports,
    get_benchmark,
    load_benchmark_registry,
    preflight_benchmarks,
    validate_realdata_output,
    validate_realdata_registry,
)

from .realdata_adapters import (
    AdapterPlan, adapter_names, get_adapter, plan_source, write_accession_lock,
)

from .biological_validation import (
    VALIDATION_VERSION, HypothesisSpec, ValidationRun,
    load_hypothesis_registry, hypothesis_registry_frame, freeze_preregistration,
    run_biological_validation, write_biological_validation,
    run_and_write_biological_validation, validate_biological_validation,
)

__all__.extend([
    "VALIDATION_VERSION", "HypothesisSpec", "ValidationRun",
    "load_hypothesis_registry", "hypothesis_registry_frame", "freeze_preregistration",
    "run_biological_validation", "write_biological_validation",
    "run_and_write_biological_validation", "validate_biological_validation",
])

from .dynamic_benchmark import (
    DynamicBenchmarkConfig,
    DynamicBenchmarkData,
    MODEL_ORDER as DYNAMIC_MODEL_ORDER,
    DYNAMIC_MODELS,
    LatestStateLinear,
    LatestStateMLP,
    HistorySummaryMLP,
    GRUDynamic,
    CausaFluxFactorizedGRU,
    IrregularTimeTransformer,
    NeuralCDE,
    PRESCIENTComparator,
    generate_dynamic_benchmark_data,
    make_split,
    run_dynamic_benchmark,
    validate_dynamic_benchmark,
    validate_external_benchmark_data,
    save_external_benchmark_npz,
    load_external_benchmark_npz,
    attach_precomputed_embeddings,
    external_benchmark_contract,
)

__all__.extend([
    "DynamicBenchmarkConfig",
    "DynamicBenchmarkData",
    "DYNAMIC_MODEL_ORDER",
    "DYNAMIC_MODELS",
    "LatestStateLinear",
    "LatestStateMLP",
    "HistorySummaryMLP",
    "GRUDynamic",
    "CausaFluxFactorizedGRU",
    "IrregularTimeTransformer",
    "NeuralCDE",
    "PRESCIENTComparator",
    "generate_dynamic_benchmark_data",
    "make_split",
    "run_dynamic_benchmark",
    "validate_dynamic_benchmark",
    "validate_external_benchmark_data",
    "save_external_benchmark_npz",
    "load_external_benchmark_npz",
    "attach_precomputed_embeddings",
    "external_benchmark_contract",
])

from .multimodal_dynamic import (
    MultimodalDynamicConfig,
    MultimodalDynamicData,
    MODALITY_ORDER as MULTIMODAL_DYNAMIC_MODALITIES,
    MODEL_ORDER as MULTIMODAL_DYNAMIC_MODEL_ORDER,
    ProductOfExpertsFusion,
    MixtureOfExpertsFusion,
    ModalityEncoder,
    CausaFluxMultimodalDynamic,
    generate_multimodal_dynamic_data,
    history_split as multimodal_history_split,
    split_audit as multimodal_split_audit,
    run_multimodal_dynamic_benchmark,
    validate_multimodal_dynamic_benchmark,
    save_external_multimodal_npz,
    load_external_multimodal_npz,
)

__all__.extend([
    "MultimodalDynamicConfig", "MultimodalDynamicData",
    "MULTIMODAL_DYNAMIC_MODALITIES", "MULTIMODAL_DYNAMIC_MODEL_ORDER",
    "ProductOfExpertsFusion", "MixtureOfExpertsFusion", "ModalityEncoder",
    "CausaFluxMultimodalDynamic", "generate_multimodal_dynamic_data",
    "multimodal_history_split", "multimodal_split_audit",
    "run_multimodal_dynamic_benchmark", "validate_multimodal_dynamic_benchmark",
    "save_external_multimodal_npz", "load_external_multimodal_npz",
])

from .foundation_pretraining import FoundationPretrainingConfig, run_foundation_pretraining, validate_foundation_pretraining

from .prospective_loop import (
    PROSPECTIVE_VERSION,
    CONTRACT_VERSION,
    ProspectiveLoopConfig,
    ProspectiveLoopResult,
    default_hypotheses as prospective_default_hypotheses,
    default_experiment_catalog,
    experiment_contract_schema,
    qc_contract_schema,
    outcome_contract_schema,
    write_contract_bundle,
    run_prospective_loop,
    validate_prospective_loop,
    ingest_experimental_qc,
    ingest_external_cycle,
)

__all__.extend([
    "PROSPECTIVE_VERSION",
    "CONTRACT_VERSION",
    "ProspectiveLoopConfig",
    "ProspectiveLoopResult",
    "prospective_default_hypotheses",
    "default_experiment_catalog",
    "experiment_contract_schema",
    "qc_contract_schema",
    "outcome_contract_schema",
    "write_contract_bundle",
    "run_prospective_loop",
    "validate_prospective_loop",
    "ingest_experimental_qc",
    "ingest_external_cycle",
])


from .virtual_cell import (
    VIRTUAL_CELL_VERSION, STATE_NAMES, ModuleEvidence, InterventionScenario, DEFAULT_SCENARIOS,
    load_module_evidence, simulate_scenario, run_virtual_cell_ensemble,
)
from .real_world_hub import (
    REAL_WORLD_HUB_VERSION, UserDatasetContract, build_real_world_evidence_matrix,
    register_user_dataset, preview_tabular_dataset,
)
from .virtual_cell_release import RELEASE_VERSION, run_virtual_cell_release
from .virtual_cell_validation import build_validation_matrix, validate_virtual_cell_release

__all__.extend([
    "VIRTUAL_CELL_VERSION", "STATE_NAMES", "ModuleEvidence", "InterventionScenario", "DEFAULT_SCENARIOS",
    "load_module_evidence", "simulate_scenario", "run_virtual_cell_ensemble",
    "REAL_WORLD_HUB_VERSION", "UserDatasetContract", "build_real_world_evidence_matrix",
    "register_user_dataset", "preview_tabular_dataset", "RELEASE_VERSION",
    "run_virtual_cell_release", "build_validation_matrix", "validate_virtual_cell_release",
])


from .evidence_ledger import EvidenceRecord, claim_registry_frame, validate_ledger
from .longitudinal_realdata import public_dataset_registry, convert_longitudinal_table, run_real_longitudinal_benchmark
from .shift_calibration import evaluate_shift_calibration, evaluate_shift_calibration_file
from .v2_release import V2_RELEASE_VERSION, run_v2_release
from .v2_release_gate import evaluate_v2_release_gate, validate_v2_output
__all__.extend(["EvidenceRecord","claim_registry_frame","validate_ledger","public_dataset_registry","convert_longitudinal_table","run_real_longitudinal_benchmark","evaluate_shift_calibration","evaluate_shift_calibration_file","V2_RELEASE_VERSION","run_v2_release","evaluate_v2_release_gate","validate_v2_output"])
