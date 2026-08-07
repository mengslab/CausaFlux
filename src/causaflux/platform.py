"""Validated research-platform utilities for CausaFlux v1.7.0.

The platform layer does not change scientific estimators. It adds reproducible
release manifests, environment snapshots, dataset cards, demo discovery, and
cross-domain validation gates for the cancer and neurobiology workflows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform as _platform
import re
import shutil
import sys
from typing import Any, Iterable

import pandas as pd

from .visualization.publication import validate_publication_bundle

PLATFORM_VERSION = "1.7.0"
FRAMEWORK_NAME = "CausaFlux"


@dataclass(frozen=True)
class DemoSpec:
    demo_id: str
    title: str
    domain: str
    description: str
    config: str
    command: str
    expected_output: str
    synthetic: bool = True


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    category: str
    status: str
    message: str
    evidence: str = ""


@dataclass
class PlatformValidationReport:
    framework: str
    version: str
    valid: bool
    generated_at_utc: str
    output_dir: str
    checks: list[ValidationCheck]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        payload["passed"] = sum(check.status == "pass" for check in self.checks)
        payload["failed"] = sum(check.status == "fail" for check in self.checks)
        payload["warnings"] = sum(check.status == "warn" for check in self.checks)
        return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, Any]:
    packages = [
        "causaflux", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "networkx", "anndata", "mudata", "h5py", "torch", "PyYAML",
    ]
    return {
        "framework": FRAMEWORK_NAME,
        "version": PLATFORM_VERSION,
        "generated_at_utc": utc_now(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": _platform.platform(),
        "machine": _platform.machine(),
        "processor": _platform.processor(),
        "implementation": _platform.python_implementation(),
        "packages": {name: _distribution_version(name) for name in packages},
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def get_demo_registry(project_root: str | Path | None = None) -> list[DemoSpec]:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    return [
        DemoSpec(
            demo_id="foundation_pretraining",
            title="Foundation adapter and pretraining benchmark",
            domain="foundation-pretraining",
            description=(
                "External scGPT, GET, Nicheformer, MrVI, ESM-2, MolFormer and DINOv2 adapter contracts plus "
                "multi-objective CausaFlux pretraining evaluated on dynamic and intervention transfer."
            ),
            config=str(root / "demos" / "foundation_pretraining" / "config.yaml"),
            command="sh demos/foundation_pretraining/run.sh",
            expected_output="demo_outputs/foundation_pretraining",
        ),
        DemoSpec(
            demo_id="spatiotemporal_digital_tissue",
            title="Spatiotemporal digital tissue benchmark",
            domain="spatiotemporal-tissue",
            description=(
                "Time-varying heterogeneous cell graphs with learned communication gates, "
                "continuous graph-conditioned dynamics, spatial interference estimands, "
                "regulatory/organelle graph layers, and tissue outcome prediction."
            ),
            config=str(root / "demos" / "spatiotemporal_digital_tissue" / "config.yaml"),
            command="sh demos/spatiotemporal_digital_tissue/run.sh",
            expected_output="demo_outputs/spatiotemporal_digital_tissue",
        ),
        DemoSpec(
            demo_id="intervention_generalization",
            title="Intervention generalization benchmark",
            domain="intervention-generalization",
            description=(
                "Unseen perturbation, dose, combination and temporal-sequence prediction with "
                "gene/compound embeddings, PK/PD, conformal uncertainty, support diagnostics and causal comparators."
            ),
            config=str(root / "demos" / "intervention_generalization" / "config.yaml"),
            command="sh demos/intervention_generalization/run.sh",
            expected_output="demo_outputs/intervention_generalization",
        ),
        DemoSpec(
            demo_id="multimodal_dynamic_state",
            title="Multimodal dynamic state benchmark",
            domain="multimodal-dynamic",
            description=(
                "Modality-specific encoders with PoE/MoE fusion, modality dropout, cross-modal decoding, "
                "early imaging/reporters, donor/cohort context, and explicit MNAR sensitivity."
            ),
            config=str(root / "demos" / "multimodal_dynamic_state" / "config.yaml"),
            command="sh demos/multimodal_dynamic_state/run.sh",
            expected_output="demo_outputs/multimodal_dynamic_state",
        ),
        DemoSpec(
            demo_id="dynamic_model_benchmark",
            title="Dynamic trajectory benchmark",
            domain="dynamic-modeling",
            description=(
                "Held-out perturbation-history comparison of static baselines, recurrent models, "
                "an irregular-time transformer, Neural CDE, and a PRESCIENT-style comparator."
            ),
            config=str(root / "demos" / "dynamic_model_benchmark" / "config.yaml"),
            command="sh demos/dynamic_model_benchmark/run.sh",
            expected_output="demo_outputs/dynamic_model_benchmark",
        ),
        DemoSpec(
            demo_id="cancer_quickstart",
            title="Cancer disease-evolution quickstart",
            domain="cancer",
            description=(
                "Longitudinal tumor–immune–stromal multimodal analysis with spatial graphs, "
                "counterfactual therapeutics, causal biomarkers, and closed-loop experiments."
            ),
            config=str(root / "demos" / "cancer_quickstart" / "config.yaml"),
            command="sh demos/cancer_quickstart/run.sh",
            expected_output="demo_outputs/cancer_quickstart",
        ),
        DemoSpec(
            demo_id="neurobiology_quickstart",
            title="Neural–glial trajectory quickstart",
            domain="neurobiology",
            description=(
                "Neural–glial trajectories integrating molecular state, live imaging, "
                "electrophysiology, APOE context, and degeneration-risk prediction."
            ),
            config=str(root / "demos" / "neurobiology_quickstart" / "config.yaml"),
            command="sh demos/neurobiology_quickstart/run.sh",
            expected_output="demo_outputs/neurobiology_quickstart",
        ),
        DemoSpec(
            demo_id="integrated_reference",
            title="Integrated cancer and neurobiology reference",
            domain="cross-domain",
            description=(
                "The complete nine-stage research-platform demonstration with all validated "
                "cancer, spatial, therapeutic, biomarker, active-learning, and neurobiology outputs."
            ),
            config=str(root / "configs" / "cancer_closed_loop_v1.7.0.yaml"),
            command="sh run_synthetic_smoke.sh",
            expected_output="causaflux_v1.7.0_output",
        ),
        DemoSpec(
            demo_id="biological_validation",
            title="Preregistered SEA-AD biological validation",
            domain="neurobiology-real-data",
            description="Public SEA-AD discovery/replication analyses with evidence-constrained conclusions and manuscript source data.",
            config=str(root / "src" / "causaflux" / "resources" / "validation" / "registry.yaml"),
            command="sh demos/biological_validation/run.sh",
            expected_output="demo_outputs/biological_validation",
            synthetic=False,
        ),
        DemoSpec(
            demo_id="realdata_registry",
            title="Accession-pinned real-data registry",
            domain="cross-domain-real-data",
            description="HTAN, GDC/CPTAC, DepMap/LINCS, SEA-AD, AMP-AD and DANDI accessions, licenses and validation cohorts.",
            config=str(root / "benchmarks" / "manifests" / "registry.yaml"),
            command="sh demos/realdata_registry/run.sh",
            expected_output="demo_outputs/realdata_registry",
            synthetic=False,
        ),
    ]


def demo_registry_frame(project_root: str | Path | None = None) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in get_demo_registry(project_root)])


def _artifact_category(relative: Path) -> str:
    if len(relative.parts) == 1:
        return "root"
    return relative.parts[0]


def build_artifact_manifest(
    output_dir: str | Path,
    *,
    exclude_prefixes: Iterable[str] = ("provenance",),
) -> pd.DataFrame:
    output = Path(output_dir).resolve()
    excluded = tuple(str(value).rstrip("/") for value in exclude_prefixes)
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        relative_text = relative.as_posix()
        if any(relative_text == prefix or relative_text.startswith(prefix + "/") for prefix in excluded):
            continue
        rows.append(
            {
                "relative_path": relative_text,
                "category": _artifact_category(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "category", "size_bytes", "sha256"])


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "No records available."
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    header = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = []
    for _, row in view.iterrows():
        body.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join([header, rule, *body])


def write_dataset_cards(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    cards = output / "cards"
    cards.mkdir(parents=True, exist_ok=True)

    cancer = pd.read_csv(output / "data" / "cancer_longitudinal.csv")
    cancer_card = cards / "cancer_demo_dataset_card.md"
    cancer_card.write_text(
        "\n".join(
            [
                "# Cancer demonstration dataset card",
                "",
                "## Status",
                "Synthetic software-verification dataset. It is not a clinical or biological cohort.",
                "",
                "## Scope",
                f"- Observations: {len(cancer):,}",
                f"- Donors: {cancer['donor_id'].nunique()}",
                f"- Cell types: {cancer['cell_type'].nunique()}",
                f"- Time points: {cancer['time_hours'].nunique()}",
                f"- Treatment arms: {cancer['treatment_arm'].nunique() if 'treatment_arm' in cancer else 'not recorded'}",
                "",
                "## Intended use",
                "Testing data validation, donor-held-out modeling, multimodal fusion, spatial graphs, "
                "causal estimation, therapeutic counterfactuals, biomarkers, and experiment design.",
                "",
                "## Prohibited interpretation",
                "Do not treat synthetic effect sizes, biomarkers, regimens, or communication circuits as discoveries.",
                "",
                "## Validation unit",
                "Donors define evaluation folds; cells are not treated as independent test subjects.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    neuro = pd.read_csv(output / "neurobiology" / "neural_glial_observations.csv")
    neuro_card = cards / "neurobiology_demo_dataset_card.md"
    neuro_card.write_text(
        "\n".join(
            [
                "# Neurobiology demonstration dataset card",
                "",
                "## Status",
                "Synthetic software-verification dataset. It is not a patient, animal, or organoid cohort.",
                "",
                "## Scope",
                f"- Observations: {len(neuro):,}",
                f"- Donors: {neuro['donor_id'].nunique()}",
                f"- Cell types: {neuro['cell_type'].nunique()}",
                f"- Time points: {neuro['time_days'].nunique()}",
                f"- APOE contexts: {neuro['apoe_genotype'].nunique()}",
                "",
                "## Intended use",
                "Testing neural–glial trajectory inference, imaging/electrophysiology integration, "
                "APOE-stratified summaries, donor-held-out risk prediction, and driver ranking.",
                "",
                "## Missing-modality policy",
                "Electrophysiology unavailable in non-neuronal populations is represented as missingness, not zero.",
                "",
                "## Prohibited interpretation",
                "Do not interpret synthetic APOE effects, risk scores, or glial drivers as biological evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"cancer": cancer_card, "neurobiology": neuro_card}


def write_platform_model_card(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    cards = output / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    path = cards / "platform_model_card.md"
    path.write_text(
        f"""# CausaFlux v{PLATFORM_VERSION} platform model card

## Purpose

CausaFlux is a research platform for interpretable causal modeling of disease-state evolution. The v1.7.0 release combines cancer and neurobiology workflows with a Nature/Cell-style vector-first graphics, source-data, and visual-regression contract.

## Included model families

- donor-aware transparent linear baselines and calibrated ensembles;
- multimodal state models for RNA, ATAC, protein, mutation, and drug response;
- tumor–immune–stromal spatial heterographs and ligand–receptor circuits;
- causal treatment-effect and counterfactual therapeutic models;
- early-warning causal biomarker ranking and donor-held-out panels;
- information-gain-based closed-loop experiment selection;
- neural–glial trajectory and degeneration-risk models with imaging/electrophysiology integration.

## Reference dimensions

- Cancer observations: {manifest.get('data_rows')}
- Neurobiology observations: {manifest.get('neurobiology_observations')}
- Spatial nodes: {manifest.get('spatial_nodes')}
- Therapeutic regimens: {manifest.get('therapeutic_regimens')}
- Closed-loop candidates: {manifest.get('closed_loop_candidates')}

## Evaluation policy

Donors define held-out folds. Bootstrap procedures resample donors. Calibration is estimated without donor leakage. Results include uncertainty and explicit evidence levels.

## Limitations

The packaged reference datasets are synthetic. High discrimination, ranked mechanisms, biomarkers, regimens, and experiments demonstrate software behavior only. Real-world deployment requires prospective study design, external cohorts, biological replication, assay validation, causal-identification review, and domain-expert oversight.

## Clinical status

Not a medical device. Not validated for diagnosis, prognosis, patient selection, or treatment decisions.
""",
        encoding="utf-8",
    )
    return path


def _check(
    checks: list[ValidationCheck],
    check_id: str,
    category: str,
    condition: bool,
    pass_message: str,
    fail_message: str,
    evidence: str = "",
    *,
    warning: bool = False,
) -> None:
    if condition:
        checks.append(ValidationCheck(check_id, category, "pass", pass_message, evidence))
    else:
        checks.append(ValidationCheck(check_id, category, "warn" if warning else "fail", fail_message, evidence))


def validate_research_platform(
    output_dir: str | Path,
    *,
    verify_hashes: bool = True,
) -> PlatformValidationReport:
    output = Path(output_dir).resolve()
    checks: list[ValidationCheck] = []

    manifest_path = output / "run_manifest.json"
    stage_path = output / "stage_status.json"
    _check(checks, "manifest_exists", "provenance", manifest_path.exists(),
           "Run manifest is present.", "Run manifest is missing.", str(manifest_path))
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check(checks, "version_consistency", "provenance",
           manifest.get("framework") == FRAMEWORK_NAME and manifest.get("version") == PLATFORM_VERSION,
           "Framework and version metadata are consistent.",
           "Framework/version metadata are inconsistent.",
           f"{manifest.get('framework')} {manifest.get('version')}")

    stage: dict[str, Any] = {}
    if stage_path.exists():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
    _check(checks, "workflow_complete", "workflow",
           stage.get("stage") == "complete" and stage.get("version") == PLATFORM_VERSION,
           "All workflow stages completed.", "Workflow completion marker is missing or stale.", str(stage_path))

    required_domains = {
        "cancer": output / "data" / "cancer_longitudinal.csv",
        "multimodal": output / "multimodal" / "causaflux_multimodal.h5mu",
        "spatial": output / "spatial_graph" / "spatial_heterograph.graphml",
        "therapeutics": output / "therapeutics" / "all_regimen_predictions.csv",
        "biomarkers": output / "biomarkers" / "causal_biomarker_ranking.csv",
        "closed_loop": output / "active_learning" / "round1_selected_batch.csv",
        "neurobiology": output / "neurobiology" / "neural_glial_observations.csv",
    }
    for domain, path in required_domains.items():
        _check(checks, f"domain_{domain}", "domain_coverage", path.exists() and path.stat().st_size > 0,
               f"{domain} workflow artifact is present.", f"{domain} workflow artifact is missing.", str(path))

    split_path = output / "baselines" / "donor_split_manifest.csv"
    no_overlap = False
    if split_path.exists():
        splits = pd.read_csv(split_path).fillna("")
        no_overlap = not splits["donor_overlap"].astype(str).str.len().gt(0).any()
    _check(checks, "donor_separation", "validation", no_overlap,
           "Donor-held-out split manifest has zero overlap.", "Donor leakage was detected or split manifest is absent.", str(split_path))

    cancer_card = output / "cards" / "cancer_demo_dataset_card.md"
    neuro_card = output / "cards" / "neurobiology_demo_dataset_card.md"
    model_card = output / "cards" / "platform_model_card.md"
    _check(checks, "research_cards", "documentation",
           all(path.exists() and path.stat().st_size > 0 for path in (cancer_card, neuro_card, model_card)),
           "Cancer, neurobiology, and platform cards are present.",
           "One or more research cards are missing.", str(output / "cards"))

    reports = [output / "report" / name for name in ("index.html", "neurobiology.html", "platform.html")]
    _check(checks, "reports", "reporting", all(path.exists() and path.stat().st_size > 0 for path in reports),
           "Integrated, neurobiology, and platform reports are present.",
           "One or more report pages are missing.", str(output / "report"))

    publication = validate_publication_bundle(output, check_hashes=False)
    _check(
        checks,
        "publication_graphics",
        "reporting",
        bool(publication.get("valid") and publication.get("n_figures", 0) >= 30),
        f"{publication.get('n_figures', 0)} publication figure bundles include vector/raster exports and panel source data.",
        "Publication graphics bundle is incomplete or invalid.",
        str(output / "publication_graphics"),
    )

    artifact_manifest = output / "provenance" / "artifact_manifest.csv"
    hashes_ok = artifact_manifest.exists()
    mismatch_count = 0
    if hashes_ok and verify_hashes:
        table = pd.read_csv(artifact_manifest)
        for row in table.itertuples(index=False):
            path = output / row.relative_path
            if not path.exists() or sha256_file(path) != row.sha256:
                mismatch_count += 1
        hashes_ok = mismatch_count == 0
    _check(checks, "artifact_hashes", "provenance", hashes_ok,
           "Artifact hashes are complete and reproducible.",
           f"Artifact hash validation failed for {mismatch_count} files.", str(artifact_manifest))

    environment_path = output / "provenance" / "environment.json"
    _check(checks, "environment_snapshot", "provenance", environment_path.exists(),
           "Environment snapshot is present.", "Environment snapshot is missing.", str(environment_path))

    synthetic_declared = False
    cancer_validation = output / "data" / "validation_report.json"
    neuro_qc = output / "neurobiology" / "neurobiology_qc.json"
    if cancer_validation.exists() and neuro_qc.exists():
        cancer_text = cancer_validation.read_text(encoding="utf-8").lower()
        neuro_payload = json.loads(neuro_qc.read_text(encoding="utf-8"))
        synthetic_declared = "synthetic" in cancer_text or bool(neuro_payload.get("synthetic_demonstration", True))
    _check(checks, "synthetic_disclosure", "governance", synthetic_declared,
           "Synthetic demonstration status is disclosed.",
           "Synthetic demonstration status could not be confirmed.", warning=True)

    valid = not any(check.status == "fail" for check in checks)
    return PlatformValidationReport(
        framework=FRAMEWORK_NAME,
        version=PLATFORM_VERSION,
        valid=valid,
        generated_at_utc=utc_now(),
        output_dir=str(output),
        checks=checks,
    )


def write_validation_report(report: PlatformValidationReport, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    json_path = provenance / "platform_validation.json"
    csv_path = provenance / "platform_validation.csv"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([asdict(check) for check in report.checks]).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def generate_platform_report(output_dir: str | Path, report: PlatformValidationReport) -> Path:
    output = Path(output_dir).resolve()
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    checks = pd.DataFrame([asdict(check) for check in report.checks])
    check_rows = "".join(
        f"<tr><td>{row.check_id}</td><td>{row.category}</td><td class='{row.status}'>{row.status.upper()}</td><td>{row.message}</td></tr>"
        for row in checks.itertuples(index=False)
    )
    demos = demo_registry_frame(Path(__file__).resolve().parents[2])
    demo_rows = "".join(
        f"<tr><td>{row.title}</td><td>{row.domain}</td><td><code>{row.command}</code></td><td>{row.description}</td></tr>"
        for row in demos.itertuples(index=False)
    )
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>CausaFlux v{PLATFORM_VERSION} platform validation</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f5f7fb;color:#172033}}.hero{{padding:44px 7%;background:#172033;color:white}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 6px 24px #18203012}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e7eaf0;vertical-align:top}}code{{background:#eef1f6;padding:2px 6px;border-radius:5px}}.pass{{color:#16794b;font-weight:700}}.fail{{color:#b42318;font-weight:700}}.warn{{color:#9a6700;font-weight:700}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}.metric{{background:#f3f6fb;border-radius:10px;padding:16px}}.metric b{{font-size:1.55rem;display:block}}</style></head>
<body><div class='hero'><h1>CausaFlux v{PLATFORM_VERSION}</h1><p>Validated cross-domain research platform: cancer and neurobiology workflows, provenance, documentation, and packaged demonstrations.</p></div><main>
<section><h2>Release summary</h2><div class='metrics'>
<div class='metric'><b>{'PASS' if report.valid else 'FAIL'}</b>platform validation</div>
<div class='metric'><b>{manifest.get('data_rows', 0):,}</b>cancer observations</div>
<div class='metric'><b>{manifest.get('neurobiology_observations', 0):,}</b>neural–glial observations</div>
<div class='metric'><b>{manifest.get('therapeutic_regimens', 0)}</b>therapeutic regimens</div>
<div class='metric'><b>{manifest.get('closed_loop_candidates', 0)}</b>experiment candidates</div>
</div></section>
<section><h2>Validation gates</h2><table><thead><tr><th>Check</th><th>Category</th><th>Status</th><th>Result</th></tr></thead><tbody>{check_rows}</tbody></table></section>
<section><h2>Packaged demonstrations</h2><table><thead><tr><th>Demo</th><th>Domain</th><th>Command</th><th>Purpose</th></tr></thead><tbody>{demo_rows}</tbody></table></section>
<section><h2>Publication graphics</h2><p>Every scientific panel is exported as 600-dpi PNG/TIFF and editable SVG/PDF, with panel-level source-data tables, figure manifests, deterministic graph layouts, and perceptual visual-regression baselines.</p><p><a href='../publication_graphics/figure_inventory.csv'>Figure inventory</a> · <a href='../publication_graphics/visual_regression_baselines.csv'>Visual baselines</a></p></section>
<section><h2>Research governance</h2><p>The bundled datasets and all derived rankings are synthetic software-verification outputs. CausaFlux v1.7.0 is a research platform, not a medical device or clinical decision system.</p><p><a href='index.html'>Integrated report</a> · <a href='neurobiology.html'>Neurobiology report</a></p></section>
</main></body></html>"""
    path = report_dir / "platform.html"
    path.write_text(html, encoding="utf-8")
    return path


def finalize_research_platform(
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> PlatformValidationReport:
    output = Path(output_dir).resolve()
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)

    write_dataset_cards(output)
    write_platform_model_card(output)
    registry = demo_registry_frame(project)
    # Persist portable project-relative config paths in packaged reports while
    # retaining absolute paths from get_demo_registry() for runtime discovery.
    def _portable_config(value: str) -> str:
        path = Path(value)
        try:
            return path.resolve().relative_to(project).as_posix()
        except (ValueError, OSError):
            return path.as_posix()
    registry["config"] = registry["config"].map(_portable_config)
    registry.to_csv(output / "demo_registry.csv", index=False)
    (provenance / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Add v1 platform fields before hashing the output tree.
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "framework": FRAMEWORK_NAME,
            "version": PLATFORM_VERSION,
            "platform_profile": "validated_research_platform",
            "domains": ["cancer", "neurobiology"],
            "validation_policy": "donor_held_out_with_donor_bootstrap",
            "synthetic_demonstration": True,
            "platform_report": "report/platform.html",
            "artifact_manifest": "provenance/artifact_manifest.csv",
            "environment_snapshot": "provenance/environment.json",
            "demo_registry": "demo_registry.csv",
            "publication_profile": "Nature/Cell vector-first",
            "publication_figure_inventory": "publication_graphics/figure_inventory.csv",
            "visual_regression_baselines": "publication_graphics/visual_regression_baselines.csv",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Create a preliminary page so it is included in the hash manifest.
    preliminary = PlatformValidationReport(
        framework=FRAMEWORK_NAME,
        version=PLATFORM_VERSION,
        valid=True,
        generated_at_utc=utc_now(),
        output_dir=str(output),
        checks=[],
    )
    generate_platform_report(output, preliminary)

    artifacts = build_artifact_manifest(output)
    artifacts.to_csv(provenance / "artifact_manifest.csv", index=False)
    summary = {
        "framework": FRAMEWORK_NAME,
        "version": PLATFORM_VERSION,
        "generated_at_utc": utc_now(),
        "artifact_count": int(len(artifacts)),
        "artifact_bytes": int(artifacts["size_bytes"].sum()) if not artifacts.empty else 0,
        "categories": artifacts.groupby("category").size().astype(int).to_dict() if not artifacts.empty else {},
    }
    (provenance / "provenance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = validate_research_platform(output, verify_hashes=True)
    write_validation_report(report, output)
    generate_platform_report(output, report)

    # Platform page changed after hashing. Refresh only its hash row.
    artifacts = build_artifact_manifest(output)
    artifacts.to_csv(provenance / "artifact_manifest.csv", index=False)
    report = validate_research_platform(output, verify_hashes=True)
    write_validation_report(report, output)

    index_path = output / "report" / "index.html"
    if index_path.exists():
        html = index_path.read_text(encoding="utf-8")
        marker = "<!-- CAUSAFLUX_PLATFORM_V1 -->"
        section = (
            f"{marker}<section class='section'><h2>Validated research platform</h2>"
            f"<p>CausaFlux v{PLATFORM_VERSION} unifies the cancer and neurobiology workflows "
            "under cross-domain validation, provenance, dataset cards, a model card, and packaged demos.</p>"
            "<p><strong>Publication bundle:</strong> 30 Nature/Cell-profile figures with editable SVG/PDF, "
            "600-dpi PNG/TIFF, panel-level source data, deterministic graph layouts, and visual-regression baselines.</p>"
            "<p><a href='platform.html'>Open platform validation report →</a> · "
            "<a href='../publication_graphics/figure_inventory.csv'>Figure inventory →</a> · "
            "<a href='../publication_graphics/visual_regression_baselines.csv'>Visual baselines →</a></p></section>"
        )
        if marker in html:
            html = re.sub(re.escape(marker) + r"<section class='section'>.*?</section>", section, html, count=1, flags=re.S)
        else:
            html = html.replace("</body>", section + "</body>") if "</body>" in html else html + section
        index_path.write_text(html, encoding="utf-8")

    # Refresh hashes after the integrated report receives the v1 platform link.
    artifacts = build_artifact_manifest(output)
    artifacts.to_csv(provenance / "artifact_manifest.csv", index=False)
    report = validate_research_platform(output, verify_hashes=True)
    write_validation_report(report, output)
    return report


def platform_doctor(project_root: str | Path | None = None) -> dict[str, Any]:
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    python_supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    required = [
        project / "pyproject.toml",
        project / "configs" / "cancer_closed_loop_v1.7.0.yaml",
        project / "configs" / "neurobiology_v1.7.0.yaml",
        project / "demos" / "cancer_quickstart" / "config.yaml",
        project / "demos" / "neurobiology_quickstart" / "config.yaml",
        project / "demos" / "dynamic_model_benchmark" / "config.yaml",
    ]
    payload = {
        "framework": FRAMEWORK_NAME,
        "version": PLATFORM_VERSION,
        "python_supported": python_supported,
        "python": sys.version.split()[0],
        "project_root": str(project),
        "required_files_present": all(path.exists() for path in required),
        "missing_files": [str(path) for path in required if not path.exists()],
        "demo_count": len(get_demo_registry(project)),
        "environment": environment_snapshot(),
    }
    payload["ready"] = bool(payload["python_supported"] and payload["required_files_present"])
    return payload
