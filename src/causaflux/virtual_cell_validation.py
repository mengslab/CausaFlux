"""Cross-module prospective virtual-cell validation for CausaFlux v1.9.0."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

import pandas as pd

VALIDATION_VERSION = "1.9.0"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_validation_matrix(project_root: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(project_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dynamic = _load(root / "dynamic_benchmark_reference" / "dynamic_benchmark_status.json")
    multimodal = _load(root / "multimodal_dynamic_reference" / "multimodal_exit_gate.json")
    intervention = _load(root / "intervention_generalization_reference" / "intervention_exit_gate.json")
    tissue = _load(root / "spatiotemporal_tissue_reference" / "spatiotemporal_exit_gate.json")
    foundation = _load(root / "foundation_pretraining_reference" / "foundation_pretraining_gate.json")
    prospective = _load(root / "prospective_loop_reference" / "prospective_exit_gate.json")
    biological = _load(root / "biological_validation_reference" / "biological_validation_status.json")
    realdata = _load(root / "realdata_reference" / "realdata_status.json")

    checks = [
        ("dynamic_state", dynamic.get("gate", {}).get("status") == "PASS", "synthetic held-out perturbation histories", "required"),
        ("multimodal_state", bool(multimodal.get("software_exit_gate_passed")), "synthetic multimodal longitudinal fixture", "required"),
        ("intervention_generalization", intervention.get("software_generalization_gate") == "PASS", "synthetic unseen intervention axes", "required"),
        ("spatiotemporal_context", tissue.get("software_spatiotemporal_gate") == "PASS", "synthetic held-out donors/sections", "required"),
        ("foundation_transfer", foundation.get("software_pretraining_gate") == "PASS", "synthetic donor/tissue/perturbation holdouts", "required"),
        ("prospective_loop", prospective.get("software_gate") == "PASS" and prospective.get("required_sequence_complete") is True, "three prospectively locked synthetic cycles", "required"),
        ("cycle3_independence", prospective.get("cycle3_independent_confirmation_or_falsification") is True, "independent confirmation/falsification role", "required"),
        ("real_world_registry", bool(realdata.get("registry_valid")), "accession-pinned real-world registry", "required"),
        ("real_world_observational_replication", int(biological.get("independent_source_cohort_replication_established", 0)) > 0, "real source-cohort replication", "required"),
        ("real_perturbational_validation", int(biological.get("perturbational_validation_established", 0)) > 0, "real perturbational validation", "real_claim"),
        ("real_three_cycle_prospective_validation", prospective.get("real_prospective_claim_authorized") is True, "real Cycle 1→2→3 locked evidence", "real_claim"),
    ]
    frame = pd.DataFrame(checks, columns=["module_or_gate", "passed", "evidence", "requirement_class"])
    frame["status"] = frame.passed.map({True: "PASS", False: "PENDING"})
    frame_path = out / "prospective_virtual_cell_validation_matrix.csv"
    frame.to_csv(frame_path, index=False)

    software_required = frame[frame.requirement_class == "required"].passed.all()
    real_required = frame[frame.requirement_class == "real_claim"].passed.all()
    status = {
        "framework": "CausaFlux",
        "version": VALIDATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_integrated_virtual_cell_gate": "PASS" if software_required else "FAIL",
        "real_prospectively_validated_virtual_cell_gate": "PASS" if (software_required and real_required) else "PENDING",
        "prospectively_validated_virtual_cell": bool(software_required and real_required),
        "software_reference_is_prospectively_locked": bool(prospective.get("software_gate") == "PASS"),
        "real_world_observational_evidence_integrated": int(biological.get("independent_source_cohort_replication_established", 0)) > 0,
        "authorization_boundary": (
            "CausaFlux may be labeled a prospectively validated virtual cell only after real perturbational evidence and "
            "three real prospectively locked cycles, including independent Cycle 3 confirmation/falsification, pass."
        ),
        "n_required_software_checks": int((frame.requirement_class == "required").sum()),
        "n_required_software_passed": int(((frame.requirement_class == "required") & frame.passed).sum()),
        "n_real_claim_checks": int((frame.requirement_class == "real_claim").sum()),
        "n_real_claim_passed": int(((frame.requirement_class == "real_claim") & frame.passed).sum()),
    }
    status_path = out / "prospective_virtual_cell_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return frame, status


def validate_virtual_cell_release(output_dir: str | Path, *, require_real_prospective: bool = False) -> dict[str, Any]:
    out = Path(output_dir)
    required = [
        "ai/ai_model_router.csv",
        "ai/virtual_cell_trajectories.csv",
        "ai/ai_guided_intervention_ranking.csv",
        "real_world/real_world_hub_status.json",
        "validation/prospective_virtual_cell_status.json",
        "report/index.html",
        "figures/figure_inventory.csv",
    ]
    missing = [item for item in required if not (out / item).exists()]
    status = _load(out / "validation" / "prospective_virtual_cell_status.json") if not missing else {}
    figure_inventory = pd.read_csv(out / "figures" / "figure_inventory.csv") if (out / "figures" / "figure_inventory.csv").exists() else pd.DataFrame()
    figure_ok = bool(len(figure_inventory) >= 5 and figure_inventory.get("validated", pd.Series(dtype=bool)).all())
    if figure_ok:
        for row in figure_inventory.itertuples(index=False):
            for field in ("png", "svg", "pdf", "tiff"):
                path = out / "figures" / str(getattr(row, field))
                if not path.exists() or path.stat().st_size <= 500:
                    figure_ok = False
            manifest = out / "figures" / "figure_manifests" / f"{row.figure_id}.json"
            if not manifest.exists():
                figure_ok = False
            else:
                payload = _load(manifest)
                if int(payload.get("dpi", 0)) < 600 or payload.get("version") != "1.9.0":
                    figure_ok = False
    software_ok = status.get("software_integrated_virtual_cell_gate") == "PASS"
    real_ok = status.get("real_prospectively_validated_virtual_cell_gate") == "PASS"
    valid = not missing and figure_ok and software_ok and (real_ok if require_real_prospective else True)
    return {
        "valid": bool(valid),
        "missing": missing,
        "figure_bundle_valid": figure_ok,
        "software_gate": status.get("software_integrated_virtual_cell_gate", "UNKNOWN"),
        "real_prospective_gate": status.get("real_prospectively_validated_virtual_cell_gate", "UNKNOWN"),
        "require_real_prospective": require_real_prospective,
    }
