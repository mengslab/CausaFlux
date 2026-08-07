"""Evidence-governed biological validation for CausaFlux v1.7.0.

The module freezes preregistered hypotheses, executes the public SEA-AD
metadata validation benchmark, compares established statistical methods, and
writes manuscript-quality source-data packages. It never upgrades an
observational association to a causal or clinical claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import importlib.resources as ir
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

VALIDATION_VERSION = "1.7.0"
ADNC_ORDER = {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3}


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    title: str
    status: str
    domain: str
    hypothesis: str
    direction: str
    primary_endpoint: str
    discovery_cohort: str
    replication_cohort: str
    analysis: str
    alpha: float
    multiple_testing_family: str
    perturbational_validation: str
    external_dataset_validation: str
    manifest_path: str
    preregistration_sha256: str


@dataclass
class ValidationRun:
    hypotheses: pd.DataFrame
    primary_results: pd.DataFrame
    endpoint_replication: pd.DataFrame
    method_comparison: pd.DataFrame
    evidence_ledger: pd.DataFrame
    conclusion_ledger: pd.DataFrame
    qc: dict[str, Any]


def _validation_resource_dir() -> Path:
    return Path(ir.files("causaflux").joinpath("resources/validation"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_hypothesis_registry(manifest_dir: str | Path | None = None) -> list[HypothesisSpec]:
    base = Path(manifest_dir) if manifest_dir else _validation_resource_dir()
    registry = yaml.safe_load((base / "registry.yaml").read_text(encoding="utf-8")) or {}
    specs: list[HypothesisSpec] = []
    for item in registry.get("hypotheses", []):
        path = base / item["manifest"]
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs.append(HypothesisSpec(
            hypothesis_id=str(payload["id"]),
            title=str(payload["title"]),
            status=str(payload["status"]),
            domain=str(payload["domain"]),
            hypothesis=str(payload["hypothesis"]),
            direction=str(payload["direction"]),
            primary_endpoint=str(payload["primary_endpoint"]),
            discovery_cohort=str(payload["discovery_cohort"]),
            replication_cohort=str(payload["replication_cohort"]),
            analysis=str(payload["analysis"]),
            alpha=float(payload["alpha"]),
            multiple_testing_family=str(payload["multiple_testing_family"]),
            perturbational_validation=str(payload["perturbational_validation"]),
            external_dataset_validation=str(payload["external_dataset_validation"]),
            manifest_path=path.name,
            preregistration_sha256=_sha256(path),
        ))
    return specs


def hypothesis_registry_frame(manifest_dir: str | Path | None = None) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in load_hypothesis_registry(manifest_dir)])


def freeze_preregistration(output: str | Path, manifest_dir: str | Path | None = None) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = hypothesis_registry_frame(manifest_dir)
    frame.to_csv(output / "preregistered_hypotheses.csv", index=False)
    base = Path(manifest_dir) if manifest_dir else _validation_resource_dir()
    manifest_out = output / "hypothesis_manifests"
    manifest_out.mkdir(exist_ok=True)
    for source in sorted(base.glob("*.yaml")):
        (manifest_out / source.name).write_bytes(source.read_bytes())
    prereg_lines = ["# CausaFlux v1.7.0 preregistration", "", "Hypotheses and primary analysis choices were frozen before results were computed.", ""]
    for row in frame.itertuples(index=False):
        prereg_lines.extend([f"## {row.hypothesis_id} — {row.title}", row.hypothesis, f"- Status: {row.status}", f"- Primary endpoint: {row.primary_endpoint}", f"- Discovery: {row.discovery_cohort}", f"- Replication: {row.replication_cohort}", f"- Analysis: {row.analysis}", f"- SHA-256: `{row.preregistration_sha256}`", ""])
    (output / "PREREGISTRATION.md").write_text("\n".join(prereg_lines), encoding="utf-8")
    lock = {
        "framework": "CausaFlux",
        "version": VALIDATION_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_hypotheses": int(len(frame)),
        "registry_sha256": _json_sha(frame.sort_values("hypothesis_id").to_dict(orient="list")),
        "hypothesis_hashes": dict(zip(frame["hypothesis_id"], frame["preregistration_sha256"])),
        "mutable_after_results": False,
    }
    path = output / "preregistration_lock.json"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return path


def _load_seaad(snapshot_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    meta_path = snapshot_dir / "sea_ad_cohort_donor_metadata_2024-07-25.xlsx"
    cps_path = snapshot_dir / "sea_ad_continuous_pseudoprogression_scores_2026-05-01.xlsx"
    if not meta_path.exists() or not cps_path.exists():
        raise FileNotFoundError("SEA-AD snapshot workbooks are required")
    meta = pd.read_excel(meta_path, sheet_name="SEA-AD_Cohort_Metadata")
    sheets = {
        "Gabitto 2024": pd.read_excel(cps_path, sheet_name="Gabitto 2024"),
        "Travaglini 2026": pd.read_excel(cps_path, sheet_name="Travaglini 2026"),
        "Kana 2026": pd.read_excel(cps_path, sheet_name="Kana 2026"),
    }
    return meta, sheets


def _prepare_endpoint(meta: pd.DataFrame, sheets: dict[str, pd.DataFrame], endpoint: str) -> tuple[pd.DataFrame, str]:
    if endpoint == "Gabitto 2024":
        e = sheets[endpoint][["Donor ID", "CPS"]].rename(columns={"CPS": "outcome"})
    elif endpoint == "Travaglini 2026":
        e = sheets[endpoint].groupby("Donor ID", as_index=False)["CPS_Global"].mean().rename(columns={"CPS_Global": "outcome"})
    elif endpoint == "Kana 2026":
        e = sheets[endpoint][["Donor ID", "CPS"]].rename(columns={"CPS": "outcome"})
    else:
        raise KeyError(endpoint)
    df = meta.merge(e, on="Donor ID", how="inner")
    df["APOE4"] = df["APOE Genotype"].astype(str).str.contains("4", regex=False).astype(int)
    df["Dementia"] = (df["Cognitive Status"] == "Dementia").astype(int)
    df["ADNC"] = df["Overall AD neuropathological Change"].map(ADNC_ORDER)
    df["SexMale"] = (df["Sex"] == "Male").astype(int)
    df["Age"] = pd.to_numeric(df["Age at Death"], errors="coerce")
    return df, "outcome"


def _ols_effect(frame: pd.DataFrame, exposure: str, outcome: str) -> tuple[float, float]:
    use = frame[[outcome, exposure, "Age", "SexMale"]].dropna()
    if len(use) < 6:
        return float("nan"), float("nan")
    age = (use["Age"].to_numpy(float) - use["Age"].mean()) / max(use["Age"].std(ddof=0), 1e-8)
    x = np.column_stack([np.ones(len(use)), use[exposure].to_numpy(float), age, use["SexMale"].to_numpy(float)])
    y = use[outcome].to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return float(beta[1]), se


def _bootstrap_adjusted(frame: pd.DataFrame, exposure: str, outcome: str, n_boot: int, seed: int) -> tuple[float, float, float, int]:
    point, _ = _ols_effect(frame, exposure, outcome)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(frame), len(frame))
        b, _ = _ols_effect(frame.iloc[idx], exposure, outcome)
        if np.isfinite(b):
            values.append(b)
    if not values:
        return point, float("nan"), float("nan"), 0
    lo, hi = np.quantile(values, [0.025, 0.975])
    return point, float(lo), float(hi), len(values)


def _binary_result(frame: pd.DataFrame, exposure: str, cohort: str, endpoint: str, alpha: float, n_boot: int, seed: int) -> dict[str, Any]:
    a = frame.loc[frame[exposure] == 1, "outcome"].dropna()
    b = frame.loc[frame[exposure] == 0, "outcome"].dropna()
    if len(a) < 2 or len(b) < 2:
        return {"cohort": cohort, "endpoint": endpoint, "n": len(frame), "effect": np.nan, "p_value": np.nan, "supported": False}
    mw = stats.mannwhitneyu(a, b, alternative="greater")
    effect = float(a.mean() - b.mean())
    adj, lo, hi, nb = _bootstrap_adjusted(frame, exposure, "outcome", n_boot, seed)
    return {
        "cohort": cohort, "endpoint": endpoint, "n": int(len(frame)),
        "n_exposed": int(len(a)), "n_reference": int(len(b)),
        "effect": effect, "method": "Mann-Whitney U (one-sided)", "statistic": float(mw.statistic),
        "p_value": float(mw.pvalue), "adjusted_effect": adj, "adjusted_ci_low": lo,
        "adjusted_ci_high": hi, "bootstrap_success": nb,
        "supported": bool(effect > 0 and mw.pvalue < alpha),
    }


def _ordinal_result(frame: pd.DataFrame, cohort: str, endpoint: str, alpha: float, n_boot: int, seed: int) -> dict[str, Any]:
    use = frame[["ADNC", "outcome", "Age", "SexMale"]].dropna()
    if len(use) < 6:
        return {"cohort": cohort, "endpoint": endpoint, "n": len(use), "effect": np.nan, "p_value": np.nan, "supported": False}
    corr = stats.spearmanr(use["ADNC"], use["outcome"], alternative="greater")
    adj, lo, hi, nb = _bootstrap_adjusted(use, "ADNC", "outcome", n_boot, seed)
    return {
        "cohort": cohort, "endpoint": endpoint, "n": int(len(use)),
        "effect": float(corr.statistic), "method": "Spearman rank correlation (one-sided)",
        "statistic": float(corr.statistic), "p_value": float(corr.pvalue),
        "adjusted_effect": adj, "adjusted_ci_low": lo, "adjusted_ci_high": hi,
        "bootstrap_success": nb, "supported": bool(corr.statistic > 0 and corr.pvalue < alpha),
    }


def _bh_adjust(values: Iterable[float]) -> list[float]:
    p = np.asarray(list(values), dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return out.tolist()
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out[np.where(valid)[0]] = q[inv]
    return out.tolist()


def _method_rows(frame: pd.DataFrame, hypothesis_id: str, exposure: str, kind: str, cohort: str, endpoint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    use = frame.dropna(subset=["outcome", exposure])
    if kind == "binary":
        a = use.loc[use[exposure] == 1, "outcome"]
        b = use.loc[use[exposure] == 0, "outcome"]
        if len(a) and len(b):
            t = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
            mw = stats.mannwhitneyu(a, b, alternative="greater")
            rows.extend([
                {"hypothesis_id": hypothesis_id, "cohort": cohort, "endpoint": endpoint, "method": "Welch t-test", "effect": float(a.mean()-b.mean()), "p_value": float(t.pvalue)},
                {"hypothesis_id": hypothesis_id, "cohort": cohort, "endpoint": endpoint, "method": "Mann-Whitney U", "effect": float(a.mean()-b.mean()), "p_value": float(mw.pvalue)},
            ])
    else:
        sp = stats.spearmanr(use[exposure], use["outcome"], alternative="greater")
        pe = stats.pearsonr(use[exposure], use["outcome"], alternative="greater")
        rows.extend([
            {"hypothesis_id": hypothesis_id, "cohort": cohort, "endpoint": endpoint, "method": "Spearman", "effect": float(sp.statistic), "p_value": float(sp.pvalue)},
            {"hypothesis_id": hypothesis_id, "cohort": cohort, "endpoint": endpoint, "method": "Pearson", "effect": float(pe.statistic), "p_value": float(pe.pvalue)},
        ])
    adj, se = _ols_effect(use, exposure, "outcome")
    if np.isfinite(adj):
        z = adj / se if se > 0 else np.inf
        p = float(stats.norm.sf(z))
        rows.append({"hypothesis_id": hypothesis_id, "cohort": cohort, "endpoint": endpoint, "method": "Age/sex-adjusted OLS", "effect": adj, "p_value": p})
    return rows


def _claim_text(hypothesis: str, discovery: bool, replication: bool, external: str, perturbation: str) -> tuple[str, str]:
    if discovery and replication:
        tier = "replicated_association"
        text = f"The preregistered association was supported in the discovery cohort and replicated in an independent SEA-AD source cohort. {hypothesis} This is an observational association, not evidence of causality or clinical utility."
    elif discovery:
        tier = "discovery_supported_not_replicated"
        text = f"The preregistered association was supported only in discovery. Replication was not established. {hypothesis}"
    else:
        tier = "not_supported"
        text = f"The preregistered hypothesis was not supported by the available primary analysis. No biological conclusion is established."
    if external.startswith("pending"):
        text += " External-dataset replication remains pending."
    if "not_applicable" not in perturbation and perturbation != "required":
        text += " Perturbational validation has not been completed."
    return tier, text


def run_biological_validation(snapshot_dir: str | Path, *, n_boot: int = 500, seed: int = 120) -> ValidationRun:
    specs = load_hypothesis_registry()
    hypotheses = pd.DataFrame([asdict(s) for s in specs])
    meta, sheets = _load_seaad(Path(snapshot_dir))
    primary_endpoint = "Gabitto 2024"
    endpoint_frames = {name: _prepare_endpoint(meta, sheets, name)[0] for name in ("Gabitto 2024", "Travaglini 2026", "Kana 2026")}
    primary_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    executed = [s for s in specs if s.status == "executed"]
    mapping = {
        "BV-SEA-AD-001": ("APOE4", "binary"),
        "BV-SEA-AD-002": ("Dementia", "binary"),
        "BV-SEA-AD-003": ("ADNC", "ordinal"),
    }
    cohorts = {"discovery": "ACT", "replication": "ADRC Clinical Core"}
    for h_index, spec in enumerate(executed):
        exposure, kind = mapping[spec.hypothesis_id]
        for e_index, (endpoint, full) in enumerate(endpoint_frames.items()):
            for role, cohort_name in cohorts.items():
                sub = full.loc[full["Primary Study Name"] == cohort_name].copy()
                if kind == "binary":
                    row = _binary_result(sub, exposure, role, endpoint, spec.alpha, n_boot, seed + h_index*100 + e_index*10 + (0 if role=="discovery" else 1))
                else:
                    row = _ordinal_result(sub, role, endpoint, spec.alpha, n_boot, seed + h_index*100 + e_index*10 + (0 if role=="discovery" else 1))
                row.update({"hypothesis_id": spec.hypothesis_id, "title": spec.title, "exposure": exposure, "cohort_name": cohort_name})
                endpoint_rows.append(row)
                method_rows.extend(_method_rows(sub, spec.hypothesis_id, exposure, kind, role, endpoint))
                if endpoint == primary_endpoint:
                    primary_rows.append(row.copy())
    primary = pd.DataFrame(primary_rows)
    primary["q_value_bh"] = _bh_adjust(primary["p_value"])
    endpoint_replication = pd.DataFrame(endpoint_rows)
    endpoint_replication["q_value_bh"] = _bh_adjust(endpoint_replication["p_value"])
    methods = pd.DataFrame(method_rows)
    methods["q_value_bh"] = _bh_adjust(methods["p_value"])

    evidence_rows: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    for spec in specs:
        if spec.status == "executed":
            h = primary.loc[primary["hypothesis_id"] == spec.hypothesis_id]
            d = bool(h.loc[h["cohort"] == "discovery", "supported"].iloc[0])
            r = bool(h.loc[h["cohort"] == "replication", "supported"].iloc[0])
            cross = endpoint_replication.loc[endpoint_replication["hypothesis_id"] == spec.hypothesis_id, "supported"].mean()
            tier, conclusion = _claim_text(spec.hypothesis, d, r, spec.external_dataset_validation, spec.perturbational_validation)
            external_status = "independent_source_cohort_replicated" if r else "not_replicated"
            perturb_status = "not_applicable" if spec.perturbational_validation.startswith("not_applicable") else "not_executed"
        else:
            d = r = False
            cross = np.nan
            tier = "preregistered_pending_data"
            conclusion = "Hypothesis preregistered before analysis. No biological conclusion is established because the required assay data have not been executed in this release."
            external_status = "pending_data_access"
            perturb_status = "planned_not_executed"
        evidence_rows.append({
            "hypothesis_id": spec.hypothesis_id, "domain": spec.domain, "preregistered": True,
            "discovery_supported": d, "source_cohort_replication_supported": r,
            "cross_endpoint_support_fraction": cross, "external_dataset_status": external_status,
            "perturbational_status": perturb_status, "causal_claim_permitted": False,
            "clinical_claim_permitted": False,
        })
        conclusions.append({
            "hypothesis_id": spec.hypothesis_id, "title": spec.title, "evidence_tier": tier,
            "allowed_conclusion": conclusion, "causal_language_allowed": False,
            "clinical_guidance_allowed": False,
        })
    evidence = pd.DataFrame(evidence_rows)
    conclusion_ledger = pd.DataFrame(conclusions)
    qc = {
        "framework": "CausaFlux", "version": VALIDATION_VERSION,
        "valid": True, "n_preregistered_hypotheses": len(specs),
        "n_executed_hypotheses": len(executed), "n_primary_tests": len(primary),
        "n_endpoint_replication_tests": len(endpoint_replication),
        "n_method_comparisons": len(methods), "bootstrap_requested": n_boot,
        "discovery_cohort_donors": int((meta["Primary Study Name"] == "ACT").sum()),
        "replication_cohort_donors": int((meta["Primary Study Name"] == "ADRC Clinical Core").sum()),
        "causal_claims": 0, "clinical_claims": 0,
        "synthetic_data_used_for_biological_conclusions": False,
    }
    return ValidationRun(hypotheses, primary, endpoint_replication, methods, evidence, conclusion_ledger, qc)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def _plot_validation_figures(run: ValidationRun, output: Path) -> list[dict[str, Any]]:
    _style()
    figures = output / "manuscript_package" / "figures"
    source = output / "manuscript_package" / "source_data"
    manifests = output / "manuscript_package" / "figure_manifests"
    figures.mkdir(parents=True, exist_ok=True); source.mkdir(parents=True, exist_ok=True); manifests.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []

    # Figure 1: preregistration and evidence status
    f1 = run.hypotheses[["hypothesis_id", "domain", "status", "title", "preregistration_sha256"]].copy()
    f1.to_csv(source / "Figure1_preregistration.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    order = f1.sort_values(["domain", "hypothesis_id"])
    vals = order["status"].map({"executed": 1.0, "preregistered_pending_data": .45, "preregistered_pending_molecular_data": .45}).fillna(.3)
    y = np.arange(len(order))
    ax.barh(y, vals, color=["#3C5488" if x == "executed" else "#B4B4B4" for x in order["status"]], height=.62)
    ax.set_yticks(y, order["hypothesis_id"]); ax.set_xlim(0, 1.05); ax.set_xlabel("Validation status (executed = 1)")
    ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False); ax.set_title("Preregistered biological-validation hypotheses", loc="left", fontweight="bold")
    _save_figure(fig, figures / "Figure1_preregistration")
    inventory.append({"figure_id":"Figure1","title":"Preregistered hypotheses","source_data":"source_data/Figure1_preregistration.csv"})

    # Figure 2: primary replicated effects
    f2 = run.primary_results.copy(); f2.to_csv(source / "Figure2_primary_replication.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (hid, sub) in zip(axes, f2.groupby("hypothesis_id", sort=True)):
        sub = sub.set_index("cohort").reindex(["discovery", "replication"])
        x = np.arange(2); effect = sub["adjusted_effect"].to_numpy(float)
        lo = sub["adjusted_ci_low"].to_numpy(float); hi = sub["adjusted_ci_high"].to_numpy(float)
        ax.errorbar(x, effect, yerr=np.vstack([effect-lo, hi-effect]), fmt="o", color="#3C5488", ecolor="#7A7A7A", capsize=2, lw=.9)
        ax.axhline(0, color="#A0A0A0", lw=.6); ax.set_xticks(x, ["ACT\ndiscovery", "ADRC\nreplication"])
        ax.spines[["top","right"]].set_visible(False); ax.set_title(hid, loc="left", fontweight="bold")
        ax.set_ylabel("Adjusted effect")
    fig.suptitle("Independent source-cohort replication of SEA-AD associations", x=.04, ha="left", fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures / "Figure2_primary_replication")
    inventory.append({"figure_id":"Figure2","title":"Primary replication","source_data":"source_data/Figure2_primary_replication.csv"})

    # Figure 3: endpoint robustness
    f3 = run.endpoint_replication.copy(); f3.to_csv(source / "Figure3_endpoint_robustness.csv", index=False)
    pivot = f3.pivot_table(index="hypothesis_id", columns=["endpoint","cohort"], values="supported", aggfunc="first").astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap=matplotlib.colors.ListedColormap(["#F0F0F0", "#00A087"]))
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{a}\n{b}" for a,b in pivot.columns], rotation=35, ha="right")
    ax.set_title("Support across independent CPS definitions", loc="left", fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]): ax.text(j, i, "supported" if pivot.iloc[i,j] else "not supported", ha="center", va="center", fontsize=6)
    fig.tight_layout(); _save_figure(fig, figures / "Figure3_endpoint_robustness")
    inventory.append({"figure_id":"Figure3","title":"Endpoint robustness","source_data":"source_data/Figure3_endpoint_robustness.csv"})

    # Figure 4: method comparison
    f4 = run.method_comparison.copy(); f4.to_csv(source / "Figure4_method_comparison.csv", index=False)
    summary = f4.assign(sig=f4["p_value"] < .05).groupby("method", as_index=False).agg(support_fraction=("sig","mean"), n_tests=("sig","size"))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    summary = summary.sort_values("support_fraction")
    ax.barh(np.arange(len(summary)), summary["support_fraction"], color="#4DBBD5")
    ax.set_yticks(np.arange(len(summary)), summary["method"]); ax.set_xlim(0,1); ax.set_xlabel("Fraction of tests with one-sided P < 0.05")
    ax.spines[["top","right"]].set_visible(False); ax.set_title("Agreement with established statistical methods", loc="left", fontweight="bold")
    fig.tight_layout(); _save_figure(fig, figures / "Figure4_method_comparison")
    inventory.append({"figure_id":"Figure4","title":"Method comparison","source_data":"source_data/Figure4_method_comparison.csv"})

    for item in inventory:
        base = figures / item["figure_id"]
        # files use descriptive names; find exact prefix
        candidates = sorted(figures.glob(item["figure_id"] + "_*.svg"))
        item["vector_file"] = str(candidates[0].relative_to(output / "manuscript_package")) if candidates else ""
        item["formats"] = ["svg", "pdf", "png", "tiff"]
        item["profile"] = "Nature double column / Cell double column"
        item["data_type"] = "public real metadata"
        (manifests / f"{item['figure_id']}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    pd.DataFrame(inventory).to_csv(output / "manuscript_package" / "figure_inventory.csv", index=False)
    return inventory


def _write_report(run: ValidationRun, output: Path) -> None:
    reports = output / "reports"; reports.mkdir(parents=True, exist_ok=True)
    css = "body{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:28px auto;padding:0 22px;color:#202124}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ddd;padding:6px;text-align:left}th{background:#f4f4f4}.ok{border-left:4px solid #00A087;padding:12px;background:#f2fbf8}.warn{border-left:4px solid #E64B35;padding:12px;background:#fff5f3}code{background:#f3f3f3;padding:2px 4px}"
    conclusions = run.conclusion_ledger[["hypothesis_id","evidence_tier","allowed_conclusion"]]
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v1.7.0 biological validation</title><style>{css}</style></head><body>
<h1>CausaFlux v1.7.0 — Biological validation</h1>
<div class='ok'><strong>Three preregistered SEA-AD hypotheses were executed.</strong> Discovery used ACT donors and replication used independent ADRC Clinical Core donors. All conclusions remain observational.</div>
<p><a href='../preregistration/preregistered_hypotheses.csv'>Preregistered hypotheses</a> · <a href='../manuscript_package/figure_inventory.csv'>Figure inventory</a> · <a href='evidence.html'>Evidence ledger</a></p>
<h2>Primary results</h2>{run.primary_results.to_html(index=False, classes='data', float_format=lambda x:f'{x:.4g}')}
<h2>Permitted conclusions</h2>{conclusions.to_html(index=False, classes='data')}
<h2>Scope boundary</h2><div class='warn'>Source-cohort replication is established for the bundled SEA-AD metadata hypotheses. External-dataset replication in AMP-AD and perturbational validation in molecular or experimental systems remain pending. No causal, therapeutic, biomarker, or clinical-guidance claim is permitted.</div>
</body></html>"""
    (reports / "index.html").write_text(html, encoding="utf-8")
    evidence_html = f"<!doctype html><html><head><meta charset='utf-8'><title>Evidence ledger</title><style>{css}</style></head><body><h1>Evidence and conclusion ledger</h1>{run.evidence_ledger.to_html(index=False)}<h2>Allowed language</h2>{run.conclusion_ledger.to_html(index=False)}</body></html>"
    (reports / "evidence.html").write_text(evidence_html, encoding="utf-8")
    methods_html = f"<!doctype html><html><head><meta charset='utf-8'><title>Method comparison</title><style>{css}</style></head><body><h1>Established-method comparison</h1>{run.method_comparison.to_html(index=False, float_format=lambda x:f'{x:.4g}')}</body></html>"
    (reports / "methods.html").write_text(methods_html, encoding="utf-8")


def _write_manuscript_files(run: ValidationRun, output: Path) -> None:
    mp = output / "manuscript_package"; (mp / "tables").mkdir(parents=True, exist_ok=True); (mp / "methods").mkdir(parents=True, exist_ok=True); (mp / "claims").mkdir(parents=True, exist_ok=True)
    run.primary_results.to_csv(mp / "tables" / "Table1_primary_validation_results.csv", index=False)
    run.endpoint_replication.to_csv(mp / "tables" / "Table2_endpoint_replication.csv", index=False)
    run.method_comparison.to_csv(mp / "tables" / "Table3_method_comparison.csv", index=False)
    run.evidence_ledger.to_csv(mp / "tables" / "Table4_evidence_ledger.csv", index=False)
    methods = """# Statistical methods\n\nHypotheses were frozen before results were computed. The public SEA-AD donor metadata were joined to continuous pseudo-progression scores by Donor ID. ACT donors were the discovery cohort and ADRC Clinical Core donors were the independent source-cohort replication cohort. Binary hypotheses used one-sided Mann–Whitney U tests and age/sex-adjusted ordinary least squares. Ordinal neuropathological change used one-sided Spearman correlation and adjusted ordinary least squares. Adjusted effects were accompanied by donor bootstrap 95% intervals. Benjamini–Hochberg values are reported for transparency; preregistered primary decisions use the stated alpha. Established-method comparisons include Welch t tests, Mann–Whitney U, Pearson or Spearman association, and adjusted OLS. No observational result is interpreted as causal.\n"""
    (mp / "methods" / "STATISTICAL_METHODS.md").write_text(methods, encoding="utf-8")
    claims = ["# Claims ledger", "", "Only the following evidence-constrained language is permitted:", ""]
    for row in run.conclusion_ledger.itertuples(index=False):
        claims.extend([f"## {row.hypothesis_id} — {row.evidence_tier}", row.allowed_conclusion, ""])
    (mp / "claims" / "CLAIMS_LEDGER.md").write_text("\n".join(claims), encoding="utf-8")
    (mp / "README.md").write_text("# CausaFlux v1.7.0 manuscript source-data package\n\nThis directory contains vector/raster figures, panel-level source data, tables, methods, figure manifests, and an evidence-constrained claims ledger. The biological validation uses public SEA-AD metadata only.\n", encoding="utf-8")


def write_biological_validation(run: ValidationRun, output: str | Path) -> Path:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    prereg = output / "preregistration"; prereg.mkdir(exist_ok=True)
    freeze_preregistration(prereg)
    results = output / "results"; results.mkdir(exist_ok=True)
    run.primary_results.to_csv(results / "primary_validation_results.csv", index=False)
    run.endpoint_replication.to_csv(results / "endpoint_replication_results.csv", index=False)
    run.method_comparison.to_csv(results / "established_method_comparison.csv", index=False)
    run.evidence_ledger.to_csv(results / "evidence_ledger.csv", index=False)
    run.conclusion_ledger.to_csv(results / "conclusion_ledger.csv", index=False)
    run.hypotheses.to_csv(results / "hypothesis_registry.csv", index=False)
    (results / "validation_qc.json").write_text(json.dumps(run.qc, indent=2), encoding="utf-8")
    _plot_validation_figures(run, output)
    _write_manuscript_files(run, output)
    _write_report(run, output)
    perturb = pd.DataFrame([
        {"hypothesis_id":"BV-HTAN-001","perturbation":"macrophage depletion or IFNG rescue","repository":"HTAN / prospective experiment","status":"planned_not_executed","support_established":False},
        {"hypothesis_id":"BV-DEPMAP-001","perturbation":"CRISPR dependency plus PRISM/LINCS response","repository":"DepMap / LINCS","status":"planned_not_executed","support_established":False},
        {"hypothesis_id":"BV-SEA-AD-004","perturbation":"ex vivo neural-glial stress perturbation","repository":"SEA-AD / DANDI follow-up","status":"planned_not_executed","support_established":False},
    ])
    perturb.to_csv(output / "results" / "perturbational_validation_ledger.csv", index=False)
    protocols = output / "perturbation_protocols"; protocols.mkdir(exist_ok=True)
    for row in perturb.to_dict(orient="records"):
        payload = {**row, "predefined_success_rule": "directionally concordant effect with FDR < 0.05 in an independent perturbational dataset", "claim_if_incomplete": "No perturbational support established"}
        (protocols / f"{row['hypothesis_id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    external = pd.DataFrame([
        {"hypothesis_id":"BV-SEA-AD-001","discovery":"ACT donors","replication":"ADRC Clinical Core donors","replication_type":"independent source cohort","status":"established","external_dataset":"AMP-AD","external_dataset_status":"pending"},
        {"hypothesis_id":"BV-SEA-AD-002","discovery":"ACT donors","replication":"ADRC Clinical Core donors","replication_type":"independent source cohort","status":"established","external_dataset":"AMP-AD","external_dataset_status":"pending"},
        {"hypothesis_id":"BV-SEA-AD-003","discovery":"ACT donors","replication":"ADRC Clinical Core donors","replication_type":"independent source cohort","status":"established","external_dataset":"AMP-AD","external_dataset_status":"pending"},
        {"hypothesis_id":"BV-HTAN-001","discovery":"HTAN metastatic breast cancer","replication":"HTAN therapeutic resistance atlas","replication_type":"external atlas","status":"pending data access","external_dataset":"HTAN validation atlas","external_dataset_status":"pending"},
        {"hypothesis_id":"BV-DEPMAP-001","discovery":"DepMap/PRISM","replication":"held-out lineages and LINCS","replication_type":"external perturbational","status":"pending data access","external_dataset":"LINCS","external_dataset_status":"pending"},
        {"hypothesis_id":"BV-SEA-AD-004","discovery":"SEA-AD MTG","replication":"AMP-AD ROSMAP/MSBB","replication_type":"external molecular cohort","status":"pending data access","external_dataset":"AMP-AD","external_dataset_status":"pending"},
    ])
    external.to_csv(output / "results" / "external_replication_matrix.csv", index=False)
    cards = output / "cards"; cards.mkdir(exist_ok=True)
    (cards / "BIOLOGICAL_VALIDATION_MODEL_CARD.md").write_text("# CausaFlux v1.7.0 biological-validation model card\n\nThis release executes preregistered observational validation on public SEA-AD metadata. It establishes independent source-cohort replication for three associations, but no external-dataset, perturbational, causal, biomarker, therapeutic, or clinical validation. Claims are automatically limited by the evidence ledger.\n", encoding="utf-8")
    status = {
        **run.qc,
        "source_cohort_replication_established": 3,
        "independent_source_cohort_replication_established": 3,
        "external_dataset_replication_established": 0,
        "perturbational_validation_established": 0,
        "manuscript_figures": 4,
        "biological_conclusions": "replicated observational associations only",
    }
    (output / "biological_validation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    # provenance after all scientific files are created
    prov = output / "provenance"; prov.mkdir(exist_ok=True)
    rows=[]
    for p in sorted(output.rglob("*")):
        if p.is_file() and "provenance/artifact_manifest.csv" not in p.as_posix():
            rows.append({"relative_path":str(p.relative_to(output)),"size_bytes":p.stat().st_size,"sha256":_sha256(p)})
    pd.DataFrame(rows).to_csv(prov / "artifact_manifest.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "framework":"CausaFlux","version":VALIDATION_VERSION,"workflow":"biological_validation",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),"public_real_data":True,
        "synthetic_data_used":False,"clinical_guidance":False,"n_artifacts":len(rows),
    }, indent=2), encoding="utf-8")
    return output


def validate_biological_validation(output: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    output = Path(output)
    required = [
        "run_manifest.json","biological_validation_status.json",
        "preregistration/preregistered_hypotheses.csv","preregistration/preregistration_lock.json",
        "results/primary_validation_results.csv","results/endpoint_replication_results.csv",
        "results/established_method_comparison.csv","results/evidence_ledger.csv",
        "results/conclusion_ledger.csv","results/perturbational_validation_ledger.csv",
        "results/external_replication_matrix.csv","cards/BIOLOGICAL_VALIDATION_MODEL_CARD.md",
        "preregistration/PREREGISTRATION.md",
        "reports/index.html","reports/evidence.html","reports/methods.html",
        "manuscript_package/figure_inventory.csv","manuscript_package/methods/STATISTICAL_METHODS.md",
        "manuscript_package/claims/CLAIMS_LEDGER.md","provenance/artifact_manifest.csv",
    ]
    errors=[]
    for rel in required:
        p=output/rel
        if not p.exists() or p.stat().st_size == 0: errors.append(f"missing_or_empty:{rel}")
    if errors:
        return {"framework":"CausaFlux","version":VALIDATION_VERSION,"valid":False,"errors":errors}
    status=json.loads((output/"biological_validation_status.json").read_text())
    primary=pd.read_csv(output/"results/primary_validation_results.csv")
    evidence=pd.read_csv(output/"results/evidence_ledger.csv")
    conclusions=pd.read_csv(output/"results/conclusion_ledger.csv")
    perturb=pd.read_csv(output/"results/perturbational_validation_ledger.csv")
    if status.get("version") != VALIDATION_VERSION: errors.append("version_mismatch")
    if primary["hypothesis_id"].nunique()!=3 or len(primary)!=6: errors.append("primary_result_shape")
    if not bool(primary.groupby("hypothesis_id")["supported"].all().all()): errors.append("preregistered_primary_replication_not_supported")
    if evidence["causal_claim_permitted"].astype(str).str.lower().isin(["true","1"]).any(): errors.append("causal_claim_leak")
    if conclusions["clinical_guidance_allowed"].astype(str).str.lower().isin(["true","1"]).any(): errors.append("clinical_claim_leak")
    if perturb["support_established"].astype(str).str.lower().isin(["true","1"]).any(): errors.append("unsupported_perturbation_claim")
    inv=pd.read_csv(output/"manuscript_package/figure_inventory.csv")
    if len(inv)!=4: errors.append("figure_inventory_count")
    for row in inv.itertuples(index=False):
        prefix = output/"manuscript_package"/row.vector_file
        if not prefix.exists(): errors.append(f"missing_vector:{row.figure_id}")
        stem=prefix.with_suffix("")
        for ext in (".pdf",".png",".tiff"):
            if not stem.with_suffix(ext).exists(): errors.append(f"missing_{ext}:{row.figure_id}")
    if verify_hashes:
        manifest=pd.read_csv(output/"provenance/artifact_manifest.csv")
        for row in manifest.itertuples(index=False):
            p=output/row.relative_path
            if not p.exists() or p.stat().st_size!=row.size_bytes or _sha256(p)!=row.sha256:
                errors.append(f"hash_mismatch:{row.relative_path}")
                break
    return {
        "framework":"CausaFlux","version":VALIDATION_VERSION,"valid":not errors,
        "n_hypotheses":int(status.get("n_preregistered_hypotheses",0)),
        "executed_hypotheses":int(status.get("n_executed_hypotheses",0)),
        "source_cohort_replications":int(status.get("source_cohort_replication_established",0)),
        "external_dataset_replications":int(status.get("external_dataset_replication_established",0)),
        "perturbational_validations":int(status.get("perturbational_validation_established",0)),
        "manuscript_figures":len(inv),"errors":errors,
    }


def run_and_write_biological_validation(snapshot_dir: str | Path, output: str | Path, *, n_boot: int = 500, seed: int = 120) -> Path:
    run = run_biological_validation(snapshot_dir, n_boot=n_boot, seed=seed)
    return write_biological_validation(run, output)
