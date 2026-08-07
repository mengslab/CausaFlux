"""Publication-grade virtual-cell figures and integrated HTML reporting."""
from __future__ import annotations

from pathlib import Path
import html
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .visualization.publication import (
    COLORS,
    EXPORT_PROFILES,
    apply_publication_style,
    export_figure,
)


def _close_export(fig, output: Path, figure_id: str, profile: str, source_data, metadata=None, synthetic_only=True):
    fig.tight_layout(pad=0.65)
    result = export_figure(
        fig, output, figure_id=figure_id, profile=profile, source_data=source_data,
        metadata=metadata or {}, synthetic_only=synthetic_only,
    )
    plt.close(fig)
    return result


def _figure1(trajectories: pd.DataFrame, ranking: pd.DataFrame, figures: Path):
    apply_publication_style("nature_double")
    top = str(ranking.iloc[0].scenario_id)
    selected = trajectories[trajectories.scenario_id == top]
    baseline = trajectories[trajectories.scenario_id == "stress_only"]
    fig, axes = plt.subplots(1, 3, figsize=(EXPORT_PROFILES["nature_double"]["width_mm"] / 25.4, EXPORT_PROFILES["nature_double"]["height_mm"] / 25.4))
    ax = axes[0]
    for data, label, ls in [(baseline, "Stress only", "--"), (selected, str(ranking.iloc[0].scenario_label), "-")]:
        ax.plot(data.time_hours, data.recovery_potential_mean, label=label, linestyle=ls, linewidth=1.6)
        if label != "Stress only":
            ax.fill_between(data.time_hours, data.recovery_potential_p05, data.recovery_potential_p95, alpha=0.16, linewidth=0)
    ax.set(title="Recovery trajectory", xlabel="Time (h)", ylabel="Recovery potential", ylim=(0, 1)); ax.legend(fontsize=6)
    ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    names = ["proteostasis_capacity", "mitochondrial_reserve", "inflammatory_dysfunction", "commitment_risk"]
    labels = ["Proteostasis", "Mitochondria", "Inflammation", "Commitment"]
    for name, label in zip(names, labels):
        ax.plot(selected.time_hours, selected[f"{name}_mean"], label=label, linewidth=1.4)
    ax.set(title="Selected virtual-cell state", xlabel="Time (h)", ylabel="State score", ylim=(0, 1)); ax.legend(fontsize=5.7, ncol=2)
    ax.spines[['top','right']].set_visible(False)

    ax = axes[2]
    final = selected.iloc[-1]
    state_labels = ["Proteostasis", "Mito reserve", "Inflammation", "Commitment", "Recovery"]
    means = [final[f"{n}_mean"] for n in ["proteostasis_capacity","mitochondrial_reserve","inflammatory_dysfunction","commitment_risk","recovery_potential"]]
    sds = [final[f"{n}_sd"] for n in ["proteostasis_capacity","mitochondrial_reserve","inflammatory_dysfunction","commitment_risk","recovery_potential"]]
    y = np.arange(len(state_labels))
    ax.barh(y, means, xerr=np.asarray(sds)*1.645, height=.62, capsize=2)
    ax.set_yticks(y, state_labels); ax.invert_yaxis(); ax.set(xlabel="Predicted state ± 90% interval", xlim=(0,1), title="72-h state and uncertainty")
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle("CausaFlux virtual cell: dynamic response and uncertainty", x=0.02, ha="left", fontsize=10, fontweight="semibold")
    return _close_export(fig, figures / "Figure1_virtual_cell_trajectory.png", "Figure1_virtual_cell_trajectory", "nature_double", {"trajectory": selected, "baseline": baseline, "ranking": ranking.head(1)}, {"selected_scenario": top})


def _figure2(ranking: pd.DataFrame, figures: Path):
    apply_publication_style("nature_double")
    data = ranking.sort_values("calibrated_utility", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(EXPORT_PROFILES["nature_double"]["width_mm"] / 25.4, EXPORT_PROFILES["nature_double"]["height_mm"] / 25.4))
    y = np.arange(len(data))
    axes[0].barh(y, data.calibrated_utility, height=.62)
    axes[0].set_yticks(y, [x.replace(" program", "") for x in data.scenario_label]); axes[0].set_xlabel("Calibrated biological utility"); axes[0].set_title("AI-guided intervention ranking", loc="left")
    axes[0].spines[['top','right']].set_visible(False)
    scatter = axes[1].scatter(data["cost"], data["final_recovery_potential"], s=30 + 500*data["mean_predictive_uncertainty"], c=data["calibrated_utility"], cmap="viridis")
    for row in data.itertuples():
        axes[1].annotate(str(row.rank), (row.cost, row.final_recovery_potential), xytext=(3,3), textcoords="offset points", fontsize=6)
    axes[1].set(xlabel="Relative experiment/intervention cost", ylabel="Predicted recovery potential", title="Utility–cost–uncertainty tradeoff")
    axes[1].spines[['top','right']].set_visible(False)
    cb=fig.colorbar(scatter, ax=axes[1], fraction=.05, pad=.03); cb.set_label("Calibrated utility")
    return _close_export(fig, figures / "Figure2_ai_intervention_ranking.png", "Figure2_ai_intervention_ranking", "nature_double", {"ranking": ranking})


def _figure3(router: pd.DataFrame, figures: Path):
    apply_publication_style("nature_double")
    data = router.sort_values("normalized_weight", ascending=True)
    fig, axes = plt.subplots(1,2, figsize=(EXPORT_PROFILES["nature_double"]["width_mm"] / 25.4, EXPORT_PROFILES["nature_double"]["height_mm"] / 25.4))
    y=np.arange(len(data)); axes[0].barh(y, data.normalized_weight, height=.62); axes[0].set_yticks(y, [x.replace('_',' ') for x in data.module]); axes[0].set_xlabel("Normalized ensemble weight"); axes[0].set_title("Validated module contribution", loc="left"); axes[0].spines[['top','right']].set_visible(False)
    metric = data.primary_value.to_numpy(float); metric_norm=(metric-metric.min())/(metric.max()-metric.min()+1e-9)
    x=np.arange(len(data)); axes[1].bar(x, data.reliability, width=.6); axes[1].scatter(x, 1-metric_norm, marker='D', s=18, label="Relative primary-metric quality")
    axes[1].set_xticks(x, [m.split('_')[0] for m in data.module], rotation=35, ha='right'); axes[1].set_ylim(0,1.05); axes[1].set_ylabel("Reliability / relative quality"); axes[1].set_title("Evidence-weighted model router", loc="left"); axes[1].legend(fontsize=5.5); axes[1].spines[['top','right']].set_visible(False)
    return _close_export(fig, figures / "Figure3_ai_model_router.png", "Figure3_ai_model_router", "nature_double", {"router": router})


def _figure4(calibration: pd.DataFrame, figures: Path):
    apply_publication_style("nature_double")
    fig, axes = plt.subplots(1,3, figsize=(EXPORT_PROFILES["nature_double"]["width_mm"] / 25.4, EXPORT_PROFILES["nature_double"]["height_mm"] / 25.4))
    axes[0].plot(calibration.cycle, calibration.prediction_rmse, marker='o'); axes[0].set(title="Prediction error", xlabel="Prospective cycle", ylabel="RMSE"); axes[0].spines[['top','right']].set_visible(False)
    axes[1].axhline(.90, linestyle='--', linewidth=.9, label="Target 90%") ; axes[1].plot(calibration.cycle, calibration.interval_coverage_90, marker='o'); axes[1].set(title="Uncertainty coverage", xlabel="Prospective cycle", ylabel="Observed coverage", ylim=(0,1.05)); axes[1].legend(fontsize=5.5); axes[1].spines[['top','right']].set_visible(False)
    axes[2].plot(calibration.cycle, calibration.brier_discovery, marker='o'); axes[2].set(title="Discovery calibration", xlabel="Prospective cycle", ylabel="Brier score"); axes[2].spines[['top','right']].set_visible(False)
    fig.suptitle("Cycle-to-cycle prospective calibration", x=.02, ha='left', fontsize=10, fontweight='semibold')
    return _close_export(fig, figures / "Figure4_prospective_calibration.png", "Figure4_prospective_calibration", "nature_double", {"calibration": calibration})


def _figure5(benchmarks: pd.DataFrame, evidence: pd.DataFrame, figures: Path):
    apply_publication_style("nature_double")
    fig, axes=plt.subplots(1,2, figsize=(EXPORT_PROFILES["nature_double"]["width_mm"] / 25.4, EXPORT_PROFILES["nature_double"]["height_mm"] / 25.4))
    data=benchmarks.sort_values('n_sources'); y=np.arange(len(data)); axes[0].barh(y,data.n_sources,height=.62,label='Registered sources'); axes[0].barh(y,data.n_validation_sources,height=.34,label='Validation sources'); axes[0].set_yticks(y,[x.replace('_',' ') for x in data.benchmark_id]); axes[0].set_xlabel('Source count'); axes[0].set_title('Real-world benchmark coverage',loc='left'); axes[0].legend(fontsize=5.5); axes[0].spines[['top','right']].set_visible(False)
    tiers=["registered_only","real_observational_replication","real_perturbational"]
    counts=evidence.evidence_tier.value_counts().reindex(tiers,fill_value=0); axes[1].bar(np.arange(len(tiers)),counts.values,width=.6); axes[1].set_xticks(np.arange(len(tiers)),['Registered','Observational\nreplication','Perturbational'],rotation=0); axes[1].set_ylabel('Hypotheses'); axes[1].set_title('Evidence maturity',loc='left'); axes[1].spines[['top','right']].set_visible(False)
    return _close_export(fig, figures / "Figure5_real_world_evidence.png", "Figure5_real_world_evidence", "nature_double", {"benchmarks":benchmarks,"evidence":evidence}, synthetic_only=False)


def _graphical_abstract(ranking: pd.DataFrame, status: dict, figures: Path):
    apply_publication_style("cell_square")
    spec=EXPORT_PROFILES['cell_square']; fig,ax=plt.subplots(figsize=(spec['width_mm']/25.4,spec['height_mm']/25.4)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    stages=[(.08,.72,.22,.14,'REAL-WORLD\nDATA'),(.39,.72,.22,.14,'MULTIMODAL\nAI'),(.70,.72,.22,.14,'VIRTUAL\nCELL'),(.70,.40,.22,.14,'AI-GUIDED\nEXPERIMENT'),(.39,.40,.22,.14,'LOCKED\nOUTCOME'),(.08,.40,.22,.14,'POSTERIOR\nUPDATE')]
    for x,y,w,h,label in stages:
        box=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.02',facecolor='#F5F7F8',edgecolor='#58636B',linewidth=.8); ax.add_patch(box); ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7,fontweight='semibold')
    arrows=[((.30,.79),(.39,.79)),((.61,.79),(.70,.79)),((.81,.72),(.81,.54)),((.70,.47),(.61,.47)),((.39,.47),(.30,.47)),((.19,.54),(.19,.72))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,linewidth=1.0,color='#3F6C8E'))
    top=ranking.iloc[0]; ax.text(.5,.25,f"Top reference recommendation\n{top.scenario_label}",ha='center',va='center',fontsize=9,fontweight='semibold')
    gate=status.get('real_prospectively_validated_virtual_cell_gate','PENDING'); ax.text(.5,.11,f"Software prospective gate: {status.get('software_integrated_virtual_cell_gate','?')}   |   Real prospective gate: {gate}",ha='center',fontsize=6.8)
    ax.text(.5,.04,'Dynamic prediction → intervention → experiment → locked evaluation → calibrated update',ha='center',fontsize=6.4,color='#687078')
    return _close_export(fig,figures/'GraphicalAbstract_virtual_cell.png','GraphicalAbstract_virtual_cell','cell_square',{'ranking':ranking.head(1)}, {'real_prospective_gate':gate})


def generate_virtual_cell_figures(output_dir: str | Path) -> pd.DataFrame:
    out=Path(output_dir); figures=out/'figures'; figures.mkdir(parents=True,exist_ok=True)
    trajectories=pd.read_csv(out/'ai'/'virtual_cell_trajectories.csv')
    ranking=pd.read_csv(out/'ai'/'ai_guided_intervention_ranking.csv')
    router=pd.read_csv(out/'ai'/'ai_model_router.csv')
    calibration=pd.read_csv(out/'prospective'/'cycle_calibration.csv')
    benchmarks=pd.read_csv(out/'real_world'/'real_world_benchmark_matrix.csv')
    evidence=pd.read_csv(out/'real_world'/'real_world_evidence_ledger.csv')
    status=json.loads((out/'validation'/'prospective_virtual_cell_status.json').read_text())
    exports=[_figure1(trajectories,ranking,figures),_figure2(ranking,figures),_figure3(router,figures),_figure4(calibration,figures),_figure5(benchmarks,evidence,figures),_graphical_abstract(ranking,status,figures)]
    rows=[]
    for item in exports:
        png=Path(item.png); svg=Path(item.svg); pdf=Path(item.pdf); tiff=Path(item.tiff)
        valid=all(p.exists() and p.stat().st_size>500 for p in (png,svg,pdf,tiff)) and item.dpi>=600
        rows.append({"figure_id":item.figure_id,"profile":item.profile,"dpi":item.dpi,"png":png.name,"svg":svg.name,"pdf":pdf.name,"tiff":tiff.name,"source_data_files":len(item.source_data),"validated":valid,"synthetic_only":item.synthetic_only})
    inv=pd.DataFrame(rows); inv.to_csv(figures/'figure_inventory.csv',index=False); return inv


def _table(frame: pd.DataFrame, columns: list[str] | None=None, max_rows: int=12) -> str:
    if columns: frame=frame[columns]
    return frame.head(max_rows).to_html(index=False,classes='data-table',border=0,escape=True,float_format=lambda x:f"{x:.3f}")


def generate_virtual_cell_report(output_dir: str | Path) -> Path:
    out=Path(output_dir); report=out/'report'; report.mkdir(parents=True,exist_ok=True)
    ranking=pd.read_csv(out/'ai'/'ai_guided_intervention_ranking.csv')
    router=pd.read_csv(out/'ai'/'ai_model_router.csv')
    validation=pd.read_csv(out/'validation'/'prospective_virtual_cell_validation_matrix.csv')
    status=json.loads((out/'validation'/'prospective_virtual_cell_status.json').read_text())
    real_status=json.loads((out/'real_world'/'real_world_hub_status.json').read_text())
    top=ranking.iloc[0]
    real_gate=status['real_prospectively_validated_virtual_cell_gate']
    gate_class='pass' if real_gate=='PASS' else 'pending'
    html_text=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CausaFlux v1.9.0 Virtual Cell</title>
<style>
:root{{--ink:#182026;--muted:#65717a;--blue:#315f7d;--teal:#267a73;--gold:#b78324;--red:#a84646;--bg:#f4f6f7;--card:#fff;--line:#dfe4e7}}*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,Arial,Helvetica,sans-serif;background:var(--bg);color:var(--ink)}}header{{padding:34px 5vw 26px;background:linear-gradient(125deg,#102532,#234e63 58%,#2d746e);color:#fff}}header h1{{margin:0;font-size:34px;letter-spacing:-.7px}}header p{{max-width:920px;line-height:1.55;color:#e4eef2}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}}nav a{{color:#fff;text-decoration:none;border:1px solid #ffffff55;padding:7px 11px;border-radius:18px;font-size:13px}}main{{max-width:1220px;margin:0 auto;padding:26px 22px 60px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 2px 10px #00000008}}.metric{{font-size:26px;font-weight:700;margin-top:6px}}.small{{font-size:13px;color:var(--muted);line-height:1.5}}.pass{{border-left:5px solid var(--teal)}}.pending{{border-left:5px solid var(--gold)}}h2{{margin-top:34px}}.figure{{background:white;border:1px solid var(--line);border-radius:14px;padding:12px;margin:16px 0}}.figure img{{width:100%;height:auto;display:block}}.data-table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff}}.data-table th,.data-table td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}.data-table th{{background:#eef2f4}}code{{background:#edf1f3;padding:2px 5px;border-radius:5px}}.warning{{background:#fff8e9;border:1px solid #eed6a4;border-left:5px solid var(--gold);padding:15px;border-radius:10px;line-height:1.5}}footer{{color:var(--muted);font-size:12px;margin-top:40px}}</style></head><body>
<header><h1>CausaFlux v1.9.0 — AI-Guided Virtual Cell</h1><p>Unified real-world evidence, validated AI modules, dynamic virtual-cell simulation, intervention ranking, prospective experimental locking, and publication-grade reporting.</p><nav><a href='#overview'>Overview</a><a href='#virtualcell'>Virtual Cell</a><a href='#models'>AI Models</a><a href='#realworld'>Real-world Data</a><a href='#validation'>Prospective Validation</a><a href='#figures'>Figures</a></nav></header><main>
<section id='overview'><div class='grid'><div class='card pass'><div class='small'>Software integrated virtual-cell gate</div><div class='metric'>{html.escape(status['software_integrated_virtual_cell_gate'])}</div></div><div class='card {gate_class}'><div class='small'>Real prospectively validated virtual-cell gate</div><div class='metric'>{html.escape(real_gate)}</div></div><div class='card'><div class='small'>Top AI-guided reference intervention</div><div class='metric' style='font-size:19px'>{html.escape(str(top.scenario_label))}</div><div class='small'>calibrated utility {top.calibrated_utility:.3f}</div></div><div class='card'><div class='small'>Real-world benchmark families</div><div class='metric'>{real_status['benchmark_families']}</div><div class='small'>{real_status['registered_sources']} registered sources</div></div></div></section>
<div class='warning'><b>Evidence boundary.</b> The bundled v1.9 reference demonstrates a prospectively locked software workflow and integrates real observational evidence. It is not yet a biologically prospectively validated virtual cell. That label remains locked until real perturbational evidence and three real prospective cycles—including independent Cycle 3 confirmation/falsification—pass.</div>
<section id='virtualcell'><h2>Virtual cell</h2><div class='figure'><img src='../figures/Figure1_virtual_cell_trajectory.svg' alt='Virtual cell trajectory'></div><div class='figure'><img src='../figures/Figure2_ai_intervention_ranking.svg' alt='Intervention ranking'></div>{_table(ranking,['rank','scenario_label','calibrated_utility','utility_per_cost','final_recovery_potential','mean_predictive_uncertainty'])}</section>
<section id='models'><h2>AI model router</h2><p class='small'>CausaFlux v1.9 combines validated modules using reliability weights derived from locked module-level benchmarks. The model router is explicit and auditable.</p><div class='figure'><img src='../figures/Figure3_ai_model_router.svg' alt='AI model router'></div>{_table(router,['module','selected_model','primary_metric','primary_value','reliability','normalized_weight','evidence_class'])}</section>
<section id='realworld'><h2>Real-world data and evidence</h2><div class='figure'><img src='../figures/Figure5_real_world_evidence.svg' alt='Real-world evidence'></div><p class='small'>Bundled large or controlled datasets are not redistributed. The hub preserves accession, access, licensing, provenance and evidence class, and can register user-provided real datasets by content hash.</p></section>
<section id='validation'><h2>Prospective validation</h2><div class='figure'><img src='../figures/Figure4_prospective_calibration.svg' alt='Prospective calibration'></div>{_table(validation,['module_or_gate','status','evidence','requirement_class'])}</section>
<section id='figures'><h2>Publication outputs</h2><div class='figure'><img src='../figures/GraphicalAbstract_virtual_cell.svg' alt='Graphical abstract'></div><p class='small'>All primary v1.9 figures export as 600-dpi PNG/TIFF plus vector SVG/PDF with panel source-data CSVs and SHA-256 figure manifests.</p></section>
<footer>CausaFlux v1.9.0 • Generated from locked reference artifacts • Real prospective claim authorization: {html.escape(real_gate)}</footer></main></body></html>"""
    path=report/'index.html'; path.write_text(html_text,encoding='utf-8'); return path
