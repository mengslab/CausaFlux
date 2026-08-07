"""Publication-grade v2 release/evidence reporting."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .visualization.publication import apply_publication_style, export_figure, COLORS


def _export(fig, out:Path, figure_id:str, source_data, profile="nature_double"):
    result=export_figure(fig,out/figure_id,figure_id=figure_id,profile=profile,source_data=source_data,metadata={"release":"2.0.0","evidence_report":True},synthetic_only=False)
    plt.close(fig); return result


def generate_v2_figures(output_dir:str|Path)->pd.DataFrame:
    out=Path(output_dir); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    matrix=pd.read_csv(out/"validation/v2_release_matrix.csv")
    ledger=pd.read_csv(out/"evidence/evidence_ledger.csv")
    rows=[]
    label=lambda x:x.replace("CF2_","").replace("_"," ").title()

    # Figure 1: software readiness versus qualifying real evidence for each release claim.
    data=matrix.copy()
    software=[]
    for claim in data.criterion:
        sub=ledger[(ledger.claim_id==claim)&(ledger.evidence_kind=="software_fixture")]
        software.append(bool(sub.status.astype(str).str.upper().isin({"SOFTWARE_PASS","PASS"}).any()))
    data["software_supported"]=software
    data["real_validated"]=data.passed.astype(bool)
    spec=apply_publication_style("nature_double"); fig,ax=plt.subplots(figsize=(spec["width_mm"]/25.4,spec["height_mm"]/25.4))
    y=np.arange(len(data)); ax.hlines(y,0,1,color=COLORS["grid"],linewidth=1.0,zorder=1)
    ax.scatter(np.zeros(len(y)),y,s=42,facecolors=[COLORS["teal"] if v else "white" for v in data.software_supported],edgecolors=COLORS["teal"],linewidths=.9,zorder=3,label="Software evidence")
    ax.scatter(np.ones(len(y)),y,s=42,facecolors=[COLORS["blue"] if v else "white" for v in data.real_validated],edgecolors=[COLORS["blue"] if v else COLORS["red"] for v in data.real_validated],linewidths=.9,zorder=3,label="Qualifying real evidence")
    ax.set_yticks(y,[label(x) for x in data.criterion]); ax.set_xticks([0,1],["Software readiness","Real validation"]); ax.set_xlim(-.18,1.18); ax.invert_yaxis(); ax.set_title("CausaFlux v2.0 evidence ladder",loc="left")
    ax.spines[["top","right","left","bottom"]].set_visible(False); ax.tick_params(axis="both",length=0); ax.legend(loc="lower right",frameon=False,fontsize=6.5)
    r=_export(fig,figdir,"Figure1_v2_release_evidence_ladder",{"panel_a":data}); rows.append(r)

    # Figure 2: claim/evidence heatmap.
    claim_ids=data.criterion.tolist(); evid_types=["software_fixture","real_longitudinal_perturbation","real_multimodal_perturbation","real_spatial_perturbation","prospective_cycle","external_lab_replication","independent_cohort_replication","distribution_shift_calibration","real_negative_result","real_failed_assay"]
    heat=pd.DataFrame(0,index=claim_ids,columns=evid_types,dtype=int)
    for row in ledger.itertuples(index=False):
        if row.claim_id in heat.index and row.evidence_kind in heat.columns:
            status=str(row.status).upper(); heat.loc[row.claim_id,row.evidence_kind]=max(heat.loc[row.claim_id,row.evidence_kind],2 if status in {"PASS","VALIDATED","CONFIRMED","SUPPORTED"} else 1 if status in {"PARTIAL","SOFTWARE_PASS"} else 0)
    spec=apply_publication_style("nature_double"); fig,ax=plt.subplots(figsize=(spec["width_mm"]/25.4,spec["height_mm"]/25.4))
    im=ax.imshow(heat.to_numpy(),aspect="auto",vmin=0,vmax=2,cmap="Blues"); ax.set_yticks(range(len(heat)),[label(x) for x in heat.index],fontsize=6); ax.set_xticks(range(len(heat.columns)),[x.replace("_"," ") for x in heat.columns],rotation=48,ha="right",fontsize=6); ax.set_title("Claim-linked evidence architecture",loc="left"); fig.colorbar(im,ax=ax,fraction=.03,pad=.02,ticks=[0,1,2],label="Evidence status")
    r=_export(fig,figdir,"Figure2_claim_evidence_matrix",{"panel_a":heat.reset_index().rename(columns={"index":"claim_id"})}); rows.append(r)

    # Figure 3: prospective sequence distinguishes software-locked rehearsal from real experiment completion.
    stages=pd.DataFrame({"stage":["Cycle 1","Cycle 2","Independent replication"],"software":[1,1,0],"real":[0,0,0]})
    real_cycles=set(pd.to_numeric(ledger.loc[(ledger.evidence_kind=="prospective_cycle")&(~ledger.synthetic.fillna(False))&ledger.status.astype(str).str.upper().isin({"PASS","VALIDATED","CONFIRMED","SUPPORTED"}),"cycle"],errors="coerce").dropna().astype(int))
    stages.loc[0,"real"]=int(1 in real_cycles); stages.loc[1,"real"]=int(2 in real_cycles)
    rep=ledger[(ledger.claim_id=="CF2_EXTERNAL_REPLICATION")&ledger.independent.fillna(False)&(~ledger.synthetic.fillna(False))&ledger.status.astype(str).str.upper().isin({"PASS","VALIDATED","CONFIRMED","SUPPORTED"})]
    stages.loc[2,"real"]=int(len(rep)>0)
    spec=apply_publication_style("nature_double"); fig,ax=plt.subplots(figsize=(spec["width_mm"]/25.4,spec["height_mm"]/25.4))
    x=np.arange(len(stages)); ax.plot(x,np.full(len(x),1.0),color=COLORS["grid"],linewidth=1.4,zorder=1); ax.plot(x,np.full(len(x),0.0),color=COLORS["grid"],linewidth=1.4,zorder=1)
    for i,row in stages.iterrows():
        ax.scatter(i,1,s=95,facecolor=COLORS["teal"] if row.software else "white",edgecolor=COLORS["teal"],linewidth=1.2,zorder=3)
        ax.scatter(i,0,s=95,facecolor=COLORS["blue"] if row.real else "white",edgecolor=COLORS["blue"] if row.real else COLORS["red"],linewidth=1.2,zorder=3)
        ax.text(i,1.16,"complete" if row.software else "not applicable",ha="center",va="bottom",fontsize=6.3,color=COLORS["muted"])
        ax.text(i,-.16,"complete" if row.real else "pending",ha="center",va="top",fontsize=6.3,color=COLORS["blue"] if row.real else COLORS["red"])
    ax.set_xticks(x,stages.stage); ax.set_yticks([0,1],["Real evidence","Software rehearsal"]); ax.set_ylim(-.38,1.38); ax.set_title("Prospective validation sequence",loc="left"); ax.spines[["top","right","left","bottom"]].set_visible(False); ax.tick_params(length=0)
    r=_export(fig,figdir,"Figure3_prospective_cycle_evidence",{"panel_a":stages}); rows.append(r)

    # Figure 4: evidence composition makes negative/pending evidence visible.
    def category(row):
        st=str(row.status).upper()
        if bool(row.synthetic): return "software fixture"
        if st in {"PASS","VALIDATED","CONFIRMED","SUPPORTED"}: return "qualifying real"
        if st=="PARTIAL": return "real partial"
        return "pending / boundary"
    comp=ledger.copy(); comp["category"]=comp.apply(category,axis=1); counts=comp.category.value_counts().reindex(["software fixture","real partial","qualifying real","pending / boundary"],fill_value=0).rename_axis("category").reset_index(name="records")
    spec=apply_publication_style("nature_single"); fig,ax=plt.subplots(figsize=(spec["width_mm"]/25.4,spec["height_mm"]/25.4)); bars=ax.barh(np.arange(len(counts)),counts.records,color=[COLORS["teal"],COLORS["gold"],COLORS["blue"],COLORS["grid"]]); ax.set_yticks(np.arange(len(counts)),counts.category); ax.invert_yaxis(); ax.set_xlabel("Evidence records"); ax.set_title("Evidence ledger composition",loc="left"); ax.spines[["top","right"]].set_visible(False)
    for b,v in zip(bars,counts.records): ax.text(b.get_width()+.08,b.get_y()+b.get_height()/2,str(int(v)),va="center",fontsize=6.5)
    r=_export(fig,figdir,"Figure4_evidence_ledger_composition",{"panel_a":counts},profile="nature_single"); rows.append(r)

    # Graphical abstract.
    spec=apply_publication_style("cell_square"); fig,ax=plt.subplots(figsize=(spec["width_mm"]/25.4,spec["height_mm"]/25.4)); ax.axis("off")
    labels=["REAL LONGITUDINAL\nPERTURBATION DATA","DYNAMIC + MULTIMODAL\nVIRTUAL CELL","UNSEEN INTERVENTION +\nSPATIAL CONTEXT","LOCKED PROSPECTIVE\nCYCLES 1 → 2","INDEPENDENT\nREPLICATION","CALIBRATED SHIFT\nUNCERTAINTY","CLAIM-LINKED\nEVIDENCE LEDGER"]
    ys=np.linspace(.90,.16,len(labels))
    for i,(text,yv) in enumerate(zip(labels,ys)):
        fc="#F1F7F6" if i<4 else "#F7F5EE"; ax.text(.5,yv,text,ha="center",va="center",fontsize=8.6,fontweight="bold",bbox=dict(boxstyle="round,pad=.52",facecolor=fc,edgecolor="#6D747A",linewidth=.8))
        if i<len(labels)-1: ax.annotate("",xy=(.5,ys[i+1]+.045),xytext=(.5,yv-.045),arrowprops=dict(arrowstyle="->",lw=1,color="#40464B"))
    status=json.loads((out/"validation/v2_release_gate.json").read_text()); footer="PROSPECTIVELY VALIDATED" if status.get("prospectively_validated_virtual_cell") else "CLAIM LOCKED — REAL EVIDENCE REQUIRED"
    ax.text(.5,.055,footer,ha="center",va="center",fontsize=7.5,fontweight="bold",color=COLORS["teal"] if status.get("prospectively_validated_virtual_cell") else COLORS["red"])
    ga=pd.DataFrame({"step":range(1,len(labels)+1),"component":[x.replace("\n"," ") for x in labels]})
    r=_export(fig,figdir,"GraphicalAbstract_v2_prospectively_validated_virtual_cell",{"panel_a":ga},profile="cell_square"); rows.append(r)

    inventory=[]
    for r in rows:
        inventory.append({"figure_id":r.figure_id,"png":Path(r.png).name,"svg":Path(r.svg).name,"pdf":Path(r.pdf).name,"tiff":Path(r.tiff).name,"dpi":r.dpi,"validated":all(Path(p).exists() and Path(p).stat().st_size>500 for p in [r.png,r.svg,r.pdf,r.tiff]),"synthetic_only":r.synthetic_only})
    inv=pd.DataFrame(inventory); inv.to_csv(figdir/"figure_inventory.csv",index=False); return inv

def generate_v2_report(output_dir:str|Path)->Path:
    out=Path(output_dir); report=out/"report"; report.mkdir(parents=True,exist_ok=True)
    status=json.loads((out/"validation/v2_release_gate.json").read_text()); matrix=pd.read_csv(out/"validation/v2_release_matrix.csv"); ledger=pd.read_csv(out/"evidence/evidence_ledger.csv"); registry=pd.read_csv(out/"real_longitudinal/longitudinal_perturbation_registry.csv")
    badge="PASS" if status.get("prospectively_validated_virtual_cell") else "LOCKED / NOT YET ELIGIBLE"
    css="""body{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:28px auto;padding:0 24px;color:#202124}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ddd;padding:6px;vertical-align:top}th{background:#f3f5f6}.ok{border-left:5px solid #2D7F78;background:#f2fbf8;padding:12px}.warn{border-left:5px solid #B64C4C;background:#fff4f2;padding:12px}.muted{color:#687078}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}img{max-width:100%;border:1px solid #e2e4e5}"""
    html=f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v2.0.0</title><style>{css}</style></head><body>
<h1>CausaFlux v2.0.0 — Prospectively Validated Virtual Cell</h1>
<div class='{'ok' if status.get('prospectively_validated_virtual_cell') else 'warn'}'><b>Prospective-validation claim: {badge}</b><br>Software release ready: {status.get('software_release_ready')}. Real criteria passed: {status.get('real_required_criteria_passed')}/{status.get('real_required_criteria')}.</div>
<p>CausaFlux v2.0.0 treats the version title as a release standard, not an automatic claim. Synthetic software fixtures remain useful for regression testing but cannot satisfy the real validation gate.</p>
<h2>Release criteria</h2>{matrix.to_html(index=False,escape=True)}
<h2>Evidence ledger</h2><p>Every release-level claim is linked to a typed evidence record. Pending, failed and negative evidence is retained rather than removed from the report.</p>{ledger.to_html(index=False,escape=True,max_rows=100)}
<h2>Real longitudinal perturbation data bridge</h2>{registry.to_html(index=False,escape=True)}
<div class='grid'><div><h3>Evidence ladder</h3><img src='../figures/Figure1_v2_release_evidence_ladder.png'></div><div><h3>Claim–evidence matrix</h3><img src='../figures/Figure2_claim_evidence_matrix.png'></div><div><h3>Prospective sequence</h3><img src='../figures/Figure3_prospective_cycle_evidence.png'></div><div><h3>Ledger composition</h3><img src='../figures/Figure4_evidence_ledger_composition.png'></div><div><h3>Virtual-cell validation flow</h3><img src='../figures/GraphicalAbstract_v2_prospectively_validated_virtual_cell.png'></div></div>
<h2>Interpretation boundary</h2><p><b>Do not describe CausaFlux as prospectively validated unless this report shows PROSPECTIVELY_VALIDATED.</b> The bundled reference package is a stable software release with a locked real-evidence gate. Real experimental evidence must be supplied through the evidence ledger and pass all criteria.</p>
</body></html>"""
    path=report/"index.html"; path.write_text(html,encoding="utf-8"); return path
