"""Streamlit user interface for CausaFlux v2.0.0."""
from __future__ import annotations
from pathlib import Path
import json, os, subprocess, sys
import pandas as pd


def _read_json(path:Path): return json.loads(path.read_text(encoding="utf-8"))
def self_test()->dict: return {"module":"causaflux.ui_app","version":"2.0.0","streamlit_optional":True,"status":"PASS"}


def render_app():
    import streamlit as st
    from causaflux.virtual_cell import InterventionScenario, load_module_evidence, simulate_scenario
    st.set_page_config(page_title="CausaFlux v2.0.0",page_icon="🧬",layout="wide")
    st.markdown("""<style>.block-container{padding-top:1.5rem;max-width:1450px}.ok{padding:.7rem 1rem;border-left:5px solid #267a73;background:#f1faf8}.locked{padding:.7rem 1rem;border-left:5px solid #b64c4c;background:#fff4f2}</style>""",unsafe_allow_html=True)
    st.title("CausaFlux v2.0.0 — Prospectively Validated Virtual Cell")
    st.caption("Real longitudinal perturbation data • AI-guided virtual cell • locked prospective cycles • distribution-shift calibration • claim-linked evidence")
    default=os.environ.get("CAUSAFLUX_OUTPUT","causaflux_v2.0.0_release"); output=Path(st.sidebar.text_input("Analysis output",default)).expanduser().resolve(); project=Path(__file__).resolve().parents[2]
    if not (output/"run_manifest.json").exists(): st.warning("Run `sh run.sh` first, or select a completed v2 output directory."); st.stop()
    status=_read_json(output/"validation/v2_release_gate.json"); matrix=pd.read_csv(output/"validation/v2_release_matrix.csv"); ledger=pd.read_csv(output/"evidence/evidence_ledger.csv"); registry=pd.read_csv(output/"real_longitudinal/longitudinal_perturbation_registry.csv")
    ranking=pd.read_csv(output/"ai/ai_guided_intervention_ranking.csv") if (output/"ai/ai_guided_intervention_ranking.csv").exists() else pd.DataFrame(); router=pd.read_csv(output/"ai/ai_model_router.csv") if (output/"ai/ai_model_router.csv").exists() else pd.DataFrame(); calibration=pd.read_csv(output/"prospective/cycle_calibration.csv") if (output/"prospective/cycle_calibration.csv").exists() else pd.DataFrame()
    tabs=st.tabs(["Release Status","Virtual Cell","AI Models","Real Longitudinal Data","Evidence Ledger","Prospective Validation","Figures & Reports"])
    with tabs[0]:
        c1,c2,c3,c4=st.columns(4); c1.metric("Software release", "READY" if status["software_release_ready"] else "FAIL"); c2.metric("Prospective claim",status["release_claim_status"]); c3.metric("Real criteria",f"{status['real_required_criteria_passed']}/{status['real_required_criteria']}"); c4.metric("Real cycles",len(status.get("real_prospective_cycles_completed",[])))
        if status["prospectively_validated_virtual_cell"]: st.markdown("<div class='ok'><b>Prospectively validated virtual-cell claim is authorized by the locked evidence ledger.</b></div>",unsafe_allow_html=True)
        else: st.markdown("<div class='locked'><b>The v2 prospective-validation claim is locked.</b> Software completeness is not enough; all real evidence criteria must pass.</div>",unsafe_allow_html=True)
        st.dataframe(matrix,use_container_width=True,hide_index=True)
    with tabs[1]:
        st.subheader("Interactive virtual-cell scenario explorer")
        left,right=st.columns([1,2])
        with left:
            stress=st.slider("Stress intensity",0.0,1.5,1.0,.05); ire1=st.slider("IRE1/XBP1 support",0.0,1.0,.5,.05); perk=st.slider("PERK/ATF4 relief",0.0,1.0,.2,.05); atf6=st.slider("ATF6 support",0.0,1.0,.35,.05); mito=st.slider("Mitochondrial support",0.0,1.0,.35,.05); anti=st.slider("Anti-inflammatory support",0.0,1.0,.2,.05); delay=st.slider("Treatment delay",0.0,.7,.1,.05)
        scenario=InterventionScenario("ui_custom","Custom UI scenario",stress=stress,ire1_support=ire1,perk_relief=perk,atf6_support=atf6,mitochondrial_support=mito,anti_inflammatory=anti,delayed_start_fraction=delay,cost=1.0); traj,_=simulate_scenario(scenario,load_module_evidence(project))
        with right:
            f=traj.set_index("time_hours")[[c for c in traj.columns if c.endswith("_mean")]]; f.columns=[c.replace("_mean","").replace("_"," ") for c in f.columns]; st.line_chart(f)
        st.caption("Interactive trajectories remain model outputs. Biological authorization is governed separately by the v2 evidence ledger.")
    with tabs[2]:
        st.subheader("AI model router"); st.dataframe(router,use_container_width=True,hide_index=True)
        if len(ranking): st.subheader("AI-guided intervention ranking"); st.dataframe(ranking,use_container_width=True,hide_index=True)
    with tabs[3]:
        st.subheader("Actual longitudinal perturbation datasets"); st.dataframe(registry,use_container_width=True,hide_index=True)
        st.code("causaflux longitudinal-convert --input experiment.csv --output experiment.npz\ncausaflux longitudinal-benchmark --input experiment.csv --output real_benchmark")
        st.caption("Repository data are not silently redistributed. Download from the authoritative source, retain provenance/checksums, then convert using the v2 contract.")
    with tabs[4]: st.subheader("Claim-linked evidence ledger"); st.dataframe(ledger,use_container_width=True,hide_index=True)
    with tabs[5]:
        st.subheader("Prospective validation")
        if len(calibration): st.line_chart(calibration.set_index("cycle")[[c for c in ["prediction_rmse","brier_discovery","interval_coverage_90"] if c in calibration.columns]])
        st.dataframe(matrix[matrix.criterion.str.contains("PROSPECTIVE|SHIFT|EXTERNAL",regex=True)],use_container_width=True,hide_index=True)
    with tabs[6]:
        for name in ["GraphicalAbstract_v2_prospectively_validated_virtual_cell","Figure1_v2_release_evidence_ladder","Figure2_claim_evidence_matrix","Figure3_prospective_cycle_evidence"]:
            p=output/"figures"/f"{name}.png"
            if p.exists(): st.image(str(p),caption=name,use_container_width=True)
        st.info(f"Static report: {output/'report/index.html'}")


def cli_main():
    if "--self-test" in sys.argv: print(json.dumps(self_test(),indent=2)); return
    raise SystemExit(subprocess.call([sys.executable,"-m","streamlit","run",str(Path(__file__).resolve()),"--server.headless=false"]))
if __name__=="__main__":
    if "--self-test" in sys.argv: print(json.dumps(self_test(),indent=2))
    else: render_app()
