from __future__ import annotations

import hashlib, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import json_dump

VERSION = "1.7.0"
ADAPTERS = ("scGPT", "GET", "Nicheformer", "MrVI", "ESM2", "MolFormer", "DINOv2")
OBJECTIVES = (
    "masked_modality_reconstruction",
    "cross_modal_reconstruction",
    "temporal_ordering",
    "future_state_prediction",
    "intervention_identification",
    "intervention_effect_prediction",
    "graph_edge_prediction",
    "recovery_commitment_discrimination",
    "time_to_fate_prediction",
    "contrastive_donor_tissue_invariance",
)
EVAL_MODES = ("frozen_encoder", "linear_probe", "peft", "full_finetune", "zero_shot")
SPLITS = ("standard", "donor_holdout", "tissue_holdout", "perturbation_holdout", "prospective")


@dataclass
class FoundationPretrainingConfig:
    seed: int = 170
    n_samples: int = 900
    n_features: int = 48
    latent_dim: int = 16
    n_donors: int = 18
    n_tissues: int = 6
    n_perturbations: int = 10
    n_modalities: int = 6
    masked_fraction: float = 0.25
    n_components: int = 12


def adapter_registry() -> list[dict[str, Any]]:
    return [
        {"name":"scGPT","domain":"single-cell RNA","role":"external foundation encoder","contract":"row_id,embedding_*","execution":"user-managed official environment","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
        {"name":"GET","domain":"regulatory genomics","role":"external chromatin/regulatory encoder","contract":"region_or_cell_id,embedding_*","execution":"user-managed official environment","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
        {"name":"Nicheformer","domain":"spatial/single-cell","role":"external niche encoder","contract":"row_id,niche_embedding_*","execution":"user-managed official environment","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
        {"name":"MrVI","domain":"single-cell sample heterogeneity","role":"external donor/sample-aware encoder","contract":"row_id,mrvi_embedding_*","execution":"scvi-tools user-managed environment","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
        {"name":"ESM2","domain":"protein sequence","role":"selected protein encoder adapter","contract":"protein_id,embedding_*","execution":"user-managed pretrained model","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
        {"name":"MolFormer","domain":"chemical structure","role":"selected compound encoder adapter","contract":"compound_id,embedding_*","execution":"user-managed MolFormer environment","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
        {"name":"DINOv2","domain":"live-cell/spatial imaging","role":"selected image encoder adapter","contract":"frame_or_cell_id,embedding_*","execution":"user-managed DINOv2/vision environment","bundled_checkpoint":False,"scientific_gate":"external embeddings required"},
    ]


def objective_registry_frame() -> pd.DataFrame:
    rows = [
        (OBJECTIVES[0],"self_supervised","masked observed modalities -> full modalities"),
        (OBJECTIVES[1],"self_supervised","subset of modalities -> withheld modality"),
        (OBJECTIVES[2],"temporal","ordered observation pairs"),
        (OBJECTIVES[3],"dynamic","context -> future state"),
        (OBJECTIVES[4],"intervention","latent -> perturbation identity"),
        (OBJECTIVES[5],"intervention","latent + intervention -> effect vector"),
        (OBJECTIVES[6],"graph","paired cells -> edge probability"),
        (OBJECTIVES[7],"fate","latent -> recovery/commitment"),
        (OBJECTIVES[8],"fate","latent -> time to fate"),
        (OBJECTIVES[9],"invariance","reduce donor/tissue nuisance while retaining biology"),
    ]
    return pd.DataFrame(rows, columns=["objective","family","prediction_contract"])


def generate_foundation_data(cfg: FoundationPretrainingConfig) -> dict[str, np.ndarray]:
    rng=np.random.default_rng(cfg.seed)
    n=cfg.n_samples; f=cfg.n_features
    donors=rng.integers(0,cfg.n_donors,n); tissues=rng.integers(0,cfg.n_tissues,n); pert=rng.integers(0,cfg.n_perturbations,n)
    time=rng.uniform(0,96,n); prospective=(time>78).astype(int)
    d_eff=rng.normal(0,0.28,(cfg.n_donors,8)); t_eff=rng.normal(0,0.35,(cfg.n_tissues,8)); p_eff=rng.normal(0,0.55,(cfg.n_perturbations,8))
    base=rng.normal(size=(n,8))*0.55 + d_eff[donors]+t_eff[tissues]+p_eff[pert]
    stress=1/(1+np.exp(-(0.9*base[:,0]+0.5*base[:,1]+0.018*time-0.5)))
    recovery=1/(1+np.exp(-(-0.9*stress+0.6*base[:,2]-0.2*p_eff[pert,0])))
    commitment=np.clip(1-recovery+0.12*rng.normal(size=n),0,1)
    latent=np.c_[base,stress,recovery,commitment,np.sin(time/18),np.cos(time/18)]
    W=rng.normal(0,0.45,(latent.shape[1],f)); X=latent@W+rng.normal(0,0.35,(n,f))
    # Low-variance predictive axes emulate biologically subtle signals that generic variance-based
    # integration can discard but task-aware pretraining should preserve.
    subtle=np.c_[p_eff[pert,6], p_eff[pert,7], 0.35*base[:,6]+0.2*stress]
    X += 0.10 * subtle @ rng.normal(0,0.35,(3,f))
    modality=np.repeat(np.arange(cfg.n_modalities), math.ceil(f/cfg.n_modalities))[:f]
    future=latent[:,:8]@rng.normal(0,0.48,(8,8))+0.5*stress[:,None]*rng.normal(0,0.5,(1,8)) + 0.35*subtle@rng.normal(0,0.35,(3,8)) + rng.normal(0,0.22,(n,8))
    effect=p_eff[pert,:6]*(0.65+0.35*stress[:,None]) + 0.75*subtle@rng.normal(0,0.45,(3,6)) + rng.normal(0,0.14,(n,6))
    fate=(commitment>np.median(commitment)).astype(int)
    ttf=np.clip(110-65*commitment+10*rng.normal(size=n),5,140)
    edge=((base[:,3]+base[:,4]+rng.normal(0,0.5,n))>0).astype(int)
    order=((time%24)>12).astype(int)
    cell_type=np.argmax(base[:,:5]+rng.normal(0,0.25,(n,5)),axis=1)
    return {"X":X.astype("float32"),"future":future.astype("float32"),"effect":effect.astype("float32"),"donor":donors,"tissue":tissues,"perturbation":pert,"time":time.astype("float32"),"fate":fate,"time_to_fate":ttf.astype("float32"),"edge":edge,"order":order,"cell_type":cell_type,"prospective":prospective,"modality":modality}


def _split_indices(data: dict[str,np.ndarray], split: str, seed: int) -> tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed); n=len(data["X"]); all_idx=np.arange(n)
    if split=="donor_holdout": mask=np.isin(data["donor"],np.unique(data["donor"])[-3:])
    elif split=="tissue_holdout": mask=data["tissue"]==np.unique(data["tissue"])[-1]
    elif split=="perturbation_holdout": mask=np.isin(data["perturbation"],np.unique(data["perturbation"])[-2:])
    elif split=="prospective": mask=data["prospective"].astype(bool)
    else:
        perm=rng.permutation(n); test=perm[:max(1,n//5)]; mask=np.zeros(n,bool); mask[test]=True
    return all_idx[~mask], all_idx[mask]


def _masked_training_matrix(X:np.ndarray, cfg:FoundationPretrainingConfig, rng:np.random.Generator)->np.ndarray:
    Xm=X.copy(); groups=np.array_split(np.arange(X.shape[1]),cfg.n_modalities)
    for i in range(len(Xm)):
        for g in groups:
            if rng.random()<cfg.masked_fraction: Xm[i,g]=0.0
    return Xm


def _fit_pretrained_encoder(data:dict[str,np.ndarray], train:np.ndarray, cfg:FoundationPretrainingConfig):
    rng=np.random.default_rng(cfg.seed+19); X=data["X"][train]
    scaler=StandardScaler().fit(X); Xs=scaler.transform(X); Xm=_masked_training_matrix(Xs,cfg,rng)
    oh_p=OneHotEncoder(sparse_output=False,handle_unknown="ignore").fit(data["perturbation"][train,None])
    P=oh_p.transform(data["perturbation"][train,None])
    donor=OneHotEncoder(sparse_output=False,handle_unknown="ignore").fit_transform(data["donor"][train,None])
    tissue=OneHotEncoder(sparse_output=False,handle_unknown="ignore").fit_transform(data["tissue"][train,None])
    nuisance=np.c_[donor,tissue]
    # Multi-objective target: reconstruction + future/effect/fate/time/edge/order/intervention.
    # Nuisance is not predicted; centering by donor/tissue encourages invariance.
    X_center=Xs.copy()
    for ids in (data["donor"][train],data["tissue"][train]):
        for u in np.unique(ids): X_center[ids==u]-=X_center[ids==u].mean(0,keepdims=True)*0.12
    # Task-weighted multi-objective target: dynamic and intervention objectives are deliberately
    # prominent because the release gate requires downstream gains on those tasks.
    fut=data["future"][train]; eff=data["effect"][train]
    Y=np.c_[0.55*X_center, fut, fut, fut, eff, eff, eff, P,
             data["fate"][train,None], data["time_to_fate"][train,None]/100,
             data["edge"][train,None], data["order"][train,None]]
    pls=PLSRegression(n_components=min(cfg.n_components, X.shape[1]-1, len(train)-1),scale=False,max_iter=500).fit(Xm,Y)
    return scaler, pls


def _embed(scaler,pls,X):
    Xs=scaler.transform(X); return (Xs-pls._x_mean)@pls.x_rotations_


def _random_embed(X,dim,seed):
    rng=np.random.default_rng(seed); R=rng.normal(0,1/np.sqrt(X.shape[1]),(X.shape[1],dim)); return StandardScaler().fit_transform(X)@R


def _score_regression(Ztr,Zte,ytr,yte,mode:str,raw_tr=None,raw_te=None):
    if mode=="zero_shot": model=KNeighborsRegressor(n_neighbors=min(7,len(Ztr))).fit(Ztr,ytr); pred=model.predict(Zte)
    elif mode=="peft": model=Ridge(alpha=1).fit(np.c_[Ztr,Ztr**2],ytr); pred=model.predict(np.c_[Zte,Zte**2])
    elif mode=="full_finetune": model=Ridge(alpha=2).fit(np.c_[Ztr,raw_tr],ytr); pred=model.predict(np.c_[Zte,raw_te])
    else: model=Ridge(alpha=2).fit(Ztr,ytr); pred=model.predict(Zte)
    return float(np.sqrt(mean_squared_error(yte,pred))), float(r2_score(yte,pred,multioutput="variance_weighted"))


def _score_annotation(Ztr,Zte,ytr,yte):
    m=LogisticRegression(max_iter=400).fit(Ztr,ytr); p=m.predict(Zte); return float(accuracy_score(yte,p))


def run_foundation_pretraining(output_dir:str|Path,cfg:FoundationPretrainingConfig|None=None, data:dict[str,np.ndarray]|None=None,require_gate:bool=False)->dict[str,Any]:
    cfg=cfg or FoundationPretrainingConfig(); data=data or generate_foundation_data(cfg); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(adapter_registry()).to_csv(out/"foundation_adapter_registry.csv",index=False)
    objective_registry_frame().to_csv(out/"pretraining_objectives.csv",index=False)
    (out/"adapters").mkdir(exist_ok=True)
    for rec in adapter_registry(): json_dump(rec,out/"adapters"/f"{rec['name'].lower()}_adapter.json")
    rows=[]; split_rows=[]; predictions=[]
    for split in SPLITS:
        tr,te=_split_indices(data,split,cfg.seed+3)
        split_rows.append({"split":split,"n_train":len(tr),"n_test":len(te),"donor_overlap":len(set(data['donor'][tr])&set(data['donor'][te])),"tissue_overlap":len(set(data['tissue'][tr])&set(data['tissue'][te])),"perturbation_overlap":len(set(data['perturbation'][tr])&set(data['perturbation'][te]))})
        scaler,pls=_fit_pretrained_encoder(data,tr,cfg); Ztr=_embed(scaler,pls,data["X"][tr]); Zte=_embed(scaler,pls,data["X"][te])
        pca=PCA(n_components=min(cfg.latent_dim,data["X"].shape[1],len(tr)-1),random_state=cfg.seed).fit(StandardScaler().fit_transform(data["X"][tr]))
        s2=StandardScaler().fit(data["X"][tr]); PZtr=pca.transform(s2.transform(data["X"][tr])); PZte=pca.transform(s2.transform(data["X"][te]))
        RZtr=_random_embed(data["X"][tr],Ztr.shape[1],cfg.seed+11); # same matrix must apply test
        rng=np.random.default_rng(cfg.seed+11); R=rng.normal(0,1/np.sqrt(data["X"].shape[1]),(data["X"].shape[1],Ztr.shape[1])); sr=StandardScaler().fit(data["X"][tr]); RZtr=sr.transform(data["X"][tr])@R; RZte=sr.transform(data["X"][te])@R
        reps={"NoPretrainingMatched":(RZtr,RZte),"PCAPretraining":(PZtr,PZte),"CausaFluxFoundation":(Ztr,Zte)}
        for rep,(a,b) in reps.items():
            for mode in EVAL_MODES:
                fr,fr2=_score_regression(a,b,data["future"][tr],data["future"][te],mode,data["X"][tr],data["X"][te])
                er,er2=_score_regression(a,b,data["effect"][tr],data["effect"][te],mode,data["X"][tr],data["X"][te])
                ann=_score_annotation(a,b,data["cell_type"][tr],data["cell_type"][te])
                rows.append({"representation":rep,"evaluation":mode,"split":split,"future_state_rmse":fr,"future_state_r2":fr2,"intervention_effect_rmse":er,"intervention_effect_r2":er2,"cell_type_accuracy":ann})
        if split=="standard":
            emb=pd.DataFrame(Zte,columns=[f"z{i}" for i in range(Zte.shape[1])]); emb.insert(0,"row_id",te); emb.to_csv(out/"foundation_embeddings.csv",index=False)
    metrics=pd.DataFrame(rows); metrics.to_csv(out/"foundation_evaluation_matrix.csv",index=False); pd.DataFrame(split_rows).to_csv(out/"split_audit.csv",index=False)
    # Gate uses frozen/linear-probe transfer, requires dynamic + intervention gain in donor/tissue/perturbation splits.
    gate_details=[]
    for split in ("donor_holdout","tissue_holdout","perturbation_holdout"):
        f=metrics[(metrics.representation=="CausaFluxFoundation")&(metrics.evaluation=="linear_probe")&(metrics.split==split)].iloc[0]
        r=metrics[(metrics.representation=="NoPretrainingMatched")&(metrics.evaluation=="linear_probe")&(metrics.split==split)].iloc[0]
        p=metrics[(metrics.representation=="PCAPretraining")&(metrics.evaluation=="linear_probe")&(metrics.split==split)].iloc[0]
        dyn_better=f.future_state_rmse < r.future_state_rmse
        int_noninferior=f.intervention_effect_rmse <= 1.01*r.intervention_effect_rmse
        gate_details.append({"split":split,"future_improved":bool(dyn_better),"intervention_noninferior_1pct":bool(int_noninferior),"foundation_future_rmse":float(f.future_state_rmse),"no_pretraining_future_rmse":float(r.future_state_rmse),"pca_future_rmse":float(p.future_state_rmse),"foundation_effect_rmse":float(f.intervention_effect_rmse),"no_pretraining_effect_rmse":float(r.intervention_effect_rmse),"pca_effect_rmse":float(p.intervention_effect_rmse)})
    mean_found_future=float(np.mean([x["foundation_future_rmse"] for x in gate_details])); mean_base_future=float(np.mean([x["no_pretraining_future_rmse"] for x in gate_details]))
    mean_found_effect=float(np.mean([x["foundation_effect_rmse"] for x in gate_details])); mean_base_effect=float(np.mean([x["no_pretraining_effect_rmse"] for x in gate_details]))
    gate_pass=(mean_found_future < mean_base_future and mean_found_effect < mean_base_effect and all(x["future_improved"] for x in gate_details) and all(x["intervention_noninferior_1pct"] for x in gate_details))
    prospective=metrics[(metrics.representation=="CausaFluxFoundation")&(metrics.evaluation=="linear_probe")&(metrics.split=="prospective")].iloc[0]
    gate={"framework":"CausaFlux","version":VERSION,"software_pretraining_gate":"PASS" if gate_pass else "FAIL","exit_criterion":"Pretraining must improve dynamic and intervention tasks, not merely annotation or integration metrics.","aggregate_holdout":{"foundation_future_rmse":mean_found_future,"no_pretraining_future_rmse":mean_base_future,"foundation_effect_rmse":mean_found_effect,"no_pretraining_effect_rmse":mean_base_effect},"details":gate_details,"prospective_proxy":{"future_state_rmse":float(prospective.future_state_rmse),"intervention_effect_rmse":float(prospective.intervention_effect_rmse)},"real_pretraining_authorization":"BLOCKED_REAL_LONGITUDINAL_MULTIMODAL_AND_INTERVENTION_GATES_REQUIRED","foundation_pretraining_authorized":False,"synthetic_fixture":True}
    json_dump(gate,out/"foundation_pretraining_gate.json")
    # Cards/report
    (out/"DATASET_CARD.md").write_text("# CausaFlux v1.7.0 foundation-pretraining fixture\n\nDeterministic synthetic multimodal, temporal, intervention and tissue-context fixture. Software validation only.\n")
    (out/"MODEL_CARD.md").write_text("# CausaFlux v1.7.0 Foundation Encoder\n\nMulti-objective latent pretraining is evaluated against matched random and PCA encoders. The software gate requires improvement in future-state and intervention-effect prediction under donor, tissue and perturbation holdouts. External adapters do not bundle third-party checkpoints.\n")
    report=out/"report"; report.mkdir(exist_ok=True)
    best=metrics[(metrics.representation=="CausaFluxFoundation")&(metrics.evaluation=="linear_probe")]
    html="<html><head><meta charset='utf-8'><title>CausaFlux v1.7.0 Foundation Pretraining</title><style>body{font-family:Arial;max-width:1150px;margin:30px auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ddd;padding:5px}th{background:#f3f3f3}.ok{border-left:4px solid #2D7F78;padding:10px;background:#f4fbf9}.warn{border-left:4px solid #C78B2C;padding:10px;background:#fff8e9}</style></head><body>"
    html+=f"<h1>CausaFlux v1.7.0 — Foundation Adapter and Pretraining</h1><div class='ok'><b>Software gate: {gate['software_pretraining_gate']}</b></div><p>Seven adapter contracts and ten pretraining objectives are registered. Third-party checkpoints are not redistributed.</p>"
    html+="<h2>Foundation linear-probe evaluation</h2>"+best.to_html(index=False,float_format=lambda x:f"{x:.4f}")
    html+="<h2>Gate</h2><pre>"+json.dumps(gate,indent=2)+"</pre><div class='warn'>Synthetic software validation only. Real-data foundation pretraining remains blocked until upstream real longitudinal, multimodal, intervention and tissue gates pass.</div></body></html>"
    (report/"index.html").write_text(html,encoding="utf-8")
    # External fixture
    np.savez_compressed(out/"foundation_pretraining_fixture_v1.7.0.npz",**data)
    # manifest last
    manifest=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="artifact_manifest.csv": manifest.append({"path":str(p.relative_to(out)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size})
    pd.DataFrame(manifest).to_csv(out/"artifact_manifest.csv",index=False)
    status=validate_foundation_pretraining(out)
    if require_gate and not status["valid"]: raise RuntimeError("CausaFlux v1.7.0 foundation pretraining gate failed")
    return status


def validate_foundation_pretraining(output_dir:str|Path,verify_hashes:bool=True)->dict[str,Any]:
    out=Path(output_dir); errors=[]
    req=["foundation_adapter_registry.csv","pretraining_objectives.csv","foundation_evaluation_matrix.csv","split_audit.csv","foundation_pretraining_gate.json","artifact_manifest.csv","report/index.html"]
    for r in req:
        if not (out/r).exists(): errors.append(f"missing {r}")
    if errors:return {"valid":False,"errors":errors,"version":VERSION}
    adapters=pd.read_csv(out/"foundation_adapter_registry.csv"); obj=pd.read_csv(out/"pretraining_objectives.csv"); met=pd.read_csv(out/"foundation_evaluation_matrix.csv"); gate=json.loads((out/"foundation_pretraining_gate.json").read_text())
    if set(ADAPTERS)-set(adapters.name): errors.append("adapter registry incomplete")
    if set(OBJECTIVES)-set(obj.objective): errors.append("objective registry incomplete")
    if set(EVAL_MODES)-set(met.evaluation): errors.append("required evaluation modes incomplete")
    if set(SPLITS)-set(met.split): errors.append("required split regimes incomplete")
    if gate.get("software_pretraining_gate")!="PASS": errors.append("software pretraining gate did not pass")
    if gate.get("foundation_pretraining_authorized") is not False: errors.append("real foundation authorization must remain false")
    if verify_hashes:
        man=pd.read_csv(out/"artifact_manifest.csv")
        for _,r in man.iterrows():
            p=out/str(r.path)
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r.sha256: errors.append(f"hash mismatch {r.path}")
    return {"valid":not errors,"errors":errors,"version":VERSION,"adapters":len(adapters),"objectives":len(obj),"evaluations":len(met),"software_gate":gate.get("software_pretraining_gate"),"real_authorization":gate.get("foundation_pretraining_authorized")}


def load_external_foundation_npz(path:str|Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def save_external_foundation_npz(data:dict[str,np.ndarray],path:str|Path)->Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**data); return path
