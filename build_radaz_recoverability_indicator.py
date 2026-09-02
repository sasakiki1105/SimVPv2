#!/usr/bin/env python3
"""Build a pre-training recoverability indicator for synthetic coarse grids."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_radaz_electric_history_hidden_band_envelope_rom import carrier_candidates
from train_radaz_g2_residual_superresolution import DEFAULT_H5, make_coarse_size_interpolated, make_grid_interpolated

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workdirs/radaz_recoverability_indicator"
CONFIGS = (("G4",64,"g4"),("C51",51,"c51"),("C43",43,"c43"),("C37",37,"c37"),("G8",32,"g8"))
FIELDS = ("phi","electron_den","ion_den")
CHANNEL_TO_FIELD = (1,2,0)  # normalized H5 electron,ion,phi -> phi,electron,ion
FEATURES = ("nyquist_ratio","direct_coherence","alias_fraction","mi_score","quadratic_coherence")
TINY = np.finfo(np.float64).tiny


def coefficients(values: np.ndarray) -> np.ndarray:
    values = values[:, CHANNEL_TO_FIELD, :257, :256]
    groups = np.array_split(np.arange(257),8)
    result=np.empty((len(values),3,8,22),np.complex64)
    for r,g in enumerate(groups):
        result[:,:,r]=np.fft.rfft(np.mean(values[:,:,g,:],axis=2),axis=-1,norm="forward")[...,:22]
    return result


def coherence(a,b):
    return float(abs(np.vdot(a,b))/np.sqrt(max(np.vdot(a,a).real*np.vdot(b,b).real,TINY)))


def phase_indicator(coarse: np.ndarray, truth: np.ndarray, field: int, mode: int, train: np.ndarray):
    vals=[]
    for radial in range(8):
        candidates,_=carrier_candidates(coarse,mode,radial); target=truth[:,field,radial,mode]
        energy=np.sum(abs(candidates[train])**2,axis=0); cross=np.sum(np.conj(candidates[train])*target[train,None],axis=0)
        coh=abs(cross)/np.sqrt(np.maximum(energy*np.sum(abs(target[train])**2),TINY)); vals.append(float(np.max(coh)))
    return float(np.mean(vals))


def mi_indicator(coarse: np.ndarray, truth: np.ndarray, field: int, mode: int, train: np.ndarray):
    visible=np.mean(abs(coarse[train,:,:,:17])**2,axis=2).reshape(np.count_nonzero(train),-1)
    target=np.mean(abs(truth[train,field,:,mode])**2,axis=1)
    mi=mutual_info_regression(np.log1p(visible),np.log1p(target),random_state=20260817,n_neighbors=5)
    top=np.sort(mi)[-3:]
    # Gaussian-equivalent bounded dependence score; 0=no information, 1=strong.
    return float(1.0-np.exp(-2.0*np.mean(top)))


def read_target_metrics(tag: str):
    p=ROOT/f"workdirs/analyze_radaz_e25_{tag}_stability_reconstruction/azimuthal_mode_metrics.csv"
    with p.open(encoding="utf-8") as f:return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with h5py.File(DEFAULT_H5,"r") as h:
        time=np.asarray(h["time_s"])*1e6; sel=np.flatnonzero((time>=20-1e-9)&(time<=30+1e-9)); fine=np.asarray(h["data_tchw"][sel[0]:sel[-1]+1],np.float32); time=time[sel]
    truth=coefficients(fine); train=(time>=20)&(time<28)
    rows=[]
    for label,coarse_size,tag in CONFIGS:
        baseline=make_grid_interpolated(fine,256//coarse_size) if 256%coarse_size==0 else make_coarse_size_interpolated(fine,coarse_size)
        coarse=coefficients(baseline); targets=read_target_metrics(tag)
        for field_index,field in enumerate(FIELDS):
            for mode in range(17,22):
                a=truth[train,field_index,:,mode]; b=coarse[train,field_index,:,mode]
                transfer=np.vdot(a,b)/max(np.vdot(a,a).real,TINY); residual=b-transfer*a
                alias=float(np.vdot(residual,residual).real/max(np.vdot(b,b).real,TINY))
                target=next(x for x in targets if x["field"]==field and int(x["mode"])==mode)
                rows.append({"configuration":label,"coarse_size":coarse_size,"effective_factor":256/coarse_size,"nyquist_mode":coarse_size//2,"field":field,"mode":mode,"nyquist_ratio":(coarse_size/2)/mode,"direct_coherence":coherence(a,b),"alias_fraction":alias,"mi_score":mi_indicator(coarse,truth,field_index,mode,train),"quadratic_coherence":phase_indicator(coarse,truth,field_index,mode,train),"model_coherence":float(target["model_coherence"]),"model_relative_error":float(target["model_relative_error"]),"recoverable":int(float(target["model_coherence"])>=0.5)})
        del baseline,coarse
        print(f"computed {label}",flush=True)
    # Grouped leave-one-resolution-out calibration.
    x=np.asarray([[r[f] for f in FEATURES] for r in rows]); y=np.asarray([r["model_coherence"] for r in rows]); y_error=np.asarray([r["model_relative_error"] for r in rows]); groups=np.asarray([r["configuration"] for r in rows])
    pred=np.empty_like(y); pred_error=np.empty_like(y_error)
    folds=[]
    for label,_,_ in CONFIGS:
        test=groups==label; model=make_pipeline(StandardScaler(),Ridge(alpha=10.0));model.fit(x[~test],y[~test]);pred[test]=np.clip(model.predict(x[test]),0,1)
        error_model=make_pipeline(StandardScaler(),Ridge(alpha=10.0));error_model.fit(x[~test],y_error[~test]);pred_error[test]=np.maximum(error_model.predict(x[test]),0)
        folds.append({"heldout_configuration":label,"coherence_mae":float(np.mean(abs(pred[test]-y[test]))),"coherence_correlation":float(np.corrcoef(pred[test],y[test])[0,1]),"relative_error_mae":float(np.mean(abs(pred_error[test]-y_error[test]))),"relative_error_correlation":float(np.corrcoef(pred_error[test],y_error[test])[0,1])})
    for r,p,e in zip(rows,pred,pred_error):r["predicted_recoverability_score"]=float(p);r["predicted_relative_error"]=float(e)
    with (OUT/"recoverability_components.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    correlations=[]
    for feature in FEATURES:
        rho,p=spearmanr([r[feature] for r in rows],y);correlations.append({"feature":feature,"spearman_rho":float(rho),"p_value":float(p)})
    classification=np.asarray([r["recoverable"] for r in rows]); auc=float(roc_auc_score(classification,pred)); mae=float(np.mean(abs(pred-y))); rho=float(spearmanr(pred,y).statistic)
    final=make_pipeline(StandardScaler(),Ridge(alpha=10.0));final.fit(x,y)
    final_error=make_pipeline(StandardScaler(),Ridge(alpha=10.0));final_error.fit(x,y_error)
    coefficients_out={f:float(v) for f,v in zip(FEATURES,final[-1].coef_)}
    calibration={"feature_order":list(FEATURES),"mean":final[0].mean_.tolist(),"scale":final[0].scale_.tolist(),"coherence_intercept":float(final[-1].intercept_),"coherence_coefficients":final[-1].coef_.tolist(),"relative_error_intercept":float(final_error[-1].intercept_),"relative_error_coefficients":final_error[-1].coef_.tolist()}
    error_mae=float(np.mean(abs(pred_error-y_error)));error_rho=float(spearmanr(pred_error,y_error).statistic)
    summary={"status":"complete","scope":"E25 stationary synthetic coarsening; indicators use only 20--28 us pre-training reference statistics and targets are held-out 29--30 us model metrics","features":list(FEATURES),"grouped_leave_one_resolution_out":{"coherence_mae":mae,"coherence_spearman_rho":rho,"recoverable_auc":auc,"relative_error_mae":error_mae,"relative_error_spearman_rho":error_rho,"folds":folds},"univariate_correlations":correlations,"full_calibration_standardized_coefficients":coefficients_out,"calibration":calibration,"score_definition":"group-calibrated predicted complex coherence; recoverable threshold=0.5","claim_boundary":"This is a same-condition empirical indicator, not yet a universal law across Ez, ion species, PPC, dt, or native coarse PIC."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6,5));
    for label,_,_ in CONFIGS:
        m=groups==label;ax.scatter(y[m],pred[m],label=label,s=34)
    ax.plot([0,1],[0,1],"k--",lw=1);ax.axvline(.5,color="gray",lw=.8);ax.axhline(.5,color="gray",lw=.8);ax.set(xlabel="observed model coherence",ylabel="pre-training recoverability score",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/"predicted_vs_observed.png",dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
