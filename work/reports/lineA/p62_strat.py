"""P62: РЕЦЕНСИ-СТРАТИФИЦИРОВАННЫЙ медианный сплит — контраст «браузинга» внутри
/ — индикаторы полос), значит должен дать больший q_ост."""
import json, os, sys
import numpy as np, polars as pl
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","4")
ROOT=Path("/Users/alexanderkondakov/ozon-cup"); SUB=ROOT/"submissions"; CANON=SUB/"canonical"
SP=Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
sys.path.insert(0,str(ROOT/"work"/"scripts")); import predict_lb as plb
MEAS={n:s for n,_,s in plb.MEASURED}
STEP,N_PUB,FPC2,NOISE=0.30,50_000,0.8,0.000022
F8=MEAS["F8_priv"]; MU,TAU=0.012,0.141
def lp(fn):
    for p in (SUB/fn,CANON/fn):
        if p.exists():
            d=pl.read_csv(p,schema_overrides={"user_id":pl.Int64}).sort("user_id")
            return np.log1p(np.clip(d["predict"].to_numpy().astype(np.float64),0,None))
    raise FileNotFoundError(fn)
base=pl.read_csv(SUB/"F8_priv.csv",schema_overrides={"user_id":pl.Int64}).sort("user_id")
uid_ref=base["user_id"].to_numpy()
lpF8=np.log1p(np.clip(base["predict"].to_numpy().astype(np.float64),0,None))
OLD32=["mdl_amber","mdl_gabbro","mdl_halite","mdl_marble","mdl_realgr","mdl_tektit","mdl_olivin","mdl_flint","mdl_gypsum","mdl_gneis2","mdl_malach","","mdl_vivian","mdl_corund","mdl_larvik","mdl_talc",
 "","","","","","seg_realgr",""]
N5=[]
N6=[]
L5,L6=lp("F5_priv.csv"),lp("F6_priv.csv")
names,cols=[],[]
for k in OLD32:
    t=pl.read_parquet(DIRS/f"{k}.parquet").sort("user_id"); names.append(k)
    cols.append(t["d"].to_numpy().astype(np.float64))
for f in N5: names.append(f.split("_")[0]); cols.append(lp(f+".csv")-L5)
for f in N6: names.append(f.split("_")[0]); cols.append(lp(f+".csv")-L6)
names.append("mdl_wulfen"); cols.append(lp("N1_ktpp.csv")-L6)
A=np.stack(cols,1); A=A-A.mean(0,keepdims=True)
U,sv,_=np.linalg.svd(A,full_matrices=False); U=U[:,:int((sv>sv[0]*1e-10).sum())]
v=pl.read_parquet(ROOT/"work/features/anchor=2026-01-14.parquet",columns=["user_id","target"]).sort("user_id")
bv=pl.read_parquet(ROOT/"work/preds/blend_opt_val.parquet").sort("user_id")
r=np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))-np.log1p(np.clip(bv["pred"].to_numpy().astype(np.float64),0,None))
F0v=float(np.sqrt(np.mean(r*r)))
t=pl.read_parquet(SP/"p62_agg.parquet").sort("user_id")
rec=t["last_di"].to_numpy().astype(np.float64); rec=np.where(np.isnan(rec),1e9,rec)
never=t["nbuyd"].to_numpy()==0
XV={c:t[c].to_numpy().astype(np.float64) for c in ("act30","act90","browse30","browse90","searches30")}

def strat_contrast(mask, x, bands):
    """±1 контраст: медиана x считается ОТДЕЛЬНО в каждой полосе давности."""
    v0=np.zeros(len(x)); info=[]
    for lo,hi,nm in bands:
        b=mask&(rec>=lo)&(rec<=hi)
        if b.sum()<200: continue
        xs=x[b]; thr=float(np.median(xs))
        ge=(xs>=thr).mean(); gt=(xs>thr).mean()
        hi_m=(x>=thr) if abs(ge-0.5)<=abs(gt-0.5) else (x>thr)
        v0[b&hi_m]=1.0; v0[b&~hi_m]=-1.0
        info.append((nm,int(b.sum()),thr,float((b&hi_m).sum()/b.sum())))
    return v0,info

def report(v0,lab,extra=""):
    d=STEP*(v0-v0.mean()); q=float(np.mean(d*d))
    raw=lpF8+d; probe=np.clip(raw,0,None); nclip=int((raw<0).sum())
    dr=probe-lpF8; dr=dr-dr.mean()
    q2=float(np.mean(dr*dr)); qr2=float(np.mean((dr-U@(U.T@dr))**2))
    h=dr/np.sqrt(q2); g=float(np.std(h*r)/F0v)
    sig=g*F8*np.sqrt(FPC2/(N_PUB*qr2)); w=TAU**2/(TAU**2+sig**2)
    gain=(w*TAU**2+MU**2)*qr2/(2*F8); nov=float(np.sqrt(qr2/q2))
    rr=(A.T@d)/np.sqrt(np.sum(A*A,0)*np.sum(d*d))
    top=[(names[i],float(rr[i])) for i in np.argsort(-np.abs(rr))[:4]]
    gate=qr2>=0.015 and nov>=0.5
    print(f"{lab:52s}q={q:.5f} обрез={nclip:6d} q_real={q2:.5f} q_ост={qr2:.5f} nov={nov:.3f} "
          f"g={g:.3f} σ={sig:.4f} w={w:.3f} V={gain/NOISE:5.2f} {'ГЕЙТ ✓' if gate else 'гейт ✗'}")
    print(f"{'':6s}corr: "+", ".join(f"{a}{b:+.3f}" for a,b in top)+("  "+extra if extra else ""))
    return dict(label=lab,q=q,n_clip=nclip,q_real=q2,q_res=qr2,nov=nov,g=g,sigma=sig,w=w,
                gain=gain,V=gain/NOISE,top=top,gate=bool(gate),d=d)

BANDS_FINE=[(31,45,"31-45"),(46,60,"46-60"),(61,90,"61-90"),(91,180,"91-180"),
            (181,365,"181-365"),(366,1e8,"366+")]
BANDS_SLEEP=[(91,180,"91-180"),(181,365,"181-365"),(366,1e8,"366+")]
BANDS_46=[(46,60,"46-60"),(61,90,"61-90"),(91,180,"91-180"),(181,365,"181-365"),(366,1e8,"366+")]
res=[]
print("="*126)
for xn in ("searches30","browse90","act90","act30"):
    for bands,mtag,mm in [(BANDS_SLEEP,"спящие 91+ (страт. 3 полосы)",(rec>=91)&(~never)),
                          (BANDS_46,"rec>=46 (страт. 5 полос)",(rec>=46)&(~never)),
                          (BANDS_FINE,"rec>=31 (страт. 6 полос)",(rec>=31)&(~never))]:
        v0,info=strat_contrast(mm,XV[xn],bands)
        res.append(report(v0,f"{mtag} × {xn}",f"m={mm.mean():.4f}"))
        res[-1].update(xvar=xn,bands=[b[2] for b in bands],info=info,m=float(mm.mean()))
# + never отдельной полосой со своим знаком
print("-"*126)
for xn in ("searches30","browse90"):
    for sgn in (+1.,-1.):
        v0,_=strat_contrast((rec>=91)&(~never),XV[xn],BANDS_SLEEP)
        xs=XV[xn][never]; thr=float(np.median(xs))
        hi=(XV[xn]>=thr) if abs((xs>=thr).mean()-.5)<=abs((xs>thr).mean()-.5) else (XV[xn]>thr)
        v0[never&hi]=sgn; v0[never&~hi]=-sgn
        res.append(report(v0,f"спящие 91+ страт. + never({'+' if sgn>0 else '−'}) × {xn}"))
        res[-1].update(xvar=xn)
np.savez_compressed(SP/"p62_strat_dirs.npz",**{f"d_{i}":x["d"] for i,x in enumerate(res)})
json.dump([{k:v for k,v in x.items() if k!="d"} for x in res],open(SP/"p62_strat.json","w"),
          ensure_ascii=False,indent=1,default=float)
