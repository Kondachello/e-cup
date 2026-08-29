"""Калибровка вал-каппы: informative ли знак κ_val? Сравниваем κ_val со ЗАМЕРЕННОЙ
κ (kappa_pair из ) по всем 47 осям."""
import json, os, sys
import numpy as np, polars as pl
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","4")
ROOT=Path("/Users/alexanderkondakov/ozon-cup"); SUB=ROOT/"submissions"; CANON=SUB/"canonical"
SP=Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
sys.path.insert(0,str(ROOT/"work"/"scripts")); import predict_lb as plb
MEAS={n:s for n,_,s in plb.MEASURED}
def lp(fn):
    for p in (SUB/fn,CANON/fn):
        if p.exists():
            d=pl.read_csv(p,schema_overrides={"user_id":pl.Int64}).sort("user_id")
            return np.log1p(np.clip(d["predict"].to_numpy().astype(np.float64),0,None))
    raise FileNotFoundError(fn)
base=pl.read_csv(SUB/"F8_priv.csv",schema_overrides={"user_id":pl.Int64}).sort("user_id")
uid_ref=base["user_id"].to_numpy()
OLD32=["mdl_amber","mdl_gabbro","mdl_halite","mdl_marble","mdl_realgr","mdl_tektit","mdl_olivin","mdl_flint","mdl_gypsum","mdl_gneis2","mdl_malach","","mdl_vivian","mdl_corund","mdl_larvik","mdl_talc",
 "","","","","","seg_realgr",""]
N5=[]
N6=[]
L5,L6=lp("F5_priv.csv"),lp("F6_priv.csv")
names,cols=[],[]
for k in OLD32:
    names.append(k); cols.append(pl.read_parquet(DIRS/f"{k}.parquet").sort("user_id")["d"].to_numpy().astype(np.float64))
for f in N5: names.append(f.split("_")[0]); cols.append(lp(f+".csv")-L5)
for f in N6: names.append(f.split("_")[0]); cols.append(lp(f+".csv")-L6)
names.append("mdl_wulfen"); cols.append(lp("N1_ktpp.csv")-L6)
v=pl.read_parquet(ROOT/"work/features/anchor=2026-01-14.parquet",columns=["user_id","target"]).sort("user_id")
bv=pl.read_parquet(ROOT/"work/preds/blend_opt_val.parquet").sort("user_id")
r=np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))-np.log1p(np.clip(bv["pred"].to_numpy().astype(np.float64),0,None))
kv,km,lab=[],[],[]
for nm,d in zip(names,cols):
    dc=d-d.mean(); q=float(np.mean(dc*dc))
    a=float(np.mean(dc*r)/q)
    if nm in K: kv.append(a); km.append(float(K[nm])); lab.append(nm)
kv=np.array(kv); km=np.array(km)
print(f"пар осей с обеими каппами: {len(kv)}")
print(f"corr(κ_val, κ_замер) = {np.corrcoef(kv,km)[0,1]:+.4f}   "
      f"совпадение знака: {float((np.sign(kv)==np.sign(km)).mean()):.3f}")
seg=[i for i,n in enumerate(lab) if n.startswith("P")]
print(f"только сегментные (P*): n={len(seg)} corr={np.corrcoef(kv[seg],km[seg])[0,1]:+.4f} "
      f"знак совпал {float((np.sign(kv[seg])==np.sign(km[seg])).mean()):.3f}")
print(f"\n{'ось':8s}{'κ_val':>9}{'κ_замер':>10}")
for i in np.argsort([lab[j] for j in range(len(lab))]):
    print(f"{lab[i]:8s}{kv[i]:+9.3f}{km[i]:+10.3f}")
