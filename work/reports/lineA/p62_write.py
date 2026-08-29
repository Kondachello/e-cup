"""P62: due-diligence по финалисту + запись направления в work/data/."""
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
never=t["nbuyd"].to_numpy()==0; s30=t["searches30"].to_numpy().astype(np.float64)

BANDS=[(31,45,"31-45"),(46,60,"46-60"),(61,90,"61-90"),(91,180,"91-180"),(181,365,"181-365"),(366,1e8,"366+")]
mask=(rec>=31)&(~never)
v0=np.zeros(len(s30)); rows=[]
for lo,hi,nm in BANDS:
    b=mask&(rec>=lo)&(rec<=hi); xs=s30[b]; thr=float(np.median(xs))
    ge=(xs>=thr).mean(); gt=(xs>thr).mean(); use_ge=abs(ge-.5)<=abs(gt-.5)
    hi_m=(s30>=thr) if use_ge else (s30>thr)
    v0[b&hi_m]=1.; v0[b&~hi_m]=-1.
    rows.append((nm,int(b.sum()),float(b.mean()),thr,">=" if use_ge else ">",
                 float((b&hi_m).sum()/b.sum()),float((xs==0).mean())))
print(f"{'полоса':10s}{'юзеров':>9}{'доля':>8}{'порог':>9}{'оп':>4}{'доля H':>9}{'доля s30=0':>12}")
for a in rows: print(f"{a[0]:10s}{a[1]:9d}{a[2]:8.4f}{a[3]:9.1f}{a[4]:>4s}{a[5]:9.3f}{a[6]:12.4f}")

d=STEP*(v0-v0.mean())
q=float(np.mean(d*d))
raw=lpF8+d; probe=np.clip(raw,0,None); nclip=int((raw<0).sum())
dr=probe-lpF8; drc=dr-dr.mean()
q2=float(np.mean(drc*drc)); qr2=float(np.mean((drc-U@(U.T@drc))**2))
h=drc/np.sqrt(q2); g=float(np.std(h*r)/F0v)
sig=g*F8*np.sqrt(FPC2/(N_PUB*qr2)); w=TAU**2/(TAU**2+sig**2)
gain=(w*TAU**2+MU**2)*qr2/(2*F8); nov=float(np.sqrt(qr2/q2))
sig_raw=g*F8*np.sqrt(FPC2/(N_PUB*q2))
print(f"\nq={q:.6f}  реализованный q={q2:.6f}  q_ост={qr2:.6f}  novelty(норма)={nov:.4f} "
      f"novelty(дисперсия)={qr2/q2:.4f}")
print(f"обрезка нулём: {nclip} строк ({nclip/len(d)*100:.3f}%)")
print(f"локальный g={g:.4f}  σ_κ(по q)={sig_raw:.4f}  σ_κ(по q_ост)={sig:.4f}  w={w:.4f}")
print(f"E[gain]=(wτ²+μ²)q_ост/(2F0)={gain:.7f} = {gain/NOISE:.2f} шума   [гейт: q_ост>=0.015 "
      f"{'✓' if qr2>=0.015 else '✗'}, novelty>=0.5 {'✓' if nov>=0.5 else '✗'}]")
print(f"sd(log1p): база {lpF8.std():.6f} -> зонд {probe.std():.6f};  mean {lpF8.mean():.6f} -> {probe.std()*0+probe.mean():.6f}")
print("парабола S²=F0²+q−2κq (полный шаг b=1):")
for k in (0.0,0.1,0.2,0.333,0.5):
    s=float(np.sqrt(F8**2+q2-2*k*q2)); print(f"   κ={k:<6} S={s:.7f} ({s-F8:+.7f})")

rr=(A.T@d)/np.sqrt(np.sum(A*A,0)*np.sum(d*d))
print("\nкорреляции с ключевыми осями:")
for nm in ("","","","","","","","","seg_realgr","mdl_gneis2","mdl_tektit","mdl_talc","mdl_amber"):
    print(f"   {nm:6s}{float(rr[names.index(nm)]):+.4f}", end="")
print()
print("все |corr|>0.15:", ", ".join(f"{names[i]}{rr[i]:+.3f}" for i in np.argsort(-np.abs(rr)) if abs(rr[i])>0.15))
cl=float(np.corrcoef(d,lpF8)[0,1]); print(f"corr(d, lp_F8) = {cl:+.4f} (проверка: не уровень/спред)")

# контаминированный вал-каппа (окно признаков = окно вал-таргета) — только справочно
kv=float(np.mean(drc*r)/q2)
print(f"κ_val (СПРАВОЧНО, ЗАГРЯЗНЕНО: окно признаков 15.01-13.02 = окно вал-таргета) = {kv:+.4f}")

outp=ROOT/"work/data/dir_P62_sleepmass.parquet"
if outp.exists(): print("СТОП: файл уже есть:", outp)
else:
    pl.DataFrame({"user_id":uid_ref,"d":d}).write_parquet(outp); print("\nзаписано направление:",outp)
json.dump(dict(q=q,q_real=q2,q_res=qr2,nov_norm=nov,nov_var=qr2/q2,n_clip=nclip,g=g,
               sigma_res=sig,sigma_raw=sig_raw,w=w,gain=gain,V=gain/NOISE,
               bands=[dict(band=a[0],n=a[1],share=a[2],thr=a[3],op=a[4],fH=a[5],zero=a[6]) for a in rows],
               corr={names[i]:float(rr[i]) for i in range(len(names))},
               corr_lp=cl,kappa_val_contaminated=kv,path=str(outp)),
          open(SP/"p62_winner.json","w"),ensure_ascii=False,indent=1,default=float)
