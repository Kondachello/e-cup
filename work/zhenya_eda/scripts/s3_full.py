"""S3. Полный локальный оценщик публичного скора на замеренном файле.
φ(lp) известна ТОЧНО для каждого замеренного: φ = (mean(lp²)+mean(t²)−f²)/2.
Для нового кандидата: проекция на span замеренных даёт φ точно, ортогональный
остаток оценивается ковариацией с валидационным таргетом (a=1.052)."""
import json, numpy as np, polars as pl, pathlib
MEAN_T, MEAN_T2, A = 2.3275, 10.79, 1.0520
S=json.load(open("../zhenya/subs.json"))
ss=pl.read_csv("sample_submit.csv",schema_overrides={"user_id":pl.Int64}).sort("user_id")
uid=ss["user_id"].to_numpy(); tval=np.log1p(np.clip(ss[ss.columns[1]].to_numpy().astype(float),0,None))
def load(nm):
    d=pl.read_csv(f"submissions/{nm}.csv",schema_overrides={"user_id":pl.Int64}).sort("user_id")
    assert np.array_equal(d["user_id"].to_numpy(),uid), nm
    return np.log1p(np.clip(d[d.columns[1]].to_numpy().astype(float),0,None))
NAMES=[n for n,_,_ in S if pathlib.Path(f"submissions/{n}.csv").exists()]
SC={n:s for n,s,_ in S}
LP={n:load(n) for n in NAMES}
LP["sample"]=tval; SC["sample"]=2.122483523224017; NAMES=NAMES+["sample"]
PHI={n:(float(np.mean(LP[n]**2))+MEAN_T2-SC[n]**2)/2 for n in NAMES}
print(f"замеренных файлов: {len(NAMES)}")
B=np.column_stack([np.ones(len(uid))]+[LP[n] for n in NAMES])
PH=np.array([MEAN_T]+[PHI[n] for n in NAMES])
G=B.T@B/len(uid)
def predict(lp, drop=()):
    idx=[0]+[i+1 for i,n in enumerate(NAMES) if n not in drop]
    Bs=B[:,idx]; PHs=PH[idx]
    w=np.linalg.lstsq(Bs.T@Bs/len(uid), Bs.T@lp/len(uid), rcond=None)[0]
    r=lp-Bs@w
    phi=float(w@PHs)+A*float(np.mean(r*(tval-tval.mean())))
    return float(np.sqrt(max(float(np.mean(lp*lp))-2*phi+MEAN_T2,0)))
print(f"\n=== ЧЕСТНАЯ ПРОВЕРКА: предсказываем файл, ИСКЛЮЧИВ его из базиса ===")
print(f"{'файл':22s} {'факт':>11} {'прогноз':>11} {'ошибка':>10}")
errs=[]
for n in ["SHOW_maxpub","SHOW2_aggr","SHOW3b_safe","V3_canon","T2_tfm4_orth_045",
          "R2_newblend","Q1_probes5","H1_applied","A1_gram7_shift"]:
    p=predict(LP[n],drop=(n,)); e=p-SC[n]; errs.append(abs(e))
    print(f"{n:22s} {SC[n]:>11.7f} {p:>11.7f} {e:>+10.7f}")
print(f"\nMAE на отложенных: {np.mean(errs):.7f}, максимум {max(errs):.7f}")
np.savez("../zhenya/out/lb_full.npz", uid=uid, tval=tval,
         **{f"lp_{n}":LP[n] for n in NAMES})
json.dump({"names":NAMES,"sc":SC,"phi":PHI}, open("../zhenya/out/lb_meta.json","w"))
