"""S5. Кандидаты на 3 попытки. База — честный T2_tfm4_orth_045 (1.6469639).

Ключевое свойство: ПРИМЕНЕНИЕ ОСИ ОДНОВРЕМЕННО ЕЁ ИЗМЕРЯЕТ.
   S² = F0² − 2bc + b²q  ->  κ = c/q = (F0² + b²q − S²)/(2bq)
Поэтому не нужно тратить попытку на «чистую» пробу: ставим сразу приорную дозу,
получаем и выигрыш, и точный замер κ для следующего шага.
"""
import json, numpy as np, polars as pl
z=np.load("../zhenya/out/lb_full.npz"); M=json.load(open("../zhenya/out/lb_meta.json"))
SC=M["sc"]; uid=z["uid"]
BASE="T2_tfm4_orth_045"; base=z[f"lp_{BASE}"]; F0=SC[BASE]
KAPPA_PRIOR, TAU = 0.333, 0.205
Q_TARGET = 0.015
print(f"база {BASE} = {F0:.7f},  min(lp)={base.min():.4f}")
print(f"приор κ = N({KAPPA_PRIOR}, {TAU}²), доза = приорное среднее\n")

# перестраиваем направления: ортогонализуем к ЭТОЙ базе и к константе
raw={nm: pl.read_parquet(f"../zhenya/out/dir_{nm}.parquet").sort("user_id")["step"].to_numpy()
     for nm in ("Y1_ya_full","Y2_pre8mar","Y3_feb23","Y4_post8mar")}
vecs=[np.ones(len(uid)), base-base.mean()]
DIRS={}
for nm,d in raw.items():
    h=d.copy()
    for v in vecs:
        h=h-v*float(np.dot(h,v))/float(np.dot(v,v))
    h=h*np.sqrt(Q_TARGET/float(np.mean(h*h)))
    DIRS[nm]=h; vecs.append(h)

print(f"{'направление':14s} {'q':>9} {'доза b':>8} {'клип':>6} {'ожид. скор':>12} {'ожид. выигрыш':>14}")
out={}
for nm,h in DIRS.items():
    q=float(np.mean(h*h)); b=KAPPA_PRIOR
    lp=base+b*h; neg=int((lp<0).sum()); lp=np.clip(lp,0,None)
    # ожидание при κ=приор: S² = F0² − 2b·κq + b²q
    S=np.sqrt(F0**2 - 2*b*KAPPA_PRIOR*q + b*b*q)
    out[nm]=(lp,q,b)
    print(f"{nm:14s} {q:>9.5f} {b:>8.3f} {neg:>6} {S:>12.7f} {F0-S:>+14.6f}")

print(f"\n=== РИСК ===")
for nm,(lp,q,b) in out.items():
    loss=b*b*q/(2*F0)
    print(f"  {nm:14s} если κ=0: {F0+loss:.7f} ({-loss:+.6f});  "
          f"P(κ<0)={100*0.5*(1+__import__("math").erf(-KAPPA_PRIOR/(TAU*np.sqrt(2)))):.1f}%")
print(f"\n=== ФАЙЛЫ ===")
import pathlib; pathlib.Path("cand").mkdir(exist_ok=True)
for nm,(lp,q,b) in out.items():
    p=f"cand/Z1_{nm}.csv"
    pl.DataFrame({"user_id":uid,"predict":np.expm1(lp)}).write_csv(p)
    print(f"  {p}  строк {len(uid):,}  отрицательных {int((np.expm1(lp)<0).sum())}")
