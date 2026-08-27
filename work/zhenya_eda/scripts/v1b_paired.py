"""V1b. Та же поправка, но ПАРНЫЕ сиды и кэш признаков.
При парном сравнении шум сида сокращается: обе руки видят одну инициализацию."""
import os, numpy as np, lightgbm as lgb, importlib.util
from pathlib import Path
OUT=Path(os.environ["ZH_OUT"]); CA=OUT/"erafix_feat.npz"
src=open("../zhenya/scripts/v1_erafix.py",encoding="utf-8").read().split("res = {}")[0]
ns={}; exec(compile(src,"v1","exec"),ns)
build,TRAIN,VAL,calibrated=ns["build"],ns["TRAIN"],ns["VAL"],ns["calibrated"]
if CA.exists():
    z=np.load(CA); D={k:z[k] for k in z.files}; print("признаки из кэша")
else:
    D={}
    for fix in (False,True):
        Xs,ys=[],[]
        for a in TRAIN:
            M,y,_=build(a,fix); Xs.append(M); ys.append(y)
        Xv,yv,_=build(VAL,fix)
        D[f"Xtr{int(fix)}"]=np.vstack(Xs); D[f"ytr{int(fix)}"]=np.concatenate(ys)
        D[f"Xv{int(fix)}"]=Xv; D["yv"]=yv
        print(f"построено fix={fix}",flush=True)
    np.savez(CA,**D)
SEEDS=[42,555,1337,7,2024,101,314,777]
print(f"\n{'сид':>6} {'без поправки':>14} {'с поправкой':>13} {'разность':>11}")
ds=[]
for s in SEEDS:
    row=[]
    for fix in (0,1):
        m=lgb.LGBMRegressor(objective="tweedie",tweedie_variance_power=1.45,learning_rate=.05,
            num_leaves=63,min_child_samples=100,subsample=.8,colsample_bytree=.8,
            n_estimators=700,verbose=-1,n_jobs=4,random_state=s).fit(D[f"Xtr{fix}"],D[f"ytr{fix}"])
        lp=np.clip(m.predict(D[f"Xv{fix}"]),0,None)
        row.append(calibrated(lp,D["yv"])[0])
    ds.append(row[1]-row[0])
    print(f"{s:>6} {row[0]:>14.6f} {row[1]:>13.6f} {row[1]-row[0]:>+11.6f}",flush=True)
ds=np.array(ds)
print(f"\nПАРНАЯ разность: среднее {ds.mean():+.6f}  sd {ds.std(ddof=1):.6f}  "
      f"SE {ds.std(ddof=1)/np.sqrt(len(ds)):.6f}")
print(f"  t = {ds.mean()/(ds.std(ddof=1)/np.sqrt(len(ds))):+.2f} при {len(ds)-1} ст.св.")
print(f"  знак в пользу поправки: {int((ds<0).sum())} из {len(ds)}")
print(f"  порог шума 0.000022; порог приёмки 0.0003")
