"""Что дают три новых конфига поверх старой пары tfm4. Протокол measure3 + плацебо."""
import numpy as np, polars as pl, itertools
from pathlib import Path
mdl_amber=Path('/mnt/user-data/uploads/ozon_cup/from_gpu/tfm4_probe')
mdl_gabbro=Path('/mnt/user-data/uploads/ozon_cup/from_gpu/tfm4_seeds23')
mdl_halite=Path('/mnt/user-data/uploads/ozon_cup/from_gpu/tfm4_s17_s23_s777')
d=pl.read_parquet('/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet').sort('user_id')
uid=d['user_id'].to_numpy(); ly=np.log1p(np.clip(d['target'].to_numpy().astype(np.float64),0,None))
models=[c for c in d.columns if c not in ('user_id','target','blend')]
M=np.stack([d[c].to_numpy().astype(np.float64) for c in models],1)
rm=lambda p: float(np.sqrt(np.mean((p-ly)**2)))
def fit_shifts(lp,lt,bins=24):
    qs=np.quantile(lp,np.linspace(0,1,bins+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    c,s=[],[]
    for i in range(bins):
        m=(lp>qs[i])&(lp<=qs[i+1])
        if m.sum()<500: continue
        c.append(lp[m].mean()); s.append(lt[m].mean()-lp[m].mean())
    return np.array(c),np.array(s)
app=lambda lp,c,s: np.clip(lp+np.interp(lp,c,s),0,None)
ROOT={1:mdl_amber,2:mdl_gabbro,3:mdl_gabbro,17:mdl_halite,23:mdl_halite,777:mdl_halite}
def load(s,sfx=''):
    r=ROOT[s]
    lp=np.load(r/f'val_logpred_tfm4_a_s{s}{sfx}.npy').astype(np.float64)
    u=np.load(r/f'val_user_ids_tfm4_a_s{s}.npy'); o=np.argsort(u)
    assert np.array_equal(u[o],uid), s
    return lp[o]
OLD=[1,2,3]; NEW=[17,23,777]
cal=lambda lp:(lambda cs: app(lp,*cs))(fit_shifts(lp,ly))
EJ={s:cal(J[s])-ly for s in J}
print("КОРРЕЛЯЦИЯ ОШИБОК joint (после калибровки)")
print("      "+"".join(f"{s:>8}" for s in OLD+NEW))
for a in OLD+NEW:
    print(f"{a:>5} "+"".join(f"{np.corrcoef(EJ[a],EJ[b])[0,1]:>8.4f}" for b in OLD+NEW))
avg=lambda ds: sum(ds)/len(ds)
def oof(extras,seed,folds=5,alpha=1e-3):
    rng=np.random.default_rng(seed); f=rng.integers(0,folds,len(uid)); out=np.zeros(len(uid))
    for k in range(folds):
        tr,te=f!=k,f==k
        ctr,cte=[M[tr]],[M[te]]
        for e in extras:
            c,s_=fit_shifts(e[tr],ly[tr])
            ctr.append(app(e[tr],c,s_)[:,None]); cte.append(app(e[te],c,s_)[:,None])
        A=np.column_stack(ctr+[np.ones(tr.sum())]); B=np.column_stack(cte+[np.ones(te.sum())])
        G=A.T@A+alpha*len(A)*np.eye(A.shape[1]); G[-1,-1]-=alpha*len(A)
        out[te]=B@np.linalg.solve(G,A.T@ly[tr])
    return rm(out)
base=np.mean([oof([],s) for s in range(5)])
rng0=np.random.default_rng(0); perm=rng0.permutation(len(uid))
pair=lambda ss: [avg([T[s] for s in ss]), avg([J[s] for s in ss])]
sets=[("СТАРАЯ ОСЬ: сиды 1,2,3", pair(OLD))]
for s in NEW: sets.append((f"  1,2,3 + конфиг {s}", pair(OLD+[s])))
sets += [("ВСЕ ШЕСТЬ", pair(OLD+NEW)),
         ("только новые 17,23,777", pair(NEW)),
         ("НУЛЬ: перемешанная пара", [pair(OLD)[0][perm], pair(OLD)[1][perm]])]
print(f"\nВКЛАД В БЛЕНД (база пака {base:.6f}, OOF 5x5, гребень 1e-3)")
print(f"{'набор':<28}{'G(вал)':>11}{'разброс':>10}{'Δ к старой оси':>16}{'порог 0.00005':>15}")
G0=None
for nm,ex in sets:
    g=np.array([base-oof(ex,s) for s in range(5)])
    if G0 is None: G0=g.mean()
    dd=g.mean()-G0
    ok='' if nm.startswith(('СТАРАЯ','НУЛЬ')) else ('ПРОХОДИТ' if dd>=0.00005 else 'не проходит')
    print(f"  {nm:<26}{g.mean():>+11.6f}{g.std():>10.6f}{dd:>+16.6f}{ok:>15}")

print("\nКОРРЕЛЯЦИЯ ОШИБОК СТВОЛОВ (они разошлись сильнее: 1.82..2.04)")
ET={s:cal(T[s])-ly for s in T}
print("      "+"".join(f"{s:>8}" for s in OLD+NEW))
for a in OLD+NEW:
    print(f"{a:>5} "+"".join(f"{np.corrcoef(ET[a],ET[b])[0,1]:>8.4f}" for b in OLD+NEW))
extra=[
 ("6 членов по отдельности (joint)", [J[s] for s in OLD+NEW]),
 ("6 стволов по отдельности",        [T[s] for s in OLD+NEW]),
 ("старая пара + ствол 777",         pair(OLD)+[T[777]]),
 ("старая пара + ствол 23",          pair(OLD)+[T[23]]),
 ("старая пара + joint 17",          pair(OLD)+[J[17]]),
 ("старая пара + все 3 новых ствола",pair(OLD)+[T[s] for s in NEW]),
 ("НУЛЬ: 3 перемешанных ствола",     pair(OLD)+[T[s][perm] for s in NEW]),
]
print(f"\n{'набор':<36}{'G(вал)':>11}{'разброс':>10}{'Δ к старой оси':>16}")
for nm,ex in extra:
    g=np.array([base-oof(ex,s) for s in range(5)])
    print(f"  {nm:<34}{g.mean():>+11.6f}{g.std():>10.6f}{g.mean()-G0:>+16.6f}")

# Последняя проверка: не лучше ли новые конфиги СТАРЫХ, а не в дополнение.
# Правило задано заранее (три лучших по одиночному калиброванному скору), без перебора.
solo={1:1.670399,2:1.669439,3:1.669774,17:1.670365,23:1.669292,777:1.668888}
best3=sorted(solo,key=solo.get)[:3]
print(f"\nтри лучших по одиночному скору: {best3}")
for nm,ss in (("замена: три лучших", best3), ("старая тройка 1,2,3", OLD)):
    g=np.array([base-oof(pair(ss),s) for s in range(5)])
    print(f"  {nm:<34}{g.mean():>+11.6f}{g.std():>10.6f}{g.mean()-G0:>+16.6f}")
