"""Уточнённый замер: гребневая регуляризация + честные нули."""
import numpy as np, polars as pl
from pathlib import Path
P = Path('/mnt/user-data/uploads/ozon_cup/from_gpu/tfm4_probe')
d = pl.read_parquet('/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet').sort('user_id')
uid = d['user_id'].to_numpy()
ly = np.log1p(np.clip(d['target'].to_numpy().astype(np.float64), 0, None))
lb = d['blend'].to_numpy().astype(np.float64)
models = [c for c in d.columns if c not in ('user_id','target','blend')]
M = np.stack([d[c].to_numpy().astype(np.float64) for c in models], 1)
def rm(p): return float(np.sqrt(np.mean((p-ly)**2)))
sb = rm(lb)

def fit_shifts(lp, lt, bins=24):
    qs = np.quantile(lp, np.linspace(0,1,bins+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    c,s=[],[]
    for i in range(bins):
        m=(lp>qs[i])&(lp<=qs[i+1])
        if m.sum()<500: continue
        c.append(lp[m].mean()); s.append(lt[m].mean()-lp[m].mean())
    return np.array(c), np.array(s)
def app(lp,c,s): return np.clip(lp+np.interp(lp,c,s),0,None)
def load(tag, sfx=''):
    lp=np.load(P/f'val_logpred_{tag}{sfx}.npy').astype(np.float64)
    u=np.load(P/f'val_user_ids_{tag}.npy'); o=np.argsort(u)
    assert np.array_equal(u[o],uid); return lp[o]


# --- кто такой kostya46_cal ---
print('корреляции с колонками пака (по предсказаниям, лог-пространство):')
for c in ('kostya46_cal','kostya46','gseq_big_s42_cal'):
    j=models.index(c)
    print(f'  {c:20s} vs tfm4off {np.corrcoef(M[:,j],ctrl)[0,1]:.4f}   vs ствол {np.corrcoef(M[:,j],trunk)[0,1]:.4f}')
print(f'\nвзаимные корреляции кандидатов: joint-ctrl {np.corrcoef(joint,ctrl)[0,1]:.4f}  '
      f'joint-ствол {np.corrcoef(joint,trunk)[0,1]:.4f}  ctrl-ствол {np.corrcoef(ctrl,trunk)[0,1]:.4f}')

def oof(extras, seed, folds=5, alpha=1e-3):
    """extras: список сырых лог-предсказаний кандидатов (может быть пустым)."""
    rng=np.random.default_rng(seed); f=rng.integers(0,folds,len(uid))
    out=np.zeros(len(uid))
    for k in range(folds):
        tr,te=f!=k,f==k
        cols_tr=[M[tr]]; cols_te=[M[te]]
        for e in extras:
            c,s=fit_shifts(e[tr],ly[tr])
            cols_tr.append(app(e[tr],c,s)[:,None]); cols_te.append(app(e[te],c,s)[:,None])
        A=np.column_stack(cols_tr+[np.ones(tr.sum())])
        B=np.column_stack(cols_te+[np.ones(te.sum())])
        # гребень: без него дублирующая колонка делает матрицу вырожденной,
        # и минимально-нормовое решение lstsq само по себе меняет качество
        G=A.T@A + alpha*len(A)*np.eye(A.shape[1]); G[-1,-1]-=alpha*len(A)
        w=np.linalg.solve(G, A.T@ly[tr])
        out[te]=B@w
    return rm(out)

rng0=np.random.default_rng(0)
tests = {
    'tfm4 (с таблицей)': [joint],
    'tfm4off (контроль)': [ctrl],
    'ствол tfm4 без таблицы': [trunk],
    'ствол + joint вместе': [trunk, joint],
    'ствол + контроль вместе': [trunk, ctrl],
    'НУЛЬ: точный дубль модели пака': [M[:,0].copy()],
    'НУЛЬ: перемешанный ствол': [trunk[rng0.permutation(len(uid))]],
}
print('\nOOF 5 фолдов по юзерам, гребень 1e-3, калибровка внутри фолда, 5 разбиений:')
base=np.array([oof([],s) for s in range(5)])
print(f'  база (25 моделей пака)        {base.mean():.6f}')
for name,ex in tests.items():
    g=np.array([base[s]-oof(ex,s) for s in range(5)])
    print(f'  {name:31} прирост {g.mean():+.6f}  разброс {g.std():.6f}  '
          f'{abs(g.mean())/0.000022:5.0f} ш.е.')

# --- сколько стоит уже стоящий в паке трансформер, и можно ли его заменить ---
j46 = models.index('kostya46_cal')
keep = [i for i in range(len(models)) if i != j46]
M_full = M
def oof_base(cols, extras, seed, folds=5, alpha=1e-3):
    global M
    M = M_full[:, cols]
    r = oof(extras, seed, folds, alpha)
    M = M_full
    return r
print('\nконтекст: чего стоит трансформер, который УЖЕ в паке (kostya46_cal)')
b_wo = np.array([oof_base(keep, [], s) for s in range(5)])
b_all = np.array([oof_base(list(range(len(models))), [], s) for s in range(5)])
print(f'  база без kostya46_cal          {b_wo.mean():.6f}')
print(f'  + вернуть kostya46_cal         {b_all.mean():.6f}   вклад {(b_wo-b_all).mean():+.6f}  '
      f'{abs((b_wo-b_all).mean())/0.000022:.0f} ш.е.')
for nm, ex in (('+ вместо него ствол tfm4', [trunk]), ('+ вместо него joint', [joint]),
               ('+ ствол И kostya46_cal', None)):
    if ex is None:
        g = np.array([b_wo[s] - oof_base(list(range(len(models))), [trunk], s) for s in range(5)])
    else:
        g = np.array([b_wo[s] - oof_base(keep, ex, s) for s in range(5)])
    print(f'  {nm:30} вклад {g.mean():+.6f}  {abs(g.mean())/0.000022:.0f} ш.е.')
