"""Три сида: одиночные вклады и усреднение. Протокол тот же (гребень, OOF)."""
import numpy as np, polars as pl
from pathlib import Path
mdl_amber = Path('/mnt/user-data/uploads/ozon_cup/from_gpu/tfm4_probe')
mdl_gabbro = Path('/mnt/user-data/uploads/ozon_cup/from_gpu/tfm4_seeds23')
d = pl.read_parquet('/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet').sort('user_id')
uid = d['user_id'].to_numpy()
ly = np.log1p(np.clip(d['target'].to_numpy().astype(np.float64), 0, None))
lb = d['blend'].to_numpy().astype(np.float64)
models = [c for c in d.columns if c not in ('user_id','target','blend')]
M = np.stack([d[c].to_numpy().astype(np.float64) for c in models], 1)
def rm(p): return float(np.sqrt(np.mean((p-ly)**2)))
sb, eb = rm(lb), lb-ly

def fit_shifts(lp, lt, bins=24):
    qs=np.quantile(lp,np.linspace(0,1,bins+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    c,s=[],[]
    for i in range(bins):
        m=(lp>qs[i])&(lp<=qs[i+1])
        if m.sum()<500: continue
        c.append(lp[m].mean()); s.append(lt[m].mean()-lp[m].mean())
    return np.array(c), np.array(s)
def app(lp,c,s): return np.clip(lp+np.interp(lp,c,s),0,None)

def load(tag, sfx=''):
    root = mdl_amber if '_s1' in tag else mdl_gabbro
    lp = np.load(root/f'val_logpred_{tag}{sfx}.npy').astype(np.float64)
    u = np.load(root/f'val_user_ids_{tag}.npy'); o = np.argsort(u)
    assert np.array_equal(u[o], uid), tag
    return lp[o]

joints = {s: load(f'tfm4_a_s{s}') for s in (1,2,3)}
ctrls  = {s: load(f'tfm4off_a_s{s}') for s in (1,2,3)}
avg = lambda ds: sum(ds.values())/len(ds)
cands = {}
for s in (1,2,3):
    cands[f'ствол сид {s}'] = trunks[s]
    cands[f'joint сид {s}'] = joints[s]
    cands[f'контроль сид {s}'] = ctrls[s]
cands['СТВОЛ среднее 3 сидов'] = avg(trunks)
cands['joint среднее 3 сидов'] = avg(joints)
cands['контроль среднее 3'] = avg(ctrls)

print(f'эталон blend {sb:.6f}\n')
print(f"{'':26} {'скор':>9} {'после кал.':>10} {'r ошибок':>9} {'ЗАПАС':>9}")
for n,lp in cands.items():
    c,s_=fit_shifts(lp,ly); lpc=app(lp,c,s_)
    sm=rm(lpc); rho=float(np.corrcoef(lpc-ly,eb)[0,1])
    print(f'  {n:24} {rm(lp):9.6f} {sm:10.6f} {rho:9.5f} {sb/sm-rho:+9.5f}')

def oof(extras, seed, folds=5, alpha=1e-3):
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

base=np.array([oof([],s) for s in range(5)])
print(f'\nбаза (25 моделей пака) {base.mean():.6f}')
print('вклад, OOF 5 фолдов x 5 разбиений, гребень 1e-3:')
rng0=np.random.default_rng(0)
sets = list(cands.items()) + [
    ('СТВОЛ ср. + joint ср.', [avg(trunks), avg(joints)]),
    ('СТВОЛ ср. + контроль ср.', [avg(trunks), avg(ctrls)]),
    ('НУЛЬ: перемешанный ствол', [avg(trunks)[rng0.permutation(len(uid))]]),
]
for n,ex in sets:
    ex = ex if isinstance(ex, list) else [ex]
    g=np.array([base[s]-oof(ex,s) for s in range(5)])
    print(f'  {n:28} {g.mean():+.6f}  разброс {g.std():.6f}  {abs(g.mean())/0.000022:5.0f} ш.е.')

print('\nдополнительно:')
extra_sets = [
    ('все три средних', [avg(trunks), avg(joints), avg(ctrls)]),
    ('ствол ср. + joint ср., без kostya46_cal в базе', None),
]
g=np.array([base[s]-oof([avg(trunks),avg(joints),avg(ctrls)],s) for s in range(5)])
print(f'  {"все три средних":34} {g.mean():+.6f}  {abs(g.mean())/0.000022:.0f} ш.е.')
j46=models.index('kostya46_cal'); keep=[i for i in range(len(models)) if i!=j46]
M_full=M
def oof_sub(cols, extras, seed):
    global M
    M=M_full[:,cols]; r=oof(extras,seed); M=M_full; return r
b_wo=np.array([oof_sub(keep,[],s) for s in range(5)])
g2=np.array([b_wo[s]-oof_sub(keep,[avg(trunks),avg(joints)],s) for s in range(5)])
print(f'  база без kostya46_cal {b_wo.mean():.6f}, + ствол ср. и joint ср. вклад {g2.mean():+.6f}  '
      f'{abs(g2.mean())/0.000022:.0f} ш.е.')
print(f'  итоговый OOF-скор с двумя членами: {(b_wo-g2).mean():.6f} против базы {base.mean():.6f}')
