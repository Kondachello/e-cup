"""Part C: differential predictability front vs tail.
C2: ridge OOF (5-fold users) on 30d-history features, per anchor, F vs T."""
import polars as pl
import numpy as np
import datetime as dt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import json

SP = '/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad'
u = pl.read_parquet(f'{SP}/user_windows.parquet')
vp = pl.read_parquet('/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet',
                     columns=['user_id', 'target', 'blend'])
df = vp.join(u, on='user_id', how='left')
b = df['blend'].to_numpy().astype(np.float64)

out = {}

def affine_r2(x, y):
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - (res ** 2).mean() / y.var(), np.sqrt((res ** 2).mean())

print('=== : blend as predictor of val halves (affine, in-sample 2 params) ===')
res_c1 = {}
for nm, col in [('F(1-15d)', 'va_F'), ('T(16-30d)', 'va_T'),
                ('S1(1-10d)', 'va_S1'), ('S2(11-20d)', 'va_S2'), ('S3(21-30d)', 'va_S3')]:
    yv = np.log1p(df[col].fill_null(0).to_numpy())
    r2, rmse = affine_r2(b, yv)
    nnz = (yv > 0).mean()
    res_c1[nm] = dict(r2=round(float(r2), 4), rmse=round(float(rmse), 4),
                      var=round(float(yv.var()), 4), nnz=round(float(nnz), 4))
    print(f'  {nm:10s} mdl_flint {r2:.4f}  RMSE {rmse:.4f}  var(y) {yv.var():.4f}  nnz {nnz:.4f}')
y30 = np.log1p(df['target'].to_numpy())
r2, rmse = affine_r2(b, y30)
print(f'  30d        mdl_flint {r2:.4f}  RMSE {rmse:.4f}  var(y) {y30.var():.4f}')
out['C1_blend_affine'] = res_c1

# corr between halves
yF = np.log1p(df['va_F'].fill_null(0).to_numpy()); yT = np.log1p(df['va_T'].fill_null(0).to_numpy())
print(f'  corr(yF,yT) {np.corrcoef(yF,yT)[0,1]:.4f}')
out['corr_val_halves'] = round(float(np.corrcoef(yF, yT)[0, 1]), 4)

print('\n=== C2: ridge OOF per anchor, same 30d-history features ===')
anchors = {'ta': '2025-02-13', 'c4': '2025-04-30', 'c6': '2025-06-30',
           'c9': '2025-09-30', 'va': '2026-01-14'}
kf = KFold(5, shuffle=True, random_state=42)
res_c2 = {}
for tag, a in anchors.items():
    ad = dt.date.fromisoformat(a)
    feats = []
    for c in [f'{tag}_gmv30', f'{tag}_ord30', f'{tag}_cart30', f'{tag}_srch30']:
        feats.append(np.log1p(df[c].fill_null(0).to_numpy()))
    feats.append(df[f'{tag}_act30'].fill_null(0).to_numpy().astype(float))
    la = df[f'{tag}_lastact'].to_list()
    gap = np.array([(ad - d).days if d is not None else 999 for d in la], dtype=float)
    feats.append(np.minimum(gap, 60.0))
    lg = df[f'{tag}_lastgmv'].to_list()
    gapg = np.array([(ad - d).days if d is not None else 999 for d in lg], dtype=float)
    feats.append(np.minimum(gapg, 400.0))
    feats.append((gap > 900).astype(float))
    X = np.column_stack(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r = {}
    for half, col in [('F', f'{tag}_F'), ('mdl_larvik', f'{tag}_T')]:
        yv = np.log1p(df[col].fill_null(0).to_numpy())
        oof = np.zeros_like(yv)
        for tr, te in kf.split(X):
            m = Ridge(alpha=10.0).fit(X[tr], yv[tr])
            oof[te] = m.predict(X[te])
        r2 = 1 - ((yv - oof) ** 2).mean() / yv.var()
        r[half] = dict(r2=round(float(r2), 4), rmse=round(float(np.sqrt(((yv-oof)**2).mean())), 4),
                       var=round(float(yv.var()), 4), nnz=round(float((yv > 0).mean()), 4))
    # ratio: how much tail predictability lags front
    r['T_over_F_r2'] = round(r['mdl_larvik']['r2'] / r['F']['r2'], 4)
    res_c2[f'{tag}_{a}'] = r
    print(f"  {tag} {a}:  F mdl_flint {r['F']['r2']:.4f} (nnz {r['F']['nnz']:.3f})   "
          f"T mdl_flint {r['mdl_larvik']['r2']:.4f} (nnz {r['mdl_larvik']['nnz']:.3f})   T/F {r['T_over_F_r2']:.3f}")
out['C2_ridge_oof'] = res_c2

# C2b: 10-day slices on va + ta (decay curve within window)
print('\n=== C2b: 10-day slice decay (ridge OOF, same features) ===')
res_c2b = {}
for tag in ['va', 'ta']:
    ad = dt.date.fromisoformat(anchors[tag])
    feats = []
    for c in [f'{tag}_gmv30', f'{tag}_ord30', f'{tag}_cart30', f'{tag}_srch30']:
        feats.append(np.log1p(df[c].fill_null(0).to_numpy()))
    feats.append(df[f'{tag}_act30'].fill_null(0).to_numpy().astype(float))
    la = df[f'{tag}_lastact'].to_list()
    gap = np.array([(ad - d).days if d is not None else 999 for d in la], dtype=float)
    feats.append(np.minimum(gap, 60.0))
    X = np.column_stack(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rr = {}
    for sl in ['S1', 'S2', 'S3']:
        col = f'{tag}_{sl}'
        if col not in df.columns: continue
        yv = np.log1p(df[col].fill_null(0).to_numpy())
        oof = np.zeros_like(yv)
        for tr, te in kf.split(X):
            m = Ridge(alpha=10.0).fit(X[tr], yv[tr])
            oof[te] = m.predict(X[te])
        r2 = 1 - ((yv - oof) ** 2).mean() / yv.var()
        rr[sl] = round(float(r2), 4)
    res_c2b[tag] = rr
    print(f'  {tag}: ' + '  '.join(f'{k} mdl_flint {v:.4f}' for k, v in rr.items()))
out['C2b_slice_decay'] = res_c2b

json.dump(out, open('/Users/alexanderkondakov/ozon-cup/work/reports/eda3_halfwin_predictability.json', 'w'), indent=1)
print('\nsaved work/reports/eda3_halfwin_predictability.json')
