"""Part D: does half-decomposition beat direct 30d prediction at equal capacity?
Ridge OOF, same folds/features. Plus residual structure between halves, concentration."""
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
kf = KFold(5, shuffle=True, random_state=42)

def build_X(tag, ad):
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
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

def oof_ridge(X, yv):
    oof = np.zeros_like(yv)
    for tr, te in kf.split(X):
        m = Ridge(alpha=10.0).fit(X[tr], yv[tr])
        oof[te] = m.predict(X[te])
    return oof

print('=== : direct vs sum-of-halves vs sum-of-thirds (ridge OOF, y30 MSE) ===')
res = {}
for tag, a in [('va', '2026-01-14'), ('ta', '2025-02-13'), ('c9', '2025-09-30')]:
    ad = dt.date.fromisoformat(a)
    X = build_X(tag, ad)
    Fv = df[f'{tag}_F'].fill_null(0).to_numpy()
    Tv = df[f'{tag}_T'].fill_null(0).to_numpy()
    y30 = np.log1p(Fv + Tv)
    o30 = oof_ridge(X, y30)
    oF = oof_ridge(X, np.log1p(Fv))
    oT = oof_ridge(X, np.log1p(Tv))
    o_sum = np.log1p(np.maximum(np.expm1(oF), 0) + np.maximum(np.expm1(oT), 0))
    mse30 = ((y30 - o30) ** 2).mean()
    msesum = ((y30 - o_sum) ** 2).mean()
    # affine-recalibrate the decomposed one on OOF (fair: direct is trained on y30 directly)
    A = np.column_stack([o_sum, np.ones_like(o_sum)])
    beta, *_ = np.linalg.lstsq(A, y30, rcond=None)
    o_sum_cal = A @ beta
    msesum_cal = ((y30 - o_sum_cal) ** 2).mean()
    # thirds where available
    third = ''
    if f'{tag}_S1' in df.columns:
        s1 = df[f'{tag}_S1'].fill_null(0).to_numpy(); s2 = df[f'{tag}_S2'].fill_null(0).to_numpy(); s3 = df[f'{tag}_S3'].fill_null(0).to_numpy()
        oo = [oof_ridge(X, np.log1p(s)) for s in (s1, s2, s3)]
        o3 = np.log1p(sum(np.maximum(np.expm1(o), 0) for o in oo))
        A3 = np.column_stack([o3, np.ones_like(o3)])
        b3, *_ = np.linalg.lstsq(A3, y30, rcond=None)
        mse3 = ((y30 - A3 @ b3) ** 2).mean()
        third = f'  thirds_cal {np.sqrt(mse3):.5f}'
    r = dict(direct=round(float(np.sqrt(mse30)), 5), halves=round(float(np.sqrt(msesum)), 5),
             halves_cal=round(float(np.sqrt(msesum_cal)), 5))
    res[tag] = r
    print(f"  {tag} {a}: direct {r['direct']}  halves {r['halves']}  halves_cal {r['halves_cal']}{third}"
          f"  d(cal-direct) {r['halves_cal']-r['direct']:+.5f}")
    if tag == 'va':
        # concentration of the halves-vs-direct gain across users
        dgain = (y30 - o30) ** 2 - (y30 - o_sum_cal) ** 2   # >0 = decomposition better
        tot = dgain.sum()
        idx = np.argsort(-np.abs(dgain))
        top1 = dgain[idx[:2500]].sum() / tot if tot != 0 else np.nan
        top01 = dgain[idx[:250]].sum() / tot if tot != 0 else np.nan
        print(f'    gain total {tot:+.1f} (RMSLE d {r["halves_cal"]-r["direct"]:+.5f}); top-1% by |contrib| carry {top1*100:.0f}%, top-0.1% {top01*100:.0f}%')
        res['va_concentration'] = dict(top1=round(float(top1), 3), top01=round(float(top01), 3))
out[''] = res

print('\n=== : half-residual correlation (common hidden intensity vs independent halves) ===')
yF = np.log1p(df['va_F'].fill_null(0).to_numpy())
yT = np.log1p(df['va_T'].fill_null(0).to_numpy())
def aff(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta
rF = aff(b, yF); rT = aff(b, yT)
c = np.corrcoef(rF, rT)[0, 1]
print(f'  corr(resid_F, resid_T | blend affine) = {c:+.4f}')
out['D2_half_resid_corr_given_blend'] = round(float(c), 4)
# same among buyers of 30d window only
mb = (df['target'].to_numpy() > 0)
c2 = np.corrcoef(rF[mb], rT[mb])[0, 1]
print(f'  among 30d buyers only: {c2:+.4f}')
out['D2_buyers_only'] = round(float(c2), 4)

json.dump(out, open('/Users/alexanderkondakov/ozon-cup/work/reports/eda3_half_decomp_test.json', 'w'), indent=1)
print('\nsaved work/reports/eda3_half_decomp_test.json')
