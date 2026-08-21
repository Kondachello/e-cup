"""Part E: observable mirror — timing of purchases WITHIN pre-anchor history window vs residual e.
Feature: split of last-30d history GMV into days -30..-16 vs -15..-1 (+ same for orders/activity).
Honest OOF (5-fold users) of e-regression, vs equal-capacity control from envelope-like cols, + concentration."""
import polars as pl
import numpy as np
import datetime as dt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import json

SP = '/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad'
va = dt.date(2026, 1, 14)
ed = pl.col('event_date')
h_f0, h_f1 = va - dt.timedelta(29), va - dt.timedelta(15)   # history front (older half)
h_t0, h_t1 = va - dt.timedelta(14), va                       # history tail (recent half)
lf = pl.scan_parquet('/Users/alexanderkondakov/ozon-cup/train.parquet').filter(
    ed.is_between(pl.lit(h_f0), pl.lit(va)))
hh = lf.group_by('user_id').agg(
    pl.col('gmv').filter(ed <= pl.lit(h_f1)).sum().alias('g_old'),
    pl.col('gmv').filter(ed >= pl.lit(h_t0)).sum().alias('g_new'),
    pl.col('to_ord').filter(ed <= pl.lit(h_f1)).sum().alias('o_old'),
    pl.col('to_ord').filter(ed >= pl.lit(h_t0)).sum().alias('o_new'),
    (ed <= pl.lit(h_f1)).sum().alias('a_old'),
    (ed >= pl.lit(h_t0)).sum().alias('a_new'),
).collect(engine='streaming')

u = pl.read_parquet(f'{SP}/user_windows.parquet')
vp = pl.read_parquet('/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet',
                     columns=['user_id', 'target', 'blend'])
df = vp.join(u, on='user_id', how='left').join(hh, on='user_id', how='left').fill_null(0)
b = df['blend'].to_numpy().astype(np.float64)
y = np.log1p(df['target'].to_numpy())
e = y - b
n = df.height
mse0 = (e ** 2).mean()

g_old = df['g_old'].to_numpy().astype(float); g_new = df['g_new'].to_numpy().astype(float)
o_old = df['o_old'].to_numpy().astype(float); o_new = df['o_new'].to_numpy().astype(float)
a_old = df['a_old'].to_numpy().astype(float); a_new = df['a_new'].to_numpy().astype(float)

# candidate timing features (history-window front/tail asymmetry)
tot_g = g_old + g_new
share_new_g = np.where(tot_g > 0, g_new / np.maximum(tot_g, 1e-9), 0.5)
has_g = (tot_g > 0).astype(float)
tot_o = o_old + o_new
share_new_o = np.where(tot_o > 0, o_new / np.maximum(tot_o, 1e-9), 0.5)
share_new_a = np.where(a_old + a_new > 0, a_new / np.maximum(a_old + a_new, 1e-9), 0.5)
C = np.column_stack([
    share_new_g, share_new_g * has_g, has_g * (share_new_g == 1.0), has_g * (share_new_g == 0.0),
    share_new_o, share_new_a,
    np.log1p(g_new) - np.log1p(g_old), np.log1p(o_new) - np.log1p(o_old),
])
C = (C - C.mean(0)) / (C.std(0) + 1e-9)
k = C.shape[1]

# envelope-like base (the model already has these): log1p sums + recency
base_cols = []
for c in ['va_gmv30', 'va_gmv90', 'va_gmv365', 'va_ord30', 'va_ord90', 'va_ord365', 'va_cart30', 'va_srch30']:
    base_cols.append(np.log1p(df[c].to_numpy().astype(float)))
la = u.join(vp.select('user_id'), on='user_id', how='right')['va_lastact'].to_list()
gap = np.array([(va - d).days if d is not None else 999 for d in la], dtype=float)
base_cols.append(np.minimum(gap, 60));
lg = u.join(vp.select('user_id'), on='user_id', how='right')['va_lastgmv'].to_list()
gapg = np.array([(va - d).days if d is not None else 999 for d in lg], dtype=float)
base_cols.append(np.minimum(gapg, 400))
B = np.column_stack(base_cols)
B = (B - B.mean(0)) / (B.std(0) + 1e-9)

kf = KFold(5, shuffle=True, random_state=0)
def oof_r2(X, target):
    oof = np.zeros(n)
    for tr, te in kf.split(X):
        m = Ridge(alpha=10.0).fit(X[tr], target[tr])
        oof[te] = m.predict(X[te])
    return 1 - ((target - oof) ** 2).mean() / target.var(), oof

r2_cand, oof_c = oof_r2(C, e)
r2_base, _ = oof_r2(B, e)
r2_both, oof_bc = oof_r2(np.column_stack([B, C]), e)
print(f'OOF mdl_flint vs e: candidates(k={k}) {r2_cand:+.6f}   base(k={B.shape[1]}) {r2_base:+.6f}   base+cand {r2_both:+.6f}   incr {r2_both-r2_base:+.6f}')

# equal-capacity control: k random cols made of base interactions/powers, 20 draws
rng = np.random.default_rng(7)
ctrl = []
for _ in range(20):
    cols = []
    for _ in range(k):
        i, j = rng.integers(0, B.shape[1], 2)
        cols.append(B[:, i] * B[:, j] if rng.random() < 0.5 else B[:, i] ** 2)
    Xc = np.column_stack(cols)
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    r2c, _ = oof_r2(np.column_stack([B, Xc]), e)
    ctrl.append(r2c - r2_base)
ctrl = np.array(ctrl)
print(f'control incr (k={k} random base-derived): mean {ctrl.mean():+.6f} +- {ctrl.std():.6f}')
z = ((r2_both - r2_base) - ctrl.mean()) / (ctrl.std() + 1e-12)
print(f'candidate excess over control: {(r2_both-r2_base)-ctrl.mean():+.6f}  ({z:+.2f} sigma)')

# RMSLE delta if applied (honest OOF corrector) + concentration
e_corr = e - oof_c
rmsle_after = np.sqrt(((e_corr) ** 2).mean())
print(f'RMSLE blend {np.sqrt(mse0):.6f} -> after cand-OOF corrector {rmsle_after:.6f}  d {np.sqrt(mse0)-rmsle_after:+.6f}')
contrib = e ** 2 - e_corr ** 2
tot = contrib.sum()
idx = np.argsort(-np.abs(contrib))
if tot != 0:
    print(f'concentration: top-1% carry {contrib[idx[:2500]].sum()/tot*100:.0f}%  top-0.1% {contrib[idx[:250]].sum()/tot*100:.0f}%')

# simple bin diagnostics: mean e by history-timing class among users with history purchases
print('\nbins (users with gmv in last 30d, n=%d):' % int(has_g.sum()))
for nm, m in [('all_old (bought -30..-16 only)', has_g.astype(bool) & (share_new_g == 0)),
              ('mixed', has_g.astype(bool) & (share_new_g > 0) & (share_new_g < 1)),
              ('all_new (bought -15..-1 only)', has_g.astype(bool) & (share_new_g == 1))]:
    print(f'  {nm:32s} n {m.sum():6d}  e {e[m].mean():+7.4f}+-{e[m].std()/np.sqrt(max(m.sum(),1)):.4f}  b {b[m].mean():6.3f}  y {y[m].mean():6.3f}')

json.dump(dict(r2_cand=float(r2_cand), r2_base=float(r2_base), incr=float(r2_both - r2_base),
               ctrl_mean=float(ctrl.mean()), ctrl_std=float(ctrl.std()), z=float(z),
               rmsle_delta=float(np.sqrt(mse0) - rmsle_after)),
          open('/Users/alexanderkondakov/ozon-cup/work/reports/eda3_hist_timing_obs.json', 'w'), indent=1)
print('\nsaved work/reports/eda3_hist_timing_obs.json')
