"""Part B: val-window residual vs intra-window timing (diagnostics on future info)."""
import polars as pl
import numpy as np
import datetime as dt

SP = '/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad'
u = pl.read_parquet(f'{SP}/user_windows.parquet')
vp = pl.read_parquet('/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet',
                     columns=['user_id', 'target', 'blend'])
df = vp.join(u, on='user_id', how='left').fill_null(0)
n = df.height
F = df['va_F'].to_numpy()
T = df['va_T'].to_numpy()
tgt = df['target'].to_numpy()
b = df['blend'].to_numpy().astype(np.float64)
y = np.log1p(tgt)
e = y - b
mse = (e ** 2).mean()
print(f'N {n}  RMSLE {np.sqrt(mse):.6f}  (expect 1.665647)')

# 1. sanity: F+T == target
diff = np.abs(F + T - tgt)
print(f'sanity |F+T-target|: max {diff.max():.6g}  n>1e-6: {(diff > 1e-6).sum()}')

# 2. groups by (F>0, T>0)
gF, gT = F > 0, T > 0
groups = {'zero': ~gF & ~gT, 'front_only': gF & ~gT, 'tail_only': ~gF & gT, 'both': gF & gT}
print('\ngroup            n      %n    mean_tgt  mean_y   mean_b   mean_e    se_e    sum_e2/Ntot  %MSE')
for k, m in groups.items():
    ni = m.sum()
    contrib = (e[m] ** 2).sum() / n
    print(f'{k:12s} {ni:8d} {ni/n*100:6.2f}  {tgt[m].mean():9.1f} {y[m].mean():7.3f} {b[m].mean():8.3f} '
          f'{e[m].mean():+8.4f} {e[m].std()/np.sqrt(ni):7.4f}  {contrib:9.4f}  {contrib/mse*100:5.1f}')

# front_only vs tail_only symmetry test
mfo, mto = groups['front_only'], groups['tail_only']
d_e = e[mfo].mean() - e[mto].mean()
se = np.sqrt(e[mfo].var()/mfo.sum() + e[mto].var()/mto.sum())
print(f'\nfront_only vs tail_only: d(mean e) {d_e:+.4f} +- {se:.4f}  z {d_e/se:+.2f}')
d_y = y[mfo].mean() - y[mto].mean()
sey = np.sqrt(y[mfo].var()/mfo.sum() + y[mto].var()/mto.sum())
d_b = b[mfo].mean() - b[mto].mean()
print(f'  d(mean y) {d_y:+.4f} +- {sey:.4f}   d(mean blend) {d_b:+.4f}')

# 3. e by first gmv day in window
d0 = dt.date(2026, 1, 15)
fd = df['va_first_gmv_date'].to_numpy()
first_day = np.array([(d - d0).days + 1 if d is not None else 0 for d in df['va_first_gmv_date'].to_list()])
ld = np.array([(d - d0).days + 1 if d is not None else 0 for d in df['va_last_gmv_date'].to_list()])
print('\nfirst_gmv_day: n, mean_e, se, mean_y, mean_b')
buyers = first_day > 0
for lo, hi in [(1,3),(4,6),(7,9),(10,12),(13,15),(16,18),(19,21),(22,24),(25,27),(28,30)]:
    m = (first_day >= lo) & (first_day <= hi)
    if m.sum() == 0: continue
    print(f'  d{lo:2d}-{hi:2d}: {m.sum():6d}  e {e[m].mean():+7.4f}+-{e[m].std()/np.sqrt(m.sum()):.4f}  '
          f'y {y[m].mean():6.3f}  b {b[m].mean():6.3f}')
m = ~buyers
print(f'  none  : {m.sum():6d}  e {e[m].mean():+7.4f}+-{e[m].std()/np.sqrt(m.sum()):.4f}  y {y[m].mean():6.3f}  b {b[m].mean():6.3f}')

# oracle ceiling of first-day bin correction (in-sample, future info -> pure upper bound)
bins = np.clip(first_day, 0, 30)
e_adj = e.copy()
for v in np.unique(bins):
    m = bins == v
    e_adj[m] -= e[m].mean()
print(f'oracle(first_day known, 31 bins) RMSLE {np.sqrt((e_adj**2).mean()):.6f}  gain {np.sqrt(mse)-np.sqrt((e_adj**2).mean()):.4f}  [FUTURE INFO]')

# 4. oracle-half decomposition: know F exactly vs know T exactly
s_front = F.sum() / (F + T).sum()
pb = np.expm1(b)
y_oF = np.log1p(F + (1 - s_front) * pb)   # know front, tail from blend split
y_oT = np.log1p(T + s_front * pb)          # know tail, front from blend split
mse_oF = ((y - y_oF) ** 2).mean()
mse_oT = ((y - y_oT) ** 2).mean()
print(f'\nglobal front share {s_front:.4f}')
print(f'oracle front known: RMSLE {np.sqrt(mse_oF):.6f}  removes {(mse-mse_oF)/mse*100:5.1f}% of MSE')
print(f'oracle tail  known: RMSLE {np.sqrt(mse_oT):.6f}  removes {(mse-mse_oT)/mse*100:5.1f}% of MSE')

# 5. allocation phi vs e among 'both' buyers (timing beyond magnitude)
m = groups['both']
phi = F[m] / (F[m] + T[m])
r = np.corrcoef(phi, e[m])[0, 1]
print(f'\nboth-buyers n {m.sum()}: corr(phi=F/(F+T), e) {r:+.4f}')
# and among all buyers, front share incl 0/1 cases
mb = buyers
phi_all = F[mb] / (F[mb] + T[mb])
r_all = np.corrcoef(phi_all, e[mb])[0, 1]
print(f'all buyers n {mb.sum()}: corr(phi, e) {r_all:+.4f}')
# mean e by phi deciles (both group)
qs = np.quantile(phi, np.linspace(0, 1, 6))
print('phi bins (both):')
for i in range(5):
    mm = (phi >= qs[i]) & (phi <= qs[i+1] if i == 4 else phi < qs[i+1])
    ee = e[m][mm]
    print(f'  [{qs[i]:.2f},{qs[i+1]:.2f}] n {mm.sum():6d}  e {ee.mean():+7.4f}+-{ee.std()/np.sqrt(mm.sum()):.4f}')

# 6. n_gmv_days and last_day diagnostics
nd = df['va_n_gmv_days'].to_numpy()
print('\nn_gmv_days: n, mean_e')
for v in [1, 2, 3, 4, 5]:
    m = nd == v
    print(f'  {v}: {m.sum():6d}  e {e[m].mean():+7.4f}+-{e[m].std()/np.sqrt(max(m.sum(),1)):.4f}')
m = nd >= 6
print(f'  6+: {m.sum():6d}  e {e[m].mean():+7.4f}+-{e[m].std()/np.sqrt(m.sum()):.4f}')
