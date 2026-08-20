"""
Разбор результатов подбора гиперпараметров.

  python analyze_sweep.py

  sweep_summary.txt   компактная сводка — её удобно скинуть целиком
  sweep_marginal.png  RMSLE против каждого гиперпараметра
  sweep_curves.png    кривые обучения лучших и худших испытаний
"""
import csv, glob, os, argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--top', type=int, default=15)
a = p.parse_args()

rows = [r for r in csv.DictReader(open(a.results)) if r['rmsle'] not in ('','nan')]
if not rows: raise SystemExit('нет завершённых испытаний')
for r in rows: r['rmsle'] = float(r['rmsle'])
HP = [k for k in rows[0] if k not in ('phase','trial','rmsle','minutes')]
rows.sort(key=lambda r: r['rmsle'])
expl = [r for r in rows if r['phase']=='explore']
fin  = [r for r in rows if r['phase']=='final']

L = []
L.append(f'испытаний завершено: {len(rows)} (разведка {len(expl)}, финал {len(fin)})')
L.append(f'RMSLE: лучший {rows[0]["rmsle"]:.4f}, медиана {np.median([r["rmsle"] for r in rows]):.4f}, худший {rows[-1]["rmsle"]:.4f}')

L.append('\n=== ЛУЧШИЕ ===')
head = f'{"rmsle":>8} {"фаза":>7} {"№":>7}  ' + '  '.join(f'{k:>10}' for k in HP)
L.append(head)
for r in rows[:a.top]:
    L.append(f'{r["rmsle"]:>8.4f} {r["phase"]:>7} {r["trial"]:>7}  ' +
             '  '.join(f'{str(r[k])[:10]:>10}' for k in HP))

L.append('\n=== ВЛИЯНИЕ КАЖДОГО ПАРАМЕТРА (только разведка) ===')
L.append('для непрерывных — по квартилям; n — число испытаний в группе')
src = expl or rows
for k in HP:
    vals = [r[k] for r in src]
    try: nums = np.array([float(v) for v in vals])
    except ValueError: nums = None
    uniq = sorted(set(vals))
    L.append(f'\n{k}:')
    if nums is not None and len(uniq) > 6:
        q = np.quantile(nums, [0,.25,.5,.75,1.0])
        for i in range(4):
            m = (nums>=q[i]) & (nums<=q[i+1] if i==3 else nums<q[i+1])
            if m.sum():
                sel = np.array([r['rmsle'] for r in src])[m]
                L.append(f'  [{q[i]:.4g}..{q[i+1]:.4g}]  n={m.sum():3d}  медиана {np.median(sel):.4f}  лучший {sel.min():.4f}')
        L.append(f'  корреляция с RMSLE: {np.corrcoef(nums, [r["rmsle"] for r in src])[0,1]:+.2f}')
    else:
        for v in uniq:
            sel = np.array([r['rmsle'] for r in src if r[k]==v])
            L.append(f'  {str(v):>10}  n={len(sel):3d}  медиана {np.median(sel):.4f}  лучший {sel.min():.4f}')

txt = '\n'.join(L)
open('sweep_summary.txt','w',encoding='utf-8').write(txt)
print(txt)

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except ImportError:
    print('\nmatplotlib не установлен, графики пропущены'); raise SystemExit

n = len(HP); c = 4; r_ = (n+c-1)//c
fig, ax = plt.subplots(r_, c, figsize=(4*c, 3.2*r_))
ax = np.atleast_1d(ax).ravel()
y = [r['rmsle'] for r in src]
for i, k in enumerate(HP):
    try: x = [float(r[k]) for r in src]
    except ValueError: x = [hash(r[k])%100 for r in src]
    ax[i].scatter(x, y, s=14, alpha=.6)
    if k in ('lr','wd'): ax[i].set_xscale('log')
    ax[i].set_title(k); ax[i].grid(alpha=.3)
for j in range(n, len(ax)): ax[j].axis('off')
fig.tight_layout(); fig.savefig('sweep_marginal.png', dpi=110); plt.close(fig)

hs = {os.path.basename(f)[8:-4]: f for f in glob.glob('history_*.csv')}
pick = [r for r in rows[:5]] + [r for r in rows[-3:]]
fig, ax = plt.subplots(1, 2, figsize=(13,4.5))
for r in pick:
    key = r['trial'] if r['phase']=='final' else f't{r["trial"]}'
    if key not in hs: continue
    d = np.array([[float(x) for x in row] for row in list(csv.reader(open(hs[key])))[1:]])
    if not len(d): continue
    lab = f'{key} {r["rmsle"]:.4f}'
    ax[0].plot(d[:,0], d[:,2], label=lab); ax[1].plot(d[:,0], d[:,3], label=lab)
ax[0].set_title('val RMSLE'); ax[1].set_title('learning rate')
for x in ax: x.set_xlabel('шаг'); x.grid(alpha=.3); x.legend(fontsize=7)
fig.tight_layout(); fig.savefig('sweep_curves.png', dpi=110); plt.close(fig)
print('\nграфики: sweep_marginal.png, sweep_curves.png\nсводка: sweep_summary.txt')
