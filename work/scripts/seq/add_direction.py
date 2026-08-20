"""
Добавляет трансформер в готовый сабмит тиммейта, не трогая его поправки.

  python add_direction.py --pack preds_pack --tfm-val tfm3_val.parquet \
      --tfm-test tfm3_test.parquet --base H1_applied.csv --out H2_tfm.csv

Как это работает. H1 несёт глобальную и сегментные поправки, измеренные
зондами лидерборда (+0.17 в log). Просто смешать H1 с нашим сырым прогнозом
нельзя — поправки размоются пропорционально весу.

Поэтому считаем не смесь, а ДОБАВКУ. На валидации подбираем МНК-веса дважды:
без трансформера и с ним. Разница этих двух блендов на тесте — это ровно тот
вклад, который вносит трансформер, очищенный от всего остального. Его и
прибавляем к H1 в log-пространстве. Поправки остаются нетронутыми.
"""
import polars as pl, numpy as np, argparse

p = argparse.ArgumentParser()
p.add_argument('--pack', default='preds_pack', help='папка с val_preds.parquet и test_preds.parquet')
p.add_argument('--tfm-val', required=True)
p.add_argument('--tfm-test', required=True)
p.add_argument('--base', required=True, help='сабмит-основа, например H1_applied.csv')
p.add_argument('--out', default='H2_tfm.csv')
p.add_argument('--scale', type=float, default=1.0, help='масштаб добавки; 0.5 = осторожный вариант')
a = p.parse_args()

L1 = lambda x: np.log1p(np.clip(np.asarray(x, np.float64), 0, None))
rm = lambda l, y: float(np.sqrt(np.mean((l - y) ** 2)))

# ---------- валидация: сколько даёт трансформер и с каким весом
v = pl.read_parquet(f'{a.pack}/val_preds.parquet').sort('user_id')
tv = pl.read_parquet(a.tfm_val).sort('user_id')
tv = tv.rename({('pred' if 'pred' in tv.columns else 'predict'): 'tfm'})
v = v.join(tv, on='user_id', how='inner')
y = L1(v['target'].to_numpy())
# берём только те модели, что есть и в val, и в test: иначе веса некуда применить
_test_cols = set(pl.read_parquet(f'{a.pack}/test_preds.parquet', n_rows=1).columns)
his = [c for c in v.columns if c not in ('user_id', 'target', 'tfm') and c in _test_cols]
_skip = [c for c in v.columns if c not in ('user_id','target','tfm') and c not in _test_cols]
if _skip: print('пропущены (нет прогнозов на тесте):', ', '.join(_skip))
Lv = {c: L1(v[c].to_numpy()) for c in his + ['tfm']}

def fit(names):
    A = np.stack([Lv[c] for c in names] + [np.ones_like(y)], 1)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w, rm(A @ w, y)

w0, r0 = fit(his)
w1, r1 = fit(his + ['tfm'])
print(f'моделей у тиммейта: {len(his)}')
print(f'валидация без трансформера {r0:.5f}')
print(f'валидация с трансформером  {r1:.5f}   ({r1-r0:+.5f}), вес {w1[-2]:.3f}')
if r1 >= r0:
    print('ВНИМАНИЕ: на валидации трансформер не помогает, дальше идти не стоит')

# ---------- тест: считаем добавку
t = pl.read_parquet(f'{a.pack}/test_preds.parquet').sort('user_id')
tt = pl.read_parquet(a.tfm_test).sort('user_id')
tt = tt.rename({('pred' if 'pred' in tt.columns else 'predict'): 'tfm'})
t = t.join(tt, on='user_id', how='inner')
assert t.height == 250000, f'на тесте {t.height} юзеров вместо 250000'
Lt = {c: L1(t[c].to_numpy()) for c in his + ['tfm']}
mk = lambda names, w: np.stack([Lt[c] for c in names] + [np.ones(t.height)], 1) @ w
delta = mk(his + ['tfm'], w1) - mk(his, w0)
print(f'\nдобавка: среднее {delta.mean():+.4f}, разброс {delta.std():.4f}, '
      f'|max| {np.abs(delta).max():.3f}')

# ---------- накладываем на готовый сабмит
b = pl.read_csv(a.base).sort('user_id')
assert np.array_equal(b['user_id'].to_numpy(), t['user_id'].to_numpy()), 'разные user_id'
lp = L1(b['predict'].to_numpy()) + a.scale * delta
out = np.expm1(np.clip(lp, 0, None))
pl.DataFrame({'user_id': b['user_id'], 'predict': out}).write_csv(a.out)
print(f'\nсохранено {a.out}')
print(f'  среднее было {float(b["predict"].mean()):.2f}, стало {float(out.mean()):.2f}')
print(f'  расхождение с основой sqrt() = {np.sqrt(np.mean((lp-L1(b["predict"].to_numpy()))**2)):.4f}')
print(f'\nожидаемый выигрыш на лидерборде ~{r0-r1:.4f} (если перенос как обычно)')
