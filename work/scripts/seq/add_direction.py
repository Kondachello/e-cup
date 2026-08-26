"""
Добавляет трансформер в готовый сабмит тиммейта, не трогая его поправки.

  python work/scripts/seq/add_direction.py --pack work/preds_pack \
      --tfm-val work/preds/tfm3_val.parquet --tfm-test work/preds/tfm3_test.parquet \
      --base H1_applied.csv --out H2_tfm.csv --center

Как это работает. H1 несёт глобальную и сегментные поправки, измеренные
зондами лидерборда (проверено: среднее log1p(H1) - blend = +0.16948). Просто
смешать H1 с сырым прогнозом нельзя — поправки размоются пропорционально весу.

Поэтому считаем не смесь, а ДОБАВКУ. На валидации подбираем МНК-веса дважды:
без трансформера и с ним. Разница этих двух блендов на тесте — вклад
трансформера, очищенный от всего остального. Его и прибавляем к H1 в
лог-пространстве. Поправки остаются нетронутыми.

ИСПРАВЛЕНО ОТНОСИТЕЛЬНО ПРЕЖНЕЙ ВЕРСИИ
--------------------------------------
1. Колонки пакета УЖЕ в log1p (см. work/preds_pack/README.md), а прежняя версия
   применяла к ним log1p ещё раз. Двойной логарифм стоил 0.0035 на валидации
   (1.669642 против 1.666128) — 160 шумовых единиц. Теперь пространство колонок
   определяется автоматически (--pack-space auto|log|raw).
2. Флаг --center: у добавки ненулевое среднее (+0.0125), а глобальная поправка
   +0.17 мерилась зондами для текущего уровня. Со сдвигом уровня она сбивается.
   --center вычитает среднее, оставляя только форму. Рекомендуется, пока уровень
   не перемерен зондом.
3. Печатается честная оценка переноса: веса и калибровка учатся на train-фолдах,
   прирост меряется на отложенных юзерах (5 фолдов x 5 разбиений).
"""
import polars as pl, numpy as np, argparse

p = argparse.ArgumentParser()
p.add_argument('--pack', default='work/preds_pack')
p.add_argument('--tfm-val', required=True)
p.add_argument('--tfm-test', required=True)
p.add_argument('--base', required=True, help='сабмит-основа, например H1_applied.csv')
p.add_argument('--out', default='H2_tfm.csv')
p.add_argument('--scale', type=float, default=1.0, help='масштаб добавки; 0.5 = осторожный вариант')
p.add_argument('--center', action='store_true', help='вычесть среднее добавки (сохранить уровень)')
p.add_argument('--pack-space', default='auto', choices=['auto', 'log', 'raw'])
p.add_argument('--folds', type=int, default=5)
a = p.parse_args()

L1 = lambda x: np.log1p(np.clip(np.asarray(x, np.float64), 0, None))
rm = lambda l, y: float(np.sqrt(np.mean((l - y) ** 2)))

v = pl.read_parquet(f'{a.pack}/val_preds.parquet').sort('user_id')
t = pl.read_parquet(f'{a.pack}/test_preds.parquet').sort('user_id')
y = L1(v['target'].to_numpy())

space = a.pack_space
if space == 'auto':
    probe = v[[c for c in v.columns if c not in ('user_id', 'target')][0]].to_numpy()
    space = 'log' if float(np.nanmax(probe)) < 50 else 'raw'
    print(f'пространство колонок пакета определено как: {space}')
conv = (lambda x: np.asarray(x, np.float64)) if space == 'log' else L1

his = [c for c in v.columns if c not in ('user_id', 'target') and c in t.columns]
skip = [c for c in v.columns if c not in ('user_id', 'target') and c not in t.columns]
if skip: print('пропущены (нет на тесте):', ', '.join(skip))

def tfm(path):
    d = pl.read_parquet(path).sort('user_id')
    col = 'pred' if 'pred' in d.columns else 'predict'
    return d['user_id'].to_numpy(), L1(d[col].to_numpy())

uid, lp3 = tfm(a.tfm_val)
tuid, lt3 = tfm(a.tfm_test)
assert np.array_equal(uid, v['user_id'].to_numpy()), 'val: другой набор user_id'
assert np.array_equal(tuid, t['user_id'].to_numpy()), 'test: другой набор user_id'

def fit_shifts(lp, yy, idx=None, bins=24):
    i = slice(None) if idx is None else idx
    qs = np.quantile(lp[i], np.linspace(0, 1, bins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    c, s = [], []
    for k in range(bins):
        m = (lp[i] > qs[k]) & (lp[i] <= qs[k + 1])
        if m.sum() < 500: continue
        c.append(lp[i][m].mean()); s.append(yy[i][m].mean() - lp[i][m].mean())
    return np.array(c), np.array(s)
ap = lambda lp, c, s: np.clip(lp + np.interp(lp, c, s), 0, None)

Xv = np.column_stack([conv(v[c].to_numpy()) for c in his] + [np.ones(len(y))])
Xt = np.column_stack([conv(t[c].to_numpy()) for c in his] + [np.ones(len(tuid))])
ins = lambda X, col: np.column_stack([X[:, :-1], col, X[:, -1]])

# --- честная оценка переноса: всё учится на train-фолдах ---------------------
gains = []
for seed in range(5):
    rng = np.random.default_rng(seed); f = rng.permutation(len(y)) % a.folds
    b_all, w_all = np.zeros(len(y)), np.zeros(len(y))
    for k in range(a.folds):
        tr, te = f != k, f == k
        c, s = fit_shifts(lp3, y, tr)
        X2 = ins(Xv, ap(lp3, c, s))
        w0 = np.linalg.lstsq(Xv[tr], y[tr], rcond=None)[0]
        w1 = np.linalg.lstsq(X2[tr], y[tr], rcond=None)[0]
        b_all[te] = Xv[te] @ w0; w_all[te] = X2[te] @ w1
    gains.append(rm(w_all, y) - rm(b_all, y))
g = np.array(gains)
print(f'\nчестный перенос (веса и калибровка только на train-фолдах, 5 разбиений):')
print(f'  прирост {g.mean():+.6f}  разброс {g.std():.6f}  = {abs(g.mean())/0.000022:.0f} шумовых единиц ЛБ')

# --- рабочие веса на всей валидации -----------------------------------------
c, s = fit_shifts(lp3, y)
Xv2, Xt2 = ins(Xv, ap(lp3, c, s)), ins(Xt, ap(lt3, c, s))
w0 = np.linalg.lstsq(Xv, y, rcond=None)[0]
w1 = np.linalg.lstsq(Xv2, y, rcond=None)[0]
print(f'\nвалидация без трансформера {rm(Xv @ w0, y):.6f}')
print(f'валидация с трансформером  {rm(Xv2 @ w1, y):.6f}  вес tfm {w1[-2]:+.4f}')
if rm(Xv2 @ w1, y) >= rm(Xv @ w0, y):
    print('ВНИМАНИЕ: на валидации трансформер не помогает, дальше идти не стоит')

delta = Xt2 @ w1 - Xt @ w0
print(f'\nдобавка: среднее {delta.mean():+.5f}  sd {delta.std():.5f}  |max| {np.abs(delta).max():.4f}')
if a.center:
    delta = delta - delta.mean(); print('  --center: среднее вычтено, уровень H1 сохранён')

b = pl.read_csv(a.base).sort('user_id')
assert np.array_equal(b['user_id'].to_numpy(), tuid), 'разные user_id с основой'
lb = L1(b['predict'].to_numpy())
lp = lb + a.scale * delta
out = np.expm1(np.clip(lp, 0, None))
pl.DataFrame({'user_id': b['user_id'], 'predict': out}).write_csv(a.out)
q = np.quantile(np.abs(lp - lb), [.5, .9, .99, 1.0])
print(f'\nсохранено {a.out}')
print(f'  среднее прогноза {float(b["predict"].mean()):.2f} -> {out.mean():.2f}')
print(f'  сдвиг уровня {(lp-lb).mean():+.5f}')
print(f'  |Δ| в логах: медиана {q[0]:.4f}  p90 {q[1]:.4f}  p99 {q[2]:.4f}  max {q[3]:.4f}')
print(f'\nожидаемый выигрыш на ЛБ ~{abs(g.mean()):.6f}, ЕСЛИ перенос как обычно.')
print('Оценка сверху: тестовые предсказания трансформера обучены по якорь 348,')
print('а остальные модели команды — по 378, то есть зазор 60 против 30.')
