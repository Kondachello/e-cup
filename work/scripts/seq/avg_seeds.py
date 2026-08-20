"""
Усреднение сидов в лог-пространстве, вывод в формате work/preds/.

  python avg_seeds.py --out work/preds/tfm3_val.parquet  work/preds/tfm_s*_val.parquet
  python avg_seeds.py --out work/preds/tfm3_test.parquet work/preds/tfm_s*_test.parquet

Веса равные: это один и тот же трансформер с разной инициализацией,
подбирать между ними нечего. Усреднять надо именно логарифмы — метрика
живёт в log1p, и среднее в исходном масштабе систематически завышает хвост.
"""
import polars as pl, numpy as np, argparse, sys

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('files', nargs='+')
a = p.parse_args()

base, acc = None, None
for f in a.files:
    d = pl.read_parquet(f).sort('user_id')
    col = 'pred' if 'pred' in d.columns else 'predict'
    if base is None:
        base = d['user_id'].to_numpy(); acc = np.zeros(len(d))
    assert np.array_equal(base, d['user_id'].to_numpy()), f'{f}: другой набор user_id'
    acc += np.log1p(np.clip(d[col].to_numpy().astype(np.float64), 0, None)) / len(a.files)
    print(f'  + {f}')
pl.DataFrame({'user_id': base.astype(np.int64),
              'pred': np.expm1(acc)}).write_parquet(a.out)
print(f'сохранено {a.out}: {len(base)} юзеров, среднее {float(np.expm1(acc).mean()):.2f}')
