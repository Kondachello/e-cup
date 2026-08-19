"""F1. Цены товаров. В днях с to_ord==1 gmv это ТОЧНАЯ цена одного товара.
Товаров и категорий в данных нет — но цена их частично выдаёт."""
import polars as pl, numpy as np
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","to_ord","gmv"])

one = df.filter(pl.col("to_ord")==1)
print(f"дней с ровно одним заказом: {one.height:,} ({100*one.height/df.height:.2f}% строк)")
p = one["gmv"].to_numpy()
print(f"цена одного товара: min={p.min():.4f} p1={np.percentile(p,1):.2f} p50={np.percentile(p,50):.2f} "
      f"p99={np.percentile(p,99):.2f} max={p.max():.2f}  mean={p.mean():.2f}")
print(f"нулевых цен (заказ есть, gmv=0): {int((p==0).sum()):,}")

print("\n=== КВАНТОВАНИЕ: сколько знаков после запятой ===")
r = np.round(p, 6)
for d in range(0, 5):
    k = np.isclose(r, np.round(r, d), atol=1e-9).mean()
    print(f"  ровно {d} знаков хватает для {100*k:6.2f}% цен")

print("\n=== ДИСКРЕТНОСТЬ: сколько уникальных цен ===")
u = np.unique(np.round(p, 4))
print(f"  уникальных цен: {len(u):,} на {len(p):,} наблюдений")
vc = one.group_by(pl.col("gmv").round(4)).len().sort("len", descending=True)
print("  топ-15 самых частых цен:")
tot = one.height
for row in vc.head(15).iter_rows():
    print(f"    {row[0]:>10.4f}  встречается {row[1]:>7,} раз ({100*row[1]/tot:.3f}%)")
print(f"  доля наблюдений, покрытая топ-100 цен:  {100*vc.head(100)['len'].sum()/tot:.2f}%")
print(f"  доля наблюдений, покрытая топ-1000 цен: {100*vc.head(1000)['len'].sum()/tot:.2f}%")

print("\n=== есть ли ЦЕНОВАЯ СЕТКА (кратность) ===")
for step in (0.01, 0.05, 0.1, 0.5, 1.0):
    k = np.isclose(p/step, np.round(p/step), atol=1e-6).mean()
    print(f"  кратно {step:>5}: {100*k:6.2f}%")

print("\n=== ЦЕНЫ ПО ВРЕМЕНИ (инфляция/переоценка) ===")
q = one.with_columns(pl.col("event_date").dt.strftime("%Y-%m").alias("m")).group_by("m").agg([
    pl.col("gmv").median().alias("med"), pl.col("gmv").mean().alias("mean"), pl.len().alias("n")]).sort("m")
print(q)
