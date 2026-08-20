"""mdl_kyanit. Инвентаризация: схема, объём, покрытие, дубли, семантика строки."""
import polars as pl, numpy as np

TR = "train.parquet"
lf = pl.scan_parquet(TR)
print("=== SCHEMA ===")
for k, v in lf.collect_schema().items():
    print(f"  {k:20s} {v}")

df = pl.read_parquet(TR)
print(f"\nrows={df.height:,}  users={df['user_id'].n_unique():,}")
print(f"dates {df['event_date'].min()} .. {df['event_date'].max()}  uniq={df['event_date'].n_unique()}")

# дубли ключа
d = df.group_by(["user_id", "event_date"]).len().filter(pl.col("len") > 1).height
print(f"дубли (user_id,event_date): {d}")

# нули/пропуски по колонкам
print("\n=== по колонкам: nulls / нулевых / min / max ===")
for c in df.columns:
    if c in ("user_id", "event_date"): continue
    s = df[c]
    print(f"  {c:20s} null={s.null_count():>8,}  zero={int((s==0).sum()):>10,}  "
          f"min={float(s.min()):>12.2f} max={float(s.max()):>14.2f}")

# сколько дней покрыто на юзера
per = df.group_by("user_id").len()
q = per["len"].quantile
print(f"\nстрок на юзера: min={per['len'].min()} p25={q(.25):.0f} p50={q(.5):.0f} "
      f"p75={q(.75):.0f} p95={q(.95):.0f} max={per['len'].max()}  mean={per['len'].mean():.1f}")

# СЕМАНТИКА СТРОКИ: бывают ли строки, где вообще ничего не произошло?
act = df.select([
    (pl.col("search") + pl.col("cat")).alias("chan"),
    pl.col("searches"), pl.col("to_cart"), pl.col("to_ord"), pl.col("gmv")])
dead = act.filter((pl.col("chan") == 0) & (pl.col("searches") == 0) &
                  (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0) & (pl.col("gmv") == 0)).height
print(f"\n«пустых» строк (ни канала, ни поиска, ни корзины, ни заказа, ни gmv): {dead:,}")
print(f"строк search=1: {int(df['search'].sum()):,}   cat=1: {int(df['cat'].sum()):,}   "
      f"оба: {int(((df['search']==1)&(df['cat']==1)).sum()):,}   ни одного: {int(((df['search']==0)&(df['cat']==0)).sum()):,}")

# тождества
chk = df.select([
    (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs().max().alias("gmv_split"),
    (pl.col("to_cart") - pl.col("search_to_cart") - pl.col("cat_to_cart")).abs().max().alias("cart_split"),
    (pl.col("to_ord") - pl.col("search_to_ord") - pl.col("cat_to_ord")).abs().max().alias("ord_split"),
])
print("\n=== тождества (max |невязка|) ===")
print(chk)
