"""E2. Что такое «пустая» строка и тождества расщепления по каналам."""
import polars as pl, numpy as np
df = pl.read_parquet("train.parquet")

chk = df.select([
    (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs().max().alias("gmv_split"),
    (pl.col("to_cart") - pl.col("search_to_cart") - pl.col("cat_to_cart")).abs().max().alias("cart_split"),
    (pl.col("to_ord") - pl.col("search_to_ord") - pl.col("cat_to_ord")).abs().max().alias("ord_split"),
]).to_dicts()[0]
print("=== max |невязка| тождеств ===")
for k, v in chk.items(): print(f"  {k:12s} {v:.3e}")

# согласованность has_* с числовыми
for pre in ["search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord"]:
    h = "has_" + pre
    bad = df.filter((pl.col(h) == 1) != (pl.col(pre) > 0)).height
    print(f"  {h:20s} рассогласований с {pre}: {bad}")

# ПУСТЫЕ СТРОКИ
empty = ((pl.col("search") == 0) & (pl.col("cat") == 0) & (pl.col("searches") == 0) &
         (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0) & (pl.col("gmv") == 0))
e = df.filter(empty)
print(f"\n=== ПУСТЫЕ СТРОКИ: {e.height:,} ({100*e.height/df.height:.2f}%) ===")
print("что в них ненулевого (должно быть ничего):")
for c in df.columns:
    if c in ("user_id", "event_date"): continue
    nz = int((e[c] != 0).sum())
    if nz: print(f"   {c}: {nz:,}")

# search=0 & cat=0 но активность есть?
ghost = df.filter((pl.col("search") == 0) & (pl.col("cat") == 0) & ~empty)
print(f"\nстрок без канала, НО с активностью: {ghost.height:,}")
if ghost.height:
    print(ghost.select(["searches","to_cart","to_ord","gmv"]).sum().to_dicts()[0])

# распределение пустых по датам и юзерам
byd = e.group_by("event_date").len().sort("event_date")
tot = df.group_by("event_date").len().sort("event_date").rename({"len": "all"})
j = byd.join(tot, on="event_date").with_columns((pl.col("len")/pl.col("all")).alias("share"))
print(f"\nдоля пустых по дням: min={j['share'].min():.4f} p50={j['share'].median():.4f} max={j['share'].max():.4f}")
print("топ-10 дней по доле пустых:"); print(j.sort("share", descending=True).head(10))
print("дно-10:"); print(j.sort("share").head(10))

peru = e.group_by("user_id").len()
print(f"\nюзеров хотя бы с одной пустой строкой: {peru.height:,} ({100*peru.height/250000:.1f}%)")
print(f"пустых строк на такого юзера: p50={peru['len'].median():.0f} p95={peru['len'].quantile(.95):.0f} max={peru['len'].max()}")
