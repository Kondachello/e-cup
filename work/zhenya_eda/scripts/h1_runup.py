"""H1. Что происходит ПЕРЕД крупной покупкой.
Контроль ВНУТРИ юзера: дни перед крупной покупкой против прочих дней того же
юзера. Так снимается уровень активности, на котором горят все наивные замеры."""
import polars as pl, numpy as np
df = pl.read_parquet("train.parquet")

# крупная покупка = день с gmv в верхних 10% среди дней с заказом
ord_days = df.filter(pl.col("to_ord")>0)
thr = float(ord_days["gmv"].quantile(0.90))
print(f"порог крупной покупки: gmv > {thr:.2f} (верхние 10% дней с заказом)")
print(f"таких дней: {ord_days.filter(pl.col('gmv')>thr).height:,}")

d = df.sort(["user_id","event_date"]).with_columns([
    (pl.col("to_ord")>0).alias("ord"),
    ((pl.col("to_ord")>0)&(pl.col("gmv")>thr)).alias("big"),
    ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0)).alias("emp"),
])
# метка: сколько дней до ближайшей БУДУЩЕЙ крупной покупки (по календарю)
d = d.with_columns(pl.when(pl.col("big")).then(pl.col("event_date")).otherwise(None).alias("bd"))
d = d.sort(["user_id","event_date"], descending=[False,True]).with_columns(
        pl.col("bd").forward_fill().over("user_id").alias("next_big")
    ).sort(["user_id","event_date"])
d = d.with_columns((pl.col("next_big")-pl.col("event_date")).dt.total_days().alias("dtb"))

# только юзеры, у которых крупная покупка вообще была
users_big = d.filter(pl.col("big"))["user_id"].unique()
sub = d.filter(pl.col("user_id").is_in(users_big.implode()) & (~pl.col("big")))
print(f"юзеров с крупной покупкой: {len(users_big):,}")

print("\n=== дневные показатели по близости к крупной покупке (внутри тех же юзеров) ===")
print(f"{'дней до крупной':>16} {'дней':>10} {'searches':>10} {'to_cart':>9} {'cat':>8} {'доля пустых':>12} {'доля с заказом':>15}")
base = None
for lo,hi,lab in [(1,1,"1"),(2,3,"2-3"),(4,7,"4-7"),(8,14,"8-14"),(15,30,"15-30"),(31,90,"31-90"),(91,10**9,"90+")]:
    s = sub.filter((pl.col("dtb")>=lo)&(pl.col("dtb")<=hi))
    if not s.height: continue
    row = (s.height, s["searches"].mean(), s["to_cart"].mean(), s["cat"].mean(),
           s["emp"].mean(), s["ord"].mean())
    if lab=="90+": base = row
    print(f"{lab:>16} {row[0]:>10,} {row[1]:>10.3f} {row[2]:>9.3f} {row[3]:>8.4f} {row[4]:>12.4f} {row[5]:>15.4f}")
if base:
    print(f"\n=== отношение «за 1 день до» к «90+ дней до» (тот же пул юзеров) ===")
    s = sub.filter(pl.col("dtb")==1)
    for i,(nm,val) in enumerate([("searches",s["searches"].mean()),("to_cart",s["to_cart"].mean()),
                                  ("cat",s["cat"].mean()),("доля пустых",s["emp"].mean()),
                                  ("доля с заказом",s["ord"].mean())]):
        print(f"   {nm:16s} x{val/base[i+1]:.3f}")
