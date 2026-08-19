"""G7. Прочёсывание на аномалии: боты, дубли траекторий, массовые события, выбросы."""
import polars as pl, numpy as np
df = pl.read_parquet("train.parquet")

print("=== 1. ЭКСТРЕМАЛЬНЫЕ ДНИ (возможные боты / массовые операции) ===")
for c, thr in [("search_to_cart",200),("cat_to_cart",200),("searches",300),("to_ord",50)]:
    s = df.filter(pl.col(c)>thr)
    print(f"  {c:16s} > {thr:>3}: строк {s.height:>6,}, юзеров {s['user_id'].n_unique():>5,}, "
          f"максимум {df[c].max()}")

print("\n=== 2. ЮЗЕРЫ-АНОМАЛИИ ===")
u = df.group_by("user_id").agg([
    pl.col("searches").sum().alias("s"), pl.col("to_cart").sum().alias("c"),
    pl.col("to_ord").sum().alias("o"), pl.col("gmv").sum().alias("g"),
    pl.len().alias("d"), pl.col("cat").sum().alias("cd")])
print(f"  0 поисков за всю историю:        {int((u['s']==0).sum()):>6,}")
print(f"  0 корзин, но заказы есть:        {int(((u['c']==0)&(u['o']>0)).sum()):>6,}")
print(f"  корзин много, заказов 0:         {int(((u['c']>50)&(u['o']==0)).sum()):>6,}")
print(f"  только каталог (0 поисков, cat>0):{int(((u['s']==0)&(u['cd']>0)).sum()):>6,}")
z = u.filter(pl.col("s")==0)
print(f"  у 'нулевых по поиску': медиана дней={z['d'].median():.0f}, медиана gmv={z['g'].median():.1f}, "
      f"доля с заказами={100*(z['o']>0).mean():.1f}%")

print("\n=== 3. МАССОВЫЕ СОБЫТИЯ: доля юзеров с заказом в один день ===")
d = df.group_by("event_date").agg([
    (pl.col("to_ord")>0).sum().alias("buyers"), pl.len().alias("rows"),
    (pl.col("to_cart")>0).sum().alias("carters")]).sort("event_date")
d = d.with_columns([(pl.col("buyers")/pl.col("rows")).alias("br"),
                    (pl.col("carters")/pl.col("rows")).alias("cr")])
br = d["br"].to_numpy()
med = np.array([np.median(br[max(0,i-7):i+8]) for i in range(len(br))])
r = br/med
for i in np.argsort(r)[-5:][::-1]: print(f"  ПИК конверсии {d['event_date'][int(i)]}  доля покупателей {br[i]:.4f}  x{r[i]:.3f}")
for i in np.argsort(r)[:5]:        print(f"  ПРОВАЛ        {d['event_date'][int(i)]}  доля покупателей {br[i]:.4f}  x{r[i]:.3f}")

print("\n=== 4. ДУБЛИ ТРАЕКТОРИЙ (одинаковые юзеры) ===")
sig = df.group_by("user_id").agg([
    pl.col("event_date").min().alias("f"), pl.col("event_date").max().alias("l"),
    pl.len().alias("n"), pl.col("gmv").sum().round(4).alias("g"),
    pl.col("to_ord").sum().alias("o"), pl.col("searches").sum().alias("s")])
dup = sig.group_by(["f","l","n","g","o","s"]).len().filter(pl.col("len")>1)
print(f"  групп юзеров с полностью совпадающей сигнатурой: {dup.height:,}")
if dup.height: print(f"  затронуто юзеров: {int(dup['len'].sum()):,}, максимум в группе {dup['len'].max()}")

print("\n=== 5. ГРАНИЦЫ ДАННЫХ: активность в первые/последние дни ===")
e = df.group_by("event_date").agg(pl.len()).sort("event_date")
print(f"  строк 2025-01-01: {e['len'][0]:,}   2025-01-02: {e['len'][1]:,}   "
      f"среднее янв-2025: {e['len'][:31].mean():,.0f}")
print(f"  строк 2026-02-13: {e['len'][-1]:,}   2026-02-12: {e['len'][-2]:,}   "
      f"среднее фев-2026: {e['len'][-13:].mean():,.0f}")
