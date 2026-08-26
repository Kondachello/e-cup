"""F2. Повторные покупки одного товара. Товаров в данных нет, но цена — отпечаток:
если юзер в разные дни покупает РОВНО ту же цену, это почти наверняка тот же SKU.
Такого признака у команды нет вообще (у них только суммы и средние)."""
import polars as pl, numpy as np
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","to_ord","gmv"])
one = df.filter(pl.col("to_ord")==1).with_columns(pl.col("gmv").round(4).alias("p"))
print(f"наблюдений (день = 1 товар): {one.height:,}, юзеров: {one['user_id'].n_unique():,}")

# сколько у юзера ПОВТОРОВ одной и той же цены
g = one.group_by(["user_id","p"]).len()
rep = g.filter(pl.col("len")>=2)
print(f"\nпар (юзер, цена) с 2+ повторами: {rep.height:,}")
print(f"юзеров хотя бы с одним повтором: {rep['user_id'].n_unique():,} "
      f"({100*rep['user_id'].n_unique()/one['user_id'].n_unique():.1f}% от покупавших поштучно)")

# ЧТО ОЖИДАЛОСЬ БЫ СЛУЧАЙНО: перемешиваем цены между юзерами, сохраняя число покупок
rng = np.random.default_rng(0)
sh = one.select(["user_id"]).with_columns(pl.Series("p", rng.permutation(one["p"].to_numpy())))
gs = sh.group_by(["user_id","p"]).len()
reps = gs.filter(pl.col("len")>=2)
print(f"\nПЛАЦЕБО (цены перемешаны между юзерами):")
print(f"  пар с 2+ повторами: {reps.height:,}   -> НАБЛЮДАЕМЫЙ ИЗБЫТОК {rep.height/max(reps.height,1):.1f}x")
print(f"  юзеров с повтором:  {reps['user_id'].n_unique():,}")

print("\n=== распределение кратности повтора ===")
d = rep.group_by("len").len().sort("len")
for row in d.head(8).iter_rows(): print(f"  цена повторилась {row[0]:>3} раз: {row[1]:>8,} пар")
print(f"  максимум повторов одной цены у юзера: {rep['len'].max()}")

print("\n=== СВЯЗЬ С БУДУЩИМ: предсказывает ли повтор покупку в следующие 30 дней ===")
from datetime import date, timedelta
A = date(2026,1,14)
hist = one.filter(pl.col("event_date")<=A)
h = hist.group_by(["user_id","p"]).len().group_by("user_id").agg([
    (pl.col("len")>=2).sum().alias("n_rep"), pl.col("len").sum().alias("n_buy"),
    pl.col("len").max().alias("max_rep")])
h = h.with_columns((pl.col("n_rep")/pl.col("n_buy")).alias("rep_share"))
fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
gg = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
j = h.join(gg, on="user_id", how="left").with_columns(pl.col("y").fill_null(0))
print(" повторов | юзеров  | доля покупающих далее | mean log1p(таргет)")
for lo,hi,lab in [(0,0,"0"),(1,1,"1"),(2,3,"2-3"),(4,7,"4-7"),(8,10**9,"8+")]:
    s = j.filter((pl.col("n_rep")>=lo)&(pl.col("n_rep")<=hi))
    if s.height:
        y = s["y"].to_numpy()
        print(f"  {lab:>7} | {s.height:>7,} |        {100*(y>0).mean():5.2f}%        |   {np.log1p(y).mean():.4f}")
