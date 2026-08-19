"""F3. Несёт ли user_id информацию? 75.8% юзеров впервые видны 2025-01-01,
их настоящий стаж ЦЕНЗУРИРОВАН. Если id выдаются по порядку регистрации, id снимает цензуру.
Признака user_id у команды нет вообще — это идентификатор, его никто не пробовал."""
import polars as pl, numpy as np
from datetime import date, timedelta
from scipy.stats import spearmanr
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","to_ord","gmv"])
u = df.group_by("user_id").agg([
    pl.col("event_date").min().alias("first"), pl.len().alias("days"),
    pl.col("gmv").sum().alias("gmv"), pl.col("to_ord").sum().alias("ord")]).sort("user_id")
uid = u["user_id"].to_numpy()
print(f"user_id: min={uid.min():,} max={uid.max():,}  уникальных={len(uid):,}")
print(f"плотность заполнения диапазона: {100*len(uid)/(uid.max()-uid.min()+1):.1f}%")
print(f"монотонность разностей: медиана шага {np.median(np.diff(uid)):.1f}")

first = u["first"].to_numpy()
fnum = np.array([(d - date(2025,1,1)).days for d in first])
cens = fnum == 0
print(f"\nцензурированных (первое событие 2025-01-01): {cens.sum():,} ({100*cens.mean():.1f}%)")

rho_all = spearmanr(uid, fnum).statistic
rho_unc = spearmanr(uid[~cens], fnum[~cens]).statistic
print(f"\nSpearman(user_id, дата первого события):")
print(f"  все юзеры:            {rho_all:+.4f}")
print(f"  НЕцензурированные:    {rho_unc:+.4f}   <- если id по порядку регистрации, было бы ~+1")

print(f"\nSpearman(user_id, прочее):")
for c in ("days","gmv","ord"):
    print(f"  {c:>5}: {spearmanr(uid, u[c].to_numpy()).statistic:+.4f}")

print("\n=== новички ПОЗДНИХ месяцев: какие у них id ===")
for m in ["2025-01","2025-04","2025-07","2025-10","2026-01"]:
    msk = np.array([str(d)[:7]==m for d in first])
    if msk.sum()>50:
        print(f"  первое событие {m}: n={msk.sum():>7,}  медиана id={np.median(uid[msk]):>9,.0f}  "
              f"p10={np.percentile(uid[msk],10):>9,.0f}  p90={np.percentile(uid[msk],90):>9,.0f}")

print("\n=== прямой замер: даёт ли user_id что-то для таргета ===")
A = date(2026,1,14)
fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
j = u.join(g, on="user_id", how="left").with_columns(pl.col("y").fill_null(0))
y = np.log1p(j["y"].to_numpy())
print(f"  Spearman(user_id, log1p таргета): {spearmanr(j['user_id'].to_numpy(), y).statistic:+.4f}")
q = np.quantile(uid, np.linspace(0,1,11))
b = np.digitize(j["user_id"].to_numpy(), q[1:-1])
print("  дециль id | mean log1p | доля нулей")
for k in range(10):
    m = b==k
    print(f"       {k:>2}    |   {y[m].mean():.4f}   |  {100*(j['y'].to_numpy()[m]==0).mean():5.2f}%")
