"""H3. Жизненный цикл юзера и непохожие юзеры — с привязкой к ошибке бленда."""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet")
A = date(2026,1,14)

print("=== ЖИЗНЕННЫЙ ЦИКЛ: активность по возрасту (только НЕцензурированные новички) ===")
u0 = df.group_by("user_id").agg(pl.col("event_date").min().alias("first"))
fresh = u0.filter(pl.col("first") > date(2025,2,1))     # цензура слева снята
print(f"юзеров с известной датой прихода: {fresh.height:,}")
d = df.join(fresh, on="user_id", how="inner").with_columns(
    (pl.col("event_date")-pl.col("first")).dt.total_days().alias("age"))
print(f"{'возраст, дней':>14} {'юзеров':>9} {'searches':>10} {'to_cart':>9} {'доля с заказом':>15} {'gmv/день':>10}")
for lo,hi,lab in [(0,0,"0 (первый)"),(1,6,"1-6"),(7,13,"7-13"),(14,29,"14-29"),(30,59,"30-59"),
                  (60,89,"60-89"),(90,179,"90-179"),(180,364,"180-364")]:
    s = d.filter((pl.col("age")>=lo)&(pl.col("age")<=hi))
    if s.height:
        print(f"{lab:>14} {s['user_id'].n_unique():>9,} {s['searches'].mean():>10.3f} "
              f"{s['to_cart'].mean():>9.3f} {(s['to_ord']>0).mean():>15.4f} {s['gmv'].mean():>10.2f}")

print("\n=== ВЫЖИВАЕМОСТЬ: доля новичков, ещё активных через N дней после прихода ===")
last = df.join(fresh, on="user_id", how="inner").group_by(["user_id","first"]).agg(
    pl.col("event_date").max().alias("last"))
life = (last["last"]-last["first"]).dt.total_days().to_numpy()
for t in (7,14,30,60,90,180,270):
    print(f"  прожили >= {t:>3} дней: {100*(life>=t).mean():5.1f}%")

print("\n=== НЕПОХОЖИЕ ЮЗЕРЫ: профиль СОСТАВА поведения (не уровня) ===")
h = df.filter(pl.col("event_date")>A-timedelta(days=180)).group_by("user_id").agg([
    pl.len().alias("n"), pl.col("searches").sum().alias("s"), pl.col("to_cart").sum().alias("c"),
    pl.col("to_ord").sum().alias("o"), (pl.col("cat")>0).sum().alias("cd"),
    ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0)).sum().alias("e"),
    pl.col("gmv").sum().alias("g")])
h = h.with_columns([
    (pl.col("s")/pl.max_horizontal(pl.col("n"),pl.lit(1))).alias("p_srch"),
    (pl.col("c")/pl.max_horizontal(pl.col("s"),pl.lit(1))).alias("p_conv1"),
    (pl.col("o")/pl.max_horizontal(pl.col("c"),pl.lit(1))).alias("p_conv2"),
    (pl.col("cd")/pl.max_horizontal(pl.col("n"),pl.lit(1))).alias("p_cat"),
    (pl.col("e")/pl.max_horizontal(pl.col("n"),pl.lit(1))).alias("p_emp"),
    (pl.col("g")/pl.max_horizontal(pl.col("o"),pl.lit(1))).alias("p_aov")])
P = h.select(["p_srch","p_conv1","p_conv2","p_cat","p_emp","p_aov"]).to_numpy()
P = np.nan_to_num(P, posinf=0, neginf=0); P = np.log1p(np.abs(P))*np.sign(P)
Z = (P - P.mean(0))/(P.std(0)+1e-9)
dist = np.sqrt((Z**2).sum(1))
print(f"  расстояние от центра профиля: p50={np.median(dist):.2f} p95={np.percentile(dist,95):.2f} max={dist.max():.2f}")
for t in (3,4,5,6):
    print(f"  дальше {t} сигм: {int((dist>t).sum()):>7,} ({100*(dist>t).mean():5.2f}%)")

v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
err = v["blend"].to_numpy().astype(np.float64) - ly
dd = pl.DataFrame({"user_id":h["user_id"], "dist":dist}).join(
     pl.DataFrame({"user_id":v["user_id"], "err":err, "ae":err**2}), on="user_id", how="inner")
print(f"\n=== СТОЯТ ЛИ ОНИ ЧЕГО-ТО: ошибка бленда по децилям непохожести ===")
q = np.quantile(dd["dist"].to_numpy(), np.linspace(0,1,11))
b = np.digitize(dd["dist"].to_numpy(), q[1:-1]); e = dd["err"].to_numpy()
tot = np.mean(e**2)
print("  дец | ср.расст | ср.ошибка | RMSE  | доля общего MSE")
for k in range(10):
    m = b==k
    print(f"   {k:>2} |  {dd['dist'].to_numpy()[m].mean():6.2f}  |  {e[m].mean():+7.4f}  | {np.sqrt(np.mean(e[m]**2)):.3f} |   {100*np.mean(e[m]**2)*m.mean()/tot:5.2f}%")
