# Адверсарная НЕЗАВИСИМАЯ репликация кандидата seg_gc_nocat
# (день gmv_cat>0 при cat=0 в истории) против остатка бленда.
# Всё своим кодом, без кода охотника.
import json
import numpy as np
import polars as pl

RNG = np.random.default_rng(7)
ANCHOR_VAL = pl.date(2026, 1, 14)
ANCHOR_TEST = pl.date(2026, 2, 13)

lf = pl.scan_parquet("train.parquet")

# --- пер-юзерные агрегаты на двух якорях ---
def agg_at(anchor_expr, pref):
    h = lf.filter(pl.col("event_date") <= anchor_expr)
    gc_nocat_day = (pl.col("gmv_cat") > 0) & (pl.col("cat") == 0)
    return h.group_by("user_id").agg(
        pl.len().alias(f"{pref}_rows"),
        gc_nocat_day.sum().alias(f"{pref}_n_gc_nocat"),
        (pl.col("gmv_cat") > 0).sum().alias(f"{pref}_n_gmvcat_days"),
        (pl.col("cat") > 0).sum().alias(f"{pref}_n_cat_days"),
        (pl.col("has_cat_to_ord") > 0).sum().alias(f"{pref}_n_c2o_days"),
        (pl.col("to_ord") > 0).sum().alias(f"{pref}_n_ord_days"),
        pl.col("gmv").sum().alias(f"{pref}_gmv_sum"),
        pl.col("gmv_cat").sum().alias(f"{pref}_gmvcat_sum"),
        pl.col("event_date").filter(pl.col("cat") > 0).max().alias(f"{pref}_last_cat"),
        pl.col("event_date").filter(pl.col("gmv_cat") > 0).max().alias(f"{pref}_last_gmvcat"),
        pl.col("event_date").filter(pl.col("to_ord") > 0).max().alias(f"{pref}_last_ord"),
        pl.col("event_date").filter(gc_nocat_day).max().alias(f"{pref}_last_gcnc"),
        pl.col("event_date").max().alias(f"{pref}_last_any"),
    )

va = agg_at(ANCHOR_VAL, "v").collect(engine="streaming")
ta = agg_at(ANCHOR_TEST, "t").collect(engine="streaming")

# глобальная проверка сырых фактов охотника
raw = lf.select(
    ((pl.col("gmv_cat") > 0) & (pl.col("cat") == 0)).sum().alias("rows_gc_nocat"),
    (pl.col("gmv_cat") > 0).sum().alias("rows_gmvcat"),
    (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs().max().alias("max_resid"),
).collect(engine="streaming")
print("RAW rows gmv_cat>0 & cat==0:", raw["rows_gc_nocat"][0],
      "of", raw["rows_gmvcat"][0], "gmv_cat days",
      f"({raw['rows_gc_nocat'][0]/raw['rows_gmvcat'][0]*100:.1f}%)",
      "| max|gmv-gs-gc|:", raw["max_resid"][0])

d = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
uid = d["user_id"].to_numpy()
y = np.log1p(d["target"].to_numpy().astype(float))
b = d["blend"].to_numpy().astype(float)
e = y - b
print(f"blend RMSE = {np.sqrt((e**2).mean()):.6f}  (эталон 1.665647)  mean_e = {e.mean():+.5f}")

base = pl.DataFrame({"user_id": uid})
va = base.join(va, on="user_id", how="left")
ta = base.join(ta, on="user_id", how="left")

def flag_of(df, pref):
    return (df[f"{pref}_n_gc_nocat"].fill_null(0).to_numpy() > 0)

fv = flag_of(va, "v")
ft = flag_of(ta, "t")
inter = (fv & ft).sum()
jac = inter / ((fv | ft).sum())
print(f"segment val {fv.sum()} ({fv.mean()*100:.1f}%)  test {ft.sum()} ({ft.mean()*100:.1f}%)  Jaccard {jac:.3f}")

# --- ключевые статистики сигнала ---
gap = e[fv].mean() - e[~fv].mean()
r = np.corrcoef(fv.astype(float), e)[0, 1]
# сегментная сигма
se_gap = np.sqrt(e[fv].var()/fv.sum() + e[~fv].var()/(~fv).sum())
print(f"corr(flag,e) = {r:+.4f}   mean_e seg {e[fv].mean():+.4f} vs rest {e[~fv].mean():+.4f}  gap {gap:+.4f} ± {se_gap:.4f}")

# матчинг по децилям бленда
qs = np.quantile(b, np.linspace(0, 1, 11))
qs[0] -= 1; qs[-1] += 1
dec = np.digitize(b, qs[1:-1])
gaps = []
ws = []
for k in range(10):
    m = dec == k
    if fv[m].sum() > 30 and (~fv[m]).sum() > 30:
        g = e[m & fv].mean() - e[m & ~fv].mean()
        w = fv[m].sum()
        gaps.append(g); ws.append(w)
gaps = np.array(gaps); ws = np.array(ws, float)
mg = (gaps * ws).sum() / ws.sum()
seg_per = [f"{g:+.3f}" for g in gaps]
print(f"decile-matched gap (weighted) = {mg:+.4f}   per-decile: {seg_per}")

# --- OOF корректор: 2-fold по чётности ---
def oof_gain(e, flag, folds, mode="full"):
    corr = np.zeros_like(e)
    betas = []
    for f in np.unique(folds):
        tr = folds != f
        te = folds == f
        b0 = e[tr & ~flag].mean()
        b1 = e[tr & flag].mean() - b0
        betas.append((b0, b1))
        if mode == "full":
            corr[te] = b0 + b1 * flag[te]
        elif mode == "level":
            m = e[tr].mean()
            corr[te] = m
        elif mode == "flag_centered":
            p = flag[tr].mean()
            corr[te] = b1 * (flag[te].astype(float) - p)
    rm0 = np.sqrt((e**2).mean())
    rm1 = np.sqrt(((e - corr)**2).mean())
    return rm0 - rm1, corr, betas

folds2 = (uid % 2).astype(int)
g_full, c_full, betas2 = oof_gain(e, fv, folds2, "full")
g_lvl, c_lvl, _ = oof_gain(e, fv, folds2, "level")
g_ctr, c_ctr, _ = oof_gain(e, fv, folds2, "flag_centered")
print(f"2-fold OOF gain: full(b0+b1*flag) {g_full*1e4:.2f}e-4 | intercept-only {g_lvl*1e4:.2f}e-4 | flag-centered-only {g_ctr*1e4:.2f}e-4")
print(f"  betas per fold (b0,b1): {[(round(a,4), round(c,4)) for a,c in betas2]}")
print(f"  marginal flag gain (full - level) = {(g_full-g_lvl)*1e4:.2f}e-4")

# 5-fold по хэшу юзера
folds5 = (uid * 2654435761 % 2**32 % 5).astype(int)
g5, c5, betas5 = oof_gain(e, fv, folds5, "full")
g5l, _, _ = oof_gain(e, fv, folds5, "level")
g5c, c5c, _ = oof_gain(e, fv, folds5, "flag_centered")
print(f"5-fold OOF gain: full {g5*1e4:.2f}e-4 | level {g5l*1e4:.2f}e-4 | flag-centered {g5c*1e4:.2f}e-4 | b1 range {min(x[1] for x in betas5):.3f}..{max(x[1] for x in betas5):.3f}")

# --- концентрация выигрыша (по центрированному флаговому корректору, 2-fold) ---
def concentration(e, corr):
    d = e**2 - (e - corr)**2  # вклад юзера в снижение MSE (положит. = выигрыш)
    tot = d.sum()
    idx = np.argsort(-np.abs(d))
    for frac in [0.01, 0.001]:
        k = int(len(d) * frac)
        drop = idx[:k]
        keep = np.ones(len(d), bool); keep[drop] = False
        surviving = d[keep].sum() / tot
        top_share = d[drop].sum() / tot
        print(f"  top-{frac*100:.1f}% by |d|: несут {top_share*100:.0f}% суммы, без них выживает {surviving*100:.0f}%")
    print(f"  users improved (d>0): {(d>0).mean()*100:.1f}%  ухудшены: {(d<0).mean()*100:.1f}%  нетронуты: {(d==0).mean()*100:.1f}%")
    return d

print("концентрация: full corrector")
d_full = concentration(e, c_full)
print("концентрация: flag-centered corrector")
d_ctr = concentration(e, c_ctr)

# --- бутстреп CI гейна (по юзерам, 2-fold full и centered) ---
def boot_ci(e, corr, n=400):
    n_u = len(e)
    gains = np.empty(n)
    d = e**2 - (e - corr)**2
    mse0 = (e**2).mean()
    for i in range(n):
        s = RNG.integers(0, n_u, n_u)
        m0 = (e[s]**2).mean()
        m1 = m0 - d[s].mean()
        gains[i] = np.sqrt(m0) - np.sqrt(m1)
    return np.percentile(gains, [2.5, 50, 97.5])
ci_f = boot_ci(e, c_full)
ci_c = boot_ci(e, c_ctr)
print(f"bootstrap CI gain full: [{ci_f[0]*1e4:.2f}, {ci_f[1]*1e4:.2f}, {ci_f[2]*1e4:.2f}]e-4")
print(f"bootstrap CI gain flag-centered: [{ci_c[0]*1e4:.2f}, {ci_c[1]*1e4:.2f}, {ci_c[2]*1e4:.2f}]e-4")

# --- воспроизводимость на других моделях пака ---
print("gap по моделям пака:")
for col in ["kostya46_cal", "fusion_v3c_avg_cal", "wklin", "gseq_small_s42_cal", "c_ts2_s42_cal"]:
    em = y - d[col].to_numpy().astype(float)
    print(f"  {col:22s} gap {em[fv].mean()-em[~fv].mean():+.4f}")

# --- сохранение пер-юзерных векторов для следующих скриптов ---
import datetime as dt
anch = dt.date(2026, 1, 14)
def rec_from(col):
    days = va[col].to_numpy()
    out = np.full(len(days), 9999.0)
    mask = va[col].is_not_null().to_numpy()
    vals = va[col].fill_null(dt.date(2025, 1, 1)).to_numpy()
    out[mask] = np.array([(anch - x).days for x in vals[mask]], float)
    return out
out = pl.DataFrame({
    "user_id": uid,
    "flag_v": fv, "flag_t": ft,
    "n_gc_nocat": va["v_n_gc_nocat"].fill_null(0),
    "n_gmvcat_days": va["v_n_gmvcat_days"].fill_null(0),
    "n_cat_days": va["v_n_cat_days"].fill_null(0),
    "n_c2o_days": va["v_n_c2o_days"].fill_null(0),
    "n_ord_days": va["v_n_ord_days"].fill_null(0),
    "gmv_sum": va["v_gmv_sum"].fill_null(0.0),
    "gmvcat_sum": va["v_gmvcat_sum"].fill_null(0.0),
    "rec_cat": rec_from("v_last_cat"),
    "rec_gmvcat": rec_from("v_last_gmvcat"),
    "rec_ord": rec_from("v_last_ord"),
    "rec_gcnc": rec_from("v_last_gcnc"),
    "e": e, "blend": b, "y": y,
})
out.write_parquet("work/reports/eda3_adv_gcnocat_user.parquet")
print("saved work/reports/eda3_adv_gcnocat_user.parquet")
