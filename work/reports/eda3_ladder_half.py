# eda3 activity ladder: 14d half-month momentum at val anchor + sharp reactivation/collapse patterns
import polars as pl
import numpy as np
import datetime as dt

anc = dt.date(2026, 1, 14)
lf = pl.scan_parquet("train.parquet")
lf = lf.with_columns(((pl.lit(anc) - pl.col("event_date")).dt.total_days()).alias("d"))
lf = lf.filter((pl.col("d") >= 0) & (pl.col("d") < 70)).with_columns((pl.col("d") // 14).alias("h").cast(pl.Int8))
agg = lf.group_by("user_id", "h").agg(pl.col("searches").sum().alias("srch")).collect(engine="streaming")

pos = agg.filter(pl.col("srch") > 0).with_columns(
    rank=pl.col("srch").rank("average").over("h"), n=pl.len().over("h")
).with_columns(lev=(pl.col("rank") / pl.col("n") * 10).ceil().clip(1, 10).cast(pl.Int8))

uni = pl.read_parquet("work/reports/eda3_ladder_uni.parquet")
w = pos.pivot(on="h", index="user_id", values="lev")
w = uni.select("user_id").join(w, on="user_id", how="left")
H = w.select([pl.col(str(k)).fill_null(0).cast(pl.Int8) for k in range(5)]).to_numpy().astype(np.int16)

e = (np.log1p(uni["target"].to_numpy()) - uni["blend"].to_numpy().astype(np.float64))
N = len(e)
mse0 = np.mean(e ** 2)
rng = np.random.default_rng(11)

def bin_report(name, key, min_n=300):
    dmse_or = 0.0
    rows = []
    for b in np.unique(key[key >= 0]):
        s = key == b
        n = s.sum()
        if n < min_n:
            continue
        mu = e[s].mean()
        t = mu / (e[s].std(ddof=1) / np.sqrt(n))
        dmse_or += n * mu * mu / N
        rows.append((int(b), int(n), round(mu, 4), round(t, 2)))
    print(f"[{name}] {rows} oracle dMSE={dmse_or:.6f} dRMSLE={np.sqrt(mse0)-np.sqrt(mse0-dmse_or):.6f}")

def oof(name, key, reps=20, min_n=50):
    gains, conc1, conc01 = [], [], []
    m = key >= 0
    for r in range(reps):
        fold = rng.random(N) < 0.5
        corr = np.zeros(N)
        for fa, fb in [(fold, ~fold), (~fold, fold)]:
            for b in np.unique(key[fa & m]):
                s = fa & m & (key == b)
                if s.sum() >= min_n:
                    corr[fb & m & (key == b)] = e[s].mean()
        g = e ** 2 - (e - corr) ** 2
        gains.append(g.mean())
        order = np.argsort(-np.abs(g))
        tot = g.sum()
        if abs(tot) > 1e-12:
            conc1.append(g[order[: N // 100]].sum() / tot)
            conc01.append(g[order[: N // 1000]].sum() / tot)
    gains = np.array(gains)
    dmse = gains.mean()
    dr = np.sqrt(mse0) - np.sqrt(mse0 - dmse)
    print(f"[{name}] OOF dMSE={dmse:.6f} +-{gains.std():.6f} -> dRMSLE={dr:.6f}; conc1%={np.mean(conc1):.2f} conc0.1%={np.mean(conc01):.2f}")

# half-month acceleration h0-2h1+h2
acc = H[:, 0].astype(int) - 2 * H[:, 1] + H[:, 2]
bins = np.clip((acc + 12) // 3, 0, 8)
bin_report("acc14d", bins)
oof("acc14d", bins)

# jerk: change of acceleration (3rd diff) sharp bins
jerk = (H[:, 0].astype(int) - 3 * H[:, 1] + 3 * H[:, 2] - H[:, 3])
jb = np.clip((jerk + 16) // 4, 0, 7)
bin_report("jerk14d", jb)
oof("jerk14d", jb)

# ---- sharp patterns on monthly ladder ----
Ls = np.load("work/reports/eda3_ladder_val_Ls.npy").astype(np.int16)
plateau_low = (Ls[:, 2:7] <= 2).all(1)
react = plateau_low & (Ls[:, 0] >= 8)          # dormant 5 months -> top-3 decile now
react_mid = plateau_low & (Ls[:, 0].astype(int) - Ls[:, 2:7].max(1) >= 4) & (Ls[:, 0] < 8)
plateau_high = (Ls[:, 2:7] >= 7).all(1)
collapse = plateau_high & (Ls[:, 0] <= 2)      # was top 5 months -> bottom now
fresh = (Ls[:, 3:] == 0).all(1) & (Ls[:, 2] == 0) & (Ls[:, 0] > 0)  # first activity in last 2 months
key = np.full(N, -1)
key[react] = 0; key[react_mid] = 1; key[collapse] = 2; key[fresh] = 3
for nm, s in [("react_hi", react), ("react_mid", react_mid), ("collapse", collapse), ("fresh<2mo", fresh)]:
    n = s.sum()
    if n:
        mu = e[s].mean(); t = mu / (e[s].std(ddof=1) / np.sqrt(n))
        print(f"pattern {nm}: n={n} mean_e={mu:.4f} t={t:.2f} dMSE_or={n*mu*mu/N:.6f}")
oof("sharp_patterns", key, min_n=30)
