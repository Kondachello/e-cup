# eda3 activity ladder: decile levels per month, transition matrices, momentum at anchor
import polars as pl
import numpy as np

df = pl.read_parquet("work/reports/eda3_ladder_monthly.parquet")

def levels(measure):
    pos = df.filter(pl.col(measure) > 0).with_columns(
        rank=pl.col(measure).rank("average").over(["anchor", "k"]),
        n=pl.len().over(["anchor", "k"]),
    ).with_columns(
        lev=(pl.col("rank") / pl.col("n") * 10).ceil().clip(1, 10).cast(pl.Int8)
    ).select("anchor", "user_id", "k", "lev")
    return pos

lev_s = levels("srch").rename({"lev": "Ls"})
lev_g = levels("gmv").rename({"lev": "Lg"})

uni = pl.read_parquet("work/preds_pack/val_preds.parquet", columns=["user_id", "target", "blend"])
users = uni.select("user_id")

# wide matrices per anchor: 250000 x 13, level 0 = inactive that month
out = {}
for anc in ["val", "test"]:
    for nm, lev in [("Ls", lev_s), ("Lg", lev_g)]:
        w = lev.filter(pl.col("anchor") == anc).pivot(on="k", index="user_id", values=nm)
        w = users.join(w, on="user_id", how="left")
        cols = [str(k) for k in range(13)]
        for c in cols:
            if c not in w.columns:
                w = w.with_columns(pl.lit(None).alias(c))
        M = w.select([pl.col(c).fill_null(0).cast(pl.Int8) for c in cols]).to_numpy()
        out[(anc, nm)] = M
        np.save(f"work/reports/eda3_ladder_{anc}_{nm}.npy", M)

# also continuous log1p sums for momentum (val anchor)
for anc in ["val", "test"]:
    w = df.filter(pl.col("anchor") == anc).pivot(on="k", index="user_id", values="srch")
    w = users.join(w, on="user_id", how="left")
    cols = [str(k) for k in range(13)]
    Z = w.select([pl.col(c).fill_null(0).alias(c) for c in cols if c in w.columns]).to_numpy()
    np.save(f"work/reports/eda3_ladder_{anc}_zsrch.npy", np.log1p(Z))
    wg = df.filter(pl.col("anchor") == anc).pivot(on="k", index="user_id", values="gmv")
    wg = users.join(wg, on="user_id", how="left")
    G = wg.select([pl.col(c).fill_null(0).alias(c) for c in cols if c in wg.columns]).to_numpy()
    np.save(f"work/reports/eda3_ladder_{anc}_zgmv.npy", np.log1p(G))

uni.write_parquet("work/reports/eda3_ladder_uni.parquet")

# ---- transition matrices, val anchor, srch ladder ----
M = out[("val", "Ls")]
N = M.shape[0]
print("users", N)

def tmat(a, b):  # from level a (month k+1) to level b (month k)
    T = np.zeros((11, 11), dtype=np.int64)
    np.add.at(T, (a, b), 1)
    return T

pooled = np.zeros((11, 11), dtype=np.int64)
per_pair = []
for k in range(12):
    T = tmat(M[:, k + 1], M[:, k])
    per_pair.append(T)
    if k >= 0:
        pooled += T

def rownorm(T):
    s = T.sum(1, keepdims=True).astype(float)
    s[s == 0] = 1
    return T / s

P = rownorm(pooled)
np.save("work/reports/eda3_ladder_tmat_pooled.npy", pooled)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("pooled transition P (rows=from level month k+1, cols=to level month k), 12 pairs, val anchor:")
print(P)

# stability: holiday pair (1->0: Dec peak -> Jan dip) vs calm pairs mean (k=4..10)
calm = rownorm(sum(per_pair[4:11]))
hol = rownorm(per_pair[0])  # k=0: month1(20.11-17.12) -> month0(18.12-14.01)
d = np.abs(hol - calm).sum(1)
print(" dist per row, pair(1->0 holiday) vs calm mean:", np.round(d, 3))

# diagonal mass (stay +-1) pooled
stay = sum(pooled[i, max(0, i - 1):i + 2].sum() for i in range(11)) / pooled.sum()
print("pooled mass within |dL|<=1:", round(stay, 4))

# ---- momentum shares at val vs test anchors (S-axis) ----
for anc in ["val", "test"]:
    A = out[(anc, "Ls")]
    d1 = A[:, 0].astype(int) - A[:, 1]
    acc = A[:, 0].astype(int) - 2 * A[:, 1] + A[:, 2]
    print(anc, "share dL01>=+2:", round((d1 >= 2).mean(), 4), "dL01<=-2:", round((d1 <= -2).mean(), 4),
          "acc>=+3:", round((acc >= 3).mean(), 4), "acc<=-3:", round((acc <= -3).mean(), 4),
          "mean dL01:", round(d1.mean(), 3), "mean acc:", round(acc.mean(), 3))
