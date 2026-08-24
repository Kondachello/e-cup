import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np
import polars as pl

SCR = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
ax = pl.read_parquet(f"{SCR}/user_axes.parquet")
vp = pl.read_parquet(
    "/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
    columns=["user_id", "target", "blend"],
)
df = vp.join(ax, on="user_id", how="left")
y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e = y - b
N = len(e)
ge = e.mean()

def col(name):
    return df[name].to_numpy().astype(np.float64)

gmv90, gmv365 = col("gmv90"), col("gmv365")
ord_days90, ord_days365 = col("ord_days90"), col("ord_days365")
act90, act28 = col("act90"), col("act28")
rec_ord = df["rec_ord"].to_numpy().astype(np.float64)
rec_ord = np.where(np.isnan(rec_ord), 999.0, rec_ord)
sd90 = col("search_days90")
d_mean90 = np.nan_to_num(col("d_mean90"), nan=90.0)
lvl = np.log1p(gmv365)
aov365 = np.where(ord_days365 > 0, gmv365 / np.maximum(ord_days365, 1), np.nan)
with np.errstate(invalid="ignore"):
    search_int = np.where(act90 > 0, sd90 / act90, np.nan)

def qbin(x, K=8):
    x = np.asarray(x, dtype=np.float64)
    nanm = np.isnan(x)
    xs = x[~nanm]
    qs = np.quantile(xs, np.linspace(0, 1, K + 1))
    qs = np.unique(qs)
    codes = np.full(len(x), -1, dtype=np.int32)
    codes[~nanm] = np.searchsorted(qs[1:-1], xs, side="right")
    nb = len(qs) - 1
    codes[nanm] = nb
    return codes, nb + (1 if nanm.any() else 0), qs

def show_map(xname, xv, yname, yv, K=8, top=6):
    cx, nx, qx = qbin(xv, K)
    cy, ny, qy = qbin(yv, K)
    cell = cx * ny + cy
    ncell = nx * ny
    cnt = np.bincount(cell, minlength=ncell).astype(float)
    s1 = np.bincount(cell, weights=e, minlength=ncell)
    s2 = np.bincount(cell, weights=e**2, minlength=ncell)
    m = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0)
    var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - m**2, 1)
    se = np.sqrt(var / np.maximum(cnt, 1))
    z = np.where(cnt > 30, m / np.maximum(se, 1e-12), 0)
    # two-way additive fit (row+col effects, weighted) to isolate interaction
    # iterate: alpha_r, beta_c
    alpha = np.zeros(nx); beta = np.zeros(ny); mu = (e).mean()
    M = m.reshape(nx, ny); C = cnt.reshape(nx, ny)
    for _ in range(50):
        alpha = np.where(C.sum(1) > 0, ((M - mu - beta[None, :]) * C).sum(1) / np.maximum(C.sum(1), 1), 0)
        beta = np.where(C.sum(0) > 0, ((M - mu - alpha[:, None]) * C).sum(0) / np.maximum(C.sum(0), 1), 0)
    fit = mu + alpha[:, None] + beta[None, :]
    inter = (M - fit).ravel()
    z_int = np.where(cnt > 30, inter / np.maximum(se, 1e-12), 0)
    print(f"\n=== {xname} x {yname} ===  bins x: {np.round(qx,2)}")
    print(f"    bins y: {np.round(qy,2)}")
    order = np.argsort(-np.abs(z))
    print(f"{'cellxy':>8} {'n':>7} {'mean_e':>8} {'se':>7} {'z':>6} | {'inter':>8} {'z_int':>6} | mean-глоб z")
    for i in order[:top]:
        if cnt[i] <= 30: continue
        r, c = divmod(i, ny)
        zc = (m[i] - ge) / max(se[i], 1e-12)
        print(f"({r},{c})   {int(cnt[i]):>7} {m[i]:+.4f} {se[i]:.4f} {z[i]:+.2f} | {inter[i]:+.4f} {z_int[i]:+.2f} | {zc:+.2f}")
    # interaction-only significance count
    n2i = int((np.abs(z_int) > 2).sum()); n3i = int((np.abs(z_int) > 3).sum())
    ncells_eff = int((cnt > 30).sum())
    print(f"interaction-only cells |z_int|>2: {n2i} (exp {0.0455*ncells_eff:.1f}), >3: {n3i}, max|z_int|={np.abs(z_int).max():.2f}")
    # potential of interaction only
    pot_i = float(np.sum(cnt * inter**2) / N)
    base = np.sqrt((e**2).mean())
    print(f"pot interaction-only in-sample rmsle: {base - np.sqrt(base**2 - pot_i):.6f}")
    return cx, cy, ny

show_map("search_int", search_int, "lvl", lvl)
show_map("d_mean90", d_mean90, "act90", act90)
show_map("act90", act90, "aov365", aov365)
show_map("rec_ord", rec_ord, "freq365", ord_days365)
show_map("act28", act28, "lvl", lvl)
