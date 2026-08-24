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
base = float(np.sqrt(np.mean(e**2)))

def col(name):
    return df[name].to_numpy().astype(np.float64)

act90 = col("act90"); rec_any = col("rec_any"); act365 = col("act365")
gs, gc = col("gmv_search365"), col("gmv_cat365")
gmv365 = col("gmv365"); ord_days365 = col("ord_days365")
cart90 = col("cart90"); ord_cnt90 = col("ord_cnt90")
rec_ord = df["rec_ord"].to_numpy().astype(np.float64)
rec_ord = np.where(np.isnan(rec_ord), 999.0, rec_ord)
lvl = np.log1p(gmv365)
with np.errstate(invalid="ignore"):
    chan = np.where(gs + gc > 0, gs / (gs + gc), np.nan)
    cart_ratio = np.where(cart90 + ord_cnt90 > 0, cart90 / (cart90 + ord_cnt90), np.nan)

print("universe sanity: min act90 =", act90.min(), " max rec_any =", rec_any.max(),
      " users act90==0:", int((act90 == 0).sum()), " min act365:", act365.min())

def qbin(x, K=8):
    x = np.asarray(x, dtype=np.float64)
    nanm = np.isnan(x)
    xs = x[~nanm]
    qs = np.unique(np.quantile(xs, np.linspace(0, 1, K + 1)))
    codes = np.full(len(x), -1, dtype=np.int32)
    codes[~nanm] = np.searchsorted(qs[1:-1], xs, side="right")
    nb = len(qs) - 1
    codes[nanm] = nb
    return codes, nb + (1 if nanm.any() else 0)

def interaction_test(xname, xv, yname, yv, K=8):
    cx, nx = qbin(xv, K)
    cy, ny = qbin(yv, K)
    cell = cx * ny + cy
    ncell = nx * ny
    cnt = np.bincount(cell, minlength=ncell).astype(float)
    s1 = np.bincount(cell, weights=e, minlength=ncell)
    s2 = np.bincount(cell, weights=e**2, minlength=ncell)
    m = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0)
    var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - m**2, 1)
    se = np.sqrt(var / np.maximum(cnt, 1))
    alpha = np.zeros(nx); beta = np.zeros(ny); mu = e.mean()
    M = m.reshape(nx, ny); C = cnt.reshape(nx, ny)
    for _ in range(80):
        alpha = np.where(C.sum(1) > 0, ((M - mu - beta[None, :]) * C).sum(1) / np.maximum(C.sum(1), 1), 0)
        beta = np.where(C.sum(0) > 0, ((M - mu - alpha[:, None]) * C).sum(0) / np.maximum(C.sum(0), 1), 0)
    inter = (M - (mu + alpha[:, None] + beta[None, :])).ravel()
    z_int = np.where(cnt > 30, inter / np.maximum(se, 1e-12), 0)
    ncf = int((cnt > 30).sum())
    n2 = int((np.abs(z_int) > 2).sum()); n3 = int((np.abs(z_int) > 3).sum())
    pot = float(np.sum(cnt * inter**2) / N)
    print(f"{xname:>10} x {yname:<8}: cells {ncf:3d}, |z_int|>2: {n2} (exp {0.0455*ncf:.1f}), >3: {n3}, "
          f"max|z_int| {np.abs(z_int).max():.2f}, pot_int {base - np.sqrt(base**2 - pot):.6f}")
    return z_int, cnt

interaction_test("chan", chan, "lvl", lvl)
interaction_test("blendv", b, "rec_ord", rec_ord)
interaction_test("blendv", b, "chan", chan)
interaction_test("cart_ratio", cart_ratio, "freq365", ord_days365)
interaction_test("blendv", b, "lvl", lvl)
interaction_test("rec_any", rec_any, "act90", act90)

# grand total: joint 2D corrector over ALL 12 maps, cross-fitted, to bound the whole lens
maps = [("act90", act90, "aov365", np.where(ord_days365 > 0, gmv365 / np.maximum(ord_days365, 1), np.nan)),
        ("rec_ord", rec_ord, "freq365", ord_days365),
        ("act28", col("act28"), "lvl", lvl),
        ("chan", chan, "lvl", lvl),
        ("blendv", b, "rec_ord", rec_ord)]
fold = np.random.default_rng(0).integers(0, 5, N)
ecur = e.copy()
for nmx, xv, nmy, yv in maps:
    cx, nx = qbin(xv, 8); cy, ny = qbin(yv, 8)
    cell = cx * ny + cy
    c = np.zeros(N)
    for f in range(5):
        tr = fold != f; te = fold == f
        cntf = np.bincount(cell[tr], minlength=nx * ny).astype(float)
        mf = np.bincount(cell[tr], weights=ecur[tr], minlength=nx * ny) / np.maximum(cntf, 1)
        mf = mf * cntf / (cntf + 100)  # shrink
        c[te] = mf[cell[te]]
    ecur = ecur - c
g_joint = base - np.sqrt((ecur**2).mean())
print(f"\nстек 5 карт 2D (cross-fit, shrink 100), последовательно: суммарный честный gain {g_joint:+.6f}")
