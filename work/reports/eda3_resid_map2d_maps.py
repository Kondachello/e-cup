import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np
import polars as pl
import json

SCR = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
ax = pl.read_parquet(f"{SCR}/user_axes.parquet")
vp = pl.read_parquet(
    "/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
    columns=["user_id", "target", "blend"],
)
df = vp.join(ax, on="user_id", how="left")
print("joined", df.shape, "missing axes rows:", df["n_rows"].null_count())

y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e = y - b
N = len(e)
base_rmsle = float(np.sqrt(np.mean(e**2)))
print(f"baseline RMSLE {base_rmsle:.7f}  mean(e) {e.mean():+.6f}")

# ---------- axis definitions ----------
def col(name):
    return df[name].to_numpy().astype(np.float64)

gmv90 = col("gmv90")
gmv365 = col("gmv365")
ord_days90 = col("ord_days90")
ord_days365 = col("ord_days365")
ord_cnt90 = col("ord_cnt90")
act90 = col("act90")
act28 = col("act28")
act365 = col("act365")
rec_any = col("rec_any")
rec_ord = df["rec_ord"].to_numpy().astype(np.float64)  # has nulls -> nan
rec_ord = np.where(np.isnan(rec_ord), 999.0, rec_ord)
gs = col("gmv_search365")
gc = col("gmv_cat365")
cart90 = col("cart90")
search90 = col("search90")
sd90 = col("search_days90")
d_mean90 = np.nan_to_num(col("d_mean90"), nan=90.0)
d_std90 = np.nan_to_num(col("d_std90"), nan=-1.0)
dow = np.stack([col(f"dow{k}") for k in range(7)], axis=1)

aov90 = np.where(ord_days90 > 0, gmv90 / np.maximum(ord_days90, 1), np.nan)
aov365 = np.where(ord_days365 > 0, gmv365 / np.maximum(ord_days365, 1), np.nan)
lvl = np.log1p(gmv365)
chan = np.where(gs + gc > 0, gs / (gs + gc), np.nan)
cart_ratio = np.where(cart90 + ord_cnt90 > 0, cart90 / (cart90 + ord_cnt90), np.nan)
dow_top = np.where(ord_days365 > 0, dow.max(axis=1) / np.maximum(ord_days365, 1), np.nan)
search_int = np.where(act90 > 0, sd90 / act90, np.nan)  # доля поисковых дней среди активных

AXES = {
    "act90": act90,
    "act28": act28,
    "aov365": aov365,
    "rec_ord": rec_ord,
    "rec_any": rec_any,
    "freq365": ord_days365,
    "lvl": lvl,
    "chan": chan,
    "cart_ratio": cart_ratio,
    "dow_top": dow_top,
    "d_mean90": d_mean90,
    "d_std90": d_std90,
    "search_int": search_int,
    "blendv": b,
}

def qbin(x, K=8):
    """quantile bins; NaN -> separate bin K (last). returns codes 0..K (K = nan/degenerate)"""
    x = np.asarray(x, dtype=np.float64)
    nanm = np.isnan(x)
    xs = x[~nanm]
    qs = np.quantile(xs, np.linspace(0, 1, K + 1))
    qs = np.unique(qs)
    codes = np.full(len(x), -1, dtype=np.int32)
    c = np.searchsorted(qs[1:-1], xs, side="right")
    codes[~nanm] = c
    nb = len(qs) - 1
    codes[nanm] = nb
    return codes, nb + (1 if nanm.any() else 0)

def map2d(name_x, name_y, K=8):
    cx, nx = qbin(AXES[name_x], K)
    cy, ny = qbin(AXES[name_y], K)
    cell = cx * ny + cy
    ncell = nx * ny
    cnt = np.bincount(cell, minlength=ncell).astype(np.float64)
    s1 = np.bincount(cell, weights=e, minlength=ncell)
    s2 = np.bincount(cell, weights=e**2, minlength=ncell)
    m = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - m**2, 0.0)
    se = np.sqrt(np.where(cnt > 1, var / np.maximum(cnt, 1), np.inf))
    z = np.where(cnt > 30, m / np.maximum(se, 1e-12), 0.0)
    mask = cnt > 30
    ncells_eff = int(mask.sum())
    n2 = int((np.abs(z) > 2).sum())
    n3 = int((np.abs(z) > 3).sum())
    n4 = int((np.abs(z) > 4).sum())
    chi2 = float(np.sum(np.where(mask, cnt * m**2 / np.maximum(var, 1e-12), 0)))
    # in-sample potential MSE reduction if all cell means removed
    pot = float(np.sum(cnt * m**2) / N)
    pot_rmsle = base_rmsle - np.sqrt(max(base_rmsle**2 - pot, 0))
    return dict(
        pair=f"{name_x} x {name_y}", nx=nx, ny=ny, ncells=ncells_eff,
        n_z2=n2, n_z3=n3, n_z4=n4, exp_z2=round(0.0455 * ncells_eff, 1),
        chi2=round(chi2, 1), maxabsz=round(float(np.abs(z).max()), 2),
        pot_mse=pot, pot_rmsle_insample=round(pot_rmsle, 6),
        cell=cell, cnt=cnt, m=m, se=se, z=z, cx=cx, cy=cy,
    )

PAIRS = [
    ("act90", "aov365"),     # активность x чек
    ("rec_ord", "freq365"),  # свежесть x частота
    ("act28", "lvl"),        # явка x уровень
    ("rec_any", "lvl"),
    ("chan", "lvl"),         # канал x уровень
    ("cart_ratio", "freq365"),
    ("dow_top", "freq365"),
    ("d_mean90", "act90"),   # темп (передний/задний фронт активности) x активность
    ("d_std90", "act90"),
    ("search_int", "lvl"),
    ("blendv", "rec_ord"),
    ("blendv", "chan"),
]

results = []
maps = {}
for px, py in PAIRS:
    r = map2d(px, py)
    maps[(px, py)] = r
    results.append({k: v for k, v in r.items() if k not in ("cell", "cnt", "m", "se", "z", "cx", "cy")})
    print(f"{r['pair']:>24}: cells={r['ncells']:3d} |z|>2: {r['n_z2']:3d} (exp {r['exp_z2']}) |z|>3: {r['n_z3']:2d} |z|>4: {r['n_z4']:2d} chi2={r['chi2']:8.1f} max|z|={r['maxabsz']:5.2f} pot_in={r['pot_rmsle_insample']:.6f}")

np.save(f"{SCR}/e.npy", e)
np.save(f"{SCR}/blend.npy", b)
import pickle
with open(f"{SCR}/maps.pkl", "wb") as f:
    pickle.dump({f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk in ("cell","cnt","m","se","z")} for k, v in maps.items()}, f)
with open(f"{SCR}/map_summary.json", "w") as f:
    json.dump(results, f, indent=1)
