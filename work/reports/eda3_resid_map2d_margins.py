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
y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e0 = y - b
N = len(e0)
base = float(np.sqrt(np.mean(e0**2)))

def col(name):
    return df[name].to_numpy().astype(np.float64)

gmv90, gmv365 = col("gmv90"), col("gmv365")
ord_days90, ord_days365 = col("ord_days90"), col("ord_days365")
ord_cnt90 = col("ord_cnt90")
act90, act28, act365 = col("act90"), col("act28"), col("act365")
rec_any = col("rec_any")
rec_ord = df["rec_ord"].to_numpy().astype(np.float64)
rec_ord = np.where(np.isnan(rec_ord), 999.0, rec_ord)
gs, gc = col("gmv_search365"), col("gmv_cat365")
cart90, search90, sd90 = col("cart90"), col("search90"), col("search_days90")
d_mean90 = np.nan_to_num(col("d_mean90"), nan=90.0)
d_std90 = np.nan_to_num(col("d_std90"), nan=-1.0)
dow = np.stack([col(f"dow{k}") for k in range(7)], axis=1)
lvl = np.log1p(gmv365)
with np.errstate(invalid="ignore"):
    aov365 = np.where(ord_days365 > 0, gmv365 / np.maximum(ord_days365, 1), np.nan)
    chan = np.where(gs + gc > 0, gs / (gs + gc), np.nan)
    cart_ratio = np.where(cart90 + ord_cnt90 > 0, cart90 / (cart90 + ord_cnt90), np.nan)
    dow_top = np.where(ord_days365 > 0, dow.max(axis=1) / np.maximum(ord_days365, 1), np.nan)
    search_int = np.where(act90 > 0, sd90 / act90, np.nan)

AXES = {
    "search_int": search_int, "chan": chan, "cart_ratio": cart_ratio,
    "dow_top": dow_top, "d_mean90": d_mean90, "d_std90": d_std90,
    "act90": act90, "act28": act28, "rec_any": rec_any, "rec_ord": rec_ord,
    "freq365": ord_days365, "lvl": lvl, "aov365": aov365, "blendv": b,
}

def qbin(x, K=10):
    x = np.asarray(x, dtype=np.float64)
    nanm = np.isnan(x)
    xs = x[~nanm]
    qs = np.unique(np.quantile(xs, np.linspace(0, 1, K + 1)))
    codes = np.full(len(x), -1, dtype=np.int32)
    codes[~nanm] = np.searchsorted(qs[1:-1], xs, side="right")
    nb = len(qs) - 1
    codes[nanm] = nb
    return codes, nb + (1 if nanm.any() else 0)

rng = np.random.default_rng(0)
fold = rng.integers(0, 5, N)

def cv_margin_correct(e, codes, ncodes, shrink=0.0):
    """5-fold user CV: bin-mean correction from out-of-fold; returns corrected e and correction c"""
    c = np.zeros(N)
    for f in range(5):
        tr = fold != f
        te = fold == f
        cnt = np.bincount(codes[tr], minlength=ncodes).astype(float)
        s1 = np.bincount(codes[tr], weights=e[tr], minlength=ncodes)
        m = s1 / np.maximum(cnt, 1)
        if shrink > 0:
            m = m * cnt / (cnt + shrink)
        c[te] = m[codes[te]]
    return e - c, c

def contribution_shares(e_new, e_old):
    s_new = np.sqrt((e_new**2).mean()); s_old = np.sqrt((e_old**2).mean())
    ci = (e_new**2 - e_old**2) / (N * (s_new + s_old))
    delta = ci.sum()
    order = np.argsort(-np.abs(ci))
    t01 = ci[order[:250]].sum() / delta if delta != 0 else np.nan
    t1 = ci[order[:2500]].sum() / delta if delta != 0 else np.nan
    return delta, t01, t1

print(f"base {base:.7f}")
print(f"{'axis':>11} | {'gain_oof':>9} | {'top0.1%':>7} {'top1%':>6} | {'gain|blendcal':>12} | {'gain|blend+seg':>12}")

# pre-computed conditioners
codes_b, nb_b = qbin(b, 20)          # blend ventiles (prediction-value axis - closed)
# closed segmentation from blend_segments: rec_ord bins x ord_days90 bins
rec_bins = np.searchsorted([7, 30, 60, 90], rec_ord, side="right")   # 5 bins
odd_bins = np.searchsorted([0.5, 2.5, 6.5], ord_days90, side="right")  # 4 bins
seg_code = rec_bins * 4 + odd_bins
codes_bs = codes_b * 20 + seg_code  # blend x segment joint  (20*20=400 cells)

e_bcal, _ = cv_margin_correct(e0, codes_b, nb_b)            # after blend-ventile calibration
e_bscal, _ = cv_margin_correct(e0, codes_bs, 400, shrink=50)  # after blend x closed-seg calibration

res = {}
for name, x in AXES.items():
    codes, nc = qbin(x, 10)
    e_new, c = cv_margin_correct(e0, codes, nc)
    d, t01, t1 = contribution_shares(e_new, e0)
    gain = base - np.sqrt((e_new**2).mean())
    # after blend calibration
    e2, _ = cv_margin_correct(e_bcal, codes, nc)
    gain2 = np.sqrt((e_bcal**2).mean()) - np.sqrt((e2**2).mean())
    # after blend + closed segments
    e3, _ = cv_margin_correct(e_bscal, codes, nc)
    gain3 = np.sqrt((e_bscal**2).mean()) - np.sqrt((e3**2).mean())
    res[name] = dict(gain_oof=gain, top01=t01, top1=t1, gain_after_blendcal=gain2, gain_after_blendseg=gain3)
    print(f"{name:>11} | {gain:+.6f} | {t01:7.2f} {t1:6.2f} | {gain2:+.6f}   | {gain3:+.6f}")

# reference: gain of blend calibration itself and of closed segmentation
g_b = base - np.sqrt((e_bcal**2).mean())
g_bs = base - np.sqrt((e_bscal**2).mean())
print(f"\nblend-ventile calibration alone (closed axis): {g_b:+.6f}")
print(f"blend x rec/ord90 segments (closed): {g_bs:+.6f}")

with open(f"{SCR}/margins_oof.json", "w") as f:
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}, f, indent=1)
