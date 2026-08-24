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

gs = df["gmv_search365"].to_numpy().astype(float)
gc = df["gmv_cat365"].to_numpy().astype(float)
gmv365 = df["gmv365"].to_numpy().astype(float)
lvl = np.log1p(gmv365)
with np.errstate(invalid="ignore"):
    chan = np.where(gs + gc > 0, gs / (gs + gc), np.nan)

def qbin(x, K=8):
    x = np.asarray(x, dtype=np.float64)
    nanm = np.isnan(x)
    xs = x[~nanm]
    qs = np.unique(np.quantile(xs, np.linspace(0, 1, K + 1)))
    codes = np.full(len(x), -1, dtype=np.int32)
    codes[~nanm] = np.searchsorted(qs[1:-1], xs, side="right")
    nb = len(qs) - 1
    codes[nanm] = nb
    return codes, nb + (1 if nanm.any() else 0), qs

cx, nx, qx = qbin(chan, 8)
cy, ny, qy = qbin(lvl, 8)
print("chan edges:", np.round(qx, 3), " nx =", nx, "(последний бин = NaN: нет ни gmv_search, ни gmv_cat за 365д)")
print("lvl edges:", np.round(qy, 2), " ny =", ny)
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
for _ in range(120):
    alpha = np.where(C.sum(1) > 0, ((M - mu - beta[None, :]) * C).sum(1) / np.maximum(C.sum(1), 1), 0)
    beta = np.where(C.sum(0) > 0, ((M - mu - alpha[:, None]) * C).sum(0) / np.maximum(C.sum(0), 1), 0)
inter = (M - (mu + alpha[:, None] + beta[None, :])).ravel()
z_int = np.where(cnt > 30, inter / np.maximum(se, 1e-12), 0)
idx = np.argsort(-np.abs(z_int))[:5]
for i in idx:
    r, c = divmod(i, ny)
    print(f"cell ({r},{c}): n={int(cnt[i])}, mean_e={m[i]:+.4f}, inter={inter[i]:+.4f}, z_int={z_int[i]:+.2f}")

# split-half stability of the two z_int>3 cells: recompute z_int on random halves
top2 = idx[:2]
rng_res = []
for s in range(20):
    half = np.random.default_rng(300 + s).integers(0, 2, N)
    signs = []
    for h in (0, 1):
        mask = half == h
        cntH = np.bincount(cell[mask], minlength=ncell).astype(float)
        mH = np.bincount(cell[mask], weights=e[mask], minlength=ncell) / np.maximum(cntH, 1)
        varH = np.bincount(cell[mask], weights=e[mask]**2, minlength=ncell) / np.maximum(cntH, 1) - mH**2
        seH = np.sqrt(varH / np.maximum(cntH, 1))
        MH = mH.reshape(nx, ny); CH = cntH.reshape(nx, ny)
        aH = np.zeros(nx); bH = np.zeros(ny); muH = e[mask].mean()
        for _ in range(120):
            aH = np.where(CH.sum(1) > 0, ((MH - muH - bH[None, :]) * CH).sum(1) / np.maximum(CH.sum(1), 1), 0)
            bH = np.where(CH.sum(0) > 0, ((MH - muH - aH[:, None]) * CH).sum(0) / np.maximum(CH.sum(0), 1), 0)
        interH = (MH - (muH + aH[:, None] + bH[None, :])).ravel()
        zH = interH / np.maximum(seH, 1e-12)
        signs.append([zH[i] for i in top2])
    rng_res.append(signs)
rng_res = np.array(rng_res)  # (20, 2 halves, 2 cells)
for j, i in enumerate(top2):
    r, c = divmod(i, ny)
    zz = rng_res[:, :, j]
    same_sign = np.mean(np.sign(zz[:, 0]) == np.sign(zz[:, 1]))
    print(f"cell ({r},{c}) split-half: z_half mean {zz.mean():+.2f}, доля совпадения знака половин {same_sign:.2f}, mean |z_half| {np.abs(zz).mean():.2f}")

# honest cross-fitted gain from correcting ONLY the interaction of this map (8x8, additive removed)
fold = np.random.default_rng(0).integers(0, 5, N)
c_corr = np.zeros(N)
for f in range(5):
    tr = fold != f; te = fold == f
    cntf = np.bincount(cell[tr], minlength=ncell).astype(float)
    mf = np.bincount(cell[tr], weights=e[tr], minlength=ncell) / np.maximum(cntf, 1)
    Mf = mf.reshape(nx, ny); Cf = cntf.reshape(nx, ny)
    af = np.zeros(nx); bf = np.zeros(ny); muf = e[tr].mean()
    for _ in range(120):
        af = np.where(Cf.sum(1) > 0, ((Mf - muf - bf[None, :]) * Cf).sum(1) / np.maximum(Cf.sum(1), 1), 0)
        bf = np.where(Cf.sum(0) > 0, ((Mf - muf - af[:, None]) * Cf).sum(0) / np.maximum(Cf.sum(0), 1), 0)
    interf = (Mf - (muf + af[:, None] + bf[None, :])).ravel()
    interf = interf * cntf / (cntf + 200)
    c_corr[te] = interf[cell[te]]
g = base - np.sqrt(((e - c_corr) ** 2).mean())
print(f"\nчестный OOF gain interaction-only корректора chan x lvl (shrink 200): {g:+.6f}")
