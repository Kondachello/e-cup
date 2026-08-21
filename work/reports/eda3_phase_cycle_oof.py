"""eda3 фаза цикла, шаг 3: честный OOF, концентрация, плацебо для сигнала u=rec/P
в срезе P[14,60] cv<=0.6; плюс n_expected = число кратных P в (rec, rec+30] (плановые
заказы в целевом окне) — резкая целочисленная композиция (rec,P), не гладкий признак.
"""
import numpy as np
import polars as pl

SCRATCH = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
rng = np.random.default_rng(0)

st = pl.read_parquet(f"{SCRATCH}/eda3_user_cycle.parquet")
vp = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
                     columns=["user_id", "target", "blend"])
d = vp.join(st, on="user_id", how="left")
d = d.with_columns(e=(pl.col("target").log1p() - pl.col("blend")), P=pl.col("gap_median"))
d = d.with_columns(u=pl.col("rec") / pl.col("P"))
N_ALL = d.height
mse0 = float((d["e"].to_numpy() ** 2).mean())
rmse0 = np.sqrt(mse0)

EDGES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

def oof_gain(sub: pl.DataFrame, col: str, edges, nfold=2, nrep=20, shrink=0.0):
    """Средние по бинам на A, применение на B (и наоборот), вклад в общий MSE."""
    x = sub[col].to_numpy()
    e = sub["e"].to_numpy()
    uid = sub["user_id"].to_numpy()
    idx = np.digitize(x, edges)
    gains, contribs = [], []
    for rep in range(nrep):
        fold = rng.permutation(len(e)) % nfold
        dmse = 0.0
        percontrib = np.zeros(len(e))
        for f in range(nfold):
            tr, te = fold != f, fold == f
            for b in np.unique(idx):
                mtr = tr & (idx == b)
                mte = te & (idx == b)
                if mtr.sum() < 10 or mte.sum() == 0:
                    continue
                c = e[mtr].mean() * (1.0 - shrink)
                delta = e[mte] ** 2 - (e[mte] - c) ** 2
                dmse += delta.sum()
                percontrib[mte] += delta
        gains.append(dmse / N_ALL)
        contribs.append(percontrib)
    g = np.array(gains)
    pc = np.mean(contribs, axis=0)
    return g, pc, uid

def report(sub, col, label, edges=EDGES):
    g, pc, uid = oof_gain(sub, col, edges)
    dmse = g.mean()
    d_rmse = rmse0 - np.sqrt(mse0 - dmse)
    # концентрация: доля суммарного выигрыша от топ-1% и топ-0.1% юзеров по вкладу
    pos_total = pc.sum()
    order = np.argsort(-pc)
    n1, n01 = max(1, len(pc) // 100), max(1, len(pc) // 1000)
    top1 = pc[order[:n1]].sum() / pos_total if pos_total > 0 else np.nan
    top01 = pc[order[:n01]].sum() / pos_total if pos_total > 0 else np.nan
    # плацебо: перестановка col внутри среза
    plac = []
    e = sub["e"].to_numpy(); x = sub[col].to_numpy()
    for _ in range(30):
        xp = rng.permutation(x)
        sp = sub.with_columns(pl.Series(col, xp))
        gp, _, _ = oof_gain(sp, col, edges, nrep=4)
        plac.append(gp.mean())
    plac = np.array(plac)
    print(f"\n== {label}: n={sub.height}")
    print(f"  OOF dMSE={dmse:.6f} (+-{g.std():.6f})  dRMSE={d_rmse:.6f}")
    print(f"  концентрация выигрыша: top1%={top1:.2f} top0.1%={top01:.2f} (сумма вклада {pos_total:.1f})")
    print(f"  плацебо dMSE: mean={plac.mean():.6f} p95={np.quantile(plac,0.95):.6f} max={plac.max():.6f}")
    return dmse, d_rmse

base = d.filter(pl.col("n_ord_days") >= 5, pl.col("P") >= 2)
sub1 = base.filter(pl.col("P") >= 14, pl.col("P") <= 60, pl.col("cv") <= 0.6)
report(sub1, "u", "u-бины, P[14,60] cv<=0.6")

# n_expected: число плановых заказов в (rec, rec+30] при периоде P
d2 = base.with_columns(
    nexp=((pl.col("rec") + 30) / pl.col("P")).floor() - (pl.col("rec") / pl.col("P")).floor()
)
for pmin, pmax, cvmax in [(14, 60, 0.6), (14, 90, 0.8), (20, 120, 1.0)]:
    sub = d2.filter(pl.col("P") >= pmin, pl.col("P") <= pmax, pl.col("cv") <= cvmax)
    x = sub["nexp"].to_numpy(); e = sub["e"].to_numpy()
    print(f"\n== n_expected, P[{pmin},{pmax}] cv<={cvmax}: n={sub.height}")
    for v in sorted(np.unique(x)):
        m = x == v
        n = int(m.sum())
        if n < 20: continue
        me, se = e[m].mean(), e[m].std(ddof=1)/np.sqrt(n)
        print(f"  nexp={v:.0f} n={n:>6} mean_e={me:+.4f} se={se:.4f} t={me/se:+.2f}")
