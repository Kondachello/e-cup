# eda3 activity ladder: single-parameter monotone test of momentum trend (low-variance) + whale concentration of the oracle signal
import numpy as np
import polars as pl

uni = pl.read_parquet("work/reports/eda3_ladder_uni.parquet")
y = np.log1p(uni["target"].to_numpy())
e = y - uni["blend"].to_numpy().astype(np.float64)
N = len(e); mse0 = np.mean(e ** 2); rmse0 = np.sqrt(mse0)
rng = np.random.default_rng(3)

Ls = np.load("work/reports/eda3_ladder_val_Ls.npy").astype(np.int16)
Zs = np.load("work/reports/eda3_ladder_val_zsrch.npy")
Zg = np.load("work/reports/eda3_ladder_val_zgmv.npy")

def lin_oof(name, x, reps=40):
    x = (x - x.mean()) / (x.std() + 1e-12)
    gains, conc1, conc01 = [], [], []
    for _ in range(reps):
        f = rng.random(N) < 0.5
        corr = np.zeros(N)
        for fa, fb in [(f, ~f), (~f, f)]:
            b = np.polyfit(x[fa], e[fa], 1)
            corr[fb] = np.polyval(b, x[fb])
        g = e ** 2 - (e - corr) ** 2
        gains.append(g.mean())
        o = np.argsort(-np.abs(g)); tot = g.sum()
        if abs(tot) > 1e-12:
            conc1.append(g[o[: N // 100]].sum() / tot); conc01.append(g[o[: N // 1000]].sum() / tot)
    gains = np.array(gains); dm = gains.mean()
    print(f"[{name}] 1-param OOF dMSE={dm:+.6f} +-{gains.std():.6f} -> dRMSLE={rmse0-np.sqrt(mse0-dm):+.6f}; "
          f"corr(x,e)={np.corrcoef(x, e)[0,1]:+.4f}; conc1%={np.mean(conc1):.2f} conc0.1%={np.mean(conc01):.2f}")

acc = (Ls[:, 0].astype(float) - 2 * Ls[:, 1] + Ls[:, 2])
lin_oof("acc_Ls", acc)
lin_oof("acc_zsrch", Zs[:, 0] - 2 * Zs[:, 1] + Zs[:, 2])
lin_oof("acc_zgmv", Zg[:, 0] - 2 * Zg[:, 1] + Zg[:, 2])
lin_oof("d1_Ls", Ls[:, 0].astype(float) - Ls[:, 1])
lin_oof("placebo_shuffle_acc", acc[rng.permutation(N)])

# concentration of the ORACLE acc-bin signal: which users hold the in-sample dMSE
bins = np.clip((acc.astype(int) + 12) // 3, 0, 8)
mu = np.array([e[bins == b].mean() for b in range(9)])
g = e ** 2 - (e - mu[bins]) ** 2
o = np.argsort(-np.abs(g))
print(f"oracle acc-bin: dMSE={g.mean():.6f}; top1% users hold {g[o[:N//100]].sum()/g.sum():.2f} of net gain, "
      f"top0.1% {g[o[:N//1000]].sum()/g.sum():.2f}")
# where the extreme-acc bins sit in target space
for b in [0, 8]:
    s = bins == b
    print(f"bin{b}: n={s.sum()} zero-rate={np.mean(y[s]==0):.3f} mean_y={y[s].mean():.3f} mean_blend={uni['blend'].to_numpy()[s].mean():.3f}")
print(f"overall: zero-rate={np.mean(y==0):.3f} mean_y={y.mean():.3f}")
