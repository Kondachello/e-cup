"""Задача 2: бутстрап сплитов public 50k / private 200k на вал-окне.
Для каждого кандидата: распределение priv−pub; для пар (файл, файл+поправка):
распределение «видимый на public выигрыш − приватный выигрыш» (цена подгонки).
Сегментный разрез — по явке (стена отбора): пул/середина/топ по p_buy.
"""
import numpy as np, polars as pl, json
from scipy.optimize import nnls

d = pl.read_parquet("/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet").sort("user_id")
y = np.log1p(d["target"].to_numpy().astype(np.float64))
N = len(y)
cols = [c for c in d.columns if c not in ("user_id", "target", "blend")]
M = np.stack([d[c].to_numpy().astype(np.float64) for c in cols], axis=1)

def fit_shifts(lp, ly, bins=24):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cs, ss = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500: continue
        cs.append(lp[m].mean()); ss.append(ly[m].mean() - lp[m].mean())
    return np.array(cs), np.array(ss)

def cal_honest(lp, seed=0):
    r = np.random.default_rng(seed); half = r.permutation(N) < N // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], y[m]); out[~m] = np.clip(lp[~m] + np.interp(lp[~m], c, s), 0, None)
    return out

L = lambda f: np.load(f"/root/work/{f}").astype(np.float64)
pz4 = (L("m2_val_pz.npy")*2 + L("m2_val_pz_s3.npy") + L("m2_val_pz_s4.npy"))/4
p4 = (L("m2_val_p.npy")*2 + L("m2_val_p_s3.npy") + L("m2_val_p_s4.npy"))/4
s4 = (L("m2_val_s.npy")*2 + L("m2_val_s_s3.npy") + L("m2_val_s_s4.npy"))/4
tw2 = (L("m2_val_twlog_s1.npy") + L("m2_val_twlog_s2.npy"))/2
m1 = 0.5*L("val_pz.npy") + 0.5*L("val_two.npy")
mine = cal_honest(0.55*(0.4*pz4 + 0.6*p4*s4) + 0.25*tw2 + 0.2*m1)
A = np.column_stack([M, mine]); w, _ = nnls(A, y)
blend_plus = A @ w
blend = d["blend"].to_numpy().astype(np.float64)

# сегменты явки (по p_buy): пул 30 / середина 40 / топ 30; факторы явки из стены отбора
q3, q7 = np.quantile(p4, [0.3, 0.7])
seg = np.where(p4 <= q3, 0, np.where(p4 <= q7, 1, 2)).astype(np.int8)
APP = {0: 0.9466, 1: 0.9900, 2: 0.9995}

CANDS = {
    "blend(пак)": blend,
    "blend+kostya46": blend_plus,
    "kostya46": mine,
}
# поправочные варианты: глобальный сдвиг и сегментный сдвиг по явке (фиксированные, из механизма)
shade = blend_plus + np.log(np.vectorize(APP.get)(seg))  # аппроксимация формы поправки
CANDS["blend+k46+shade_seg"] = shade

e2 = {nm: (v - y) ** 2 for nm, v in CANDS.items()}
rng = np.random.default_rng(2026)
B = 600
res = {nm: [] for nm in CANDS}
pairdiff = []      # (public gain of shade vs base) - (private gain)
probe_overfit = [] # сегментная поправка, ПОДОГНАННАЯ на public, применённая на private
for b in range(B):
    perm = rng.permutation(N)
    pub, prv = perm[:50000], perm[50000:]
    for nm in CANDS:
        res[nm].append(np.sqrt(e2[nm][prv].mean()) - np.sqrt(e2[nm][pub].mean()))
    gp = np.sqrt(e2["blend+kostya46"][pub].mean()) - np.sqrt(e2["blend+k46+shade_seg"][pub].mean())
    gv = np.sqrt(e2["blend+kostya46"][prv].mean()) - np.sqrt(e2["blend+k46+shade_seg"][prv].mean())
    pairdiff.append(gp - gv)
    # probe-fitted: 3 сегментных сдвига подобраны на public-50k, применены на private
    fitted = np.zeros(N)
    for s_ in (0, 1, 2):
        mseg = pub[seg[pub] == s_]
        fitted_shift = (y - blend_plus)[mseg].mean()
        fitted[seg == s_] = fitted_shift
    corr = blend_plus + fitted
    ec = (corr - y) ** 2
    gain_pub = np.sqrt(e2["blend+kostya46"][pub].mean()) - np.sqrt(ec[pub].mean())
    gain_prv = np.sqrt(e2["blend+kostya46"][prv].mean()) - np.sqrt(ec[prv].mean())
    probe_overfit.append((gain_pub, gain_prv))

print(f"{'кандидат':28s}  E[priv-pub]      p5        p95       sd")
for nm in CANDS:
    a = np.array(res[nm])
    print(f"{nm:28s}  {a.mean():+.6f}  {np.percentile(a,5):+.6f}  {np.percentile(a,95):+.6f}  {a.std():.6f}")
pd_ = np.array(pairdiff)
print(f"\nфикс. сегментная поправка: (pub-выигрыш − priv-выигрыш): {pd_.mean():+.6f} ± {pd_.std():.6f}  p95 {np.percentile(pd_,95):+.6f}")
po = np.array(probe_overfit)
print(f"поправка, ПОДОГНАННАЯ на public (3 сегмента): pub-выигрыш {po[:,0].mean():+.6f}, priv-выигрыш {po[:,1].mean():+.6f}")
print(f"  => переносится {po[:,1].mean()/max(po[:,0].mean(),1e-12):.0%}; p5 приватного: {np.percentile(po[:,1],5):+.6f}")
# сегментная декомпозиция дисперсии priv-pub для бленда
a = np.array(res["blend+kostya46"])
print(f"\nsd(priv−pub) бленда: {a.std():.6f}  (шум LB одного замера 0.000022 — сплит-шум х{a.std()/0.000022:.0f})")
json.dump({nm: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                "p5": float(np.percentile(v,5)), "p95": float(np.percentile(v,95))} for nm, v in res.items()},
          open("/root/work/private_sim_results.json", "w"), indent=1)
