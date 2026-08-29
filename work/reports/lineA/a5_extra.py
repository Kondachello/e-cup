"""-довесок: одноосевой LOO против группового, калибровка приора, гребень геометрии."""
import json
import numpy as np

L = "/Users/alexanderkondakov/ozon-cup/work/reports/lineA/"
z = np.load(L + "a5_loo_state_a5.npz", allow_pickle=True)
J = json.load(open(L + "a5_loo_groups_a5.json"))

names = list(map(str, z["names"]))
Q, cQ, cP, mu_c = z["Q"], z["cQ"], z["cP"], z["mu_c"]
Sig_e, Sig_p, Lam = z["Sig_e"], z["Sig_p"], z["Lam"]
dF9, dF8, q, tau, kap, sig = z["d_F9"], z["d_F8"], z["q"], z["tau"], z["kap"], z["sig"]
fam = list(map(str, z["fam"]))
dot = z["dot"]
F, NOISE, GAM = float(z["F_SCALE"]), float(z["NOISE"]), 0.1
n = len(names)
pos = {k: i for i, k in enumerate(names)}
GROUPS = J["groups"]


def g_at(d, c, Qm=Q):
    return float((2 * d @ c - d @ Qm @ d) / (2 * F))


def solve_sub(idx, tau_scale=1.0):
    ix = np.array(idx)
    ta = tau[ix] / tau_scale
    qk = q[ix]
    Sp = np.diag((qk * ta) ** 2)
    Se = Sig_e[np.ix_(ix, ix)]
    muc = mu_c[ix] if tau_scale == 1.0 else mu_c[ix]
    K = Sp @ np.linalg.inv(Sp + Se)
    ms = muc + K @ (cP[ix] - muc)
    cq = 1.25 * ms - 0.25 * cP[ix]
    w = ta ** 2 / (ta ** 2 + sig[ix] ** 2)
    Lm = np.diag(qk * (1 - w) * ta ** 2)
    Qs = Q[np.ix_(ix, ix)]
    rg = 1e-9 * np.trace(Qs) / len(Qs) * np.eye(len(Qs))
    d = np.linalg.solve(Qs + Lm + rg + GAM * np.diag(np.diag(Qs)), cq)
    return d, cq, Qs, float((2 * d @ cq - d @ Qs @ d) / (2 * F))


d0, cq0, Q0, g0 = solve_sub(range(n))
print(f"контроль: g_full={g0:+.8f}  max|d-dF9|={np.abs(d0-dF9).max():.2e}")

# ---------------- 1) ОДНООСЕВОЙ LOO (приор семейства заморожен) ----------------
print("\n=== одноосевой LOO против группового ===")
rows_ax = []
for i, k in enumerate(names):
    idx = [j for j in range(n) if j != i]
    _, _, _, gi = solve_sub(idx)
    opt_i = float(dF9[i] * (cQ[i] - mu_c[i]) / F)
    rows_ax.append(dict(ax=k, fam=fam[i], d_in=g0 - gi, opt=opt_i, d_oof=g0 - gi - opt_i))
s_in_ax = sum(r["d_in"] for r in rows_ax)
s_oof_ax = sum(r["d_oof"] for r in rows_ax)
print(f"  Σ Δ_in (по одной оси)  = {s_in_ax:+.6f}  ({s_in_ax/NOISE:+.1f} шума)")
print(f"  Σ Δ_oof (по одной оси) = {s_oof_ax:+.6f}")
print(f"  для сравнения: групповой Σ Δ_in = {J['sum_delta_in']:+.6f}, "
      f"Σ Δ_oof = {J['sum_delta_oof']:+.6f}")
top = sorted(rows_ax, key=lambda r: -r["d_in"])[:10]
for r in top:
    print(f"    {r['ax']:6s} [{r['fam']:7s}] Δ_in {r['d_in']:+.6f} "
          f"({r['d_in']/NOISE:+.2f}ш)  Δ_oof {r['d_oof']:+.6f}")

# групповой Δ_in по группам против суммы одноосевых внутри группы
print("\n  группа: Δ_in(группа) против Σ Δ_in(оси внутри) — мера коллинеарности")
coll = {}
for gname, axes in GROUPS.items():
    s1 = sum(r["d_in"] for r in rows_ax if r["ax"] in axes)
    gi = [r for r in J["rows"] if r["group"] == gname][0]["delta_in"]
    coll[gname] = dict(group_delta_in=gi, sum_axis_delta_in=s1,
                       ratio=(gi / s1 if s1 else None))
    print(f"    {gname:11s} групповой {gi:+.6f}  сумма одноосевых {s1:+.6f}  "
          f"отношение x{gi/s1 if s1 else float('nan'):.2f}")

# ---------------- 2) калибровка: pooled <z^2> ----------------
print("\n=== калибровка held-out ===")
tot_z2, tot_n = 0.0, 0
for r in J["rows"]:
    tot_z2 += r["mean_z2"] * r["n_axes"]
    tot_n += r["n_axes"]
    print(f"  {r['group']:11s} n={r['n_axes']:2d} <z²>={r['mean_z2']:.3f}  "
          f"tau_ML(held-out)={r['tau_heldout_ml']:.4f} против приорного "
          f"{r['tau_prior_used']:.4f}  (mu_приор {r['mu_prior_used']:+.4f})")
pool = tot_z2 / tot_n
print(f"  ПУЛ по 46 осям: <z²> = {pool:.3f}  -> приоры {'ТЕСНЫ' if pool>1 else 'ШИРОКИ'}; "
      f"калиброванный масштаб tau: x{np.sqrt(pool):.2f}")

# ---------------- 3) кривая по lam (включая lam<1) ----------------
print("\n=== E[priv] действующих векторов под приором tau/lam ===")
lam_rows = {}
for lam in (0.5, 0.6, 0.74, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
    ta = tau / lam
    Sp = np.diag((q * ta) ** 2)
    K = Sp @ np.linalg.inv(Sp + Sig_e)
    ms = mu_c + K @ (cP - mu_c)
    cqt = 1.25 * ms - 0.25 * cP
    a, b = g_at(dF9, cqt), g_at(dF8, cqt)
    lam_rows[lam] = dict(F9=a, F8=b, diff=a - b, diff_noise=(a - b) / NOISE)
    print(f"  lam={lam:4.2f} (tau x{1/lam:.2f}): F9 {a:+.6f}  F8 {b:+.6f}  "
          f"F9−F8 {a-b:+.7f} = {(a-b)/NOISE:+.2f} шума")

# ---------------- 4) геометрия: бутстрап грама ----------------
print("\n=== геометрия: бутстрап грама (gram_boot_a2.npz) ===")
gb = np.load(L + "gram_boot_a2.npz", allow_pickle=True)
nb = list(map(str, gb["names"]))
ix = np.array([nb.index(k) for k in names])
QB = gb["QB"][:, ix][:, :, ix]
DOTB = gb["DOTB"][:, ix]
gs9, gs8 = [], []
for b in range(QB.shape[0]):
    Qb = QB[b]
    cPb = q * kap - DOTB[b]          # dot меняется вместе с грамом
    # c_priv остаётся оценкой; двигаем только геометрию квадратичной части
    gs9.append(float((2 * dF9 @ cQ - dF9 @ Qb @ dF9) / (2 * F)))
    gs8.append(float((2 * dF8 @ cQ - dF8 @ Qb @ dF8) / (2 * F)))
gs9, gs8 = np.array(gs9), np.array(gs8)
print(f"  F9: среднее {gs9.mean():+.6f}  sd {gs9.std():.6f} "
      f"({gs9.std()/NOISE:.2f} шума), полный грам {g_at(dF9, cQ):+.6f}")
print(f"  F8: среднее {gs8.mean():+.6f}  sd {gs8.std():.6f}")
d98 = gs9 - gs8
print(f"  F9−F8 по бутстрапу грама: среднее {d98.mean():+.7f} sd {d98.std():.7f} "
      f"({d98.std()/NOISE:.2f} шума); доля бутстрапов с F9>F8: "
      f"{float((d98>0).mean()):.2f}")

# ---------------- 5) паблик-репрей F8 ----------------
print("\n=== замер F8: алгебра против факта ===")
PUB_ALG, PUB_MEAS = 1.645765225926265, 1.6458057389
d2 = PUB_ALG ** 2 - PUB_MEAS ** 2
print(f"  алгебра {PUB_ALG:.7f}, факт {PUB_MEAS:.7f}, промах {PUB_MEAS-PUB_ALG:+.3e} "
      f"({(PUB_MEAS-PUB_ALG)/NOISE:+.2f} шума)")
print(f"  d·(cP − c_true^pub) = {d2/2:+.3e}; в единицах скора "
      f"{d2/2/F:+.3e}")
print(f"  приватный аналог (Q_priv−Q_all = −0.25·(Q_pub−Q_all)): "
      f"{-0.25*(PUB_MEAS-PUB_ALG):+.3e} = {-0.25*(PUB_MEAS-PUB_ALG)/NOISE:+.2f} шума")

# ---------------- 6) итоговые числа для вердикта ----------------
rows = J["rows"]
pos_oof = sum(max(r["delta_oof"], 0.0) for r in rows)
print("\n=== ИТОГ ===")
print(f"  in-sample алгебра F9−F7        = {J['insample_gain']:+.6f} "
      f"({J['insample_gain']/NOISE:+.1f} шума)")
print(f"  Σ Δ_oof по группам             = {J['sum_delta_oof']:+.6f} "
      f"({J['sum_delta_oof']/NOISE:+.2f} шума)")
print(f"  полный OOF (все оси held-out)  = {J['oof_full']:+.6f} "
      f"({J['oof_full']/NOISE:+.2f} шума)")
print(f"  Σ положительных Δ_oof          = {pos_oof:+.6f}  "
      f"-> мягкий дисконт x{J['insample_gain']/pos_oof:.2f}")
f98 = J["f9_vs_f8"]
print(f"  F9−F8 in-sample {f98['in_sample']:+.7f} ({f98['in_sample_noises']:+.2f} шума); "
      f"после мягкого дисконта {f98['in_sample']/(J['insample_gain']/pos_oof):+.7f} "
      f"({f98['in_sample_noises']/(J['insample_gain']/pos_oof):+.2f} шума)")

json.dump(dict(axis_loo=rows_ax, sum_axis_delta_in=s_in_ax, sum_axis_delta_oof=s_oof_ax,
               collinearity=coll, pooled_z2=pool, lam_rows={str(k): v for k, v in lam_rows.items()},
               gram_boot=dict(sd_F9=float(gs9.std()), sd_F8=float(gs8.std()),
                              sd_diff=float(d98.std()), frac_F9_better=float((d98 > 0).mean())),
               pub_replay=dict(alg=PUB_ALG, meas=PUB_MEAS,
                               priv_implication=-0.25 * (PUB_MEAS - PUB_ALG)),
               sum_positive_oof=pos_oof,
               soft_discount=J["insample_gain"] / pos_oof),
          open(L + "a5_extra.json", "w"), ensure_ascii=False, indent=1)
print("\njson:", L + "a5_extra.json")
