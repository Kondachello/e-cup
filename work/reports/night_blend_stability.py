"""Ночная проверка устойчивости нового бленда (1.665647) против старого (1.666302).

Вопросы: (1) бутстрап юзеров 50k/200k — вероятность потерять выигрыш на привате;
(2) концентрация выигрыша в топ-юзерах по |вкладу|, сравнение с прежними приростами;
(3) терцили ранга прогноза — где живёт выигрыш; (4) вклад новых членов
(kostya46/gseq_small/gseq_big/lagd28) в дельту по сегментам.

Только чтение готовых данных + numpy. Ничего не обучает, ничего не пишет вне
work/reports/night_*. Старые паки берутся из git-извлечений в скретчпаде.

Запуск: POLARS_MAX_THREADS=3 .venv/bin/python work/reports/night_blend_stability.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
SCRATCH = Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/"
               "b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad")
OUT_JSON = ROOT / "work/reports/night_blend_stability.json"
NOISE = 0.000022          # шум одного замера лидерборда

rng = np.random.default_rng(20260821)


def load_pack(path: Path, columns=None) -> pl.DataFrame:
    return pl.read_parquet(path, columns=columns).sort("user_id")


def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a)))


# ---------------------------------------------------------------- загрузка
new = load_pack(ROOT / "work/preds_pack/val_preds.parquet")
old = load_pack(SCRATCH / "old_val_preds.parquet")
uid = new["user_id"].to_numpy()
assert np.array_equal(uid, old["user_id"].to_numpy())
y = np.log1p(np.clip(new["target"].to_numpy().astype(np.float64), 0, None))
assert np.allclose(y, np.log1p(np.clip(old["target"].to_numpy().astype(np.float64), 0, None)))
n = len(y)

b_new = new["blend"].to_numpy().astype(np.float64)
b_old = old["blend"].to_numpy().astype(np.float64)
e_new, e_old = b_new - y, b_old - y
se_new, se_old = e_new ** 2, e_old ** 2
s_new, s_old = rmse(se_new), rmse(se_old)
delta = s_new - s_old                      # < 0 — новый бленд лучше
d_user = (se_new - se_old) / (n * (s_new + s_old))   # точный аддитивный вклад юзера
assert abs(d_user.sum() - delta) < 1e-12

res: dict = {"s_new": s_new, "s_old": s_old, "delta": delta, "n": n}
print(f"новый {s_new:.6f}  старый {s_old:.6f}  дельта {delta:+.6f}")

# ------------------------------------------------- (1) сплит-симуляция 50k/200k
# Приват = 200k из тех же 250k, публика = остальные 50k. Симулируем сам сплит:
# случайная перестановка, первые 50k — «публика», остальные 200k — «приват».
B_SPLIT = 4000
S_new_full, S_old_full = se_new.sum(), se_old.sum()
pub_d = np.empty(B_SPLIT)
priv_d = np.empty(B_SPLIT)
for i in range(B_SPLIT):
    perm = rng.permutation(n)
    p = perm[:50_000]
    sn_p, so_p = se_new[p].sum(), se_old[p].sum()
    pub_d[i] = np.sqrt(sn_p / 50_000) - np.sqrt(so_p / 50_000)
    priv_d[i] = (np.sqrt((S_new_full - sn_p) / 200_000)
                 - np.sqrt((S_old_full - so_p) / 200_000))

# Бутстрап С возвращением — свежая выборка юзеров того же размера (обобщение,
# консервативнее сплита: без поправки конечной популяции).
def boot(size: int, B: int, chunk: int) -> np.ndarray:
    out = np.empty(B)
    done = 0
    while done < B:
        k = min(chunk, B - done)
        idx = rng.integers(0, n, size=(k, size))
        out[done:done + k] = (np.sqrt(se_new[idx].mean(axis=1))
                              - np.sqrt(se_old[idx].mean(axis=1)))
        done += k
    return out

boot50 = boot(50_000, 3000, 200)
boot200 = boot(200_000, 3000, 50)

# аналитика (дельта-метод): SE дельты на свежей выборке размера m
g = se_new / (2 * s_new) - se_old / (2 * s_old)
sd_g = float(g.std(ddof=1))
se_an = {m: sd_g / np.sqrt(m) for m in (50_000, 200_000, n)}
fpc = np.sqrt(1 - 200_000 / n)             # поправка конечной популяции для сплита

def stats(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "q05": float(np.quantile(a, .05)), "q95": float(np.quantile(a, .95)),
            "p_delta_pos": float((a > 0).mean()),   # новый бленд ПРОИГРЫВАЕТ
            "p_delta_neg": float((a < 0).mean())}   # выигрыш сохраняется

res["split_sim"] = {"B": B_SPLIT, "pub50k": stats(pub_d), "priv200k": stats(priv_d),
                    "corr_pub_priv": float(np.corrcoef(pub_d, priv_d)[0, 1]),
                    "p_priv_neg_given_pub_pos":
                        float((priv_d[pub_d > 0] < 0).mean()) if (pub_d > 0).any() else None}
res["bootstrap_fresh"] = {"B": 3000, "n50k": stats(boot50), "n200k": stats(boot200)}
res["analytic"] = {"se_fresh": {str(k): float(v) for k, v in se_an.items()},
                   "se_split_priv200k": float(se_an[200_000] * fpc),
                   "z_full": float(delta / se_an[n])}
for tag, a in (("сплит: публика 50k", pub_d), ("сплит: приват 200k", priv_d),
               ("свежие 50k", boot50), ("свежие 200k", boot200)):
    print(f"{tag:<22} mean {a.mean():+.6f} std {a.std():.6f} "
          f"P(дельта>0, проигрыш)={np.mean(a > 0):.4f}")

# ------------------------------------------------- (2) концентрация выигрыша
def concentration(dd: np.ndarray) -> dict:
    tot = dd.sum()
    order = np.argsort(-np.abs(dd))
    out = {}
    for frac, label in ((0.001, "top0.1%"), (0.01, "top1%"), (0.1, "top10%")):
        k = max(1, int(round(frac * len(dd))))
        top = dd[order[:k]]
        out[label] = {"k": k, "net_share": float(top.sum() / tot),
                      "gross_share": float(np.abs(top).sum() / np.abs(dd).sum())}
    out["max_single_user_share"] = float(dd[order[0]] / tot)
    out["frac_users_worse"] = float((dd > 0).mean())   # у кого новый бленд хуже
    out["frac_users_better"] = float((dd < 0).mean())
    return out

res["concentration_new"] = concentration(d_user)

# прежние приросты: переходы между историческими паками
# пак pack-early колонки blend не имел — сравнение только с переходом pack-prev→pack-old
d = load_pack(SCRATCH / "pack_prev.parquet", columns=["user_id", "target", "blend"])
assert np.array_equal(d["user_id"].to_numpy(), uid)
b_ea = d["blend"].to_numpy().astype(np.float64)

transitions = [("pack-prev→pack-old(старый)", b_ea, b_old),
               ("pack-old→текущий", b_old, b_new)]
res["prior_transitions"] = {}
for label, ba, bb in transitions:
    ea, eb = ba - y, bb - y
    sa, sb2 = rmse(ea ** 2), rmse(eb ** 2)
    dt = sb2 - sa
    if abs(dt) < 1e-9:
        res["prior_transitions"][label] = {"delta": dt, "note": "бленд не менялся"}
        print(f"{label}: дельта {dt:+.7f} (не менялся)")
        continue
    dd = (eb ** 2 - ea ** 2) / (n * (sa + sb2))
    cc = concentration(dd)
    res["prior_transitions"][label] = {"delta": dt, "s_from": sa, "s_to": sb2, **cc}
    print(f"{label}: дельта {dt:+.6f} top0.1% net {cc['top0.1%']['net_share']:+.3f} "
          f"top1% net {cc['top1%']['net_share']:+.3f}")

# ------------------------------------------------- (3) терцили ранга прогноза
def tercile_masks(pred: np.ndarray):
    r = np.argsort(np.argsort(pred, kind="stable"))
    edges = [0, n // 3, 2 * n // 3, n]
    return [(r >= edges[i]) & (r < edges[i + 1]) for i in range(3)]

for base, tag in ((b_new, "by_new_rank"), (b_old, "by_old_rank")):
    segs = []
    for i, m in enumerate(tercile_masks(base)):
        k = int(m.sum())
        ssn, sso = rmse(se_new[m]), rmse(se_old[m])
        segs.append({"seg": f"T{i+1}", "n": k,
                     "s_new": ssn, "s_old": sso, "delta_inseg": ssn - sso,
                     "contrib_to_total": float(d_user[m].sum()),
                     "share_of_delta": float(d_user[m].sum() / delta)})
    res.setdefault("terciles", {})[tag] = segs
segs = res["terciles"]["by_new_rank"]
for s in segs:
    print(f"{s['seg']} n={s['n']} skor {s['s_new']:.6f}/{s['s_old']:.6f} "
          f"дельта в сегменте {s['delta_inseg']:+.6f} вклад {s['contrib_to_total']:+.6f} "
          f"({s['share_of_delta']:+.1%})")

# --------------------------------- (4) поколоночная декомпозиция дельты по членам
W_NEW = json.loads((ROOT / "work/reports/blend_reopt.json").read_text())["winner"]["weights"]
old_reopt = subprocess.run(["git", "show", "pack-old:work/reports/blend_reopt.json"],
                           capture_output=True, text=True, cwd=ROOT).stdout
W_OLD = json.loads(old_reopt)["winner"]["weights"]
W_NEW = {k: v for k, v in W_NEW.items() if abs(v) > 1e-4}
W_OLD = {k: v for k, v in W_OLD.items() if abs(v) > 1e-4}

def col(name: str) -> np.ndarray:
    if name in new.columns:
        return new[name].to_numpy().astype(np.float64)
    return old[name].to_numpy().astype(np.float64)

# сверка: колонка blend = линейная комбинация членов?
rec_new = sum(w * col(k) for k, w in W_NEW.items())
rec_old = sum(w * col(k) for k, w in W_OLD.items())
res["reconstruction"] = {
    "max_abs_new": float(np.max(np.abs(rec_new - b_new))),
    "max_abs_old": float(np.max(np.abs(rec_old - b_old))),
    "clip_negative_new": float((rec_new < 0).mean())}
print("реконструкция бленда: max|err| new "
      f"{res['reconstruction']['max_abs_new']:.2e} old {res['reconstruction']['max_abs_old']:.2e}")

# общие колонки должны совпадать между паками
shared = [c for c in set(W_NEW) & set(W_OLD)]
mism = {c: float(np.max(np.abs(new[c].to_numpy() - old[c].to_numpy())))
        for c in shared if c in new.columns and c in old.columns}
res["shared_col_max_diff"] = mism

# точная аддитивная декомпозиция: ΔMSE = mean(Δpred·(e_new+e_old)); Δpred = Σ δw_c·col_c
# (правило трапеций точно для квадратичной MSE; сумма по колонкам = дельта без остатка)
allc = sorted(set(W_NEW) | set(W_OLD))
dw = {c: W_NEW.get(c, 0.0) - W_OLD.get(c, 0.0) for c in allc}
esum = e_new + e_old
denom = s_new + s_old
masks = tercile_masks(b_new)
NEW_MEMBERS = ["kostya46_cal", "gseq_small_s42_cal", "gseq_big_s42_cal", "lagd28"]

decomp = []
tot_check = 0.0
for c in allc:
    v = col(c)
    contrib = dw[c] * float(np.mean(v * esum)) / denom
    per_seg = [dw[c] * float((v[m] * esum[m]).sum()) / (n * denom) for m in masks]
    tot_check += contrib
    decomp.append({"col": c, "dw": dw[c], "contrib": contrib,
                   "per_seg": per_seg,
                   "group": ("новый член" if c in NEW_MEMBERS else
                             "добавлен" if c not in W_OLD else
                             "исключён" if c not in W_NEW else "перевесовка")})
resid = (b_new - b_old) - sum(dw[c] * col(c) for c in allc)
resid_contrib = float(np.mean(resid * esum)) / denom
resid_seg = [float((resid[m] * esum[m]).sum()) / (n * denom) for m in masks]
decomp.append({"col": "_residual_", "dw": 0.0, "contrib": resid_contrib,
               "per_seg": resid_seg, "group": "остаток"})
tot_check += resid_contrib
assert abs(tot_check - delta) < 1e-10, (tot_check, delta)
decomp.sort(key=lambda r: r["contrib"])
res["member_decomp"] = decomp
res["member_decomp_note"] = ("contrib<0 — колонка тянет дельту вниз (улучшает); "
                             "per_seg — терцили T1/T2/T3 по рангу нового прогноза; "
                             "сумма всех contrib = полная дельта точно")
print(f"\n{'колонка':<26}{'δw':>9}{'вклад':>11}   T1/T2/T3")
for r in decomp:
    if abs(r["contrib"]) < 2e-6 and r["col"] not in NEW_MEMBERS:
        continue
    ps = " ".join(f"{v:+.6f}" for v in r["per_seg"])
    print(f"{r['col']:<26}{r['dw']:+9.4f}{r['contrib']:+11.6f}   {ps}  [{r['group']}]")

# --------- (4б) LOO-перефит: сколько теряет НОВЫЙ бленд без каждого нового члена
# Трапецеидальные per-seg числа выше страдают от коллинеарности (уровень, снятый с
# исключённых членов, приписан новым). LOO-перефит чист: убираем колонку, перефитим
# nnls_free (метод победителя, in-sample как и веса пака) на остальных 13.
from scipy.optimize import nnls as _nnls

cols_new = list(W_NEW)
A = np.column_stack([col(c) for c in cols_new])

def nnls_fit(Asub: np.ndarray) -> np.ndarray:
    G = Asub.T @ Asub
    bb = Asub.T @ y
    L = np.linalg.cholesky(G + 1e-10 * np.eye(len(G)))
    return _nnls(L.T, np.linalg.solve(L, bb))[0]

w_full = nnls_fit(A)
p_full = A @ w_full
s_full = rmse((p_full - y) ** 2)
seg_full = [rmse(((p_full - y) ** 2)[m]) for m in masks]
res["loo"] = {"s_full_refit": s_full,
              "note": "потеря = скор без члена − скор полного перефита; per_seg — то же в терцилях T1/T2/T3 по рангу нового прогноза; drop_all4 — без всех четырёх новых"}
print(f"\nLOO-перефит (полный перефит {s_full:.6f}, пак {s_new:.6f})")
print(f"{'вариант':<26}{'скор':>10}{'потеря':>10}   T1/T2/T3")
for drop in [[m] for m in NEW_MEMBERS] + [NEW_MEMBERS]:
    keep = [i for i, c in enumerate(cols_new) if c not in drop]
    w = nnls_fit(A[:, keep])
    p = A[:, keep] @ w
    s = rmse((p - y) ** 2)
    per_seg = [float(rmse(((p - y) ** 2)[m]) - seg_full[i]) for i, m in enumerate(masks)]
    lab = drop[0] if len(drop) == 1 else "drop_all4"
    res["loo"][lab] = {"score": s, "loss": s - s_full, "per_seg": per_seg,
                      "delta_vs_old": s - s_old}
    ps = " ".join(f"{v:+.6f}" for v in per_seg)
    print(f"{lab:<26}{s:10.6f}{s - s_full:+10.6f}   {ps}")

OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(f"\nсохранено: {OUT_JSON}")
