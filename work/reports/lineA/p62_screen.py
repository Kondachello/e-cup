"""P62 «крупная масса»: построение направления, q, ортогонализация против всех
существующих осей + константы, q_ост, novelty, гейт слота.

над F5 + 7 над F6 + mdl_wulfen = 47 осей.
"""
import json, os, sys
import numpy as np, polars as pl
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path("/Users/alexanderkondakov/ozon-cup")
SUB = ROOT / "submissions"; CANON = SUB / "canonical"
SP = Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
sys.path.insert(0, str(ROOT / "work" / "scripts"))
import predict_lb as plb
MEAS = {n: s for n, _, s in plb.MEASURED}

STEP = 0.30
N_PUB = 50_000
FPC2 = 0.8
NOISE = 0.000022
F8 = MEAS["F8_priv"]

def lp(fn):
    for p in (SUB / fn, CANON / fn):
        if p.exists():
            d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
            return np.log1p(np.clip(d["predict"].to_numpy().astype(np.float64), 0, None))
    raise FileNotFoundError(fn)

# ------------------------------------------------------------------ универсум
base = pl.read_csv(SUB / "F8_priv.csv", schema_overrides={"user_id": pl.Int64}).sort("user_id")
uid_ref = base["user_id"].to_numpy()
lp_F8 = np.log1p(np.clip(base["predict"].to_numpy().astype(np.float64), 0, None))
n = len(uid_ref); assert n == 250_000

# ------------------------------------------------------------------ 47 осей
OLD32 = ["mdl_amber","mdl_gabbro","mdl_halite","mdl_marble","mdl_realgr","mdl_tektit","mdl_olivin","mdl_flint","mdl_gypsum","mdl_gneis2","mdl_malach","","mdl_vivian","mdl_corund","mdl_larvik","mdl_talc",
         "","","seg_realgr",""]
L5, L6 = lp("F5_priv.csv"), lp("F6_priv.csv")
names, cols = [], []
for k in OLD32:
    t = pl.read_parquet(DIRS / f"{k}.parquet").sort("user_id")
    assert np.array_equal(t["user_id"].to_numpy(), uid_ref), k
    names.append(k); cols.append(t["d"].to_numpy().astype(np.float64))
for fn in NEW_F5:
    names.append(fn.split("_")[0]); cols.append(lp(fn + ".csv") - L5)
for fn in NEW_F6:
    names.append(fn.split("_")[0]); cols.append(lp(fn + ".csv") - L6)
names.append("mdl_wulfen"); cols.append(lp("N1_ktpp.csv") - L6)
A = np.stack(cols, 1)                      # (n, 47)
A = A - A.mean(0, keepdims=True)            # + константа: центрируем всё
print(f"осей в базисе ортогонализации: {A.shape[1]} (+ константа)")

# SVD-проектор на span(A) с отбраковкой численно вырожденных мод
U, sv, _ = np.linalg.svd(A, full_matrices=False)
rank = int((sv > sv[0] * 1e-10).sum())
U = U[:, :rank]
print(f"ранг базиса: {rank} (сингулярные {sv[0]:.3e} .. {sv[rank-1]:.3e})")

# ------------------------------------------------------------------ агрегаты
t = pl.read_parquet(SP / "p62_agg.parquet").sort("user_id")
assert np.array_equal(t["user_id"].to_numpy(), uid_ref)
rec = t["last_di"].to_numpy().astype(np.float64); rec = np.where(np.isnan(rec), 1e9, rec)
never = t["nbuyd"].to_numpy() == 0
V = {c: t[c].to_numpy().astype(np.float64) for c in
     ("act30","act30s","act90","browse30","browse90","searches30","cart30","browse7")}

def med_split(mask, x):
    """Медианный сплит ВНУТРИ mask: порог = медиана x на mask; H = x >= thr,
    если это ближе к 50/50, иначе H = x > thr. Возвращает (H, L, thr, доля H)."""
    xs = x[mask]
    thr = float(np.median(xs))
    hi_ge = xs >= thr; hi_gt = xs > thr
    use_ge = abs(hi_ge.mean() - 0.5) <= abs(hi_gt.mean() - 0.5)
    hi = (x >= thr) if use_ge else (x > thr)
    H = mask & hi; L = mask & ~hi
    return H, L, thr, float(H.sum() / max(mask.sum(), 1)), (">=" if use_ge else ">")

def screen(d_raw, label, note):
    d = STEP * (d_raw - d_raw.mean())
    q = float(np.mean(d * d))
    proj = U @ (U.T @ d)
    dperp = d - proj
    q_res = float(np.mean(dperp * dperp))
    nov_norm = float(np.sqrt(q_res / q)) if q > 0 else 0.0
    r = (A.T @ d) / np.sqrt(np.sum(A * A, 0) * np.sum(d * d))
    worst = int(np.argmax(np.abs(r)))
    # ценность: E[gain] = (w*tau^2 + mu^2)*q_ост/(2F0), sigma по K1b с локальным g
    return dict(label=label, note=note, q=q, q_res=q_res, nov_norm=nov_norm,
                nov_var=q_res / q, worst_axis=names[worst], worst_r=float(r[worst]),
                top5=[(names[i], float(r[i])) for i in np.argsort(-np.abs(r))[:5]],
                d=d, dperp=dperp)


S = {
  "S1 rec 91-365 (родитель /)": (rec >= 91) & (rec <= 365),
  "S2 rec>=91 покупавшие":            (rec >= 91) & (~never),
  "S3 rec>=91 ИЛИ never":             (rec >= 91) | never,
  "S4 rec>=60 покупавшие":            (rec >= 60) & (~never),
  "S5 rec>=46 покупавшие":            (rec >= 46) & (~never),
  "S6 rec>=31 покупавшие":            (rec >= 31) & (~never),
  "S7 rec>=60 ИЛИ never":             (rec >= 60) | never,
}
XS = ["act30", "searches30", "act90", "browse90", "browse30"]

res = []
print("\n" + "=" * 118)
print(f"{'кандидат':46s}{'m':>8}{'дол.H':>7}{'q':>10}{'q_ост':>10}{'nov':>7}{'nov²':>7}  худшая ось")
print("-" * 118)
for sk, mask in S.items():
    for xn in XS:
        H, L, thr, fH, op = med_split(mask, V[xn])
        v = H.astype(np.float64) - L.astype(np.float64)
        r = screen(v, f"{sk} × {xn}", f"порог {xn}{op}{thr:g}, доля H внутри {fH:.3f}")
        r.update(m=float(mask.mean()), fH=fH, thr=thr, seg=sk, xvar=xn, op=op)
        res.append(r)
        print(f"{sk+' × '+xn:46s}{r['m']:8.4f}{fH:7.3f}{r['q']:10.5f}{r['q_res']:10.5f}"
              f"{r['nov_norm']:7.3f}{r['nov_var']:7.3f}  {r['worst_axis']} {r['worst_r']:+.3f}")

# --- стратифицированные варианты для S3 (медиана отдельно в never и в покупавших)
print("-" * 118)
for xn in XS:
    mb = (rec >= 91) & (~never); mn = never
    Hb, Lb, tb, fb, ob = med_split(mb, V[xn])
    Hn, Ln, tn, fn_, on = med_split(mn, V[xn])
    for sgn, tag in ((+1.0, "оба знака +"), (-1.0, "never со знаком − (механизм )")):
        v = (Hb.astype(float) - Lb.astype(float)) + sgn * (Hn.astype(float) - Ln.astype(float))
        r = screen(v, f"S3strat × {xn} [{tag}]", f"мед. по слоям: покупавшие {tb:g}, never {tn:g}")
        r.update(m=float((mb | mn).mean()), fH=float("nan"), thr=tb, seg="S3 страт.", xvar=xn, op=ob)
        res.append(r)
        print(f"{('S3 страт. × '+xn+' ['+tag+']'):46s}{r['m']:8.4f}{'—':>7}{r['q']:10.5f}"
              f"{r['q_res']:10.5f}{r['nov_norm']:7.3f}{r['nov_var']:7.3f}  {r['worst_axis']} {r['worst_r']:+.3f}")

np.savez_compressed(SP / "p62_dirs.npz", **{f"d_{i}": r["d"] for i, r in enumerate(res)},
                    uid=uid_ref)
json.dump([{k: v for k, v in r.items() if k not in ("d", "dperp")} for r in res],
          open(SP / "p62_grid.json", "w"), ensure_ascii=False, indent=1, default=float)
print("\nсетка сохранена:", SP / "p62_grid.json")
