"""blend_segments.py — строгая перепроверка ПОСЕГМЕНТНОГО бленда.

Гипотеза: модели разной природы (секвенс, бустинги, симулятор, декомпозиции) сильны
в разных зонах пользовательского пространства, поэтому единый набор весов для всех
250k юзеров теряет часть выигрыша.

Ранее направление было закрыто (KNOWLEDGE Н4), но тот тест делался на паре моделей
с корреляцией ошибок 0.995 и без кросс-фита. Здесь — 21 КАЛИБРОВАННАЯ чистая модель
и полностью честный протокол.

Протокол (всё в log1p — пространство RMSLE):
  * внешняя 5-фолдовая CV по юзерам (тот же seed/разбиение, что у blend_reopt);
  * ВНУТРИ каждого внешнего трейна — вложенная CV для выбора (lambda, alpha);
    lambda — усадка посегментных весов к глобальным: w_s = λ·w_s^loc + (1−λ)·w_glob,
    alpha — ridge на локальном NNLS. Выбор гиперпараметров НИКОГДА не видит внешний фолд;
  * метрика — pooled OOF RMSLE (каждый юзер оценён весами, подобранными без него);
  * контроль — случайная сегментация тех же размеров (placebo): показывает, сколько
    «выигрыша» даёт сам факт дробления выборки.

Все подгонки сведены к алгебре Грама: для каждой пары (сегмент, фолд) хранятся
G = X'X, b = X'ly, yy = ly'ly, n. Любая подвыборка — сумма блоков.

Запуск: .venv/bin/python work/scripts/blend_segments.py [--save] [--folds 5]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.linalg import cholesky, solve_triangular  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

EXCLUDE = {"blend"}                       # blend_cal — сам бленд, не модель
CONTAMINATED = {"lgblog_final", "xgblog_final", "cblog_final", "mlp_final", "gru_final",
                "hjit37", "hjit44"}

N_FOLDS = 5
SEED = 42
MIN_CELL = 8000
LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]     # относительный ridge на локальном NNLS
NAME = "blend_seg"


# ------------------------------------------------------------------ пул моделей
def build_pool() -> list[str]:
    names = []
    for p in sorted(PREDS_DIR.glob("*_cal_test.parquet")):
        stem = p.name[: -len("_cal_test.parquet")]
        if stem in EXCLUDE or stem in CONTAMINATED:
            continue
        if not (PREDS_DIR / f"{stem}_cal_val.parquet").exists():
            continue
        names.append(stem + "_cal")
    if (PREDS_DIR / "channel3_chcal_test.parquet").exists() and \
       (PREDS_DIR / "channel3_chcal_val.parquet").exists():
        names.append("channel3_chcal")
    return names


def load_lp(name: str, split: str, uid_ref: np.ndarray) -> np.ndarray:
    d = pl.read_parquet(PREDS_DIR / f"{name}_{split}.parquet").sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid_ref):
        raise ValueError(f"{name}_{split}: user_id не совпадает с базисом")
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


# --------------------------------------------------------------------- решатели
def fit_nnls(G: np.ndarray, b: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """min ||Xw - y||, w >= 0, ridge alpha; G, b нормированы на n."""
    m = G.shape[0]
    jitter = 1e-10 * float(np.trace(G)) / m
    R = cholesky(G + (alpha + jitter) * np.eye(m), lower=False)
    z = solve_triangular(R, b, trans="mdl_larvik", lower=False)
    w, _ = nnls(R, z)
    return w


def sse(G: np.ndarray, b: np.ndarray, yy: float, w: np.ndarray) -> float:
    """Сумма квадратов ошибок при весах w (G, b, yy — НЕнормированные суммы)."""
    return float(w @ G @ w - 2.0 * (b @ w) + yy)


# ------------------------------------------------------------------ сегментации
def bins_recency(rec: np.ndarray) -> np.ndarray:
    """0-7 / 8-30 / 31-90 / 91+ / никогда (null)."""
    out = np.full(len(rec), 4, dtype=np.int32)          # 4 = никогда
    ok = ~np.isnan(rec)
    r = rec[ok]
    lab = np.where(r <= 7, 0, np.where(r <= 30, 1, np.where(r <= 90, 2, 3)))
    out[ok] = lab
    return out


def bins_orddays(od: np.ndarray) -> np.ndarray:
    """0 / 1-2 / 3-5 / 6+."""
    return np.where(od == 0, 0, np.where(od <= 2, 1, np.where(od <= 5, 2, 3))).astype(np.int32)


def bins_decile(x: np.ndarray, uid: np.ndarray) -> np.ndarray:
    """Ранговые децили равного размера (устойчиво к массе нулей)."""
    order = np.lexsort((uid, x))
    ranks = np.empty(len(x), dtype=np.int64)
    ranks[order] = np.arange(len(x))
    return (ranks * 10 // len(x)).astype(np.int32)


def merge_cells(ai: np.ndarray, bi: np.ndarray,
                na: int, nb: int, min_n: int) -> tuple[np.ndarray, dict]:
    """Жадное объединение мелких ячеек сетки (na x nb) до размера >= min_n.

    Соседство — по ребру сетки. Возвращает (map (a,b)->group_id, описание групп).
    """
    groups = {}                       # gid -> set((a,b))
    counts = np.zeros((na, nb), dtype=np.int64)
    for a in range(na):
        for b in range(nb):
            counts[a, b] = int(((ai == a) & (bi == b)).sum())
    gid_of = {}
    for a in range(na):
        for b in range(nb):
            gid = a * nb + b
            groups[gid] = {(a, b)}
            gid_of[(a, b)] = gid

    def gcount(g):
        return sum(counts[a, b] for a, b in groups[g])

    def neighbours(g):
        out = set()
        for a, b in groups[g]:
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (a + da, b + db)
                if 0 <= p[0] < na and 0 <= p[1] < nb and gid_of[p] != g:
                    out.add(gid_of[p])
        return out

    while True:
        small = [g for g in groups if gcount(g) < min_n]
        if not small:
            break
        g = min(small, key=lambda x: (gcount(x), x))
        nb_ = neighbours(g)
        if not nb_:
            break
        h = min(nb_, key=lambda x: (gcount(x), x))
        keep, drop = (h, g) if gcount(h) >= gcount(g) else (g, h)
        groups[keep] |= groups[drop]
        for c in groups[drop]:
            gid_of[c] = keep
        del groups[drop]

    gids = sorted(groups, key=lambda g: (-gcount(g), g))
    remap = {g: i for i, g in enumerate(gids)}
    lab = np.array([[remap[gid_of[(a, b)]] for b in range(nb)] for a in range(na)],
                   dtype=np.int32)
    desc = {i: sorted(groups[g]) for g, i in remap.items()}
    return lab, desc


# --------------------------------------------------------------- ядро протокола
class SegBlocks:
    """Блоки Грама по (сегмент, фолд)."""

    def __init__(self, X: np.ndarray, ly: np.ndarray, seg: np.ndarray,
                 fold: np.ndarray, n_seg: int, n_folds: int):
        m = X.shape[1]
        self.n_seg, self.n_folds, self.m = n_seg, n_folds, m
        self.G = np.zeros((n_seg, n_folds, m, m))
        self.b = np.zeros((n_seg, n_folds, m))
        self.yy = np.zeros((n_seg, n_folds))
        self.n = np.zeros((n_seg, n_folds), dtype=np.int64)
        for s in range(n_seg):
            ms = seg == s
            for f in range(n_folds):
                idx = ms & (fold == f)
                if not idx.any():
                    continue
                Xf, lyf = X[idx], ly[idx]
                self.G[s, f] = Xf.T @ Xf
                self.b[s, f] = Xf.T @ lyf
                self.yy[s, f] = float(lyf @ lyf)
                self.n[s, f] = int(idx.sum())
        self.Gg = self.G.sum(0)      # глобальные блоки по фолдам
        self.bg = self.b.sum(0)
        self.yyg = self.yy.sum(0)
        self.ng = self.n.sum(0)
        self._wcache: dict = {}
        self._bcache: dict = {}

    def sub_global(self, folds):
        f = list(folds)
        return self.Gg[f].sum(0), self.bg[f].sum(0), float(self.yyg[f].sum()), int(self.ng[f].sum())

    def sub_seg(self, s, folds):
        key = (s, folds)
        v = self._bcache.get(key)
        if v is None:
            f = list(folds)
            v = (self.G[s][f].sum(0), self.b[s][f].sum(0),
                 float(self.yy[s][f].sum()), int(self.n[s][f].sum()))
            self._bcache[key] = v
        return v

    def fits(self, folds: tuple, alpha: float):
        """(w_glob, [w_loc по сегментам]) на подвыборке folds; кэшируется."""
        key = (folds, alpha)
        v = self._wcache.get(key)
        if v is not None:
            return v
        Gg, bg, _, ng = self.sub_global(folds)
        w_glob = fit_nnls(Gg / ng, bg / ng, 0.0)
        ws = []
        for s in range(self.n_seg):
            Gs, bs, _, ns = self.sub_seg(s, folds)
            ws.append(fit_nnls(Gs / ns, bs / ns, alpha) if ns > 0 else w_glob)
        v = (w_glob, ws)
        self._wcache[key] = v
        return v


def eval_shrunk(B: SegBlocks, tr_folds, te_folds, lam: float, alpha: float,
                ret_w: bool = False):
    """Фит на tr_folds (глобально + посегментно), оценка SSE на te_folds."""
    tr_folds, te_folds = tuple(tr_folds), tuple(te_folds)
    w_glob, w_locs = B.fits(tr_folds, alpha)
    tot, cnt = 0.0, 0
    ws = {}
    for s in range(B.n_seg):
        w = lam * w_locs[s] + (1.0 - lam) * w_glob
        ws[s] = w
        Gt, bt, yyt, nt = B.sub_seg(s, te_folds)
        if nt == 0:
            continue
        tot += sse(Gt, bt, yyt, w)
        cnt += nt
    if ret_w:
        return tot, cnt, ws, w_glob
    return tot, cnt


def nested_cv(B: SegBlocks, n_folds: int, lam_grid, alpha_grid, verbose=True):
    """Вложенная CV: (lam, alpha) выбираются внутри трейна, оценка — на внешнем фолде."""
    tot, cnt = 0.0, 0
    picks = []
    for f in range(n_folds):
        tr = [g for g in range(n_folds) if g != f]
        best, best_key = np.inf, None
        for alpha in alpha_grid:
            for lam in lam_grid:
                inner = 0.0
                for g in tr:
                    tr2 = [h for h in tr if h != g]
                    s_, _ = eval_shrunk(B, tr2, [g], lam, alpha)
                    inner += s_
                if inner < best - 1e-12:
                    best, best_key = inner, (lam, alpha)
        lam, alpha = best_key
        s_, c_ = eval_shrunk(B, tr, [f], lam, alpha)
        tot += s_
        cnt += c_
        picks.append({"fold": f, "lam": lam, "alpha": alpha})
        if verbose:
            print(f"    outer fold {f}: выбрано lam={lam} alpha={alpha} "
                  f"-> RMSLE({f})={np.sqrt(s_ / c_):.6f}", flush=True)
    return float(np.sqrt(tot / cnt)), picks


def eval_shift(B1: SegBlocks, m: int, tr_folds, te_folds, lam: float, per_seg: bool = True):
    """Контроль: ГЛОБАЛЬНЫЕ веса + свободный сдвиг уровня (посегментный либо один общий).

    Отделяет «специализацию моделей» от банальной посегментной калибровки уровня
    (KNOWLEDGE Ф22/Н5: потолок посегментного сдвига на val ~+0.001).
    """
    tr_folds, te_folds = tuple(tr_folds), tuple(te_folds)
    Gg, bg, _, ng = B1.sub_global(tr_folds)
    wg = fit_nnls(Gg[:m, :m] / ng, bg[:m] / ng, 0.0)
    c_glob = float((bg[m] - Gg[m, :m] @ wg) / ng)
    tot, cnt = 0.0, 0
    shifts = {}
    for s in range(B1.n_seg):
        Gs, bs, _, ns = B1.sub_seg(s, tr_folds)
        c = float((bs[m] - Gs[m, :m] @ wg) / ns) if (per_seg and ns > 0) else c_glob
        shifts[s] = c
        w = np.concatenate([wg, [lam * c]])
        Gt, bt, yyt, nt = B1.sub_seg(s, te_folds)
        if nt == 0:
            continue
        tot += sse(Gt, bt, yyt, w)
        cnt += nt
    return tot, cnt, shifts


def nested_cv_shift(B1: SegBlocks, m: int, n_folds: int, lam_grid, per_seg=True):
    tot, cnt, picks = 0.0, 0, []
    for f in range(n_folds):
        tr = [g for g in range(n_folds) if g != f]
        best, best_lam = np.inf, None
        for lam in lam_grid:
            inner = sum(eval_shift(B1, m, [h for h in tr if h != g], [g], lam, per_seg)[0]
                        for g in tr)
            if inner < best - 1e-12:
                best, best_lam = inner, lam
        s_, c_, _ = eval_shift(B1, m, tr, [f], best_lam, per_seg)
        tot += s_
        cnt += c_
        picks.append(best_lam)
    return float(np.sqrt(tot / cnt)), picks


def lam_curve(B: SegBlocks, n_folds: int, lam_grid, alpha: float = 0.0):
    """Диагностика: OOF при ФИКСИРОВАННОЙ lam (оракульный выбор, оптимистичен)."""
    out = {}
    for lam in lam_grid:
        tot, cnt = 0.0, 0
        for f in range(n_folds):
            tr = [g for g in range(n_folds) if g != f]
            s_, c_ = eval_shrunk(B, tr, [f], lam, alpha)
            tot += s_
            cnt += c_
        out[lam] = float(np.sqrt(tot / cnt))
    return out


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    a = ap.parse_args()
    t0 = time.time()

    # ---------------- данные ----------------
    fv = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=["user_id", "target", "rec_order", "ord_days_90",
                                  "gmv_sum_365"]).sort("user_id")
    ft = pl.read_parquet(FEATURES_DIR / f"anchor={TEST_ANCHOR.isoformat()}.parquet",
                         columns=["user_id", "rec_order", "ord_days_90",
                                  "gmv_sum_365"]).sort("user_id")
    uid = fv["user_id"].to_numpy()
    uid_t = ft["user_id"].to_numpy()
    ly = np.log1p(np.clip(fv["target"].to_numpy().astype(np.float64), 0, None))
    N = len(uid)

    pool = build_pool()
    print(f"[пул] {len(pool)} калиброванных чистых моделей: {', '.join(pool)}", flush=True)
    X = np.column_stack([load_lp(n, "val", uid) for n in pool])
    Xt = np.column_stack([load_lp(n, "test", uid_t) for n in pool])
    m = X.shape[1]

    rng = np.random.default_rng(SEED)
    fold = rng.permutation(N) % a.folds

    # ---------------- глобальный NNLS (база) ----------------
    Gf = np.zeros((a.folds, m, m)); bf = np.zeros((a.folds, m))
    yyf = np.zeros(a.folds); nf = np.zeros(a.folds, dtype=np.int64)
    for f in range(a.folds):
        idx = fold == f
        Xf, lyf = X[idx], ly[idx]
        Gf[f] = Xf.T @ Xf; bf[f] = Xf.T @ lyf
        yyf[f] = float(lyf @ lyf); nf[f] = int(idx.sum())
    tot = 0.0
    for f in range(a.folds):
        tr = [g for g in range(a.folds) if g != f]
        ntr = int(nf[tr].sum())
        w = fit_nnls(Gf[tr].sum(0) / ntr, bf[tr].sum(0) / ntr, 0.0)
        tot += sse(Gf[f], bf[f], float(yyf[f]), w)
    global_oof = float(np.sqrt(tot / N))
    w_glob_full = fit_nnls(Gf.sum(0) / N, bf.sum(0) / N, 0.0)
    print(f"\n[глобальный NNLS] OOF = {global_oof:.6f}  (ожидалось 1.666791)")
    print("  веса:", {pool[i]: round(float(w_glob_full[i]), 4)
                      for i in np.argsort(-w_glob_full) if w_glob_full[i] > 1e-4}, flush=True)

    # ---------------- сегментации ----------------
    rec_v = bins_recency(fv["rec_order"].to_numpy().astype(np.float64))
    rec_t = bins_recency(ft["rec_order"].to_numpy().astype(np.float64))
    od_v = bins_orddays(fv["ord_days_90"].to_numpy().astype(np.float64))
    od_t = bins_orddays(ft["ord_days_90"].to_numpy().astype(np.float64))
    dec_v = bins_decile(fv["gmv_sum_365"].to_numpy().astype(np.float64), uid)
    dec_t = bins_decile(ft["gmv_sum_365"].to_numpy().astype(np.float64), uid_t)

    cell_map, cell_desc = merge_cells(rec_v, od_v, 5, 4, MIN_CELL)
    cross_v = cell_map[rec_v, od_v]
    cross_t = cell_map[rec_t, od_t]

    REC_LAB = ["ord 0-7д", "ord 8-30д", "ord 31-90д", "ord 90+д", "никогда"]
    OD_LAB = ["дней 0", "дней 1-2", "дней 3-5", "дней 6+"]

    SEGS = {
        "recency": (rec_v, rec_t, REC_LAB),
        "orddays90": (od_v, od_t, OD_LAB),
        "gmv365_decile": (dec_v, dec_t, [f"дец {i}" for i in range(10)]),
        "rec_x_ord": (cross_v, cross_t,
                      [" + ".join(f"{REC_LAB[x]}/{OD_LAB[y]}" for x, y in cell_desc[i])
                       for i in sorted(cell_desc)]),
    }

    results = {}
    blocks = {}
    for sname, (sv, st, labs) in SEGS.items():
        n_seg = int(sv.max()) + 1
        sizes_v = np.bincount(sv, minlength=n_seg)
        sizes_t = np.bincount(st, minlength=n_seg)
        print(f"\n=== сегментация «{sname}» : {n_seg} сегментов ===")
        for s in range(n_seg):
            print(f"  [{s}] n_val={sizes_v[s]:>7} n_test={sizes_t[s]:>7}  {labs[s][:90]}")
        B = SegBlocks(X, ly, sv, fold, n_seg, a.folds)
        blocks[sname] = (B, sv, st, labs)
        curve = lam_curve(B, a.folds, LAM_GRID, 0.0)
        print("  OOF по фиксированной lam (оракул, оптимистично):",
              {k: round(v, 6) for k, v in curve.items()}, flush=True)
        oof, picks = nested_cv(B, a.folds, LAM_GRID, ALPHA_GRID)
        print(f"  ЧЕСТНЫЙ вложенный OOF = {oof:.6f}   gain = {global_oof - oof:+.6f}",
              flush=True)
        results[sname] = {
            "n_seg": n_seg,
            "sizes_val": sizes_v.tolist(), "sizes_test": sizes_t.tolist(),
            "labels": labs,
            "lam_curve_oracle": {str(k): round(v, 6) for k, v in curve.items()},
            "oracle_best_lam": float(min(curve, key=curve.get)),
            "oracle_best_oof": round(float(min(curve.values())), 6),
            "nested_oof": round(oof, 6),
            "nested_gain": round(global_oof - oof, 6),
            "fold_picks": picks,
        }

    # ---------------- placebo: случайная сегментация тех же размеров ----------------
    best_seg = min(results, key=lambda k: results[k]["nested_oof"])
    n_seg_b = results[best_seg]["n_seg"]
    prng = np.random.default_rng(777)
    plac = []
    for rep in range(3):
        sizes = results[best_seg]["sizes_val"]
        lab = np.concatenate([np.full(sz, i) for i, sz in enumerate(sizes)]).astype(np.int32)
        lab = lab[prng.permutation(N)]
        Bp = SegBlocks(X, ly, lab, fold, n_seg_b, a.folds)
        cp = lam_curve(Bp, a.folds, LAM_GRID, 0.0)
        op, _ = nested_cv(Bp, a.folds, LAM_GRID, ALPHA_GRID, verbose=False)
        plac.append({"rep": rep, "oracle_best_oof": round(float(min(cp.values())), 6),
                     "oracle_best_lam": float(min(cp, key=cp.get)),
                     "nested_oof": round(op, 6), "nested_gain": round(global_oof - op, 6)})
        print(f"[placebo {rep}] случайные {n_seg_b} сегментов: оракул "
              f"{min(cp.values()):.6f} (lam={min(cp, key=cp.get)}), вложенный {op:.6f} "
              f"gain {global_oof - op:+.6f}", flush=True)

    # ------- декомпозиция: сколько из выигрыша даёт просто посегментный сдвиг -------
    B, sv, st, labs = blocks[best_seg]
    X1 = np.column_stack([X, np.ones(N)])
    B1 = SegBlocks(X1, ly, sv, fold, B.n_seg, a.folds)
    shift_seg_oof, shift_picks = nested_cv_shift(B1, m, a.folds, LAM_GRID, True)
    shift_glob_oof, _ = nested_cv_shift(B1, m, a.folds, [1.0], False)
    print(f"\n[декомпозиция уровня] глобальный NNLS {global_oof:.6f} | "
          f"+ ОДИН общий сдвиг {shift_glob_oof:.6f} | + ПОСЕГМЕНТНЫЙ сдвиг "
          f"{shift_seg_oof:.6f} | + полный посегментный перевес "
          f"{results[best_seg]['nested_oof']:.6f}", flush=True)
    out_shift = {
        "global_only": round(global_oof, 6),
        "plus_one_global_shift": round(shift_glob_oof, 6),
        "plus_per_segment_shift": round(shift_seg_oof, 6),
        "plus_full_reweight": results[best_seg]["nested_oof"],
        "shift_lam_picks": shift_picks,
    }

    # ---------------- карта «кто где силён» ----------------
    def _mode(vals):
        vals = list(vals)
        return float(min(set(vals), key=lambda v: (-vals.count(v), v)))

    lam_full = _mode(p["lam"] for p in results[best_seg]["fold_picks"])
    alpha_full = _mode(p["alpha"] for p in results[best_seg]["fold_picks"])
    all_folds = tuple(range(a.folds))
    _, _, ws_full, wg_full = eval_shrunk(B, all_folds, all_folds, 1.0, alpha_full, ret_w=True)
    seg_solo = {}
    model_map_lines = []
    print(f"\n=== карта весов (сегментация {best_seg}, чистые локальные NNLS, alpha={alpha_full}) ===")
    for s in range(B.n_seg):
        w = ws_full[s]
        top = sorted(((pool[i], float(w[i])) for i in range(m) if w[i] > 5e-3),
                     key=lambda kv: -kv[1])[:6]
        Gs, bs, yys, ns = B.sub_seg(s, all_folds)
        r_loc = np.sqrt(sse(Gs, bs, yys, w) / ns)
        r_gl = np.sqrt(sse(Gs, bs, yys, wg_full) / ns)
        seg_solo[s] = {"n": ns, "rmsle_global_w": round(float(r_gl), 6),
                       "rmsle_local_w": round(float(r_loc), 6),
                       "insample_gain": round(float(r_gl - r_loc), 6),
                       "top": [(n, round(v, 4)) for n, v in top],
                       "sumw": round(float(w.sum()), 4)}
        line = (f"  [{s}] n={ns:>7} {labs[s][:60]:60s} rmsle {r_gl:.4f}->{r_loc:.4f} "
                f"| " + ", ".join(f"{n} {v:.3f}" for n, v in top))
        model_map_lines.append(line)
        print(line, flush=True)

    # ---------------- устойчивость: другие разбиения на фолды ----------------
    seeds = [SEED, 1337, 7, 2024, 99]
    seed_rows = []
    for sd in seeds:
        r2 = np.random.default_rng(sd)
        fold2 = r2.permutation(N) % a.folds
        Gf2 = np.zeros((a.folds, m, m)); bf2 = np.zeros((a.folds, m))
        yyf2 = np.zeros(a.folds); nf2 = np.zeros(a.folds, dtype=np.int64)
        for f in range(a.folds):
            idx = fold2 == f
            Xf, lyf = X[idx], ly[idx]
            Gf2[f] = Xf.T @ Xf; bf2[f] = Xf.T @ lyf
            yyf2[f] = float(lyf @ lyf); nf2[f] = int(idx.sum())
        t2 = 0.0
        for f in range(a.folds):
            tr = [g for g in range(a.folds) if g != f]
            ntr = int(nf2[tr].sum())
            t2 += sse(Gf2[f], bf2[f], float(yyf2[f]),
                      fit_nnls(Gf2[tr].sum(0) / ntr, bf2[tr].sum(0) / ntr, 0.0))
        g_oof = float(np.sqrt(t2 / N))
        B2 = SegBlocks(X, ly, sv, fold2, B.n_seg, a.folds)
        s_oof, _ = nested_cv(B2, a.folds, LAM_GRID, ALPHA_GRID, verbose=False)
        seed_rows.append({"seed": sd, "global_oof": round(g_oof, 6),
                          "seg_oof": round(s_oof, 6), "gain": round(g_oof - s_oof, 6)})
        print(f"[seed {sd}] глоб {g_oof:.6f}  сегм {s_oof:.6f}  gain {g_oof - s_oof:+.6f}",
              flush=True)
    g_arr = np.array([r["gain"] for r in seed_rows])
    print(f"[устойчивость] gain по 5 разбиениям: {g_arr.mean():+.6f} ± {g_arr.std():.6f}"
          f"  (min {g_arr.min():+.6f}, max {g_arr.max():+.6f})", flush=True)

    # ---------------- независимая проверка на ТЕСТОВОМ окне (predict_lb) ----------
    lb = None
    try:
        import predict_lb as PL
        basis = PL.load_basis()
        P = PL.LBPredictor(basis)
        if np.array_equal(uid_t, P.uid):
            def scored(lp):
                """(прогноз как есть, прогноз ПОСЛЕ оптимального глобального сдвига).

                f²(c) = f²(0) − c², c* = MEAN_T − mean(lp): уровень пайплайн всё равно
                перемеряет на LB (KNOWLEDGE R9/F17), поэтому честно сравнивать форму.
                """
                r = P.predict(lp)
                c = PL.MEAN_T - float(lp.mean())
                return r, float(np.sqrt(max(r["pred"] ** 2 - c ** 2, 1e-12))), c

            def seg_lp(ws_dict, wg, seg_lab, lam):
                z = np.zeros(len(uid_t))
                for s in range(len(ws_dict)):
                    z[seg_lab == s] = Xt[seg_lab == s] @ (lam * ws_dict[s] + (1 - lam) * wg)
                return z

            lt_glob = Xt @ w_glob_full
            lt_seg = seg_lp(ws_full, wg_full, st, lam_full)
            # placebo: случайная сегментация тех же размеров, та же процедура
            prng2 = np.random.default_rng(4242)
            lab_v = np.concatenate([np.full(int(c), i)
                                    for i, c in enumerate(np.bincount(sv, minlength=B.n_seg))])
            lab_v = lab_v[prng2.permutation(N)].astype(np.int32)
            lab_t = np.concatenate([np.full(int(c), i)
                                    for i, c in enumerate(np.bincount(st, minlength=B.n_seg))])
            lab_t = lab_t[prng2.permutation(len(uid_t))].astype(np.int32)
            Bp2 = SegBlocks(X, ly, lab_v, fold, B.n_seg, a.folds)
            _, _, ws_p, wg_p = eval_shrunk(Bp2, all_folds, all_folds, 1.0, alpha_full,
                                           ret_w=True)
            lt_plac = seg_lp(ws_p, wg_p, lab_t, lam_full)

            rows = {}
            for tag, lp_ in (("global", lt_glob), ("segmented", lt_seg),
                             ("placebo_seg", lt_plac)):
                r, f_sh, c = scored(lp_)
                rows[tag] = {"pred": round(r["pred"], 6), "pred_shift_opt": round(f_sh, 6),
                             "mean_lp": round(float(lp_.mean()), 5),
                             "opt_shift": round(c, 5),
                             "sigma68": round(r["sigma68"], 6),
                             "sd_resid": round(r["sd_resid"], 5)}
            lb = {"lam_used": lam_full, "variants": rows,
                  "delta_raw": round(rows["segmented"]["pred"] - rows["global"]["pred"], 6),
                  "delta_shift_opt": round(rows["segmented"]["pred_shift_opt"]
                                           - rows["global"]["pred_shift_opt"], 6),
                  "placebo_delta_shift_opt": round(rows["placebo_seg"]["pred_shift_opt"]
                                                   - rows["global"]["pred_shift_opt"], 6)}
            print("\n[predict_lb] (уровень пайплайн перемеряет на LB -> смотреть колонку "
                  "«со сдвигом»)")
            for tag, d in rows.items():
                print(f"  {tag:12s} как есть {d['pred']:.6f}  со сдвигом {d['pred_shift_opt']:.6f}"
                      f"  mean_lp {d['mean_lp']:.4f}  c* {d['opt_shift']:+.4f}")
            print(f"  дельта (со сдвигом): сегментный {lb['delta_shift_opt']:+.6f}, "
                  f"ПЛАЦЕБО {lb['placebo_delta_shift_opt']:+.6f}", flush=True)
        else:
            print("[predict_lb] пропуск: user_id теста не совпадает с базисом")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[predict_lb] пропуск: {exc}")

    out = {
        "pool": pool,
        "n_users": N,
        "folds": a.folds,
        "global_oof": round(global_oof, 6),
        "global_weights": {pool[i]: round(float(w_glob_full[i]), 6)
                           for i in np.argsort(-w_glob_full) if w_glob_full[i] > 1e-6},
        "segmentations": results,
        "best_segmentation": best_seg,
        "placebo": plac,
        "level_decomposition": out_shift,
        "seed_robustness": seed_rows,
        "gain_mean_over_seeds": round(float(g_arr.mean()), 6),
        "gain_std_over_seeds": round(float(g_arr.std()), 6),
        "lb_check": lb,
        "best_lambda": lam_full,
        "best_alpha": alpha_full,
        "segment_map": {str(k): v for k, v in seg_solo.items()},
        "runtime_s": round(time.time() - t0, 1),
    }

    # ---------------- финальный вердикт / сабмит ----------------
    best = results[best_seg]
    gain = best["nested_gain"]
    out["gain"] = gain
    out["shipped"] = False

    (REPORTS_DIR / "blend_segments.json").write_text(json.dumps(out, indent=1,
                                                               ensure_ascii=False))
    print(f"\nJSON: work/reports/blend_segments.json  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
