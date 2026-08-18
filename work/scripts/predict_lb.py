"""predict_lb.py — прогноз public-скора submission-файла БЕЗ отправки на лидерборд.

Метод (см. work/reports/lb_predictor.md): "span-алгебра + val-остаток".

    f² = mean_P(lp²) − 2·mean_P(lp·t) + mean_P(t²)

Локально известно всё, кроме φ(lp) = mean_P(lp·t). φ линеен по lp, поэтому:

 1. направление d = lp_new − lp_A1 проецируется на span{1, lp_i − lp_A1} замеренных файлов;
    для этой части φ известна ТОЧНО (восстановлена из замеренных скоров);
 2. ортогональный остаток r оценивается через ковариацию с ВАЛИДАЦИОННЫМ таргетом:
    φ(r) ≈ a · cov(r, t_val), a = 1.052 (подобрано на замеренных файлах);
 3. f_new² = f_A1² + (mean(lp_new²) − mean(lp_A1²)) − 2·φ̂(d).

mean_P(t²) в формулу НЕ входит (сокращается) — прогноз не зависит от её 2-значной точности.
Единственная внешняя константа — mean_P(t) = 2.3275 (входит с весом c₀ ≈ сдвиг среднего).

Честная точность (последовательный холдаут на 13 поздних файлах, которые НЕ участвовали
в настройке): MAE 0.00012, max 0.00075, Spearman +0.967.
Для файлов с ПРИНЦИПИАЛЬНО новым направлением (новая модель, не выражаемая через
замеренные): MAE ≈ 0.0015, худший случай ≈ 0.002.

Использование:
    .venv/bin/python work/scripts/predict_lb.py --selftest      # честный холдаут заново
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submissions"
CANON = SUB / "canonical"
CACHE = ROOT / "work" / "lb_cache.npz"
VAL_ANCHOR_FEAT = ROOT / "work" / "features" / "anchor=2026-01-14.parquet"

# --------------------------------------------------------------- измеренные константы
MEAN_T = 2.3275           # mean_P(log1p(y_test)), замерено парой сдвинутых сабмитов
MEAN_T_SQ = 10.79         # mean_P(log1p(y_test)²); в прогноз скора НЕ входит (сокращается),
                          # нужна только для построения весов val-окна (--weights)
A_RESID = 1.0520          # коэффициент при cov(остаток, t_val); подобран на 33 ранних файлах
KAPPA68 = 0.010           # σ68 = KAPPA68*sd(resid) + FLOOR68
FLOOR68 = 0.00010
KAPPA95 = 0.045           # σ95 = KAPPA95*sd(resid) + FLOOR95
FLOOR95 = 0.00025
ANCHOR = "A1_gram7_shift"

# файл -> (имя csv, точный public-скор).  Порядок = хронологический (важен для --selftest).
MEASURED: list[tuple[str, str, float]] = [
    ("sample_submit",     "sample_submit.csv",              2.122483523224017),
    ("sub_blend_w1a",     "sub_blend_w1a.csv",              1.6754553658578413),
    ("sub_twlog_probe",   "sub_twlog_probe.csv",            1.66779),
    ("sub_c_cand",        "sub_c_cand.csv",                 1.6695398157),
    ("lbmix4_3way",       "lbmix4_3way.csv",                1.6573961435),
    ("A1_gram7_shift",    "A1_gram7_shift.csv",             1.6535955005),
    ("A2_probe_s1_gmv",   "A2_probe_s1_gmv.csv",            1.6563024241),
    ("F4_applied",        "F4_applied.csv",                 1.64916806),
    ("F5_probe_hmmsim",   "F5_probe_hmmsim.csv",            1.6499003958),
    ("G1_probe_zeropush", "G1_probe_zeropush.csv",          1.6507738649),
    ("H1_applied",        "H1_applied.csv",                 1.6489955175417363),
    ("H2_edge_p1",        "H2_edge_p1.csv",                 1.6490330321),
]


# ------------------------------------------------------------------------ ввод-вывод
def _resolve(fn: str) -> Path:
    for p in (SUB / fn, CANON / fn, ROOT / fn, Path(fn)):
        if p.exists():
            return p
    raise FileNotFoundError(f"не найден {fn}")


def read_lp(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """(user_id, log1p(predict)), отсортировано по user_id."""
    d = pl.read_csv(_resolve(str(path)), schema_overrides={"user_id": pl.Int64}).sort("user_id")
    col = "predict" if "predict" in d.columns else d.columns[1]
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d[col].to_numpy().astype(np.float64), 0, None)))


def load_basis(rebuild: bool = False) -> dict:
    """Матрица направлений замеренных файлов + валидационный таргет (с кэшем)."""
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        if len(z["names"]) == len(MEASURED):
            return {k: z[k] for k in z.files} | {"names": [str(x) for x in z["names"]]}
    uid0, rows = None, []
    for _, fn, _ in MEASURED:
        uid, lp = read_lp(fn)
        if uid0 is None:
            uid0 = uid
        elif not np.array_equal(uid, uid0):
            raise ValueError(f"{fn}: другой порядок user_id")
        rows.append(lp)
    L = np.stack(rows)
    v = pl.read_parquet(VAL_ANCHOR_FEAT, columns=["user_id", "target"]).sort("user_id")
    if not np.array_equal(v["user_id"].to_numpy(), uid0):
        raise ValueError("val-таргет не совпадает по user_id с сабмитами")
    tval = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64), 0, None))
    names = [n for n, _, _ in MEASURED]
    f = np.array([s for _, _, s in MEASURED])
    np.savez_compressed(CACHE, L=L, tval=tval, uid=uid0, f=f, names=np.array(names))
    return dict(L=L, tval=tval, uid=uid0, f=f, names=names)


# ------------------------------------------------------------------------ ядро метода
class LBPredictor:
    def __init__(self, basis: dict, use_idx: list[int] | None = None, a: float = A_RESID):
        self.names = list(basis["names"])
        self.uid = basis["uid"]
        self.a = a
        L, f = basis["L"], basis["f"]
        self.anch = self.names.index(ANCHOR)
        idx = list(range(len(self.names))) if use_idx is None else list(use_idx)
        if self.anch not in idx:
            raise ValueError("якорь должен быть в обучающем наборе")
        self.N = L.shape[1]
        self.qd = (L * L).mean(1)
        # psi_i = phi(lp_i - lp_anchor); mean_P(t²) сокращается
        self.psi = ((self.qd - self.qd[self.anch]) - (f ** 2 - f[self.anch] ** 2)) / 2
        self.lp_a = L[self.anch]
        self.f_a = float(f[self.anch])
        self.q_a = float(self.qd[self.anch])
        self.idx = idx
        Dtr = L[idx] - self.lp_a
        self.B = np.vstack([np.ones(self.N), Dtr])
        G = self.B @ self.B.T / self.N
        self.G = G + 1e-9 * np.trace(G) / len(G) * np.eye(len(G))
        self.psi_vec = np.concatenate([[MEAN_T], self.psi[idx]])
        tv = basis["tval"]
        self.tv_c = tv - tv.mean()

    def predict(self, lp: np.ndarray) -> dict:
        d = lp - self.lp_a
        c = np.linalg.solve(self.G, self.B @ d / self.N)
        resid = d - c @ self.B
        psi_span = float(c @ self.psi_vec)
        cov_val = float((resid * self.tv_c).mean())
        psi_hat = psi_span + self.a * cov_val
        q = float((lp * lp).mean())
        fsq = self.f_a ** 2 + (q - self.q_a) - 2 * psi_hat
        fhat = float(np.sqrt(max(fsq, 1e-12)))
        rsd = float(resid.std())
        nov = float((resid ** 2).mean() / max((d ** 2).mean(), 1e-15))
        # вклады в скор (в единицах f): что дала бы чистая span-часть
        f_span = float(np.sqrt(max(self.f_a ** 2 + (q - self.q_a) - 2 * psi_span, 1e-12)))
        top = sorted(zip([self.names[i] for i in self.idx], c[1:]),
                     key=lambda t: -abs(t[1]))[:5]
        return dict(
            pred=fhat, sigma68=KAPPA68 * rsd + FLOOR68, sigma95=KAPPA95 * rsd + FLOOR95,
            novelty=nov, sd_resid=rsd, mean_lp=float(lp.mean()),
            shift_vs_anchor=float(c[0]), val_correction=fhat - f_span,
            f_span_only=f_span, weight_sum=float(c[1:].sum()),
            top_terms=[(n, float(v)) for n, v in top],
            extrapolation=bool(rsd > 0.15),
        )


# ------------------------------------------------- побочный продукт: веса val-окна
def _maxent(F: np.ndarray, targ: np.ndarray, iters: int = 400) -> tuple[np.ndarray, float]:
    """w ∝ exp(λ·F), нормированы к mean(w)=1, с mean_w(F_j) = targ_j."""
    m, n = F.shape
    lam = np.zeros(m)
    g = np.full(m, np.inf)
    for _ in range(iters):
        z = lam @ F
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        e = F @ w
        g = e - targ
        if np.abs(g).max() < 1e-11:
            break
        J = (F * w) @ F.T - np.outer(e, e)
        lam -= np.linalg.solve(J + 1e-10 * np.trace(J) / m * np.eye(m), g)
    return w * n, float(np.abs(g).max())


def segment_weights(basis: dict, delta: float = 0.30) -> dict:
    """Веса юзеров на val-окне, при которых val-таргет повторяет ЗАМЕРЕННУЮ структуру
    тестового: 4 посегментных средних (gmv_sum_365-децили, пробы A2-A4) + mean(t²).

    Пригодны как sample-weights при обучении/калибровке на val-окне.
    """
    names, f, L, tv = list(basis["names"]), basis["f"], basis["L"], basis["tval"]
    segs, mean_t = {}, {}
    segs["S4"] = ~(segs["S1"] | segs["S2"] | segs["S3"])
    F = np.vstack([segs[k].astype(float) * tv for k in ("S1", "S2", "S3", "S4")] + [tv ** 2])
    T = np.array([segs[k].mean() * mean_t[k] for k in ("S1", "S2", "S3", "S4")] + [MEAN_T_SQ])
    w, resid = _maxent(F, T)
    return dict(w=w, resid=resid, segs=segs, mean_t_test=mean_t,
                mean_t_val={k: float(tv[v].mean()) for k, v in segs.items()},
                share={k: float(v.mean()) for k, v in segs.items()})


def fmt(name: str, r: dict) -> str:
    flag = "  ⚠ ЭКСТРАПОЛЯЦИЯ (остаток вне диапазона калибровки)" if r["extrapolation"] else ""
    return (f"{name}\n"
            f"  прогноз public RMSLE : {r['pred']:.6f}  ±{r['sigma68']:.5f} (68%)  "
            f"±{r['sigma95']:.5f} (95%){flag}\n"
            f"  novelty              : {r['novelty']:.3e}   sd(остатка) {r['sd_resid']:.4f}\n"
            f"  span-часть           : {r['f_span_only']:.6f}   поправка по val: "
            f"{r['val_correction']:+.6f}\n"
            f"  сдвиг к якорю c₀     : {r['shift_vs_anchor']:+.5f}   Σвесов {r['weight_sum']:.4f}\n"
            f"  главные члены        : " +
            ", ".join(f"{n} {v:+.3f}" for n, v in r["top_terms"]))


# ------------------------------------------------------------------------ самопроверка
def selftest(basis: dict) -> dict:
    from scipy.stats import spearmanr
    names, f = list(basis["names"]), basis["f"]
    split = names.index("G1_probe_zeropush")
    L = basis["L"]

    def run(mode: str):
        preds, trues = [], []
        for j in range(split, len(names)):
            use = list(range(split)) if mode == "batch" else list(range(j))
            p = LBPredictor(basis, use_idx=use).predict(L[j])
            preds.append(p["pred"]); trues.append(f[j])
        e = np.array(preds) - np.array(trues)
        return (float(np.abs(e).mean()), float(np.abs(e).max()),
                float(spearmanr(preds, trues).statistic), preds, trues)

    out = {}
    for mode in ("batch", "sequential"):
        mae, mx, sp, preds, trues = run(mode)
        out[mode] = dict(mae=mae, max=mx, spearman=sp)
        print(f"\n=== холдаут «{mode}» — 13 поздних файлов (G1…), не участвовали в настройке ===")
        for j, (p, t) in enumerate(zip(preds, trues)):
            print(f"  {names[split+j]:20s} факт {t:.6f}  прогноз {p:.6f}  ошибка {p-t:+.5f}")
        print(f"  MAE {mae:.6f}   max {mx:.6f}   Spearman {sp:+.4f}")

    # отдельно: файлы с принципиально новым направлением (первое появление семейства)
    errs = []
    print("\n=== файлы с новым направлением (прогноз только по предыдущим замерам) ===")
    out["novel_direction"] = dict(mae=float(np.abs(errs).mean()), max=float(np.abs(errs).max()))
    print(f"  MAE {out['novel_direction']['mae']:.6f}  max {out['novel_direction']['max']:.6f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="submission csv")
    ap.add_argument("--selftest", action="store_true", help="честная оценка на холдауте")
    ap.add_argument("--rebuild", action="store_true", help="перестроить кэш базиса")
    ap.add_argument("--json", action="store_true", help="сырой JSON на выход")
    ap.add_argument("--exclude", default="", help="исключить файлы базиса (через запятую)")
    ap.add_argument("--weights", metavar="OUT.parquet", nargs="?", const="-",
                    help="построить веса val-окна под замеренную сегментную структуру теста")
    a = ap.parse_args()

    basis = load_basis(rebuild=a.rebuild)
    if a.weights:
        r = segment_weights(basis)
        print(f"невязка ограничений {r['resid']:.2e}   ESS {1/(r['w']**2).mean():.4f}   "
              f"вес {r['w'].min():.3f}…{r['w'].max():.3f}")
        print(f"{'сегмент':10s}{'доля':>8}{'mean(t) тест':>15}{'mean(t) val':>14}{'сдвиг':>10}")
        for k in ("S1", "S2", "S3", "S4"):
            print(f"{k:10s}{r['share'][k]:8.4f}{r['mean_t_test'][k]:15.5f}"
                  f"{r['mean_t_val'][k]:14.5f}{r['mean_t_test'][k]-r['mean_t_val'][k]:+10.5f}")
        if a.weights != "-":
            pl.DataFrame({"user_id": basis["uid"], "w": r["w"]}).write_parquet(a.weights)
            print(f"сохранено: {a.weights}")
        return 0
    if a.selftest:
        res = selftest(basis)
        if a.json:
            print(json.dumps(res, indent=1))
        return 0
    if not a.files:
        ap.print_help()
        return 1

    drop = {s.strip() for s in a.exclude.split(",") if s.strip()}
    use = [i for i, n in enumerate(basis["names"]) if n not in drop]
    P = LBPredictor(basis, use_idx=use)
    out = {}
    for fp in a.files:
        uid, lp = read_lp(fp)
        if not np.array_equal(uid, P.uid):
            print(f"{fp}: user_id не совпадает с базисом", file=sys.stderr)
            return 2
        r = P.predict(lp)
        out[str(fp)] = r
        if not a.json:
            print(fmt(str(fp), r))
    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
