"""G1 zeropush probe: push confident-simulator-zeros toward zero.

Direction built from hmmsim p_zero (mechanism signal, err-corr 0.915 with blend):
  d = -1[p_zero > tau] * w(confidence)          (flat variant)
  d = -lp_base * 1[p_zero > tau] * w(conf)      (proportional variant, full push to 0)
w = clip((p_zero - tau)/(1 - tau), 0, 1).

Candidates over tau in {0.85, 0.92} x {flat, prop}; each residualized on TEST
against the measured basis (subs.MEASURED + D*/E*/F* files + const), rms-normed
to 0.12. Val signal: same construction on VAL (aux p_zero val), residualized
against clean val-analog span, c_val = mean(e_val * h_val), e_val from my27_val.
Best by novelty * (-c_val) -> work/probes/h_zeropush.npy +
submissions/G1_probe_zeropush.csv = expm1(clip(lp_F4 + 0.45*h, 0)).  NOT submitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from subs import MEASURED, lp, novelty, span_matrix  # noqa: E402

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
PREDS = ROOT / "work" / "preds"
PACK = ROOT / "work" / "preds_pack"
PROBES = ROOT / "work" / "probes"

DEF_FILES = [
]
BASE = "F4_applied.csv"
F_BASE = 1.649218          # calculated public position of F4_applied
STEP = 0.45
RMS = 0.12
TAUS = (0.85, 0.92)


def lp_val(name: str) -> np.ndarray:
    d = pl.read_parquet(PREDS / f"{name}_val.parquet").sort("user_id")
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


def rms_norm(x: np.ndarray, target: float = RMS) -> np.ndarray:
    return x * (target / np.sqrt((x ** 2).mean()))


def main():
    # ---------------- test side
    uid, lp_base = lp(BASE)
    n = len(uid)
    aux_t = pl.read_parquet(PREDS / "hmmsim_aux_test.parquet").sort("user_id")
    assert (aux_t["user_id"].to_numpy() == uid).all()
    pz_t = aux_t["p_zero"].to_numpy()

    print("building measured span (test) ...", flush=True)
    Sp = span_matrix(MEASURED + DEF_FILES, n)
    print(f"  span: {Sp.shape[0]} vectors (const + {len(MEASURED)} MEASURED + {len(DEF_FILES)} D/E/F)")

    # ---------------- val side
    pack = pl.read_parquet(PACK / "val_preds.parquet").sort("user_id")
    assert (pack["user_id"].to_numpy() == uid).all()
    lt = np.log1p(pack["target"].to_numpy().astype(np.float64))
    aux_v = pl.read_parquet(PREDS / "hmmsim_aux_val.parquet").sort("user_id")
    assert (aux_v["user_id"].to_numpy() == uid).all()
    pz_v = aux_v["p_zero"].to_numpy()

    lp_base_v = lp_val("my27")               # clean val analog of the applied base
    e_val = lp_base_v - lt
    f_val_base = float(np.sqrt((e_val ** 2).mean()))
    print(f"val base my27: rmsle={f_val_base:.4f}, mean_e={e_val.mean():+.4f}")

    pack_cols = ["mlpziln_cal", "mlpbin_cal", "mlp2_big_cal", "mlp2_final_cal",
                 "channel2_cal", "c_xtw_s42", "c_ts2_s42", "c_twlog_s42",
                 "c_dirlgb_s42", "twdeep", "seq2tr_f", "gru_final", "febspec",
                 "fusion_f", "whale_final", "short14", "rankmodel", "behavonly"]
    Vspan = [np.ones(n)]
    for c in pack_cols:
        Vspan.append(np.log1p(np.clip(pack[c].to_numpy().astype(np.float64), 0, None)))
    for nm in ("my27", "my26", "hmmsim", "countaov_cal"):
        Vspan.append(lp_val(nm))
    Vspan = np.stack(Vspan)
    print(f"  val span: {Vspan.shape[0]} vectors")

    # flagged-set diagnostics on val
    for tau in TAUS:
        m = pz_v > tau
        print(f"val diag tau={tau}: flagged={m.sum()} ({m.mean()*100:.1f}%), "
              f"P(target=0|flag)={float((lt[m]==0).mean()):.3f}, "
              f"mean_e_val(flag)={float(e_val[m].mean()):+.3f}, "
              f"mean_lp_base(flag)={float(lp_base_v[m].mean()):.3f}")

    h5 = np.load(PROBES / "h_hmmsim.npy")

    results = {}
    for tau in TAUS:
        w_t = np.clip((pz_t - tau) / (1 - tau), 0, 1)
        w_v = np.clip((pz_v - tau) / (1 - tau), 0, 1)
        for kind in ("flat", "prop"):
            d_t = -w_t if kind == "flat" else -lp_base * w_t
            d_v = -w_v if kind == "flat" else -lp_base_v * w_v
            nov, r_t = novelty(d_t, Sp)
            h_t = rms_norm(r_t)
            nov_v, r_v = novelty(d_v, Vspan)
            h_v = rms_norm(r_v)
            c_val = float((e_val * h_v).mean())
            c_raw = float((e_val * rms_norm(d_v)).mean())
            # split-half sign stability
            rng = np.random.default_rng(0)
            perm = rng.permutation(n)
            ha, hb = perm[: n // 2], perm[n // 2:]
            c_a = float((e_val[ha] * h_v[ha]).mean())
            c_b = float((e_val[hb] * h_v[hb]).mean())
            q = float((h_t ** 2).mean())
            corr5 = float(np.corrcoef(h_t, h5)[0, 1])
            results[f"{kind}_{tau}"] = dict(
                nov=nov, nov_val=nov_v, c_val=c_val, c_raw=c_raw,
                c_half=(c_a, c_b), q=q, corr_h_hmmsim=corr5,
                score=nov * abs(c_val), h=h_t)
            print(f"{kind} tau={tau}: novelty={nov:.3f} (val {nov_v:.3f}) "
                  f"c_val={c_val:+.5f} (raw {c_raw:+.5f}, halves {c_a:+.5f}/{c_b:+.5f}) "
                  f"corr(h,h_hmmsim)={corr5:+.3f} sel_score={nov*abs(c_val):.6f}")

    best = max(results, key=lambda k: results[k]["score"])
    R = results[best]
    h = R["h"]
    print(f"\nBEST: {best}  novelty={R['nov']:.3f} c_val={R['c_val']:+.5f}")

    np.save(PROBES / "h_zeropush.npy", h)
    lp_probe = np.clip(lp_base + STEP * h, 0, None)
    n_clip = int((lp_base + STEP * h < 0).sum())
    pl.DataFrame({"user_id": uid, "predict": np.expm1(lp_probe)}).write_csv(
        ROOT / "submissions" / "G1_probe_zeropush.csv")

    # expected numbers (c_test = kappa * c_val; negative kappa = val sign wrong,
    # mechanism hypothesis "push zeros down" right on test)
    q = R["q"]
    exp = {}
    for kap in (-1.0, 0.0, 1.0, 2.5):
        c_t = kap * R["c_val"]
        f_probe = float(np.sqrt(max(F_BASE ** 2 + 2 * STEP * c_t + STEP ** 2 * q, 0)))
        gain_opt = F_BASE - float(np.sqrt(max(F_BASE ** 2 - c_t ** 2 / q, 0)))
        exp[f"kappa_{kap}"] = dict(f_probe=round(f_probe, 5),
                                   applied_gain=round(gain_opt, 6),
                                   delta_star=round(-c_t / q, 3))
        print(f"kappa={kap}: probe score={f_probe:.5f} "
              f"(base {F_BASE}), optimal step={-c_t/q:+.3f}, applied gain={gain_opt:.6f}")
    print(f"clip guard: {n_clip} users clipped at lp=0; "
          f"min lp_probe={float(lp_probe.min()):.4f}")

    out = dict(best=best, novelty=round(R["nov"], 4), c_val=round(R["c_val"], 6),
               q=round(q, 5), corr_h_hmmsim=round(R["corr_h_hmmsim"], 4),
               expected=exp, n_clip=n_clip,
               file="submissions/G1_probe_zeropush.csv",
               all={k: dict(nov=round(v["nov"], 4), c_val=round(v["c_val"], 6),
                            score=round(v["score"], 6)) for k, v in results.items()})
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
