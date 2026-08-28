"""Team acceptance: honest calibration + exact pair algebra vs the pack blend.
Formulas verbatim from work/scripts/margin.py (origin/sasha)."""
import sys
import numpy as np
import polars as pl

PACK = "/root/rw/val_preds.parquet"


def fit_shifts(lp, ly, bins=24):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    centers, shifts = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500:
            continue
        centers.append(lp[m].mean()); shifts.append(ly[m].mean() - lp[m].mean())
    return np.array(centers), np.array(shifts)


def calibrate_honest(lp, ly, bins=24, seed=0):
    rng = np.random.default_rng(seed + 100_003)
    half = rng.permutation(len(ly)) < len(ly) // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], ly[m], bins)
        out[~m] = np.clip(lp[~m] + np.interp(lp[~m], c, s), 0, None)
    return out


def pair_contribution(sb, sm, margin):
    z = max(float(margin), 0.0)
    den = (sm * sm - sb * sb + 2.0 * sb * sm * z) * 2.0 * sb
    return (sb * sb * sm * sm * z * z) / den if den > 1e-12 else 0.0


def main(paths):
    pack = pl.read_parquet(PACK).sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    eb = lb - ly
    sb = float(np.sqrt(np.mean(eb ** 2)))
    print(f"эталон: бленд пака {sb:.6f}")
    for p in paths:
        df = pl.read_parquet(p).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid)
        lp_raw = np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None))
        for tag, lp in (("raw", lp_raw), ("cal", calibrate_honest(lp_raw, ly))):
            sm = float(np.sqrt(np.mean((lp - ly) ** 2)))
            e = lp - ly
            rho = float(np.mean(e * eb) / (sm * sb))
            margin = sb / sm - rho
            contrib = pair_contribution(sb, sm, margin)
            v = ("ГОДИТСЯ" if contrib >= 0.0003 else
                 "слабо, но не шум" if contrib >= 0.000044 else "шум")
            print(f"{p:<38} {tag:<4} скор {sm:.6f} корр {rho:.5f} "
                  f"ЗАПАС {margin:+.5f} вклад {contrib:.6f}  {v}")


if __name__ == "__main__":
    main(sys.argv[1:])
