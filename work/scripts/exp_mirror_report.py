"""Aggregate work/reports/mirror_window_results.jsonl -> ranking tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR  # noqa: E402

MARCH = ["2025-02-06", "2025-02-13", "2025-02-20"]
NORMAL = ["2025-11-12", "2025-12-31", "2026-01-14"]
MAIN_ROUNDS = "1000"
POP = "act42"


def load():
    rows = {}
    with open(REPORTS_DIR / "mirror_window_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["config"]] = r
    return rows


def get(r, anchor, rounds, pop, key):
    return r["scores"][anchor][rounds][pop][key]


def fam_mean(r, fam, rounds, pop, key):
    return float(np.mean([get(r, a, rounds, pop, key) for a in fam]))


def ranks(vals: dict) -> dict:
    order = sorted(vals, key=lambda k: vals[k])
    return {k: i + 1 for i, k in enumerate(order)}


def spearman(a: dict, b: dict) -> float:
    ks = sorted(a)
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    x -= x.mean(); y -= y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def main():
    rows = load()
    cfgs = [c for c in rows if not c.endswith("_s1337")]
    pop = sys.argv[1] if len(sys.argv) > 1 else POP
    rounds = sys.argv[2] if len(sys.argv) > 2 else MAIN_ROUNDS

    print(f"=== population={pop} rounds={rounds} ===\n")
    hdr = f"{'config':22s}" + "".join(f"{a[5:]:>9s}" for a in MARCH + NORMAL) + f"{'MARmean':>9s}{'NORmean':>9s}"
    print(hdr)
    for c in cfgs:
        r = rows[c]
        line = f"{c:22s}" + "".join(f"{get(r,a,rounds,pop,'rmsle'):9.4f}" for a in MARCH + NORMAL)
        line += f"{fam_mean(r,MARCH,rounds,pop,'rmsle'):9.4f}{fam_mean(r,NORMAL,rounds,pop,'rmsle'):9.4f}"
        print(line)

    m = {c: fam_mean(rows[c], MARCH, rounds, pop, "rmsle") for c in cfgs}
    n = {c: fam_mean(rows[c], NORMAL, rounds, pop, "rmsle") for c in cfgs}
    mirror = {c: get(rows[c], "2025-02-13", rounds, pop, "rmsle") for c in cfgs}
    val = {c: get(rows[c], "2026-01-14", rounds, pop, "rmsle") for c in cfgs}
    rm, rn, rmir, rval = ranks(m), ranks(n), ranks(mirror), ranks(val)
    print(f"\n{'config':22s}{'rkMAR':>7s}{'rkNOR':>7s}{'rkMIRROR':>10s}{'rkVAL2601':>10s}")
    for c in cfgs:
        print(f"{c:22s}{rm[c]:7d}{rn[c]:7d}{rmir[c]:10d}{rval[c]:10d}")
    print(f"\nspearman(MARmean, NORmean)      = {spearman(rm, rn):+.3f}")
    print(f"spearman(mirror0213, val2601)   = {spearman(rmir, rval):+.3f}")
    print(f"MARCH winner  : {min(m, key=m.get)}   (mirror-only: {min(mirror, key=mirror.get)})")
    print(f"NORMAL winner : {min(n, key=n.get)}   (val2026-01-14 only: {min(val, key=val.get)})")

    # deltas vs baseline
    base = "base_tw13_n8"
    print(f"\n--- delta rmsle vs {base} (negative = better) ---")
    print(f"{'config':22s}{'dMAR':>9s}{'dNOR':>9s}{'dMIRROR':>10s}{'dVAL':>9s}")
    for c in cfgs:
        print(f"{c:22s}{m[c]-m[base]:+9.4f}{n[c]-n[base]:+9.4f}"
              f"{mirror[c]-mirror[base]:+10.4f}{val[c]-val[base]:+9.4f}")
    if base + "_s1337" in rows:
        s = rows[base + "_s1337"]
        b = rows[base]
        print("\nseed noise (seed1337 - seed42):")
        for a in MARCH + NORMAL:
            print(f"  {a}: {get(s,a,rounds,pop,'rmsle')-get(b,a,rounds,pop,'rmsle'):+.4f}")
        print(f"  MARmean {fam_mean(s,MARCH,rounds,pop,'rmsle')-m[base]:+.4f}  "
              f"NORmean {fam_mean(s,NORMAL,rounds,pop,'rmsle')-n[base]:+.4f}")

    # bias / shape decomposition
    print(f"\n--- bias (mean log-residual; >0 = model UNDER-predicts) / shape (std) ---")
    print(f"{'config':22s}" + "".join(f"{a[5:]+'_b':>11s}" for a in MARCH + NORMAL))
    for c in cfgs:
        print(f"{c:22s}" + "".join(f"{get(rows[c],a,rounds,pop,'bias'):11.4f}" for a in MARCH + NORMAL))
    print(f"{'config':22s}" + "".join(f"{a[5:]+'_s':>11s}" for a in MARCH + NORMAL))
    for c in cfgs:
        print(f"{c:22s}" + "".join(f"{get(rows[c],a,rounds,pop,'shape'):11.4f}" for a in MARCH + NORMAL))

    # ranking on SHAPE only (level-calibrated)
    ms = {c: fam_mean(rows[c], MARCH, rounds, pop, "shape") for c in cfgs}
    ns = {c: fam_mean(rows[c], NORMAL, rounds, pop, "shape") for c in cfgs}
    rms, rns = ranks(ms), ranks(ns)
    print(f"\n--- level-calibrated (shape) ranking ---")
    print(f"{'config':22s}{'MARshape':>10s}{'NORshape':>10s}{'rkMAR':>7s}{'rkNOR':>7s}")
    for c in cfgs:
        print(f"{c:22s}{ms[c]:10.4f}{ns[c]:10.4f}{rms[c]:7d}{rns[c]:7d}")
    print(f"spearman(shape MAR, NOR) = {spearman(rms, rns):+.3f}")
    print(f"MARCH shape winner={min(ms,key=ms.get)}  NORMAL shape winner={min(ns,key=ns.get)}")

    # round sensitivity
    print(f"\n--- rmsle by rounds ({pop}) : MARmean / NORmean ---")
    grid = sorted(rows[cfgs[0]]["scores"][MARCH[0]], key=int)
    print(f"{'config':22s}" + "".join(f"{k:>16s}" for k in grid))
    for c in cfgs:
        cells = []
        for k in grid:
            cells.append(f"{fam_mean(rows[c],MARCH,k,pop,'rmsle'):7.4f}/"
                         f"{fam_mean(rows[c],NORMAL,k,pop,'rmsle'):.4f}")
        print(f"{c:22s}" + "".join(f"{x:>16s}" for x in cells))
    # best-over-rounds ranking (robustness to the fixed-round choice)
    mb = {c: min(fam_mean(rows[c], MARCH, k, pop, "rmsle") for k in grid) for c in cfgs}
    nb = {c: min(fam_mean(rows[c], NORMAL, k, pop, "rmsle") for k in grid) for c in cfgs}
    print(f"best-over-rounds  MARCH winner={min(mb,key=mb.get)}  NORMAL winner={min(nb,key=nb.get)}"
          f"  spearman={spearman(ranks(mb),ranks(nb)):+.3f}")


if __name__ == "__main__":
    main()
