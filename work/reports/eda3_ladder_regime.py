# eda3 activity ladder: non-stationarity of the ladder near anchors (val month0 = holiday window, test month0 = calm)
import numpy as np
import datetime as dt

def rownorm(T):
    s = T.sum(1, keepdims=True).astype(float); s[s == 0] = 1
    return T / s

def tmat(a, b):
    T = np.zeros((11, 11), dtype=np.int64); np.add.at(T, (a, b), 1); return T

for anc_name, anc in [("val", dt.date(2026, 1, 14)), ("test", dt.date(2026, 2, 13))]:
    L = np.load(f"work/reports/eda3_ladder_{anc_name}_Ls.npy").astype(int)
    pairs = [tmat(L[:, k + 1], L[:, k]) for k in range(12)]
    calm = rownorm(sum(pairs[4:11]))
    last = rownorm(pairs[0])
    d = np.abs(last - calm).sum(1)
    w0 = (anc - dt.timedelta(days=27), anc)
    print(f"{anc_name}: month0 window {w0[0]}..{w0[1]}; -dist(last pair vs calm) per from-level:", np.round(d, 3))
    print(f"  weighted mean -dist = {np.average(d, weights=pairs[0].sum(1)):.4f}")

L = np.load("work/reports/eda3_ladder_val_Ls.npy").astype(int)
tot, wsum = 0.0, 0
for l1 in range(11):
    m1 = L[:, 1] == l1
    if m1.sum() < 500:
        continue
    p1 = np.bincount(L[m1, 0], minlength=11) / m1.sum()
    for l2 in range(11):
        m = m1 & (L[:, 2] == l2)
        if m.sum() < 200:
            continue
        p2 = np.bincount(L[m, 0], minlength=11) / m.sum()
        tot += m.sum() * np.abs(p2 - p1).sum(); wsum += m.sum()

# how much does month-0 level differ in meaning: share inactive (L=0) per month, both anchors
for anc_name in ["val", "test"]:
    A = np.load(f"work/reports/eda3_ladder_{anc_name}_Ls.npy").astype(int)
    print(anc_name, "share L=0 per month k=0..12:", np.round((A == 0).mean(0), 4))
