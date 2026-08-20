import numpy as np
from sklearn.isotonic import IsotonicRegression

def rmsle_from_log(pred_z, z):
    return float(np.sqrt(np.mean((np.maximum(pred_z, 0) - z) ** 2)))

def cal_bins(pred, z, bins=24):
    """Team-style: equal-count bins on pred, bin value = mean z, monotone-enforced.
    Returns calibrated in-sample values and a transfer function for new preds."""
    order = np.argsort(pred, kind="stable")
    n = len(pred)
    vals, cuts = [], []
    for i in range(bins):
        lo, hi = int(n * i / bins), int(n * (i + 1) / bins)
        idx = order[lo:hi]
        vals.append(float(z[idx].mean()))
        cuts.append(float(pred[idx[-1]]))
    vals = np.maximum.accumulate(np.array(vals))
    cuts = np.array(cuts)
    out = np.empty(n, dtype=np.float64)
    for i in range(bins):
        lo, hi = int(n * i / bins), int(n * (i + 1) / bins)
        out[order[lo:hi]] = vals[i]
    def transfer(pnew):
        j = np.searchsorted(cuts[:-1], pnew, side="left")
        return vals[j]
    return out, transfer

def iso_crossfit(pred, z, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.random(len(z)) < 0.5
    out = np.empty_like(z, dtype=np.float64)
    for f in [fold, ~fold]:
        iso = IsotonicRegression(y_min=0, out_of_bounds="clip").fit(pred[~f], z[~f])
        out[f] = iso.predict(pred[f])
    return out

def iso_full(pred, z):
    iso = IsotonicRegression(y_min=0, out_of_bounds="clip").fit(pred, z)
    return iso.predict(pred), iso

def boot_delta(zA, zB, z, iters=300, seed=0):
    """bootstrap SE of RMSLE(A)-RMSLE(B) over users."""
    rng = np.random.default_rng(seed)
    n = len(z)
    eA = (np.maximum(zA, 0) - z) ** 2
    eB = (np.maximum(zB, 0) - z) ** 2
    d = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        d.append(np.sqrt(eA[idx].mean()) - np.sqrt(eB[idx].mean()))
    d = np.array(d)
    return float(np.mean(d)), float(np.std(d))
