"""Feature builder at any anchor day on a cube's 7-day grid.
X = horizon sums from cube + day-level recency/interval features.
All features use ONLY data with day < anchor_day.

LOCAL ADAPTATION (delta vs ../scripts/features.py): /root/work -> <repo>/work_kostya/work.
No other changes.
"""
import polars as pl, numpy as np
from datetime import date
from pathlib import Path

_W = str(Path(__file__).resolve().parents[1] / "work")  # was /root/work

DAY0 = date(2025, 1, 1)
METRICS = ["active", "searches", "to_cart", "to_ord", "gmv", "buyday",
           "s_day", "c_day", "cartday", "s2c"]
HORIZONS_W = [1, 2, 4, 8, 13, 26, 52]  # weeks; plus ALL

_act_cache = {}

def _load_day_arrays():
    if _act_cache:
        return _act_cache
    users = pl.read_parquet(f"{_W}/users_order.parquet")["user_id"]
    uid_to_idx = pl.DataFrame({"user_id": users, "uix": np.arange(len(users), dtype=np.int32)})
    act = pl.read_parquet(f"{_W}/act.parquet", columns=["user_id", "event_date", "to_ord", "to_cart", "gmv"])
    act = act.join(uid_to_idx, on="user_id")
    day = act.select((pl.col("event_date") - pl.lit(DAY0)).dt.total_days())["event_date"].to_numpy().astype(np.int16)
    uix = act["uix"].to_numpy().astype(np.int32)
    to_ord = act["to_ord"].to_numpy()
    to_cart = act["to_cart"].to_numpy()
    gmv = act["gmv"].to_numpy()
    _act_cache.update(dict(N=len(users), day=day, uix=uix,
                           is_buy=to_ord > 0, is_cart=to_cart > 0, gmv=gmv))
    return _act_cache

def _last_day_before(anchor, mask_extra=None):
    c = _load_day_arrays()
    m = c["day"] < anchor
    if mask_extra is not None:
        m = m & mask_extra
    out = np.full(c["N"], -10000, dtype=np.int32)
    np.maximum.at(out, c["uix"][m], c["day"][m].astype(np.int32))
    return out

def _first_day_before(anchor):
    c = _load_day_arrays()
    m = c["day"] < anchor
    out = np.full(c["N"], 10000, dtype=np.int32)
    np.minimum.at(out, c["uix"][m], c["day"][m].astype(np.int32))
    return out

def interval_feats(anchor):
    """Inter-buy-day interval stats from full history < anchor."""
    c = _load_day_arrays()
    m = (c["day"] < anchor) & c["is_buy"]
    uix = c["uix"][m]; day = c["day"][m].astype(np.int32); gmv = c["gmv"][m]
    order = np.argsort(uix, kind="stable")  # day already sorted within user
    uix, day, gmv = uix[order], day[order], gmv[order]
    N = c["N"]
    nbd = np.bincount(uix, minlength=N).astype(np.float32)
    same = np.empty(len(uix), dtype=bool)
    same[0] = False
    same[1:] = uix[1:] == uix[:-1]
    gaps = np.where(same, day - np.concatenate(([0], day[:-1])), 0).astype(np.float32)
    gsum = np.zeros(N, np.float64); gsq = np.zeros(N, np.float64); gmax = np.zeros(N, np.float32)
    np.add.at(gsum, uix[same], gaps[same])
    np.add.at(gsq, uix[same], gaps[same] ** 2)
    np.maximum.at(gmax, uix[same], gaps[same])
    ngap = np.maximum(nbd - 1, 0)
    mean_gap = np.where(ngap > 0, gsum / np.maximum(ngap, 1), 9999).astype(np.float32)
    var = np.where(ngap > 1, gsq / np.maximum(ngap, 1) - (gsum / np.maximum(ngap, 1)) ** 2, 0)
    std_gap = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    # last buy gmv and mean log-gmv per buyday
    last_gmv = np.zeros(N, np.float32)
    # last occurrence per user = last in order
    lastpos = np.zeros(N, dtype=np.int64)
    lastpos[uix] = np.arange(len(uix))  # later rows overwrite
    has = nbd > 0
    last_gmv[has] = gmv[lastpos[has.nonzero()[0]]] if has.any() else 0
    lg = np.log1p(gmv)
    lgsum = np.zeros(N, np.float64); np.add.at(lgsum, uix, lg)
    mean_lg = np.where(nbd > 0, lgsum / np.maximum(nbd, 1), 0).astype(np.float32)
    return dict(mean_gap=mean_gap, std_gap=std_gap, max_gap=gmax,
                last_gmv=np.log1p(last_gmv), mean_log_gmv=mean_lg)

def build_features(anchor_day, cube, boundary_day):
    """cube: (N, NW, M) memmap; anchor must satisfy (boundary_day - anchor_day) % 7 == 0"""
    NW = cube.shape[1]
    assert (boundary_day - anchor_day) % 7 == 0
    wa = NW - (boundary_day - anchor_day) // 7  # weeks [0, wa) are history
    assert 0 < wa <= NW
    feats = {}
    hist = cube[:, :wa, :]
    for h in HORIZONS_W:
        hh = min(h, wa)
        s = hist[:, wa - hh:, :].sum(axis=1)
        for mi, m in enumerate(METRICS):
            feats[f"{m}_w{h}"] = s[:, mi]
    s_all = hist.sum(axis=1)
    for mi, m in enumerate(METRICS):
        feats[f"{m}_all"] = s_all[:, mi]
    # recencies
    la = _last_day_before(anchor_day)
    c = _load_day_arrays()
    lb = _last_day_before(anchor_day, c["is_buy"])
    lc = _last_day_before(anchor_day, c["is_cart"])
    fa = _first_day_before(anchor_day)
    feats["rec_active"] = np.clip(anchor_day - la, 0, 5000).astype(np.float32)
    feats["rec_buy"] = np.clip(anchor_day - lb, 0, 5000).astype(np.float32)
    feats["rec_cart"] = np.clip(anchor_day - lc, 0, 5000).astype(np.float32)
    feats["tenure"] = np.clip(anchor_day - fa, 0, 5000).astype(np.float32)
    iv = interval_feats(anchor_day)
    for k, v in iv.items():
        feats[k] = v
    feats["rec_over_gap"] = (feats["rec_buy"] / np.maximum(iv["mean_gap"], 1)).astype(np.float32)

    # --- derived block (all from already-computed sums; cheap) ---
    eps = 1e-6
    def F(n): return feats[n]
    # size / intensity
    feats["gmv_per_buyday_all"] = (F("gmv_all") / np.maximum(F("buyday_all"), 1)).astype(np.float32)
    feats["gmv_per_ord_all"] = (F("gmv_all") / np.maximum(F("to_ord_all"), 1)).astype(np.float32)
    feats["gmv_per_buyday_w13"] = (F("gmv_w13") / np.maximum(F("buyday_w13"), 1)).astype(np.float32)
    feats["searches_per_active_w13"] = (F("searches_w13") / np.maximum(F("active_w13"), 1)).astype(np.float32)
    feats["cart_per_active_w13"] = (F("to_cart_w13") / np.maximum(F("active_w13"), 1)).astype(np.float32)
    # conversion
    for h in ["w8", "w26", "all"]:
        feats[f"ord_per_cart_{h}"] = (F(f"to_ord_{h}") / np.maximum(F(f"to_cart_{h}"), 1)).astype(np.float32)
        feats[f"buyday_per_active_{h}"] = (F(f"buyday_{h}") / np.maximum(F(f"active_{h}"), 1)).astype(np.float32)
    # channel mix
    feats["search_share_days_w13"] = (F("s_day_w13") / np.maximum(F("active_w13"), 1)).astype(np.float32)
    feats["s2c_share_w13"] = (F("s2c_w13") / np.maximum(F("to_cart_w13"), 1)).astype(np.float32)
    # trends
    for m in ["active", "searches", "to_cart", "buyday", "gmv"]:
        feats[f"{m}_tr_4_13"] = (F(f"{m}_w4") / np.maximum(F(f"{m}_w13"), eps)).astype(np.float32)
        feats[f"{m}_tr_13_52"] = (F(f"{m}_w13") / np.maximum(F(f"{m}_w52"), eps)).astype(np.float32)
    # weekly regularity from cube: weeks active / max zero-run over last 26w
    wa26 = min(26, wa)
    aw = hist[:, wa - wa26:, 0] > 0  # active weeks bool
    feats["weeks_active_26"] = aw.sum(1).astype(np.float32)
    feats["weeks_active_frac_26"] = (aw.sum(1) / wa26).astype(np.float32)
    bw = hist[:, wa - wa26:, 5] > 0  # buy weeks
    feats["buy_weeks_26"] = bw.sum(1).astype(np.float32)
    # longest inactive run (weeks) in last 26
    run = np.zeros(aw.shape[0], np.float32); best = np.zeros(aw.shape[0], np.float32)
    for wcol in range(aw.shape[1]):
        run = np.where(aw[:, wcol], 0, run + 1)
        best = np.maximum(best, run)
    feats["max_inactive_run_26w"] = best
    feats["cur_inactive_run_w"] = run
    # year-ago same-season (4 weeks centered 52w back), if history long enough
    if wa >= 52:
        ya = hist[:, wa - 54:wa - 50, :] if wa >= 54 else hist[:, 0:wa - 50, :]
        feats["ya_active_4w"] = ya[:, :, 0].sum(1)
        feats["ya_buyday_4w"] = ya[:, :, 5].sum(1)
        feats["ya_gmv_4w"] = ya[:, :, 4].sum(1)
    else:
        for nm in ["ya_active_4w", "ya_buyday_4w", "ya_gmv_4w"]:
            feats[nm] = np.zeros(cube.shape[0], np.float32)
    for k, v in extra_feats(anchor_day).items():
        feats[k] = v
    names = list(feats.keys())
    X = np.stack([feats[n] for n in names], axis=1).astype(np.float32)
    return X, names

def extra_feats(anchor_day):
    """Day-of-week profile + order-size quantiles, full history < anchor."""
    import polars as pl
    c = _load_day_arrays()
    N = c["N"]
    m = c["day"] < anchor_day
    d = c["day"][m].astype(np.int32); u = c["uix"][m]
    wknd = ((d % 7 == 3) | (d % 7 == 4))
    tot = np.bincount(u, minlength=N).astype(np.float32)
    wk = np.bincount(u[wknd], minlength=N).astype(np.float32)
    out = {"wknd_share": (wk / np.maximum(tot, 1)).astype(np.float32)}
    mb = m & c["is_buy"]
    ub = c["uix"][mb]; lg = np.log1p(c["gmv"][mb]).astype(np.float32)
    df = pl.DataFrame({"u": ub, "lg": lg})
    q = df.group_by("u").agg(pl.col("lg").quantile(0.25).alias("q25"),
                             pl.col("lg").quantile(0.75).alias("q75"),
                             pl.col("lg").std().alias("sd"))
    q25 = np.zeros(N, np.float32); q75 = np.zeros(N, np.float32); sd = np.zeros(N, np.float32)
    uu = q["u"].to_numpy()
    q25[uu] = q["q25"].to_numpy(); q75[uu] = q["q75"].to_numpy()
    sdv = q["sd"].fill_null(0.0).to_numpy(); sd[uu] = sdv
    out["ordsize_q25"] = q25; out["ordsize_q75"] = q75; out["ordsize_sd"] = sd
    return out
