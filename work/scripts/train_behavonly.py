"""behavonly: the champion GBDT trained without a single feature that carries money,
and without the volume counts that arithmetically stand in for money.

Rationale. Every model in the zoo sees GMV sums and their transforms, so they all
inherit the same view of "how much this user spends" and repeat each other's
mistakes on users whose spend level is misleading (a one-off big purchase, a gift,
a seasonal spike). A model that has never seen a rouble is structurally unable to
repeat those mistakes -- which is the point (FOR_TEAM_what_we_have.md: value = low
error correlation, not low RMSLE).

Two exclusion layers, both RULES rather than frozen lists, so a newly built gmv_*
feature is excluded automatically instead of leaking in.

1. MONEY (money_cols)
   * any column whose name contains "gmv" -- sums, logs, per-order/per-day stats,
     search/cat split, concentration, decays, year-ago bands, cross-sectional
     ranks, recency of the last paid day, weekend GMV share;
   * btyd_exp_monetary / btyd_exp_ltv30 -- the BTYD money head;
   * hv{50,200,1000}_days_* -- counts of days with gmv above a rouble threshold;
   * gift_spike_flag -- thresholded gmv_ya_t3 against median daily gmv;
   * search_share_trend -- difference of two GMV shares;
   * seasonal_index -- global GMV level of the target window.

2. VOLUME PROXIES (volume_cols, disable with --with-counts)
   Money is count x average check, so raw event counts are the closest arithmetic
   stand-in for the signal layer 1 removed: ord_cnt_*, cart_cnt_*, searches_*,
   s2o_cnt_90, c2o_cnt_90, cart_minus_ord_30, the exponential-decay volume sums
   dec_*, the cross-sectional ranks rk_*, and the last-active-day snapshot
   last_day_*. Measured on a 2-anchor smoke: keeping them gives RMSLE 1.7086 at
   error correlation 0.9869; dropping them gives 1.7165 at 0.9825 -- 0.008 RMSLE
   for 0.004 correlation, which is the trade this project exists to make.

What survives (85 features): day-level presence windows (ord_days, active_days,
cart_days, search_days, cat_days, surface-transition days, both_days), recency of
each event type, inter-order and inter-activity gaps and their spread, tenure and
history length, activity/search trends and conversion rates, year-ago ACTIVITY
windows and their coverage, weekend order share, burstiness, and the BTYD
frequency head (p_alive, exp_purch30, freq, recency, T).

Config, protocol, retrain and saving are delegated to train_gbdt.py, so this stays
in sync with the champion by construction. Default config = champion
tweedie-on-log, --gap-days 30.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, feature_cols, load_anchor

MONEY_EXTRA = {"btyd_exp_monetary", "btyd_exp_ltv30", "gift_spike_flag",
               "search_share_trend", "seasonal_index"}
VOLUME_EXTRA = {"s2o_cnt_90", "c2o_cnt_90", "cart_minus_ord_30"}
CHAMPION = ["--model", "lgb", "--objective", "log_mse",
            "--params", '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}']


def money_cols(cols: list[str]) -> set[str]:
    out = {c for c in cols if "gmv" in c.lower()}
    out |= {c for c in cols if re.match(r"^hv\d+_days_", c)}
    return out | (MONEY_EXTRA & set(cols))


def volume_cols(cols: list[str]) -> set[str]:
    out = {c for c in cols if re.match(r"^(ord_cnt|cart_cnt|searches)_", c)}
    out |= {c for c in cols if c.startswith(("dec_", "rk_", "last_day_"))}
    return out | (VOLUME_EXTRA & set(cols))


def main():
    argv = sys.argv[1:]
    with_counts = "--with-counts" in argv
    argv = [a for a in argv if a != "--with-counts"]

    cols = feature_cols(load_anchor(VAL_ANCHOR))
    drop = money_cols(cols)
    if not with_counts:
        drop |= volume_cols(cols)
    drop = sorted(drop)
    keep = [c for c in cols if c not in set(drop)]
    print(f"behavonly: dropping {len(drop)} money/volume features, keeping {len(keep)}"
          f"{' (--with-counts)' if with_counts else ''}", flush=True)
    print("kept:", ",".join(keep), flush=True)

    if "--name" not in argv:
        argv = ["--name", "behavonly"] + argv
    if "--gap-days" not in argv:
        argv += ["--gap-days", "30"]
    if not any(a in argv for a in ("--model", "--objective", "--params")):
        argv += CHAMPION
    argv += ["--drop-cols", ",".join(drop)]
    if "--notes" not in argv:
        argv += ["--notes", f"behavonly: champion tweedie-on-log on {len(keep)} money-free "
                            f"{'(counts kept)' if with_counts else 'count-free'} behaviour feats, "
                            f"{len(drop)} cols dropped by rule"]

    import train_gbdt
    sys.argv = ["train_gbdt.py"] + argv
    train_gbdt.main()


if __name__ == "__main__":
    main()
