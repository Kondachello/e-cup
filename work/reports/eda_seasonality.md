# Seasonality & Trend Research (Ozon E-CUP LTV)

Scripts: `work/scripts/seas_01_aggregate.py`, `work/scripts/seas_02_analysis.py`.
Data: one row per active user-day confirmed (rows == distinct users every day).

## 1. Global daily series 2025-01-01 .. 2026-02-13

Monthly view (gmv/day is the level indicator; AOV = gmv/orders):

| month | gmv (M) | orders (K) | avg DAU (K) | gmv/day (K) | AOV |
|---|---|---|---|---|---|
| 2025-01 | 14.56 | 402 | 57.1 | 469.6 | 36.2 |
| 2025-02 | 14.48 | 417 | 61.8 | 517.3 | 34.8 |
| 2025-03 | 17.42 | 489 | 64.2 | 561.9 | 35.6 |
| 2025-04 | 17.01 | 475 | 64.9 | 566.9 | 35.8 |
| 2025-05 | 17.76 | 510 | 65.4 | 572.8 | 34.8 |
| 2025-06 | 18.72 | 535 | 67.1 | 623.9 | 35.0 |
| 2025-07 | 20.44 | 591 | 69.2 | 659.4 | 34.6 |
| 2025-08 | 22.57 | 672 | 73.8 | 728.1 | 33.6 |
| 2025-09 | 21.85 | 643 | 78.1 | 728.4 | 34.0 |
| 2025-10 | 23.54 | 715 | 84.5 | 759.2 | 32.9 |
| 2025-11 | 24.00 | 739 | 92.5 | 799.9 | 32.5 |
| 2025-12 | 28.81 | 851 | 95.8 | 929.5 | 33.9 |
| 2026-01 | 21.80 | 704 | 90.2 | 703.2 | 30.9 |
| 2026-02 (1-13) | 9.15 | 308 | 93.9 | 703.8 | 29.7 |

Shape: strong secular growth all through 2025 (gmv/day x2.0 from Jan-25 trough to Dec-25 peak; DAU 57K -> 96K), December peak, sharp New-Year slump, and a **flat plateau in Jan-Feb 2026 (~700-704K/day)**. Orders grow faster than gmv (AOV drifts down 36 -> 30, -15% YoY).

Calendar events (daily detail):
- **Feb-Mar 2025 / March 8**: gradual ramp through late Feb, elevated Mar 1-6 (peak Mar 3 at 656K, ~+15% vs mid-Feb), then a **dip on Mar 7-8 itself** (518/511K, DAU drops 64K->54K), recovery Mar 9-10. So the gifting effect is a pre-holiday build-up, not a spike on the day. Feb 23 shows no lift (mild dip). The whole Feb14-Mar15 window sits well above Jan15-Feb13 (per-day: 469 -> 493 -> 538 -> 577K across half-month blocks).
- **May holidays 2025**: dips on the holiday days (May 9 = 431K, the lowest non-January day; May 1-3 soft), no pre-spike. Mild plateau overall.
- **November 2025**: clear **11.11 sale spike** (937K on Nov 11 vs ~800K neighbors, +15%), elevated Nov 15-16; **Black Friday week (Nov 24-28) shows no spike** (actually soft, 750-790K). December ramp starts Dec 1.
- **New Year**: top-15 gmv days are ALL Dec 13-28, 2025 (0.98-1.09M/day). Collapse Dec 29-31 (527K on Dec 31), Jan 1 low (554K), recovery to ~720-740K by Jan 5. January slump: Jan-26 is -24% vs Dec-25 per-day; late January is the annual trough in both years.

## 2. Key window ratios (2025 analog of val -> test windows)

Windows (both 30 days): W1 = Jan 15 - Feb 13 (val-like), mdl_onyx = Feb 14 - Mar 15 (test-like, incl. Feb 23 + Mar 8).

|---|---|---|---|
| gmv | 14.39M | 16.73M | **R_season = 1.163** |
| orders | 406.5K | 474.1K | 1.166 |
| distinct active users | 190.8K | 195.7K | 1.026 |
| gmv per active user | 75.4 | 85.5 | 1.134 |

For reference, actual val window (2026-01-15..02-13): gmv 21.01M (700K/day), orders 700K.

## 3. YoY growth and deceleration

| calendar span | gmv YoY | orders YoY | DAU YoY |
|---|---|---|---|
| Jan 1-14 | 1.510 | 1.742 | 1.566 |
| Jan 15-31 | 1.487 | 1.756 | 1.588 |
| **Jan full** | **1.497** | 1.750 | 1.579 |
| Feb 1-13 | 1.427 | 1.681 | 1.552 |

(No Dec/Dec possible - data starts 2025-01-01.) GMV YoY is *declining* through the observable stretch (1.51 -> 1.49 -> 1.43): 2026 growth has stalled. Within-2026 momentum confirms it: Jan15-31 -> Feb1-13 per-day is +0.9% in 2026 vs +5.1% in 2025.

## 4. Global multiplier M for 2026 (test window level / val window level)

- Pure seasonal carry (assume 2026 window-over-window dynamics = 2025's): M = R_season = **1.163** (upper bound).
- YoY-drift adjusted: fitting log-YoY over the three observable spans gives a drift of x0.946 per 30.5 days; extrapolated YoY for the test window = 1.381. M = R_season x drift = 16.73M x 1.381 / 21.01M = **1.100**.
- Cross-check: 2025 underlying trend was ~+5-6%/month in Feb-Mar; removing it from 1.163 with 2026 trend ~0 gives 1.09-1.11. Converges.

**Point estimate M = 1.10, plausible range [1.06, 1.16].** The seasonal (gifting) component ~+10% should recur since it is calendar-driven (pre-Mar-8 build-up was visible and strong in 2025); the risk on the low side is continued deceleration, on the high side a normal promo calendar restoring some ramp.

## 5. Weekday pattern

gmv relative to centered 7-day MA: Mon 1.015, Tue 1.016, Wed 1.021, Thu 0.998, **Fri 0.958, Sat 0.968**, Sun 1.021. Amplitude max/min = 1.065 - weak. Window weekday-composition effect: val window mean factor 0.9981 vs test window 0.9992 -> **+0.1% on M, negligible**. Weekday features are of minor value for 30-day-sum targets.

## 6. Per-user log-scale seasonal shift (what RMSLE actually sees)

Cohort = users active 2025-01-01..14 (anchor-like, defined before both windows), n = 165,489:

- mean log1p(user gmv 30d): W1 = 2.157, mdl_onyx = 2.345 -> **R_user_logshift = +0.188** (ratio of means 1.087)
- buyer share (gmv>0): 51.1% -> 54.6% (+3.5pp extensive margin)
- mean gmv per user: 82.3 -> 93.9 (x1.141)
- among users buying in both windows: mean d(log1p) = +0.105 (intensive margin)

Sensitivity, cohort = active any day in Jan 2025 (n=189,446): shift +0.174. Robust.

The shift is NOT whale-driven: it appears in buyer share, in the intensive margin, and in log-means. Note +0.188 is the 2025 value and embeds 2025 trend; scaling by the same deceleration logic as M (mean-gmv ratio 1.141 -> ~1.08 expected in 2026) gives an **expected 2026 per-user log shift of ~ +0.10 to +0.12**, upper bound +0.19.

## 7. Implications for modeling

1. Val-anchor models will be trained on a systematically LOWER target regime than the test window. A global calibration of predictions upward (in log space, ~+0.10..0.12 additive on log1p, i.e. multiply predicted gmv by ~1.08-1.13 for mid-size preds) is worth testing; tune it on the val split via the 2025 analog rather than trusting the raw 2025 shift.
2. Feature windows crossing Dec/NY (for the test anchor, "last 60-90d" includes the December peak and NY collapse) need care; the val anchor sees the same distortion shifted by 30 days, so val is a reasonably honest proxy.
3. Big-sale days to know: 11.11 (real spike), Dec 13-28 (peak), Dec 29 - Jan 1 + late Jan (troughs), May 9 (dip), pre-Mar-8 week (lift), Mar 7-8 itself (dip). Black Friday: no effect in this data.
4. Growth means recent-activity features dominate: DAU +57% YoY while gmv +43-50% YoY, and AOV is falling - per-order value features should not be extrapolated upward.
