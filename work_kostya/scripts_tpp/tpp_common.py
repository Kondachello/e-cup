"""Neural TPP v0 (Program 1, research week): discrete-time hazard model on the day grid.

Per user-day t the model sees only history <= t-1, expressed as:
  - Hawkes-style EMA banks (6 half-lives x 6 channels, unnormalized exponential kernels)
  - days-since-last counters (buy / cart / active / search-to-cart)
  - personal cycle-phase block (inter-purchase intervals, dsl/mean-interval ratio)
  - amount memory (last log-check, mean log-check)
  - calendar (DOW one-hot, day-of-month and day-of-year harmonics, global trend)
Heads: hazard P(buy at day t) and E[log1p(day GMV) | buy].
Forecast = Monte-Carlo rollout of 30 days with self-excitation (sampled purchases
feed back into the state); exogenous browsing is mean-field (recent per-day rates).
"""
import numpy as np
from datetime import date, timedelta

T_DATA = 409          # days 0..408 = 2025-01-01 .. 2026-02-13
HALF_LIVES = np.array([2.0, 5.0, 11.0, 24.0, 52.0, 113.0])
DECAYS = np.exp(-np.log(2.0) / HALF_LIVES).astype(np.float32)      # (6,)
NHL = len(HALF_LIVES)
CHANNELS = ["buy", "lgmv", "srch", "cart", "s2c", "active"]        # EMA channels
NC = len(CHANNELS)

# calendar tables out to day 438 (test rollout end)
_dates = [date(2025, 1, 1) + timedelta(days=int(k)) for k in range(T_DATA + 30)]
DOW = np.array([d.weekday() for d in _dates], dtype=np.int64)
DOM = np.array([d.day for d in _dates], dtype=np.float32)
DOY = np.array([d.timetuple().tm_yday for d in _dates], dtype=np.float32)

N_EMA = NC * NHL                      # 36
FEAT_NAMES = (
    [f"ema_{c}_{int(h)}" for c in CHANNELS for h in HALF_LIVES]
    + ["dsl_buy", "dsl_cart", "dsl_act", "dsl_s2c",
       "nev_buy", "nev_cart", "nev_act", "nev_s2c",
       "n_buys", "mean_int", "last_int", "phase_ratio", "multi_buy",
       "last_amt", "mean_amt", "ever_buy"]
    + [f"dow{k}" for k in range(7)]
    + ["dom_sin", "dom_cos", "doy_sin", "doy_cos", "trend"]
)
NF = len(FEAT_NAMES)                  # 64


def load_cube(path="/root/work/tpp/day_cube.npz"):
    z = np.load(path)
    return {k: z[k] for k in z.files}


class State:
    """Vectorized per-user scan state. All history strictly <= current-1."""

    def __init__(self, n):
        self.n = n
        self.E = np.zeros((n, NC, NHL), dtype=np.float32)
        self.last = np.full((n, 4), -100000, dtype=np.int32)   # buy, cart, act, s2c
        self.nbuys = np.zeros(n, dtype=np.float32)
        self.sum_amt = np.zeros(n, dtype=np.float32)
        self.last_amt = np.zeros(n, dtype=np.float32)
        self.prev_buy = np.full(n, -100000, dtype=np.int32)
        self.last_int = np.zeros(n, dtype=np.float32)
        self.sum_int = np.zeros(n, dtype=np.float32)

    def features(self, t, out=None):
        n = self.n
        f = out if out is not None else np.empty((n, NF), dtype=np.float32)
        f[:, :N_EMA] = np.log1p(self.E.reshape(n, N_EMA))
        dsl = np.minimum(t - self.last, 500).astype(np.float32)   # (n,4)
        nev = self.last < 0
        dsl[nev] = 500.0
        f[:, N_EMA:N_EMA + 4] = np.log1p(dsl)
        f[:, N_EMA + 4:N_EMA + 8] = nev
        i = N_EMA + 8
        nb = self.nbuys
        multi = nb >= 2
        mean_int = np.where(multi, self.sum_int / np.maximum(nb - 1, 1), 0.0)
        dslb = dsl[:, 0]
        phase = np.where(multi, np.minimum(dslb / np.maximum(mean_int, 1.0), 8.0), 0.0)
        f[:, i] = np.log1p(nb)
        f[:, i + 1] = np.log1p(mean_int)
        f[:, i + 2] = np.log1p(self.last_int)
        f[:, i + 3] = phase
        f[:, i + 4] = multi
        ever = nb > 0
        f[:, i + 5] = self.last_amt
        f[:, i + 6] = np.where(ever, self.sum_amt / np.maximum(nb, 1), 0.0)
        f[:, i + 7] = ever
        i += 8
        f[:, i:i + 7] = 0.0
        f[:, i + DOW[t]] = 1.0
        i += 7
        f[:, i] = np.sin(2 * np.pi * DOM[t] / 30.44)
        f[:, i + 1] = np.cos(2 * np.pi * DOM[t] / 30.44)
        f[:, i + 2] = np.sin(2 * np.pi * DOY[t] / 365.25)
        f[:, i + 3] = np.cos(2 * np.pi * DOY[t] / 365.25)
        f[:, i + 4] = t / 408.0
        return f

    def update(self, t, buy, lgmv, srch, cart, s2c, active):
        """Feed day-t observations (arrays over users) into the state."""
        self.E *= DECAYS[None, None, :]
        x = np.stack([buy, lgmv, srch, cart, s2c, active], axis=1).astype(np.float32)
        self.E += x[:, :, None]
        b = buy > 0
        if b.any():
            had = self.prev_buy[b] >= 0
            intv = (t - self.prev_buy[b]).astype(np.float32)
            self.sum_int[b] += np.where(had, intv, 0.0)
            self.last_int[b] = np.where(had, intv, 0.0)
            self.prev_buy[b] = t
            self.nbuys[b] += 1
            self.sum_amt[b] += lgmv[b]
            self.last_amt[b] = lgmv[b]
            self.last[b, 0] = t
        self.last[cart > 0, 1] = t
        self.last[active > 0, 2] = t
        self.last[s2c > 0, 3] = t


def day_slices(cube, t):
    nord = cube["nord"][:, t]
    return {
        "buy": (nord > 0).astype(np.float32),
        "lgmv": cube["lgmv"][:, t].astype(np.float32),
        "srch": np.log1p(cube["srch"][:, t].astype(np.float32)),
        "cart": np.log1p(cube["cart"][:, t].astype(np.float32)),
        "s2c": np.log1p(cube["s2c"][:, t].astype(np.float32)),
        "active": (cube["active"][:, t] > 0).astype(np.float32),
    }


def scan_collect(cube, t_end, rows_by_t=None, state_out_day=None):
    """Scan days 0..t_end-1 updating state; optionally collect features for
    sampled rows (dict t -> user index array) and/or return the state snapshot
    *as of* day `state_out_day` (features at that day use history <= day-1).
    Returns (collected_feats, collected_keys, state) — state only if requested.
    """
    n = cube["nord"].shape[0]
    st = State(n)
    total = sum(len(v) for v in rows_by_t.values()) if rows_by_t else 0
    X = np.empty((total, NF), dtype=np.float16) if total else None
    off = 0
    for t in range(t_end):
        if state_out_day is not None and t == state_out_day:
            return X[:off] if X is not None else None, None, st
        if rows_by_t is not None and t in rows_by_t:
            u = rows_by_t[t]
            sub = _SubView(st, u)
            X[off:off + len(u)] = sub.features(t)
            off += len(u)
        d = day_slices(cube, t)
        st.update(t, **d)
    if state_out_day is not None and state_out_day == t_end:
        return X[:off] if X is not None else None, None, st
    return X, None, None


class _SubView:
    """Feature computation for a subset of users without copying full state."""

    def __init__(self, st, idx):
        self.n = len(idx)
        self.E = st.E[idx]
        self.last = st.last[idx]
        self.nbuys = st.nbuys[idx]
        self.sum_amt = st.sum_amt[idx]
        self.last_amt = st.last_amt[idx]
        self.prev_buy = st.prev_buy[idx]
        self.last_int = st.last_int[idx]
        self.sum_int = st.sum_int[idx]
    features = State.features


def build_mlp(seed, h1=128, h2=96):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(NF, h1), nn.GELU(),
                nn.Linear(h1, h2), nn.GELU(),
            )
            self.h_buy = nn.Linear(h2, 1)
            self.h_amt = nn.Linear(h2, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.h_buy(z).squeeze(-1), self.h_amt(z).squeeze(-1)

    return Net()
