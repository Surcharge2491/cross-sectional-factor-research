"""
Data layer: a survivorship-aware synthetic equity market, plus a loader for real data.

Why synthetic first?
    Real point-in-time data (CRSP/Compustat/IBES via WRDS, or a commercial
    survivorship-free vendor) is the gold standard but is licensed. A simulator
    with *known* true alphas lets us test the research pipeline itself: a good
    pipeline must recover the planted signals, reject the placebos, and show
    the inflation caused by survivorship and look-ahead bias.

Panel convention (one row per stock-month, month-end dates):
    date        month-end timestamp
    id          stock identifier (permanent, never reused)
    sector      sector label
    ret         total return realised DURING month `date` (includes the
                delisting return in the stock's final month)
    mcap        market cap at the END of month `date`
    log_bm      log book-to-price for the fiscal period ending at `date`
                (NOT yet public at `date`; published with a lag)
    roe         return on equity for the fiscal period ending at `date`
                (announced during month date+1)
    eps_rev     net analyst EPS revisions observed during month `date`
                (public in real time)
    delisted    True in the stock's final month

The signals module is responsible for lagging fundamentals to their
publication date. Keeping raw period values here makes look-ahead bias
easy to demonstrate (and easy to avoid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = [
    "Tech", "Health", "Financials", "Energy", "Industrials",
    "ConsDisc", "ConsStaples", "Utilities", "Materials", "RealEstate",
]

# True monthly alpha loadings planted in the simulator (per 1 s.d. of the
# latent driver). These are the "answers" the pipeline should recover.
TRUE_EFFECTS = {
    "momentum":  "persistent latent alpha (AR(1), rho=0.92) -> picked up by past returns",
    "reversal":  "-0.03 x last month's idiosyncratic return",
    "value":     "+0.20%/month per s.d. of latent value; observed B/P also carries a sector tilt with NO alpha",
    "quality":   "+0.18%/month per s.d. of latent quality; quality also lowers delisting risk",
    "revisions": "revisions lead the earnings announcement jump (+1.2% per s.d. of earnings news)",
    "low_vol":   "no alpha (only beta exposure) -> should die after beta neutralisation",
    "size":      "no alpha",
    "noise_a/b": "pure placebos",
}


def simulate_market(
    n_init: int = 1200,
    n_months: int = 252,
    start: str = "2004-01-31",
    entry_per_month: int = 9,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate a monthly equity panel with listings, delistings and planted alphas."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="ME")
    K = len(SECTORS)

    # ---- common factors -------------------------------------------------
    mkt = rng.normal(0.0, 0.045, n_months)
    mkt = mkt - mkt.mean() + 0.007                      # realised premium fixed at ~8.4%/yr
    sec_ret = rng.normal(0.0, 0.025, (n_months, K))     # sector shocks
    sector_bm_tilt = rng.normal(0.0, 0.6, K)            # structural sector cheapness, no alpha

    # ---- per-stock static attributes (allocated for every stock ever) ----
    cap_n = n_init + entry_per_month * n_months + 1
    sector = rng.integers(0, K, cap_n)
    beta = np.clip(rng.normal(1.0, 0.35, cap_n), 0.2, 2.2)
    idio_vol = np.exp(rng.normal(np.log(0.10), 0.3, cap_n))
    bm_fixed = rng.normal(0.0, 0.5, cap_n)              # accounting noise in B/P
    shares = np.exp(rng.normal(0.0, 1.0, cap_n))

    # ---- per-stock dynamic state ----------------------------------------
    a = np.zeros(cap_n)          # persistent alpha (momentum source)
    v = rng.normal(0, 1, cap_n)  # latent value
    q = rng.normal(0, 1, cap_n)  # latent quality
    eps_prev = np.zeros(cap_n)   # last month's idiosyncratic return
    e_prev = np.zeros(cap_n)     # earnings news of last fiscal period (announced this month)
    rev_prev = np.zeros(cap_n)
    price = np.exp(rng.normal(3.0, 0.8, cap_n))
    ret_hist = np.zeros((cap_n, 6))  # rolling 6m returns for delisting hazard

    alive = np.zeros(cap_n, dtype=bool)
    alive[:n_init] = True
    next_id = n_init
    a_sd, rho_a = 0.0060, 0.92
    a_innov = a_sd * np.sqrt(1 - rho_a**2)

    rows = []
    for t in range(n_months):
        # new listings: smaller, lower-quality on average
        if t > 0:
            new = np.arange(next_id, next_id + entry_per_month)
            alive[new] = True
            price[new] = np.exp(rng.normal(2.3, 0.6, entry_per_month))
            q[new] = rng.normal(-0.3, 1.0, entry_per_month)
            next_id += entry_per_month

        idx = np.flatnonzero(alive)
        n = idx.size

        # expected return set at t-1 (uses only last month's state)
        mu = (a[idx] + 0.0020 * v[idx] + 0.0018 * q[idx]
              - 0.03 * eps_prev[idx]
              + 0.012 * e_prev[idx])         # earnings announcement jump

        eps = rng.normal(0, 1, n) * idio_vol[idx]
        r = beta[idx] * mkt[t] + sec_ret[t, sector[idx]] + mu + eps

        # this month's fiscal-period earnings news (announced NEXT month)
        e_now = rng.normal(0, 1, n)
        roe = 0.12 + 0.05 * q[idx] + 0.03 * e_now
        # analysts partly anticipate the news -> revisions lead the announcement
        rev = 0.30 * rev_prev[idx] + 0.45 * e_now + rng.normal(0, 0.8, n)

        # delisting: performance-related, more likely for low quality / losers
        ret_hist[idx] = np.column_stack([ret_hist[idx, 1:], r])
        trail6 = ret_hist[idx].sum(axis=1)
        hazard = 1 / (1 + np.exp(-(-6.6 - 0.9 * q[idx] - 2.5 * trail6)))
        die = rng.random(n) < hazard
        if t == 0:
            die[:] = False
        perf = rng.random(n) < 0.55   # rest are mergers / takeovers
        delist_ret = np.where(perf, rng.normal(-0.35, 0.15, n), rng.normal(0.20, 0.10, n))
        r = np.where(die, (1 + r) * (1 + delist_ret) - 1, r)
        r = np.maximum(r, -0.99)

        price[idx] *= 1 + r
        log_bm = 0.7 * v[idx] + sector_bm_tilt[sector[idx]] + bm_fixed[idx] + rng.normal(0, 0.15, n)

        rows.append(pd.DataFrame({
            "date": dates[t],
            "id": idx,
            "sector": np.array(SECTORS)[sector[idx]],
            "ret": r,
            "mcap": price[idx] * shares[idx],
            "log_bm": log_bm,
            "roe": roe,
            "eps_rev": rev,
            "delisted": die,
            "true_beta": beta[idx],
        }))

        # state transitions
        a[idx] = rho_a * a[idx] + a_innov * rng.normal(0, 1, n)
        v[idx] = 0.99 * v[idx] + np.sqrt(1 - 0.99**2) * rng.normal(0, 1, n)
        q[idx] = 0.98 * q[idx] + np.sqrt(1 - 0.98**2) * rng.normal(0, 1, n)
        eps_prev[idx] = eps
        e_prev[idx] = e_now
        rev_prev[idx] = rev
        alive[idx[die]] = False

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["id", "date"]).reset_index(drop=True)


REQUIRED_COLUMNS = ["date", "id", "sector", "ret", "mcap", "log_bm", "roe", "eps_rev", "delisted"]


def load_panel_csv(path: str) -> pd.DataFrame:
    """Load a real point-in-time panel with the same schema as `simulate_market`.

    Requirements for a valid research panel:
      * include stocks that later delisted, with their delisting return folded
        into `ret` for the final month (CRSP: DLRET);
      * `log_bm` and `roe` keyed to fiscal period end, NOT publication date
        (the signals module applies the reporting lag);
      * `eps_rev` measured from estimates available during the month (IBES).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")
    df["delisted"] = df["delisted"].astype(bool)
    return df.sort_values(["id", "date"]).reset_index(drop=True)
