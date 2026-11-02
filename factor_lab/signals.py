"""
Signal construction, universe definition and cross-sectional neutralisation.

Timing convention (the single most important thing to get right):
    A signal stamped at month-end t may only use information public by t.
    It is evaluated against `fwd_ret`, the return realised during month t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Publication lags in months, applied to fiscal-period-keyed fundamentals.
BM_LAG = 4    # book value: annual/quarterly filings arrive with a delay
ROE_LAG = 2   # period-t earnings are announced during t+1; +1 month buffer

SIGNALS = ["momentum", "reversal", "value", "quality", "revisions",
           "low_vol", "size", "noise_a", "noise_b"]


def add_returns_and_universe(panel: pd.DataFrame, min_history: int = 13,
                             size_pct: float = 0.20) -> pd.DataFrame:
    """Forward returns, beta estimates and the investable universe."""
    df = panel.sort_values(["id", "date"]).copy()
    g = df.groupby("id", sort=False)

    # Forward return: next month's return, INCLUDING delisting returns.
    df["fwd_ret"] = g["ret"].shift(-1)
    # Returns k months ahead, from the FULL panel (so delisting returns are
    # kept even if the stock later leaves the universe). Used for IC decay.
    for k in range(1, 13):
        df[f"fwd_ret_{k}"] = g["ret"].shift(-k)
    df["age"] = g.cumcount() + 1

    # Cap-weighted market return (weights = previous month-end cap).
    df["mcap_lag"] = g["mcap"].shift(1)
    mkt = (df.dropna(subset=["mcap_lag"])
             .groupby("date")
             .apply(lambda x: np.average(x["ret"], weights=x["mcap_lag"]), include_groups=False))
    df["mkt"] = df["date"].map(mkt)

    # Rolling 36m beta (min 12 obs), using data up to and including t.
    df["rm"] = df["ret"] * df["mkt"]
    roll = lambda s: s.groupby(df["id"]).transform(lambda x: x.rolling(36, min_periods=12).mean())
    m_r, m_m, m_rm = roll(df["ret"]), roll(df["mkt"]), roll(df["rm"])
    m_mm = roll(df["mkt"] ** 2)
    df["beta_hat"] = ((m_rm - m_r * m_m) / (m_mm - m_m ** 2)).clip(-1, 4)

    # Universe: enough history, beta available, above the size cut-off at t.
    size_cut = df.groupby("date")["mcap"].transform(lambda x: x.quantile(size_pct))
    df["in_universe"] = (df["age"] >= min_history) & df["beta_hat"].notna() & (df["mcap"] >= size_cut)
    return df.drop(columns=["rm"])


def build_signals(df: pd.DataFrame, seed: int = 11, lookahead: bool = False) -> pd.DataFrame:
    """Raw signals (higher = expected to outperform).

    lookahead=True deliberately uses unpublished fundamentals (lag 0) to
    show how much a timing bug inflates results. Never use it for research.
    """
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id", sort=False)
    logret = np.log1p(df["ret"])

    # 12-1 momentum: months t-11..t-1, skipping the most recent month.
    df["momentum"] = logret.groupby(df["id"]).transform(lambda x: x.shift(1).rolling(11, min_periods=11).sum())
    # 1-month reversal.
    df["reversal"] = -df["ret"]
    # Value: book-to-price published with a lag.
    df["value"] = g["log_bm"].shift(0 if lookahead else BM_LAG)
    # Quality: ROE published with a lag.
    df["quality"] = g["roe"].shift(0 if lookahead else ROE_LAG)
    # Earnings revisions: 3-month sum of net revisions (real-time data).
    df["revisions"] = df.groupby("id")["eps_rev"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    # Low volatility: negative 12m return volatility.
    df["low_vol"] = -df.groupby("id")["ret"].transform(lambda x: x.rolling(12, min_periods=12).std())
    # Size: negative log market cap.
    df["size"] = -np.log(df["mcap"])

    # Placebos: persistent random scores with no relation to returns.
    rng = np.random.default_rng(seed)
    for name in ("noise_a", "noise_b"):
        z = pd.Series(rng.normal(0, 1, len(df)), index=df.index)
        df[name] = z.groupby(df["id"]).transform(lambda x: x.ewm(alpha=0.15).mean())
    return df


def _rank_gauss(x: pd.Series) -> pd.Series:
    """Cross-sectional rank -> standard normal scores (robust to outliers)."""
    r = x.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / r.count()), index=x.index)


def neutralize(df: pd.DataFrame, cols: list[str], sector_col: str = "sector",
               beta_col: str = "beta_hat") -> pd.DataFrame:
    """Rank-gauss each signal, then regress out sector dummies + beta each month.

    Returns z-scored residuals: the part of each signal that is NOT a
    sector bet or a market (beta) bet.
    """
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for date, grp in df.groupby("date"):
        X = pd.get_dummies(grp[sector_col], dtype=float)
        X["beta"] = grp[beta_col] - grp[beta_col].mean()
        Xv = X.to_numpy()
        for c in cols:
            y = grp[c]
            ok = y.notna().to_numpy()
            if ok.sum() < 50:
                continue
            z = _rank_gauss(y[ok]).to_numpy()
            coef, *_ = np.linalg.lstsq(Xv[ok], z, rcond=None)
            resid = z - Xv[ok] @ coef
            out.loc[y.index[ok], c] = (resid - resid.mean()) / resid.std()
    return out
