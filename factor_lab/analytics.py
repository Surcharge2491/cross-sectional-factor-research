"""
Signal evaluation: information coefficients, decile portfolios, turnover,
transaction costs, decay, and signal combination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- statistics
def newey_west_t(x: pd.Series, lags: int = 3) -> float:
    """t-statistic of the mean with Newey-West (HAC) standard errors."""
    x = pd.Series(x).dropna().to_numpy()
    n = x.size
    if n < 10:
        return np.nan
    u = x - x.mean()
    s = u @ u / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (u[L:] @ u[:-L]) / n
    return x.mean() / np.sqrt(s / n)


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    return {"ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "t_stat": newey_west_t(r), "max_dd": max_drawdown(r), "months": int(r.size)}


# ----------------------------------------------------------------------- IC
def ic_series(df: pd.DataFrame, col: str, ret_col: str = "fwd_ret") -> pd.Series:
    """Monthly Spearman rank IC between a signal and forward returns."""
    d = df[["date", col, ret_col]].dropna()
    def _ic(x):
        return x[col].rank().corr(x[ret_col].rank()) if len(x) > 30 else np.nan
    return d.groupby("date").apply(_ic, include_groups=False)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    return {"mean_ic": ic.mean(), "ic_vol": ic.std(),
            "icir_ann": ic.mean() / ic.std() * np.sqrt(12),
            "t_nw": newey_west_t(ic), "hit_rate": (ic > 0).mean(), "months": int(ic.size)}


def ic_decay(df: pd.DataFrame, col: str, horizons=range(1, 13)) -> pd.Series:
    """Mean IC of the signal at t against the return in month t+k.

    Expects columns fwd_ret_1..fwd_ret_12 computed on the full panel
    (see signals.add_returns_and_universe), so delisting returns are kept.
    """
    out = {k: ic_series(df, col, f"fwd_ret_{k}").mean() for k in horizons}
    return pd.Series(out, name=col)


# ---------------------------------------------------------------- portfolios
def decile_portfolios(df: pd.DataFrame, col: str, n: int = 10,
                      ret_col: str = "fwd_ret") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weighted decile returns and the long-short weights per month.

    Returns (decile_returns[date x decile], weights[long format]).
    Weights are dated at formation t and earn fwd_ret (month t+1).
    """
    d = df[["date", "id", col, ret_col]].dropna().copy()
    d["dec"] = d.groupby("date")[col].transform(
        lambda x: pd.qcut(x.rank(method="first"), n, labels=False) + 1)
    dec_ret = d.groupby(["date", "dec"])[ret_col].mean().unstack()

    ls = d[d["dec"].isin([1, n])].copy()
    cnt = ls.groupby(["date", "dec"])["id"].transform("count")
    ls["w"] = np.where(ls["dec"] == n, 1.0, -1.0) / cnt
    return dec_ret, ls[["date", "id", "w"]]


def score_weighted_weights(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Dollar-neutral weights proportional to the (neutralised) score; gross = 2."""
    d = df[["date", "id", col]].dropna().copy()
    d["w"] = d.groupby("date")[col].transform(lambda x: 2 * (x - x.mean()) / (x - x.mean()).abs().sum())
    return d[["date", "id", "w"]]


def portfolio_returns(weights: pd.DataFrame, df: pd.DataFrame, ret_col: str = "fwd_ret") -> pd.Series:
    m = weights.merge(df[["date", "id", ret_col]], on=["date", "id"], how="left")
    m[ret_col] = m[ret_col].fillna(0.0)
    return m.assign(pnl=m["w"] * m[ret_col]).groupby("date")["pnl"].sum()


def turnover(weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Monthly one-way turnover = 0.5 * sum|w_t - w_{t-1}| / gross book (100% = full replacement).

    Also returns traded notional sum|w_t - w_{t-1}|, which is what costs are charged on.

    Simplification: ignores intra-month drift of the weights.
    """
    W = weights.pivot_table(index="date", columns="id", values="w", fill_value=0.0)
    traded = W.diff().abs().sum(axis=1)
    traded.iloc[0] = W.iloc[0].abs().sum()
    gross = W.abs().sum(axis=1)
    return (0.5 * traded / gross).rename("turnover"), traded.rename("traded")


def net_returns(gross_ret: pd.Series, traded: pd.Series, cost_bps: float) -> pd.Series:
    """Subtract one-way cost (bps of traded notional) from gross returns."""
    return gross_ret - traded.reindex(gross_ret.index).fillna(0) * cost_bps / 1e4


# --------------------------------------------------------------- combination
def ic_weighted_composite(df: pd.DataFrame, z: pd.DataFrame, cols: list[str],
                          window: int = 36, min_obs: int = 24) -> pd.Series:
    """Composite score with weights = trailing mean IC, floored at zero.

    No look-ahead: at formation date t we may only use ICs whose forward
    return has already been realised, i.e. ICs stamped at t-1 and earlier.
    """
    tmp = pd.concat([df[["date", "fwd_ret"]], z[cols]], axis=1)
    ics = pd.DataFrame({c: ic_series(tmp, c) for c in cols})
    w = ics.shift(1).rolling(window, min_periods=min_obs).mean().clip(lower=0)
    w = w.div(w.sum(axis=1), axis=0)
    wd = df[["date"]].join(w, on="date")[cols]
    comp = (z[cols].fillna(0) * wd).sum(axis=1)
    comp[wd.isna().all(axis=1) | (wd.sum(axis=1) == 0)] = np.nan
    return comp, w


def equal_weight_composite(z: pd.DataFrame, cols: list[str]) -> pd.Series:
    return z[cols].mean(axis=1, skipna=True)
