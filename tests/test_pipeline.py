"""Sanity tests for the research pipeline. Run: python -m pytest -q"""
import numpy as np
import pandas as pd

from factor_lab import analytics as A
from factor_lab.data import simulate_market
from factor_lab.signals import add_returns_and_universe, build_signals, neutralize


def small_panel():
    return simulate_market(n_init=300, n_months=60, entry_per_month=3, seed=1)


def test_forward_return_includes_delisting():
    df = add_returns_and_universe(small_panel())
    assert df["delisted"].sum() > 0
    # in the month before a delisting, fwd_ret must equal the final-month
    # return, which already includes the delisting return
    before = df.groupby("id")["delisted"].shift(-1, fill_value=False).astype(bool)
    final_ret = df.groupby("id")["ret"].shift(-1)
    assert before.sum() > 0
    assert np.allclose(df.loc[before, "fwd_ret"], final_ret[before])


def test_fundamentals_are_lagged():
    df = build_signals(add_returns_and_universe(small_panel()))
    g = df.groupby("id")
    assert np.allclose(df["quality"].dropna(), g["roe"].shift(2).dropna())
    assert np.allclose(df["value"].dropna(), g["log_bm"].shift(4).dropna())


def test_momentum_skips_last_month():
    df = build_signals(add_returns_and_universe(small_panel()))
    one = df[df["id"] == df["id"].iloc[0]].reset_index(drop=True)
    t = 20
    expected = np.log1p(one["ret"].iloc[t - 11:t]).sum()  # months t-11..t-1
    assert np.isclose(one["momentum"].iloc[t], expected)


def test_neutralized_signal_is_orthogonal_to_sector_and_beta():
    df = build_signals(add_returns_and_universe(small_panel()))
    U = df[df["in_universe"]].reset_index(drop=True)
    z = neutralize(U, ["value"])
    d = U.assign(z=z["value"]).dropna(subset=["z"])
    last = d[d["date"] == d["date"].max()]
    assert last.groupby("sector")["z"].mean().abs().max() < 1e-8
    assert abs(np.corrcoef(last["z"], last["beta_hat"])[0, 1]) < 1e-8


def test_ic_recovers_a_known_signal():
    rng = np.random.default_rng(0)
    n, T = 500, 60
    dates = np.repeat(pd.date_range("2020-01-31", periods=T, freq="ME"), n)
    s = rng.normal(size=n * T)
    r = 0.05 * s + rng.normal(size=n * T)
    ic = A.ic_series(pd.DataFrame({"date": dates, "s": s, "fwd_ret": r}), "s")
    assert 0.03 < ic.mean() < 0.07


def test_turnover_of_static_book_is_zero_after_first_month():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    w = pd.DataFrame({"date": np.repeat(dates, 2), "id": [1, 2] * 3, "w": [1.0, -1.0] * 3})
    to, traded = A.turnover(w)
    assert to.iloc[1:].abs().max() == 0
