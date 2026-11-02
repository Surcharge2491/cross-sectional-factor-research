"""
End-to-end cross-sectional factor research pipeline.

    python run_research.py                 # synthetic market (default)
    python run_research.py --panel my.csv  # real point-in-time panel

Outputs go to ./outputs: results.json, CSV tables and PNG figures.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from factor_lab import analytics as A
from factor_lab.data import load_panel_csv, simulate_market
from factor_lab.signals import SIGNALS, add_returns_and_universe, build_signals, neutralize

COST_BPS = 10          # base-case one-way cost, bps of traded notional
COST_GRID = [0, 5, 10, 20, 30, 50]
IS_FRACTION = 0.6      # first 60% of months = in-sample (signal selection)
T_SELECT = 3.0         # in-sample NW t needed (Harvey-Liu-Zhu style hurdle, not 2.0)


def rank_gauss_by_date(df, cols):
    def rg(x):
        r = x.rank()
        return pd.Series(norm.ppf((r - 0.5) / r.count()), index=x.index)
    return pd.DataFrame({c: df.groupby("date")[c].transform(rg) for c in cols})


def prepare(panel, lookahead=False):
    df = add_returns_and_universe(panel)
    df = build_signals(df, lookahead=lookahead)
    U = df[df["in_universe"] & df["fwd_ret"].notna()].copy()
    U = U.dropna(subset=["momentum", "low_vol"])  # need full return history
    return U.reset_index(drop=True)


def evaluate_signal(U, z, col, split_date):
    tmp = pd.concat([U[["date", "id", "fwd_ret"]], z[[col]]], axis=1)
    ic = A.ic_series(tmp, col)
    dec, w = A.decile_portfolios(tmp, col)
    ls = A.portfolio_returns(w, tmp)
    to, traded = A.turnover(w)
    net = A.net_returns(ls, traded, COST_BPS)
    res = {
        "ic_full": A.ic_summary(ic),
        "ic_is": A.ic_summary(ic[ic.index <= split_date]),
        "ic_oos": A.ic_summary(ic[ic.index > split_date]),
        "ls_gross": A.perf_stats(ls),
        "ls_net": A.perf_stats(net),
        "ls_gross_oos": A.perf_stats(ls[ls.index > split_date]),
        "ls_net_oos": A.perf_stats(net[net.index > split_date]),
        "turnover": float(to.iloc[1:].mean()),
        "breakeven_bps": float(ls.mean() / traded.iloc[1:].mean() * 1e4),
        "decile_mean": (dec.mean() * 12).to_dict(),
    }
    return res, ic, ls, net


def main(panel_path=None, out="outputs"):
    t0 = time.time()
    outdir = Path(out)
    outdir.mkdir(exist_ok=True)

    panel = load_panel_csv(panel_path) if panel_path else simulate_market()
    U = prepare(panel)
    dates = np.sort(U["date"].unique())
    split_date = dates[int(len(dates) * IS_FRACTION)]
    print(f"panel {len(panel):,} rows | universe {len(U):,} stock-months | "
          f"{len(dates)} months | {panel['id'].nunique():,} stocks ever | "
          f"{int(panel['delisted'].sum()):,} delistings | IS ends {pd.Timestamp(split_date).date()}")

    raw = rank_gauss_by_date(U, SIGNALS)
    neu = neutralize(U, SIGNALS)

    results = {"meta": {}, "signals": {}, "series": {}}
    cum = {}
    for col in SIGNALS:
        r_raw, ic_raw, ls_raw, _ = evaluate_signal(U, raw, col, split_date)
        r_neu, ic_neu, ls_neu, net_neu = evaluate_signal(U, neu, col, split_date)
        decay = A.ic_decay(pd.concat([U[["date", "id"] + [f"fwd_ret_{k}" for k in range(1, 13)]], neu[[col]]], axis=1), col)
        results["signals"][col] = {"raw": r_raw, "neutral": r_neu, "decay": decay.to_dict()}
        cum[col] = ls_neu
        print(f"{col:10s} raw IC {r_raw['ic_full']['mean_ic']:+.4f} (t {r_raw['ic_full']['t_nw']:+5.1f}) | "
              f"neutral IC {r_neu['ic_full']['mean_ic']:+.4f} (t {r_neu['ic_full']['t_nw']:+5.1f}) | "
              f"LS SR gross {r_neu['ls_gross']['sharpe']:+.2f} net {r_neu['ls_net']['sharpe']:+.2f} | "
              f"TO {r_neu['turnover']:.0%}")

    # --- signal correlations (mean cross-sectional) ---------------------
    corr = sum(neu.loc[g.index, SIGNALS].corr() for _, g in U.groupby("date")) / len(dates)

    # --- in-sample selection and composites ------------------------------
    selected = [c for c in SIGNALS if results["signals"][c]["neutral"]["ic_is"]["t_nw"] > T_SELECT]
    print("selected in-sample:", selected)

    comp_df = U[["date", "id", "fwd_ret", "sector", "beta_hat"]].copy()
    comp_df["eq"] = A.equal_weight_composite(neu, selected)
    icw, icw_weights = A.ic_weighted_composite(U, neu, selected)
    comp_df["icw"] = icw
    comp_neu = neutralize(comp_df, ["eq", "icw"])

    composites = {}
    for name in ["eq", "icw"]:
        res, ic, ls, net = evaluate_signal(U, comp_neu, name, split_date)
        tmp = pd.concat([U[["date", "id", "fwd_ret"]], comp_neu[[name]]], axis=1)
        _, w = A.decile_portfolios(tmp, name)
        _, traded = A.turnover(w)
        grid = {c: A.perf_stats(A.net_returns(ls, traded, c)[ls.index > split_date])["sharpe"] for c in COST_GRID}
        res["cost_grid_oos_sharpe"] = grid
        composites[name] = res
        cum[f"composite_{name}"] = ls
        cum[f"composite_{name}_net"] = net
        print(f"composite {name}: OOS IC {res['ic_oos']['mean_ic']:+.4f} (t {res['ic_oos']['t_nw']:+.1f}) | "
              f"OOS SR gross {res['ls_gross_oos']['sharpe']:.2f} net {res['ls_net_oos']['sharpe']:.2f} | "
              f"TO {res['turnover']:.0%} | breakeven {res['breakeven_bps']:.0f}bps")

    # --- bias demonstrations ---------------------------------------------
    bias = {}
    # (a) look-ahead: fundamentals used before publication
    U_la = prepare(panel, lookahead=True)
    neu_la = neutralize(U_la, ["value", "quality"])
    for col in ["value", "quality"]:
        r_la, *_ = evaluate_signal(U_la, neu_la, col, split_date)
        bias[f"lookahead_{col}"] = {
            "pit_ic": results["signals"][col]["neutral"]["ic_full"]["mean_ic"],
            "biased_ic": r_la["ic_full"]["mean_ic"],
            "pit_sharpe": results["signals"][col]["neutral"]["ls_gross"]["sharpe"],
            "biased_sharpe": r_la["ls_gross"]["sharpe"]}

    # (b) survivorship: only stocks still listed at the end of the sample,
    #     i.e. what you get by downloading today's constituents' history.
    survivors = panel.loc[panel["date"] == panel["date"].max(), "id"]
    U_sv = prepare(panel[panel["id"].isin(survivors)])
    neu_sv = neutralize(U_sv, SIGNALS)
    for col in SIGNALS:
        r_sv, *_ = evaluate_signal(U_sv, neu_sv, col, split_date)
        bias[f"survivor_{col}"] = {
            "pit_ic": results["signals"][col]["neutral"]["ic_full"]["mean_ic"],
            "biased_ic": r_sv["ic_full"]["mean_ic"],
            "pit_sharpe": results["signals"][col]["neutral"]["ls_gross"]["sharpe"],
            "biased_sharpe": r_sv["ls_gross"]["sharpe"]}
    # universe-level return inflation
    ew_true = U.groupby("date")["fwd_ret"].mean()
    ew_sv = U_sv.groupby("date")["fwd_ret"].mean()
    bias["universe_ew_return"] = {"pit": float(ew_true.mean() * 12), "survivors_only": float(ew_sv.mean() * 12)}

    # --- save --------------------------------------------------------------
    cum_df = pd.DataFrame(cum)
    cum_df.to_csv(outdir / "long_short_returns.csv")
    corr.to_csv(outdir / "signal_correlations.csv")
    icw_weights.to_csv(outdir / "ic_weights.csv")
    results["composites"] = composites
    results["selected"] = selected
    results["bias"] = bias
    results["correlation"] = corr.round(3).to_dict()
    results["ic_weights_last"] = icw_weights.dropna().iloc[-1].round(3).to_dict()
    results["meta"] = {
        "months": int(len(dates)), "start": str(pd.Timestamp(dates[0]).date()),
        "end": str(pd.Timestamp(dates[-1]).date()), "is_end": str(pd.Timestamp(split_date).date()),
        "stocks_ever": int(panel["id"].nunique()), "delistings": int(panel["delisted"].sum()),
        "avg_universe": float(U.groupby("date").size().mean()),
        "survivor_stocks": int(survivors.size), "cost_bps": COST_BPS, "t_select": T_SELECT,
        "data": "synthetic" if panel_path is None else panel_path,
    }
    results["series"]["cum_log"] = {c: np.log1p(cum_df[c].fillna(0)).cumsum().round(4).tolist() for c in cum_df}
    results["series"]["dates"] = [str(pd.Timestamp(d).date()) for d in cum_df.index]

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.integer):
            return int(o)
        return o
    (outdir / "results.json").write_text(json.dumps(clean(results), indent=1))
    make_figures(results, cum_df, outdir, split_date)
    print(f"done in {time.time() - t0:.0f}s -> {outdir}/")


def make_figures(results, cum_df, outdir, split_date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    sig = results["signals"]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(SIGNALS))
    ax.bar(x - 0.2, [sig[c]["raw"]["ic_full"]["mean_ic"] for c in SIGNALS], 0.4, label="raw", color="#9aa4b1")
    ax.bar(x + 0.2, [sig[c]["neutral"]["ic_full"]["mean_ic"] for c in SIGNALS], 0.4, label="sector + beta neutral", color="#2f6fdf")
    ax.set_xticks(x, SIGNALS, rotation=30)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean monthly rank IC")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "fig_ic.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c in SIGNALS:
        ax.plot(range(1, 13), [sig[c]["decay"][k] for k in range(1, 13)], marker="o", ms=3, label=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("horizon (months ahead)"); ax.set_ylabel("mean rank IC")
    ax.legend(ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "fig_decay.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c in ["composite_icw", "composite_icw_net", "composite_eq"]:
        ax.plot(cum_df.index, np.log1p(cum_df[c].fillna(0)).cumsum(), label=c)
    ax.axvline(split_date, color="k", ls="--", lw=0.8)
    ax.text(split_date, ax.get_ylim()[1] * 0.95, "  out-of-sample →", fontsize=8)
    ax.set_ylabel("cumulative log return (decile L/S)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "fig_composite.png"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", help="CSV panel with the schema in factor_lab/data.py")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    main(a.panel, a.out)
