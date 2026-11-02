# Cross-sectional equity factor research

An end-to-end quant research pipeline: survivorship-aware universe, nine signals,
sector/beta neutralisation, IC and decile tests, turnover and costs, IC decay,
in-sample selection with an out-of-sample test, signal combination, and
deliberate look-ahead and survivorship bias experiments.

## Why a synthetic market first
The default run uses a simulated market with **known planted alphas** (momentum,
reversal, value, quality, revisions) and **known zero-alpha signals** (low vol,
size, two placebos). That turns the project into a test of the research process:
a sound pipeline must find the real signals, reject the fake ones, and show how
much timing and survivorship errors distort results. The same code runs on a
real point-in-time panel.

## Headline results (synthetic, seed 7)
| | |
|---|---|
| Signals selected in-sample (NW t > 3) | reversal, value, quality, revisions |
| Composite OOS rank IC (IC-weighted) | 0.031 (t = 15.8) |
| Composite OOS L/S Sharpe, gross / net of 10 bps | 4.4 / 3.5 |
| Monthly one-way turnover | 68% |
| Look-ahead bias on quality | IC 0.011 → 0.069, Sharpe 1.5 → 7.2 |

Momentum is planted but weak; it missed the t > 3 hurdle (a deliberate type-II
error you should be able to discuss in an interview). Both placebos were rejected.
Absolute Sharpes are optimistic: Gaussian noise, no crashes, no capacity limits.

## Layout
```
factor_lab/data.py      simulator + real-data CSV loader (schema documented)
factor_lab/signals.py   universe, beta, signals with publication lags, neutralisation
factor_lab/analytics.py IC, Newey-West, deciles, turnover, costs, decay, composites
run_research.py         full study -> outputs/results.json, CSVs, PNGs
build_report.py         research-note HTML from results.json
tests/test_pipeline.py  timing, lag, delisting and neutralisation checks
```

## Run
```
pip install numpy pandas scipy matplotlib pytest
python -m pytest -q
python run_research.py                    # ~1 minute
python run_research.py --panel panel.csv  # real data
python build_report.py
```

## Using real data
Monthly panel columns: `date, id, sector, ret, mcap, log_bm, roe, eps_rev, delisted`.
- Include dead stocks, and fold the delisting return into `ret` for the final month.
- Key `log_bm` and `roe` to fiscal period end; the code applies the publication lag.
- Build `eps_rev` from estimates available during the month.

Typical source: CRSP (returns, delistings) + Compustat (fundamentals) + IBES
(estimates) via WRDS, often available through a university.

## Next steps
Risk-model optimiser with turnover penalty, square-root impact costs, slower
rebalancing for fast signals, industry-level (not sector) neutralisation,
regime analysis of factor crashes.
