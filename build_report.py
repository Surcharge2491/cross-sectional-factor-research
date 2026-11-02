"""Render outputs/results.json into a self-contained research-note HTML page.

    python build_report.py   ->  outputs/report.html
"""
import json
from pathlib import Path

R = json.loads(Path("outputs/results.json").read_text())
TEMPLATE = Path("report_template.html").read_text()

TRUE = {"momentum": True, "reversal": True, "value": True, "quality": True, "revisions": True,
        "low_vol": False, "size": False, "noise_a": False, "noise_b": False}
LABEL = {"momentum": "Momentum 12-1", "reversal": "1m reversal", "value": "Value (B/P)",
         "quality": "Quality (ROE)", "revisions": "EPS revisions", "low_vol": "Low volatility",
         "size": "Size", "noise_a": "Placebo A", "noise_b": "Placebo B"}

sig, comp, meta, bias = R["signals"], R["composites"], R["meta"], R["bias"]
selected = R["selected"]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def verdict(name):
    s = sig[name]["neutral"]["ic_is"]["t_nw"]
    if name in selected:
        return "pass", "Selected"
    if TRUE[name]:
        return "warn", "Missed"
    return "fail", "Rejected"


rows = []
for name in sig:
    n = sig[name]["neutral"]
    v_cls, v_txt = verdict(name)
    rows.append({
        "name": LABEL[name], "planted": TRUE[name],
        "raw_ic": sig[name]["raw"]["ic_full"]["mean_ic"], "ic": n["ic_full"]["mean_ic"],
        "t_is": n["ic_is"]["t_nw"], "ic_oos": n["ic_oos"]["mean_ic"], "t_oos": n["ic_oos"]["t_nw"],
        "sr": n["ls_gross"]["sharpe"], "sr_net": n["ls_net"]["sharpe"], "to": n["turnover"],
        "be": n["breakeven_bps"], "v_cls": v_cls, "v_txt": v_txt,
    })

dec = comp["icw"]["decile_mean"]
dec_vals = [dec[k] for k in sorted(dec, key=float)]
corr_keys = ["momentum", "reversal", "value", "quality", "revisions"]

page_data = {
    "dates": R["series"]["dates"],
    "cum": {k: R["series"]["cum_log"][k] for k in ["composite_icw", "composite_icw_net", "composite_eq"]},
    "is_end": meta["is_end"],
    "decay": {LABEL[k]: [sig[k]["decay"][str(h)] for h in range(1, 13)] for k in corr_keys},
    "cost": {"IC-weighted": [comp["icw"]["cost_grid_oos_sharpe"][c] for c in ["0", "5", "10", "20", "30", "50"]],
             "Equal-weight": [comp["eq"]["cost_grid_oos_sharpe"][c] for c in ["0", "5", "10", "20", "30", "50"]]},
    "cost_x": [0, 5, 10, 20, 30, 50],
    "deciles": dec_vals,
    "rows": rows,
    "corr": {"keys": [LABEL[k] for k in corr_keys],
             "m": [[R["correlation"][a][b] for b in corr_keys] for a in corr_keys]},
    "bias": [
        {"test": "Look-ahead", "sig": "Quality (ROE)", **bias["lookahead_quality"]},
        {"test": "Look-ahead", "sig": "Value (B/P)", **bias["lookahead_value"]},
        {"test": "Survivorship", "sig": "Quality (ROE)", **bias["survivor_quality"]},
        {"test": "Survivorship", "sig": "Momentum 12-1", **bias["survivor_momentum"]},
        {"test": "Survivorship", "sig": "EPS revisions", **bias["survivor_revisions"]},
        {"test": "Survivorship", "sig": "Value (B/P)", **bias["survivor_value"]},
    ],
}

icw, eq = comp["icw"], comp["eq"]
lq, sq = bias["lookahead_quality"], bias["survivor_quality"]
ew = bias["universe_ew_return"]
weights = R["ic_weights_last"]
tokens = {
    "START": meta["start"][:4], "END": meta["end"][:4], "MONTHS": str(meta["months"]),
    "IS_END": meta["is_end"][:4], "OOS_START": str(int(meta["is_end"][:4]) + 1),
    "STOCKS": f"{meta['stocks_ever']:,}", "DELIST": f"{meta['delistings']:,}",
    "UNIV": f"{meta['avg_universe']:,.0f}", "SURV": f"{meta['survivor_stocks']:,}",
    "N_SEL": str(len(selected)), "SEL_LIST": ", ".join(LABEL[s] for s in selected),
    "ICW_IC": f"{icw['ic_oos']['mean_ic']:.3f}", "ICW_T": f"{icw['ic_oos']['t_nw']:.1f}",
    "ICW_SR": f"{icw['ls_gross_oos']['sharpe']:.1f}", "ICW_SRN": f"{icw['ls_net_oos']['sharpe']:.1f}",
    "ICW_RET": pct(icw["ls_gross_oos"]["ann_ret"]), "ICW_RETN": pct(icw["ls_net_oos"]["ann_ret"]),
    "ICW_TO": pct(icw["turnover"], 0), "ICW_BE": f"{icw['breakeven_bps']:.0f}",
    "ICW_DD": pct(icw["ls_net_oos"]["max_dd"]),
    "EQ_IC": f"{eq['ic_oos']['mean_ic']:.3f}", "EQ_SR": f"{eq['ls_gross_oos']['sharpe']:.1f}",
    "EQ_SRN": f"{eq['ls_net_oos']['sharpe']:.1f}",
    "SR50": f"{icw['cost_grid_oos_sharpe']['50']:.1f}",
    "LQ_PIT": f"{lq['pit_ic']:.3f}", "LQ_B": f"{lq['biased_ic']:.3f}",
    "LQ_SRP": f"{lq['pit_sharpe']:.1f}", "LQ_SRB": f"{lq['biased_sharpe']:.1f}",
    "SQ_SRP": f"{sq['pit_sharpe']:.1f}", "SQ_SRB": f"{sq['biased_sharpe']:.1f}",
    "EW_PIT": pct(ew["pit"]), "EW_SV": pct(ew["survivors_only"]),
    "MOM_T": f"{sig['momentum']['neutral']['ic_is']['t_nw']:.1f}",
    "MOM_TF": f"{sig['momentum']['neutral']['ic_full']['t_nw']:.1f}",
    "REV_TO": pct(sig["reversal"]["neutral"]["turnover"], 0),
    "REV_SR": f"{sig['reversal']['neutral']['ls_gross']['sharpe']:.1f}",
    "REV_SRN": f"{sig['reversal']['neutral']['ls_net']['sharpe']:.1f}",
    "VAL_RAW": f"{sig['value']['raw']['ic_full']['mean_ic']:.3f}",
    "VAL_NEU": f"{sig['value']['neutral']['ic_full']['mean_ic']:.3f}",
    "W_REV": pct(weights.get("revisions", 0), 0), "W_REVR": pct(weights.get("reversal", 0), 0),
    "COST": str(meta["cost_bps"]), "TSEL": f"{meta['t_select']:.1f}",
    "DATA_JSON": json.dumps(page_data, separators=(",", ":")),
}
html = TEMPLATE
for k, v in tokens.items():
    html = html.replace(f"%%{k}%%", v)
assert "%%" not in html, "unreplaced token"
Path("outputs/report.html").write_text(html)
print("wrote outputs/report.html", len(html) // 1024, "KB")
