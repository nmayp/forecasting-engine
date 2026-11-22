"""Builds the small parameter files of the user-facing what-if simulators (forecasting_engine/whatif/mech_*.py) from the lab's mechanisms and back-test results.

    python -m forecasting_engine.whatif.backtests.build heat|strike|rates|choke|calib [--as-of YYYY-MM-DD]

Why a build step: the simulators run in the deployed image, which has no lab store (ENTSO-E load, Oil Bulletin history, ECB series live in the store or in
results/). They read only the compact JSON written here (forecasting_engine/whatif/data/<name>.json): fitted coefficients, the last observed state needed to
anchor a path, and the back-test numbers each result must quote. Raw series that licences keep out of products (ENTSO-E actual load, EUROCONTROL flights,
Drewry / SCFI freight levels, IMF PortWatch) are never written: only coefficients, shares and ratios. Every file carries `as_of`, the code spec and where
the numbers come from. docs/methods.md has the tests behind each file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "forecasting_engine" / "whatif" / "data"
RES = ROOT / "results" / "cascade"


CALIB_KEYS = {"heat_load": ["heat_load"], "rates_credit": ["rates_firms", "rates_house"], "oil_pump_hicp": ["oil_pump", "hicp_fuel"]}
CALIB_NODES = {"oil_pump": ("1 pump", ("chain", "single")), "hicp_fuel": ("1 HICP FUEL", ("chain", "S1")), "heat_load": ("2 load (heat_load)", ("chain",)),
               "rates_firms": ("4 firms lending rate", ("chain",)), "rates_house": ("4 house_purchase lending rate", ("chain",))}


def calib_block(name: str) -> dict | None:
    """Calibrated p10-p90 bands and exceedance of a wired mechanism (forecasting_engine.lab.calib_run: conformal on development events, frozen, one check look): the frozen
    parameters, the layer's normal spread and normal band per series and step, and the coverage / score numbers each result quotes. Read from results/cascade (no store)."""
    keys = CALIB_KEYS.get(name)
    f, g = RES / "calib_engine.json", RES / "calib_nodes.json"
    if not keys or not f.exists() or not g.exists():
        return None
    eng, nodes = json.loads(f.read_text()), json.loads(g.read_text())
    out = {}
    for k in keys:
        d = dict(eng[k])
        node, models = CALIB_NODES[k]
        meta = {}
        for mode, model in zip(("chain", "given"), models):
            r = nodes.get(f"{node} | {model}")
            if not r:
                continue
            m = dict(exceed_method=d["modes"][mode]["exceed"])
            ek = "calibrated" if d["modes"][mode]["exceed"] == "calibrated" else "current"
            for sp in ("dev", "check"):
                if sp in r:
                    m[sp] = dict(n_events=r[sp]["calibrated"]["n_events"], coverage_before=round(r[sp]["current"]["coverage"], 2), coverage_after=round(r[sp]["calibrated"]["coverage"], 2),
                                 interval_score_before=round(r[sp]["current"]["interval_score"], 2), interval_score_after=round(r[sp]["calibrated"]["interval_score"], 2),
                                 exceed_brier_normal_spread=round(r[sp]["current"]["brier"], 3), exceed_brier_shipped=round(r[sp][ek]["brier"], 3),
                                 exceed_bss_normal_spread=round(r[sp]["current"]["bss"], 2), exceed_bss_shipped=round(r[sp][ek]["bss"], 2), exceed_rate=round(r[sp]["current"]["exceed_rate"], 2))
            meta[mode] = m
        d["meta"] = meta
        out[k] = d
    return out


def _w(name: str, doc: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    cb = calib_block(name)
    if cb:
        doc = {**doc, "calib": cb}
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(doc, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    return p


def sigma_from_within(share: float, half_width: float) -> float:
    """Normal sigma such that P(|e| < half_width) = share (two-sided): how a reported 'within +-20 %' share becomes a p10-p90 band."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + min(max(share, 0.05), 0.95) / 2)
    return half_width / z


# --------------------------------------------------------------------------------------------------------------------------------- heat -> load
def build_heat(as_of: str) -> Path:
    os.environ.setdefault("HEAT_CACHE", str(Path(os.environ.get("TMPDIR", "/tmp")) / "heat_cache"))
    Path(os.environ["HEAT_CACHE"]).mkdir(parents=True, exist_ok=True)
    from forecasting_engine.whatif.backtests import heat_load as H
    inp = H.inputs()
    cut = pd.Timestamp(as_of, tz="UTC") - pd.Timedelta(days=28)
    curves = H.fit_all(inp, cut)
    out = {}
    for cc, cv in curves.items():
        hc = H.HourlyCurve(cv, inp["load"][cc], inp["temp"][cc])
        nn = H.n_nonneg(cv.names)
        sh_h = [float(hc.b[h][-nn]) for h in range(24)]           # local hour 0..23: heating slope (log points per degC below Th)
        sc_h = [float(hc.b[h][-nn + 1]) for h in range(24)]       # cooling slope per local hour
        clim = [float(x) for x in cv.clim.reindex(range(1, 367)).to_numpy()]
        out[cc] = dict(th=float(cv.th), tc=float(cv.tc), sh=float(cv.sh), sc=float(cv.sc), sh_h=sh_h, sc_h=sc_h, clim_te=[round(x, 3) for x in clim],
                       n_fit_days=int(len(cv.hist_y.dropna())))
    sc = json.loads((RES / "heat_load_scen.json").read_text())
    w20 = [e["path_scen"]["within20"] for e in sc]
    w10 = [e["path_scen"]["within10"] for e in sc]
    med = [e["path_scen"]["med_ratio"] for e in sc]
    sig = sigma_from_within(float(np.mean(w20)), 0.20)
    doc = dict(name="heat_load", as_of=as_of, fit_data_before=str(cut.date()), spec=dict(K=H.CFG["K"], anchor_days=H.CFG["anc"], inertia_w=list(H.CFG["w"])),
               countries=out, inertia_w=list(H.CFG["w"]),
               band=dict(sigma_log=round(sig, 4), basis=f"within +-20 % of the realised excess in {np.mean(w20):.2f} of country-events (mean over the five heat-wave tests) -> normal sigma {sig:.3f}"),
               backtest=dict(events=[e["event"] for e in sc], median_size_ratio=[round(x, 2) for x in med], within10=[round(x, 2) for x in w10],
                             within20=[round(x, 2) for x in w20], direction_right=[round(e["path_scen"]["dir"], 2) for e in sc],
                             skill_scen=[round(e["skill_scen"], 3) for e in sc], skill_engine=[(round(e["skill_eng"], 3) if e.get("skill_eng") is not None else None) for e in sc],
                             hotday_level_ratio=None))
    return _w("heat_load", doc)


# --------------------------------------------------------------------------------------------------------------------------------- strikes -> air traffic
def build_strike(as_of: str) -> Path:
    cells = []
    for f, tag in (("strike_air_dev_S2.json", "development (k = 1, u = 0, nothing fitted)"), ("strike_air_check_S0.json", "check set (k fitted on development, 1.02)")):
        d = json.loads((RES / f).read_text())
        for c in d["cells"]:
            if c.get("moved") and c.get("size_ratio"):
                cells.append(dict(event=c["event"], cc=c["cc"], basis=c["basis"], verified=c["verified"], split=tag.split(" ")[0],
                                  realised_over_predicted=1.0 / float(c["size_ratio"])))
    allr = np.array([c["realised_over_predicted"] for c in cells])
    ann = np.array([c["realised_over_predicted"] for c in cells if c["basis"] == "announced"])

    def q(a):
        return {k: round(float(np.quantile(a, p)), 3) for k, p in (("p10", .1), ("p50", .5), ("p90", .9))} | {"n": int(len(a))}
    dev = json.loads((RES / "strike_air_dev_S2.json").read_text())["summary"]
    chk = json.loads((RES / "strike_air_check_S0.json").read_text())["summary"]
    doc = dict(name="strike_air", as_of=as_of, k=1.0, catchup_u=0.0, cap=0.98,
               ratio_realised_over_predicted=dict(announced=q(ann), estimate=q(allr)),
               backtest=dict(dev=dict(n=dev["n"], moved=dev["moved"], size_median=round(dev["size_median"], 2), within20=dev["within20"], skill_vs_engine_off=round(dev["skill_vs_off"], 3)),
                             check=dict(n=chk["n"], moved=chk["moved"], size_median=round(chk["size_median"], 2), within20=chk["within20"], skill_vs_engine_off=round(chk["skill_vs_off"], 3),
                                        cells_beating_off=chk["cells_beating_off"])),
               cells=[{k: v for k, v in c.items()} for c in cells])
    return _w("strike_air", doc)


# --------------------------------------------------------------------------------------------------------------------------------- policy rate -> borrowing cost
def build_rates(as_of: str) -> Path:
    """Stage 1 of docs "Mechanism: interest rates to lending and investment" and its follow-up: T6 (market-implied expected path, 150-month window) for a path that is
    announced and priced, E (policy-rate ARDL, 84-month window) for a surprise. Only the fitted coefficients, the last observed lending rate and the last three months of
    policy and Euribor rates are stored (ECB Data Portal: reuse with credit, statistics not modified)."""
    from forecasting_engine.whatif.backtests import rates_credit as RC
    S = RC.Store()
    o = pd.Timestamp(as_of)
    o0 = o.to_period("M").to_timestamp()
    out: dict = {}
    prem = None
    for spec in ("T6", "E"):
        RC.use_spec(spec)
        for key in ("ccob_nfc", "ccob_hh"):
            _, coef = RC.stage1(S, key, o, modes=("mech",))
            for g, c in coef.items():
                last = S.known(key, o)[g].dropna()
                d = out.setdefault(key, {}).setdefault(g, dict(last_obs=float(last.iloc[-1]), last_obs_month=str(last.index[-1].date())))
                d[spec] = dict(braw=c["braw"], resp6=c["resp6"], resp12=c["resp12"], lr=c["lr"], halflife=c["halflife"], n=c["n"])
    RC.use_spec("T6")
    held = S.policy_path(o, "held")
    grid = pd.date_range(o0 - pd.DateOffset(months=6), o0 + pd.DateOffset(months=12), freq="MS")
    A = RC.expect_gap(S, o, held, grid)
    prem = float(A.loc[o0])
    eu = S.ext["eur12m"]
    hist_m = [o0 - pd.DateOffset(months=k) for k in (3, 2, 1)]
    doc = dict(name="rates_credit", as_of=as_of, origin_month=str(o0.date()), spec_priced="T6 (Euribor 12m expected path, 150 months)", spec_surprise="E (policy-rate ARDL, 84 months)",
               policy_in_force_pct=float(S.pol.loc[o]), euribor12m_latest=dict(month=str(eu.dropna().index[-1].date()), pct=float(eu.dropna().iloc[-1])), premium_pp=prem, horizon_months=12,
               history=dict(months=[str(m.date()) for m in hist_m], policy_pct=[float(S.ipol.get(m, np.nan)) for m in hist_m], euribor12m_pct=[float(eu.get(m, np.nan)) for m in hist_m]),
               series=out,
               backtest=dict(note="docs/methods.md, follow-up: T6 on 4 development events and the cuts_2024 check (one look)",
                             firms=dict(dev_skill=0.39, dev_off=0.07, dev_best_open=0.06, check_skill=0.40, check_off=-0.06, check_best_open=0.33, direction=1.00, moved=22,
                                        size_range=[0.75, 0.92], within10=0.41, within20=0.59, timing_months_late=[4, 6]),
                             house=dict(dev_skill=0.29, dev_off=0.14, dev_best_open=0.14, check_skill=0.16, check_off=0.11, check_best_open=0.25, direction=1.00, moved=20,
                                        size_range=[1.05, 1.13], within10=0.25, within20=0.55, timing_months_late=[5, 6])),
               band=dict(sigma_firms=round(sigma_from_within(0.59, 0.20), 4), sigma_house=round(sigma_from_within(0.55, 0.20), 4),
                         basis="share of series within +-20 % of the realised move (0.59 firms, 0.55 house purchase; first-pass table, the T6 shares were not reported)"))
    return _w("rates_credit", doc)


def build_choke(as_of: str) -> Path:
    """Link 1 of docs "Mechanism: shipping chokepoints to freight and import prices": ln(R_t / R_0) = beta * e * phi * EWMA(d_t). beta, phi and the exposure shares are the frozen spec
    (choke_spec.json, not fitted); the band is the size spread over the beta x phi grid of choke_posthoc.json. No Drewry / SCFI level is stored."""
    spec = json.loads((RES / "choke_spec.json").read_text())
    post = json.loads((RES / "choke_posthoc.json").read_text())
    grid = post["rtm_size_sensitivity"]
    central = grid["beta10_phi0.21"]
    rel = np.array([v / central for v in grid.values()])
    doc = dict(name="choke_freight", as_of=as_of, beta=spec["L1"]["beta"], phi=spec["L1"]["phi"], exposure=spec["L1"]["exposure"], ewma_a=0.5,
               size_factor=dict(p10=round(float(np.quantile(rel, 0.1)), 3), p50=1.0, p90=round(float(np.quantile(rel, 0.9)), 3), n=len(rel), min=round(float(rel.min()), 3), max=round(float(rel.max()), 3),
                                basis="size ratio predicted / realised of the Shanghai-Rotterdam rise over beta in {7, 10, 15} x phi in {0.15, 0.21, 0.27}, relative to the central beta 10, phi 0.21 (choke_posthoc.json)"),
               backtest=dict(check_event="Red Sea from 11 Dec 2023 (Suez container transits -54 % over 26 weeks)", lane_26w=dict(realised_ln=1.04, predicted_ln=1.09, size=1.05), lane_52w=dict(realised_ln=1.19, predicted_ln=1.14, size=0.96),
                             composite_26w=dict(realised_ln=0.77, predicted_ln=0.38, size=0.50), composite_52w=dict(realised_ln=0.91, predicted_ln=0.40, size=0.44),
                             mae=dict(mechanism=0.333, no_change=0.979, chronos2=0.587, tirex2=0.905), direction="4 of 4", timing_weeks=10.5, dev_cells=6, dev_direction=0.50, dev_note="no development event has an exposed container lane with a transit cut: beta is not identified on 2021-23"))
    return _w("choke_freight", doc)


def build_calib(as_of: str) -> Path:
    """Attach the calibrated bands to the parameter files that exist, without refitting any coefficient (the files keep their as_of)."""
    for name in CALIB_KEYS:
        p = OUT / f"{name}.json"
        if p.exists():
            doc = json.loads(p.read_text())
            doc.pop("calib", None)
            cb = calib_block(name)
            if cb:
                doc["calib"] = cb
            p.write_text(json.dumps(doc, indent=1) + "\n")
    return OUT


BUILDERS = {"heat": build_heat, "strike": build_strike, "rates": build_rates, "choke": build_choke, "calib": build_calib}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(BUILDERS) + ["all"])
    ap.add_argument("--as-of", default=str(pd.Timestamp.today().date()))
    a = ap.parse_args(argv)
    for k in (sorted(BUILDERS) if a.what == "all" else [a.what]):
        print(k, BUILDERS[k](a.as_of))
    return 0


if __name__ == "__main__":
    sys.exit(main())
