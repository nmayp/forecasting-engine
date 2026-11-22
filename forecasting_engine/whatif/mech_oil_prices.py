"""`oil_pump_hicp`: an oil / fuel price path -> pump prices (diesel, petrol) and the energy components of consumer prices (HICP fuels; electricity, gas and heat), by EU country.
Back-tested mechanisms of docs/methods.md: "Mechanism: energy prices to consumer prices", "Scenario test: pump prices given the realised oil path" and
"Energy chain: HICP what-if" (forecasting_engine/lab/pump_mech.py, energy_hicp.py).

    pump price   dN(s) = sum_{k=0..5} b_k dX(s-k) + r1 dN(s-1) + r2 dN(s-2),   W = W_last + v (N - N_last)      X = wholesale product price (EUR/l), v = ad valorem tax factor
    FUEL (HICP)  x_t = b0 dp_t + b1 dp_{t-1} + b2 dh_t                   p = monthly mean log pump price (petrol and diesel), h = heating oil
    ELC_GAS      x_t = sum_k w_k (A0 dgas_{t-k-d} + A1 delec_{t-k-d})     gas and electricity changes in EUR/MWh, country-specific lag kernel

The user gives the PATH (a product or Brent price path, or the pump-price path itself, and gas / electricity changes); nothing here forecasts oil. Excise is held at its last
published level (tax cuts and rebates, which explained about 0.17 of skill in 2022, are not modelled). Every effect is the ADDITIONAL change the scenario brings against the same
run with prices held flat from now: price moves already under way before the origin are not in it. IMF gas and ENTSO-E electricity series are not republished: you give changes.

Request: area {"country": [EU member ISO-2 codes]};
  scenario {"kind": "custom", "product_change_pct": number | [cumulative % change of the wholesale product price vs now, per week]    (diesel and petrol products)
                              | "brent_usd_bbl": [Brent levels per week] (+ optional "brent_now_usd_bbl")  -> product moves by the same EUR/l (x params.crude_to_product)
                              | "pump_change_pct": number | [cumulative % change of the pump price per week]  (skips the pass-through mechanism: the HICP link alone)
            optional, product by product (the mechanism's input IS the wholesale product price, so these are supported, not context):
              "diesel_change_pct" / "gasoline_change_pct": number | [cumulative % change of that product's wholesale price per week]  (replaces the common path for that fuel)
              "crack_change_usd_bbl" (both) / "diesel_crack_change_usd_bbl" / "gasoline_crack_change_usd_bbl": number | [change per week of the product's margin over Brent, USD/bbl]
                  (added to the path above, or alone with the product otherwise flat: e.g. a diesel crack +20 USD/bbl with Brent unchanged)
            "gas_delta_eur_mwh": number | [per month from the first month after the last published HICP, cumulative change vs now],
            "elec_delta_eur_mwh": same (8 countries have an electricity term), "weeks": n (default the path length, else 12), "months": 1-6 (HICP, default 4)}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.whatif import kit as K
from forecasting_engine.whatif import mech_kit as MK
from forecasting_engine.whatif.registry import Simulator, register

ID = "oil_pump_hicp"
VERSION = "1.0"
MAX_WEEKS = 26
HOURS_PER_MONTH = 730.5
EU = ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK")
L_G = 12
PARAMS = {"crude_to_product": {"default": 1.0, "unit": "x", "doc": "share of a Brent move (in EUR/l) that reaches the wholesale product price (1 = the same EUR/l move; the refining margin is not modelled)"}}
SOURCES = ["EU Weekly Oil Bulletin (European Commission, reuse authorised with acknowledgement): pump prices with and without taxes, Monday prices",
           "EIA spot prices (public domain) and ECB reference rates for the product price; Eurostat HICP (CC BY 4.0 equivalent, Source: Eurostat)", "energy-price mechanisms of this repository (docs/methods.md)"]


def _pump(c: dict, prod: dict, dX: np.ndarray, n: int) -> np.ndarray:
    """Weekly pump price with taxes (EUR/l) for n weeks after the last published Monday, given the weekly product-price changes dX (n,)."""
    b = np.array(c["b"], float)
    hist = list(prod["dx_hist"])                                   # dX of the last 5 published weeks, oldest first
    dxs = hist + list(dX)
    dn = list(c["dn_hist"])
    cum, out = 0.0, []
    for u in range(1, n + 1):
        i = len(hist) + u - 1                                      # index of week u in dxs
        x = np.r_[[dxs[i - k] if i - k >= 0 else 0.0 for k in range(6)], dn[-1], dn[-2]]
        d = float(x @ b)
        dn.append(d)
        cum += d
        out.append(c["w_last_eur_l"] + c["v"] * cum)
    return np.array(out)


def _kernel(rho: float, d: int, x: np.ndarray, m: int) -> np.ndarray:
    """sum_k w_k x[t-k-d], w_k = rho^k / sum, k < 12, x indexed from the first forecast month (zero before)."""
    w = rho ** np.arange(L_G)
    w = w / w.sum()
    return np.array([sum(w[k] * (x[t - k - d] if 0 <= t - k - d < len(x) else 0.0) for k in range(L_G)) for t in range(m)])


PRODUCT_KEYS = ("product_change_pct", "brent_usd_bbl", "diesel_change_pct", "gasoline_change_pct", "crack_change_usd_bbl", "diesel_crack_change_usd_bbl", "gasoline_crack_change_usd_bbl")
FUEL_KEY = {"diesel": ("diesel_change_pct", "diesel_crack_change_usd_bbl"), "gasoline": ("gasoline_change_pct", "gasoline_crack_change_usd_bbl")}


def _weekly_x(sc: dict, prod: dict, brent: dict, fuel: str, n: int, ctp: float):
    """Product-price changes per week (EUR/l, length n) from the scenario; None when the scenario gives a pump path instead (or nothing for this product).
    Order: the product's own % path, else the common % path, else Brent (x params.crude_to_product), else flat; then the product's crack-spread change is added."""
    pct_key, crack_key = FUEL_KEY[fuel]
    per_l = brent["bbl_l"] * brent["usd_per_eur"]                  # USD/bbl -> EUR/l
    x = None
    for k in (pct_key, "product_change_pct"):
        if sc.get(k) is not None:
            pct = MK.num_list(sc[k], k, n, -90, 500)
            x = prod["x_last_eur_l"] * (1 + pct / 100)
            break
    if x is None and sc.get("brent_usd_bbl") is not None:
        b = MK.num_list(sc["brent_usd_bbl"], "brent_usd_bbl", n, 5, 500)
        b0 = float(sc.get("brent_now_usd_bbl") or brent["usd_bbl"])
        x = prod["x_last_eur_l"] + ctp * (b - b0) / per_l
    ck = sc.get(crack_key) if sc.get(crack_key) is not None else sc.get("crack_change_usd_bbl")
    if ck is not None:
        c = MK.num_list(ck, crack_key if sc.get(crack_key) is not None else "crack_change_usd_bbl", n, -150, 300)
        x = (np.full(n, prod["x_last_eur_l"]) if x is None else x) + c / per_l
    if x is None:
        return None
    x = np.maximum(x, 0.05)
    return np.diff(np.r_[prod["x_last_eur_l"], x])


def run(req: dict, ctx: dict) -> dict:
    tm = K.Timer()
    P = MK.params("oil_pump_hicp")
    sc = dict(req.get("scenario") or {})
    if str(sc.get("kind") or "custom") != "custom":
        raise ValueError("scenario.kind must be custom")
    prm = {k: v["default"] for k, v in PARAMS.items()} | (req.get("params") or {})
    ccs = MK.codes(req.get("area"), EU)
    given = [k for k in PRODUCT_KEYS if sc.get(k) is not None]
    first = next((sc[k] for k in (*PRODUCT_KEYS, "pump_change_pct") if sc.get(k) is not None), None)
    if first is None and sc.get("gas_delta_eur_mwh") is None and sc.get("elec_delta_eur_mwh") is None:
        raise ValueError("give scenario.product_change_pct, scenario.brent_usd_bbl, a diesel / gasoline product or crack path or scenario.pump_change_pct (and/or gas_delta_eur_mwh / elec_delta_eur_mwh)")
    n = int(sc.get("weeks") or (len(first) if isinstance(first, (list, tuple)) else 12))
    if not 1 <= n <= MAX_WEEKS:
        raise ValueError(f"the price path must have 1-{MAX_WEEKS} weeks")
    M = int(sc.get("months") or 4)
    if not 1 <= M <= 6:
        raise ValueError("scenario.months must be 1-6")
    ctp = float(prm["crude_to_product"])
    if not 0 <= ctp <= 3:
        raise ValueError("params.crude_to_product must be 0-3")
    pump_mode = sc.get("pump_change_pct") is not None and not given
    K_m = pd.Timestamp(P["known_hicp_month"])
    first_month = (K_m + pd.DateOffset(months=1)).to_period("M").to_timestamp()
    sL = pd.Timestamp(P["pump"]["diesel"]["product"]["last_monday"])
    # internal horizon: through the end of the last HICP month asked for
    end_month = first_month + pd.DateOffset(months=M) - pd.Timedelta(days=1)
    n_int = max(n, int(np.ceil((end_month - sL).days / 7)))
    mondays = [sL + pd.Timedelta(weeks=u) for u in range(1, n_int + 1)]
    midx = np.array([(d.year - first_month.year) * 12 + d.month - first_month.month for d in mondays])
    n_mon = {j: sum(1 for d in pd.date_range(first_month + pd.DateOffset(months=j), first_month + pd.DateOffset(months=j + 1) - pd.Timedelta(days=1)) if d.weekday() == 0) for j in range(M)}
    tw = np.arange(1, n_int + 1) * 168.0
    tmo = np.arange(M) * HOURS_PER_MONTH
    gas = sc.get("gas_delta_eur_mwh")
    elec = sc.get("elec_delta_eur_mwh")
    g_d = MK.num_list(gas, "gas_delta_eur_mwh", M, -300, 1000, max_n=M) if gas is not None else np.zeros(M)
    e_d = MK.num_list(elec, "elec_delta_eur_mwh", M, -500, 2000, max_n=M) if elec is not None else np.zeros(M)
    series, summary, cols_w, cols_m, moments, warnings = {}, {}, {}, {}, [], []
    sig_pump, sig_elc = P["band"]["sigma_pump"], P["band"]["sigma_elc_gas"]
    sig_fuel = P["band"]["sigma_fuel_s1"] if pump_mode else P["band"]["sigma_fuel_s2"]
    # calibrated ranges (conformal on development events, docs "Calibrated what-if ranges"): the pump layer when the model builds the pump path from a Brent path ("chain": the
    # product leg is predicted too) or from a product path given ("given"); the HICP fuel component given the pump path (S1 = "given") or from the oil path ("chain")
    product_given = bool(set(given) & {"product_change_pct", "diesel_change_pct", "gasoline_change_pct"})        # a Brent / crack path leaves the product leg to the model ("chain")
    cal_pump = None if pump_mode or first is None else MK.cal_mode(P, "oil_pump", "given" if product_given else "chain")
    cal_fuel = MK.cal_mode(P, "hicp_fuel", "given" if pump_mode else "chain")
    exceed = {}
    if sc.get("pump_change_pct") is not None and given:
        warnings.append("pump_change_pct is ignored because a wholesale product / Brent / crack path was also given (the pump path is only used alone)")
    if any(k.endswith("crack_change_usd_bbl") for k in given):
        warnings.append("Crack-spread paths move the wholesale product price (product = Brent + crack) and feed the same tested pass-through; the back-test used realised product prices, "
                        "not crack paths as such, and the crack you give is added to whatever Brent / product path you also give (give the margin OR a product path that already contains it)")
    if len({"diesel_change_pct", "gasoline_change_pct", "diesel_crack_change_usd_bbl", "gasoline_crack_change_usd_bbl"} & set(given)) and "product_change_pct" not in given and "brent_usd_bbl" not in given:
        warnings.append("Only the product(s) you gave a path for move; the other product's wholesale price is held flat")
    if M > 4:
        warnings.append("HICP months beyond the fourth after the last published month were not tested (targets were K+2..K+4)")
    for cc in ccs:
        s = {}
        h = P["hicp"].get(cc, {})
        dl = {}                                              # per fuel: weekly log differences (scenario vs flat) of the pump price
        for fuel, col in (("diesel", "diesel"), ("gasoline", "gasoline")):
            prod = P["pump"][fuel]["product"]
            c = P["pump"][fuel]["countries"].get(cc)
            if c is None:
                continue
            if first is None:
                continue                                                 # gas / electricity only: no pump path
            Wl = c["w_last_eur_l"]
            flat = _pump(c, prod, np.zeros(n_int), n_int)
            if pump_mode:
                pc = MK.num_list(sc["pump_change_pct"], "pump_change_pct", n, -90, 500)
                scen = Wl * (1 + np.r_[pc, np.full(n_int - n, pc[-1])] / 100)
            else:
                dX = _weekly_x(sc, prod, P["brent"], fuel, n, ctp)
                if dX is None:
                    scen = flat
                else:
                    scen = _pump(c, prod, np.r_[dX, np.zeros(n_int - n)], n_int)
            if cal_pump:
                band, s_u = MK.cal_band(scen - Wl, cal_pump, cc)
                pe = MK.exceed_prob(scen - Wl, s_u, cal_pump, cc)
                series[f"{cc}_{fuel}_pump_p_leaves_normal_band"] = MK.prob_ser(tw, pe)
                exceed[f"{cc}_{fuel}_pump"] = pe
            else:
                band = MK.mult_band(scen - Wl, sig_pump)
            lvl = {k: Wl + v for k, v in band.items()}
            series[f"{cc}_{fuel}_pump_eur_l"] = MK.ser("EUR/l with taxes (band on the change from the last published price)", tw, lvl)
            series[f"{cc}_{fuel}_pump_if_prices_flat_eur_l"] = MK.ser("EUR/l with taxes, wholesale price held flat (no band)", tw, {"p10": flat, "p50": flat, "p90": flat})
            cols_w[f"{cc}_{fuel}_pump_eur_l"] = scen
            dl[fuel] = np.log(scen / flat)
            i = n - 1 if n <= n_int else n_int - 1
            s[f"{fuel}_pump"] = dict(last_published_eur_l=MK.rnd(Wl, 3), last_monday=prod["last_monday"],
                                     change_at_week_n_eur_l=MK.qd3(band["p10"][i], band["p50"][i], band["p90"][i]),
                                     change_at_week_n_pct=MK.qd3(*(100 * band[k][i] / Wl for k in ("p10", "p50", "p90"))),
                                     effect_vs_prices_flat_pct=MK.rnd(100 * (scen[i] / flat[i] - 1), 2), tax_factor_v=MK.rnd(c["v"], 3))
            if cal_pump:
                s[f"{fuel}_pump"]["prob_leaves_normal_band_at_week_n"] = MK.rnd(float(exceed[f"{cc}_{fuel}_pump"][i]), 3)
        # HICP: FUEL
        comp = {}
        if "diesel" in dl and "gasoline" in dl and h.get("fuel_b"):
            b0, b1, b2 = h["fuel_b"]
            dtr = np.zeros(M)
            dho = np.zeros(M)
            for u in range(n_int):
                j = midx[u]
                if 0 <= j < M:
                    dtr[j] += 0.5 * (dl["diesel"][u] + dl["gasoline"][u]) / n_mon[j]
                    dho[j] += dl["diesel"][u] / n_mon[j]
            dp = 100 * np.diff(np.r_[0.0, dtr])
            dh = 100 * np.diff(np.r_[0.0, dho])
            fx = b0 * dp + b1 * np.r_[0.0, dp[:-1]] + b2 * dh
            comp["FUEL"] = (np.cumsum(fx), cal_fuel or sig_fuel, h["w_fuel"])
        # HICP: ELC_GAS (gas, and electricity where the country has an electricity term)
        k = h.get("elc")
        if k and (gas is not None or elec is not None):
            A = k["A"]
            ex = A[0] * _kernel(k["rho"], k["d"], np.diff(np.r_[0.0, g_d]), M)
            if k["has_el"] and len(A) > 1:
                ex = ex + A[1] * _kernel(k["rho"], k["d"], np.diff(np.r_[0.0, e_d]), M)
            elif elec is not None and not k["has_el"]:
                warnings.append(f"{cc}: no electricity price term in the model for this country; elec_delta_eur_mwh is ignored")
            if k["has_el"] and elec is None:
                warnings.append(f"{cc}: the model has an electricity term and no elec_delta_eur_mwh was given (taken as zero)")
            comp["ELC_GAS"] = (np.cumsum(ex), sig_elc, h["w_elc_gas"])
        tot_c, tot_h = np.zeros(M), np.zeros(M)
        for nm, (cum, sg, w) in comp.items():
            if isinstance(sg, dict):                                # calibrated (FUEL): month j after the last published month = benchmark step j - 1
                band, s_u = MK.cal_band(cum, sg, cc, np.maximum(np.arange(M) - 1, 0))
                pe = MK.exceed_prob(cum, s_u, sg, cc, np.maximum(np.arange(M) - 1, 0))
                series[f"{cc}_hicp_{nm.lower()}_p_leaves_normal_band"] = MK.prob_ser(tmo, pe)
            else:
                band, pe = MK.mult_band(cum, sg), None
            series[f"{cc}_hicp_{nm.lower()}_cum_pct"] = MK.ser(f"% change of the HICP component {nm} caused by the scenario (cumulative from the first month after the last published)", tmo, band)
            cols_m[f"{cc}_hicp_{nm.lower()}_cum_pct"] = cum
            tot_c += (w or 0) * cum
            tot_h += ((w or 0) * (band["p90"] - band["p10"]) / 2) ** 2
            s[f"hicp_{nm.lower()}"] = dict(cumulative_effect_pct=MK.qd3(band["p10"][-1], band["p50"][-1], band["p90"][-1]), months=M, weight_in_basket=MK.rnd(w, 4),
                                           prob_leaves_normal_band_at_end=(MK.rnd(float(pe[-1]), 3) if pe is not None else None),
                                           status="screening" if nm == "FUEL" else "experimental",
                                           status_note=("direction right in 0.96 of country-months given the pump path, 0.82 given only the wholesale path" if nm == "FUEL"
                                                        else "direction indicator only (0.71 right), size 0.08-0.58 of the realised: not a size estimate"))
        if comp:
            hw = np.sqrt(tot_h)
            series[f"{cc}_energy_headline_contribution_pp"] = MK.ser("pp of the headline HICP index level, indicative (component weight x component effect; chain-linking ignored)", tmo,
                                                                      {"p10": tot_c - hw, "p50": tot_c, "p90": tot_c + hw})
            cols_m[f"{cc}_energy_headline_contribution_pp"] = tot_c
            s["headline_contribution_pp"] = MK.qd3(tot_c[-1] - hw[-1], tot_c[-1], tot_c[-1] + hw[-1])
        if not s:
            raise ValueError(f"no model for {cc}")
        summary[cc] = s
        if "diesel_pump" in s:
            d = s["diesel_pump"]["change_at_week_n_pct"]
            moments.append(MK.moment((n - 1) * 168.0 + 168, f"{cc}: diesel at the pump {d['p50']:+.1f} % after {n} weeks (p10-p90 {d['p10']:+.1f} to {d['p90']:+.1f} %)", min(1.0, abs(d["p50"]) / 30)))
    frames = sorted(MK.timeline(tw, cols_w) + MK.timeline(tmo, cols_m), key=lambda f: f["t_h"])
    bt = P["backtest"]
    pb, f1, f2, eg = bt["pump"], bt["fuel_s1"], bt["fuel_s2"], bt["elc_gas"]
    line = (f"Pump prices given the realised wholesale path: {pb['cells']} shock cells in {pb['events']} events (+{pb['check_cells']} check cell), direction right in {pb['direction']:.2f}, median predicted / realised cumulative change "
            f"{pb['median_size_ratio']:.2f}, {pb['within20']:.2f} of cells within +-20 %, skill {pb['skill']:.2f} vs the engine's {pb['engine']:.3f} and {pb['brent_1to1']:.2f} for pump moving 1:1 with Brent; peak timing {pb['peak_timing_weeks']} week; "
            f"HICP fuels (5 events): direction {f1['direction']:.2f} given the pump path / {f2['direction']:.2f} given only the wholesale path, size {f1['size'][0]:.2f}-{f1['size'][1]:.2f} / {f2['size'][0]:.2f}-{f2['size'][1]:.2f} of the realised; "
            f"electricity-gas: direction {eg['direction']:.2f}, size {eg['size'][0]:.2f}-{eg['size'][1]:.2f}")
    wi = MK.whatif_block(
        mechanism="Oil / fuel price path -> pump prices -> HICP fuels and electricity-gas (EU countries)", status="screening",
        status_note="the pump-price what-if given an oil path passed its direction and size checks on development events (and one check cell); the HICP fuels link is screening; electricity-gas is a direction indicator (experimental)",
        as_of=P["as_of"], tested_scope="weekly diesel and petrol pump prices (Oil Bulletin) of 27 countries given the realised wholesale path, 8 events 2018-2024; HICP fuels and electricity-gas for 27 countries, 5 events, 2-4 months ahead, price paths given",
        backtest=line,
        ranges=((f"p10-p90 CALIBRATED (80 % target): central effect + conformal quantiles x (the layer's normal spread + b x |effect|); {MK.calib_text({'pump': cal_pump, 'HICP fuels': cal_fuel})}. " if (cal_pump or cal_fuel) else "")
                + (f"p10-p90 = the central effect x exp(-+1.28 x sigma): pump {sig_pump:.2f} (0.64 of cells within +-20 %)" if not cal_pump else "pump band: see above")
                + (f"; HICP fuels {sig_fuel:.2f} ({'0.44' if pump_mode else '0.20'} of country-months within +-20 %)" if not cal_fuel else "")
                + f"; electricity-gas {sig_elc:.2f} (0.11 within +-20 %: the band is very wide on purpose, not calibrated). "
                "Series *_p_leaves_normal_band: probability that the layer ends outside its normal 90 % band (Brier-scored in the benchmark)"),
        caveats=[
            "A conditional what-if, not a forecast: you give the oil path; the mechanism does not predict it. As a forecast the pass-through loses to the engine in shocks (skill -0.02 vs +0.03); only the what-if given the path is the validated part.",
            "Tax regime: excise is held at its last published level and VAT as the fitted factor. Fuel-tax cuts and rebates (2022) explained about 0.17 of the skill in the 2022 cells and are not modelled; announced measures are not used. French refinery strikes (size ratio about 0) are not an oil-price effect.",
            "Pass-through is fast: about 75 % of a wholesale move reaches the pump within three weeks and over 90 % within five (ECB, Meyler 2009); the product price is the US NYH / Gulf spot, not the European ARA price, so the refining margin is not in a Brent path.",
            "Effects are incremental: the scenario against the same run with wholesale prices held flat; moves already under way before the origin are not in the numbers.",
            "HICP fuels given only the wholesale path: direction 0.82 and about 60 % of the realised size, because the pump step under-reacts when excise changes. Electricity-gas: direction 0.71, the 2022 moves were far larger than the pre-origin fits imply; not a size estimate.",
            "Headline HICP is dominated by non-energy components and base effects: the contribution shown is arithmetic (weight x component effect), not a headline forecast; the euro-area aggregate over 26 months over-corrects.",
            "Five HICP events (two share the same 2022 shock), the electricity-gas variant was chosen on them; ENTSO-E day-ahead prices cover 8 countries only."],
        licence="Coefficients were fitted with IMF gas (FRED) and ENTSO-E day-ahead electricity prices (back-tests only): neither series is stored or shown, and you supply gas and electricity changes. "
                "EU Weekly Oil Bulletin: credit \"European Commission, Weekly Oil Bulletin\"; Eurostat: \"Source: Eurostat\".",
        not_tested=["Oil price forecasts", "Diesel and petrol in non-EU countries", "Gas and electricity retail prices beyond the HICP components", "Headline inflation as a forecast"],
        links_not_wired=["Energy price shocks -> industrial output / demand (failed)", "Freight -> import prices and goods HICP (failed)"],
        inputs_used=dict(wholesale_product_paths=given, pump_path_only=bool(pump_mode), gas=gas is not None, electricity=elec is not None),
        known_hicp_month=P["known_hicp_month"][:7], last_oil_bulletin_monday=P["pump"]["diesel"]["product"]["last_monday"], brent_reference=P["brent"],
        calibration=MK.calib_info({"pump": cal_pump, "hicp_fuel": cal_fuel}))
    rule = ("Tested as a conditional scenario on shock cells: direction right in >= 0.80 of cells and median predicted / realised change 0.8-1.25 (pump: 1.00 and 0.98); "
            "as a forecast the mechanism does not beat the engine, so only the what-if is offered.")
    cases = [{"name": "pump prices, realised wholesale path given", "metric": "median predicted / realised cumulative change (1 = exact)", "observed": 1.0, "simulated": pb["median_size_ratio"], "error": round(pb["median_size_ratio"] - 1, 2)},
             {"name": "HICP fuels, pump path given (S1)", "metric": "median predicted / realised size (best event range)", "observed": 1.0, "simulated": f1["size"][1], "error": round(f1["size"][1] - 1, 2)}]
    return MK.result(ID, VERSION, tm, wi, series, summary, frames, moments, MK.validation(rule, line, cases),
                     assumptions=[f"Week 1 = the Oil Bulletin Monday after {P['pump']['diesel']['product']['last_monday']}; the product price is held after the last week of your path.",
                                  f"HICP months run from {first_month.strftime('%Y-%m')} (the month after the last published HICP).", "Heating oil follows diesel; excise held at the last published level."],
                     warnings=warnings, sources=SOURCES)


register(Simulator(id=ID, name="Oil prices to pump prices and inflation", domain="economy", hazards=[], run=run, scenarios=["custom"], params=PARAMS, status="screening",
                   description="An oil / product price path -> pump prices (diesel, petrol) and the fuel and electricity-gas components of consumer prices by EU country, with p10-p90 (given-the-path what-if; electricity-gas is a direction indicator)",
                   max_horizon_h=24 * 7 * MAX_WEEKS))
