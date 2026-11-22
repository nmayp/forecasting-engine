"""`rates_credit`: an ECB policy-rate path -> firms' and households' cost of borrowing (composite cost of new loans to non-financial corporations, and for house purchase),
by euro-area country. Stage 1 of docs/methods.md and its follow-up (forecasting_engine/lab/rates_credit.py).

    r_t = c + rho r_{t-1} + b0 E_t + b1 E_{t-1} + b2 E_{t-2}       country by country, ridge-shrunk to the euro-area fit
    E_t = the market-implied expected rate: Euribor 12m up to the origin; from the origin month on E_t = mean(policy path over t..t+11) + premium   (spec T6; "announced" path)
    E_t = the policy rate itself                                                                                                          (spec E; "surprise": not priced)

The policy path is the user's (monthly mean of the deposit facility rate, from the origin month). T6 treats the path as announced and fully priced twelve months ahead
(perfect foresight inside the horizon): if the market had not priced it, use pricing "surprise". What this does NOT do is the real side: credit volumes, loan stocks, building
permits, construction and investment failed their tests (stage 2 is not identified) and are not in the answer.

Request: area {"country": ["DE", ...]} (euro-area aggregate U2 and AT BE DE ES FI FR GR IE IT NL PT);
  scenario {"kind": "custom", "rate_path_pct": [policy rate, monthly mean, from the origin month; held after the last value]}
        or {"kind": "custom", "steps": [{"month": "YYYY-MM", "rate_pct": x} | {"month": "YYYY-MM", "change_pp": d}, ...]}  (cumulative; each step holds until the next)
        optional "months": 1-12 (default 12), "pricing": "announced" (default) | "surprise"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.whatif import kit as K
from forecasting_engine.whatif import mech_kit as MK
from forecasting_engine.whatif.registry import Simulator, register

ID = "rates_credit"
VERSION = "1.0 (T6 stage 1, 22 Oct 2026)"
GEOS = ("U2", "AT", "BE", "DE", "ES", "FI", "FR", "GR", "IE", "IT", "NL", "PT")
HOURS_PER_MONTH = 730.5
PARAMS = {}
SOURCES = ["ECB Data Portal: MIR composite cost of borrowing (2240 firms, 2250 house purchase), FM policy rates, Euribor 12m; Source: ECB Data Portal (statistics not modified)",
           "Rates mechanism (docs/methods.md)"]


def _policy_path(sc: dict, o0: pd.Timestamp, now: float, n: int) -> np.ndarray:
    """Monthly mean policy rate for the n months from the origin month."""
    if sc.get("rate_path_pct") is not None:
        a = MK.num_list(sc["rate_path_pct"], "rate_path_pct", None, -1.0, 10.0, max_n=12)
        return np.r_[a, np.full(max(0, n - len(a)), a[-1])][:n]
    st = sc.get("steps")
    if not st or not isinstance(st, list):
        raise ValueError("give scenario.rate_path_pct (monthly levels from the origin month) or scenario.steps")
    cur = now
    path = np.full(n, now)
    rows = []
    for s in st:
        if not isinstance(s, dict) or "month" not in s:
            raise ValueError('each step is {"month": "YYYY-MM", "rate_pct": x} or {"month": "YYYY-MM", "change_pp": d}')
        m = pd.Timestamp(str(s["month"]) + ("-01" if len(str(s["month"])) == 7 else "")).to_period("M").to_timestamp()
        rows.append((m, s))
    for m, s in sorted(rows, key=lambda r: r[0]):
        i = max(0, (m.year - o0.year) * 12 + m.month - o0.month)
        if i >= n:
            continue
        if "rate_pct" in s:
            cur = float(s["rate_pct"])
        elif "change_pp" in s:
            cur = cur + float(s["change_pp"])
        else:
            raise ValueError("a step needs rate_pct or change_pp")
        if not -1.0 <= cur <= 10.0:
            raise ValueError("policy rate must stay within -1 % and 10 %")
        path[i:] = cur
    return path


def _run_path(c: dict, y_last: float, hist_x: list, xs_future: list) -> np.ndarray:
    """Lending-rate path: rows (1, y_{t-1}, X_t, X_{t-1}, X_{t-2}) @ braw. hist_x = X for [Jul, Aug, Sep]; xs_future = X from the origin month; the first
    forecast month is Sep (the last observed lending rate is Aug)."""
    b = np.array(c["braw"], float)
    X = list(hist_x) + list(xs_future)                       # index 0 = Jul, 1 = Aug, 2 = Sep, 3 = origin month ...
    y_prev, out = y_last, []
    for t in range(2, len(X)):                               # Sep onwards
        row = np.array([1.0, y_prev, X[t], X[t - 1], X[t - 2]])
        y_prev = float(row @ b)
        out.append(y_prev)
    return np.array(out)


def run(req: dict, ctx: dict) -> dict:
    tm = K.Timer()
    P = MK.params("rates_credit")
    sc = dict(req.get("scenario") or {})
    if str(sc.get("kind") or "custom") != "custom":
        raise ValueError("scenario.kind must be custom")
    ccs = MK.codes(req.get("area"), GEOS, "area")
    pricing = str(sc.get("pricing") or "announced")
    if pricing not in ("announced", "surprise"):
        raise ValueError("scenario.pricing must be announced or surprise")
    n = int(sc.get("months") or P["horizon_months"])
    if not 1 <= n <= P["horizon_months"]:
        raise ValueError(f"scenario.months must be 1-{P['horizon_months']} (the mechanism was tested over 12 months)")
    o0 = pd.Timestamp(P["origin_month"])
    now = P["policy_in_force_pct"]
    path = _policy_path(sc, o0, now, n)
    held = np.full(n, now)
    prem = P["premium_pp"]
    hist = P["history"]
    spec = "T6" if pricing == "announced" else "E"

    def regressors(I: np.ndarray) -> list:
        """X (the regressor series) from the origin month for a policy path I over n months (held beyond the path)."""
        if pricing == "surprise":
            return list(I)
        ext = np.r_[I, np.full(11, I[-1])]
        return [float(ext[t:t + 12].mean() + prem) for t in range(n)]            # E_t = mean(I[t..t+11]) + premium
    hist_x = hist["euribor12m_pct"] if pricing == "announced" else hist["policy_pct"]
    x_user, x_held = regressors(path), regressors(held)
    market = P["euribor12m_latest"]["pct"]
    warnings = []
    mean12 = float(np.r_[path, np.full(max(0, 12 - n), path[-1])][:12].mean())
    if pricing == "announced" and abs(mean12 + prem - market) > 0.5:
        warnings.append(f"The path's 12-month average ({mean12:.2f} %) is {abs(mean12 + prem - market):.1f} pp from what Euribor 12m prices today ({market:.2f} %): the announced-and-priced "
                        "model (T6) treats your path as already priced; if the market has not priced it use scenario.pricing = \"surprise\". The unpriced-move failure case: the July 2008 origin predicted -1.66 pp against +0.36 realised.")
    t_h = np.arange(n) * HOURS_PER_MONTH                  # hours from the start of the origin month
    series, summary, cols, moments = {}, {}, {}, []
    cals = {}
    bt = P["backtest"]
    for cc in ccs:
        s = {}
        for key, nm, bkey, sig in (("ccob_nfc", "firms", "firms", P["band"]["sigma_firms"]), ("ccob_hh", "house_purchase", "house", P["band"]["sigma_house"])):
            c = P["series"][key].get(cc)
            if c is None:
                continue
            y0 = c["last_obs"]
            pu = _run_path(c[spec], y0, hist_x, x_user)[-n:]             # origin month onwards
            ph = _run_path(c[spec], y0, hist_x, x_held)[-n:]
            cal = MK.cal_mode(P, "rates_" + bkey, "chain")
            cals[nm] = cal
            if cal:
                band, s_u = MK.cal_band(pu - y0, cal, cc)
                pe = MK.exceed_prob(pu - y0, s_u, cal, cc)
                series[f"{cc}_{nm}_rate_p_leaves_normal_band"] = MK.prob_ser(t_h, pe)
            else:
                band, pe = MK.mult_band(pu - y0, sig), None
            lvl = {k: y0 + v for k, v in band.items()}
            series[f"{cc}_{nm}_rate_pct"] = MK.ser("% (composite cost of new loans; band on the change from the last observed rate)", t_h, lvl)
            series[f"{cc}_{nm}_rate_if_policy_held_pct"] = MK.ser("% (policy rate held at today's level, no band)", t_h, {"p10": y0 + ph - y0, "p50": ph, "p90": ph})
            cols[f"{cc}_{nm}_rate_pct"] = pu
            cols[f"{cc}_{nm}_rate_if_policy_held_pct"] = ph
            i6 = min(5, n - 1)
            s[nm] = dict(last_observed_pct=MK.rnd(y0, 2), last_observed_month=c["last_obs_month"],
                         end_of_path_pct=MK.qd3(lvl["p10"][-1], lvl["p50"][-1], lvl["p90"][-1]),
                         change_after_6_months_pp=MK.qd3(band["p10"][i6], band["p50"][i6], band["p90"][i6]),
                         change_at_end_pp=MK.qd3(band["p10"][-1], band["p50"][-1], band["p90"][-1]),
                         prob_leaves_normal_band_at_end=(MK.rnd(float(pe[-1]), 3) if pe is not None else None),
                         effect_of_your_path_vs_policy_held_pp=MK.rnd(pu[-1] - ph[-1], 2),
                         long_run_pass_through=MK.rnd(c[spec]["lr"], 2), half_life_months=MK.rnd(c[spec]["halflife"], 1),
                         status="screening" if nm == "firms" else "experimental",
                         status_note=("passed on development and on the cuts_2024 check; timing fails" if nm == "firms"
                                      else "passed on development, failed the check set against the open models (skill +0.16 vs +0.25); timing fails"))
        summary[cc] = s
        if "firms" in s:
            moments.append(MK.moment(0.0, f"{cc}: firms' cost of borrowing moves {s['firms']['change_at_end_pp']['p50']:+.2f} pp over {n} months (p10-p90 {s['firms']['change_at_end_pp']['p10']:+.2f} to "
                                          f"{s['firms']['change_at_end_pp']['p90']:+.2f}); the lab's back-tests put the model's half-response 4-6 months LATER than the realised one", 0.5))
    if not summary or not any(summary.values()):
        raise ValueError("no series for these countries")
    f, h = bt["firms"], bt["house"]
    line = (f"Cost of borrowing, firms: skill +{f['dev_skill']:.2f} on 4 development events (same model without the rate terms +{f['dev_off']:.2f}, best open model +{f['dev_best_open']:.2f}) and "
            f"+{f['check_skill']:.2f} on the 2024 cuts (check set, one look; {f['check_off']:+.2f} without, best open +{f['check_best_open']:.2f}); direction right in {f['direction']:.2f} of {f['moved']} series that moved, "
            f"size {f['size_range'][0]:.2f}-{f['size_range'][1]:.2f} of the realised, {f['within20']:.2f} within +-20 %; house purchase: +{h['dev_skill']:.2f} on development but +{h['check_skill']:.2f} on the check set "
            f"(best open +{h['check_best_open']:.2f}); timing FAILS: the half-response comes {f['timing_months_late'][0]}-{f['timing_months_late'][1]} months later than the realised one")
    wi = MK.whatif_block(
        mechanism="ECB policy-rate path -> cost of borrowing of firms and households (euro-area countries)", status="screening",
        status_note="firms' lending rate passed its skill, direction and size rules on development and on the check set; timing failed in every window; house-purchase rates failed the check set (experimental for them)",
        as_of=P["as_of"], tested_scope="composite cost of new loans to firms (and for house purchase), 11 euro-area countries + euro area, 12 months ahead, the policy path given; 4 development events (2008 cuts, 2011 hikes, 2014 negative rate, 2022 hikes) + the 2024 cuts",
        backtest=line,
        ranges=((f"p10-p90 CALIBRATED (80 % target): change from the last observed lending rate + conformal quantiles x (the rate's normal spread + b x |change|); {MK.calib_text({'firms': cals.get('firms'), 'house purchase': cals.get('house_purchase')})}; "
                 "series *_p_leaves_normal_band = probability that the rate ends outside its normal 90 % band; " if any(cals.values()) else
                 f"p10-p90 = the change from the last observed lending rate x exp(-+1.28 x sigma), sigma {P['band']['sigma_firms']:.2f} (firms) / {P['band']['sigma_house']:.2f} (house purchase), calibrated on the share of series within +-20 % of the realised move; ")
                + "the timing error is NOT in the band"),
        caveats=[
            "Timing: the model's half-response comes 4-6 months later than the realised one in every window (banks priced the 2022 hikes from June, before the first hike). Do not read the timing of the path.",
            "The unpriced-move failure: pricing \"announced\" treats the path as fully priced twelve months ahead. A move the market had not priced (July 2008 origin) was predicted -1.66 pp against +0.36 realised. Use \"surprise\" for unpriced paths.",
            "House-purchase rates are not explained by the policy rate alone (fixation periods and long rates matter): skill +0.16 on the check set against +0.25 for an open model; experimental.",
            "The 2024 check window had one series that moved, so direction and size there are not testable; banks passed the 2024 cuts through much less than the model's average (stock repricing still catching up).",
            "The 2022 fit is unreliable at the periphery (GR, PT, IT: the 84 months before were the zero-rate era); Greece and Ireland have the lowest fitted pass-through here.",
            "Real-side effects are NOT forecast: credit volumes, loan stocks, building permits, construction and investment failed their tests (stage 2 is not identified by the ECB shocks)."],
        licence="ECB Data Portal: free reuse incl. commercial, credit \"Source: ECB Data Portal\"; the statistics are not modified.",
        not_tested=["Policy paths longer than 12 months", "Countries outside the euro area", "Bank-level or sector-level lending conditions"],
        links_not_wired=["Rates -> new loan volumes, loan stocks, building permits, construction output, investment (all failed; stage 2 not identified)"],
        data_to=P["history"]["months"][-1], calibration=MK.calib_info({"firms": cals.get("firms"), "house_purchase": cals.get("house_purchase")}), pricing=pricing, spec=P["spec_priced"] if pricing == "announced" else P["spec_surprise"])
    rule = ("Pre-registered rules: skill above the same model without the rate terms and above the best open model; direction >= 0.80 of moved series (at least 5); median size ratio 0.8-1.25; "
            "median timing error <= 2 months. Firms' rate: skill, direction, size pass; timing FAIL.")
    cases = [{"name": "firms, development (4 events)", "metric": "MAE skill over the last-value reference", "observed": f["dev_off"], "simulated": f["dev_skill"], "error": round(f["dev_skill"] - f["dev_off"], 2)},
             {"name": "firms, check set (cuts 2024)", "metric": "MAE skill over the last-value reference", "observed": f["check_off"], "simulated": f["check_skill"], "error": round(f["check_skill"] - f["check_off"], 2)}]
    return MK.result(ID, VERSION, tm, wi, series, summary, MK.timeline(t_h, cols), moments, MK.validation(rule, line, cases),
                     assumptions=[f"Policy path (monthly mean deposit facility rate): {', '.join(f'{x:.2f}' for x in path[:6])}{' ...' if n > 6 else ''} %; held after the last value.",
                                  f"Origin month {P['origin_month'][:7]}; last observed lending rates {P['series']['ccob_nfc'][ccs[0]]['last_obs_month'][:7]}; policy rate in force {now:.2f} %; Euribor 12m {market:.2f} %.",
                                  "The path is treated as " + ("announced and fully priced twelve months ahead (premium " + f"{prem:.2f} pp)." if pricing == "announced" else "a surprise: banks react to the policy rate itself.")],
                     warnings=warnings, sources=SOURCES)


register(Simulator(id=ID, name="Policy rates to cost of borrowing", domain="economy", hazards=[], run=run, scenarios=["custom"], params=PARAMS, status="screening",
                   description="An ECB policy-rate path -> the cost of new loans to firms and for house purchase by euro-area country, with p10-p90 (direction and size back-tested; timing is 4-6 months late; no real-side effects)",
                   max_horizon_h=int(12 * HOURS_PER_MONTH)))
