"""`strike_air`: a strike calendar -> flights lost per day in a country. Back-tested mechanism of docs/methods.md (forecasting_engine/lab/strike_mech.py), air traffic only.

    on strike day j the country's flights fall by the share k * I_j,  k = 1 (fitted 0.99-1.23 on the development events, 1.02 on the check set), no catch-up (u = 0)

I_j is the ANNOUNCED share of the country's scheduled flights cancelled that day (the regulator's or airport's notice: DGAC asks airlines to cut 25 % of Paris flights -> 0.25;
Brussels cancels all departures and half of arrivals -> 0.75). The mechanism takes it as given; it does not predict how many flights a strike will stop.

Status experimental: every back-test number rests on EUROCONTROL daily flights under a back-test-only licence (). So only the RELATIVE effect is shown
(share of flights lost); flights lost in numbers appear only on a baseline of daily flights the caller gives. No EUROCONTROL level is stored or shown.

Request: area {"country": ["BE", ...]};
  scenario {"kind": "custom", "start": "YYYY-MM-DD", "days": n, "intensity": share | [share per day]}   consecutive strike days from `start`
        or {"kind": "custom", "dates": ["YYYY-MM-DD", ...], "intensity": share | [share per date]}       explicit strike days (days between them are zero)
  params {"intensity_basis": "announced" | "estimate", "baseline_flights": number | [per day] | {cc: ..}}
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from forecasting_engine.whatif import kit as K
from forecasting_engine.whatif import mech_kit as MK
from forecasting_engine.whatif.registry import Simulator, register

ID = "strike_air"
VERSION = "1.0"
TESTED = ("FR", "BE", "DE", "IT", "PT", "ES", "FI", "NL")
MAX_STRIKE_SPAN = 60
PARAMS = {"intensity_basis": {"default": "announced", "unit": "announced | estimate",
                              "doc": "announced: the intensity is the cancelled share stated in a notice of the regulator or airport (narrower band); estimate: your own class guess (wider band)"},
          "baseline_flights": {"default": None, "unit": "flights per day",
                               "doc": "optional: the country's scheduled flights per day (your own data); gives flights lost in numbers, otherwise only shares are shown"}}
SOURCES = ["Strike mechanism (docs/methods.md); strike catalogue forecasting_engine/whatif/data/strike_catalogue_2015_2025.csv (dates and announced shares from operators' and regulators' notices)",
           "Back-test series: EUROCONTROL daily IFR flights (back-test only, no level is shown)"]


def _days(sc: dict):
    if sc.get("dates") is not None:
        ds = sorted({MK.parse_date(x) for x in (sc["dates"] if isinstance(sc["dates"], list) else [sc["dates"]])})
        if not ds:
            raise ValueError("scenario.dates is empty")
        if (ds[-1] - ds[0]).days > MAX_STRIKE_SPAN:
            raise ValueError(f"strike dates span more than {MAX_STRIKE_SPAN} days")
        return ds, True
    n = int(sc.get("days") or 1)
    if not 1 <= n <= MAX_STRIKE_SPAN:
        raise ValueError(f"scenario.days must be 1-{MAX_STRIKE_SPAN}")
    s = MK.parse_date(sc.get("start"))
    return [s + dt.timedelta(days=i) for i in range(n)], False


def run(req: dict, ctx: dict) -> dict:
    tm = K.Timer()
    P = MK.params("strike_air")
    sc = dict(req.get("scenario") or {})
    if str(sc.get("kind") or "custom") != "custom":
        raise ValueError("scenario.kind must be custom")
    prm = {k: v["default"] for k, v in PARAMS.items()} | (req.get("params") or {})
    basis = str(prm["intensity_basis"])
    if basis not in ("announced", "estimate"):
        raise ValueError("params.intensity_basis must be announced or estimate")
    a = req.get("area") or {}
    c = a.get("country")
    if c is None:
        raise ValueError('area must be {"country": ["XX", ...]} (ISO-2 codes)')
    ccs = [str(x).upper() for x in ([c] if isinstance(c, str) else c)]
    if not ccs or any(len(x) != 2 or not x.isalpha() for x in ccs):
        raise ValueError("area.country: ISO-2 codes")
    strike_days, explicit = _days(sc)
    inten = MK.per_key(sc.get("intensity"), ccs[0]) if isinstance(sc.get("intensity"), dict) else sc.get("intensity")
    if inten is None:
        raise ValueError("scenario.intensity is required: the announced share of the country's flights cancelled per strike day (0-1)")
    ns = len(strike_days)
    warnings = []
    d0, d1 = strike_days[0], strike_days[-1]
    grid = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    n = len(grid)
    t_h = np.arange(n) * 24.0
    k = P["k"]
    q = P["ratio_realised_over_predicted"][basis]
    series, summary, cols, moments = {}, {}, {}, []
    for cc in ccs:
        ic = MK.per_key(sc.get("intensity"), cc) if isinstance(sc.get("intensity"), dict) else sc.get("intensity")
        if ic is None:
            raise ValueError(f"scenario.intensity has no value for {cc}")
        I = MK.num_list(ic, "intensity", ns, 0.0, 1.0)
        share = np.zeros(n)
        for dd, v in zip(strike_days, I):
            share[(dd - d0).days] = min(max(k * v, 0.0), P["cap"])
        if cc not in TESTED:
            warnings.append(f"{cc} is not among the tested countries ({', '.join(TESTED)}); the mechanism is generic (k = 1) but only those eight were back-tested")
        band = MK.ratio_band(share, q)
        band = {kk: np.clip(v, 0.0, P["cap"]) for kk, v in band.items()}
        series[f"{cc}_flights_lost_share"] = MK.ser("share of the country's scheduled flights", t_h, band)
        cols[f"{cc}_flights_lost_share"] = share
        i = int(np.argmax(share))
        s = dict(peak_loss_share=MK.qd3(band["p10"][i], band["p50"][i], band["p90"][i]), peak_day=grid[i].isoformat(), strike_days=ns,
                 mean_loss_share_on_strike_days=MK.qd3(*(float(np.mean([band[kk][(dd - d0).days] for dd in strike_days])) for kk in ("p10", "p50", "p90"))),
                 catch_up="none inside 3 days (u = 0): rebooked passengers fly on other days' existing flights, so the flights are a pure loss")
        bl = MK.per_key(prm.get("baseline_flights"), cc)
        if bl is not None:
            base = MK.num_list(bl, f"baseline_flights[{cc}]", n, 0, 1e6)
            lost = MK.ratio_band(share * base, q)
            series[f"{cc}_flights_lost"] = MK.ser("flights per day (on the baseline you gave)", t_h, lost)
            cols[f"{cc}_flights_lost"] = share * base
            s["flights_lost_total"] = MK.qd3(float(np.sum(lost["p10"])), float(np.sum(lost["p50"])), float(np.sum(lost["p90"])))
        else:
            s["flights_lost_total"] = "not shown: no baseline_flights given and the EUROCONTROL levels may not be shown (back-test licence); the share of flights lost is the answer"
        summary[cc] = s
        moments.append(MK.moment(i * 24.0, f"{cc}: peak strike day {grid[i].isoformat()}, {100 * share[i]:.0f} % of flights cancelled (p10-p90 {100 * band['p10'][i]:.0f}-{100 * band['p90'][i]:.0f} %)",
                                 min(1.0, share[i])))
    bt = P["backtest"]
    line = (f"Air: {bt['dev']['n'] + bt['check']['n']} strike clusters, {bt['dev']['moved'] + bt['check']['moved']} that moved flights (11 development, 14 check; check set looked at once): check set pooled error 5.7 % "
            f"of the level vs 6.8 % without the strike term (skill +{bt['check']['skill_vs_engine_off']:.2f}, better in {bt['check']['cells_beating_off']} of {bt['check']['n']} cells), median predicted / realised "
            f"{bt['check']['size_median']:.2f} (development {bt['dev']['size_median']:.2f}), direction right in 10 of 10 moved cells on both sets, largest-loss day exact (median offset 0 days), within +-20 % in {bt['check']['within20']:.1f}")
    wi = MK.whatif_block(
        mechanism="Strike calendar -> flights lost per day (air traffic, country level)", status="experimental",
        status_note="the mechanism passed its pre-declared rules on the check set, but every number rests on EUROCONTROL data under a back-test-only licence; it therefore stays experimental until the data terms are settled",
        as_of=P["as_of"], tested_scope="country-level scheduled flights of FR BE DE IT PT ES FI NL, strikes 2017-2025 (air traffic control, airport, airline and general strikes), the announced intensity given",
        backtest=line,
        ranges=f"p10-p90 = the empirical quantiles of realised / predicted loss in the back-test ({q['n']} cells that moved, {basis} intensities: p10 {q['p10']:.2f}, p50 {q['p50']:.2f}, p90 {q['p90']:.2f}) applied to the central loss",
        caveats=[
            "Intensity in, loss out: it needs the announced share of flights cancelled. Class guesses (Italy, Portugal, Belgium 2022) were off by factors of 0.4-3; the verified-notice subset is the better guide to the size (skill +0.30 on 6 check cells).",
            "General strikes in Italy moved flights only 4-6 % (flights are largely exempt) against 15 % predicted: do not enter the share of strikers as the share of flights.",
            "Counts movements at the country's airports; French ATC overflights are not modelled. No catch-up inside three days (u about 0).",
            "Two check events (Belgium 14 Oct 2025, France 3-4 Jul 2025) were seen before the check look; one development event (Italy 2021) and three check cells did not move.",
            "The lab's own air_d champion gate was not qualified (+0.0001, 2 of 4 blocks): this adds no forecast skill to the standing air_d family, it is a what-if given the announced share.",
            "Knock-ons are not claimed: industrial output and retail are not measurable at monthly frequency from one-day strikes; diesel demand and rail freight have no data.",
            "Ports under strike are NOT covered: the port mechanism failed its check set (skill -0.095, size ratio 1.88, timing 9 days)."],
        licence="Back-tests used EUROCONTROL daily flights (copying allowed with the source named, not for commercial purposes, no modification without written permission). No EUROCONTROL value is shown here; "
                "ask EUROCONTROL before any paid use or trial.",
        not_tested=["Port and rail-freight stoppages (port mechanism failed)", "Passenger numbers, delays and rebooking", "Countries outside the eight tested"],
        links_not_wired=["Ports under strikes (failed on the check set)"])
    rule = ("Pre-declared rules: skill above the same baseline without the strike term and above the best open model; direction >= 0.80 of moved cells; median size ratio 0.8-1.25; "
            "median timing offset <= 1 day. Air passed all four on the check set; the licence keeps the status experimental.")
    cases = [{"name": f"{c_['event']} ({c_['cc']})", "metric": "realised / predicted loss share", "observed": 1.0, "simulated": round(c_["realised_over_predicted"], 2),
              "error": round(c_["realised_over_predicted"] - 1, 2)} for c_ in P["cells"][:8]]
    return MK.result(ID, VERSION, tm, wi, series, summary, MK.timeline(t_h, cols), moments, MK.validation(rule, line, cases),
                     assumptions=["Loss on strike day j = k x intensity_j with k = 1, capped at 0.98; days without a strike are not changed.", "Intensity is the share of the country's scheduled flights cancelled, as announced."],
                     warnings=warnings, sources=SOURCES)


register(Simulator(id=ID, name="Strikes to air traffic", domain="logistics", hazards=[], run=run, scenarios=["custom"], params=PARAMS, status="experimental",
                   description="A strike calendar with the announced share of flights cancelled -> flights lost per day in the country, with p10-p90 (experimental: back-test data under a research-only licence, relative effect only)",
                   max_horizon_h=24 * MAX_STRIKE_SPAN))
