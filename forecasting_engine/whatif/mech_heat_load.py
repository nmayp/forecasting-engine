"""`heat_load`: a temperature path -> electricity demand, by country (energy system response to heat, and to cold). Back-tested mechanism of
docs/methods.md (forecasting_engine/lab/heat_load.py, second pass: 10 annual harmonics, 7-day level anchor).

    log load(t) = ... + sh * HDD(Te) + sc * CDD(Te),   Te(t) = (T(t) + 0.5 T(t-1) + 0.25 T(t-2)) / 1.75   (building inertia)
    HDD = max(0, Th - Te), CDD = max(0, Te - Tc), Th and Tc the country's balance points, sh and sc the heating and cooling slopes (share of load per degree C)

The user gives a daily population-weighted mean temperature path per country (or an anomaly on the normal) and, optionally, the reference path to compare with
(default: the day-of-year normal of the same effective temperature). The answer is the load the TEMPERATURE adds (or removes) against that reference, per day and
per local hour, in % with p10-p90, and in MW when a baseline is known (the caller's own, or the median day-ahead forecast daily peak of the same ISO week and day
type, ENTSO-E free re-use data). It is not a load forecast: calendar and season are not in the answer (the validated size is the lab's total excess, see caveats).

Request: area {"country": ["ES", ...]} (ES PT FR DE IT NL BE PL);
  scenario {"kind": "custom", "tmean_c": [daily means] | {"ES": [...]}, "start": "YYYY-MM-DD", "reference_c": scalar | list | {cc: ..} (optional), "pre_c": [2 days before] (optional),
            "baseline_mw": scalar | list | {cc: ..} (optional),
            "wind_mw": / "solar_mw": scalar | list | {cc: ..} (optional; daily mean generation in MW: CONTEXT only, see below)}
           or {"kind": "anomaly", "anomaly_c": scalar | list | {cc: ..}, "days": n, "start": ...}: the normal plus an anomaly (a scalar is held for `days` days)
  params {"hourly": true|false}

Wind and solar generation paths do NOT enter the demand mechanism (the temperature response is a demand response and wind / solar do not move demand): when given, they are shown
as context only: the generation you supplied and, where a baseline_mw is given, the arithmetic residual load = baseline + temperature excess - wind - solar (no band, not tested).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from forecasting_engine.whatif import kit as K
from forecasting_engine.whatif import mech_kit as MK
from forecasting_engine.whatif.registry import Simulator, register

ID = "heat_load"
VERSION = "1.0 (second pass, 22 Oct 2026)"
CCS = ("ES", "PT", "FR", "DE", "IT", "NL", "BE", "PL")
NORMS = K.DATA / "entsoe_live_norms.json"
PARAMS = {"hourly": {"default": True, "unit": "bool", "doc": "also return the excess by local hour of the day (paths of up to 14 days)"}}
MAX_HOURLY_DAYS = 14

SOURCES = ["Heat-load mechanism (docs/methods.md); fitted on ENTSO-E actual load (restricted: coefficients only, no load values) and population-weighted ERA5-type temperature",
           "ENTSO-E Transparency Platform (CC BY 4.0), day-ahead total load forecast norms: only for the MW scale when no baseline is given"]


def _norms() -> dict:
    try:
        return json.loads(NORMS.read_text()).get("load_da", {})
    except (OSError, ValueError):
        return {}


def _eff(T: np.ndarray, pre: np.ndarray, w) -> np.ndarray:
    """Effective temperature with the inertia weights: Te(t) = sum w_i T(t-i) / sum w; `pre` = the days before the first (oldest first)."""
    full = np.r_[pre, T]
    k = len(w) - 1
    return sum(wi * full[k - i:len(full) - i] for i, wi in enumerate(w)) / sum(w)


def _path(sc: dict, cc: str, n: int | None, dates: list, clim: np.ndarray, kind: str):
    """(T path, pre values) of the scenario for one country."""
    if kind == "anomaly":
        an = MK.per_key(sc.get("anomaly_c"), cc)
        a = MK.num_list(an, "anomaly_c", n, -15, 25)
        return np.array([clim[min(d.timetuple().tm_yday, 366) - 1] for d in dates]) + a, None
    t = MK.per_key(sc.get("tmean_c"), cc)
    if t is None:
        raise ValueError(f"scenario.tmean_c has no path for {cc}")
    T = MK.num_list(t, f"tmean_c[{cc}]", n, -40, 55)
    pre = MK.per_key(sc.get("pre_c"), cc)
    return T, (MK.num_list(pre, "pre_c", 2, -40, 55) if pre is not None else None)


def run(req: dict, ctx: dict) -> dict:
    tm = K.Timer()
    P = MK.params("heat_load")
    sc = dict(req.get("scenario") or {})
    kind = str(sc.get("kind") or "custom")
    if kind not in ("custom", "anomaly"):
        raise ValueError("scenario.kind must be custom or anomaly")
    ccs = MK.codes(req.get("area"), CCS)
    prm = {k: v["default"] for k, v in PARAMS.items()} | (req.get("params") or {})
    start = MK.parse_date(sc.get("start"))
    if kind == "anomaly":
        an = sc.get("anomaly_c")
        first = next(iter(an.values())) if isinstance(an, dict) and an else an
        if isinstance(first, (list, tuple)):
            n = len(first)
        else:
            n = int(sc.get("days") or 7)
    else:
        tt = sc.get("tmean_c")
        first = next(iter(tt.values())) if isinstance(tt, dict) and tt else tt
        if not isinstance(first, (list, tuple)) or not first:
            raise ValueError("scenario.tmean_c must be a daily list of mean temperatures (degC) or {country: list}")
        n = len(first)
    if not 1 <= n <= MK.MAX_DAYS:
        raise ValueError(f"the path must have 1-{MK.MAX_DAYS} days")
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    t_h = np.arange(n) * 24.0
    w = P["inertia_w"]
    sigma = P["band"]["sigma_log"]
    cal = MK.cal_mode(P, "heat_load", "chain")
    norms = _norms()
    series, summary, cols, moments, warnings = {}, {}, {}, [], []
    hourly = bool(prm.get("hourly")) and n <= MAX_HOURLY_DAYS
    for cc in ccs:
        c = P["countries"][cc]
        clim = np.array(c["clim_te"], float)
        T, pre = _path(sc, cc, n, dates, clim, kind)
        pre = np.full(len(w) - 1, T[0]) if pre is None else pre
        Te = _eff(T, pre, w)
        if Te.max() > c["tc"] + 14 or T.max() > 35 or T.min() < -15:
            warnings.append(f"{cc}: the path goes beyond the range of the five tested heat waves (population-weighted daily means up to about 32 degC): the response is extrapolated linearly")
        ref = MK.per_key(sc.get("reference_c"), cc)
        if ref is None:
            Tr = np.array([clim[min(d.timetuple().tm_yday, 366) - 1] for d in dates])        # the normal of the effective temperature (already smoothed)
            hdd_r, cdd_r = np.maximum(0, c["th"] - Tr), np.maximum(0, Tr - c["tc"])
            ref_txt = "day-of-year normal of the effective temperature"
        else:
            R = MK.num_list(ref, f"reference_c[{cc}]", n, -40, 55)
            Re = _eff(R, np.full(len(w) - 1, R[0]), w)
            hdd_r, cdd_r = np.maximum(0, c["th"] - Re), np.maximum(0, Re - c["tc"])
            ref_txt = "the reference path given"
        hdd, cdd = np.maximum(0, c["th"] - Te), np.maximum(0, Te - c["tc"])
        lx = c["sh"] * (hdd - hdd_r) + c["sc"] * (cdd - cdd_r)
        pct = 100 * np.expm1(lx)
        if cal:
            band, s_u = MK.cal_band(pct, cal, cc)
            pe = MK.exceed_prob(pct, s_u, cal, cc)
            series[f"{cc}_p_load_leaves_normal_band"] = MK.prob_ser(t_h, pe)
        else:
            band, pe = MK.mult_band(pct, sigma), None
        series[f"{cc}_excess_pct"] = MK.ser("% of the daily mean load (against " + ref_txt + ")", t_h, band)
        cols[f"{cc}_excess_pct"] = pct
        i = int(np.argmax(np.abs(pct)))
        # MW scale: the caller's baseline, else the median day-ahead forecast daily peak (same ISO week and day type)
        bl = MK.per_key(sc.get("baseline_mw"), cc)
        if bl is not None:
            base = MK.num_list(bl, f"baseline_mw[{cc}]", n, 1, 500000)
            basis = "baseline_mw given by the caller (daily mean load)"
        else:
            base = np.full(n, np.nan)
            for j, d in enumerate(dates):
                q = norms.get(cc, {}).get(f"{d.isocalendar()[1]}:{'we' if d.weekday() >= 5 else 'wd'}")
                if q:
                    base[j] = q["q"][5]
            basis = "median day-ahead forecast daily PEAK of the same ISO week and day type, 2021-25 (ENTSO-E, CC BY 4.0); public holidays count as working days" if np.isfinite(base).any() else None
        if basis:
            mw = {k_: v_ / 100 * base for k_, v_ in band.items()} if cal else MK.mult_band(pct / 100 * base, sigma)
            series[f"{cc}_excess_mw"] = MK.ser("MW (" + ("on the baseline given" if bl is not None else "at the daily peak, approximate") + ")", t_h, mw)
            cols[f"{cc}_excess_mw"] = pct / 100 * base
        s = dict(peak_excess_pct=MK.qd3(band["p10"][i], band["p50"][i], band["p90"][i]), peak_day=dates[i].isoformat(), mean_excess_pct=MK.qd3(*(float(np.mean(band[k])) for k in ("p10", "p50", "p90"))),
                 days_with_excess_over_1pct=int(np.sum(pct > 1.0)), prob_load_leaves_normal_band_at_peak=(MK.rnd(float(pe[i]), 3) if pe is not None else None), balance_points_c=dict(heating_below=c["th"], cooling_above=c["tc"]),
                 slopes_pct_per_degC=dict(cooling=MK.rnd(100 * np.expm1(c["sc"]), 2), heating=MK.rnd(100 * np.expm1(c["sh"]), 2)), baseline_basis=basis)
        if bl is not None:
            gwh = pct / 100 * base * 24 / 1000
            gb = ({k_: float(np.sum(v_ / 100 * base * 24 / 1000)) for k_, v_ in band.items()} if cal else MK.mult_band(float(np.sum(gwh)), sigma))      # calibrated: days comonotone
            s["cumulative_excess_gwh"] = MK.qd3(float(gb["p10"]), float(gb["p50"]), float(gb["p90"]))
        if hourly:
            sh_h, sc_h = np.array(c["sh_h"]), np.array(c["sc_h"])
            hh = [100 * np.expm1(sh_h[h] * (hdd - hdd_r) + sc_h[h] * (cdd - cdd_r)) for h in range(24)]       # (24 local hours) x (days)
            flat = np.array([hh[h][d] for d in range(n) for h in range(24)])
            tt = np.array([d * 24.0 + h for d in range(n) for h in range(24)])
            hb = MK.mult_band(flat, sigma)
            if cal:                                              # the daily band's relative widening applied to each hour of that day
                dd = np.repeat(np.arange(n), 24)
                with np.errstate(all="ignore"):
                    up = np.where(np.abs(pct) > 1e-9, (band["p90"] - pct) / np.where(np.abs(pct) > 1e-9, pct, 1.0), 0.0)
                    dn = np.where(np.abs(pct) > 1e-9, (band["p10"] - pct) / np.where(np.abs(pct) > 1e-9, pct, 1.0), 0.0)
                a_, b_ = flat * (1 + dn[dd]), flat * (1 + up[dd])
                hb = {"p10": np.minimum(a_, b_), "p50": flat, "p90": np.maximum(a_, b_)}
            series[f"{cc}_hourly_excess_pct"] = MK.ser("% of the load of that local hour", tt, hb)
            j = int(np.argmax(np.abs(flat)))
            s["hour_of_peak_excess_local"] = int(j % 24)
        gen = {k: MK.per_key(sc.get(f"{k}_mw"), cc) for k in ("wind", "solar")}
        if any(v is not None for v in gen.values()):
            g = {k: MK.num_list(v, f"{k}_mw[{cc}]", n, 0, 500000) for k, v in gen.items() if v is not None}
            ctx = dict(status="context", note="wind / solar generation does not enter the demand mechanism and was not tested; shown as supplied, with the arithmetic residual load where a baseline_mw is given",
                       **{f"{k}_mean_mw": MK.rnd(float(np.mean(v)), 1) for k, v in g.items()})
            for k, v in g.items():
                series[f"{cc}_{k}_context_mw"] = MK.ser(f"MW {k} generation as supplied by the caller (context; not modelled, no band)", t_h, {"p10": v, "p50": v, "p90": v})
                cols[f"{cc}_{k}_context_mw"] = v
            if bl is not None:
                res_mw = base + pct / 100 * base - sum(g.values())
                series[f"{cc}_residual_load_context_mw"] = MK.ser("MW: baseline_mw + temperature excess - wind - solar supplied (arithmetic context; no band, not tested)", t_h, {"p10": res_mw, "p50": res_mw, "p90": res_mw})
                cols[f"{cc}_residual_load_context_mw"] = res_mw
                ctx["residual_load_mean_mw"] = MK.rnd(float(np.mean(res_mw)), 1)
                ctx["residual_load_peak_mw"] = MK.rnd(float(np.max(res_mw)), 1)
            else:
                warnings.append(f"{cc}: wind / solar given without baseline_mw: the generation is shown but no residual load is computed (the default MW scale is a daily peak, not a mean)")
            s["generation_context"] = ctx
            warnings.append(f"{cc}: wind_mw / solar_mw are context only: the heat-to-load mechanism is a demand response and does not use them")
        summary[cc] = s
        moments.append(MK.moment(i * 24.0, f"{cc}: largest temperature effect on load ({pct[i]:+.1f} %, p10-p90 {band['p10'][i]:+.1f} to {band['p90'][i]:+.1f} %) on {dates[i].isoformat()}",
                                 min(1.0, abs(pct[i]) / 20)))
    bt = P["backtest"]
    line = (f"{len(bt['events'])} heat-wave tests (2017 Jun and Aug, 2019, 2022, 2023; 8 countries; the realised temperature given): median predicted / realised excess "
            f"{min(bt['median_size_ratio']):.2f}-{max(bt['median_size_ratio']):.2f} (mean {np.mean(bt['median_size_ratio']):.2f}), within +-10 % in {np.mean(bt['within10']):.2f} and within +-20 % "
            f"in {np.mean(bt['within20']):.2f} of countries, direction right in {np.mean(bt['direction_right']):.2f}; mean skill {np.mean(bt['skill_scen']):.2f} over the strict reference (engine 0.39 on 4 events); development events only")
    wi = MK.whatif_block(
        mechanism="Temperature path -> electricity demand (degree-day response per country)", status="screening",
        status_note="size good to about +-20 % on five development heat waves (no check set); tuned on those events (10 harmonics, 7-day anchor); lab champion load_d by holdout +0.0035 only",
        as_of=P["as_of"], tested_scope="8 countries (ES PT FR DE IT NL BE PL), daily mean load and local-hour profile, heat waves of June-August 2017-2023, the realised temperature given",
        backtest=line, ranges=((f"p10-p90 CALIBRATED (80 % target): central effect + conformal quantiles x (the layer's normal daily-load spread + b x |effect|); {MK.calib_text({'load': cal})}; the ratio between the lab's total excess and "
                 "the temperature-only part shown here is not in the band; series *_p_load_leaves_normal_band = probability that the daily load ends outside its normal 90 % band; " if cal else
                 f"p10-p90 = the central temperature effect x exp(-+1.28 x {sigma:.2f}) (log-normal error calibrated on the share of country-events within +-20 % of the realised excess); ")
                + "temperature-path error is the caller's assumption and is not in the band"),
        caveats=[
            "A size tool good to roughly +-20 %, not a metering tool: fewer than a third of the countries came within +-5 % of the realised excess.",
            "The validated size is the lab's total excess against a frozen pre-shock week (season and calendar included). This what-if returns only the part the temperature adds against the reference "
            "(default: the normal). The doc states the temperature response itself was not shown to be sized right; the excess is mostly season and calendar.",
            "Tuned on its test events: the number of annual harmonics (10) and the anchor length (7 d) were chosen on the five development heat waves; there is no check set for heat waves.",
            "Known regressions: in June 2017 the peak day of the excess was 6.5 days off; as a forecast without a weather forecast it scores 0.31 against the engine's 0.39. The temperature path is yours, not a forecast.",
            "Hourly profile: the daily peak hour is exact in 55-76 % and within one hour in 77-91 % of days; one regression per local hour with the daily balance points.",
            "Covers 8 countries only (the ENTSO-E load in the lab store); heating response (cold) uses the same fit and was not tested separately.",
            "Calendar effects (public holidays, the August shutdown) are not in the answer; the MW scale without a baseline is the median day-ahead DAILY PEAK, so an approximate MW at the peak hour."],
        licence="Coefficients were fitted on ENTSO-E actual load, which is not on ENTSO-E's free re-use list (back-tests only 18 Oct 2026). The answer holds percentages and "
                "coefficients, no load values. Whether a model trained on restricted values may publish its outputs is an open question: check ENTSO-E's terms before any wider use.",
        not_tested=["Heat to excess deaths (not part of this mechanism: see the heat_health simulator)", "Countries outside the eight", "Temperature paths beyond 120 days"],
        fit_data_before=P["fit_data_before"], calibration=MK.calib_info({"load": cal}))
    rule = "Pre-registered size criterion of the lab test: median predicted / realised excess 0.8-1.25 and direction right; passed on the five development events (partly on the +-10 % share)."
    cases = [{"name": e, "metric": "median predicted / realised excess (1 = exact)", "observed": 1.0, "simulated": r, "error": round(r - 1, 3)}
             for e, r in zip(bt["events"], bt["median_size_ratio"])]
    return MK.result(ID, VERSION, tm, wi, series, summary, MK.timeline(t_h, cols), moments, MK.validation(rule, line, cases),
                     assumptions=[f"Effective temperature Te = (T + 0.5 T-1 + 0.25 T-2) / 1.75; the first days use pre_c or the first value repeated.",
                                  f"Reference: {ref_txt}.", "Coefficients fitted before " + P["fit_data_before"] + "; load assumed to move by the same share at every hour except where the hourly slopes differ."],
                     warnings=warnings, sources=SOURCES)


register(Simulator(id=ID, name="Heat waves to electricity demand", domain="infrastructure", hazards=[], run=run, scenarios=["custom", "anomaly"], params=PARAMS, status="screening",
                   description="A temperature path per country -> the extra (or lower) electricity demand it causes, per day and local hour, with p10-p90 (back-tested on five heat waves, size good to about +-20 %)",
                   max_horizon_h=24 * MK.MAX_DAYS))
