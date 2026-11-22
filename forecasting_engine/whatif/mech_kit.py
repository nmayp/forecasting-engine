"""Shared helpers of the back-tested what-if mechanisms (heat -> load, strikes -> air traffic, rates -> borrowing cost, oil -> pump prices -> HICP,
chokepoints -> freight): forecasting_engine/whatif/mech_heat_load.py, mech_strike_air.py, mech_rates.py, mech_oil_prices.py, mech_choke_freight.py.

These are not hazard simulators: the user supplies the driver path (a temperature path, a strike calendar, a policy-rate path, an oil path, a transit cut) and
gets the mechanism's response. Each was tested in the lab (docs/methods.md) and the answer says how far: every result carries

    meta.whatif = {mechanism, status, status_note, as_of, run_date, tested_scope, backtest (ONE line), caveats [..], ranges, licence, not_tested, links_not_wired}

next to the usual p10 / p50 / p90 series. The parameters come from small JSON files built by forecasting_engine.whatif.backtests.build (forecasting_engine/whatif/data/),
so the deployed image needs no lab store. Statuses use the registry's words: validated = passed its pre-registered rule on held-out events; screening =
plausible order of magnitude; experimental = failed part of its rule or not yet tested on held-out events (docs/methods.md).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from forecasting_engine.whatif import kit as K

DATA = K.DATA
Z90 = 1.2815515655446004                      # p10 / p90 of a standard normal
MAX_DAYS = 120


@lru_cache(maxsize=None)
def params(name: str) -> dict:
    try:
        return json.loads((DATA / f"{name}.json").read_text())
    except FileNotFoundError as e:
        raise LookupError(f"parameter file {name}.json is missing: run `python -m forecasting_engine.whatif.backtests.build`") from e


def run_date() -> str:
    return dt.date.today().isoformat()


def parse_date(v, default: dt.date | None = None) -> dt.date:
    if v is None or v == "":
        return default or dt.date.today() + dt.timedelta(days=1)
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except ValueError:
        raise ValueError(f"{v!r} is not a date (use YYYY-MM-DD)")


def num_list(v, name: str, n: int | None = None, lo: float | None = None, hi: float | None = None, max_n: int = MAX_DAYS) -> np.ndarray:
    """A number or a list of numbers -> float array (a scalar is repeated n times when n is given); finite and within [lo, hi]."""
    if v is None:
        raise ValueError(f"{name} is required")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if n is None:
            raise ValueError(f"{name}: give a list of numbers")
        a = np.full(n, float(v))
    elif isinstance(v, (list, tuple)) and v:
        try:
            a = np.array([float(x) for x in v], float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numbers")
    else:
        raise ValueError(f"{name} must be a number or a list of numbers")
    if len(a) > max_n:
        raise ValueError(f"{name}: at most {max_n} values")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite numbers")
    if lo is not None and a.min() < lo or hi is not None and a.max() > hi:
        raise ValueError(f"{name} must be within [{lo}, {hi}]")
    if n is not None and len(a) != n:
        raise ValueError(f"{name}: {len(a)} values, expected {n}")
    return a


def per_key(v, key: str):
    """A value given per country / series ({"ES": ..., "IT": ...}) or for all (anything else)."""
    if isinstance(v, dict):
        return v.get(key, v.get(key.lower()))
    return v


def codes(area: dict | None, allowed, what: str = "country") -> list[str]:
    """area {"country": ["ES", ...]} -> upper-case codes, each within `allowed`."""
    a = area or {}
    c = a.get("country")
    if c is None:
        raise ValueError(f'area must be {{"country": ["XX", ...]}} (this what-if is national; supported: {", ".join(sorted(allowed))})')
    c = [c] if isinstance(c, str) else list(c)
    out = [str(x).upper() for x in c]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"{what} {', '.join(bad)} not covered; supported: {', '.join(sorted(allowed))}")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------------------------ ranges
def mult_band(central, sigma: float) -> dict:
    """p10 / p50 / p90 of a central effect with a multiplicative (log-normal) error of the given sigma: central x exp(-+Z90 sigma); symmetric in sign."""
    c = np.asarray(central, float)
    f = math.exp(Z90 * sigma)
    lo, hi = c / f, c * f
    return {"p10": np.minimum(lo, hi), "p50": c, "p90": np.maximum(lo, hi)}


def ratio_band(central, q: dict) -> dict:
    """p10 / p50 / p90 of an effect from the empirical quantiles of realised / predicted in the back-test (q = {p10, p50, p90}); p50 is the central x q50."""
    c = np.asarray(central, float)
    lo, md, hi = c * q["p10"], c * q["p50"], c * q["p90"]
    return {"p10": np.minimum(lo, hi), "p50": md, "p90": np.maximum(lo, hi)}


# ------------------------------------------------------------------------------------------------ calibrated ranges
# forecasting_engine.lab.calib_core / calib_run fit these on development events only (leave-one-event-out conformal residuals, scaled by the layer's normal spread, the shock size
# and the horizon) and write them next to the other parameters (P["calib"][key]); docs/methods.md.
ZP = np.round(np.arange(0.02, 0.981, 0.02), 2)


def cal_mode(P: dict, key: str, mode: str = "chain") -> dict | None:
    """The frozen calibration of one mechanism and information mode ('chain': the layer is predicted from the previous link's prediction; 'given': its upstream is given), or None."""
    c = (P.get("calib") or {}).get(key)
    if not c:
        return None
    m = c["modes"].get(mode) or c["modes"].get("chain")
    md = mode if mode in c["modes"] else "chain"
    return dict(m, ext=c["ext"], unit=c["unit"], compounding=c.get("compounding"), key=key, mode=md, meta=(c.get("meta") or {}).get(md))


def _row(m: dict, which: str, name: str, k):
    tab = m[which]
    row = tab.get(name) or tab["*"]
    star = tab["*"]
    row = [star[i] if v is None else v for i, v in enumerate(row)]
    n = len(row)
    k = np.asarray(k, int)
    v = np.array(row, float)[np.clip(k, 0, n - 1)]
    if m["ext"] == "sqrt":
        v = np.where(k > n - 1, v * np.sqrt((k + 1) / n), v)
    return v


def cal_band(central, m: dict, name: str, k=None):
    """(band {p10, p50, p90}, s): p10 / p90 = central + s x the frozen z-quantiles, s = (floor + b |central|)(1 + c h): floor = the layer's normal spread at that step (per
    series), b = growth with the shock size, c = growth with the horizon. p50 stays the central effect. Units of `central` must be those of the table (m['unit'])."""
    c = np.asarray(central, float)
    k = np.arange(len(c)) if k is None else np.asarray(k, int)
    fl = _row(m, "floor", name, k)
    hn = np.minimum(np.maximum(k, 0) / max(m["hsteps"] - 1, 1), 1.0)
    s = (fl + m["b"] * np.abs(c)) * (1 + m["c"] * hn)
    z10, z90 = np.interp([0.1, 0.9], ZP, m["zq"])
    return {"p10": np.minimum(c + s * z10, c), "p50": c, "p90": np.maximum(c + s * z90, c)}, s


def exceed_prob(central, s, m: dict, name: str, k=None) -> np.ndarray:
    """P(the layer leaves its normal band) = P(|central + s z| > q), q = the 90th percentile of |placebo change| of that series and step over the 3 years before the origin
    (stored per series), z from the frozen normalised-residual quantiles; Brier-scored in the benchmark (docs, "Calibrated what-if ranges")."""
    c = np.asarray(central, float)
    k = np.arange(len(c)) if k is None else np.asarray(k, int)
    q = _row(m, "qn", name, k)
    if m.get("exceed") == "normal":          # where the calibrated distribution did not beat the benchmark's normal-spread probability on development events
        sg = _row(m, "floor", name, k)
        cdf = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))
        return np.clip(1.0 - (cdf((q - c) / sg) - cdf((-q - c) / sg)), 0.0, 1.0)
    zq = np.asarray(m["zq"], float)
    lo = np.interp((-q - c) / s, zq, ZP, left=0.01, right=0.99)
    hi = np.interp((q - c) / s, zq, ZP, left=0.01, right=0.99)
    return np.clip(1.0 - (hi - lo), 0.0, 1.0)


def calib_info(label_modes: dict) -> dict | None:
    """{label: cal_mode} -> the block every calibrated result carries in meta.whatif.calibration: how the band was made and how it scored (development leave-one-event-out, check one look)."""
    out = {}
    for label, m in label_modes.items():
        if not m:
            continue
        out[label] = dict(method="conformal on leave-one-event-out residuals of the development events, scaled by the layer's normal spread, the shock size and the horizon; frozen, one check look",
                          mode=m["mode"], coverage_target=0.8, compounding_vs_upstream_given=m.get("compounding"), **(m.get("meta") or {}))
    return out or None


def calib_text(label_modes: dict) -> str:
    parts = []
    for label, m in label_modes.items():
        if not m or not m.get("meta"):
            continue
        d, c = m["meta"].get("dev"), m["meta"].get("check")
        t = f"{label}: {d['coverage_after']:.2f} of development values inside (was {d['coverage_before']:.2f}; leave-one-event-out, {d['n_events']} events)"
        if c:
            t += f", {c['coverage_after']:.2f} on the check set (was {c['coverage_before']:.2f}; {c['n_events']} event{'s' if c['n_events'] != 1 else ''}, one look)"
        parts.append(t)
    return "; ".join(parts)


def prob_ser(t_h, p) -> dict:
    return {"unit": "probability (0-1) that the layer ends outside its normal 90 % band", "t_h": [K._f(x) for x in np.asarray(t_h, float)], "p10": [K._f(x) for x in p],
            "p50": [K._f(x) for x in p], "p90": [K._f(x) for x in p]}


def ser(unit: str, t_h, band: dict) -> dict:
    t = [K._f(x) for x in np.asarray(t_h, float)]
    return {"unit": unit, "t_h": t, "p10": [K._f(x) for x in band["p10"]], "p50": [K._f(x) for x in band["p50"]], "p90": [K._f(x) for x in band["p90"]]}


def qd3(a_p10, a_p50, a_p90) -> dict:
    return {"p10": K._f(a_p10), "p50": K._f(a_p50), "p90": K._f(a_p90)}


# ------------------------------------------------------------------------------------------------ result
def whatif_block(mechanism: str, status: str, status_note: str, as_of: str, tested_scope: str, backtest: str, caveats: list, ranges: str,
                 licence: str | None = None, not_tested: list | None = None, links_not_wired: list | None = None, **extra) -> dict:
    """The block every back-tested what-if returns in meta.whatif (and run_simulation passes through): status, as_of, ONE back-test line, caveats, ranges."""
    b = dict(mechanism=mechanism, status=status, status_note=status_note, as_of=as_of, run_date=run_date(), tested_scope=tested_scope, backtest=backtest,
             caveats=list(caveats), ranges=ranges, doc="docs/methods.md")
    if licence:
        b["licence"] = licence
    if not_tested:
        b["not_tested"] = list(not_tested)
    if links_not_wired:
        b["links_not_wired"] = list(links_not_wired)
    b.update(extra)
    return b


def result(sim: str, version: str, timer, wi: dict, series: dict, summary: dict, frames: list, moments: list, validation: dict,
           assumptions=(), warnings=(), sources=(), label: str | None = None, **extra) -> dict:
    """Registry-shaped result (forecasting_engine.whatif.registry): meta (status, assumptions, warnings, sources, whatif), frames (timeline), series, moments, summary, validation."""
    m = K.meta(sim, version, timer, wi["status"], assumptions=assumptions, warnings=list(warnings), sources=sources, whatif=wi,
               label=label or wi["mechanism"], as_of=wi["as_of"])
    out = {"meta": m, "frames": frames, "series": series, "moments": moments, "summary": summary, "validation": validation}
    out.update(extra)
    return out


def validation(rule: str, backtest_line: str, cases: list | None = None, adopted: bool = False) -> dict:
    return {"cases": cases or [], "rule": rule, "adopted": adopted, "note": backtest_line}


def timeline(t_h, columns: dict) -> list:
    """Non-spatial frames [{t_h, values: {name: number}}] from per-name arrays."""
    t = list(np.asarray(t_h, float))
    return [{"t_h": K._f(t[i]), "values": {k: K._f(v[i]) for k, v in columns.items() if i < len(v) and np.isfinite(v[i])}} for i in range(len(t))]


def moment(t_h: float, label: str, severity: float = 0.5) -> dict:
    return {"t_h": K._f(t_h), "label": label, "severity": K._f(severity)}


def rnd(x, nd: int = 2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)
