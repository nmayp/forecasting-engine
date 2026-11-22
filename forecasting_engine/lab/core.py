"""Lab v2 core: origins, what was known at each origin, the strict reference, features, pooled models, scoring, the ledger.

    python -m forecasting_engine.lab.core register                 # once: windows per family (refuses to rewrite)
    python -m forecasting_engine.lab.core run --method M --families a,b [--holdout] [--note ..]
    python -m forecasting_engine.lab.core report

Same rules as lab v1 (docs/methods.md): frozen data (forecasting_engine.lab.panel), windows registered once, development first, one
counted holdout look for a variant that beats the champion on development, champion only if it wins on both, every run logged.

What changes from v1: every country (or currency, commodity) of a family is scored, and the models can learn across them ("pooled"); the
histories go back to 2000-2015; every driver enters as it was known at the origin, with its publication delay. A family's score is the pooled
skill: 1 - sum of scaled absolute errors / the same for the strict reference, over every series, origin and step of the window (each series'
errors scaled by its own typical one-step change, so a large country does not dominate). The median of the per-series skills is reported too.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.lab import panel as PN

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(os.environ.get("FE_LAB_DIR") or ROOT / "results" / "lab")
REGISTRY = LAB / "registry.json"
TRIALS = LAB / "trials.jsonl"
REPORT = LAB / "report.json"
DEV_FROM, HOLDOUT_SHARE = 0.4, 0.2
MAX_EVAL = 240                    # evaluation origins per window (evenly spaced; dense enough for stable family scores)
QS = (0.1, 0.5, 0.9)
ORIGIN_OFFSET_H = {"load_d": 12, "fx_d": 16, "commod_d": 12, "kev_d": 12, "diesel_w": 108, "gasoline_w": 108, "unrest_w": 108}
REFIT = {"h": pd.Timedelta(days=182), "D": pd.Timedelta(days=182), "W": pd.Timedelta(days=364), "M": pd.Timedelta(days=365)}
SKIP = {"M": -1}                  # monthly: forecast the origin's month and the next H-1 (nowcast + forecast)
YEAR_ROWS = {"h": 364 * 24, "D": 364, "W": 52, "M": 12}
LOOKBACK = {"h": pd.Timedelta(days=7), "D": pd.Timedelta(days=7), "W": pd.Timedelta(days=28), "M": pd.Timedelta(days=91)}


# ------------------------------------------------------------------------------------------------------------------------------ time and knowledge
def known_times(stamps: pd.DatetimeIndex, avail_h: float, dayahead_hour) -> np.ndarray:
    """When each timestamp's value is known (int64 ns)."""
    if dayahead_hour is not None:
        t = stamps.floor("D") - pd.Timedelta(hours=24 - int(dayahead_hour))
    else:
        t = stamps + pd.Timedelta(hours=float(avail_h))
    return t.as_unit("ns").asi8


def origins_of(fam: PN.Family) -> pd.DatetimeIndex:
    ix = fam.Y.index
    if fam.freq == "h":
        return ix[ix.hour == fam.origin_hour]
    return ix + pd.Timedelta(hours=ORIGIN_OFFSET_H.get(fam.id, 0))


def _ffill_idx(v: np.ndarray) -> np.ndarray:
    """Per row, the index of the last finite value at or before it (-1 if none)."""
    ok = np.isfinite(v)
    idx = np.where(ok, np.arange(len(v)), -1)
    return np.maximum.accumulate(idx)


class Design:
    """Everything known at every origin, for one family: target rows, references, features."""

    def __init__(self, fam: PN.Family):
        self.fam = fam
        self.Y = fam.Y.to_numpy(float)
        self.stamps = pd.DatetimeIndex(fam.Y.index)
        self.T, self.J = self.Y.shape
        self.series = [str(c) for c in fam.Y.columns]
        self.skip = SKIP.get(fam.freq, fam.skip)
        self.H = fam.H
        self.kt = known_times(self.stamps, fam.avail_h, fam.dayahead_hour)
        orr = origins_of(fam)
        self.tau = orr.as_unit("ns").asi8
        st = self.stamps.as_unit("ns").asi8
        self.p0 = np.searchsorted(st, self.tau, "right") - 1                      # the origin's period
        self.L = np.searchsorted(self.kt, self.tau, "right") - 1                  # last row known at the origin (any series)
        self.R = self.p0[:, None] + self.skip + np.arange(1, self.H + 1)[None, :]  # target rows
        self.ok_origin = (self.L >= 0) & (self.R[:, -1] < self.T)
        self.ffi = np.stack([_ffill_idx(self.Y[:, j]) for j in range(self.J)], 1)
        tr_end = int(self.T * DEV_FROM)
        if fam.span_min:                                     # series that start late: each one's scale from its own first 40 % of data
            Ys = np.full_like(self.Y, np.nan)
            for j in range(self.J):
                ok = np.where(np.isfinite(self.Y[:, j]))[0]
                if len(ok):
                    cut = ok[0] + max(24, int((ok[-1] - ok[0]) * DEV_FROM))
                    Ys[:cut, j] = self.Y[:cut, j]
            d1 = np.abs(np.diff(Ys, axis=0))
            self.mu, self.sd = np.nanmean(Ys, axis=0), np.nanstd(Ys, axis=0)
        else:
            d1 = None
        d1 = np.abs(np.diff(self.Y[:tr_end], axis=0)) if d1 is None else d1
        self.scale = np.nanmean(d1, axis=0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 0), self.scale, np.nanstd(self.Y[:tr_end], axis=0))
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 0), self.scale, 1.0)
        if not fam.span_min:
            self.mu = np.nanmean(self.Y[:tr_end], axis=0)
            self.sd = np.nanstd(self.Y[:tr_end], axis=0)
        self.sd = np.where(self.sd > 0, self.sd, 1.0)

    def country(self, j: int) -> str:
        s = self.series[j]
        return s.split(":", 1)[1] if ":" in s else s

    # -------------------------------------------------------------------------------------------- references
    def last_known(self, j: int) -> np.ndarray:
        L = np.clip(self.L, 0, self.T - 1)
        li = self.ffi[L, j]
        return np.where((self.L >= 0) & (li >= 0), li, -1)

    def naive(self, j: int) -> np.ndarray:
        li = self.last_known(j)
        v = np.where(li >= 0, self.Y[np.clip(li, 0, None), j], np.nan)
        return np.repeat(v[:, None], self.H, 1)

    def seasonal(self, j: int, s: int) -> np.ndarray:
        """Value one (or more) seasons before each target, the latest such value that was known at the origin; the naive value if none."""
        m = np.ceil((self.R - self.L[:, None]) / s).clip(min=1)
        rr = (self.R - m * s).astype(int)
        rr = np.clip(rr, 0, self.T - 1)
        v = self.Y[rr, j]
        nv = self.naive(j)
        return np.where(np.isfinite(v), v, nv)

    def bases(self, j: int) -> dict[str, np.ndarray]:
        out = {"naive": self.naive(j)}
        for s in self.fam.seasons:
            out[f"s{s}"] = self.seasonal(j, int(s))
        return out

    def truth(self, j: int) -> np.ndarray:
        R = np.clip(self.R, 0, self.T - 1)
        v = self.Y[R, j]
        return np.where(self.R < self.T, v, np.nan)


# ------------------------------------------------------------------------------------------------------------------------------ registry
def _span_rows(d: Design) -> tuple[int, int]:
    have = np.isfinite(d.Y).sum(1) >= max(1, d.fam.span_min or d.J // 2)
    w = np.where(have)[0]
    return int(w[0]), int(w[-1])


def windows(d: Design) -> dict:
    a, b = _span_rows(d)
    n = b - a
    ho_row = a + int(n * (1 - HOLDOUT_SHARE))
    dev_row = a + int(n * DEV_FROM)
    warm = max(int(max(d.fam.seasons) * 2 + 10), 0)
    ok = d.ok_origin & (d.L >= a + warm)
    st = d.stamps
    ho_t = st[ho_row].as_unit("ns").value
    target_end = d.kt[np.clip(d.R[:, -1], 0, d.T - 1)]
    dev = np.where(ok & (d.p0 >= dev_row) & (target_end < ho_t))[0]
    hold = np.where(ok & (d.p0 >= ho_row))[0]
    pick = lambda w: w[np.unique(np.linspace(0, len(w) - 1, min(MAX_EVAL, len(w))).round().astype(int))] if len(w) else w  # noqa: E731
    return dict(dev=pick(dev), hold=pick(hold), train_from=int(a + warm), holdout_start=str(st[ho_row]), dev_start=str(st[dev_row]))


def ref_choice(d: Design, upto_origin: int) -> list[str]:
    """Per series, the simple forecast (naive or a seasonal naive) with the lowest MAE on origins before `upto_origin`: the strict reference."""
    tr = np.where(d.ok_origin[:upto_origin])[0]
    tr = tr[np.linspace(0, len(tr) - 1, min(400, len(tr))).round().astype(int)] if len(tr) else tr
    out = []
    for j in range(d.J):
        y = d.truth(j)[tr]
        best, bk = np.inf, "naive"
        for k, P in d.bases(j).items():
            e = np.nanmean(np.abs(y - P[tr]))
            if np.isfinite(e) and e < best:
                best, bk = e, k
        out.append(bk)
    return out


def _family_entry(fid: str) -> dict:
    from forecasting_engine import histories as HS
    fam = PN.load(fid)
    d = Design(fam)
    w = windows(d)
    ok = len(w["dev"]) >= 20 and len(w["hold"]) >= 20
    refs = ref_choice(d, int(w["dev"][0])) if ok else []
    man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
    return dict(id=fid, available=ok, label=fam.label, layer=fam.layer, freq=fam.freq, unit=fam.unit, H=fam.H, skip=d.skip, series=d.series,
                drivers=[dict(name=x.name, group=x.group, avail_h=x.avail_h, dayahead_hour=x.dayahead_hour, ahead=x.ahead) for x in fam.drivers],
                start=str(d.stamps[_span_rows(d)[0]]), end=str(d.stamps[_span_rows(d)[1]]), holdout_start=w["holdout_start"], dev_start=w["dev_start"],
                train_from=str(d.stamps[w["train_from"]]), dev_origins=[str(pd.Timestamp(x)) for x in d.tau[w["dev"]]],
                holdout_origins=[str(pd.Timestamp(x)) for x in d.tau[w["hold"]]], reference=dict(zip(d.series, refs)), sha256=man["sha256"])


def register_family(fid: str) -> dict:
    """Add ONE new family to the registry, once, before any method is tried on it (the existing entries are not touched)."""
    reg = registry()
    if any(f["id"] == fid for f in reg["families"]):
        raise SystemExit(f"{fid} is already registered (its windows are locked)")
    e = _family_entry(fid)
    e["registered_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reg["families"].append(e)
    REGISTRY.write_text(json.dumps(reg, indent=1))
    return e


def register(force: bool = False) -> dict:
    if REGISTRY.exists() and not force:
        raise SystemExit(f"{REGISTRY} exists: the registry is written once (the holdout stays locked)")
    fams = []
    for fid in PN.FAMILIES:
        try:
            fam = PN.load(fid)
        except Exception as e:  # noqa: BLE001
            fams.append(dict(id=fid, available=False, reason=f"not frozen: {type(e).__name__}"))
            continue
        d = Design(fam)
        w = windows(d)
        ok = len(w["dev"]) >= 20 and len(w["hold"]) >= 20
        refs = ref_choice(d, int(w["dev"][0])) if ok else []
        from forecasting_engine import histories as HS
        man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
        fams.append(dict(id=fid, available=ok, label=fam.label, layer=fam.layer, freq=fam.freq, unit=fam.unit, H=fam.H, skip=d.skip, series=d.series,
                         drivers=[dict(name=x.name, group=x.group, avail_h=x.avail_h, dayahead_hour=x.dayahead_hour, ahead=x.ahead) for x in fam.drivers],
                         start=str(d.stamps[_span_rows(d)[0]]), end=str(d.stamps[_span_rows(d)[1]]), holdout_start=w["holdout_start"], dev_start=w["dev_start"],
                         train_from=str(d.stamps[w["train_from"]]), dev_origins=[str(pd.Timestamp(x)) for x in d.tau[w["dev"]]],
                         holdout_origins=[str(pd.Timestamp(x)) for x in d.tau[w["hold"]]], reference=dict(zip(d.series, refs)), sha256=man["sha256"]))
        print(json.dumps({k: fams[-1].get(k) for k in ("id", "available", "start", "holdout_start")} | dict(dev=len(w["dev"]), hold=len(w["hold"]))), flush=True)
    reg = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), version=2, holdout_share=HOLDOUT_SHARE, max_eval=MAX_EVAL,
               rule="variants are tried on development only; the holdout is scored only for a variant that beat the champion on development (pooled "
                    "skill), every holdout look is counted, and a variant becomes champion only if it also beats the champion's pooled skill on the "
                    "holdout without lowering the median series skill", families=fams)
    LAB.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1))
    return reg


def registry() -> dict:
    return json.loads(REGISTRY.read_text())


# ------------------------------------------------------------------------------------------------------------------------------ features
def _driver_known(dr: PN.Driver, col: str, tau: np.ndarray, at_stamps: np.ndarray | None = None) -> np.ndarray:
    """The driver's latest value known at each origin (or, for ahead drivers with at_stamps, its value AT those stamps when known by the origin)."""
    if col not in dr.data.columns:
        return np.full(len(tau) if at_stamps is None else at_stamps.shape, np.nan)
    s = dr.data[col].dropna()
    if not len(s):
        return np.full(len(tau) if at_stamps is None else at_stamps.shape, np.nan)
    st = pd.DatetimeIndex(s.index)
    kt = known_times(st, dr.avail_h, dr.dayahead_hour)
    v = s.to_numpy(float)
    if at_stamps is None:
        order = np.argsort(kt, kind="stable")
        kt, v2 = kt[order], v[order]
        i = np.searchsorted(kt, tau, "right") - 1
        return np.where(i >= 0, v2[np.clip(i, 0, None)], np.nan)
    sti = st.as_unit("ns").asi8
    i = np.searchsorted(sti, at_stamps.ravel(), "left")
    i = np.clip(i, 0, len(sti) - 1)
    hit = sti[i] == at_stamps.ravel()
    known = kt[i] <= np.broadcast_to(tau[:, None], at_stamps.shape).ravel()
    out = np.where(hit & known, v[i], np.nan)
    return out.reshape(at_stamps.shape)


def driver_col(dr, series: str, cc: str) -> str:
    """Which column of a driver feeds a series: the series itself (per-port drivers in ports_m: "a coruna:ES"), else the whole family ("*"),
    else the series' country. Before 19 Oct 2026 only "*" and the country were looked up, so per-series columns were silently empty."""
    if series in dr.data.columns:
        return series
    return "*" if "*" in dr.data.columns else cc


def _cal(stamps: pd.DatetimeIndex, cc: str) -> pd.DataFrame:
    from forecasting_engine import histories as HS
    try:
        c = HS.calendar_frame(stamps, cc if len(cc) == 2 and cc.isalpha() else "BE")
    except Exception:  # noqa: BLE001
        c = HS.calendar_frame(stamps, "BE")
    c["month"] = stamps.month
    return c


RG_RMAX = 100                     # BOCPD on whole series: run lengths truncated at 100 steps
RG_HAZARD = 1 / 100               # prior chance of a new regime at each step of the series


def bocpd_series(x: np.ndarray, hazard: float = RG_HAZARD, rmax: int = RG_RMAX):
    """Online (causal) BOCPD on one standardised stream x (T), Gaussian with unit noise and an unknown mean (prior variance 4); missing
    values leave the posterior as it was. -> (p_new, exp_run): the posterior mass on a run of <= 3 steps, and the expected run length,
    each using only x[:t+1]."""
    T = len(x)
    R = np.zeros(rmax)
    R[0] = 1.0
    mu, v = np.zeros(rmax), np.full(rmax, 4.0)
    p_new, exp_run = np.full(T, np.nan), np.full(T, np.nan)
    rl = np.arange(rmax)
    for t in range(T):
        xt = x[t]
        if np.isfinite(xt):
            s2 = v + 1.0
            pred = np.exp(-0.5 * (xt - mu) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
            g = R * pred
            cp = g.sum() * hazard
            nR = np.empty(rmax)
            nR[1:] = g[:-1] * (1 - hazard)
            nR[-1] += g[-1] * (1 - hazard)                         # the last bin keeps the long runs
            nR[0] = cp
            tot = nR.sum()
            R = nR / tot if tot > 0 else np.r_[1.0, np.zeros(rmax - 1)]
            k = v / s2
            nmu, nv = np.empty(rmax), np.empty(rmax)
            nmu[1:], nv[1:] = (mu + k * (xt - mu))[:-1], (v * (1 - k))[:-1]
            nmu[0], nv[0] = 0.0, 4.0
            mu, v = nmu, nv
        p_new[t] = R[:4].sum()
        exp_run[t] = (R * rl).sum()
    return p_new, exp_run


def regime_stream(y: np.ndarray, season: int) -> np.ndarray:
    """The stream BOCPD watches: the seasonal difference (or the plain one for short series), robustly standardised."""
    s = season if len(y) > 3 * season else 1
    x = np.full(len(y), np.nan)
    x[s:] = y[s:] - y[:-s]
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if np.isfinite(mad) and mad > 0 else (np.nanstd(x) or 1.0))


_STACK: tuple = ()                # stack[m1+m2]:<method> = those foundation models' forecasts as features (a learned combiner)


def _stack_wrap(fn, members: tuple):
    """Stacking: the trees see each listed foundation model's forecast relative to their reference (usually another foundation model, via
    anchor-<fm>:), next to every layer's drivers, and learn per series, step and situation how much of each to take. Unlike blend: and
    adapt: (fixed or error-weighted means), a member that is only right in some regimes or countries gets weight only there."""
    def f(d, refs, eo, tf):
        global _STACK
        _STACK = members
        try:
            return fn(d, refs, eo, tf)
        finally:
            _STACK = ()
    return f


_REGIME = 0                       # rg:<method> = own and cross-series regime features; rgd:<method> = also each driver's regime


def _regime_wrap(fn, level: int):
    """Theory change 2 as features: the BOCPD change-point probability and run length of the series itself, of the other series, and (rgd)
    of every driver, as known at the origin, so the model learns how shocks in one layer show up in another."""
    def f(d, refs, eo, tf):
        global _REGIME
        _REGIME = level
        try:
            return fn(d, refs, eo, tf)
        finally:
            _REGIME = 0
    return f


_RG_CACHE: dict = {}


def _regime_of(d: "Design", j: int):
    key = (d.fam.id, "y", j)
    if key not in _RG_CACHE:
        _RG_CACHE[key] = bocpd_series(regime_stream(d.Y[:, j], int(max(d.fam.seasons))))
    return _RG_CACHE[key]


def _regime_driver(dr):
    """A driver's BOCPD change-point probability per column, as a driver with the same publication lag."""
    key = ("drv", dr.name, id(dr.data))
    if key not in _RG_CACHE:
        df = dr.data.sort_index()
        seas = {"h": 24, "D": 7}.get(pd.infer_freq(df.index[:50]) or "", 1) if len(df) > 50 else 1
        out = pd.DataFrame({c: bocpd_series(regime_stream(df[c].to_numpy(float), seas))[0] for c in df.columns}, index=df.index)
        _RG_CACHE[key] = PN.Driver(f"{dr.name}_shock", dr.group, out, dr.avail_h, False, "", dr.dayahead_hour)
    return _RG_CACHE[key]


class Features:
    """Rows (origin n, series j, step k) with: step, calendar at the target, the simple forecasts relative to the reference, own recent state,
    other series' recent change, and every driver as known at the origin (standardised). Targets: (truth - reference) / scale."""

    def __init__(self, d: Design, refs: list[str], groups: set[str] | None = None, cross: bool = True, own: bool = True, yearly: bool = False):
        self.d, self.refs, self.groups, self.cross, self.own, self.yearly = d, refs, groups, cross, own, yearly
        self.names: list[str] = []

    def build(self, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """-> X (rows x features), z (scaled target), ref (rows), scale (rows), keys (rows x 3: n, j, k)."""
        d = self.d
        tau = d.tau[origins]
        H, J = d.H, d.J
        lb = LOOKBACK[d.fam.freq].value
        # other series' recent change (scaled), for the cross-series feature
        chg = np.full((len(origins), J), np.nan)
        for j in range(J):
            nv = d.naive(j)[origins, 0]
            li = d.last_known(j)[origins]
            back = np.searchsorted(d.kt, tau - lb, "right") - 1
            bi = d.ffi[np.clip(back, 0, d.T - 1), j]
            pv = np.where((back >= 0) & (bi >= 0), d.Y[np.clip(bi, 0, None), j], np.nan)
            chg[:, j] = (nv - pv) / d.scale[j]
            del li
        csum, ccnt = np.nansum(chg, 1), np.isfinite(chg).sum(1)
        drivers = [x for x in d.fam.drivers if self.groups is None or x.group in self.groups or f"@{x.name}" in self.groups]
        dstat = {}
        for dr in drivers:
            arr = dr.data.to_numpy(float)
            cut = int(len(arr) * DEV_FROM)
            dstat[dr.name] = (np.nanmean(arr[:cut]) if np.isfinite(arr[:cut]).any() else 0.0, np.nanstd(arr[:cut]) if np.isfinite(arr[:cut]).any() else 1.0)
        Xs, zs, refs, scs, keys = [], [], [], [], []
        for j in range(J):
            cc = d.country(j)
            B = {k: v[origins] for k, v in d.bases(j).items()}
            if str(self.refs[j]).startswith("fm:"):                 # anchored on a foundation model's zero-shot forecast (lab.bench)
                fm = str(self.refs[j])[3:]
                if getattr(self, "_fm", None) is None or self._fm[0] is not origins or self._fm[2] != fm:
                    from forecasting_engine.lab import bench as BN
                    self._fm = (origins, FM[fm](d, origins), fm)
                anc = self._fm[1][:, j, :]
                fb = B.get(f"s{int(max(d.fam.seasons))}", B["naive"])
                ref = np.where(np.isfinite(anc), anc, fb)
                add_anchor = np.isfinite(anc).astype(np.float32)
            elif str(self.refs[j]).startswith("drv:"):              # anchored: an external forecast (e.g. the operator's) as the model's reference
                dr_a = next(x for x in d.fam.drivers if x.name == self.refs[j][4:])
                col_a = "*" if "*" in dr_a.data.columns else cc
                tsi = d.stamps.as_unit("ns").asi8[np.clip(d.R[origins], 0, d.T - 1)]
                anc = _driver_known(dr_a, col_a, tau, tsi)           # only where it was published by the origin
                fb = B.get(f"s{int(max(d.fam.seasons))}", B["naive"])
                ref = np.where(np.isfinite(anc), anc, fb)
                add_anchor = np.isfinite(anc).astype(np.float32)
            else:
                ref = B[self.refs[j]]
                add_anchor = None
            sc = d.scale[j]
            y = d.truth(j)[origins]
            R = d.R[origins]
            tst = d.stamps[np.clip(R, 0, d.T - 1).ravel()]
            cal = _cal(tst, cc)
            cols, names = [], []
            def add(name, v):  # noqa: E306
                cols.append(np.broadcast_to(v, (len(origins), H)).astype(np.float32).ravel())
                names.append(name)
            add("step", np.arange(1, H + 1)[None, :])
            for cn in ("hour", "weekday", "holiday", "month"):
                if cn == "hour" and d.fam.freq != "h":
                    continue
                add(f"cal_{cn}", cal[cn].to_numpy(float).reshape(len(origins), H))
            add("series_id", np.full((1, 1), j))
            if add_anchor is not None:
                add("anchor_known", add_anchor)
            for k, v in B.items():
                add(f"base_{k}", (v - ref) / sc)
            for fm in _STACK:                                 # stacking: the other foundation models' forecasts relative to the anchor
                if getattr(self, "_stk", None) is None or self._stk[0] is not origins:
                    self._stk = (origins, {})
                if fm not in self._stk[1]:
                    self._stk[1][fm] = FM[fm](d, origins)
                add(f"fm_{fm}", (self._stk[1][fm][:, j, :] - ref) / sc)
            if self.yearly:                                   # the same period a year before (aligned on the weekday for hourly and daily)
                yr = YEAR_ROWS[d.fam.freq]
                rr = R - yr
                okr = (rr >= 0) & (rr <= d.L[origins][:, None])
                vy = np.where(okr, d.Y[np.clip(rr, 0, d.T - 1), j], np.nan)
                add("base_year", (vy - ref) / sc)
            if self.own:
                nv = B["naive"][:, :1]
                add("own_level", (nv - d.mu[j]) / d.sd[j])
                add("own_change", chg[:, j:j + 1])
                if d.fam.freq == "h":
                    li = d.last_known(j)[origins]
                    m24 = np.array([np.nanmean(d.Y[max(0, i - 23):i + 1, j]) if i >= 0 else np.nan for i in li])
                    add("own_dev24", (nv[:, 0] - m24)[:, None] / sc)
            if _REGIME:                                       # theory change 2: regimes, as known at the origin (online BOCPD)
                li = d.last_known(j)[origins]
                pn, er = _regime_of(d, j)
                add("own_shock", np.where(li >= 0, pn[np.clip(li, 0, None)], np.nan)[:, None])
                add("own_run", np.log1p(np.where(li >= 0, er[np.clip(li, 0, None)], np.nan))[:, None])
                if J > 1:
                    oth = [np.where(d.last_known(i)[origins] >= 0, _regime_of(d, i)[0][np.clip(d.last_known(i)[origins], 0, None)], np.nan)
                           for i in range(J) if i != j]
                    add("others_shock", np.nanmean(np.stack(oth), 0)[:, None])
            if self.cross and J > 1:
                own_c = np.nan_to_num(chg[:, j])
                n_o = ccnt - np.isfinite(chg[:, j]).astype(int)
                add("others_change", np.where(n_o > 0, (csum - own_c) / np.maximum(n_o, 1), np.nan)[:, None])
            for dr in drivers:
                col = driver_col(dr, d.series[j], cc)
                m, s = dstat[dr.name]
                s = s if s and np.isfinite(s) and s > 0 else 1.0
                if dr.ahead:
                    tsi = d.stamps.as_unit("ns").asi8[np.clip(R, 0, d.T - 1)]
                    v = _driver_known(dr, col, tau, tsi)
                    seas = int(max(d.fam.seasons))
                    tsb = d.stamps.as_unit("ns").asi8[np.clip(R - seas, 0, d.T - 1)]
                    vb = _driver_known(dr, col, tau, tsb)
                    add(f"{dr.name}@target", (v - m) / s)
                    add(f"{dr.name}@target_vs_season", (v - vb) / s)
                else:
                    v = _driver_known(dr, col, tau)
                    vb = _driver_known(dr, col, tau - lb)
                    add(f"{dr.name}", ((v - m) / s)[:, None])
                    add(f"{dr.name}_chg", ((v - vb) / s)[:, None])
                if _REGIME >= 2:
                    rd = _regime_driver(dr)
                    add(f"{dr.name}_shock", _driver_known(rd, driver_col(rd, d.series[j], cc), tau)[:, None])
            Xs.append(np.stack(cols, 1))
            zs.append(((y - ref) / sc).astype(np.float32).ravel())
            refs.append(ref.ravel())
            scs.append(np.full(ref.size, sc))
            nn, kk = np.meshgrid(origins, np.arange(1, H + 1), indexing="ij")
            keys.append(np.stack([nn.ravel(), np.full(nn.size, j), kk.ravel()], 1))
            self.names = names
        return np.vstack(Xs), np.concatenate(zs), np.concatenate(refs), np.concatenate(scs), np.vstack(keys)


# ------------------------------------------------------------------------------------------------------------------------------ methods
def _train_origins(d: Design, train_from: int, upto_t: int, stride: int) -> np.ndarray:
    """Origins whose whole truth window was known by time upto_t (ns)."""
    te = d.kt[np.clip(d.R[:, -1], 0, d.T - 1)]
    w = np.where(d.ok_origin & (d.p0 >= train_from) & (te <= upto_t))[0]
    if _XCOVID:                                   # xcovid:<method>: no training rows whose targets fall in the pandemic regime
        st = d.stamps.as_unit("ns").asi8[np.clip(d.R[w], 0, d.T - 1)]
        w = w[~((st >= XCOVID[0]) & (st < XCOVID[1])).any(1)]
    return w[::stride]


XCOVID = (pd.Timestamp("2020-03-01", tz="UTC").value, pd.Timestamp("2021-07-01", tz="UTC").value)
_XCOVID = False


def _xcovid(fn):
    """The method trained without the COVID regime (Mar 2020 - Jun 2021 targets): tourism, travel and mobility were not themselves then.
    Evaluation origins are untouched (scored on every origin as usual)."""
    def f(d, refs, eo, tf):
        global _XCOVID
        _XCOVID = True
        try:
            return fn(d, refs, eo, tf)
        finally:
            _XCOVID = False
    return f


def _stride(d: Design) -> int:
    return {"h": 2, "D": 1, "W": 1, "M": 1}[d.fam.freq]


GBM = dict(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0)
GBM_BIG = dict(loss="absolute_error", max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0)
GBM_XL = dict(loss="absolute_error", max_iter=1500, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=2.0)
GBM_REG = dict(loss="absolute_error", max_iter=200, learning_rate=0.03, max_depth=3, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=5.0)


_FEAT_CACHE: dict = {}
_FEAT_NAMES: dict = {}


def _features(d: Design, refs, groups, cross, origins: np.ndarray, yearly: bool = False):
    """Feature rows for these origins, cached per (family, driver groups, cross, yearly) for the process (the loop runs many methods on one design)."""
    key = (d.fam.id, len(d.fam.drivers), None if groups is None else frozenset(groups), cross, yearly, tuple(refs), _REGIME, _STACK)
    c = _FEAT_CACHE.get(key)
    if c is not None and np.isin(origins, c[0]).all():
        o, X, Z, REF, SC, K = c
        m = np.isin(K[:, 0], origins)
        return X[m], Z[m], REF[m], SC[m], K[m]
    F = Features(d, refs, groups=groups, cross=cross, yearly=yearly)
    X, Z, REF, SC, K = F.build(origins)
    _FEAT_NAMES[key] = F.names
    if len(_FEAT_CACHE) >= 3:
        _FEAT_CACHE.pop(next(iter(_FEAT_CACHE)))
    _FEAT_CACHE[key] = (origins, X, Z, REF, SC, K)
    return X, Z, REF, SC, K


def pooled(d: Design, refs: list[str], eval_origins: np.ndarray, train_from: int, model: str = "gbm", groups=None, cross=True, single=False,
           params: dict | None = None, shrink: float = 1.0, yearly: bool = False) -> np.ndarray:
    """Pooled model on the scaled residual over the strict reference, refitted every REFIT period on what was known then. -> (origins x J x H)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    if _SCREEN["on"] and model == "gbm":                         # screening stage: smaller trees (ranking ideas, never promoting them)
        params = dict(params or GBM)
        params.update(max_iter=min(int(params.get("max_iter", 300)), 300), max_leaf_nodes=min(int(params.get("max_leaf_nodes", 31)), 31),
                      learning_rate=max(float(params.get("learning_rate", 0.05)), 0.05))
    if groups and "pcmci" in groups:                               # E4: the family's causal parents (lab.causal), by driver name
        from forecasting_engine.lab import causal
        groups = (set(groups) - {"pcmci"}) | {f"@{n}" for n in causal.parents(d.fam.id)}
    out = np.full((len(eval_origins), d.J, d.H), np.nan)
    tau = d.tau[eval_origins]
    step = REFIT[d.fam.freq].value
    t0 = tau.min()
    blocks = ((tau - t0) // step).astype(int)
    stride = _stride(d)
    last_rho = t0 + int(blocks.max()) * step
    need = np.union1d(_train_origins(d, train_from, last_rho, stride), eval_origins)
    X, Z, REF, SC, K = _features(d, refs, groups, cross, need, yearly)  # every row once; each refit slices what it may see
    for b in np.unique(blocks):
        sel = np.where(blocks == b)[0]
        rho = t0 + b * step                                        # refit time: the start of the block
        tr = _train_origins(d, train_from, rho, stride)
        if len(tr) < 30:
            continue
        rt = np.isin(K[:, 0], tr)
        Xt, zt, kt = X[rt], Z[rt], K[rt]
        okr = np.isfinite(zt)
        re = np.isin(K[:, 0], eval_origins[sel])
        Xe, refe, sce, ke = X[re], REF[re], SC[re], K[re]
        names = _FEAT_NAMES[(d.fam.id, len(d.fam.drivers), None if groups is None else frozenset(groups), cross, yearly, tuple(refs), _REGIME, _STACK)]
        groups_j = [np.arange(d.J)] if not single else [np.array([j]) for j in range(d.J)]
        for gj in groups_j:
            mt = okr & np.isin(kt[:, 1], gj)
            me = np.isin(ke[:, 1], gj)
            if mt.sum() < 50:
                continue
            Xa, za = Xt[mt], zt[mt]
            if model == "gbm":
                pp = dict(params or GBM)
                m = HistGradientBoostingRegressor(**{k: v for k, v in pp.items() if k != "random_state"}, early_stopping=False,
                                                  random_state=pp.get("random_state", 0),
                                                  categorical_features=[names.index("series_id")] if not single else None)
                keep = np.isfinite(Xa).any(0)
                m.fit(Xa[:, keep], za)
                if not single:
                    pass
                zp = m.predict(Xe[me][:, keep])
            else:
                mu = np.nanmean(Xa, 0)
                sd = np.nanstd(Xa, 0)
                sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
                mu = np.where(np.isfinite(mu), mu, 0.0)
                Xs = np.clip(np.nan_to_num((Xa - mu) / sd), -5, 5)          # (not Z: that name holds the targets for the next refit blocks)
                m = Ridge(alpha=float(len(Xs)) * 0.1)
                m.fit(Xs, np.clip(za, -20, 20))
                zp = np.clip(m.predict(np.clip(np.nan_to_num((Xe[me] - mu) / sd), -5, 5)), -20, 20)
            pred = refe[me] + sce[me] * zp * shrink
            kk = ke[me]
            pos = np.searchsorted(eval_origins[sel], kk[:, 0])
            out[sel[pos], kk[:, 1], kk[:, 2] - 1] = pred
    return out


def m_ref(d, refs, eo, tf):
    return np.stack([d.bases(j)[refs[j]][eo] for j in range(d.J)], 1)


def m_naive(d, refs, eo, tf):
    return np.stack([d.naive(j)[eo] for j in range(d.J)], 1)


def m_ets(d, refs, eo, tf):
    """Exponential smoothing per series, refitted at each origin on what was known (damped trend; additive season for hourly 24 and daily 7)."""
    import warnings
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    sp = {"h": 24, "D": 7}.get(d.fam.freq)
    keep = {"h": 24 * 28, "D": 7 * 30, "W": 156, "M": 120}[d.fam.freq]
    out = np.full((len(eo), d.J, d.H), np.nan)
    for j in range(d.J):
        li = d.last_known(j)
        for i, n in enumerate(eo):
            L = li[n]
            if L < 0:
                continue
            seg = d.Y[max(0, L + 1 - keep):L + 1, j]
            seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy()
            if len(seg) < (2 * sp + 5 if sp else 12) or not np.isfinite(seg).all():
                continue
            gap = int(d.R[n, 0] - L)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    f = ExponentialSmoothing(seg, trend="add", damped_trend=True, seasonal="add" if sp else None, seasonal_periods=sp,
                                             initialization_method="estimated").fit().forecast(gap + d.H - 1)
                    out[i, j] = f[gap - 1:gap - 1 + d.H]
                except Exception:  # noqa: BLE001
                    pass
    ref = m_ref(d, refs, eo, tf)
    return np.where(np.isfinite(out), out, ref)


def m_yoy_struct(d, refs, eo, tf, shrink: float = 1.0):
    """Year-on-year rates from their index (hicp_m): the last known rate + each coming month's expected change - the change 12 months before
    (which leaves the rate). Expected change of a month = the mean change of that calendar month over the 3 previous years (seasonality)
    + the mean de-seasonalised change of the last 3 known months (momentum). Needs the hicp_mom driver (panel.extra_drivers)."""
    dr = next((x for x in d.fam.drivers if x.name in ("hicp_mom", "cpi_mom")), None)
    ref = m_ref(d, refs, eo, tf)
    if dr is None:
        return ref
    out = ref.copy()
    st = d.stamps
    for j in range(d.J):
        cc = d.country(j)
        if cc not in dr.data.columns:
            continue
        mom = dr.data[cc].reindex(st)                                     # monthly change, on the family's month stamps
        momv = mom.to_numpy(float)
        kt = known_times(st, dr.avail_h, None)
        li = d.last_known(j)
        for i, n in enumerate(eo):
            L = li[n]
            if L < 40:
                continue
            tau = d.tau[n]
            known = kt <= tau
            mk = np.where(known, momv, np.nan)
            def seas(r):                                                  # mean change of the same calendar month, previous 3 years, known
                v = [mk[r - 12 * y] for y in (1, 2, 3) if r - 12 * y >= 0]
                v = [x for x in v if np.isfinite(x)]
                return float(np.mean(v)) if v else np.nan
            rec = [mk[r] - seas(r) for r in range(L - 2, L + 1) if np.isfinite(mk[r])]
            mo = float(np.nanmean(rec)) if rec else 0.0
            yv = d.Y[L, j]
            acc = yv
            for k in range(d.H):
                t = d.R[n, k]
                for r in range(int(d.R[n, k - 1]) + 1 if k else L + 1, int(t) + 1):
                    e = seas(r)
                    if not np.isfinite(e) or r - 12 < 0 or not np.isfinite(mk[r - 12]):
                        acc = np.nan
                        break
                    acc = acc + (e + mo) - mk[r - 12]
                if np.isfinite(acc):
                    out[i, j, k] = ref[i, j, k] + shrink * (acc - ref[i, j, k])
    return out


def _p(model="gbm", groups=None, cross=True, single=False, params=None, shrink=1.0, yearly=False):
    return lambda d, refs, eo, tf: pooled(d, refs, eo, tf, model=model, groups=groups, cross=cross, single=single, params=params, shrink=shrink,
                                          yearly=yearly)


ADAPT_K = 12                      # adaptive blend: the latest 12 forecasts whose outcomes were known


def _adapt_j(*fns, k: int | None = None, power: float = 2.0):
    """As _adapt, but each series (country) gets its own weights from its own recent errors (k = how many earlier origins; default ADAPT_K).
    Where one country's regime shifts (a recovery, a shock) its mix moves without dragging the others."""
    def f(d, refs, eo, tf):
        K = k or ADAPT_K
        Ps = [g(d, refs, eo, tf) for g in fns]
        end = d.kt[np.clip(d.R[eo, -1], 0, d.T - 1)]
        Ytrue = np.stack([d.truth(j)[eo] for j in range(d.J)], 1)
        errs = [np.nanmean(np.abs(Ytrue - P) / d.scale[None, :, None], axis=2) for P in Ps]           # per origin and series
        out = np.full_like(Ps[0], np.nan)
        for i in range(len(eo)):
            past = np.where(end < d.tau[eo[i]])[0][-K:]
            stack = np.stack([P[i] for P in Ps])                                                 # (M, J, H)
            if len(past) >= 4:
                e = np.stack([np.nanmean(x[past], axis=0) for x in errs])                        # (M, J)
                w = 1.0 / np.maximum(e, 1e-9) ** power
                w = np.where(np.isfinite(w), w, 0)
                sw = w.sum(0, keepdims=True)
                w = np.where(sw > 0, w / np.where(sw > 0, sw, 1), 1.0 / len(Ps))
            else:
                w = np.full((len(Ps), d.J), 1.0 / len(Ps))
            ok = np.isfinite(stack)
            ww = w[:, :, None] * ok
            out[i] = np.where(ok.any(0), np.nansum(np.nan_to_num(stack) * ww, 0) / np.maximum(ww.sum(0), 1e-12), np.nan)
        return out
    return f


def _adapt(*fns, k: int | None = None, power: float = 2.0):
    """Weights the methods at each origin by 1 / (their recent scaled MAE)^2, from the latest ADAPT_K earlier origins whose whole truth window
    was known by then (equal weights until there are 4). Shock-sensitive models lead in turbulent stretches, calm ones in calm stretches."""
    def f(d, refs, eo, tf):
        Ps = [g(d, refs, eo, tf) for g in fns]
        end = d.kt[np.clip(d.R[eo, -1], 0, d.T - 1)]
        Ytrue = np.stack([d.truth(j)[eo] for j in range(d.J)], 1)
        errs = [np.nanmean(np.abs(Ytrue - P) / d.scale[None, :, None], axis=(1, 2)) for P in Ps]   # per origin, pooled
        out = np.full_like(Ps[0], np.nan)
        for i in range(len(eo)):
            past = np.where(end < d.tau[eo[i]])[0][-(k or ADAPT_K):]
            if len(past) >= 4:
                e = np.array([np.nanmean(x[past]) for x in errs])
                w = 1.0 / np.maximum(e, 1e-9) ** power
                w = np.where(np.isfinite(w), w, 0)
                w = w / w.sum() if w.sum() > 0 else np.ones(len(Ps)) / len(Ps)
            else:
                w = np.ones(len(Ps)) / len(Ps)
            stack = np.stack([P[i] for P in Ps])
            ok = np.isfinite(stack)
            ww = w[:, None, None] * ok
            out[i] = np.where(ok.any(0), np.nansum(np.nan_to_num(stack) * ww, 0) / np.maximum(ww.sum(0), 1e-12), np.nan)
        return out
    return f


def _recon(fn):
    """Coupled engine E5: cross-temporal reconciliation of a daily family with its hourly sibling. For each step whose target day the sibling
    covers (its chained d1 forecast, lab.chain), the daily forecast and the mean of the sibling's hourly forecasts are combined by inverse
    recent squared error (MinT with a diagonal error covariance), from the latest ADAPT_K earlier origins whose truth was known by then."""
    def f(d, refs, eo, tf):
        P = fn(d, refs, eo, tf)
        dr = next((x for x in d.fam.drivers if x.name.startswith("chain_") and x.name.endswith("_d1")), None)
        if dr is None:
            return P
        tau = d.tau[eo]
        Q = np.full_like(P, np.nan)
        for j in range(d.J):
            st = d.stamps.as_unit("ns").asi8[np.clip(d.R[eo], 0, d.T - 1)]
            Q[:, j, :] = _driver_known(dr, d.country(j), tau, st)
        end = d.kt[np.clip(d.R[eo, -1], 0, d.T - 1)]
        Yt = np.stack([d.truth(j)[eo] for j in range(d.J)], 1)
        eP = np.abs(Yt - P) / d.scale[None, :, None]
        eQ = np.abs(Yt - Q) / d.scale[None, :, None]
        out = P.copy()
        for i in range(len(eo)):
            past = np.where(end < tau[i])[0][-ADAPT_K:]
            for k in range(d.H):
                q = Q[i, :, k]
                if not np.isfinite(q).any() or len(past) < 4:
                    continue
                a, b = np.nanmean(eP[past, :, k]), np.nanmean(eQ[past, :, k])
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                w = (1 / max(b, 1e-9) ** 2) / (1 / max(a, 1e-9) ** 2 + 1 / max(b, 1e-9) ** 2)
                out[i, :, k] = np.where(np.isfinite(q), (1 - w) * P[i, :, k] + w * q, P[i, :, k])
        return out
    return f


_MEMO: dict = {}


def _lagsig(d, drivers: bool) -> str:
    """Cache-key suffix after the publication-lag audit (panel.relag sets fam.lagfix): the delays a saved forecast depends on (the
    target's when the audit changed it, and the drivers' when the method reads drivers). Empty for families the audit did not change, so
    their saved forecasts stay valid."""
    lf = getattr(d.fam, "lagfix", "") or ""
    if not lf or (not drivers and "target" not in lf):
        return ""
    sig = repr((lf, float(d.fam.avail_h), [(x.name, float(x.avail_h)) for x in d.fam.drivers] if drivers else None))
    return "_lf" + hashlib.sha1(sig.encode()).hexdigest()[:8]


def memo_method(name: str):
    """method(name), memoised per (family, origins, train start, references), and saved to the forecast cache (keyed on the method name,
    the target data, the driver set, the origins and FC_VERSION) so later rounds reuse a champion's forecasts instead of refitting it."""
    fn = method(name)

    def f(d, refs, eo, tf):
        key = (d.fam.id, name, np.asarray(eo).tobytes(), tf, tuple(map(str, refs)))
        if key not in _MEMO:
            P = None
            if _PERSIST["on"] and not _SCREEN["on"]:
                h = hashlib.sha256(repr((FC_VERSION, name, tf, key[4], [x.name for x in d.fam.drivers])).encode() + _lagsig(d, True).encode())
                h.update(np.ascontiguousarray(d.Y).tobytes())
                h.update(key[2])
                fname = f"fc_{d.fam.id}_{h.hexdigest()[:20]}.npy"
                P = _fm_cache_get(fname)
                if P is None:
                    P = fn(d, refs, eo, tf)
                    _fm_cache_put(fname, P)
            _MEMO[key] = P if P is not None else fn(d, refs, eo, tf)
        return _MEMO[key]
    return f


FC_VERSION = 1                   # bump when a method's code changes what it forecasts: saved forecasts (memo_method) are keyed on it
_PERSIST = {"on": os.environ.get("FE_FC_CACHE", "1") == "1"}   # saved forecasts next to the foundation-model cache


def _shock_wrap(base: str, alts: list[str], amax: float, w: int, family_wide: bool):
    """Shock-aware switch (event replays I, Oct 2026: the champions lead in calm periods but lag the foundation models just after a shock).
    For each origin and series, s = the largest BOCPD change-point probability (run of <= 3 steps) over the last w rows known at the
    origin (over all series if family_wide). The forecast is (1 - a) * base + a * mean(alts) with a = amax * clip((s - 0.1) / 0.4, 0, 1):
    the base everywhere calm, the foundation models where a break was just seen, back to the base as the run grows."""
    fb, fa = memo_method(base), [memo_method(x) for x in alts]

    def f(d, refs, eo, tf):
        P = fb(d, refs, eo, tf)
        Q = np.nanmean(np.stack([g(d, refs, eo, tf) for g in fa]), axis=0)
        S = np.zeros((len(eo), d.J))
        for j in range(d.J):
            pn = np.nan_to_num(_regime_of(d, j)[0])
            li = d.last_known(j)[eo]
            for i, l in enumerate(li):
                if l >= 0:
                    S[i, j] = pn[max(0, l - w + 1): l + 1].max()
        if family_wide:
            S = np.repeat(S.max(1, keepdims=True), d.J, 1)
        a = (amax * np.clip((S - 0.1) / 0.4, 0, 1))[:, :, None]
        return np.where(np.isfinite(Q), (1 - a) * P + a * Q, P)
    return f


def split_members(body: str) -> list[str]:
    """Split a combiner's member list on top-level "|" only: "[...]" and "{...}" nest, and a member wrapped in {} is taken whole
    (so a champion that is itself a blend can be one member of median:/trim:)."""
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "|" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x[1:-1] if x.startswith("{") and x.endswith("}") else x for x in out]


def _median(*fns):
    """Robust combination: the per-point median of the members' forecasts (one member far off cannot pull it, unlike blend:)."""
    def f(d, refs, eo, tf):
        return np.nanmedian(np.stack([g(d, refs, eo, tf) for g in fns]), axis=0)
    return f


def _trim(*fns, drop: int = 1):
    """Robust combination: at each origin, the mean of the members left after dropping the `drop` with the largest recent scaled MAE
    (latest ADAPT_K earlier origins whose truth was known then; plain mean until there are 4). Unlike adapt:, the weights do not chase
    the latest winner: it only removes the members that are currently doing worst."""
    def f(d, refs, eo, tf):
        Ps = [g(d, refs, eo, tf) for g in fns]
        end = d.kt[np.clip(d.R[eo, -1], 0, d.T - 1)]
        Ytrue = np.stack([d.truth(j)[eo] for j in range(d.J)], 1)
        errs = [np.nanmean(np.abs(Ytrue - P) / d.scale[None, :, None], axis=(1, 2)) for P in Ps]
        stack = np.stack(Ps)                                                        # (M, O, J, H)
        out = np.full_like(Ps[0], np.nan)
        for i in range(len(eo)):
            past = np.where(end < d.tau[eo[i]])[0][-ADAPT_K:]
            keep = np.arange(len(Ps))
            if len(past) >= 4 and len(Ps) > drop + 1:
                e = np.array([np.nanmean(x[past]) for x in errs])
                e = np.where(np.isfinite(e), e, np.inf)
                keep = np.argsort(e)[: len(Ps) - drop]
            out[i] = np.nanmean(stack[keep, i], axis=0)
        return out
    return f


def _refit_wrap(fn, n: int):
    unit = {"h": pd.Timedelta(hours=1), "D": pd.Timedelta(days=1), "W": pd.Timedelta(days=7), "M": pd.Timedelta(days=30.44)}

    def f(d, refs, eo, tf):
        old = REFIT[d.fam.freq]
        REFIT[d.fam.freq] = unit[d.fam.freq] * n
        try:
            return fn(d, refs, eo, tf)
        finally:
            REFIT[d.fam.freq] = old
    return f


def _wmean(Ps, w):
    stack = np.stack(Ps)
    ok = np.isfinite(stack)
    ww = np.asarray(w, float)[:, None, None, None] * ok
    return np.where(ok.any(0), np.nansum(np.nan_to_num(stack) * ww, 0) / np.maximum(ww.sum(0), 1e-12), np.nan)


def _wblend(*fns, weights):
    def f(d, refs, eo, tf):
        return _wmean([g(d, refs, eo, tf) for g in fns], weights)
    return f


def _sblend(*fns, win, wout):
    """Fixed weights that depend on the origin's season (October-March = respiratory season, otherwise the off-season)."""
    def f(d, refs, eo, tf):
        Ps = [g(d, refs, eo, tf) for g in fns]
        m = pd.DatetimeIndex(d.tau[eo].astype("datetime64[ns]")).month
        inn = (m >= 10) | (m <= 3)
        return np.where(inn[:, None, None], _wmean(Ps, win), _wmean(Ps, wout))
    return f


def _blend(*fns):
    """The mean of several methods' forecasts (forecast combination); where one is missing, the others."""
    def f(d, refs, eo, tf):
        return np.nanmean(np.stack([g(d, refs, eo, tf) for g in fns]), axis=0)
    return f


def m_chronos(d, refs, eo, tf):
    """Chronos-Bolt (a pretrained time-series foundation model), zero-shot on each series' own history as known at the origin: a member of
    blends and adaptive blends, so the cross-layer trees add what the other systems know on top of its pattern memory (lab.bench)."""
    from forecasting_engine.lab import bench as BN
    return BN.chronos_P(d, eo)


_FM_MEMO: dict = {}


def _fm_chronos2(d, eo, cov=False, cross=False, select=False):
    """Memoised per (family, rows, origins, variant): several methods of one round reuse the same zero-shot forecasts."""
    from forecasting_engine.lab import bench as BN
    import hashlib
    key = (d.fam.id, d.T, cov, cross, select, hashlib.sha1(np.asarray(eo, np.int64).tobytes()).hexdigest())
    if key not in _FM_MEMO:
        drv = hashlib.sha1("|".join(sorted(x.name for x in d.fam.drivers)).encode()).hexdigest()[:10] if (cov or select) else "-"
        name = f"{d.fam.id}_c2_{int(cov)}{int(cross)}{int(select)}_{d.T}_{drv}_{key[-1][:16]}{_lagsig(d, cov or select)}.npy"
        P = _fm_cache_get(name)
        if P is None or P.shape != (len(eo), d.J, d.H):
            P = BN.chronos2_P(d, eo, cov, cross, select=select)
            _fm_cache_put(name, P)
        _FM_MEMO[key] = P
    return _FM_MEMO[key]


_FM_LOCAL = Path(os.environ.get("FE_FM_LOCAL", str(Path.home() / ".cache" / "forecasting_engine_fm")))


def _fm_cache_get(name: str):
    """Foundation-model forecasts are deterministic for (family, rows, variant, drivers, origins): cached on disk, shared across rounds."""
    f = _FM_LOCAL / name
    try:
        return np.load(f) if f.exists() else None
    except Exception:
        return None


def _fm_cache_put(name: str, P) -> None:
    try:
        _FM_LOCAL.mkdir(parents=True, exist_ok=True)
        np.save(_FM_LOCAL / name, P)
    except Exception:
        pass


def _fm_new(model: str, cross: bool = False, cov: bool = False):
    """TiRex-2 / Toto 2.0 zero-shot forecasts (lab.bench) as members, through the same shared cache as Chronos-2."""
    def f(d, eo):
        from forecasting_engine.lab import bench as BN
        import hashlib
        key = (d.fam.id, d.T, model, cov, cross, hashlib.sha1(np.asarray(eo, np.int64).tobytes()).hexdigest())
        if key not in _FM_MEMO:
            drv = hashlib.sha1("|".join(sorted(x.name for x in d.fam.drivers)).encode()).hexdigest()[:10] if cov else "-"
            name = f"{d.fam.id}_{model}_{int(cov)}{int(cross)}_{d.T}_{drv}_{key[-1][:16]}{_lagsig(d, cov)}.npy"
            P = _fm_cache_get(name)
            if P is None or P.shape != (len(eo), d.J, d.H):
                P = BN.toto2_P(d, eo, cross=cross) if model == "toto2" else BN.tirex2_P(d, eo, covariates=cov, cross=cross)
                _fm_cache_put(name, P)
            _FM_MEMO[key] = P
        return _FM_MEMO[key]
    return f


FM = {"chronos": lambda d, eo: m_chronos(d, None, eo, 0),                     # Chronos-Bolt, own history only
      "chronos2": lambda d, eo: _fm_chronos2(d, eo),                          # Chronos-2, own history only
      "chronos2x": lambda d, eo: _fm_chronos2(d, eo, True, True),             # Chronos-2 with every layer as covariates + cross-country context
      "chronos2s": lambda d, eo: _fm_chronos2(d, eo, True, True, True),
      "toto2": _fm_new("toto2"), "toto2x": _fm_new("toto2", cross=True),       # Toto 2.0 (Datadog, Apache 2.0): own history / cross-series
      "tirex2": _fm_new("tirex2"), "tirex2x": _fm_new("tirex2", cross=True),   # TiRex-2 (NX-AI, Apache 2.0): own history / cross-series
      "tirex2c": _fm_new("tirex2", cov=True)}                                  # TiRex-2 with every layer as covariates       # Chronos-2 with the PCMCI parents + day-ahead drivers only


ALL_GROUPS = {"energy", "weather", "markets", "society", "economy", "cyber", "fuel", "trade", "health", "policy", "network", "chain", "pcmci", "transport", "security", "environment", "water", "tourism", "information", "calendar", "housing", "labour", "attention", "migration", "agriculture", "space", "emissions", "regulation", "credit", "freight", "construction", "firms", "sentiment", "services", "insolvency", "research", "transit", "traffic", "linkgraph", "linkdense"}
METHODS = {
    "ref": m_ref, "naive": m_naive, "ets": m_ets, "chronos": m_chronos,
    "chronos2": lambda d, refs, eo, tf: FM["chronos2"](d, eo), "chronos2x": lambda d, refs, eo, tf: FM["chronos2x"](d, eo),
    "chronos2s": lambda d, refs, eo, tf: FM["chronos2s"](d, eo),
    **{k: (lambda k: lambda d, refs, eo, tf: FM[k](d, eo))(k) for k in ("toto2", "toto2x", "tirex2", "tirex2x", "tirex2c")},
    "gbm_own": _p("gbm", groups=set(), cross=False),                 # own history only, but learned across every country
    "gbm_cross": _p("gbm", groups=set(), cross=True),                # + what the other countries are doing
    "gbm_all": _p("gbm"),                                            # + every driver of every layer
    "gbm_single": _p("gbm", single=True),                            # every driver, but one model per series (no pooling)
    "ridge_all": _p("ridge"),
    "gbm_all_reg": _p("gbm", params=GBM_REG),                        # shallower, slower, larger leaves: for short, noisy (monthly) families
    "gbm_all_half": _p("gbm", shrink=0.5),                           # half way between the reference and the model
    "gbm_all_reg_half": _p("gbm", params=GBM_REG, shrink=0.5),
    "gbm_all_y": _p("gbm", yearly=True),                             # + the same period a year before
    "gbm_all_reg_y_half": _p("gbm", params=GBM_REG, shrink=0.5, yearly=True),
    "blend_all_own": _blend(_p("gbm"), _p("gbm", groups=set(), cross=False)),
    "blend_all_ets": _blend(_p("gbm"), m_ets),
    "gbm_own_y": _p("gbm", groups=set(), cross=False, yearly=True),
    "gbm_cross_y": _p("gbm", groups=set(), cross=True, yearly=True),
    "gbm_all_cs": _p("gbm", params=dict(GBM, max_features=0.3)),     # each split sees 30 % of the columns: less room to fit noise drivers
    "gbm_all_cs_y": _p("gbm", params=dict(GBM, max_features=0.3), yearly=True),
    "yoy_struct": m_yoy_struct,                                      # inflation from its index (base effects, seasonality, momentum)
    "yoy_struct_half": lambda d, refs, eo, tf: m_yoy_struct(d, refs, eo, tf, shrink=0.5),
    "gbm_own_big": _p("gbm", groups=set(), cross=False, params=GBM_BIG),     # bigger trees for the large hourly families
    "gbm_own_xl": _p("gbm", groups=set(), cross=False, params=GBM_XL),
    "gbm_own_big_y": _p("gbm", groups=set(), cross=False, params=GBM_BIG, yearly=True),
    "gbm_all_big": _p("gbm", params=GBM_BIG),
    "gbm_cross_big": _p("gbm", groups=set(), cross=True, params=GBM_BIG),
    "gbm_own_reg": _p("gbm", groups=set(), cross=False, params=GBM_REG),
    "gbm_all_big_cs_y": _p("gbm", params=dict(GBM_BIG, max_features=0.3), yearly=True),
    "gbm_ew_big_y": _p("gbm", groups={"energy", "weather"}, params=GBM_BIG, yearly=True),
}


def method(name: str):
    """A method by name; "gbm_only-<layer>+<layer>" = trees with own history, other countries and only those layers' drivers
    (optionally suffixed "_y" for the year-before feature, "_sq" for squared loss)."""
    if name in METHODS:
        return METHODS[name]
    if re.search(r"xl[tr]_|chronos2[xs]g", name):           # the cross-layer round's learners and ragged-edge Chronos-2 members register themselves on import
        from forecasting_engine.lab import xl_c2
        from forecasting_engine.lab import xl_lib
        if name in METHODS:
            return METHODS[name]
    if name.startswith("tabpfn_ref"):                          # research #7: TabPFN-v2 on the feature table (registers itself on import)
        from forecasting_engine.lab import tabpfn_ref as TP
        TP.register(name)
        if name in METHODS:
            return METHODS[name]
    m_ad = re.match(r"^adaptj(\d*):", name)
    if m_ad:                                                    # adaptj[K]:<method>|<method>: per-series weights, memory K origins
        return _adapt_j(*[method(x) for x in name[m_ad.end():].split("|")], k=int(m_ad.group(1)) if m_ad.group(1) else None)
    m_aw = re.match(r"^adapt(j?)w(\d+)p(\d+):", name)
    if m_aw:                                                    # adapt[j]w<K>p<P>:<m>|<m>: memory K origins, weights 1/error^(P/10): P=0 is an equal weighting
        fa = _adapt_j if m_aw.group(1) else _adapt
        return fa(*[method(x) for x in name[m_aw.end():].split("|")], k=int(m_aw.group(2)), power=int(m_aw.group(3)) / 10.0)
    m_rf = re.match(r"^rf(\d+):", name)
    if m_rf:                                                    # rf<N>:<method>: the trees refitted every N periods (default: a year)
        return _refit_wrap(method(name[m_rf.end():]), int(m_rf.group(1)))
    m_wb = re.match(r"^wblend\[([0-9,]+)\]:", name)
    if m_wb:                                                    # wblend[w1,w2,..]:<m>|<m>..: fixed weights (percent) fitted on development
        return _wblend(*[memo_method(x) for x in split_members(name[m_wb.end():])], weights=[int(x) for x in m_wb.group(1).split(",")])
    m_sb = re.match(r"^sblend\[([0-9,]+);([0-9,]+)\]:", name)
    if m_sb:                                                    # sblend[in;out]:<m>|<m>..: fixed weights by the origin's season (Oct-Mar / Apr-Sep)
        return _sblend(*[memo_method(x) for x in split_members(name[m_sb.end():])],
                       win=[int(x) for x in m_sb.group(1).split(",")], wout=[int(x) for x in m_sb.group(2).split(",")])
    m_sh = re.match(r"^shock(\d+)w(\d+)(f?)\[([a-z0-9|]+)\]:", name)
    if m_sh:                                                    # shock<A>w<W>[f][fm|fm]:<method>: lean A % on those models after a break
        return _shock_wrap(name[m_sh.end():], m_sh.group(4).split("|"), int(m_sh.group(1)) / 100, int(m_sh.group(2)), bool(m_sh.group(3)))
    m_st = re.match(r"^stack\[([a-z0-9+]+)\]:", name)
    if m_st:                                                    # stack[m1+m2]:<method>: those models' forecasts as tree features
        return _stack_wrap(method(name[m_st.end():]), tuple(m_st.group(1).split("+")))
    if name.startswith("rg:"):                                  # rg:<method>: with own / cross-series regime features (BOCPD)
        return _regime_wrap(method(name[len("rg:"):]), 1)
    if name.startswith("rgd:"):                                 # rgd:<method>: and every driver's regime
        return _regime_wrap(method(name[len("rgd:"):]), 2)
    if name.startswith("adapt:"):                               # adapt:<method>|<method>: weighted by recent errors (as known then)
        return _adapt(*[method(x) for x in name[len("adapt:"):].split("|")])
    if name.startswith("anchor-"):                              # anchor-<driver>:<method>: the method learns corrections to that forecast
        drv, inner = name[len("anchor-"):].split(":", 1)
        fn = method(inner)
        tag = f"fm:{drv}" if drv in FM else f"drv:{drv}"
        return lambda d, refs, eo, tf: fn(d, [tag] * d.J, eo, tf)
    m_tf = re.match(r"^tfm(\d+):", name)
    if m_tf:                                                    # tfm<N>:<method>: trees start training N rows before the registered
        inner, n_back = method(name[m_tf.end():]), int(m_tf.group(1))   # train_from (the warm-up for the yearly lag); targets must still be
        return lambda d, refs, eo, tf: inner(d, refs, eo, max(0, tf - n_back))   # known before each origin (_train_origins): no look-ahead
    if name.startswith("xcovid:"):
        return _xcovid(method(name[len("xcovid:"):]))
    if name.startswith("recon:"):                               # recon:<method>: reconciled with the hourly sibling's chained forecast
        return _recon(method(name[len("recon:"):]))
    if name.startswith("median:"):                              # median:<method>|<method>[|...]: the per-point median
        return _median(*[memo_method(x) for x in split_members(name[len("median:"):])])
    m_tr = re.match(r"^trim(\d*):", name)
    if m_tr:                                                    # trim[N]:<method>|...: mean after dropping the N recently worst (default 1)
        return _trim(*[memo_method(x) for x in split_members(name[m_tr.end():])], drop=int(m_tr.group(1) or 1))
    if name.startswith("blend:"):                               # blend:<method>|<method>[|...]: the mean of their forecasts
        return _blend(*[method(x) for x in name[len("blend:"):].split("|")])
    if name.startswith("gbm_only-"):                             # gbm_only-<layer>+<layer>[_y][_big|_xl][_cs][_reg][_half]
        body = name[len("gbm_only-"):]
        seed = 0
        m_seed = re.search(r"_s(\d+)$", body)
        if m_seed:                                               # _sN: another random seed (with _cs: a different column draw, for bagging)
            seed = int(m_seed.group(1))
            body = body[: m_seed.start()]
        flags, changed = set(), True
        while changed:                                           # strip suffixes in any order (one pass missed "_big" in "..._big_y")
            changed = False
            for f in ("_half", "_reg", "_cs", "_big", "_xl", "_y", "_sq"):
                if body.endswith(f):
                    flags.add(f)
                    body = body[: -len(f)]
                    changed = True
        groups = set(body.split("+")) - {"", "none"}
        if groups - ALL_GROUPS:
            raise KeyError(f"unknown layer(s) {groups - ALL_GROUPS} in {name}")
        params = GBM_XL if "_xl" in flags else GBM_BIG if "_big" in flags else GBM_REG if "_reg" in flags else GBM
        if "_cs" in flags:
            params = dict(params, max_features=0.3)
        if "_sq" in flags:                                       # squared loss: absolute loss predicts the median, which is zero for zero-inflated
            params = dict(params, loss="squared_error")          # targets (wildfire_w), so the trees never moved the anchor

        if seed:
            params = dict(params, random_state=seed)
        return _p("gbm", groups=groups, yearly="_y" in flags, params=params, shrink=0.5 if "_half" in flags else 1.0)
    raise KeyError(name)
_NEW_FM = ("timesfm25", "timesfm25x", "flowstate")                  # TimesFM 2.5 (Google) and FlowState (IBM), both Apache 2.0 (lab.bench)


def _fm_extra(k):
    def f(d, eo):
        import hashlib
        key = (d.fam.id, d.T, k, hashlib.sha1(np.asarray(eo, np.int64).tobytes()).hexdigest())
        if key not in _FM_MEMO:
            name = f"{d.fam.id}_{k}_{d.T}_{key[-1][:16]}{_lagsig(d, k.endswith('x'))}.npy"
            P = _fm_cache_get(name)
            if P is None or P.shape != (len(eo), d.J, d.H):
                from forecasting_engine.lab import bench as BN
                P = BN.FM_EXTRA[k](d, eo)
                _fm_cache_put(name, P)
            _FM_MEMO[key] = P
        return _FM_MEMO[key]
    return f


FM.update({k: _fm_extra(k) for k in _NEW_FM})
METHODS.update({k: (lambda k: lambda d, refs, eo, tf: FM[k](d, eo))(k) for k in _NEW_FM})
for _g in sorted(ALL_GROUPS):                                         # leave-one-layer-out: what each layer adds
    METHODS[f"gbm_all-{_g}"] = _p("gbm", groups=ALL_GROUPS - {_g})


# ------------------------------------------------------------------------------------------------------------------------------ scoring
VOL_K = 6                         # volatility scaling: the last 6 known steps
ROLL_K = 60                       # rolling calibration: each series' latest 60 resolved errors per step


def _roll_q(d: Design, i_tau: int, roll: tuple) -> np.ndarray | None:
    """(3, J, H) quantiles of the scaled errors of the latest ROLL_K origins whose truth was known at i_tau. Modes: "roll" each series' own;
    "rollp" pooled over the family's series (many more errors per band); "rollmax" (applied in score) the wider of that and the long-run band."""
    end, Zr = roll[0], roll[1]
    mode = roll[2] if len(roll) > 2 else "roll"
    m = np.where(end < i_tau)[0][-ROLL_K:]
    if len(m) < 20:
        return None
    if mode in ("rollp", "rollmax"):
        q = np.nanquantile(Zr[m].reshape(-1, d.H), QS, axis=0)          # (3, H)
        return np.broadcast_to(q[:, None, :], (3, d.J, d.H))
    return np.nanquantile(Zr[m], QS, axis=0)


def score(d: Design, eo: np.ndarray, P: np.ndarray, refs: list[str], calib: np.ndarray | None = None, roll: tuple | None = None) -> dict:
    """Pooled skill and per-series skills against the strict reference; with calib (scaled residuals per step from development) coverage of the
    10-90 % band and pinball loss; with roll = (truth-known times, scaled residuals) of earlier origins, per-series bands from recent errors."""
    Rf = m_ref(d, refs, eo, None)
    num = den = 0.0
    per = {}
    zres = np.full((len(eo), d.J, d.H), np.nan)
    for j in range(d.J):
        y = d.truth(j)[eo]
        ok = np.isfinite(y) & np.isfinite(P[:, j]) & np.isfinite(Rf[:, j])
        if ok.sum() < 10:
            continue
        e, er = np.abs(y - P[:, j])[ok], np.abs(y - Rf[:, j])[ok]
        num += e.sum() / d.scale[j]
        den += er.sum() / d.scale[j]
        per[d.series[j]] = round(1 - e.mean() / er.mean(), 4) if er.mean() > 0 else None
        zres[:, j] = (y - P[:, j]) / d.scale[j]
    vals = [v for v in per.values() if v is not None]
    out = dict(pooled_skill=round(1 - num / den, 4) if den > 0 else None, median_skill=round(float(np.median(vals)), 4) if vals else None,
               share_better=round(float(np.mean([v > 0 for v in vals])), 3) if vals else None, n_series=len(vals),
               points=int(np.isfinite(zres).sum()), per_series=per)
    if calib is not None:
        q = np.nanquantile(calib.reshape(-1, d.H), QS, axis=0)          # (3, H) on scaled residuals, per step
        Q = np.broadcast_to(q[:, None, None, :], (3, len(eo), d.J, d.H)).copy()
        if roll is not None and len(roll) > 2 and roll[2] == "rollv":
            # volatility scaling: each series' long-run band times its recent typical change (last VOL_K known steps) / its long-run one
            ratio = np.ones((len(eo), d.J))
            for j in range(d.J):
                li = d.last_known(j)[eo]
                for i, L in enumerate(li):
                    if L > VOL_K:
                        seg = d.Y[L - VOL_K:L + 1, j]
                        v = np.nanmean(np.abs(np.diff(seg)))
                        if np.isfinite(v):
                            ratio[i, j] = np.clip(v / d.scale[j], 1.0, 4.0)        # widen in turbulent stretches, never narrower
            Q = Q * ratio[None, :, :, None]
            roll = None
        if roll is not None:
            for i, n in enumerate(eo):
                rq = _roll_q(d, int(d.tau[n]), roll)
                if rq is not None:
                    if len(roll) > 2 and roll[2] == "rollmax":       # never narrower than the long-run band
                        rq = np.stack([np.minimum(rq[0], Q[0, i]), rq[1], np.maximum(rq[2], Q[2, i])])
                    Q[:, i] = np.where(np.isfinite(rq), rq, Q[:, i])
        inside = (zres >= Q[0]) & (zres <= Q[2])
        okz = np.isfinite(zres)
        out["coverage80"] = round(float(inside[okz].mean()), 4) if okz.any() else None
        pl = []
        for k, qq in enumerate(QS):
            dlt = zres - Q[k]
            pl.append(np.where(dlt >= 0, qq * dlt, (qq - 1) * dlt)[okz])
        out["pinball"] = round(float(np.mean(np.concatenate(pl))), 5) if okz.any() else None
    out["_zres"] = zres
    return out


# ------------------------------------------------------------------------------------------------------------------------------ ledger
def code_hash() -> str:
    h = hashlib.sha256()
    for p in ("forecasting_engine/lab/core.py", "forecasting_engine/lab/panel.py"):
        h.update((ROOT / p).read_bytes())
    return h.hexdigest()[:16]


def ledger() -> list[dict]:
    return [json.loads(x) for x in TRIALS.read_text().splitlines() if x.strip()] if TRIALS.exists() else []


REOPEN_BELOW = 0.02              # a champion under 2 % holdout skill has failed its test; its family reopens
_REF_ROW = {"dev": {"pooled_skill": 0.0, "median_skill": 0.0}, "holdout": {"pooled_skill": 0.0, "median_skill": 0.0}}


def _failed(r: dict | None) -> bool:
    """A champion that did not hold up on the holdout (and is not the reference itself): its development score is no longer the bar."""
    return bool(r and r.get("method") != "ref" and ((r.get("holdout") or {}).get("pooled_skill") or 0.0) < REOPEN_BELOW)


def bar_row(r: dict | None) -> dict | None:
    """What a candidate must beat on both windows: the champion, or the reference when the champion failed its holdout (reopened family).
    Candidates accepted against this bar used the holdout a second time; their rows say so (second_look)."""
    return _REF_ROW if _failed(r) else r


def _corrected(r: dict, led: list[dict], i: int) -> dict:
    """A champion's row with its scores replaced by the latest re-score of the same family and method written after it (rescore=True rows,
    lab.rescore: e.g. round "lagfix", the publication-lag audit of 20 Oct 2026). A re-score never selects anything: it only corrects a
    contaminated score, so the corrected row keeps the champion's identity and the old scores are kept under "uncorrected"."""
    c = next((x for x in reversed(led[i + 1:]) if x.get("rescore") and x.get("family") == r["family"] and x.get("method") == r.get("method")
              and x.get("dev") and x.get("holdout") and not x.get("error")
              and (x.get("drivers") is None or r.get("drivers") is None or x.get("drivers") == r.get("drivers"))), None)
    if c is None:
        return r
    return dict(r, dev=c["dev"], holdout=c["holdout"], corrected_by=dict(round=c.get("round"), utc=c.get("utc"), note=c.get("note")),
                uncorrected=dict(dev=r.get("dev"), holdout=r.get("holdout")))


def champion_row(fid: str) -> dict | None:
    """The accepted rows in ledger order, each replacing the current champion only if it beats it on both windows. Machines that run in
    parallel each compare with the champion they started from, so a later acceptance can be worse than one collected meanwhile. A failed
    champion (holdout under REOPEN_BELOW) is compared through the reference instead (bar_row). A champion re-scored later (rescore rows,
    _corrected) carries its corrected scores, which are then also the bar for the rows after it."""
    cur = None
    led = ledger()
    for i, r in enumerate(led):
        if r["family"] != fid or not r.get("accepted") or r.get("qualified") is False:   # rows before the dev gate carry no "qualified"
            continue
        b = bar_row(cur)
        if cur is None or str(r.get("method")).endswith(("+roll", "+rollp", "+rollmax", "+rollv")) or (
                _better(r.get("dev") or {}, b.get("dev") or {}) and _better(r.get("holdout") or {}, b.get("holdout") or {})):
            cur = _corrected(r, led, i)
    return cur


# ------------------------------------------------------------------------------------------------------------------------------ dev gate
# Selection is decided on development only. The development origins are split into GATE_K consecutive time blocks (rolling-origin test); a candidate qualifies
# for a holdout look only if it beats the champion's pooled skill on the whole development window AND on a strict majority of the
# blocks (a tie counts against). The holdout is then a confirmation only.
GATE_K = 4
GATE_MIN_BLOCK = 12              # fewest origins in a block (score needs >= 10 points per series): fewer blocks when dev is short


def dev_blocks(times: np.ndarray, k: int | None = None) -> list[np.ndarray]:
    """Positions of k consecutive time blocks of the development origins (times = their origin times, e.g. d.tau[dev]); k shrinks so
    each block has at least GATE_MIN_BLOCK origins (one block = the whole window when development is short)."""
    k = GATE_K if k is None else k
    n = len(times)
    k = max(1, min(k, n // GATE_MIN_BLOCK))
    order = np.argsort(np.asarray(times), kind="stable")
    return [np.sort(b) for b in np.array_split(order, k)]


def block_scores(d: "Design", eo: np.ndarray, P: np.ndarray, refs: list[str], blocks: list[np.ndarray]) -> list[float | None]:
    """Pooled skill (core.score) on each development block."""
    return [score(d, eo[b], P[b], refs)["pooled_skill"] for b in blocks]


def _skill(x) -> float:
    return -9.0 if x is None else float(x)


def gate(cand_dev: float | None, cand_blocks: list, bar_dev: float | None, bar_blocks: list) -> dict:
    """The dev gate: strictly better pooled skill on the whole development window and on a strict majority of the blocks."""
    won = sum(_skill(a) > _skill(b) for a, b in zip(cand_blocks, bar_blocks))
    n = len(bar_blocks)
    return dict(qualified=bool(_skill(cand_dev) > _skill(bar_dev) and n > 0 and won * 2 > n), blocks_won=int(won), n_blocks=int(n))


def _bar_dev(d: "Design", refs: list[str], dev: np.ndarray, tf: int, blocks: list[np.ndarray], bar: str, nd: int | None = None) -> tuple[float | None, list]:
    """The bar's development pooled skill and per-block skills, recomputed on today's data with the same scoring (memo_method caches the
    champion's forecasts). The reference scores 0 everywhere by definition."""
    if bar == "ref":
        return 0.0, [0.0] * len(blocks)
    if nd and nd < len(d.fam.drivers):                 # drivers added since the champion's row: the bar keeps the inputs it was chosen
        import copy                                    # with (new drivers are appended, so its set is the first nd), so new data is
        import dataclasses                             # measured against the champion as it was, not handed to it for free
        d = copy.copy(d)
        d.fam = dataclasses.replace(d.fam, drivers=list(d.fam.drivers[:nd]))
    P = memo_method(bar)(d, refs, dev, tf)
    return score(d, dev, P, refs)["pooled_skill"], block_scores(d, dev, P, refs, blocks)


def holdout_looks(rows: list[dict], fid: str | None = None) -> int:
    """Holdout looks actually taken (rows with holdout scores); candidates stopped by the dev gate did not spend one."""
    return sum(1 for r in rows if r.get("holdout") and (fid is None or r.get("family") == fid))


def champion(fid: str) -> str | None:
    cur = champion_row(fid)
    return cur["method"] if cur else None


def _better(a: dict, b: dict) -> bool:
    sa, sb = a.get("pooled_skill"), b.get("pooled_skill")
    sa = -9.0 if sa is None else sa
    sb = -9.0 if sb is None else sb
    return sa > sb


_DESIGNS: dict = {}


def design(fid: str) -> tuple[Design, dict]:
    if fid not in _DESIGNS:
        reg = next(f for f in registry()["families"] if f["id"] == fid)
        _DESIGNS[fid] = (Design(PN.load(fid)), reg)
    return _DESIGNS[fid]


def _idx(d: Design, stamps: list[str]) -> np.ndarray:
    pos = {int(t): i for i, t in enumerate(d.tau)}
    return np.array([pos[pd.Timestamp(s).as_unit("ns").value] for s in stamps if pd.Timestamp(s).as_unit("ns").value in pos], int)


_SCREEN = {"on": False}


def run(method: str, families: list[str] | None = None, round_: str = "", holdout: bool = False, note: str = "", screen: bool = False) -> list[dict]:
    """screen=True: the cheap first stage (every 2nd development origin, smaller trees, no holdout); its rows carry screen=True and are never
    a champion or a promotion candidate: only finalists rerun in full can be."""
    if screen and holdout:
        raise ValueError("a screening run never looks at the holdout")
    _SCREEN["on"] = screen
    rmode = next((m for m in ("+rollmax", "+rollp", "+rollv", "+roll") if method.endswith(m)), None)
    roll = rmode is not None
    fn = globals()["method"](method[: -len(rmode)] if roll else method)
    rmode = rmode[1:] if roll else None
    out = []
    for f in registry()["families"]:
        if not f.get("available") or (families and f["id"] not in families):
            continue
        t0 = time.time()
        row = dict(family=f["id"], method=method, round=round_, note=note, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code=code_hash())
        if f["id"] in _DESIGNS:
            row["drivers"] = len(_DESIGNS[f["id"]][0].fam.drivers)
        try:
            d, reg = design(f["id"])
            row["drivers"] = len(d.fam.drivers)                      # the driver set grows between rounds (panel.extend); the targets never change
            refs = [reg["reference"][s] for s in d.series]
            dev, hold = _idx(d, reg["dev_origins"]), _idx(d, reg["holdout_origins"])
            if screen:
                dev = dev[::2]
                row["screen"] = True
            tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
            Pd = fn(d, refs, dev, tf)
            half = len(dev) // 2
            s_dev = score(d, dev, Pd, refs)
            end_dev = d.kt[np.clip(d.R[dev, -1], 0, d.T - 1)]
            s_cal = score(d, dev[half:], Pd[half:], refs, calib=score(d, dev[:half], Pd[:half], refs)["_zres"],
                          roll=(end_dev, s_dev["_zres"], rmode) if roll else None)
            row["dev"] = {k: v for k, v in s_dev.items() if not k.startswith("_")} | dict(coverage80=s_cal.get("coverage80"), pinball=s_cal.get("pinball"))
            blocks = dev_blocks(d.tau[dev])
            row["dev_blocks"] = block_scores(d, dev, Pd, refs, blocks)
            if holdout:
                ch = champion(f["id"])
                prev = champion_row(f["id"])
                if _failed(prev) and not roll:                     # reopened family: the bar is the reference; this is a second holdout look
                    row["second_look"] = True
                    prev = _REF_ROW
                row["vs"] = ch

                def cal_ok(a, b):
                    if a.get("coverage80") is None or b.get("coverage80") is None or a.get("pinball") is None or b.get("pinball") is None:
                        return False
                    return abs(a["coverage80"] - 0.8) < abs(b["coverage80"] - 0.8) and a["pinball"] <= b["pinball"] * 1.001
                same_points = lambda a, b: (a.get("pooled_skill") or -9) >= (b.get("pooled_skill") or -9) - 0.001  # noqa: E731  (data can change under a name)
                # the dev gate (selection on development only): no holdout look unless it passes
                if prev is None:                                   # no champion yet (the reference's first look): nothing to beat
                    g = dict(qualified=True, blocks_won=None, n_blocks=len(blocks))
                elif roll:                                         # same forecasts, other bands: development calibration decides
                    g = dict(qualified=bool(cal_ok(row["dev"], prev["dev"]) and same_points(row["dev"], prev["dev"])), blocks_won=None,
                             n_blocks=len(blocks))
                else:
                    bar = "ref" if prev is _REF_ROW else str(prev["method"])
                    bar = next((bar[: -len(m)] for m in ("+rollmax", "+rollp", "+rollv", "+roll") if bar.endswith(m)), bar)
                    try:
                        bd, bb = _bar_dev(d, refs, dev, tf, blocks, bar, None if prev is _REF_ROW else prev.get("drivers"))
                        row["vs_dev"], row["vs_dev_blocks"] = bd, bb
                        g = gate(row["dev"]["pooled_skill"], row["dev_blocks"], bd, bb)
                    except Exception as e:  # noqa: BLE001  (the bar cannot be recomputed: fail closed, no look)
                        row["gate_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                        g = dict(qualified=False, blocks_won=None, n_blocks=len(blocks))
                row.update(g)
                if not g["qualified"]:
                    row["accepted"] = False                        # no holdout forecasts, no holdout scores: the look is not spent
                else:
                    Ph = fn(d, refs, hold, tf)
                    rl = None
                    if roll:
                        zh = score(d, hold, Ph, refs)["_zres"]
                        rl = (np.r_[end_dev, d.kt[np.clip(d.R[hold, -1], 0, d.T - 1)]], np.concatenate([s_dev["_zres"], zh]), rmode)
                    s_h = score(d, hold, Ph, refs, calib=s_dev["_zres"], roll=rl)
                    row["holdout"] = {k: v for k, v in s_h.items() if not k.startswith("_")}
                    if roll:                                       # judged on calibration on both windows
                        row["accepted"] = bool(prev is not None and cal_ok(row["dev"], prev["dev"]) and cal_ok(row["holdout"], prev["holdout"])
                                               and same_points(row["dev"], prev["dev"]) and same_points(row["holdout"], prev["holdout"]))
                    else:
                        row["accepted"] = bool(prev is None or (_better(row["dev"], prev["dev"]) and _better(row["holdout"], prev["holdout"]) and
                                                                (row["holdout"].get("median_skill") or -9) >= (prev["holdout"].get("median_skill") or -9) - 0.005))
                    row["holdout_paths"] = _paths(d, hold, Ph)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            row["trace"] = traceback.format_exc()[-1500:]
        row["run_s"] = round(time.time() - t0, 1)
        LAB.mkdir(parents=True, exist_ok=True)
        with TRIALS.open("a") as fh:                            # parallel runs share the ledger
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps(row, default=str) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
        out.append(row)
        print(json.dumps({k: (row.get(k) if k not in ("dev", "holdout") else {kk: vv for kk, vv in (row.get(k) or {}).items() if kk != "per_series"})
                          for k in ("family", "method", "dev", "holdout", "qualified", "blocks_won", "accepted", "error", "run_s")}, default=str), flush=True)
    return out


def _paths(d: Design, eo: np.ndarray, P: np.ndarray, n_origins: int = 8, n_series: int = 4) -> dict:
    """A few forecast paths for the page: up to n_series series, n_origins origins each."""
    pick = np.unique(np.linspace(0, len(eo) - 1, min(n_origins, len(eo))).round().astype(int))
    out = {}
    for j in range(min(n_series, d.J)):
        out[d.series[j]] = [dict(origin=str(pd.Timestamp(d.tau[eo[i]], tz="UTC")), stamps=[str(d.stamps[r]) for r in d.R[eo[i]] if r < d.T],
                                 pred=[None if not np.isfinite(v) else round(float(v), 4) for v in P[i, j]]) for i in pick]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "register-family", "run", "report"])
    ap.add_argument("--method")
    ap.add_argument("--families")
    ap.add_argument("--round", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "register":
        register()
    elif a.cmd == "register-family":
        e = register_family(a.families)
        print(json.dumps({k: e[k] for k in ("id", "available", "start", "dev_start", "holdout_start", "train_from")} | dict(series=len(e["series"]),
                         dev=len(e["dev_origins"]), hold=len(e["holdout_origins"]))))
    elif a.cmd == "run":
        run(a.method, a.families.split(",") if a.families else None, a.round, a.holdout, a.note)
    else:
        from forecasting_engine.lab import report as RP
        r = RP.report()
        print(json.dumps({k: r[k] for k in ("trials", "holdout_looks")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
