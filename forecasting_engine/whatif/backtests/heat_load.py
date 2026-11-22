"""Mechanism "heat waves -> electricity demand" (docs/methods.md).

Explicit temperature-response curve of load per country, not a learned black box. For country g and day t (log of the daily mean load, ENTSO-E actual load):

    log y(t) = a + b*trend(t) + dow(t) + holiday(t) + annual Fourier(3) + sh * HDD(Te, Th) + sc * CDD(Te, Tc) + e
    Te(t)    = (T(t) + 0.5 T(t-1) + 0.25 T(t-2)) / 1.75           population-weighted daily mean temperature, with building inertia
    HDD = max(0, Th - Te), CDD = max(0, Te - Tc), sh >= 0, sc >= 0

Th and Tc (balance points, deg C) are country-specific and found by a grid search of the fit error; sh and sc are the heating and cooling slopes (share of
load per deg C). Weekday profile, public holiday and a slow trend are in the fit; the trend is extrapolated and each origin re-anchors the level on the mean
residual of its last 28 known days. Everything is fitted on data strictly before the first origin of the cell (no data of the window is used).
The hourly version uses the same Th, Tc and one regression per local hour of day (profile and slopes by hour).

Modes: `scenario` = the realised temperature path of the target days is given (a what-if, not a forecast); `forecast` = temperature of the target days is
the day-of-year climatology plus the last observed anomaly decaying with a 3-day time constant (no weather forecast).
Data: ENTSO-E actual load (restricted: back-test only), population-weighted ERA5-type daily temperature of the lab store.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

CCS = ("ES", "PT", "FR", "DE", "IT", "NL", "BE", "PL")
TH_GRID, TC_GRID = np.arange(8, 19), np.arange(15, 27)
COVID = (pd.Timestamp("2020-03-10", tz="UTC"), pd.Timestamp("2020-06-15", tz="UTC"))   # first lockdown: load far from any temperature response
TRAIN_FROM = pd.Timestamp("2015-01-25", tz="UTC")
TAU_ANOM = 3.0
CACHE = os.environ.get("HEAT_CACHE", "")
QUAD = os.environ.get("HEAT_QUAD", "0") == "1"      # variant: convex cooling term (CDD^2 / 10); off in the reported run
# Size-fix variants (21 Oct 2026, second pass). Defaults = the model of the first pass. Override with HEAT_CFG='{"w":[1,.5,.25,.12],"c2":5}'.
#   w       weights of T(t), T(t-1), ... in the effective temperature Te (normalised)
#   c2      None | offset (deg C) of a second cooling threshold Tc2 = Tc + c2: extra slope above it (piecewise, non-negative: saturation or acceleration)
#   slow    0 | N: extra cooling term on the mean of the last N days (heat build-up), above Tc
#   tn      None | offset: extra cooling term on the implied night minimum Tn = 2 T - Tx (popw mean and max), above Tc + offset
#   K       number of annual Fourier harmonics (seasonal and holiday-season shape: the August trough, school terms)
#   anc     length of the level anchor in days
#   anchor  "last28" | "frozen" (anchor window ends at the first origin of the cell minus 2 d, so event days never enter it) | "cool" (last 28 days,
#           only days with Te <= Tc, so a heat spell does not feed the level)
CFG = dict(w=(1.0, 0.5, 0.25), c2=None, slow=0, tn=None, anchor="last28", K=10, anc=7)     # second pass; first pass was K=3, anc=28
if os.environ.get("HEAT_CFG"):
    import json as _json
    CFG.update(_json.loads(os.environ["HEAT_CFG"]))
EXTRA_TERMS = ("cdd2", "slow", "tn")


def n_nonneg(names: list[str]) -> int:
    return sum(n in ("hdd", "cdd", "cdd2", "slow", "tn") for n in names)


def inputs() -> dict:
    """hourly load (UTC), daily temperature (popw), both for the 8 ENTSO-E countries. HEAT_CACHE/<name>.parquet if present (speed), else the lab panel."""
    out = {}
    for k in ("load", "temp", "tx"):
        f = f"{CACHE}/{k}.parquet"
        if CACHE and os.path.exists(f):
            out[k] = pd.read_parquet(f)
    if "tx" not in out:
        from forecasting_engine.lab import panel as PA
        out["tx"] = PA._weather("tx")
        if CACHE:
            out["tx"].to_parquet(f"{CACHE}/tx.parquet")
    if len(out) < 3:
        from forecasting_engine.lab import panel as PA
        P = PA.Pool()
        out.update({"load": P.get("load"), "temp": P.get("temp")})
        if CACHE:
            for k in ("load", "temp"):
                out[k].to_parquet(f"{CACHE}/{k}.parquet")
    return out


def calendar(idx: pd.DatetimeIndex, cc: str) -> pd.DataFrame:
    from forecasting_engine import histories as HS
    return HS.calendar_frame(idx, cc)


def eff_temp(T: pd.Series) -> pd.Series:
    w = CFG["w"]
    return sum(wi * T.shift(i) for i, wi in enumerate(w)) / sum(w)


def feats(T: pd.Series, Tx: pd.Series | None = None) -> np.ndarray | pd.DataFrame:
    """daily feature frame: te (effective temperature), slow (mean of the last CFG.slow days), tn (implied night minimum 2T - Tx, same inertia weights).
    With no extra term configured it is the plain Te series (first-pass behaviour)."""
    te = eff_temp(T)
    if not (CFG["slow"] or CFG["tn"] is not None):
        return te
    F = pd.DataFrame({"te": te})
    F["slow"] = T.rolling(int(CFG["slow"]), min_periods=int(CFG["slow"])).mean() if CFG["slow"] else 0.0
    F["tn"] = eff_temp(2 * T - Tx.reindex(T.index)) if (CFG["tn"] is not None and Tx is not None) else 0.0
    return F


def design(idx: pd.DatetimeIndex, Te: np.ndarray, cc: str, t0: pd.Timestamp, th: float | None, tc: float | None) -> tuple[np.ndarray, list[str]]:
    """Te: effective temperature (n,) or, with CFG.slow / CFG.tn, the array (n, 3) [te, slow, tn] of `feats`"""
    X2 = None
    if np.ndim(Te) == 2:
        X2 = Te[:, 1:]
        Te = Te[:, 0]
    cal = calendar(idx, cc)
    doy = idx.dayofyear.to_numpy() / 365.25
    cols, names = [np.ones(len(idx)), (idx - t0).days.to_numpy() / 365.25], ["const", "trend"]
    for d in range(1, 7):
        cols.append((cal["weekday"].to_numpy() == d).astype(float)); names.append(f"dow{d}")
    cols.append(cal["holiday"].to_numpy()); names.append("holiday")
    for k in range(1, int(CFG["K"]) + 1):
        cols += [np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)]; names += [f"s{k}", f"c{k}"]
    if th is not None:
        cols.append(np.maximum(0.0, th - Te)); names.append("hdd")
    if tc is not None:
        cols.append(np.maximum(0.0, Te - tc)); names.append("cdd")
        if QUAD:
            cols.append(np.maximum(0.0, Te - tc) ** 2 / 10); names.append("cdd2")
        if CFG["c2"] is not None:
            cols.append(np.maximum(0.0, Te - tc - CFG["c2"])); names.append("cdd2")
        if CFG["slow"]:
            cols.append(np.maximum(0.0, X2[:, 0] - tc)); names.append("slow")
        if CFG["tn"] is not None:
            cols.append(np.maximum(0.0, X2[:, 1] - (tc + CFG["tn"]))); names.append("tn")
    return np.column_stack(cols), names


def _fit(X, y, nonneg_from: int):
    lb = np.r_[np.full(nonneg_from, -np.inf), np.zeros(X.shape[1] - nonneg_from)]
    r = lsq_linear(X, y, bounds=(lb, np.full(X.shape[1], np.inf)), method="bvls")
    return r.x, float(np.mean((y - X @ r.x) ** 2))


class Curve:
    """Fitted temperature-response curve of one country (daily). `fit(y, T, cutoff)` uses days with index < cutoff only."""

    def __init__(self, cc: str):
        self.cc = cc

    def fit(self, y: pd.Series, T: pd.Series, cutoff: pd.Timestamp, Tx: pd.Series | None = None):
        ly = np.log(y.where(y > 0))
        F = feats(T, Tx)
        Te = F["te"] if isinstance(F, pd.DataFrame) else F
        ok = F.reindex(ly.index).notna().all(axis=1).to_numpy() if isinstance(F, pd.DataFrame) else Te.reindex(ly.index).notna().to_numpy()
        m = (ly.index < cutoff) & (ly.index >= TRAIN_FROM) & ly.notna().to_numpy() & ok
        m &= ~((ly.index >= COVID[0]) & (ly.index <= COVID[1]))
        idx = ly.index[m]
        self.t0, self.cut = idx[0], cutoff
        e, yy = self._arr(F, idx), ly[m].to_numpy()
        best = None
        for th in TH_GRID:
            for tc in TC_GRID:
                if tc < th + 2:
                    continue
                X, nm = design(idx, e, self.cc, self.t0, th, tc)
                b, mse = _fit(X, yy, len(nm) - n_nonneg(nm))
                if best is None or mse < best[0]:
                    best = (mse, th, tc, b)
        self.mse, self.th, self.tc, self.b = best
        self.names = design(idx[:3], e[:3], self.cc, self.t0, self.th, self.tc)[1]
        nn = n_nonneg(self.names)
        self.sh, self.sc = self.b[-nn], self.b[-nn + 1]
        self.extra = dict(zip(self.names[-nn + 2:], self.b[-nn + 2:])) if nn > 2 else {}
        # climatology of the effective temperature by day of year (training years only), 15-day smoothing: for forecast mode
        cl = pd.Series(Te.reindex(idx).to_numpy(), index=idx).groupby(idx.dayofyear).mean()
        cl = pd.concat([cl.iloc[-15:], cl, cl.iloc[:15]]).rolling(15, center=True).mean().iloc[15:-15]
        self.clim = cl.reindex(range(1, 367)).interpolate().ffill().bfill()
        self.hist_y, self.hist_T, self.anchor_cap = ly, T, None
        return self

    @staticmethod
    def _arr(F, idx) -> np.ndarray:
        return F.reindex(idx).to_numpy()

    def predict_log(self, idx: pd.DatetimeIndex, Te: np.ndarray) -> np.ndarray:
        X, _ = design(idx, Te, self.cc, self.t0, self.th, self.tc)
        return X @ self.b

    def anchor(self, y_log: pd.Series, T: pd.Series, origin_day: pd.Timestamp, Tx: pd.Series | None = None) -> float:
        """mean residual of the last CFG["anc"] days up to origin_day (inclusive: those days' load is known at the origin).
        CFG.anchor "frozen": the window ends at self.anchor_cap (first origin of the cell minus 2 d) at the latest; "cool": only days with Te <= Tc."""
        F = feats(T, Tx)
        Te = F["te"] if isinstance(F, pd.DataFrame) else F
        if CFG["anchor"] == "frozen" and self.anchor_cap is not None:
            origin_day = min(origin_day, self.anchor_cap)
        sel = (y_log.index <= origin_day) & (y_log.index > origin_day - pd.Timedelta(days=int(CFG["anc"])))
        if CFG["anchor"] == "cool":
            sel &= (Te.reindex(y_log.index) <= self.tc).to_numpy()
        idx = y_log.index[sel]
        ok = F.reindex(idx).notna().all(axis=1).to_numpy() if isinstance(F, pd.DataFrame) else Te.reindex(idx).notna().to_numpy()
        idx = idx[y_log.reindex(idx).notna().to_numpy() & ok]
        if len(idx) < 7:
            return 0.0
        return float(np.mean(y_log.reindex(idx).to_numpy() - self.predict_log(idx, self._arr(F, idx))))

    def forecast_temp(self, T: pd.Series, origin_day: pd.Timestamp, targets: pd.DatetimeIndex) -> np.ndarray:
        """effective temperature of the target days without a weather forecast: climatology + last anomaly decaying with TAU_ANOM days"""
        te = eff_temp(T.loc[:origin_day].dropna())
        an = float(te.iloc[-1] - self.clim.loc[min(te.index[-1].dayofyear, 366)])
        h = np.array([(t - te.index[-1]).days for t in targets], float)
        return np.array([self.clim.loc[min(t.dayofyear, 366)] for t in targets]) + an * np.exp(-h / TAU_ANOM)


def fit_all(inp: dict, cutoff: pd.Timestamp) -> dict[str, Curve]:
    L = inp["load"]
    Ld = L.resample("D").mean()
    return {cc: Curve(cc).fit(Ld[cc], inp["temp"][cc], cutoff, inp["tx"][cc] if "tx" in inp and cc in inp["tx"] else None) for cc in CCS if cc in Ld.columns}


_DAILY: dict = {}


def daily_predict(cv: Curve, inp: dict, origin_day: pd.Timestamp, targets: pd.DatetimeIndex, mode: str = "scenario") -> np.ndarray:
    """MW forecast of the daily mean load of `targets` from the origin (anchor from days <= origin_day)."""
    T = inp["temp"][cv.cc]
    k = (id(inp), cv.cc)
    if k not in _DAILY:     # daily load and its log, computed once per input set (the hourly resample dominated the run time)
        Ld = inp["load"][cv.cc].resample("D").mean()
        _DAILY[k] = (Ld, np.log(Ld.where(Ld > 0)))
    Ld, lLd = _DAILY[k]
    Tx = inp["tx"][cv.cc] if "tx" in inp and cv.cc in inp["tx"] else None
    if mode == "scenario":
        Te = cv._arr(feats(T, Tx), targets)
    else:
        Te = cv.forecast_temp(T, origin_day, targets)
        if CFG["slow"] or CFG["tn"] is not None:
            raise NotImplementedError("forecast mode with extra terms")
    a = cv.anchor(lLd, T, origin_day, Tx)
    return np.exp(cv.predict_log(targets, Te) + a)


# ---------------------------------------------------------------------------------------------------------------------------------- hourly
TZS = {"ES": "Europe/Madrid", "PT": "Europe/Lisbon", "FR": "Europe/Paris", "DE": "Europe/Berlin", "IT": "Europe/Rome", "NL": "Europe/Amsterdam",
       "BE": "Europe/Brussels", "PL": "Europe/Warsaw"}


class HourlyCurve:
    """One regression per local hour of day (log hourly load) with the country's Th, Tc: profile, weekday, holiday, trend, Fourier and the two slopes by hour."""

    def __init__(self, cv: Curve, Lh: pd.Series, T: pd.Series):
        self.cv, self.b = cv, {}
        Te = eff_temp(T)
        loc = Lh.dropna().index.tz_convert(TZS[cv.cc])
        s = Lh.dropna()
        s.index = s.index
        hr = loc.hour.to_numpy()
        day = pd.DatetimeIndex(s.index.floor("D"))
        ok = (s.index < cv.cut) & (s.index >= TRAIN_FROM) & ~((s.index >= COVID[0]) & (s.index <= COVID[1])) & (s > 0).to_numpy()
        ok &= Te.reindex(day).notna().to_numpy()
        self.t0 = cv.t0
        for h in range(24):
            m = ok & (hr == h)
            ix = s.index[m]
            X, nm = design(pd.DatetimeIndex(ix), Te.reindex(pd.DatetimeIndex(ix).floor("D")).to_numpy(), cv.cc, self.t0, cv.th, cv.tc)
            b, _ = _fit(X, np.log(s[m].to_numpy()), len(nm) - n_nonneg(nm))
            self.b[h] = b
        self.Lh, self.Te = s, Te

    def _pred(self, idx_utc: pd.DatetimeIndex, Te_day: np.ndarray) -> np.ndarray:
        hr = idx_utc.tz_convert(TZS[self.cv.cc]).hour.to_numpy()
        X, _ = design(idx_utc, Te_day, self.cv.cc, self.t0, self.cv.th, self.cv.tc)
        return np.array([X[i] @ self.b[hr[i]] for i in range(len(idx_utc))])

    def predict(self, origin: pd.Timestamp, targets: pd.DatetimeIndex, mode: str = "scenario", T: pd.Series | None = None) -> np.ndarray:
        """MW per target hour; anchor = mean residual per local hour of the last 28 days up to origin - 2 h."""
        if mode == "scenario":
            Te = self.Te.reindex(targets.floor("D")).to_numpy()
        else:
            Te = self.cv.forecast_temp(T, origin.floor("D"), pd.DatetimeIndex(targets.floor("D")))
        lo = origin - pd.Timedelta(hours=2)
        past = self.Lh.loc[lo - pd.Timedelta(days=CFG["anc"]):lo]
        past = past[self.Te.reindex(past.index.floor("D")).notna().to_numpy()]
        pr = self._pred(pd.DatetimeIndex(past.index), self.Te.reindex(past.index.floor("D")).to_numpy())
        res = np.log(past.to_numpy()) - pr
        hp = past.index.tz_convert(TZS[self.cv.cc]).hour.to_numpy()
        ht = targets.tz_convert(TZS[self.cv.cc]).hour.to_numpy()
        an = {h: (np.mean(res[hp == h]) if (hp == h).sum() >= 5 else 0.0) for h in range(24)}
        return np.exp(self._pred(targets, Te) + np.array([an[h] for h in ht]))
