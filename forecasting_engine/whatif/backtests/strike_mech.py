"""Strikes and closures -> daily transport throughput (air flights per country, port calls / cargo per country). Docs: docs/methods.md,
"Mechanism: strikes and closures to transport and logistics".

baseline(y, cutoff, avoid)  : log y = trend + weekday + public holiday + annual Fourier, fitted on data up to `cutoff` only (COVID months masked, strike days
                              of the catalogue masked), plus a level anchor on the mean residual of the last ANCH clean days.
counterfactual(y, days)     : the same regression fitted on both sides of an event (non-causal, truth side only) with a local residual level; the realised
                              effect of a strike day is e = log y - log counterfactual, a loss share is 1 - exp(e). Placebo sigma = spread of e on clean days.
Strike term (the mechanism)  : on a listed strike day j with intensity I_j the series falls by the share k * I_j; a share u of the loss returns over the next
                              CATCH days (catch-up: rebooked passengers, backlog of ships). k and u are fitted on development events only (leave-one-event-out
                              when scoring a development event).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-09-30"))
CATCH = 3
ANCH = 28
LAG = {"air": 46, "port": 12}          # publication delay in days (lab convention: Eurocontrol 46, PortWatch 12)
K_FOUR = {"air": 6, "port": 4}


def _cal(idx: pd.DatetimeIndex, cc: str) -> pd.DataFrame:
    try:
        from forecasting_engine import histories as HS
        c = HS.calendar_frame(idx, cc)
        return c[["weekday", "holiday"]]
    except Exception:  # noqa: BLE001
        return pd.DataFrame({"weekday": idx.weekday, "holiday": 0.0}, index=idx)


def design(idx: pd.DatetimeIndex, cc: str, K: int, t_ref: pd.Timestamp) -> np.ndarray:
    c = _cal(idx, cc)
    t = ((idx - t_ref).days.to_numpy() / 365.25)
    doy = idx.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    cols = [np.ones(len(idx)), t] + [(c["weekday"].to_numpy() == w).astype(float) for w in range(1, 7)] + [c["holiday"].to_numpy()]
    for k in range(1, K + 1):
        cols += [np.sin(k * doy), np.cos(k * doy)]
    return np.column_stack(cols)


def _mask_days(idx, days, before=0, after=CATCH + 1):
    m = np.zeros(len(idx), bool)
    for d in days:
        m |= (idx >= d - pd.Timedelta(days=before)) & (idx <= d + pd.Timedelta(days=after))
    return m


def _fit(y: pd.Series, cc: str, K: int, train: pd.DatetimeIndex, t_ref):
    X = design(train, cc, K, t_ref)
    ly = np.log(y.reindex(train).to_numpy())
    ok = np.isfinite(ly)
    lam = 1e-3 * np.eye(X.shape[1])
    lam[0, 0] = 0
    beta = np.linalg.solve(X[ok].T @ X[ok] + lam, X[ok].T @ ly[ok])
    return beta


def baseline(y: pd.Series, cc: str, cutoff: pd.Timestamp, horizon: pd.DatetimeIndex, avoid: list, kind: str = "air", years: float = 3.0,
             anchor: int = ANCH) -> pd.Series:
    """Prediction of the strike-free path on `horizon` from data up to `cutoff` (inclusive)."""
    h = y.loc[:cutoff].dropna()
    h = h[h > 0]
    h = h[h.index >= cutoff - pd.Timedelta(days=int(365.25 * years))]
    bad = ((h.index >= COVID[0]) & (h.index <= COVID[1])) | _mask_days(h.index, avoid)
    tr = h.index[~bad]
    span = (tr.max() - tr.min()).days
    K = K_FOUR[kind] if span > 700 else (2 if span > 300 else 0)
    t_ref = cutoff
    beta = _fit(h, cc, K, tr, t_ref)
    res = np.log(h.reindex(tr).to_numpy()) - design(tr, cc, K, t_ref) @ beta
    r = pd.Series(res, index=tr)
    lvl = float(r.iloc[-anchor:].mean()) if len(r) else 0.0
    return pd.Series(np.exp(design(horizon, cc, K, t_ref) @ beta + lvl), index=horizon)


def counterfactual(y: pd.Series, cc: str, days: list, centre: pd.Timestamp, kind: str = "air", half: int = 240) -> tuple[pd.Series, pd.Series]:
    """(counterfactual path, mask of clean days) on [centre-half, centre+half]; strike days (all catalogue days) masked with a catch-up buffer."""
    w = y.loc[centre - pd.Timedelta(days=half):centre + pd.Timedelta(days=half)].dropna()
    w = w[w > 0]
    bad = ((w.index >= COVID[0]) & (w.index <= COVID[1])) | _mask_days(w.index, days, 0, CATCH + 1)
    tr = w.index[~bad]
    K = K_FOUR[kind] if (tr.max() - tr.min()).days > 700 else 2
    beta = _fit(w, cc, K, tr, centre)
    fit = pd.Series(np.exp(design(w.index, cc, K, centre) @ beta), index=w.index)
    r = pd.Series(np.log(w.reindex(tr)) - np.log(fit.reindex(tr)), index=tr)
    loc = r.rolling(43, center=True, min_periods=15).median().reindex(w.index).interpolate(limit_direction="both")
    return fit * np.exp(loc), pd.Series(~bad, index=w.index)


def strike_term(idx: pd.DatetimeIndex, days: list, inten: list, k: float, u: float) -> pd.Series:
    """multiplicative factor on idx: 1 - k I_j on strike day j; + u k I_j / CATCH on the CATCH days after"""
    f = np.ones(len(idx))
    pos = {d: i for i, d in enumerate(idx)}
    for d, a in zip(days, inten):
        loss = min(max(k * a, 0.0), 0.98)
        if d in pos:
            f[pos[d]] *= (1 - loss)
        for m in range(1, CATCH + 1):
            dd = d + pd.Timedelta(days=m)
            if dd in pos and dd not in set(days):
                f[pos[dd]] *= 1 + u * loss / CATCH
    return pd.Series(f, index=idx)
