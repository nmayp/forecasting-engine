"""Mechanism link "energy prices -> pump prices" (docs/methods.md).

Explicit pass-through, not a learned black box. For a country g and fuel p (diesel / gasoline95), at forecast origin o:
  pump price with taxes   W(s) = N(s) + tax wedge(s)        (Weekly Oil Bulletin Monday prices, EUR per 1000 l, with and without taxes)
  pre-tax price           N(s) = product cost + margin      dN(s) = sum_{k=0..K} b_k dX(s-k) + r1 dN(s-1) + r2 dN(s-2) + e
  X(s)                    wholesale product price in EUR/l  (EIA spot ULSD NY Harbor / gasoline US Gulf Coast, USD/gal, / ECB USD per EUR)
  tax wedge               ad valorem part: dW = v dN, v = (1 + VAT rate) estimated from the Oil Bulletin's own with / without-tax prices
                          (slope of dW on dN over the last 104 weeks, clipped to [1, 1.30]); excise held at its last published level
Estimation uses only weeks public at the origin: Oil Bulletin Monday s is public at s + 4 d (OB_DELAY), EIA spot (the Brent family) at its
date + 14 d (lag audit). Coefficients per country are ridge-shrunk towards the pooled (all countries) distributed lag; both are fitted on the last
156 public weeks. Forecast: dX beyond the last public EIA date is zero (no futures), dN is rolled forward recursively, W = W_last + v (N - N_last).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

GAL_L = 3.78541
BBL_L = 158.987
EIA_DELAY = pd.Timedelta(days=14)
OB_DELAY = pd.Timedelta(days=4)
K_LAGS = 6                      # b_0 .. b_5, weeks
WIN = 156
VAT_WIN = 104
LAM_POOL = 5.0                  # ridge of the pooled fit (on standardised dX units), fixed in advance
LAM_CTRY = 40.0                 # shrinkage of a country's coefficients towards the pooled ones, fixed in advance
PROD = {"diesel": ("diesel", "ulsd_nyh"), "gasoline": ("gasoline95", "gasoline_gulf")}
_C: dict = {}


def load():
    if _C:
        return _C
    ob = HS.get_parquet("oil_bulletin/prices.parquet").reset_index()
    ob["date"] = pd.to_datetime(ob["date"], utc=True)
    ob = ob[ob["product"].isin(["diesel", "gasoline95"])]
    for (p, t), x in ob.groupby(["product", "tax"]):
        _C[(p, t)] = x.pivot_table(index="date", columns="country", values="eur_per_1000l").sort_index() / 1000.0
    e = HS.get_parquet("lab/raw/eia_daily.parquet").reset_index()
    e["date"] = pd.to_datetime(e["date"], utc=True)
    _C["eia"] = e.pivot_table(index="date", columns="id", values="value").sort_index()
    fx = HS.get_parquet("lab/raw/ecb_fx.parquet").reset_index()
    fx["date"] = pd.to_datetime(fx["date"], utc=True)
    c = [x for x in fx.columns if x not in ("date", "id", "value")]
    usd = fx[fx["id"] == "USD"].set_index("date")["value"] if "id" in fx else fx.set_index("date")["USD"]
    _C["usd"] = usd.sort_index()
    return _C


def wholesale(key: str, o: pd.Timestamp) -> pd.Series:
    """Price in EUR/l public at o (EIA date + 14 d), daily; key brent_spot is USD/bbl, the products USD/gal."""
    C = load()
    x = C["eia"][key].dropna()
    x = x[x.index + EIA_DELAY <= o]
    usd = C["usd"].reindex(x.index, method="ffill")
    return (x / (BBL_L if key == "brent_spot" else GAL_L) / usd).dropna()


def _ridge(X, y, lam, b0=None):
    b0 = np.zeros(X.shape[1]) if b0 is None else b0
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y + lam * b0)


def _design(dN: np.ndarray, dX: np.ndarray, i: int):
    """Row for week i: dX lags 0..K-1 and dN lags 1, 2."""
    return np.r_[[dX[i - k] for k in range(K_LAGS)], dN[i - 1], dN[i - 2]]


def fit_forecast(fuel: str, o: pd.Timestamp, countries, H: int = 4, src: str | None = None, targets=None, xext: pd.Series | None = None) -> dict:
    """{country: (pump-price forecasts for the next H Mondays after the last public one [EUR/l], last public Monday)} at origin o.
    targets: the dates of the H published rows that follow (the harness steps over weeks the Oil Bulletin skipped at Easter / New Year); the weekly
    recursion is rolled to the last of them and the weeks asked for are returned. Without it, the next H Mondays."""
    C = load()
    ps, key = PROD[fuel]
    key = src or key                                  # src='brent_spot': crude only, no refining margin
    Wt, Nt = C[(ps, "with")], C[(ps, "without")]
    ok = Wt.index + OB_DELAY <= o
    Wt, Nt = Wt[ok], Nt.reindex(Wt.index[ok])
    sL = Wt.index[-1]
    wk = list(range(1, H + 1)) if targets is None else [int(round((pd.Timestamp(t) - sL).days / 7)) for t in targets]
    H = max(wk)
    mon = pd.date_range(sL - pd.Timedelta(weeks=WIN + K_LAGS + 2), sL + pd.Timedelta(weeks=H), freq="7D", tz="UTC")
    wx = wholesale(key, o)
    if xext is not None:                                            # a sampled / scenario path of the product price (EUR/l) after the last public date
        wx = pd.concat([wx, xext[xext.index > wx.index[-1]]])
    X = wx.reindex(wx.index.union(mon)).ffill().reindex(mon)            # Monday value = last public daily value on or before (flat after the last public date)
    nL = len(mon) - H
    dX = np.r_[0.0, np.diff(X.to_numpy())]
    out, rows, ys, who = {}, {}, {}, []
    for g in countries:
        if g not in Wt or g not in Nt:
            continue
        # weeks the Oil Bulletin skipped (Easter, New Year: 1-2 Mondays) are interpolated; left as NaN they made dN NaN for the two origins after every gap
        n = pd.Series(Nt[g].reindex(mon[:nL]).to_numpy()).interpolate(limit=3, limit_area="inside").to_numpy()
        w = pd.Series(Wt[g].reindex(mon[:nL]).to_numpy()).interpolate(limit=3, limit_area="inside").to_numpy()
        if np.isfinite(n[-1]) and np.isfinite(w[-1]) and np.isfinite(n[-WIN:]).sum() > 80:
            dN = np.r_[np.nan, np.diff(n)]
            idx = [i for i in range(K_LAGS, nL) if np.isfinite(dN[i]) and np.isfinite(dN[i - 1]) and np.isfinite(dN[i - 2]) and np.isfinite(dX[i - K_LAGS + 1:i + 1]).all()]
            if len(idx) > 60:
                rows[g] = (np.stack([_design(dN, dX, i) for i in idx]), dN[idx], dN, n, w)
                who.append(g)
    if not who:
        return out
    Xp = np.vstack([rows[g][0] for g in who])
    yp = np.concatenate([rows[g][1] for g in who])
    sc = np.r_[np.full(K_LAGS, np.nanstd(dX[1:nL]) + 1e-9), np.full(2, np.std(yp) + 1e-9)]
    bp = np.linalg.solve((Xp / sc).T @ (Xp / sc) + LAM_POOL * len(who) * np.eye(8), (Xp / sc).T @ yp)
    bp = bp / sc
    for g in who:
        Xg, yg, dN, n, w = rows[g]
        Z = Xg / sc
        b = _ridge(Z, yg, LAM_CTRY, bp * sc) / sc
        dW = np.diff(w[-VAT_WIN:])
        dn = np.diff(n[-VAT_WIN:])
        m = np.isfinite(dW) & np.isfinite(dn)
        v = float(np.clip((dW[m] @ dn[m]) / max(dn[m] @ dn[m], 1e-12), 1.0, 1.30)) if m.sum() > 30 else 1.2
        dn_h = list(dN[nL - 2:nL])                                       # dN(sL-1), dN(sL)
        N_l, W_l = n[-1], w[-1]
        res = []
        cum = 0.0
        for u in range(1, H + 1):
            i = nL - 1 + u
            x = np.r_[[dX[i - k] for k in range(K_LAGS)], dn_h[-1], dn_h[-2]]
            d = float(x @ b)
            dn_h.append(d)
            cum += d
            res.append(W_l + v * cum)
        res = [res[k - 1] for k in wk]
        out[g] = (np.array(res), sL, dict(b=b, v=v, pooled=bp))
    return out


FAM = {"diesel_w": ("diesel", "diesel"), "gasoline_w": ("gasoline", "gasoline95")}


def series_names() -> list[str]:
    """The 27 country series of the pump-price families (Oil Bulletin countries without the EA / EU aggregates and the UK), in design order."""
    return [c for c in sorted(load()[("diesel", "with")].columns) if c not in ("EA", "EU", "UK")]


def capture_origins(ev: str, fid: str, cap: str = "/tmp/casc_cap"):
    """Window origins of a captured cell without loading the design: the row k of the last public Oil Bulletin Monday at each origin, found by matching
    the captured truth (the next four published rows) of the affected series. Returns (affected series indices, [k...])."""
    W = load()[(FAM[fid][1], "with")]
    z = np.load(f"{cap}/{ev}__{fid}.npz", allow_pickle=True)
    ser = series_names()
    A = [int(a) for a in z["A"]]
    cols = W[[ser[j] for j in A]].to_numpy()
    ks, k0 = [], 0
    for i in range(z["Y"].shape[0]):
        y = z["Y"][i][A]
        k = next(k for k in range(k0, len(W) - 4) if np.allclose(cols[k + 1:k + 5].T, y, atol=3e-4, equal_nan=True))
        ks.append(k)
        k0 = k + 1
    return A, ks
