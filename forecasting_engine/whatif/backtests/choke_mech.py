"""Shipping chokepoint what-if: transits -> freight rates -> EU import prices -> goods HICP (docs/methods.md).

The user supplies the transit-reduction path at a chokepoint (weekly container or total transits). Three links, each a function of the previous
link's *given* path in scenario mode (and chained in `chain`):

  L1  transits -> freight rate (weekly)     effective-capacity loss K_t = e * phi * EWMA(d_t), d_t = 1 - transits_t / pre-onset mean, d clipped to [0, 1];
                                            ln(R_t / R_0) = beta * K_t.  phi = share of a lane's capacity absorbed by the longer route (Suez -> Cape: +11
                                            days each way on a ~84-day Asia-Europe round trip = 0.21; Panama alternatives +15 %: 0.15), e = share of the
                                            freight index that sails the lane, beta = 10 = log rate rise per unit of effective capacity lost, from the
                                            2021 supply crunch (about 10 % of the fleet absorbed by congestion in 2021-22, WCI composite x3-x7 on 2019)
  L2  freight -> import prices (monthly)    import prices are measured CIF, so freight enters directly:
                                            ln(P_t / P_0) = eps * sum_k w_k (ln F_{t-k} - ln F_0), w_k ~ exp(-k / tau), k = 0..6.
                                            M1: eps = 0.065 (published: OECD 2021, UNCTAD 2021, Carriere-Swallow et al. 2021: a doubling of container
                                            freight raises import prices 4.4-4.9 %), tau = 2 months. M2: eps, tau fitted on development months
  L3  import prices -> goods HICP (monthly) ln(H_t / H_0) = eta * sum_k v_k (ln P_{t-k} - ln P_0) + seasonal(t) - seasonal(0), v_k gamma-shaped over 0..12
                                            months with mean lag 6; M1: eta = 0.25, M2 fitted on development months.
Publication delays: transits 12 d, WCI same week (Thursday print), SCFI weekly, imports prices month + 35 d, HICP month + 17 d (the origin of a
monthly cell is the last month published at the onset date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import choke_data as C

BETA = 10.0
PHI = {"suez": 0.21, "panama": 0.15}
EWMA_A = 0.5                                  # weekly smoothing of the diverted share (half-life one week)
EPS0, TAU0, ETA0 = 0.065, 2.0, 0.25
LAG_L2, LAG_L3 = 7, 13
LAG_IMP_D, LAG_HICP_D = 35, 17                # publication delays, days after month end... see `origin_month`
# share of each freight index that sails each chokepoint lane (assumed for the composites; the Shanghai-Rotterdam lane is the Asia-Europe lane itself)
EXPO = {"wci_rtm": {"suez": 1.0, "panama": 0.0}, "wci": {"suez": 0.35, "panama": 0.05}, "scfi": {"suez": 0.25, "panama": 0.10}}

EVENTS = {   # onset = Monday of the first week the shock was public; chokepoints = (lane key, PortWatch name, column)
    "ever_given": dict(split="dev", onset="2021-03-22", end="2021-07-12", chokes=[("suez", "Suez Canal", "n_container")], label="Ever Given blockage (23-29 Mar 2021)"),
    "crunch_2021": dict(split="dev", onset="2021-05-03", end="2022-04-25", chokes=[("suez", "Suez Canal", "n_container")],
                        label="2021 container crunch (negative control: no chokepoint closure; COVID demand and port congestion)"),
    "blacksea_2022": dict(split="dev", onset="2022-02-21", end="2022-09-05", chokes=[("suez", "Bosporus Strait", "n_total")],
                          label="Black Sea / Bosphorus after 24 Feb 2022 (bulk and tankers; no container lane exposure)"),
    "panama_2023": dict(split="dev", onset="2023-06-05", end="2023-12-04", chokes=[("panama", "Panama Canal", "n_container")],
                        label="Panama Canal drought restrictions 2023"),
    "redsea": dict(split="check", onset="2023-12-11", end="2024-12-02", chokes=[("suez", "Suez Canal", "n_container")],
                   label="Red Sea / Houthi diversions from mid-Dec 2023"),
}
LANE_OF_CHOKE = {"Suez Canal": "suez", "Panama Canal": "panama"}


# ------------------------------------------------------------------------------------------------------------------------ weekly inputs
def weekly_transits(name: str, col: str) -> pd.Series:
    s = C.transits(name, col)
    w = s.resample("W-MON", label="left", closed="left")
    full = w.count() >= 7
    return w.mean().where(full).dropna()


def diverted_share(name: str, col: str, onset: pd.Timestamp, weeks: pd.DatetimeIndex, base_weeks: int = 26) -> np.ndarray:
    """d_t = 1 - transits_t / mean of the base_weeks weeks before the onset week (clipped to [0, 1]); NaN weeks are filled by the last value."""
    w = weekly_transits(name, col)
    base = w[(w.index < onset) & (w.index >= onset - pd.Timedelta(weeks=base_weeks))].mean()
    d = (1 - w.reindex(weeks) / base).clip(0, 1)
    return d.ffill().fillna(0.0).to_numpy()


def ewma(x: np.ndarray, a: float = EWMA_A) -> np.ndarray:
    out, z = np.zeros(len(x)), 0.0
    for i, v in enumerate(x):
        z = a * z + (1 - a) * v
        out[i] = z
    return out


def l1_path(ev: str, series: str, weeks: pd.DatetimeIndex, beta: float = BETA, phi: dict | None = None, expo: dict | None = None,
            d_override: dict | None = None) -> np.ndarray:
    """Predicted ln(R_t / R_0) over `weeks` for freight series `series` given the transit-reduction paths (or d_override = {lane: d array})."""
    phi, expo = phi or PHI, expo or EXPO
    e = EVENTS[ev]
    onset = pd.Timestamp(e["onset"])
    K = np.zeros(len(weeks))
    for lane, name, col in e["chokes"]:
        lk = LANE_OF_CHOKE.get(name)
        if lk is None:                                    # Bosphorus: no container lane in the freight indices
            continue
        d = d_override[lk] if d_override and lk in d_override else diverted_share(name, col, onset, weeks)
        K += expo[series].get(lk, 0.0) * phi[lk] * ewma(d)
    return beta * K


# ------------------------------------------------------------------------------------------------------------------------ monthly series
def freight_monthly() -> pd.DataFrame:
    """Monthly means (Thursday prints / weekly SCFI): scfi, wci, wci_rtm, and the spliced container index lnF (SCFI to 2023-03, then WCI changes)."""
    sc = C.scfi().resample("MS").mean()
    w = C.wci_series()
    wm = w[["wci", "sha_rtm"]].resample("MS").mean()
    wm.loc[(wm["sha_rtm"] / wm["wci"] > 2.5) | (wm["sha_rtm"] / wm["wci"] < 0.4), "sha_rtm"] = np.nan
    out = pd.DataFrame({"scfi": sc, "wci": wm["wci"], "wci_rtm": wm["sha_rtm"]})
    ln = np.log(out["scfi"]).copy()
    lw = np.log(out["wci"])
    last = out["scfi"].dropna().index[-1]
    for t in out.index[out.index > last]:
        prev = ln.get(t - pd.DateOffset(months=1))
        if prev is not None and not np.isnan(prev) and not np.isnan(lw.get(t - pd.DateOffset(months=1), np.nan)) and not np.isnan(lw.get(t, np.nan)):
            ln.loc[t] = prev + lw[t] - lw[t - pd.DateOffset(months=1)]
    out["lnF"] = ln
    return out


def monthly_panel() -> pd.DataFrame:
    f = freight_monthly()
    d = pd.DataFrame({"lnF": f["lnF"], "wci": f["wci"], "wci_rtm": f["wci_rtm"], "scfi": f["scfi"],
                      "fx": np.log(C.eurusd())})
    for k in ("consumer", "interm", "nonfood", "all"):
        d["imp_" + k] = np.log(C.import_price(k))
    d["hicp"] = np.log(C.hicp("nrg_goods"))
    d["hicp_core"] = np.log(C.hicp("core"))
    return d


# ------------------------------------------------------------------------------------------------------------------------ L2, L3
def lag_weights(tau: float, n: int) -> np.ndarray:
    w = np.exp(-np.arange(n) / tau)
    return w / w.sum()


def gamma_weights(mean: float, n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    shape = 2.0
    w = k ** (shape - 1) * np.exp(-k * shape / mean)
    return w / w.sum()


def distributed(x: np.ndarray, w: np.ndarray, x0: float) -> np.ndarray:
    """y_t = sum_k w_k (x_{t-k} - x0) with x_{s} = x0 for s before the first element (the origin level)."""
    n = len(w)
    pad = np.concatenate([np.full(n - 1, x0), x])
    return np.array([sum(w[k] * (pad[t + n - 1 - k] - x0) for k in range(n)) for t in range(len(x))])


def l2_path(lnF: np.ndarray, lnF0: float, eps: float = EPS0, tau: float = TAU0) -> np.ndarray:
    return eps * distributed(lnF, lag_weights(tau, LAG_L2), lnF0)


def l3_path(lnP: np.ndarray, lnP0: float, eta: float = ETA0, mean: float = 6.0) -> np.ndarray:
    return eta * distributed(lnP, gamma_weights(mean, LAG_L3), lnP0)


def seasonal_factors(lnH: pd.Series, upto: pd.Timestamp, years: int = 8) -> pd.Series:
    """Mean month-over-month log change by calendar month over the `years` years up to `upto`, centred (the non-seasonal drift is removed)."""
    d = lnH.diff().loc[upto - pd.DateOffset(years=years):upto].dropna()
    f = d.groupby(d.index.month).mean()
    return f - f.mean()


def seasonal_path(lnH: pd.Series, origin: pd.Timestamp, months: pd.DatetimeIndex) -> np.ndarray:
    f = seasonal_factors(lnH, origin)
    cum, out, t = 0.0, [], origin
    for m in months:
        while t < m:
            t = t + pd.DateOffset(months=1)
            cum += f.get(t.month, 0.0)
        out.append(cum)
    return np.array(out)


def origin_month(onset: pd.Timestamp, delay_days: int) -> pd.Timestamp:
    """Last month whose value was public on the onset date (month t public at the end of month t + delay_days)."""
    m = onset.to_period("M").to_timestamp()
    while m + pd.DateOffset(months=1) + pd.Timedelta(days=delay_days - 1) > onset:
        m = m - pd.DateOffset(months=1)
    return m
