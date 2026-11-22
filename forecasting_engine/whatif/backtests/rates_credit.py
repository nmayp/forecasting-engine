"""Interest-rate what-if (docs/methods.md).

Given a path of the ECB policy rate (the what-if input), how do bank lending rates, credit volumes and construction / investment move, country by country.

    policy rate i (monthly mean of the rate in force)                                   given, not forecast
      -> stage 1  composite cost of borrowing r (ECB MIR; NFC, households house purchase)
         ARDL(1; 0..2):  r_t = c + rho r_{t-1} + b0 i_t + b1 i_{t-1} + b2 i_{t-2}      long-run pass-through (b0+b1+b2)/(1-rho), speed 1-rho
      -> stage 2  y/y growth z of a real or credit target (new-loan volumes, MFI loan growth, construction output, building permits, investment)
         z_t = c + phi z_{t-1} + sum_blocks theta_blk * mean(D r over lag block)       D r = r_t - r_{t-12}; blocks of lags 0-3, 4-9, 10-15 months
         (quarterly investment: D4 on quarterly means, blocks 0-1, 2-3, 4-5 quarters)
Coefficients: ridge on standardised inputs, pooled over the euro-area countries first, then each country shrunk towards the pooled vector (fixed penalties
ALPHA_POOL, ALPHA_C per observation, set before any scoring and not tuned), fitted on the last WIN months known at the origin only. Every series enters with its
publication lag (rates_credit_data.LAG). Modes of the policy path after the origin: "mech" the realised path (the what-if), "held" the rate in force at the
origin kept constant, "off" no policy input at all (autoregression only; the mechanism-off ablation, a stand-in for the engine on these new targets).

Policy rate used: main refinancing rate (fixed rate / minimum bid) until 14 Oct 2008, deposit facility rate from 15 Oct 2008 (when the corridor narrowed and
the deposit rate became the floor of the money market); the 1 pp corridor shift is an intercept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import rates_credit_data as RD
from forecasting_engine.whatif.backtests import rates_credit_ext_data as RE

WIN = 150                       # months of fit window
ALPHA_POOL, ALPHA_C = 0.05, 0.5
MIN_ROWS = 30
GEOS = RD.GEO
H, HQ = 12, 4
BLOCKS = {"M": ((0, 3), (4, 9), (10, 15)), "Q": ((0, 1), (2, 3), (4, 5))}
# Pre-registered specifications, compared on the development events only (leave-one-event-out) before the check set is run once.
#   win1: months of the stage-1 fit window; blocks: lag blocks of the rate change in stage 2; alpha2: country shrinkage in stage 2; sign: rate effects clipped to <= 0
SPECS = {"A": dict(win1=150, blocks=BLOCKS, alpha2=0.5, sign=False, feat="actual"),
         "B": dict(win1=150, blocks={"M": ((0, 11),), "Q": ((0, 3),)}, alpha2=2.0, sign=False, feat="actual"),
         "C": dict(win1=150, blocks={"M": ((0, 11),), "Q": ((0, 3),)}, alpha2=2.0, sign=True, feat="actual"),
         "D": dict(win1=150, blocks={"M": ((0, 5), (6, 11)), "Q": ((0, 1), (2, 3))}, alpha2=2.0, sign=True, feat="actual"),
         "F": dict(win1=150, blocks={"M": ((0, 11),), "Q": ((0, 3),)}, alpha2=2.0, sign=True, feat="policy"),
         "G": dict(win1=150, blocks={"M": ((0, 5), (6, 11)), "Q": ((0, 1), (2, 3))}, alpha2=2.0, sign=True, feat="policy"),
         "H": dict(win1=84, blocks={"M": ((0, 11),), "Q": ((0, 3),)}, alpha2=2.0, sign=True, feat="actual", lead=True),
         "E": dict(win1=84, blocks={"M": ((0, 11),), "Q": ((0, 3),)}, alpha2=2.0, sign=True, feat="actual")}
# Follow-up specifications (22 Oct 2026; docs "Follow-up: identification and timing"). exp = (Euribor series, horizon months): stage 1 gets the market-implied
# expected rate gap A_t = Euribor_t - i_t as an extra regressor; for the what-if months it is derived from the given path (mean of the announced path over the
# next h months minus i_t plus the average Euribor premium of the last 24 completed months). iv = shock series used as instrument for the lending-rate change in stage 2.
SPECS.update({
    "T1": dict(SPECS["E"], exp=("eur6m", 6)),
    "T2": dict(SPECS["E"], exp=("eur3m", 3)),
    "T3": dict(SPECS["E"], exp=("eur12m", 12)),
    "T4": dict(SPECS["E"], exp=("eur12m", 12), win1=150),
    "T5": dict(SPECS["E"], exp=("eur12m", 12), lead=True),
    "T6": dict(SPECS["E"], exp=("eur12m", 12), win1=150, mkt=True),       # banks price off the market-implied expected path: regressors are E_t, E_{t-1}, E_{t-2}, E = i + A
    "T7": dict(SPECS["E"], exp=("eur6m", 6), win1=150, mkt=True),
})
# Stage 2 identified with exogenous shocks, on top of the fixed stage-1 choice T6 (pooled 2SLS, see instrument())
SPECS.update({"I1": dict(SPECS["T6"], iv="jk_mp"), "I2": dict(SPECS["T6"], iv="mpd_ois1y"), "I3": dict(SPECS["T6"], iv="jk_pc1")})
# Nowcast follow-up: months between the last published MIR month and the origin are estimated from market rates (bridge equation, see `bridge_fit`)
# instead of the policy-driven recursion; keys of S.ext (monthly means of Euribor / euro-area AAA yield curve, known at month end), nc_win = fit window in months.
SPECS.update({"N1": dict(SPECS["T6"], nc=dict(keys=("eur3m", "eur12m", "yc2y"), win=84)),
              "N2": dict(SPECS["T6"], nc=dict(keys=("eur1m", "eur3m", "eur12m", "yc2y", "yc10y"), win=84)),
              "N3": dict(SPECS["T6"], nc=dict(keys=("eur3m", "eur12m", "yc2y"), win=150)),
              "N4": dict(SPECS["T6"], nc=dict(keys=("eur3m",), win=84, spread=True)),       # spread over Euribor 3m: unit long-run pass-through, AR(1) spread, short-run change
              "N5": dict(SPECS["T6"], nc=dict(keys=("yc2y",), win=84, spread=True)),
              "N6": dict(SPECS["T6"], nc=dict(keys=("eur12m",), win=84, spread=True))})
# Asymmetric and pooled variants (24 Oct 2026; docs "Asymmetric and pooled links"): asym = separate pass-through and lags for rises and falls of the
# driving rate (nonlinear ARDL: partial sums of the positive and of the negative changes of the rate path); eb = country shrinkage learned from the data
# (DerSimonian-Laird between-country variance per coefficient) instead of the fixed penalty ALPHA_C.
SPECS.update({"AS": dict(SPECS["N6"], asym=True), "EB": dict(SPECS["N6"], eb=True), "AE": dict(SPECS["N6"], asym=True, eb=True)})
CFG = dict(SPECS["E"])        # FINAL spec: best mean skill over the 9 targets on the 4 development events (see the doc), fixed before the check set


def use_spec(name: str):
    CFG.clear()
    CFG.update(SPECS[name])
# target -> (kind, driver keys); kind: "lvl" (stage 1 rate level), "yy" (100 dlog yoy), "raw" (already a growth rate)
TARGETS = {"ccob_nfc": ("lvl", ["ccob_nfc"], "M"), "ccob_hh": ("lvl", ["ccob_hh"], "M"),
           "vol_nfc": ("yy", ["ccob_nfc"], "M"), "vol_hh": ("yy", ["ccob_hh"], "M"),
           "lg_nfc": ("raw", ["ccob_nfc"], "M"), "lg_hh": ("raw", ["ccob_hh"], "M"),
           "constr": ("yy", ["ccob_hh", "ccob_nfc"], "M"), "permits": ("yy", ["ccob_hh"], "M"),
           "gfcf": ("yy", ["ccob_nfc"], "Q")}


def _ridge(X, y, lam, b0):
    pen = lam * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + pen, X.T @ y + pen @ b0)


class Store:
    def __init__(self):
        d = RD.load()
        self.W = {k: g.pivot(index="date", columns="geo", values="value").sort_index() for k, g in d[~d.key.isin(["dfr", "mro_fr", "mro_mbr"])].groupby("key")}
        p = {k: d[d.key == k].set_index("date")["value"].sort_index() for k in ("dfr", "mro_fr", "mro_mbr")}
        mro = pd.concat([p["mro_fr"], p["mro_mbr"]]).sort_index()
        mro = mro[~mro.index.duplicated(keep="last")]
        days = pd.date_range("1999-01-01", "2029-12-31", freq="D")
        dfr = p["dfr"].reindex(days.union(p["dfr"].index)).ffill().reindex(days)
        mro_d = mro.reindex(days.union(mro.index)).ffill().reindex(days)
        self.pol = mro_d.where(days < pd.Timestamp("2008-10-15"), dfr)
        self.dfr, self.mro = dfr, mro_d
        self.ipol = self.pol.resample("MS").mean()                         # monthly mean of the rate in force (realised)
        self.ext = RE.load()                                               # shocks and Euribor (lag 0)

    # -- information --------------------------------------------------------------------------------------------------------------------------
    def known(self, key: str, o: pd.Timestamp) -> pd.DataFrame:
        """Series as public at origin o (a timestamp): the reference period must have ended LAG days before."""
        w = self.W[key]
        per = 3 if key == "gfcf" else 1
        av = w.index + pd.DateOffset(months=per) + pd.Timedelta(days=RD.LAG[key])
        k = w[av <= o]
        full = pd.date_range(w.index[0], o, freq="QS" if per == 3 else "MS")
        return k.reindex(full[full <= k.index.max()] if len(k) else full[:0])

    def policy_path(self, o: pd.Timestamp, mode: str) -> pd.Series:
        """Monthly policy rate: realised before o; from o on the realised path ("mech") or the rate in force at o held ("held")."""
        s = self.ipol.copy()
        if mode == "held":
            s[s.index >= o.to_period("M").to_timestamp()] = float(self.pol.loc[o])
        return s


def expect_gap(S: Store, o: pd.Timestamp, I: pd.Series, grid) -> pd.Series:
    """Market-implied expected gap A_t = (expected mean policy rate over the next h months incl. premium) - i_t on `grid`.
    Months before the origin month: observed, A_t = Euribor_t - i_t. From the origin month on: derived from the given policy path I (assumed announced and fully
    priced h months ahead): A_t = mean(I[t..t+h-1]) - I_t + premium, premium = mean over the last 24 completed windows of (Euribor_t - mean realised policy[t..t+h-1])."""
    key, h = CFG["exp"]
    eu = S.ext[key]
    o0 = o.to_period("M").to_timestamp()
    ip = S.ipol
    fwd = pd.concat([ip.shift(-k) for k in range(h)], axis=1).mean(axis=1)           # realised mean over t..t+h-1
    done = fwd.index[fwd.index + pd.DateOffset(months=h - 1) < o0]
    prem = float((eu.reindex(done) - fwd.reindex(done)).dropna().iloc[-24:].mean())
    Ifwd = pd.concat([I.shift(-k) for k in range(h)], axis=1).mean(axis=1)
    A = pd.Series(np.nan, index=grid)
    for t in grid:
        A.loc[t] = (eu.get(t, np.nan) - ip.get(t, np.nan)) if t < o0 else float(Ifwd.get(t, np.nan) - I.get(t, np.nan) + prem)
    return A


# -- helpers ----------------------------------------------------------------------------------------------------------------------------------
def to_target(w: pd.DataFrame, kind: str, per: int) -> pd.DataFrame:
    """Series in the space that is modelled and scored: level (rates), 100 dlog yoy (volumes, indices), or the published growth rate."""
    if kind == "yy":
        lag = 12 if per == 1 else 4
        return 100 * np.log(w.where(w > 0)).diff(lag)
    return w


def qmean(r: pd.Series) -> pd.Series:
    """Quarterly mean (complete quarters only) of a monthly series, on quarter-start stamps."""
    q = r.resample("QS").agg(["mean", "count"])
    return q["mean"].where(q["count"] == 3)


def _updown(I: pd.Series):
    """Partial sums of the rises and of the falls of the rate path (nonlinear ARDL): I = I_0 + Ip + Im."""
    d = I.diff().fillna(0.0)
    return d.clip(lower=0).cumsum(), d.clip(upper=0).cumsum()


def _ard_stage1(y: pd.Series, I: pd.Series, o: pd.Timestamp, A=None):
    """Design rows (1, y_{t-1}, I_t, I_{t-1}, I_{t-2}) -> y_t on the grid of y (asym: the three lags of the rises, then of the falls)."""
    idx = y.index
    if CFG.get("asym"):
        Ip, Im = (z.reindex(idx) for z in _updown(I))
        cols = [pd.Series(1.0, index=idx), y.shift(1), Ip, Ip.shift(1), Ip.shift(2), Im, Im.shift(1), Im.shift(2)]
        X = pd.concat(cols, axis=1)
        X.columns = ["c", "y1", "p0", "p1", "p2", "m0", "m1", "m2"]
        return X
    Ii = I.reindex(idx)
    cols = [pd.Series(1.0, index=idx), y.shift(1), Ii, Ii.shift(1), Ii.shift(2)]
    names = ["c", "y1", "i0", "i1", "i2"]
    if CFG.get("lead"):                                            # anticipation: mean of the next three months of the given policy path
        cols.append(pd.concat([I.reindex(idx + pd.DateOffset(months=k)).set_axis(idx) for k in (1, 2, 3)], axis=1).mean(axis=1))
        names.append("ilead")
    if A is not None:
        cols.append(A.reindex(idx))
        names.append("A")
    X = pd.concat(cols, axis=1)
    X.columns = names
    return X


def _row1(P, I, t, A=None):
    if isinstance(I, tuple):                                       # asym: (partial sums of rises, of falls)
        Ip, Im = I
        o1, o2 = pd.DateOffset(months=1), pd.DateOffset(months=2)
        return np.array([1.0, P.loc[t - o1], Ip.get(t, np.nan), Ip.get(t - o1, np.nan), Ip.get(t - o2, np.nan), Im.get(t, np.nan), Im.get(t - o1, np.nan), Im.get(t - o2, np.nan)])
    r = [1.0, P.loc[t - pd.DateOffset(months=1)], I.get(t, np.nan), I.get(t - pd.DateOffset(months=1), np.nan), I.get(t - pd.DateOffset(months=2), np.nan)]
    if CFG.get("lead"):
        r.append(float(np.mean([I.get(t + pd.DateOffset(months=k), np.nan) for k in (1, 2, 3)])))
    if A is not None:
        r.append(A.get(t, np.nan))
    return np.array(r)


def _bridge_X(y: pd.Series, S: Store, keys) -> pd.DataFrame:
    """rows (1, y_{t-1}, M_t, M_{t-1}) -> y_t, M = market rates of the month (complete months only: known at the origin)"""
    M = S.ext[list(keys)].reindex(y.index)
    if CFG["nc"].get("spread"):                      # y is the spread over market rate M0 (first key); regressors: spread_{t-1}, change of M0 and M1 (if given)
        return pd.concat([pd.Series(1.0, index=y.index), y.shift(1), M.iloc[:, 0].diff()], axis=1)
    X = pd.concat([pd.Series(1.0, index=y.index), y.shift(1), M, M.shift(1).add_suffix("_1")], axis=1)
    return X


def bridge_fit(S: Store, own: dict, nc: dict):
    """pooled ridge, then each country shrunk to the pooled vector (same penalties as stage 1); returns {geo: (raw coefficients)}"""
    rows = {}
    for g, y in own.items():
        if nc.get("spread"):
            y = y - S.ext[nc["keys"][0]].reindex(y.index)
        X = _bridge_X(y, S, nc["keys"]).iloc[-nc["win"]:]
        yw = y.iloc[-nc["win"]:]
        ok = X.notna().all(axis=1) & yw.notna()
        if ok.sum() >= MIN_ROWS:
            rows[g] = (X[ok].to_numpy(), yw[ok].to_numpy())
    if len(rows) < 3:
        return {}
    Xp, yp = np.vstack([r[0] for r in rows.values()]), np.concatenate([r[1] for r in rows.values()])
    mu, sd = Xp.mean(0), Xp.std(0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd < 1e-9] = 1.0
    bp = _ridge((Xp - mu) / sd, yp, ALPHA_POOL * len(yp), np.zeros(Xp.shape[1]))
    out = {}
    for g, (X, yy) in rows.items():
        b = _ridge((X - mu) / sd, yy, ALPHA_C * len(yy), bp) / sd
        b[0] = _ridge((X - mu) / sd, yy, ALPHA_C * len(yy), bp)[0] - float((b[1:] * mu[1:]).sum())
        out[g] = b
    return out


def stage1(S: Store, key: str, o: pd.Timestamp, modes=("mech", "held", "off"), geos=GEOS):
    """Fit the pass-through per country on data known at o; return {mode: DataFrame(monthly path, columns geos, history + forecast to o+H-1)}, and the coefficients."""
    yk = S.known(key, o)
    end = o.to_period("M").to_timestamp() + pd.DateOffset(months=H - 1)
    grid = pd.date_range(yk.index[0], end, freq="MS")
    I_hist = S.policy_path(o, "mech")
    rows, ys, own = {}, {}, {}
    for g in geos:
        y = yk[g].dropna() if g in yk else pd.Series(dtype=float)
        if len(y) < MIN_ROWS + 3:
            continue
        y = y.reindex(pd.date_range(y.index[0], y.index[-1], freq="MS")).interpolate(limit=2)
        if CFG.get("mkt"):
            X = _ard_stage1(y, I_hist.reindex(grid) + expect_gap(S, o, I_hist, grid), o, None)
        else:
            X = _ard_stage1(y, I_hist, o, expect_gap(S, o, I_hist, y.index) if CFG.get("exp") else None)
        Xw = X.iloc[-CFG["win1"]:]
        yw = y.iloc[-CFG["win1"]:]
        ok = Xw.notna().all(axis=1) & yw.notna()
        if ok.sum() >= MIN_ROWS:
            rows[g] = (Xw[ok].to_numpy(), yw[ok].to_numpy())
        own[g] = y
    if len(rows) < 3:
        return {}, {}
    Xp = np.vstack([r[0] for r in rows.values()])
    yp = np.concatenate([r[1] for r in rows.values()])
    mu, sd = Xp.mean(0), Xp.std(0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd < 1e-9] = 1.0
    bp = _ridge((Xp - mu) / sd, yp, ALPHA_POOL * len(yp), np.zeros(Xp.shape[1]))
    lam_eb = None
    if CFG.get("eb"):                                              # between-country sd per coefficient (standardised units), DerSimonian-Laird; penalty = s2 / tau^2
        from forecasting_engine.whatif.backtests import ap_core as APC
        Zp = (Xp - mu) / sd
        s2 = float(np.mean((yp - Zp @ bp) ** 2))
        est, var = {}, {}
        for g, (X, yy) in rows.items():
            Z = (X - mu) / sd
            A_ = Z.T @ Z + 1e-6 * len(yy) * np.eye(Z.shape[1])
            Ai = np.linalg.inv(A_)
            est[g], var[g] = Ai @ (Z.T @ yy), s2 * np.diag(Ai)
        tau = np.array([np.sqrt(max(APC.dl_tau2(np.array([est[g][j] for g in est]), np.array([var[g][j] for g in est])), 1e-8)) for j in range(Zp.shape[1])])
        lam_eb = np.minimum(s2 / tau ** 2, 1e8)
        lam_eb[0] = 0.0
    # "off": random walk with damped momentum, d y_t = a1 d y_{t-1} + a2 d y_{t-2} (no policy input); pooled then shrunk
    coef, paths, sims = {}, {m: {} for m in modes}, {}
    drows = {}
    for g, y in own.items():
        dy = y.diff()
        Xd = pd.concat([pd.Series(1.0, index=y.index), dy.shift(1), dy.shift(2)], axis=1).iloc[-CFG["win1"]:]
        yd = dy.iloc[-CFG["win1"]:]
        ok = Xd.notna().all(axis=1) & yd.notna()
        if ok.sum() >= MIN_ROWS:
            drows[g] = (Xd[ok].to_numpy(), yd[ok].to_numpy())
    Xdp = np.vstack([r[0] for r in drows.values()])
    ydp = np.concatenate([r[1] for r in drows.values()])
    mud, sdd = Xdp.mean(0), Xdp.std(0)
    mud[0], sdd[0] = 0.0, 1.0
    bdp = _ridge((Xdp - mud) / sdd, ydp, ALPHA_POOL * len(ydp), np.zeros(3))
    bridge = bridge_fit(S, own, CFG["nc"]) if CFG.get("nc") else {}
    for g, y in own.items():
        if g not in rows:
            continue
        X, yy = rows[g]
        if lam_eb is not None:
            Z = (X - mu) / sd
            b = np.linalg.solve(Z.T @ Z + np.diag(lam_eb), Z.T @ yy + lam_eb * bp)
        else:
            b = _ridge((X - mu) / sd, yy, ALPHA_C * len(yy), bp)
        # coefficients back in raw units
        braw = b / sd
        braw[0] = b[0] - float((b[1:] * mu[1:] / sd[1:]).sum())
        rho, bi = braw[1], braw[2:].sum()
        if CFG.get("asym"):
            bi = braw[2:5].sum()
        def _step(o0):
            st = np.zeros(13)                                       # cumulative response (pp) of the lending rate to a permanent +1 pp policy step at month 0
            for k in range(13):
                st[k] = (rho * st[k - 1] if k else 0.0) + braw[o0] + (braw[o0 + 1] if k >= 1 else 0.0) + (braw[o0 + 2] if k >= 2 else 0.0) \
                    + (braw[5] if CFG.get("lead") and not CFG.get("asym") else 0.0)
            return st
        step = _step(2)
        if CFG.get("asym"):
            step_dn = _step(5)                                      # response to a permanent -1 pp step is -step_dn (its sign convention: partial sum of falls is negative)
        coef[g] = dict(resp3_dn=float(step_dn[3]) if CFG.get("asym") else None, resp6_dn=float(step_dn[6]) if CFG.get("asym") else None, resp12_dn=float(step_dn[12]) if CFG.get("asym") else None,
                       lr_dn=float(braw[5:8].sum() / (1 - rho)) if CFG.get("asym") and rho < 0.995 else None, resp3=float(step[3]), resp6=float(step[6]), resp12=float(step[12]), rho=float(rho), lr=float(bi / (1 - rho)) if rho < 0.995 else float("nan"), speed=float(1 - rho), impact=float(braw[2]),
                       halflife=float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("nan"), n=int(len(yy)), braw=[float(x) for x in braw])
        L = y.index[-1]
        for m in modes:
            P = pd.Series(np.nan, index=grid)
            paths[m][g] = P
            P.loc[y.index] = y
            if m == "off":
                if g in drows:
                    bd = _ridge((drows[g][0] - mud) / sdd, drows[g][1], ALPHA_C * len(drows[g][1]), bdp)
                    for t in grid[grid > L]:
                        d1 = P.loc[t - pd.DateOffset(months=1)] - P.loc[t - pd.DateOffset(months=2)]
                        d2 = P.loc[t - pd.DateOffset(months=2)] - P.loc[t - pd.DateOffset(months=3)]
                        P.loc[t] = P.loc[t - pd.DateOffset(months=1)] + float(((np.array([1.0, d1, d2]) - mud) / sdd) @ bd)
                continue
            I = S.policy_path(o, m)
            Ag = expect_gap(S, o, I, grid) if CFG.get("exp") else None
            if CFG.get("mkt"):
                I, Ag = I.reindex(grid) + Ag, None
            Iu = _updown(I) if CFG.get("asym") else I
            o0 = o.to_period("M").to_timestamp()
            for t in grid[grid > L]:
                if CFG.get("nc") and g in bridge and t < o0:        # months already over at the origin but not yet published: nowcast from market rates
                    if CFG["nc"].get("spread"):
                        m0 = S.ext[CFG["nc"]["keys"][0]]
                        P.loc[t] = float(m0.loc[t] + np.r_[1.0, P.loc[t - pd.DateOffset(months=1)] - m0.loc[t - pd.DateOffset(months=1)], m0.loc[t] - m0.loc[t - pd.DateOffset(months=1)]] @ bridge[g])
                        continue
                    Mrow = np.r_[1.0, P.loc[t - pd.DateOffset(months=1)], S.ext[list(CFG["nc"]["keys"])].loc[t].to_numpy(),
                                 S.ext[list(CFG["nc"]["keys"])].loc[t - pd.DateOffset(months=1)].to_numpy()]
                    P.loc[t] = float(Mrow @ bridge[g])
                else:
                    P.loc[t] = float(_row1(P, Iu, t, Ag) @ braw)
            # policy-implied lending rate: the same recursion run from the start of the fit window on the policy path alone (no observed y after it),
            # i.e. the part of the lending rate that the policy rate explains (spread and credit-risk moves left out)
            sim = pd.Series(np.nan, index=grid)
            t0 = y.index[-min(CFG["win1"], len(y))]
            sim.loc[t0] = y.loc[t0]
            for t in grid[grid > t0]:
                sim.loc[t] = float(_row1(sim, Iu, t, Ag) @ braw)
            sims.setdefault(m + "_sim", {})[g] = sim
    out = {m: pd.DataFrame(paths[m]) for m in modes} | {k: pd.DataFrame(v) for k, v in sims.items()}
    return out, coef


def _feat(z: pd.Series, Rs: list[pd.Series], per: str):
    P = 12 if per == "M" else 4
    cols = {"c": pd.Series(1.0, index=z.index), "z1": z.shift(1)}
    for n, R in enumerate(Rs):
        D = R - R.shift(P)
        for b, (lo, hi) in enumerate(CFG["blocks"][per]):
            cols[f"r{n}b{b}"] = D.shift(lo).rolling(hi - lo + 1).mean().reindex(z.index)
    return pd.DataFrame(cols)


def instrument(S: Store, shock: str, o: pd.Timestamp, per: str) -> pd.Series:
    """Triangular 23-month sum of the exogenous monetary-policy shock, the weights with which the monthly rate changes enter the stage-2 regressor (mean over lags 0-11
    of the 12-month change). Only shocks of months before the origin month are used. Quarterly targets: quarterly mean of the monthly instrument."""
    sh = S.ext[shock]
    o0 = o.to_period("M").to_timestamp()
    sh = sh[sh.index < o0].fillna(0.0)
    sh = sh.reindex(pd.date_range(sh.index[0], o0 - pd.DateOffset(months=1), freq="MS"), fill_value=0.0)
    w = np.array([(min(m, 22 - m) + 1) / 12.0 for m in range(23)])
    z = pd.Series(np.convolve(sh.to_numpy(), w)[:len(sh)], index=sh.index)
    return qmean(z) if per == "Q" else z


def _tsls(Xe: np.ndarray, Xx: np.ndarray, Z: np.ndarray, y: np.ndarray):
    """2SLS: y on endogenous Xe (n x k) and exogenous Xx (n x m, incl. intercept dummies), instruments Z (n x k) + Xx. Returns coefficients on Xe, first-stage F (first column)."""
    W = np.hstack([Z, Xx])
    PW = W @ np.linalg.pinv(W.T @ W) @ W.T
    Xh = PW @ Xe
    A = np.hstack([Xh, Xx])
    b = np.linalg.pinv(A.T @ A) @ A.T @ y
    # first-stage F of the excluded instruments for the first endogenous column
    r1 = Xe[:, 0] - Xx @ (np.linalg.pinv(Xx.T @ Xx) @ Xx.T @ Xe[:, 0])
    r2 = Xe[:, 0] - W @ (np.linalg.pinv(W.T @ W) @ W.T @ Xe[:, 0])
    n, q = len(y), Z.shape[1]
    F = ((r1 @ r1 - r2 @ r2) / q) / (r2 @ r2 / max(n - W.shape[1], 1))
    return b[:Xe.shape[1]], float(F)


def stage2(S: Store, tgt: str, o: pd.Timestamp, rpaths: dict, geos=GEOS, use_rates=True):
    """Forecast target `tgt` for the months (quarters) o ... o+H-1 given rate paths rpaths[key] (DataFrame monthly history + forecast). Returns
    DataFrame (target months x geos) of the modelled space, plus the pooled theta."""
    kind, dkeys, per = TARGETS[tgt]
    P = 3 if per == "Q" else 1
    w = S.known(tgt, o)
    z_all = to_target(w, kind, P)
    n_f = HQ if per == "Q" else H
    first = o.to_period("Q" if per == "Q" else "M").to_timestamp()
    fut = pd.date_range(first, periods=n_f, freq="QS" if per == "Q" else "MS")
    data, Rg = {}, {}
    for g in geos:
        if g not in z_all:
            continue
        z = z_all[g].dropna()
        if len(z) < MIN_ROWS + 2:
            continue
        z = z.reindex(pd.date_range(z.index[0], z.index[-1], freq="QS" if per == "Q" else "MS"))
        Rs = []
        for k in (dkeys if use_rates else []):
            r = rpaths[k][g] if g in rpaths.get(k, {}) else None
            if r is None or r.isna().all():
                Rs = None
                break
            r = r.dropna()
            Rs.append(qmean(r) if per == "Q" else r)
        if Rs is None:
            continue
        grid = pd.date_range(z.index[0], fut[-1], freq="QS" if per == "Q" else "MS")
        Rg[g] = [x.reindex(grid) for x in Rs]
        zf = z.reindex(grid)
        X = _feat(zf, Rg[g], per)
        data[g] = (zf, X, z.index[-1])
    if len(data) < 3:
        return None, None
    rows = {}
    for g, (zf, X, L) in data.items():
        Xw = X.loc[:L].iloc[-(WIN if per == "M" else WIN // 3):]
        yw = zf.loc[:L].iloc[-(WIN if per == "M" else WIN // 3):]
        ok = Xw.notna().all(axis=1) & yw.notna()
        if ok.sum() >= (MIN_ROWS if per == "M" else 24):
            rows[g] = (Xw[ok].to_numpy(), yw[ok].to_numpy())
    if len(rows) < 3:
        return None, None
    Xp = np.vstack([r[0] for r in rows.values()])
    yp = np.concatenate([r[1] for r in rows.values()])
    mu, sd = Xp.mean(0), Xp.std(0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd < 1e-9] = 1.0
    bp = _ridge((Xp - mu) / sd, yp, ALPHA_POOL * len(yp), np.zeros(Xp.shape[1]))
    out = pd.DataFrame(np.nan, index=fut, columns=list(data))
    theta = {}
    step = pd.DateOffset(months=3 if per == "Q" else 1)
    iv = None
    if CFG.get("iv") and use_rates:                                  # identification: pooled 2SLS with country fixed effects on the full known history
        Zs = instrument(S, CFG["iv"], o, per)
        ycols, rcols = [], [c for c in next(iter(data.values()))[1].columns if c.startswith("r")]
        blk = {}
        for n_ in sorted({c[:2] for c in rcols}):
            blk[n_] = [c for c in rcols if c.startswith(n_)]
        gl = [g for g in data if g in rows]
        ys, Xe_, Xx_, Zz_ = [], [], [], []
        for k_, g in enumerate(gl):
            zf, X, L = data[g]
            Xk = X.loc[:L]
            ok = Xk.notna().all(axis=1) & zf.loc[:L].notna()
            ok &= Zs.reindex(Xk.index).notna()
            if ok.sum() < 24:
                continue
            Xk = Xk[ok]
            xe = Xk[rcols].mean(axis=1).to_numpy()[:, None] if len(rcols) > 1 else Xk[rcols].to_numpy()
            ys.append(zf.loc[:L][ok].to_numpy() - 0.0)
            Xe_.append(xe)
            Xx_.append(Xk[["z1"]].to_numpy())
            Zz_.append(Zs.reindex(Xk.index).to_numpy()[:, None])
            ys[-1] = np.column_stack([ys[-1]])[:, 0]
            Xx_[-1] = np.column_stack([Xx_[-1], np.ones(len(ys[-1]))])          # intercept per country below by demeaning
        if len(ys) >= 3:
            # country fixed effects by within-demeaning
            dm = lambda a: a - a.mean(axis=0, keepdims=True)
            y_ = np.concatenate([dm(a) for a in ys])
            Xe = np.vstack([dm(a) for a in Xe_])
            Xx = np.vstack([dm(a[:, :1]) for a in Xx_])
            Z_ = np.vstack([dm(a) for a in Zz_])
            thv, F = _tsls(Xe, Xx, Z_, y_)
            iv = dict(theta=float(thv[0]), F=F, n=int(len(y_)), nc=len(ys))
    ivr, bp2, mu2, sd2 = {}, None, None, None
    if iv is not None:
        th = min(iv["theta"], 0.0) if CFG["sign"] else iv["theta"]
        for g, (zf, X, L) in data.items():
            if g not in rows:
                continue
            rc = [c for c in X.columns if c.startswith("r")]
            xr = X[rc].mean(axis=1) if len(rc) > 1 else X[rc[0]]
            Xw = X.loc[:L].iloc[-(WIN if per == "M" else WIN // 3):]
            yw = (zf - th * xr).loc[:L].iloc[-(WIN if per == "M" else WIN // 3):]
            ok2 = Xw[["c", "z1"]].notna().all(axis=1) & yw.notna() & xr.loc[Xw.index].notna()
            if ok2.sum() >= (MIN_ROWS if per == "M" else 24):
                ivr[g] = (Xw[["c", "z1"]][ok2].to_numpy(), yw[ok2].to_numpy(), xr)
        if len(ivr) >= 3:
            Xq = np.vstack([r[0] for r in ivr.values()])
            yq = np.concatenate([r[1] for r in ivr.values()])
            mu2, sd2 = Xq.mean(0), Xq.std(0)
            mu2[0], sd2[0] = 0.0, 1.0
            sd2[sd2 < 1e-9] = 1.0
            bp2 = _ridge((Xq - mu2) / sd2, yq, ALPHA_POOL * len(yq), np.zeros(2))
        else:
            iv = None
    for g, (zf, X, L) in data.items():
        if g not in rows:
            continue
        if iv is not None:
            if g not in ivr:
                continue
            Xc, yc, xr = ivr[g]
            b2 = _ridge((Xc - mu2) / sd2, yc, CFG["alpha2"] * len(yc), bp2)
            c_, ph_ = b2[0], b2[1] / sd2[1]
            c_ = c_ - float(ph_ * mu2[1])
            zz = zf.copy()
            for t in zz.index[zz.index > L]:
                x_ = xr.get(t, np.nan)
                if not np.isfinite(x_):
                    break
                zz.loc[t] = float(c_ + ph_ * zz.loc[t - step] + th * x_)
            out[g] = zz.reindex(fut).to_numpy()
            theta[g] = dict(c=round(c_, 4), z1=round(float(ph_), 4), r=round(th, 4), iv_theta=round(iv["theta"], 4), F=round(iv["F"], 2))
            continue
        b = _ridge((rows[g][0] - mu) / sd, rows[g][1], CFG["alpha2"] * len(rows[g][1]), bp)
        if CFG["sign"]:
            for ci, cn in enumerate(data[g][1].columns):
                if cn.startswith("r") and b[ci] > 0:
                    b[ci] = 0.0
        zz = zf.copy()
        for t in zz.index[zz.index > L]:
            x = X.loc[t].to_numpy().copy()
            x[1] = zz.loc[t - step]
            if not np.isfinite(x).all():
                break
            zz.loc[t] = float(((x - mu) / sd) @ b)
        out[g] = zz.reindex(fut).to_numpy()
        theta[g] = dict(zip(X.columns, (b / sd).round(4).tolist()))
    return out, theta
