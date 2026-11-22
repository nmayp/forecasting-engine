"""Cross-layer nowcasting round: processing variants of the pooled learner that read the other layers as they were KNOWN at the origin.

    import forecasting_engine.lab.xl_lib      # registers the methods below in core.METHODS (use them inside anchor-<fm>: like any other tree layer)

What is new against core.pooled (same rows, same refits, same anchor, same scoring):
  fac    k factors of the layer-index panel (xl_panel: every other layer, ragged edge kept), the loadings re-estimated at every refit on what was
         known then (a diffusion index, Stock and Watson; with missing values set to the mean this is the usual ragged-edge treatment)
  gap    the bridge-equation quantity: what the other layers learned AFTER the end of the period the target last printed. Per layer, its newest known
         value minus its value at the end of the target's last period (zero when the layer has nothing newer than the target). core's driver features
         are the latest value and its change over the last 91 days, which is not that: a driver that lags the target by less than the target's own
         delay has its whole advantage in the gap
  tp     targeted predictors (Bai and Ng): at each refit the 4 layer indices whose level, and the 4 whose gap, correlate most with the anchor's
         error on the training rows (a hard threshold chosen on what was known then) enter as features: the bridge equation on the best indicators
  chain  each factor carried to the target's period with its own AR(1) (estimated up to the refit): the chained driver forecast
  reg    stress switch known at the origin (the first factor beyond 1.5 sd) and its interactions with the factors and gaps: effects that exist
         only in stress
Learners: xlt_* trees as core's gbm_all_reg (identical parameters), xlr_* ridge on the core features (calendar, simple forecasts, own state) plus the
extras, so that the control without extras (xlr_0) and the old way (xlr_raw: every driver's latest value and 3-month change) are the same model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C
from forecasting_engine.lab import xl_panel as XP

G0 = XP.GRID[0].value
DAYNS = 86400 * 10**9
K_FAC = 4
PCA_DAYS = 2500
PER_D = {"h": 1 / 24, "D": 1.0, "W": 7.0, "M": 30.44}
SAME = {"ip_m": ("ip_long_m",), "ip_long_m": ("ip_m",), "unemp_m": ("ur_long_m",), "ur_long_m": ("unemp_m",), "hicp_m": ("cpi_long_m", "hicp_ea_m"),
        "hicp_ea_m": ("hicp_m", "cpi_long_m"), "cpi_long_m": ("hicp_m", "hicp_ea_m")}
FIXED = ("step", "cal_", "anchor_known", "base_", "fm_", "own_", "others_")
XL_EXTRA: dict = {}                    # family id -> (Zk, Zd, EK, names): columns added to the panel (the synthetic positive control)
_PAN: dict = {}
_VER = {"n": 0}
_LOAD: dict = {}
_LK: dict = {}


def set_extra(fid: str, extra) -> None:
    """Install (or clear, extra=None) synthetic panel columns for a family; every cache that depends on the panel is dropped."""
    if extra is None:
        XL_EXTRA.pop(fid, None)
    else:
        XL_EXTRA[fid] = extra
    _VER["n"] += 1
    _PAN.clear()
    _LOAD.clear()


def _day(ns) -> np.ndarray:
    return np.clip((np.asarray(ns, np.int64) - G0) // DAYNS, 0, len(XP.GRID) - 1).astype(int)


def panel(fid: str):
    """(Zk, Zd, EK, column names) for this family: its own layer and its same-concept layers left out, plus any synthetic extra columns."""
    ex = XL_EXTRA.get(fid)
    key = (fid, _VER["n"])
    if key not in _PAN:
        Zk, Zd, EK = XP.zidx(), XP.zdata(), XP.zend()
        drop = set(SAME.get(fid, ())) | {fid}
        cols = [c for c in Zk.columns if c.split(":")[0] not in drop]
        a, b, c = Zk[cols].to_numpy(np.float32), Zd[cols].to_numpy(np.float32), EK[cols].to_numpy(np.float32)
        names = list(cols)
        if ex is not None:
            a, b, c, names = np.hstack([a, ex[0]]), np.hstack([b, ex[1]]), np.hstack([c, ex[2]]), names + list(ex[3])
        _PAN.clear()
        _PAN[key] = (a, b, c, names)
    return _PAN[key]


def _last_known(d, j):
    key = (d.fam.id, d.T, j)
    if key not in _LK:
        _LK[key] = d.last_known(j)
    return _LK[key]


def state(d, K: np.ndarray, fid: str):
    """Per (origin, series) pair of the rows: the day of the origin, the day the target's last printed period ends, the layers' state at both."""
    Zk, Zd, EK, names = panel(fid)
    n, j = K[:, 0].astype(int), K[:, 1].astype(int)
    uniq, inv = np.unique(n.astype(np.int64) * d.J + j, return_inverse=True)
    un, uj = uniq // d.J, uniq % d.J
    gi = _day(d.tau[un])
    li = np.empty(len(un), int)
    for jj in np.unique(uj):
        m = uj == jj
        li[m] = _last_known(d, jj)[un[m]]
    ok = li >= 0
    tl = _day(np.where(ok, d.stamps.as_unit("ns").asi8[np.clip(li, 0, d.T - 1)], d.tau[un]) + int(PER_D[d.fam.freq] * DAYNS))
    lv = Zk[gi]
    newer = EK[gi] > tl[:, None]
    gap = np.where(newer & ok[:, None], lv - Zd[tl], np.nan)
    return lv, gap, inv, un, gi, tl, ok, uj, li


def loadings(fid: str, rho_ns: int, k: int = K_FAC):
    """Loadings V (columns x k), column scales, factor scales and the factor series on the grid, estimated on the PCA_DAYS days before rho."""
    key = (fid, int(rho_ns // DAYNS), k, _VER["n"])
    if key in _LOAD:
        return _LOAD[key]
    Zk, _, _, names = panel(fid)
    i1 = int(_day(rho_ns))
    i0 = max(0, i1 - PCA_DAYS)
    sub = Zk[i0:i1].astype(np.float64)
    cov = np.isfinite(sub).mean(0)
    keep = cov >= 0.5
    if keep.sum() < k + 2:
        _LOAD[key] = None
        return None
    sd = np.nanstd(sub, 0)
    sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    X = np.nan_to_num(sub[:, keep] / sd[keep])
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = np.zeros((Zk.shape[1], k))
    V[keep] = Vt[:k].T
    Fg = np.nan_to_num(Zk.astype(np.float64) / sd) @ V
    fs = Fg[i0:i1].std(0)
    fs = np.where(fs > 0, fs, 1.0)
    _LOAD[key] = (V, sd, fs, Fg / fs, (S[:k] ** 2) / (S ** 2).sum(), int(keep.sum()))
    return _LOAD[key]


def _phi(Fg: np.ndarray, i1: int) -> np.ndarray:
    a, b = Fg[max(0, i1 - 2500):i1 - 30], Fg[max(0, i1 - 2500) + 30:i1]
    out = []
    for c in range(Fg.shape[1]):
        x, y = a[:, c], b[:, c]
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) + np.abs(y) > 0)
        out.append(float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 100 else 0.0)
    return np.array(out)


def row_extras(d, K, st, L, kinds, rho_ns):
    """Factor-based extras (fac, gap, chain, reg) for the rows of K, from the state of every layer at the origin and at the end of the target's last period."""
    lv, gap, inv, un, gi, tl, ok, uj, li = st
    V, sd, fs, Fg, ev, nk = L
    k = V.shape[1]
    f_u = np.nan_to_num(lv / sd) @ V / fs
    g_u = np.nan_to_num(gap / sd) @ V / fs
    f, g = f_u[inv], g_u[inv]
    cols, names = [], []
    if "fac" in kinds:
        cols.append(f)
        names += [f"xl_f{i + 1}" for i in range(k)]
    if "gap" in kinds:
        cols.append(g)
        names += [f"xl_gap{i + 1}" for i in range(k)]
    if "chain" in kinds:
        phi = _phi(Fg, int(_day(rho_ns)))
        R = np.clip(d.R[K[:, 0].astype(int), K[:, 2].astype(int) - 1], 0, d.T - 1)
        h = (d.stamps.as_unit("ns").asi8[R] - d.tau[K[:, 0].astype(int)]) / DAYNS / 30.44
        cols.append(f * np.power(np.clip(phi, 0, 1)[None, :], np.clip(h, 0, None)[:, None]))
        names += [f"xl_ch{i + 1}" for i in range(k)]
    if "reg" in kinds:
        s = (np.abs(f[:, 0]) > 1.5).astype(float)
        cols.append(s[:, None])
        names.append("xl_stress")
        cols.append(f * s[:, None])
        names += [f"xl_f{i + 1}_x" for i in range(k)]
        cols.append(g * s[:, None])
        names += [f"xl_gap{i + 1}_x" for i in range(k)]
    return (np.hstack(cols).astype(np.float32) if cols else np.zeros((len(K), 0), np.float32)), names


def dgap_extras(d, K, st):
    """The same bridge quantity for the family's own drivers (as core builds them: aggregated to the driver's own grid, with its publication delay):
    the driver's newest known value minus its value at the end of the target's last period, in the driver's standard deviations; 0 when the driver has
    nothing newer than the target, missing when it is not known yet at all."""
    _, _, inv, un, gi, tl, ok, uj, li = st
    tau_u = d.tau[un]
    stamp_ns = d.stamps.as_unit("ns").asi8
    stL = np.where(ok, stamp_ns[np.clip(li, 0, d.T - 1)], 0)
    cols, names = [], []
    for dr in d.fam.drivers:
        arr = dr.data.to_numpy(float)
        cut = int(len(arr) * C.DEV_FROM)
        sd = np.nanstd(arr[:cut]) if np.isfinite(arr[:cut]).any() else 1.0
        sd = sd if sd and np.isfinite(sd) and sd > 0 else 1.0
        out = np.full(len(un), np.nan)
        for jj in np.unique(uj):
            m = uj == jj
            col = C.driver_col(dr, d.series[jj], d.country(jj))
            if col not in dr.data.columns:
                continue
            sr = dr.data[col].dropna()
            if len(sr) < 3:
                continue
            st_t = pd.DatetimeIndex(sr.index).as_unit("ns").asi8
            v = sr.to_numpy(float)
            o = np.argsort(st_t, kind="stable")
            st_t, v = st_t[o], v[o]
            kt = C.known_times(pd.DatetimeIndex(sr.index[o]), dr.avail_h, dr.dayahead_hour)
            ko = np.argsort(kt, kind="stable")
            kts, vs, sts = kt[ko], v[ko], st_t[ko]
            i = np.searchsorted(kts, tau_u[m], "right") - 1
            has = i >= 0
            lat_v = np.where(has, vs[np.clip(i, 0, None)], np.nan)
            lat_s = np.where(has, sts[np.clip(i, 0, None)], -1)
            j2 = np.searchsorted(st_t, stL[m], "right") - 1
            at_v = np.where(j2 >= 0, v[np.clip(j2, 0, None)], np.nan)
            g = np.where(lat_s > stL[m], (lat_v - at_v) / sd, np.where(has, 0.0, np.nan))
            out[m] = np.where(ok[m], g, np.nan)
        cols.append(out[inv])
        names.append(f"xl_dg_{dr.name}")
    return np.stack(cols, 1).astype(np.float32), names


def tp_extras(d, K, st, tr_mask, z, m=4):
    """Targeted predictors: choose by |correlation| with the training errors the m layer indices (levels) and m gaps, then build them for every row."""
    lv_u, gap_u, inv, *_ = st
    zt = np.clip(z[tr_mask], -5, 5)
    out, names = [], []
    for tag, Mu in (("lv", lv_u), ("gap", gap_u)):
        A = Mu[inv][tr_mask]
        ok = np.isfinite(A)
        cnt = ok.sum(0)
        Ac = np.where(ok, A, 0.0)
        mu = Ac.sum(0) / np.maximum(cnt, 1)
        zm = np.where(ok, zt[:, None], 0.0).sum(0) / np.maximum(cnt, 1)
        cov = np.where(ok, (A - mu) * (zt[:, None] - zm), 0.0).sum(0)
        va = np.where(ok, (A - mu) ** 2, 0.0).sum(0)
        vz = np.where(ok, (zt[:, None] - zm) ** 2, 0.0).sum(0)
        r = cov / np.sqrt(np.maximum(va * vz, 1e-12))
        r = np.where(cnt >= max(200, 0.3 * len(zt)), np.abs(r), 0.0)
        top = np.argsort(-r)[:m]
        out.append(Mu[inv][:, top])
        names += [f"xl_tp_{tag}{i + 1}" for i in range(len(top))]
    return np.hstack(out).astype(np.float32), names


def pooled_x(d, refs, eval_origins, train_from, model="gbm", kinds=(), params=None, shrink=1.0, raw="all", k=K_FAC):
    """core.pooled with extra feature columns (kinds) built per refit block from the layer panel. raw: "all" keeps every driver column of the default
    features, "none" keeps only the fixed ones (calendar, simple forecasts, own state), for the ridge learners."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    out = np.full((len(eval_origins), d.J, d.H), np.nan)
    tau = d.tau[eval_origins]
    step = C.REFIT[d.fam.freq].value
    t0 = tau.min()
    blocks = ((tau - t0) // step).astype(int)
    stride = C._stride(d)
    last_rho = t0 + int(blocks.max()) * step
    need = np.union1d(C._train_origins(d, train_from, last_rho, stride), eval_origins)
    X, Z, REF, SC, K = C._features(d, refs, None, True, need, False)
    key = (d.fam.id, len(d.fam.drivers), None, True, False, tuple(refs), C._REGIME, C._STACK)
    names = list(C._FEAT_NAMES[key])
    if model == "ridge":
        idx = [i for i, nm in enumerate(names) if nm != "series_id" and (raw == "all" or nm.startswith(FIXED))]
    else:
        idx = list(range(len(names)))
    Xb = X[:, idx]
    base_names = [names[i] for i in idx]
    st = state(d, K, d.fam.id) if kinds else None
    dg = None
    for b in np.unique(blocks):
        sel = np.where(blocks == b)[0]
        rho = t0 + b * step
        tr = C._train_origins(d, train_from, rho, stride)
        if len(tr) < 30:
            continue
        XX, nn_ = Xb, base_names
        if kinds:
            E, en = np.zeros((len(K), 0), np.float32), []
            if set(kinds) & {"fac", "gap", "chain", "reg"}:
                L = loadings(d.fam.id, int(rho), k)
                if L is None:
                    continue
                E, en = row_extras(d, K, st, L, kinds, int(rho))
            if "dgap" in kinds:
                D_, dn = dgap_extras(d, K, st)
                E, en = np.hstack([E, D_]), en + dn
            if "tp" in kinds:
                T_, tn = tp_extras(d, K, st, np.isin(K[:, 0], tr) & np.isfinite(Z), Z)
                E, en = np.hstack([E, T_]), en + tn
            XX, nn_ = np.hstack([Xb, E]), base_names + en
        rt = np.isin(K[:, 0], tr)
        Xt, zt, kt = XX[rt], Z[rt], K[rt]
        okr = np.isfinite(zt)
        re = np.isin(K[:, 0], eval_origins[sel])
        Xe, refe, sce, ke = XX[re], REF[re], SC[re], K[re]
        if okr.sum() < 50:
            continue
        Xa, za = Xt[okr], zt[okr]
        if model == "gbm":
            pp = dict(params or C.GBM_REG)
            m = HistGradientBoostingRegressor(**{kk: v for kk, v in pp.items() if kk != "random_state"}, early_stopping=False,
                                              random_state=pp.get("random_state", 0), categorical_features=[nn_.index("series_id")])
            keep = np.isfinite(Xa).any(0)
            m.fit(Xa[:, keep], za)
            zp = m.predict(Xe[:, keep])
        else:
            mu = np.nanmean(Xa, 0)
            sd = np.nanstd(Xa, 0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            Xs = np.clip(np.nan_to_num((Xa - mu) / sd), -5, 5)
            m = Ridge(alpha=float(len(Xs)) * 0.1)
            m.fit(Xs, np.clip(za, -20, 20))
            zp = np.clip(m.predict(np.clip(np.nan_to_num((Xe - mu) / sd), -5, 5)), -20, 20)
        pred = refe + sce * zp * shrink
        pos = np.searchsorted(eval_origins[sel], ke[:, 0])
        out[sel[pos], ke[:, 1], ke[:, 2] - 1] = pred
    return out


def _mk(model, kinds, **kw):
    return lambda d, refs, eo, tf: pooled_x(d, refs, eo, tf, model=model, kinds=kinds, **kw)


VARIANTS = {"f": ("fac",), "g": ("gap",), "fg": ("fac", "gap"), "fgc": ("fac", "gap", "chain"), "fgr": ("fac", "gap", "reg"), "tp": ("tp",),
            "tpg": ("tp", "fac", "gap"), "dg": ("dgap",), "dgt": ("dgap", "tp")}
for _t, _k in VARIANTS.items():
    C.METHODS[f"xlt_{_t}"] = _mk("gbm", _k, params=C.GBM_REG)
    C.METHODS[f"xlr_{_t}"] = _mk("ridge", _k, raw="none")
C.METHODS["xlr_0"] = _mk("ridge", (), raw="none")                # control: the same ridge without the other layers
C.METHODS["xlr_raw"] = _mk("ridge", (), raw="all")               # the old way: every driver's latest value and change, no factors
C.METHODS["xlr_rawfg"] = _mk("ridge", ("fac", "gap"), raw="all")
C.METHODS["xlt_0"] = _mk("gbm", (), params=C.GBM_REG)           # = core gbm_all_reg (control, checks that the copy is faithful)
