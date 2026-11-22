"""Cascade benchmark (docs/methods.md): does the engine predict how a shock in one system layer spreads to the others?

    python -m forecasting_engine.cascade.cascade run [--events id,id] [--split dev|check] [--families a,b] [--models m1,m2] [--out NAME]
    python -m forecasting_engine.cascade.cascade aggregate [--dir results/cascade] [--split dev|check|all]

A cell is (event x affected layer = lab family). For each cell every model forecasts from origins that start `K_CAL` steps before the onset
(calibration segment, scored for intervals only) through the event window; trees are fitted once on what was public at the first origin (never
refitted after the onset, core.pooled), every input enters as published (core.known_times), zero-shot models read only the history public at
each origin. Scored on the window origins only:
  skill      pooled MAE skill vs the strict reference (core.m_ref), all steps, affected series
  direction  share of moved series where sign(sum of forecast deviation) = sign(sum of realised deviation), both from the pre-shock path
  timing     |argmax time of the forecast deviation - argmax time of the realised deviation|, days, moved series, median
  magnitude  median ratio mean forecast deviation / mean realised deviation (moved series), and median |log ratio|
  moved      the layer "moved" if >= 30 % of its affected series have |mean deviation| above their own 90th percentile of placebo windows
             (same-length windows on the 3 years before the onset); precision / recall of the predicted layer set per event, in `aggregate`
  coverage   80 % and 90 % intervals from each model's own latest 30 resolved errors per step (as known at each origin)
Pre-shock path = the family's strict-reference rule (naive or seasonal naive) frozen at the onset origin and extended over the window.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C

ROOT = Path(__file__).resolve().parents[2]
CASC = ROOT / "results" / "cascade"
EVENTS = ROOT / "data" / "cascade_events.json"
K_CAL = {"h": 14, "D": 14, "W": 13, "M": 12}          # calibration origins before the onset (hourly / daily families: every 2nd or 3rd day)
MAX_WIN = 30                                           # most window origins for hourly / daily families (every k-th day beyond that)
MIN_WIN = {"h": 1, "D": 1, "W": 8, "M": 6}            # fewest window origins (weekly / monthly families get a longer window)
K_ROLL = 30
MOVED_Q, MOVED_SHARE = 0.90, 0.30
PLACEBO_YEARS = 3
FIRST_K = 5                                            # the first window origins, for the size of the initial reaction (published-estimate check)
OWN_EXTRA = {"energy": {"weather"}, "water": {"weather"}, "environment": {"weather"}}   # physical forcing that is part of the layer's own model
FIXFILE = Path(__file__).with_name("cascade_fix.json")
CANDFILE = Path(__file__).with_name("cascade_cand.json")
FIXAUDIT = Path(__file__).with_name("cascade_fix_audit.json")     # the three fixes the stand-alone audit accepted (air_d, hicp_m, retail_m); cascade_fix.json keeps those the in-ensemble re-score confirmed
OPEN = ("chronos2", "tirex2", "toto2")
ALL_MODELS = ("ref", "K_full", "K_off", "chronos2", "tirex2", "toto2", "c2cov", "LP")


# ------------------------------------------------------------------------------------------------------------------------------ per-origin FM cache
# The champions' trees are anchored on foundation-model forecasts over the whole training history (core.Features, "fm:<model>" references), and
# core caches those by the exact origin set, so every new event (a new refit time) would recompute years of zero-shot forecasts. A foundation
# model forecasts each origin independently, so the forecasts are cached per origin (in the process and on disk, per family and model): a later
# cell of the same family only computes the origins it has not seen.
import hashlib
import os
import pickle

PO_DIR = Path(os.environ.get("FE_PO_LOCAL", "/tmp/forecasting_engine_po"))
_PO: dict = {}
_PO_DIRTY: set = set()


def _po_file(key) -> Path:
    return PO_DIR / (hashlib.sha1(repr(key).encode()).hexdigest()[:24] + ".pkl")


def _po_load(key) -> dict:
    f = _po_file(key)
    try:
        return pickle.loads(f.read_bytes()) if f.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def po_flush() -> None:
    for key in list(_PO_DIRTY):
        f = _po_file(key)
        PO_DIR.mkdir(parents=True, exist_ok=True)
        f.write_bytes(pickle.dumps(_PO[key]))
    _PO_DIRTY.clear()


def _po_wrap(name, fn):
    def f(d, eo):
        eo = np.asarray(eo)
        key = (d.fam.id, name, d.T, len(d.fam.drivers), C._lagsig(d, True))
        if key not in _PO:
            _PO[key] = _po_load(key)
        store, tau = _PO[key], d.tau[eo]
        miss = [i for i, t in enumerate(tau) if int(t) not in store]
        if miss:
            P = fn(d, eo[miss])
            for k, i in enumerate(miss):
                store[int(tau[i])] = np.asarray(P[k])
            _PO_DIRTY.add(key)
        return np.stack([store[int(t)] for t in tau])
    return f


def install_po_cache() -> None:
    for name in list(C.FM):
        if not getattr(C.FM[name], "_po", False):
            w = _po_wrap(name, C.FM[name])
            w._po = True
            C.FM[name] = w


# ------------------------------------------------------------------------------------------------------------------------------ links off
def own_groups(layer: str) -> set[str]:
    return {layer} | OWN_EXTRA.get(layer, set())


def links_off(name: str, layer: str, keep: set[str] | None = None) -> str:
    """The champion with every cross-layer input removed: trees keep only the family's own-layer driver groups (no network / chain / pcmci /
    other layers), covariate foundation models lose their covariates, the sibling-chain reconciliation is dropped. Members that read only the
    family's own history (and other countries of the same layer) are kept. With `keep`, the trees get exactly that group set instead."""
    own = own_groups(layer) if keep is None else set(keep)        # keep: any chosen driver-group set (the link-fix step)

    def tree(groups: list[str], flags: str) -> str:
        g = [x for x in groups if x in own and (keep is not None or x not in ("network", "chain", "pcmci"))]
        return "gbm_only-" + ("+".join(g) if g else "none") + flags

    def leaf(n: str) -> str:
        m = re.match(r"^gbm_only-([a-z+]+?)((?:_(?:half|reg|cs|big|xl|y|s\d+))*)$", n)
        if m:
            return tree(m.group(1).split("+"), m.group(2))
        m = re.match(r"^gbm_all((?:_(?:half|reg|cs|big|xl|y))*)$", n)
        if m:                                           # every driver -> the family's own-layer drivers
            return tree(sorted(own), m.group(1))
        if n == "blend_all_ets":
            return "blend:" + tree(sorted(own), "") + "|ets"
        if n == "blend_all_own":
            return "blend:" + tree(sorted(own), "") + "|gbm_own"
        if n in ("chronos2x", "chronos2s"):
            return "chronos2"
        if n == "tirex2c":
            return "tirex2"
        return n

    def rec(n: str) -> str:
        for pre in ("recon:",):
            if n.startswith(pre):
                return rec(n[len(pre):])
        m = re.match(r"^(adaptj\d*:|adapt:|blend:|median:|trim\d*:)", n)
        if m:
            return m.group(1) + "|".join(rec(x) for x in C.split_members(n[m.end():]))
        m = re.match(r"^(shock\d+w\d+f?\[[a-z0-9|]+\]:|xcovid:|rg:|rgd:|tfm\d+:)", n)
        if m:
            return m.group(1) + rec(n[m.end():])
        m = re.match(r"^anchor-([^:]+):", n)
        if m:
            a = m.group(1)
            a = "chronos2" if a in ("chronos2x", "chronos2s") else a
            return f"anchor-{a}:" + rec(n[m.end():])
        return leaf(n)
    return rec(name)


# ------------------------------------------------------------------------------------------------------------------------------ shock-gated / event-learned links
SHOCKFILE = Path(__file__).with_name("cascade_shock.json")       # {"thr", "ramp", "weights": {"group|family": w}, "w_default"} chosen on development events only
IND_RECENT_D, IND_BASE_D = 7, 28                                  # days: the source's recent mean against its own path over the 28 days before


CTX: dict = {}                                                     # run_cell sets the window positions of the cell being scored


def shock_indicator(d, layer: str, E: np.ndarray) -> dict:
    """Per source group, at each origin of E: how far the group's drivers have just left their own pre-origin path, from data public at that
    origin only. For a driver, stat(t) = mean of the last 7 days up to day t minus the mean of the 28 days before them (daily means of the driver
    published up to t); z = |stat| / (1.4826 x median absolute stat over days public at the first origin of E, i.e. before the cell's onset
    except for the calibration origins); the group's value is the 75th percentile of z over its drivers. Only groups other than the target
    layer's own (own_groups) and 'calendar' count: these are the cross-layer sources."""
    own = own_groups(layer) | {"calendar"}
    tau0 = int(d.tau[E[0]])
    out: dict[str, list] = {}
    for dr in d.fam.drivers:
        if dr.group in own or dr.data is None or len(dr.data) == 0:
            continue
        try:
            x = dr.data.astype(float)
            x = x.mean(axis=1) if x.shape[1] > 1 else x.iloc[:, 0]
            x = x.groupby(x.index.floor("D")).mean().asfreq("D")
            x = x.ffill(limit=10)
            stat = x.rolling(IND_RECENT_D, min_periods=3).mean() - x.shift(IND_RECENT_D).rolling(IND_BASE_D, min_periods=10).mean()
            kt = (x.index.as_unit("ns").asi8 + int((dr.avail_h + 24) * 3600e9))      # day t is complete 24 h later, plus the publication delay
            m0 = kt <= tau0
            sv = stat.to_numpy()[m0]
            sv = sv[np.isfinite(sv)]
            if len(sv) < 60:
                continue
            sc = 1.4826 * np.median(np.abs(sv - np.median(sv)))
            if not sc > 0:
                continue
            pos = np.searchsorted(kt, d.tau[E], "right") - 1
            z = np.where(pos >= 0, np.abs(stat.to_numpy()[np.clip(pos, 0, None)]) / sc, np.nan)
            out.setdefault(dr.group, []).append(z)
        except Exception:  # noqa: BLE001
            continue
    return {g: np.nanpercentile(np.stack(v), 75, axis=0) if len(v) > 1 else v[0] for g, v in out.items()}


def source_of(ind: dict, widx) -> tuple[str | None, np.ndarray]:
    """The source group (the one with the highest indicator over the first 3 window origins, public then) and the gate series max over groups."""
    if not ind:
        return None, np.zeros(0)
    first = {g: float(np.nanmax(v[widx[:3]])) if np.isfinite(v[widx[:3]]).any() else -1.0 for g, v in ind.items()}
    return max(first, key=first.get), np.nanmax(np.stack(list(ind.values())), axis=0)


def gate_alpha(g: np.ndarray, thr: float, ramp: float) -> np.ndarray:
    """Weight of the links-on forecast: 0 below thr, rising to 1 over `ramp` indicator units; NaN indicator = off."""
    return np.clip((np.nan_to_num(g, nan=-1e9) - thr) / max(ramp, 1e-9), 0.0, 1.0)


def _blend(P_on, P_off, a):
    a = np.asarray(a, float).reshape((-1,) + (1,) * (P_on.ndim - 1))
    ok = np.isfinite(P_on) & np.isfinite(P_off)
    return np.where(ok, a * P_on + (1 - a) * P_off, np.where(np.isfinite(P_off), P_off, P_on))


def m_gate(champ: str, layer: str, thr: float | None = None, ramp: float | None = None):
    """(A) Shock-gated links: the champion (links on) only where the cross-layer shock indicator is above thr (ramping over `ramp`), else links off."""
    on, off = C.memo_method(champ), C.memo_method(links_off(champ, layer))

    def f(d, refs, eo, tf):
        cfg = json.loads(SHOCKFILE.read_text()) if SHOCKFILE.exists() else {}
        t, r = thr if thr is not None else cfg.get("thr", 3.0), ramp if ramp is not None else cfg.get("ramp", 1.5)
        _, g = source_of(shock_indicator(d, layer, eo), CTX.get("widx", np.arange(len(eo))))
        return _blend(on(d, refs, eo, tf), off(d, refs, eo, tf), gate_alpha(g, t, r))
    return f


def m_learned(champ: str, layer: str, weights: dict | None = None):
    """(B) Event-learned link strength: w x links-on + (1 - w) x links-off with w per (source group, family) from past shock windows
    (cascade_shock.json 'weights'; fallback 'w_default'). The source group is the one with the highest indicator over the first 3 window origins."""
    on, off = C.memo_method(champ), C.memo_method(links_off(champ, layer))

    def f(d, refs, eo, tf):
        cfg = json.loads(SHOCKFILE.read_text()) if SHOCKFILE.exists() else {}
        W = weights if weights is not None else cfg.get("weights", {})
        ind = shock_indicator(d, layer, eo)
        wi = CTX.get("widx", np.arange(len(eo))[-8:])
        src, _ = source_of(ind, wi)
        w = W.get(f"{src}|{d.fam.id}", W.get(f"*|{d.fam.id}", cfg.get("w_default", 0.5)))
        return _blend(on(d, refs, eo, tf), off(d, refs, eo, tf), np.full(len(eo), w))
    return f


# ------------------------------------------------------------------------------------------------------------------------------ models
def _fm(name):
    return lambda d, refs, eo, tf: C.FM[name](d, eo)


def m_c2cov(shock_names: list[str], shock_group: str | None):
    """Chronos-2 with the shock layer's observed series (the event's carrying drivers, or else the drivers of the origin layer's group) as
    past covariates, plus the calendar; no cross-learning. Equals plain Chronos-2 + calendar where the family has no such driver."""
    def f(d, refs, eo, tf):
        from forecasting_engine.lab import bench as BN
        drs = [x for x in d.fam.drivers if x.name in shock_names]
        if not drs and shock_group:
            drs = [x for x in d.fam.drivers if x.group == shock_group][:8]
        orig = BN._cov_drivers
        BN._cov_drivers = lambda dd, groups=None, select=False: drs
        try:
            return BN.chronos2_P(d, eo, covariates=True, cross=False)
        finally:
            BN._cov_drivers = orig
    f.n_cov = lambda d: len([x for x in d.fam.drivers if x.name in shock_names]) or (
        len([x for x in d.fam.drivers if x.group == shock_group][:8]) if shock_group else 0)
    return f


def m_lp(shock_names: list[str], shock_group: str | None):
    """Local projections (Jorda 2005): for each step k a ridge regression of the scaled deviation from the strict reference on the series' own
    recent change, the simple forecasts, the other countries' change and the shock layer's drivers (level and change) as known at the origin,
    fitted once on origins whose targets were public at the first origin of the cell."""
    from sklearn.linear_model import Ridge

    def f(d, refs, eo, tf):
        drs = [x.name for x in d.fam.drivers if x.name in shock_names]
        if not drs and shock_group:
            drs = [x.name for x in d.fam.drivers if x.group == shock_group][:8]
        F = C.Features(d, refs, groups={f"@{n}" for n in drs}, cross=True)
        tr = C._train_origins(d, tf, int(d.tau[eo].min()), C._stride(d))
        out = np.full((len(eo), d.J, d.H), np.nan)
        if len(tr) < 30:
            return out
        Xt, zt, _, _, kt = F.build(tr)
        Xe, _, refe, sce, ke = F.build(eo)
        use = [i for i, n in enumerate(F.names) if n != "series_id" and not n.startswith("cal_") and n != "step"]
        Xt, Xe = Xt[:, use], Xe[:, use]
        mu, sd = np.nanmean(Xt, 0), np.nanstd(Xt, 0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        prep = lambda X: np.clip(np.nan_to_num((X - mu) / sd), -5, 5)  # noqa: E731
        Xt, Xe = prep(Xt), prep(Xe)
        for k in range(1, d.H + 1):
            mt = (kt[:, 2] == k) & np.isfinite(zt)
            me = ke[:, 2] == k
            if mt.sum() < 50 or not me.any():
                continue
            m = Ridge(alpha=float(mt.sum()) * 0.1).fit(Xt[mt], np.clip(zt[mt], -20, 20))
            zp = np.clip(m.predict(Xe[me]), -20, 20)
            kk = ke[me]
            pos = np.searchsorted(eo, kk[:, 0])
            out[pos, kk[:, 1], k - 1] = refe[me] + sce[me] * zp
        return out
    return f


def fm_tokens(*names) -> list[str]:
    """Foundation-model members named in method strings (the ones whose forecasts are cached on disk)."""
    toks = set()
    for n in names:
        for t in re.findall(r"[a-z0-9_]+", n):
            if t in C.FM or t in C._NEW_FM or t == "chronos":
                toks.add(t)
    return sorted(toks)


def _prewarm(*names):
    """Computes (and caches on disk) every foundation-model member of the given methods for the cell's origins; returns nothing to score."""
    def f(d, refs, eo, tf):
        for t in fm_tokens(*names):
            try:
                C.method(t)(d, refs, eo, tf)
            except Exception as e:  # noqa: BLE001
                print(f"  prewarm {d.fam.id} {t} failed: {e!r:.160}", flush=True)
        return None
    return f


def build_models(d, layer: str, ev: dict, names) -> dict:
    champ = C.champion(d.fam.id) or "gbm_all"
    sd, sg = ev.get("shock_drivers", []), ev.get("shock_group")
    M = {}
    for n in names:
        try:
            _add_model(M, n, d, layer, champ, sd, sg)
        except KeyError as e:                 # a champion that is not a plain lab method (e.g. the bottom-up hicp_ea_m): no engine row for the cell
            print(f"  {d.fam.id}: {n} skipped, unknown method {e}", flush=True)
    return M, champ


def _add_model(M, n, d, layer, champ, sd, sg) -> None:
    if True:
        if n == "K_full":
            M[n] = C.memo_method(champ)
        elif n == "K_off":
            M[n] = C.memo_method(links_off(champ, layer))
        elif n == "K_gate":
            M[n] = m_gate(champ, layer)
        elif n == "K_learn":
            M[n] = m_learned(champ, layer)
        elif n in OPEN:
            M[n] = _fm(n)
        elif n == "c2cov":
            M[n] = m_c2cov(sd, sg)
        elif n == "LP":
            M[n] = m_lp(sd, sg)
        elif n == "ref":
            M[n] = C.m_ref
        elif n == "K_fix":                     # the champion with the development-selected driver groups (cascade_links.select -> cascade_fix.json)
            fx = json.loads(FIXFILE.read_text()).get(d.fam.id) if FIXFILE.exists() else None
            M[n] = C.memo_method(links_off(champ, layer, keep=set(fx["groups"])) if fx else champ)
        elif n == "K_fix0":                    # exploratory: every fix of the audit, including the two the in-ensemble re-score did not confirm
            fx = json.loads(FIXAUDIT.read_text()).get(d.fam.id)
            M[n] = C.memo_method(links_off(champ, layer, keep=set(fx["groups"])) if fx else champ)
        elif n in ("K_c1", "K_c2"):             # candidate link sets from the audit, tried inside the champion (cascade_links.candidates -> cascade_cand.json)
            cd = json.loads(CANDFILE.read_text()).get(d.fam.id, []) if CANDFILE.exists() else []
            i = int(n[-1]) - 1
            if i < len(cd):
                M[n] = C.memo_method(links_off(champ, layer, keep=set(cd[i]["groups"])))
        elif n == "PREWARM":
            M[n] = _prewarm(champ, links_off(champ, layer))
        else:
            M[n] = C.memo_method(n)           # any lab method by name (ablations)


# ------------------------------------------------------------------------------------------------------------------------------ paths
def _steps(d, lead: str = "1") -> list[int]:
    """Steps of an origin's forecast that form the path: lead "1" = the next step (hourly families: the next 24 hours), lead "H" = the longest
    lead (hourly: hours 25-48; otherwise the last step), where a persistence forecast lags the shock by the whole horizon."""
    if lead == "1":
        return list(range(24)) if d.fam.freq == "h" else [0]
    return list(range(24, d.H)) if d.fam.freq == "h" else [d.H - 1]


def frozen_ref(d, j: int, kind: str, L0: int, rows: np.ndarray) -> np.ndarray:
    """The strict-reference rule as of row L0 (last row public at the onset origin), extended to target rows `rows`."""
    li = d.ffi[min(max(L0, 0), d.T - 1), j]
    nv = d.Y[li, j] if li >= 0 else np.nan
    if kind == "naive" or not str(kind).startswith("s"):
        return np.full(rows.shape, nv)
    s = int(str(kind)[1:])
    m = np.ceil((rows - L0) / s).clip(min=1)
    rr = np.clip((rows - m * s).astype(int), 0, d.T - 1)
    v = d.Y[rr, j]
    return np.where(np.isfinite(v), v, nv)


def path_of(d, X: np.ndarray, j: int, st: list[int]) -> np.ndarray:
    """(origins,) path of X[:, j, st] (mean over the steps)."""
    return np.nanmean(X[:, j, :][:, st], axis=1) if len(st) > 1 else X[:, j, st[0]]


def base_path(d, refs, j: int, n_idx: np.ndarray, L0: int, st: list[int]) -> np.ndarray:
    rows = d.R[n_idx][:, st]
    return np.nanmean(frozen_ref(d, j, refs[j], L0, rows), axis=1)


def truth_path(d, j: int, n_idx: np.ndarray, st: list[int]) -> np.ndarray:
    return np.nanmean(d.truth(j)[n_idx][:, st], axis=1)


def placebo_stats(d, refs, j: int, nw: int, o0_tau: int, sub: int = 1) -> np.ndarray:
    """|mean scaled deviation from the frozen reference| (lead-1 path) over same-length windows of origins that ended before the onset (3 years)."""
    ok = np.where(d.ok_origin)[0]
    lo = o0_tau - int(PLACEBO_YEARS * 365.25 * 86400 * 1e9)
    last = d.R.shape[1] - 1
    out = []
    stride = {"h": 14, "D": 14, "W": 4, "M": 1}[d.fam.freq]
    ok = ok[(d.tau[ok] >= lo)]
    st = _steps(d)
    for a in range(0, len(ok) - nw * sub, stride):
        idx = ok[a:a + nw * sub:sub]
        if len(idx) < nw:
            break
        if d.kt[np.clip(d.R[idx[-1], last], 0, d.T - 1)] > o0_tau or d.R[idx[-1], last] >= d.T:
            break
        y = truth_path(d, j, idx, st)
        b = base_path(d, refs, j, idx, int(d.L[idx[0]]), st)
        v = np.nanmean(y - b) / d.scale[j]
        if np.isfinite(v):
            out.append(abs(v))
    return np.array(out)


# ------------------------------------------------------------------------------------------------------------------------------ metrics
def roll_coverage(d, E: np.ndarray, P: np.ndarray, widx: np.ndarray, A: list[int]) -> dict:
    """80 / 90 % interval coverage on the window origins from each series' latest K_ROLL resolved scaled errors per step (known at each origin)."""
    Y = np.stack([d.truth(j)[E] for j in range(d.J)], 1)                 # (n, J, H)
    Z = (Y - P) / d.scale[None, :, None]
    tgt_known = d.kt[np.clip(d.R[E], 0, d.T - 1)]                        # (n, H) when each target is public
    ins = {80: [], 90: []}
    for i in widx:
        tau_i = d.tau[E[i]]
        for k in range(d.H):
            res = np.where(tgt_known[:i, k] <= tau_i)[0][-K_ROLL:]
            if len(res) < 8:
                continue
            Zr = Z[res, :, k][:, A]
            for lvl, (lo, hi) in ((80, (0.1, 0.9)), (90, (0.05, 0.95))):
                q = np.nanquantile(Zr, [lo, hi], axis=0)
                zi = Z[i, A, k]
                ok = np.isfinite(zi) & np.isfinite(q[0])
                ins[lvl].append(((zi >= q[0]) & (zi <= q[1]))[ok])
    return {f"cov{l}": (round(float(np.concatenate(v).mean()), 4) if v and sum(len(x) for x in v) else None) for l, v in ins.items()}


def _lead_block(d, P, widx, A, qplace, caches, lead: str, step_days: float) -> dict:
    """Direction / timing / magnitude / layer-moved for one lead (caches[lead] = {j: (truth path, base path)})."""
    st = _steps(d, lead)
    dirs, tim, mag, moved_pred, pcts, pcts_first = [], [], [], 0, [], []
    nA = 0
    for j in A:
        q = qplace.get(j)
        if q is None or not np.isfinite(q):
            continue
        y, b, y1, b1 = caches[lead][j] + caches["1"][j]
        ph = path_of(d, P[widx], j, st)
        dy, dh = (y - b) / d.scale[j], (ph - b) / d.scale[j]
        okp = np.isfinite(dy) & np.isfinite(dh)
        if okp.sum() < 2:
            continue
        nA += 1
        sh = abs(np.mean(dh[okp]))
        mb = np.nanmean(np.abs(b[okp]))
        if np.isfinite(mb) and mb > 0:
            pcts.append(100 * float(np.mean((ph - b)[okp])) / mb)
            pcts_first.append(100 * float(np.nanmean((ph - b)[:FIRST_K])) / mb)
        moved_pred += int(sh > q)
        sy1 = abs(np.nanmean((y1 - b1) / d.scale[j]))
        if sy1 > q:                                                      # the series really moved (lead-1 realised path)
            dirs.append(int(np.sign(np.sum(dy[okp])) == np.sign(np.sum(dh[okp]))))
            ty, th = int(np.argmax(np.abs(np.where(okp, dy, 0)))), int(np.argmax(np.abs(np.where(okp, dh, 0))))
            tim.append(abs(ty - th) * step_days)
            mag.append(np.mean(dh[okp]) / np.mean(dy[okp]))
    sfx = "" if lead == "1" else "_H"
    return {f"direction{sfx}": round(float(np.mean(dirs)), 3) if dirs else None, f"n_moved{sfx}": len(dirs),
            f"timing_days{sfx}": round(float(np.median(tim)), 2) if tim else None,
            f"mag_ratio{sfx}": round(float(np.median(mag)), 3) if mag else None,
            f"mag_logerr{sfx}": round(float(np.median([abs(math.log(min(max(abs(r), 0.1), 10.0))) for r in mag])), 3) if mag else None,
            f"moved_pred{sfx}": bool(nA and moved_pred / nA >= MOVED_SHARE), f"moved_pred_share{sfx}": round(moved_pred / nA, 3) if nA else None,
            f"pct_pred{sfx}": round(float(np.median(pcts)), 2) if pcts else None,
            f"pct_pred_first{sfx}": round(float(np.median(pcts_first)), 2) if pcts_first else None}


def _onset_block(d, refs, E, widx, P, A, qplace, caches, L0: int) -> dict:
    """What the model says at the onset origin alone (the first window origin): its whole multi-step forecast against the pre-shock path, for the
    series that really moved. A persistence forecast has zero deviation there, so this is the part of the cascade that is anticipated."""
    n0 = E[widx[0]]
    rows = d.R[n0]
    unit = {"h": 1 / 24, "D": 1.0, "W": 7.0, "M": 30.44}[d.fam.freq]
    dirs, mag, tim = [], [], []
    for j in A:
        q = qplace.get(j)
        if q is None or not np.isfinite(q):
            continue
        y1, b1 = caches["1"][j]
        if not abs(np.nanmean((y1 - b1) / d.scale[j])) > q:
            continue
        b = frozen_ref(d, j, refs[j], L0, rows)
        dy, dh = (d.truth(j)[n0] - b) / d.scale[j], (P[widx[0], j, :] - b) / d.scale[j]
        okp = np.isfinite(dy) & np.isfinite(dh)
        if okp.sum() < 2:
            continue
        sy, sh = np.sum(dy[okp]), np.sum(dh[okp])
        dirs.append(int(sy != 0 and sh != 0 and np.sign(sy) == np.sign(sh)))
        mag.append(np.mean(dh[okp]) / np.mean(dy[okp]) if np.mean(dy[okp]) != 0 else np.nan)
        tim.append(abs(int(np.argmax(np.abs(np.where(okp, dy, 0)))) - int(np.argmax(np.abs(np.where(okp, dh, 0))))) * unit)
    mag = [r for r in mag if np.isfinite(r)]
    return dict(direction_O=round(float(np.mean(dirs)), 3) if dirs else None, n_moved_O=len(dirs),
                timing_days_O=round(float(np.median(tim)), 2) if tim else None,
                mag_ratio_O=round(float(np.median(mag)), 3) if mag else None,
                mag_logerr_O=round(float(np.median([abs(math.log(min(max(abs(r), 0.1), 10.0))) for r in mag])), 3) if mag else None)


def cell_metrics(d, refs, E, widx, P, A, qplace, caches, Rf) -> dict:
    """All metrics for one model's forecasts P (len(E) x J x H) in one cell."""
    nidx = E[widx]
    num = den = 0.0
    n1 = d1 = 0.0
    for j in A:
        y = d.truth(j)[nidx]
        ok = np.isfinite(y) & np.isfinite(P[widx][:, j]) & np.isfinite(Rf[widx][:, j])
        num += (np.abs(y - P[widx][:, j]) / d.scale[j])[ok].sum()
        den += (np.abs(y - Rf[widx][:, j]) / d.scale[j])[ok].sum()
        ok1 = ok.copy()
        ok1[1:] = False
        n1 += (np.abs(y - P[widx][:, j]) / d.scale[j])[ok1].sum()
        d1 += (np.abs(y - Rf[widx][:, j]) / d.scale[j])[ok1].sum()
    out = dict(skill=round(1 - num / den, 4) if den > 0 else None, skill_onset=round(1 - n1 / d1, 4) if d1 > 0 else None)
    sp = d.tau[nidx]
    step_days = float(np.median(np.diff(sp)) / 86400e9) if len(sp) > 1 else 1.0
    out.update(_lead_block(d, P, widx, A, qplace, caches, "1", step_days))
    out.update(_lead_block(d, P, widx, A, qplace, caches, "H", step_days))
    out.update(_onset_block(d, refs, E, widx, P, A, qplace, caches, int(d.L[E[widx[0]]])))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ run
def run_cell(ev: dict, cell: dict, names, log=print) -> dict:
    fid, onset, win = cell["family"], pd.Timestamp(ev["onset"], tz="UTC"), int(ev["window_days"])
    t0 = time.time()
    d, reg = C.design(fid)
    refs = [reg["reference"][s] for s in d.series]
    layer = reg["layer"]
    freq = d.fam.freq
    o0 = onset.as_unit("ns").value
    o1 = (onset + pd.Timedelta(days=win)).as_unit("ns").value
    ok = np.where(d.ok_origin)[0]
    w = ok[(d.tau[ok] >= o0) & (d.tau[ok] <= o1)]
    if len(w) < MIN_WIN[freq]:                                   # weekly / monthly families: extend to the fewest origins
        after = ok[d.tau[ok] >= o0][:MIN_WIN[freq]]
        w = after if len(after) >= MIN_WIN[freq] else w
    if len(w) < 2:
        return dict(event=ev["id"], family=fid, error="no origins in the window (outside the family's data)")
    sub = int(math.ceil(len(w) / MAX_WIN)) if freq in ("h", "D") else 1
    w = w[::sub]
    cal = ok[d.tau[ok] < o0][::-1][::sub][:K_CAL[freq]][::-1]
    E = np.concatenate([cal, w])
    widx = np.arange(len(cal), len(E))
    tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
    if len(cal) < 8 or int(d.tau[E[0]]) < d.stamps.as_unit("ns").asi8[tf]:
        return dict(event=ev["id"], family=fid, error=f"too little history before the onset ({len(cal)} calibration origins)")
    ser = cell.get("series", "*")
    A = list(range(d.J)) if ser == "*" else [d.series.index(x) for x in ser if x in d.series]
    # series with data in the window
    A = [j for j in A if np.isfinite(d.truth(j)[E[widx]]).sum() >= 2]
    if not A:
        return dict(event=ev["id"], family=fid, error="no affected series with data")
    L0 = int(d.L[E[widx[0]]])
    caches = {lead: {} for lead in ("1", "H")}
    qplace = {}
    nw = len(widx)
    for lead in ("1", "H"):
        st = _steps(d, lead)
        for j in A:
            caches[lead][j] = (truth_path(d, j, E[widx], st), base_path(d, refs, j, E[widx], L0, st))
    truth_cache = {j: caches["1"][j][0] for j in A}
    base_cache = {j: caches["1"][j][1] for j in A}
    for j in A:
        pl = placebo_stats(d, refs, j, nw, o0, sub)
        qplace[j] = float(np.quantile(pl, MOVED_Q)) if len(pl) >= 10 else None
    pt, ptf = [], []
    for j in A:
        mb = np.nanmean(np.abs(base_cache[j]))
        if np.isfinite(mb) and mb > 0:
            pt.append(100 * float(np.nanmean(truth_cache[j] - base_cache[j])) / mb)
            ptf.append(100 * float(np.nanmean((truth_cache[j] - base_cache[j])[:FIRST_K])) / mb)
    # realised movement
    sy = {j: abs(np.nanmean(truth_cache[j] - base_cache[j]) / d.scale[j]) for j in A}
    judge = [j for j in A if qplace[j] is not None]
    moved_true_j = [j for j in judge if sy[j] > qplace[j]]
    row = dict(event=ev["id"], label=ev.get("label"), split=ev.get("split"), onset=ev["onset"], family=fid, layer=layer, freq=freq, n_series=len(A),
               n_judged=len(judge), n_moved=len(moved_true_j), moved_true=bool(judge and len(moved_true_j) / len(judge) >= MOVED_SHARE),
               origins_window=int(nw), origins_cal=int(len(cal)), expect=cell.get("expect"),
               pct_true=round(float(np.median(pt)), 2) if pt else None, pct_true_first=round(float(np.median(ptf)), 2) if ptf else None, models={})
    Rf = C.m_ref(d, refs, E, tf)
    CTX["widx"] = widx
    M, champ = build_models(d, layer, ev, names)
    row["champion"], row["links_off"] = champ, links_off(champ, layer)
    st1 = _steps(d)
    for mname, fn in M.items():
        t1 = time.time()
        try:
            P = fn(d, refs, E, tf)
        except Exception as e:  # noqa: BLE001
            row["models"][mname] = dict(error=f"{type(e).__name__}: {e}"[:200])
            log(f"  {fid} {mname} failed: {e!r:.200}")
            continue
        if P is None:
            continue
        m = cell_metrics(d, refs, E, widx, P, A, qplace, caches, Rf)
        m.update(roll_coverage(d, E, P, widx, A))
        mv = [j for j in moved_true_j]
        if mv:
            ph = {j: path_of(d, P[widx], j, st1) for j in mv}
            m["path_pred"] = [round(float(np.nanmean([(ph[j][i] - base_cache[j][i]) / d.scale[j] for j in mv])), 2) for i in range(nw)]
        m["s"] = round(time.time() - t1, 1)
        row["models"][mname] = m
    if moved_true_j:
        row["path_true"] = [round(float(np.nanmean([(truth_cache[j][i] - base_cache[j][i]) / d.scale[j] for j in moved_true_j])), 2) for i in range(nw)]
    row["run_s"] = round(time.time() - t0, 1)
    return row


def capture(ids=None, split=None, outdir: str = "/tmp/casc_cap") -> None:
    """Saves per cell the links-on / links-off forecasts, the truth, the reference and the shock indicator of each source group (npz), so gates and
    weights can be tuned offline on development events without refitting."""
    install_po_cache()
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    todo = [(ev, c) for ev in load_events(ids, split) for c in ev["cells"]]
    todo.sort(key=lambda x: (x[1]["family"], x[0]["onset"]))
    for ev, cell in todo:
        f = od / f"{ev['id']}__{cell['family']}.npz"
        if f.exists():
            continue
        try:
            fid, onset, win = cell["family"], pd.Timestamp(ev["onset"], tz="UTC"), int(ev["window_days"])
            d, reg = C.design(fid)
            refs = [reg["reference"][x] for x in d.series]
            layer, freq = reg["layer"], d.fam.freq
            o0, o1 = onset.as_unit("ns").value, (onset + pd.Timedelta(days=win)).as_unit("ns").value
            ok = np.where(d.ok_origin)[0]
            w = ok[(d.tau[ok] >= o0) & (d.tau[ok] <= o1)]
            if len(w) < MIN_WIN[freq]:
                after = ok[d.tau[ok] >= o0][:MIN_WIN[freq]]
                w = after if len(after) >= MIN_WIN[freq] else w
            if len(w) < 2:
                continue
            sub = int(math.ceil(len(w) / MAX_WIN)) if freq in ("h", "D") else 1
            w = w[::sub]
            cal = ok[d.tau[ok] < o0][::-1][::sub][:K_CAL[freq]][::-1]
            E = np.concatenate([cal, w])
            widx = np.arange(len(cal), len(E))
            tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
            if len(cal) < 8:
                continue
            ser = cell.get("series", "*")
            A = list(range(d.J)) if ser == "*" else [d.series.index(x) for x in ser if x in d.series]
            A = [j for j in A if np.isfinite(d.truth(j)[E[widx]]).sum() >= 2]
            if not A:
                continue
            champ = C.champion(fid) or "gbm_all"
            Pon = C.memo_method(champ)(d, refs, E, tf)
            Poff = C.memo_method(links_off(champ, layer))(d, refs, E, tf)
            Rf = C.m_ref(d, refs, E, tf)
            ind = shock_indicator(d, layer, E)
            Y = np.stack([d.truth(j)[E] for j in range(d.J)], 1)
            np.savez_compressed(f, Pon=Pon[widx].astype("f4"), Poff=Poff[widx].astype("f4"), Rf=Rf[widx].astype("f4"), Y=Y[widx].astype("f4"),
                                scale=d.scale, A=np.array(A), groups=np.array(list(ind)), ind=np.stack(list(ind.values())) if ind else np.zeros((0, len(E))),
                                widx=widx, ncal=len(cal), onset=ev["onset"], split=ev["split"], family=fid, event=ev["id"], H=d.H)
            print("cap", ev["id"], fid, list(ind), flush=True)
            po_flush()
        except Exception as e:  # noqa: BLE001
            print("cap fail", ev["id"], cell["family"], repr(e)[:200], flush=True)


def load_events(ids=None, split=None) -> list[dict]:
    ev = json.loads(EVENTS.read_text())["events"]
    return [e for e in ev if (not ids or e["id"] in ids) and (not split or e["split"] == split)]


def run(ids=None, split=None, families=None, names=ALL_MODELS, out: str | None = None, log=print) -> list[dict]:
    rows = []
    install_po_cache()
    todo = [(ev, cell) for ev in load_events(ids, split) for cell in ev["cells"] if not families or cell["family"] in families]
    fam_order = {f: i for i, f in enumerate(dict.fromkeys(c["family"] for _, c in todo))}
    todo.sort(key=lambda x: (fam_order[x[1]["family"]], x[0]["onset"]))                 # cells of one family together: the per-origin cache pays off
    for ev, cell in todo:
        if True:
            try:
                r = run_cell(ev, cell, names, log)
            except Exception as e:  # noqa: BLE001
                r = dict(event=ev["id"], family=cell["family"], error=f"{type(e).__name__}: {e}"[:300])
            try:
                po_flush()
            except Exception as e:  # noqa: BLE001
                print(f"po flush failed: {e!r:.100}", flush=True)
            rows.append(r)
            print("CASC " + json.dumps(r, default=str), flush=True)             # rows can be parsed back from a saved log
    if out:
        CASC.mkdir(parents=True, exist_ok=True)
        (CASC / f"cells_{out}.json").write_text(json.dumps(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code=C.code_hash(),
                                                                models=list(names), rows=rows), default=str))
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ aggregate
def _mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return round(float(np.mean(x)), 4) if x else None


BEST_OPEN = ("chronos2", "tirex2", "toto2", "c2cov", "LP")


def _get(r, m, key):
    c = r["models"].get(m)
    return None if not c or "error" in c else c.get(key)


def _best(r, key, names=BEST_OPEN, hi=True):
    v = [_get(r, m, key) for m in names]
    v = [x for x in v if x is not None]
    return (max(v) if hi else min(v)) if v else None


def _boot(rows, fn, n=2000, seed=0):
    """Mean of fn(row) with a 90 % interval, resampling whole events (cells of one event are not independent)."""
    by = {}
    for r in rows:
        v = fn(r)
        if v is not None:
            by.setdefault(r["event"], []).append(v)
    if len(by) < 3:
        return None
    ev = list(by)
    rng = np.random.default_rng(seed)
    allv = [x for e in ev for x in by[e]]
    bs = []
    for _ in range(n):
        pick = rng.choice(len(ev), len(ev))
        vv = [x for i in pick for x in by[ev[i]]]
        bs.append(np.mean(vv))
    return dict(mean=round(float(np.mean(allv)), 4), lo=round(float(np.quantile(bs, 0.05)), 4), hi=round(float(np.quantile(bs, 0.95)), 4), cells=len(allv), events=len(ev))


def aggregate(rows: list[dict], split: str = "all") -> dict:
    rows = [r for r in rows if "error" not in r and (split == "all" or r.get("split") == split)]
    names = sorted({m for r in rows for m in r["models"]})
    tab = {}
    for m in names:
        cells = [r["models"][m] for r in rows if m in r["models"] and "error" not in r["models"][m]]
        t = dict(cells=len(cells), skill=_mean([c.get("skill") for c in cells]), skill_onset=_mean([c.get("skill_onset") for c in cells]),
                 cov80=_mean([c.get("cov80") for c in cells]), cov90=_mean([c.get("cov90") for c in cells]))
        for sfx in ("", "_H", "_O"):
            dirn = [(c[f"direction{sfx}"], c[f"n_moved{sfx}"]) for c in cells if c.get(f"direction{sfx}") is not None]
            t[f"direction_series{sfx}"] = round(sum(a * n for a, n in dirn) / sum(n for _, n in dirn), 3) if dirn else None
            t[f"direction_cells{sfx}"] = _mean([c.get(f"direction{sfx}") for c in cells])
            t[f"timing_days{sfx}"] = _mean([c.get(f"timing_days{sfx}") for c in cells])
            t[f"mag_ratio{sfx}"] = _mean([c.get(f"mag_ratio{sfx}") for c in cells])
            t[f"mag_logerr{sfx}"] = _mean([c.get(f"mag_logerr{sfx}") for c in cells])
            tp = fp = fn = tn = 0
            for r in rows:
                c = r["models"].get(m)
                if not c or "error" in c or r.get("n_judged", 0) == 0:
                    continue
                tr, pr_ = r["moved_true"], c.get(f"moved_pred{sfx}")
                if pr_ is None:
                    continue
                tp += tr and pr_
                fp += (not tr) and pr_
                fn += tr and (not pr_)
                tn += (not tr) and (not pr_)
            t[f"moved_precision{sfx}"] = round(tp / (tp + fp), 3) if tp + fp else None
            t[f"moved_recall{sfx}"] = round(tp / (tp + fn), 3) if tp + fn else None
            t[f"moved_n{sfx}"] = int(tp + fp + fn + tn)
        tab[m] = t
    h2h = {}
    for key in ("skill", "direction_O", "direction_H", "timing_days_H", "mag_logerr_O", "mag_logerr_H"):
        hi = not key.startswith(("timing", "mag_logerr"))
        for a, b in (("K_full", "K_off"), ("K_full", "best_open"), ("K_off", "best_open"), ("K_full", "chronos2"), ("K_full", "c2cov"), ("K_full", "LP")):
            def diff(r, a=a, b=b, key=key, hi=hi):
                va = _get(r, a, key)
                vb = _best(r, key, hi=hi) if b == "best_open" else _get(r, b, key)
                if va is None or vb is None:
                    return None
                return (va - vb) if hi else (vb - va)           # positive = a better
            vals = [diff(r) for r in rows]
            vals = [v for v in vals if v is not None]
            h2h[f"{key}:{a}>{b}"] = dict(cells=len(vals), wins=int(sum(v > 1e-9 for v in vals)), ties=int(sum(abs(v) <= 1e-9 for v in vals)),
                                          gain=_boot(rows, diff))
    return dict(split=split, cells=len(rows), events=len({r["event"] for r in rows}), table=tab, head_to_head=h2h)


def diagnose(rows: list[dict], split: str = "dev") -> list[dict]:
    """Per cell, where the cross-layer links go wrong: K_full against K_off and the best open model. Flags: `sign` (a moved series whose onset or
    long-lead forecast deviation has the wrong sign while the best open model's is right), `missing` (the layer moved, K_full predicts it did not,
    K_off or an open model does), `timing` (K_full's peak error at the long lead is more than 7 days worse than K_off's), `worse` (K_full skill
    below K_off by more than 0.02), `better` (above by more than 0.02)."""
    out = []
    for r in rows:
        if "error" in r or r.get("split") != split:
            continue
        g = lambda m, k: _get(r, m, k)  # noqa: E731
        flags = []
        for sfx in ("_O", "_H"):
            kd, od = g("K_full", f"direction{sfx}"), _best(r, f"direction{sfx}")
            if kd is not None and od is not None and kd + 0.2 < od:
                flags.append("sign" + sfx)
        if r.get("moved_true"):
            kp = g("K_full", "moved_pred_H")
            if kp is False and (g("K_off", "moved_pred_H") or any(_get(r, m, "moved_pred_H") for m in BEST_OPEN)):
                flags.append("missing")
        kt, ot = g("K_full", "timing_days_H"), g("K_off", "timing_days_H")
        if kt is not None and ot is not None and kt - ot > 7:
            flags.append("timing")
        ks, os_ = g("K_full", "skill"), g("K_off", "skill")
        if ks is not None and os_ is not None:
            if ks < os_ - 0.02:
                flags.append("worse")
            elif ks > os_ + 0.02:
                flags.append("better")
        out.append(dict(event=r["event"], family=r["family"], moved=r.get("moved_true"), champion=r.get("champion"), flags=flags,
                        d_skill=round(ks - os_, 3) if ks is not None and os_ is not None else None,
                        dir_O=(g("K_full", "direction_O"), g("K_off", "direction_O"), _best(r, "direction_O")),
                        dir_H=(g("K_full", "direction_H"), g("K_off", "direction_H"), _best(r, "direction_H"))))
    return out


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(rows: list[dict], split: str, order=None) -> str:
    """Markdown tables for the doc: per-baseline table, head to head, per layer, per event."""
    ag = aggregate(rows, split)
    order = order or ["ref", "K_full", "K_off", "K_fix", "c2cov", "chronos2", "tirex2", "toto2", "LP"]
    names = [m for m in order if m in ag["table"]]
    out = [f"**{split}: {ag['cells']} cells, {ag['events']} events.** Metrics are means over cells (direction and timing over the series that really moved; "
           "layers-moved precision / recall over all cells).", "",
           "| Model | MAE skill | skill at onset | direction at onset | direction long lead | timing err long lead (d) | magnitude err long lead | layers moved P / R (long lead) | cov 80 | cov 90 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        t = ag["table"][m]
        out.append(f"| {m} | {_f(t['skill'])} | {_f(t['skill_onset'])} | {_f(t['direction_series_O'])} | {_f(t['direction_series_H'])} | {_f(t['timing_days_H'], 1)} | "
                   f"{_f(t['mag_logerr_H'], 2)} | {_f(t['moved_precision_H'], 2)} / {_f(t['moved_recall_H'], 2)} | {_f(t['cov80'], 2)} | {_f(t['cov90'], 2)} |")
    out += ["", "Head to head (positive = first model better; gain with a 90 % bootstrap interval over events):", "",
            "| Comparison | metric | cells | first wins | ties | mean gain [90 % interval] |", "|---|---|---|---|---|---|"]
    for k, v in ag["head_to_head"].items():
        key, pair = k.split(":")
        g = v["gain"]
        out.append(f"| {pair} | {key} | {v['cells']} | {v['wins']} | {v['ties']} | " + (f"{g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]" if g else "-") + " |")
    # per family
    fams = sorted({r["family"] for r in rows if "error" not in r and (split == "all" or r.get("split") == split)})
    out += ["", "By layer (family): mean MAE skill, and direction at the long lead.", "",
            "| Layer | cells | moved | K_full skill | K_off skill | best open skill | K_full dir | K_off dir | best open dir |", "|---|---|---|---|---|---|---|---|---|"]
    for f in fams:
        rr = [r for r in rows if r["family"] == f and "error" not in r and (split == "all" or r.get("split") == split)]
        mv = sum(bool(r["moved_true"]) for r in rr)
        out.append(f"| {f} | {len(rr)} | {mv} | {_f(_mean([_get(r, 'K_full', 'skill') for r in rr]))} | {_f(_mean([_get(r, 'K_off', 'skill') for r in rr]))} | "
                   f"{_f(_mean([_best(r, 'skill') for r in rr]))} | {_f(_mean([_get(r, 'K_full', 'direction_H') for r in rr]), 2)} | "
                   f"{_f(_mean([_get(r, 'K_off', 'direction_H') for r in rr]), 2)} | {_f(_mean([_best(r, 'direction_H') for r in rr]), 2)} |")
    return "\n".join(out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "run", "aggregate", "links", "ingest", "diagnose", "report"])
    ap.add_argument("--log")
    ap.add_argument("--events")
    ap.add_argument("--split")
    ap.add_argument("--families")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        capture(a.events.split(",") if a.events else None, a.split, a.out or "/tmp/casc_cap")
    elif a.cmd == "run":
        run(a.events.split(",") if a.events else None, a.split, a.families.split(",") if a.families else None, tuple(a.models.split(",")), a.out)
    elif a.cmd == "ingest":                           # CASC lines of a VM log (gsutil cat .../out/log.txt) -> cells_<out>.json
        rows = {}
        for line in Path(a.log).read_text(errors="replace").splitlines():
            if line.startswith("CASC "):
                r = json.loads(line[5:])
                rows[(r.get("event"), r.get("family"))] = r
        CASC.mkdir(parents=True, exist_ok=True)
        (CASC / f"cells_{a.out}.json").write_text(json.dumps(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rows=list(rows.values())), default=str))
        print(len(rows), "cells ->", CASC / f"cells_{a.out}.json")
    elif a.cmd == "diagnose":
        rows = []
        for f in sorted(CASC.glob("cells_*.json")):
            rows += json.loads(f.read_text())["rows"]
        for x in diagnose(rows, a.split or "dev"):
            print(json.dumps(x))
    elif a.cmd == "report":
        rows = []
        for f in sorted(CASC.glob("cells_*.json")):
            if "audit" not in f.name:
                rows += json.loads(f.read_text())["rows"]
        print(report(rows, a.split or "dev"))
    elif a.cmd == "aggregate":
        rows = []
        for f in sorted(CASC.glob("cells_*.json")):
            rows += json.loads(f.read_text())["rows"]
        print(json.dumps(aggregate(rows, a.split or "all"), indent=1))
    elif a.cmd == "links":
        reg = C.registry()
        for f in reg["families"]:
            if f.get("available") and C.champion(f["id"]):
                print(f["id"], "|", C.champion(f["id"]), "->", links_off(C.champion(f["id"]), f["layer"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
