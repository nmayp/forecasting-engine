"""Other forecasting engines on the lab's own windows, data and scoring (docs/methods.md).
Benchmarks are NOT trials: they never become champions and do not enter the ledger; results go to results/lab/bench.json.

    python -m forecasting_engine.lab.bench operator load_h res_h     # grid operators' own day-ahead forecasts (ENTSO-E 6.1.b, 14.1.d)
    python -m forecasting_engine.lab.bench lear price_h              # LEAR (Lago et al. 2021, the open EPF benchmark), re-implemented
    python -m forecasting_engine.lab.bench chronos load_h price_h ... # Chronos-Bolt zero-shot (Amazon, open weights), own history only
    python -m forecasting_engine.lab.bench chronos2 ports_m ...     # Chronos-2 zero-shot: own history; + every layer as covariates; + cross-country
    python -m forecasting_engine.lab.bench tirex2 ports_m ...       # TiRex-2 zero-shot (NX-AI): own history; + covariates; + cross-series
    python -m forecasting_engine.lab.bench toto2 ports_m ...        # Toto 2.0 zero-shot (Datadog): own history; + cross-series
    python -m forecasting_engine.lab.bench fm3 ports_m ...          # both (FE_FM_VARIANTS=tag,tag to pick variants)
    python -m forecasting_engine.lab.bench fm2 ports_m ...          # TimesFM 2.5 (Google) and FlowState (IBM) zero-shot
    python -m forecasting_engine.lab.bench collect run1.log run2.log  # merge TiRex-2 / Toto 2.0 / ... rows printed in run logs into bench.json
    python -m forecasting_engine.lab.bench champion load_h ...       # our champion on exactly the same cells (for the step-restricted ones)

Same everything: the registry's development and holdout origins, the strict reference, the per-series scale, pooled skill
S = 1 - sum|y - f| / sum|y - r| (scaled) over the cells a benchmark forecasts. A benchmark that only covers the next calendar day (operators,
LEAR) is scored on those steps only, and our champion is rescored on the same cells. "as known" variants use only what was published by the
origin; "hindsight" variants (marked) use a value published later (14.1.d is due at 18:00 Brussels, after our 10 UTC origin), as an upper bound.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C

OUT = C.LAB / "bench.json"


def _setup(fid: str):
    d, reg = C.design(fid)
    refs = [reg["reference"][s] for s in d.series]
    dev, hold = C._idx(d, reg["dev_origins"]), C._idx(d, reg["holdout_origins"])
    tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
    return d, reg, refs, dev, hold, tf


def _next_day_mask(d, eo) -> np.ndarray:
    """(origins, H) True where the target lies on the calendar day after the origin's day."""
    tau = pd.to_datetime(d.tau[eo], utc=True).floor("D")
    st = d.stamps.as_unit("ns").asi8[np.clip(d.R[eo], 0, d.T - 1)]
    day = pd.to_datetime(st.ravel(), utc=True).floor("D").to_numpy().reshape(st.shape)
    return day == (tau + pd.Timedelta(days=1)).to_numpy()[:, None]


def skill(d, eo, P, refs, mask=None) -> dict:
    R = C.method("ref")(d, refs, eo, 0)
    Y = np.stack([d.truth(j)[eo] for j in range(d.J)], 1)
    sc = d.scale[None, :, None]
    m = np.isfinite(Y) & np.isfinite(P) & np.isfinite(R)
    if mask is not None:
        m &= mask[:, None, :]
    if not m.any():
        return {}
    num, den = (np.abs(Y - P) / sc)[m].sum(), (np.abs(Y - R) / sc)[m].sum()
    per = []
    for j in range(d.J):
        mj = m[:, j, :]
        if mj.any():
            a, b = (np.abs(Y - P)[:, j, :][mj]).sum(), (np.abs(Y - R)[:, j, :][mj]).sum()
            per.append(1 - a / b if b > 0 else np.nan)
    return dict(pooled_skill=round(float(1 - num / den), 4), median_skill=round(float(np.nanmedian(per)), 4), cells=int(m.sum()))


# ---------------------------------------------------------------------------------------------------------- operators' forecasts
def _known_at(d, name, eo, hindsight=False):
    dr = next(x for x in d.fam.drivers if x.name == name)
    st = d.stamps.as_unit("ns").asi8[np.clip(d.R[eo], 0, d.T - 1)]
    out = np.full((len(eo), d.J, d.H), np.nan)
    for j in range(d.J):
        if hindsight:
            dr2 = C.PN.Driver(dr.name, dr.group, dr.data, -10 ** 6, True, dr.note, None)   # "known" long before: ignores publication time
            out[:, j, :] = C._driver_known(dr2, d.country(j), d.tau[eo], st)
        else:
            out[:, j, :] = C._driver_known(dr, d.country(j), d.tau[eo], st)
    return out


def operator(fid: str) -> dict:
    d, reg, refs, dev, hold, tf = _setup(fid)
    res = {}
    for wname, eo in (("dev", dev), ("holdout", hold)):
        mask = _next_day_mask(d, eo)
        if fid in ("load_h",):
            P = _known_at(d, "load_da", eo)
            res.setdefault("operator_day_ahead_load_6.1.b", {})[wname] = skill(d, eo, P, refs, mask)
        if fid == "res_h":
            # the operators' wind + solar forecast as a share of their load forecast, mapped to our target (share of generation) by a
            # per-country linear fit on data before the development window (as known then)
            for hs in (False, True):
                X = _known_at(d, "res_da_share", eo, hindsight=hs)
                Xt = _known_at(d, "res_da_share", np.arange(max(0, dev[0] - 400), dev[0], 7), hindsight=True)
                Yt = np.stack([d.truth(j)[np.arange(max(0, dev[0] - 400), dev[0], 7)] for j in range(d.J)], 1)
                P = np.full_like(X, np.nan)
                for j in range(d.J):
                    x, y = Xt[:, j, :].ravel(), Yt[:, j, :].ravel()
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() > 100:
                        b, a = np.polyfit(x[ok], y[ok], 1)
                        P[:, j, :] = a + b * X[:, j, :]
                key = "operator_wind_solar_14.1.d" + ("_hindsight" if hs else "_as_known")
                res.setdefault(key, {})[wname] = skill(d, eo, P, refs, mask)
    return res


# ---------------------------------------------------------------------------------------------------------- LEAR (price)
def _lear_country(d, Y, load, resd, variant, eo, j, cc, calib_days):
    from sklearn.linear_model import LassoLarsIC
    P = np.full((len(eo), 1, d.H), np.nan)
    p = Y[cc].groupby([Y.index.floor("D"), Y.index.hour]).mean().unstack().reindex(columns=range(24))
    L = load[cc].groupby([load.index.floor("D"), load.index.hour]).mean().unstack().reindex(columns=range(24)) if cc in load.columns else None
    Rz = None
    if variant.endswith("hindsight") and resd is not None and cc in resd.data.columns:
        r = resd.data[cc]
        Rz = r.groupby([r.index.floor("D"), r.index.hour]).mean().unstack().reindex(columns=range(24))
    days = pd.date_range(p.index.min(), p.index.max(), freq="D")
    p = p.reindex(days)
    blocks = [p.shift(k).to_numpy() for k in (1, 2, 3, 7)]          # row t: prices of t-1, t-2, t-3, t-7
    if L is not None:
        L = L.reindex(days)
        blocks += [L.shift(k).to_numpy() for k in (0, 1, 7)]
    if Rz is not None:
        blocks += [Rz.reindex(days).to_numpy()]
    dow = np.eye(7)[days.dayofweek]
    F = np.hstack(blocks + [dow])                                      # (days, features), as known for the target day t
    Yd = p.to_numpy()
    pos = {t: n for n, t in enumerate(days)}
    for i, o in enumerate(eo):
        D = pd.Timestamp(d.tau[o], tz="UTC").floor("D") + pd.Timedelta(days=1)          # target day
        n = pos.get(D)
        if n is None:
            continue
        lo = max(0, n - 1 - calib_days)
        X, Yt, xq = F[lo:n - 1], Yd[lo:n - 1], F[n:n + 1]               # training days end at D-2 (D-1 not complete at the origin)
        ok = np.isfinite(X).all(1) & np.isfinite(Yt).all(1)
        if ok.sum() < 100 or not np.isfinite(xq).all():
            continue
        X, Yt = X[ok], Yt[ok]
        med, mad = np.median(X, 0), np.median(np.abs(X - np.median(X, 0)), 0) + 1e-6
        Xs, xs = np.arcsinh((X - med) / mad), np.arcsinh((xq - med) / mad)
        st = d.stamps[np.clip(d.R[o], 0, d.T - 1)]
        for h in range(24):
            ym, ys = np.median(Yt[:, h]), np.median(np.abs(Yt[:, h] - np.median(Yt[:, h]))) + 1e-6
            m = LassoLarsIC(criterion="aic").fit(Xs, np.arcsinh((Yt[:, h] - ym) / ys))
            yh = ym + ys * np.sinh(m.predict(xs)[0])
            k = np.where((st.floor("D") == D) & (st.hour == h))[0]
            if len(k):
                P[i, 0, k[0]] = yh
    return j, P[:, 0, :]


def lear(fid: str = "price_h", calib_days: int = 728) -> dict:
    """LEAR (Uniejewski et al. 2016; Lago et al. 2021): per country and target hour, LASSO (LassoLarsIC, AIC) on asinh-standardised prices
    of D-1, D-2, D-3, D-7 (24 hours each), the day-ahead load forecast of the target day, D-1 and D-7, and day-of-week dummies; recalibrated
    at every origin on the last `calib_days` days known then. Here 'D' = the target day (the day after the origin); its D-1 prices are the
    origin day's, published at noon the day before. Variant '+res' adds the operators' wind + solar forecast of the target day (hindsight)."""
    from sklearn.linear_model import LassoLarsIC
    d, reg, refs, dev, hold, tf = _setup(fid)
    Y = d.fam.Y.copy()
    Y.columns = [d.country(j) for j in range(d.J)]
    load = next(x for x in d.fam.drivers if x.name == "load_da").data
    resd = next((x for x in d.fam.drivers if x.name == "res_da_share"), None)
    res = {}
    for variant in ("lear", "lear+res_hindsight"):
        for wname, eo in (("dev", dev), ("holdout", hold)):
            P = np.full((len(eo), d.J, d.H), np.nan)
            mask = _next_day_mask(d, eo)
            from joblib import Parallel, delayed
            for j, Pj in Parallel(n_jobs=-1)(delayed(_lear_country)(d, Y, load, resd, variant, eo, j, cc, calib_days)
                                             for j, cc in enumerate(Y.columns)):
                P[:, j, :] = Pj
            res.setdefault(variant, {})[wname] = skill(d, eo, P, refs, mask)
    return res


# ---------------------------------------------------------------------------------------------------------- Chronos (zero-shot)
_PIPE: dict = {}
FM_DEVICE = __import__("os").environ.get("FE_FM_DEVICE", "cpu")      # "cuda" on a GPU machine


def model_src(repo: str) -> str:
    """A local copy of a Hugging Face repo if FE_MODEL_DIR has one (one folder per repo with '/' as '__'), else the repo id (downloaded from Hugging Face)."""
    import os
    from pathlib import Path
    base = os.environ.get("FE_MODEL_DIR")
    if base:
        p = Path(base) / repo.replace("/", "__")
        if p.is_dir() and any(p.glob("*.json")) and any(x.suffix in (".safetensors", ".ckpt", ".bin") for x in p.iterdir()):
            return str(p)                                    # a complete copy (config + weights); a half-written mirror falls back
    return repo


def chronos_P(d, eo, model: str = "amazon/chronos-bolt-small", ctx: int = 512) -> np.ndarray:
    """Chronos-Bolt median forecasts (origins x J x H) from each series' own history as known at the origin (zero-shot)."""
    import torch
    from chronos import BaseChronosPipeline
    if model not in _PIPE:
        _PIPE[model] = BaseChronosPipeline.from_pretrained(model_src(model), device_map=FM_DEVICE, torch_dtype=torch.float32)
    pipe = _PIPE[model]
    if True:
        P = np.full((len(eo), d.J, d.H), np.nan)
        for j in range(d.J):
            batch, idx = [], []
            for i, o in enumerate(eo):
                L = d.last_known(j)[o]
                if L < 32:
                    continue
                x = d.Y[max(0, L - ctx + 1): L + 1, j]
                x = pd.Series(x).ffill().bfill().to_numpy()
                if not np.isfinite(x).all():
                    continue
                batch.append(torch.tensor(x, dtype=torch.float32))
                idx.append((i, o, L))
            for b in range(0, len(batch), 64):
                steps = int(max(d.R[o, -1] - L for _, o, L in idx[b:b + 64]))
                q, mean = pipe.predict_quantiles(batch[b:b + 64], prediction_length=min(steps, 1024), quantile_levels=[0.5])
                med = q[:, :, 0].cpu().numpy()
                for r, (i, o, L) in enumerate(idx[b:b + 64]):
                    k = d.R[o] - L - 1                                   # forecast index of each target row
                    okk = (k >= 0) & (k < med.shape[1])
                    P[i, j, okk] = med[r, k[okk]]
    return P


C2_MAX_BATCH = 1024               # variates per Chronos-2 batch: a weekly family (60 series x 41 variates) overflowed a 24 GB L4
MAX_COV = 40                      # Chronos-2 covariates per series (group attention cost grows with the variates)


def _parents(d) -> set:
    try:
        from forecasting_engine.lab import causal as CA
        return set(CA.parents(d.fam.id) or ())
    except Exception:  # noqa: BLE001
        return set()


def _cov_drivers(d, groups=None, select: bool = False) -> list:
    """The family's drivers, the same ones the cross-layer trees see; with more than MAX_COV, the PCMCI causal parents first.
    select=True: only the PCMCI causal parents and the drivers known ahead (day-ahead forecasts), so Chronos-2, which cannot pick its
    inputs as the trees do, is not swamped by irrelevant layers (its best fair use, not a handicap)."""
    drs = [x for x in d.fam.drivers if groups is None or x.group in groups]
    par = _parents(d)
    if select:
        return [x for x in drs if x.name in par or x.ahead][:MAX_COV]
    if len(drs) <= MAX_COV:
        return drs
    return (sorted(drs, key=lambda x: x.name not in par))[:MAX_COV]


_FT: dict = {}
FT_STEPS = int(__import__("os").environ.get("FE_C2_FT_STEPS", "600"))


def chronos2_finetuned(d, cutoff_ns: int, model: str = "amazon/chronos-2"):
    """Chronos-2 fine-tuned (LoRA) on the family's own series as known before cutoff_ns: each series' values whose known time is before the
    cutoff (nothing later). One fine-tune per (family, cutoff); the window's forecasts then use it zero-shot as usual."""
    key = (d.fam.id, int(cutoff_ns))
    if key in _FT:
        return _FT[key]
    import torch
    from chronos import BaseChronosPipeline
    base = BaseChronosPipeline.from_pretrained(model_src(model), device_map=FM_DEVICE, torch_dtype=torch.float32)
    known = d.kt < cutoff_ns                                        # rows whose value was published before the cutoff
    series = []
    for j in range(d.J):
        y = pd.Series(np.where(known, d.Y[:, j], np.nan)).iloc[: int(np.where(known)[0].max()) + 1 if known.any() else 0]
        y = y.ffill().dropna().to_numpy(np.float32)
        if len(y) >= 3 * d.H + 32:
            series.append(y)
    if not series:
        _FT[key] = base
        return base
    import tempfile
    ft = base.fit(series, prediction_length=d.H, finetune_mode="lora", num_steps=FT_STEPS, learning_rate=1e-4,
                  batch_size=min(64, max(8, len(series))), output_dir=tempfile.mkdtemp(prefix="c2ft_"), remove_printer_callback=True)
    _FT[key] = ft
    return ft


def chronos2ft_P(d, eo, covariates: bool = False, cross: bool = False) -> np.ndarray:
    """Chronos-2 fine-tuned on what was known before the first origin of eo (one fine-tune per evaluation window)."""
    ft = chronos2_finetuned(d, int(d.tau[eo[0]]))
    return chronos2_P(d, eo, covariates, cross, pipe=ft)


def chronos2_P(d, eo, covariates: bool = True, cross: bool = True, model: str = "amazon/chronos-2", ctx: int = 512, select: bool = False,
               pipe=None) -> np.ndarray:
    """Chronos-2 (Amazon, 2025; open weights) median forecasts (origins x J x H), zero-shot. The fair version (covariates=True): each series
    with its own history, every driver of the family as known at the origin (past values only where published by then; ahead drivers, e.g.
    day-ahead forecasts, also at the target times where published by then) and the calendar at both; cross=True lets the model share
    context across the family's countries at the same origin (cross_learning). Nothing published after the origin enters."""
    import torch
    from chronos import BaseChronosPipeline
    if pipe is None:
        if model not in _PIPE:
            _PIPE[model] = BaseChronosPipeline.from_pretrained(model_src(model), device_map=FM_DEVICE, torch_dtype=torch.float32)
        pipe = _PIPE[model]
    drs = _cov_drivers(d, select=select) if covariates else []
    hourly = d.fam.freq == "h"
    st_ns = d.stamps.as_unit("ns").asi8
    P = np.full((len(eo), d.J, d.H), np.nan)
    for i, o in enumerate(eo):
        tau = np.array([d.tau[o]])
        items, idx = [], []
        for j in range(d.J):
            L = d.last_known(j)[o]
            if L < 32:
                continue
            a = max(0, L - ctx + 1)
            y = pd.Series(d.Y[a: L + 1, j]).ffill().bfill().to_numpy()
            if not np.isfinite(y).all():
                continue
            steps = int(d.R[o, -1] - L)
            if steps < 1:
                continue
            hist = st_ns[a: L + 1]
            fut = st_ns[np.clip(np.arange(L + 1, L + 1 + steps), 0, d.T - 1)]
            item = dict(target=y.astype(np.float32))
            if covariates:
                cc = d.country(j)
                past, future = {}, {}
                cal_h, cal_f = C._cal(pd.to_datetime(hist, utc=True), cc), C._cal(pd.to_datetime(fut, utc=True), cc)
                for cn in (("hour", "weekday", "holiday", "month") if hourly else ("weekday", "holiday", "month")):
                    past[f"cal_{cn}"] = cal_h[cn].to_numpy(np.float32)
                    future[f"cal_{cn}"] = cal_f[cn].to_numpy(np.float32)
                for dr in drs:
                    col = C.driver_col(dr, d.series[j], d.country(j))
                    v = C._driver_known(dr, col, tau, hist[None, :])[0]          # its value at each history stamp, if published by the origin
                    v = pd.Series(v).ffill().fillna(0.0).to_numpy(np.float32)
                    past[dr.name] = v
                    if dr.ahead:
                        vf = C._driver_known(dr, col, tau, fut[None, :])[0]       # at the target times, only where published by the origin
                        if np.isfinite(vf).any():
                            future[dr.name] = pd.Series(vf).ffill().fillna(float(v[-1])).to_numpy(np.float32)
                        else:
                            future[dr.name] = np.full(steps, v[-1], np.float32)
                item["past_covariates"], item["future_covariates"] = past, future
            items.append(item)
            idx.append((j, L, steps))
        if not items:
            continue
        med = float(np.median([s for _, _, s in idx]))       # a series whose data stopped long before the origin (e.g. aq_d countries without
        keep = [k for k, (_, _, s) in enumerate(idx) if s <= 4 * med + d.H]   # E2a after 2025) would make the whole call forecast >1000 steps
        items, idx = [items[k] for k in keep], [idx[k] for k in keep]          # (slow, GPU OOM): left NaN, like a series with too little history
        steps = max(s for _, _, s in idx)
        for it, (_, _, s) in zip(items, idx):                 # one schema per call: future covariates padded to the longest horizon
            for k, v in (it.get("future_covariates") or {}).items():
                if len(v) < steps:
                    it["future_covariates"][k] = np.concatenate([v, np.full(steps - len(v), v[-1], np.float32)])
        # batch_size counts every variate (targets and covariates): all of this origin's series in one batch, so cross_learning shares across them
        nv = sum(1 + len(it.get("past_covariates") or {}) for it in items)
        q, _ = pipe.predict_quantiles(items, prediction_length=steps, quantile_levels=[0.5], cross_learning=cross, batch_size=max(min(nv, C2_MAX_BATCH), 1))
        for r, (j, L, s) in enumerate(idx):
            med = q[r][0, :, 0].cpu().numpy()
            k = d.R[o] - L - 1
            okk = (k >= 0) & (k < len(med))
            P[i, j, okk] = med[k[okk]]
    return P


def chronos2(fid: str) -> dict:
    """Chronos-2 three ways on the same cells: own history only, with every layer as covariates, and with covariates + cross-country context."""
    d, reg, refs, dev, hold, tf = _setup(fid)
    out = {}
    if __import__("os").environ.get("FE_C2_FT") == "1":     # Chronos-2 fine-tuned on the family (known before each window), own history
        out["chronos-2_finetuned"] = {w: skill(d, eo, chronos2ft_P(d, eo), refs) for w, eo in (("dev", dev), ("holdout", hold))}
        print(json.dumps({fid: {"chronos-2_finetuned": out["chronos-2_finetuned"]}}), flush=True)
    variants = (("chronos-2_univariate", False, False, False), ("chronos-2_covariates", True, False, False),
                ("chronos-2_covariates_cross", True, True, False), ("chronos-2_selected_cross", True, True, True))
    only = __import__("os").environ.get("FE_C2_VARIANTS")             # e.g. "chronos-2_covariates_cross,chronos-2_selected_cross"
    for tag, cov, cross, sel in variants:
        if only and tag not in only.split(","):
            continue
        out[tag] = {w: skill(d, eo, chronos2_P(d, eo, cov, cross, select=sel), refs) for w, eo in (("dev", dev), ("holdout", hold))}
        print(json.dumps({fid: {tag: out[tag]}}), flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------- TiRex-2 and Toto 2.0 (zero-shot)
# The bar: a layer is won only if the engine's champion beats the BEST of Chronos-2, TiRex-2 and Toto 2.0, all three
# commercial-use licences (Apache 2.0;). Same cells, same 512-step context, same drivers as Chronos-2 where the model
# takes covariates (TiRex-2 does, past and known-future; Toto 2.0 does not yet), every value as known at the origin.
#   TiRex-2   NX-AI, pip tirex-2==0.3.1, HF NX-AI/TiRex-2 (xLSTM, 38M + 44M multivariate parameters)
#   Toto 2.0  Datadog, pip toto-2==2.0.0, HF Datadog/Toto-2.0-1B (1B: fits an L4 with batch headroom; FE_TOTO_MODEL to change)
_ENV = __import__("os").environ
TIREX_MODEL = _ENV.get("FE_TIREX_MODEL", "NX-AI/TiRex-2")
TOTO_MODEL = _ENV.get("FE_TOTO_MODEL", "Datadog/Toto-2.0-1B")
FM_MAX_VAR = 1024                 # variates per TiRex-2 cross item (targets + covariates), as C2_MAX_BATCH
FM_ORIGINS_PER_CALL = int(_ENV.get("FE_FM_CHUNK", "16"))
FM_VAR_BATCH = int(_ENV.get("FE_FM_VAR_BATCH", "1024"))   # variates per TiRex-2 GPU batch (64 items x 45 variates overflowed an L4)
_DRK: dict = {}
_CALK: dict = {}


def _drv_arrays(d, dr, col):
    """(stamps ns, known times, values) of one driver column, computed once (what C._driver_known recomputes at every call)."""
    key = (d.fam.id, dr.name, col)
    if key not in _DRK:
        if col not in dr.data.columns or not dr.data[col].notna().any():
            _DRK[key] = None
        else:
            s = dr.data[col].dropna()
            st = pd.DatetimeIndex(s.index)
            _DRK[key] = (st.as_unit("ns").asi8, C.known_times(st, dr.avail_h, dr.dayahead_hour), s.to_numpy(float))
    return _DRK[key]


def _drv_at(d, dr, col, tau: float, at_ns: np.ndarray) -> np.ndarray:
    """Exactly C._driver_known(dr, col, [tau], at_ns[None])[0]: the driver's value AT each stamp, where published by the origin tau."""
    arr = _drv_arrays(d, dr, col)
    if arr is None:
        return np.full(len(at_ns), np.nan)
    sti, kt, v = arr
    i = np.clip(np.searchsorted(sti, at_ns, "left"), 0, len(sti) - 1)
    return np.where((sti[i] == at_ns) & (kt[i] <= tau), v[i], np.nan)


def _cal_full(d, cc: str) -> dict:
    """The calendar covariates of Chronos-2 (C._cal) over every row of the family, once per country; sliced per origin."""
    key = (d.fam.id, d.T, cc)
    if key not in _CALK:
        c = C._cal(pd.to_datetime(d.stamps.as_unit("ns").asi8, utc=True), cc)
        names = ("hour", "weekday", "holiday", "month") if d.fam.freq == "h" else ("weekday", "holiday", "month")
        _CALK[key] = {f"cal_{n}": c[n].to_numpy(np.float32) for n in names}
    return _CALK[key]


def _fm_inputs(d, o, j, drs, ctx: int, steps: int | None = None):
    """One series at one origin with the Chronos-2 covariate semantics (chronos2_P): history y (ffill/bfill), every driver's value at each
    history stamp as published by the origin (ffill, 0 before the first), ahead drivers also at the target stamps where published by then
    (else carried forward), calendar at both. -> (L, a, steps, y, past {name: (T,)}, ahead {name: (T + steps,)}) or None.
    `ahead` holds the calendar and the ahead drivers over history + horizon (TiRex-2's known-future covariates)."""
    L = d.last_known(j)[o]
    if L < 32:
        return None
    a = max(0, L - ctx + 1)
    y = pd.Series(d.Y[a: L + 1, j]).ffill().bfill().to_numpy()
    if not np.isfinite(y).all():
        return None
    steps = int(d.R[o, -1] - L) if steps is None else steps
    if steps < 1:
        return None
    past, ahead = {}, {}
    if drs is not None:
        st_ns = d.stamps.as_unit("ns").asi8
        hi, fi = np.arange(a, L + 1), np.clip(np.arange(L + 1, L + 1 + steps), 0, d.T - 1)
        tau, cc = float(d.tau[o]), d.country(j)
        for n, v in _cal_full(d, cc).items():
            ahead[n] = np.concatenate([v[hi], v[fi]])
        for dr in drs:
            col = C.driver_col(dr, d.series[j], cc)
            v = pd.Series(_drv_at(d, dr, col, tau, st_ns[hi])).ffill().fillna(0.0).to_numpy(np.float32)
            if dr.ahead:
                vf = _drv_at(d, dr, col, tau, st_ns[fi])
                vf = pd.Series(vf).ffill().fillna(float(v[-1])).to_numpy(np.float32) if np.isfinite(vf).any() else np.full(steps, v[-1], np.float32)
                ahead[dr.name] = np.concatenate([v, vf])
            else:
                past[dr.name] = v
    return L, a, steps, y.astype(np.float32), past, ahead


def _place(P, i, j, med, o, L, d):
    k = d.R[o] - L - 1
    ok = (k >= 0) & (k < len(med))
    P[i, j, ok] = med[k[ok]]


def _groups_by_L(d, o, J):
    """Series of one origin grouped by their last known row: cross-series context is shared among series with the same information cutoff
    (a series published later would otherwise have targets inside the others' context)."""
    g: dict = {}
    for j in range(J):
        g.setdefault(int(d.last_known(j)[o]), []).append(j)
    return list(g.values())


_TIREX: dict = {}


def _tirex():
    if "m" not in _TIREX:
        from tirex2 import load_model
        try:
            _TIREX["m"] = load_model(model_src(TIREX_MODEL), device=FM_DEVICE)
            _TIREX["dev"] = FM_DEVICE
        except Exception as e:  # noqa: BLE001  (FlashRNN needs nvcc matching torch's CUDA major; CPU is the same model, slower)
            print(f"tirex-2 on {FM_DEVICE} failed ({e!r:.200}); using cpu", flush=True)
            _TIREX["m"], _TIREX["dev"] = load_model(model_src(TIREX_MODEL), device="cpu"), "cpu"
    return _TIREX["m"]


def tirex2_P(d, eo, covariates: bool = False, cross: bool = False, select: bool = False, ctx: int = 512) -> np.ndarray:
    """TiRex-2 median forecasts (origins x J x H), zero-shot. covariates: the drivers of Chronos-2's covariate variant (_cov_drivers) as past
    covariates, and the calendar and ahead drivers (day-ahead forecasts) as known-future covariates, all as known at the origin. cross: the
    family's series with the same last known row forecast jointly as one multivariate item (TiRex-2's variate mixing), with their covariates
    (identical rows, e.g. family-wide drivers and shared calendar columns, kept once; at most FM_MAX_VAR variates)."""
    import torch
    from tirex2 import TimeseriesType
    m = _tirex()
    drs = _cov_drivers(d, select=select) if covariates else None
    P = np.full((len(eo), d.J, d.H), np.nan)
    t = lambda x: torch.tensor(np.stack(x), dtype=torch.float32)  # noqa: E731
    for c0 in range(0, len(eo), FM_ORIGINS_PER_CALL):
        items, meta = [], []
        for i in range(c0, min(c0 + FM_ORIGINS_PER_CALL, len(eo))):
            o = eo[i]
            groups = _groups_by_L(d, o, d.J) if cross else [[j] for j in range(d.J)]
            for g in groups:
                ins = [(j, _fm_inputs(d, o, j, drs, ctx)) for j in g]
                ins = [(j, x) for j, x in ins if x is not None]
                if not ins:
                    continue
                Ts = min(len(x[3]) for _, x in ins)           # same L in a group; a differs only near the start of a series
                steps = max(x[2] for _, x in ins)
                tg, pc, fc, seen = [], [], [], set()
                for j, (L, a, s, y, past, ahead) in ins:
                    tg.append(y[-Ts:])
                    for v in past.values():
                        h = hash(v[-Ts:].tobytes())
                        if h not in seen:
                            seen.add(h); pc.append(v[-Ts:])
                    for v in ahead.values():
                        v = v[-(Ts + s):] if s == steps else np.concatenate([v[-(Ts + s):], np.full(steps - s, v[-1], np.float32)])
                        h = hash(v.tobytes())
                        if h not in seen:
                            seen.add(h); fc.append(v)
                room = max(FM_MAX_VAR - len(tg), 0)
                fc = fc[:room]
                pc = pc[: max(room - len(fc), 0)]
                items.append(TimeseriesType(target=t(tg), past_covariates=t(pc) if pc else None, future_covariates=t(fc) if fc else None))
                meta.append((i, o, [j for j, _ in ins], ins[0][1][0], steps))
        if not items:
            continue
        H = max(x[4] for x in meta)
        for it, x in zip(items, meta):                        # one prediction length per call: future covariates padded to the longest
            if it.future_covariates is not None and x[4] < H:
                fcv = it.future_covariates
                it.future_covariates = torch.cat([fcv, fcv[:, -1:].repeat(1, H - x[4])], 1)
        nv = max(it.target.shape[0] + it.n_past_covariates + it.n_future_covariates for it in items)
        fc_out = m.forecast(items, prediction_length=H, output_type="numpy", batch_size=max(1, FM_VAR_BATCH // nv))
        for f, (i, o, js, L, _) in zip(fc_out, meta):
            for r, j in enumerate(js):
                _place(P, i, j, f[r, 4, :], o, L, d)          # quantile 0.5 of (0.1 .. 0.9)
    return P


_TOTO: dict = {}


def _toto():
    if "m" not in _TOTO:
        import torch
        from toto2 import Toto2Model
        dev = FM_DEVICE if (FM_DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
        _TOTO["m"] = Toto2Model.from_pretrained(model_src(TOTO_MODEL)).to(dev).eval()
        _TOTO["dev"] = dev
    return _TOTO["m"]


def toto2_P(d, eo, cross: bool = False, ctx: int = 512, batch: int = 256) -> np.ndarray:
    """Toto 2.0 median forecasts (origins x J x H), zero-shot, own history (Toto 2.0 has no covariate input yet). cross: the series with
    the same last known row as the variates of one item (alternating time / variate attention). Shorter histories are left-padded and
    masked."""
    import torch
    m = _toto()
    dev = _TOTO["dev"]
    med_i = list(m.output_head.knots).index(0.5)
    P = np.full((len(eo), d.J, d.H), np.nan)
    work: dict = {}                                            # n variates -> [(i, o, js, L, steps, X (n, ctx), M)]
    for i, o in enumerate(eo):
        groups = _groups_by_L(d, o, d.J) if cross else [[j] for j in range(d.J)]
        for g in groups:
            ins = [(j, _fm_inputs(d, o, j, None, ctx)) for j in g]
            ins = [(j, x) for j, x in ins if x is not None]
            if not ins:
                continue
            X, M = np.zeros((len(ins), ctx), np.float32), np.zeros((len(ins), ctx), bool)
            for r, (j, x) in enumerate(ins):
                y = x[3]
                X[r, ctx - len(y):], M[r, ctx - len(y):] = y, True
            work.setdefault(len(ins), []).append((i, o, [j for j, _ in ins], ins[0][1][0], max(x[2] for _, x in ins), X, M))
    for n, ws in work.items():
        bs = max(1, batch // n)
        for b in range(0, len(ws), bs):
            chunk = ws[b:b + bs]
            H = max(w[4] for w in chunk)
            X = torch.tensor(np.stack([w[5] for w in chunk]), device=dev)
            Mk = torch.tensor(np.stack([w[6] for w in chunk]), device=dev)
            with torch.no_grad():
                q = m.forecast({"target": X, "target_mask": Mk, "series_ids": torch.zeros(X.shape[:2], dtype=torch.long, device=dev)},
                               horizon=H, decode_block_size=None, has_missing_values=bool((~Mk).any()))
            med = q[med_i].float().cpu().numpy()                      # (batch, n, H)
            for r, (i, o, js, L, *_ ) in enumerate(chunk):
                for v, j in enumerate(js):
                    _place(P, i, j, med[r, v], o, L, d)
    return P


# ---------------------------------------------------------------------------------------------------------- TimesFM 2.5 and FlowState
#   TimesFM 2.5  Google, pip timesfm==2.0.2 (the 2.5 code; never the 3.0 weights, non-commercial), HF google/timesfm-2.5-200m-pytorch, Apache 2.0
#   FlowState    IBM, pip granite-tsfm==0.3.10 (installed --no-deps: it pins torch < 2.12), HF ibm-granite/granite-timeseries-flowstate-r1
#                revision r1.1, Apache 2.0 (state-space encoder + functional decoder, 18.5M parameters)
TIMESFM_MODEL = _ENV.get("FE_TIMESFM_MODEL", "google/timesfm-2.5-200m-pytorch")
FLOWSTATE_MODEL = _ENV.get("FE_FLOWSTATE_MODEL", "ibm-granite/granite-timeseries-flowstate-r1")
FLOWSTATE_REV = _ENV.get("FE_FLOWSTATE_REV", "r1.1")
TIMESFM_XREG_MODE = _ENV.get("FE_TIMESFM_XREG", "xreg + timesfm")   # better than "timesfm + xreg" on a 4-origin check (hicp_m)
TIMESFM_RIDGE = float(_ENV.get("FE_TIMESFM_RIDGE", "1.0"))   # ridge 0 (the default) over-fits the calendar dummies on short monthly histories
FLOWSTATE_SCALE = {"h": 1.0, "D": 3.43, "d": 3.43, "W": 0.46, "M": 2.0}   # the model card's scale factor per sampling rate (daily: weekly cycle)
_TFM: dict = {}


def _timesfm(horizon: int, xreg: bool):
    """TimesFM 2.5 compiled for this context (512, as every model here) and a horizon covering the family's steps (multiple of 128)."""
    import torch
    import timesfm
    hz = int(np.ceil(max(horizon, 1) / 128) * 128)
    key = (hz, xreg)
    if _TFM.get("key") != key:
        if "m" not in _TFM:
            _TFM["m"] = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_src(TIMESFM_MODEL), torch_compile=False)
            if FM_DEVICE == "cuda" and torch.cuda.is_available():
                _TFM["m"].model.to("cuda")
        _TFM["m"].compile(timesfm.ForecastConfig(max_context=512, max_horizon=hz, normalize_inputs=True, use_continuous_quantile_head=True,
                                                 force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
                                                 per_core_batch_size=256, return_backcast=xreg))
        _TFM["key"] = key
    return _TFM["m"]


def timesfm25_P(d, eo, cross: bool = False, ctx: int = 512) -> np.ndarray:
    """TimesFM 2.5 (200M) point (median) forecasts (origins x J x H), zero-shot from each series' own history as known at the origin.
    TimesFM has no cross-series attention; cross=True adds the known-future covariates (calendar and ahead drivers such as day-ahead
    forecasts, as published by the origin: the only kind its XReg takes) with one linear model shared by the origin's series
    ("xreg + timesfm": a ridge linear model on the covariates, TimesFM on its residuals)."""
    P = np.full((len(eo), d.J, d.H), np.nan)
    drs = [x for x in _cov_drivers(d) if x.ahead] if cross else None
    per = []
    for i, o in enumerate(eo):
        for j in range(d.J):
            x = _fm_inputs(d, o, j, drs if cross else None, ctx)
            if x is not None and x[2] <= 1000:     # the continuous quantile head stops at 1024 steps: stale series (aq_d) stay NaN as for other members
                per.append((i, o, j, x))
    if not per:
        return P
    m = _timesfm(max(x[2] for *_, x in per), cross)
    if not cross:
        H = max(x[2] for *_, x in per)
        pt, _ = m.forecast(horizon=H, inputs=[x[3].astype(np.float64) for *_, x in per])
        for (i, o, j, x), f in zip(per, pt):
            _place(P, i, j, f, o, x[0], d)
        return P
    by_o: dict = {}
    for r in per:
        by_o.setdefault(r[0], []).append(r)
    for i, rows in by_o.items():
        H = max(x[2] for *_, x in rows)
        cov: dict = {}
        for *_, x in rows:
            for k, v in x[5].items():                         # ahead: calendar + ahead drivers over history + horizon
                v = v if x[2] == H else np.concatenate([v, np.full(H - x[2], v[-1], np.float32)])
                cov.setdefault(k, []).append(v.astype(np.float64))
        keys = [k for k, v in cov.items() if len(v) == len(rows)]
        out, _ = m.forecast_with_covariates(inputs=[x[3].astype(np.float64) for *_, x in rows],
                                            dynamic_numerical_covariates={k: cov[k] for k in keys}, xreg_mode=TIMESFM_XREG_MODE,
                                            ridge=TIMESFM_RIDGE, force_on_cpu=True)
        for (i_, o, j, x), f in zip(rows, out):
            _place(P, i_, j, np.asarray(f), o, x[0], d)
    return P


_FLOW: dict = {}


def flowstate_P(d, eo, ctx: int = 512, batch: int = 512) -> np.ndarray:
    """IBM FlowState (r1.1) median forecasts (origins x J x H), zero-shot from each series' own history as known at the origin (FlowState is
    univariate, no covariates), with the model card's scale factor for the family's sampling rate."""
    import torch
    if "m" not in _FLOW:
        from tsfm_public import FlowStateForPrediction
        src = model_src(FLOWSTATE_MODEL)
        kw = {} if src != FLOWSTATE_MODEL else {"revision": FLOWSTATE_REV}
        dev = FM_DEVICE if (FM_DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
        _FLOW["m"], _FLOW["dev"] = FlowStateForPrediction.from_pretrained(src, **kw).to(dev).eval(), dev
    m, dev = _FLOW["m"], _FLOW["dev"]
    sf = FLOWSTATE_SCALE.get(str(d.fam.freq)[:1], 1.0)
    P = np.full((len(eo), d.J, d.H), np.nan)
    work: dict = {}                                            # context length -> [(i, o, j, L, steps, y)]
    for i, o in enumerate(eo):
        for j in range(d.J):
            x = _fm_inputs(d, o, j, None, ctx)
            if x is not None:
                work.setdefault(len(x[3]), []).append((i, o, j, x[0], x[2], x[3]))
    for n, ws in work.items():
        for b in range(0, len(ws), batch):
            chunk = ws[b:b + batch]
            H = max(w[4] for w in chunk)
            X = torch.tensor(np.stack([w[5] for w in chunk], 1)[:, :, None], dtype=torch.float32, device=dev)   # (context, batch, 1)
            Hq = max(H, int(m.config.decoder_patch_len / sf + 1e-6))   # granite-tsfm 0.3.10 returns fewer steps for some lengths (5 -> 3 at
            with torch.no_grad():                                         # scale 2): ask for at least one decoder patch, then a step more
                for _ in range(8):
                    q = m(X, scale_factor=sf, prediction_length=Hq, batch_first=False).quantile_outputs   # (batch, 9, steps, 1)
                    if q.shape[2] >= H:
                        break
                    Hq += 1
            med = q[:, 4, :H, 0].float().cpu().numpy()
            for r, (i, o, j, L, *_ ) in enumerate(chunk):
                _place(P, i, j, med[r], o, L, d)
    return P


FM_VARIANTS = {   # bench tag -> (lab member name, forecast function); members mirror core.FM's chronos2 / chronos2x / chronos2s
    "tirex-2_univariate": ("tirex2", lambda d, eo: tirex2_P(d, eo)),
    "tirex-2_covariates": ("tirex2c", lambda d, eo: tirex2_P(d, eo, covariates=True)),
    "tirex-2_cross": ("tirex2m", lambda d, eo: tirex2_P(d, eo, cross=True)),
    "tirex-2_covariates_cross": ("tirex2x", lambda d, eo: tirex2_P(d, eo, covariates=True, cross=True)),
    "tirex-2_selected_cross": ("tirex2s", lambda d, eo: tirex2_P(d, eo, covariates=True, cross=True, select=True)),
    "toto-2_univariate": ("toto2", lambda d, eo: toto2_P(d, eo)),
    "toto-2_cross": ("toto2x", lambda d, eo: toto2_P(d, eo, cross=True)),
    "timesfm-2.5_univariate": ("timesfm25", lambda d, eo: timesfm25_P(d, eo)),
    "timesfm-2.5_covariates_cross": ("timesfm25x", lambda d, eo: timesfm25_P(d, eo, cross=True)),
    "flowstate_univariate": ("flowstate", lambda d, eo: flowstate_P(d, eo)),
}
_FM2_MEMO: dict = {}


def _memo(tag):
    def f(d, eo):
        import hashlib
        key = (tag, d.fam.id, d.T, hashlib.sha1(np.asarray(eo, np.int64).tobytes()).hexdigest())
        if key not in _FM2_MEMO:
            _FM2_MEMO[key] = FM_VARIANTS[tag][1](d, eo)
        return _FM2_MEMO[key]
    return f


FM_EXTRA = {name: _memo(tag) for tag, (name, _) in FM_VARIANTS.items()}     # core.FM-style: fn(d, eo) -> (origins, J, H)


def register(core=None) -> None:
    """Make tirex2* / toto2* lab members (core.FM and core.METHODS entries, and "fm:<name>" anchors) without editing core.py."""
    core = core or C
    core.FM.update(FM_EXTRA)
    core.METHODS.update({k: (lambda k: lambda d, refs, eo, tf: core.FM[k](d, eo))(k) for k in FM_EXTRA})


def _fm_bench(fid: str, prefix: str) -> dict:
    d, reg, refs, dev, hold, tf = _setup(fid)
    only = _ENV.get("FE_FM_VARIANTS")                       # e.g. "tirex-2_univariate,toto-2_cross"
    out = {}
    for tag, (_, fn) in FM_VARIANTS.items():
        if not tag.startswith(prefix) or (only and tag not in only.split(",")) or tag in _ENV.get("FE_FM_SKIP", "").split(","):
            continue
        t0 = time.time()
        try:
            out[tag] = {w: skill(d, eo, fn(d, eo), refs) for w, eo in (("dev", dev), ("holdout", hold))}
        except Exception as e:  # noqa: BLE001
            print(json.dumps({fid: {tag: {"error": repr(e)[:300]}}}), flush=True)
            continue
        out[tag]["s"] = round(time.time() - t0)
        print(json.dumps({fid: {tag: out[tag]}}), flush=True)
    return out


def tirex2(fid: str) -> dict:
    return _fm_bench(fid, "tirex-2")


def toto2(fid: str) -> dict:
    return _fm_bench(fid, "toto-2")


def timesfm25(fid: str) -> dict:
    return _fm_bench(fid, "timesfm-2.5")


def flowstate(fid: str) -> dict:
    return _fm_bench(fid, "flowstate")


def fm2(fid: str) -> dict:
    """TimesFM 2.5 and FlowState, every variant, on one family."""
    return {**flowstate(fid), **timesfm25(fid)}


def fm3(fid: str) -> dict:
    """TiRex-2 and Toto 2.0, every variant, on one family."""
    return {**toto2(fid), **tirex2(fid)}


def collect(logs: list[str], prefixes=("tirex-2", "toto-2", "timesfm-2.5", "flowstate")) -> dict:
    """Merge the benchmark rows printed in run logs (one JSON object per line) into bench.json: `python -m forecasting_engine.lab.bench collect LOG ...`."""
    from pathlib import Path
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    n = 0
    for path in logs:
        for line in Path(path).read_text().splitlines():
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for fid, v in r.items():
                for tag, x in (v.items() if isinstance(v, dict) else ()):
                    if tag.startswith(tuple(prefixes)) and isinstance(x, dict) and "holdout" in x:
                        allr.setdefault(fid, {})[tag] = dict(x, source=Path(path).name)
                        n += 1
    OUT.write_text(json.dumps(allr, indent=1))
    return {"rows": n}


def chronos(fid: str, model: str = "amazon/chronos-bolt-small", ctx: int = 512) -> dict:
    d, reg, refs, dev, hold, tf = _setup(fid)
    res = {wname: skill(d, eo, chronos_P(d, eo, model, ctx), refs) for wname, eo in (("dev", dev), ("holdout", hold))}
    return {model.split("/")[-1] + "_zero_shot": res}


# ---------------------------------------------------------------------------------------------------------- our champion, same cells
def champion(fid: str) -> dict:
    d, reg, refs, dev, hold, tf = _setup(fid)
    ch = C.champion(fid)
    base = next((m for m in ("+rollmax", "+rollp", "+rollv", "+roll") if ch.endswith(m)), None)
    point = ch[: -len(base)] if base else ch
    out = {}
    for wname, eo in (("dev", dev), ("holdout", hold)):
        P = C.method(point)(d, refs, eo, tf)
        out.setdefault(f"engine_champion ({point})", {})[wname] = skill(d, eo, P, refs)
        if d.fam.freq == "h":
            out.setdefault(f"engine_champion next day only ({point})", {})[wname] = skill(d, eo, P, refs, _next_day_mask(d, eo))
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) < 2:
        print(__doc__)
        return 2
    if a[0] == "collect":
        print(collect(a[1:]))
        return 0
    fn = {"operator": operator, "lear": lear, "chronos": chronos, "chronos2": chronos2, "champion": champion,
          "tirex2": tirex2, "toto2": toto2, "fm3": fm3, "timesfm25": timesfm25, "flowstate": flowstate, "fm2": fm2}[a[0]]
    for fid in a[1:]:
        t0 = time.time()
        r = fn(fid)
        allr = json.loads(OUT.read_text()) if OUT.exists() else {}
        allr.setdefault(fid, {}).update(r)
        OUT.write_text(json.dumps(allr, indent=1))
        print(json.dumps({fid: r, "s": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
