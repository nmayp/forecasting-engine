"""Research #7 (week 2): TabPFN-v2 as an in-context regressor on the lab's own feature table (experiment A of
docs/methods.md). Registered as lab methods on import (core.method imports this module for "tabpfn*").

    tabpfn_ref            the scaled residual over the strict reference (the GBM methods' target, clipped to +-20), every driver of
                          every layer as known at the origin (the `gbm_all` table), pooled over the family's series; refitted every
                          core.REFIT period on what was known then; point = the median of TabPFN's predictive distribution
    tabpfn_ref_c<N>       the same with a context of N rows (default CTX)

Context: the most recent CTX training rows known at the refit time (newest origins first), all series and steps of an origin kept
together. Columns that are empty in the context are dropped; if more than MAX_FEAT remain, the first MAX_FEAT in table order are kept
(step, calendar, series id, simple forecasts and own state come first, then drivers in their registered order).
q10 / q50 / q90 of each forecast (in the target's units) are kept in QUANTS for the round script (tabpfn_round), which reports
TabPFN's own 80 % band next to the lab's calibrated one; the lab's scoring itself uses the point (q50), as for every other method.

Licence: TabPFN-v2 regressor weights (Prior-Labs/TabPFN-v2-reg) are under the Prior Labs License 1.1 = Apache 2.0 + an attribution
clause (show "Built with PriorLabs-TabPFN" and ship the licence with any product or service that contains or uses the weights;
internal benchmarking needs no attribution). Commercial production use is allowed under those conditions. The package default
(TabPFN-3.5) and TabPFN-2.5 / 2.6 / 3 weights are NON-commercial: this module always asks for ModelVersion.V2 explicitly.
CPU only; no data leaves the machine (no tabpfn-client / API).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np

from forecasting_engine.lab import core as C

CTX = int(os.environ.get("FE_TABPFN_CTX", "2000"))
N_EST = int(os.environ.get("FE_TABPFN_NEST", "2"))
MAX_FEAT = 400
VERSION = 1
QUANTS: dict = {}          # (family, origins sha) -> (3, O, J, H) q10 / q50 / q90 in the target's units
TIMES: dict = {}           # (family, origins sha) -> seconds spent in TabPFN fit + predict
_MEMO: dict = {}
_MODEL_DIR = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "tabpfn")))


def _regressor():
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(_MODEL_DIR))
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=N_EST, ignore_pretraining_limits=True,
                                                      fit_mode="fit_with_cache", memory_saving_mode=False, random_state=0)


def _context(Kt: np.ndarray, ok: np.ndarray, ctx: int) -> np.ndarray:
    """Row mask: the newest training origins until ctx rows (whole origins)."""
    o = Kt[:, 0]
    keep = np.zeros(len(o), bool)
    n = 0
    for org in np.unique(o[ok])[::-1]:
        m = ok & (o == org)
        keep |= m
        n += int(m.sum())
        if n >= ctx:
            break
    return keep


def tabpfn_pooled(d, refs, eval_origins, train_from, ctx: int = CTX):
    """As core.pooled(model="gbm") with TabPFN-v2 in place of the trees. -> (origins x J x H)."""
    sha = hashlib.sha1(np.asarray(eval_origins, np.int64).tobytes()).hexdigest()
    key = (d.fam.id, sha, train_from, tuple(map(str, refs)), ctx, N_EST, len(d.fam.drivers))
    if key in _MEMO:
        return _MEMO[key]
    h = hashlib.sha256(repr((VERSION, key[2:], [x.name for x in d.fam.drivers], d.T)).encode() + C._lagsig(d, True).encode())
    h.update(np.ascontiguousarray(d.Y).tobytes())
    h.update(np.asarray(eval_origins, np.int64).tobytes())
    fname = f"tabpfn_{d.fam.id}_{h.hexdigest()[:20]}.npy"
    Q = C._fm_cache_get(fname)
    if Q is not None and Q.shape == (3, len(eval_origins), d.J, d.H):
        QUANTS[(d.fam.id, sha)] = Q
        _MEMO[key] = Q[1]
        return Q[1]
    out = np.full((3, len(eval_origins), d.J, d.H), np.nan)
    tau = d.tau[eval_origins]
    step = C.REFIT[d.fam.freq].value
    t0 = tau.min()
    blocks = ((tau - t0) // step).astype(int)
    stride = C._stride(d)
    need = np.union1d(C._train_origins(d, train_from, t0 + int(blocks.max()) * step, stride), eval_origins)
    X, Z, REF, SC, K = C._features(d, refs, None, True, need, False)
    spent = 0.0
    for b in np.unique(blocks):
        sel = np.where(blocks == b)[0]
        rho = t0 + b * step
        tr = C._train_origins(d, train_from, rho, stride)
        if len(tr) < 30:
            continue
        rt = np.isin(K[:, 0], tr)
        Xt, zt, kt = X[rt], Z[rt], K[rt]
        okr = np.isfinite(zt)
        if okr.sum() < 50:
            continue
        m = _context(kt, okr, ctx)
        Xa, za = Xt[m], np.clip(zt[m], -20, 20)
        keep = np.where(np.isfinite(Xa).any(0) & (np.nanstd(Xa, 0) > 0))[0][:MAX_FEAT]
        re_ = np.isin(K[:, 0], eval_origins[sel])
        Xe, refe, sce, ke = X[re_][:, keep], REF[re_], SC[re_], K[re_]
        t1 = time.time()
        reg = _regressor()
        reg.fit(Xa[:, keep], za)
        q = np.asarray(reg.predict(Xe, output_type="quantiles", quantiles=[0.1, 0.5, 0.9]))   # (3, rows)
        spent += time.time() - t1
        q = np.clip(q, -20, 20)
        pos = np.searchsorted(eval_origins[sel], ke[:, 0])
        for i in range(3):
            out[i, sel[pos], ke[:, 1], ke[:, 2] - 1] = refe + sce * q[i]
        print(f"  tabpfn {d.fam.id} block {b}: ctx {len(za)} x {len(keep)}, eval {len(Xe)}, {time.time() - t1:.0f}s", flush=True)
    QUANTS[(d.fam.id, sha)] = out
    TIMES[(d.fam.id, sha)] = round(spent, 1)
    C._fm_cache_put(fname, out)
    _MEMO[key] = out[1]
    return out[1]


def _method(ctx: int):
    return lambda d, refs, eo, tf: tabpfn_pooled(d, refs, eo, tf, ctx=ctx)


C.METHODS["tabpfn_ref"] = _method(CTX)


def register(name: str) -> None:
    m = re.match(r"^tabpfn_ref_c(\d+)$", name)
    if m:
        C.METHODS[name] = _method(int(m.group(1)))
