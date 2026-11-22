"""Chronos-2 with the ragged edge (cross-layer round, 24 Oct 2026).

bench.chronos2_P gives every covariate to the model only over the TARGET's history window: a driver that is published before the target is (a
survey, a weekly or daily indicator) is cut off where the target's own newest value ends, so everything the driver learned during the target's
publication delay is thrown away (non-"ahead" drivers get no future covariate). Here the covariates are flagged "ahead", which makes bench.chronos2_P
also pass each driver's values at the gap and beyond wherever they were published by the origin (the rest carried forward from the last one known).
Nothing published after the origin enters. Members: chronos2xg (every driver, cross context) and chronos2sg (the selected drivers).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib

import numpy as np

from forecasting_engine.lab import core as C


def _gap_design(d, select: bool):
    from forecasting_engine.lab import bench as BN
    names = {x.name for x in BN._cov_drivers(d, select=select)}
    drs = [dataclasses.replace(x, ahead=True) for x in d.fam.drivers if x.name in names]
    d2 = copy.copy(d)
    d2.fam = dataclasses.replace(d.fam, drivers=drs)
    return d2


def fm_c2g(select: bool):
    def f(d, eo):
        from forecasting_engine.lab import bench as BN
        key = (d.fam.id, d.T, "c2gap", select, hashlib.sha1(np.asarray(eo, np.int64).tobytes()).hexdigest())
        if key not in C._FM_MEMO:
            drv = hashlib.sha1("|".join(sorted(x.name for x in d.fam.drivers)).encode()).hexdigest()[:10]
            name = f"{d.fam.id}_c2gap_{int(select)}_{d.T}_{drv}_{key[-1][:16]}{C._lagsig(d, True)}.npy"
            P = C._fm_cache_get(name)
            if P is None or P.shape != (len(eo), d.J, d.H):
                P = BN.chronos2_P(_gap_design(d, select), eo, True, True, select=False)
                C._fm_cache_put(name, P)
            C._FM_MEMO[key] = P
        return C._FM_MEMO[key]
    return f


C.FM["chronos2xg"] = fm_c2g(False)
C.FM["chronos2sg"] = fm_c2g(True)
C.METHODS["chronos2xg"] = lambda d, refs, eo, tf: C.FM["chronos2xg"](d, eo)
C.METHODS["chronos2sg"] = lambda d, refs, eo, tf: C.FM["chronos2sg"](d, eo)
