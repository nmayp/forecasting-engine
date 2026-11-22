"""Coupled engine E4 (docs/methods.md, 2.3): each family's drivers pruned to its causal parents by PCMCI+ (Runge 2020, tigramite),
found on data from BEFORE the family's development window only, so the selection never sees the windows it is scored on.

    python -m forecasting_engine.lab.causal discover price_h load_h ...      # -> forecasting_engine/lab/pcmci_parents.json (merged)

Per series (country): the target and every driver of that country, hourly families as daily means, standardised; links at lags 0..TAU_MAX
(lag 0 matters for drivers published ahead, such as forecasts). A driver is a parent of the family if PCMCI+ keeps a link driver -> target (or an
unoriented contemporaneous link) in the graph pooled over all series (tigramite 'multiple' datasets mode, missing values flagged). Drivers with too little data before development
(GFS forecasts start in 2021) cannot be judged and are kept, listed apart. Method: gbm_only-pcmci[_big|_xl][_y] (core.pooled expands "pcmci").
A discovered graph is consistent with the data, not proof of causation: in the lab it is a feature selector, judged by skill like any other.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C

OUT = Path(__file__).with_name("pcmci_parents.json")
TAU_MAX = {"h": 3, "D": 3, "W": 8, "M": 12}           # daily means: energy reacts within days; 7 lags did not finish in 2 h
MIN_POINTS = {"h": 365, "D": 365, "W": 104, "M": 48}
PC_ALPHA = 0.05


def _series(df: pd.DataFrame, col: str, daily: bool) -> pd.Series | None:
    c = "*" if "*" in df.columns else col
    if c not in df.columns:
        return None
    s = df[c].astype(float)
    return s.resample("D").mean() if daily else s


def discover(fid: str, log=print) -> dict:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    t0 = time.time()
    d, reg = C.design(fid)
    freq = d.fam.freq
    daily = freq == "h"
    cut = pd.Timestamp(reg["dev_origins"][0])
    cut = cut.tz_localize("UTC") if cut.tz is None else cut.tz_convert("UTC")
    MISS = -9999.0
    per, untestable = [], set()
    for j in range(d.J):                                       # one frame per series (country): target + that country's drivers
        cc = d.country(j)
        y = d.fam.Y.iloc[:, j].astype(float)
        y = y.resample("D").mean() if daily else y
        cols = {"target": y}
        for dr in d.fam.drivers:
            s = _series(dr.data, cc, daily)
            if s is None:
                continue
            s = s.reindex(y.index)
            if s[s.index < cut].notna().sum() < MIN_POINTS[freq]:
                untestable.add(dr.name)
                continue
            cols[dr.name] = s
        X = pd.DataFrame(cols)
        X = X[(X.index < cut) & X["target"].notna()].ffill(limit=3)
        if len(X) >= MIN_POINTS[freq]:
            per.append(X)
    if not per:
        raise SystemExit(f"{fid}: no series with enough data before {cut}")
    names = ["target"] + sorted({c for X in per for c in X.columns if c != "target"})
    names = [n for n in names if n == "target" or sum(n in X.columns and X[n].notna().mean() > 0.5 for X in per) >= 2 * len(per) / 3]
    data = {}
    for k, X in enumerate(per):
        if not all(n in X.columns for n in names):
            continue
        X = X[names]
        Z = ((X - X.mean()) / X.std().replace(0, 1)).to_numpy()
        data[len(data)] = np.where(np.isfinite(Z), Z, MISS)
    pc = PCMCI(dataframe=pp.DataFrame(data, var_names=names, analysis_mode="multiple", missing_flag=MISS),
               cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    res = pc.run_pcmciplus(tau_min=0, tau_max=TAU_MAX[freq], pc_alpha=PC_ALPHA)
    g, pv = res["graph"], res["p_matrix"]
    votes = {n: 1 for i, n in enumerate(names) if i and (any(g[i, 0, t] == "-->" for t in range(1, g.shape[2])) or g[i, 0, 0] in ("-->", "o-o", "x-x"))}
    tested = {n: 1 for n in names[1:]}
    log(json.dumps(dict(family=fid, datasets=len(data), vars=len(names), samples=int(sum(len(v) for v in data.values())), s=round(time.time() - t0))))
    parents = sorted(votes)
    keep = sorted(untestable - set(tested))
    best_p = {n: round(float(np.nanmin(pv[i, 0])), 5) for i, n in enumerate(names) if i}
    out = dict(family=fid, parents=parents, kept_untestable=keep, tested=sorted(tested), min_p=best_p, datasets=len(data),
               cut=str(cut), pc_alpha=PC_ALPHA, tau_max=TAU_MAX[freq], s=round(time.time() - t0))
    log(json.dumps(out))
    return out


def parents(fid: str) -> list[str]:
    if not OUT.exists():
        return []
    r = json.loads(OUT.read_text()).get(fid)
    return (r["parents"] + r["kept_untestable"]) if r else []


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:])
    if not fams or fams[0] != "discover":
        print(__doc__)
        return 2
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for f in fams[1:]:
        res[f] = discover(f)
        OUT.write_text(json.dumps(res, indent=1))
        print(json.dumps({"pcmci": f, "result": res[f]}, default=str), flush=True)      # also in the log
    return 0


if __name__ == "__main__":
    sys.exit(main())
