"""Coupled engine E3 (docs/methods.md): one family's forecasts, as the lab would have made them at every origin, stored so other
families can use them as drivers (panel._chain). The point is the train / predict discrepancy of regressor chains (Spyromitros-Xioufis et al.
2016): a child must learn from the parent's FORECASTS, not its actual values, so every stored value comes from a model refitted only on what
was known at its own refit time (core.pooled's blocks), with features as known at its origin.

    python -m forecasting_engine.lab.chain build res_h [--method <name>]     # default: the family's champion (as in the ledger)
        [--nd N]      the family's first N drivers only (the inputs the chain was first built with; drivers are appended)
        [--out NAME]  another store name than lab/chain/<family>.parquet (e.g. a second method's chain for one driver)

Output (history store): lab/chain/<family>.parquet, long: valid time, cc, lead ("d1" = made at the origin of the day before the valid day,
"d2" = two days before), value, method, origin_hour. The first refit block of history has too little training data and stays empty.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab import core as C


def build(fid: str, method: str | None = None, log=print, nd: int | None = None, out: str | None = None) -> dict:
    t0 = time.time()
    method = method or C.champion(fid)
    if not method:
        raise SystemExit(f"{fid}: no champion yet; give --method")
    base = next((m for m in ("+rollmax", "+rollp", "+rollv", "+roll") if method.endswith(m)), None)
    point = method[: -len(base)] if base else method          # calibration suffixes change bands, not the point forecast
    d, reg = C.design(fid)
    if nd and nd < len(d.fam.drivers):
        import copy
        import dataclasses
        d = copy.copy(d)
        d.fam = dataclasses.replace(d.fam, drivers=list(d.fam.drivers[:nd]))
    refs = [reg["reference"][s] for s in d.series]
    tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
    eo = np.where(d.ok_origin & (d.p0 >= tf))[0]
    P = C.method(point)(d, refs, eo, tf)                      # (origins, J, H): each block refitted on what was known at its start
    tau = pd.to_datetime(d.tau[eo], utc=True)
    rows = []
    hourly = d.fam.freq == "h"
    for k in range(d.H):
        valid = d.stamps[np.clip(d.R[eo, k], 0, d.T - 1)]
        lead_days = (valid.floor("D") - tau.floor("D")).days
        for j, cc in enumerate(d.series):
            v = P[:, j, k]
            ok = np.isfinite(v) & (np.isin(lead_days, (1, 2)) if hourly else True)
            lead = np.where(lead_days[ok] == 1, "d1", "d2") if hourly else np.array([f"k{k + 1}"] * int(ok.sum()))
            rows.append(pd.DataFrame(dict(valid=valid[ok], cc=d.country(j), series=d.series[j], lead=lead, step=k + 1, origin=tau[ok], value=v[ok])))
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["valid", "origin"]).drop_duplicates(["valid", "series", "lead"], keep="last")
    df["method"] = point
    df["origin_hour"] = int(pd.Timestamp(d.tau[eo[0]]).hour)
    HS.put_parquet(out or f"lab/chain/{fid}.parquet", df.set_index("valid"))
    out = dict(family=fid, store=out or f"lab/chain/{fid}.parquet", drivers=len(d.fam.drivers), method=point, origins=len(eo), rows=len(df), first=str(df["valid"].min()), last=str(df["valid"].max()),
               filled=round(float(np.isfinite(P).mean()), 3), s=round(time.time() - t0))
    log(json.dumps(out))
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("family")
    ap.add_argument("--method")
    ap.add_argument("--nd", type=int)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    build(a.family, a.method, nd=a.nd, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
