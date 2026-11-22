"""results/lab/report.json: per family the champion and its locked-window scores, what each layer adds (leave-one-layer
out on development), every trial; the shock replays."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C

# families built on ENTSO-E items outside its free re-use list (actual load, prices, generation): back-testing only, so the public report
# carries their scores but not their values (docs/methods.md, licences)
RESTRICTED = {"price_h", "load_h", "res_h", "load_d"}


def _actual(d: C.Design, paths: dict) -> dict:
    out = {}
    for sid, ps in (paths or {}).items():
        if sid not in d.series or not ps:
            continue
        j = d.series.index(sid)
        utc = lambda x: pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tz is None else pd.Timestamp(x).tz_convert("UTC")  # noqa: E731
        t0 = utc(ps[0]["origin"]) - pd.Timedelta(days={"h": 3, "D": 30, "W": 120, "M": 730}[d.fam.freq])
        t1 = utc(ps[-1]["stamps"][-1]) if ps[-1]["stamps"] else utc(ps[-1]["origin"])
        s = d.fam.Y.iloc[:, j]
        s = s[(s.index >= t0) & (s.index <= t1)].dropna()
        if d.fam.freq == "h" and len(s) > 1500:
            s = s.iloc[::3]
        out[sid] = [[str(i), round(float(v), 4)] for i, v in s.items()]
    return out


def report() -> dict:
    reg = C.registry()
    led = C.ledger()
    fams = []
    for f in reg["families"]:
        tr = [r for r in led if r["family"] == f["id"]]
        ch = C.champion(f["id"])
        best = next((r for r in reversed(tr) if r["method"] == ch and r.get("holdout")), None) if ch else None
        dev_by = {}
        for r in tr:
            if r.get("dev") and r["dev"].get("pooled_skill") is not None:
                dev_by[r["method"]] = r["dev"]["pooled_skill"]
        layers = []
        if "gbm_all" in dev_by:
            for m, v in dev_by.items():
                if m.startswith("gbm_all-"):
                    layers.append(dict(layer=m.split("-", 1)[1], adds=round(dev_by["gbm_all"] - v, 4)))
        layers.sort(key=lambda x: -x["adds"])
        ablation = {k: dev_by.get(k) for k in ("ref", "ets", "gbm_own", "gbm_cross", "gbm_all", "gbm_single")}
        actual = None
        if best and best.get("holdout_paths") and f["id"] not in RESTRICTED:
            try:
                d, _ = C.design(f["id"])
                actual = _actual(d, best["holdout_paths"])
            except Exception:  # noqa: BLE001
                actual = None
        fams.append(dict({k: f.get(k) for k in ("id", "label", "layer", "freq", "unit", "H", "available", "start", "end", "holdout_start", "dev_start")},
                         n_series=len(f.get("series") or []), series=f.get("series"), drivers=f.get("drivers"),
                         n_dev=len(f.get("dev_origins") or []), n_hold=len(f.get("holdout_origins") or []),
                         trials=len(tr), holdout_looks=C.holdout_looks(tr), gated_out=sum(1 for r in tr if r.get("qualified") is False), champion=ch, layers=layers, ablation_dev=ablation,
                         methods=[{k: (r.get(k) if k not in ("dev", "holdout") else {kk: vv for kk, vv in (r.get(k) or {}).items() if kk != "per_series"})
                                   for k in ("method", "round", "note", "utc", "dev", "dev_blocks", "holdout", "qualified", "blocks_won", "accepted", "vs", "error", "code")}
                                  for r in tr],
                         best=dict(method=ch, dev={k: v for k, v in best["dev"].items() if k != "per_series"}, holdout=best["holdout"],
                                   paths=None if f["id"] in RESTRICTED else best.get("holdout_paths")) if best else None, actual=actual,
                         values_withheld=f["id"] in RESTRICTED))
    shocks = None
    p = C.LAB / "shocks.json"
    if p.exists():
        shocks = json.loads(p.read_text())
    out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), version=2, registry_utc=reg["created_utc"], rule=reg["rule"],
               holdout_share=reg["holdout_share"], trials=len(led), holdout_looks=C.holdout_looks(led),
               series=sum(len(f.get("series") or []) for f in reg["families"] if f.get("available")), families=fams, shocks=shocks)
    C.REPORT.write_text(json.dumps(out, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)))
    return out
