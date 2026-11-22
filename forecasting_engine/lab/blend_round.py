"""Open-model blending round (Oct 2026): the champion blended with the strongest open-model variants, weights fitted on development only.

    python -m forecasting_engine.lab.blend_round explore FAM[,FAM..]     # dev table of members and candidates (no holdout, no ledger row)
    python -m forecasting_engine.lab.blend_round look FAM [--method M]   # one core.run with the holdout look for the dev-best candidate

Members: the champion and the open-model variants (Chronos-2, TiRex-2, Toto 2.0, FlowState, TimesFM 2.5) whose forecasts are in the
shared cache. Open members are ranked by their DEVELOPMENT score only. Candidates are fixed-weight blends (core "wblend[..]:"):
champion + best open (equal), champion + best two (equal thirds), weights fitted on development (grid, step 10 %), and that fit
shrunk half way to equal weights. The dev gate and the single holdout look are core.run's. The bar for a "win" stays the best single
open model, so a blend that contains it must beat it; the scoreboard wording says these are ensembles that include open models.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C

TAG = {"chronos-2_univariate": "chronos2", "chronos-2_covariates_cross": "chronos2x", "chronos-2_selected_cross": "chronos2s",
       "toto-2_univariate": "toto2", "toto-2_cross": "toto2x", "tirex-2_univariate": "tirex2", "tirex-2_cross": "tirex2x",
       "tirex-2_covariates": "tirex2c", "flowstate_univariate": "flowstate", "timesfm-2.5_univariate": "timesfm25"}
OPEN = ("chronos2", "chronos2x", "chronos2s", "toto2", "toto2x", "tirex2", "tirex2x", "tirex2c", "flowstate", "timesfm25")
OUT = C.LAB / "blend_round.json"


def _out(fid):
    return C.LAB / "blend_round" / f"{fid}.json"


def _no_compute():
    from forecasting_engine.lab import bench as BN

    def boom(*a, **k):
        raise LookupError("not in the forecast cache")
    for n in ("toto2_P", "tirex2_P", "chronos2_P", "chronos_P"):
        if hasattr(BN, n):
            setattr(BN, n, boom)
    for k in list(getattr(BN, "FM_EXTRA", {})):
        BN.FM_EXTRA[k] = boom


def _trunc(fid: str):
    """The design with the driver set the champion was chosen with (core.run's gate does the same for the bar): new drivers are not handed
    to the blend, and the champion's saved forecasts (keyed on the driver names) are found."""
    import copy
    import dataclasses
    d, reg = C.design(fid)
    row = C.champion_row(fid)
    nd = (row or {}).get("drivers")
    if nd and nd < len(d.fam.drivers):
        d = copy.copy(d)
        d.fam = dataclasses.replace(d.fam, drivers=list(d.fam.drivers[:nd]))
        C._DESIGNS[fid] = (d, reg)
    return d, reg


def _setup(fid: str):
    d, reg = _trunc(fid)
    refs = [reg["reference"][s] for s in d.series]
    dev, hold = C._idx(d, reg["dev_origins"]), C._idx(d, reg["holdout_origins"])
    tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
    return d, refs, dev, hold, tf


def _champ(fid: str) -> str:
    c = C.champion(fid)
    for m in ("+rollmax", "+rollp", "+rollv", "+roll"):
        if c.endswith(m):
            return c[: -len(m)]
    return c


def simplex(n: int, step: int = 10):
    for w in itertools.product(range(0, 101, step), repeat=n - 1):
        if sum(w) <= 100:
            yield list(w) + [100 - sum(w)]


def bench_members(fid: str, k: int = 2) -> list[str]:
    """The k open-model variants with the best DEVELOPMENT score in bench.json that the member grammar can name (never the holdout)."""
    b = json.loads((C.LAB / "bench.json").read_text()).get(fid, {})
    sc = sorted([(v["dev"]["pooled_skill"], TAG[t]) for t, v in b.items() if t in TAG and isinstance(v, dict) and v.get("dev")
                 and v["dev"].get("pooled_skill") is not None], reverse=True)
    return [m for _, m in sc[:k]]


def explore(fid: str, compute: bool = False, only_members: bool = False) -> dict:
    """compute=False: only forecasts already in the cache (a missing member is reported). only_members: build the pool from the two
    best-on-development open variants of bench.json (computing them if compute=True) instead of every cached variant."""
    if not compute:
        _no_compute()
    d, refs, dev, hold, tf = _setup(fid)
    blocks = C.dev_blocks(d.tau[dev])
    ch = _champ(fid)
    mem = {}
    P = {}
    for name in ("{champ}",) + (tuple(bench_members(fid)) if only_members else OPEN):
        try:
            P[name] = C.memo_method(ch if name == "{champ}" else name)(d, refs, dev, tf)
        except Exception as e:  # noqa: BLE001
            mem[name] = f"missing: {type(e).__name__}"
    if "{champ}" not in P:
        return dict(family=fid, champion=ch, missing=mem)
    sk = {n: C.score(d, dev, p, refs)["pooled_skill"] for n, p in P.items()}
    bl = {n: C.block_scores(d, dev, p, refs, blocks) for n, p in P.items()}
    opens = sorted([n for n in P if n != "{champ}"], key=lambda n: -(sk[n] or -9))
    res = dict(family=fid, champion=ch, dev=sk, missing=mem, ranked_open=opens)
    if len(opens) < 1:
        return res
    cand = {}

    def add(label, names, w):
        name = "wblend[" + ",".join(str(x) for x in w) + "]:" + "|".join("{" + ch + "}" if n == "{champ}" else n for n in names)
        Pb = C._wmean([P[n] for n in names], w)
        s = C.score(d, dev, Pb, refs)["pooled_skill"]
        b = C.block_scores(d, dev, Pb, refs, blocks)
        g = C.gate(s, b, sk["{champ}"], bl["{champ}"])
        cand[label] = dict(method=name, dev=s, gate=g, w=w, names=names)
    top1 = ["{champ}", opens[0]]
    add("c+b1", top1, [50, 50])
    if len(opens) > 1:
        top2 = top1 + [opens[1]]
        add("c+b2", top2, [34, 33, 33])
        grid = top2
    else:
        grid = top1
    # weights fitted on development: pooled skill is a sum of absolute errors, so minimise it directly (ties to the equal weights)
    best, bw = None, None
    for w in simplex(len(grid)):
        if w[0] == 0 and len(grid) > 1 and False:
            continue
        s = C.score(d, dev, C._wmean([P[n] for n in grid], w), refs)["pooled_skill"]
        if best is None or s > best + 1e-9:
            best, bw = s, w
    add("fit", grid, bw)
    eq = 100 / len(grid)
    sh = [int(round(0.5 * x + 0.5 * eq)) for x in bw]
    sh[0] += 100 - sum(sh)
    add("fit_shrunk", grid, sh)
    res["cands"] = cand
    return res


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    cmd = a[0]
    if cmd == "explore":
        (C.LAB / "blend_round").mkdir(exist_ok=True)
        for fid in a[1].split(","):
            try:
                r = explore(fid)
            except Exception as e:  # noqa: BLE001
                r = dict(family=fid, error=f"{type(e).__name__}: {e}")
            print(fid, json.dumps({k: v for k, v in r.items() if k != "cands"}, default=str)[:600], flush=True)
            for k, c in (r.get("cands") or {}).items():
                print("   ", k, c["w"], c["names"], round((c["dev"] or 0) * 100, 2), c["gate"], flush=True)
            _out(fid).write_text(json.dumps(r, indent=1, default=str))
    elif cmd in ("auto", "auto-cached"):
        # dev-best candidate (gate-qualified ones first) -> core.run with the single holdout look
        fid = a[1]
        r = explore(fid, compute=cmd == "auto", only_members=cmd == "auto")
        (C.LAB / "blend_round").mkdir(exist_ok=True)
        _out(fid).write_text(json.dumps(r, indent=1, default=str))
        cs = r.get("cands") or {}
        if not cs:
            print(fid, "no candidates or missing members", r.get("missing"))
            return 1
        pick = max(cs.values(), key=lambda c: (c["gate"]["qualified"], c["dev"] or -9))
        print(fid, "picked", pick["method"], pick["dev"], pick["gate"], flush=True)
        if len(a) > 2 and a[2] == "dry":
            return 0
        _trunc(fid)
        C.run(pick["method"], [fid], round_="blend1", holdout=True, note="champion blended with open-model variants, weights on development only")
    elif cmd == "look":
        fid, method = a[1], a[2]
        _trunc(fid)
        C.run(method, [fid], round_="blend1", holdout=True, note="open-model blend, weights on development only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
