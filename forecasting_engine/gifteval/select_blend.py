"""Validation-only selection of the blend.

    python -m forecasting_engine.gifteval.select_blend val        # score every (member subset, combiner) on the validation windows of each config
    python -m forecasting_engine.gifteval.select_blend select     # per (domain x term) group, pick the members and combiner -> selection.json

Validation windows are the last windows before the test region (the series cut at T - prediction_length * windows), at most 300 series per
config (seed 0). The test labels are never read. Needs the validation predictions of `forecast.py`.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import numpy as np

from forecasting_engine.gifteval import blend as B
from forecasting_engine.gifteval import data as gdata
from forecasting_engine.gifteval import harness as H


def gm(v):
    return float(np.exp(np.mean(np.log(np.maximum(v, 1e-6)))))


def stage_val(only=()):
    (B.OUT / "val").mkdir(parents=True, exist_ok=True)
    for ds, term in sorted(H.all_configs(), key=lambda c: (c[0], c[1])):
        name = H.config_name(ds, term)
        if only and name not in only:
            continue
        f = B.OUT / "val" / f"{B.safe(name)}.json"
        if f.exists():
            continue
        t0 = time.time()
        sp = gdata.build_split(ds, term, "val", H.GIFT)
        sc = B.FastScorer(sp.ctxs, sp.tgts, sp.pl, sp.season)
        sn = sc(H.seasonal_naive_quantiles(sp.ctxs, sp.pl, sp.season))
        P = {m: B.load_pred("val", name, m) for m in B.MEMBERS}
        P = {k: v for k, v in P.items() if v is not None}
        tab = {}
        for sub, comb in B.combos(B.MEMBERS):
            if not all(x in P for x in sub):
                continue
            s = sc(B.combine(np.stack([P[x] for x in sub]), comb))
            tab[B.ckey(sub, comb)] = [s["MASE"] / sn["MASE"] if sn["MASE"] > 0 else 1.0, s["CRPS"] / sn["CRPS"] if sn["CRPS"] > 0 else 1.0]
        f.write_text(json.dumps({"name": name, "n": len(sp.ctxs), "have": sorted(P), "sn": sn, "tab": tab}))
        print(name, f"{time.time() - t0:.0f}s", flush=True)


def stage_select():
    props = json.loads((H.OFFICIAL / "dataset_properties.json").read_text())
    V = {}
    for f in sorted((B.OUT / "val").glob("*.json")):
        d = json.loads(f.read_text())
        V[d["name"]] = d
    groups = defaultdict(list)
    for c in V:
        groups[f"{props.get(c.split('/')[0], {}).get('domain', 'unknown')}_{c.split('/')[-1]}"].append(c)
    final = defaultdict(list)
    for g, cs in groups.items():                      # a group with fewer than 3 configs is merged into its term
        if len(cs) < 3:
            final[cs[0].split("/")[-1]].extend(cs)
        else:
            final[g].extend(cs)
    sel = {}
    for g, cs in final.items():
        agg = {}
        for k in V[cs[0]]["tab"]:
            if all(k in V[c]["tab"] for c in cs):
                agg[k] = (gm([V[c]["tab"][k][0] for c in cs]), gm([V[c]["tab"][k][1] for c in cs]))
        bm, bms, bmc, bs, bss = None, float("inf"), float("inf"), None, float("inf")
        for k, (ma, cr) in agg.items():
            n = len(k.split("#")[0].split("|"))
            if n > 1:
                if ma < bms - 1e-6 or (abs(ma - bms) < 1e-6 and cr < bmc):
                    bm, bms, bmc = k, ma, cr
            elif ma < bss:
                bs, bss = k, ma
        w = bs if (bs is not None and bss < 0.97 * bms) else (bm if bm is not None else bs)    # a single member must be clearly better to win
        sub, comb = w.split("#")
        allk = B.ckey(tuple(B.MEMBERS), "mean")
        sel[g] = {"configs": cs, "members": sub.split("|"), "combiner": comb, "val_MASE": agg[w][0], "val_CRPS": agg[w][1],
                  "all_four_mean_MASE": agg[allk][0] if allk in agg else None}
    json.dump(sel, open(B.OUT / "selection.json", "w"), indent=2)
    print({g: (s["members"], s["combiner"], round(s["val_MASE"], 4)) for g, s in sel.items()})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "val":
        stage_val(sys.argv[2:])
    elif cmd == "select":
        stage_select()
    else:
        sys.exit(__doc__)
