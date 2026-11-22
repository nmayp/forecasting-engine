"""Scenario test of the strike mechanism (docs/methods.md). Usage:
  STRIKE_CACHE=dir python -m forecasting_engine.whatif.backtests.strike_run air|port dev|check [spec]
Per cell (event cluster x country x series): window = first strike day - 7 .. last strike day + 14; the strike calendar of the window is GIVEN; the baseline is
fitted on data public at the window start (publication lag LAG); k and u are fitted on development events only (leave-one-event-out for development cells).
Writes results/cascade/strike_<measure>_<split>_<spec>.json (+ arrays in strike_arrays_*.npz for the open-model comparison)."""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import strike_cat as C
from forecasting_engine.whatif.backtests import strike_data as D
from forecasting_engine.whatif.backtests import strike_mech as M

OUT = Path(__file__).resolve().parents[3] / "results" / "cascade"
MOVE = {"air": 0.02, "port": 0.03}        # a cell "moved" if the mean realised loss over its strike days is at least this share and 1.5 placebo sigma
# Pass rules, fixed before the check look: skill > engine-off AND > best open; direction >= 0.80 of moved cells; median size ratio 0.8-1.25; timing: median
# |trough-day offset| <= 1 day
SERIES = {"air": ["flights"], "port": ["calls", "vol"]}


def clusters(ev: dict, gap: int = 21) -> list[dict]:
    pairs = sorted(zip(ev["days"], ev["inten"]))
    out, cur = [], [pairs[0]]
    for p in pairs[1:]:
        if (p[0] - cur[-1][0]).days > gap:
            out.append(cur); cur = []
        cur.append(p)
    out.append(cur)
    return [dict(days=[a for a, _ in c], inten=[b for _, b in c]) for c in out]


def get_series(measure: str, cc: str, name: str) -> pd.Series | None:
    if measure == "air":
        a = D.air()
        return a[cc] if cc in a.columns else None
    p = D.ports().get(cc)
    return None if p is None else p[name].where(p[name] > 0)


def all_days(cc: str, measure: str) -> list:
    return [d for e in C.load() if e["country"] == cc and e["measure"] == measure for d in e["days"]]


def truth(measure, e, cl, name, y):
    """realised per-day loss shares on the cluster's strike days from the centred counterfactual"""
    centre = cl["days"][len(cl["days"]) // 2]
    cf, clean = M.counterfactual(y, e["country"], all_days(e["country"], measure), centre, kind=measure)
    idx = cf.index
    ef = np.log(y.reindex(idx)) - np.log(cf)
    sig = float(ef[clean & (abs((idx - centre).days) <= 90)].std())
    loss = {d: float(1 - np.exp(ef.get(d, np.nan))) for d in cl["days"]}
    return cf, ef, sig, loss


def fit_ku(rows: list[dict], sector: str | None, bysector: bool) -> tuple[float, float]:
    use = [r for r in rows if (not bysector) or r["sector"] == sector]
    if bysector and len({r["event"] for r in use}) < 2:
        use = rows
    I = np.array([r["I"] for r in use]); L = np.clip(np.array([r["loss"] for r in use]), 0, None)
    ok = np.isfinite(L) & (I > 0)
    k = float(L[ok].sum() / I[ok].sum()) if ok.any() else 0.0
    cu = np.array([r["catch"] for r in use]); ok2 = np.isfinite(cu) & np.isfinite(L)
    u = float(np.clip(np.nansum(cu[ok2]) / max(np.nansum(L[ok2]), 1e-9), 0, 1)) if ok2.any() else 0.0
    return k, u


def run(measure: str, split: str, spec: str = "S0", lag: int | None = None) -> dict:
    lag = M.LAG[measure] if lag is None else lag
    bysector = spec == "S1"
    cells = []
    for e in C.events(measure=measure):
        if e["split"] not in ("dev", "check"):
            continue
        for name in SERIES[measure]:
            y = get_series(measure, e["country"], name)
            if y is None:
                continue
            for ci, cl in enumerate(clusters(e)):
                cf, ef, sig, loss = truth(measure, e, cl, name, y)
                # catch-up: mean relative excess over the CATCH days after each strike day not themselves strike days
                catch = {}
                for d in cl["days"]:
                    nxt = [d + pd.Timedelta(days=m) for m in range(1, M.CATCH + 1)]
                    if any(x in set(all_days(e["country"], measure)) for x in nxt):
                        catch[d] = np.nan
                    else:
                        catch[d] = float(np.nansum([np.exp(ef.get(x, np.nan)) - 1 for x in nxt]))
                cells.append(dict(event=e["event_id"], cluster=ci, cc=e["country"], sector=e["sector"], split=e["split"], name=name, cl=cl, y=y, cf=cf, ef=ef,
                                  sigma=sig, loss=loss, catch=catch, verified=e["verified"], basis=e["intensity_basis"]))
    dev_rows = {}
    for c in cells:
        if c["split"] != "dev":
            continue
        dev_rows.setdefault(c["name"], []).extend(
            dict(event=c["event"], sector=c["sector"], I=i, loss=c["loss"][d], catch=c["catch"][d]) for d, i in zip(c["cl"]["days"], c["cl"]["inten"]))
    res = []
    for c in cells:
        if c["split"] != split:
            continue
        rows = [r for r in dev_rows.get(c["name"], []) if r["event"] != c["event"]]
        k, u = (1.0, 0.0) if spec == "S2" else fit_ku(rows, c["sector"], bysector)
        u = 0.0 if spec == "S3" else u
        days = c["cl"]["days"]
        w = pd.date_range(days[0] - pd.Timedelta(days=7), days[-1] + pd.Timedelta(days=14))
        cutoff = w[0] - pd.Timedelta(days=1 + lag)
        base = M.baseline(c["y"], c["cc"], cutoff, w, all_days(c["cc"], measure), kind=measure)
        fac = M.strike_term(w, days, c["cl"]["inten"], k, u)
        pred = base * fac
        act = c["y"].reindex(w)
        ref = c["y"].reindex(w - pd.Timedelta(days=364)).set_axis(w)
        ok = act.notna() & ref.notna()
        mae = lambda p: float(np.abs(p[ok] - act[ok]).mean() / base[ok].mean())  # noqa: E731
        pl = np.array([1 - fac[d] for d in days]); rl = np.array([c["loss"][d] for d in days])
        ml = float(np.nanmean(rl))
        moved = bool(ml >= MOVE[measure] and ml >= 1.5 * c["sigma"] / np.sqrt(max(len(days), 1)))
        # trough: window day of the largest loss share (predicted vs realised), only where both exist
        rl_w = (1 - np.exp(c["ef"].reindex(w))).to_numpy(); pl_w = (1 - fac).to_numpy()
        trough = int(abs(int(np.nanargmax(np.where(np.isfinite(rl_w), rl_w, -9))) - int(np.argmax(pl_w)))) if moved else None
        res.append(dict(event=c["event"], cluster=c["cluster"], cc=c["cc"], sector=c["sector"], name=c["name"], k=k, u=u, n_days=len(days), verified=c["verified"],
                        basis=c["basis"], sigma=c["sigma"], mean_real_loss=ml, mean_pred_loss=float(pl.mean()), moved=moved,
                        size_ratio=float(pl.sum() / np.nansum(np.clip(rl, 0, None))) if moved else None, trough_off=trough,
                        dir_days=float(np.mean(rl > 0)), mae_mech=mae(pred), mae_off=mae(base), mae_ref=mae(ref),
                        window=[str(w[0].date()), str(w[-1].date())], real_days=[float(x) for x in rl], pred_days=[float(x) for x in pl],
                        arrays=dict(w=[str(x.date()) for x in w], act=act.tolist(), pred=pred.tolist(), base=base.tolist(), ref=ref.tolist(), cutoff=str(cutoff.date()))))
    return dict(measure=measure, split=split, spec=spec, lag=lag, cells=res)


def summarise(r: dict) -> dict:
    c = r["cells"]
    mv = [x for x in c if x["moved"]]
    ratio = [x["size_ratio"] for x in mv if x["size_ratio"] is not None]
    pooled = lambda k: float(np.sum([x[k] for x in c]))  # noqa: E731
    return dict(n=len(c), moved=len(mv), dir_moved=float(np.mean([x["mean_pred_loss"] > 0 for x in mv])) if mv else None,
                dir_days=float(np.mean([x["dir_days"] for x in mv])) if mv else None,
                size_median=float(np.median(ratio)) if ratio else None, size_pooled=float(sum(x["mean_pred_loss"] * x["n_days"] for x in mv) /
                                                                                       max(sum(max(x["mean_real_loss"], 0) * x["n_days"] for x in mv), 1e-9)) if mv else None,
                within20=float(np.mean([abs(v - 1) <= .2 for v in ratio])) if ratio else None,
                trough_off_median=float(np.median([x["trough_off"] for x in mv if x["trough_off"] is not None])) if mv else None,
                skill_vs_ref=1 - pooled("mae_mech") / pooled("mae_ref"), skill_off_vs_ref=1 - pooled("mae_off") / pooled("mae_ref"),
                skill_vs_off=1 - pooled("mae_mech") / pooled("mae_off"),
                cells_beating_off=int(sum(x["mae_mech"] < x["mae_off"] for x in c)))


if __name__ == "__main__":
    measure, split = sys.argv[1], sys.argv[2]
    spec = sys.argv[3] if len(sys.argv) > 3 else "S0"
    lag = int(sys.argv[4]) if len(sys.argv) > 4 else None
    r = run(measure, split, spec, lag)
    r["summary"] = summarise(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"strike_{measure}_{split}_{spec}{'' if lag is None else f'_lag{lag}'}.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["summary"], indent=1))
    for x in r["cells"]:
        print(f"{x['event']:<24}{x['cc']} {x['name']:<7} k={x['k']:.2f} u={x['u']:.2f} real={x['mean_real_loss']:+.3f} pred={x['mean_pred_loss']:+.3f} moved={x['moved']!s:<5} "
              f"ratio={x['size_ratio']} mae mech/off/ref={x['mae_mech']:.3f}/{x['mae_off']:.3f}/{x['mae_ref']:.3f}")
