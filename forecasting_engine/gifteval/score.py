"""Score members, the plain four-model mean and the selected blend on the test windows; aggregate like the leaderboard.

    python -m forecasting_engine.gifteval.score test        # per config -> out/test/<config>.json
    python -m forecasting_engine.gifteval.score aggregate   # geometric mean of metric / Seasonal_Naive; writes out/all_results.csv

Needs out/selection.json (from select_blend.py), the test predictions of forecast.py and the official GIFT-Eval files under $GIFT_ROOT/_official.
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

from forecasting_engine.gifteval import blend as B
from forecasting_engine.gifteval import data as gdata
from forecasting_engine.gifteval import harness as H


def extra(y, q):
    """The other leaderboard columns, computed on the 0.5 quantile (no 0.05 / 0.95 quantiles, so MSIS stays empty)."""
    y = np.asarray(y, dtype=np.float64)
    q5 = np.asarray(q[:, 4, :], dtype=np.float64)
    valid = ~np.isnan(y)
    ys = np.where(valid, y, np.nan)
    q5 = np.where(valid, q5, np.nan)
    mse = float(np.nanmean((ys - q5) ** 2))
    mae = float(np.nanmean(np.abs(ys - q5)))
    rmse = float(np.sqrt(mse))
    may = float(np.nanmean(np.abs(ys)))
    say = float(np.nansum(np.abs(ys)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs(ys - q5) / np.abs(ys)
        smape = 2 * np.abs(ys - q5) / (np.abs(ys) + np.abs(q5))
    return {"MSE[mean]": mse, "MSE[0.5]": mse, "MAE[0.5]": mae,
            "MAPE[0.5]": float(np.nanmean(np.where(np.abs(ys) > 0, mape, np.nan))),
            "sMAPE[0.5]": float(np.nanmean(np.where((np.abs(ys) + np.abs(q5)) > 0, smape, np.nan))), "MSIS": float("nan"),
            "RMSE[mean]": rmse, "NRMSE[mean]": rmse / may if may > 0 else float("nan"), "ND[0.5]": float(np.nansum(np.abs(ys - q5)) / say) if say > 0 else float("nan")}


def stage_test(only=()):
    sel = json.load(open(B.OUT / "selection.json"))
    c2g = {c: g for g, s in sel.items() for c in s["configs"]}
    (B.OUT / "test").mkdir(parents=True, exist_ok=True)
    for ds, term in sorted(H.all_configs(), key=lambda c: (c[0], c[1])):
        name = H.config_name(ds, term)
        if only and name not in only:
            continue
        f = B.OUT / "test" / f"{B.safe(name)}.json"
        if f.exists():
            continue
        t0 = time.time()
        s = gdata.build_split(ds, term, "test", H.GIFT)
        ref = H.load_task(ds, term, H.GIFT)
        assert np.array_equal(np.nan_to_num(ref.targets, nan=-9e9), np.nan_to_num(s.tgts, nan=-9e9)), "order mismatch " + name
        sc = B.FastScorer(s.ctxs, s.tgts, s.pl, s.season)
        P = {m: B.load_pred("test", name, m) for m in B.MEMBERS}
        P = {k: v for k, v in P.items() if v is not None}
        r = {"name": name, "members": {m: sc(P[m]) for m in P}}
        if len(P) == len(B.MEMBERS):
            r["mean4"] = sc(np.mean(np.stack([P[m] for m in B.MEMBERS]), axis=0))
        info = sel[c2g[name]]
        if all(m in P for m in info["members"]):
            q = B.combine(np.stack([P[m] for m in info["members"]]), info["combiner"])
            b = sc(q)
            b.update(extra(s.tgts, q))
            r["blend"] = b
        f.write_text(json.dumps(r))
        print(name, f"{time.time() - t0:.0f}s", flush=True)


def rel(T, sn, get):
    a, b = [], []
    for c in sorted(T):
        s = get(T[c])
        if s is None:
            return None
        a.append(s["MASE"] / sn[c]["MASE"])
        b.append(s["CRPS"] / sn[c]["CRPS"])
    gm = lambda v: float(np.exp(np.mean(np.log(np.maximum(v, 1e-6)))))
    return round(gm(a), 4), round(gm(b), 4), len(a)


def aggregate():
    import pandas as pd
    T = {}
    for f in (B.OUT / "test").glob("*.json"):
        d = json.load(open(f))
        T[d["name"]] = d
    sn = H.published("Seasonal_Naive")
    print("configs scored:", len(T))
    for m in B.MEMBERS:
        print(f"{m:12s}", rel(T, sn, lambda d: d["members"].get(m)))
    print("mean4       ", rel(T, sn, lambda d: d.get("mean4")))
    print("blend       ", rel(T, sn, lambda d: d.get("blend")))
    ref = pd.read_csv(H.OFFICIAL / "leaderboard" / "Chronos-2__all_results.csv")      # column layout of the leaderboard files
    mp = {"eval_metrics/MSE[mean]": "MSE[mean]", "eval_metrics/MSE[0.5]": "MSE[0.5]", "eval_metrics/MAE[0.5]": "MAE[0.5]", "eval_metrics/MAPE[0.5]": "MAPE[0.5]",
          "eval_metrics/sMAPE[0.5]": "sMAPE[0.5]", "eval_metrics/MSIS": "MSIS", "eval_metrics/RMSE[mean]": "RMSE[mean]", "eval_metrics/NRMSE[mean]": "NRMSE[mean]",
          "eval_metrics/ND[0.5]": "ND[0.5]", "eval_metrics/MASE[0.5]": "MASE", "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS"}
    rows = []
    for _, row in ref.iterrows():
        r = row.copy()
        r["model"] = "FM-Blend"
        for col, k in mp.items():
            r[col] = T[row["dataset"]]["blend"][k]
        rows.append(r)
    pd.DataFrame(rows).to_csv(B.OUT / "all_results.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "test":
        stage_test(sys.argv[2:])
    elif cmd == "aggregate":
        aggregate()
    else:
        sys.exit(__doc__)
