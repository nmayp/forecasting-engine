"""Held-out check of the heat-load mechanism on heat waves of 2024-2025 (spec frozen in results/cascade/heat_load_holdout_spec.json before scoring).
Usage: HEAT_CACHE=dir python -m forecasting_engine.whatif.backtests.heat_load_holdout   -> results/cascade/heat_load_holdout.json (one scoring run)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import heat_load as H
from forecasting_engine.whatif.backtests import heat_load_run as R

import os
EV = {"heatwave_2023": "2023-07-10"} if os.environ.get("HOLDOUT_TEST") else {"heatwave_2024_jul": "2024-07-23", "heatwave_2025_jun": "2025-06-18", "heatwave_2025_aug": "2025-08-03"}
MOVE = R.MOVE


class OffCurve(H.Curve):
    """temperature-blind ablation: same calendar/season/trend/dow/holiday terms, no heating or cooling term (refit)"""
    def fit(self, y, T, cutoff, Tx=None):
        ly = np.log(y.where(y > 0))
        m = (ly.index < cutoff) & (ly.index >= H.TRAIN_FROM) & ly.notna().to_numpy()
        m &= ~((ly.index >= H.COVID[0]) & (ly.index <= H.COVID[1]))
        idx = ly.index[m]
        self.t0, self.cut = idx[0], cutoff
        X, nm = H.design(idx, np.zeros(len(idx)), self.cc, self.t0, None, None)
        self.b, _ = H._fit(X, ly[m].to_numpy(), len(nm))
        self.th = self.tc = 0.0
        return self

    def predict_log(self, idx, Te):
        X, _ = H.design(idx, np.zeros(len(idx)), self.cc, self.t0, None, None)
        return X @ self.b

    def anchor(self, y_log, T, origin_day, Tx=None):
        sel = (y_log.index <= origin_day) & (y_log.index > origin_day - pd.Timedelta(days=int(H.CFG["anc"])))
        idx = y_log.index[sel]
        idx = idx[y_log.reindex(idx).notna().to_numpy()]
        if len(idx) < 7:
            return 0.0
        return float(np.mean(y_log.reindex(idx).to_numpy() - self.predict_log(idx, None)))


def off_predict(cv, inp, d, tg):
    Ld = inp["load"][cv.cc].resample("D").mean()
    return np.exp(cv.predict_log(tg, None) + cv.anchor(np.log(Ld.where(Ld > 0)), None, d))


def open_models(Ld, t0):
    import torch
    from forecasting_engine.lab import bench as B
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(B.model_src("amazon/chronos-2"), device_map="cpu", torch_dtype=torch.float32)
    tx = B._tirex()
    from tirex2 import TimeseriesType
    out = {m: np.full((len(t0), 8, 7), np.nan) for m in ("chronos2", "tirex2")}
    for i, d0 in enumerate(t0):
        last = d0 - pd.Timedelta(days=2)
        hs = [Ld[c].loc[:last].dropna().iloc[-512:].to_numpy(np.float32) for c in H.CCS]
        q, _ = pipe.predict_quantiles([dict(target=h) for h in hs], prediction_length=8, quantile_levels=[0.5], cross_learning=False, batch_size=8)
        for j in range(8):
            out["chronos2"][i, j] = q[j][0, 1:8, 0].cpu().numpy()
        fo = tx.forecast([TimeseriesType(target=torch.tensor(h[None, :]), past_covariates=None, future_covariates=None) for h in hs], prediction_length=8, output_type="numpy", batch_size=8)
        for j in range(8):
            out["tirex2"][i, j] = fo[j][0, 4, 1:8]
    return out


def series_stats(Y, P, base):
    """per country, step 1: cumulative excess ratio, sign, peak offsets; only movers"""
    y, p, r = Y[:, :, 0], P[:, :, 0], base
    ey, ep = y - r, p - r
    lvl = np.nanmean(np.abs(r), 0)
    mv = np.abs(np.nanmean(ey, 0)) >= MOVE * lvl
    rows = []
    for k in range(y.shape[1]):
        if not mv[k]:
            continue
        rows.append(dict(cc=H.CCS[k], ratio=float(np.nansum(ep[:, k]) / np.nansum(ey[:, k])), dir=bool(np.sign(np.nansum(ep[:, k])) == np.sign(np.nansum(ey[:, k]))),
                         peak_off=int(abs(np.nanargmax(ep[:, k]) - np.nanargmax(ey[:, k]))), load_peak_off=int(abs(np.nanargmax(p[:, k]) - np.nanargmax(y[:, k]))),
                         real_pct=float(100 * np.nansum(ey[:, k]) / np.nansum(r[:, k]))))
    return rows


def main():
    inp = H.inputs()
    Ld = inp["load"].resample("D").mean()[list(H.CCS)]
    res, allrows = [], {}
    for ev, on in R.EVENTS_NO_CAPTURE.items() if False else EV.items():
        R.EVENTS_NO_CAPTURE[ev] = on
        Y, Rf, sc, t0 = R.synthetic_capture(ev, Ld)
        A = list(range(8))
        curves = H.fit_all(inp, t0[0] - pd.Timedelta(days=28))
        offs = {cc: OffCurve(cc).fit(Ld[cc], None, t0[0] - pd.Timedelta(days=28)) for cc in H.CCS}
        P = {m: np.full(Y.shape, np.nan) for m in ("scen", "off", "naive364")}
        for i, d0 in enumerate(t0):
            tg = pd.date_range(d0, periods=7, freq="D", tz="UTC")
            for j, cc in enumerate(H.CCS):
                P["scen"][i, j] = H.daily_predict(curves[cc], inp, d0 - pd.Timedelta(days=2), tg, "scenario")
                P["off"][i, j] = off_predict(offs[cc], inp, d0 - pd.Timedelta(days=2), tg)
                P["naive364"][i, j] = Ld[cc].reindex(tg - pd.Timedelta(days=364)).to_numpy()
        P.update(open_models(Ld, t0))
        base = R.frozen_base(Ld, t0)
        row = dict(event=ev, onset=on, n_origins=len(t0), nan_truth=int(np.isnan(Y).sum()), curves={cc: dict(th=float(c.th), tc=float(c.tc), cool_pct=float(100 * c.sc)) for cc, c in curves.items()})
        for m, p in P.items():
            row["skill_" + m] = R.skill(Y, p, Rf, sc, A)
        row["per_country_skill_scen"] = {cc: R.skill(Y, P["scen"], Rf, sc, [j]) for j, cc in enumerate(H.CCS)}
        row["series"] = series_stats(Y, P["scen"], base)
        row["series_off"] = series_stats(Y, P["off"], base)
        res.append(row)
        print(ev, {k: round(v, 3) for k, v in row.items() if k.startswith("skill_")}, flush=True)
    json.dump(res, open(os.environ.get("HOLDOUT_OUT", "results/cascade/heat_load_holdout.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
