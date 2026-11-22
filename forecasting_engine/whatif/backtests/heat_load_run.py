"""Scenario (what-if) test of the heat -> electricity demand mechanism on heat-wave windows (docs/methods.md). Per captured load_d cell (cascade.capture: Y truth, Rf strict reference s7, Pon engine, Poff links-off, 8 countries x 7 daily steps x
window origins): the curve (heat_load.Curve) is fitted on data before the first window origin - 28 days; at every origin it forecasts the 7 target days
(a) in SCENARIO mode, given the realised temperature path of the target days, (b) in FORECAST mode (climatology + decaying anomaly, no weather forecast).
Also scored: naive same-weekday-last-year rule (value 364 days earlier).
Usage: HEAT_CACHE=dir python -m forecasting_engine.whatif.backtests.heat_load_run [event ...]   (needs /tmp/casc_cap/<event>__load_d.npz)
Writes results/cascade/heat_load_scen.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import heat_load as H

CAP = "/tmp/casc_cap"
MOVE = 0.01          # a series "moved" if the mean realised excess over the window is >= 1 % of the pre-shock path


def origins(Y: np.ndarray, Ld: pd.DataFrame) -> list[pd.Timestamp]:
    """first target day of each origin, recovered from the captured truth"""
    out, k0 = [], 0
    for i in range(Y.shape[0]):
        k = next(k for k in range(k0, len(Ld) - 7) if np.allclose(Ld.iloc[k:k + 7].to_numpy(), Y[i].T, atol=1, equal_nan=True))
        out.append(Ld.index[k]); k0 = k
    return out


def frozen_base(Ld: pd.DataFrame, t0: list) -> np.ndarray:
    """pre-shock path (as in the benchmark design): the last week public at the first window origin repeated; (origins, series)"""
    last = t0[0] - pd.Timedelta(days=2)
    out = np.full((len(t0), len(H.CCS)), np.nan)
    for i, d in enumerate(t0):
        n = int(np.ceil(((d - last).days) / 7))
        out[i] = Ld.reindex([d - pd.Timedelta(days=7 * n)]).iloc[0][list(H.CCS)].to_numpy()
    return out


def path_stats(Y, P, base, A):
    """step-1 path over window origins per series: excess over the frozen pre-shock path; direction, size ratio (cumulative predicted / realised excess,
    series whose realised mean excess is >= MOVE of the base), peak-day offset (days between the day of the largest realised and predicted excess),
    share within +-5 / 10 / 20 %; pooled = all 8 countries together"""
    if not np.isfinite(P[:, A, 0]).any():
        return dict(moved=0)
    y, p, r = Y[:, A, 0], P[:, A, 0], base[:, A]
    ey, ep = y - r, p - r
    lvl = np.nanmean(np.abs(r), 0)
    mv = np.abs(np.nanmean(ey, 0)) >= MOVE * lvl
    pooled = float(np.nansum(ep) / np.nansum(ey))
    if not mv.any():
        return dict(moved=0, pooled_ratio=pooled)
    ratio = np.nansum(ep, 0)[mv] / np.nansum(ey, 0)[mv]
    pk = np.nanargmax(ep, 0)[mv] - np.nanargmax(ey, 0)[mv]
    return dict(moved=int(mv.sum()), dir=float(np.mean(np.sign(np.nansum(ep, 0)[mv]) == np.sign(np.nansum(ey, 0)[mv]))), med_ratio=float(np.median(ratio)),
                pooled_ratio=pooled, within5=float(np.mean(np.abs(ratio - 1) <= .05)), within10=float(np.mean(np.abs(ratio - 1) <= .10)),
                within20=float(np.mean(np.abs(ratio - 1) <= .20)), peak_off=float(np.median(np.abs(pk))), cum_real_pct=float(100 * np.mean(np.nansum(ey, 0)[mv] / np.nansum(r, 0)[mv])),
                all_dir=float(np.mean(np.sign(np.nansum(ep, 0)) == np.sign(np.nansum(ey, 0)))), real_pct_all=[round(float(100 * np.nansum(ey[:, k]) / np.nansum(r[:, k])), 2) for k in range(len(A))])


def peak_day(Y, P, A, dates):
    if not np.isfinite(P[:, A, 0]).any():
        return float("nan")
    """within the window of origins: day of the highest realised daily load (step 1) vs predicted, per series; median |offset| in days"""
    y, p = Y[:, A, 0], P[:, A, 0]
    return float(np.median(np.abs(np.nanargmax(p, 0) - np.nanargmax(y, 0))))


def skill(Y, P, Rf, sc, A):
    if not np.isfinite(P[:, A]).any():
        return float('nan')
    ok = np.isfinite(Y[:, A]) & np.isfinite(Rf[:, A])
    Pf = np.where(np.isfinite(P[:, A]), P[:, A], Rf[:, A])
    s = sc[A][None, :, None]
    return float(1 - (np.abs(Y[:, A] - Pf) / s)[ok].sum() / (np.abs(Y[:, A] - Rf[:, A]) / s)[ok].sum())


EVENTS_NO_CAPTURE = {"heatwave_2023": "2023-07-10", "heatwave_2017_jun": "2017-06-17", "heatwave_2017_aug": "2017-08-01"}   # 2017: the /tmp captures are gone; engine columns are carried over from the first-pass json     # no engine capture (the design run did not finish): truth and s7 reference rebuilt from the load, engine columns NaN


def synthetic_capture(ev: str, Ld: pd.DataFrame):
    on = pd.Timestamp(EVENTS_NO_CAPTURE[ev], tz="UTC")
    t0 = [on + pd.Timedelta(days=i + 1) for i in range(21)]
    Y = np.full((21, 8, 7), np.nan); Rf = np.full((21, 8, 7), np.nan)
    for i, d in enumerate(t0):
        for h in range(7):
            t = d + pd.Timedelta(days=h)
            n = int(np.ceil((t - (d - pd.Timedelta(days=2))).days / 7))
            Y[i, :, h] = Ld.loc[t].to_numpy(); Rf[i, :, h] = Ld.loc[t - pd.Timedelta(days=7 * n)].to_numpy()
    sc = np.array([np.nanmean(np.abs(np.diff(Ld[c].dropna().iloc[:int(.4 * Ld[c].notna().sum())].to_numpy()))) for c in H.CCS])
    return Y, Rf, sc, t0


def run_event(ev: str, inp: dict, cut_days: int = 28) -> dict:
    Ld = inp["load"].resample("D").mean()[list(H.CCS)]
    if ev in EVENTS_NO_CAPTURE:
        Y, Rf, sc, t0 = synthetic_capture(ev, Ld)
        z = dict(Pon=np.full(Y.shape, np.nan), Poff=np.full(Y.shape, np.nan), split="dev", onset=EVENTS_NO_CAPTURE[ev]); A = list(range(8))
    else:
        z = np.load(f"{CAP}/{ev}__load_d.npz", allow_pickle=True)
        Y, Rf, sc, A = z["Y"].astype(float), z["Rf"].astype(float), z["scale"], [int(a) for a in z["A"]]
        t0 = origins(Y, Ld)
    curves = H.fit_all(inp, t0[0] - pd.Timedelta(days=cut_days))
    for c in curves.values():
        c.anchor_cap = t0[0] - pd.Timedelta(days=2)      # used only by CFG.anchor == "frozen"
    out = {m: np.full(Y.shape, np.nan) for m in ("scen", "fcst", "naive364")}
    for i, d0 in enumerate(t0):
        tg = pd.date_range(d0, periods=7, freq="D", tz="UTC")
        for j, cc in enumerate(H.CCS):
            out["scen"][i, j] = H.daily_predict(curves[cc], inp, d0 - pd.Timedelta(days=2), tg, "scenario")
            if not os.environ.get("HEAT_SKIP_FCST"):
                out["fcst"][i, j] = H.daily_predict(curves[cc], inp, d0 - pd.Timedelta(days=2), tg, "forecast")
            out["naive364"][i, j] = Ld[cc].reindex(tg - pd.Timedelta(days=364)).to_numpy()
    base = frozen_base(Ld, t0)
    P = {"eng": z["Pon"].astype(float), "off": z["Poff"].astype(float), "ref": Rf, **out}
    row = dict(event=ev, split=str(z["split"]), onset=str(z["onset"]), n_origins=len(t0), fit_cutoff=str(t0[0] - pd.Timedelta(days=cut_days)),
               curves={cc: dict(th=float(c.th), tc=float(c.tc), heat_pct_per_deg=float(100 * c.sc), cold_pct_per_deg=float(100 * c.sh)) for cc, c in curves.items()})
    for m, p in P.items():
        row["skill_" + m] = skill(Y, p, Rf, sc, A)
        row["path_" + m] = path_stats(Y, p, base, A)
        row["peakday_" + m] = peak_day(Y, p, A, t0)
    row["per_country_skill_scen"] = {cc: skill(Y, P["scen"], Rf, sc, [j]) for j, cc in enumerate(H.CCS)}
    row["per_country_skill_eng"] = {cc: skill(Y, P["eng"], Rf, sc, [j]) for j, cc in enumerate(H.CCS)}
    np.savez_compressed(f"{CAP}/../heat_{ev}.npz", **{k: v for k, v in out.items()}, Y=Y, Rf=Rf, base=base)
    return row


def main(evs):
    inp = H.inputs()
    evs = evs or sorted(f.split("/")[-1].split("__")[0] for f in glob.glob(f"{CAP}/heatwave*__load_d.npz"))
    rows = [run_event(e, inp) for e in evs]
    for r in rows:
        print(r["event"], {k: round(v, 3) for k, v in r.items() if k.startswith("skill_")})
    old = {r["event"]: r for r in json.load(open("results/cascade/heat_load_scen.json"))} if os.path.exists("results/cascade/heat_load_scen.json") else {}
    for r in rows:     # engine / links-off / reference columns that could not be recomputed (no capture on disk) are kept from the previous file
        o = old.get(r["event"])
        if o and not np.isfinite(r["skill_eng"]) and np.isfinite(o.get("skill_eng", np.nan)):
            for k in ("eng", "off"):
                for pre in ("skill_", "path_", "peakday_"):
                    r[pre + k] = o[pre + k]
    json.dump(rows, open("results/cascade/heat_load_scen.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
