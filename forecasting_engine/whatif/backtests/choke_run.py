"""Cells and scores for the shipping-chokepoint what-if (docs/methods.md).

    python -m forecasting_engine.whatif.backtests.choke_run cells dev|check     -> results/cascade/choke_cells_<split>.json (inputs, realised, predictions, histories)
    HF_HUB_OFFLINE=1 python -m forecasting_engine.lab.choke_fm dev|check   -> choke_fm_<split>.json (Chronos-2 / TiRex-2 on the same cells)
    python -m forecasting_engine.whatif.backtests.choke_run score dev|check     -> choke_scores_<split>.json + printed table

Cell = event x link x series. Quantities are log changes against the origin level (the last value public on the onset date).
L1 weekly freight (WCI Shanghai-Rotterdam, WCI composite, SCFI composite); L2 monthly import prices from outside the euro area (consumer goods,
intermediate goods); L3 monthly HICP non-energy industrial goods (seasonal path removed). Each link is run in scenario mode: the previous link's REALISED
path is the input (transits for L1, freight for L2, consumer-goods import prices for L3). `chain` also runs L1 -> L2 -> L3 with predicted inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import choke_data as C
from forecasting_engine.whatif.backtests import choke_mech as M

OUT = Path(__file__).resolve().parents[3] / "results" / "cascade"
L2END = {"ever_given": "2021-12-01", "crunch_2021": "2022-04-01", "blacksea_2022": "2022-12-01", "panama_2023": "2023-11-01", "redsea": "2024-12-01"}
L3END = {"ever_given": "2022-06-01", "crunch_2021": "2022-12-01", "blacksea_2022": "2023-06-01", "redsea": "2025-06-01"}
L1WIN = {"redsea": (26, 52)}                                      # weeks: the 26-week ramp and the 52-week window are separate cells
HIST_W, HIST_M = 156, 96
MOVED = {"L1": 0.15, "L2": 0.015, "L3": 0.010}                    # |window-mean realised log change| for a cell to count as moved
DEV_FIT_START, DEV_FIT_END = pd.Timestamp("2010-02-01"), pd.Timestamp("2023-11-01")


# ------------------------------------------------------------------------------------------------------------------------ weekly freight
def weekly_series(key: str) -> pd.Series:
    """ln freight index by week (Monday stamp of the week of the print); cleaned of parse outliers by a 5-point rolling median."""
    if key == "scfi":
        s = C.scfi()
    else:
        w = C.wci_series()
        s = w["wci"] if key == "wci" else w["sha_rtm"].where((w["sha_rtm"] / w["wci"] < 2.5) & (w["sha_rtm"] / w["wci"] > 0.4))
    s = s.dropna()
    s.index = s.index.normalize() - pd.to_timedelta(s.index.weekday, unit="D")
    s = np.log(s[~s.index.duplicated()])
    s = s.asfreq("W-MON")
    med = s.rolling(5, center=True, min_periods=3).median()
    return s.where((s - med).abs() < 0.30)


def l1_cells(ev: str) -> list:
    e = M.EVENTS[ev]
    onset, end = pd.Timestamp(e["onset"]), pd.Timestamp(e["end"])
    out = []
    wins = L1WIN.get(ev, (None,))
    for key in ("wci_rtm", "wci", "scfi"):
        s = weekly_series("wci" if key == "wci" else ("wci_rtm" if key == "wci_rtm" else "scfi"))
        pre = s[s.index < onset].dropna()
        if len(pre) < 2 or (onset - pre.index[-1]).days > 56:
            continue
        r0 = float(pre.iloc[-2:].mean())
        for nw in wins:
            weeks = pd.date_range(onset, onset + pd.Timedelta(weeks=nw - 1), freq="W-MON") if nw else pd.date_range(onset, end, freq="W-MON")
            y = (s.reindex(weeks) - r0).to_numpy()
            if np.isfinite(y).sum() < 6:
                continue
            hist = s[s.index < onset].dropna().tail(HIST_W)
            histf = hist.reindex(pd.date_range(hist.index[0], onset - pd.Timedelta(weeks=1), freq="W-MON")).interpolate(limit=6).ffill()
            out.append(dict(event=ev, link="L1", series=key, weeks=[str(w.date()) for w in weeks], real=[None if not np.isfinite(v) else float(v) for v in y],
                            pred=[float(v) for v in M.l1_path(ev, key, weeks)], off=[0.0] * len(weeks), hist=[float(v) for v in histf.to_numpy()],
                            ref=r0, win=nw or len(weeks)))
    return out


# ------------------------------------------------------------------------------------------------------------------------ monthly
def _conv(x: pd.Series, w: np.ndarray) -> pd.Series:
    return sum(w[k] * x.shift(k) for k in range(len(w)))


def fit_l2(P: pd.DataFrame, target: str, excl: list) -> dict:
    d = P[["lnF", "fx", target]].diff().loc[DEV_FIT_START:DEV_FIT_END]
    keep = pd.Series(True, index=d.index)
    for a, b in excl:
        keep &= ~((d.index >= a) & (d.index <= b))
    best = None
    for tau in (1.0, 2.0, 3.0, 4.0):
        x = _conv(P["lnF"].diff(), M.lag_weights(tau, M.LAG_L2)).loc[d.index]
        X = np.column_stack([x, d["fx"], np.ones(len(d))])[keep.to_numpy()]
        y = d[target].to_numpy()[keep.to_numpy()]
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        b, res, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        e = y[ok] - X[ok] @ b
        s2 = e @ e / (ok.sum() - 3)
        se = np.sqrt(s2 * np.linalg.inv(X[ok].T @ X[ok])[0, 0])
        if best is None or e @ e < best["sse"]:
            best = dict(eps=float(b[0]), se=float(se), tau=tau, fx=float(b[1]), n=int(ok.sum()), sse=float(e @ e))
    return best


def fit_l3(P: pd.DataFrame, excl: list) -> dict:
    h = P["hicp"].diff()
    d = pd.DataFrame({"h": h, "p": P["imp_consumer"].diff()}).loc[DEV_FIT_START:DEV_FIT_END]
    keep = pd.Series(True, index=d.index)
    for a, b in excl:
        keep &= ~((d.index >= a) & (d.index <= b))
    best = None
    for mean in (4.0, 6.0, 8.0, 10.0):
        x = _conv(P["imp_consumer"].diff(), M.gamma_weights(mean, M.LAG_L3)).loc[d.index]
        mon = pd.get_dummies(d.index.month).astype(float).to_numpy()
        X = np.column_stack([x, mon])[keep.to_numpy()]
        y = d["h"].to_numpy()[keep.to_numpy()]
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        e = y[ok] - X[ok] @ b
        s2 = e @ e / (ok.sum() - X.shape[1])
        se = np.sqrt(s2 * np.linalg.inv(X[ok].T @ X[ok])[0, 0])
        if best is None or e @ e < best["sse"]:
            best = dict(eta=float(b[0]), se=float(se), mean=mean, n=int(ok.sum()), sse=float(e @ e))
    return best


def excl_of(ev: str) -> list:
    on = pd.Timestamp(M.EVENTS[ev]["onset"]).to_period("M").to_timestamp()
    return [(on - pd.DateOffset(months=3), pd.Timestamp(L3END.get(ev, L2END[ev])) + pd.DateOffset(months=12))]


def clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def monthly_cells(ev: str, P: pd.DataFrame, split: str) -> list:
    e = M.EVENTS[ev]
    onset = pd.Timestamp(e["onset"])
    on_m = onset.to_period("M").to_timestamp()
    out = []
    f0 = float(P["lnF"].loc[on_m - pd.DateOffset(months=3):on_m - pd.DateOffset(months=1)].mean())
    o2 = M.origin_month(onset, M.LAG_IMP_D)
    months = pd.date_range(o2 + pd.DateOffset(months=1), L2END[ev], freq="MS")
    for tgt in ("imp_consumer", "imp_interm"):
        h = P[tgt].loc[:o2].dropna().tail(HIST_M)
        p0 = float(h.iloc[-1])
        y = (P[tgt].reindex(months) - p0).to_numpy()
        lnf = P["lnF"].reindex(months).to_numpy()
        if not np.isfinite(lnf).all() or not np.isfinite(y).any() or np.isnan(f0):
            continue
        ex = excl_of(ev) if split == "dev" else []
        fit = fit_l2(P, tgt, ex)
        eps2 = clip(fit["eps"], 0.0, 0.3)
        out.append(dict(event=ev, link="L2", series=tgt, weeks=[str(m.date()) for m in months], real=[None if not np.isfinite(v) else float(v) for v in y],
                        pred=[float(v) for v in M.l2_path(lnf, f0)], pred2=[float(v) for v in M.l2_path(lnf, f0, eps2, fit["tau"])],
                        off=[0.0] * len(months), hist=[float(v) for v in h.to_numpy()], ref=p0, fit=fit, origin=str(o2.date()), f0=f0,
                        input=[float(v) for v in lnf], win=len(months)))
    if ev in L3END:
        o3 = M.origin_month(onset, M.LAG_HICP_D)
        months = pd.date_range(o3 + pd.DateOffset(months=1), L3END[ev], freq="MS")
        hh = P["hicp"].loc[:o3].dropna().tail(HIST_M)
        h0 = float(hh.iloc[-1])
        sea = M.seasonal_path(P["hicp"].dropna(), o3, months)
        y = (P["hicp"].reindex(months) - h0).to_numpy() - sea
        p0 = float(P["imp_consumer"].loc[o3])
        lnp = P["imp_consumer"].reindex(months).to_numpy()
        if np.isfinite(lnp).all():
            ex = excl_of(ev) if split == "dev" else []
            fit = fit_l3(P, ex)
            eta2 = clip(fit["eta"], 0.0, 0.6)
            out.append(dict(event=ev, link="L3", series="hicp_nrg_goods", weeks=[str(m.date()) for m in months], real=[None if not np.isfinite(v) else float(v) for v in y],
                            pred=[float(v) for v in M.l3_path(lnp, p0)], pred2=[float(v) for v in M.l3_path(lnp, p0, eta2, fit["mean"])],
                            off=[0.0] * len(months), hist=[float(v) for v in hh.to_numpy()], ref=h0, fit=fit, origin=str(o3.date()), seas=[float(v) for v in sea],
                            input=[float(v) for v in (lnp - p0)], win=len(months)))
    return out


def chain_cells(ev: str, P: pd.DataFrame) -> dict:
    """End-to-end: transits -> L1 (composite WCI) -> L2 (consumer-goods import prices) -> L3 (HICP goods), predicted inputs all the way. Monthly means of the L1 weekly path."""
    e = M.EVENTS[ev]
    onset = pd.Timestamp(e["onset"])
    weeks = pd.date_range(onset - pd.Timedelta(weeks=12), pd.Timestamp(L3END.get(ev, L2END[ev])) + pd.offsets.MonthEnd(0), freq="W-MON")
    p1 = np.zeros(len(weeks))
    on = weeks >= onset
    p1[on] = M.l1_path(ev, "wci", weeks[on])
    mf = pd.Series(p1, index=weeks).resample("MS").mean()
    on_m = onset.to_period("M").to_timestamp()
    f0 = float(P["lnF"].loc[on_m - pd.DateOffset(months=3):on_m - pd.DateOffset(months=1)].mean())
    lnf_pred = f0 + mf                                                    # predicted ln freight index by month (composite, WCI-scale changes)
    o3 = M.origin_month(onset, M.LAG_HICP_D)
    months = pd.date_range(o3 + pd.DateOffset(months=1), L3END.get(ev, L2END[ev]), freq="MS")
    pp = M.l2_path(lnf_pred.reindex(months).ffill().to_numpy(), f0)
    ph = M.l3_path(pp, 0.0)
    return dict(months=[str(m.date()) for m in months], freight=[float(v) for v in mf.reindex(months).to_numpy()], import_price=[float(v) for v in pp], hicp=[float(v) for v in ph])


def build(split: str) -> list:
    P = M.monthly_panel()
    cells = []
    for ev, e in M.EVENTS.items():
        if e["split"] != split:
            continue
        cells += l1_cells(ev) + monthly_cells(ev, P, split)
    return cells


# ------------------------------------------------------------------------------------------------------------------------ scoring
def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[ok] - b[ok]).mean()) if ok.any() else float("nan")


def cell_metrics(c: dict, key: str = "pred", fm: dict | None = None) -> dict:
    y, p = np.array([np.nan if v is None else v for v in c["real"]], float), np.array(c[key], float)
    ok = np.isfinite(y)
    mr, mp = float(np.nanmean(y)), float(np.mean(p[ok]))
    moved = abs(mr) >= MOVED[c["link"]]
    d = dict(event=c["event"], link=c["link"], series=c["series"], win=c["win"], real_mean=mr, pred_mean=mp, moved=bool(moved),
             mae=mae(y, p), mae_off=mae(y, c["off"]))
    if fm:
        for k in ("chronos2", "tirex2"):
            d["mae_" + k] = mae(y, fm[k][:len(y)])
    d["sd_placebo"] = placebo_sd(c["link"], c["series"] if c["link"] != "L3" else "hicp", c["win"] if c["link"] == "L1" else len(c["real"]))
    d["detectable"] = bool(abs(mr) > 2 * d["sd_placebo"])
    d["dir_ok"] = bool(np.sign(mr) == np.sign(mp)) if moved else None
    d["size"] = float(mp / mr) if moved and mr != 0 else None
    if moved:
        sm = lambda v: pd.Series(v).rolling(3, center=True, min_periods=1).mean().to_numpy()       # noqa: E731
        yi = np.where(ok, y, np.nan)
        ia = int(np.nanargmax(sm(yi) * np.sign(mr))) if np.isfinite(yi).any() else 0
        ip = int(np.argmax(sm(p) * np.sign(mr)))
        d["peak_real"], d["peak_pred"], d["timing"] = ia, ip, ip - ia
    return d


_PLC: dict = {}


def placebo_sd(link: str, series: str, n: int) -> float:
    """SD of the window-mean log change against the origin level over placebo windows of the same length in 2012-2019 (no event): the noise floor of a cell."""
    k = (link, series, n)
    if k in _PLC:
        return _PLC[k]
    if link == "L1":
        s = weekly_series("scfi").interpolate(limit=4)
        s = s.loc["2012-01-01":"2019-12-31"]
        v = [float(s.iloc[i + 1:i + 1 + n].mean() - s.iloc[i - 1:i + 1].mean()) for i in range(2, len(s) - n - 1)]
    else:
        P = M.monthly_panel()
        if link == "L2":
            x = P[series].loc["2012-01-01":"2019-12-31"]
            v = [float(x.iloc[i + 1:i + 1 + n].mean() - x.iloc[i]) for i in range(len(x) - n - 1)]
        else:
            h = P["hicp"].dropna()
            v = []
            for o in pd.date_range("2013-01-01", "2019-12-01", freq="MS")[: -n - 1]:
                months = pd.date_range(o + pd.DateOffset(months=1), periods=n, freq="MS")
                sea = M.seasonal_path(h, o, months)
                v.append(float(np.mean(h.reindex(months).to_numpy() - h.loc[o] - sea)))
    _PLC[k] = float(np.std(v))
    return _PLC[k]


def pooled(rows: list, fm_keys=("chronos2", "tirex2")) -> dict:
    out = {}
    if not rows:
        return out
    m, o = np.nanmean([r["mae"] for r in rows]), np.nanmean([r["mae_off"] for r in rows])
    out.update(n=len(rows), mae=float(m), mae_off=float(o), skill_off=float(1 - m / o) if o > 0 else None)
    fms = {k: float(np.nanmean([r.get("mae_" + k, np.nan) for r in rows])) for k in fm_keys}
    if all(np.isfinite(v) for v in fms.values()) and fms:
        b = min(fms.values())
        out.update(fm=fms, skill_fm=float(1 - m / b))
    mv = [r for r in rows if r["moved"]]
    out["moved"] = len(mv)
    out["detectable"] = int(sum(r["detectable"] for r in rows))
    if mv:
        out["dir"] = float(np.mean([r["dir_ok"] for r in mv]))
        out["size_median"] = float(np.median([r["size"] for r in mv]))
        out["timing_median_abs"] = float(np.median([abs(r["timing"]) for r in mv]))
    return out


def main(a):
    cmd, split = a[0], a[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "cells":
        cells = build(split)
        (OUT / f"choke_cells_{split}.json").write_text(json.dumps(cells))
        print(split, len(cells), "cells", {(c["event"], c["link"], c["series"], c["win"]) for c in cells})
    elif cmd == "score":
        cells = json.loads((OUT / f"choke_cells_{split}.json").read_text())
        fmp = OUT / f"choke_fm_{split}.json"
        fms = {(f["event"], f["link"], f["series"], f["win"]): f for f in json.loads(fmp.read_text())} if fmp.exists() else {}
        res = {"cells": [], "pooled": {}}
        for c in cells:
            fm = fms.get((c["event"], c["link"], c["series"], c["win"]))
            for key, tag in (("pred", "M1"), ("pred2", "M2")):
                if key in c:
                    d = cell_metrics(c, key, fm)
                    d["model"] = tag
                    if "fit" in c:
                        d["fit"] = {k: v for k, v in c["fit"].items() if k != "sse"}
                    res["cells"].append(d)
        for link in ("L1", "L2", "L3"):
            for tag in ("M1", "M2"):
                rows = [r for r in res["cells"] if r["link"] == link and r["model"] == tag]
                if rows:
                    res["pooled"][f"{link}_{tag}"] = pooled(rows)
        (OUT / f"choke_scores_{split}.json").write_text(json.dumps(res, indent=1))
        for r in res["cells"]:
            print(f"{r['event']:14s} {r['link']} {r['series']:15s} {r['model']} win {r['win']:3d} real {r['real_mean']:+.3f} pred {r['pred_mean']:+.3f} moved {r['moved']!s:5s}"
                  f" mae {r['mae']:.3f} off {r['mae_off']:.3f} c2 {r.get('mae_chronos2', float('nan')):.3f} tx {r.get('mae_tirex2', float('nan')):.3f}"
                  f" sd {r['sd_placebo']:.3f} det {r['detectable']}" f" dir {r['dir_ok']} size {r['size'] if r['size'] is None else round(r['size'], 2)} timing {r.get('timing')}")
        for k, v in res["pooled"].items():
            print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    elif cmd == "chain":
        P = M.monthly_panel()
        res = {}
        for ev, e in M.EVENTS.items():
            if e["split"] != split or ev not in L3END:
                continue
            ch = chain_cells(ev, P)
            months = pd.DatetimeIndex(ch["months"])
            onset = pd.Timestamp(e["onset"])
            on_m = onset.to_period("M").to_timestamp()
            f0 = float(P["lnF"].loc[on_m - pd.DateOffset(months=3):on_m - pd.DateOffset(months=1)].mean())
            o3 = M.origin_month(onset, M.LAG_HICP_D)
            sea = M.seasonal_path(P["hicp"].dropna(), o3, months)
            ch["real_freight"] = [None if not np.isfinite(v) else float(v) for v in (P["lnF"].reindex(months) - f0).to_numpy()]
            ch["real_import_price"] = [float(v) for v in (P["imp_consumer"].reindex(months) - P["imp_consumer"].loc[o3]).to_numpy()]
            ch["real_hicp"] = [float(v) for v in (P["hicp"].reindex(months) - P["hicp"].loc[o3]).to_numpy() - sea]
            ch["mean"] = {k: float(np.nanmean([v for v in ch[k] if v is not None])) for k in ("freight", "import_price", "hicp", "real_freight", "real_import_price", "real_hicp")}
            res[ev] = ch
            print(ev, {k: round(v, 4) for k, v in ch["mean"].items()})
        (OUT / f"choke_chain_{split}.json").write_text(json.dumps(res))
    elif cmd == "fits":
        P = M.monthly_panel()
        for tgt in ("imp_consumer", "imp_interm"):
            print(tgt, fit_l2(P, tgt, []))
        print("hicp", fit_l3(P, []))


if __name__ == "__main__":
    main(sys.argv[1:])
