"""Event test of the interest-rate mechanism (rates_credit.py): builds forecasts at every window origin, scores them (docs/methods.md,
"Mechanism: interest rates to lending and investment").

    python -m forecasting_engine.whatif.backtests.rates_credit_run build  <event|dev>     # mechanism + baselines -> results/cascade/rates_credit/<event>.pkl
    python -m forecasting_engine.whatif.backtests.rates_credit_run fm     <event> <model>  # chronos2 | tirex2 | toto2 zero-shot -> .../<event>_<model>.pkl
    python -m forecasting_engine.whatif.backtests.rates_credit_run score  <dev|check|all>  # -> results/cascade/rates_credit_scores.json

Cell = event x target (9 targets). Origins: the first day of each month from onset-3 to onset+9 (13), steps 0..11 months from the origin month (quarters 0..3 for
investment), 11 countries (+ the euro-area aggregate U2, reported apart). Information at an origin: only what was public (data lags in rates_credit_data.LAG);
the realised policy path from the origin on is the given what-if input.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.whatif.backtests import rates_credit as RC

OUT = Path(__file__).resolve().parents[3] / "results" / "cascade" / "rates_credit"
EVENTS = {"cuts_2008": ("2008-10-01", "dev"), "hikes_2011": ("2011-04-01", "dev"), "negative_2014": ("2014-06-01", "dev"),
          "hikes_2022": ("2022-07-01", "dev"), "cuts_2024": ("2024-06-01", "check"),
          "hikes_2026": ("2026-06-01", "unseen")}      # unseen: the ECB policy path rises from June 2026 (asym / pooled work, 24 Oct 2026); truth only to 2026-08
PRE, POST = 3, 9
FM_USE = ("chronos2", "tirex2", "toto2")
GEOS = RC.GEOS
CTRY = [g for g in GEOS if g != "U2"]
TG = list(RC.TARGETS)


def origins(onset: str):
    o = pd.Timestamp(onset)
    return [x for x in (o + pd.DateOffset(months=k) for k in range(-PRE, POST + 1)) if x <= pd.Timestamp('2026-10-01')]     # origins in the future have no truth


def tmonths(o: pd.Timestamp, tgt: str):
    per = RC.TARGETS[tgt][2]
    if per == "Q":
        return pd.date_range(o.to_period("Q").to_timestamp(), periods=RC.HQ, freq="QS")
    return pd.date_range(o.to_period("M").to_timestamp(), periods=RC.H, freq="MS")


def full_space(S: RC.Store, tgt: str) -> pd.DataFrame:
    kind, _, per = RC.TARGETS[tgt]
    return RC.to_target(S.W[tgt], kind, 3 if per == "Q" else 1)


def known_space(S: RC.Store, tgt: str, o) -> pd.DataFrame:
    kind, _, per = RC.TARGETS[tgt]
    return RC.to_target(S.known(tgt, o), kind, 3 if per == "Q" else 1)


def lastval(df: pd.DataFrame, g):
    s = df[g].dropna() if g in df else pd.Series(dtype=float)
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, np.nan)


def place(arr, i, tg, df, o, tgt):
    for j, g in enumerate(GEOS):
        if df is not None and g in df:
            arr[i, j, :] = df[g].reindex(tmonths(o, tgt)).to_numpy()


def build(event: str):
    onset, split = EVENTS[event]
    S = RC.Store()
    os_ = origins(onset)
    nO, nG = len(os_), len(GEOS)
    res = dict(event=event, onset=onset, origins=[str(o.date()) for o in os_], geos=GEOS)
    F = {m: {t: np.full((nO, nG, RC.HQ if RC.TARGETS[t][2] == "Q" else RC.H), np.nan) for t in TG} for m in ("mech", "held", "off")}
    T = {t: np.full_like(F["mech"][t], np.nan) for t in TG}
    N, SNL = {t: np.full_like(T[t], np.nan) for t in TG}, {t: np.full_like(T[t], np.nan) for t in TG}
    coef0, theta0 = None, {}
    for i, o in enumerate(os_):
        s1 = {k: RC.stage1(S, k, o) for k in ("ccob_nfc", "ccob_hh")}
        if i == 0:
            coef0 = {k: s1[k][1] for k in s1}
        for t in TG:
            kind, dk, per = RC.TARGETS[t]
            full, kn = full_space(S, t), known_space(S, t, o)
            tm = tmonths(o, t)
            for j, g in enumerate(GEOS):
                if g in full:
                    T[t][i, j, :] = full[g].reindex(tm).to_numpy()
                L, v = lastval(kn, g)
                N[t][i, j, :] = v
                step = pd.DateOffset(months=3 if per == "Q" else 12 if False else 12)
                for h, m in enumerate(tm):
                    sm = m - (pd.DateOffset(months=12))
                    SNL[t][i, j, h] = full[g].get(sm, np.nan) if (g in full and L is not None and sm <= L) else v
            if kind == "lvl":
                for m in F:
                    if t in s1 and m in s1[t][0]:
                        place(F[m][t], i, None, s1[t][0][m], o, t)
                continue
            for m in ("mech", "held", "off"):
                rp = {k: s1[k][0][m + ("_sim" if RC.CFG["feat"] == "policy" else "")] for k in s1 if m in s1[k][0]} if m != "off" else {}
                df, th = RC.stage2(S, t, o, {k: {g: v[g] for g in v} for k, v in rp.items()}, use_rates=(m != "off"))
                if df is not None:
                    place(F[m][t], i, None, df, o, t)
                    if i == 0 and m == "mech":
                        theta0[t] = th
        print(event, "origin", i + 1, "/", nO, flush=True)
    res.update(F=F, T=T, N=N, SNL=SNL, coef0=coef0, theta0=theta0)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"{event}.pkl", "wb"))
    return res


# ------------------------------------------------------------------------------------------------------------------------ zero-shot models
def hist_arrays(S, t, o):
    kn = known_space(S, t, o)
    out = {}
    for g in GEOS:
        if g in kn:
            s = kn[g].dropna()
            if len(s) >= 24:
                s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="QS" if RC.TARGETS[t][2] == "Q" else "MS")).ffill()
                out[g] = s
    return out


def fm(event: str, model: str):
    from forecasting_engine.lab import bench as B
    import torch
    onset, _ = EVENTS[event]
    S = RC.Store()
    os_ = origins(onset)
    P = {t: np.full((len(os_), len(GEOS), RC.HQ if RC.TARGETS[t][2] == "Q" else RC.H), np.nan) for t in TG}
    if model == "chronos2":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(B.model_src("amazon/chronos-2"), device_map="cpu", torch_dtype=torch.float32)
    elif model == "tirex2":
        from tirex2 import TimeseriesType
        m = B._tirex()
    else:
        m = B._toto()
        med_i = list(m.output_head.knots).index(0.5)
    for i, o in enumerate(os_):
        for t in TG:
            per = RC.TARGETS[t][2]
            hs = hist_arrays(S, t, o)
            tm = tmonths(o, t)
            gl = list(hs)
            if not gl:
                continue
            unit = 3 if per == "Q" else 1
            steps = {g: (tm[-1].year - hs[g].index[-1].year) * 12 + tm[-1].month - hs[g].index[-1].month for g in gl}
            steps = {g: s // unit for g, s in steps.items()}
            if model == "chronos2":
                ST = max(steps.values())
                q, _ = pipe.predict_quantiles([dict(target=hs[g].to_numpy(np.float32)[-512:]) for g in gl], prediction_length=ST, quantile_levels=[0.5],
                                              cross_learning=False, batch_size=64)
                med = {g: q[r][0, :, 0].cpu().numpy() for r, g in enumerate(gl)}
            elif model == "tirex2":
                ST = max(steps.values())
                items = [TimeseriesType(target=torch.tensor(hs[g].to_numpy(np.float32)[-512:][None, :]), past_covariates=None, future_covariates=None) for g in gl]
                fo = m.forecast(items, prediction_length=ST, output_type="numpy", batch_size=64)
                med = {g: fo[r][0, 4, :] for r, g in enumerate(gl)}
            else:
                ST, ctx = max(steps.values()), 512
                X, Mk = np.zeros((len(gl), ctx), np.float32), np.zeros((len(gl), ctx), bool)
                for r, g in enumerate(gl):
                    y = hs[g].to_numpy(np.float32)[-ctx:]
                    X[r, ctx - len(y):], Mk[r, ctx - len(y):] = y, True
                with torch.no_grad():
                    qq = m.forecast({"target": torch.tensor(X[None]), "target_mask": torch.tensor(Mk[None]),
                                     "series_ids": torch.zeros((1, len(gl)), dtype=torch.long)}, horizon=ST, decode_block_size=None, has_missing_values=True)
                arr = qq[med_i].float().cpu().numpy()[0]
                med = {g: arr[r] for r, g in enumerate(gl)}
            for j, g in enumerate(GEOS):
                if g not in med:
                    continue
                last = hs[g].index[-1]
                for h, mm in enumerate(tm):
                    k = ((mm.year - last.year) * 12 + mm.month - last.month) // unit - 1
                    if 0 <= k < len(med[g]):
                        P[t][i, j, h] = med[g][k]
        print(event, model, "origin", i + 1, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(P, open(OUT / f"{event}_{model}.pkl", "wb"))


# ------------------------------------------------------------------------------------------------------------------------------- scoring
def scale_of(S, t):
    full = full_space(S, t)
    sc = {}
    for g in full:
        s = full[g].dropna()
        k = max(int(len(s) * 0.4), 12)
        d = s.iloc[:k].diff().abs().mean()
        sc[g] = float(d) if np.isfinite(d) and d > 1e-9 else float(s.diff().abs().mean() or 1.0)
    return sc


def moved_flags(S, t, event_res):
    """Placebo test: the window-mean deviation from the frozen reference against the same statistic on windows 25-60 months earlier."""
    full = full_space(S, t)
    os_ = [pd.Timestamp(x) for x in event_res["origins"]]
    per = RC.TARGETS[t][2]
    f0, f1 = os_[0].to_period("M").to_timestamp(), os_[-1].to_period("M").to_timestamp() + pd.DateOffset(months=11)
    out = {}
    for j, g in enumerate(GEOS):
        if g not in full:
            continue
        z = full[g].dropna()
        ref = event_res["N"][t][0, j, 0]

        def stat(a, b, r):
            w = z[(z.index >= a) & (z.index <= b)]
            return abs(float((w - r).mean())) if len(w) >= (6 if per == "M" else 2) else np.nan
        s0 = stat(f0, f1, ref)
        pl = []
        for sft in range(25, 61):
            a, b = f0 - pd.DateOffset(months=sft), f1 - pd.DateOffset(months=sft)
            r = z[z.index <= a - pd.DateOffset(months=3)]
            if len(r) and a >= z.index[0]:
                pl.append(stat(a, b, float(r.iloc[-1])))
        pl = [x for x in pl if np.isfinite(x)]
        out[g] = bool(np.isfinite(s0) and len(pl) >= 10 and s0 > np.percentile(pl, 90))
    return out


def tfirst(path, kind):
    """t50: first index where |path| reaches half its maximum; 'inc': index of the largest one-step change in the direction of the final deviation."""
    p = np.asarray(path, float)
    if np.isfinite(p).sum() < 6:
        return np.nan
    p = pd.Series(p).interpolate(limit_direction="both").to_numpy()
    if kind == "t50":
        a = np.abs(p - p[0])
        return float(np.argmax(a >= 0.5 * a.max())) if a.max() > 0 else np.nan
    sgn = np.sign(p[-1] - p[0]) or 1.0
    d = np.diff(p) * sgn
    return float(np.argmax(d))


def cell_metrics(S, t, ev, models, scale, moved):
    """models: {name: array (O,G,H)}; truth ev['T'][t]; naive ev['N'][t]. Returns {model: metrics}."""
    Tt, Nn = ev["T"][t], ev["N"][t]
    cidx = [j for j, g in enumerate(GEOS) if g != "U2"]
    sc = np.array([scale.get(g, np.nan) for g in GEOS])[None, :, None]
    mask = np.isfinite(Tt) & np.isfinite(Nn) & np.isfinite(sc)
    for a in models.values():
        mask &= np.isfinite(a)
    mask[:, [j for j in range(len(GEOS)) if j not in cidx], :] = False
    out = {"n": int(mask.sum())}
    if mask.sum() == 0:
        return out
    den = float((np.abs(Nn - Tt) / sc)[mask].sum())
    kind = "inc" if RC.TARGETS[t][0] == "lvl" else "t50"
    ref = Nn[0, :, 0]
    for name, a in models.items():
        e = float((np.abs(a - Tt) / sc)[mask].sum())
        r = dict(skill=1 - e / den if den > 0 else np.nan, mae_scaled=e / mask.sum())
        dirs, rat, tl = [], [], []
        for j in cidx:
            g = GEOS[j]
            if not moved.get(g) or not np.isfinite(ref[j]):
                continue
            m = mask[:, j, :]
            if m.sum() < 6:
                continue
            dF, dR = np.mean((a[:, j, :] - ref[j])[m]), np.mean((Tt[:, j, :] - ref[j])[m])
            if abs(dR) < 1e-12:
                continue
            dirs.append(np.sign(dF) == np.sign(dR))
            rat.append(dF / dR)
            h3 = 3 if a.shape[2] > 3 else 1
            pf, pr = a[:, j, h3] - ref[j], Tt[:, j, h3] - ref[j]
            tf, trr = tfirst(pf, kind), tfirst(pr, kind)
            if np.isfinite(tf) and np.isfinite(trr):
                tl.append(tf - trr)
        r.update(n_moved=len(dirs), direction=float(np.mean(dirs)) if dirs else np.nan, ratio_median=float(np.median(rat)) if rat else np.nan,
                 within10=float(np.mean([abs(x - 1) <= 0.1 for x in rat])) if rat else np.nan,
                 within20=float(np.mean([abs(x - 1) <= 0.2 for x in rat])) if rat else np.nan,
                 timing_median_abs=float(np.median(np.abs(tl))) if tl else np.nan, timing_median=float(np.median(tl)) if tl else np.nan,
                 ratios={GEOS[j]: float(x) for j, x in zip([j for j in cidx if moved.get(GEOS[j])], rat)} if False else None)
        out[name] = r
    return out


def score(split: str):
    S = RC.Store()
    evs = [e for e, (_, sp) in EVENTS.items() if split in ("all", sp)]
    allr = {}
    for e in evs:
        ev = pickle.load(open(OUT / f"{e}.pkl", "rb"))
        FMD = OUT.parent / "rates_credit"
        fms = {m: pickle.load(open(FMD / f"{e}_{m}.pkl", "rb")) for m in FM_USE if (FMD / f"{e}_{m}.pkl").exists()}
        cells = {}
        for t in TG:
            scale = scale_of(S, t)
            moved = moved_flags(S, t, ev)
            models = {"mech": ev["F"]["mech"][t], "held": ev["F"]["held"][t], "off": ev["F"]["off"][t], "snl": ev["SNL"][t]}
            for m, P in fms.items():
                models[m] = P[t]
            cm = cell_metrics(S, t, ev, models, scale, moved)
            ok = [m for m in fms if m in cm]
            if ok:
                best = max(ok, key=lambda m: cm[m]["skill"])
                cm["best_open"] = dict(cm[best], which=best)
            cm["moved"] = [g for g, v in moved.items() if v]
            cells[t] = cm
        allr[e] = cells
    json.dump(allr, open(OUT.parent / f"{OUT.name}_scores_{split}.json", "w"), indent=1, default=lambda x: None if x != x else float(x))
    return allr


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a and a[0].startswith("spec="):                       # e.g. spec=B build dev
        RC.use_spec(a[0][5:])
        global OUT
        OUT = OUT.parent / f"rates_credit_spec{a[0][5:]}"
        a = a[1:]
    if a and a[0].startswith("only="):                      # e.g. only=ccob_nfc,ccob_hh : restrict to some targets (stage-1 trials)
        global TG
        TG = a[0][5:].split(",")
        a = a[1:]
    if a[0] == "build":
        for e in (EVENTS if a[1] == "dev" else [a[1]]):
            if a[1] == "dev" and EVENTS[e][1] != "dev":
                continue
            build(e)
    elif a[0] == "fm":
        fm(a[1], a[2])
    elif a[0] == "score":
        score(a[1])


if __name__ == "__main__":
    sys.exit(main())
