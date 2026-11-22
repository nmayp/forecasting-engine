"""Shock replays: does the engine see the consequences of a real shock once the shock is known?

    python -m forecasting_engine.cascade.shocks run [--methods gbm_all,gbm_own,champion,chronos2x] [--shocks id,id] [--families load_h]
("champion" = each family's champion; any lab method name works for the window skill, the intervention test is gbm_all's)

For each shock in SHOCKS (public onset date, the families and series it hit, the drivers that carry it) and each family it hit:
1. Window skill. Every origin from the onset to onset + window is forecast by each method, trained only on what was known at the onset, and
   scored against the strict reference (pooled skill, as in core.score). This says whether the engine forecast the aftermath better than the
   simple forecasts, and whether the cross-layer drivers helped (gbm_all vs gbm_own, which sees only the target series' own histories).
2. Intervention test (what the engine thinks the shock does). With the same fitted model, the same origins are forecast twice: with the real
   inputs, and with the shock's drivers held at their values the day before the onset. The difference is the engine's predicted effect of
   the shock. The realised surprise is the truth minus the held-input forecast. Scores: the share of the surprise the predicted effect explains
   (1 - sum (surprise - effect)^2 / sum surprise^2; 0 = no help, 1 = all of it, negative = it made the forecast worse) and the share of points
   where the predicted effect has the sign of the surprise (only points with an effect above 0.1 typical steps).

Replays are run once per method version, on the champions and the ablations, after the development loop (they overlap both windows: they are
a stress test of behaviour around shocks, not a second holdout). Results: results/lab/shocks.json.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from forecasting_engine.lab import core as C

# id, label, onset (UTC), window days, {family: [series] or "*"}, drivers carrying the shock (held for the intervention test), kind
SHOCKS = [
    ("covid_lockdowns", "COVID-19 lockdowns in Europe", "2020-03-12", 60, {"load_d": "*", "load_h": "*", "price_h": "*", "commod_d": "*", "fx_d": "*",
     "ip_m": "*", "retail_m": "*", "unemp_m": "*", "diesel_w": "*", "gasoline_w": "*", "unrest_w": "*"}, ["vix", "brent_eur", "us10y", "dollar_index"], "health"),
    ("ukraine_invasion", "Russia's full-scale invasion of Ukraine", "2022-02-24", 60, {"commod_d": "*", "fx_d": "*", "diesel_w": "*", "gasoline_w": "*",
     "price_h": "*", "hicp_m": "*", "unrest_w": "*"}, ["oil_region_violence", "brent_eur", "henry_hub_eur", "vix", "wheat_usd", "gas_eu_usd"], "geopolitics"),
    ("nord_stream_cuts", "Gas flows through Nord Stream cut to 40 %, then 20 %, then stopped", "2022-06-14", 90, {"price_h": "*", "load_d": "*",
     "hicp_m": "*", "ip_m": "*"}, ["henry_hub_eur", "gas_eu_usd", "brent_eur"], "supply"),
    ("fr_nuclear_2022", "French nuclear fleet outages (stress corrosion)", "2022-01-13", 120, {"price_h": "*", "load_h": ["FR"]},
     ["gen_outage_mw", "nuclear_share", "net_imports"], "outage"),
    ("fr_refinery_strikes", "French refinery strikes and fuel shortages", "2022-09-27", 35, {"diesel_w": ["FR", "BE", "DE", "ES", "IT"],
     "gasoline_w": ["FR", "BE", "DE", "ES", "IT"], "unrest_w": ["strike:FR"]}, ["gdelt_strike", "gdelt_protest"], "strike"),
    ("fr_pension_strikes", "French pension reform strikes", "2023-01-19", 90, {"unrest_w": ["strike:FR", "protest:FR"], "load_h": ["FR"],
     "price_h": ["FR"]}, ["gdelt_strike", "gdelt_protest", "gdelt_tone"], "strike"),
    ("iberian_blackout", "Iberian peninsula blackout", "2025-04-28", 21, {"load_h": ["ES", "PT"], "price_h": ["ES", "PT"], "res_h": ["ES", "PT"],
     "load_d": ["ES", "PT"]}, ["load", "net_imports", "gen_outage_mw"], "outage"),
    ("red_sea_attacks", "Houthi attacks on Red Sea shipping", "2023-12-15", 60, {"commod_d": "*", "diesel_w": "*", "gasoline_w": "*"},
     ["oil_region_violence", "brent_eur"], "geopolitics"),
    ("ever_given", "Suez Canal blocked by the Ever Given", "2021-03-23", 21, {"commod_d": "*"}, ["oil_region_violence", "brent_eur"], "logistics"),
    ("opec_cut_2023", "Surprise OPEC+ production cut", "2023-04-02", 30, {"commod_d": "*", "diesel_w": "*", "gasoline_w": "*"}, ["brent_eur"], "markets"),
    ("filomena", "Storm Filomena snowfall in Spain", "2021-01-08", 14, {"load_h": ["ES"], "price_h": ["ES"], "load_d": ["ES"]}, ["temp", "hdd"], "weather"),
    ("heatwave_2022", "July 2022 western Europe heatwave", "2022-07-12", 21, {"load_h": "*", "price_h": "*", "load_d": "*"}, ["temp", "cdd"], "weather"),
    ("power_crisis_2021", "Autumn 2021 gas and power price spike", "2021-09-01", 60, {"price_h": "*"}, ["henry_hub_eur", "gas_eu_usd"], "markets"),
    ("de_nuclear_exit", "Germany shuts its last nuclear plants", "2023-04-15", 45, {"price_h": ["DE", "FR", "NL", "BE", "PL"]}, ["nuclear_share", "net_imports"],
     "policy"),
    ("uk_minibudget", "UK mini-budget market turmoil", "2022-09-23", 30, {"fx_d": ["GBP"]}, ["us10y", "vix", "dollar_index"], "policy"),
    ("svb_collapse", "Silicon Valley Bank collapse", "2023-03-10", 30, {"fx_d": "*", "commod_d": "*"}, ["vix", "us10y", "us2y_minus_ecb"], "finance"),
    ("try_crash_2021", "Turkish lira crash", "2021-11-23", 30, {"fx_d": ["TRY"]}, ["us2y_minus_ecb", "dollar_index"], "finance"),
    ("ecb_hikes_2022", "ECB starts raising rates", "2022-07-21", 60, {"fx_d": "*"}, ["ecb_deposit_rate", "us2y_minus_ecb"], "policy"),
    ("log4shell", "Log4Shell vulnerability", "2021-12-10", 30, {"kev_d": "*"}, [], "cyber"),
    ("moveit", "MOVEit mass exploitation", "2023-05-31", 30, {"kev_d": "*"}, [], "cyber"),
    ("farmer_protests_2024", "European farmer protests", "2024-01-08", 60, {"unrest_w": ["protest:FR", "protest:DE", "protest:BE", "protest:NL", "protest:PL",
     "protest:ES", "protest:IT", "protest:RO"]}, ["diesel", "hicp", "gdelt_tone"], "society"),
    ("energy_inflation_2022", "Energy-led inflation surge", "2022-03-01", 180, {"hicp_m": "*", "unrest_w": "*"}, ["brent_eur", "gas_eu_usd", "price_month",
     "diesel_month", "diesel"], "markets"),
]

# every layer is a target too: the same real shocks on the non-energy, non-economy families, and shocks that hit them first
_MORE_LAYERS = {
    "covid_lockdowns": ("air_d", "tourism_m", "ports_m", "info_w", "security_w"),
    "ukraine_invasion": ("ports_m", "security_w", "info_w", "air_d", "tourism_m", "kev_d"),
    "red_sea_attacks": ("ports_m", "security_w", "info_w"),
    "ever_given": ("ports_m",),
    "heatwave_2022": ("wildfire_w", "water_w", "aq_d", "health_w", "air_d"),
    "iberian_blackout": ("air_d", "info_w", "aq_d"),
    "filomena": ("air_d",),
    "fr_pension_strikes": ("air_d", "security_w", "info_w"),
    "fr_refinery_strikes": ("security_w", "info_w"),
    "farmer_protests_2024": ("security_w", "info_w", "ports_m"),
    "log4shell": ("info_w", "security_w"),
    "moveit": ("info_w", "security_w"),
}
SHOCKS += [
    ("crowdstrike_outage", "A faulty CrowdStrike update takes down Windows systems worldwide (airlines, hospitals, banks)", "2024-07-19", 14,
     {"air_d": "*", "kev_d": "*", "info_w": "*", "security_w": "*"}, [], "cyber"),
    ("paris_olympics", "Paris 2024 Olympic Games (security perimeter, crowds, travel)", "2024-07-26", 21,
     {"air_d": "*", "tourism_m": "*", "security_w": "*", "info_w": "*", "load_h": "*"}, [], "events"),
    ("winter_resp_2022", "The 2022-23 winter wave of flu, RSV and COVID across Europe", "2022-11-15", 90,
     {"health_w": "*", "load_d": "*", "air_d": "*", "unemp_m": "*"}, [], "health"),
    ("omicron_2021", "Omicron: travel bans and new restrictions across Europe", "2021-11-26", 45,
     {"air_d": "*", "tourism_m": "*", "commod_d": "*", "fx_d": "*", "load_d": "*"}, [], "health"),
    ("drought_2023", "Spring 2023 drought in Iberia and southern France (reservoirs, fires, farm output)", "2023-03-01", 120,
     {"water_w": "*", "wildfire_w": "*", "load_d": "*", "price_h": "*"}, [], "environment"),
]
SHOCKS = [(a, b, c, d, {**e, **{f: "*" for f in _MORE_LAYERS.get(a, ())}}, g, h) for a, b, c, d, e, g, h in SHOCKS]


def _freeze_tau(d: C.Design, origins: np.ndarray, onset: pd.Timestamp) -> np.ndarray:
    return np.full(len(origins), (onset - pd.Timedelta(days=1)).as_unit("ns").value)


class _FrozenFeatures(C.Features):
    """Features with some drivers read at a fixed earlier time (the day before the onset) instead of at each origin."""

    def __init__(self, *a, freeze: set[str] | None = None, freeze_tau: int | None = None, **k):
        super().__init__(*a, **k)
        self.freeze, self.freeze_tau = freeze or set(), freeze_tau

    def build(self, origins):
        if not self.freeze:
            return super().build(origins)
        orig = C._driver_known
        fz, ft = self.freeze, self.freeze_tau
        names = {id(x.data): x.name for x in self.d.fam.drivers}

        def patched(dr, col, tau, at_stamps=None):
            if dr.name in fz:
                tau = np.minimum(tau, ft)
            return orig(dr, col, tau, at_stamps)
        C._driver_known = patched
        try:
            return super().build(origins)
        finally:
            C._driver_known = orig
            del names


def fit_predict(d: C.Design, refs, origins: np.ndarray, train_from: int, upto: int, groups=None, cross=True, freeze=None, freeze_tau=None):
    """One pooled tree model fitted on what was known at `upto`; predictions with real inputs and (if freeze) with held drivers."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = C.Features(d, refs, groups=groups, cross=cross)
    tr = C._train_origins(d, train_from, upto, C._stride(d))
    if len(tr) < 30:
        return None, None
    Xt, zt, _, _, _ = F.build(tr)
    ok = np.isfinite(zt)
    keep = np.isfinite(Xt[ok]).any(0)
    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                      l2_regularization=1.0, early_stopping=False, random_state=0,
                                      categorical_features=[F.names.index("series_id")])
    m.fit(Xt[ok][:, keep], zt[ok])

    def pred(FF):
        Xe, _, refe, sce, ke = FF.build(origins)
        p = refe + sce * m.predict(Xe[:, keep])
        out = np.full((len(origins), d.J, d.H), np.nan)
        pos = np.searchsorted(origins, ke[:, 0])
        out[pos, ke[:, 1], ke[:, 2] - 1] = p
        return out
    P = pred(F)
    Pf = None
    if freeze:
        FF = _FrozenFeatures(d, refs, groups=groups, cross=cross, freeze=set(freeze), freeze_tau=freeze_tau)
        Pf = pred(FF)
    return P, Pf


VARIANTS = {"gbm_all": dict(groups=None, cross=True), "gbm_own": dict(groups=set(), cross=False), "gbm_cross": dict(groups=set(), cross=True)}


def replay(sid: str, methods=("gbm_all", "gbm_own"), families=None) -> list[dict]:
    s = next(x for x in SHOCKS if x[0] == sid)
    _, label, onset, win, fams, drivers, kind = s
    onset = pd.Timestamp(onset, tz="UTC")
    rows = []
    for fid, ser in fams.items():
        if families and fid not in families:
            continue
        try:
            d, reg = C.design(fid)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(shock=sid, family=fid, error=f"{type(e).__name__}: {e}"))
            continue
        refs = [reg["reference"][x] for x in d.series]
        o0, o1 = onset.as_unit("ns").value, (onset + pd.Timedelta(days=win)).as_unit("ns").value
        eo = np.where(d.ok_origin & (d.tau >= o0) & (d.tau <= o1))[0]
        if len(eo) < 2:
            rows.append(dict(shock=sid, family=fid, error="no origins in the window (outside the family's data)"))
            continue
        tf = int(np.searchsorted(d.stamps.as_unit("ns").asi8, pd.Timestamp(reg["train_from"]).as_unit("ns").value))
        js = list(range(d.J)) if ser == "*" else [d.series.index(x) for x in ser if x in d.series]
        held = [x for x in drivers if any(dr.name == x for dr in d.fam.drivers)]
        row = dict(shock=sid, label=label, kind=kind, onset=str(onset.date()), window_days=win, family=fid, series=[d.series[j] for j in js],
                   origins=len(eo), held_drivers=held, methods={})
        Rf = C.m_ref(d, refs, eo, tf)
        for mname in methods:
            if mname == "champion":                  # the family's champion (point part), compared under its own name
                ch = C.champion(fid) or "gbm_all"
                base = next((x for x in ("+rollmax", "+rollp", "+rollv", "+roll") if ch.endswith(x)), None)
                mname = ch[: -len(base)] if base else ch
            if mname in VARIANTS:
                P, Pf = fit_predict(d, refs, eo, tf, o0, freeze=held if mname == "gbm_all" and held else None,
                                    freeze_tau=(onset - pd.Timedelta(days=1)).as_unit("ns").value, **VARIANTS[mname])
            else:                                   # any lab method (a champion): window skill only, refitted on what was known at each block
                P, Pf = C.method(mname)(d, refs, eo, tf), None
            if P is None:
                continue
            num = den = 0.0
            for j in js:
                y = d.truth(j)[eo]
                ok = np.isfinite(y) & np.isfinite(P[:, j]) & np.isfinite(Rf[:, j])
                num += np.abs(y - P[:, j])[ok].sum() / d.scale[j]
                den += np.abs(y - Rf[:, j])[ok].sum() / d.scale[j]
            res = dict(window_skill=round(1 - num / den, 4) if den > 0 else None)
            if Pf is not None:
                S, E = [], []
                for j in js:
                    y = d.truth(j)[eo]
                    ok = np.isfinite(y) & np.isfinite(P[:, j]) & np.isfinite(Pf[:, j])
                    S.append(((y - Pf[:, j]) / d.scale[j])[ok])
                    E.append(((P[:, j] - Pf[:, j]) / d.scale[j])[ok])
                S, E = np.concatenate(S), np.concatenate(E)
                big = np.abs(E) > 0.1
                res.update(explained=round(1 - float(np.sum((S - E) ** 2) / np.sum(S ** 2)), 4) if np.sum(S ** 2) > 0 else None,
                           sign_agree=round(float(np.mean(np.sign(E[big]) == np.sign(S[big]))), 3) if big.sum() >= 10 else None,
                           n_effect=int(big.sum()), mean_effect=round(float(np.mean(E)), 3), mean_surprise=round(float(np.mean(S)), 3))
            row["methods"][mname] = res
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    return rows


def run(methods=("gbm_all", "gbm_own"), shocks=None, families=None) -> dict:
    out = []
    for s in SHOCKS:
        if shocks and s[0] not in shocks:
            continue
        t0 = time.time()
        rr = replay(s[0], methods, families)
        for r in rr:
            r["run_s"] = round(time.time() - t0, 1)
            print(json.dumps(r, default=str), flush=True)              # also in the log
        out += rr
    res = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code=C.code_hash(), methods=list(methods), rows=out)
    C.LAB.mkdir(parents=True, exist_ok=True)
    (C.LAB / "shocks.json").write_text(json.dumps(res, default=str))
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--methods", default="gbm_all,gbm_own")
    ap.add_argument("--shocks")
    ap.add_argument("--families")
    a = ap.parse_args(argv)
    run(tuple(a.methods.split(",")), a.shocks.split(",") if a.shocks else None, a.families.split(",") if a.families else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
