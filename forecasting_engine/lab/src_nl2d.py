"""Newer layers as drivers of the existing layers.

    python -m forecasting_engine.lab.panel extend ip_m retail_m unemp_m asylum_m     # adds the drivers below to the frozen families

| Link (layer -> layer) | Families | Input | Real publication lag (worst case used) | Licence |
|---|---|---|---|---|
| freight -> economy | ip_m (DE) | German lorry-toll mileage index (Destatis / BALM): monthly mean and its y/y change | the daily file lags 12 days: month + 12 days | dl-de/by-2.0 |
| credit -> economy | retail_m, ip_m | ECB new-business loan volumes: consumer loans -> retail, loans to firms -> industry, y/y log change per country | month + 65 days | ECB reuse |
| firms -> labour | unemp_m (ES) | Spanish companies constituted and dissolved per month, national totals, y/y | month + 85 days | INE |
| border -> migration | asylum_m | Frontex detections of illegal border-crossings by route (monthly) and in total, log | month + 75 days (the file is refreshed in the first week of the month with the month before) | Frontex (source acknowledged) |
| attention + labour -> migration | asylum_m | Wikipedia migration-group views (home-language edition, monthly mean), Indeed job postings (6 EU countries) | month + 2 days; month + 10 days | CC0; CC BY 4.0 |
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, MONTH = 24.0, 24.0 * 30.44


def _utc(x):
    x = x.copy()
    x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
    return x


def _yoy(x):
    return (100 * np.log(x / x.shift(12))).replace([np.inf, -np.inf], np.nan)      # a zero month would give an infinite change


def _credit(kind: str) -> pd.DataFrame:
    c = _utc(HS.get_parquet("lab/raw/ecb_credit.parquet"))
    c = c[[k for k in c.columns if k.startswith(kind + ":")]]
    c.columns = [k.split(":")[1] for k in c.columns]
    return c.drop(columns=["EA"], errors="ignore")


def drivers(fid: str, P, PN) -> list:
    Driver, D = PN.Driver, []
    if fid == "ip_m":
        t = _utc(HS.get_parquet("lab/raw/lkw_toll_daily.parquet"))
        n = t["toll:DE"].resample("MS").count()
        m = t["toll:DE"].resample("MS").mean().where(n >= 27).dropna().to_frame("DE")
        D += [Driver("nl_toll", "freight", m, MONTH + 12 * DAY, note="Destatis lorry-toll mileage index (unadjusted), Germany, monthly mean"),
              Driver("nl_toll_yoy", "freight", _yoy(m), MONTH + 12 * DAY)]
        D.append(Driver("nl_credit_nfc_yoy", "credit", _yoy(_credit("nfc")), MONTH + 65 * DAY, note="ECB new-business loans to firms, y/y log change %"))
    if fid == "retail_m":
        D.append(Driver("nl_credit_cons_yoy", "credit", _yoy(_credit("cons")), MONTH + 65 * DAY, note="ECB new-business consumer loans, y/y log change %"))
    if fid == "unemp_m":
        f = _utc(HS.get_parquet("lab/raw/ine_firms.parquet"))
        cr = f[[c for c in f.columns if c.startswith("created:")]].sum(axis=1, min_count=19).to_frame("ES")
        ds = f[[c for c in f.columns if c.startswith("dissolved:")]].sum(axis=1, min_count=19).to_frame("ES")
        D += [Driver("nl_firms_created_yoy", "firms", _yoy(cr), MONTH + 85 * DAY, note="Spain: companies constituted, y/y log change %"),
              Driver("nl_firms_dissolved_yoy", "firms", _yoy(ds), MONTH + 85 * DAY)]
    if fid == "asylum_m":
        fx = _utc(HS.get_parquet("lab/raw/frontex_ibc.parquet"))
        for r in ("Total", "Eastern_Mediterranean", "Central_Mediterranean", "Western_Balkan", "Western_Mediterranean", "Western_African", "Eastern_Borders"):
            if r in fx.columns:
                D.append(Driver(f"nl_frontex_{r}", "migration", np.log1p(fx[[r]].clip(lower=0)).rename(columns={r: "*"}), MONTH + 75 * DAY, note=f"Frontex detections of illegal border-crossings, {r}, log"))
        from forecasting_engine.lab import src_nl as SNL
        w = SNL.wiki_group_daily()["migration"]
        lv = np.log1p(w).rolling(7, min_periods=7).mean()
        D.append(Driver("nl_wiki_migration", "attention", lv.resample("MS").mean(), MONTH + 2 * DAY, note="Wikipedia migration-group articles, log views, monthly mean (home-language edition)"))
        j = _utc(HS.get_parquet("lab/raw/indeed_jobs.parquet"))
        j = j[j["key"].str.startswith("total:")].reset_index().pivot_table(index="date", columns="key", values="value").sort_index()
        j.columns = [c.split(":")[1] for c in j.columns]
        j = j[[c for c in j.columns if c in ("DE", "ES", "FR", "IE", "IT", "NL")]]
        D.append(Driver("nl_jobs_total", "labour", j.resample("MS").mean(), MONTH + 10 * DAY, note="Indeed total postings (NSA), monthly mean"))
    return D
