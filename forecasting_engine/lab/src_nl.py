"""New layers as drivers of the existing layers.

    python -m forecasting_engine.lab.panel extend health_w unrest_w info_w tourism_m unemp_m   # adds the drivers below to the frozen families

| Link (layer -> layer) | Families | Input | Real publication lag (worst case used) | Licence |
|---|---|---|---|---|
| attention -> health | health_w | Wikipedia views (21 editions, country by home language) of the health articles (influenza, COVID-19, vaccine, pneumonia, common cold, measles): log of the 7-day mean, and its change on the 7 days before | a day is served the next morning: stamped at the day, known 2 days later | CC0 |
| attention -> society | unrest_w, info_w | the same for the conflict group (war, terrorism, protest, strike, coup, NATO), the prices group (inflation, recession, unemployment, crypto) and the migration group | 2 days | CC0 |
| attention -> tourism | tourism_m | the travel group (tourism, airline, hotel, airport, beach) and the hazard group (heat wave, flood, drought, earthquake, storm), monthly mean of the daily 7-day mean | month + 2 days | CC0 |
| labour (job postings) -> economy | unemp_m | Indeed total and new postings (NSA, 6 EU countries: DE ES FR IE IT NL), monthly mean and 3-month change | month + 10 days (weekly repository refresh, 1 week of data lag) | CC BY 4.0 |
| migration -> society | unrest_w | Eurostat first-time asylum applicants, log, 3-month change, per country | month + 85 days | Eurostat |
| agriculture -> economy | hicp_m | raw milk collected, y/y change per country | month + 55 days | Eurostat |
| space weather -> transport | air_d | GFZ Ap, Kp max and F10.7 (global) | 30 h | CC BY 4.0 |
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab import src_wiki as SW

DAY, WEEK, MONTH = 24.0, 24.0 * 7, 24.0 * 30.44
WIKI_GROUPS = {"health_w": ("health",), "unrest_w": ("conflict", "prices", "migration"), "info_w": ("conflict", "prices", "migration", "cyber"),
               "tourism_m": ("travel", "hazard")}


def _utc(x):
    x = x.copy()
    x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
    return x


def wiki_group_daily() -> dict[str, pd.DataFrame]:
    """group -> daily frame (columns = country codes) of the summed user views of the group's articles in the country's language edition."""
    tp = json.loads(HS.get_bytes(SW.TOPIC_STORE))
    pv = _utc(HS.get_parquet("lab/raw/wiki_pv_daily.parquet"))
    out = {}
    for g in SW.TOPICS:
        cols = {}
        for lang, cc in SW.LANG_CC.items():
            qs = [c for c in pv.columns if c.startswith(lang + ":") and tp[c.split(":")[1]]["group"] == g and tp[c.split(":")[1]]["en"] not in SW.DROP]
            if qs:
                cols[cc] = pv[qs].sum(axis=1, min_count=1)
        out[g] = pd.DataFrame(cols)
    return out


def drivers(fid: str, P, PN) -> list:
    Driver, D = PN.Driver, []
    if fid in WIKI_GROUPS:
        monthly = fid.endswith("_m")
        for g, x in wiki_group_daily().items():
            if g not in WIKI_GROUPS[fid]:
                continue
            lv = np.log1p(x).rolling(7, min_periods=7).mean()
            chg = lv - lv.shift(7)
            if monthly:
                D += [Driver(f"nl_wiki_{g}", "attention", lv.resample("MS").mean(), MONTH + 2 * DAY, note=f"Wikipedia {g} articles, log views, monthly mean of the 7-day mean"),
                      Driver(f"nl_wiki_{g}_chg", "attention", chg.resample("MS").mean(), MONTH + 2 * DAY)]
            else:
                D += [Driver(f"nl_wiki_{g}", "attention", lv, 2 * DAY, note=f"Wikipedia {g} articles, log of the 7-day mean of daily user views, country = home language edition"),
                      Driver(f"nl_wiki_{g}_chg", "attention", chg, 2 * DAY, note="its change on the 7 days before")]
    if fid == "unemp_m":
        j = _utc(HS.get_parquet("lab/raw/indeed_jobs.parquet"))
        j = j[j["key"].str.match(r"^(total|new):")]
        for kind in ("total", "new"):
            w = j[j["key"].str.startswith(kind + ":")].reset_index().pivot_table(index="date", columns="key", values="value").sort_index()
            w.columns = [c.split(":")[1] for c in w.columns]
            w = w[[c for c in w.columns if c in ("DE", "ES", "FR", "IE", "IT", "NL")]]
            m = w.resample("MS").mean()
            full = w.notna().resample("MS").sum() >= w.index.to_series().resample("MS").count().to_numpy()[:, None] * 0.9
            m = m.where(full)
            D += [Driver(f"nl_jobs_{kind}", "labour", m, MONTH + 10 * DAY, note=f"Indeed {kind} job postings index (NSA), monthly mean"),
                  Driver(f"nl_jobs_{kind}_chg3", "labour", 100 * np.log(m / m.shift(3)), MONTH + 10 * DAY)]
    if fid == "unrest_w":
        a = _utc(HS.get_parquet("lab/raw/estat_asylum.parquet"))
        a = a[[c for c in a.columns if not c.startswith("EU")]]
        la = np.log1p(a)
        D += [Driver("nl_asylum", "migration", la, MONTH + 85 * DAY, note="first-time asylum applicants, log"),
              Driver("nl_asylum_chg3", "migration", la - la.shift(3), MONTH + 85 * DAY)]
    if fid == "hicp_m":
        m = _utc(HS.get_parquet("lab/raw/estat_milk.parquet"))
        m = m[[c for c in m.columns if len(c) == 2 and c not in ("IT", "LU", "CH", "NO", "UA", "TR", "AL", "BA", "ME", "MK", "RS", "IS")]]
        D += [Driver("nl_milk_yoy", "agriculture", 100 * np.log(m / m.shift(12)), MONTH + 55 * DAY, note="raw milk collected, y/y log change %")]
    if fid == "air_d":
        s = _utc(HS.get_parquet("lab/raw/space_daily.parquet"))
        for c in ("ap", "kp_max", "f107"):
            D.append(Driver(f"nl_space_{c}", "space", s[[c]].rename(columns={c: "*"}), 30.0, note="GFZ Potsdam geomagnetic / solar index, global"))
    return D
