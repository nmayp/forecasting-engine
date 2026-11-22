"""Cross-layer links: newer layers as drivers of existing layers, and of each other.

    python -m forecasting_engine.lab.panel extend milk_m hicp_ea_m unrest_w traffic_d wiki_w mort_w ports_m     # adds the drivers below to the frozen families

Every link has a mechanism, a real publication lag (the worst case is used) and goes through the normal dev gate (`forecasting_engine.lab.nldriver`): the champion
re-run with the driver added, one holdout look only if it passes on development and on most blocks.

| Link (layer -> layer) | Families | Input | Lag used |
|---|---|---|---|
| agri-food prices -> milk | milk_m | EC weekly dairy, cereal and oilseed prices (monthly mean), World Bank maize, soybean meal, wheat: feed costs set the milk price | month + 5 days (EC), month + 3 days (WB) |
| agricultural commodities and agri-food prices -> food inflation | hicp_ea_m | World Bank food groups (grains, oils, sugar, meat, softs) and EC meat, dairy and cereal prices, y/y | month + 5 days |
| food commodities -> unrest | unrest_w | World Bank grain, vegetable-oil, sugar and fertiliser prices, y/y (food-price protests) | month + 3 days |
| agricultural commodities -> ports | ports_m | World Bank grain, oilseed and fertiliser prices, y/y (bulk cargo value and volume) | month + 3 days |
| cycling -> road traffic | traffic_d | Norwegian bicycle counts, national index (shared weather, holidays and commuting) | 2 days |
| earthquakes -> attention | wiki_w | USGS weekly counts of M4.5+, M5.5+, M6.5+ (a large quake sends readers to the Earthquake article) | week + 2 days |
| air quality, vehicles -> mortality | mort_w | EEA daily PM2.5 and NO2 in Amsterdam and Brussels (weekly mean) | 3 days |
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK, MONTH = 24.0, 24.0 * 7, 24.0 * 30.44


def _utc(x):
    x = x.copy()
    ix = pd.DatetimeIndex(x.index)
    x.index = ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")
    return x


def _yoy(x):
    return (100 * np.log(x / x.shift(12))).replace([np.inf, -np.inf], np.nan)      # a zero month would give an infinite change


def _star(s, name=None):
    return s.dropna().to_frame("*")


def _pink():
    return _utc(HS.get_parquet("lab/raw/pink_monthly.parquet"))


def _agri_m():
    a = _utc(HS.get_parquet("lab/raw/agri_weekly.parquet"))
    return a.resample("MS").mean().where(a.resample("MS").count() >= 3)


def drivers(fid: str, P, PN) -> list:
    Driver, D = PN.Driver, []
    if fid == "milk_m":
        pk, ag = _pink(), _agri_m()
        for k in ("maize", "soybean_meal", "wheat_us_hrw", "barley"):
            D.append(Driver(f"nl4_wb_{k}", "agricom", _star(pk[f"{k}:WW"]), MONTH + 3 * DAY, note=f"World Bank {k} price, USD (feed cost)"))
        for k in [c for c in ag.columns if c.startswith(("dairy_", "cereal_", "oilseeds_"))][:12]:
            D.append(Driver(f"nl4_eu_{k.split(':')[0]}", "agri-food prices", _star(ag[k]), MONTH + 5 * DAY, note="EC Agri-food data portal weekly price, monthly mean"))
    if fid == "hicp_ea_m":
        pk, ag = _pink(), _agri_m()
        grp = {"grains": ["wheat_us_hrw", "maize", "barley", "rice_thai_5"], "oils": ["palm_oil", "soybean_oil", "sunflower_oil", "rapeseed_oil"], "sugar": ["sugar_world", "sugar_eu"],
               "meat": ["beef", "chicken", "lamb"], "softs": ["cocoa", "coffee_arabica", "coffee_robusta"], "fert": ["urea", "dap", "potassium_chloride"]}
        for g, ks in grp.items():
            x = np.log(pk[[f"{k}:WW" for k in ks]]).mean(axis=1)
            D.append(Driver(f"nl4_wb_{g}_yoy", "agricom", _star(100 * (x - x.shift(12))), MONTH + 3 * DAY, note=f"World Bank {g} prices, mean log y/y %"))
        for pre in ("dairy_", "pigmeat_", "beef_", "poultry_", "cereal_"):
            cols = [c for c in ag.columns if c.startswith(pre)]
            if cols:
                x = np.log(ag[cols]).mean(axis=1)
                D.append(Driver(f"nl4_eu_{pre.strip('_')}_yoy", "agri-food prices", _star(100 * (x - x.shift(12))), MONTH + 5 * DAY, note="EC Agri-food data portal weekly prices, monthly mean, mean log y/y %"))
    if fid == "unrest_w":
        pk = _pink()
        for g, ks in {"grains": ["wheat_us_hrw", "maize", "rice_thai_5"], "oils": ["palm_oil", "soybean_oil", "sunflower_oil"], "sugar": ["sugar_world"], "fert": ["urea", "dap", "potassium_chloride"]}.items():
            x = np.log(pk[[f"{k}:WW" for k in ks]]).mean(axis=1)
            D.append(Driver(f"nl4_wb_{g}_yoy", "agricom", _star(100 * (x - x.shift(12))), MONTH + 3 * DAY, note=f"World Bank {g} prices, mean log y/y % (food-price protests)"))
    if fid == "ports_m":
        pk = _pink()
        for g, ks in {"grains": ["wheat_us_hrw", "maize", "barley"], "oilseeds": ["soybeans", "palm_oil", "soybean_meal"], "fert": ["urea", "dap", "potassium_chloride"]}.items():
            x = np.log(pk[[f"{k}:WW" for k in ks]]).mean(axis=1)
            D.append(Driver(f"nl4_wb_{g}_yoy", "agricom", _star(100 * (x - x.shift(12))), MONTH + 3 * DAY, note=f"World Bank {g} prices, mean log y/y % (bulk food and fertiliser cargo)"))
    if fid == "traffic_d":
        b = _utc(HS.get_parquet("lab/raw/npra_bike_daily.parquet"))
        D.append(Driver("nl4_bike_all", "cycling", b[["all:NO"]].rename(columns={"all:NO": "*"}), 2 * DAY, note="Norwegian bicycle counts, national index (bike_d)"))
    if fid == "wiki_w":
        q = _utc(HS.get_parquet("lab/raw/usgs_quakes_weekly.parquet"))
        for k in ("m45", "m55", "m65"):
            D.append(Driver(f"nl4_quake_{k}", "earthquakes", _star(q[f"{k}:WW"]), WEEK + 2 * DAY, note=f"USGS earthquakes per week, {k}"))
    if fid == "mort_w":
        ccs = ["NL", "BE"]
        for k in ("aq_pm25", "aq_no2"):
            x = _utc(HS.get_parquet(f"lab/raw/{k}.parquet")).astype(float).reindex(columns=ccs)
            D.append(Driver(f"nl4_{k}", "environment", x.resample("W-MON", label="left", closed="left").mean(), WEEK + 3 * DAY, note="EEA urban background daily mean in the capital, weekly mean (air quality layer)"))
    return D
