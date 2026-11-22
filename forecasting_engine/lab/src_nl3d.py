"""Cross-layer links: newer layers as drivers of existing layers, and of each other.

    python -m forecasting_engine.lab.panel extend retail_m ip_m housing_m credit_m ports_m transit_d c311_d     # adds the drivers below to the frozen families

Every link has a mechanism, a real publication lag (the worst case is used) and goes through the normal dev gate (`forecasting_engine.lab.nldriver`): the champion
re-run with the driver added, one holdout look only if it passes on development and on most blocks.

| Link (layer -> layer) | Families | Input | Lag used |
|---|---|---|---|
| sentiment -> retail, industry | retail_m, ip_m | EC retail and services confidence balances (sent_m's own series), 13 countries | the survey month (published on the penultimate working day) |
| services -> retail, industry | retail_m (hospitality), ip_m (transport) | Eurostat services turnover, y/y log change | month + 65 days |
| construction -> housing, credit | housing_m (ES), credit_m | construction output y/y (constr_m's own series); the EC construction confidence (ES) | month + 55 days; survey month |
| freight -> ports | ports_m | German lorry-toll mileage index, monthly mean and y/y | month + 12 days |
| procurement -> construction | constr_m | TED works notices (EU weekly counts, mean per week of the month) and all notices by buyer country (DE FR ES IT PL NL CZ RO) | month + 3 days |
| consumer spending -> insolvency | insolv_m (GB) | UK debit card spending (spend_w), monthly mean, total and groceries | month + 35 days |
| city services <-> transit | transit_d, c311_d | NYC 311 requests (total, NYPD) into MTA ridership; MTA subway, bus and bridge-and-tunnel counts into 311 | 2 days |
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


def _bcs(kind: str) -> pd.DataFrame:
    b = _utc(HS.get_parquet("lab/raw/estat_bcs.parquet"))
    b = b[[c for c in b.columns if c.startswith(kind + ":")]]
    b.columns = [c.split(":")[1] for c in b.columns]
    return b


def _serv(kind: str) -> pd.DataFrame:
    s = _utc(HS.get_parquet("lab/raw/estat_services.parquet"))
    s = s[[c for c in s.columns if c.startswith(kind + ":")]]
    s.columns = [c.split(":")[1] for c in s.columns]
    return s


def drivers(fid: str, P, PN) -> list:
    Driver, D = PN.Driver, []
    if fid == "retail_m":
        D += [Driver("nl3_bcs_retail", "sentiment", _bcs("retail"), MONTH, note="EC retail confidence balance (sent_m's series)"),
              Driver("nl3_bcs_services", "sentiment", _bcs("services"), MONTH, note="EC services confidence balance (sent_m's series)"),
              Driver("nl3_hospitality_yoy", "services", _yoy(_serv("hospitality")), MONTH + 65 * DAY, note="Eurostat turnover of accommodation and food service, y/y log change %")]
    if fid == "ip_m":
        D += [Driver("nl3_bcs_services", "sentiment", _bcs("services"), MONTH, note="EC services confidence balance (sent_m's series)"),
              Driver("nl3_transport_yoy", "services", _yoy(_serv("transport")), MONTH + 65 * DAY, note="Eurostat turnover of transport and storage, y/y log change %")]
    if fid == "housing_m":
        c = _utc(HS.get_parquet("lab/raw/estat_constr.parquet"))
        D += [Driver("nl3_constr_yoy", "construction", _yoy(c[["ES"]]), MONTH + 55 * DAY, note="Eurostat construction output volume, Spain, y/y log change %"),
              Driver("nl3_bcs_construction", "sentiment", _bcs("construction")[["ES"]], MONTH, note="EC construction confidence balance, Spain")]
    if fid == "credit_m":
        c = _utc(HS.get_parquet("lab/raw/estat_constr.parquet"))
        c = c.drop(columns=[k for k in c.columns if k not in set(PN.EU27)], errors="ignore")
        D.append(Driver("nl3_constr_yoy", "construction", _yoy(c), MONTH + 55 * DAY, note="Eurostat construction output volume, y/y log change %, per country"))
    if fid == "ports_m":
        t = _utc(HS.get_parquet("lab/raw/lkw_toll_daily.parquet"))["toll:DE"]
        m = t.resample("MS").mean().where(t.resample("MS").count() >= 27).dropna().to_frame("*")
        D += [Driver("nl3_toll", "freight", m, MONTH + 12 * DAY, note="Destatis lorry-toll mileage index, monthly mean"),
              Driver("nl3_toll_yoy", "freight", _yoy(m), MONTH + 12 * DAY)]
    if fid == "constr_m":
        t = _utc(HS.get_parquet("lab/raw/ted_weekly.parquet"))
        wk = t.resample("MS").mean().where(t.resample("MS").count() >= 4)      # mean weekly count of the Monday-stamped weeks in the month (4 or 5 of them); a short last month is dropped
        D.append(Driver("nl3_ted_works", "services", np.log1p(wk[["works:EU"]]).rename(columns={"works:EU": "*"}), MONTH + 3 * DAY,
                        note="TED works notices, mean per week of the month, log: calls for works precede construction output"))
        cc = {"de": "DE", "fr": "FR", "es": "ES", "it": "IT", "pl": "PL", "nl": "NL", "cz": "CZ", "ro": "RO"}
        D.append(Driver("nl3_ted_country", "services", np.log1p(wk[[f"{k}:EU" for k in cc]].rename(columns={f"{k}:EU": v for k, v in cc.items()})), MONTH + 3 * DAY,
                        note="TED notices, mean per week of the month, by buyer country, log"))
    if fid == "insolv_m":
        sp = _utc(HS.get_parquet("lab/raw/ons_revolut_weekly.parquet"))[["total:GB", "groceries:GB"]].resample("MS").mean()
        D += [Driver("nl3_spend_total", "economy", sp[["total:GB"]].rename(columns={"total:GB": "GB"}), MONTH + 35 * DAY, note="UK debit card spending index (spend_w), monthly mean"),
              Driver("nl3_spend_groceries", "economy", sp[["groceries:GB"]].rename(columns={"groceries:GB": "GB"}), MONTH + 35 * DAY)]
    if fid == "transit_d":
        c = _utc(HS.get_parquet("lab/raw/nyc311_daily.parquet"))
        D += [Driver("nl3_c311_total", "services", c.sum(axis=1, min_count=5).to_frame("*"), 2 * DAY, note="NYC 311 requests per day, all agencies"),
              Driver("nl3_c311_nypd", "services", c[["NYPD"]].rename(columns={"NYPD": "*"}), 2 * DAY, note="NYC 311 requests routed to the NYPD (noise, parking)")]
    if fid == "c311_d":
        m = _utc(HS.get_parquet("lab/raw/mta_daily.parquet"))
        D += [Driver(f"nl3_mta_{k}", "transit", m[[f"{k}:US"]].rename(columns={f"{k}:US": "*"}), 2 * DAY + 6, note=f"MTA {k} daily count")
              for k in ("subway", "bus", "bridges_tunnels")]
    return D
