"""Energy -> industry link for ip_m: gas delivered from the transmission grid straight to final consumers (large industrial sites and
gas power plants), per country and day, from the ENTSOG Transparency Platform. Energy use leads output: the month just ended is known at the
forecast origin, while industrial production is known only to m-2.

    python -m forecasting_engine.lab.src_gas            # downloads 2015- (one request per country and year, 6 in parallel)

| Store name | Content | Source, delay | Licence |
|---|---|---|---|
| lab/raw/entsog_final_consumers.parquet | daily GWh/d per ISO2 country: sum over TSOs of indicator "Physical Flow", direction exit, adjacent system "Final Consumers" | ENTSOG Transparency Platform API v1 AggregatedData (transparency.entsog.eu); daily flows are published the next day (operators revise for weeks): a whole month is used 3 days after it ends | ENTSOG Transparency Platform data: free re-use with attribution ("Source: ENTSOG Transparency Platform") |

Caveats: the platform API serves only the last five years (older data is archived: "Data on the TP is archived on a rolling basis"), so
the series starts 2021 and covers the ip_m holdout window only (none of development); 17 of the 25 countries report final consumers.
Operators join or leave the platform (level breaks), so the panel uses the y/y change of the month mean only where both months have the
same number of reporting TSOs; latest vintage, not the first publication (revisions are small for physical flows).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from forecasting_engine import histories as HS

URL = "https://transparency.entsog.eu/api/v1/AggregatedData"
CCS = ("AT", "BE", "BG", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "NL", "PL", "PT", "RO", "SE",
       "SI", "SK")
LAG_DAYS = 3.0
ATTR = "Source: ENTSOG Transparency Platform (aggregated physical flows to final consumers). Aggregated by the forecasting engine per country and month."


def _year(cc: str, y: int) -> list[dict]:
    import requests
    to = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp.now().normalize())
    params = {"countryKey": cc, "periodType": "day", "from": f"{y}-01-01", "to": f"{to:%Y-%m-%d}", "limit": "-1", "directionKey": "exit",
              "adjacentSystemsKey": "Final Consumers", "indicator": "Physical Flow"}
    for i in range(5):
        try:
            r = requests.get(URL, params=params, timeout=600)
            if r.status_code == 404:                        # no data that year (the API answers 404 for an empty result)
                return []
            r.raise_for_status()
            return [dict(date=x["periodFrom"][:10], cc=cc, op=x.get("operatorKey"), kwh=x.get("value")) for x in r.json().get("AggregatedData", [])
                    if x.get("value") is not None]
        except Exception:  # noqa: BLE001
            time.sleep(2 ** (i + 2))
    return []


def gas_load(years=range(2015, 2027), log=print) -> dict:
    jobs = [(cc, y) for cc in CCS for y in years]
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(lambda a: _year(*a), jobs))
    df = pd.DataFrame([r for p in parts for r in p])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(["date", "cc", "op"], keep="last")
    g = df.groupby(["date", "cc"]).agg(gwh=("kwh", lambda v: v.sum() / 1e6), n_tso=("op", "nunique")).reset_index()
    HS.put_parquet("lab/raw/entsog_final_consumers.parquet", g.set_index("date"))
    out = dict(rows=len(g), countries=int(g["cc"].nunique()), first=str(g["date"].min().date()), last=str(g["date"].max().date()))
    log(f"entsog final consumers: {out}")
    return out


def monthly_yoy() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(month mean GWh/d per country, y/y % where the month and the month a year before have the same TSO count), index = month start UTC."""
    g = HS.get_parquet("lab/raw/entsog_final_consumers.parquet")
    if g is None:
        return None
    g = g.reset_index()
    g["date"] = pd.to_datetime(g["date"], utc=True)
    g["m"] = g["date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    lv = g.pivot_table(index="m", columns="cc", values="gwh", aggfunc="mean").sort_index()
    nt = g.pivot_table(index="m", columns="cc", values="n_tso", aggfunc="max").sort_index()
    cnt = g.pivot_table(index="m", columns="cc", values="gwh", aggfunc="count").sort_index()
    lv = lv.where(cnt >= 25)                                  # complete months only
    lv = lv[lv.index < pd.Timestamp.now(tz="UTC").normalize().replace(day=1)]
    same = nt.eq(nt.shift(12))
    yoy = ((lv / lv.shift(12) - 1) * 100).where(same).replace([float("inf"), float("-inf")], float("nan")).clip(-80, 200)
    return lv, yoy


if __name__ == "__main__":
    gas_load()
    sys.exit(0)
