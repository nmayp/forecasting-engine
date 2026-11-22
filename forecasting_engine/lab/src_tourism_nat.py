"""National early releases of tourist nights for tourism_m.

tourism_m's target (Eurostat tour_occ_nim) is known 73 days after the month: at a 1st-of-month origin only month m-4 is in. National statistical
offices publish the same survey (the Eurostat series is built from it) much sooner, so their latest month is a nowcast of the months Eurostat
has not published yet. Each driver is stamped at its month and known at its national release (the lag below); the back-test uses the latest
vintage of the national series, as it does for Eurostat (revisions are small: the same survey).

    python -m forecasting_engine.lab.src_tourism_nat           # load and print a summary

| Store name | Country | Source | Real lag used | Licence |
|---|---|---|---|---|
| lab/raw/tourism_nat.parquet (column ES) | Spain | INE, Encuesta de Ocupacion Hotelera, series EOT25 (hotel nights, all categories, all residences; servicios.ine.es Tempus3 API), 1999- | month end + 26 days (INE calendar 2026: day 21-26 after the month, the latest Dec 2025 on 26 Jan 2026; August 2026 on 22 Sep): month m-2 at the origin; the store keeps the revised values, the first release is provisional | INE aviso legal: reuse of INE statistics allowed, including commercial, citing "Fuente: INE" (Ley 37/2007 reuse; INE terms) |
| same (column PT) | Portugal | INE Portugal, indicator 0010415 (nights in tourist accommodation, all residences; ine.pt json_indicador API), 2013- | month end + 35 days (the full release is about 30 days after the month: August 2026 on 30 Sep 2026; the flash estimate at ~15 days is not used): month m-3 | INE Portugal: free reuse of disseminated statistics with the source cited ("Fonte: INE, Statistics Portugal") |
| same (column FI) | Finland | Statistics Finland, table 11iz (nights spent, all accommodation establishments, whole country; PxWeb API), 1995- | month end + 32 days (2026 releases: June on 30 Jul, July on 27 Aug, August on 24 Sep): month m-3 | CC BY 4.0, "Source: Statistics Finland" |

Hotels only for Spain (the EOH; apartments, campsites and rural lodgings are separate surveys published later): about 70 % of Spain's nights
in tour_occ_nim, and the drivers are year-on-year changes, which move together.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

ES_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_SERIE/EOT25"
PT_URL = "https://www.ine.pt/ine/json_indicador/pindica.jsp"
FI_URL = "https://statfin.stat.fi/PxWeb/api/v1/en/StatFin/matk/11iz.px"
LAG_DAYS = {"ES": 26, "PT": 35, "FI": 32}   # worst-case release delays seen in 2026 (see the table)
MONTHS_PT = {"Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10,
             "Novembro": 11, "Dezembro": 12}
ATTR = {"ES": "Fuente: INE (Encuesta de Ocupacion Hotelera). Adapted by the forecasting engine (year-on-year changes).",
        "PT": "Fonte: INE, Statistics Portugal (Inquerito a permanencia de hospedes). Adapted by the forecasting engine (year-on-year changes).",
        "FI": "Source: Statistics Finland, Accommodation statistics (CC BY 4.0). Adapted by the forecasting engine (year-on-year changes)."}


def _es() -> pd.Series:
    d = _get(ES_URL, params={"nult": 1000}).json()["Data"]
    return pd.Series({pd.Timestamp(year=int(x["Anyo"]), month=int(x["FK_Periodo"]), day=1, tz="UTC"): float(x["Valor"]) for x in d
                      if x.get("Valor") is not None and 1 <= int(x["FK_Periodo"]) <= 12}).sort_index()


def _pt() -> pd.Series:
    j = _get(PT_URL, params={"op": 2, "varcd": "0010415", "lang": "PT", "Dim1": "T", "Dim2": "PT", "Dim3": "T"}).json()[0]
    out = {}
    for per, rows in j["Dados"].items():
        mname, _, yr = per.partition(" de ")
        for r in rows:
            if r.get("dim_3") == "T" and r.get("geocod") == "PT" and r.get("valor") not in (None, ""):
                out[pd.Timestamp(year=int(yr), month=MONTHS_PT[mname], day=1, tz="UTC")] = float(r["valor"])
    return pd.Series(out).sort_index()


def _fi() -> pd.Series:
    import requests
    q = {"query": [{"code": "maakunta_26_20190101", "selection": {"filter": "item", "values": ["SSS"]}},
                   {"code": "majoitus_3_20181130", "selection": {"filter": "item", "values": ["01"]}},
                   {"code": "valtio_51_19950101", "selection": {"filter": "item", "values": ["A"]}},
                   {"code": "contentscode", "selection": {"filter": "item", "values": ["yopyneet"]}}], "response": {"format": "json"}}
    r = requests.post(FI_URL, json=q, timeout=120)
    r.raise_for_status()
    out = {}
    for row in r.json()["data"]:
        t, v = row["key"][-1], row["values"][0]
        if v not in ("", ".", "..", None):
            out[pd.Timestamp(year=int(t[:4]), month=int(t[5:]), day=1, tz="UTC")] = float(v)
    return pd.Series(out).sort_index()


def load(log=print) -> dict:
    df = pd.DataFrame({"ES": _es(), "PT": _pt(), "FI": _fi()}).sort_index()
    df.index.name = "date"
    HS.put_parquet("lab/raw/tourism_nat.parquet", df)
    out = {c: (str(df[c].first_valid_index().date()), str(df[c].last_valid_index().date())) for c in df.columns}
    log(f"national tourist nights: {out}")
    return out


def features() -> dict[str, tuple[pd.DataFrame, float, str]]:
    """name -> (frame, avail_h, note); one column per country, each with its own lag (one driver per country and transform)."""
    df = HS.get_parquet("lab/raw/tourism_nat.parquet")
    if df is None:
        return {}
    ix = pd.DatetimeIndex(df.index)
    df.index = ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")
    lx = np.log(df.where(df > 0))
    out = {}
    for c in df.columns:
        av = 30.44 * 24.0 + LAG_DAYS[c] * 24.0
        nt = f"{ATTR[c]} Known month end + {LAG_DAYS[c]} days"
        yoy = (lx[c] - lx[c].shift(12)).to_frame(c)
        out[f"nat_nights_yoy_{c.lower()}"] = (yoy, av, f"{nt}: log change vs the same month a year before")
        out[f"nat_nights_yoy3_{c.lower()}"] = (yoy.rolling(3, min_periods=2).mean(), av, f"{nt}: the same, mean of the last 3 months")
    # the same countries in ONE driver (columns ES, FI: equal 26-day lag), so the pooled trees learn a single feature from several series
    # (appended after the per-country drivers above; PT keeps its own, its lag differs)
    cc = [c for c in ("ES", "FI") if c in df.columns]
    if cc:                                                # one lag for the driver: the slower of the two (Finland, 32 days)
        yoy = lx[cc] - lx[cc].shift(12)
        av, nt = 30.44 * 24.0 + 32 * 24.0, "INE Spain (EOH hotels) and Statistics Finland (CC BY 4.0) nights spent; known month end + 32 days"
        out["nat_nights_yoy_pool"] = (yoy, av, f"{nt}: log change vs the same month a year before")
        out["nat_nights_yoy3_pool"] = (yoy.rolling(3, min_periods=2).mean(), av, f"{nt}: the same, mean of the last 3 months")
    return out


if __name__ == "__main__":
    print(load())
    for k, (x, av, nt) in features().items():
        print(k, x.shape, x.dropna().index[-1].date(), round(av / 24, 1))
    sys.exit(0)
