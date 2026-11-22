"""Lab v2 input histories that the history store (forecasting_engine.histories) did not have yet. Everything goes to the same store, under lab/raw/.

    python -m forecasting_engine.lab.sources all            # or: fred | ecb | estat | kev | gdelt

| Source | Store name | Content | Licence (for back-testing; public use checked per item before any product use) |
|---|---|---|---|
| FRED (St. Louis Fed) | lab/raw/fred.parquet | daily and monthly market series (long format: date, id, value) | public domain except the copyrighted items marked in FRED |
| ECB Data Portal | lab/raw/ecb_fx.parquet, ecb_rates.parquet, ecb_hicp.parquet | euro reference rates (all currencies, daily, 1999-), policy and money-market rates, HICP y/y per country | ECB reuse with acknowledgement |
| Eurostat | lab/raw/estat_unemployment.parquet | monthly unemployment rate (SA, % of labour force) per country | CC BY 4.0 equivalent |
| CISA KEV | lab/raw/kev.parquet | every catalogue entry with dateAdded (Nov 2021-) | US Government work (public domain) |
| GDELT 2.0 events | gdelt/events_daily-*.parquet (BigQuery export, see build_gdelt_sql) | daily event counts per country and CAMEO root, with strikes split out, by the day the events were added (DATEADDED, UTC; since 21 Oct 2026) | GDELT: free use with citation |
"""
from __future__ import annotations

import io
import json
import os
import sys

import pandas as pd

from forecasting_engine import histories as HS

UA = {"User-Agent": "forecasting-engine/0.1"}

FRED_DAILY = {
    "DCOILBRENTEU": "Brent crude, USD/bbl", "DCOILWTICO": "WTI crude, USD/bbl", "DHHNGSP": "Henry Hub gas, USD/MMBtu",
    "DGS10": "US 10-year Treasury yield, %", "DGS2": "US 2-year Treasury yield, %", "T10YIE": "US 10-year breakeven inflation, %",
    "DTWEXBGS": "Broad US dollar index", "VIXCLS": "VIX equity volatility (CBOE, copyrighted)",
    "BAMLH0A0HYM2": "US high-yield spread (ICE BofA, copyrighted)", "DEXCHUS": "CNY per USD",
}
FRED_MONTHLY = {
    "PNGASEUUSDM": "Natural gas, Europe, USD/MMBtu (IMF)", "PCOPPUSDM": "Copper, USD/t (IMF)", "PWHEAMTUSDM": "Wheat, USD/t (IMF)",
    "PMAIZMTUSDM": "Maize, USD/t (IMF)", "PALUMUSDM": "Aluminium, USD/t (IMF)", "PIORECRUSDM": "Iron ore, USD/t (IMF)",
    "GEPUCURRENT": "Global economic policy uncertainty index", "PCU325311325311": "US producer prices, nitrogen fertilisers",
}

ECB = "https://data-api.ecb.europa.eu/service/data/"
ECB_RATES = {"FM/B.U2.EUR.4F.KR.MRR_FR.LEV": "ECB main refinancing rate, %", "FM/B.U2.EUR.4F.KR.DFR.LEV": "ECB deposit facility rate, %",
             "FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA": "Euribor 3 months, %"}

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# GDELT 2.0 events, aggregated in BigQuery (about 41 GB scanned, USD 0.26) and exported to the store
GDELT_COUNTRIES = {  # FIPS 10-4 (GDELT) -> ISO 3166 alpha-2
    "SP": "ES", "PO": "PT", "FR": "FR", "GM": "DE", "IT": "IT", "NL": "NL", "BE": "BE", "PL": "PL", "AU": "AT", "EI": "IE", "GR": "GR", "SW": "SE",
    "DA": "DK", "FI": "FI", "EZ": "CZ", "RO": "RO", "HU": "HU", "LO": "SK", "BU": "BG", "HR": "HR", "SI": "SI", "LH": "LT", "LG": "LV", "EN": "EE",
    "LU": "LU", "MT": "MT", "CY": "CY", "UK": "GB", "RS": "RU", "UP": "UA", "US": "US", "CH": "CN", "IR": "IR", "IZ": "IQ", "SA": "SA", "EG": "EG",
    "YM": "YE", "IS": "IL", "TU": "TR"}


# Keyed on DATEADDED (lag audit, 21 Oct 2026): `d` is the UTC day the event was ADDED to GDELT, not its event date (SQLDATE). Counting by
# SQLDATE over every later DATEADDED partition let events added days later (1-1.5 % of a day's events) revise days a back-test had
# already read; by DATEADDED a day's value is complete at the end of that day and never changes afterwards (docs/methods.md).
GDELT_DAY = "CAST(DIV(DATEADDED, 1000000) AS INT64)"


def build_gdelt_sql() -> str:
    cc = ",".join(f"'{c}'" for c in GDELT_COUNTRIES)
    return f"""SELECT {GDELT_DAY} AS d, ActionGeo_CountryCode AS cc, EventRootCode AS root,
  IF(EventCode IN ('143','1431','1432','1433','1434'), 'strike', IF(EventRootCode='14','protest', IF(EventRootCode IN ('18','19','20'),'violence',
     IF(EventRootCode IN ('16','17'),'coerce', 'other')))) AS kind,
  COUNT(*) AS n, SUM(NumMentions) AS mentions, AVG(GoldsteinScale) AS goldstein, AVG(AvgTone) AS tone
FROM `gdelt-bq.gdeltv2.events_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('2015-02-18') AND ActionGeo_CountryCode IN ({cc})
GROUP BY d, cc, root, kind"""


def _get(url: str, ua: bool = True, **kw):
    import time
    import requests
    for i in range(5):
        try:
            r = requests.get(url, headers=UA if ua else None, timeout=300, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == 4:
                raise
            time.sleep(2 ** (i + 1))


def fred_load(log=print) -> dict:
    rows = []
    for sid in {**FRED_DAILY, **FRED_MONTHLY}:
        r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", ua=False, params={"id": sid})   # FRED refuses our agent string
        d = pd.read_csv(io.BytesIO(r.content))
        d.columns = ["date", "value"]
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna()
        d["id"] = sid
        rows.append(d)
        log(f"fred {sid}: {len(d)} rows from {d['date'].iloc[0]}")
    out = pd.concat(rows)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    HS.put_parquet("lab/raw/fred.parquet", out.set_index("date"))
    return dict(series=len(rows), rows=len(out))


def _ecb(key: str, since: str = "1999-01-01") -> pd.DataFrame:
    flow, k = key.split("/", 1)
    d = pd.read_csv(io.BytesIO(_get(f"{ECB}{flow}/{k}", params={"format": "csvdata", "startPeriod": since}).content))
    return d


def ecb_load(log=print) -> dict:
    fx = _ecb("EXR/D..EUR.SP00.A")
    fx = pd.DataFrame({"date": pd.to_datetime(fx["TIME_PERIOD"], utc=True), "id": fx["CURRENCY"], "value": pd.to_numeric(fx["OBS_VALUE"], errors="coerce")}).dropna()
    HS.put_parquet("lab/raw/ecb_fx.parquet", fx.set_index("date"))
    log(f"ecb fx: {fx['id'].nunique()} currencies, {len(fx)} rows")
    rr = []
    for k in ECB_RATES:
        d = _ecb(k)
        t = d["TIME_PERIOD"].astype(str)
        rr.append(pd.DataFrame({"date": pd.to_datetime(t.where(t.str.len() > 7, t + "-01"), utc=True), "id": k, "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")}))
    rates = pd.concat(rr).dropna()
    HS.put_parquet("lab/raw/ecb_rates.parquet", rates.set_index("date"))
    h = _ecb("ICP/M..N.000000.4.ANR", "2000-01")
    hicp = pd.DataFrame({"date": pd.to_datetime(h["TIME_PERIOD"] + "-01", utc=True), "geo": h["REF_AREA"], "value": pd.to_numeric(h["OBS_VALUE"], errors="coerce")}).dropna()
    HS.put_parquet("lab/raw/ecb_hicp.parquet", hicp.set_index("date"))
    log(f"ecb rates {len(rates)} rows; hicp {hicp['geo'].nunique()} areas")
    return dict(fx=len(fx), rates=len(rates), hicp=len(hicp))


ESTAT = {"unemployment": ("une_rt_m", dict(s_adj="SA", age="TOTAL", sex="T", unit="PC_ACT")),
         "hicp": ("prc_hicp_manr", dict(coicop="CP00", unit="RCH_A")),       # HICP annual rate (the ECB country series stop at the 2025 rebase)
         # leading indicators (business and consumer surveys, published at the END of the month they describe; producer prices)
         "esi": ("ei_bssi_m_r2", dict(indic="BS-ESI-I", s_adj="SA")),
         "consumer_conf": ("ei_bssi_m_r2", dict(indic="BS-CSMCI-BAL", s_adj="SA")),
         "industry_conf": ("ei_bssi_m_r2", dict(indic="BS-ICI-BAL", s_adj="SA")),
         "ppi": ("sts_inppd_m", dict(nace_r2="B-E36", s_adj="NSA", unit="I21", indic_bt="PRC_PRR_DOM")),
         # the HICP index itself (monthly changes give the base effects of the y/y rate) and its energy component
         "hicp_index": ("prc_hicp_midx", dict(coicop="CP00", unit="I15")),
         "hicp_energy_index": ("prc_hicp_midx", dict(coicop="NRG", unit="I15"))}

OXCGRT_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_national_v1.csv"
ISO3 = {"AUT": "AT", "BEL": "BE", "BGR": "BG", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE", "ESP": "ES", "FIN": "FI", "FRA": "FR",
        "GRC": "GR", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ITA": "IT", "LTU": "LT", "LUX": "LU", "LVA": "LV", "MLT": "MT", "NLD": "NL", "POL": "PL",
        "PRT": "PT", "ROU": "RO", "SWE": "SE", "SVN": "SI", "SVK": "SK", "GBR": "GB", "UKR": "UA", "TUR": "TR", "USA": "US", "CHN": "CN"}


def oxcgrt_load(log=print) -> dict:
    """Oxford COVID-19 Government Response Tracker (CC BY 4.0): daily stringency and economic support per country, 2020-2022."""
    d = pd.read_csv(io.BytesIO(_get(OXCGRT_URL).content), usecols=["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average",
                                                                     "EconomicSupportIndex", "ConfirmedCases"], low_memory=False)
    d = d[(d["Jurisdiction"] == "NAT_TOTAL") & d["CountryCode"].isin(ISO3)]
    out = pd.DataFrame({"date": pd.to_datetime(d["Date"].astype(str), format="%Y%m%d", utc=True), "geo": d["CountryCode"].map(ISO3),
                        "stringency": d["StringencyIndex_Average"], "support": d["EconomicSupportIndex"], "cases": d["ConfirmedCases"]})
    HS.put_parquet("lab/raw/oxcgrt.parquet", out.set_index("date"))
    log(f"oxcgrt: {out['geo'].nunique()} countries, {len(out)} rows, {out['date'].min().date()} - {out['date'].max().date()}")
    return dict(rows=len(out))


def estat_load(log=print) -> dict:
    from forecasting_engine.eurostat import parse_jsonstat
    out = {}
    for k, (ds, flt) in ESTAT.items():
        j = _get(HS.ESTAT_BASE + ds, params=dict(format="JSON", lang="en", sinceTimePeriod="2000-01", **flt)).json()
        d = parse_jsonstat(j).dropna(subset=["value"])
        d = d[d["time"].str.match(r"^\d{4}-\d{2}$")]
        d = pd.DataFrame({"date": pd.to_datetime(d["time"] + "-01", utc=True), "geo": d["geo"].astype(str), "value": d["value"].astype(float)})
        HS.put_parquet(f"lab/raw/estat_{k}.parquet", d.set_index("date"))
        log(f"estat {k}: {d['geo'].nunique()} geos, {len(d)} rows, last {d['date'].max().date()}")
        out[k] = len(d)
    return out


def kev_load(log=print) -> dict:
    j = _get(KEV_URL).json()
    d = pd.DataFrame(j["vulnerabilities"])
    d["date"] = pd.to_datetime(d["dateAdded"], utc=True)
    d = d[["date", "cveID", "vendorProject", "product", "knownRansomwareCampaignUse"]].astype({"cveID": str, "vendorProject": str, "product": str,
                                                                                            "knownRansomwareCampaignUse": str})
    HS.put_parquet("lab/raw/kev.parquet", d.set_index("date"))
    log(f"kev: {len(d)} entries from {d['date'].min().date()}")
    return dict(rows=len(d))


_GD: dict = {}


def gdelt_frame() -> pd.DataFrame:
    """The BigQuery export, ISO country codes, daily timestamps (UTC)."""
    if "d" in _GD:
        return _GD["d"]
    pre = os.environ.get("FE_GDELT_PREFIX", "gdelt/")     # the SQLDATE-keyed export before 21 Oct 2026 is archived under gdelt_sqldate/
    parts = [HS.get_parquet(n) for n in HS.list_names(pre)]
    d = pd.concat([p.reset_index(drop=True) for p in parts if p is not None])
    d["date"] = pd.to_datetime(d["d"].astype(str), format="%Y%m%d", utc=True)
    d["cc"] = d["cc"].map(GDELT_COUNTRIES)
    _GD["d"] = d.drop(columns=["d"]).dropna(subset=["cc"])
    return _GD["d"]


OECD_PRICES = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
OECD_ATTR = "OECD Data Explorer (Prices, all items), OECD terms: free use with attribution"


def oecd_load(log=print) -> dict:
    """consumer prices for every OECD country since 1960, y/y and the index, plus monthly WTI since 1946
    (FRED WTISPLC), so models can learn energy-to-prices pass-through from 1973, 1979, 1990, 2008 instead of the calm 2010s only."""
    out = {}
    for tr, key in (("GY", "yoy"), ("_Z", "index")):
        url = OECD_PRICES + f".M.N.CPI.{'PA' if tr == 'GY' else 'IX'}._T.N.{tr}"
        r = _get_browser(url, {"startPeriod": "1960-01", "format": "csvfilewithlabels"})
        d = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        d = pd.DataFrame({"date": pd.to_datetime(d["TIME_PERIOD"] + "-01", utc=True), "geo": d["REF_AREA"].astype(str),
                          "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")}).dropna()
        HS.put_parquet(f"lab/raw/oecd_cpi_{key}.parquet", d.set_index("date"))
        out[key] = dict(rows=len(d), geos=int(d["geo"].nunique()), first=str(d["date"].min().date()), last=str(d["date"].max().date()))
        log(f"oecd cpi {key}: {out[key]}")
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", ua=False, params={"id": "WTISPLC"})
    w = pd.read_csv(io.BytesIO(r.content))
    w.columns = ["date", "value"]
    w["date"] = pd.to_datetime(w["date"], utc=True)
    w["value"] = pd.to_numeric(w["value"], errors="coerce")
    HS.put_parquet("lab/raw/fred_wti_monthly.parquet", w.dropna().set_index("date"))
    out["wti_monthly"] = len(w)
    return out


OECD_LONG = {   # long monthly histories for the families that failed after 2021 (unemployment, industry)
    "ip": ("OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3", ".M.PRVM.IX.BTE.Y._Z._Z.N"),       # industrial production, SA index
    "cli": ("OECD.SDD.STES,DSD_STES@DF_CLI,4.1", ".M.LI.IX._Z.AA.IX._Z.H"),            # composite leading indicator
    "bci": ("OECD.SDD.STES,DSD_STES@DF_CLI,4.1", ".M.BCICP.IX._Z.AA.IX._Z.H"),         # business confidence
    "cci": ("OECD.SDD.STES,DSD_STES@DF_CLI,4.1", ".M.CCICP.IX._Z.AA.IX._Z.H"),         # consumer confidence
    "unemployment": ("OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0", "..._Z.Y._T.Y_GE15..M"),  # harmonised unemployment rate, SA
}


def oecd_long_load(log=print) -> dict:
    out = {}
    for k, (flow, key) in OECD_LONG.items():
        r = _get_browser(f"https://sdmx.oecd.org/public/rest/data/{flow}/{key}", {"startPeriod": "1955-01", "format": "csvfilewithlabels"})
        d = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        d = pd.DataFrame({"date": pd.to_datetime(d["TIME_PERIOD"] + "-01", utc=True), "geo": d["REF_AREA"].astype(str),
                          "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")}).dropna()
        d = d.groupby(["date", "geo"], as_index=False)["value"].last()
        HS.put_parquet(f"lab/raw/oecd_{k}.parquet", d.set_index("date"))
        out[k] = dict(rows=len(d), geos=int(d["geo"].nunique()), first=str(d["date"].min().date()), last=str(d["date"].max().date()))
        log(f"oecd {k}: {out[k]}")
    return out


ENTSOG = "https://transparency.entsog.eu/api/v1/operationaldatas"
# Russian pipeline gas into the EU, daily physical flow (entry), ENTSOG Transparency Platform (free re-use with attribution)
GAS_ROUTES = {
    "nord_stream": [("ITP-00247", "NEL"), ("ITP-00251", "OPAL")],       # Greifswald, split between the two onshore pipelines
    "ukraine_transit": [("ITP-00117", "eustream")],                      # Velke Kapusany (UA -> SK)
    "yamal": [("ITP-00104", "GAZ-SYSTEM")],                              # Kondratki (BY -> PL)
    "turkstream": [("ITP-00549", "Bulgartransgaz")],                     # Strandzha 2 (TR -> BG)
    "finland": [("ITP-00024", "Gasgrid")],                               # Imatra (RU -> FI)
}


def entsog_load(years=range(2015, 2027), log=print) -> dict:
    import requests
    rows = []
    for route, pts in GAS_ROUTES.items():
        for key, op in pts:
            for y in years:
                params = {"from": f"{y}-01-01", "to": f"{y}-12-31", "indicator": "Physical Flow", "periodType": "day", "pointKey": key,
                          "directionKey": "entry", "limit": "-1"}
                for i in range(4):
                    try:
                        r = requests.get(ENTSOG, params=params, timeout=300)
                        if r.status_code == 404:
                            js = {}
                            break
                        r.raise_for_status()
                        js = r.json()
                        break
                    except requests.RequestException:
                        import time
                        time.sleep(2 ** (i + 2))
                        js = {}
                d = [x for x in (js.get("operationaldatas") or []) if op.lower() in (x.get("operatorLabel") or "").lower()]
                for x in d:
                    rows.append(dict(date=x["periodFrom"][:10], route=route, point=key, gwh=(x.get("value") or 0) / 1e6))
            log(f"entsog {route} {key}: {sum(1 for r in rows if r['point'] == key)} days")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    daily = df.groupby(["date", "route"])["gwh"].sum().unstack().sort_index()
    HS.put_parquet("lab/raw/entsog_russian_gas.parquet", daily)
    return dict(days=len(daily), routes=list(daily.columns), first=str(daily.index.min().date()), last=str(daily.index.max().date()))


def _get_browser(url: str, params: dict):
    """OECD's API refuses non-browser agents through the proxy."""
    import time
    import requests
    for i in range(5):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == 4:
                raise
            time.sleep(2 ** (i + 1))


def main(argv=None) -> int:
    what = (argv or sys.argv[1:] or ["all"])[0]
    out = {}
    for k, fn in (("fred", fred_load), ("ecb", ecb_load), ("estat", estat_load), ("kev", kev_load), ("oxcgrt", oxcgrt_load), ("oecd", oecd_load), ("oecd_long", oecd_long_load), ("entsog", entsog_load)):
        if what in (k, "all"):
            out[k] = fn()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
