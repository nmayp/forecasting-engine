"""Cross-layer links, upstream port calls and cargo-type mix for choke_w, ignition / human-activity inputs for wildfire_w,
supply-chain pressure, house prices and labour costs for cpi_long_m.

    python -m forecasting_engine.lab.src_x6 load        # PortWatch ports, NY Fed GSCPI, OECD house prices, Eurostat labour cost index -> lab/raw/x6_*.parquet
    python -m forecasting_engine.lab.panel extend choke_w wildfire_w cpi_long_m

| Link (layer -> layer) | Families | Input | Real publication lag (worst case used) | Licence |
|---|---|---|---|---|
| transport (upstream port calls) -> transport | choke_w | weekly port calls (all and cargo) at 38 of the world's largest container, bulk and transhipment ports, summed by region (East Asia, South-East Asia and India, Gulf and Red Sea, North Europe, Mediterranean, Americas): ships that call there reach a strait one to three weeks later | week + 10 days (same feed and lag as the target) | IMF PortWatch (Kpler / UN Global Platform AIS): same terms as the target, back-tests only |
| transport (cargo mix) -> transport | choke_w | per strait: container share, tanker share and mean capacity per transit of the week | week + 10 days | the same |
| calendar, society (ignition) -> environment | wildfire_w | weekday public holidays in the target week and the one before (known years ahead), Monday / Friday holidays (long weekends), school-holiday share of the week and its change (OpenHolidays) | calendar: known ahead; school: published months ahead | holidays package (MIT); OpenHolidays ODbL |
| transport (supply chains) -> economy | cpi_long_m | NY Fed Global Supply Chain Pressure Index, level and 3-month change (global) | month + 7 days (released the first working days of the next month; the series is REVISED each month, the stored copy is the latest vintage: an upper bound) | NY Fed, free with citation |
| housing -> economy | cpi_long_m | OECD real house price index (deflated by consumer prices), y/y %, by country (quarterly) | quarter + 100 days | OECD (CC BY 4.0) |
| labour -> economy | cpi_long_m | Eurostat labour cost index (business economy, total, calendar and seasonally adjusted), y/y %, EU members | quarter + 80 days | Commission reuse policy (CC BY 4.0 equivalent) |

Lightning (a natural-ignition driver) was looked for and not used: Blitzortung's licence forbids commercial use and the ECMWF / ERA5 CAPE fields are
not in the store; the fire-weather and drought drivers of x2 / x5 stay the dryness side. Population exposure is static per country; the pooled
models scale every series by its own history, so a constant adds nothing a country effect does not already carry.
"""
from __future__ import annotations

import io
import sys
import time

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK, MONTH = 24.0, 24.0 * 7, 24.0 * 30.44
PORT_PREFIXES = {
    "east_asia": ["Shanghai", "Ningbo", "Shenzhen", "Guangzhou", "Qingdao", "Tianjin", "Xiamen", "Busan", "Hong Kong", "Kaohsiung", "Yokohama", "Tokyo"],
    "sea_india": ["Singapore", "Port Klang", "Tanjung Pelepas", "Laem Chabang", "Ho Chi Minh", "Colombo", "Mundra", "Nhava Sheva"],
    "gulf_red": ["Jebel Ali", "Jeddah", "Salalah", "Port Said"],
    "n_europe": ["Rotterdam", "Antwerp", "Hamburg", "Bremerhaven", "Felixstowe"],
    "med": ["Piraeus", "Algeciras", "Valencia", "Tanger"],
    "americas": ["Los Angeles", "Long Beach", "New York", "Savannah", "Houston", "Santos"],
}
PORTS_STORE, GSCPI_STORE, HPI_STORE, LCI_STORE = ("lab/raw/x6_ports_global.parquet", "lab/raw/x6_gscpi.parquet", "lab/raw/x6_oecd_hpi.parquet",
                                                   "lab/raw/x6_labour_cost.parquet")


# ------------------------------------------------------------------------------------------------------------------------ loaders
def load_ports(log=print) -> dict:
    import requests
    from forecasting_engine.lab.src_leading import PORTS_URL, _arcgis_all
    names = [p for v in PORT_PREFIXES.values() for p in v]
    ids: dict = {}
    for i in range(0, len(names), 6):          # resolve the port ids on one day (a filter on the day keeps the query valid and fast)
        like = " OR ".join(f"portname LIKE '{n}%'" for n in names[i:i + 6])
        for k in range(4):
            j = requests.get(PORTS_URL, params=dict(where=f"({like}) AND date=DATE '2024-03-05'", outFields="portid,portname", returnGeometry="false", f="json"),
                             timeout=300).json()
            if "error" not in j:
                break
            time.sleep(10 * (k + 1))
        for f in j.get("features", []):
            ids[f["attributes"]["portid"]] = f["attributes"]["portname"]
    log(f"x6 ports resolved: {len(ids)}")
    inl = ",".join(f"'{k}'" for k in ids)
    parts = []
    for y in range(2019, pd.Timestamp.now("UTC").year + 1):
        parts.append(_arcgis_all(PORTS_URL, f"portid IN ({inl}) AND year={y}", ("date", "portid", "portname", "ISO3", "portcalls", "portcalls_cargo"), log, order=False))
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "portid"])
    d["date"] = pd.to_datetime(d["date"], utc=True)
    reg = {n: r for r, v in PORT_PREFIXES.items() for n in v}
    d["region"] = [next((reg[n] for n in reg if str(p).startswith(n)), None) for p in d["portname"]]
    d = d.dropna(subset=["region"]).sort_values(["date", "portid"]).set_index("date")
    HS.put_parquet(PORTS_STORE, d)
    return dict(rows=len(d), ports=int(d["portid"].nunique()), first=str(d.index.min().date()), last=str(d.index.max().date()))


def load_gscpi() -> dict:
    from forecasting_engine.lab.sources import _get
    r = _get("https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx", ua=False)
    x = pd.read_excel(io.BytesIO(r.content), sheet_name="GSCPI Monthly Data", header=None)
    x = x.iloc[:, :2].dropna()
    x.columns = ["date", "value"]
    x["date"] = pd.to_datetime(x["date"], errors="coerce", format="mixed")
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna()
    x["date"] = x["date"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    HS.put_parquet(GSCPI_STORE, x.set_index("date"))
    return dict(rows=len(x), last=str(x["date"].max().date()))


def load_hpi() -> dict:
    import requests
    url = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_RHPI_TARGET@DF_RHPI_TARGET,1.0/all"
    r = requests.get(url, params=dict(startPeriod="1999-Q1", dimensionAtObservation="AllDimensions"), timeout=600,
                     headers={"Accept": "application/vnd.sdmx.data+csv; charset=utf-8", "User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_csv(io.BytesIO(r.content))
    d = d[(d["REF_AREA_TYPE"] == "COU") & (d["FREQ"] == "Q") & (d["MEASURE"] == "RHPI") & (d["UNIT_MEASURE"] == "PA") & (d["TRANSFORMATION"] == "GY") & (d["VINTAGE"] == "_T")
          & (d["DWELLINGS"] == "_T") & (d["ADJUSTMENT"] == "N")]
    q = d["TIME_PERIOD"].astype(str).str.replace(r"^(\d{4})-Q(\d)$", lambda m: f"{m.group(1)}-{int(m.group(2)) * 3 - 2:02d}-01", regex=True)
    out = pd.DataFrame({"date": pd.to_datetime(q, utc=True), "geo": d["REF_AREA"].astype(str), "value": d["OBS_VALUE"].astype(float)}).dropna().set_index("date")
    HS.put_parquet(HPI_STORE, out)
    return dict(rows=len(out), geos=int(out["geo"].nunique()), last=str(out.index.max().date()))


def load_lci() -> dict:
    from forecasting_engine.eurostat import parse_jsonstat
    from forecasting_engine.lab.sources import _get
    j = _get(HS.ESTAT_BASE + "lc_lci_r2_q", params=dict(format="JSON", lang="en", sinceTimePeriod="2000-Q1", lcstruct="D1_D4_MD5", nace_r2="B-S",
                                                          s_adj="SCA", unit="I20")).json()
    d = parse_jsonstat(j).dropna(subset=["value"])
    d = d[d["time"].astype(str).str.match(r"^\d{4}-Q\d$")]
    q = d["time"].str.replace(r"^(\d{4})-Q(\d)$", lambda m: f"{m.group(1)}-{int(m.group(2)) * 3 - 2:02d}-01", regex=True)
    out = pd.DataFrame({"date": pd.to_datetime(q, utc=True), "geo": d["geo"].astype(str), "value": d["value"].astype(float)}).set_index("date")
    HS.put_parquet(LCI_STORE, out)
    return dict(rows=len(out), geos=int(out["geo"].nunique()), last=str(out.index.max().date()))


# ------------------------------------------------------------------------------------------------------------------------ helpers
def _utc(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    i = pd.DatetimeIndex(x.index)
    x.index = i.tz_convert("UTC") if i.tz is not None else i.tz_localize("UTC")
    return x.sort_index()


def _wk(x: pd.DataFrame, how="sum", min_days=7) -> pd.DataFrame:
    r = x.resample("W-MON", label="left", closed="left")
    full = x.notna().resample("W-MON", label="left", closed="left").sum() >= min_days
    return (r.sum(min_count=1) if how == "sum" else r.mean()).where(full)


# ------------------------------------------------------------------------------------------------------------------------ choke_w
def choke_drivers(PN, cols) -> list:
    Driver, D = PN.Driver, []
    p = HS.get_parquet(PORTS_STORE)
    av = WEEK + 10 * DAY
    if p is not None and len(p):
        p = p.reset_index()
        p["date"] = pd.to_datetime(p["date"], utc=True)
        for v, tag in (("portcalls", "calls"), ("portcalls_cargo", "cargo")):
            w = p.pivot_table(index="date", columns="region", values=v, aggfunc="sum").sort_index()
            wk = _wk(w, "sum", 7)
            lw = np.log(wk.where(wk > 0))
            for reg in lw.columns:
                D.append(Driver(f"x6_port_{tag}_{reg}", "transport", lw[[reg]].rename(columns={reg: "*"}), av,
                                note=f"IMF PortWatch: weekly port {tag} summed over the {reg} ports, log (week + 10 days)"))
                if tag == "cargo":
                    D.append(Driver(f"x6_port_{tag}_{reg}_chg4", "transport", (lw[reg] - lw[reg].shift(4)).to_frame("*"), av,
                                    note=f"the same, 4-week log change"))
    c = HS.get_parquet("lab/raw/portwatch_chokepoints_daily.parquet").reset_index()
    c["date"] = pd.to_datetime(c["date"], utc=True)
    r = lambda v: c.pivot_table(index="date", columns="portname", values=v, aggfunc="sum").sort_index()   # noqa: E731
    tot, cont, tank = _wk(r("n_total")), _wk(r("n_container")), _wk(r("n_tanker"))
    cap = _wk(r("capacity"))
    keep = [s for s in cols if s in tot.columns]
    D.append(Driver("x6_container_share", "transport", (cont / tot.where(tot > 0))[keep].clip(0, 1), av, note="PortWatch: container ships' share of the strait's weekly transits"))
    D.append(Driver("x6_tanker_share", "transport", (tank / tot.where(tot > 0))[keep].clip(0, 1), av, note="PortWatch: tankers' share of the strait's weekly transits"))
    D.append(Driver("x6_mean_capacity", "transport", np.log((cap / tot.where(tot > 0))[keep].where(lambda x: x > 0)), av, note="PortWatch: log mean capacity (dwt) per transit"))
    return D


# ------------------------------------------------------------------------------------------------------------------------ wildfire_w
def wildfire_drivers(PN, ccs) -> list:
    import holidays as HL
    from forecasting_engine.lab import src_society as SS
    Driver, D = PN.Driver, []
    ahead = -365 * DAY
    hol = SS.holidays_weekly(ccs)
    hol.index = _utc(hol).index
    idx = hol.index
    long_w = {}
    for cc in ccs:                                   # holidays on a Monday or Friday: the week has a long weekend
        try:
            h = HL.country_holidays(cc, years=range(2015, 2028))
        except Exception:  # noqa: BLE001
            continue
        days = pd.DatetimeIndex(sorted(pd.Timestamp(x) for x in h.keys())).tz_localize("UTC")
        days = days[(days.dayofweek == 0) | (days.dayofweek == 4)]
        mon = days - pd.to_timedelta(days.dayofweek, unit="D")
        long_w[cc] = pd.Series(1.0, index=mon).groupby(level=0).sum().reindex(idx, fill_value=0.0)
    D.append(Driver("x6_pub_hol", "calendar", hol, ahead, ahead=True, note="weekday public holidays in the target week (holidays package)"))
    D.append(Driver("x6_pub_hol_prev", "calendar", hol.shift(1), ahead, ahead=True, note="weekday public holidays in the week before"))
    D.append(Driver("x6_pub_hol_next", "calendar", hol.shift(-1), ahead, ahead=True, note="weekday public holidays in the week after (people leave for it)"))
    if long_w:
        D.append(Driver("x6_long_weekend", "calendar", pd.DataFrame(long_w), ahead, ahead=True, note="Monday or Friday public holidays in the target week (long weekends)"))
    sch = HS.get_parquet("lab/raw/school_holidays.parquet")
    if sch is not None:
        sw = _wk(_utc(sch).reindex(columns=ccs), "mean", 1)
        D.append(Driver("x6_school_share", "calendar", sw, -180 * DAY, ahead=True, note="share of the country's regions on school holiday, weekly mean (OpenHolidays, ODbL)"))
        D.append(Driver("x6_school_chg", "calendar", sw - sw.shift(1), -180 * DAY, ahead=True, note="school-holiday share, this week minus last week"))
    return D


# ------------------------------------------------------------------------------------------------------------------------ cpi_long_m
def cpi_drivers(PN) -> list:
    from forecasting_engine.lab.sources import ISO3
    Driver, D = PN.Driver, []
    inv = {v: k for k, v in ISO3.items()}

    def to3(w: pd.DataFrame) -> pd.DataFrame:
        w = w.rename(columns={c: inv.get({'EL': 'GR', 'UK': 'GB'}.get(c, c), c) for c in w.columns})
        return w[[c for c in w.columns if len(c) == 3 and c in ISO3]]

    g = HS.get_parquet(GSCPI_STORE)
    if g is not None:
        s = _utc(g)["value"].astype(float)
        D.append(Driver("x6_gscpi", "transport", s.to_frame("*"), MONTH + 7 * DAY, note="NY Fed Global Supply Chain Pressure Index (latest vintage: revised each month), month + 7 days"))
        D.append(Driver("x6_gscpi_chg3", "transport", (s - s.shift(3)).to_frame("*"), MONTH + 7 * DAY, note="the same, 3-month change"))
    h = HS.get_parquet(HPI_STORE)
    if h is not None:
        w = _utc(h.reset_index().pivot_table(index="date", columns="geo", values="value", aggfunc="last"))
        w = w[[c for c in w.columns if c in ISO3]]
        D.append(Driver("x6_hpi_yoy", "housing", w.clip(-40, 60), 3 * MONTH + 100 * DAY, note="OECD real house price index (deflated by consumer prices), y/y %, quarter start stamp, known quarter + 100 days"))
        D.append(Driver("x6_hpi_chg4", "housing", (w - w.shift(4)).clip(-40, 40), 3 * MONTH + 100 * DAY, note="the same, change of the y/y rate over four quarters"))
    l = HS.get_parquet(LCI_STORE)
    if l is not None:
        w = _utc(l.reset_index().pivot_table(index="date", columns="geo", values="value", aggfunc="last"))
        w = to3(w)
        y = (100 * np.log(w / w.shift(4))).clip(-20, 40)
        D.append(Driver("x6_lci_yoy", "labour", y, 3 * MONTH + 80 * DAY, note="Eurostat labour cost index, business economy, y/y log change %, quarter + 80 days; EU members"))
        D.append(Driver("x6_lci_chg4", "labour", (y - y.shift(4)).clip(-20, 20), 3 * MONTH + 80 * DAY, note="the same, change of the y/y rate over four quarters"))
    return D


def drivers(fid: str, P, PN) -> list:
    if fid == "choke_w":
        c = HS.get_parquet("lab/raw/portwatch_chokepoints_daily.parquet")
        return choke_drivers(PN, sorted(set(c["portname"])))
    if fid == "wildfire_w":
        fam = PN.load(fid)
        return wildfire_drivers(PN, list(fam.Y.columns))
    if fid == "cpi_long_m":
        return cpi_drivers(PN)
    return []


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["load"]:
        what = a[1:] or ["gscpi", "hpi", "lci", "ports"]
        for k in what:
            try:
                print(k, {"gscpi": load_gscpi, "hpi": load_hpi, "lci": load_lci, "ports": load_ports}[k](), flush=True)
            except Exception as e:  # noqa: BLE001
                print(k, "FAILED", type(e).__name__, str(e)[:300], flush=True)
