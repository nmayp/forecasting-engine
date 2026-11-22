"""Leading inputs for two layers where the lab only ties Chronos-2: maritime trade (ports_m) and wildfire (wildfire_w).

    python -m forecasting_engine.lab.src_leading            # or: ports | chokepoints | fwi

| Store name | Content | Source, delay |
|---|---|---|
| lab/raw/portwatch_ports_daily.parquet | daily port calls (all, cargo ships) and import / export volume estimates (metric tons) per port, ISO2 'geo', 2019- | IMF PortWatch Daily_Ports_Data (AIS, UN Global Platform / Kpler), updated weekly on Tuesdays 9 AM ET, data up to the previous ~Friday |
| lab/raw/portwatch_chokepoints_daily.parquet | daily transit calls and transit capacity (deadweight-based volume estimate, t) per chokepoint (28), 2019- | IMF PortWatch Daily_Chokepoints_Data, same cadence |
| lab/raw/fwi_points_daily.parquet | daily inputs per sample point (country, weight, T2M_MAX, RH at Tmax, WS10M, PRECTOTCORR) | NASA POWER daily API (MERRA-2 + GEOS-IT), about 2 days of latency |
| lab/raw/fwi_country_daily.parquet | daily Canadian FWI system per country ('fwi:ES', 'dc:ES', 'dmc:ES', 'isi:ES', 'ffmc:ES'), weighted by burnable area | computed here (Van Wagner 1987) |

PortWatch. Monthly sums per country only over complete months (the month's last day is in the data). The weekly release covers days up to
about 4-5 days before it, so a month is complete at most ~12 days after it ends: avail_h = one month + 12 days from the month-start stamp.
History was backfilled (Oct 2024) and is revised with methodology changes: the back-test uses the latest vintage (slightly optimistic).
Licence: IMF Copyright and Usage, "The Use of IMF Data" special terms (imf.org/en/About/copyright-and-terms): Data may be downloaded,
transformed and used with attribution, material transformation stated; "for any potential commercial reuse of IMF Data, please email
the IMF"; the PortWatch FAQ asks the same for commercial redistribution, and the vessel data come from Kpler via the UN Global
Platform. So: back-tests and internal research now; ask the IMF before a paid product or trial uses it. Credit:
"Sources: Kpler; UN Global Platform; IMF PortWatch (portwatch.imf.org). Aggregated to country-month by the forecasting engine." The download uses the
official ArcGIS API (paged, 5000 records per page, one query per country and year), not a scraper.

Fire danger. The Copernicus CEMS fire danger reanalysis (CDS, CC BY 4.0) needs a CDS account and licence acceptance that this container does
not have, so the Canadian Forest Fire Weather Index System is computed here from NASA POWER daily weather at a few points per country (the
1x1 degree boxes with the most burnable land - tree + shrub + grass cover of the project grid - each box's weight its burnable area; Ukraine
and Bosnia, outside the grid, use fixed points with equal weights). Equations: Van Wagner, C.E. (1987) "Development and structure of the
Canadian Forest Fire Weather Index System", Forestry Technical Report 35, Canadian Forestry Service; as coded in Wang, Anderson & Suddaby
(2015) "Updated source code for calculating fire danger indices in the Canadian Forest Fire Weather Index System", NRCan Information Report
NOR-X-424 (day-length factors for latitudes >= 30 N). Approximations, stated: noon temperature = daily maximum (T2M_MAX), noon humidity = the
relative humidity at the daily maximum (from QV2M and surface pressure), wind = daily mean 10 m wind, rain = the day's total (UTC/LST day,
not noon to noon); start-up FFMC 85, DMC 6, DC 15 on 2004-01-01, computed through winter (no overwintering). It is a past/known value
(ahead=False): a week (Mon-Sun) is known on the Thursday after it (POWER ~2 days + 1 day margin): avail_h = one week + 4 days.
Licence: NASA Earth science data are open with no restriction on use (NASA Earth Science Data and Information Policy); POWER asks for the
acknowledgement "These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/
Applied Science Program." Commercial use allowed.
"""
from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

PW = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
PORTS_URL = f"{PW}/Daily_Ports_Data/FeatureServer/0/query"
CHOKE_URL = f"{PW}/Daily_Chokepoints_Data/FeatureServer/0/query"
PW_ATTR = "Sources: Kpler; UN Global Platform; IMF PortWatch (portwatch.imf.org). Aggregated to country-month by the forecasting engine."
PORT_FIELDS = ("portcalls", "portcalls_cargo", "import", "export")
CHOKE_FIELDS = ("n_total", "n_cargo", "n_container", "n_tanker", "capacity", "capacity_container", "capacity_tanker")
PORT_ISO3 = {"ESP": "ES"}             # ports_m is Spain's 29 port authorities; the lab matches drivers on the series' country ("ES")
CHOKEPOINTS = {"suez": "Suez Canal", "bab_el_mandeb": "Bab el-Mandeb Strait", "gibraltar": "Gibraltar Strait", "panama": "Panama Canal",
               "bosporus": "Bosporus Strait", "hormuz": "Strait of Hormuz", "cape": "Cape of Good Hope", "dover": "Dover Strait"}
PW_LAG_DAYS = 12.0

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = "T2M_MAX,QV2M,PS,WS10M,PRECTOTCORR"
POWER_ATTR = ("These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/"
              "Applied Science Program. Fire Weather Index computed by the forecasting engine (Van Wagner 1987).")
FWI_START = "2004-01-01"
FWI_LAG_DAYS = 4.0
GRID = os.environ.get("FE_GRID", "data/cells_europe.parquet")      # land-cover / population grid of Europe (parquet), not shipped
FIXED_POINTS = {"UA": [(50.45, 30.5), (51.4, 28.0), (49.8, 24.0), (48.6, 22.6), (46.6, 30.7), (48.5, 35.0), (50.0, 36.2), (47.0, 33.5)],
                "BA": [(44.0, 17.8), (44.75, 17.2), (43.35, 17.8)]}
NUTS_ISO = {"EL": "GR", "UK": "GB"}


def _ts(x) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(pd.to_datetime(x))
    return ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")


# ------------------------------------------------------------------------------------------------------------------------------ PortWatch
def _arcgis_all(url: str, where: str, fields: tuple, log=print, order: bool = True) -> pd.DataFrame:
    """Paged download (5000 records per page). order=False for filtered port queries: ordering (and server-side statistics) time out on the
    IMF server there; the hosted layer then pages in storage order, and the caller drops duplicates."""
    rows, off = [], 0
    while True:
        prm = dict(where=where, outFields=",".join(fields), returnGeometry="false", resultOffset=off, resultRecordCount=5000,
                   maxRecordCountFactor=5, f="json")
        if order:
            prm["orderByFields"] = "ObjectId"
        for i in range(4):
            j = _get(url, params=prm).json()
            if "error" not in j:
                break
            if i == 3:
                raise RuntimeError(f"PortWatch: {j['error']} [{where}, offset {off}]")
            time.sleep(15 * (i + 1))
        feats = [f["attributes"] for f in j.get("features", [])]
        rows += feats
        off += len(feats)
        if not feats or not j.get("exceededTransferLimit"):
            break
        time.sleep(0.5)                                       # gentle on the IMF server
    log(f"portwatch {url.split('/')[-4]} [{where}]: {len(rows)} rows")
    return pd.DataFrame(rows)


def ports_load(iso3=tuple(PORT_ISO3), log=print) -> dict:
    """Daily per-port records, one paged query per country and year (whole-country queries time out on the IMF server)."""
    parts = []
    for c in iso3:
        for y in range(2019, pd.Timestamp.now("UTC").year + 1):
            parts.append(_arcgis_all(PORTS_URL, f"ISO3='{c}' AND year={y}", ("date", "portid", "portname", "ISO3") + PORT_FIELDS, log, order=False))
    d = pd.concat(parts, ignore_index=True)
    n0 = len(d)
    d = d.drop_duplicates(["date", "portid"])
    d["date"] = _ts(d["date"])
    d["geo"] = d["ISO3"].map(PORT_ISO3)
    d = d.drop(columns="ISO3").sort_values(["date", "portid"])
    HS.put_parquet("lab/raw/portwatch_ports_daily.parquet", d.set_index("date"))
    return dict(rows=len(d), duplicates_dropped=n0 - len(d), ports=int(d["portid"].nunique()), first=str(d["date"].min().date()),
                last=str(d["date"].max().date()))


def chokepoints_load(log=print) -> dict:
    d = _arcgis_all(CHOKE_URL, "1=1", ("date", "portid", "portname") + CHOKE_FIELDS, log)
    d["date"] = _ts(d["date"])
    d = d.sort_values(["date", "portid"])
    HS.put_parquet("lab/raw/portwatch_chokepoints_daily.parquet", d.set_index("date"))
    return dict(rows=len(d), chokepoints=int(d["portid"].nunique()), first=str(d["date"].min().date()), last=str(d["date"].max().date()))


def _complete_months(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly sums stamped at the month start (UTC), only months whose last day is in the data."""
    last = daily.index.max()
    m = daily.resample("MS").sum(min_count=1)
    return m[m.index + pd.offsets.MonthEnd(0) <= last.normalize()]


def ports_monthly(metric: str = "portcalls") -> pd.DataFrame | None:
    """Wide: month start (UTC) x ISO2, sum over the country's ports of a daily metric (PORT_FIELDS); complete months only."""
    d = HS.get_parquet("lab/raw/portwatch_ports_daily.parquet")
    if d is None:
        return None
    d.index = _ts(d.index)
    w = d.pivot_table(index=d.index, columns="geo", values=metric, aggfunc="sum").sort_index()
    return _complete_months(w).astype(float)


def chokepoints_monthly(metric: str = "n_total") -> pd.DataFrame | None:
    """Wide: month start (UTC) x short chokepoint name (CHOKEPOINTS keys)."""
    d = HS.get_parquet("lab/raw/portwatch_chokepoints_daily.parquet")
    if d is None:
        return None
    d.index = _ts(d.index)
    inv = {v: k for k, v in CHOKEPOINTS.items()}
    d = d[d["portname"].isin(inv)]
    w = d.pivot_table(index=d.index, columns="portname", values=metric, aggfunc="sum").sort_index().rename(columns=inv)
    return _complete_months(w).astype(float)


# ------------------------------------------------------------------------------------------------------------------------------ FWI
def fwi_points() -> pd.DataFrame:
    """Sample points per country: the 1x1 degree boxes with the most burnable land (tree + shrub + grass), weight = burnable km2."""
    c = pd.read_parquet(GRID, columns=["lat", "lon", "nuts3", "area_km2", "land_share", "lc_tree", "lc_shrub", "lc_grass"])
    c["cc"] = c["nuts3"].astype(str).str[:2].replace(NUTS_ISO)
    c = c[c["cc"].str.fullmatch(r"[A-Z]{2}") & (c["cc"] != "LI")]
    c["w"] = c["area_km2"] * c["land_share"].fillna(1) * (c["lc_tree"].fillna(0) + c["lc_shrub"].fillna(0) + c["lc_grass"].fillna(0))
    c["box"] = np.floor(c["lat"]).astype(int).astype(str) + "_" + np.floor(c["lon"]).astype(int).astype(str)
    out = []
    for cc, g in c.groupby("cc"):
        b = g.groupby("box").apply(lambda x: pd.Series(dict(lat=np.average(x["lat"], weights=x["w"] + 1e-9), lon=np.average(x["lon"], weights=x["w"] + 1e-9),
                                                            w=x["w"].sum())), include_groups=False).sort_values("w", ascending=False)
        n = int(np.clip(round(np.sqrt(g["area_km2"].sum() / 2500.0)), 2, 10))
        b = b.head(n)
        out += [dict(cc=cc, lat=round(r.lat, 3), lon=round(r.lon, 3), w=float(r.w)) for r in b.itertuples()]
    for cc, pts in FIXED_POINTS.items():
        out += [dict(cc=cc, lat=la, lon=lo, w=1.0) for la, lo in pts]
    p = pd.DataFrame(out)
    p["point"] = p["cc"] + "_" + p.groupby("cc").cumcount().astype(str)
    return p


def _power(lat: float, lon: float, end: str) -> pd.DataFrame:
    r = _get(POWER_URL, params=dict(parameters=POWER_PARAMS, community="AG", latitude=lat, longitude=lon, start=FWI_START.replace("-", ""),
                                    end=end, format="JSON", **{"time-standard": "lst"}))
    p = r.json()["properties"]["parameter"]
    d = pd.DataFrame(p)
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    return d.replace(-999.0, np.nan)


def _rh_at_tmax(t: np.ndarray, q_gkg: np.ndarray, ps_kpa: np.ndarray) -> np.ndarray:
    q = q_gkg / 1000.0
    e = q * ps_kpa * 10.0 / (0.622 + 0.378 * q)                       # vapour pressure, hPa
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))                       # saturation at Tmax (Bolton 1980), hPa
    return np.clip(100.0 * e / es, 1.0, 100.0)


LE = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])      # DMC effective day length, lat >= 30 N
LF = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])      # DC day-length adjustment, lat > 20 N


def fwi_system(T, H, W, P, month, ffmc0=85.0, dmc0=6.0, dc0=15.0) -> dict:
    """Canadian FWI system (Van Wagner 1987; Wang et al. 2015), vectorised over points. T (C), H (%), W (km/h), P (mm): arrays (days x points);
    month: (days,) 1-12. Days with missing inputs carry the codes forward unchanged and give NaN indices."""
    nd, npt = T.shape
    out = {k: np.full((nd, npt), np.nan) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    ffmc, dmc, dc = np.full(npt, ffmc0), np.full(npt, dmc0), np.full(npt, dc0)
    for i in range(nd):
        t, h, w, ro = T[i], H[i], W[i], P[i]
        ok = np.isfinite(t) & np.isfinite(h) & np.isfinite(w) & np.isfinite(ro)
        t, h, w, ro = np.where(ok, t, 0.0), np.where(ok, h, 50.0), np.where(ok, w, 0.0), np.where(ok, ro, 0.0)
        mi = int(month[i]) - 1
        # FFMC
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        rf = ro - 0.5
        wet = ro > 0.5
        rfs = np.where(wet, rf, 1.0)
        add = 42.5 * rfs * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rfs))
        add = np.where(mo > 150.0, add + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rfs), add)
        mo = np.where(wet, np.minimum(mo + add, 250.0), mo)
        ed = 0.942 * h ** 0.679 + 11.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
        ew = 0.618 * h ** 0.753 + 10.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
        ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * t)
        k1 = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)
        kw = k1 * 0.581 * np.exp(0.0365 * t)
        m = np.where(mo > ed, ed + (mo - ed) * 10.0 ** (-kd), np.where(mo < ew, ew - (ew - mo) * 10.0 ** (-kw), mo))
        f_new = np.clip(59.5 * (250.0 - m) / (147.2 + m), 0.0, 101.0)
        # DMC
        tt = np.maximum(t, -1.1)
        rk = 1.894 * (tt + 1.1) * (100.0 - h) * LE[mi] * 1e-6
        re = 0.92 * ro - 1.27
        mo_d = 20.0 + np.exp(5.6348 - dmc / 43.43)
        lg = np.log(np.maximum(dmc, 1e-6))
        b = np.where(dmc <= 33.0, 100.0 / (0.5 + 0.3 * dmc), np.where(dmc <= 65.0, 14.0 - 1.3 * lg, 6.2 * lg - 17.2))
        mr = mo_d + 1000.0 * re / (48.77 + b * re)
        pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-6)), 0.0)
        d_new = np.maximum(np.where(ro > 1.5, pr, dmc), 0.0) + 100.0 * rk
        # DC
        td = np.maximum(t, -2.8)
        pe = np.maximum((0.36 * (td + 2.8) + LF[mi]) / 2.0, 0.0)
        rd = 0.83 * ro - 1.27
        qr = 800.0 * np.exp(-dc / 400.0) + 3.937 * rd
        dr = np.maximum(400.0 * np.log(800.0 / np.maximum(qr, 1e-6)), 0.0)
        c_new = np.where(ro > 2.8, dr, dc) + pe
        ffmc, dmc, dc = np.where(ok, f_new, ffmc), np.where(ok, d_new, dmc), np.where(ok, c_new, dc)
        # ISI, BUI, FWI
        fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        isi = 0.208 * np.exp(0.05039 * w) * 91.9 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)
        den = np.maximum(dmc + 0.4 * dc, 1e-9)
        bui = np.where(dmc <= 0.4 * dc, 0.8 * dmc * dc / den, dmc - (1.0 - 0.8 * dc / den) * (0.92 + (0.0114 * dmc) ** 1.7))
        bui = np.where((dmc == 0) & (dc == 0), 0.0, np.maximum(bui, 0.0))
        fd = np.where(bui <= 80.0, 0.626 * bui ** 0.809 + 2.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
        bb = 0.1 * isi * fd
        fwi = np.where(bb > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1.0 + 1e-12))) ** 0.647), bb)
        for k, v in (("ffmc", ffmc), ("dmc", dmc), ("dc", dc), ("isi", isi), ("bui", bui), ("fwi", fwi)):
            out[k][i] = np.where(ok, v, np.nan)
    return out


def fwi_load(workers: int = 3, log=print) -> dict:
    pts = fwi_points()
    end = (pd.Timestamp.now("UTC") - pd.Timedelta(days=1)).strftime("%Y%m%d")

    def one(r):
        d = _power(r.lat, r.lon, end)
        return r.point, d
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, list(pts.itertuples())))
    log(f"power: {len(res)} points read")
    idx = sorted(set().union(*[set(d.index) for d in res.values()]))
    idx = pd.DatetimeIndex(idx)
    cols = list(pts["point"])
    g = lambda k: np.column_stack([res[p][k].reindex(idx).to_numpy(float) for p in cols])   # noqa: E731
    T, Q, PS, W, P = g("T2M_MAX"), g("QV2M"), g("PS"), g("WS10M") * 3.6, g("PRECTOTCORR")
    H = _rh_at_tmax(T, Q, PS)
    raw = pd.concat({k: pd.DataFrame(v, index=idx, columns=cols) for k, v in (("t", T), ("rh", H), ("wind_kmh", W), ("rain", P))}, axis=1)
    raw.columns = [f"{a}:{b}" for a, b in raw.columns]
    HS.put_parquet("lab/raw/fwi_points_daily.parquet", raw.astype("float32"))
    HS.put_parquet("lab/raw/fwi_points.parquet", pts)
    S = fwi_system(T, H, W, P, idx.month.to_numpy())
    w = pts["w"].to_numpy(float)
    frames = {}
    for k in ("fwi", "dc", "dmc", "isi", "ffmc"):
        for cc in sorted(pts["cc"].unique()):
            j = np.flatnonzero(pts["cc"].to_numpy() == cc)
            v, ww = S[k][:, j], w[j]
            ok = np.isfinite(v)
            frames[f"{k}:{cc}"] = np.where(ok.any(1), np.nansum(np.where(ok, v, 0) * ww, 1) / np.maximum((ok * ww).sum(1), 1e-12), np.nan)
    out = pd.DataFrame(frames, index=_ts(idx))
    out.index.name = "date"
    HS.put_parquet("lab/raw/fwi_country_daily.parquet", out.astype("float32"))
    last = out["fwi:ES"].last_valid_index()
    return dict(points=len(pts), countries=int(pts["cc"].nunique()), first=str(out.index[0].date()), last=str(last.date() if last is not None else None))


def fwi_daily(index: str = "fwi") -> pd.DataFrame | None:
    """Wide: UTC day x ISO2 of one FWI-system index (fwi, dc, dmc, isi, ffmc)."""
    d = HS.get_parquet("lab/raw/fwi_country_daily.parquet")
    if d is None:
        return None
    d = d[[c for c in d.columns if c.startswith(f"{index}:")]].astype(float)
    d.columns = [c.split(":", 1)[1] for c in d.columns]
    d.index = _ts(d.index)
    return d


def fwi_weekly(index: str = "fwi", how: str = "mean") -> pd.DataFrame | None:
    """Weekly (Mon-Sun, stamped on the Monday like wildfire_w) mean or max; only complete weeks (7 valid days)."""
    d = fwi_daily(index)
    if d is None:
        return None
    r = d.resample("W-MON", label="left", closed="left")
    v = r.mean() if how == "mean" else r.max()
    return v.where(r.count() >= 7)


def main(argv=None) -> int:
    import json
    what = (argv or sys.argv[1:] or ["all"])[0]
    out = {}
    for k, fn in (("ports", ports_load), ("chokepoints", chokepoints_load), ("fwi", fwi_load)):
        if what in (k, "all"):
            out[k] = fn()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
