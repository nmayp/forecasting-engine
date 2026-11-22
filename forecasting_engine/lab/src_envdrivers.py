"""More drivers for the water, wildfire and air-quality families: rain over each Spanish river basin, weather at the
EU capitals, and satellite active-fire detections per country.

    python -m forecasting_engine.lab.src_envdrivers            # or: basins | capitals | firms

| Store name | Content | Source, delay |
|---|---|---|
| lab/raw/basin_power_daily.parquet | daily rain (mm) and mean 2 m temperature per Spanish hydrographic basin ('rain:duero:ES', 't2m:duero:ES'), mean of 2-5 points in the basin's reservoir zone, 1985- | NASA POWER daily API (MERRA-2 / GEOS-IT), about 2 days of latency |
| lab/raw/capital_power_daily.parquet | daily mean 10 m wind (m/s), rain (mm), relative humidity (%) and 2 m temperature at each EU capital ('ws10:FR', ...), 2014- | NASA POWER daily API, about 2 days of latency |
| lab/raw/firms_modis_daily.parquet | daily count and summed fire radiative power (MW) of MODIS active-fire detections per country, vegetation fires only (type 0), 2005-2025 | NASA FIRMS country archive (MODIS C6.1, Terra + Aqua); the live feed (NRT) is out within 3 hours; the archive files stop at the last processed year |

Why these: reservoir storage is the balance of inflow (rain over the basin) and of human draw (irrigation, cities, tourism,
hydro power); rain per basin is the inflow side, the other side is in tourism nights, hydro and demand drivers. Satellite fire detections
show the fires burning NOW, which the EFFIS mapped burnt area publishes a week later; most European fires are lit by people (land clearing,
holidays, negligence), so the calendar and school holidays go with them. Capital weather (wind, rain) is the dispersion side of air quality;
traffic, heating and industry are the emission side (calendar, temperature, energy and economy drivers).

Licences.
- NASA POWER: NASA Earth science data, no restriction on use, commercial use allowed; acknowledgement "These data were obtained from the NASA
  Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program." (src_leading.POWER_ATTR)
- NASA FIRMS: NASA Earth Science Data and Information Policy (open, no restriction, commercial use allowed); citation asked: "We acknowledge
  the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part
  of NASA's Earth Science Data and Information System (ESDIS)." MODIS Collection 6.1 (DOI 10.5067/FIRMS/MODIS/MCD14ML). The country yearly
  archive needs no key; the near-real-time API needs a free MAP_KEY (not available here): 2026 is missing in the back-test (NaN, so the
  models fall back to the other drivers there; honest, slightly pessimistic).
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FIRMS_ATTR = ("We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) "
              "(https://earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information System (ESDIS).")
FIRMS_ZIP = "https://firms.modaps.eosdis.nasa.gov/data/country/zips/modis_{y}_all_countries.zip"
POWER_LAG_DAYS = 4.0           # observed 20 Oct 2026 08 UTC: the last day served was 17 Oct (3.3 days after its start, 18 Oct not yet out); 4 days after the day ends = worst case with margin
FIRMS_LAG_DAYS = 1.0           # NRT detections within 3 hours; a day is complete the next morning

# reservoir zones of each basin (approximate, lat, lon): 2-5 points where the big reservoirs and their catchments are
BASIN_POINTS = {
    "cantabrico_occidental": [(43.2, -6.0), (43.1, -4.5), (43.3, -5.3)],
    "cantabrico_oriental": [(43.1, -2.5), (43.2, -1.9), (43.2, -3.2)],
    "cuenca_mediterranea_andaluza": [(36.9, -4.4), (36.8, -3.2), (36.6, -5.1)],
    "cuencas_internas_de_cataluna": [(41.9, 2.4), (42.0, 1.8), (41.6, 1.6)],
    "cuencas_internas_del_pais_vasco": [(43.2, -2.9), (43.1, -2.2)],
    "duero": [(41.8, -5.5), (42.5, -4.5), (41.5, -3.5), (41.0, -6.4)],
    "ebro": [(42.6, -0.5), (42.0, -1.5), (42.3, 1.0), (41.3, -0.5), (42.9, -3.8)],
    "galicia_costa": [(42.8, -8.6), (43.2, -8.2), (42.4, -8.4)],
    "guadalete_barbate": [(36.7, -5.6), (36.3, -5.8)],
    "guadalquivir": [(37.9, -3.0), (38.1, -4.5), (37.4, -5.6), (37.2, -3.5)],
    "guadiana": [(39.0, -3.5), (38.9, -5.5), (38.5, -6.8), (39.3, -4.8)],
    "jucar": [(39.5, -1.0), (40.0, -1.8), (38.9, -0.6)],
    "mino_sil": [(42.4, -7.2), (42.7, -6.6), (42.2, -8.0)],
    "segura": [(38.1, -1.8), (38.2, -2.6), (37.8, -1.2)],
    "tajo": [(40.3, -3.7), (40.5, -2.5), (39.8, -5.3), (40.2, -6.2), (40.8, -3.4)],
    "tinto_odiel_y_piedras": [(37.5, -6.9), (37.7, -6.5)],
}
CAPITALS = {"AT": (48.208, 16.373), "BE": (50.847, 4.357), "BG": (42.698, 23.322), "HR": (45.815, 15.982), "CY": (35.173, 33.364),
            "CZ": (50.075, 14.438), "DK": (55.676, 12.568), "EE": (59.437, 24.754), "FI": (60.170, 24.938), "FR": (48.857, 2.352),
            "DE": (52.520, 13.405), "GR": (37.984, 23.728), "HU": (47.498, 19.040), "IE": (53.350, -6.260), "IT": (41.903, 12.496),
            "LV": (56.950, 24.105), "LT": (54.687, 25.280), "LU": (49.612, 6.130), "MT": (35.899, 14.514), "NL": (52.370, 4.895),
            "PL": (52.230, 21.012), "PT": (38.722, -9.139), "RO": (44.427, 26.103), "SK": (48.149, 17.107), "SI": (46.057, 14.506),
            "ES": (40.417, -3.704), "SE": (59.329, 18.069)}
FIRMS_NAMES = {"Albania": "AL", "Austria": "AT", "Bosnia_and_Herzegovina": "BA", "Belgium": "BE", "Bulgaria": "BG", "Switzerland": "CH",
               "Cyprus": "CY", "Czech_Republic": "CZ", "Czechia": "CZ", "Germany": "DE", "Denmark": "DK", "Estonia": "EE", "Spain": "ES",
               "Finland": "FI", "France": "FR", "United_Kingdom": "GB", "Greece": "GR", "Croatia": "HR", "Hungary": "HU", "Ireland": "IE",
               "Iceland": "IS", "Italy": "IT", "Lithuania": "LT", "Luxembourg": "LU", "Latvia": "LV", "Montenegro": "ME", "Macedonia": "MK",
               "North_Macedonia": "MK", "Malta": "MT", "Netherlands": "NL", "Norway": "NO", "Poland": "PL", "Portugal": "PT", "Romania": "RO",
               "Serbia": "RS", "Republic_of_Serbia": "RS", "Sweden": "SE", "Slovenia": "SI", "Slovakia": "SK", "Turkey": "TR", "Turkiye": "TR",
               "Ukraine": "UA", "Kosovo": "XK"}


def _ts(x) -> pd.DatetimeIndex:
    i = pd.DatetimeIndex(x)
    return i.tz_convert("UTC") if i.tz is not None else i.tz_localize("UTC")


def _power(lat: float, lon: float, params: str, start: str) -> pd.DataFrame:
    end = (pd.Timestamp.now("UTC") - pd.Timedelta(days=1)).strftime("%Y%m%d")
    r = _get(POWER_URL, params=dict(parameters=params, community="AG", latitude=lat, longitude=lon, start=start.replace("-", ""), end=end,
                                    format="JSON", **{"time-standard": "lst"}))
    d = pd.DataFrame(r.json()["properties"]["parameter"])
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    return d.replace(-999.0, np.nan)


def basins_load(workers: int = 3, log=print) -> dict:
    pts = [(b, la, lo) for b, ps in BASIN_POINTS.items() for la, lo in ps]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda p: (p[0], _power(p[1], p[2], "PRECTOTCORR,T2M", "1985-01-01")), pts))
    cols = {}
    for b in BASIN_POINTS:
        fr = [d for bb, d in res if bb == b]
        cols[f"rain:{b}:ES"] = pd.concat([d["PRECTOTCORR"] for d in fr], axis=1).mean(axis=1)
        cols[f"t2m:{b}:ES"] = pd.concat([d["T2M"] for d in fr], axis=1).mean(axis=1)
    out = pd.DataFrame(cols)
    out["rain:total:ES"] = out[[c for c in out if c.startswith("rain:")]].mean(axis=1)
    out["t2m:total:ES"] = out[[c for c in out if c.startswith("t2m:")]].mean(axis=1)
    out.index = _ts(out.index)
    out.index.name = "date"
    HS.put_parquet("lab/raw/basin_power_daily.parquet", out.astype("float32"))
    log(f"basins: {len(pts)} points")
    return dict(points=len(pts), first=str(out.index[0].date()), last=str(out["rain:total:ES"].last_valid_index().date()))


def capitals_load(workers: int = 3, log=print) -> dict:
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(lambda kv: (kv[0], _power(kv[1][0], kv[1][1], "WS10M,PRECTOTCORR,RH2M,T2M", "2014-01-01")), CAPITALS.items()))
    cols = {}
    for cc, d in res.items():
        for k, nm in (("WS10M", "ws10"), ("PRECTOTCORR", "rain"), ("RH2M", "rh"), ("T2M", "t2m")):
            cols[f"{nm}:{cc}"] = d[k]
    out = pd.DataFrame(cols)
    out.index = _ts(out.index)
    out.index.name = "date"
    HS.put_parquet("lab/raw/capital_power_daily.parquet", out.astype("float32"))
    return dict(capitals=len(res), first=str(out.index[0].date()), last=str(out.index[-1].date()))


def _firms_year(b: bytes, y: int) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(b))
    rows = []
    for n in z.namelist():
        nm = os.path.basename(n)[len(f"modis_{y}_"):-4] if n.endswith(".csv") else None
        cc = FIRMS_NAMES.get(nm or "")
        if not cc:
            continue
        d = pd.read_csv(z.open(n), usecols=lambda c: c in ("acq_date", "frp", "type"))
        if "type" in d:
            d = d[d["type"] == 0]
        g = d.groupby("acq_date").agg(n=("frp", "size"), frp=("frp", "sum")).reset_index()
        g["cc"] = cc
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def firms_load(work_dir: str | None = None, years=range(2005, 2026), log=print) -> dict:
    """One year at a time: download the zip (or use <work_dir>/m<year>.zip), aggregate to country-days, delete the zip; the per-year
    aggregates are kept in <work_dir>/agg_<year>.parquet (small) so a rerun resumes."""
    import time
    work_dir = work_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "firms")
    os.makedirs(work_dir, exist_ok=True)
    rows = []
    for y in years:
        ap = os.path.join(work_dir, f"agg_{y}.parquet")
        if not os.path.exists(ap):
            zp = os.path.join(work_dir, f"m{y}.zip")
            g = None
            for i in range(5):
                try:
                    b = open(zp, "rb").read() if os.path.exists(zp) else _get(FIRMS_ZIP.format(y=y)).content
                    g = _firms_year(b, y)
                    break
                except (zipfile.BadZipFile, OSError):
                    if os.path.exists(zp):
                        os.remove(zp)
                    time.sleep(10 * (i + 1))
            if os.path.exists(zp):
                os.remove(zp)
            if g is None:
                raise RuntimeError(f"FIRMS {y}: no valid zip")
            g.to_parquet(ap)
        rows.append(pd.read_parquet(ap))
        log(f"firms {y}: done")
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["acq_date"])
    w = df.pivot_table(index="date", columns="cc", values=["n", "frp"], aggfunc="sum")
    w.columns = [f"{a}:{b}" for a, b in w.columns]
    full = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    w = w.reindex(full).fillna(0.0)                 # a country-day without detections is a zero (inside the archive's span)
    w.index = _ts(w.index)
    w.index.name = "date"
    HS.put_parquet("lab/raw/firms_modis_daily.parquet", w.astype("float32"))
    return dict(countries=len({c.split(':')[1] for c in w.columns}), first=str(w.index[0].date()), last=str(w.index[-1].date()))


# ------------------------------------------------------------------------------------------------------------------------------ readers
def _wk(x: pd.DataFrame, how: str = "sum", min_days: int = 7) -> pd.DataFrame:
    r = x.resample("W-MON", label="left", closed="left")
    v = r.sum(min_count=1) if how == "sum" else r.mean()
    return v.where(r.count() >= min_days)


def basin_weekly(var: str = "rain", weeks: int = 1) -> pd.DataFrame | None:
    """Weekly (Mon-Sun, stamped on the Monday) rain sum or mean temperature per basin series id ('duero:ES'); weeks > 1: rolling sum/mean."""
    d = HS.get_parquet("lab/raw/basin_power_daily.parquet")
    if d is None:
        return None
    d = d[[c for c in d.columns if c.startswith(f"{var}:")]].astype(float)
    d.columns = [c.split(":", 1)[1] for c in d.columns]
    d.index = _ts(d.index)
    w = _wk(d, "sum" if var == "rain" else "mean")
    if weeks > 1:
        w = w.rolling(weeks, min_periods=weeks).sum() if var == "rain" else w.rolling(weeks, min_periods=weeks).mean()
    return w


def basin_rain_anomaly(weeks: int = 13) -> pd.DataFrame | None:
    """Rain over the last `weeks` weeks minus the mean of the same calendar weeks in the PREVIOUS years only (expanding, causal): an SPI-like
    drought index per basin, mm."""
    w = basin_weekly("rain", weeks)
    if w is None:
        return None
    woy = w.index.isocalendar().week.clip(upper=52).to_numpy()
    out = w.copy()
    for k in np.unique(woy):
        m = woy == k
        sub = w[m]
        out[m] = sub - sub.shift(1).expanding(min_periods=3).mean()
    return out


def capital_daily(var: str) -> pd.DataFrame | None:
    d = HS.get_parquet("lab/raw/capital_power_daily.parquet")
    if d is None:
        return None
    d = d[[c for c in d.columns if c.startswith(f"{var}:")]].astype(float)
    d.columns = [c.split(":", 1)[1] for c in d.columns]
    d.index = _ts(d.index)
    return d


def firms_weekly(var: str = "n") -> pd.DataFrame | None:
    """Weekly (Mon-Sun, stamped on the Monday) sum of MODIS vegetation-fire detections (n) or FRP (frp) per ISO2; NaN after the archive."""
    d = HS.get_parquet("lab/raw/firms_modis_daily.parquet")
    if d is None:
        return None
    d = d[[c for c in d.columns if c.startswith(f"{var}:")]].astype(float)
    d.columns = [c.split(":", 1)[1] for c in d.columns]
    d.index = _ts(d.index)
    return _wk(d, "sum")


def daily_rolling(var: str, windows=(7,)) -> dict:
    """{window: daily-stamped rolling sum over the last `window` days} of basin rain ('rain', series ids) or FIRMS detections ('firms_n',
    log1p of the sum, ISO2; NaN after the archive)."""
    if var == "rain":
        d = HS.get_parquet("lab/raw/basin_power_daily.parquet")
        d = d[[c for c in d.columns if c.startswith("rain:")]].astype(float)
    else:
        d = HS.get_parquet("lab/raw/firms_modis_daily.parquet")
        d = d[[c for c in d.columns if c.startswith("n:")]].astype(float)
    d.columns = [c.split(":", 1)[1] for c in d.columns]
    d.index = _ts(d.index)
    out = {}
    for w in windows:
        x = d.rolling(w, min_periods=w).sum()
        out[w] = np.log1p(x) if var != "rain" else x
    return out


def main(argv=None) -> int:
    import json
    a = argv or sys.argv[1:] or ["all"]
    out = {}
    for k, fn in (("basins", basins_load), ("capitals", capitals_load)):
        if a[0] in (k, "all"):
            out[k] = fn()
    if a[0] in ("firms", "all"):
        out["firms"] = firms_load(a[1] if len(a) > 1 else None)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
