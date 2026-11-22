"""Long histories used by the forecasting lab: one compact parquet per (source, series, year) or per series, in a local store folder.

Store: $FE_HISTORY_STORE (default ./data/history), laid out as

    entsoe/<dataset>/<key>/year=YYYY.parquet     ENTSO-E Transparency, hourly UTC means (load, load_da, price, gen, flows) or outage events
    oil_bulletin/prices.parquet                  EU Weekly Oil Bulletin, weekly consumer prices with taxes per country (EUR per 1000 l)
    ports_es/monthly.parquet                     Puertos del Estado monthly traffic per port authority (t)
    eurostat/<dataset>.parquet                   monthly Eurostat indicators per country (industrial production, retail, tourism nights)
    weather/popw_daily.parquet                   population-weighted daily T mean / max, HDD, CDD per country (ERA5-type reanalysis)

Loaders are resumable (a chunk that exists is skipped; the current year is refreshed). Readers (`series(...)`) return a UTC-indexed float
pd.Series and memoise in process.

Licences: ENTSO-E data items on its CC-BY 4.0 free re-use list (day-ahead load forecast 6.1.b, physical flows 12.1.g, transmission
unavailability 10.1) can be used with attribution; actual load (6.1.a), day-ahead prices (12.1.d), actual generation per type (16.1.b/c) and
generation unavailability (15.1) are NOT on that list, so they are used for back-testing only unless FE_ALLOW_RESTRICTED=1 (see `restricted_ok`).

The ENTSO-E token is read from the environment variable ENTSOE_TOKEN only; it is never logged or stored, and error messages are scrubbed of
the request URL.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from forecasting_engine import store

DEFAULT_BASE = "data/history"
_MEMO: dict = {}
_LOCK = threading.Lock()


def base() -> str:
    return os.environ.get("FE_HISTORY_STORE", DEFAULT_BASE).rstrip("/")


def restricted_ok() -> bool:
    """ENTSO-E items outside its CC-BY list (actual load, prices, generation, generation outages): back-testing only unless a flag is set."""
    return os.environ.get("FE_ALLOW_RESTRICTED", "0") == "1"


# ------------------------------------------------------------------------------------------------------------------------------ store
def _uri(name: str) -> str:
    return f"{base()}/{name}"


def put_bytes(name: str, data: bytes, ctype: str = "application/octet-stream") -> None:
    store.write_bytes(_uri(name), data, ctype)


def get_bytes(name: str) -> bytes | None:
    try:
        return store.read_bytes(_uri(name))
    except Exception:  # noqa: BLE001  (absent)
        return None


def put_parquet(name: str, df: pd.DataFrame) -> int:
    buf = io.BytesIO()
    df.to_parquet(buf, compression="zstd", index=True)
    b = buf.getvalue()
    put_bytes(name, b)
    return len(b)


def get_parquet(name: str) -> pd.DataFrame | None:
    b = get_bytes(name)
    if not b:
        return None
    return pd.read_parquet(io.BytesIO(b))


def list_names(prefix: str) -> list[str]:
    """Object names (relative to the store) below a prefix, recursively."""
    return store.list_names(base(), prefix)


# ------------------------------------------------------------------------------------------------------------------------------ ENTSO-E
ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
COUNTRIES = ("ES", "PT", "FR", "DE", "IT", "NL", "BE", "PL")
AREA = {"ES": "10YES-REE------0", "PT": "10YPT-REN------W", "FR": "10YFR-RTE------C", "DE": "10Y1001A1001A83F", "IT": "10YIT-GRTN-----B",
        "NL": "10YNL----------L", "BE": "10YBE----------2", "PL": "10YPL-AREA-----S"}
# day-ahead price zones: DE-AT-LU until 30 Sep 2018, DE-LU after; Italy: the North zone (largest; the PUN is not on the platform)
PRICE_ZONE = {"ES": [("2015-01-01", "10YES-REE------0")], "PT": [("2015-01-01", "10YPT-REN------W")], "FR": [("2015-01-01", "10YFR-RTE------C")],
              "DE": [("2015-01-01", "10Y1001A1001A63L"), ("2018-10-01", "10Y1001A1001A82H")], "IT": [("2015-01-01", "10Y1001A1001A73I")],
              "NL": [("2015-01-01", "10YNL----------L")], "BE": [("2015-01-01", "10YBE----------2")], "PL": [("2015-01-01", "10YPL-AREA-----S")]}
BORDERS = (("ES", "PT"), ("ES", "FR"), ("FR", "BE"), ("FR", "DE"), ("FR", "IT"), ("BE", "NL"), ("BE", "DE"), ("DE", "NL"), ("DE", "PL"))
PSR = {"B01": "biomass", "B02": "lignite", "B03": "coal_gas", "B04": "gas", "B05": "hard_coal", "B06": "oil", "B07": "oil_shale", "B08": "peat",
       "B09": "geothermal", "B10": "hydro_pumped", "B11": "hydro_ror", "B12": "hydro_reservoir", "B13": "marine", "B14": "nuclear", "B15": "other_renewable",
       "B16": "solar", "B17": "waste", "B18": "wind_offshore", "B19": "wind_onshore", "B20": "other", "B25": "storage"}
#: dataset -> (licence class, description). "open" = on ENTSO-E's CC-BY 4.0 free re-use list (list of 18 Oct 2023); "restricted" = not on it.
ENTSOE_DATASETS = {
    "load": ("restricted", "actual total load (6.1.a), A65/A16"),
    "load_da": ("open", "day-ahead total load forecast (6.1.b), A65/A01"),
    "price": ("restricted", "day-ahead prices (12.1.d), A44"),
    "gen": ("restricted", "actual generation per production type (16.1.b/c), A75/A16"),
    "flows": ("open", "cross-border physical flows (12.1.g), A11"),
    "res_da": ("restricted", "day-ahead generation forecasts for wind and solar (14.1.d), A69/A01 - not confirmed on the free re-use list: back-tests only"),
    "out_gen": ("restricted", "unavailability of generation units (15.1.a-d), A80"),
    "out_tx": ("open", "unavailability of transmission infrastructure (10.1.a-b), A78"),
}
ATTR_ENTSOE = "ENTSO-E Transparency Platform (transparency.entsoe.eu); data items on the free re-use list under CC BY 4.0; values aggregated to hourly means by the forecasting engine"
_TOKEN: dict = {}


class EntsoeError(RuntimeError):
    pass


def _token() -> str:
    """The ENTSO-E token (env ENTSOE_TOKEN), in memory only. Never print or log it."""
    if "t" not in _TOKEN:
        t = os.environ.get("ENTSOE_TOKEN", "").strip()
        if not t:
            raise EntsoeError("no ENTSO-E token: set the environment variable ENTSOE_TOKEN")
        _TOKEN["t"] = t
    return _TOKEN["t"]


_RATE = {"t": [], "lock": threading.Lock()}


def _throttle(max_per_min: int = 300) -> None:
    with _RATE["lock"]:
        now = time.time()
        _RATE["t"] = [x for x in _RATE["t"] if now - x < 60]
        if len(_RATE["t"]) >= max_per_min:
            time.sleep(60 - (now - _RATE["t"][0]) + 0.1)
        _RATE["t"].append(time.time())


def entsoe_request(params: dict, tries: int = 4) -> bytes:
    """GET with the token; returns the raw body (XML or ZIP). Errors are scrubbed (no URL, no token)."""
    import requests
    last = ""
    for k in range(tries):
        _throttle()
        try:
            r = requests.get(ENTSOE_URL, params={**params, "securityToken": _token()}, timeout=180)
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
            time.sleep(5 * (k + 1))
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 503, 502, 504):
            last = f"HTTP {r.status_code}"
            time.sleep(20 * (k + 1))
            continue
        if r.status_code == 400 and b"Acknowledgement" in r.content:
            return r.content            # "no matching data" and similar: parsed as empty
        if r.status_code == 401:
            raise EntsoeError("ENTSO-E rejected the token (401)")
        last = f"HTTP {r.status_code}"
        time.sleep(3)
    raise EntsoeError(f"ENTSO-E request failed ({last})")


def _ln(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_RES = {"PT15M": pd.Timedelta(minutes=15), "PT30M": pd.Timedelta(minutes=30), "PT60M": pd.Timedelta(hours=1), "P1D": pd.Timedelta(days=1)}


def ack_reason(xml: bytes) -> str | None:
    """Acknowledgement document -> its reason text ('' when none); None for a data document."""
    if b"Acknowledgement_MarketDocument" not in xml[:400]:
        return None
    from lxml import etree
    root = etree.fromstring(xml)
    return " ".join((e.text or "") for e in root.iter() if _ln(e.tag) in ("code", "text")).strip()


def parse_series(xml: bytes) -> pd.DataFrame:
    """GL / Publication market document -> long frame (time UTC, ts, psr, in, out, value). Handles 'quantity' and 'price.amount', any
    resolution, and curve type A03 (omitted positions repeat the previous value). Acknowledgements (no data) give an empty frame."""
    cols = ["time", "ts", "psr", "in", "out", "value"]
    if not xml or ack_reason(xml) is not None:
        return pd.DataFrame(columns=cols)
    from lxml import etree
    root = etree.fromstring(xml)
    rows_t, rows_v, rows_meta = [], [], []
    for i, ts in enumerate(e for e in root.iter() if _ln(e.tag) == "TimeSeries"):
        psr, din, dout = "", "", ""
        for e in ts.iter():
            n = _ln(e.tag)
            if n == "psrType":
                psr = e.text or ""
            elif n in ("inBiddingZone_Domain.mRID", "in_Domain.mRID"):
                din = e.text or ""
            elif n in ("outBiddingZone_Domain.mRID", "out_Domain.mRID"):
                dout = e.text or ""
        for per in (e for e in ts if _ln(e.tag) == "Period"):
            start = end = res = None
            pts = {}
            for e in per:
                n = _ln(e.tag)
                if n == "timeInterval":
                    for x in e:
                        if _ln(x.tag) == "start":
                            start = pd.Timestamp(x.text)
                        elif _ln(x.tag) == "end":
                            end = pd.Timestamp(x.text)
                elif n == "resolution":
                    res = _RES.get(e.text or "")
                elif n == "Point":
                    pos, val = None, None
                    for x in e:
                        xn = _ln(x.tag)
                        if xn == "position":
                            pos = int(x.text)
                        elif xn in ("quantity", "price.amount"):
                            val = float(x.text)
                    if pos is not None and val is not None:
                        pts[pos] = val
            if start is None or end is None or res is None or not pts:
                continue
            npos = int((end - start) / res)
            vals = np.full(npos, np.nan)
            for p, v in pts.items():
                if 1 <= p <= npos:
                    vals[p - 1] = v
            vals = pd.Series(vals).ffill().to_numpy()      # omitted positions repeat the previous value (curve A03); no-op on complete curves
            rows_t.append(pd.date_range(start, periods=npos, freq=res))
            rows_v.append(vals)
            rows_meta.append((i, psr, din, dout, npos))
    if not rows_t:
        return pd.DataFrame(columns=cols)
    t = np.concatenate([x.to_numpy() for x in rows_t])
    v = np.concatenate(rows_v)
    meta = np.repeat(np.array([m[:4] for m in rows_meta], dtype=object), [m[4] for m in rows_meta], axis=0)
    d = pd.DataFrame({"time": pd.to_datetime(t, utc=True), "ts": meta[:, 0].astype(int), "psr": meta[:, 1], "in": meta[:, 2], "out": meta[:, 3], "value": v})
    return d


def hourly(d: pd.DataFrame, how: str = "first") -> pd.Series:
    """Long frame (time, ts, value) of ONE quantity -> hourly mean. Several TimeSeries covering the same instant (re-sent or sequenced
    documents) are reduced per instant with `how` before the hourly mean."""
    if d.empty:
        return pd.Series(dtype="float64")
    s = d.groupby("time")["value"].agg(how)
    return s.resample("h").mean().dropna()


def n_documents(body: bytes) -> int:
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            return sum(1 for n in z.namelist() if n.lower().endswith(".xml"))
    return 0 if ack_reason(body) is not None else 1


def parse_unavailability_zip(body: bytes) -> pd.DataFrame:
    """A80 / A78 answer (one XML or a ZIP of XML documents) -> one row per (document, time series): mrid, rev, unit, psr, nominal_mw,
    available_mw (lowest point of the available periods), start, end, business_type (A53 planned / A54 forced), in, out."""
    docs = []
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            docs = [z.read(n) for n in z.namelist() if n.lower().endswith(".xml")]
    else:
        docs = [body]
    from lxml import etree
    rows = []
    for x in docs:
        if not x or ack_reason(x) is not None:
            continue
        root = etree.fromstring(x)
        mrid = next((e.text for e in root if _ln(e.tag) == "mRID"), None)
        rev = int(next((e.text for e in root if _ln(e.tag) == "revisionNumber"), "1") or 1)
        for ts in (e for e in root.iter() if _ln(e.tag) == "TimeSeries"):
            g = {}
            for e in ts.iter():
                n = _ln(e.tag)
                if e.text and e.text.strip() and n not in g:
                    g[n] = e.text.strip()

            def suf(*ends):
                return next((v for k, v in g.items() for x in ends if k.endswith(x)), None)
            d0, t0, d1, t1 = g.get("start_DateAndOrTime.date"), g.get("start_DateAndOrTime.time"), g.get("end_DateAndOrTime.date"), g.get("end_DateAndOrTime.time")
            av = [float(q.text) for ap in ts if _ln(ap.tag) == "Available_Period" for q in ap.iter() if _ln(q.tag) == "quantity" and q.text]
            nom = suf("nominalP")
            rows.append(dict(mrid=mrid, rev=rev, unit=suf("RegisteredResource.name", "powerSystemResources.name"), psr=suf("psrType"),
                             nominal_mw=float(nom) if nom else np.nan, available_mw=min(av) if av else np.nan,
                             start=pd.Timestamp(f"{d0}T{t0 or '00:00:00Z'}") if d0 else pd.NaT, end=pd.Timestamp(f"{d1}T{t1 or '00:00:00Z'}") if d1 else pd.NaT,
                             business_type=g.get("businessType"), din=suf("in_Domain.mRID"), dout=suf("out_Domain.mRID")))
    out = pd.DataFrame(rows)
    if len(out):
        for c in ("start", "end"):
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    return out


def _fmt(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y%m%d%H%M")


def _zone_for(cc: str, t0: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    zs = PRICE_ZONE[cc]
    out = []
    for i, (since, z) in enumerate(zs):
        a = pd.Timestamp(since, tz="UTC")
        b = pd.Timestamp(zs[i + 1][0], tz="UTC") if i + 1 < len(zs) else pd.Timestamp("2100-01-01", tz="UTC")
        out.append((a, b, z))
    return out


def _bz(cc: str, t: pd.Timestamp) -> str:
    """Bidding zone valid at t (outage documents are published per bidding zone: Germany is DE-AT-LU to Sep 2018, then DE-LU)."""
    z = [zz for a, b, zz in _zone_for(cc, t) if a <= t < b] if cc == "DE" else []
    return z[0] if z else AREA[cc]


def fetch_chunk(dataset: str, key: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """One ENTSO-E chunk [t0, t1) -> hourly frame (index UTC) or an events frame (outages)."""
    if dataset in ("load", "load_da"):
        b = entsoe_request(dict(documentType="A65", processType="A16" if dataset == "load" else "A01", outBiddingZone_Domain=AREA[key],
                                periodStart=_fmt(t0), periodEnd=_fmt(t1)))
        s = hourly(parse_series(b))
        return pd.DataFrame({"load_mw" if dataset == "load" else "load_da_mw": s.astype("float32")})
    if dataset == "price":
        parts = []
        for a, bnd, z in _zone_for(key, t0):
            lo, hi = max(a, t0), min(bnd, t1)
            if lo >= hi:
                continue
            body = entsoe_request(dict(documentType="A44", in_Domain=z, out_Domain=z, periodStart=_fmt(lo), periodEnd=_fmt(hi)))
            d = parse_series(body)
            if len(d):
                parts.append(hourly(d, how="first"))
        s = pd.concat(parts).sort_index() if parts else pd.Series(dtype="float64")
        s = s[~s.index.duplicated(keep="last")]
        return pd.DataFrame({"price_eur_mwh": s.astype("float32")})
    if dataset == "gen":
        b = entsoe_request(dict(documentType="A75", processType="A16", in_Domain=AREA[key], periodStart=_fmt(t0), periodEnd=_fmt(t1)))
        d = parse_series(b)
        if d.empty:
            return pd.DataFrame()
        d["col"] = np.where(d["out"] != "", "cons_", "gen_") + d["psr"].map(PSR).fillna(d["psr"].str.lower())
        w = d.groupby(["time", "col"])["value"].first().unstack("col")
        return w.resample("h").mean().dropna(axis=1, how="all").astype("float32")
    if dataset == "res_da":
        b = entsoe_request(dict(documentType="A69", processType="A01", in_Domain=AREA[key], periodStart=_fmt(t0), periodEnd=_fmt(t1)))
        d = parse_series(b)
        if d.empty:
            return pd.DataFrame()
        d["col"] = "da_" + d["psr"].map(PSR).fillna(d["psr"].str.lower())
        w = d.groupby(["time", "col"])["value"].first().unstack("col")
        return w.resample("h").mean().dropna(axis=1, how="all").astype("float32")
    if dataset == "flows":
        a, c = key.split("_")
        b = entsoe_request(dict(documentType="A11", out_Domain=AREA[a], in_Domain=AREA[c], periodStart=_fmt(t0), periodEnd=_fmt(t1)))
        s = hourly(parse_series(b), how="sum")
        return pd.DataFrame({"flow_mw": s.astype("float32")})
    if dataset in ("out_gen", "out_tx"):
        frames, off = [], 0
        while True:
            if dataset == "out_gen":
                p = dict(documentType="A80", biddingZone_Domain=_bz(key, t0), periodStart=_fmt(t0), periodEnd=_fmt(t1), offset=off)
            else:
                a, c = key.split("_")
                p = dict(documentType="A78", in_Domain=_bz(c, t0), out_Domain=_bz(a, t0), periodStart=_fmt(t0), periodEnd=_fmt(t1), offset=off)
            body = entsoe_request(p)
            ev = parse_unavailability_zip(body)
            if ev.empty:
                break
            frames.append(ev)
            ndoc = n_documents(body)            # a page is full at 200 DOCUMENTS (one mRID can come in several revisions)
            if ndoc < 200:
                break
            off += 200
            if off > 4800:                 # API ceiling on offset
                break
        if not frames:
            return pd.DataFrame()
        ev = pd.concat(frames, ignore_index=True)
        ev = ev.sort_values("rev").drop_duplicates(["mrid", "unit", "start", "end"], keep="last")
        return ev.reset_index(drop=True)
    raise ValueError(dataset)


def entsoe_keys(dataset: str, countries=COUNTRIES) -> list[str]:
    if dataset in ("flows", "out_tx"):
        out = []
        for a, b in BORDERS:
            if a in countries and b in countries:
                out += [f"{a}_{b}", f"{b}_{a}"] if dataset == "flows" else [f"{a}_{b}"]
        return out
    return list(countries)


def _chunks(dataset: str, year: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    t0 = pd.Timestamp(f"{year}-01-01", tz="UTC")
    t1 = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), pd.Timestamp.now("UTC").floor("h"))
    if dataset in ("gen", "out_gen", "out_tx"):
        edges = list(pd.date_range(t0, t1, freq="MS", tz="UTC")) + [t1] if dataset != "gen" else list(pd.date_range(t0, t1, freq="QS", tz="UTC")) + [t1]
        edges = sorted(set(edges))
        return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if a < b]
    return [(t0, t1)] if t0 < t1 else []


def entsoe_year(dataset: str, key: str, year: int) -> pd.DataFrame:
    parts = []
    for a, b in _chunks(dataset, year):
        try:
            parts.append(fetch_chunk(dataset, key, a, b))
        except EntsoeError as e:
            if "Acknowledgement" in str(e):
                continue
            raise
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    if dataset in ("out_gen", "out_tx"):
        return df.sort_values("rev").drop_duplicates(["mrid", "unit", "start", "end"], keep="last").reset_index(drop=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    return df


def entsoe_backfill(datasets=tuple(ENTSOE_DATASETS), countries=COUNTRIES, years=range(2015, 2027), workers: int = 4, log=print, refresh_current: bool = True) -> dict:
    """Resumable: every (dataset, key, year) is one parquet; existing ones are skipped (the current year is refetched when refresh_current)."""
    have = set(list_names("entsoe"))
    cur_year = pd.Timestamp.now("UTC").year
    jobs = []
    for ds in datasets:
        for key in entsoe_keys(ds, countries):
            for y in years:
                name = f"entsoe/{ds}/{key}/year={y}.parquet"
                if name in have and not (refresh_current and y == cur_year):
                    continue
                jobs.append((ds, key, y, name))
    stats = dict(jobs=len(jobs), written=0, empty=0, failed=0, bytes=0)

    def run(job):
        ds, key, y, name = job
        t = time.time()
        try:
            df = entsoe_year(ds, key, y)
        except Exception as e:  # noqa: BLE001
            msg = str(e) if isinstance(e, EntsoeError) else type(e).__name__
            log(f"FAIL {name}: {msg}")
            with _LOCK:
                stats["failed"] += 1
            return
        if df is None or not len(df):
            put_parquet(name, pd.DataFrame({"empty": pd.Series(dtype="float32")}))      # marks the chunk as done (no data published)
            with _LOCK:
                stats["empty"] += 1
            log(f"empty {name}")
            return
        n = put_parquet(name, df)
        with _LOCK:
            stats["written"] += 1
            stats["bytes"] += n
        log(f"ok {name} rows={len(df)} kb={n // 1024} s={time.time() - t:.0f}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))
    return stats


def entsoe_series(dataset: str, key: str, col: str | None = None, since: str | None = None) -> pd.Series | pd.DataFrame:
    """All stored years of one ENTSO-E series (hourly UTC). col=None returns the whole frame."""
    mk = ("entsoe", dataset, key, col, since)
    if mk in _MEMO:
        return _MEMO[mk]
    names = [n for n in list_names(f"entsoe/{dataset}/{key}") if n.endswith(".parquet")]
    parts = []
    for n in names:
        if since and n.split("year=")[-1][:4] < since[:4]:
            continue
        d = get_parquet(n)
        if d is not None and len(d) and "empty" not in d.columns:
            parts.append(d)
    if not parts:
        out = pd.Series(dtype="float64") if col else pd.DataFrame()
    else:
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        out = df[col].astype("float64").dropna() if col else df
    _MEMO[mk] = out
    return out


def outage_daily(events: pd.DataFrame, start=None, end=None) -> pd.Series:
    """Outage events -> daily mean unavailable MW (nominal - available over each event's overlap with the day, hourly resolution)."""
    if events is None or not len(events):
        return pd.Series(dtype="float64")
    e = events.dropna(subset=["start", "end"]).copy()
    e["lost"] = (e["nominal_mw"] - e["available_mw"].fillna(0)).clip(lower=0)
    e = e[e["lost"] > 0]
    if e.empty:
        return pd.Series(dtype="float64")
    lo = pd.Timestamp(start, tz="UTC") if start is not None else e["start"].min().floor("h")
    hi = pd.Timestamp(end, tz="UTC") if end is not None else e["end"].max().ceil("h")
    idx = pd.date_range(lo.floor("h"), hi.ceil("h"), freq="h")
    delta = np.zeros(len(idx) + 1)
    ns = lambda x: pd.DatetimeIndex(x).as_unit("ns").asi8  # noqa: E731
    ii = ns(idx)
    a = np.clip(np.searchsorted(ii, ns(e["start"].dt.floor("h"))), 0, len(idx))
    b = np.clip(np.searchsorted(ii, ns(e["end"].dt.ceil("h"))), 0, len(idx))
    np.add.at(delta, a, e["lost"].to_numpy())
    np.add.at(delta, b, -e["lost"].to_numpy())
    h = pd.Series(np.cumsum(delta)[:-1], index=idx)
    return h.resample("D").mean()


# ------------------------------------------------------------------------------------------------------------------------------ calendar
def calendar_frame(index: pd.DatetimeIndex, country: str) -> pd.DataFrame:
    """Known-in-advance calendar drivers on a UTC index: weekday (0-6), weekend, public holiday (national; `holidays` package when present)."""
    idx = pd.DatetimeIndex(index)
    tzs = {"ES": "Europe/Madrid", "PT": "Europe/Lisbon", "FR": "Europe/Paris", "DE": "Europe/Berlin", "IT": "Europe/Rome", "NL": "Europe/Amsterdam",
           "BE": "Europe/Brussels", "PL": "Europe/Warsaw"}
    loc = idx.tz_convert(tzs.get(country, "Europe/Brussels")) if idx.tz is not None else idx
    hol = np.zeros(len(idx), bool)
    try:
        import holidays as HL
        yrs = sorted(set(loc.year))
        h = HL.country_holidays(country, years=yrs)
        days = set(pd.Timestamp(d).date() for d in h.keys())
        hol = np.array([d in days for d in loc.date])
    except Exception:  # noqa: BLE001
        md = {(1, 1), (5, 1), (8, 15), (11, 1), (12, 25), (12, 26)}
        hol = np.array([(t.month, t.day) in md for t in loc])
    wd = loc.weekday.to_numpy()
    return pd.DataFrame({"weekday": wd, "weekend": (wd >= 5).astype(float), "holiday": hol.astype(float), "hour": loc.hour.to_numpy()}, index=idx)


# ------------------------------------------------------------------------------------------------------------------------------ EU Weekly Oil Bulletin
OIL_BULLETIN_URL = ("https://energy.ec.europa.eu/document/download/906e60ca-8b6a-44e7-8589-652854d2fd3f_en"
                    "?filename=Weekly_Oil_Bulletin_Prices_History_maticni_4web.xlsx")
ATTR_OIL = "European Commission, Weekly Oil Bulletin (DG ENER), (c) European Union, reuse authorised with acknowledgement (Commission Decision 2011/833/EU)"
OIL_PRODUCTS = {"euro95": "gasoline95", "diesel": "diesel", "heating_oil": "heating_oil", "LPG": "lpg"}


def parse_oil_bulletin(xlsx: bytes) -> pd.DataFrame:
    """History workbook -> long frame (date, country, product, tax: with|without, eur_per_1000l). Country blocks start at a 'CTR' column;
    headers look like 'ES_price_with_tax_diesel'. 'EU_' (EU weighted) and 'EUR_' (euro area) are kept as country EU / EA."""
    import openpyxl
    import re
    wb = openpyxl.load_workbook(io.BytesIO(xlsx), read_only=True, data_only=True)
    out = []
    for sheet, tax in (("Prices with taxes", "with"), ("Prices wo taxes", "without")):
        if sheet not in wb.sheetnames:
            continue
        rows = list(wb[sheet].iter_rows(values_only=True))
        head = rows[0]
        cols = []
        for j, h in enumerate(head):
            m = re.match(r"^([A-Z]{2,3})_price_(?:with|wo)_tax_(euro95|diesel|heating_oil|LPG)$", str(h or "").strip())
            if m:
                cc = {"EU": "EU", "EUR": "EA"}.get(m.group(1), m.group(1))
                cols.append((j, cc, OIL_PRODUCTS[m.group(2)]))
        for r in rows[3:]:
            if not r or r[0] is None:
                continue
            try:
                d = pd.Timestamp(r[0])
            except Exception:  # noqa: BLE001
                continue
            for j, cc, prod in cols:
                v = r[j] if j < len(r) else None
                if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                    out.append((d, cc, prod, tax, float(v)))
    df = pd.DataFrame(out, columns=["date", "country", "product", "tax", "eur_per_1000l"])
    return df.drop_duplicates(["date", "country", "product", "tax"]).sort_values(["country", "product", "tax", "date"]).reset_index(drop=True)


def oil_bulletin_load(log=print) -> dict:
    import requests
    r = requests.get(OIL_BULLETIN_URL, timeout=180, headers={"User-Agent": "forecasting-engine/0.1"})
    r.raise_for_status()
    df = parse_oil_bulletin(r.content)
    if len(df) < 1000:
        raise RuntimeError("Oil Bulletin workbook could not be parsed")
    n = put_parquet("oil_bulletin/prices.parquet", df.set_index("date"))
    st = dict(rows=len(df), countries=int(df["country"].nunique()), first=str(df["date"].min().date()), last=str(df["date"].max().date()), bytes=n)
    log(f"oil bulletin {st}")
    return st


def oil_series(country: str, product: str = "diesel", tax: str = "with") -> pd.Series:
    """Weekly pump price in EUR per litre (Monday of the bulletin week)."""
    mk = ("oil", None)
    if mk not in _MEMO:
        d = get_parquet("oil_bulletin/prices.parquet")
        _MEMO[mk] = d.reset_index() if d is not None else pd.DataFrame(columns=["date", "country", "product", "tax", "eur_per_1000l"])
    d = _MEMO[mk]
    x = d[(d["country"] == country.upper()) & (d["product"] == product) & (d["tax"] == tax)]
    s = pd.Series(x["eur_per_1000l"].to_numpy(float) / 1000.0, index=pd.to_datetime(x["date"], utc=True)).sort_index()
    return s[~s.index.duplicated(keep="last")]


# ------------------------------------------------------------------------------------------------------------------------------ Puertos del Estado monthly
PORTS_PAGE = "https://www.puertos.es/en/data/statistics/monthly?date_value={y}"
PORTS_HOST = "https://www.puertos.es"
ATTR_PORTS = ("Puertos del Estado, Resumen mensual del tráfico portuario (Ministerio de Transportes y Movilidad Sostenible). Licence NOT cleared for "
              "commercial reuse: the puertos.es legal notice requires prior written authorisation; used for back-testing only until it is granted")
MONTHS_ES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
             "octubre": 10, "noviembre": 11, "diciembre": 12}
PORT_SHEETS = {"tonnes": ("total tráfico", "total trafico"), "teu": ("teus",), "passengers": ("pasajeros total",)}


def _wb_rows(b: bytes) -> dict[str, list[tuple]]:
    """Workbook bytes (xlsx or legacy xls) -> {sheet name: rows}."""
    if b[:2] == b"PK":
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(b), read_only=True, data_only=True)
        return {ws.title: list(ws.iter_rows(values_only=True)) for ws in wb.worksheets}
    import xlrd
    wb = xlrd.open_workbook(file_contents=b)
    return {sh.name: [tuple(sh.row_values(i)) for i in range(sh.nrows)] for sh in wb.sheets()}


def port_key(name: str) -> str:
    """'BAHÍA DE ALGECIRAS' / 'Bahía de Algeciras' -> 'bahia de algeciras' (the files changed case and accents over the years)."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def parse_port_month(b: bytes) -> pd.DataFrame:
    """One monthly summary workbook -> rows (port, metric, year, month, value, vintage) for the month of the file and the same month a year
    earlier (the earlier value is the revised one). vintage = the file's year (newer vintages win when consolidating)."""
    import re
    sheets = _wb_rows(b)
    out = []
    for metric, names in PORT_SHEETS.items():
        sn = next((s for s in sheets if s.strip().lower() in names), None)
        if sn is None:
            continue
        rows = sheets[sn]
        month = years = None
        hdr_i = None
        for i, r in enumerate(rows[:15]):
            cells = [c for c in r if c not in (None, "")]
            for c in cells:
                if isinstance(c, str) and month is None:
                    w = re.sub(r"^mes\s+", "", c.strip().lower())
                    if w in MONTHS_ES:
                        month, hdr_i = MONTHS_ES[w], i
            ys = [int(c) for c in cells if isinstance(c, (int, float)) and 1990 < float(c) < 2100]
            if month is not None and len(ys) >= 2 and years is None and i > (hdr_i or 0):
                years = ys[:2]
                start = i + 1
                break
        if month is None or years is None:
            continue
        for r in rows[start:]:
            if not r or not isinstance(r[0], str) or not r[0].strip():
                continue
            name = port_key(r[0])
            if name.startswith("total"):
                name = "total"
            elif name.startswith("incluye") or len(name) > 40:
                continue
            nums = [c for c in r[1:] if isinstance(c, (int, float))]
            if len(nums) < 2:
                continue
            for y, v in zip(years, nums[:2]):
                out.append((name, metric, int(y), int(month), float(v), int(years[1])))
            if name == "total":
                break
    return pd.DataFrame(out, columns=["port", "metric", "year", "month", "value", "vintage"])


def ports_es_load(years=range(2006, 2027), log=print) -> dict:
    """Every monthly summary since 2006 (about 250 workbooks of ~1 MB, read in memory) -> ports_es/monthly.parquet. Resumable per year
    (ports_es/_parts/year=Y.parquet; the current and previous year are refetched)."""
    import re
    import requests
    have = set(list_names("ports_es/_parts"))
    cur = pd.Timestamp.now("UTC").year
    sess = requests.Session()
    sess.headers["User-Agent"] = "forecasting-engine/0.1"
    for y in years:
        part = f"ports_es/_parts/year={y}.parquet"
        if part in have and y < cur - 1:
            continue
        page = sess.get(PORTS_PAGE.format(y=y), timeout=60).text
        links = list(dict.fromkeys(re.findall(r'href="(/[a-z]{2}/file-download/download/(?:private|public)/\d+)"[^>]*>\s*Descargar XLS', page)))
        rows = []
        for href in links:
            try:
                b = sess.get(PORTS_HOST + href, timeout=120).content
                d = parse_port_month(b)
                if len(d):
                    rows.append(d)
            except Exception as e:  # noqa: BLE001
                log(f"ports {y} {href}: {type(e).__name__}")
            time.sleep(0.5)
        if rows:
            df = pd.concat(rows, ignore_index=True)
            put_parquet(part, df)
            log(f"ports {y}: {len(links)} files, {len(df)} rows, months {sorted(df[df.year == y].month.unique().tolist())}")
        else:
            log(f"ports {y}: no workbook parsed ({len(links)} links)")
    parts = [get_parquet(n) for n in list_names("ports_es/_parts") if n.endswith(".parquet")]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    df = df.sort_values("vintage").drop_duplicates(["port", "metric", "year", "month"], keep="last")
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), utc=True)
    df = df[["date", "port", "metric", "value"]].sort_values(["port", "metric", "date"]).reset_index(drop=True)
    n = put_parquet("ports_es/monthly.parquet", df.set_index("date"))
    return dict(rows=len(df), ports=int(df["port"].nunique()), first=str(df["date"].min().date()), last=str(df["date"].max().date()), bytes=n)


def port_series(port: str, metric: str = "tonnes") -> pd.Series:
    mk = ("ports", None)
    if mk not in _MEMO:
        d = get_parquet("ports_es/monthly.parquet")
        _MEMO[mk] = d.reset_index() if d is not None else pd.DataFrame(columns=["date", "port", "metric", "value"])
    d = _MEMO[mk]
    x = d[(d["port"] == port_key(port)) & (d["metric"] == metric)]
    return pd.Series(x["value"].to_numpy(float), index=pd.to_datetime(x["date"], utc=True)).sort_index()


# ------------------------------------------------------------------------------------------------------------------------------ Eurostat monthly
ESTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
ATTR_ESTAT = "Eurostat, (c) European Union, reuse with acknowledgement (Commission Decision 2011/833/EU; CC BY 4.0)"
#: id -> (dataset, filters, label). Seasonally and calendar adjusted where Eurostat publishes it (headline indices); tourism nights are raw.
ESTAT_MONTHLY = {
    "industrial_production": ("sts_inpr_m", dict(nace_r2="B-D", s_adj="SCA", unit="I21", indic_bt="PRD"), "Industrial production index (B-D, SCA, 2021=100)"),
    "retail_volume": ("sts_trtu_m", dict(nace_r2="G47", s_adj="SCA", unit="I21", indic_bt="VOL_SLS"), "Retail trade volume (G47, SCA, 2021=100)"),
    "tourism_nights": ("tour_occ_nim", dict(c_resid="TOTAL", nace_r2="I551-I553", unit="NR"), "Nights spent in tourist accommodation (number)"),
}


def eurostat_load(ids=tuple(ESTAT_MONTHLY), since: str = "2000-01", log=print) -> dict:
    import requests
    from forecasting_engine.eurostat import parse_jsonstat
    out = {}
    for k in ids:
        ds, flt, _ = ESTAT_MONTHLY[k]
        r = requests.get(ESTAT_BASE + ds, params=dict(format="JSON", lang="en", sinceTimePeriod=since, **flt), timeout=300,
                         headers={"User-Agent": "forecasting-engine/0.1"})
        r.raise_for_status()
        d = parse_jsonstat(r.json()).dropna(subset=["value"])
        d = d[d["time"].str.match(r"^\d{4}-\d{2}$")]
        d = pd.DataFrame({"date": pd.to_datetime(d["time"] + "-01", utc=True), "geo": d["geo"].astype(str), "value": d["value"].astype(float)})
        n = put_parquet(f"eurostat/{k}.parquet", d.set_index("date"))
        out[k] = dict(rows=len(d), geos=int(d["geo"].nunique()), first=str(d["date"].min().date()), last=str(d["date"].max().date()), bytes=n)
        log(f"eurostat {k}: {out[k]}")
    return out


def eurostat_series(k: str, geo: str) -> pd.Series:
    mk = ("estat", k)
    if mk not in _MEMO:
        d = get_parquet(f"eurostat/{k}.parquet")
        _MEMO[mk] = d.reset_index() if d is not None else pd.DataFrame(columns=["date", "geo", "value"])
    d = _MEMO[mk]
    x = d[d["geo"] == geo.upper()]
    return pd.Series(x["value"].to_numpy(float), index=pd.to_datetime(x["date"], utc=True)).sort_index()


# ------------------------------------------------------------------------------------------------------------------------------ weather driver (ERA5)
ATTR_ERA5 = ("Contains modified Copernicus Climate Change Service information (ERA5 hourly 2 m temperature, Copernicus licence: commercial use with "
             "attribution) via ARCO-ERA5 (Google Research); GHSL population weights")
HDD_BASE, CDD_BASE = 18.0, 21.0


def weather_series(country: str, col: str = "t") -> pd.Series:
    """Daily population-weighted series of a country (t, tx, hdd, cdd); index = the UTC day."""
    mk = ("weather", None)
    if mk not in _MEMO:
        d = get_parquet("weather/popw_daily.parquet")
        if d is not None:
            d = d.reset_index()
            d = d[d["geo"].str.len() == 2]          # countries only in memory
        _MEMO[mk] = d if d is not None else pd.DataFrame(columns=["date", "geo", "t", "tx", "hdd", "cdd"])
    d = _MEMO[mk]
    cc = {"GR": "EL", "GB": "UK"}.get(country.upper(), country.upper())
    x = d[d["geo"] == cc]
    idx = pd.to_datetime(x["date"])
    idx = idx.dt.tz_localize("UTC") if idx.dt.tz is None else idx.dt.tz_convert("UTC")
    return pd.Series(x[col].to_numpy(float), index=pd.DatetimeIndex(idx)).sort_index()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="forecasting_engine.histories", description="load the long histories into $FE_HISTORY_STORE (resumable)")
    ap.add_argument("cmd", choices=["entsoe", "oil", "ports", "eurostat", "all"])
    ap.add_argument("--datasets", default=",".join(ENTSOE_DATASETS))
    ap.add_argument("--years", default="2015-2026")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    y0, _, y1 = a.years.partition("-")
    if a.cmd in ("entsoe", "all"):
        print(entsoe_backfill(a.datasets.split(","), years=range(int(y0), int(y1 or y0) + 1), workers=a.workers))
    if a.cmd in ("oil", "all"):
        print(oil_bulletin_load())
    if a.cmd in ("ports", "all"):
        print(ports_es_load())
    if a.cmd in ("eurostat", "all"):
        print(eurostat_load())
