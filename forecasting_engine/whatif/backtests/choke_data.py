"""Data for the shipping-chokepoint what-if (docs/methods.md).

Everything is cached under data/choke_whatif/ (git-ignored; licences in) and every loader is resumable.

    python -m forecasting_engine.whatif.backtests.choke_data all          # or: portwatch | fred | gscpi | scfi | estat | wci

Series (name -> source, publication delay used by the scenario code):
  transits   IMF PortWatch daily transit calls per chokepoint (Kpler / UN Global Platform), 12 days
  wci        Drewry World Container Index composite and Shanghai-Rotterdam / Shanghai-Genoa, weekly (Thursday), from the weekly public releases
             re-published by AJOT; same day. Back-test use only (Drewry terms).
  scfi       Shanghai Containerized Freight Index composite, weekly, 2009-10 to 2023-03 only (public copy of the SSE series; back-test only)
  ppi_ds     BLS PPI deep sea freight transportation (FRED PCU483111483111), monthly, month + 35 days (public domain)
  gscpi      NY Fed Global Supply Chain Pressure Index, monthly (public, citation), month + 5 days
  imp_*      Eurostat sts_inpi_m import prices from outside the euro area (PRC_IMP_NEU, EA20), monthly, month + 35 days (CC BY 4.0)
  hicp_*     Eurostat prc_hicp_midx HICP, euro area (EA20), non-energy industrial goods IGD_NNRG / TOT_X_NRG_FOOD / CP00, month + 17 days
"""
from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "data" / "choke_whatif"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) forecasting-engine"}
ESTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
PW = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"
CHOKES = ("Suez Canal", "Bab el-Mandeb Strait", "Cape of Good Hope", "Panama Canal", "Bosporus Strait", "Gibraltar Strait", "Strait of Hormuz",
          "Dover Strait", "Kerch Strait", "Malacca Strait")


def _p(name: str) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    return DIR / name


def _get(url, **kw):
    for k in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=kw.pop("timeout", 60), **kw)
            if r.status_code == 200:
                return r
            time.sleep(2 + 3 * k)
        except requests.RequestException:
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"GET failed {url}")


# ------------------------------------------------------------------------------------------------------------------------ PortWatch
def portwatch(refresh: bool = False) -> pd.DataFrame:
    """Daily chokepoint transits: columns n_total n_cargo n_container n_tanker capacity capacity_container capacity_tanker, index date (naive), portname."""
    p = _p("portwatch.parquet")
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    rows, off = [], 0
    flds = "date,portid,portname,n_total,n_cargo,n_container,n_tanker,capacity,capacity_container,capacity_tanker"
    while True:
        r = _get(PW, params=dict(where="1=1", outFields=flds, orderByFields="date", resultOffset=off, resultRecordCount=2000, f="json"), timeout=120).json()
        f = r.get("features", [])
        rows += [x["attributes"] for x in f]
        if len(f) < 2000 and not r.get("exceededTransferLimit"):
            break
        off += len(f)
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"], unit="ms") if np.issubdtype(d["date"].dtype, np.number) else pd.to_datetime(d["date"])
    d["date"] = d["date"].dt.tz_localize(None).dt.normalize()
    d = d[d["portname"].isin(CHOKES)].drop_duplicates(["date", "portname"]).sort_values(["portname", "date"])
    d.to_parquet(p)
    return d


def transits(name: str, col: str = "n_total") -> pd.Series:
    d = portwatch()
    s = d[d["portname"] == name].set_index("date")[col].astype(float).sort_index()
    return s.asfreq("D")


# ------------------------------------------------------------------------------------------------------------------------ freight
def fred(sid: str) -> pd.Series:
    p = _p(f"fred_{sid}.csv")
    if not p.exists():
        p.write_bytes(_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}").content)
    d = pd.read_csv(p)
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.set_index(pd.to_datetime(d["date"]))["v"].dropna()


def gscpi() -> pd.Series:
    p = _p("gscpi.xlsx")
    if not p.exists():
        p.write_bytes(_get("https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx").content)
    x = pd.read_excel(p, sheet_name="GSCPI Monthly Data", header=None).iloc[:, :2].dropna()
    x = x[x[0].astype(str).str.match(r"^\d{2}-[A-Za-z]{3}-\d{4}$")]
    s = pd.Series(x[1].astype(float).to_numpy(), index=pd.to_datetime(x[0], format="%d-%b-%Y").dt.to_period("M").dt.to_timestamp())
    return s


def scfi() -> pd.Series:
    p = _p("scfi.xls")
    if not p.exists():
        u = "https://raw.githubusercontent.com/jordanbell2357/datasets/main/SCFI/Shanghai%20(Export)%20Containerized%20Freight%20Index%20(SCFI).xls"
        p.write_bytes(_get(u).content)
    x = pd.read_excel(p, header=None).iloc[4:, :2].dropna()
    x = x[x[0].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")]
    return pd.Series(x[1].astype(float).to_numpy(), index=pd.to_datetime(x[0])).sort_index()


MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_NUM = r"\$\s?([\d,]{3,6})"


def _parse_wci(html: str, target: pd.Timestamp):
    """Composite and route values from one AJOT page of Drewry's weekly release; None when the page is not the release of `target`."""
    t = re.sub(r"<[^>]+>", " ", html)
    t = re.sub(r"\s+", " ", t)
    m = re.search(r"assessment for (?:Thursday|Wednesday|Friday),? (\d{1,2})(?:st|nd|rd|th)? ([A-Za-z]+) (\d{4})", t)
    if not m:
        return None
    try:
        dt = pd.Timestamp(f"{m.group(1)} {m.group(2)[:3]} {m.group(3)}")
    except Exception:  # noqa: BLE001
        return None
    if abs((dt - target).days) > 1:
        return None
    seg = t[m.end():m.end() + 3000]
    out = {"date": dt}
    c = re.search(r"composite index[^$]{0,120}?" + _NUM, seg)
    if c:
        out["wci"] = float(c.group(1).replace(",", ""))
    sep = r"\s*(?:to|[\u2013\u2014-])\s*"
    for key, pat in (("sha_rtm", "Shanghai" + sep + "Rotterdam"), ("sha_gen", "Shanghai" + sep + "Genoa"), ("sha_la", "Shanghai" + sep + "Los Angeles"),
                     ("sha_ny", "Shanghai" + sep + "New York")):
        r = re.search(pat + r"[^.]{0,110}?\bto (?:reach )?" + _NUM, seg)
        if r:
            out[key] = float(r.group(1).replace(",", ""))
    return out if "wci" in out else None


def _wci_one(d: pd.Timestamp, ses) -> dict | None:
    cands = []
    for off in (0, 1, -1):
        t = d + pd.Timedelta(days=off)
        for stem in ("drewry-world-container-index", "drewrys-world-container-index"):
            cands.append(f"{stem}-{t.day}-{MON[t.month - 1]}")
    for stem in ("drewry-world-container-index", "drewrys-world-container-index"):
        cands.append(f"{stem}-{d.day}-{MON[d.month - 1]}-1")
    for c in cands:
        try:
            h = ses.get(f"https://www.ajot.com/news/{c}", timeout=40).text
        except requests.RequestException:
            continue
        got = _parse_wci(h, d)
        if got:
            got["date"] = d
            return got
    return None


def wci(start="2019-01-03", end="2025-12-31", budget_s: float = 90.0, workers: int = 8) -> pd.DataFrame:
    """Weekly Drewry WCI prints from the public weekly releases (AJOT.com news pages named drewry-world-container-index-<d>-<mon>). Resumable."""
    from concurrent.futures import ThreadPoolExecutor
    p = _p("wci.csv")
    have = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame({"date": pd.to_datetime([]), "wci": []})
    miss_p = _p("wci_missing.txt")
    miss = set(miss_p.read_text().split()) if miss_p.exists() else set()
    done = set(have["date"].dt.strftime("%Y-%m-%d"))
    todo = [d for d in pd.date_range(start, end, freq="W-THU") if d.strftime("%Y-%m-%d") not in done | miss]
    ses = requests.Session()
    ses.headers.update(UA)
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), workers * 2):
            if time.time() - t0 > budget_s:
                break
            chunk = todo[i:i + workers * 2]
            for d, g in zip(chunk, ex.map(lambda x: _wci_one(x, ses), chunk)):
                if g:
                    rows.append(g)
                else:
                    miss.add(d.strftime("%Y-%m-%d"))
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("date").sort_values("date")
        have.to_csv(p, index=False)
    miss_p.write_text("\n".join(sorted(miss)))
    return have


def wci_hsn(pages: int = 60, budget_s: float = 90.0, workers: int = 8) -> pd.DataFrame:
    """Same weekly releases, from Hellenic Shipping News article pages found through its search listing (resumable; links cached)."""
    from concurrent.futures import ThreadPoolExecutor
    ses = requests.Session()
    ses.headers.update(UA)
    lp = _p("hsn_links.txt")
    links = set(lp.read_text().split()) if lp.exists() else set()
    seen_p = _p("hsn_seen.txt")
    seen = set(seen_p.read_text().split()) if seen_p.exists() else set()
    t0 = time.time()
    pg_done = int(_p("hsn_pages.txt").read_text()) if _p("hsn_pages.txt").exists() else 0
    for pg in range(pg_done + 1, pages + 1):
        if time.time() - t0 > budget_s / 2:
            break
        try:
            h = ses.get(f"https://www.hellenicshippingnews.com/page/{pg}/?s=drewry+world+container+index", timeout=40).text
        except requests.RequestException:
            break
        new = re.findall(r'<h[23][^>]*>\s*<a href="(https://www.hellenicshippingnews.com/drewry[^"]+)"', h)
        links.update(new)
        _p("hsn_pages.txt").write_text(str(pg))
        if not new:
            break
    lp.write_text("\n".join(sorted(links)))
    p = _p("wci.csv")
    have = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame({"date": pd.to_datetime([]), "wci": []})
    todo = [u for u in sorted(links) if u not in seen]

    def one(u):
        try:
            h = ses.get(u, timeout=40).text
        except requests.RequestException:
            return u, None
        t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))
        m = re.search(r"assessment for (?:Thursday|Wednesday|Friday),? (\d{1,2}) ([A-Za-z]+) (\d{4})", t)
        if not m:
            return u, False
        try:
            dt = pd.Timestamp(f"{m.group(1)} {m.group(2)[:3]} {m.group(3)}")
        except Exception:  # noqa: BLE001
            return u, False
        return u, _parse_wci(h, dt)

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), workers * 2):
            if time.time() - t0 > budget_s:
                break
            for u, g in ex.map(one, todo[i:i + workers * 2]):
                if g is None:
                    continue
                seen.add(u)
                if g:
                    rows.append(g)
    seen_p.write_text("\n".join(sorted(seen)))
    if rows:
        have = pd.concat([pd.DataFrame(rows), have], ignore_index=True).drop_duplicates("date").sort_values("date")
        have.to_csv(p, index=False)
    return have


def wci_series() -> pd.DataFrame:
    p = _p("wci.csv")
    return pd.read_csv(p, parse_dates=["date"]).set_index("date") if p.exists() else pd.DataFrame()


# ------------------------------------------------------------------------------------------------------------------------ Eurostat
def _jsonstat(j) -> pd.Series:
    dims, size = j["id"], j["size"]
    cats = [list(j["dimension"][d]["category"]["index"]) for d in dims]
    v = np.full(int(np.prod(size)), np.nan)
    for k, x in j["value"].items():
        v[int(k)] = x
    ix = pd.MultiIndex.from_product(cats, names=dims)
    return pd.Series(v, index=ix)


def estat_series(tag: str, ds: str, since="2005-01", **flt) -> pd.Series:
    p = _p(f"estat_{tag}.csv")
    if not p.exists():
        j = _get(ESTAT + ds, params=dict(format="JSON", lang="en", sinceTimePeriod=since, **flt), timeout=120).json()
        s = _jsonstat(j)
        s = s.droplevel([n for n in s.index.names if n != "time"]).dropna()
        s.index = pd.to_datetime(s.index + "-01")
        s.sort_index().to_csv(p, header=["v"], index_label="date")
    d = pd.read_csv(p, parse_dates=["date"])
    return d.set_index("date")["v"]


IMP = {"all": "CPA_B-D", "nonfood": "CPA_B-D_X_FOOD", "capital": "CPA_MIG_CAG", "consumer": "CPA_MIG_COG_X_FOOD", "interm": "CPA_MIG_ING",
       "nonenergy": "CPA_B_C_X_MIG_NRG", "energy": "CPA_MIG_NRG_X_E"}
HICP = {"nrg_goods": "IGD_NNRG", "core": "TOT_X_NRG_FOOD", "total": "CP00", "services": "SERV"}


def eurusd() -> pd.Series:
    """ECB reference rate, monthly mean USD per EUR (SDW, free reuse with source credit)."""
    p = _p("ecb_usd_m.csv")
    if not p.exists():
        p.write_bytes(_get("https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A?format=csvdata&startPeriod=2005-01").content)
    d = pd.read_csv(p)
    return pd.Series(d["OBS_VALUE"].astype(float).to_numpy(), index=pd.to_datetime(d["TIME_PERIOD"] + "-01"))


def import_price(kind: str = "consumer", geo: str = "EA20", src: str = "PRC_IMP_NEU") -> pd.Series:
    return estat_series(f"imp_{src}_{kind}_{geo}", "sts_inpi_m", since="2005-01", geo=geo, indic_bt=src, cpa2_1=IMP[kind], s_adj="NSA", unit="I21")


def hicp(kind: str = "nrg_goods", geo: str = "EA") -> pd.Series:
    return estat_series(f"hicp_{kind}_{geo}", "prc_hicp_midx", since="2005-01", geo=geo, coicop=HICP[kind], unit="I15")


def main(a):
    what = a[0] if a else "all"
    if what in ("all", "portwatch"):
        d = portwatch()
        print("portwatch", d.shape, d["date"].min().date(), d["date"].max().date())
    if what in ("all", "fred"):
        print("ppi_ds", fred("PCU483111483111").tail(2).to_dict())
    if what in ("all", "gscpi"):
        print("gscpi", gscpi().tail(2).to_dict())
    if what in ("all", "scfi"):
        print("scfi", scfi().tail(2).to_dict())
    if what in ("all", "estat"):
        for k in IMP:
            print("imp", k, import_price(k).tail(1).to_dict())
        for k in HICP:
            print("hicp", k, hicp(k).tail(1).to_dict())
    if what == "hsn":
        w = wci_hsn()
        print("wci rows", len(w), w["date"].min(), w["date"].max())
    if what == "wci":
        w = wci(start=a[1] if len(a) > 1 else "2019-01-03", end=a[2] if len(a) > 2 else "2025-12-31")
        print("wci rows", len(w), w["date"].min(), w["date"].max())


if __name__ == "__main__":
    main(sys.argv[1:])
