"""Inputs for the ECB policy-rate family (ecb_w) and the euro-area inflation family (hicp_ea_m), into the history store under lab/raw/.

    python -m forecasting_engine.lab.src_ecb all          # or: rates | meetings | yc | hicp_ea

| Store name | Content | Source / licence |
|---|---|---|
| lab/raw/ecb_dfr_daily.parquet | deposit facility rate, daily step function from its change (effective) dates, 1999- | ECB Data Portal FM/B.U2.EUR.4F.KR.DFR.LEV; ECB reuse with acknowledgement |
| lab/raw/ecb_meetings.parquet | dates of every Governing Council monetary-policy decision, 1999- (ad hoc ones flagged) | ecb.europa.eu monetary policy decisions pages |
| lab/raw/ecb_yc.parquet | euro-area AAA government spot yields 3M, 6M, 1Y, 2Y, daily, 2004- | ECB Data Portal YC dataset; ECB reuse with acknowledgement |
| lab/raw/estat_hicp_ea.parquet | euro-area HICP: all items, core (excl. energy, food, alcohol, tobacco), energy, food, services; y/y and m/m; member states' headline y/y and m/m, 1996- | Eurostat prc_hicp_minr (ECOICOP 2, full back-calculated history); Eurostat reuse policy, credit |

Credit: "Source: ECB Data Portal; Eurostat (prc_hicp_minr)". The HICP values are the current vintage (HICP is rarely revised; the Nov and
Dec 2025 headline rates equal the first releases that settled the Polymarket markets).
"""
from __future__ import annotations

import io
import re
import sys

import pandas as pd

from forecasting_engine import histories as HS

ECB = "https://data-api.ecb.europa.eu/service/data/"
UA = {"User-Agent": "forecasting-engine/0.1"}
YC = {"yc_3m": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M", "yc_6m": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_6M",
      "yc_1y": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y", "yc_2y": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"}
HICP_EA = {"total": "TOTAL", "core": "TOT_X_NRG_FOOD", "energy": "NRG", "food": "FOOD", "services": "SERV"}


def _get(url, params=None):
    import requests
    r = requests.get(url, params=params, headers=UA, timeout=120)
    r.raise_for_status()
    return r


def _ecb(key: str, since: str = "1999-01-01") -> pd.DataFrame:
    flow, k = key.split("/", 1)
    d = pd.read_csv(io.BytesIO(_get(f"{ECB}{flow}/{k}", params={"format": "csvdata", "startPeriod": since}).content))
    return pd.DataFrame({"date": pd.to_datetime(d["TIME_PERIOD"], utc=True), "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")}).dropna()


def rates_load(log=print) -> dict:
    d = _ecb("FM/B.U2.EUR.4F.KR.DFR.LEV").sort_values("date")
    days = pd.date_range(pd.Timestamp("1999-01-01", tz="UTC"), pd.Timestamp.now(tz="UTC").normalize(), freq="D")
    s = d.set_index("date")["value"].reindex(days.union(d["date"])).ffill().reindex(days).dropna()   # rate in force on each calendar day
    HS.put_parquet("lab/raw/ecb_dfr_daily.parquet", s.to_frame("dfr"))
    log(f"dfr: {len(d)} changes, {s.index[0].date()} - {s.index[-1].date()}, now {s.iloc[-1]}")
    return dict(changes=len(d))


def meetings_load(log=print) -> dict:
    """Every monetary-policy decision date from the ECB's yearly decision pages (press release codes ecb.mpYYMMDD / ecb.prYYMMDD)."""
    out = set()
    for y in range(1999, pd.Timestamp.now().year + 1):
        s = _get(f"https://www.ecb.europa.eu/press/govcdec/mopo/{y}/html/index_include.en.html").text
        for c in re.findall(r"/(?:ecb\.)?(?:mp|pr)(\d{6})[~.]", s):
            t = pd.Timestamp(f"20{c[:2]}-{c[2:4]}-{c[4:]}" if int(c[:2]) < 90 else f"19{c}", tz="UTC")
            if t.year == y:
                out.add(t)
    past = set(out)
    import html as H                                                 # upcoming decisions: the published calendar (two-day meetings: day 2)
    t = H.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", _get("https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html").text)))
    cal = sorted({pd.Timestamp(pd.to_datetime(d, format="%d/%m/%Y"), tz="UTC")
                  for d in re.findall(r"(\d{2}/\d{2}/\d{4}) Governing Council of the ECB: monetary policy meeting", t)})
    for d in cal:
        if d + pd.Timedelta(days=1) not in cal and d > max(past):
            out.add(d)
    m = pd.DataFrame({"date": sorted(out)})
    m["upcoming"] = ~m["date"].isin(past)
    m["scheduled"] = ~m["date"].isin(UNSCHEDULED)                   # every other decision was on the calendar published months ahead
    HS.put_parquet("lab/raw/ecb_meetings.parquet", m.set_index("date"))
    log(f"meetings: {len(m)} decisions, {int((~m['scheduled']).sum())} unscheduled")
    return dict(meetings=len(m))


def yc_load(log=print) -> dict:
    rows = [(_ecb(k, "2004-09-01").assign(id=n)) for n, k in YC.items()]
    y = pd.concat(rows)
    HS.put_parquet("lab/raw/ecb_yc.parquet", y.set_index("date"))
    log(f"yield curve: {y['id'].nunique()} tenors, {len(y)} rows, last {y['date'].max().date()}")
    return dict(rows=len(y))


def hicp_ea_load(log=print) -> dict:
    rows = []
    for name, code in HICP_EA.items():
        for unit, sfx in (("RCH_A", "yoy"), ("RCH_M", "mom")):
            j = _get("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_minr",
                     params={"geo": "EA", "coicop18": code, "unit": unit}).json()
            idx = j["dimension"]["time"]["category"]["index"]
            inv = {v: k for k, v in idx.items()}
            for k, v in j["value"].items():
                rows.append((pd.Timestamp(inv[int(k)] + "-01", tz="UTC"), f"{name}_{sfx}", float(v)))
    from forecasting_engine.lab.panel import EU27                          # member states: headline y/y and m/m (Greece is EL at Eurostat)
    for unit, sfx in (("RCH_A", "yoy"), ("RCH_M", "mom")):
        j = _get("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_minr",
                 params=[("coicop18", "TOTAL"), ("unit", unit)] + [("geo", "EL" if c == "GR" else c) for c in EU27]).json()
        geo = {v: k for k, v in j["dimension"]["geo"]["category"]["index"].items()}
        tim = {v: k for k, v in j["dimension"]["time"]["category"]["index"].items()}
        nt = len(tim)
        for k, v in j["value"].items():
            g, t = divmod(int(k), nt)
            cc = "GR" if geo[g] == "EL" else geo[g]
            rows.append((pd.Timestamp(tim[t] + "-01", tz="UTC"), f"{cc}_{sfx}", float(v)))
    df = pd.DataFrame(rows, columns=["date", "id", "value"]).sort_values(["id", "date"])
    HS.put_parquet("lab/raw/estat_hicp_ea.parquet", df.set_index("date"))
    log(f"hicp_ea: {df['id'].nunique()} series, last {df['date'].max().date()}")
    return dict(rows=len(df))


UNSCHEDULED = pd.to_datetime(["2001-09-17", "2008-10-08", "2020-03-18", "2022-06-15"], utc=True)   # ad hoc meetings (9/11, Lehman, PEPP, spreads)
LOADERS = {"rates": rates_load, "meetings": meetings_load, "yc": yc_load, "hicp_ea": hicp_ea_load}


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    for k in (LOADERS if not a or a[0] == "all" else a):
        LOADERS[k]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
