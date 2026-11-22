"""Cyber inputs for kev_d.

    python -m forecasting_engine.lab.src_cyber load NVD_JSONL   # rows {id, pub, score, vendors} from the NVD CVE API 2.0 (fetch_nvd)

| Store name | Content | Delay | Licence |
|---|---|---|---|
| lab/raw/nvd_daily.parquet | new CVE records published by NVD per UTC day (2019-) | end of the day + 6 h | NVD: US Government work, public domain; NVD asks for the notice "This product uses data from the NVD API but is not endorsed or certified by the NVD" |
| (computed) CISA KEV by vendor (lab/raw/kev.parquet) | 30-day rolling KEV additions: Microsoft, network-edge vendors, mobile and browsers, vendor breadth (not the ransomware flag: it is set after the addition) | end of day | US Government work (public domain) |
| (computed) Patch Tuesday, US federal holidays | second Tuesday of each month (Microsoft, Adobe, SAP releases); US federal holidays (holidays package) | known years ahead | - |

Only the publication count is used from NVD: the CPE vendor lists and NVD's own CVSS scores are added by NVD analysis days or months later
(a large backlog since 2024), so using them as of today would leak the future.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from forecasting_engine import histories as HS

DAY = 24.0
NVD_STORE = "lab/raw/nvd_daily.parquet"
NVD_NOTICE = "This product uses data from the NVD API but is not endorsed or certified by the NVD."
EDGE = ("Cisco", "Ivanti", "Fortinet", "Citrix", "Palo Alto Networks", "SonicWall", "Juniper", "F5", "Zyxel", "D-Link", "VMware", "Progress",
        "Atlassian", "Pulse Secure", "Check Point", "Barracuda", "TP-Link", "DrayTek", "QNAP", "Synology")
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def fetch_nvd(out_path: str, start: str = "2019-01-01", end: str | None = None, log=print) -> None:
    """Every CVE record by its NVD publication date, 110-day windows, 2000 per page, 7 s between calls (the keyless rate limit)."""
    import datetime as dt
    import time
    import requests
    a = dt.datetime.fromisoformat(start)
    e = dt.datetime.fromisoformat(end) if end else dt.datetime.utcnow()
    with open(out_path, "w") as fh:
        while a < e:
            b = min(a + dt.timedelta(days=110), e)
            i0 = 0
            while True:
                p = dict(pubStartDate=a.strftime("%Y-%m-%dT00:00:00.000"), pubEndDate=b.strftime("%Y-%m-%dT00:00:00.000"), resultsPerPage=2000,
                         startIndex=i0)
                r = None
                for k in range(8):
                    try:
                        r = requests.get(NVD_API, params=p, headers={"User-Agent": "forecasting-engine/0.1"}, timeout=120)
                        if r.status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(10 * (k + 1))
                j = r.json()
                for v in j.get("vulnerabilities", []):
                    fh.write(json.dumps(dict(id=v["cve"]["id"], pub=v["cve"]["published"])) + "\n")
                i0 += 2000
                time.sleep(7)
                if i0 >= j.get("totalResults", 0):
                    break
            log(f"nvd {a.date()}..{b.date()}")
            a = b


def load(path: str, log=print) -> dict:
    rows = [json.loads(x) for x in open(path)]
    d = pd.DataFrame(rows).drop_duplicates("id")
    d["date"] = pd.to_datetime(d["pub"], utc=True, format="ISO8601").dt.floor("D")
    s = d.groupby("date").size().astype(float)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)
    HS.put_parquet(NVD_STORE, s.to_frame("n"))
    out = dict(cves=len(d), first=str(s.index.min().date()), last=str(s.index.max().date()), mean_per_day=round(float(s.iloc[-365:].mean()), 1))
    log(out)
    return out


def _kev() -> pd.DataFrame:
    d = HS.get_parquet("lab/raw/kev.parquet").reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.floor("D")
    return d


def _count(mask, d, idx) -> pd.Series:
    return d[mask].groupby("date").size().reindex(idx, fill_value=0).astype(float)


def drivers() -> list:
    from forecasting_engine.lab.panel import Driver
    k = _kev()
    idx = pd.date_range(pd.Timestamp("2021-11-03", tz="UTC"), k["date"].max(), freq="D")
    r30 = lambda s: s.rolling(30, min_periods=1).sum().to_frame("*")   # noqa: E731
    kn = "CISA KEV (public domain), additions in the last 30 days"
    D = [Driver("kev_ms_30d", "cyber", r30(_count(k["vendorProject"].eq("Microsoft"), k, idx)), DAY, note=f"{kn}: Microsoft"),
         Driver("kev_edge_30d", "cyber", r30(_count(k["vendorProject"].isin(EDGE), k, idx)), DAY, note=f"{kn}: network-edge and IT vendors"),
         Driver("kev_mobile_web_30d", "cyber", r30(_count(k["vendorProject"].isin(["Apple", "Google", "Mozilla", "Samsung"]), k, idx)), DAY,
                note=f"{kn}: Apple, Google, Mozilla, Samsung"),
         Driver("kev_vendors_30d", "cyber", (k.groupby("date")["vendorProject"].nunique().reindex(idx, fill_value=0).astype(float)
                                              .rolling(30, min_periods=1).sum()).to_frame("*"), DAY, note=f"{kn}: vendor breadth")]
    n = HS.get_parquet(NVD_STORE)
    if n is not None:
        s = n["n"].astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_convert("UTC") if pd.DatetimeIndex(s.index).tz is not None else pd.DatetimeIndex(s.index).tz_localize("UTC")
        D += [Driver("nvd_new_1d", "cyber", s.to_frame("*"), DAY + 6, note=f"NVD CVE records published that day (public domain). {NVD_NOTICE}"),
              Driver("nvd_new_7d", "cyber", s.rolling(7, min_periods=7).sum().to_frame("*"), DAY + 6, note="NVD, 7-day sum"),
              Driver("nvd_new_28d", "cyber", s.rolling(28, min_periods=28).sum().to_frame("*"), DAY + 6, note="NVD, 28-day sum")]
    full = pd.date_range(pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2027-12-31", tz="UTC"), freq="D")
    pt = (full.dayofweek == 1) & (full.day >= 8) & (full.day <= 14)
    pts = pd.Series(pt.astype(float), index=full)
    import holidays as HL
    us = HL.country_holidays("US", years=range(2019, 2028))
    ush = pd.Series([float(t.date() in us) for t in full], index=full)
    D += [Driver("patch_tuesday", "calendar", pts.to_frame("*"), -365 * DAY, ahead=True, note="second Tuesday of the month (Microsoft, Adobe, SAP)"),
          Driver("patch_tuesday_next", "calendar", pts.shift(1, fill_value=0.0).to_frame("*"), -365 * DAY, ahead=True, note="the day after"),
          Driver("us_holiday", "calendar", ush.to_frame("*"), -365 * DAY, ahead=True, note="US federal holiday (CISA adds fewer entries)")]
    return D


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    if a and a[0] == "load":
        load(a[1])
    elif a and a[0] == "fetch":
        fetch_nvd(a[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
