"""Mixed-frequency nowcasting inputs: the part of a month or week that is already published at a forecast origin, for
the layers where the lab loses to Chronos-2 (hicp_m, tourism_m, health_w). Every value carries the moment it was really known.

    python -m forecasting_engine.lab.src_nowcast            # or: flash | wastewater | school

| Store name | Content | Source, delay | Licence |
|---|---|---|---|
| lab/raw/hicp_flash.parquet | per (month, geo): flash y/y, first-published final y/y (Eurostat prc_hicp_fp / prc_hicp_fpd, release FLS / FIN, CP00 / TOTAL, RCH_A), flash release time (UTC) | flash of month M: Eurostat euro indicators release, last working day of M or the first days of M+1, 11:00 CET (10:00 UTC used); dates read from the release pages themselves (ec.europa.eu/eurostat/web/products-euro-indicators/w/2-DDMMYYYY-xp) | Eurostat reuse policy (free reuse incl. commercial, with attribution) |
| lab/raw/wastewater_de.parquet | weekly population-weighted virus load in German wastewater, normalised (viruslast_normalisiert), one column per pathogen (sars2, flu_a, flu_b, rsv) | RKI / UBA AMELAG (GitHub robert-koch-institut/Abwassersurveillance_AMELAG), weekly points stamped on the week's Wednesday, released the Tuesday after (version 2026-10-13 held the 2026-10-07 point): 8 days used | CC BY 4.0 ("Robert Koch-Institut, Umweltbundesamt: Abwassersurveillance AMELAG") |
| lab/raw/school_holidays.parquet | daily share of a country's top-level regions on school holiday (0-1), NaN where the source has no school calendar that year | OpenHolidays API (openholidaysapi.org), school calendars are set months to years ahead | ODbL 1.0 (attribution + share-alike for a derived database) |

Not built here, derived in panel from data already in the store: month-to-date Oil Bulletin fuel prices (weekly, Monday + 4 days), month-to-date
day-ahead power prices (ENTSO-E, back-tests only), month-to-date flights (EUROCONTROL airport panel standing in for the Daily Traffic Variation
dashboard, D-1; back-tests only). See panel._mtd.

Skipped (stated so nobody looks again): the EU Wastewater Observatory (wastewater-observatory.jrc.ec.europa.eu) resets the connection from this
container; RIVM's national SARS-CoV-2 wastewater file (data.rivm.nl/covid-19/COVID-19_rioolwaterdata_landelijk.csv) is reported weekly with
6-12 days of delay and SARS-CoV-2 only, so it is staler than the ILI target itself; no open daily or weekly hotel / short-stay indicator with
history exists (Eurostat's platform data are monthly with a longer lag than tour_occ_nim, STR / AirDNA are commercial).
"""
from __future__ import annotations

import io
import re
import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import UA, _get

EI_URL = "https://ec.europa.eu/eurostat/web/products-euro-indicators/w/2-{d:%d%m%Y}-{s}"
FLASH_FROM = "2017-08"                 # prc_hicp_fp has country flash rates from 2017-08 (euro area members; AT from 2021, HR from 2023)
FLASH_UTC_HOUR = 10                    # 11:00 CET (09:00 UTC in summer: 10:00 is the later, safe bound)
AMELAG_URL = "https://raw.githubusercontent.com/robert-koch-institut/Abwassersurveillance_AMELAG/main/amelag_aggregierte_kurve.tsv"
AMELAG_LAG_DAYS = 8.0
AMELAG_TYPES = {"SARS-CoV-2": "sars2", "Influenza A": "flu_a", "Influenza B": "flu_b", "RSV A/B": "rsv"}
OH = "https://openholidaysapi.org"
SCHOOL_CC = ("AT", "BE", "BG", "HR", "CZ", "EE", "FR", "DE", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "CH")
SCHOOL_AHEAD_DAYS = 180.0              # a school calendar is fixed at least half a year before (most are set for several years)
ATTR = {"flash": "Source: Eurostat, prc_hicp_fp / prc_hicp_fpd and euro indicators releases. Release dates read by the forecasting engine.",
        "wastewater": "Robert Koch-Institut, Umweltbundesamt (2026): Abwassersurveillance AMELAG, CC BY 4.0. Adapted by the forecasting engine (one series per pathogen).",
        "school": "School holidays: OpenHolidays API (openholidaysapi.org), ODbL 1.0. Aggregated by the forecasting engine to the share of regions on holiday."}


# ------------------------------------------------------------------------------------------------------------------------------ HICP flash
def _estat(ds: str, **flt) -> pd.DataFrame:
    from forecasting_engine.eurostat import parse_jsonstat
    return parse_jsonstat(_get(HS.ESTAT_BASE + ds, params=dict(format="JSON", lang="en", unit="RCH_A", sinceTimePeriod=FLASH_FROM, **flt)).json())


def _is_flash(day: pd.Timestamp) -> bool:
    """True if a euro indicators release of that day is the HICP flash (title 'Euro area annual inflation ...'; the scan window holds no final)."""
    import time
    import requests
    for s in ("ap", "bp", "cp", "dp", "ep", "fp"):
        for i in range(8):
            try:
                r = requests.get(EI_URL.format(d=day, s=s), headers=UA, timeout=60, allow_redirects=False, stream=True)
                if r.status_code != 429:
                    break
                r.close()
            except requests.RequestException:
                if i == 7:
                    raise
            time.sleep(5 * 2 ** min(i, 4))                     # the portal rate-limits (429): back off
        if r.status_code == 429:
            raise RuntimeError(f"rate-limited on {day.date()}")
        if r.status_code != 200:                           # ids run on per day (an -ap can be withdrawn: 31 Aug 2017 starts at -bp)
            if s == "ap":
                continue
            return False
        head = next(r.iter_content(40000), b"").decode("utf-8", "ignore")
        r.close()
        m = re.search(r"<title>([^<]*)", head)
        if m and re.search(r"(?i)euro area (annual )?inflation", m.group(1)):
            return True
        time.sleep(0.3)
    return False


def flash_dates(months, log=print) -> dict:
    """Release day of the flash for each month, most likely days first (M's last working day, then the first working days of M+1, then
    the days before from the 24th): the first whose release is the inflation flash. Found days are kept in lab/raw/hicp_flash_dates.json
    (the scan is slow: the portal rate-limits), so a rerun only looks up new months."""
    import json
    raw = HS.get_bytes("lab/raw/hicp_flash_dates.json")
    cache = json.loads(raw) if raw else {}
    out = {}
    for M in map(pd.Timestamp, months):
        k = str(M.date())
        if cache.get(k):
            out[M] = pd.Timestamp(cache[k])
            continue
        nxt = M + pd.offsets.MonthBegin(1)
        bd = lambda a, b: [x for x in pd.date_range(a, b, freq="D") if x.dayofweek < 5]   # noqa: E731
        late = bd(M + pd.Timedelta(days=23), nxt - pd.Timedelta(days=1))
        cand = late[-1:] + bd(nxt, nxt + pd.Timedelta(days=9)) + late[:-1][::-1]
        out[M] = next((day for day in cand if _is_flash(day)), None)
        if out[M] is not None:
            cache[k] = str(out[M].date())
            HS.put_bytes("lab/raw/hicp_flash_dates.json", json.dumps(cache, indent=0, sort_keys=True).encode(), "application/json")
    miss = [str(k.date()) for k, v in out.items() if v is None]
    log(f"flash dates: {len(out) - len(miss)} found, missing {miss}")
    return out


def flash_load(log=print) -> dict:
    a = pd.concat([_estat("prc_hicp_fp", coicop="CP00"), _estat("prc_hicp_fpd", coicop18="TOTAL").rename(columns={"coicop18": "coicop"})])
    a = a[a["time"].str.match(r"^\d{4}-\d{2}$")].dropna(subset=["value"])
    a["geo"] = a["geo"].astype(str).replace({"EL": "GR"})
    w = a.pivot_table(index=["time", "geo"], columns="release", values="value", aggfunc="last").reset_index()
    w = w.rename(columns={"FLS": "flash", "FIN": "final_first"})
    w["month"] = pd.to_datetime(w["time"] + "-01", utc=True)
    months = sorted(w.loc[w["flash"].notna(), "month"].dt.tz_localize(None).unique())
    rel = flash_dates(months, log)
    w["release"] = w["month"].dt.tz_localize(None).map(lambda m: rel.get(pd.Timestamp(m)))
    w["release"] = pd.to_datetime(w["release"]).dt.tz_localize("UTC") + pd.Timedelta(hours=FLASH_UTC_HOUR)
    out = w[["month", "geo", "flash", "final_first", "release"]].sort_values(["month", "geo"]).set_index("month")
    HS.put_parquet("lab/raw/hicp_flash.parquet", out)
    d = (out["release"] - (out.index + pd.offsets.MonthBegin(1))).dt.days.dropna()
    log(f"hicp flash: {out['flash'].notna().sum()} country-months, {out.index.min().date()} - {out.index.max().date()}; release day vs next month start "
        f"min {d.min()} max {d.max()}; known by the 1st 00:00 UTC in {(d < 0).groupby(level=0).first().mean():.0%} of months")
    return dict(rows=int(out["flash"].notna().sum()), months=len(months), release_days_vs_month_end=d.groupby(level=0).first().value_counts().to_dict())


def flash_news(final_lag_h: float) -> pd.DataFrame | None:
    """Event series (stamped when known, avail 0): at the flash release of month M, flash(M) minus the first-published final of M-1 (the latest
    target then known); when the final of M comes out (month start + final_lag_h, the target's own delay), 0: the flash no longer adds anything."""
    f = HS.get_parquet("lab/raw/hicp_flash.parquet")
    if f is None:
        return None
    f = f.reset_index()
    ff = f.pivot_table(index="month", columns="geo", values="final_first")
    fl = f.pivot_table(index="month", columns="geo", values="flash")
    rel = f.dropna(subset=["release"]).groupby("month")["release"].first()
    news = (fl - ff.shift(1, freq="MS").reindex(fl.index)).reindex(rel.index)
    news.index = pd.DatetimeIndex(rel.to_list())
    zero = (fl.notna().astype(float) * 0).where(fl.notna())
    zero.index = zero.index + pd.Timedelta(hours=final_lag_h)
    out = pd.concat([news, zero]).sort_index()
    return out[~out.index.duplicated(keep="last")]


# ------------------------------------------------------------------------------------------------------------------------------ wastewater
def wastewater_load(log=print) -> dict:
    d = pd.read_csv(io.BytesIO(_get(AMELAG_URL).content), sep="\t")
    d = d[d["typ"].isin(AMELAG_TYPES)]
    d["date"] = pd.to_datetime(d["datum"], utc=True)
    w = d.assign(k=d["typ"].map(AMELAG_TYPES)).pivot_table(index="date", columns="k", values="viruslast_normalisiert").sort_index()
    HS.put_parquet("lab/raw/wastewater_de.parquet", w)
    log(f"wastewater DE (AMELAG): {list(w.columns)}, {w.index.min().date()} - {w.index.max().date()}, {len(w)} weeks")
    return dict(series={c: [str(w[c].first_valid_index().date()), str(w[c].last_valid_index().date())] for c in w.columns})


# ------------------------------------------------------------------------------------------------------------------------------ school holidays
def _top(code: str) -> str:
    return "-".join(code.split("-")[:2])


def school_load(years=range(2015, 2028), log=print) -> dict:
    """Unweighted share of top-level regions (Bundeslaender, regioni, comunidades; Belgium's three language communities) on school holiday."""
    cols, info = {}, {}
    for cc in SCHOOL_CC:
        subs = _get(f"{OH}/Subdivisions", params=dict(countryIsoCode=cc, languageIsoCode="EN")).json()
        evs = [e for y in years for e in _get(f"{OH}/SchoolHolidays", params=dict(countryIsoCode=cc, languageIsoCode="EN", validFrom=f"{y}-01-01",
                                                                                     validTo=f"{y}-12-31")).json()]
        evs = list({e["id"]: e for e in evs}.values())
        g2t: dict = {}
        def walk(s):  # noqa: E306
            for g in s.get("groups") or []:
                g2t.setdefault(g["code"], set()).add(_top(s["code"]))
            for c in s.get("children") or []:
                walk(c)
        for s in subs:
            walk(s)
        tops = sorted({_top(s["code"]) for s in subs})
        if not tops:                                                            # no subdivisions (BE): the groups are the regions
            tops = sorted({g["code"] for e in evs for g in e.get("groups") or []}) or [cc]
            g2t = {t: {t} for t in tops}
        idx = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D", tz="UTC")
        on = {t: np.zeros(len(idx), bool) for t in tops}
        for e in evs:
            a, b = idx.searchsorted(pd.Timestamp(e["startDate"], tz="UTC")), idx.searchsorted(pd.Timestamp(e["endDate"], tz="UTC"), "right")
            if e.get("nationwide") or not (e.get("subdivisions") or e.get("groups")):
                hit = set(tops)
            else:
                hit = {_top(s["code"]) for s in e.get("subdivisions") or []}
                for g in e.get("groups") or []:
                    hit |= g2t.get(g["code"], set())
            for r in hit & set(tops):
                on[r][a:b] = True
        sh = pd.Series(np.mean([on[r] for r in tops], axis=0), index=idx)
        yr = sh.groupby(sh.index.year).mean()
        have = sorted(yr.index[yr >= 0.6 * yr[yr > 0].median()]) if (yr > 0).any() else []   # partial years (calendar not loaded yet) are unknown
        sh[~sh.index.year.isin(have)] = np.nan                                  # no school calendar that year: unknown, not "no holiday"
        cols[cc] = sh
        info[cc] = dict(regions=len(tops), years=[int(y) for y in have])
    df = pd.DataFrame(cols)
    df.index.name = "date"
    HS.put_parquet("lab/raw/school_holidays.parquet", df)
    log(f"school holidays: {df.shape[1]} countries, " + ", ".join(f"{k} {v['years'][0] if v['years'] else '-'}-{v['years'][-1] if v['years'] else ''}"
                                                                    for k, v in info.items()))
    return info


def main(argv=None) -> int:
    args = argv or sys.argv[1:] or ["flash", "wastewater", "school"]
    for a in args:
        print({"flash": flash_load, "wastewater": wastewater_load, "school": school_load}[a]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
