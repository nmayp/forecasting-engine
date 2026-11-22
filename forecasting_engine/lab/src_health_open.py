"""Open national respiratory histories from before 2022 for health_w.

health_w's target (ERVISS) starts in 2022, so these series enter as DRIVERS, never as target changes: one EU-wide column ("*") each, as the
countries are outside the 28 target series (respiratory seasons in Europe move within a few weeks of each other). Each source gives
(a) its live weekly level and change, known at its worst-case publication lag, and (b) its pre-2020 seasonal climatology, built like the
other clim_* drivers (src_health_long.climatology), known years ahead.

    python -m forecasting_engine.lab.src_health_open load     # download, store lab/raw/health_open_week.parquet
    python -m forecasting_engine.lab.src_health_open          # print the drivers

| Column | Source | Weeks | Lag used (Monday stamp +) and evidence | Licence |
|---|---|---|---|---|
| ch_ili | Swiss FOPH, Sentinella, ILI consultations per 100 000, extrapolated, Switzerland (api.idd.bag.admin.ch export INFLUENZA_sentinella, incValue) | 2013-W01 - | 16 days. The IDD is updated every Wednesday; on Tue 20 Oct 2026 the latest week is 2026-W41 (5-11 Oct), out Wed 14 Oct = +9 days; one skipped Wednesday (holidays) gives +16 | opendata.swiss "terms_by": open use, source must be cited, commercial use allowed ("Source: FOPH, Sentinella") |
| en_flu_pos, en_rsv_pos | UKHSA data dashboard API, England, PCR test positivity per week, influenza and RSV, all ages | 2017-07 - | 18 days. Weekly until May 2026, fortnightly since 18 May 2026; on 20 Oct 2026 the latest week begins 28 Sep, so a week can be out as late as +17 days | Open Government Licence v3.0 ("Contains public sector information licensed under the OGL v3.0", UKHSA) |
| sc_flu, sc_rsv, sc_hmpv, sc_rhino | Public Health Scotland open data (opendata.nhs.scot, Viral respiratory diseases, Weekly cases for all pathogens, Scotland), laboratory-confirmed cases per week, log(1 + n) | 2016-W40 - | 18 days. Updated every Thursday: week ending Sun 11 Oct 2026 out Thu 15 Oct = +10 days from its Monday; a skipped Thursday gives +17 | UK Open Government Licence (dataset licence on the portal; "Public Health Scotland, OGL v3.0") |

Checked and not used: Belgium Sciensano and the Netherlands RIVM / Nivel publish ILI only in weekly PDF bulletins or charts (no open
machine-readable history found); Spain ISCIII SiVIRA, Italy ISS InfluNet / RespiVirNet, Sweden, Norway, Denmark: no open-licensed
machine-readable pre-2022 series found in this pass (search tools were rate-limited; worth a second look).
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import requests

from forecasting_engine import histories as HS

STORE = "lab/raw/health_open_week.parquet"
LAG_H = {"ch": 16 * 24.0, "en": 18 * 24.0, "sc": 18 * 24.0}
CH_URL = "https://api.idd.bag.admin.ch/api/v1/export/current/INFLUENZA_sentinella/csv"
UK_URL = ("https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/{t}/geography_types/Nation/"
          "geographies/England/metrics/{m}")
SC_URL = "https://www.opendata.nhs.scot/api/3/action/datastore_search"
SC_RES = "d08c0fbd-71e3-47b3-8957-e47490a130b9"
SC_PATH = {"Influenza (All)": "sc_flu", "Respiratory Syncytial Virus": "sc_rsv", "RSV": "sc_rsv", "HMPV": "sc_hmpv", "Rhinovirus": "sc_rhino"}
NOTE = {"ch": "Source: FOPH, Sentinella (opendata.swiss, open use with source, commercial allowed); ILI consultations per 100 000",
        "en": "UKHSA data dashboard (OGL v3.0), England PCR positivity",
        "sc": "Public Health Scotland open data (OGL v3.0), laboratory-confirmed cases, log(1 + n)"}


def _monday_iso(s: pd.Series) -> pd.DatetimeIndex:
    yw = s.astype(str).str.extract(r"(\d{4})-W(\d{1,2})")
    return pd.to_datetime(yw[0] + "-W" + yw[1].str.zfill(2) + "-1", format="%G-W%V-%u", utc=True, errors="coerce")


def _ch() -> pd.Series:
    r = requests.get(CH_URL, timeout=180)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text), low_memory=False)
    d = d[(d.temporal_type == "week") & (d.georegion == "CH") & (d.agegroup == "all") & (d.sex == "all")]
    return pd.Series(pd.to_numeric(d.incValue, errors="coerce").to_numpy(), index=_monday_iso(d.temporal)).dropna().sort_index()


def _uk(topic: str, metric: str) -> pd.Series:
    out, url = {}, UK_URL.format(t=topic, m=metric)
    params = {"age": "all", "page_size": 365, "page": 1, "format": "json"}
    while True:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        j = r.json()
        for x in j["results"]:
            if x.get("stratum", "default") == "default" and x.get("sex", "all") == "all":
                out[pd.Timestamp(x["date"], tz="UTC")] = float(x["metric_value"])
        if not j.get("next"):
            break
        params["page"] += 1
    return pd.Series(out).sort_index()


def _sc() -> pd.DataFrame:
    rows, off = [], 0
    while True:
        r = requests.get(SC_URL, params={"resource_id": SC_RES, "limit": 5000, "offset": off}, timeout=120)
        r.raise_for_status()
        rec = r.json()["result"]["records"]
        rows += rec
        if len(rec) < 5000:
            break
        off += 5000
    d = pd.DataFrame(rows)
    d = d[d.Pathogen.isin(SC_PATH)]
    d["date"] = pd.to_datetime(d.WeekBeginning.astype(str), format="%Y%m%d", utc=True)
    d["col"] = d.Pathogen.map(SC_PATH)
    return d.pivot_table(index="date", columns="col", values="NumberCasesPerWeek", aggfunc="sum").sort_index()


def load(log=print) -> dict:
    parts = {"ch_ili": _ch(), "en_flu_pos": _uk("Influenza", "influenza_testing_positivityByWeek"),
             "en_rsv_pos": _uk("RSV", "RSV_testing_positivityByWeek")}
    df = pd.DataFrame(parts).join(_sc(), how="outer").sort_index()
    df.index.name = "date"
    HS.put_parquet(STORE, df)
    out = {c: (str(df[c].first_valid_index().date()), str(df[c].last_valid_index().date())) for c in df.columns}
    log(f"open respiratory histories: {out}")
    return out


def weekly() -> pd.DataFrame | None:
    """log(1 + value) per column on Monday stamps."""
    try:
        w = HS.get_parquet(STORE)
    except Exception:  # noqa: BLE001
        return None
    if w is None:
        return None
    ix = pd.DatetimeIndex(w.index)
    w.index = ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")
    return np.log1p(w.clip(lower=0))


def features() -> dict[str, tuple[pd.DataFrame, float, bool, str]]:
    """name -> (frame, avail_h, ahead, note), appended to health_w in this order (never reorder)."""
    from forecasting_engine.lab import src_health_long as SHL
    w = weekly()
    out = {}
    if w is None:
        return out
    for c in w.columns:
        src = c[:2]
        x = w[[c]].dropna().rename(columns={c: "*"})
        out[f"op_{c}"] = (x, LAG_H[src], False, f"{NOTE[src]}: {c}, weekly level")
        out[f"op_{c}_chg"] = (x.diff(), LAG_H[src], False, f"{NOTE[src]}: {c}, change from the week before")
    for c in ("ch_ili", "en_flu_pos", "sc_flu", "sc_rsv"):            # pre-2020 seasonal shapes (CH: 7 seasons, England 2, Scotland 3)
        if c in w.columns:
            cl = SHL.climatology(w[[c]].dropna().rename(columns={c: "*"}), min_seasons=2)
            if cl is None or cl.empty:
                continue
            f = SHL.on_mondays(cl)
            src = c[:2]
            out[f"clim_op_{c}"] = (f, SHL.KNOWN_BEFORE_H, True, f"{NOTE[src]}: {c}, typical seasonal log level at the target week, pre-2020")
            out[f"clim_op_{c}_slope"] = (f.diff().bfill(), SHL.KNOWN_BEFORE_H, True, f"{NOTE[src]}: {c}, typical weekly change at the target week")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["load"]:
        print(load())
        sys.exit(0)
    for k, (f, av, ah, nt) in features().items():
        print(k, f.shape, f.dropna().index.max().date(), round(av / 24, 1), "ahead" if ah else "")
    sys.exit(0)
