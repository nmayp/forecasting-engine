"""Health layer: weekly primary-care respiratory illness (ILI, else ARI) consultation rates per EU/EEA country, from ECDC ERVISS.

    python -c "from forecasting_engine.lab.src_health import health_load; print(health_load())"

| Source | Store name | Content | Licence |
|---|---|---|---|
| ECDC ERVISS (GitHub EU-ECDC/Respiratory_viruses_weekly_data) | lab/raw/health_ervis.parquet | wide: Monday of ISO week (UTC) x ISO2, all ages, rate | ECDC data: reuse with acknowledgement (see ERVISS_ATTR) |
| same | lab/raw/health_ervis_long.parquet | long: date, geo, indicator (ILI/ARI), value, all ages | same |
| same, data/sentinelTestsDetectionsPositivity.csv + data/nonSentinelTestsDetections.csv | lab/raw/health_ervis_viro.parquet | long: date, geo, pathogen (flu / rsv / sars2), kind (sent = primary-care sentinel, nonsent = other laboratories), tests, detections | same |
| same, data/SARIRates.csv | lab/raw/health_ervis_sari.parquet | wide: Monday of ISO week x ISO2, SARI hospital admission rate, all ages (16 countries) | same |

every ERVISS file is published together on the Friday after the week, so positivity and SARI carry the
target's own delay (week + 5 days); they lead consultations because laboratory positivity rises before the consultation rate does.

Indicator per country: ILI consultation rate when the country reports it, else ARI (BG, CY, DE); ILI_OR_ARI in the result says which.
Units: per 100 000 population, except CY, LU, MT (per 100 consultations) and FI (per 100 000 consultations), so scale per country.
Series start 2022-W25 (ERVISS launch); pre-2022 Flu News Europe / TESSy history is not in this repository. Week W (Mon-Sun) is published
on the Friday after it ends (5 days); data/snapshots/<date>_ILIARIRates.csv keep each weekly vintage for real-time back-tests.
"""
from __future__ import annotations

import io

import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

ERVISS_URL = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/ILIARIRates.csv"
ERVISS_BASE = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/"
PATHOGENS = {"Influenza": "flu", "RSV": "rsv", "SARS-CoV-2": "sars2"}
ERVISS_ATTR = ("Dataset provided by ECDC based on data provided by public health authorities, scientific institutes or health care providers in the "
               "relevant reporting countries and/or by WHO (ERVISS, erviss.org). Adapted by the forecasting engine: total ages only, one indicator per country.")
COUNTRIES = {"Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Cyprus": "CY", "Czechia": "CZ", "Denmark": "DK", "Estonia": "EE",
             "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR", "Hungary": "HU", "Iceland": "IS", "Ireland": "IE", "Italy": "IT",
             "Latvia": "LV", "Liechtenstein": "LI", "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL", "Norway": "NO",
             "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK", "Slovenia": "SI", "Spain": "ES", "Sweden": "SE"}
ILI_OR_ARI: dict = {}


def health_load(log=print) -> dict:
    d = pd.read_csv(io.BytesIO(_get(ERVISS_URL).content))
    d = d[(d["age"] == "total") & d["countryname"].isin(COUNTRIES)]
    d = pd.DataFrame({"date": pd.to_datetime(d["yearweek"] + "-1", format="%G-W%V-%u", utc=True), "geo": d["countryname"].map(COUNTRIES),
                      "indicator": d["indicator"].str[:3], "value": pd.to_numeric(d["value"], errors="coerce")}).dropna()
    HS.put_parquet("lab/raw/health_ervis_long.parquet", d.set_index("date"))
    ili = set(d.loc[d["indicator"] == "ILI", "geo"])
    ILI_OR_ARI.update({g: "ILI" if g in ili else "ARI" for g in d["geo"].unique()})
    w = d[d["indicator"] == d["geo"].map(ILI_OR_ARI)].pivot_table(index="date", columns="geo", values="value").sort_index()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="7D", tz="UTC")).sort_index(axis=1)
    HS.put_parquet("lab/raw/health_ervis.parquet", w)
    miss = float(w.isna().mean().mean())
    log(f"health (ERVISS ILI/ARI): {w.shape[1]} countries, {len(w)} weeks, {w.index.min().date()} - {w.index.max().date()}, missing {miss:.0%}")
    return dict(countries=w.shape[1], weeks=len(w), first=str(w.index.min().date()), last=str(w.index.max().date()), missing=round(miss, 3),
                ari=sorted(g for g, i in ILI_OR_ARI.items() if i == "ARI"))


def _week(yw: pd.Series) -> pd.Series:
    return pd.to_datetime(yw + "-1", format="%G-W%V-%u", utc=True)


def viro_load(log=print) -> dict:
    """sentinel and non-sentinel tests and detections per pathogen (total of each pathogen, all ages), and the SARI rate."""
    out, parts = {}, []
    for f, kind in (("sentinelTestsDetectionsPositivity.csv", "sent"), ("nonSentinelTestsDetections.csv", "nonsent")):
        d = pd.read_csv(io.BytesIO(_get(ERVISS_BASE + f).content))
        d = d[(d["age"] == "total") & d["countryname"].isin(COUNTRIES) & d["pathogen"].isin(PATHOGENS) & (d["pathogentype"] == d["pathogen"])
              & d["indicator"].isin(("tests", "detections"))]
        d = d[(d["pathogen"] != "Influenza") | (d["pathogensubtype"] == "total")]
        d = d.assign(date=_week(d["yearweek"]), geo=d["countryname"].map(COUNTRIES), pathogen=d["pathogen"].map(PATHOGENS), kind=kind,
                     value=pd.to_numeric(d["value"], errors="coerce"))
        w = d.pivot_table(index=["date", "geo", "pathogen", "kind"], columns="indicator", values="value", aggfunc="sum").reset_index()
        parts.append(w)
        out[kind] = dict(rows=len(w), countries=int(w["geo"].nunique()))
    v = pd.concat(parts, ignore_index=True)
    HS.put_parquet("lab/raw/health_ervis_viro.parquet", v.set_index("date"))
    s = pd.read_csv(io.BytesIO(_get(ERVISS_BASE + "SARIRates.csv").content))
    s = s[(s["age"] == "total") & s["countryname"].isin(COUNTRIES) & (s["indicator"] == "SARIrate")]
    s = s.assign(date=_week(s["yearweek"]), geo=s["countryname"].map(COUNTRIES), value=pd.to_numeric(s["value"], errors="coerce"))
    sw = s.pivot_table(index="date", columns="geo", values="value").sort_index()
    HS.put_parquet("lab/raw/health_ervis_sari.parquet", sw)
    out["sari"] = dict(weeks=len(sw), countries=sw.shape[1])
    log(f"health virology / SARI (ERVISS): {out}")
    return out


def positivity(kind: str | None = None, min_tests: int = 10) -> dict[str, pd.DataFrame]:
    """Weekly positivity (%) per pathogen, date x ISO2: detections / tests where tests >= min_tests. kind None = sentinel where it has enough
    tests, else non-sentinel (more countries; laboratories outside the sentinel networks)."""
    v = HS.get_parquet("lab/raw/health_ervis_viro.parquet").reset_index()
    v["date"] = pd.to_datetime(v["date"], utc=True)
    v["pos"] = (v["detections"] / v["tests"] * 100).where(v["tests"] >= min_tests)
    out = {}
    for p in PATHOGENS.values():
        w = {k: v[(v["pathogen"] == p) & (v["kind"] == k)].pivot_table(index="date", columns="geo", values="pos").sort_index() for k in ("sent", "nonsent")}
        out[p] = w[kind] if kind else w["sent"].combine_first(w["nonsent"])
    return out
