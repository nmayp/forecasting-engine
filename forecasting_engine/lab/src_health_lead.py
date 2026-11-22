"""Leading inputs for health_w: what is published before ERVISS at a Monday 00:00 origin, or at a finer grain.

At a health_w origin (Monday 00:00 of week p0) ERVISS holds weeks up to p0-2 (week + 5 days). The national systems were checked for a faster
release; none publishes week p0-1 by Monday 00:00 (real lags below), so the one input that covers p0-1 is Wikipedia reading of the illness
articles in each country's languages (an established ILI nowcasting signal: McIver and Brownstein 2014, Generous et al. 2014).

    python -m forecasting_engine.lab.src_health_lead          # or: wiki | are

| Store name | Content | Source, real publication lag | Licence |
|---|---|---|---|
| lab/raw/wiki_health_daily.parquet | long: date (UTC day), lang, topic (flu / cold), views (user agents, all access, summed over the candidate titles) | Wikimedia REST pageviews API (per-article, daily, 2015-07-); a day is served within 24 h. The weekly driver uses Monday-Friday only and is stamped Monday + 7 days: at the origin the newest day is 48 h old | Wikimedia pageview data: CC0 1.0 (public domain dedication, analytics.wikimedia.org) |
| lab/raw/rki_are_age.parquet | wide: Monday of ISO week x age group (0-4, 5-14, 15-34, 35-59, 60+, 00+), ARE consultation incidence per 100 000, Germany | RKI ARE-Konsultationsinzidenz (GitHub robert-koch-institut), evaluated Tuesday, available from Thursday for the week before: Monday stamp + 10.5 days used (same week as ERVISS at the origin, finer grain: children lead adults) | CC BY 4.0 ("Robert Koch-Institut: ARE-Konsultationsinzidenz") |

Checked, not loaded (real lags at a Monday 00:00 origin; none adds a week):
- France, Sentinelles (sentiweb.fr, Inserm/Sorbonne): weekly incidence published on the Tuesday/Wednesday after the week; same week as ERVISS.
- Germany, RKI GrippeWeb (CC BY 4.0): published every Friday for the week before (README); RKI SARI / ICOSARI hospital incidence: evaluated
  Tuesdays, weekly; AMELAG wastewater is already in the panel (Wednesday stamp + 8 days).
- Spain SiVIRA (ISCIII) and Italy RespiVirNet / InfluNet (ISS): weekly bulletins on Thursday/Friday for the week before (PDF or dashboard).
- Netherlands, Nivel primary-care ILI: weekly, Thursday; RIVM wastewater is SARS-CoV-2 only and 6-12 days late (src_nowcast).
- The EU Wastewater Observatory resets connections from this container (src_nowcast).

Language -> country: one Wikipedia per language; countries that share one (DE/AT, FR/BE/LU, NL/BE, GR/CY, IE/MT on English) take the mean of
their languages' log changes. English is dominated by readers outside Ireland and Malta: a weak signal there.
"""
from __future__ import annotations

import io
import sys
import time
from urllib.parse import quote

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

PV = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{title}/daily/{a}/{b}"
UA = {"User-Agent": "forecasting-engine/0.1"}
START = "20150701"
TITLES = {   # candidate titles per language (article and its usual redirects; a missing title answers 404 and counts zero)
    "flu": {"de": ["Influenza", "Grippe"], "fr": ["Grippe"], "nl": ["Griep", "Influenza"], "it": ["Influenza"], "es": ["Gripe"],
            "pt": ["Gripe"], "pl": ["Grypa"], "cs": ["Chřipka"], "sk": ["Chrípka"], "hu": ["Influenza"], "ro": ["Gripă"], "bg": ["Грип"],
            "el": ["Γρίπη"], "hr": ["Gripa"], "sl": ["Gripa"], "et": ["Gripp"], "lv": ["Gripa"], "lt": ["Gripas"], "fi": ["Influenssa"],
            "sv": ["Influensa"], "da": ["Influenza"], "no": ["Influensa"], "is": ["Inflúensa"], "en": ["Influenza", "Flu"]},
    "cold": {"de": ["Erkältung", "Grippaler Infekt"], "fr": ["Rhume", "Rhinopharyngite"], "nl": ["Verkoudheid"], "it": ["Raffreddore comune", "Raffreddore"],
             "es": ["Resfriado común", "Resfriado"], "pt": ["Resfriado comum", "Constipação"], "pl": ["Przeziębienie"], "cs": ["Nachlazení", "Rýma"],
             "sk": ["Prechladnutie", "Nádcha"], "hu": ["Nátha", "Megfázás"], "ro": ["Răceală", "Răceală comună"], "bg": ["Настинка", "Простуда"],
             "el": ["Κοινό κρυολόγημα", "Κρυολόγημα"], "hr": ["Prehlada"], "sl": ["Prehlad"], "et": ["Nohu"], "lv": ["Saaukstēšanās"],
             "lt": ["Peršalimas", "Sloga"], "fi": ["Flunssa", "Nuhakuume"], "sv": ["Förkylning"], "da": ["Forkølelse"], "no": ["Forkjølelse"],
             "is": ["Kvef"], "en": ["Common cold"]}}
LANGS = {"AT": ("de",), "BE": ("fr", "nl"), "BG": ("bg",), "CY": ("el",), "CZ": ("cs",), "DE": ("de",), "DK": ("da",), "EE": ("et",), "ES": ("es",),
         "FI": ("fi",), "FR": ("fr",), "GR": ("el",), "HR": ("hr",), "HU": ("hu",), "IE": ("en",), "IS": ("is",), "IT": ("it",), "LT": ("lt",),
         "LU": ("fr", "de"), "LV": ("lv",), "MT": ("en", "it"), "NL": ("nl",), "NO": ("no",), "PL": ("pl",), "PT": ("pt",), "RO": ("ro",),
         "SI": ("sl",), "SK": ("sk",)}
WIKI_LAG_H = 7 * 24.0              # Monday stamp, Monday-Friday views: known at the next Monday 00:00 (Friday's views served by Saturday)
ARE_URL = "https://raw.githubusercontent.com/robert-koch-institut/ARE-Konsultationsinzidenz/main/ARE-Konsultationsinzidenz.tsv"
ARE_LAG_H = 10.5 * 24.0
ATTR = {"wiki": "Wikimedia Foundation pageview data (CC0). Aggregated by the forecasting engine per language, topic and week.",
        "are": "Robert Koch-Institut (2026): ARE-Konsultationsinzidenz, CC BY 4.0. Adapted by the forecasting engine (national, by age group)."}


def _pv(lang: str, title: str, b: str) -> pd.Series:
    import requests
    url = PV.format(lang=lang, title=quote(title.replace(" ", "_"), safe=""), a=START, b=b)
    for i in range(8):
        r = requests.get(url, headers=UA, timeout=120)
        if r.status_code == 404:
            return pd.Series(dtype=float)
        if r.status_code == 200:
            it = r.json().get("items", [])
            return pd.Series({pd.Timestamp(x["timestamp"][:8], tz="UTC"): float(x["views"]) for x in it}, dtype=float)
        time.sleep(min(10 * 2 ** i, 300))                    # 429: the API rate-limits shared addresses
    raise RuntimeError(f"pageviews {lang}:{title}: HTTP {r.status_code}")


def wiki_load(log=print, pause: float = 1.5) -> dict:
    b = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    rows = []
    for topic, per in TITLES.items():
        for lang, titles in per.items():
            tot = None
            for t in titles:
                s = _pv(lang, t, b)
                time.sleep(pause)
                if len(s):
                    tot = s if tot is None else tot.add(s, fill_value=0)
            if tot is None:
                log(f"wiki {topic} {lang}: no title found")
                continue
            rows.append(pd.DataFrame({"date": tot.index, "lang": lang, "topic": topic, "views": tot.to_numpy()}))
            log(f"wiki {topic} {lang}: {len(tot)} days, median {tot.median():.0f} views/day")
    d = pd.concat(rows)
    HS.put_parquet("lab/raw/wiki_health_daily.parquet", d.set_index("date"))
    return dict(series=len(rows), rows=len(d))


def are_load(log=print) -> dict:
    d = pd.read_csv(io.BytesIO(_get(ARE_URL).content), sep="\t")
    d = d[d["Bundesland"] == "Bundesweit"]
    d = d.assign(date=pd.to_datetime(d["Kalenderwoche"] + "-1", format="%G-W%V-%u", utc=True))
    w = d.pivot_table(index="date", columns="Altersgruppe", values="ARE_Konsultationsinzidenz", aggfunc="last").sort_index()
    HS.put_parquet("lab/raw/rki_are_age.parquet", w)
    log(f"rki ARE by age: {w.shape}, {w.index.min().date()} - {w.index.max().date()}")
    return dict(weeks=len(w), groups=list(w.columns))


# ------------------------------------------------------------------------------------------------------------------------------ features
def wiki_weekly(topic: str) -> pd.DataFrame | None:
    """Per country: mean of log(1 + views) over Monday-Friday of each week (stamped Monday), languages averaged per country."""
    d = HS.get_parquet("lab/raw/wiki_health_daily.parquet")
    if d is None:
        return None
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[(d["topic"] == topic) & (d["date"].dt.dayofweek < 5)]
    x = d.pivot_table(index="date", columns="lang", values="views", aggfunc="sum").sort_index()
    lx = np.log1p(x).resample("W-MON", label="left", closed="left").mean()
    n = x.notna().resample("W-MON", label="left", closed="left").sum()
    lx = lx.where(n >= 4)                                      # at least 4 of the 5 working days
    return pd.DataFrame({cc: lx.reindex(columns=[l for l in ls if l in lx.columns]).mean(axis=1, skipna=False) for cc, ls in LANGS.items()}).sort_index()


def wiki_features() -> dict[str, tuple[pd.DataFrame, float, str]]:
    out = {}
    for topic in TITLES:
        w = wiki_weekly(topic)
        if w is None:
            continue
        nt = f"Wikipedia pageviews (CC0) of the {topic} article(s) in the country's language(s), Monday-Friday mean of log views"
        out[f"wiki_{topic}_logchg"] = (w.diff().clip(-2, 2), WIKI_LAG_H, f"{nt}: week-on-week change (the week before the origin vs the one before)")
        out[f"wiki_{topic}_logchg2"] = (w.diff(2).clip(-3, 3), WIKI_LAG_H, f"{nt}: change over two weeks")
        base = w.shift(1).rolling(52, min_periods=26).median()
        out[f"wiki_{topic}_rel"] = (w - base, WIKI_LAG_H, f"{nt}: minus the median of the 52 weeks before")
    return out


def are_features() -> dict[str, tuple[pd.DataFrame, float, str]]:
    w = HS.get_parquet("lab/raw/rki_are_age.parquet")
    if w is None:
        return {}
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC") if pd.DatetimeIndex(w.index).tz is not None else pd.DatetimeIndex(w.index).tz_localize("UTC")
    lw = np.log(w.where(w > 0))
    nt = "RKI ARE-Konsultationsinzidenz (CC BY 4.0), Germany"
    return {"de_are_child_logchg": (lw[["0-4", "5-14"]].mean(axis=1).diff().to_frame("DE"), ARE_LAG_H, f"{nt}: ages 0-14, week-on-week log change"),
            "de_are_child_vs_all": ((lw[["0-4", "5-14"]].mean(axis=1) - lw["00+"]).to_frame("DE"), ARE_LAG_H,
                                    f"{nt}: log ratio of the 0-14 incidence to all ages (children lead adults)")}


if __name__ == "__main__":
    what = sys.argv[1:] or ["are", "wiki"]
    if "are" in what:
        print(are_load())
    if "wiki" in what:
        print(wiki_load())
    sys.exit(0)
