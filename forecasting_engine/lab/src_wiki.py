"""Public attention layer: Wikipedia page views per language edition and topic group.

    python -m forecasting_engine.lab.src_wiki topics            # English titles -> Wikidata items -> one article title per edition
    python -m forecasting_engine.lab.src_wiki load [budget_s]   # per edition: daily user page views of every topic article -> lab/raw/wiki/<lang>.parquet
    python -m forecasting_engine.lab.src_wiki merge             # all editions -> lab/raw/wiki_pv_daily.parquet (columns 'lang:qid')

Source: Wikimedia Analytics API, /metrics/pageviews/per-article, access all-access, agent user (automated and spider traffic excluded),
granularity daily, from 2015-07-01 (the start of that endpoint). Licence: Wikimedia page-view data is released under CC0 (public domain
dedication, https://dumps.wikimedia.org/legal.html and the API documentation: "data is freely available, CC0"). Terms of use: a descriptive
User-Agent with a contact, no more than about 200 requests/s (the API rate-limits shared IPs with HTTP 429: we go one request at a time and
back off). Publication: the API serves day D from about 04:00-08:00 UTC on D+1 (checked on 23 Oct 2026: the latest day available is the day
before); lag audit uses 2 days (a day D is treated as known at D + 2 days, one day more than seen).

Language edition is a proxy for a country: de DE, fr FR, es ES, it IT, nl NL, pl PL, pt PT, sv SE, cs CZ, ro RO, hu HU, el GR, da DK, fi FI,
bg BG, hr HR, sk SK, sl SI, lt LT, lv LV, et EE. The editions of large languages are read all over the world (Spanish: Latin America;
Portuguese: Brazil; German: Austria and Switzerland; French: Africa and Canada): the series are "attention in that language", labelled
by the home country only so that country-keyed drivers can be matched; the notes of the family say so.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

from forecasting_engine import histories as HS

UA = {"User-Agent": "forecasting-engine/0.1"}
LANG_CC = {"de": "DE", "fr": "FR", "es": "ES", "it": "IT", "nl": "NL", "pl": "PL", "pt": "PT", "sv": "SE", "cs": "CZ", "ro": "RO", "hu": "HU",
           "el": "GR", "da": "DK", "fi": "FI", "bg": "BG", "hr": "HR", "sk": "SK", "sl": "SI", "lt": "LT", "lv": "LV", "et": "EE"}
# topic group -> English article titles (resolved to Wikidata items, then to each edition's article)
TOPICS = {
    "health": ["Influenza", "COVID-19", "Vaccine", "Pneumonia", "Common cold", "Measles"],
    "energy": ["Natural gas", "Electricity", "Renewable energy", "Nuclear power", "Power outage"],
    "prices": ["Inflation", "Recession", "Unemployment", "Interest rate", "Stock market", "Cryptocurrency"],
    "migration": ["Refugee", "Human migration", "Illegal immigration", "Right of asylum"],
    "conflict": ["War", "Terrorism", "Protest", "Strike action", "Coup d'état", "NATO"],
    "travel": ["Tourism", "Airline", "Hotel", "Airport", "Beach", "Vacation"],
    "hazard": ["Heat wave", "Wildfire", "Flood", "Drought", "Earthquake", "Storm"],
    "cyber": ["Ransomware", "Cyberattack", "Phishing", "Data breach"],
}
DROP = {"Data breach", "Interest rate", "Stock market", "Illegal immigration", "Right of asylum", "Wildfire", "Power outage", "Vacation"}   # fewer than 15 editions
START, TOPIC_STORE = "20150701", "lab/raw/wiki/topics.json"
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


def topics() -> dict:
    """English titles -> Wikidata items -> sitelink titles for each edition."""
    allt = [t for v in TOPICS.values() for t in v]
    items = {}
    for i in range(0, len(allt), 20):
        chunk = allt[i:i + 20]
        for k in range(10):
            r = requests.get("https://www.wikidata.org/w/api.php", headers=UA, timeout=60, params={
                "action": "wbgetentities", "sites": "enwiki", "titles": "|".join(chunk), "props": "sitelinks|labels", "languages": "en", "format": "json"})
            if r.status_code != 429:
                break
            time.sleep(10 + 10 * k)
        r.raise_for_status()
        for q, e in r.json()["entities"].items():
            if q.startswith("Q"):
                en = e.get("sitelinks", {}).get("enwiki", {}).get("title")
                items[en] = dict(qid=q, label=e.get("labels", {}).get("en", {}).get("value"),
                                 titles={lg: e["sitelinks"][f"{lg}wiki"]["title"] for lg in LANG_CC if f"{lg}wiki" in e.get("sitelinks", {})})
        time.sleep(1)
    out = {}
    for g, ts in TOPICS.items():
        for t in ts:
            if t in items:
                out[items[t]["qid"]] = dict(group=g, en=t, label=items[t]["label"], titles=items[t]["titles"])
            else:
                print("topic not resolved:", t)
    HS.put_bytes(TOPIC_STORE, json.dumps(out, indent=1).encode(), "application/json")
    Path(__file__).with_name("wiki_topics.json").write_text(json.dumps(out, indent=1))
    return out


def _views(lang: str, title: str, end: str) -> pd.Series | None:
    url = f"{API}/{lang}.wikipedia/all-access/user/{urllib.parse.quote(title.replace(' ', '_'), safe='')}/daily/{START}/{end}"
    for i in range(12):
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(3 + 2 * i)
            continue
        if r.status_code == 200:
            it = r.json()["items"]
            s = pd.Series([x["views"] for x in it], index=pd.to_datetime([x["timestamp"][:8] for x in it], utc=True), dtype="float64")
            return s
        if r.status_code == 404:
            return pd.Series(dtype="float64")      # no data (title not viewed on any day of the range, or renamed)
        time.sleep(min(5 + 4 * i, 40) if r.status_code == 429 else 3 + 2 * i)
    return None


def load(budget_s: float = 500.0, log=print) -> dict:
    """Resumable: an edition whose parquet exists is skipped; stops starting editions when the time budget is used."""
    tp = json.loads(HS.get_bytes(TOPIC_STORE) or b"{}") or topics()
    end = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    t0, done, left = time.time(), [], []
    import os
    only = [x for x in os.environ.get("FE_WIKI_LANGS", "").split(",") if x]
    for lang in LANG_CC:
        if only and lang not in only:
            continue
        name = f"lab/raw/wiki/{lang}.parquet"
        if HS.get_bytes(name):
            continue
        if time.time() - t0 > budget_s:
            left.append(lang)
            continue
        part = f"lab/raw/wiki/_part_{lang}.parquet"        # resumable inside an edition (container restarts): articles done so far
        px = HS.get_parquet(part)
        cols, bad = ({c: px[c] for c in px.columns} if px is not None else {}), 0
        n_new = 0
        for q, t in tp.items():
            ti = t["titles"].get(lang)
            if not ti or q in cols or t["en"] in DROP:
                continue
            s = _views(lang, ti, end)
            if s is None:
                bad += 1
                continue
            cols[q] = s
            n_new += 1
            if n_new % 8 == 0:
                HS.put_parquet(part, pd.DataFrame(cols).sort_index())
            time.sleep(0.3)
        df = pd.DataFrame(cols).sort_index()
        HS.put_parquet(name, df)
        log(f"wiki {lang}: {df.shape[1]} articles, {df.index.min().date()} .. {df.index.max().date()}, {int(time.time() - t0)} s")
        done.append(lang)
    return dict(done=done, left=left)


def merge(log=print) -> dict:
    parts = {}
    for lang in LANG_CC:
        x = HS.get_parquet(f"lab/raw/wiki/{lang}.parquet")
        if x is not None:
            x.index = pd.DatetimeIndex(x.index)
            for q in x.columns:
                parts[f"{lang}:{q}"] = x[q]
    df = pd.DataFrame(parts).sort_index()
    HS.put_parquet("lab/raw/wiki_pv_daily.parquet", df)
    log(f"wiki merged: {df.shape}, {df.index.min().date()} .. {df.index.max().date()}")
    return dict(shape=df.shape)


def main(a=None):
    a = a or sys.argv[1:]
    if a[0] == "topics":
        print(len(topics()))
    elif a[0] == "load":
        print(json.dumps(load(float(a[1]) if len(a) > 1 else 500.0)))
    elif a[0] == "merge":
        merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
