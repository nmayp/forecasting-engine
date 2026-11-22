"""Eurostat dissemination API (JSON-stat 2.0) client with a disk cache. Eurostat data: free reuse with source acknowledgement
(Commission Decision 2011/833/EU). Aggregates by area only; no personal data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
UA = "forecasting-engine/0.1"


def cache_dir() -> Path:
    p = Path(os.environ.get("FE_EUROSTAT_CACHE", "data/eurostat_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_jsonstat(j: dict) -> pd.DataFrame:
    """JSON-stat 2.0 dataset -> long DataFrame (one column per dimension + `value`). Missing cells are absent."""
    ids, size = j["id"], j["size"]
    cats = [list(j["dimension"][k]["category"]["index"]) if isinstance(j["dimension"][k]["category"]["index"], list)
            else sorted(j["dimension"][k]["category"]["index"], key=j["dimension"][k]["category"]["index"].get) for k in ids]
    vals = j.get("value", {})
    if isinstance(vals, list):
        vals = {i: v for i, v in enumerate(vals) if v is not None}
    idx = np.array([int(k) for k in vals], dtype=np.int64)
    out = {"value": np.array(list(vals.values()), dtype="float64")}
    cols = np.unravel_index(idx, size) if len(idx) else [np.array([], dtype=int)] * len(ids)
    for k, c, cat in zip(ids, cols, cats):
        out[k] = np.array(cat, dtype=object)[c]
    df = pd.DataFrame(out)
    return df[[*ids, "value"]]


def fetch(dataset: str, refresh: bool = False, **params) -> pd.DataFrame:
    """GET one dataset with filters (e.g. unit='NR', sinceTimePeriod='2019'); cached by request. Raises on HTTP errors."""
    import requests
    key = hashlib.sha1(json.dumps([dataset, params], sort_keys=True).encode()).hexdigest()[:16]
    f = cache_dir() / f"{dataset}_{key}.json"
    if f.exists() and not refresh:
        return parse_jsonstat(json.loads(f.read_text()))
    r = requests.get(BASE + dataset, params=dict(format="JSON", lang="en", **params), headers={"User-Agent": UA}, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"Eurostat {dataset} {params}: HTTP {r.status_code} {r.text[:200]}")
    f.write_text(r.text)
    return parse_jsonstat(r.json())
