"""Dated strike and closure catalogue for the what-if "strikes and closures -> transport and logistics throughput"
(docs/methods.md). Data: forecasting_engine/whatif/data/strike_catalogue_2015_2025.csv.
One row = one event (a cluster of strike days in one country and sector). `days` is a ';' list of dates or 'a..b' ranges; `intensity` is one value for all days
or one per listed day: the ex-ante estimate (from announcements, written before any outcome was looked at) of the share of the country's scheduled
flights (air) or port throughput (port) expected to be stopped that day. `split`: dev = onset before 2024, check = 2024-25 (looked at once)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "data" / "strike_catalogue_2015_2025.csv"


def _days(spec: str) -> list[pd.Timestamp]:
    out = []
    for part in str(spec).split(";"):
        if ".." in part:
            a, b = part.split("..")
            out += list(pd.date_range(a, b))
        else:
            out.append(pd.Timestamp(part))
    return out


def load() -> list[dict]:
    """events as dicts with `days` (list of Timestamps, one entry per listed item, ranges expanded) and `inten` (same length)."""
    ev = []
    for r in pd.read_csv(CSV, dtype=str).fillna("").to_dict("records"):
        items = r["days"].split(";")
        iv = [float(x) for x in r["intensity"].split(";")] if r["intensity"] else []
        days, inten = [], []
        for k, it in enumerate(items):
            dd = _days(it)
            v = iv[k] if len(iv) == len(items) else (iv[0] if iv else float("nan"))
            days += dd
            inten += [v] * len(dd)
        r["days"], r["inten"] = days, inten
        ev.append(r)
    return ev


def events(measure: str | None = None, split: str | None = None) -> list[dict]:
    return [e for e in load() if (measure is None or e["measure"] == measure) and (split is None or e["split"] == split)]


if __name__ == "__main__":
    for e in load():
        print(e["event_id"], e["country"], e["measure"], e["split"], len(e["days"]), e["days"][0].date(), e["days"][-1].date(), e["verified"])
