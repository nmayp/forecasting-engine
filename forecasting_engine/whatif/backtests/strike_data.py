"""Throughput series for the strikes what-if. Air: EUROCONTROL daily IFR flights per country (store `lab/raw/air_traffic.parquet`, back-tests only).
Ports: IMF PortWatch daily port calls and AIS import+export volume (t), summed per country; the shared store has Spain only, other countries are downloaded
with `fetch_portwatch` to a local scratch dir (STRIKE_CACHE) and never written to the store. Run once: python -m forecasting_engine.whatif.backtests.strike_data air|portwatch NLD BEL ..."""
from __future__ import annotations

import os
import sys

import pandas as pd

CACHE = os.environ.get("STRIKE_CACHE", "/tmp/strike_cache")
ISO = {"NLD": "NL", "BEL": "BE", "DEU": "DE", "FRA": "FR", "FIN": "FI", "SWE": "SE", "PRT": "PT", "ITA": "IT", "GRC": "GR", "DNK": "DK", "POL": "PL", "ESP": "ES"}


def fetch_air() -> None:
    from forecasting_engine import histories as HS
    a = HS.get_parquet("lab/raw/air_traffic.parquet")
    a.index = pd.DatetimeIndex(a.index).tz_convert("UTC").tz_localize(None)
    os.makedirs(CACHE, exist_ok=True)
    a.to_parquet(f"{CACHE}/air.parquet")


def fetch_portwatch(iso3: list[str], years: list[int] | None = None) -> None:
    """quarterly date windows: whole-country-year queries time out on the IMF server (minutes, 504s), quarters take seconds"""
    from forecasting_engine.lab import src_leading as L
    os.makedirs(CACHE, exist_ok=True)
    years = years or list(range(2019, pd.Timestamp.now("UTC").year + 1))
    for c in iso3:
        parts = []
        for y in years:
            for a, b in (("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")):
                parts.append(L._arcgis_all(L.PORTS_URL, f"ISO3='{c}' AND date >= DATE '{y}-{a}' AND date <= DATE '{y}-{b}'",
                                           ("date", "portid", "portname", "ISO3") + L.PORT_FIELDS, lambda *x: None, order=False))
        d = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "portid"])
        d["date"] = L._ts(d["date"]).tz_localize(None)
        d.to_parquet(f"{CACHE}/pw_{c}.parquet")


def air() -> pd.DataFrame:
    return pd.read_parquet(f"{CACHE}/air.parquet")


def ports() -> dict[str, pd.DataFrame]:
    """country ISO2 -> daily frame with columns calls (all port calls), vol (import+export, t), n (ports reporting)"""
    import glob
    frames = {}
    for f in sorted(glob.glob(f"{CACHE}/pw_*.parquet")):
        d = pd.read_parquet(f)
        ix = pd.DatetimeIndex(d["date"])
        d["date"] = ix.tz_localize(None) if ix.tz is not None else ix
        d["vol"] = d["import"].fillna(0) + d["export"].fillna(0)
        for iso3, g in d.groupby("ISO3"):
            frames.setdefault(iso3, []).append(g)
    out = {}
    for iso3, gs in frames.items():
        g = pd.concat(gs).drop_duplicates(["date", "portid"])
        out[ISO[iso3]] = g.groupby("date").agg(calls=("portcalls", "sum"), vol=("vol", "sum"), n=("portid", "nunique")).sort_index()
    return out


if __name__ == "__main__":
    if sys.argv[1] == "air":
        fetch_air()
    else:
        fetch_portwatch(sys.argv[2:])   # e.g. NLD BEL DEU FRA FIN
