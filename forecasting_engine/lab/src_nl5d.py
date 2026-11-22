"""Cross-layer links: newer layers as drivers of existing layers.

    python -m forecasting_engine.lab.panel extend firms_m insolv_m retail_m     # adds the drivers below to the frozen families

Every link has a mechanism, a real publication lag (the worst case is used) and goes through the normal dev gate (`forecasting_engine.lab.nldriver`): the champion
re-run with the driver added, one holdout look only if it passes on development and on most blocks.

| Link (layer -> layer) | Families | Input | Lag used |
|---|---|---|---|
| corporate disclosure -> firm demography | firms_m | EDGAR registration statements, prospectuses and Form D private placements, monthly sums (risk appetite for raising capital) | month + 8 days |
| corporate disclosure -> insolvency | insolv_m | EDGAR late-filing notices (NT 10-K, NT 10-Q: a distress signal) and 8-K current reports, monthly sums | month + 8 days |
| births -> retail | retail_m | births per day in the Netherlands, Sweden and Finland (household formation and baby-goods demand) | month + 60 days |
"""
from __future__ import annotations

import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK, MONTH = 24.0, 24.0 * 7, 24.0 * 30.44


def _utc(x):
    x = x.copy()
    ix = pd.DatetimeIndex(x.index)
    x.index = ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")
    return x


def _msum(x):
    """Weekly counts (Monday stamps) -> monthly sums by the week's Monday; months with fewer than 4 weeks are left out."""
    s = x.resample("MS").sum(min_count=4)
    return s.where(x.resample("MS").count() >= 4)


def drivers(fid: str, P, PN) -> list:
    Driver, D = PN.Driver, []
    if fid in ("firms_m", "insolv_m"):
        e = _utc(HS.get_parquet("lab/raw/edgar_weekly.parquet"))
        e = e[e.index >= "2010-01-04"]
        keys = ("registration", "prospectus", "private") if fid == "firms_m" else ("late", "8k")
        for k in keys:
            D.append(Driver(f"nl5_edgar_{k}", "corporate disclosure", _msum(e[[f"{k}:US"]].rename(columns={f"{k}:US": "*"})), MONTH + 8 * DAY, note=f"SEC EDGAR {k} filings per month (sec_w)"))
    if fid == "retail_m":
        b = _utc(HS.get_parquet("lab/raw/births_monthly.parquet"))
        x = pd.DataFrame({"NL": b["nl_total:NL"], "SE": b["se_total:SE"], "FI": b["fi_total:FI"]})
        D.append(Driver("nl5_births", "births", x, MONTH + 60 * DAY, note="births per day, national (births_m): household formation"))
    return D
