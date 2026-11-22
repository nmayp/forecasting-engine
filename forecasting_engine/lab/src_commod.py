"""Leading inputs for commod_d: US oil and gas inventories, futures positioning and refined-product spreads, each
stamped with the moment it is really published, so a daily origin (12:00 UTC) sees only what was out by then.

    python -m forecasting_engine.lab.src_commod            # downloads everything to the store (a few MB)

| Store name | Content | Source, real publication lag | Licence |
|---|---|---|---|
| lab/raw/eia_weekly.parquet | long: date (week-ending Friday, UTC), id, value. Petroleum: commercial crude stocks excl. SPR, Cushing stocks, gasoline and distillate stocks (kbbl), refinery utilisation (%), crude production, products supplied, crude imports (kb/d). Natural gas: working gas in storage, Lower 48 (Bcf) | EIA Weekly Petroleum Status Report (dnav/pet/hist_xls), released Wednesday 10:30 ET for the week to the previous Friday, Thursday after a Monday holiday (week to 9 Oct 2026 came out Thu 15 Oct): week-ending Friday + 7 days used (an origin at 12:00 UTC sees it from the Friday after the release). EIA Weekly Natural Gas Storage Report, Thursday 10:30 ET, a day later in holiday weeks: + 8.5 days used | US Government work, public domain (EIA copyright and reuse page: "public domain and may be used freely") |
| lab/raw/cftc_cot.parquet | long: date (report Tuesday, UTC), market (wti / brent / ng), managed money long, short, open interest | CFTC Commitments of Traders, disaggregated futures only (publicreporting.cftc.gov dataset 72hh-3qpy, 2006-06-), positions as of Tuesday, released Friday 15:30 ET, the Monday after in holiday weeks: Tuesday + 6 days 21 h used | US Government work, public domain |
| lab/raw/commod_fred.parquet | long: date, id, value: NY Harbor conventional gasoline and No. 2 heating oil spot (USD/gal, EIA via FRED DGASNYH, DHOILNYH) | FRED, end of day + 12 h (as the panel's other FRED dailies) | EIA data, public domain |

Checked and not used (stated so nobody looks again): GIE AGSI+ EU gas storage needs a registered API key (the API answers "Invalid or missing
API key"; its terms allow reuse with attribution once registered) - left to the operator to register; IMF PortWatch chokepoint transits are
already in the store (back-tests only, IMF permission needed for commercial reuse) and stay off commod_d; ECB and FRED market series the panel
lacked are the two product spot prices above (crack spreads).
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab import panel as PN
from forecasting_engine.lab.sources import _get

EIA_PET = {"WCESTUS1": "crude_stocks", "W_EPC0_SAX_YCUOK_MBBL": "cushing_stocks", "WGTSTUS1": "gasoline_stocks", "WDISTUS1": "distillate_stocks",
           "WPULEUS3": "refinery_util", "WCRFPUS2": "crude_production", "WRPUPUS2": "products_supplied", "WCRIMUS2": "crude_imports"}
EIA_NG = {"NW2_EPG0_SWO_R48_BCF": "ng_storage"}
PET_LAG_H = 12 * 24.0              # lag audit 20 Oct 2026: Wednesday release (Thursday after a holiday), but the week to 19 Dec 2025 came out
#                                    Mon 29 Dec after 17:00 ET (10 days 22 h); was 7 days
NG_LAG_H = 8.5 * 24.0              # Friday week end -> Saturday 12:00 after the (Thursday, Friday in holiday weeks) release
COT_LAG_H = 6 * 24.0 + 21          # Tuesday positions -> Monday 21:00 UTC (holiday-week release 15:30 ET Monday)
COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
COT_MARKETS = {"067651": "wti", "06765T": "brent", "023651": "ng"}
FRED_EXTRA = {"DGASNYH": "NY Harbor conventional gasoline spot, USD/gal", "DHOILNYH": "NY Harbor No. 2 heating oil spot, USD/gal"}
ATTR = {"eia": "Source: U.S. Energy Information Administration (Weekly Petroleum Status Report, Weekly Natural Gas Storage Report), public domain.",
        "cftc": "Source: U.S. Commodity Futures Trading Commission, Commitments of Traders (disaggregated), public domain."}


def _eia_xls(sid: str, area: str) -> pd.Series:
    r = _get(f"https://www.eia.gov/dnav/{area}/hist_xls/{sid}w.xls", ua=False)
    x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
    x.columns = ["date", "value"] + list(x.columns[2:])
    s = pd.Series(pd.to_numeric(x["value"], errors="coerce").to_numpy(), index=pd.to_datetime(x["date"], errors="coerce", utc=True)).dropna()
    return s[s.index.notna()].sort_index()


def eia_load(log=print) -> dict:
    rows = []
    for ids, area in ((EIA_PET, "pet"), (EIA_NG, "ng")):
        for sid, nm in ids.items():
            s = _eia_xls(sid, area)
            rows.append(pd.DataFrame({"date": s.index, "id": nm, "value": s.to_numpy()}))
            log(f"eia {nm}: {len(s)} weeks {s.index.min().date()} - {s.index.max().date()}")
    d = pd.concat(rows)
    HS.put_parquet("lab/raw/eia_weekly.parquet", d.set_index("date"))
    return dict(series=len(rows), rows=len(d))


def cot_load(log=print) -> dict:
    parts = []
    for code, nm in COT_MARKETS.items():
        r = _get(COT_URL, ua=False, params={"cftc_contract_market_code": code, "$limit": 50000, "$order": "report_date_as_yyyy_mm_dd",
                                            "$select": "report_date_as_yyyy_mm_dd,m_money_positions_long_all,m_money_positions_short_all,open_interest_all"})
        d = pd.DataFrame(r.json())
        d = pd.DataFrame({"date": pd.to_datetime(d["report_date_as_yyyy_mm_dd"], utc=True), "market": nm,
                          "mm_long": pd.to_numeric(d["m_money_positions_long_all"], errors="coerce"),
                          "mm_short": pd.to_numeric(d["m_money_positions_short_all"], errors="coerce"),
                          "oi": pd.to_numeric(d["open_interest_all"], errors="coerce")}).dropna()
        parts.append(d)
        log(f"cftc {nm}: {len(d)} weeks {d['date'].min().date()} - {d['date'].max().date()}")
    d = pd.concat(parts)
    HS.put_parquet("lab/raw/cftc_cot.parquet", d.set_index("date"))
    return dict(rows=len(d))


def fred_extra_load(log=print) -> dict:
    rows = []
    for sid in FRED_EXTRA:
        d = pd.read_csv(io.BytesIO(_get("https://fred.stlouisfed.org/graph/fredgraph.csv", ua=False, params={"id": sid}).content))
        d.columns = ["date", "value"]
        d = d.assign(value=pd.to_numeric(d["value"], errors="coerce"), id=sid).dropna()
        rows.append(d)
        log(f"fred {sid}: {len(d)} rows")
    out = pd.concat(rows)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    HS.put_parquet("lab/raw/commod_fred.parquet", out.set_index("date"))
    return dict(rows=len(out))


# ------------------------------------------------------------------------------------------------------------------------------ features
def _wide(name: str, key: str) -> pd.DataFrame | None:
    d = HS.get_parquet(name)
    if d is None:
        return None
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    return d.pivot_table(index="date", columns=key, values="value", aggfunc="last").sort_index()


def _vs5y(s: pd.Series, pct: bool = True) -> pd.Series:
    """Deviation from the mean of the same week in the 5 previous years (52-week steps), as known at the week itself."""
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    past = pd.concat([s.shift(52 * k) for k in range(1, 6)], axis=1).mean(axis=1, skipna=False)
    return ((s / past - 1) * 100 if pct else s - past).dropna()


def eia_features() -> dict[str, tuple[pd.Series, float, str]]:
    """name -> (weekly series stamped at the week-ending Friday, avail_h, note)."""
    w = _wide("lab/raw/eia_weekly.parquet", "id")
    if w is None:
        return {}
    out = {}
    nt = "EIA Weekly Petroleum Status Report (public domain)"
    for k in ("crude_stocks", "cushing_stocks", "gasoline_stocks", "distillate_stocks"):
        out[f"eia_{k}_vs5y"] = (_vs5y(w[k]), PET_LAG_H, f"{nt}: {k} vs the same week's 5-year mean, %")
    c = w["crude_stocks"].dropna()
    out["eia_crude_wchg_vs5y"] = (_vs5y(c.diff(), pct=False) / 1000, PET_LAG_H, f"{nt}: weekly change of commercial crude stocks minus the 5-year mean change of that week, million bbl")
    ps = w["products_supplied"].dropna().rolling(4).mean()
    out["eia_products_supplied_yoy"] = ((ps / ps.shift(52) - 1) * 100).dropna(), PET_LAG_H, f"{nt}: total products supplied (implied demand), 4-week mean y/y %"
    out["eia_refinery_util"] = (w["refinery_util"].dropna(), PET_LAG_H, f"{nt}: refinery utilisation, %")
    pr = w["crude_production"].dropna()
    out["eia_crude_production_yoy"] = (((pr / pr.shift(52) - 1) * 100).dropna(), PET_LAG_H, f"{nt}: US crude production y/y %")
    g = w["ng_storage"].dropna()
    ntg = "EIA Weekly Natural Gas Storage Report (public domain)"
    out["eia_ng_storage_vs5y"] = (_vs5y(g), NG_LAG_H, f"{ntg}: Lower 48 working gas vs the same week's 5-year mean, %")
    out["eia_ng_inj_vs5y"] = (_vs5y(g.diff(), pct=False), NG_LAG_H, f"{ntg}: weekly net injection minus the 5-year mean of that week, Bcf")
    return {k: (v[0].replace([np.inf, -np.inf], np.nan).dropna(), v[1], v[2]) for k, v in out.items()}


def cot_features() -> dict[str, tuple[pd.Series, float, str]]:
    d = HS.get_parquet("lab/raw/cftc_cot.parquet")
    if d is None:
        return {}
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    out = {}
    for m in COT_MARKETS.values():
        x = d[d["market"] == m].drop_duplicates("date", keep="last").set_index("date").sort_index()
        net = PN.cot_mask(((x["mm_long"] - x["mm_short"]) / x["oi"] * 100).where(x["oi"] > 0).dropna())   # late (shutdown) reports dropped
        out[f"cot_mm_net_{m}"] = (net, COT_LAG_H, f"CFTC COT disaggregated (public domain): {m} managed-money net long, % of open interest (Tuesday positions)")
    return out


def crack_features(f: pd.DataFrame) -> dict[str, tuple[pd.Series, float, str]]:
    """Refined-product cracks in USD/bbl: product spot x 42 minus crude (gasoline vs WTI, heating oil vs Brent). f: the panel's FRED dailies."""
    p = _wide("lab/raw/commod_fred.parquet", "id")
    if p is None:
        return {}
    gas = (p["DGASNYH"] * 42 - f["DCOILWTICO"]).dropna()
    ho = (p["DHOILNYH"] * 42 - f["DCOILBRENTEU"]).dropna()
    nt = "EIA spot prices via FRED (public domain), USD/bbl"
    lag = PN.EIA_SPOT_LAG_H                 # lag audit 20 Oct 2026: EIA spot prices are released weekly (was 36 h)
    return {"crack_gasoline_wti": (gas, lag, f"{nt}: NY Harbor gasoline x 42 minus WTI"),
            "crack_heating_oil_brent": (ho, lag, f"{nt}: NY Harbor heating oil x 42 minus Brent"),
            "brent_wti_spread": ((f["DCOILBRENTEU"] - f["DCOILWTICO"]).dropna(), lag, "FRED (EIA): Brent minus WTI, USD/bbl")}


if __name__ == "__main__":
    print(eia_load(), cot_load(), fred_extra_load())
    sys.exit(0)
