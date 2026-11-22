"""Lab v2 panels: per family, the target series of every country (or port, currency, commodity) and every driver, each with the time after its
timestamp when it is known (publication delay included). Built from the history store and frozen once (lab/frozen/<family>/ in the store, hashes
in the registry).

A driver is a frame indexed by its own timestamps with one column per series it belongs to (a country code) or a single column "*" (global).
`avail_h` is the delay from a timestamp to the moment its value is known: a daily value stamped at 00:00 and published the next morning has
avail_h = 24 + a few hours. `ahead=True` marks drivers published before the period they describe (the TSO's day-ahead load forecast): their value
AT THE TARGET time is a feature when it is known at the origin.

Delays assumed (stated so a reader can check them; Eurostat values are the latest vintage, not the first release, which flatters the monthly
families a little):
    ENTSO-E actual load / generation / flows 2 h; day-ahead price: the whole day D is known at 12:00 UTC on D-1; day-ahead load forecast: day D at
    09:00 UTC on D-1; generation outages: from their start; temperature (ERA5, standing in for live observations) end of day + 6 h;
    FRED daily markets end of day + 12 h (EXCEPT the weekly releases, lag audit of 20 Oct 2026, audit_lag: EIA spot prices 14 days, H.10 dollar
    index 9 days, IMF monthly commodities month + 28 days, policy uncertainty month + 100 days); ECB reference rates 15:00 UTC the same day; GDELT end of day + 1 h (days = the UTC day events were ADDED, since 21 Oct 2026); KEV end of day;
    Oil Bulletin Monday prices + 4 days; Eurostat industrial production month end + 45 days, retail + 35, unemployment + 36 (+ 30 before the lag audit), HICP + 17;
    IMF monthly commodities month end + 7 days; Puertos del Estado month end + 45 days.
    Nowcast inputs (lab.src_nowcast): Eurostat HICP flash at its real release (10:00 UTC on the day read from the release page);
    RKI AMELAG wastewater Wednesday stamp + 8 days; EUROCONTROL dashboard D + 2 days from 2020-04 (whole months + 46 days before);
    school calendars 180 days ahead.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

ENTSOE_CC = ("ES", "PT", "FR", "DE", "IT", "NL", "BE", "PL")
EU27 = ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
        "SE", "SI", "SK")
GEO = {"EL": "GR", "UK": "GB"}                      # Eurostat -> ISO 3166
DAY, MONTH, WEEK = 24.0, 24.0 * 30.44, 24.0 * 7
FX = ("USD", "GBP", "JPY", "CHF", "CNY", "PLN", "CZK", "HUF", "SEK", "NOK", "DKK", "RON", "TRY", "AUD", "CAD", "BRL", "MXN", "INR", "KRW", "ZAR")
OIL_REGION = ("IR", "IQ", "SA", "YE", "IL", "RU", "UA", "EG")

# ------------------------------------------------------------------------------------------------------------------------------ lag audit
# Publication-lag audit (20 Oct 2026, docs/methods.md): worst-case delays from a value's stamp to the moment
# it is public, from each source's release calendar and the FRED/ALFRED first-vintage dates of every observation from Oct 2025 to Oct 2026.
# audit_lag() applies them to every driver built here (build_family, extra_drivers) and, through `relag`, to the frozen manifests. It only
# ever lengthens a delay (max with the old one), so it is idempotent; driver names and order never change.
EIA_SPOT_LAG_H = 14 * DAY      # EIA daily spot prices (Brent, WTI, Henry Hub, NY Harbor / Gulf products, propane; FRED DCOILBRENTEU, DCOILWTICO,
#                                DHHNGSP, DGASNYH, DHOILNYH): released WEEKLY (Wednesday with data to the Tuesday, Thursday in holiday weeks).
#                                ALFRED Oct 2025-Oct 2026: 1-9 days after the date, 13 days over Christmas 2025, + 21 h (FRED update 15:12 CT)
H10_LAG_H = 9 * DAY            # Fed H.10 broad dollar index (DTWEXBGS): weekly, Monday 16:15 ET with the days to the Friday before; ALFRED: up to
#                                8 days after the date (Tuesday release after a Monday holiday) + 21 h
IMF_PCPS_DAYS = 28             # IMF primary commodity prices via FRED (PNGASEUUSDM, PWHEAMTUSDM, PCOPPUSDM): ALFRED 2024-26, 4-26 days after the month
EPU_DAYS = 100                 # FRED GEPUCURRENT (global economic policy uncertainty): ALFRED 2024-26, 8-96 days after the month
WTISPLC_DAYS = 8               # FRED WTISPLC (monthly WTI): ALFRED 2024-26, 0-7 days after the month
ECB_YC_LAG_H = 5 * DAY + 12    # ECB AAA yield curve: 12:00 CET the next TARGET day; Thursday before Easter -> Tuesday 11:00 UTC (5 days 11 h)
ESTR_LAG_H = 4 * DAY + 8       # euro short-term rate: 08:00 CET the next TARGET day; Easter: Thursday -> Tuesday 07:00 UTC
EUROSTAT_DAYS = {"unemp": 36, "retail": 40, "ip": 49, "hicp": 27, "ppi": 38}   # first release, days after the month (2025-26 euro indicators:
#                                unemployment 30-32, retail 35-38, industrial production 44-47, HICP final 15-17 but 25 for January, PPI 35;
#                                + about a day because MONTH (30.44 days) ends before the end of a 31-day month)
EIA_SPOT_DRIVERS = ("brent_eur", "henry_hub_eur", "brent_wti_spread", "crack_gasoline_wti", "crack_heating_oil_brent", "propane_vs_wti")
EIA_SPOT_PREFIX = ("crack_ulsd", "crack_jet", "crack_gasoline_gulf", "ulsd_nyh_eur_l", "ulsd_gulf_eur_l", "gasoline_gulf_eur_l", "jet_gulf_eur_l")
EIA_PET_W_LAG_H = 12 * DAY     # EIA Weekly Petroleum Status Report: Wednesday 10:30 ET for the week to the Friday before, Thursday after a holiday,
#                                but the week to 19 Dec 2025 came out Monday 29 Dec after 17:00 ET (10 days 22 h after the week's Friday)
# CFTC Commitments of Traders: positions of these report Tuesdays were published weeks late (US government shutdowns of 2013, 2018-19 and
# 2025; the 2025 catch-up schedule is cftc.gov press release 9147-25): they are dropped (never known on time), not given a longer delay.
COT_LATE = (("2013-09-30", "2013-10-29"), ("2018-12-24", "2019-03-01"), ("2025-09-29", "2025-12-16"))


def cot_mask(x):
    """Drop the COT report dates published late (COT_LATE)."""
    ix = pd.DatetimeIndex(x.index)
    ix = ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")
    bad = np.zeros(len(ix), bool)
    for a, b in COT_LATE:
        lo, hi = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(hours=23)
        bad |= np.asarray((ix >= lo) & (ix <= hi), bool)
    return x[~bad]


def audit_lag(fid: str, freq: str, name: str, avail_h: float) -> float:
    """The audited (worst-case) delay of a driver of family `fid` (freq of the family: monthly families read month means of the dailies)."""
    m = MONTH if freq == "M" else 0.0
    new = None
    if name in ("brent_eur", "henry_hub_eur"):
        new = m + EIA_SPOT_LAG_H
    elif name in EIA_SPOT_DRIVERS or name.startswith(EIA_SPOT_PREFIX):
        new = EIA_SPOT_LAG_H
    elif name == "dollar_index":
        new = m + H10_LAG_H
    elif name in ("gas_eu_usd", "wheat_usd", "copper_usd"):
        new = MONTH + IMF_PCPS_DAYS * DAY
    elif name == "policy_uncertainty":
        new = MONTH + EPU_DAYS * DAY
    elif name in ("wti_usd", "wti_yoy"):
        new = MONTH + WTISPLC_DAYS * DAY
    elif name.startswith("eia_") and not name.startswith("eia_ng_"):
        new = EIA_PET_W_LAG_H
    elif name in ("unemp", "unemp_ea", "unemployment"):      # Eurostat / OECD harmonised unemployment (national releases: Eurostat for the EU)
        new = MONTH + EUROSTAT_DAYS["unemp"] * DAY
    elif name in ("retail", "ip", "ppi"):
        new = MONTH + EUROSTAT_DAYS[name] * DAY
    elif name == "hicp" and fid != "hicp_m":
        new = MONTH + EUROSTAT_DAYS["hicp"] * DAY
    elif name.startswith("hicp_") and name.endswith(("_ea", "_ea_mom")) and fid == "ecb_w":   # euro-area HICP (stamped at the release month)
        new = EUROSTAT_DAYS["hicp"] * DAY
    elif fid == "ecb_w" and (name.startswith("yc_") or name.startswith("slope_")):
        new = ECB_YC_LAG_H
    elif name in ("ea_curve_10_2", "us_ea_2y"):
        new = ECB_YC_LAG_H
    elif name in ("yc_1y_month", "yc_2y_month"):
        new = MONTH + ECB_YC_LAG_H
    elif name == "estr_minus_dfr":
        new = ESTR_LAG_H
    elif name == "euribor3m":
        new = MONTH + 14 * DAY
    return float(avail_h) if new is None else max(float(avail_h), float(new))


def _audit(fid: str, freq: str, D: list) -> list:
    out = []
    for d in D:
        a = audit_lag(fid, freq, d.name, d.avail_h)
        x = cot_mask(d.data) if d.name.startswith("cot_") else d.data
        out.append(d if (a == d.avail_h and x is d.data) else Driver(d.name, d.group, x, a, d.ahead, d.note, d.dayahead_hour))
    return out


@dataclass
class Driver:
    name: str
    group: str                       # layer the driver belongs to (energy, weather, markets, society, economy, cyber, trade, calendar)
    data: pd.DataFrame               # index: timestamps (UTC); columns: series ids or "*"
    avail_h: float
    ahead: bool = False
    note: str = ""
    dayahead_hour: int | None = None  # published for a whole day at this UTC hour of the day before (overrides avail_h)


@dataclass
class Family:
    id: str
    label: str
    layer: str
    freq: str                        # "h", "D", "W", "M"
    unit: str
    Y: pd.DataFrame                  # index: timestamps; columns: series ids
    avail_h: float                   # delay of the target itself
    skip: int                        # periods after the origin's period that are not forecast (already known or past the decision)
    H: int                           # forecast steps after the skip
    seasons: tuple                   # seasonal-naive candidates (periods)
    origin_hour: int = 0             # hourly families: origins at this UTC hour every day
    drivers: list = field(default_factory=list)
    notes: str = ""
    dayahead_hour: int | None = None
    span_min: int | None = None          # windows start where at least this many series have data (default: half of them)
    lagfix: str = ""                     # set by relag when the lag audit changed a delay of this frozen family (forecast caches key on it)


# ------------------------------------------------------------------------------------------------------------------------------ readers
def _geo(x: str) -> str:
    return GEO.get(x, x)


def _long(name: str, key: str = "geo", keep=None) -> pd.DataFrame:
    d = HS.get_parquet(name).reset_index()
    d[key] = d[key].astype(str).map(_geo)
    if keep is not None:
        d = d[d[key].isin(keep)]
    w = d.pivot_table(index="date", columns=key, values="value", aggfunc="last").sort_index()
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC") if pd.DatetimeIndex(w.index).tz is not None else pd.DatetimeIndex(w.index).tz_localize("UTC")
    return w


def _fred(ids) -> pd.DataFrame:
    d = HS.get_parquet("lab/raw/fred.parquet").reset_index()
    d = d[d["id"].isin(ids)]
    w = d.pivot_table(index="date", columns="id", values="value", aggfunc="last").sort_index()
    return w


def _ecb_fx() -> pd.DataFrame:
    d = HS.get_parquet("lab/raw/ecb_fx.parquet").reset_index()
    return d.pivot_table(index="date", columns="id", values="value", aggfunc="last").sort_index()


def _entsoe_wide(dataset: str, col: str, ccs=ENTSOE_CC, how="mean") -> pd.DataFrame:
    cols = {}
    for cc in ccs:
        s = HS.entsoe_series(dataset, cc, col)
        if len(s):
            s = s.resample("h").mean() if how == "mean" else s.resample("h").sum()
            cols[cc] = s
    return pd.DataFrame(cols).sort_index()


def _gen_shares(ccs=ENTSOE_CC) -> dict[str, pd.DataFrame]:
    """Hourly generation per type -> shares of total generation: wind, solar, nuclear, gas, hydro, coal (each a frame country x hour)."""
    groups = {"wind": ("gen_wind_onshore", "gen_wind_offshore"), "solar": ("gen_solar",), "nuclear": ("gen_nuclear",),
              "gas": ("gen_gas", "gen_coal_gas"), "hydro": ("gen_hydro_reservoir", "gen_hydro_ror", "gen_hydro_pumped"),
              "coal": ("gen_hard_coal", "gen_lignite")}
    out = {g: {} for g in groups}
    out["total"] = {}
    for cc in ccs:
        df = HS.entsoe_series("gen", cc)
        if df is None or not len(df):
            continue
        df = df.resample("h").mean()
        gen = df[[c for c in df.columns if c.startswith("gen_")]].clip(lower=0)
        tot = gen.sum(axis=1).where(lambda x: x > 0)
        out["total"][cc] = tot
        for g, cols in groups.items():
            have = [c for c in cols if c in gen.columns]
            out[g][cc] = gen[have].sum(axis=1) / tot if have else tot * np.nan
    return {g: pd.DataFrame(v).sort_index() for g, v in out.items()}


def _net_imports(ccs=ENTSOE_CC) -> pd.DataFrame:
    """Hourly net physical imports (MW) per country from the stored border flows (only the 9 stored borders: a partial balance)."""
    tot = {cc: None for cc in ccs}
    for a, b in HS.BORDERS:
        for f, t in ((a, b), (b, a)):
            s = HS.entsoe_series("flows", f"{f}_{t}", "flow_mw")
            if not len(s):
                continue
            s = s.resample("h").mean()
            for cc, sign in ((t, 1.0), (f, -1.0)):
                if cc in tot:
                    tot[cc] = s * sign if tot[cc] is None else tot[cc].add(s * sign, fill_value=0)
    return pd.DataFrame({k: v for k, v in tot.items() if v is not None}).sort_index()


def _outages(ccs=ENTSOE_CC) -> pd.DataFrame:
    cols = {}
    for cc in ccs:
        names = [n for n in HS.list_names(f"entsoe/out_gen/{cc}") if n.endswith(".parquet")]
        parts = [HS.get_parquet(n) for n in names]
        ev = pd.concat([p for p in parts if p is not None and len(p) and "start" in p.columns])
        if "mrid" in ev.columns:
            ev = ev.sort_values("rev").drop_duplicates("mrid", keep="last")
        ev["start"] = pd.to_datetime(ev["start"], utc=True)
        ev["end"] = pd.to_datetime(ev["end"], utc=True)
        cols[cc] = HS.outage_daily(ev, "2015-01-01", None)
    return pd.DataFrame(cols).sort_index()


def _weather(col: str = "t") -> pd.DataFrame:
    d = HS.get_parquet("weather/popw_daily.parquet").reset_index()
    d = d[d["geo"].str.len() == 2]
    d["geo"] = d["geo"].map(_geo)
    w = d.pivot_table(index="date", columns="geo", values=col, aggfunc="last").sort_index()
    w.index = pd.DatetimeIndex(w.index).tz_localize("UTC") if pd.DatetimeIndex(w.index).tz is None else pd.DatetimeIndex(w.index)
    return w


def _gdelt(kind: str, value: str = "n", freq: str = "D", ccs=None) -> pd.DataFrame:
    from forecasting_engine.lab import sources as SR
    d = SR.gdelt_frame()
    if kind != "*":
        d = d[d["kind"] == kind]
    if value == "tone":
        d = d.assign(w=d["tone"] * d["n"])
        g = d.groupby(["date", "cc"])[["w", "n"]].sum()
        w = (g["w"] / g["n"]).unstack()
    else:
        w = d.groupby(["date", "cc"])[value].sum().unstack()
    if ccs:
        w = w.reindex(columns=[c for c in ccs if c in w.columns])
    w = w.sort_index()
    if kind != "*" and value != "tone":
        w = w.fillna(0.0)
    if freq == "W":
        w = w.resample("W-MON", label="left", closed="left").sum()
    return w


def _oxcgrt(col: str) -> pd.DataFrame:
    d = HS.get_parquet("lab/raw/oxcgrt.parquet").reset_index()
    w = d.pivot_table(index="date", columns="geo", values=col, aggfunc="last").sort_index()
    full = pd.date_range(pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"), freq="D")
    return w.reindex(full).fillna(0.0)                       # no measures before 2020 and after the tracker ended (end of 2022)


def _kev_daily() -> pd.DataFrame:
    d = HS.get_parquet("lab/raw/kev.parquet").reset_index()
    s = d.groupby(d["date"].dt.floor("D")).size()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    s = s.reindex(pd.date_range(pd.Timestamp("2021-11-03", tz="UTC"), s.index.max(), freq="D"), fill_value=0).astype(float)
    return s.to_frame("*")


def _oil(product: str, ccs=EU27) -> pd.DataFrame:
    return pd.DataFrame({cc: HS.oil_series(cc, product) for cc in ccs}).sort_index()


def _g(df: pd.DataFrame | pd.Series, name: str = "*") -> pd.DataFrame:
    """A global series as a one-column frame."""
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    return s.dropna().to_frame(name)


def _to_eur(usd: pd.Series, fx: pd.DataFrame) -> pd.Series:
    r = fx["USD"].reindex(usd.index).ffill()
    return usd / r


# ------------------------------------------------------------------------------------------------------------------------------ shared drivers
class Pool:
    """Every driver frame, read once."""

    def __init__(self):
        self._m = {}

    def get(self, k: str):
        if k not in self._m:
            self._m[k] = BUILD[k]()
        return self._m[k]


def _fred_one(sid):
    return lambda: _g(_fred([sid])[sid])


def _gdelt_w(kind):
    return lambda: _gdelt(kind, "n", "D")


BUILD = {
    "load": lambda: _entsoe_wide("load", "load_mw"),
    "load_da": lambda: _entsoe_wide("load_da", "load_da_mw"),
    "price": lambda: _entsoe_wide("price", "price_eur_mwh"),
    "gen": _gen_shares,
    "net_imp": _net_imports,
    "outage": _outages,
    "temp": lambda: _weather("t"),
    "hdd": lambda: _weather("hdd"),
    "cdd": lambda: _weather("cdd"),
    "fx": _ecb_fx,
    "fred": lambda: _fred(["DCOILBRENTEU", "DCOILWTICO", "DHHNGSP", "DGS10", "DGS2", "T10YIE", "DTWEXBGS", "VIXCLS", "DEXCHUS", "PNGASEUUSDM",
                           "PCOPPUSDM", "PWHEAMTUSDM", "PMAIZMTUSDM", "PALUMUSDM", "PIORECRUSDM", "GEPUCURRENT"]),
    "ecb_rates": lambda: HS.get_parquet("lab/raw/ecb_rates.parquet").reset_index().pivot_table(index="date", columns="id", values="value").sort_index(),
    "strike": _gdelt_w("strike"),
    "protest": _gdelt_w("protest"),
    "violence": _gdelt_w("violence"),
    "coerce": _gdelt_w("coerce"),
    "tone": lambda: _gdelt("*", "tone", "D"),
    "events": lambda: _gdelt("*", "n", "D"),
    "kev": _kev_daily,
    "ip": lambda: _long("eurostat/industrial_production.parquet"),
    "retail": lambda: _long("eurostat/retail_volume.parquet"),
    "unemp": lambda: _long("lab/raw/estat_unemployment.parquet"),
    "hicp": lambda: _long("lab/raw/estat_hicp.parquet"),
    "esi": lambda: _long("lab/raw/estat_esi.parquet"),
    "consumer_conf": lambda: _long("lab/raw/estat_consumer_conf.parquet"),
    "industry_conf": lambda: _long("lab/raw/estat_industry_conf.parquet"),
    "ppi": lambda: _long("lab/raw/estat_ppi.parquet"),
    "stringency": lambda: _oxcgrt("stringency"),
    "hicp_index": lambda: _long("lab/raw/estat_hicp_index.parquet"),
    "hicp_energy_index": lambda: _long("lab/raw/estat_hicp_energy_index.parquet"),
    "diesel": lambda: _oil("diesel"),
    "gasoline": lambda: _oil("gasoline95"),
}


def _daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("D").mean()


def _hsutc(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
    return x


def _monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("MS").mean()


def _regional(df: pd.DataFrame, ccs) -> pd.DataFrame:
    return df.reindex(columns=[c for c in ccs if c in df.columns]).sum(axis=1).to_frame("*")


# ------------------------------------------------------------------------------------------------------------------------------ families
def _markets(P: Pool, freq: str) -> list[Driver]:
    """Global market drivers on any family (daily values; monthly families read their month means)."""
    f, fx, r = P.get("fred"), P.get("fx"), P.get("ecb_rates")
    brent_eur = _to_eur(f["DCOILBRENTEU"].dropna(), fx)
    hh_eur = _to_eur(f["DHHNGSP"].dropna(), fx)
    agg = _monthly_mean if freq == "M" else (lambda x: x)
    D = [Driver("brent_eur", "markets", agg(_g(brent_eur)), DAY + 12), Driver("henry_hub_eur", "markets", agg(_g(hh_eur)), DAY + 12),
         Driver("usd_per_eur", "markets", agg(_g(fx["USD"])), 15), Driver("vix", "markets", agg(_g(f["VIXCLS"])), DAY + 12),
         Driver("us10y", "markets", agg(_g(f["DGS10"])), DAY + 12), Driver("us2y_minus_ecb", "markets",
         agg(_g((f["DGS2"].dropna() - r["FM/B.U2.EUR.4F.KR.DFR.LEV"].reindex(f["DGS2"].dropna().index, method="ffill")).dropna())), DAY + 12),
         Driver("dollar_index", "markets", agg(_g(f["DTWEXBGS"])), DAY + 12),
         Driver("gas_eu_usd", "markets", _g(f["PNGASEUUSDM"]), MONTH + 7 * DAY, note="IMF monthly"),
         Driver("wheat_usd", "markets", _g(f["PWHEAMTUSDM"]), MONTH + 7 * DAY), Driver("copper_usd", "markets", _g(f["PCOPPUSDM"]), MONTH + 7 * DAY),
         Driver("policy_uncertainty", "society", _g(f["GEPUCURRENT"]), MONTH + 10 * DAY),
         Driver("ecb_deposit_rate", "economy", _g(r["FM/B.U2.EUR.4F.KR.DFR.LEV"]), 12),
         Driver("oil_region_violence", "society", agg(_regional(P.get("violence"), OIL_REGION)) if freq != "M" else _monthly_mean(_regional(P.get("violence"), OIL_REGION)), DAY + 1,
                note="GDELT violent events in oil-producing and shipping-lane countries"),
         Driver("kev_new", "cyber", agg(P.get("kev")), DAY)]
    return D


def _society(P: Pool, freq: str, ccs) -> list[Driver]:
    agg = _monthly_mean if freq == "M" else (lambda x: x)
    return [Driver(f"gdelt_{k}", "society", agg(P.get(k).reindex(columns=list(ccs))), DAY + 1) for k in ("strike", "protest")] + \
           [Driver("gdelt_tone", "society", agg(P.get("tone").reindex(columns=list(ccs))), DAY + 1)]


def _energy_h(P: Pool, ccs=ENTSOE_CC) -> list[Driver]:
    g = P.get("gen")
    return [Driver("load_da", "energy", P.get("load_da"), 0, ahead=True, note="ENTSO-E day-ahead total load forecast (6.1.b)", dayahead_hour=9),
            Driver("load", "energy", P.get("load"), 2), Driver("price", "energy", P.get("price"), 0, dayahead_hour=12),
            Driver("wind_share", "energy", g["wind"], 2), Driver("solar_share", "energy", g["solar"], 2), Driver("nuclear_share", "energy", g["nuclear"], 2),
            Driver("gas_share", "energy", g["gas"], 2), Driver("hydro_share", "energy", g["hydro"], 2),
            Driver("net_imports", "energy", P.get("net_imp"), 2), Driver("gen_outage_mw", "energy", P.get("outage"), 2),
            Driver("temp", "weather", P.get("temp").reindex(columns=list(ccs)), DAY + 6), Driver("hdd", "weather", P.get("hdd").reindex(columns=list(ccs)), DAY + 6),
            Driver("cdd", "weather", P.get("cdd").reindex(columns=list(ccs)), DAY + 6)]


def _indeed_driver(cc, per: str) -> "Driver":
    """Indeed Hiring Lab total job postings index (lab.src_jobs; labour layer) as a cross-layer driver: one country (cc) or all ('M': monthly mean, one column per country)."""
    j = HS.get_parquet("lab/raw/indeed_jobs.parquet")
    j = _hsutc(j)
    j = j[j["key"].str.match(r"^total:")]
    daily = j.reset_index().pivot_table(index="date", columns="key", values="value").sort_index()
    daily.columns = [c.split(":")[1] for c in daily.columns]
    if per == "M":
        return Driver("indeed_postings", "labour", daily.resample("MS").mean(), MONTH + 14 * DAY, note="Indeed Hiring Lab job postings index (CC BY 4.0), monthly mean; known 14 days after the month")
    return Driver("indeed_postings", "labour", daily[[cc]].rename(columns={cc: "*"}), 14 * DAY, note="Indeed Hiring Lab job postings index (CC BY 4.0), 7-day trailing; refreshed weekly: 14 days of delay")


def build_family(fid: str, P: Pool | None = None) -> Family:
    P = P or Pool()
    if fid in ("price_h", "load_h", "res_h"):
        if fid == "price_h":
            Y, lab, unit, av = P.get("price"), "Day-ahead electricity price", "EUR/MWh", -12.0
        elif fid == "load_h":
            Y, lab, unit, av = P.get("load"), "Electricity demand (actual load)", "MW", 2.0
        else:
            g = P.get("gen")
            Y, lab, unit, av = (g["wind"] + g["solar"]) * 100, "Wind and solar share of generation", "%", 2.0
        D = _energy_h(P) + _markets(P, "D") + _society(P, "D", ENTSOE_CC)
        if fid == "price_h":
            D = [d for d in D if d.name != "price"]
        fam = Family(fid, lab, "energy", "h", unit, Y, av, skip=13, H=48, seasons=(24, 168), origin_hour=10, drivers=D,
                     notes="origin 10:00 UTC (before the day-ahead auction); forecasts the next two calendar days (hours 14-61)",
                     dayahead_hour=12 if fid == "price_h" else None)
    elif fid == "load_d":
        D = [Driver(d.name, d.group, _daily_mean(d.data), d.avail_h + DAY, d.ahead, d.note, d.dayahead_hour) for d in _energy_h(P)] + _markets(P, "D") + \
            _society(P, "D", ENTSOE_CC)
        fam = Family(fid, "Electricity demand, daily mean", "energy", "D", "MW", _daily_mean(P.get("load")), DAY + 2, 0, 7, (7,), drivers=D)
    elif fid in ("diesel_w", "gasoline_w"):
        Y = P.get("diesel" if fid == "diesel_w" else "gasoline")
        other = P.get("gasoline" if fid == "diesel_w" else "diesel")
        D = [Driver("other_fuel", "markets", other, 4 * DAY)] + _markets(P, "D") + _society(P, "D", EU27)
        fam = Family(fid, "Diesel pump price" if fid == "diesel_w" else "Petrol (95) pump price", "fuel", "W", "EUR/l", Y, 4 * DAY, 0, 4, (52,), drivers=D,
                     notes="EU Weekly Oil Bulletin, with taxes, 27 countries")
    elif fid in ("ip_m", "retail_m", "unemp_m", "hicp_m"):
        k = fid[:-2]
        Y = P.get(k).reindex(columns=[c for c in EU27 if c in P.get(k).columns])
        lag = {"ip": 45, "retail": 35, "unemp": EUROSTAT_DAYS["unemp"], "hicp": 17}[k]   # unemp: 36 since the lag audit (target)
        D = _markets(P, "M") + _society(P, "M", EU27)
        for o, ol in (("ip", 45), ("retail", 35), ("unemp", EUROSTAT_DAYS["unemp"]), ("hicp", 17)):
            if o != k:
                D.append(Driver(o, "economy", P.get(o), MONTH + ol * DAY))
        D += [Driver("load_month", "energy", _monthly_mean(P.get("load")), MONTH + 2), Driver("price_month", "energy", _monthly_mean(P.get("price")), MONTH),
              Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY),
              Driver("hdd_month", "weather", _monthly_mean(P.get("hdd")), MONTH + DAY), Driver("gen_outage_month", "energy", _monthly_mean(P.get("outage")), MONTH)]
        lab = {"ip": "Industrial production", "retail": "Retail trade volume", "unemp": "Unemployment rate", "hicp": "Consumer price inflation (HICP, y/y)"}[k]
        unit = {"ip": "index 2021=100", "retail": "index 2021=100", "unemp": "% of labour force", "hicp": "%"}[k]
        fam = Family(fid, lab, "economy", "M", unit, Y, MONTH + lag * DAY, 0, 3, (12,), drivers=D)
    elif fid == "fx_d":
        fx = P.get("fx").reindex(columns=list(FX))
        D = [d for d in _markets(P, "D") if d.name != "usd_per_eur"] + [Driver("tone_world", "society", _g(P.get("tone").mean(axis=1)), DAY + 1)]
        fam = Family(fid, "Euro exchange rates (ECB reference)", "markets", "D", "units per EUR", fx, 15, 0, 7, (5,), drivers=D)
    elif fid == "commod_d":
        f, fx = P.get("fred"), P.get("fx")
        Y = pd.DataFrame({"brent_eur": _to_eur(f["DCOILBRENTEU"].dropna(), fx), "wti_eur": _to_eur(f["DCOILWTICO"].dropna(), fx),
                          "henry_hub_eur": _to_eur(f["DHHNGSP"].dropna(), fx)})
        D = [d for d in _markets(P, "D") if d.name not in ("brent_eur", "henry_hub_eur")]
        fam = Family(fid, "Oil and gas prices in euros", "markets", "D", "EUR", Y, EIA_SPOT_LAG_H, 0, 7, (5,), drivers=D,
                     notes="EIA daily spot prices, released weekly: the target is known EIA_SPOT_LAG_H after its date (lag audit, 20 Oct 2026; "
                           "DAY + 12 before)")
    elif fid == "unrest_w":
        ccs = [c for c in EU27 + ("GB", "UA", "TR") if c in P.get("strike").columns]
        Y = pd.concat({"strike": P.get("strike")[ccs].resample("W-MON", label="left", closed="left").sum(),
                       "protest": P.get("protest")[ccs].resample("W-MON", label="left", closed="left").sum()}, axis=1)
        Y.columns = [f"{a}:{b}" for a, b in Y.columns]
        D = _markets(P, "D") + [Driver("hicp", "economy", P.get("hicp"), MONTH + 17 * DAY), Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                                Driver("diesel", "fuel", P.get("diesel"), 4 * DAY), Driver("tone", "society", P.get("tone"), DAY + 1)]
        fam = Family(fid, "Strikes and protests in the news (GDELT events per week)", "society", "W", "events/week", Y, WEEK + 1, 0, 4, (52,),
                     drivers=D, notes="series id 'strike:FR' etc.; country drivers matched on the part after ':'")
    elif fid == "cpi_long_m":
        # (docs/methods.md): decades of consumer prices, so the oil shocks of 1973 and 1979,
        # 1990 and 2008 are in the training data and the 2021-23 surge is in the locked window
        yoy = _long("lab/raw/oecd_cpi_yoy.parquet")
        idx = _long("lab/raw/oecd_cpi_index.parquet")
        keep = [c for c in yoy.columns if len(c) == 3 and c.isalpha() and c not in ("OECD", "EA20", "EU27", "G20", "G7", "OEU", "EA19", "EU28", "EA")]
        Y = yoy[keep]
        mom = idx.reindex(columns=keep).pct_change(fill_method=None) * 100
        f = P.get("fred")
        wti = HS.get_parquet("lab/raw/fred_wti_monthly.parquet")["value"]
        wti.index = pd.DatetimeIndex(wti.index).tz_convert("UTC") if pd.DatetimeIndex(wti.index).tz is not None else pd.DatetimeIndex(wti.index).tz_localize("UTC")
        D = [Driver("wti_usd", "markets", _g(wti), MONTH + 5 * DAY, note="FRED WTISPLC, monthly since 1946"),
             Driver("wti_yoy", "markets", _g(wti.pct_change(12, fill_method=None) * 100), MONTH + 5 * DAY),
             Driver("us10y", "markets", _monthly_mean(_g(f["DGS10"])), MONTH),
             Driver("cpi_mom", "economy", mom, MONTH + 15 * DAY, note="OECD CPI index, monthly change"),
             Driver("base_effect", "economy", mom.shift(12, freq="MS"), MONTH + 15 * DAY - 12 * MONTH, ahead=True,
                    note="the monthly change 12 months before each month (it leaves the y/y rate that month)"),
             Driver("gas_eu_usd", "markets", _g(f["PNGASEUUSDM"]), MONTH + 7 * DAY), Driver("wheat_usd", "markets", _g(f["PWHEAMTUSDM"]), MONTH + 7 * DAY),
             Driver("policy_uncertainty", "society", _g(f["GEPUCURRENT"]), MONTH + 10 * DAY)]
        fam = Family(fid, "Consumer price inflation, OECD countries since 1960", "economy", "M", "%", Y, MONTH + 15 * DAY, 0, 3, (12,), drivers=D,
                     notes="OECD CPI y/y, ~45 countries; series ids are ISO3", span_min=10)
    elif fid in ("ip_long_m", "ur_long_m"):
        # the same treatment as cpi_long_m for the families that lost their gains after 2021 (industry, unemployment)
        AGG = ("OECD", "EA20", "EU27", "G20", "G7", "OEU", "EA19", "EU28", "EA", "OECDE", "G4E", "NAFTA", "OXE", "A5M", "EA17")
        rd = lambda k: _long(f"lab/raw/oecd_{k}.parquet")  # noqa: E731
        ip, ur, cpi = rd("ip"), rd("unemployment"), _long("lab/raw/oecd_cpi_yoy.parquet")
        Y = ip if fid == "ip_long_m" else ur
        Y = Y[[c for c in Y.columns if len(c) == 3 and c.isalpha() and c not in AGG]]
        f = P.get("fred")
        wti = HS.get_parquet("lab/raw/fred_wti_monthly.parquet")["value"]
        wti.index = pd.DatetimeIndex(wti.index).tz_convert("UTC") if pd.DatetimeIndex(wti.index).tz is not None else pd.DatetimeIndex(wti.index).tz_localize("UTC")
        other = (ur, "unemployment", MONTH + EUROSTAT_DAYS["unemp"] * DAY) if fid == "ip_long_m" else (ip, "industrial_production", MONTH + 45 * DAY)
        D = [Driver("cli", "economy", rd("cli"), MONTH + 10 * DAY, note="OECD composite leading indicator"),
             Driver("business_conf", "economy", rd("bci"), MONTH), Driver("consumer_conf", "economy", rd("cci"), MONTH),
             Driver(other[1], "economy", other[0], other[2]), Driver("cpi_yoy", "economy", cpi, MONTH + 15 * DAY),
             Driver("wti_usd", "markets", _g(wti), MONTH + 5 * DAY), Driver("us10y", "markets", _monthly_mean(_g(f["DGS10"])), MONTH),
             Driver("policy_uncertainty", "society", _g(f["GEPUCURRENT"]), MONTH + 10 * DAY),
             Driver("gas_eu_usd", "markets", _g(f["PNGASEUUSDM"]), MONTH + 7 * DAY)]
        lab = "Industrial production, OECD countries since 1955" if fid == "ip_long_m" else "Unemployment rate, OECD countries since 1955"
        fam = Family(fid, lab, "economy", "M", "index" if fid == "ip_long_m" else "%", Y, MONTH + (45 if fid == "ip_long_m" else EUROSTAT_DAYS["unemp"]) * DAY, 0, 3, (12,),
                     drivers=D, notes="OECD monthly, series ids ISO3", span_min=10)
    elif fid == "ports_m":
        # transport and logistics layer: monthly cargo tonnes of Spain's 29 port authorities (Puertos del Estado, open data)
        g = HS.get_parquet("ports_es/monthly.parquet")
        piv = lambda m: g[g["metric"] == m].pivot_table(index=g[g["metric"] == m].index, columns="port", values="value").sort_index()   # noqa: E731
        Y = piv("tonnes")
        Y.columns = [f"{c}:ES" for c in Y.columns]
        teu, pax = piv("teu"), piv("passengers")
        teu.columns = [f"{c}:ES" for c in teu.columns]
        pax.columns = [f"{c}:ES" for c in pax.columns]
        lag = 60 * DAY                                         # monthly summaries come out about 6-8 weeks after the month (conservative)
        D = _markets(P, "M") + _society(P, "M", ("ES",)) + [
            Driver("teu", "trade", teu, MONTH + lag, note="the port's own container traffic (TEU)"),
            Driver("passengers", "transport", pax, MONTH + lag, note="the port's own passengers"),
            Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY), Driver("retail", "economy", P.get("retail"), MONTH + 35 * DAY),
            Driver("hicp", "economy", P.get("hicp"), MONTH + 17 * DAY), Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY),
            Driver("hdd_month", "weather", _monthly_mean(P.get("hdd")), MONTH + DAY)]   # (no ENTSO-E prices: restricted, would block live use)
        fam = Family(fid, "Port cargo, Spain's 29 port authorities", "trade", "M", "tonnes", Y, MONTH + lag, 0, 3, (12,), drivers=D,
                     notes="Puertos del Estado monthly statistics; series ids 'port:ES'; country drivers matched on ES")
    elif fid == "health_w":
        # health layer: weekly influenza-like / acute respiratory illness consultation rate (ECDC ERVISS, from mid-2022),
        # published about 5 days after the week; units differ by country (each series is scaled on its own)
        Y = HS.get_parquet("lab/raw/health_ervis.parquet")
        Y.index = pd.DatetimeIndex(Y.index).tz_convert("UTC") if pd.DatetimeIndex(Y.index).tz is not None else pd.DatetimeIndex(Y.index).tz_localize("UTC")
        wk = lambda x: x.resample("W-MON", label="left", closed="left").mean()   # noqa: E731
        ccs = list(Y.columns)
        air = HS.get_parquet("lab/raw/air_traffic.parquet")
        D = [Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)), WEEK + 6 * DAY), Driver("hdd", "weather", wk(P.get("hdd").reindex(columns=ccs)), WEEK + 6 * DAY),
             Driver("flights", "transport", wk(air.reindex(columns=ccs)), WEEK + 46 * DAY, note="EUROCONTROL IFR flights (mobility; back-tests only)"),
             Driver("stringency", "health", wk(P.get("stringency").reindex(columns=ccs)), WEEK + DAY)] + \
            [Driver(f"gdelt_{k}", "society", wk(P.get(k).reindex(columns=ccs)), WEEK + DAY) for k in ("protest",)] + _markets(P, "D")
        fam = Family(fid, "Respiratory illness (ILI / ARI consultation rate, ECDC ERVISS)", "health", "W", "rate", Y, WEEK + 5 * DAY, 0, 4, (52,),
                     drivers=D, notes="weeks start Monday; BG, CY, DE report ARI; units per country differ (scaled per series); from 2022-W25")
    elif fid == "air_d":
        # air transport layer: daily IFR arrivals + departures at a fixed set of 263 airports (EUROCONTROL; NOT for commercial
        # use: back-tests only); monthly files come out 2-6 weeks after the month: a day is known 46 days later at the latest
        Y = HS.get_parquet("lab/raw/air_traffic.parquet")
        Y.index = pd.DatetimeIndex(Y.index).tz_convert("UTC") if pd.DatetimeIndex(Y.index).tz is not None else pd.DatetimeIndex(Y.index).tz_localize("UTC")
        ccs = list(Y.columns)
        h = HS.get_parquet("lab/raw/health_ervis.parquet")
        h.index = pd.DatetimeIndex(h.index).tz_convert("UTC") if pd.DatetimeIndex(h.index).tz is not None else pd.DatetimeIndex(h.index).tz_localize("UTC")
        D = _markets(P, "D") + _society(P, "D", ccs) + [
            Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
            Driver("ili_rate", "health", h.reindex(columns=ccs), WEEK + 5 * DAY, note="ECDC ERVISS weekly rate (from 2022)"),
            Driver("temp", "weather", P.get("temp").reindex(columns=ccs), DAY + 6),
            Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("diesel", "fuel", P.get("diesel"), 4 * DAY)]
        fam = Family(fid, "Air traffic: daily IFR flights per country (EUROCONTROL)", "transport", "D", "flights/day", Y, 46 * DAY, 0, 14, (7, 364),
                     drivers=D, notes="263 airports present in 2016 and 2026; back-tests only (EUROCONTROL licence)")
    elif fid in ("security_w", "info_w"):
        # security and information layers: GDELT events per week per country, violence (CAMEO 18-20)
        # and coercion (16-17) as security; the average tone of all events as information climate; Europe and its neighbours
        ccs = [c for c in EU27 + ("GB", "UA", "TR", "RU", "IL", "EG") if c in P.get("violence").columns]
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        if fid == "security_w":
            Y = pd.concat({"violence": wk(P.get("violence")[ccs]).sum(), "coerce": wk(P.get("coerce")[ccs]).sum()}, axis=1)
            lab, unit = "Security: violence and coercion events in the news (GDELT, per week)", "events/week"
        else:
            Y = pd.concat({"tone": wk(P.get("tone").reindex(columns=ccs)).mean()}, axis=1)
            lab, unit = "Information climate: average tone of news events (GDELT, per week)", "tone"
        Y.columns = [f"{a}:{b}" for a, b in Y.columns]
        D = _markets(P, "D") + [Driver("hicp", "economy", P.get("hicp"), MONTH + 17 * DAY), Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                                Driver("diesel", "fuel", P.get("diesel"), 4 * DAY), Driver("gdelt_strike", "society", P.get("strike"), DAY + 1),
                                Driver("gdelt_protest", "society", P.get("protest"), DAY + 1),
                                Driver("stringency", "health", P.get("stringency"), DAY + 1)]
        if fid == "security_w":
            D.append(Driver("tone", "society", P.get("tone"), DAY + 1))
        else:
            D += [Driver("violence", "security", P.get("violence"), DAY + 1), Driver("coerce", "security", P.get("coerce"), DAY + 1)]
        fam = Family(fid, lab, "society", "W", unit, Y, WEEK + 1, 0, 4, (52,), drivers=D,
                     notes="series ids 'violence:FR' etc.; country drivers matched on the part after ':'; GDELT volume drifts over the years")
    elif fid == "tourism_m":
        # tourism layer: nights spent in tourist accommodation, Eurostat tour_occ_nim (CC BY 4.0 reuse policy), 30 countries,
        # published about 2-2.5 months after the month (73 days in Oct 2026); COVID 2020-21 is a break the models must live through
        Y = HS.get_parquet("lab/raw/tourism_nights.parquet")
        Y.index = pd.DatetimeIndex(Y.index).tz_convert("UTC") if pd.DatetimeIndex(Y.index).tz is not None else pd.DatetimeIndex(Y.index).tz_localize("UTC")
        ccs = list(Y.columns)
        air = HS.get_parquet("lab/raw/air_traffic.parquet")
        D = _markets(P, "M") + _society(P, "M", ccs) + [
            Driver("violence_month", "security", _monthly_mean(P.get("violence").reindex(columns=ccs)), MONTH + DAY),
            Driver("stringency_month", "health", _monthly_mean(P.get("stringency").reindex(columns=ccs)), MONTH + DAY),
            Driver("flights_month", "transport", _monthly_mean(air.reindex(columns=ccs)), MONTH + 46 * DAY, note="EUROCONTROL (back-tests only)"),
            Driver("hicp", "economy", P.get("hicp"), MONTH + 17 * DAY), Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
            Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY),
            Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY)]
        fam = Family(fid, "Tourism: nights spent in tourist accommodation (Eurostat)", "tourism", "M", "nights", Y, MONTH + 73 * DAY, 0, 3, (12,),
                     drivers=D, notes="30 countries (EU27 + NO, CH, IS); Greece = GR; COVID collapse 2020-21 inside the history")
    elif fid in ("water_w", "wildfire_w"):
        # water and environment layers: Spain's reservoir storage by basin, % of capacity (MITECO Boletin Hidrologico, weekly
        # since 1988, published the next day); burnt area per country per week (EFFIS rapid damage assessment, >= ~30 ha fires, CC BY 4.0)
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        res = HS.get_parquet("lab/raw/reservoirs_es.parquet")
        res.index = pd.DatetimeIndex(res.index).tz_convert("UTC") if pd.DatetimeIndex(res.index).tz is not None else pd.DatetimeIndex(res.index).tz_localize("UTC")
        fire = HS.get_parquet("lab/raw/wildfire_burnt.parquet")
        fire.index = pd.DatetimeIndex(fire.index).tz_convert("UTC") if pd.DatetimeIndex(fire.index).tz is not None else pd.DatetimeIndex(fire.index).tz_localize("UTC")
        if fid == "water_w":
            Y = wk(res).mean()
            ccs = ["ES"]
            D = [Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                 Driver("cdd", "weather", wk(P.get("cdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                 Driver("burnt_area", "environment", wk(fire.reindex(columns=ccs)).sum(), WEEK + 7 * DAY),
                 Driver("gas_eu_usd", "markets", _g(P.get("fred")["PNGASEUUSDM"]), MONTH + 7 * DAY),
                 Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY)] + _society(P, "D", ccs)
            fam = Family(fid, "Water: reservoir storage in Spain by basin (MITECO, % of capacity)", "water", "W", "% of capacity", Y, WEEK + DAY, 0, 4, (52,),
                         drivers=D, notes="series 'basin:ES' and 'total:ES'; Spain-level drivers matched on ES")
        else:
            Y = wk(fire).sum()
            ccs = list(Y.columns)
            D = [Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                 Driver("cdd", "weather", wk(P.get("cdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                 Driver("reservoirs_es", "water", wk(res[["total:ES"]].rename(columns={"total:ES": "ES"})).mean(), WEEK + DAY, note="drought proxy for ES"),
                 Driver("tourism_nights", "tourism", HS.get_parquet("lab/raw/tourism_nights.parquet"), MONTH + 73 * DAY)] + \
                _society(P, "D", ccs) + _markets(P, "D")
            fam = Family(fid, "Environment: wildfire burnt area per country (EFFIS, ha per week)", "environment", "W", "ha/week", Y, WEEK + 7 * DAY, 0, 4, (52,),
                         drivers=D, notes="38 countries; fires >= ~30 ha; last 1-2 weeks provisional (hence 7 days of delay)")
    elif fid == "wiki_w":
        # public attention layer: Wikipedia page views (users only) in 21 language editions, summed over the articles of
        # a topic group, per week (Monday-Sunday). Wikimedia pageview data is CC0; the API serves day D on D+1; a week is known 2 days after it ends.
        # A language edition stands for its home country only for matching country drivers (Spanish and Portuguese are read mostly outside ES / PT).
        from forecasting_engine.lab import src_wiki as SW
        tp = json.loads(HS.get_bytes(SW.TOPIC_STORE))
        pv = HS.get_parquet("lab/raw/wiki_pv_daily.parquet")
        pv.index = pd.DatetimeIndex(pv.index).tz_convert("UTC") if pd.DatetimeIndex(pv.index).tz is not None else pd.DatetimeIndex(pv.index).tz_localize("UTC")
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        cols = {}
        keep = ("DE", "FR", "IT", "PL", "NL", "SE", "CZ", "RO")      # 8 editions (cost: the 21-edition version, 168 series, needed hours of CPU per method); es and pt left out (read mostly outside ES / PT)
        for lang, cc in SW.LANG_CC.items():
            if cc not in keep:
                continue
            for g in SW.TOPICS:
                qs = [c for c in pv.columns if c.startswith(lang + ":") and tp[c.split(":")[1]]["group"] == g and tp[c.split(":")[1]]["en"] not in SW.DROP]
                if qs:
                    cols[f"{g}:{cc}"] = pv[qs].sum(axis=1, min_count=1)
        daily = pd.DataFrame(cols)
        full = wk(daily.notna()).sum() >= 7
        Y = wk(daily).sum().where(full).dropna(how="all")
        Y = Y[Y.index >= pd.Timestamp("2015-07-06", tz="UTC")]
        ccs = sorted({c.split(":")[1] for c in Y.columns})
        h = HS.get_parquet("lab/raw/health_ervis.parquet")
        h.index = pd.DatetimeIndex(h.index).tz_convert("UTC") if pd.DatetimeIndex(h.index).tz is not None else pd.DatetimeIndex(h.index).tz_localize("UTC")
        D = [Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
             Driver("hdd", "weather", wk(P.get("hdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
             Driver("cdd", "weather", wk(P.get("cdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
             Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
             Driver("ili_rate", "health", h.reindex(columns=ccs), WEEK + 5 * DAY, note="ECDC ERVISS weekly rate (from 2022)"),
             Driver("hicp", "economy", P.get("hicp"), MONTH + EUROSTAT_DAYS["hicp"] * DAY), Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
             Driver("diesel", "fuel", P.get("diesel"), 4 * DAY), Driver("violence", "security", P.get("violence").reindex(columns=ccs), DAY + 1)] + \
            _markets(P, "D") + _society(P, "D", ccs)
        fam = Family(fid, "Public attention: Wikipedia page views per topic group and language edition", "attention", "W", "views/week", Y, WEEK + 2 * DAY, 0, 4, (52,),
                     drivers=D, notes="series 'health:DE' = German-edition views of 4-6 health articles; 8 editions (DE FR IT PL NL SE CZ RO); user traffic only; "
                                      "edition stands for its home country only to match country drivers; Wikimedia CC0")
    elif fid == "jobs_w":
        # labour market layer: Indeed Hiring Lab job postings index (CC BY 4.0), NOT seasonally adjusted, total and new postings,
        # 10 countries (DE ES FR IE IT NL GB US AU CA), weekly mean of the daily 7-day-trailing index; the repo is refreshed weekly (data to the Friday
        # of the week before the pull on 23 Oct 2026): a week is known 10 days after it ends (WEEK + 10 days from its Monday stamp)
        j = HS.get_parquet("lab/raw/indeed_jobs.parquet")
        j.index = pd.DatetimeIndex(j.index).tz_convert("UTC") if pd.DatetimeIndex(j.index).tz is not None else pd.DatetimeIndex(j.index).tz_localize("UTC")
        j = j[j["key"].str.match(r"^(total|new):")]
        daily = j.reset_index().pivot_table(index="date", columns="key", values="value").sort_index()
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        full = wk(daily.notna()).sum() >= 7
        Y = wk(daily).mean().where(full).dropna(how="all")
        ccs = sorted({c.split(":")[1] for c in Y.columns})
        D = [Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
             Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
             Driver("hicp", "economy", P.get("hicp"), MONTH + EUROSTAT_DAYS["hicp"] * DAY), Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
             Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY), Driver("retail", "economy", P.get("retail"), MONTH + 35 * DAY),
             Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("industry_conf", "economy", P.get("industry_conf"), MONTH),
             Driver("esi", "economy", P.get("esi"), MONTH), Driver("diesel", "fuel", P.get("diesel"), 4 * DAY)] + _markets(P, "D") + _society(P, "D", ccs)
        fam = Family(fid, "Labour market: Indeed job postings index (not seasonally adjusted)", "labour", "W", "index (1 Feb 2020 = 100)", Y, WEEK + 10 * DAY, 0, 4, (52,),
                     drivers=D, notes="series 'total:DE' / 'new:DE'; 10 countries; from 2020-02; Indeed Hiring Lab, CC BY 4.0; the series is the latest vintage")
    elif fid in ("asylum_m", "milk_m"):
        # migration layer: first-time asylum applicants per month (Eurostat migr_asyappctzm); agriculture layer: raw cow's milk collected per month
        # (Eurostat apro_mk_colm). Eurostat reuse licence (CC BY 4.0 equivalent). Lags from the latest month present per country on 23 Oct 2026
        # (asylum: up to 2026-07 for DE, ES, PL, SE, SI: month + 85 days; milk: 2026-08 for most countries: month + 55 days; Italy ends 2026-04: left out)
        k = "asylum" if fid == "asylum_m" else "milk"
        x = HS.get_parquet(f"lab/raw/estat_{k}.parquet")
        x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
        keep = [c for c in EU27 + ("CH", "NO") if c in x.columns and x[c].notna().sum() >= 100 and x[c].iloc[-60:].mean() >= (20 if k == "asylum" else 20)]
        Y = x[keep].dropna(how="all")
        if k == "milk":
            Y = Y.drop(columns=[c for c in ("IT", "LU", "CH", "NO") if c in Y.columns])
        ccs = list(Y.columns)
        lagd = 85 if k == "asylum" else 55
        D = _markets(P, "M") + _society(P, "M", ccs) + [
            Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY), Driver("hicp", "economy", P.get("hicp"), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
            Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY), Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH),
            Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY),
            Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY)]
        if k == "asylum":
            vi = P.get("violence")
            D += [Driver("violence_world", "security", _monthly_mean(vi.sum(axis=1).to_frame("*")), MONTH + DAY, note="GDELT violent events, all countries"),
                  Driver("violence_conflict_zones", "security", _monthly_mean(_regional(vi, ("SY", "AF", "UA", "SD", "SS", "IQ", "YE", "SO", "ML", "NG", "CD", "MM", "IR", "PK", "ET", "HT", "VE"))), MONTH + DAY,
                         note="GDELT violent events in the main origin countries of asylum seekers")]
        else:
            D += [Driver("wheat_cycle", "markets", _g(P.get("fred")["PWHEAMTUSDM"]), MONTH + 28 * DAY), Driver("hdd_month", "weather", _monthly_mean(P.get("hdd").reindex(columns=ccs)), MONTH + DAY),
                  Driver("cdd_month", "weather", _monthly_mean(P.get("cdd").reindex(columns=ccs)), MONTH + DAY)]
        fam = Family(fid, "Migration: first-time asylum applicants (Eurostat)" if k == "asylum" else "Agriculture: raw cow's milk collected (Eurostat)",
                     "migration" if k == "asylum" else "agriculture", "M", "persons" if k == "asylum" else "thousand tonnes", Y, MONTH + lagd * DAY, 0, 3, (12,), drivers=D,
                     notes=f"Eurostat {'migr_asyappctzm' if k == 'asylum' else 'apro_mk_colm'}; {len(ccs)} countries; latest vintage")
    elif fid == "space_d":
        # space weather layer: daily geomagnetic activity (Ap, Kp) and solar radio flux (F10.7), GFZ Potsdam (CC BY 4.0), 1995-.
        # The Sun drives it, nothing on Earth does: no Earth driver enters as an input; the other three series are the cross-series context.
        x = HS.get_parquet("lab/raw/space_daily.parquet")
        x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
        fam = Family(fid, "Space weather: geomagnetic activity and solar flux (GFZ Potsdam)", "space", "D", "index", x[["ap", "kp_max", "kp_mean", "f107"]], 30, 0, 14, (27, 365),
                     drivers=[Driver("kev_new", "cyber", P.get("kev"), DAY)], notes="series ap, kp_max, kp_mean (geomagnetic), f107 (solar radio flux); 27-day solar rotation; nowcast values replaced by definitive ones later (latest vintage)")
    elif fid == "carbon_w":
        # emissions layer: Carbon Monitor weekly (Monday-Sunday sums of daily) CO2 emissions per sector and country (CC BY 4.0), 2019-.
        # Last day 2026-08-31 on 23 Oct: a week is known 60 days after its end (WEEK + 60 d from its Monday stamp); series 'sector:CC'. A monthly family would
        # have had 18 holdout origins (the lab needs 20), hence weekly.
        x = HS.get_parquet("lab/raw/carbon_weekly.parquet")
        x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
        Y = x[[c for c in x.columns if c.split(":")[1] in EU27 + ("GB", "NO", "CH") and c.split(":")[0] in ("power", "industry", "ground", "residential")
                   and c.split(":")[1] in ("DE", "FR", "IT", "ES", "PL", "NL")]].dropna(axis=1, thresh=300)
        ccs = sorted({c.split(":")[1] for c in Y.columns})
        wk = lambda y: y.resample("W-MON", label="left", closed="left")   # noqa: E731
        D = _markets(P, "D") + _society(P, "D", ccs) + [
            Driver("hdd", "weather", wk(P.get("hdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY), Driver("cdd", "weather", wk(P.get("cdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
            Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
            Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY), Driver("retail", "economy", P.get("retail"), MONTH + 35 * DAY),
            Driver("hicp", "economy", P.get("hicp"), MONTH + EUROSTAT_DAYS["hicp"] * DAY), Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
            Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("diesel", "fuel", P.get("diesel"), 4 * DAY)]
        fam = Family(fid, "Emissions: weekly CO2 by sector and country (Carbon Monitor)", "emissions", "W", "Mt CO2/week", Y, WEEK + 60 * DAY, 0, 4, (52,), drivers=D,
                     notes="series 'power:DE' etc.; Carbon Monitor CC BY 4.0; 2019-; its power and transport sectors are modelled from other lab inputs; COVID 2020 in the history; published about 2 months late")
    elif fid in ("credit_m", "constr_m", "firms_m", "housing_m", "sent_m"):
        # layers. credit_m: ECB MIR new-business bank loan volumes (EUR m) per euro-area country: loans to firms (two size
        # classes summed), to households for consumption and for house purchase; ECB publishes MIR about 23 working days after the month (latest month on 23 Oct
        # 2026: 2026-08), so a month is known 65 days after its start. constr_m: Eurostat construction output volume (NSA, 2021 = 100), 20 countries, month + 55 days
        # (latest 2026-08). firms_m: Spain, companies constituted and dissolved per month and autonomous community (INE; latest 2026-07): month + 85 days;
        # series 'created_AN:ES' so that national drivers (ES) and the Spanish calendar apply.
        econ = lambda ccs: [Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY), Driver("hicp", "economy", P.get("hicp"), MONTH + EUROSTAT_DAYS["hicp"] * DAY),  # noqa: E731
                            Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY), Driver("retail", "economy", P.get("retail"), MONTH + 35 * DAY),
                            Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("industry_conf", "economy", P.get("industry_conf"), MONTH),
                            Driver("esi", "economy", P.get("esi"), MONTH), Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY)]
        if fid == "credit_m":
            x = _hsutc(HS.get_parquet("lab/raw/ecb_credit.parquet"))
            Y = x[[c for c in x.columns if x[c].notna().sum() >= 200 and x[c].iloc[-24:].mean() > 20]]
            ccs = sorted({c.split(":")[1] for c in Y.columns})
            D = _markets(P, "M") + _society(P, "M", ccs) + econ(ccs) + [Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY)]
            fam = Family(fid, "Bank credit: new-business loan volumes (ECB MIR)", "credit", "M", "EUR million", Y, MONTH + 65 * DAY, 0, 3, (12,), drivers=D,
                         notes=f"series 'nfc:DE' (loans to firms), 'cons:DE' (consumer loans), 'house:DE' (house purchase); {len(ccs)} areas incl. the euro area (EA); ECB Data Portal, latest vintage")
        elif fid == "constr_m":
            x = _hsutc(HS.get_parquet("lab/raw/estat_constr.parquet"))
            keep = [c for c in x.columns if c in EU27 + ("NO",) and x[c].notna().sum() >= 250 and x[c].last_valid_index() >= pd.Timestamp("2026-07-01", tz="UTC")]
            Y = x[keep].dropna(how="all")
            Y.columns = [f"output:{c}" for c in Y.columns]
            ccs = keep
            pm = _hsutc(HS.get_parquet("lab/raw/estat_permits.parquet"))
            cr = _hsutc(HS.get_parquet("lab/raw/ecb_credit.parquet"))
            D = _markets(P, "M") + _society(P, "M", ccs) + econ(ccs) + [
                Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY),
                Driver("hdd_month", "weather", _monthly_mean(P.get("hdd").reindex(columns=ccs)), MONTH + DAY),
                Driver("permits_m2", "construction", pm.reindex(columns=[c for c in ccs if c in pm.columns]), MONTH + 90 * DAY, note="Eurostat building permits (m2), a leading indicator"),
                Driver("credit_house", "credit", cr[[c for c in cr.columns if c.startswith("house:")]].rename(columns=lambda c: c.split(":")[1]).reindex(columns=[c for c in ccs if c in {k.split(":")[1] for k in cr.columns}]), MONTH + 65 * DAY,
                       note="ECB new-business house-purchase loan volumes"),
                Driver("credit_nfc", "credit", cr[[c for c in cr.columns if c.startswith("nfc:")]].rename(columns=lambda c: c.split(":")[1]).reindex(columns=[c for c in ccs if c in {k.split(":")[1] for k in cr.columns}]), MONTH + 65 * DAY,
                       note="ECB new-business loan volumes to firms")]
            fam = Family(fid, "Construction: output volume (Eurostat sts_copr_m)", "construction", "M", "index (2021 = 100)", Y, MONTH + 55 * DAY, 0, 3, (12,), drivers=D,
                         notes=f"series 'output:DE'; {len(ccs)} countries; Eurostat NSA volume index, latest vintage; building permits and house-purchase credit enter as drivers")
        elif fid == "sent_m":
            # sentiment layer: EC Business and Consumer Surveys balances (Eurostat ei_bssi_m_r2, NSA): consumer, services, retail and construction confidence in 13
            # large EU countries; published on the penultimate working day of the survey month (known at the end of the month: delay = one month from the stamp)
            x = _hsutc(HS.get_parquet("lab/raw/estat_bcs.parquet"))
            Y = x[[c for c in x.columns if x[c].notna().sum() >= 200]].dropna(how="all")
            ccs = sorted({c.split(":")[1] for c in Y.columns})
            D = _markets(P, "M") + _society(P, "M", ccs) + econ(ccs) + [Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY)]
            fam = Family(fid, "Sentiment: EC consumer, services, retail and construction confidence (Eurostat BCS)", "sentiment", "M", "balance (points)", Y, MONTH, 0, 3, (12,), drivers=D,
                         notes=f"series 'consumer:DE', 'services:DE', 'retail:DE', 'construction:DE'; {len(ccs)} countries; NSA balances; latest vintage")
        elif fid == "housing_m":
            # housing layer: Spain, home sales (compraventas de viviendas, INE ETDP) per autonomous community: all, new and second-hand; latest month 2026-07 on 23 Oct
            # 2026 (the August figure was not out): month + 100 days. House-purchase credit and building permits (Spain) enter as drivers.
            x = _hsutc(HS.get_parquet("lab/raw/ine_housing.parquet"))
            Y = x.dropna(how="all")
            ccs = ["ES"]
            cr = _hsutc(HS.get_parquet("lab/raw/ecb_credit.parquet"))
            pm = _hsutc(HS.get_parquet("lab/raw/estat_permits.parquet"))
            D = _markets(P, "M") + _society(P, "M", ccs) + econ(ccs) + [
                Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY),
                Driver("permits_m2", "construction", pm[["ES"]], MONTH + 90 * DAY),
                Driver("credit_house", "credit", cr[["house:ES"]].rename(columns={"house:ES": "ES"}), MONTH + 65 * DAY, note="ECB new-business house-purchase loans, Spain"),
                Driver("credit_nfc", "credit", cr[["nfc:ES"]].rename(columns={"nfc:ES": "ES"}), MONTH + 65 * DAY)]
            fam = Family(fid, "Housing market: home sales per autonomous community (INE)", "housing", "M", "homes sold", Y, MONTH + 100 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'total_AN:ES' / 'new_AN:ES' / 'used_AN:ES' for 19 autonomous communities; INE Estadistica de Transmisiones de Derechos de la Propiedad, 2007-; latest vintage")
        else:
            x = _hsutc(HS.get_parquet("lab/raw/ine_firms.parquet"))
            Y = x.dropna(how="all")
            Y.columns = [f"{c.split(':')[0]}_{c.split(':')[1]}:ES" for c in Y.columns]
            ccs = ["ES"]
            D = _markets(P, "M") + _society(P, "M", ccs) + econ(ccs) + [Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY)]
            fam = Family(fid, "Firm demography: Spanish companies constituted and dissolved per month (INE)", "firms", "M", "companies", Y, MONTH + 85 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'created_AN:ES' / 'dissolved_AN:ES' per autonomous community (19); INE Estadistica de Sociedades Mercantiles, 2007-12-; latest vintage")
    elif fid == "freight_d":
        # road freight layer: Destatis daily lorry-toll mileage index (2021 = 100, unadjusted), German motorways, 2008-; the xlsx is
        # refreshed every Thursday with data to the Saturday before (5 to 12 days of lag): 12 days of delay. One series (the unadjusted index).
        x = _hsutc(HS.get_parquet("lab/raw/lkw_toll_daily.parquet"))
        Y = x[["toll:DE"]].dropna()
        ccs = ["DE"]
        D = _markets(P, "D") + _society(P, "D", ccs) + [
            Driver("temp", "weather", P.get("temp").reindex(columns=ccs), DAY + 6), Driver("hdd", "weather", P.get("hdd").reindex(columns=ccs), DAY + 6),
            Driver("cdd", "weather", P.get("cdd").reindex(columns=ccs), DAY + 6), Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
            Driver("ip", "economy", P.get("ip"), MONTH + EUROSTAT_DAYS["ip"] * DAY), Driver("industry_conf", "economy", P.get("industry_conf"), MONTH),
            Driver("esi", "economy", P.get("esi"), MONTH), Driver("diesel", "fuel", P.get("diesel"), 4 * DAY)]
        fam = Family(fid, "Road freight: daily lorry-toll mileage index, Germany (Destatis / BALM)", "freight", "D", "index (2021 = 100)", Y, 12 * DAY, 0, 14, (7, 365), drivers=D,
                     notes="series 'toll:DE' (unadjusted); trucks with 4+ axles on motorways; 2008-; Destatis EXSTAT, dl-de/by-2.0; latest vintage (the file is refreshed weekly)")
    elif fid in ("services_m", "insolv_m", "research_m", "water_m", "transit_d", "c311_d", "spend_w", "traffic_d"):
        # layers.
        econ = lambda: [Driver("unemp", "economy", P.get("unemp"), MONTH + EUROSTAT_DAYS["unemp"] * DAY), Driver("hicp", "economy", P.get("hicp"), MONTH + EUROSTAT_DAYS["hicp"] * DAY),  # noqa: E731
                        Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY), Driver("retail", "economy", P.get("retail"), MONTH + 35 * DAY),
                        Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("industry_conf", "economy", P.get("industry_conf"), MONTH),
                        Driver("esi", "economy", P.get("esi"), MONTH), Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY)]
        indeed = lambda cc, per: _indeed_driver(cc, per)  # noqa: E731
        if fid == "services_m":
            # Eurostat sts_setu_m net turnover (NSA, 2021 = 100): transport and storage, accommodation and food, ICT, professional services; countries with 250+ months.
            # Published about 60 days after the month (latest 2026-08 for ES, CZ, PT, SK, FI on 24 Oct 2026): month + 65 days.
            Y = _hsutc(HS.get_parquet("lab/raw/estat_services.parquet")).dropna(how="all")
            ccs = sorted({c.split(":")[1] for c in Y.columns})
            bcs = _hsutc(HS.get_parquet("lab/raw/estat_bcs.parquet"))
            D = _markets(P, "M") + _society(P, "M", ccs) + econ() + [
                Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY),
                Driver("hdd_month", "weather", _monthly_mean(P.get("hdd").reindex(columns=ccs)), MONTH + DAY),
                Driver("cdd_month", "weather", _monthly_mean(P.get("cdd").reindex(columns=ccs)), MONTH + DAY)]
            for k in ("services", "consumer"):
                x = bcs[[c for c in bcs.columns if c.startswith(k + ":")]].rename(columns=lambda c: c.split(":")[1])
                D.append(Driver(f"bcs_{k}", "sentiment", x, MONTH, note=f"EC {k} confidence balance (sent_m's own series; published at the end of the survey month)"))
            D.append(indeed(None, "M"))
            fam = Family(fid, "Services: net turnover of transport, hospitality, ICT and professional services (Eurostat sts_setu_m)", "services", "M", "index (2021 = 100)", Y,
                         MONTH + 65 * DAY, 0, 3, (12,), drivers=D, notes=f"series 'transport:ES', 'hospitality:DE', 'ict:FR', 'professional:SE'; {len(ccs)} countries; Eurostat NSA turnover index, latest vintage")
        elif fid == "insolv_m":
            # Insolvency Service (OGL 3.0) long-run monthly series: company insolvencies (England and Wales by type, Scotland) and individual insolvencies (bankruptcies,
            # debt relief orders, IVAs), not seasonally adjusted, 2000-. Published about the 20th of the next month (September 2026 on 20 Oct): month + 25 days.
            Y = _hsutc(HS.get_parquet("lab/raw/uk_insolv.parquet")).dropna(how="all")
            ccs = ["GB"]
            D = _markets(P, "M") + _society(P, "M", ccs) + [
                Driver("unemp", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY), Driver("hicp", "economy", P.get("hicp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
                Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY), Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY),
                indeed("GB", "M")]
            fam = Family(fid, "Insolvencies: UK company and individual insolvencies per month (Insolvency Service)", "insolvency", "M", "cases", Y, MONTH + 25 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'co_total:GB', 'co_cvl:GB', 'ind_bankruptcy:GB' ...; England and Wales (company Scotland separate); Insolvency Service, OGL 3.0, 2000-; latest vintage (recent months are provisional)",
                         span_min=7)
        elif fid == "research_m":
            # Crossref REST API: DOIs first registered ('created') per month by work type (journal articles, chapters, conference papers, preprints, datasets, peer reviews,
            # dissertations, reports, books, other); metadata CC0. A month is complete at its end; counted known 3 days after.
            Y = _hsutc(HS.get_parquet("lab/raw/crossref_monthly.parquet")).dropna(how="all")
            Y.columns = [f"{c}:WW" for c in Y.columns]
            D = _markets(P, "M")
            fam = Family(fid, "Research output: DOIs registered per month by work type (Crossref)", "research", "M", "DOIs", Y, MONTH + 3 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'journal_article:WW', 'preprint:WW', 'dataset:WW' ...; Crossref created date (CC0); 2002- (types start when they reach 50 a month); bulk deposits are in the data",
                         span_min=5)
        elif fid == "water_m":
            # California urban water suppliers (State Water Board, data.ca.gov, ODC Attribution): residential gallons per capita per day (R-GPCD), population-weighted, 10 hydrologic
            # regions and the state, over the 355 suppliers reporting 135+ of 147 months, 2014-06-. Latest month 2026-08 on 24 Oct 2026: month + 60 days.
            Y = _hsutc(HS.get_parquet("lab/raw/ca_water.parquet")).dropna(how="all")
            wx = _hsutc(HS.get_parquet("lab/raw/ca_weather.parquet"))
            D = _markets(P, "M") + [Driver("ca_tmax_month", "weather", wx[["tmax"]].rename(columns={"tmax": "*"}), MONTH + 3 * DAY, note="NOAA GHCN-D, 5 California airports, mean daily maximum"),
                                    Driver("ca_prcp_month", "weather", wx[["prcp"]].rename(columns={"prcp": "*"}), MONTH + 3 * DAY, note="NOAA GHCN-D, 5 California airports, total precipitation (mm)"),
                                    Driver("us_unemp", "economy", P.get("unemp").reindex(columns=["US"]).rename(columns={"US": "*"}), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                                    Driver("us_hicp", "economy", P.get("hicp").reindex(columns=["US"]).rename(columns={"US": "*"}), MONTH + EUROSTAT_DAYS["hicp"] * DAY)]
            fam = Family(fid, "Water demand: residential water use per person per day, California urban suppliers (State Water Board)", "water demand", "M", "gallons per capita per day", Y,
                         MONTH + 60 * DAY, 0, 3, (12,), drivers=D, notes="series 'rgpcd_south_coast:US' ... 'rgpcd_state:US'; population-weighted R-GPCD of 355 suppliers; data.ca.gov, ODC-BY; 2014-06-; latest vintage")
        elif fid == "spend_w":
            # ONS real-time indicators, UK spending on Revolut debit cards (Revolut; OGL 3.0): daily 7-day rolling average (index 100 = 2023 average, NSA) by sector and age and
            # the online share, weekly means (Monday-Sunday), 2020-01-. Released with the ONS real-time indicators about monthly (data to 11 Oct 2026 in the 15 Oct file): a week
            # counts as known 35 days after it ends.
            Y = _hsutc(HS.get_parquet("lab/raw/ons_revolut_weekly.parquet")).dropna(how="all")
            ccs = ["GB"]
            wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
            D = _markets(P, "D") + _society(P, "D", ccs) + [
                Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY), Driver("hdd", "weather", wk(P.get("hdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                Driver("cdd", "weather", wk(P.get("cdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
                Driver("hicp", "economy", P.get("hicp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["hicp"] * DAY), Driver("unemp", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                Driver("diesel", "fuel", P.get("diesel"), 4 * DAY), indeed("GB", "D")]
            fam = Family(fid, "Consumer spending: UK debit card spending by sector and age (Revolut via ONS)", "consumer spending", "W", "index (2023 average = 100)", Y, WEEK + 35 * DAY, 0, 4, (52,),
                         drivers=D, notes="series 'groceries:GB', 'restaurants:GB', 'travel:GB', '18_34:GB', 'online:GB' ...; weekly mean of a 7-day rolling daily index; Revolut customers (a sample, not the population); ONS real-time indicators, OGL 3.0; 2020-01-; latest vintage")
        else:
            if fid == "transit_d":
                # MTA daily ridership (subway, bus, LIRR, Metro-North, Access-A-Ride, Staten Island Railway) and bridge and tunnel crossings, data.ny.gov, 2020-03-;
                # the latest day was 2 days back on 24 Oct 2026: 2 days + 6 h of delay.
                Y, cc, lag, nm = _hsutc(HS.get_parquet("lab/raw/mta_daily.parquet")).dropna(how="all"), "US", 2 * DAY + 6, "MTA daily ridership and traffic, New York (data.ny.gov)"
                note = "series 'subway:US', 'bus:US', 'lirr:US', 'mnr:US', 'bridges_tunnels:US', 'access_a_ride:US', 'staten_island_rail:US'; 2020-03-; no usage limits stated; estimates revised for late data (latest vintage)"
                layer, unit = "transit", "trips per day"
            elif fid == "c311_d":
                # NYC 311 service requests per day by agency (NYC Open Data: no usage restrictions), 2010-; the next day's file is complete enough: 2 days of delay.
                x = _hsutc(HS.get_parquet("lab/raw/nyc311_daily.parquet"))
                keep = [c for c in x.columns if x[c].notna().sum() >= 6100]
                Y, cc, lag, nm = x[keep].dropna(how="all"), "US", 2 * DAY, "NYC 311 service requests per day by agency (NYC Open Data)"
                Y.columns = [f"{c.lower()}:US" for c in Y.columns]
                note = f"series 'nypd:US', 'hpd:US', 'dsny:US' ...; {len(keep)} agencies with a complete daily history; NYC Open Data (no usage restrictions), 2010-; latest vintage"
                layer, unit = "city services", "requests per day"
            elif fid == "traffic_d":
                # Norwegian Public Roads Administration (Statens vegvesen) traffic registration points, daily vehicle volume, NLOD 2.0 (commercial use allowed): county indices (mean of up to
                # 10 fixed points' volumes relative to each point's 2017-2018 level) and the national index; the Traffic Data API shows the day after it ends: 2 days of delay.
                Y, cc, lag, nm = _hsutc(HS.get_parquet("lab/raw/npra_traffic_daily.parquet")).dropna(how="all"), "NO", 2 * DAY, "Road traffic: daily vehicle volume by county, Norway (Statens vegvesen)"
                note = f"series 'akershus:NO' ... 'all:NO'; {Y.shape[1] - 1} counties, index of fixed counting points (100 = their 2017-2018 level); NPRA Traffic Data API, NLOD 2.0, 2017-; days with too few covered points are missing; latest vintage"
                layer, unit = "road traffic", "vehicles per day"
            ccs = [cc]
            D = _markets(P, "D") + _society(P, "D", ccs) + [Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1), indeed(cc, "D") if cc != "NO" else None,
                                                             Driver("diesel", "fuel", P.get("diesel"), 4 * DAY)]
            D = [d for d in D if d is not None]
            if fid == "traffic_d":
                ow = _hsutc(HS.get_parquet("lab/raw/oslo_weather.parquet"))
                D += [Driver(f"oslo_{k}", "weather", ow[[k]].rename(columns={k: "*"}), DAY + 24, note="NOAA GHCN-D Oslo Blindern (public domain)") for k in ("tmax", "tmin", "prcp", "snwd")]
            if cc == "US":
                wx = _hsutc(HS.get_parquet("lab/raw/nyc_weather.parquet"))
                D += [Driver(f"nyc_{k}", "weather", wx[[k]].rename(columns={k: "*"}), DAY + 24, note="NOAA GHCN-D Central Park (public domain)") for k in ("tmax", "tmin", "prcp", "snow", "snwd")]
            else:
                D += [Driver("temp", "weather", P.get("temp").reindex(columns=ccs), DAY + 6), Driver("hdd", "weather", P.get("hdd").reindex(columns=ccs), DAY + 6),
                      Driver("cdd", "weather", P.get("cdd").reindex(columns=ccs), DAY + 6),
                      Driver("unemp_m", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY)]
            fam = Family(fid, nm, layer, "D", unit, Y, lag, 0, 14, (7, 365), drivers=D, notes=note, span_min=max(3, Y.shape[1] // 2))
    elif fid in ("mort_w", "bike_d", "reg_d", "agri_w", "agricom_m", "quake_w", "ev_m", "arxiv_m"):
        # layers.
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        eu_mean = lambda k: P.get(k)[[c for c in EU27 if c in P.get(k).columns]].mean(axis=1).to_frame("*")   # noqa: E731
        if fid == "mort_w":
            # deaths per ISO week: Netherlands (CBS, by age band) and Belgium (Statbel, by region and age band), 2010-. CBS showed week 40 on 16 Oct (12 days after the week);
            # Statbel's daily file ended on the Sunday 20 days before 24 Oct: the target is known 21 days after the week.
            Y = _hsutc(HS.get_parquet("lab/raw/mort_weekly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2010-01-04", tz="UTC")]
            ccs = ["NL", "BE"]
            D = _markets(P, "D") + [Driver("temp", "weather", wk(P.get("temp").reindex(columns=ccs)).mean(), WEEK + 6 * DAY), Driver("hdd", "weather", wk(P.get("hdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY),
                                    Driver("cdd", "weather", wk(P.get("cdd").reindex(columns=ccs)).mean(), WEEK + 6 * DAY), Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
                                    Driver("unemp_m", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY)]
            from forecasting_engine.lab import src_health as SH
            for k, x in SH.positivity().items():            # ERVISS weekly laboratory positivity (health layer, from 2022-06): the cause of the winter excess
                D.append(Driver(f"pos_{k}", "health", _hsutc(x).reindex(columns=ccs), WEEK + 5 * DAY, note="ECDC ERVISS test positivity %, primary-care sentinel (health layer)"))
            ili = _hsutc(HS.get_parquet("lab/raw/health_ervis.parquet"))
            D.append(Driver("ili", "health", ili.reindex(columns=ccs), WEEK + 5 * DAY, note="ECDC ERVISS ILI / ARI consultation rate (health layer, from 2022-06)"))
            fam = Family(fid, "Mortality: deaths per week, Netherlands and Belgium (CBS, Statbel)", "mortality", "W", "deaths per week", Y, WEEK + 21 * DAY, 0, 4, (52,), drivers=D,
                         notes="series 'nl_total:NL', 'nl_80p:NL', 'be_flanders:BE', 'be_85p:BE' ...; ISO weeks (CBS part weeks at the turn of the year added up); CBS CC BY 4.0 and Statbel open data (CC BY 4.0); 2010-; latest vintage (recent weeks are provisional)",
                         span_min=5)
        elif fid == "ev_m":
            # Statistics Norway table 14020 (CC BY 4.0): vehicles registered for the first time per month by registration type, fuel and vehicle, 2008-; published in the first days of the
            # next month (September 2026 was in on 24 Oct): month + 6 days.
            Y = _hsutc(HS.get_parquet("lab/raw/ssb_vehicles_monthly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2008-01-01", tz="UTC")]
            ow = _hsutc(HS.get_parquet("lab/raw/oslo_weather.parquet"))
            fx = P.get("fx")
            D = _markets(P, "M") + [Driver("nok_per_eur", "markets", _monthly_mean(_g(fx["NOK"])), MONTH + 15 * DAY, note="ECB reference rate, monthly mean (imported vehicles are priced in EUR)"),
                                    Driver("oslo_tmax_month", "weather", _monthly_mean(ow[["tmax"]].rename(columns={"tmax": "*"})), MONTH + 3 * DAY, note="NOAA GHCN-D Oslo Blindern, monthly mean of the daily maximum")]
            fam = Family(fid, "Vehicle electrification: first registrations per month by fuel, Norway (Statistics Norway)", "vehicle electrification", "M", "vehicles per month", Y, MONTH + 6 * DAY, 0, 3, (12,),
                         drivers=D, notes="series 'car_new_electric:NO', 'car_new_fossil:NO', 'van_new_electric:NO', 'car_used_electric:NO' ...; new and imported second-hand vehicles by fuel (electric, fossil, hybrid); SSB 14020, CC BY 4.0; 2008-",
                         span_min=max(4, Y.shape[1] // 2))
        elif fid == "arxiv_m":
            # arXiv API (metadata CC0): new submissions per month by category, by the submission date of the first version, 2012-; a month is final the day after it ends.
            Y = _hsutc(HS.get_parquet("lab/raw/arxiv_monthly.parquet")).dropna(how="all")
            D = _markets(P, "M")
            fam = Family(fid, "Academic preprints: arXiv submissions per month by category", "academic preprints", "M", "submissions per month", Y, MONTH + 2 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'cs_ai:WW', 'cs_lg:WW', 'math_all:WW', 'astro_ph:WW', 'hep_th:WW', 'q_bio:WW', 'econ:WW' ...; counts of the category as primary or cross-list; arXiv API, metadata CC0; 2012-; latest vintage",
                         span_min=max(4, Y.shape[1] // 2))
        elif fid == "bike_d":
            # Norwegian Public Roads Administration bicycle counting points, daily counts (NLOD 2.0), county indices (100 = the points' 2018-2019 mean), 2018-; 2 days of delay.
            Y = _hsutc(HS.get_parquet("lab/raw/npra_bike_daily.parquet")).dropna(how="all")
            ccs = ["NO"]
            ow = _hsutc(HS.get_parquet("lab/raw/oslo_weather.parquet"))
            tr = _hsutc(HS.get_parquet("lab/raw/npra_traffic_daily.parquet"))[["all:NO"]].rename(columns={"all:NO": "*"})
            D = _markets(P, "D") + _society(P, "D", ccs) + [Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1)]
            D += [Driver(f"oslo_{k}", "weather", ow[[k]].rename(columns={k: "*"}), DAY + 24, note="NOAA GHCN-D Oslo Blindern (public domain)") for k in ("tmax", "tmin", "prcp", "snwd")]
            D += [Driver("npra_traffic", "road traffic", tr, 2 * DAY, note="road traffic layer (traffic_d): national vehicle volume index, Norway; cars and bikes share weather and holidays")]
            fam = Family(fid, "Cycling: daily bicycle counts by county, Norway (Statens vegvesen)", "cycling", "D", "index (2018-19 mean = 100)", Y, 2 * DAY, 0, 14, (7, 365), drivers=D,
                         notes=f"series 'oslo:NO', 'akershus:NO' ... 'all:NO'; {Y.shape[1] - 1} counties, mean of fixed bicycle counting points relative to each point's own 2018-2019 mean; NPRA Traffic Data API, NLOD 2.0; latest vintage; days with too few points are missing",
                         span_min=max(3, Y.shape[1] // 2))
        elif fid == "reg_d":
            # RDW (Netherlands) register of licensed vehicles, open data CC0: vehicles per day of first admission by type, 2010-; counts are complete at the end of the day, 3 days of delay.
            Y = _hsutc(HS.get_parquet("lab/raw/rdw_registrations_daily.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2010-01-01", tz="UTC")]
            ccs = ["NL"]
            D = _markets(P, "D") + _society(P, "D", ccs) + [
                Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1), Driver("diesel", "fuel", P.get("diesel").reindex(columns=ccs), 4 * DAY),
                Driver("temp", "weather", P.get("temp").reindex(columns=ccs), DAY + 6), Driver("hdd", "weather", P.get("hdd").reindex(columns=ccs), DAY + 6),
                Driver("cdd", "weather", P.get("cdd").reindex(columns=ccs), DAY + 6),
                Driver("unemp_m", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY), Driver("hicp_m", "economy", P.get("hicp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
                Driver("consumer_conf", "economy", P.get("consumer_conf").reindex(columns=ccs), MONTH), Driver("industry_conf", "economy", P.get("industry_conf").reindex(columns=ccs), MONTH),
                _indeed_driver("NL", "D")]
            fam = Family(fid, "Vehicle registrations: vehicles first admitted per day by type, Netherlands (RDW)", "vehicle registrations", "D", "vehicles per day", Y, 3 * DAY, 0, 14, (7, 365), drivers=D,
                         notes="series 'car:NL', 'van:NL', 'motorcycle:NL', 'moped:NL', 'bus:NL', 'tractor:NL', 'semitrailer:NL', 'trailer:NL'; date of first admission (new vehicles; used imports keep their old date); RDW open data, CC0; 2010-",
                         span_min=4)
        elif fid == "agri_w":
            # EC Agri-food data portal weekly prices (Commission reuse, CC BY 4.0): one series per product, mean over member states, markets and stages; 2016-. The week to Sunday 18 Oct
            # was published on Thursday 22 Oct (the portal's reference date): 4 days; 5 days used.
            Y = _hsutc(HS.get_parquet("lab/raw/agri_weekly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2016-01-04", tz="UTC")]
            Y = Y[[c for c in Y.columns if Y[c].notna().mean() >= 0.8 and Y[c].last_valid_index() >= Y.index[-1] - pd.Timedelta(days=28)]]   # oilseeds, poultry and some dairy start in 2020-21; discontinued grades end
            bf = [c for c in Y.columns if c.startswith("beef_")]          # 38 beef categories and grades move together: the 8 best covered
            Y = Y.drop(columns=[c for c in bf if c not in sorted(bf, key=lambda c: -Y[c].notna().mean())[:8]])
            pk = _hsutc(HS.get_parquet("lab/raw/pink_monthly.parquet"))
            D = _markets(P, "D") + [Driver("temp_eu", "weather", wk(eu_mean("temp")).mean(), WEEK + 6 * DAY, note="mean temperature of the EU27 countries (ERA5)"),
                                    Driver("cdd_eu", "weather", wk(eu_mean("cdd")).mean(), WEEK + 6 * DAY),
                                    Driver("diesel_eu", "fuel", eu_mean("diesel"), 4 * DAY, note="mean EU diesel pump price (Oil Bulletin)"),
                                    Driver("hicp_eu", "economy", _g(P.get("hicp").reindex(columns=list(EU27)).mean(axis=1)), MONTH + EUROSTAT_DAYS["hicp"] * DAY)]
            for k in ("urea", "dap", "potassium_chloride", "wheat_us_hrw", "maize", "soybeans", "palm_oil"):
                D.append(Driver(f"wb_{k}", "agricom", _g(pk[f"{k}:WW"].dropna()), MONTH + 3 * DAY,
                                note="World Bank Pink Sheet monthly price (agricultural commodities and fertiliser layer, agricom_m): input costs and world grain prices feed EU farm-gate prices"))
            fam = Family(fid, "Agri-food prices: EU weekly prices of cereals, oilseeds, pigmeat, beef, poultry, dairy and wine (EC Agri-food data portal)", "agri-food prices", "W", "EUR per tonne (or 100 kg)", Y,
                         WEEK + 5 * DAY, 0, 4, (52,), drivers=D, notes="series 'cereal_milling_wheat:EU', 'pigmeat_...:EU', 'dairy_butter:EU' ...; mean over member states and markets; Commission reuse (CC BY 4.0); 2016-; latest vintage",
                         span_min=max(3, Y.shape[1] // 2))
        elif fid == "agricom_m":
            # World Bank Commodity Price Data (Pink Sheet), monthly nominal USD, agriculture and fertilisers, 2000- (CC BY 4.0); the 2 Oct 2026 release held September: month + 3 days.
            Y = _hsutc(HS.get_parquet("lab/raw/pink_monthly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2000-01-01", tz="UTC")]
            Y = Y.drop(columns=[c for c in Y.columns if c.split(":")[0] in ("tea_colombo", "tea_kolkata", "tea_mombasa", "rice_thai_25", "rice_thai_a_1", "banana_us", "rubber_rss3", "sawnwood_malaysian",
                                                                                "logs_malaysian", "palm_kernel_oil", "groundnut_oil", "coconut_oil", "fish_meal", "shrimps_mexican")])
            ag = _hsutc(HS.get_parquet("lab/raw/agri_weekly.parquet"))
            agm = ag.resample("MS").mean().where(ag.resample("MS").count() >= 3)
            D = _markets(P, "M") + [Driver("hicp_eu", "economy", _g(P.get("hicp").reindex(columns=list(EU27)).mean(axis=1)), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
                                    Driver("unemp_eu", "economy", _g(P.get("unemp").reindex(columns=list(EU27)).mean(axis=1)), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                                    Driver("temp_eu_month", "weather", _monthly_mean(eu_mean("temp")), MONTH + DAY), Driver("esi_eu", "economy", _g(P.get("esi").reindex(columns=list(EU27)).mean(axis=1)), MONTH)]
            for k in [c for c in agm.columns if c.split(":")[0] in ("cereal_milling_wheat", "cereal_feed_barley", "cereal_feed_maize", "oilseeds_rapeseed_n_a", "pigmeat_e", "dairy_butter", "dairy_smp")] or list(agm.columns)[:4]:
                D.append(Driver(f"eu_{k.split(':')[0]}", "agri-food prices", _g(agm[k].dropna()), MONTH + 5 * DAY, note="EC Agri-food data portal weekly price (agri_w), monthly mean of 3+ weeks"))
            fam = Family(fid, "Agricultural commodities and fertiliser: World Bank monthly prices of food, oils, grains, softs, meat, cotton, rubber, timber and fertilisers", "agricultural commodities", "M", "USD (as published per unit)", Y,
                         MONTH + 3 * DAY, 0, 3, (12,), drivers=D, notes="series 'urea:WW', 'dap:WW', 'wheat_us_hrw:WW', 'cocoa:WW' ...; nominal USD; World Bank Commodity Markets, CC BY 4.0; 2000-; energy and metals series left out (other layers)",
                         span_min=max(5, Y.shape[1] // 2))
        else:
            # USGS ComCat earthquakes M4.5+ (public domain), weekly counts 2000-; a week is catalogued within a day or two: 2 days of delay. Almost no outside driver is plausible: markets only.
            Y = _hsutc(HS.get_parquet("lab/raw/usgs_quakes_weekly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2000-01-03", tz="UTC")]
            D = _markets(P, "D")
            fam = Family(fid, "Earthquakes: events per week, magnitude 4.5+, world and three regions (USGS)", "earthquakes", "W", "events per week", Y, WEEK + 2 * DAY, 0, 4, (52,), drivers=D,
                         notes="series 'm45:WW', 'm55:WW', 'm65:WW', 'americas:WW', 'asia_pacific:WW', 'med_europe:WW'; USGS ComCat, public domain; clustered (aftershocks); latest vintage", span_min=3)
    elif fid in ("births_m", "fire911_w", "sec_w", "library_m", "launch_w"):
        # layers.
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        if fid == "births_m":
            # live births per day (the month's births divided by its days): Netherlands (CBS, national and 12 provinces), Sweden (SCB, national and 21 counties), Finland (Statistics Finland).
            # CBS and SCB had August on 24 Oct (54 days after the month): the target is known 60 days after the month. 2000-; latest vintage (the newest months are preliminary).
            Y = _hsutc(HS.get_parquet("lab/raw/births_monthly.parquet")).dropna(how="all")
            Y = Y[Y.notna().mean(axis=1) >= 0.5]                       # the newest month of the fastest source (Finland) alone is left out
            ccs = ["NL", "SE", "FI"]
            D = _markets(P, "M") + [Driver("unemp_m", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                                    Driver("hicp_m", "economy", P.get("hicp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
                                    Driver("consumer_conf", "economy", P.get("consumer_conf").reindex(columns=ccs), MONTH), Driver("industry_conf", "economy", P.get("industry_conf").reindex(columns=ccs), MONTH),
                                    Driver("stringency", "health", _monthly_mean(P.get("stringency").reindex(columns=ccs)), MONTH + DAY, note="Oxford COVID-19 stringency index, monthly mean (the 2020-22 births dip)")]
            fam = Family(fid, "Births: live births per day, Netherlands, Sweden and Finland (CBS, SCB, Statistics Finland)", "births", "M", "births per day", Y, MONTH + 60 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'nl_total:NL', 'nl_pv20:NL' (12 provinces), 'se_total:SE', 'se_c01:SE' (21 counties), 'fi_total:FI'; births in the month divided by its days; CBS CC BY 4.0, SCB CC0, Statistics Finland CC BY 4.0; 2000- (provinces 2002-); latest vintage",
                         span_min=12)
        elif fid == "fire911_w":
            # Seattle Fire Department 911 dispatches per week by call group (data.seattle.gov, public domain dedication), 2003-11-; the local Pacific date; a week is complete on Monday: 1 day of delay.
            Y = _hsutc(HS.get_parquet("lab/raw/seattle_fire_weekly.parquet")).dropna(how="all")
            sw = _hsutc(HS.get_parquet("lab/raw/seattle_weather.parquet"))
            D = _markets(P, "D") + [Driver("sea_tmax", "weather", wk(sw[["tmax"]].rename(columns={"tmax": "*"})).mean(), WEEK + 24, note="NOAA GHCN-D Seattle-Tacoma, weekly mean of the daily maximum (public domain)"),
                                    Driver("sea_tmin", "weather", wk(sw[["tmin"]].rename(columns={"tmin": "*"})).mean(), WEEK + 24),
                                    Driver("sea_prcp", "weather", wk(sw[["prcp"]].rename(columns={"prcp": "*"})).sum(min_count=5), WEEK + 24, note="weekly precipitation, mm")]
            D += _society(P, "D", ["US"])
            fam = Family(fid, "Emergency calls: Seattle Fire Department 911 dispatches per week by call group (City of Seattle)", "emergency calls", "W", "dispatches per week", Y, WEEK + DAY, 0, 4, (52,), drivers=D,
                         notes="series 'total:US', 'aid:US', 'medic:US', 'alarm:US', 'mva:US', 'rescue:US', 'fire:US', 'other:US'; call types grouped by name; local Pacific date; City of Seattle open data, public domain dedication; 2003-11-; latest vintage",
                         span_min=4)
        elif fid == "sec_w":
            # SEC EDGAR filings per week by form family, date filed (Eastern), 2006-; US government work, public domain. The full index of a day is complete the next morning: 1 day of delay.
            Y = _hsutc(HS.get_parquet("lab/raw/edgar_weekly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2010-01-04", tz="UTC")]       # electronic Form D (2009) and 13F-HR (2013, then growing) make earlier weeks incomparable
            D = _markets(P, "D") + _society(P, "D", ["US"]) + [_indeed_driver("US", "D")]
            fam = Family(fid, "Corporate disclosure: SEC EDGAR filings per week by form family (SEC)", "corporate disclosure", "W", "filings per week", Y, WEEK + DAY, 0, 4, (52,), drivers=D,
                         notes="series 'total:US', '8k:US', '10k:US', '10q:US', 'insider:US', 'ownership:US', 'registration:US', 'prospectus:US', 'foreign:US', 'funds:US', 'late:US', 'private:US' ...; date filed; EDGAR full index, public domain; 2006-; loan-level ABS, fund and correspondence forms left out",
                         span_min=max(4, Y.shape[1] // 2))
        elif fid == "library_m":
            # Seattle Public Library checkouts per month by material type (data.seattle.gov, public domain dedication), 2005-04-; August 2026 was the newest on 24 Oct: month + 60 days.
            Y = _hsutc(HS.get_parquet("lab/raw/seattle_library_monthly.parquet")).dropna(how="all")
            sw = _hsutc(HS.get_parquet("lab/raw/seattle_weather.parquet"))
            D = _markets(P, "M") + [Driver("sea_tmax_month", "weather", _monthly_mean(sw[["tmax"]].rename(columns={"tmax": "*"})), MONTH + 3 * DAY, note="NOAA GHCN-D Seattle-Tacoma, monthly mean of the daily maximum"),
                                    Driver("sea_prcp_month", "weather", sw[["prcp"]].rename(columns={"prcp": "*"}).resample("MS").sum(min_count=20), MONTH + 3 * DAY, note="monthly precipitation, mm")]
            fam = Family(fid, "Library loans: Seattle Public Library checkouts per month by material type (City of Seattle)", "library loans", "M", "checkouts per month", Y, MONTH + 60 * DAY, 0, 3, (12,), drivers=D,
                         notes="series 'total:US', 'book:US', 'ebook:US', 'audiobook:US', 'videodisc:US', 'sounddisc:US', 'use_digital:US', 'use_physical:US'; City of Seattle open data, public domain dedication; 2005-04-; latest vintage",
                         span_min=max(3, Y.shape[1] // 2))
        else:
            # GCAT (J. McDowell, CC BY 4.0): orbital and deep-space launch attempts and payloads put in orbit per week, 1998-; catalogued within days: 5 days of delay.
            Y = _hsutc(HS.get_parquet("lab/raw/launches_weekly.parquet")).dropna(how="all")
            D = _markets(P, "D")
            fam = Family(fid, "Space launches: orbital launch attempts and payloads orbited per week (GCAT)", "space launches", "W", "per week", Y, WEEK + 5 * DAY, 0, 4, (52,), drivers=D,
                         notes="series 'all:WW', 'us:WW', 'china:WW', 'russia:WW', 'spacex:WW', 'pay_all:WW', 'pay_starlink:WW' ...; GCAT, CC BY 4.0; 1998-; latest vintage (late catalogued objects are added)",
                         span_min=4)
    elif fid == "hmrc_m":
        # HMRC tax receipts and NICs, monthly GBP million, April 2017- (OGL 3.0): income tax (PAYE, self-assessment), NICs, VAT, corporation tax, capital gains, stamp duties, fuel, alcohol,
        # tobacco, gambling, air passenger and insurance premium duties, customs; published about the 20th of the next month (September 2026 on 21 Oct): month + 25 days.
        Y = _hsutc(HS.get_parquet("lab/raw/hmrc_receipts.parquet")).dropna(how="all")
        ccs = ["GB"]
        sp = _hsutc(HS.get_parquet("lab/raw/ons_revolut_weekly.parquet"))[["total:GB", "groceries:GB", "restaurants:GB"]].resample("MS").mean()
        iv = _hsutc(HS.get_parquet("lab/raw/uk_insolv.parquet"))
        D = _markets(P, "M") + _society(P, "M", ccs) + [
            Driver("unemp", "economy", P.get("unemp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["unemp"] * DAY), Driver("hicp", "economy", P.get("hicp").reindex(columns=ccs), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
            Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY), Driver("temp_month", "weather", _monthly_mean(P.get("temp").reindex(columns=ccs)), MONTH + DAY),
            _indeed_driver("GB", "M"),
            Driver("spend_total", "economy", sp[["total:GB"]].rename(columns={"total:GB": "GB"}), MONTH + 35 * DAY, note="spend_w: UK debit card spending index (Revolut via ONS), monthly mean; weeks known 35 days late"),
            Driver("spend_groceries", "economy", sp[["groceries:GB"]].rename(columns={"groceries:GB": "GB"}), MONTH + 35 * DAY),
            Driver("insolv_companies", "insolvency", iv[["co_total:GB"]].rename(columns={"co_total:GB": "GB"}), MONTH + 25 * DAY, note="insolv_m: company insolvencies (Insolvency Service)")]
        fam = Family(fid, "Tax receipts: HMRC receipts by tax, UK (HMRC)", "tax receipts", "M", "GBP million", Y, MONTH + 25 * DAY, 0, 3, (12,), drivers=D,
                     notes="series 'vat:GB', 'paye:GB', 'corporation:GB', 'stamp_land:GB' ...; HMRC tax and NICs receipts, OGL 3.0, 2017-04-; first-release values are revised a little (latest vintage); spend_w and insolv_m enter as drivers",
                     span_min=10)
    elif fid in ("rasff_w", "ted_w"):
        # EU-wide weekly counts. rasff_w: RASFF Window food and feed safety notifications per week (total, by classification,
        # product category and country of origin), validated by the Commission, 2012-; known the next day (a Monday-Sunday week: 2 days after it ends). ted_w: TED public
        # procurement notices per week (total, competition and result notices, works / services / supplies, buyer country), 2017-; 3 days after the week.
        wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        eu = lambda df: df[[c for c in ("EU27_2020", "EU") if c in df.columns][:1]].rename(columns=lambda c: "*")   # noqa: E731
        if fid == "rasff_w":
            Y = _hsutc(HS.get_parquet("lab/raw/rasff_weekly.parquet")).dropna(how="all")
            Y = Y[Y.index >= pd.Timestamp("2020-01-06", tz="UTC")]      # the public window holds under 10 notifications a week before 2019-12 (the old portal's restored few): 2020- only
            lag, nm, layer = WEEK + 2 * DAY, "Food safety: RASFF notifications per week (EC RASFF Window)", "food safety"
            note = "series 'total:EU', 'cls_alert:EU', 'cat_fruits_and_vegetables:EU', 'origin_cn:EU' ...; weekly counts by validation date, 2020- (the public window is nearly empty before 2019-12); EC reuse notice (Decision 2011/833/EU), CC BY 4.0 on the EU Open Data Portal; latest vintage"
        else:
            Y = _hsutc(HS.get_parquet("lab/raw/ted_weekly.parquet")).dropna(how="all")
            lag, nm, layer = WEEK + 3 * DAY, "Public procurement: TED notices per week (EU Publications Office)", "public procurement"
            note = "series 'total:EU', 'competition:EU', 'result:EU', 'works:EU', 'de:EU' ...; weekly counts by publication date, 2017-; the eForms change of Oct 2023-Jan 2024 moves some form counts; TED, Decision 2011/833/EU, CC BY 4.0; latest vintage"
        tm = P.get("temp")[[c for c in EU27 if c in P.get("temp").columns]].mean(axis=1).to_frame("*")
        cd = P.get("cdd")[[c for c in EU27 if c in P.get("cdd").columns]].mean(axis=1).to_frame("*")
        D = _markets(P, "D") + [Driver("temp_eu", "weather", wk(tm).mean(), WEEK + 6 * DAY, note="mean temperature of the EU27 countries (ERA5)"),
                                Driver("cdd_eu", "weather", wk(cd).mean(), WEEK + 6 * DAY, note="mean cooling degree-days of the EU27 countries"),
                                Driver("stringency", "health", P.get("stringency")[[c for c in EU27 if c in P.get("stringency").columns]].mean(axis=1).to_frame("*"), DAY + 1),
                                Driver("diesel", "fuel", P.get("diesel")[[c for c in EU27 if c in P.get("diesel").columns]].mean(axis=1).to_frame("*"), 4 * DAY)]
        D += [Driver(f"{k}_eu", "economy", eu(P.get(k)), av) for k, av in (("unemp", MONTH + EUROSTAT_DAYS["unemp"] * DAY), ("hicp", MONTH + EUROSTAT_DAYS["hicp"] * DAY), ("ip", MONTH + 45 * DAY),
                                                                           ("retail", MONTH + 35 * DAY), ("esi", MONTH), ("consumer_conf", MONTH), ("industry_conf", MONTH))
              if any(c in P.get(k).columns for c in ("EU27_2020", "EU"))]
        fam = Family(fid, nm, layer, "W", "notifications per week" if fid == "rasff_w" else "notices per week", Y, lag, 0, 4, (52,), drivers=D, notes=note, span_min=max(3, Y.shape[1] // 2))
    elif fid == "regulation_w":
        # regulation flow: EU legal acts adopted per week by type (Publications Office Cellar, EUR-Lex reuse licence), 2011-:
        # regulations, implementing and delegated regulations, decisions, implementing decisions, directives; dated by the act's document date;
        # published in the Official Journal and indexed days to weeks later: a week is known 28 days after it ends
        x = HS.get_parquet("lab/raw/eurlex_daily.parquet")
        x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
        x = x[x.index >= pd.Timestamp("2011-01-03", tz="UTC")]
        Y = x.resample("W-MON", label="left", closed="left").sum()
        Y = Y.iloc[:-1]
        Y.columns = [f"{c}:EU" for c in Y.columns]
        D = _markets(P, "D") + [Driver("hicp_ea", "economy", _g(P.get("hicp").reindex(columns=list(EU27)).mean(axis=1)), MONTH + EUROSTAT_DAYS["hicp"] * DAY),
                                Driver("unemp_ea", "economy", _g(P.get("unemp").reindex(columns=list(EU27)).mean(axis=1)), MONTH + EUROSTAT_DAYS["unemp"] * DAY),
                                Driver("strike_eu", "society", _g(P.get("strike").reindex(columns=list(EU27)).sum(axis=1)), DAY + 1),
                                Driver("protest_eu", "society", _g(P.get("protest").reindex(columns=list(EU27)).sum(axis=1)), DAY + 1),
                                Driver("tone_eu", "society", _g(P.get("tone").reindex(columns=list(EU27)).mean(axis=1)), DAY + 1)]
        fam = Family(fid, "Regulation flow: EU legal acts adopted per week by type (EUR-Lex / Cellar)", "regulation", "W", "acts/week", Y, WEEK + 28 * DAY, 0, 4, (52,), drivers=D,
                     notes="series 'REG_IMPL:EU' etc. (REG regulation, REG_IMPL implementing, REG_DEL delegated, DEC decision, DEC_IMPL implementing decision, DIR directive); global drivers on '*'; latest Cellar vintage")
    elif fid == "choke_w":
        # transport and logistics layer: weekly vessel transits through the world's chokepoints (IMF PortWatch, AIS; Kpler and the
        # UN Global Platform), Monday-Sunday sums over complete weeks. PortWatch updates on Tuesdays with data up to about the Friday before, so a
        # week is complete in the data about 9 days after it ends: 10 days of delay (WEEK + 10 days from the Monday stamp)
        c = HS.get_parquet("lab/raw/portwatch_chokepoints_daily.parquet").reset_index()
        c["date"] = pd.to_datetime(c["date"], utc=True)
        nn = c.pivot_table(index="date", columns="portname", values="n_total", aggfunc="sum").sort_index()
        wkr = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
        full = wkr(nn.notna()).sum() >= 7
        Y = wkr(nn).sum().where(full).dropna(how="all")
        Y = Y[[k for k in Y.columns if Y[k].median() >= 50]]               # drop the near-empty straits (Kerch, Bering, Magellan...)
        from forecasting_engine.lab import src_x2 as SX2
        D = _markets(P, "D")
        cs = SX2.gdelt_conflict_share()
        if cs is not None:
            D.append(Driver("gdelt_conflict_7d", "society", cs, DAY + 1, note="GDELT: global share of material-conflict events, 7-day mean"))
        D += [Driver("diesel_eu", "fuel", _g(P.get("diesel").mean(axis=1)), 4 * DAY, note="EU Weekly Oil Bulletin diesel, mean of countries"),
              Driver("hicp_ea_proxy", "economy", _g(P.get("hicp").mean(axis=1)), MONTH + EUROSTAT_DAYS["hicp"] * DAY, note="mean HICP inflation of the euro-area countries")]
        fam = Family(fid, "Maritime chokepoints: weekly vessel transits (IMF PortWatch)", "transport", "W", "ships/week", Y, WEEK + 10 * DAY, 0, 4, (52,),
                     drivers=D, notes="28 straits and canals filtered to those with a median of 50 or more transits a week; complete Monday-Sunday weeks; global drivers matched on '*'")
    elif fid == "aq_d":
        # environment / health layer: measured daily PM2.5 and NO2 in each EU capital (EEA E1a verified + E2a up-to-date,
        # CC BY 4.0), urban background stations within 20 km; most countries published within 1-3 days (6-12 for a few): 3 days of delay
        def rd(n):
            x = HS.get_parquet(f"lab/raw/{n}.parquet")
            x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
            return x.astype(float)
        Y = pd.concat({"pm25": rd("aq_pm25"), "no2": rd("aq_no2")}, axis=1)
        Y.columns = [f"{a}:{b}" for a, b in Y.columns]
        ccs = sorted({c.split(":")[1] for c in Y.columns})
        air = HS.get_parquet("lab/raw/air_traffic.parquet")
        D = [Driver("temp", "weather", P.get("temp").reindex(columns=ccs), DAY + 6), Driver("hdd", "weather", P.get("hdd").reindex(columns=ccs), DAY + 6),
             Driver("cdd", "weather", P.get("cdd").reindex(columns=ccs), DAY + 6),
             Driver("stringency", "health", P.get("stringency").reindex(columns=ccs), DAY + 1),
             Driver("flights", "transport", air.reindex(columns=ccs), 46 * DAY, note="EUROCONTROL (back-tests only)"),
             Driver("diesel", "fuel", P.get("diesel"), 4 * DAY), Driver("ip", "economy", P.get("ip"), MONTH + 45 * DAY)] + _society(P, "D", ccs)
        g = HS.get_parquet("lab/raw/gfs_country.parquet")
        if g is not None:                                     # GFS wind and temperature forecasts (8 countries, 2021-): dispersion
            g = g.reset_index()
            g["valid"] = pd.to_datetime(g["valid"], utc=True)
            for var in ("ws100", "t2m"):
                x = g[(g["var"] == var) & (g["lead"] == "d1")].pivot_table(index="valid", columns="cc", values="value").sort_index().resample("D").mean()
                D.append(Driver(f"fc_{var}_d1", "weather", x, 0, ahead=True, dayahead_hour=5, note="NOAA GFS forecast, daily mean"))
        fam = Family(fid, "Air quality: daily PM2.5 and NO2 in EU capitals (EEA, measured)", "environment", "D", "ug/m3", Y, 3 * DAY, 0, 7, (7, 364),
                     drivers=D, notes="series 'pm25:FR' / 'no2:FR'; urban background stations; some countries end 2025-12-31 (no E2a)")
    elif fid in ("hicp_ea_m", "ecb_w"):
        # euro-area inflation (EA aggregate, its components and the member states, up to 26 months ahead: the horizons of the SPF, the ECB staff
        # projections and the Commission forecasts) and the ECB deposit facility rate decided each week, up to a year ahead (8 meetings).
        # Inputs: forecasting_engine.lab.src_ecb (Eurostat prc_hicp_minr, ECB Data Portal, ECB decision pages). hicp_m is untouched.
        hx = HS.get_parquet("lab/raw/estat_hicp_ea.parquet").reset_index().pivot_table(index="date", columns="id", values="value").sort_index()
        hx.index = pd.DatetimeIndex(hx.index)
        comps = ("total", "core", "energy", "food", "services")
        cols = lambda sfx: {f"{c}:EA": hx[f"{c}_{sfx}"] for c in comps} | {f"total:{cc}": hx[f"{cc}_{sfx}"] for cc in EU27 if f"{cc}_{sfx}" in hx}  # noqa: E731
        rel = lambda df: df.set_axis(df.index + pd.DateOffset(months=1))   # noqa: E731  (reference month m stamped at the 1st of month m+1)
        yoy, mom = pd.DataFrame(cols("yoy")), pd.DataFrame(cols("mom"))
        dfr = HS.get_parquet("lab/raw/ecb_dfr_daily.parquet")["dfr"]
        dfr.index = pd.DatetimeIndex(dfr.index)
        yc = HS.get_parquet("lab/raw/ecb_yc.parquet").reset_index().pivot_table(index="date", columns="id", values="value").sort_index()
        yc.index = pd.DatetimeIndex(yc.index)
        spread = yc.sub(dfr.reindex(yc.index, method="ffill"), axis=0)            # market-implied change of the policy rate
        rd_ea = lambda k, col: _g(P.get(k)[col]) if col in P.get(k).columns else None   # noqa: E731
        fuel = lambda k: _g(P.get(k).reindex(columns=list(EU27)).mean(axis=1))   # noqa: E731
        common = [Driver("esi_ea", "economy", rd_ea("esi", "EA20"), MONTH, note="Eurostat ESI, euro area (end of the month)"),
                  Driver("consumer_conf_ea", "economy", rd_ea("consumer_conf", "EA20"), MONTH),
                  Driver("industry_conf_ea", "economy", rd_ea("industry_conf", "EA20"), MONTH),
                  Driver("unemp_ea", "economy", rd_ea("unemp", "EA21"), MONTH + EUROSTAT_DAYS["unemp"] * DAY, note="Eurostat unemployment, euro area"),
                  Driver("diesel_eu", "fuel", fuel("diesel"), 4 * DAY), Driver("gasoline_eu", "fuel", fuel("gasoline"), 4 * DAY),
                  Driver("strike_eu", "society", _g(P.get("strike").reindex(columns=list(EU27)).sum(axis=1)), DAY + 1),
                  Driver("protest_eu", "society", _g(P.get("protest").reindex(columns=list(EU27)).sum(axis=1)), DAY + 1),
                  Driver("tone_eu", "society", _g(P.get("tone").reindex(columns=list(EU27)).mean(axis=1)), DAY + 1),
                  Driver("stringency_eu", "health", _g(P.get("stringency").reindex(columns=list(EU27)).mean(axis=1)), DAY + 1)]
        common = [d for d in common if d.data is not None]
        if fid == "hicp_ea_m":
            Y = rel(yoy)
            M_ = rel(mom)
            D = _markets(P, "M") + [Driver(f"{d.name}_month", d.group, _monthly_mean(d.data), d.avail_h + MONTH) if d.avail_h < MONTH else d
                                    for d in common] + [
                Driver("mom", "economy", M_, 20 * DAY, note="monthly change of each series (same release)"),
                Driver("base_effect", "economy", M_.shift(12, freq="MS"), 20 * DAY - 12 * MONTH, ahead=True,
                       note="the monthly change 12 months before each target month (it leaves the y/y rate)"),
                Driver("dfr_month", "policy", _monthly_mean(_g(dfr)), MONTH, note="ECB deposit facility rate, monthly mean"),
                Driver("yc_1y_month", "markets", _monthly_mean(_g(yc["yc_1y"])), MONTH + DAY),
                Driver("yc_2y_month", "markets", _monthly_mean(_g(yc["yc_2y"])), MONTH + DAY),
                Driver("diesel_month", "fuel", _monthly_mean(P.get("diesel")), MONTH + 4 * DAY),
                Driver("gasoline_month", "fuel", _monthly_mean(P.get("gasoline")), MONTH + 4 * DAY),
                Driver("hdd_month", "weather", _monthly_mean(P.get("hdd")), MONTH + DAY)]
            fam = Family(fid, "Euro-area inflation: HICP y/y, components and member states (Eurostat)", "economy", "M", "%", Y, 20 * DAY, 0, 26,
                         (12,), drivers=D, notes="reference month m stamped at the 1st of month m+1 (the release month; final release ~17-20th): "
                         "the origin of the 1st forecasts the month just ended and 25 more; series 'total:EA', 'core:EA', ..., 'total:FR'")
        else:
            # the week's row s (a Monday) holds the deposit facility rate in force on Wednesday s+2: the rate decided at a meeting in the week
            # BEFORE s (decisions on Thursday, in force the next Wednesday), known from the Thursday 14:15 CET press release (s - 83 h). The
            # Monday origin t therefore forecasts at step 1 the decision of its own week, up to 52 weeks ahead.
            wk = pd.date_range(pd.Timestamp("1999-01-04", tz="UTC"), dfr.index.max() - pd.Timedelta(days=2), freq="W-MON")
            Y = pd.DataFrame({"dfr:EA": dfr.reindex(wk + pd.Timedelta(days=2)).to_numpy()}, index=wk)
            mt = HS.get_parquet("lab/raw/ecb_meetings.parquet")
            sched = pd.DatetimeIndex(mt.index[mt["scheduled"]])
            cal_wk = (sched - pd.to_timedelta(sched.dayofweek, unit="D")).normalize() + pd.Timedelta(days=7)   # the row whose step it decides
            full = pd.date_range(wk[0], cal_wk.max() + pd.Timedelta(days=7), freq="W-MON")
            meet = pd.DataFrame({"*": full.isin(cal_wk).astype(float)}, index=full)
            hy = rel(yoy)
            D = _markets(P, "D") + common + [
                Driver("meeting", "policy", meet, -365 * DAY, ahead=True, note="a scheduled monetary-policy meeting decides this week's step "
                       "(the ECB calendar is published more than a year ahead; ad hoc meetings excluded)"),
                Driver("euribor3m", "markets", _g(P.get("ecb_rates")["FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"].dropna()), MONTH + DAY)]
            one = lambda df, c: _g(df[c].dropna())   # noqa: E731  (one column per driver: the family has a single series)
            D += [Driver(c, "markets", one(yc, c), DAY + 12, note="euro-area AAA government spot yield (ECB)") for c in yc.columns]
            D += [Driver(f"{c}_minus_dfr", "markets", one(spread, c), DAY + 12, note="yield minus the rate in force: the market-implied path")
                  for c in spread.columns]
            D += [Driver(f"hicp_{c.split(':')[0]}_ea", "economy", one(hy, c), 20 * DAY, note="euro-area HICP y/y (Eurostat)")
                  for c in ("total:EA", "core:EA", "energy:EA", "services:EA")]
            D += [Driver(f"hicp_{c.split(':')[0]}_ea_mom", "economy", one(rel(mom), c), 20 * DAY) for c in ("total:EA", "core:EA")]
            g = HS.get_parquet("lab/chain/hicp_ea_m.parquet")
            if g is not None:                                  # the lab's own euro-area inflation forecast, as made at each origin
                g = g.reset_index()
                g = g[g["series"].isin(["total:EA", "core:EA"])]
                g["origin"] = pd.to_datetime(g["origin"], utc=True)
                for k in (1, 6, 12):
                    for c in ("total:EA", "core:EA"):
                        x = g[(g["step"] == k) & (g["series"] == c)].set_index("origin")["value"].sort_index()
                        D.append(Driver(f"chain_hicp_{c.split(':')[0]}_ea_k{k}", "chain", _g(x), 0,
                                        note=f"the lab's hicp_ea_m forecast ({g['method'].iloc[0]}), step {k}, made at its origin"))
            fam = Family(fid, "ECB deposit facility rate decided each week (ECB)", "policy", "W", "%", Y, -83.0, 0, 52, (52,), drivers=D,
                         notes="row s = rate in force on s+2 days = decision of the week before s; step 1 from a Monday origin = that week's meeting")
    elif fid == "kev_d":
        D = [d for d in _markets(P, "D") if d.name != "kev_new"]
        fam = Family(fid, "New exploited vulnerabilities (CISA KEV)", "cyber", "D", "per day", P.get("kev"), DAY, 0, 7, (7,), drivers=D)
    else:
        raise KeyError(fid)
    fam.Y = fam.Y.dropna(how="all").sort_index()
    fam.drivers = _audit(fid, fam.freq, fam.drivers)                 # publication-lag audit, 20 Oct 2026 (audit_lag)
    return fam


def extra_drivers(fid: str, P: Pool) -> list[Driver]:
    """_extra_drivers with the publication-lag audit of 20 Oct 2026 applied (audit_lag)."""
    return _audit(fid, "M" if fid.endswith("_m") else "", _extra_drivers(fid, P))


def _extra_drivers(fid: str, P: Pool) -> list[Driver]:
    """Drivers added after the registry: leading economic surveys, producer prices, COVID-19 policy stringency (health layer)."""
    M = fid.endswith("_m")
    if fid.endswith("_long_m"):                              # the health layer for the OECD families (ISO3 series ids)
        from forecasting_engine.lab.sources import ISO3
        st = _monthly_mean(P.get("stringency"))
        st = st.rename(columns={v: k for k, v in ISO3.items()})
        D0 = [Driver("stringency", "health", st, MONTH, note="Oxford COVID-19 Government Response Tracker (CC BY 4.0), monthly mean")]
        if fid in ("ip_long_m", "cpi_long_m"):                # surveys, trade, energy / freight / commodity indices
            from forecasting_engine.lab import src_x4 as SX4
            D0 += SX4.drivers(fid, P, sys.modules[__name__])
            from forecasting_engine.lab import src_x5 as SX5          # US / Asia manufacturing cycle, exchange rates, US prices
            D0 += SX5.drivers(fid, P, sys.modules[__name__])
        return D0
    D = [Driver("stringency", "health", _monthly_mean(P.get("stringency")) if M else P.get("stringency"), (MONTH if M else DAY + 1),
                note="Oxford COVID-19 Government Response Tracker (CC BY 4.0)")]
    if M or fid in ("unrest_w", "diesel_w", "gasoline_w"):
        D += [Driver("esi", "economy", P.get("esi"), MONTH, note="Eurostat economic sentiment (surveys published at the end of the month)"),
              Driver("consumer_conf", "economy", P.get("consumer_conf"), MONTH), Driver("industry_conf", "economy", P.get("industry_conf"), MONTH),
              Driver("ppi", "economy", P.get("ppi"), MONTH + 35 * DAY)]
    if fid in ("price_h", "load_h", "res_h", "load_d", "hicp_m", "ip_m", "cpi_long_m", "ip_long_m"):   # Russian pipeline gas (2021-)
        g = HS.get_parquet("lab/raw/entsog_russian_gas.parquet")
        g.index = pd.DatetimeIndex(g.index).tz_convert("UTC") if pd.DatetimeIndex(g.index).tz is not None else pd.DatetimeIndex(g.index).tz_localize("UTC")
        g = g.assign(total=g.sum(axis=1, min_count=1))
        agg = _monthly_mean if fid.endswith("_m") else (lambda x: x)
        av = MONTH if fid.endswith("_m") else DAY + 12
        D += [Driver(f"ru_gas_{c}", "energy", agg(g[[c]].rename(columns={c: "*"})), av, note="ENTSOG physical flow, GWh/d (data from 2021)")
              for c in ("total", "nord_stream", "ukraine_transit", "yamal", "turkstream")]
    if fid in ("price_h", "load_h", "res_h", "load_d"):       # GFS weather forecasts (2021-), known before the target day
        g = HS.get_parquet("lab/raw/gfs_country.parquet")
        if g is not None:
            g = g.reset_index()
            g["valid"] = pd.to_datetime(g["valid"], utc=True)
            for var in ("ws100", "ws100_cube", "dswrf", "t2m"):
                for lead, dh in (("d1", 5), ("d2", -19)):   # run of the day before (05 UTC) / two days before: known floor(t) - 19 h / - 43 h
                    x = g[(g["var"] == var) & (g["lead"] == lead)].pivot_table(index="valid", columns="cc", values="value").sort_index()
                    x = x.resample("h").interpolate(limit=2)
                    if fid == "load_d":
                        x = x.resample("D").mean()
                    D.append(Driver(f"fc_{var}_{lead}", "weather", x, 0, ahead=True, dayahead_hour=dh,
                                    note="NOAA GFS 0.5 deg forecast, country box mean (public domain)"))
    if fid in ("price_h", "load_h", "res_h", "load_d"):       # coupled engine E1: the operators' day-ahead wind + solar forecast (14.1.d)
        rd = _res_da(P, daily=fid == "load_d")
        for k, x in rd.items():                               # published by 18:00 Brussels the day before: known floor(t) - 7 h (17 UTC)
            D.append(Driver(f"res_da_{k}", "energy", x, 0, ahead=True, dayahead_hour=17,
                            note="ENTSO-E day-ahead generation forecast, wind and solar (14.1.d; not confirmed on the free re-use list: back-tests only)"))
        D += _network(fid, P, rd)                             # coupled engine E2: neighbours, weighted by cross-border flows
        D += _chain(fid)                                      # coupled engine E3: other families' own forecasts (lab.chain)
    if fid in ("diesel_w", "gasoline_w"):                     # local currency and the tax component of the pump price
        prod = "diesel" if fid == "diesel_w" else "gasoline95"
        tax = pd.DataFrame({cc: HS.oil_series(cc, prod, "with") - HS.oil_series(cc, prod, "without") for cc in EU27}).sort_index()
        cur = {"PL": "PLN", "CZ": "CZK", "HU": "HUF", "RO": "RON", "SE": "SEK", "DK": "DKK", "BG": "BGN"}
        fx = P.get("fx")
        fxl = pd.DataFrame({cc: fx[c] for cc, c in cur.items() if c in fx.columns})
        D += [Driver("tax_component", "policy", tax, 4 * DAY, note="Oil Bulletin: price with taxes minus without (EUR/l)"),
              Driver("local_fx", "markets", fxl, 15, note="ECB: local currency per EUR (non-euro countries)")]
    if fid in ("hicp_m", "ports_m", "tourism_m"):             # coupled engine E6: weekly / daily forecasts (fuel, unrest, air traffic) into monthly families
        D += _chain(fid)
    if fid == "hicp_m":                                       # base effects
        idx, nrg = P.get("hicp_index"), P.get("hicp_energy_index")
        mom = idx.pct_change(fill_method=None) * 100
        D += [Driver("hicp_mom", "economy", mom, MONTH + 17 * DAY, note="monthly change of the HICP index"),
              Driver("base_effect", "economy", mom.shift(12, freq="MS"), MONTH + 17 * DAY - 12 * MONTH, ahead=True,
                     note="the monthly change 12 months before each month: it leaves the y/y rate that month (known a year ahead)"),
              Driver("energy_mom", "energy", nrg.pct_change(fill_method=None) * 100, MONTH + 17 * DAY, note="HICP energy, monthly change"),
              Driver("energy_yoy", "energy", nrg.pct_change(12, fill_method=None) * 100, MONTH + 17 * DAY, note="HICP energy, y/y")]
    if fid in ("hicp_m", "tourism_m", "health_w"):            # mixed-frequency nowcasting (the published part of the month / week)
        D += _nowcast(fid, P)
    if fid == "health_w":                                     # virology, hospitals and neighbour spread (same ERVISS release)
        D += _health_r63()
    if fid == "health_w":                                     # Wikipedia reading (covers the week ERVISS has not published yet), RKI ARE by age
        D += _health_n1()
    if fid == "commod_d":                                     # US inventories (EIA), futures positioning (CFTC), product cracks
        D += _commod_n1(P)
    if fid == "ip_m":                                         # energy use leads output (load, industrial gas, the load_d chain)
        D += _ip_r63(P) + _chain(fid)
    if fid in ("ports_m", "wildfire_w"):                      # leading inputs (lab.src_leading): AIS port calls / chokepoints, fire danger
        from forecasting_engine.lab import src_leading as SL
        if fid == "ports_m":                                  # complete months only, known one month + 12 days after the month start
            av, nt = MONTH + SL.PW_LAG_DAYS * DAY, "IMF PortWatch (AIS; Kpler, UN Global Platform), 2019-; commercial reuse needs IMF permission"
            for m, nm in (("portcalls", "pw_calls"), ("portcalls_cargo", "pw_cargo_calls"), ("import", "pw_import_t"), ("export", "pw_export_t")):
                x = SL.ports_monthly(m)
                if x is not None:
                    D.append(Driver(nm, "trade", x, av, note=f"{nt}; Spain's ports, monthly sum of {m}"))
            ck = SL.chokepoints_monthly("capacity")
            if ck is not None:
                D += [Driver(f"ck_{k}_t", "trade", ck[[k]].rename(columns={k: "*"}), av, note=f"{nt}; {k} transit capacity, t per month")
                      for k in ("suez", "bab_el_mandeb", "cape", "gibraltar", "panama", "hormuz") if k in ck.columns]
        else:                                                 # computed from NASA POWER weather; a Mon-Sun week known on the Thursday after it
            for nm, (ix, how) in (("fwi_w", ("fwi", "mean")), ("fwi_max_w", ("fwi", "max")), ("dc_w", ("dc", "mean"))):
                x = SL.fwi_weekly(ix, how)
                if x is not None:
                    D.append(Driver(nm, "environment", x, WEEK + SL.FWI_LAG_DAYS * DAY,
                                    note=f"Canadian Fire Weather Index system ({ix}, weekly {how}; Van Wagner 1987) from NASA POWER daily weather"))
    if fid in ("commod_d", "fx_d", "diesel_w", "gasoline_w"):  # products, positioning, policy rates
        D += _mktfuel_m1(fid, P)
    if fid == "health_w":                                     # pre-2022 seasons as climatology, THL Finland pathogens
        from forecasting_engine.lab import src_health_long as SHX
        D += [Driver(nm, "health", x, av, ahead=ah, note=nt) for nm, (x, av, ah, nt) in SHX.features().items()]
        from forecasting_engine.lab import src_health_open as SHO    # open pre-2022 national histories (CH, England, Scotland)
        D += [Driver(nm, "health", x, av, ahead=ah, note=nt) for nm, (x, av, ah, nt) in SHO.features().items()]
    if fid == "tourism_m":                                    # national early releases (ES, PT, FI)
        from forecasting_engine.lab import src_tourism_nat as STN
        D += [Driver(nm, "tourism", x, av, note=nt) for nm, (x, av, nt) in STN.features().items()]
    if fid in ("unrest_w", "security_w", "info_w"):          # GDELT actors / roots / quad classes, holidays
        from forecasting_engine.lab import src_society as SSO
        D += SSO.drivers(fid, EU27 + ("GB", "UA", "TR") + (("RU", "IL", "EG") if fid != "unrest_w" else ()))
    if fid == "kev_d":                                        # KEV by vendor, NVD CVE counts, Patch Tuesday
        from forecasting_engine.lab import src_cyber as SCY
        D += SCY.drivers()
    if fid in ("water_w", "wildfire_w", "aq_d"):              # basin rain, active fires, capital weather
        D += _envdrivers_e1(fid, P)
    if fid in ("unrest_w", "security_w", "info_w", "kev_d", "wildfire_w", "ports_m", "choke_w"):   # cross-layer links
        from forecasting_engine.lab import src_x2 as SX2
        D += SX2.drivers(fid, P, sys.modules[__name__])
    if fid == "wildfire_w":                                   # vapour pressure deficit, dry run, drought index
        from forecasting_engine.lab import src_x5 as SX5
        D += SX5.drivers(fid, P, sys.modules[__name__])
    if fid in ("health_w", "choke_w", "ip_long_m", "cpi_long_m", "diesel_w"):   # cross-layer links
        from forecasting_engine.lab import src_x4 as SX4
        D += SX4.drivers(fid, P, sys.modules[__name__])
    if fid in ("health_w", "unrest_w", "info_w", "tourism_m", "unemp_m", "hicp_m", "air_d"):   # new layers as drivers
        from forecasting_engine.lab import src_nl as SNL
        D += SNL.drivers(fid, P, sys.modules[__name__])
    if fid in ("ip_m", "retail_m", "unemp_m", "asylum_m"):      # new layers as drivers
        from forecasting_engine.lab import src_nl2d as SN2
        D += SN2.drivers(fid, P, sys.modules[__name__])
    if fid in ("choke_w", "wildfire_w", "cpi_long_m"):           # upstream ports, ignition calendar, supply chains, housing, labour
        from forecasting_engine.lab import src_x6 as SX6
        D += SX6.drivers(fid, P, sys.modules[__name__])
    if fid in ("retail_m", "ip_m", "housing_m", "credit_m", "ports_m", "transit_d", "c311_d", "constr_m", "insolv_m"):   # links
        from forecasting_engine.lab import src_nl3d as SN3
        D += SN3.drivers(fid, P, sys.modules[__name__])
    if fid in ("milk_m", "hicp_ea_m", "unrest_w", "traffic_d", "wiki_w", "mort_w", "ports_m"):   # links
        from forecasting_engine.lab import src_nl4d as SN4
        D += SN4.drivers(fid, P, sys.modules[__name__])
    if fid in ("firms_m", "insolv_m", "retail_m"):                # links
        from forecasting_engine.lab import src_nl5d as SN5
        D += SN5.drivers(fid, P, sys.modules[__name__])
    if fid in ("ip_m", "retail_m", "unemp_m", "hicp_m", "ports_m"):   # EC surveys, registrations /
        from forecasting_engine.lab import src_econ as SEC                    # bankruptcies, construction, services / transport turnover, trade volumes,
        D += SEC.drivers(fid)                                           # import prices (Eurostat)
        if fid in ("retail_m", "unemp_m", "ports_m"):                   # energy use as an activity proxy (the ip_m round-63 inputs, reused)
            D += _ip_r63(P)
        D += SEC.drivers2(fid)                                          # ECB credit growth, borrowing costs, deposits (finance layer)
        D += SEC.drivers3(fid)                                          # x3: consumers' savings expectations (after the delay correction)
    return D

def _envdrivers_e1(fid: str, P: Pool) -> list[Driver]:
    """Water: rain over each basin (inflow) next to the human draw (tourism nights) and the drought code. Wildfire: NASA FIRMS
    active fires (the fires burning now, a week before EFFIS maps them) and school holidays (people light most fires). Air quality: wind, rain,
    humidity at the capital (dispersion and washout) and school holidays (traffic). Appended after every earlier driver."""
    from forecasting_engine.lab import src_envdrivers as SE
    from forecasting_engine.lab import src_leading as SL
    from forecasting_engine.lab import src_nowcast as SN
    D = []
    pw = "NASA POWER daily (MERRA-2 / GEOS-IT), mean of 2-5 points in the basin's reservoir zone"
    if fid == "water_w":
        av = WEEK + SE.POWER_LAG_DAYS * DAY
        for nm, x, nt in (("basin_rain_w", SE.basin_weekly("rain", 1), "weekly rain, mm"), ("basin_rain_4w", SE.basin_weekly("rain", 4), "4-week rain, mm"),
                          ("basin_rain_13w", SE.basin_weekly("rain", 13), "13-week rain, mm"),
                          ("basin_rain_anom_13w", SE.basin_rain_anomaly(13), "13-week rain minus the same weeks of the previous years (causal), mm"),
                          ("basin_rain_anom_52w", SE.basin_rain_anomaly(52), "52-week rain anomaly (causal), mm"),
                          ("basin_t2m_4w", SE.basin_weekly("t2m", 4), "4-week mean 2 m temperature (evaporation, irrigation demand)")):
            if x is not None:
                D.append(Driver(nm, "weather", x, av, note=f"{pw}; {nt}"))
        dc = SL.fwi_weekly("dc", "mean")
        if dc is not None:
            D.append(Driver("drought_code_es", "environment", dc[["ES"]], WEEK + SL.FWI_LAG_DAYS * DAY,
                            note="Canadian FWI drought code for Spain (deep soil dryness), weekly mean, from NASA POWER"))
        tn = HS.get_parquet("lab/raw/tourism_nights.parquet")
        if tn is not None and "ES" in tn.columns:
            D.append(Driver("tourism_nights_es", "tourism", tn[["ES"]], MONTH + 73 * DAY, note="Eurostat tour_occ_nim nights in Spain (water demand)"))
    elif fid == "wildfire_w":
        av = WEEK + SE.FIRMS_LAG_DAYS * DAY
        for nm, v in (("firms_n_w", "n"), ("firms_frp_w", "frp")):
            x = SE.firms_weekly(v)
            if x is not None:
                D.append(Driver(nm, "environment", np.log1p(x), av,
                                note=f"NASA FIRMS MODIS C6.1 vegetation-fire detections ({v}), weekly, log1p; archive 2005-2025 (2026 not in the archive)"))
    elif fid == "aq_d":
        for nm, v in (("cap_wind", "ws10"), ("cap_rain", "rain"), ("cap_rh", "rh"), ("cap_t2m", "t2m")):
            x = SE.capital_daily(v)
            if x is not None:
                D.append(Driver(nm, "weather", x, DAY + SE.POWER_LAG_DAYS * DAY, note=f"NASA POWER daily {v} at the capital (dispersion / washout)"))
    if fid in ("wildfire_w", "aq_d"):
        sch = HS.get_parquet("lab/raw/school_holidays.parquet")
        if sch is not None:
            sch.index = pd.DatetimeIndex(sch.index).tz_convert("UTC") if pd.DatetimeIndex(sch.index).tz is not None else pd.DatetimeIndex(sch.index).tz_localize("UTC")
            agg = sch.resample("W-MON", label="left", closed="left").mean() if fid == "wildfire_w" else sch
            D.append(Driver("school_holiday", "society", agg, -SN.SCHOOL_AHEAD_DAYS * DAY, ahead=True,
                            note="share of the country's regions on school holiday in the target period (OpenHolidays API, ODbL; 23 countries, 2020-)"))
    if fid in ("water_w", "wildfire_w"):                      # e2: rolling sums stamped DAILY, so the week EFFIS / the weekly sums have not
        rs = SE.daily_rolling("rain" if fid == "water_w" else "firms_n", (7, 30) if fid == "water_w" else (7, 14))   # published yet is seen
        for k, x in rs.items():
            nm = f"basin_rain_roll{k}d" if fid == "water_w" else f"firms_n_roll{k}d"
            lag = SE.POWER_LAG_DAYS if fid == "water_w" else SE.FIRMS_LAG_DAYS
            D.append(Driver(nm, "weather" if fid == "water_w" else "environment", x, DAY + lag * DAY,
                            note=f"{k}-day sum ending on the stamped day ({'NASA POWER rain per basin' if fid == 'water_w' else 'NASA FIRMS MODIS detections, log1p'}), "
                                 f"known {lag:.0f} day(s) after the day ends"))
    if fid == "aq_d":                                         # cross-layer: fires in the country (smoke) into PM2.5
        f = HS.get_parquet("lab/raw/firms_modis_daily.parquet")
        if f is not None:
            f = f[[c for c in f.columns if c.startswith("frp:")]].astype(float)
            f.columns = [c.split(":", 1)[1] for c in f.columns]
            f.index = pd.DatetimeIndex(f.index).tz_convert("UTC") if pd.DatetimeIndex(f.index).tz is not None else pd.DatetimeIndex(f.index).tz_localize("UTC")
            D.append(Driver("firms_frp_d", "environment", np.log1p(f), DAY + SE.FIRMS_LAG_DAYS * DAY,
                            note="NASA FIRMS MODIS vegetation-fire radiative power in the country, daily, log1p (smoke); archive 2005-2025"))
    return D


def _mktfuel_m1(fid: str, P: Pool) -> list[Driver]:
    """commod_d gets refined-product cracks and more futures positioning; diesel_w / gasoline_w get the oil-market layer (EIA weekly
    inventories, CFTC positioning, product cracks and product prices in EUR/l); fx_d gets policy-rate differentials (BIS), currency-futures
    positioning (CFTC TFF) and the euro / US curves. Appended after every earlier driver (the champions keep their first N)."""
    from forecasting_engine.lab import src_commod as SC
    from forecasting_engine.lab import src_mktfuel as SM
    f = P.get("fred")
    rows = []
    if fid == "commod_d":
        rows = SM.commod_features(f) + SM.cot_features(("rbob", "ulsd"), ("mm", "swap", "prod")) + SM.cot_features(("wti", "brent", "ng"), ("swap", "prod"))
    elif fid in ("diesel_w", "gasoline_w"):
        rows = [(n, "energy", _g(x), av, nt) for n, (x, av, nt) in SC.eia_features().items()] + \
               [(n, "markets", _g(x), av, nt) for n, (x, av, nt) in SC.cot_features().items()] + \
               [(n, "fuel", _g(x), av, nt) for n, (x, av, nt) in SC.crack_features(f).items()] + \
               SM.commod_features(f) + SM.fuel_features(P.get("fx")) + SM.cot_features(("rbob", "ulsd"), ("mm",))
    elif fid == "fx_d":
        rows = SM.fx_features(f)
    return [Driver(n, g, x, av, note=nt) for n, g, x, av, nt in rows]


def _res_da(P: Pool, daily: bool = False) -> dict[str, pd.DataFrame]:
    """Day-ahead wind (onshore + offshore) and solar forecasts per country (MW), and their share of the day-ahead load forecast (%)."""
    w, so = {}, {}
    for cc in ENTSOE_CC:
        f = HS.entsoe_series("res_da", cc)
        if f is None or not len(f):
            continue
        f = f.resample("h").mean()
        wc = [c for c in f.columns if c.startswith("da_wind")]
        if wc:
            w[cc] = f[wc].sum(axis=1, min_count=1)
        if "da_solar" in f.columns:
            so[cc] = f["da_solar"]
    W, S = pd.DataFrame(w).sort_index(), pd.DataFrame(so).sort_index()
    L = P.get("load_da")
    sh = (W.add(S, fill_value=0) / L.reindex(W.index.union(S.index)) * 100).clip(0, 200)
    out = dict(wind=W, solar=S, share=sh)
    return {k: v.resample("D").mean() for k, v in out.items()} if daily else out


def flow_weights() -> dict[str, dict[str, float]]:
    """w[i][j]: the share of country i's cross-border exchange that goes to or comes from j (mean |flow| both ways, 2015-2017, before every
    development window), so a neighbour that trades more counts more. Each row sums to 1."""
    from forecasting_engine.histories import BORDERS
    raw: dict = {}
    for a, b in BORDERS:
        tot = 0.0
        for k in (f"{a}_{b}", f"{b}_{a}"):
            s = HS.entsoe_series("flows", k, "flow_mw")
            s = s[(s.index >= "2015-01-01") & (s.index < "2018-01-01")] if len(s) else s
            tot += float(s.abs().mean()) if len(s) else 0.0
        raw.setdefault(a, {})[b] = tot
        raw.setdefault(b, {})[a] = tot
    return {i: {j: v / sum(r.values()) for j, v in r.items()} for i, r in raw.items() if sum(r.values()) > 0}


def _nb(x: pd.DataFrame, W: dict) -> pd.DataFrame:
    """x*_i = sum_j w_ij x_j (GVAR-style foreign aggregate), only where every weighted neighbour has a value."""
    out = {}
    for i, r in W.items():
        cols = [j for j in r if j in x.columns]
        if not cols:
            continue
        ww = pd.Series({j: r[j] for j in cols})
        out[i] = (x[cols] * ww).sum(axis=1, min_count=len(cols)) / ww.sum()
    return pd.DataFrame(out).sort_index()


def _network(fid: str, P: Pool, rd: dict) -> list[Driver]:
    W = flow_weights()
    daily = fid == "load_d"
    agg = (lambda x: x.resample("D").mean()) if daily else (lambda x: x)
    D = [Driver("nb_load_da", "network", agg(_nb(P.get("load_da"), W)), 0, ahead=True, dayahead_hour=9,
                note="neighbours' day-ahead load forecast, weighted by cross-border flows"),
         Driver("nb_res_da_share", "network", _nb(rd["share"], W), 0, ahead=True, dayahead_hour=17,
                note="neighbours' day-ahead wind + solar share, weighted by cross-border flows"),
         Driver("nb_price", "network", agg(_nb(P.get("price"), W)), 24 if daily else 0, dayahead_hour=12,
                note="neighbours' day-ahead price (known at noon for the next day), weighted by cross-border flows")]
    if fid != "load_d":
        sh = P.get("gen")
        D.append(Driver("nb_wind_share", "network", _nb(sh["wind"], W), 2, note="neighbours' actual wind share, weighted"))
    g = HS.get_parquet("lab/raw/gfs_country.parquet")
    if g is not None:
        g = g.reset_index()
        g["valid"] = pd.to_datetime(g["valid"], utc=True)
        for var in ("ws100_cube", "dswrf", "t2m"):
            x = g[(g["var"] == var) & (g["lead"] == "d1")].pivot_table(index="valid", columns="cc", values="value").sort_index()
            x = x.resample("h").interpolate(limit=2)
            D.append(Driver(f"nb_fc_{var}_d1", "network", agg(_nb(x, W)), 0, ahead=True, dayahead_hour=5,
                            note="neighbours' GFS forecast (run of the day before), weighted by cross-border flows"))
    return D


def _kinds(g: pd.DataFrame):
    """A parent with several series per country (unrest_w: strike:FR, protest:FR) gives one driver per kind; older chain files (no `series`
    column) and one-series-per-country parents give one."""
    if "series" not in g.columns or (g["series"].astype(str) == g["cc"].astype(str)).all():
        yield "", g
        return
    kind = g["series"].astype(str).str.split(":", n=1).str[0]
    for k in sorted(kind.unique()):
        yield f"_{k}", g[kind == k]


def _chain(fid: str) -> list[Driver]:
    """Parents' forecasts as made at each origin (lab.chain build): lead d1 = from the origin of the day before the valid day, d2 = two days before."""
    D = []
    for parent in CHAIN_PARENTS.get(fid, ()):
        g = HS.get_parquet(f"lab/chain/{parent}.parquet")
        if g is None:
            continue
        g = g.reset_index()
        g["valid"] = pd.to_datetime(g["valid"], utc=True)
        if fid.endswith("_m") and "origin" in g.columns:       # E6: a weekly parent into a monthly child (nowcast of the origin's month)
            D += [_chain_monthly(fid, parent, gk, sfx) for sfx, gk in _kinds(g)]
            continue
        oh = int(g["origin_hour"].iloc[0])
        for lead, dh in (("d1", oh), ("d2", oh - 24)):
            if fid == "load_d" and lead == "d2":               # d2 covers only the first hours of the day: no daily mean
                continue
            x = g[g["lead"] == lead].pivot_table(index="valid", columns="cc", values="value").sort_index()
            if fid == "load_d":                                 # the daily mean of the hourly forecast, only for (nearly) complete days
                n = x.resample("D").count()
                x = x.resample("D").mean().where(n >= 20)
            D.append(Driver(f"chain_{parent}_{lead}", "chain", x, 0, ahead=True, dayahead_hour=dh,
                            note=f"the lab's own {parent} forecast ({g['method'].iloc[0]}), made at the origin of the day(s) before"))
    return D


def _chain_monthly(fid: str, parent: str, g: pd.DataFrame, sfx: str = "") -> Driver:
    """For each month m and country: the mean of the parent's forecasts for the weeks starting in m, all taken from the parent's LAST origin
    strictly before m starts (so the whole path was known 1 h before the month began: avail -1 h at the month stamp)."""
    g = g.copy()
    g["origin"] = pd.to_datetime(g["origin"], utc=True)
    g["month"] = g["valid"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    g = g[g["origin"] < g["month"] - pd.Timedelta(hours=1)]
    last = g.groupby(["month", "cc"])["origin"].transform("max")
    g = g[g["origin"] == last]
    x = g.groupby(["month", "cc"])["value"].mean().unstack("cc").sort_index()
    return Driver(f"chain_{parent}{sfx}_path", "chain", x, -1, ahead=True,
                  note=f"mean of the lab's own {parent} forecasts ({g['method'].iloc[0]}) for the month's weeks, made before the month began")


def _mtd(x: pd.DataFrame, same_days: bool = True, base: str = "yoy", min_day: int = 1) -> pd.DataFrame:
    """Month-to-date mean at each stamp (the month's values up to and including it), as % change against the same month a year before (its
    same days when same_days, else its whole mean) or, base='mom', against the whole previous month. Stamps before day `min_day` of a month
    are dropped, so the previous month's complete value stays the latest known one until a few days of the new month are in."""
    x = x.sort_index()
    mk = x.index.tz_convert("UTC").tz_localize(None).to_period("M")
    cm = x.groupby(mk).cumsum() / x.notna().astype(float).groupby(mk).cumsum().replace(0, np.nan)
    mm = x.groupby(mk).mean()
    if base == "mom":
        ref = mm.reindex(mk - 1).set_axis(x.index)
    elif same_days:                                           # the same days of the month a year before (Feb 29 -> Feb 28)
        ly = (x.index - pd.DateOffset(years=1))
        ref = cm.reindex(ly).set_axis(x.index)
    else:
        ref = mm.reindex(mk - 12).set_axis(x.index)
    out = (cm / ref - 1) * 100
    return out[x.index.day >= min_day].replace([np.inf, -np.inf], np.nan).dropna(how="all")


DASH_FROM = pd.Timestamp("2020-04-01", tz="UTC")             # EUROCONTROL Daily Traffic Variation dashboard (D-1, every morning): from spring 2020


def _flights_mtd(air: pd.DataFrame) -> pd.DataFrame:
    """Latest flights y/y known, stamped WHEN KNOWN (avail 0). Before DASH_FROM only the monthly airport files existed: a whole month, y/y,
    known 46 days after it ends. From DASH_FROM the daily dashboard gives D-1: the month-to-date y/y of the same days, known 2 days after D
    (the fixed airport panel stands in for the dashboard's state totals: the y/y ratios of the two agree, levels do not)."""
    m = _mtd(air.resample("D").sum(min_count=1), same_days=True, min_day=4)
    m = m[m.index >= DASH_FROM]
    m.index = m.index + pd.Timedelta(days=2)
    mo = air.resample("MS").mean()
    mo = (mo / mo.shift(12) - 1) * 100
    mo.index = mo.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=46)
    mo = mo[mo.index < DASH_FROM + pd.Timedelta(days=2)]
    return pd.concat([mo, m]).sort_index().replace([np.inf, -np.inf], np.nan)


def _nowcast(fid: str, P: Pool) -> list[Driver]:
    """mixed-frequency nowcasting for the layers that lose to Chronos-2. Monthly origins sit at 00:00 UTC on
    the 1st (core.ORIGIN_OFFSET_H has no entry for them; skip -1 = targets m, m+1, m+2), when the target is known only to m-2 (HICP, 17 days)
    or m-4 (tourism, 73 days): the month-to-date at the origin is the just-ended month m-1, from data published daily or weekly. health_w
    origins are Monday 00:00 of week p0 (target known to p0-2)."""
    from forecasting_engine.lab import src_nowcast as SN
    D = []
    if fid == "hicp_m":
        for k, nm in (("diesel", "diesel"), ("gasoline", "petrol")):   # Oil Bulletin Monday prices, + 4 days: m-1 up to its last Monday <= day 27
            x = P.get(k)
            D += [Driver(f"{nm}_mtd_yoy", "fuel", _mtd(x, same_days=False), 4 * DAY,
                         note="Oil Bulletin pump price, month-to-date mean of the Monday prices published so far vs the same month a year before, %"),
                  Driver(f"{nm}_mtd_mom", "fuel", _mtd(x, base="mom"), 4 * DAY,
                         note="Oil Bulletin pump price, month-to-date mean vs the whole previous month, %")]
        pr = P.get("price").resample("D").mean()                # day D's day-ahead prices are all known at 12:00 UTC on D-1
        D.append(Driver("power_mtd_yoy", "energy", _mtd(pr, same_days=True, min_day=4), 0, dayahead_hour=12,
                        note="ENTSO-E day-ahead price, month-to-date mean vs the same days a year before, % (8 countries; back-tests only)"))
        fn = SN.flash_news(MONTH + 17 * DAY)
        if fn is not None:                                     # the flash is out at the 1st 00:00 only when Eurostat releases it on M's last working day
            cc = [c for c in fn.columns if c in EU27]
            D += [Driver("hicp_flash_news", "economy", fn[cc], 0,
                         note="Eurostat flash y/y of the latest flashed month minus the first-published final y/y of the month before; 0 once the "
                              "final of the flashed month is out; stamped at the real release (euro area members, 2017-08-)"),
                  Driver("hicp_flash_news_ea", "economy", fn[["EA"]].rename(columns={"EA": "*"}), 0, note="the same for the euro area aggregate")]
    if fid == "tourism_m":
        air = HS.get_parquet("lab/raw/air_traffic.parquet")
        air.index = pd.DatetimeIndex(air.index).tz_convert("UTC") if pd.DatetimeIndex(air.index).tz is not None else pd.DatetimeIndex(air.index).tz_localize("UTC")
        D.append(Driver("flights_mtd_yoy", "transport", _flights_mtd(air), 0,
                        note="EUROCONTROL flights y/y, the latest known: month-to-date of the same days from the D-1 dashboard (2020-04-), "
                             "whole months 46 days late before; stamped when known (back-tests only)"))
    sch = HS.get_parquet("lab/raw/school_holidays.parquet")
    if sch is not None and fid in ("tourism_m", "health_w"):  # school calendars: set months to years ahead (ahead driver, known 180 days before)
        sch.index = pd.DatetimeIndex(sch.index).tz_convert("UTC") if pd.DatetimeIndex(sch.index).tz is not None else pd.DatetimeIndex(sch.index).tz_localize("UTC")
        agg = sch.resample("MS").mean() if fid == "tourism_m" else sch.resample("W-MON", label="left", closed="left").mean()
        D.append(Driver("school_holiday", "society", agg, -SN.SCHOOL_AHEAD_DAYS * DAY, ahead=True,
                        note="share of the country's regions on school holiday in the target period (OpenHolidays API, ODbL; 23 countries, 2020-)"))
    if fid == "health_w":
        ww = HS.get_parquet("lab/raw/wastewater_de.parquet")
        if ww is not None:                                     # weekly points stamped Wednesday, released the Tuesday after (8 days used)
            D += [Driver(f"wastewater_{k}", "health", ww[[k]].rename(columns={k: "DE"}), SN.AMELAG_LAG_DAYS * DAY,
                         note=f"RKI/UBA AMELAG wastewater virus load ({k}, population-weighted, normalised; Germany only; CC BY 4.0)")
                  for k in ww.columns]
    return D


# ------------------------------------------------------------------------------------------------------------------------------ round 63
# health_w neighbours: land borders between the 28 ERVISS countries plus fixed links and short sea crossings (DK-SE Oresund bridge, FI-EE and
# DK-NO / SE-DE ferries, IE-FR, MT-IT, CY-GR, GR-IT, IS-NO, IS-DK). Countries outside the panel (GB, CH, the Western Balkans, UA, TR) are not
# nodes. Each neighbour is weighted by its population (Eurostat 1 Jan 2024, millions, rounded), so a large neighbour counts more.
HEALTH_ADJ = {"AT": ("DE", "CZ", "SK", "HU", "SI", "IT"), "BE": ("FR", "NL", "DE", "LU", "IE"), "BG": ("RO", "GR"), "HR": ("SI", "HU"),
              "CY": ("GR",), "CZ": ("DE", "PL", "SK", "AT"), "DK": ("DE", "SE", "NO", "IS"), "EE": ("LV", "FI"), "FI": ("SE", "NO", "EE"),
              "FR": ("BE", "LU", "DE", "IT", "ES", "IE"), "DE": ("DK", "PL", "CZ", "AT", "FR", "LU", "BE", "NL", "SE"), "GR": ("BG", "CY", "IT"),
              "HU": ("AT", "SK", "RO", "HR", "SI"), "IS": ("NO", "DK"), "IE": ("FR", "BE"), "IT": ("FR", "AT", "SI", "MT", "GR"), "LV": ("EE", "LT"),
              "LT": ("LV", "PL"), "LU": ("BE", "FR", "DE"), "MT": ("IT",), "NL": ("BE", "DE"), "NO": ("SE", "FI", "DK", "IS"),
              "PL": ("DE", "CZ", "SK", "LT"), "PT": ("ES",), "RO": ("HU", "BG"), "SK": ("CZ", "PL", "HU", "AT"), "SI": ("IT", "AT", "HU", "HR"),
              "ES": ("FR", "PT"), "SE": ("NO", "FI", "DK", "DE")}
POP_M = {"AT": 9.2, "BE": 11.8, "BG": 6.4, "CY": 0.9, "CZ": 10.9, "DE": 83.4, "DK": 6.0, "EE": 1.4, "ES": 48.6, "FI": 5.6, "FR": 68.4, "GR": 10.4,
         "HR": 3.9, "HU": 9.6, "IE": 5.3, "IS": 0.4, "IT": 59.0, "LT": 2.9, "LU": 0.7, "LV": 1.9, "MT": 0.6, "NL": 17.9, "NO": 5.6, "PL": 36.6,
         "PT": 10.6, "RO": 19.1, "SI": 2.1, "SK": 5.4, "SE": 10.6}


def health_weights() -> dict[str, dict[str, float]]:
    """w[i][j]: neighbour j's population share among i's neighbours (HEALTH_ADJ is symmetric; each row sums to 1)."""
    return {i: {j: POP_M[j] / sum(POP_M[k] for k in nb) for j in nb} for i, nb in HEALTH_ADJ.items()}


def _nb_avail(x: pd.DataFrame, W: dict) -> pd.DataFrame:
    """Like _nb, but over the neighbours that reported that week (weights renormalised); NaN when none did."""
    out = {}
    for i, r in W.items():
        cols = [j for j in r if j in x.columns]
        if not cols:
            continue
        ww = pd.Series({j: r[j] for j in cols})
        m = x[cols].notna()
        out[i] = (x[cols].fillna(0) * ww).sum(axis=1) / (m * ww).sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(out).sort_index()


def _live_cols(x: pd.DataFrame, since: str = "2024-03-01", min_share: float = 0.1) -> pd.DataFrame:
    """Drop the countries that (nearly) stopped reporting: the lab reads a driver's latest value however old it is, so a series that ended in
    2024 would be carried into every later origin."""
    c = x[x.index >= pd.Timestamp(since, tz="UTC")].notna().mean()
    return x[[k for k in x.columns if c.get(k, 0) >= min_share]]


def _health_r63() -> list[Driver]:
    """ERVISS laboratory positivity per pathogen and SARI hospital rates, the EU-wide positivity,
    and the neighbour-spread of consultations and positivity. All from the same weekly ERVISS release as the target: week + 5 days."""
    from forecasting_engine.lab import src_health as SH
    av, D = WEEK + 5 * DAY, []
    nt = "ECDC ERVISS (reuse with acknowledgement, see src_health.ERVISS_ATTR)"
    pos = {k: _live_cols(x) for k, x in SH.positivity().items()}
    v = HS.get_parquet("lab/raw/health_ervis_viro.parquet").reset_index()
    v["date"] = pd.to_datetime(v["date"], utc=True)
    W = health_weights()
    for p, x in pos.items():
        D.append(Driver(f"pos_{p}", "health", x, av, note=f"{nt}: {p} test positivity %, primary-care sentinel (else other laboratories), >= 10 tests"))
        g = v[v["pathogen"] == p].groupby("date")[["detections", "tests"]].sum()
        D.append(Driver(f"eu_pos_{p}", "health", _g((g["detections"] / g["tests"] * 100).where(g["tests"] >= 50)),
                        av, note=f"{nt}: {p} positivity pooled over every reporting country and laboratory, %"))
        D.append(Driver(f"nb_pos_{p}", "network", _nb_avail(x, W), av,
                        note=f"{nt}: neighbours' {p} positivity, population-weighted over land borders and short sea links (panel.HEALTH_ADJ)"))
    sari = HS.get_parquet("lab/raw/health_ervis_sari.parquet")
    if sari is not None:
        sari.index = pd.DatetimeIndex(sari.index).tz_convert("UTC") if pd.DatetimeIndex(sari.index).tz is not None else pd.DatetimeIndex(sari.index).tz_localize("UTC")
        D.append(Driver("sari_rate", "health", _live_cols(sari), av, note=f"{nt}: severe acute respiratory infection hospital rate per 100 000, all ages (16 countries)"))
    ili = HS.get_parquet("lab/raw/health_ervis.parquet")
    ili.index = pd.DatetimeIndex(ili.index).tz_convert("UTC") if pd.DatetimeIndex(ili.index).tz is not None else pd.DatetimeIndex(ili.index).tz_localize("UTC")
    lc = np.log(ili.where(ili > 0)).diff().clip(-2, 2)        # week-on-week log change: units differ by country, relative changes do not
    D.append(Driver("nb_ili_logchg", "network", _nb_avail(lc, W), av,
                    note=f"{nt}: neighbours' week-on-week log change of the ILI / ARI rate, population-weighted (panel.HEALTH_ADJ)"))
    D.append(Driver("nb_ili_logchg3", "network", _nb_avail(lc.rolling(3, min_periods=2).mean(), W), av,
                    note=f"{nt}: the same, mean of the last 3 weeks"))
    return D


def _health_n1() -> list[Driver]:
    """leading health inputs (lab.src_health_lead). Wikipedia illness reading per country, Monday-Friday of the week
    before the origin (ERVISS has it only on the Friday after), its neighbour spread, and German ARE consultations by age (finer grain)."""
    from forecasting_engine.lab import src_health_lead as SHL
    D, W = [], health_weights()
    for nm, (x, av, nt) in SHL.wiki_features().items():
        D.append(Driver(nm, "health", x, av, note=nt))
        if nm.endswith("_logchg"):
            D.append(Driver(f"nb_{nm}", "network", _nb_avail(x, W), av, note=f"neighbours' {nm}, population-weighted (panel.HEALTH_ADJ); {nt}"))
    for nm, (x, av, nt) in SHL.are_features().items():
        D.append(Driver(nm, "health", x, av, note=nt))
    return D


def _commod_n1(P: Pool) -> list[Driver]:
    """leading commodity inputs (lab.src_commod), each stamped at its real publication: EIA weekly petroleum and gas
    storage (vs 5-year norms), CFTC managed-money positioning, refined-product cracks."""
    from forecasting_engine.lab import src_commod as SC
    D = []
    for nm, (x, av, nt) in SC.eia_features().items():
        D.append(Driver(nm, "energy", _g(x), av, note=nt))
    for nm, (x, av, nt) in SC.cot_features().items():
        D.append(Driver(nm, "markets", _g(x), av, note=nt))
    for nm, (x, av, nt) in SC.crack_features(P.get("fred")).items():
        D.append(Driver(nm, "fuel", _g(x), av, note=nt))
    return D


def _ip_r63(P: Pool) -> list[Driver]:
    """energy use leads output. At a monthly origin (00:00 on the 1st) industrial production is known
    to m-3 or m-2, the electricity load and the gas delivered to large consumers to the day before."""
    from forecasting_engine.lab import src_gas as SG
    ld = P.get("load").resample("D").mean()                   # ENTSO-E actual load, a day is complete + 2 h (back-tests only, never displayed)
    wd = ld[ld.index.dayofweek < 5]
    D = [Driver("load_mtd_yoy", "energy", _mtd(ld, same_days=True), DAY + 2,
                note="ENTSO-E actual load, month-to-date mean vs the same days a year before, % (8 countries; back-tests only)"),
         Driver("load_wd_mtd_yoy", "energy", _mtd(wd, same_days=False), DAY + 2,
                note="ENTSO-E actual load on working days (Mon-Fri), month-to-date mean vs the working days of the same month a year before, %")]
    g = HS.get_parquet("lab/raw/entsog_final_consumers.parquet")
    if g is not None:
        g = g.reset_index()
        g["date"] = pd.to_datetime(g["date"], utc=True)
        x = g.pivot_table(index="date", columns="cc", values="gwh").sort_index()
        nt = g.pivot_table(index="date", columns="cc", values="n_tso").sort_index()
        mk = x.index.tz_convert("UTC").tz_localize(None).to_period("M")
        ntm = nt.groupby(mk).max()
        ok = ntm.eq(ntm.shift(12)).reindex(mk).set_axis(x.index)   # the same TSOs reported a year before (no platform level break)
        y = _mtd(x, same_days=False)
        D.append(Driver("gas_fc_mtd_yoy", "energy", y.where(ok.reindex(y.index)).clip(-80, 200), SG.LAG_DAYS * DAY,
                        note="ENTSOG Transparency Platform (free re-use with attribution): gas from the transmission grid to final consumers "
                             "(industry, gas power plants), month-to-date mean vs the same month a year before, %; known 3 days after the day"))
        mo = SG.monthly_yoy()
        if mo is not None:
            D.append(Driver("gas_fc_yoy", "energy", mo[1], MONTH + SG.LAG_DAYS * DAY, note="the same, whole months, y/y %"))
    return D


CHAIN_PARENTS = {"price_h": ("res_h", "load_h"), "load_d": ("load_h",), "hicp_m": ("diesel_w", "gasoline_w"),
                 "ports_m": ("diesel_w", "unrest_w"),   # fuel and labour unrest (the lab's own forecasts) into port cargo
                 "tourism_m": ("air_d",),               # the lab's own air-traffic forecast (the first days of the month, made before it) into tourism nights
                 "ip_m": ("load_d",)}                   # the load_d forecast for the month's first days (E6 path; needs lab/chain/load_d)
# one direction only (no cycles): the physical layer feeds the market


def extend(fid: str, P: Pool | None = None) -> dict:
    """Add extra_drivers to a frozen family: new driver files and manifest entries; the target file (and the registry) are untouched."""
    P = P or Pool()
    man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
    have = {d["name"] for d in man["drivers"]}
    for d in extra_drivers(fid, P):
        if d.name in have:
            continue
        df = d.data.copy()
        df.columns = [str(c) for c in df.columns]
        b = _pq(df.astype("float32"))
        HS.put_bytes(f"lab/frozen/{fid}/X_{d.name}.parquet", b)
        man["sha256"][f"X_{d.name}"] = hashlib.sha256(b).hexdigest()
        man["drivers"].append(dict(name=d.name, group=d.group, avail_h=d.avail_h, ahead=d.ahead, note=d.note, dayahead_hour=d.dayahead_hour,
                                   cols=[str(c) for c in df.columns], added="after the registry"))
    HS.put_bytes(f"lab/frozen/{fid}/manifest.json", json.dumps(man, indent=1).encode(), "application/json")
    return dict(id=fid, drivers=len(man["drivers"]))


FAMILIES = ("price_h", "load_h", "res_h", "load_d", "diesel_w", "gasoline_w", "ip_m", "retail_m", "unemp_m", "hicp_m", "fx_d", "commod_d", "unrest_w", "kev_d")


# ------------------------------------------------------------------------------------------------------------------------------ freeze
def _pq(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_parquet(b, compression="zstd")
    return b.getvalue()


def freeze(fam: Family) -> dict:
    """Write the family to lab/frozen/<id>/ and return its manifest (with SHA-256 per file)."""
    files = {"Y": fam.Y}
    for d in fam.drivers:
        files[f"X_{d.name}"] = d.data
    man = dict(id=fam.id, label=fam.label, layer=fam.layer, freq=fam.freq, unit=fam.unit, avail_h=fam.avail_h, skip=fam.skip, H=fam.H, seasons=list(fam.seasons),
               origin_hour=fam.origin_hour, notes=fam.notes, series=[str(c) for c in fam.Y.columns], span_min=fam.span_min,
               drivers=[dict(name=d.name, group=d.group, avail_h=d.avail_h, ahead=d.ahead, note=d.note, dayahead_hour=d.dayahead_hour,
                             cols=[str(c) for c in d.data.columns]) for d in fam.drivers], dayahead_hour=fam.dayahead_hour,
               sha256={})
    for k, df in files.items():
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        b = _pq(df.astype("float32"))
        HS.put_bytes(f"lab/frozen/{fam.id}/{k}.parquet", b)
        man["sha256"][k] = hashlib.sha256(b).hexdigest()
    HS.put_bytes(f"lab/frozen/{fam.id}/manifest.json", json.dumps(man, indent=1).encode(), "application/json")
    return man


def load(fid: str) -> Family:
    man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
    rd = lambda k: pd.read_parquet(io.BytesIO(HS.get_bytes(f"lab/frozen/{fid}/{k}.parquet"))).astype("float64")  # noqa: E731
    D = [Driver(d["name"], d["group"], rd(f"X_{d['name']}"), d["avail_h"], d["ahead"], d.get("note", ""), d.get("dayahead_hour")) for d in man["drivers"]]
    return Family(fid, man["label"], man["layer"], man["freq"], man["unit"], rd("Y"), man["avail_h"], man["skip"], man["H"], tuple(man["seasons"]),
                  man["origin_hour"], D, man["notes"], man.get("dayahead_hour"), man.get("span_min"), man.get("lagfix", ""))


# Target delays fixed by the audit (a target delay is part of a family's design). commod_d:
# EIA spot prices are released weekly. unemp_m, ur_long_m: Eurostat publishes unemployment 30-32 days after the month at
# 11:00 CET, so at an origin on the 1st, month m-2 was counted as known although it comes out later that day or the next.
TARGET_LAG_H = {"commod_d": EIA_SPOT_LAG_H, "unemp_m": MONTH + EUROSTAT_DAYS["unemp"] * DAY, "ur_long_m": MONTH + EUROSTAT_DAYS["unemp"] * DAY}


def _lagfix_tag(old: str, tag: str, flags: set) -> str:
    """The manifest's lagfix string: the latest audit tag, then every flag set so far (target = the target's own delay or data changed,
    gdadded = the GDELT files were rebuilt keyed on DATEADDED). Any change of it changes core._lagsig, so no saved forecast is reused."""
    have = set((old or "").split("+")[1:]) if old else set()
    return tag + "".join(f"+{x}" for x in sorted(have | set(flags)))


def relag(fid: str, tag: str = "20261020") -> dict:
    """Apply the publication-lag audit (audit_lag, cot_mask) to a frozen family in place: longer delays in the manifest, the late COT report
    dates dropped from the cot_* driver files, the commod_d target delay. Names, order, the target data and the registry are untouched."""
    man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
    ch = []
    if fid in TARGET_LAG_H and man["avail_h"] < TARGET_LAG_H[fid]:
        ch.append(dict(name="(target)", old=man["avail_h"], new=TARGET_LAG_H[fid]))
        man["avail_h"] = TARGET_LAG_H[fid]
    for d in man["drivers"]:
        a = audit_lag(fid, man["freq"], d["name"], d["avail_h"])
        if a != d["avail_h"]:
            ch.append(dict(name=d["name"], old=d["avail_h"], new=a))
            d["avail_h"] = a
        if d["name"].startswith("cot_"):
            x = pd.read_parquet(io.BytesIO(HS.get_bytes(f"lab/frozen/{fid}/X_{d['name']}.parquet")))
            y = cot_mask(x)
            if len(y) < len(x):
                b = _pq(y.astype("float32"))
                HS.put_bytes(f"lab/frozen/{fid}/X_{d['name']}.parquet", b)
                man["sha256"][f"X_{d['name']}"] = hashlib.sha256(b).hexdigest()
                ch.append(dict(name=d["name"], dropped_rows=len(x) - len(y)))
    if ch:
        tgt = "+target" in man.get("lagfix", "") or any(c["name"] == "(target)" for c in ch)
        man["lagfix"] = _lagfix_tag(man.get("lagfix", ""), tag, {"target"} if tgt else set())   # core._lagsig: saved forecasts not reused
        man.setdefault("lagfix_log", []).append(dict(tag=tag, changes=ch))
        HS.put_bytes(f"lab/frozen/{fid}/manifest.json", json.dumps(man, indent=1).encode(), "application/json")
    return dict(id=fid, changed=len(ch), changes=ch)


# ------------------------------------------------------------------------------------------------------------------------------ GDELT by DATEADDED
GDELT_TARGET = ("unrest_w", "security_w", "info_w")       # families whose target is a GDELT count (or tone)
GDELT_TAG = "gdadded"
_GD_BASE = ("strike", "protest", "violence", "coerce", "tone")


def _gdelt_builders(fid: str, man: dict, P: Pool, chain: bool) -> dict:
    """name -> callable returning the GDELT-derived frame of a frozen family, built the way build_family / extra_drivers built it, from
    the store's current GDELT export ("Y" = the target of GDELT_TARGET). Names not here are not GDELT-derived and are left alone."""
    freq = man["freq"]
    M = freq == "M"
    cols = {d["name"]: d["cols"] for d in man["drivers"]}
    out = {}
    agg = _monthly_mean if M else (lambda x: x)
    for d in man["drivers"]:
        n = d["name"]
        k = n[len("gdelt_"):] if n.startswith("gdelt_") else n
        if k in _GD_BASE and (n.startswith("gdelt_") or fid in GDELT_TARGET):
            wkm = (lambda x: x.resample("W-MON", label="left", closed="left").mean()) if fid == "health_w" else agg   # noqa: E731
            out[n] = (lambda k=k, c=cols[n], f=wkm: f(P.get(k).reindex(columns=c)))
        elif n == "oil_region_violence":
            out[n] = lambda: agg(_regional(P.get("violence"), OIL_REGION))
        elif n == "tone_world":
            out[n] = lambda: _g(P.get("tone").mean(axis=1))
        elif n in ("strike_eu", "protest_eu", "tone_eu", "strike_eu_month", "protest_eu_month", "tone_eu_month"):
            b = n.split("_")[0]
            how = "mean" if b == "tone" else "sum"
            out[n] = (lambda b=b, how=how, m=n.endswith("_month"):
                      (_monthly_mean if m else (lambda x: x))(_g(getattr(P.get(b).reindex(columns=list(EU27)), how)(axis=1))))
        elif n == "violence_month":
            out[n] = (lambda c=cols[n]: _monthly_mean(P.get("violence").reindex(columns=c)))
        elif n.startswith("gd_"):
            def soc(n=n):
                from forecasting_engine.lab import src_society as SSO
                key = ("sso", fid)
                if key not in P._m:
                    P._m[key] = {x.name: x.data for x in SSO.drivers(fid, EU27 + ("GB", "UA", "TR") + (("RU", "IL", "EG") if fid != "unrest_w" else ()))}
                return P._m[key][n]
            out[n] = soc
        elif chain and n.startswith("chain_unrest_w"):
            def ch(n=n):
                sfx = n[len("chain_unrest_w"):-len("_path")]
                if sfx == "":                                   # the first chain driver came from an older gbm_own chain file
                    g = HS.get_parquet("lab/chain/unrest_w_gbm_own.parquet")
                    if g is None:
                        return None
                    g = g.reset_index()
                    g = g.drop(columns=[c for c in ("series",) if c in g.columns])
                else:
                    g = HS.get_parquet("lab/chain/unrest_w.parquet").reset_index()
                    g = dict(_kinds(g))[sfx]
                g["valid"] = pd.to_datetime(g["valid"], utc=True)
                return _chain_monthly(fid, "unrest_w", g, sfx).data
            out[n] = ch
    if fid in GDELT_TARGET:
        def tgt():
            if fid == "unrest_w":
                c = [x.split(":", 1)[1] for x in man["series"] if x.startswith("strike:")]
                Y = pd.concat({"strike": P.get("strike").reindex(columns=c).resample("W-MON", label="left", closed="left").sum(),
                               "protest": P.get("protest").reindex(columns=c).resample("W-MON", label="left", closed="left").sum()}, axis=1)
            elif fid == "security_w":
                c = [x.split(":", 1)[1] for x in man["series"] if x.startswith("violence:")]
                wk = lambda x: x.resample("W-MON", label="left", closed="left")   # noqa: E731
                Y = pd.concat({"violence": wk(P.get("violence").reindex(columns=c)).sum(), "coerce": wk(P.get("coerce").reindex(columns=c)).sum()}, axis=1)
            else:
                c = [x.split(":", 1)[1] for x in man["series"]]
                Y = pd.concat({"tone": P.get("tone").reindex(columns=c).resample("W-MON", label="left", closed="left").mean()}, axis=1)
            Y.columns = [f"{a}:{b}" for a, b in Y.columns]
            return Y
        out["Y"] = tgt
    return out


def _cmp(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    a, b = old.to_numpy(float), new.to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    rel = np.abs(a - b)[both] / np.maximum(np.abs(a[both]), 1e-9)
    return dict(cells=int(both.sum()), nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                share_changed=round(float((rel > 1e-4).mean()), 4) if both.any() else None,
                mean_abs_rel=round(float(rel.mean()), 5) if both.any() else None, max_rel=round(float(rel.max()), 4) if both.any() else None)


def regdelt(fid: str, write: bool = False, chain: bool = False, tag: str = "20261021", P: Pool | None = None) -> dict:
    """Rebuild the GDELT-derived files of a frozen family from the store's GDELT export (lag audit, 21 Oct 2026: keyed on DATEADDED, so no
    later addition revises a past day). Same index, columns, names, order and delays; only the values change (and the target's for
    GDELT_TARGET, where the frozen NaN pattern is kept so the registered windows stay valid). write=False compares only: run against the
    OLD export it must reproduce the frozen files (the check that each builder is the one the family was frozen with)."""
    P = P or Pool()
    man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
    rd = lambda k: pd.read_parquet(io.BytesIO(HS.get_bytes(f"lab/frozen/{fid}/{k}.parquet")))  # noqa: E731
    rep, files = {}, {}
    for n, fn in _gdelt_builders(fid, man, P, chain).items():
        key = "Y" if n == "Y" else f"X_{n}"
        old = rd(key)
        new = fn()
        if new is None:                                       # an input not built yet (a chain file): left as it is
            rep[n] = dict(missing=True)
            continue
        new = new.copy()
        new.columns = [str(c) for c in new.columns]
        new = new.reindex(index=old.index, columns=old.columns).astype("float32")
        if n == "Y":
            new = new.where(old.notna())
        rep[n] = _cmp(old, new)
        files[key] = new
    if write and any(v.get("missing") for v in rep.values()):
        raise SystemExit(f"{fid}: inputs missing for {[k for k, v in rep.items() if v.get('missing')]}: nothing written")
    if write and files:
        for key, df in files.items():
            b = _pq(df)
            HS.put_bytes(f"lab/frozen/{fid}/{key}.parquet", b)
            man["sha256"][key] = hashlib.sha256(b).hexdigest()
        man["lagfix"] = _lagfix_tag(man.get("lagfix", ""), tag, {GDELT_TAG} | ({"target"} if "Y" in files else set()))
        man.setdefault("lagfix_log", []).append(dict(tag=tag, gdelt="rebuilt keyed on DATEADDED", files=sorted(files),
                                                      changes={k: v for k, v in rep.items()}))
        HS.put_bytes(f"lab/frozen/{fid}/manifest.json", json.dumps(man, indent=1).encode(), "application/json")
    return dict(id=fid, files=len(files), written=bool(write and files), lagfix=man.get("lagfix"), report=rep)


def main(argv=None) -> int:
    import sys
    args = argv or sys.argv[1:]
    if args and args[0] == "relag":                          # publication-lag audit on frozen families
        tag = "20261020"
        if len(args) > 2 and args[1] == "--tag":
            tag, args = args[2], args[:1] + args[3:]
        for fid in args[1:]:
            print(json.dumps(relag(fid, tag)), flush=True)
        return 0
    if args and args[0] == "regdelt":                        # GDELT keyed on DATEADDED: regdelt [--write] [--chain] fam ...
        P = Pool()
        w, c = "--write" in args, "--chain" in args
        for fid in [a for a in args[1:] if not a.startswith("--")]:
            print(json.dumps(regdelt(fid, write=w, chain=c, P=P), default=str), flush=True)
        return 0
    if args and args[0] == "extend":
        P = Pool()
        for fid in args[1:] or FAMILIES:
            print(json.dumps(extend(fid, P)), flush=True)
        return 0
    ids = args or list(FAMILIES)
    P = Pool()
    for fid in ids:
        fam = build_family(fid, P)
        man = freeze(fam)
        print(json.dumps(dict(id=fid, series=len(man["series"]), drivers=len(man["drivers"]), start=str(fam.Y.index[0]), end=str(fam.Y.index[-1]),
                              rows=len(fam.Y))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
