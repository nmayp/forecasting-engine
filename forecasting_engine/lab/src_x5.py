"""Cross-layer links, the global industrial cycle and exchange rates for ip_long_m / cpi_long_m, dryness for wildfire_w.

    python -m forecasting_engine.lab.src_x5 load        # FRED -> lab/raw/x5_fred.parquet
    python -m forecasting_engine.lab.panel extend ip_long_m cpi_long_m wildfire_w

| Link (layer -> layer) | Families | Input | Real publication lag (worst case used) | Licence |
|---|---|---|---|---|
| economy (US / Asia manufacturing cycle) -> economy | ip_long_m | US industrial production (total, manufacturing), capacity utilisation, durable-goods and manufacturers' new orders, y/y; Philadelphia, Empire State and Dallas Fed manufacturing surveys (level); Korean, Chinese and Japanese exports (OECD MEI on FRED), y/y; global columns, the same for every country | surveys: published in the reference month (3rd Thursday / 15th / last Monday): used from the month after (stamp + 1 month); INDPRO / TCU mid next month (+ 20 d); orders next month + 27 d; OECD exports + 62 d (last value on 23 Oct 2026 was August); FRED serves the latest vintage (revisions of the US series: noted, small) | FRED: public domain (Fed, Census), OECD MEI via FRED: OECD terms |
| markets (exchange rates) -> economy | cpi_long_m, ip_long_m | local currency per US dollar, y/y and 3-month change of the monthly mean (depreciation = positive; euro-area members take the euro rate; the US has none), Fed H.10 and OECD MEI via FRED | monthly mean known one month + 7 days after the month start | FRED: Fed H.10 public domain; OECD MEI: OECD terms |
| markets, trade (US prices) -> economy | cpi_long_m | US producer prices (all commodities), US import price index (all commodities), US CPI, y/y | month + 20 days (BLS mid next month; FRED serves the latest vintage) | public domain |
| weather -> environment (fuel dryness) | wildfire_w | vapour pressure deficit (from the capital's daily mean temperature and relative humidity), its anomaly against the same week of earlier years, consecutive dry days at the week's end, days above 25 C, and a Keetch-Byram-type drought index from daily temperature and rain | NASA POWER: week + 4 days | NASA POWER: open |
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK, MONTH = 24.0, 24.0 * 7, 24.0 * 30.44
STORE = "lab/raw/x5_fred.parquet"
FRED_IDS = {  # id -> short name
    "GACDFSA066MSFRBPHI": "philly", "GACDISA066MSFRBNY": "empire", "BACTSAMFRBDAL": "dallas", "INDPRO": "us_ip", "IPMAN": "us_ipman", "TCU": "us_tcu",
    "DGORDER": "us_durable", "NEWORDER": "us_neworder", "XTEXVA01KRM667S": "kr_exports", "XTEXVA01CNM667S": "cn_exports", "XTEXVA01JPM667S": "jp_exports",
    "PPIACO": "us_ppi", "IR14200": "us_import_px", "CPIAUCSL": "us_cpi",
    # exchange rates, local currency per USD unless noted
    "EXUSEU": "fx_EUR_inv", "EXJPUS": "fx_JPN", "EXCAUS": "fx_CAN", "EXMXUS": "fx_MEX", "EXBZUS": "fx_BRA", "EXINUS": "fx_IND", "EXKOUS": "fx_KOR",
    "EXSZUS": "fx_CHE", "EXUSUK": "fx_GBR_inv", "EXUSAL": "fx_AUS_inv", "EXNOUS": "fx_NOR", "EXDNUS": "fx_DNK", "EXSDUS": "fx_SWE", "EXSFUS": "fx_ZAF",
    "EXCHUS": "fx_CHN",
    "CCUSMA02CLM618N": "fx_CHL", "CCUSMA02CZM618N": "fx_CZE", "CCUSMA02HUM618N": "fx_HUN", "CCUSMA02IDM618N": "fx_IDN", "CCUSMA02ILM618N": "fx_ISR",
    "CCUSMA02PLM618N": "fx_POL", "CCUSMA02RUM618N": "fx_RUS", "CCUSMA02TRM618N": "fx_TUR",
}
EURO = ("AUT", "BEL", "BGR", "DEU", "ESP", "EST", "FIN", "FRA", "GRC", "HRV", "IRL", "ITA", "LTU", "LUX", "LVA", "NLD", "PRT", "SVK", "SVN")


def load(log=print) -> dict:
    from forecasting_engine.lab.sources import _get
    parts = []
    for sid, nm in FRED_IDS.items():
        try:
            r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", ua=False, params={"id": sid})
            d = pd.read_csv(io.BytesIO(r.content))
            d.columns = ["date", "value"]
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d = d.dropna()
            d["date"] = pd.to_datetime(d["date"], utc=True)
            d["id"] = nm
            parts.append(d)
            log(f"x5 fred {nm}: {len(d)} rows, last {d['date'].max().date()}")
        except Exception as e:   # noqa: BLE001
            log(f"x5 fred {nm}: FAILED {type(e).__name__}: {str(e)[:120]}")
    out = pd.concat(parts).set_index("date")
    HS.put_parquet(STORE, out)
    return dict(rows=len(out), ids=int(out["id"].nunique()))


def _fred(nm: str) -> pd.Series | None:
    d = HS.get_parquet(STORE)
    if d is None:
        return None
    d = d[d["id"] == nm]
    if d.empty:
        return None
    s = d["value"].astype(float)
    i = pd.DatetimeIndex(s.index)
    s.index = i.tz_convert("UTC") if i.tz is not None else i.tz_localize("UTC")
    return s.sort_index().resample("MS").mean().dropna()


def _yoy(s: pd.Series) -> pd.Series:
    return 100 * np.log(s / s.shift(12))


def fx_frame(countries) -> pd.DataFrame:
    """Local currency per USD, monthly, as a frame of ISO3 columns (euro members share the euro rate)."""
    out = {}
    eur = _fred("fx_EUR_inv")
    for c in countries:
        s = None
        if c in EURO and eur is not None:
            s = 1.0 / eur
        elif _fred(f"fx_{c}") is not None:
            s = _fred(f"fx_{c}")
        elif _fred(f"fx_{c}_inv") is not None:
            s = 1.0 / _fred(f"fx_{c}_inv")
        if s is not None:
            out[c] = s
    return pd.DataFrame(out)


def long_drivers(fid: str, PN) -> list:
    Driver, D = PN.Driver, []
    if fid == "ip_long_m":
        for nm, key, lag, note in (
                ("x5_us_ip_yoy", "us_ip", 20, "US industrial production, y/y log change %, month + 20 days"),
                ("x5_us_ipman_yoy", "us_ipman", 20, "US manufacturing output, y/y log change %, month + 20 days"),
                ("x5_us_durable_yoy", "us_durable", 27, "US durable-goods orders, y/y log change %, month + 27 days"),
                ("x5_us_neworder_yoy", "us_neworder", 27, "US manufacturers' new orders, y/y log change %, month + 27 days"),
                ("x5_kr_exports_yoy", "kr_exports", 62, "Korean exports (OECD MEI), y/y log change %, month + 62 days"),
                ("x5_cn_exports_yoy", "cn_exports", 62, "Chinese exports (OECD MEI), y/y log change %, month + 62 days"),
                ("x5_jp_exports_yoy", "jp_exports", 62, "Japanese exports (OECD MEI), y/y log change %, month + 62 days")):
            s = _fred(key)
            if s is not None:
                D.append(Driver(nm, "economy" if "export" not in nm else "trade", _yoy(s).to_frame("*"), MONTH + lag * DAY, note=note))
        s = _fred("us_tcu")
        if s is not None:
            D.append(Driver("x5_us_tcu", "economy", s.to_frame("*"), MONTH + 20 * DAY, note="US capacity utilisation, level %, month + 20 days"))
        for nm, key in (("x5_philly", "philly"), ("x5_empire", "empire"), ("x5_dallas", "dallas")):
            s = _fred(key)
            if s is not None:    # published in the reference month; used from the month after to stay clear of the month-start stamp
                D.append(Driver(nm, "economy", s.to_frame("*"), MONTH, note=f"{key} Fed manufacturing survey, general activity, published in the reference month; known one month after its start"))
    if fid in ("cpi_long_m", "ip_long_m"):
        ctry = sorted(set(EURO) | {nm[3:] for nm in FRED_IDS.values() if nm.startswith("fx_") and "_inv" not in nm and nm != "fx_EUR_inv"}
                      | {"GBR", "AUS"})
        fx = fx_frame(ctry)
        if not fx.empty:
            lag = MONTH + 7 * DAY
            D.append(Driver("x5_fx_yoy", "markets", (100 * np.log(fx / fx.shift(12))).replace([np.inf, -np.inf], np.nan), lag,
                            note="local currency per US dollar, y/y log change % (depreciation positive); monthly mean, known 7 days after the month ends; euro members share the euro rate"))
            D.append(Driver("x5_fx_chg3", "markets", (100 * np.log(fx / fx.shift(3))).replace([np.inf, -np.inf], np.nan), lag,
                            note="local currency per US dollar, 3-month log change % (depreciation positive)"))
    if fid == "cpi_long_m":
        for nm, key, note in (("x5_us_ppi_yoy", "us_ppi", "US producer prices, all commodities"), ("x5_us_import_yoy", "us_import_px", "US import price index, all commodities"),
                              ("x5_us_cpi_yoy", "us_cpi", "US consumer prices")):
            s = _fred(key)
            if s is not None:
                D.append(Driver(nm, "markets", _yoy(s).to_frame("*"), MONTH + 20 * DAY, note=f"{note}, y/y log change %, month + 20 days (latest vintage)"))
    return D


# ------------------------------------------------------------------------------------------------------------------------ wildfire_w
def _vpd(t: pd.DataFrame, rh: pd.DataFrame) -> pd.DataFrame:
    """Vapour pressure deficit (kPa) from daily mean temperature (C) and relative humidity (%)."""
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh.clip(0, 100) / 100)


def _kbdi(t: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    """Keetch-Byram drought index (0-800 hundredths of an inch of soil water deficit), daily, from daily mean temperature (+4 C as a stand-in
    for the maximum, which POWER at the capital does not carry) and rain; mean annual rain fixed at 25 in (630 mm)."""
    out = pd.DataFrame(np.nan, index=t.index, columns=t.columns)
    denom = 1 + 10.88 * np.exp(-0.0441 * 25.0)
    for c in t.columns:
        tv, rv = t[c].to_numpy(float) + 4.0, rain[c].to_numpy(float)
        q, spell = 0.0, 0.0
        res = np.full(len(tv), np.nan)
        for i in range(len(tv)):
            if not (np.isfinite(tv[i]) and np.isfinite(rv[i])):
                continue
            r_in = rv[i] / 25.4
            if r_in > 0:                                        # the first 0.2 in of a rain spell is intercepted
                net = max(spell + r_in - 0.2, 0.0) - max(spell - 0.2, 0.0)
                spell += r_in
            else:
                net, spell = 0.0, 0.0
            q = max(q - 100 * net, 0.0)
            tf = tv[i] * 9 / 5 + 32
            if tf > 50:
                q = min(q + (800 - q) * (0.968 * np.exp(0.0486 * tf) - 8.30) * 0.001 / denom, 800.0)
            res[i] = q
        out[c] = res
    return out


def wildfire_drivers(PN) -> list:
    from forecasting_engine.lab import src_envdrivers as SE
    Driver, D = PN.Driver, []
    t, rh, rain = SE.capital_daily("t2m"), SE.capital_daily("rh"), SE.capital_daily("rain")
    if t is None or rh is None or rain is None:
        return D
    cols = [c for c in t.columns if c in rh.columns and c in rain.columns]
    t, rh, rain = t[cols], rh[cols], rain[cols]
    av = WEEK + 4 * DAY
    nt = "NASA POWER daily at the capital (4 days of latency)"

    def wk(x, how="mean", min_days=6):
        r = x.resample("W-MON", label="left", closed="left")
        return (r.mean() if how == "mean" else r.max() if how == "max" else r.sum()).where(r.count() >= min_days)
    vpd = _vpd(t, rh)
    vw = wk(vpd)
    D.append(Driver("x5_vpd_w", "weather", vw, av, note=f"{nt}: vapour pressure deficit (kPa) from mean temperature and relative humidity, weekly mean"))
    woy = pd.Series(vw.index.isocalendar().week.to_numpy().clip(max=52), index=vw.index)
    an = pd.DataFrame(np.nan, index=vw.index, columns=vw.columns)
    for k in range(1, 53):
        idx = np.where(woy.to_numpy() == k)[0]
        for pos, i in enumerate(idx):                  # earlier years only (causal)
            if pos >= 2:
                an.iloc[i] = vw.iloc[i] - vw.iloc[idx[:pos]].mean()
    D.append(Driver("x5_vpd_anom", "weather", an, av, note=f"{nt}: weekly VPD minus the mean of the same week in earlier years"))
    dry = (rain.fillna(0) < 1.0)
    run = dry.astype(int).copy()
    for c in run.columns:                              # consecutive dry days ending on each day
        v, n = run[c].to_numpy(), 0
        for i in range(len(v)):
            n = n + 1 if v[i] else 0
            v[i] = n
        run[c] = v
    D.append(Driver("x5_dry_run", "weather", np.log1p(run.resample("W-MON", label="left", closed="left").last()), av,
                    note=f"{nt}: consecutive days with under 1 mm of rain at the end of the week, log1p"))
    D.append(Driver("x5_hot_days", "weather", wk((t > 25).astype(float).where(t.notna()), "sum", 6), av, note=f"{nt}: days of the week with mean temperature above 25 C"))
    kb = _kbdi(t, rain)
    D.append(Driver("x5_kbdi", "environment", wk(kb, "mean", 6), av, note=f"{nt}: Keetch-Byram-type drought index from temperature and rain (daily, weekly mean)"))
    return D


def drivers(fid: str, P, PN) -> list:
    if fid in ("ip_long_m", "cpi_long_m"):
        return long_drivers(fid, PN)
    if fid == "wildfire_w":
        return wildfire_drivers(PN)
    return []


if __name__ == "__main__":
    if sys.argv[1:2] == ["load"]:
        print(load())
