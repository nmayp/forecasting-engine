"""Cross-layer links, new inputs for the layers that are lost or thin.

    python -m forecasting_engine.lab.src_x4 load        # FRED freight / IMF commodity indices -> lab/raw/x4_fred.parquet
    python -m forecasting_engine.lab.panel extend health_w choke_w ip_long_m cpi_long_m diesel_w wildfire_w

| Link (layer -> layer) | Families | Input | Real publication lag (worst case used) | Licence |
|---|---|---|---|---|
| transport (air) -> health | health_w | EUROCONTROL flights per state, weekly mean, y/y and 4-week change, stamped when known | airport files: week + 46 days (before the D-1 dashboard, DASH_FROM 2020-04); dashboard weeks: week end + 3 days | EUROCONTROL: back-tests only |
| environment (air quality) -> health | health_w | EEA PM2.5 and NO2 in the capital, weekly mean of log, and minus the trailing 52-week median | day + 3 days for most states, a few 6-12: week + 5 days after the Monday stamp (= Sunday + 6 days) | EEA: CC BY 4.0 |
| weather -> health | health_w | NASA POWER at the capital: absolute humidity (g/m3) and 2 m temperature, weekly mean, anomaly vs the same week of earlier years, 2-week change | POWER: 4 days after the day (src_envdrivers.POWER_LAG_DAYS): week + 4 days | NASA POWER: open |
| tourism -> health | health_w | Eurostat tour_occ_nim nights spent, y/y log change | month + 73 days (as the registered tourism driver) | Eurostat CC BY 4.0 |
| calendar (schools, public holidays) -> health | health_w | change of the school-holiday share week on week, next week's share, weekday public holidays this and last week | published months to years ahead | OpenHolidays ODbL, `holidays` package |
| health -> health (neighbours) | health_w | neighbours' ILI / ARI rate relative to their own trailing 52-week mean, the same one week earlier, neighbours minus own week-on-week log change, share of countries rising | ERVISS week + 5 days | ECDC ERVISS: acknowledgement |
| society (conflict news) -> transport | choke_w | GDELT 7-day share of material-conflict, verbal-conflict and military / rebel violence events in the countries around each strait (Suez <- EG, IL; Bab el-Mandeb <- YE, SA, EG; Hormuz <- IR, IQ, SA; Bosporus <- TR, UA, RU; ...) | day + 1 | GDELT: citation |
| fuel, economy -> transport | choke_w | freight price index (BLS PPI deep sea freight, FRED) y/y, IMF energy index y/y | month + 35 days (BLS first print mid-month; revised after 4 months: noted) / IMF month + 28 days | public domain / IMF |
| transport -> fuel | diesel_w | weekly tanker transits through Hormuz, Suez, Bab el-Mandeb, Cape, Malacca, Bosporus (IMF PortWatch) and their 4-week change | week + 10 days (PortWatch Tuesday update) | IMF PortWatch: permission for commercial reuse |
| society (labour protest) -> fuel | diesel_w | GDELT labour and farmer protest share by country (blockades) and public-holiday weeks | day + 1 | GDELT |
| economy (surveys, trade, prices) -> economy | ip_long_m, cpi_long_m | EC business and consumer surveys (production, order books, export orders, price expectations), Eurostat export / import volumes and import prices, mapped to ISO3 for the EU members; IMF energy, food, metals, coal indices y/y; freight PPI | surveys: end of the month; trade month + 80 days; import prices month + 35 days; IMF month + 28 days | Eurostat CC BY 4.0, IMF |
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK, MONTH = 24.0, 24.0 * 7, 24.0 * 30.44
STORE = "lab/raw/x4_fred.parquet"
FRED_IDS = {"PCU483111483111": "freight_ppi", "PNRGINDEXM": "imf_energy", "PFOODINDEXM": "imf_food", "PALLFNFINDEXM": "imf_allfuel",
            "PMETAINDEXM": "imf_metals", "PCOALAUUSDM": "coal_aus", "PRAWMINDEXM": "imf_rawmat"}
IMF_LAG = 28 * DAY
FREIGHT_LAG = 35 * DAY
POWER_LAG_DAYS = 4.0
HEALTH_EXTRA_AHEAD = -365 * DAY


def load(log=print) -> dict:
    from forecasting_engine.lab.sources import _get
    parts = []
    for sid, nm in FRED_IDS.items():
        try:
            r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", ua=False, params={"id": sid})   # FRED refuses our agent string
            d = pd.read_csv(io.BytesIO(r.content))
            d.columns = ["date", "value"]
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d = d.dropna()
            d["date"] = pd.to_datetime(d["date"], utc=True)
            d["id"] = nm
            parts.append(d)
            log(f"x4 fred {nm}: {len(d)} rows, last {d['date'].max().date()}")
        except Exception as e:   # noqa: BLE001
            log(f"x4 fred {nm}: FAILED {type(e).__name__}: {str(e)[:120]}")
    out = pd.concat(parts).set_index("date")
    HS.put_parquet(STORE, out)
    return dict(rows=len(out), ids=int(out["id"].nunique()))


def _ts(x) -> pd.DatetimeIndex:
    i = pd.DatetimeIndex(x)
    return i.tz_convert("UTC") if i.tz is not None else i.tz_localize("UTC")


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = _ts(df.index)
    return df.sort_index()


def _wk(x: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    """Monday-Sunday weekly mean of a daily frame (weeks with fewer than min_days values are NaN), stamped on the Monday."""
    r = x.resample("W-MON", label="left", closed="left")
    return r.mean().where(r.count() >= min_days)


def _fred(nm: str) -> pd.Series | None:
    d = HS.get_parquet(STORE)
    if d is None:
        return None
    d = d[d["id"] == nm]
    s = d["value"].astype(float)
    s.index = _ts(s.index)
    return s.sort_index().resample("MS").mean().dropna()


def _clim_anom(w: pd.DataFrame, years_min: int = 2) -> pd.DataFrame:
    """Weekly value minus the mean of the same ISO week (+-1) in EARLIER years only (causal)."""
    woy = pd.Series(w.index.isocalendar().week.to_numpy().clip(max=52), index=w.index)
    yr = pd.Series(w.index.year, index=w.index)
    out = pd.DataFrame(np.nan, index=w.index, columns=w.columns)
    for c in w.columns:
        s = w[c]
        for k in range(1, 53):
            near = woy.isin([(k - 2) % 52 + 1, k, k % 52 + 1])
            idx = np.where(woy.to_numpy() == k)[0]
            if not len(idx):
                continue
            for i in idx:
                y0 = yr.iloc[i]
                prev = s[near.to_numpy() & (yr.to_numpy() < y0)].dropna()
                if prev.index.year.nunique() >= years_min:
                    out.iloc[i, out.columns.get_loc(c)] = s.iloc[i] - prev.mean()
    return out


def _abs_humidity(t: pd.DataFrame, rh: pd.DataFrame) -> pd.DataFrame:
    """g/m3 from temperature (C) and relative humidity (%) (Magnus formula)."""
    es = 6.112 * np.exp(17.62 * t / (243.12 + t))
    return 216.7 * (rh.clip(0, 100) / 100 * es) / (273.15 + t)


# ------------------------------------------------------------------------------------------------------------------------ health_w
def _flights_known(air: pd.DataFrame, dash_from: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly flights: y/y log change and the 4-week log change, stamped when KNOWN (avail 0): airport files week + 46 days before the daily
    dashboard (D-1) existed, week end + 3 days after."""
    w = _wk(air.resample("D").sum(min_count=1), min_days=6).where(lambda x: x > 0)
    lw = np.log(w)
    out = []
    for f in (lw - lw.shift(52), lw - lw.shift(4)):
        f = f.replace([np.inf, -np.inf], np.nan)
        early = f[f.index + pd.Timedelta(days=7) < dash_from].copy()
        early.index = early.index + pd.Timedelta(days=7 + 46)
        late = f[f.index + pd.Timedelta(days=7) >= dash_from].copy()
        late.index = late.index + pd.Timedelta(days=7 + 3)
        # the airport files also carry the early part until the dashboard begins; the join keeps the earliest known value of each week
        out.append(pd.concat([early, late]).sort_index())
    return out[0], out[1]


def health_drivers(PN) -> list:
    from forecasting_engine.lab import src_envdrivers as SE
    from forecasting_engine.lab import src_society as SS
    Driver, D = PN.Driver, []
    ili = _utc(HS.get_parquet("lab/raw/health_ervis.parquet"))
    ccs = list(ili.columns)
    W = PN.health_weights()
    # 1. transport -> health: air traffic (mobility)
    air = HS.get_parquet("lab/raw/air_traffic.parquet")
    if air is not None:
        air = _utc(air).reindex(columns=ccs)
        yy, ch = _flights_known(air.dropna(axis=1, how="all"), PN.DASH_FROM)
        nt = "EUROCONTROL IFR flights per state (back-tests only), weekly; stamped when known (airport files +46 d, dashboard week end + 3 d)"
        D += [Driver("x4_flights_yoy", "transport", yy, 0, note=f"{nt}: y/y log change"),
              Driver("x4_flights_chg4", "transport", ch, 0, note=f"{nt}: 4-week log change")]
    # 2. environment -> health: air quality (EEA, CC BY 4.0)
    for pol, key in (("pm25", "aq_pm25"), ("no2", "aq_no2")):
        a = HS.get_parquet(f"lab/raw/{key}.parquet")
        if a is None:
            continue
        a = np.log1p(_utc(a).astype(float).clip(lower=0))
        w = _wk(a, 5).reindex(columns=[c for c in ccs if c in a.columns])
        D.append(Driver(f"x4_{pol}_wk", "environment", w, WEEK + 5 * DAY, note=f"EEA {pol} in the capital, weekly mean of log1p (CC BY 4.0), most states within 3 days"))
        D.append(Driver(f"x4_{pol}_rel", "environment", w - w.shift(1).rolling(52, min_periods=26).median(), WEEK + 5 * DAY,
                        note=f"EEA {pol}, weekly mean of log1p minus the median of the 52 weeks before"))
    # 3. weather -> health: absolute humidity and temperature anomalies at the capital (NASA POWER)
    t, rh = SE.capital_daily("t2m"), SE.capital_daily("rh")
    if t is not None and rh is not None:
        cols = [c for c in ccs if c in t.columns and c in rh.columns]
        t, rh = _utc(t)[cols], _utc(rh)[cols]
        ah = _wk(_abs_humidity(t, rh), 6)
        tw = _wk(t, 6)
        av = WEEK + POWER_LAG_DAYS * DAY
        nt = "NASA POWER daily at the capital (4 days of latency)"
        D += [Driver("x4_abs_hum", "weather", ah, av, note=f"{nt}: absolute humidity g/m3, weekly mean (low humidity favours influenza spread)"),
              Driver("x4_abs_hum_anom", "weather", _clim_anom(ah), av, note=f"{nt}: absolute humidity minus the same week (+-1) of earlier years"),
              Driver("x4_abs_hum_chg2", "weather", ah - ah.shift(2), av, note=f"{nt}: absolute humidity, 2-week change"),
              Driver("x4_t2m_anom", "weather", _clim_anom(tw), av, note=f"{nt}: 2 m temperature minus the same week (+-1) of earlier years"),
              Driver("x4_t2m_chg2", "weather", tw - tw.shift(2), av, note=f"{nt}: 2 m temperature, 2-week change (cold snaps)")]
    # 4. tourism -> health: nights spent (monthly, 73 days)
    tn = HS.get_parquet("lab/raw/tourism_nights.parquet")
    if tn is not None:
        tn = _utc(tn).reindex(columns=ccs).where(lambda x: x > 0)
        D.append(Driver("x4_tourism_yoy", "tourism", np.log(tn / tn.shift(12)).replace([np.inf, -np.inf], np.nan).clip(-1.5, 1.5), MONTH + 73 * DAY,
                        note="Eurostat tour_occ_nim nights spent, y/y log change (month + 73 days)"))
    # 5. calendar -> health: schools and public holidays (known long before)
    sch = HS.get_parquet("lab/raw/school_holidays.parquet")
    if sch is not None:
        sw = _wk(_utc(sch).reindex(columns=ccs), 1)
        D += [Driver("x4_school_chg", "calendar", sw - sw.shift(1), -180 * DAY, ahead=True, note="school-holiday share, this week minus last week (term starts and ends); published months ahead"),
              Driver("x4_school_next", "calendar", sw.shift(-1), -180 * DAY, ahead=True, note="school-holiday share of the NEXT week (calendar, published months ahead)")]
    hol = SS.holidays_weekly(ccs)
    hol.index = _ts(hol.index)
    D += [Driver("x4_pub_hol", "calendar", hol, HEALTH_EXTRA_AHEAD, ahead=True, note="weekday public holidays in the target week (holidays package)"),
          Driver("x4_pub_hol_prev", "calendar", hol.shift(1), HEALTH_EXTRA_AHEAD, ahead=True, note="weekday public holidays in the week before the target week")]
    # 6. neighbours' ERVISS signals, lagged and relative (units differ by country)
    lg = np.log(ili.where(ili > 0))
    rel = (lg - lg.rolling(52, min_periods=26).mean()).clip(-3, 3)
    chg = lg.diff().clip(-2, 2)
    nb_rel = PN._nb_avail(rel, W)
    nb_chg = PN._nb_avail(chg, W)
    av = WEEK + 5 * DAY
    nt = "ECDC ERVISS ILI / ARI rate"
    D += [Driver("x4_nb_rel", "network", nb_rel, av, note=f"{nt}: neighbours' log rate minus their own trailing 52-week mean, population-weighted"),
          Driver("x4_nb_rel_l1", "network", nb_rel.shift(1), av, note="the same, one week earlier"),
          Driver("x4_nb_lead", "network", (nb_chg - chg.reindex(columns=nb_chg.columns)).clip(-3, 3), av,
                 note="neighbours' week-on-week log change minus the country's own (positive: the neighbours are ahead)"),
          Driver("x4_own_rel", "health", rel, av, note=f"{nt}: the country's own log rate minus its trailing 52-week mean"),
          Driver("x4_eu_rising", "network", ((chg.rolling(3, min_periods=2).mean() > 0.03).where(chg.notna()).mean(axis=1)).to_frame("*"), av,
                 note="share of reporting countries whose 3-week mean log change is above +3 %")]
    return D


# ------------------------------------------------------------------------------------------------------------------------ choke_w
STRAIT_COUNTRIES = {"Suez Canal": ("EG", "IL"), "Bab el-Mandeb Strait": ("YE", "SA", "EG"), "Strait of Hormuz": ("IR", "IQ", "SA"),
                    "Bosporus Strait": ("TR", "UA", "RU"), "Kerch Strait": ("RU", "UA"), "Cape of Good Hope": ("YE", "EG", "IL"),
                    "Taiwan Strait": ("CN",), "Gibraltar Strait": ("ES",), "Dover Strait": ("GB", "FR"), "Oresund Strait": ("DK", "SE"),
                    "Panama Canal": ("US",), "Malacca Strait": ("CN",)}


def choke_drivers(PN, cols) -> list:
    from forecasting_engine.lab import src_society as SS
    Driver, D = PN.Driver, []
    nt = f"{SS.ATTR}; 7-day share of the events of the countries around the strait, per 1000"
    series = {"x4_loc_material": ["q:4"], "x4_loc_verbal": ["q:3"], "x4_loc_cooperation": ["q:2"]}
    for nm, keys in series.items():
        sh = SS._share(keys)
        out = {}
        for s in cols:
            cc = [c for c in STRAIT_COUNTRIES.get(s, ()) if c in sh.columns]
            if cc:
                out[s] = sh[cc].mean(axis=1)
        if out:
            f = pd.DataFrame(out)
            D.append(Driver(nm, "society", f, DAY + 1, note=f"{nt} ({keys[0]})"))
            if nm == "x4_loc_material":
                D.append(Driver(nm + "_chg28", "society", f - f.shift(28), DAY + 1, note="the same material-conflict share, 28-day change"))
    viol = SS._share(SS._ak(("MIL", "REB", "UAF", "RAD"), SS.VIOL))
    out = {s: viol[[c for c in STRAIT_COUNTRIES[s] if c in viol.columns]].mean(axis=1) for s in cols if s in STRAIT_COUNTRIES
           and any(c in viol.columns for c in STRAIT_COUNTRIES[s])}
    if out:
        D.append(Driver("x4_loc_armed", "security", pd.DataFrame(out), DAY + 1, note=f"{nt}: military and armed-group actors in violent events"))
    rs = SS._share(["q:4"])
    red = rs[[c for c in ("EG", "YE", "SA", "IL") if c in rs.columns]].mean(axis=1)
    D.append(Driver("x4_red_sea_material", "security", red.to_frame("*"), DAY + 1, note=f"{nt}: Egypt, Yemen, Saudi Arabia, Israel, material conflict"))
    gulf = rs[[c for c in ("IR", "IQ", "SA") if c in rs.columns]].mean(axis=1)
    D.append(Driver("x4_gulf_material", "security", gulf.to_frame("*"), DAY + 1, note=f"{nt}: Iran, Iraq, Saudi Arabia, material conflict"))
    fr, en = _fred("freight_ppi"), _fred("imf_energy")
    if fr is not None:
        D.append(Driver("x4_freight_yoy", "markets", (100 * np.log(fr / fr.shift(12))).to_frame("*"), FREIGHT_LAG + MONTH,
                        note="BLS PPI deep sea freight (FRED PCU483111483111), y/y log change %; first print mid-month, revised after 4 months"))
    if en is not None:
        D.append(Driver("x4_imf_energy_yoy", "energy", (100 * np.log(en / en.shift(12))).to_frame("*"), IMF_LAG + MONTH, note="IMF energy price index (FRED PNRGINDEXM), y/y log change %"))
    return D


# ------------------------------------------------------------------------------------------------------------------------ diesel_w
TANKER_TAG = {"Strait of Hormuz": "hormuz", "Suez Canal": "suez", "Bab el-Mandeb Strait": "bab", "Cape of Good Hope": "cape", "Malacca Strait": "malacca",
              "Bosporus Strait": "bosporus"}
TANKER_STRAITS = ("Strait of Hormuz", "Suez Canal", "Bab el-Mandeb Strait", "Cape of Good Hope", "Malacca Strait", "Bosporus Strait")


def diesel_drivers(PN, cols) -> list:
    from forecasting_engine.lab import src_society as SS
    Driver, D = PN.Driver, []
    c = HS.get_parquet("lab/raw/portwatch_chokepoints_daily.parquet").reset_index()
    c["date"] = pd.to_datetime(c["date"], utc=True)
    r = lambda v: c.pivot_table(index="date", columns="portname", values=v, aggfunc="sum").sort_index()   # noqa: E731
    for v, tag in (("n_tanker", "tankers"), ("capacity_tanker", "tanker_dwt")):
        w = r(v)[[s for s in TANKER_STRAITS if s in set(c["portname"])]]
        wk = w.resample("W-MON", label="left", closed="left").sum()
        full = w.notna().resample("W-MON", label="left", closed="left").sum() >= 7
        wk = wk.where(full)
        for s in wk.columns:
            nm = TANKER_TAG[s]
            lw = np.log(wk[[s]].where(wk[[s]] > 0)).rename(columns={s: "*"})
            D.append(Driver(f"x4_{tag}_{nm}", "transport", lw, WEEK + 10 * DAY, note=f"IMF PortWatch {s} weekly {v}, log (week + 10 days)"))
            if tag == "tankers":
                D.append(Driver(f"x4_{tag}_{nm}_chg4", "transport", lw - lw.shift(4), WEEK + 10 * DAY, note=f"IMF PortWatch {s} weekly tanker transits, 4-week log change"))
    ccs = [c_ for c_ in cols]
    pro = SS._share(SS._ak(("LAB", "AGR"), ("14",)))
    wkm = lambda x: x.resample("W-MON", label="left", closed="left").mean()   # noqa: E731
    D.append(Driver("x4_lab_agr_protest", "society", wkm(pro.reindex(columns=ccs)), WEEK + DAY,
                    note=f"{SS.ATTR}; labour and farmer protests (blockades), weekly mean of the 7-day share per 1000"))
    hol = SS.holidays_weekly(ccs)
    hol.index = _ts(hol.index)
    D.append(Driver("x4_pub_hol", "calendar", hol, -365 * DAY, ahead=True, note="weekday public holidays in the target week (holidays package)"))
    en = _fred("imf_energy")
    if en is not None:
        D.append(Driver("x4_imf_energy_yoy", "energy", (100 * np.log(en / en.shift(12))).to_frame("*"), IMF_LAG + MONTH, note="IMF energy price index, y/y log change %"))
    return D


# ------------------------------------------------------------------------------------------------------------------------ ip_long_m, cpi_long_m
def long_drivers(fid: str, PN) -> list:
    from forecasting_engine.lab import src_econ as SEC
    from forecasting_engine.lab.sources import ISO3
    Driver, D = PN.Driver, []
    inv = {v: k for k, v in ISO3.items()}

    def to3(w: pd.DataFrame) -> pd.DataFrame:
        w = w.rename(columns={c: inv[c] for c in w.columns if c in inv})
        return w[[c for c in w.columns if len(c) == 3 and c in ISO3]]

    surveys = {"ip_long_m": ["ind_prod_exp", "ind_orders", "ind_export_orders", "ind_stocks", "ind_prod_trend", "con_conf", "svc_conf"],
               "cpi_long_m": ["ind_price_exp", "ret_price_exp", "svc_price_exp", "con_price_exp", "cons_price_exp", "cons_price_past"]}[fid]
    for k in surveys:
        w = SEC.wide(k)
        if w is not None:
            D.append(Driver(f"x4_es_{k}", "economy", to3(w), MONTH, note=f"EC survey {k} (SA balance, Eurostat), end of the reference month; EU members only"))
    trade = ["exp_vol", "imp_vol", "exp_vol_xeu", "imp_vol_xeu"] if fid == "ip_long_m" else ["imp_price", "imp_price_nrg", "imp_uv_xeu"]
    for k in trade:
        w = SEC.wide(k)
        if w is None:
            continue
        lag = 80 if "vol" in k else 35
        D.append(Driver(f"x4_es_{k}_yoy", "trade", to3(SEC.yoy(w)), MONTH + lag * DAY, note=f"Eurostat {k}, y/y log change %, month + {lag} days; EU members only"))
    for nm, key, lag in (("x4_imf_energy_yoy", "imf_energy", IMF_LAG), ("x4_imf_metals_yoy", "imf_metals", IMF_LAG), ("x4_freight_yoy", "freight_ppi", FREIGHT_LAG),
                         ("x4_imf_food_yoy", "imf_food", IMF_LAG), ("x4_coal_yoy", "coal_aus", IMF_LAG), ("x4_imf_rawmat_yoy", "imf_rawmat", IMF_LAG)):
        if fid == "ip_long_m" and key in ("imf_food",):
            continue
        s = _fred(key)
        if s is not None:
            D.append(Driver(nm, "markets" if "freight" not in nm else "transport", (100 * np.log(s / s.shift(12))).to_frame("*"), MONTH + lag,
                            note=f"{key} (FRED, IMF / BLS), y/y log change %; IMF month + 28 days, freight PPI month + 35 days"))
    return D


def drivers(fid: str, P, PN) -> list:
    """The extra drivers of one family (PN = the panel module, passed in to avoid a circular import)."""
    if fid == "health_w":
        return health_drivers(PN)
    if fid in ("ip_long_m", "cpi_long_m"):
        return long_drivers(fid, PN)
    if fid in ("choke_w", "diesel_w"):
        c = HS.get_parquet("lab/raw/portwatch_chokepoints_daily.parquet")
        if fid == "choke_w":
            return choke_drivers(PN, sorted(set(c["portname"])))
        y = P.get("diesel")
        return diesel_drivers(PN, list(y.columns))
    return []


if __name__ == "__main__":
    if sys.argv[1:2] == ["load"]:
        print(load())
