"""Cross-layer links, new inputs for the thinly won layers.

    python -m forecasting_engine.lab.src_x2 wiki        # load the Wikipedia reading below into lab/raw/wiki_x2_daily.parquet (run on a VM: the shared proxy gets 429)

| Link (layer -> layer) | Families | Input | Real publication lag (worst case used) | Licence |
|---|---|---|---|---|
| information -> cyber | kev_d | Wikipedia pageviews of Ransomware / Zero-day / Cyberattack / Data breach (English), 7-day mean of log views | a day is served within 24 h; stamped at the day, known 2 days later | Wikimedia pageviews: CC0 |
| information -> society | unrest_w, security_w, info_w | Wikipedia pageviews of the strike article in the country's languages, Monday-Friday mean, week-on-week change | Monday-Friday of the week before the origin, known at the next Monday (src_health_lead.WIKI_LAG_H) | CC0 |
| information -> environment | wildfire_w | Wikipedia pageviews of the forest-fire article in the country's languages, same construction | same | CC0 |
| economy -> society | unrest_w, security_w, info_w | HICP food and energy y/y by country (Eurostat prct_hicp), 3-month change | month + 17 days (the HICP delay of the audit, panel.EUROSTAT_DAYS) | Eurostat: CC BY 4.0 |
| energy -> society | unrest_w, security_w, info_w | weekly mean ENTSO-E day-ahead price (log) | week end + 1 day | ENTSO-E: back-tests only |
| society -> society (neighbours) | unrest_w, security_w, info_w | the neighbours' 7-day GDELT strike / protest counts (log1p), population-weighted over land borders and short sea links | GDELT day + 1 day | GDELT: free use with citation |
| geopolitics -> cyber | kev_d | GDELT global share of material-conflict events (QuadClass 4), 7-day mean | day + 1 day | GDELT |
| vendor calendar -> cyber | kev_d | Oracle Critical Patch Update days (third Tuesday of Jan / Apr / Jul / Oct), the Chrome four-week stable release Tuesdays | published a year ahead (known ahead) | - |
| water, weather -> environment | wildfire_w | 30 / 60 / 90 day rain sums and 7-day temperature anomaly in the capital (NASA POWER), and the neighbours' maximum fire-weather index | POWER: 4 days (src_envdrivers.POWER_LAG_DAYS); FWI day + 4 days | NASA POWER and Copernicus: open |
| energy, economy -> transport | ports_m | HICP energy and food y/y, weekly ENTSO-E price mean as a monthly mean | month + 17 days / month + 1 day | as above |
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK = 24.0, 24.0 * 7
STORE = "lab/raw/wiki_x2_daily.parquet"
WIKI_LAG_W = 7 * DAY                       # Monday-Friday views, stamped Monday: known at the next Monday 00:00
WIKI_LAG_D = 2 * DAY                       # a day's views are served within 24 h; stamped at the day
TOPICS = {   # topic -> language -> candidate titles (a missing title answers 404 and is skipped)
    "cyber": {"en": ["Ransomware", "Zero-day (computing)", "Cyberattack", "Data breach"]},
    "strike": {"de": ["Streik"], "fr": ["Grève"], "es": ["Huelga"], "it": ["Sciopero"], "nl": ["Staking"], "pl": ["Strajk"], "en": ["Strike action"],
               "pt": ["Greve"], "el": ["Απεργία"], "cs": ["Stávka"], "sk": ["Štrajk"], "hu": ["Sztrájk"], "ro": ["Grevă"], "bg": ["Стачка"],
               "hr": ["Štrajk"], "sl": ["Stavka"], "et": ["Streik"], "lv": ["Streiks"], "lt": ["Streikas"], "fi": ["Lakko"], "sv": ["Strejk"],
               "da": ["Strejke"], "no": ["Streik"]},
    "fire": {"de": ["Waldbrand"], "fr": ["Incendie de forêt", "Feu de forêt"], "es": ["Incendio forestal"], "it": ["Incendio boschivo"],
             "pt": ["Incêndio florestal"], "el": ["Δασική πυρκαγιά"], "en": ["Wildfire"], "pl": ["Pożar lasu"], "cs": ["Lesní požár"],
             "hu": ["Erdőtűz"], "ro": ["Incendiu de pădure"], "bg": ["Горски пожар"], "hr": ["Šumski požar"], "fi": ["Metsäpalo"],
             "sv": ["Skogsbrand"], "da": ["Skovbrand"], "no": ["Skogbrann"], "sk": ["Lesný požiar"], "sl": ["Gozdni požar"]},
}
CALENDAR_AHEAD_H = -365 * DAY


def wiki_load(log=print, pause: float = 1.5) -> dict:
    from forecasting_engine.lab import src_health_lead as SHL
    b = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    rows = []
    for topic, per in TOPICS.items():
        for lang, titles in per.items():
            tot = None
            for t in titles:
                s = SHL._pv(lang, t, b)
                time.sleep(pause)
                if len(s):
                    tot = s if tot is None else tot.add(s, fill_value=0)
            if tot is None:
                log(f"wiki {topic} {lang}: no title found")
                continue
            rows.append(pd.DataFrame({"date": tot.index, "lang": lang, "topic": topic, "views": tot.to_numpy()}))
            log(f"wiki {topic} {lang}: {len(tot)} days, median {tot.median():.0f} views/day")
    d = pd.concat(rows)
    HS.put_parquet(STORE, d.set_index("date"))
    return dict(series=len(rows), rows=len(d))


def _wiki_daily(topic: str) -> pd.DataFrame | None:
    d = HS.get_parquet(STORE)
    if d is None:
        return None
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[d["topic"] == topic]
    return d.pivot_table(index="date", columns="lang", values="views", aggfunc="sum").sort_index() if len(d) else None


def wiki_weekly(topic: str) -> pd.DataFrame | None:
    """Per country: Monday-Friday mean of log(1 + views) per week (stamped Monday), the country's languages averaged."""
    from forecasting_engine.lab import src_health_lead as SHL
    x = _wiki_daily(topic)
    if x is None:
        return None
    x = x[x.index.dayofweek < 5]
    lx = np.log1p(x).resample("W-MON", label="left", closed="left").mean()
    n = x.notna().resample("W-MON", label="left", closed="left").sum()
    lx = lx.where(n >= 4)
    return pd.DataFrame({cc: lx.reindex(columns=[l for l in ls if l in lx.columns]).mean(axis=1, skipna=False) for cc, ls in SHL.LANGS.items()
                         if any(l in lx.columns for l in ls)}).sort_index()


def wiki_cyber_daily() -> dict[str, pd.DataFrame]:
    x = _wiki_daily("cyber")
    if x is None:
        return {}
    lx = np.log1p(x["en"]).rename("*").to_frame()
    r7 = lx.rolling(7, min_periods=5).mean()
    return {"x2_wiki_cyber_7d": r7, "x2_wiki_cyber_rel": r7 - r7.shift(1).rolling(90, min_periods=45).median()}


def _third_tuesdays(months=(1, 4, 7, 10)) -> list[pd.Timestamp]:
    out = []
    for y in range(2015, 2028):
        for m in months:
            d = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
            d += pd.Timedelta(days=(1 - d.dayofweek) % 7 + 14)
            out.append(d)
    return out


def vendor_calendar(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    cpu = pd.Series(0.0, index=index)
    for d in _third_tuesdays():
        if d in cpu.index:
            cpu[d] = 1.0
    anchor = pd.Timestamp("2025-06-24", tz="UTC")               # Chrome 138 stable; four-week cycle (the year-end gap is ignored)
    chrome = pd.Series(((index - anchor).days % 28 == 0).astype(float), index=index)
    return {"x2_oracle_cpu_day": cpu.to_frame("*"), "x2_chrome_release_day": chrome.to_frame("*")}


def gdelt_conflict_share() -> pd.DataFrame | None:
    """Global share of QuadClass 4 (material conflict) events among all events, 7-day mean, per day (by the day events were added)."""
    d = HS.get_parquet("lab/raw/gdelt_actor_quad.parquet")
    if d is None:
        return None
    d = d.reset_index()
    d = d[d["k"].astype(str).str.startswith("q:")]
    d["date"] = pd.to_datetime(d["date"], utc=True)
    g = d.groupby(["date", "k"])["n"].sum().unstack()
    if "q:4" not in g.columns:
        return None
    sh = (g["q:4"] / g.sum(axis=1)).rolling(7, min_periods=5).mean()
    return sh.to_frame("*")


def hicp_links() -> dict[str, pd.DataFrame]:
    """HICP food and energy: year-on-year and 3-month change of the index, per country, stamped at the reference month."""
    from forecasting_engine.lab import panel as PN
    x = HS.get_parquet("lab/raw/estat_hicp_comp.parquet")
    if x is None:
        return {}
    x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if pd.DatetimeIndex(x.index).tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
    out = {}
    for comp, tag in (("FOOD", "food"), ("NRG", "nrg")):
        cols = {c.split("|")[0]: c for c in x.columns if c.endswith("|" + comp)}
        w = pd.DataFrame({PN._geo(cc): x[c] for cc, c in cols.items()}).sort_index()
        out[f"x2_hicp_{tag}_yoy"] = (w / w.shift(12) - 1) * 100
        out[f"x2_hicp_{tag}_3m"] = (w / w.shift(3) - 1) * 100
    return out


def rain_temp_links(ccs=None) -> dict[str, pd.DataFrame]:
    """Capital rain sums (30 / 60 / 90 days) and the 7-day temperature anomaly against the same days of the earlier years (causal)."""
    d = HS.get_parquet("lab/raw/capital_power_daily.parquet")
    if d is None:
        return {}
    d.index = pd.DatetimeIndex(d.index).tz_convert("UTC") if pd.DatetimeIndex(d.index).tz is not None else pd.DatetimeIndex(d.index).tz_localize("UTC")
    rain = d[[c for c in d.columns if c.startswith("rain:")]].astype(float)
    rain.columns = [c.split(":", 1)[1] for c in rain.columns]
    t = d[[c for c in d.columns if c.startswith("t2m:")]].astype(float)
    t.columns = [c.split(":", 1)[1] for c in t.columns]
    out = {f"x2_rain_{k}d": rain.rolling(k, min_periods=k).sum() for k in (30, 60, 90)}
    r7 = t.rolling(7, min_periods=7).mean()
    doy = r7.index.dayofyear
    clim = r7.groupby(doy).transform(lambda s: s.shift(1).expanding(min_periods=3).mean())   # earlier years only
    out["x2_t2m_anom7"] = r7 - clim
    return out


def fwi_daily() -> pd.DataFrame | None:
    d = HS.get_parquet("lab/raw/fwi_country_daily.parquet")
    if d is None:
        return None
    d.index = pd.DatetimeIndex(d.index).tz_convert("UTC") if pd.DatetimeIndex(d.index).tz is not None else pd.DatetimeIndex(d.index).tz_localize("UTC")
    x = d[[c for c in d.columns if c.startswith("fwi:")]].astype(float)
    x.columns = [c.split(":", 1)[1] for c in x.columns]
    return x


def shipping_calendar() -> dict[str, pd.DataFrame]:
    """Weekly (Monday-stamped) holiday features for the shipping lanes: weeks from the Chinese New Year, and the public-holiday days of the week in
    China, the US, Germany and Japan (the busiest cargo and consumer ends of the chokepoints). Known years ahead."""
    import holidays as HL
    weeks = pd.date_range(pd.Timestamp("2015-01-05", tz="UTC"), pd.Timestamp("2027-12-27", tz="UTC"), freq="W-MON")
    cny = []
    for y in range(2014, 2029):
        h = HL.country_holidays("CN", years=y)
        ds = sorted(d for d, n in h.items() if "春节" in n)
        if ds:
            cny.append(pd.Timestamp(ds[0], tz="UTC"))
    cny = pd.DatetimeIndex(cny)
    rel = pd.Series([(float(np.clip(((w - cny[np.argmin(np.abs((cny - w).days))]).days) / 7.0, -8, 10))) for w in weeks], index=weeks)
    out = {"x2_cny_weeks": rel.to_frame("*")}
    for cc in ("CN", "US", "DE", "JP"):
        h = HL.country_holidays(cc, years=range(2015, 2029))
        days = pd.Series(1.0, index=pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in h]))
        wk = days.resample("W-MON", label="left", closed="left").sum().reindex(weeks).fillna(0.0)
        out[f"x2_hol_{cc.lower()}"] = wk.to_frame("*")
    return out


def drivers(fid: str, P, PN) -> list:
    """The extra drivers of one family (PN = the panel module, passed in to avoid a circular import)."""
    from forecasting_engine.lab import src_envdrivers as SE
    Driver, D = PN.Driver, []
    W = PN.health_weights()
    social = fid in ("unrest_w", "security_w", "info_w")
    if social or fid == "wildfire_w":
        topic = "fire" if fid == "wildfire_w" else "strike"
        w = wiki_weekly(topic)
        if w is not None:
            nt = f"Wikipedia pageviews (CC0) of the {'forest-fire' if topic == 'fire' else 'strike'} article in the country's language(s), Monday-Friday mean of log views"
            D += [Driver(f"x2_wiki_{topic}_chg", "information", w.diff().clip(-2, 2), WIKI_LAG_W, note=nt + ": week-on-week change"),
                  Driver(f"x2_wiki_{topic}_rel", "information", w - w.shift(1).rolling(52, min_periods=26).median(), WIKI_LAG_W,
                         note=nt + ": minus the median of the 52 weeks before")]
    if social:
        for nm, x in hicp_links().items():
            D.append(Driver(nm, "economy", x, PN.MONTH + PN.EUROSTAT_DAYS["hicp"] * DAY, note="Eurostat HICP " + ("food" if "food" in nm else "energy") + ", " + nm.rsplit("_", 1)[1]))
        pr = P.get("price")
        D.append(Driver("x2_elec_price_w", "energy", np.log(pr.where(pr > 1)).resample("W-MON", label="left", closed="left").mean(), WEEK + DAY,
                        note="ENTSO-E day-ahead price, weekly mean of log (restricted: back-tests only)"))
        for kind in ("protest", "strike"):
            g = np.log1p(P.get(kind).rolling(7, min_periods=5).mean())
            D.append(Driver(f"x2_nb_{kind}_7d", "network", PN._nb_avail(g, W), DAY + 1,
                            note=f"neighbours' GDELT {kind} events, 7-day mean of log1p, population-weighted over borders and short sea links"))
    if fid == "kev_d":
        for nm, x in wiki_cyber_daily().items():
            D.append(Driver(nm, "information", x, WIKI_LAG_D, note="Wikipedia pageviews (CC0) of Ransomware, Zero-day, Cyberattack, Data breach (English), " + nm.rsplit("_", 1)[1]))
        g = gdelt_conflict_share()
        if g is not None:
            D.append(Driver("x2_gdelt_conflict_7d", "society", g, DAY + 1, note="GDELT: global share of material-conflict events, 7-day mean"))
        idx = pd.date_range(pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2027-12-31", tz="UTC"), freq="D")
        for nm, x in vendor_calendar(idx).items():
            D.append(Driver(nm, "calendar", x, CALENDAR_AHEAD_H, ahead=True, note="vendor release calendar, published a year ahead"))
    if fid == "wildfire_w":
        for nm, x in rain_temp_links().items():
            D.append(Driver(nm, "water" if "rain" in nm else "weather", x, DAY + SE.POWER_LAG_DAYS * DAY,
                            note="NASA POWER at the capital: " + ("rain sum" if "rain" in nm else "7-day temperature anomaly vs earlier years")))
        f = fwi_daily()
        if f is not None:
            fx = f.rolling(7, min_periods=5).max()
            D.append(Driver("x2_nb_fwi_max7", "network", PN._nb_avail(fx, W), DAY + 4 * DAY,
                            note="neighbours' 7-day maximum fire-weather index (Copernicus CEMS), population-weighted; FWI served within 4 days"))
    if fid == "choke_w":
        for nm, x in shipping_calendar().items():
            D.append(Driver(nm, "calendar", x, CALENDAR_AHEAD_H, ahead=True, note="public-holiday calendars of the big trading nations (the `holidays` package), published years ahead"))
    if fid == "ports_m":
        for nm, x in hicp_links().items():
            if "3m" in nm:
                continue
            D.append(Driver(nm, "economy", x, PN.MONTH + PN.EUROSTAT_DAYS["hicp"] * DAY, note="Eurostat HICP " + ("food" if "food" in nm else "energy") + " y/y"))
    return D


if __name__ == "__main__":
    what = sys.argv[1:] or ["wiki"]
    if "wiki" in what:
        print(wiki_load())
    sys.exit(0)
