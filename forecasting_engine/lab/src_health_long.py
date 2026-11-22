"""Long-history health inputs for health_w: the seasons before ERVISS (2022-W25) as drivers.

ERVISS starts in mid-2022, so health_w's trees train only from 2024-08 and see two seasons. The registered target cannot be extended backwards
(the windows are locked), so the older seasons enter as drivers: each country's typical respiratory season shape, learned from open series
that go back before 2022, given at the TARGET week (an ahead driver: a calendar fact known years before) and at the origin's week.

    python -m forecasting_engine.lab.src_health_long          # print a summary of the climatologies

| Driver | Source (already in the history store) | Seasons used | Licence |
|---|---|---|---|
| clim_wiki_flu, clim_wiki_cold (+ _now, _slope) | lab/raw/wiki_health_daily.parquet: Wikipedia pageviews of the influenza / common cold articles in each country's languages (src_health_lead) | 2015/16 - 2019/20 (to 2020-02, before COVID-19 changed both reading and illness) | CC0 1.0 (Wikimedia) |
| clim_are, clim_are_child (+ _now, _slope) | lab/raw/rki_are_age.parquet: RKI ARE-Konsultationsinzidenz, Germany, all ages and 0-14 | 2012/13 - 2019/20 | CC BY 4.0 ("Robert Koch-Institut: ARE-Konsultationsinzidenz") |
| clim_fi_flu (+ _now, _slope) | lab/raw/thl_flu_week.parquet: THL Finland, Infectious Diseases Register, laboratory-confirmed influenza A + B per week | 2018 - 2019/20 | CC BY 4.0 ("Terveyden ja hyvinvoinnin laitos (THL), tartuntatautirekisteri") |

Each climatology: per series, log of the weekly value minus that season's mean log (a season runs ISO week 27 to week 26), averaged over the
pre-2020 seasons per ISO week (week 53 takes week 52), then smoothed over 3 weeks (circular). The value at any date is a function of its ISO
week only, built from data published before March 2020, so it is known long before every health_w origin (stamped known 365 days before;
ahead=True gives the value at each target week). The *_now drivers are the same value at the origin's week (known at the origin) and
*_slope the climatological change from one week to the next; trees combine them with the current level. The German ARE shape is also
given as an EU-wide column ("*") for the countries without a long series of their own: respiratory seasons in Europe peak within a few weeks.

Checked and NOT used (licence): WHO FluNet / FluID (xmart-api-public.who.int FLUMART views, 1997- / 2010-): the downloads sit on who.int, whose
"Terms and conditions of use for WHO data compilations" limit use to "public health purposes" and bar use "in conjunction with the promotion of
a commercial enterprise and/or its product(s) or service(s)"; only datasets on data.who.int are CC BY 4.0 and FluNet / FluID are not listed
there. France Sentinelles (sentiweb.fr, 1984-): the site says "All rights reserved" and states no open licence. Both: ask before any use.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

CLIM_END = pd.Timestamp("2020-03-01", tz="UTC")          # seasons before COVID-19 only
KNOWN_BEFORE_H = -365 * 24.0                              # ahead: a week's climatology is known a year before it (built from pre-2020 data)
THL_URL = "https://sampo.thl.fi/pivot/prod/fi/ttr/casesweek/fact_ttr_casesweek.csv"
ATTR = {"wiki": "Wikimedia Foundation pageview data (CC0). Seasonal climatology by the forecasting engine.",
        "are": "Robert Koch-Institut: ARE-Konsultationsinzidenz, CC BY 4.0. Seasonal climatology by the forecasting engine.",
        "thl": "Terveyden ja hyvinvoinnin laitos (THL), tartuntatautirekisteri, CC BY 4.0. Weekly sum of influenza A and B, adapted by the forecasting engine."}


def _utc(x: pd.DataFrame) -> pd.DataFrame:
    ix = pd.DatetimeIndex(x.index)
    x.index = ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")
    return x.sort_index()


def _isoweek(ix: pd.DatetimeIndex) -> np.ndarray:
    return np.minimum(ix.isocalendar().week.to_numpy(int), 52)


def climatology(lw: pd.DataFrame, end: pd.Timestamp = CLIM_END, min_seasons: int = 2) -> pd.DataFrame:
    """lw: weekly log values (Monday stamps) per column -> 52 x columns: mean over seasons of (log value - the season's mean log), per ISO week,
    smoothed over 3 weeks (circular). A season runs from ISO week 27 to week 26; seasons with fewer than 40 weeks are dropped."""
    x = lw[lw.index < end]
    wk = _isoweek(x.index)
    season = x.index.year - (wk < 27)
    out = {}
    for c in x.columns:
        s = pd.DataFrame({"v": x[c].to_numpy(float), "wk": wk, "se": season}).dropna()
        n = s.groupby("se")["v"].transform("size")
        s = s[n >= 40]
        if s["se"].nunique() < min_seasons:
            continue
        s["a"] = s["v"] - s.groupby("se")["v"].transform("mean")
        m = s.groupby("wk")["a"].mean().reindex(range(1, 53)).interpolate(limit_direction="both")
        v = m.to_numpy()
        out[c] = (np.roll(v, 1) + v + np.roll(v, -1)) / 3
    return pd.DataFrame(out, index=range(1, 53))


def on_mondays(cl: pd.DataFrame, start: str = "2010-01-04", end: str = "2028-12-25") -> pd.DataFrame:
    """The climatology spread over every Monday (UTC) between start and end."""
    ix = pd.date_range(start, end, freq="W-MON", tz="UTC")
    return pd.DataFrame(cl.loc[_isoweek(ix)].to_numpy(), index=ix, columns=cl.columns)


def _wiki_weekly(topic: str) -> pd.DataFrame | None:
    from forecasting_engine.lab import src_health_lead as SHL
    w = SHL.wiki_weekly(topic)
    return None if w is None else _utc(w)


def _are_weekly() -> pd.DataFrame | None:
    w = HS.get_parquet("lab/raw/rki_are_age.parquet")
    if w is None:
        return None
    w = _utc(w)
    lw = np.log(w.where(w > 0))
    return pd.DataFrame({"all": lw["00+"], "child": lw[["0-4", "5-14"]].mean(axis=1)})


# ------------------------------------------------------------------------------------------------------------------------------ THL Finland
THL_GROUPS = {"Influenssa": "flu", "Influenssa A": "flu_a", "Influenssa B": "flu_b", "RSV": "rsv", "Rinovirus": "rhino",
              "Metapneumovirus": "hmpv", "Adenovirus": "adeno", "Parainfluenssavirus": "parainfl", "Mykoplasma": "myco",
              "Koronavirus SARS-CoV-2": "sars2", "Hinkuyskä": "pertussis", "Pneumokokki": "pneumococcus"}
THL_LAG_H = 8 * 24.0             # the cube is refreshed every day: week W (Mon-Sun) counted by Tuesday 00:00 after it (Monday stamp + 8 days)


def thl_load(log=print) -> dict:
    """THL Infectious Diseases Register, weekly laboratory-confirmed cases, whole country (cube ttr/casesweek, from 2018-W01). The row
    'yearweek-<a week's sid>L' expands to every week, the column 'nidrreportgroup-<a group's sid>L' to every report group."""
    from forecasting_engine.lab.sources import _get
    r = _get(THL_URL, params=[("row", "yearweek-1201465L"), ("column", "nidrreportgroup-877942L"), ("filter", "measure-877837")])
    t = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), sep=";")
    t.columns = ["group", "week", "value"]
    t = t[t["group"].isin(THL_GROUPS)]
    yw = t["week"].astype(str).str.extract(r"Vuosi (\d{4}) Viikko (\d{1,2})")
    t = t.assign(date=pd.to_datetime(yw[0] + "-W" + yw[1].str.zfill(2) + "-1", format="%G-W%V-%u", utc=True, errors="coerce"),
                 value=pd.to_numeric(t["value"], errors="coerce"), group=t["group"].map(THL_GROUPS)).dropna(subset=["date"])
    w = t.pivot_table(index="date", columns="group", values="value", aggfunc="sum").sort_index()
    w = w[w.index < pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=7)]   # drop the running week
    w.index.name = "date"
    HS.put_parquet("lab/raw/thl_flu_week.parquet", w)
    log(f"THL weekly cases: {w.shape}, {w.index.min().date()} - {w.index.max().date()}")
    return dict(weeks=len(w), groups=list(w.columns), first=str(w.index.min().date()), last=str(w.index.max().date()))


def thl_weekly() -> pd.DataFrame | None:
    """log(1 + weekly cases) per report group (columns as THL_GROUPS values)."""
    try:
        w = HS.get_parquet("lab/raw/thl_flu_week.parquet")
    except Exception:  # noqa: BLE001
        return None
    return None if w is None else np.log1p(_utc(w).clip(lower=0))


def _thl_weekly() -> pd.DataFrame | None:
    w = thl_weekly()
    return None if w is None or "flu" not in w.columns else w[["flu"]].rename(columns={"flu": "FI"})


# ------------------------------------------------------------------------------------------------------------------------------ drivers
def features() -> dict[str, tuple[pd.DataFrame, float, bool, str]]:
    """name -> (frame on Mondays, avail_h, ahead, note). Appended to health_w in this order (never reorder: the champion keeps its first N)."""
    out = {}

    def put(nm, cl, note):
        if cl is None or cl.empty:
            return
        f = on_mondays(cl)
        out[f"clim_{nm}"] = (f, KNOWN_BEFORE_H, True, f"{note}: typical seasonal log level at the target week (pre-2020 seasons)")
        out[f"clim_{nm}_now"] = (f, 0.0, False, f"{note}: the same at the origin's week")
        out[f"clim_{nm}_slope"] = (f.diff().bfill(), KNOWN_BEFORE_H, True, f"{note}: typical change from the week before, at the target week")

    for topic in ("flu", "cold"):
        w = _wiki_weekly(topic)
        if w is not None:
            put(f"wiki_{topic}", climatology(w), f"Wikipedia {topic} reading (CC0), per country language(s), 2015/16-2019/20")
    a = _are_weekly()
    if a is not None:
        ca = climatology(a)
        if "all" in ca:
            put("are", pd.DataFrame({"*": ca["all"]}), "RKI ARE consultations, Germany, all ages (CC BY 4.0), 2012/13-2019/20, EU-wide shape")
        if "child" in ca:
            put("are_child", pd.DataFrame({"*": ca["child"]}), "RKI ARE consultations, Germany, ages 0-14 (CC BY 4.0), EU-wide shape")
    t = _thl_weekly()
    if t is not None:
        put("fi_flu", climatology(t, min_seasons=2), "THL Finland laboratory-confirmed influenza (CC BY 4.0), 2018-2019/20")
    w = thl_weekly()
    if w is not None:                                     # Finland's pathogen mix, weekly (a finer grain than ERVISS's three pathogens)
        nt = "THL Finland, Infectious Diseases Register (CC BY 4.0), laboratory-confirmed cases per week, log(1 + n)"
        for g in ("flu", "rsv", "rhino", "hmpv", "myco", "parainfl", "adeno"):
            if g in w.columns:
                x = w[[g]].rename(columns={g: "FI"})
                out[f"fi_{g}"] = (x, THL_LAG_H, False, f"{nt}: {g}")
                out[f"fi_{g}_logchg"] = (x.diff(), THL_LAG_H, False, f"{nt}: {g}, week-on-week change")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["thl"]:
        print(thl_load())
        sys.exit(0)
    for k, (f, av, ah, nt) in features().items():
        print(k, f.shape, "ahead" if ah else "", round(float(np.nanstd(f.to_numpy())), 3), nt[:80])
    sys.exit(0)
