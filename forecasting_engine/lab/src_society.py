"""Society, security and information inputs for unrest_w, security_w and info_w.

    python -m forecasting_engine.lab.src_society load DIR     # DIR: the BigQuery export of build_sql() (parquet parts)

| Store name | Content | Delay | Licence |
|---|---|---|---|
| lab/raw/gdelt_actor_quad.parquet | GDELT 2.0 events per day and country (ActionGeo): counts and articles by actor type (either actor: LAB, EDU, MED, AGR, OPP, CVL, COP, MIL, REB, GOV, ...) and event root (10-20 kept, rest 'x'), and by QuadClass, by the day the events were ADDED (DATEADDED, UTC; since 21 Oct 2026, so later additions never revise a past day) | end of day + 1 h, the same as the GDELT events already in the lab | GDELT: free use, including commercial, with citation ("GDELT Project, gdeltproject.org") |
| (computed) public holidays per country and week | the `holidays` package (MIT code; the dates are the laws' own facts) | known years ahead | - |

Every driver is a 7-day rolling sum (or mean) of the daily values stamped at the day it ends, so a weekly family sees the last full week known
at its origin, not one noisy day. Counts are shares of all the country's events in the same 7 days (per 1000): GDELT's volume drifts.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

DAY, WEEK = 24.0, 24.0 * 7
ATTR = "GDELT Project (gdeltproject.org), GDELT 2.0 Event Database"
STORE = __import__("os").environ.get("FE_GDELT_AQ", "lab/raw/gdelt_actor_quad.parquet")   # before 21 Oct 2026 (SQLDATE): lab/raw/gdelt_actor_quad_sqldate.parquet
ACTORS = ("LAB", "EDU", "MED", "GOV", "MIL", "COP", "REB", "OPP", "CVL", "BUS", "REF", "CRM", "JUD", "LEG", "AGR", "SPY", "UAF", "ELI", "MOD", "RAD")


def build_sql() -> str:
    from forecasting_engine.lab.sources import GDELT_COUNTRIES, GDELT_DAY
    cc = ",".join(f"'{c}'" for c in GDELT_COUNTRIES)
    act = ",".join(f"'{a}'" for a in ACTORS)
    return f"""WITH e AS (
  SELECT {GDELT_DAY} AS d, ActionGeo_CountryCode AS cc, Actor1Type1Code AS a1, Actor2Type1Code AS a2, QuadClass AS q, EventRootCode AS root, NumArticles AS art
  FROM `gdelt-bq.gdeltv2.events_partitioned` WHERE _PARTITIONTIME >= TIMESTAMP('2015-02-18') AND ActionGeo_CountryCode IN ({cc}))
SELECT d, cc, CONCAT('a:', t, ':', IF(root IN ('14','18','19','20','16','17','13','10','11','12'), root, 'x')) AS k, COUNT(*) AS n, SUM(art) AS art FROM (
  SELECT d, cc, root, art, a1 AS t FROM e WHERE a1 IN ({act})
  UNION ALL SELECT d, cc, root, art, a2 AS t FROM e WHERE a2 IN ({act}) AND (a1 IS NULL OR a1 != a2)) GROUP BY d, cc, k
UNION ALL SELECT d, cc, CONCAT('q:', CAST(q AS STRING)), COUNT(*), SUM(art) FROM e GROUP BY d, cc, q"""


def load(path: str, log=print) -> dict:
    """The BigQuery export (bq query --destination_table ... then bq extract --destination_format=PARQUET) -> the store."""
    import glob
    from forecasting_engine.lab.sources import GDELT_COUNTRIES
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{path}/*.parquet"))])
    d["cc"] = d["cc"].map(GDELT_COUNTRIES)
    d = d.dropna(subset=["cc"])
    d["date"] = pd.to_datetime(d["d"].astype(str), format="%Y%m%d", utc=True, errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[(d["date"] >= "2015-02-18")][["date", "cc", "k", "n", "art"]]
    d["n"], d["art"] = d["n"].astype("int32"), d["art"].astype("int64")
    HS.put_parquet(STORE, d.set_index("date"))
    out = dict(rows=len(d), keys=int(d["k"].nunique()), first=str(d["date"].min().date()), last=str(d["date"].max().date()))
    log(out)
    return out


_M: dict = {}


def _raw() -> pd.DataFrame:
    if "aq" not in _M:
        d = HS.get_parquet(STORE).reset_index()
        d["date"] = pd.to_datetime(d["date"], utc=True)
        _M["aq"] = d
    return _M["aq"]


def _daily(keys, value: str = "n") -> pd.DataFrame:
    d = _raw()
    d = d[d["k"].isin(keys)]
    w = d.groupby(["date", "cc"])[value].sum().unstack().sort_index()
    full = pd.date_range(pd.Timestamp("2015-02-18", tz="UTC"), _raw()["date"].max(), freq="D")
    return w.reindex(full).fillna(0.0)


def _roll(w: pd.DataFrame) -> pd.DataFrame:
    return w.rolling(7, min_periods=7).sum()


def _total() -> pd.DataFrame:
    if "tot" not in _M:
        _M["tot"] = _roll(_daily([f"q:{q}" for q in (1, 2, 3, 4)]))
    return _M["tot"]


def _share(keys) -> pd.DataFrame:
    tot = _total()
    x = _roll(_daily(keys)).reindex(columns=tot.columns)
    return (x / tot.where(tot > 0) * 1000).replace([np.inf, -np.inf], np.nan)


def _ak(actors, roots) -> list[str]:
    return [f"a:{a}:{r}" for a in actors for r in roots]


VIOL = ("18", "19", "20")
ALL_R = ("10", "11", "12", "13", "14", "16", "17", "18", "19", "20", "x")


def _root_share(roots) -> pd.DataFrame:
    """Shares of CAMEO roots from the original GDELT export (lab.sources.gdelt_frame), 7-day rolling."""
    from forecasting_engine.lab import sources as SR
    g = SR.gdelt_frame()
    w = g[g["root"].isin(roots)].groupby(["date", "cc"])["n"].sum().unstack().sort_index()
    tot = _total()
    w = _roll(w.reindex(tot.index).fillna(0.0)).reindex(columns=tot.columns)
    return w / tot.where(tot > 0) * 1000


def _goldstein() -> pd.DataFrame:
    from forecasting_engine.lab import sources as SR
    g = SR.gdelt_frame().assign(w=lambda x: x["goldstein"] * x["n"])
    s = g.groupby(["date", "cc"])[["w", "n"]].sum()
    tot = _total()
    w = _roll(s["w"].unstack().reindex(tot.index).fillna(0.0))
    n = _roll(s["n"].unstack().reindex(tot.index).fillna(0.0))
    return (w / n.where(n > 0)).reindex(columns=tot.columns)


def _attention(kind: str) -> pd.DataFrame:
    """Mentions per event of the kind (how much coverage each strike or protest gets), 7-day rolling."""
    from forecasting_engine.lab import sources as SR
    g = SR.gdelt_frame()
    g = g[g["kind"] == kind]
    s = g.groupby(["date", "cc"])[["mentions", "n"]].sum()
    tot = _total()
    m = _roll(s["mentions"].unstack().reindex(tot.index).fillna(0.0))
    n = _roll(s["n"].unstack().reindex(tot.index).fillna(0.0))
    return (m / n.where(n > 0)).reindex(columns=tot.columns)


def holidays_weekly(ccs) -> pd.DataFrame:
    """Public holidays (national) falling Monday-Sunday in each week, per country, stamped at the week's Monday (known years ahead)."""
    import holidays as HL
    idx = pd.date_range(pd.Timestamp("2015-01-05", tz="UTC"), pd.Timestamp("2027-12-27", tz="UTC"), freq="W-MON")
    out = {}
    for cc in ccs:
        try:
            h = HL.country_holidays(cc, years=range(2015, 2028))
        except Exception:  # noqa: BLE001
            continue
        days = pd.DatetimeIndex(sorted(pd.Timestamp(x) for x in h.keys())).tz_localize("UTC")
        days = days[days.dayofweek < 5]                       # weekday holidays only (a Sunday holiday changes little)
        mon = days - pd.to_timedelta(days.dayofweek, unit="D")
        out[cc] = pd.Series(1.0, index=mon).groupby(level=0).sum().reindex(idx, fill_value=0.0)
    return pd.DataFrame(out, index=idx)


def drivers(fid: str, ccs) -> list:
    """The new drivers for unrest_w / security_w / info_w (appended after the registered ones, in this order)."""
    from forecasting_engine.lab.panel import Driver
    ccs = list(ccs)
    av = DAY + 1
    nt = f"{ATTR}; 7-day rolling share of the country's events, per 1000"
    sh = lambda k: _share(k).reindex(columns=ccs)          # noqa: E731
    D = [Driver("gd_lab_any", "society", sh(_ak(("LAB",), ALL_R)), av, note=f"labour actors, any event; {nt}"),
         Driver("gd_lab_protest", "society", sh(_ak(("LAB",), ("14",))), av, note=f"labour actors protesting; {nt}"),
         Driver("gd_agr_protest", "society", sh(_ak(("AGR",), ("14",))), av, note=f"farmers protesting; {nt}"),
         Driver("gd_edu_med_protest", "society", sh(_ak(("EDU", "MED"), ("14",))), av, note=f"education and health actors protesting; {nt}"),
         Driver("gd_civil_protest", "society", sh(_ak(("OPP", "CVL", "MOD", "RAD"), ("14",))), av, note=f"opposition and civilians protesting; {nt}"),
         Driver("gd_police_coerce", "security", sh(_ak(("COP",), ("16", "17", "18", "19", "20"))), av, note=f"police coercion and force; {nt}"),
         Driver("gd_mil_violence", "security", sh(_ak(("MIL",), VIOL)), av, note=f"military actors in violent events; {nt}"),
         Driver("gd_reb_violence", "security", sh(_ak(("REB", "UAF", "RAD"), VIOL)), av, note=f"rebels and armed groups in violent events; {nt}"),
         Driver("gd_crime", "security", sh(_ak(("CRM",), ALL_R)), av, note=f"criminal actors; {nt}"),
         Driver("gd_gov_threat", "policy", sh(_ak(("GOV", "LEG", "JUD"), ("10", "12", "13"))), av, note=f"government demands, rejections, threats; {nt}"),
         Driver("gd_quad_verbal_conflict", "society", sh(["q:3"]), av, note=f"QuadClass 3; {nt}"),
         Driver("gd_quad_material_conflict", "security", sh(["q:4"]), av, note=f"QuadClass 4; {nt}"),
         Driver("gd_quad_material_coop", "society", sh(["q:2"]), av, note=f"QuadClass 2; {nt}"),
         Driver("gd_demand", "society", _root_share(["10"]).reindex(columns=ccs), av, note=f"CAMEO 10 demand; {nt}"),
         Driver("gd_disapprove", "society", _root_share(["11"]).reindex(columns=ccs), av, note=f"CAMEO 11 disapprove; {nt}"),
         Driver("gd_threaten", "security", _root_share(["13"]).reindex(columns=ccs), av, note=f"CAMEO 13 threaten; {nt}"),
         Driver("gd_force_posture", "security", _root_share(["15"]).reindex(columns=ccs), av, note=f"CAMEO 15 force posture; {nt}"),
         Driver("gd_goldstein_7d", "society", _goldstein().reindex(columns=ccs), av, note=f"{ATTR}; mean Goldstein scale, 7 days"),
         Driver("gd_protest_attention", "information", _attention("protest").reindex(columns=ccs), av, note=f"{ATTR}; mentions per protest event, 7 days"),
         Driver("gd_strike_attention", "information", _attention("strike").reindex(columns=ccs), av, note=f"{ATTR}; mentions per strike event, 7 days"),
         Driver("gd_volume_7d", "information", np.log1p(_total().reindex(columns=ccs)), av, note=f"{ATTR}; log events in 7 days (coverage volume)"),
         Driver("holidays_week", "calendar", holidays_weekly(ccs), -365 * DAY, ahead=True,
                note="weekday public holidays in the target week (holidays package; laws' own dates), known years ahead")]
    return D


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    if a and a[0] == "load":
        load(a[1])
    elif a and a[0] == "sql":
        print(build_sql())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
