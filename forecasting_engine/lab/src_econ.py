"""Leading inputs for the economy and trade families (ip_m, retail_m, unemp_m, hicp_m, ports_m).

Everything here is Eurostat (reuse policy: CC BY 4.0 equivalent, commercial reuse allowed, credit "Source: Eurostat"; the business and consumer
surveys are the European Commission's DG ECFIN surveys redistributed by Eurostat, same policy). Each frame is stored long (date, geo, value) in
lab/raw/econ_<key>.parquet and enters the panel as a driver (panel.extra_drivers -> drivers()).

    python -m forecasting_engine.lab.src_econ load            # fetch everything (about 30 requests)
    python -m forecasting_engine.lab.src_econ extend ip_m retail_m unemp_m hicp_m ports_m

Timing (stamp = first day of the reference month; avail = delay from the stamp, checked on the dataset "updated" dates of 16-17 Oct 2026):
| Key | Dataset | Content | Delay used |
|---|---|---|---|
| survey balances | ei_bsin_m_r2, ei_bsco_m, ei_bsrt_m_r2, ei_bsse_m_r2, ei_bsbu_m_r2 (SA balances) | sector surveys: production, order books, employment, price expectations, consumers' unemployment / price / purchase expectations | 1 month (released the last working days of the reference month; September 2026 was up on 29 Sep) |
| reg / bkrt | sts_rb_m (I21, SCA, B-S_X_O_S94) | business registrations, bankruptcy declarations, y/y % | registrations month + 45 days, bankruptcies + 95 (on 20 Oct the slowest reporters had registrations to Aug, bankruptcies to Jul (DE) and Jun (PT)) |
| constr | sts_copr_m (I21, SCA, F) | construction production, y/y % | month + 60 days (on 20 Oct HR, IT, LU, RO had only July) |
| svc | sts_setu_m (NETTUR, I21, SCA, H-N_X_K / H / H50 / H52) | services and transport/warehousing turnover, y/y % | month + 100 days (on 20 Oct most countries had July, CY and SE June) |
| trade | ei_eteu27_2020_m (IVOL-SA, partner WORLD and EXT_EU27_2020) | export / import volume, y/y %; extra-EU unit values y/y | month + 80 days (on 20 Oct the volume indices stopped at July for every country, although August values were out on 16 Oct) |
| imp_prices | sts_inpi_m (PRC_IMP, CPA_B-D and energy MIG, I21) | industrial import prices, y/y % | month + 35 days |

Values are the latest vintage (as for every Eurostat input in the lab): revisions flatter the back-test a little.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

DAY, MONTH = 24.0, 24.0 * 30.44
GEO = {"EL": "GR", "UK": "GB"}
SINCE = "2000-01"

SURVEY = {   # key: (dataset, indic, extra filters)
    "ind_prod_exp": ("ei_bsin_m_r2", "BS-IPE", {"unit": "BAL"}), "ind_orders": ("ei_bsin_m_r2", "BS-IOB", {"unit": "BAL"}),
    "ind_export_orders": ("ei_bsin_m_r2", "BS-IEOB", {"unit": "BAL"}), "ind_stocks": ("ei_bsin_m_r2", "BS-ISFP", {"unit": "BAL"}),
    "ind_prod_trend": ("ei_bsin_m_r2", "BS-IPT", {"unit": "BAL"}), "ind_empl_exp": ("ei_bsin_m_r2", "BS-IEME-BAL", {"unit": "BAL"}),
    "ind_price_exp": ("ei_bsin_m_r2", "BS-ISPE", {"unit": "BAL"}),
    "cons_unemp_exp": ("ei_bsco_m", "BS-UE-NY", {"unit": "BAL"}), "cons_price_exp": ("ei_bsco_m", "BS-PT-NY", {"unit": "BAL"}),
    "cons_price_past": ("ei_bsco_m", "BS-PT-LY", {"unit": "BAL"}), "cons_major_purch": ("ei_bsco_m", "BS-MP-NY", {"unit": "BAL"}),
    "cons_fin_exp": ("ei_bsco_m", "BS-FS-NY", {"unit": "BAL"}), "cons_econ_exp": ("ei_bsco_m", "BS-GES-NY", {"unit": "BAL"}),
    "ret_sales_past": ("ei_bsrt_m_r2", "BS-RPBS", {"unit": "BAL"}), "ret_sales_exp": ("ei_bsrt_m_r2", "BS-REBS", {"unit": "BAL"}),
    "ret_empl_exp": ("ei_bsrt_m_r2", "BS-REM", {"unit": "BAL"}), "ret_price_exp": ("ei_bsrt_m_r2", "BS-RPE", {"unit": "BAL"}),
    "ret_conf": ("ei_bsrt_m_r2", "BS-RCI", {"unit": "BAL"}),
    "svc_conf": ("ei_bsse_m_r2", "BS-SCI", {"unit": "BAL"}), "svc_demand_exp": ("ei_bsse_m_r2", "BS-SAEM", {"unit": "BAL"}),
    "svc_empl_exp": ("ei_bsse_m_r2", "BS-SEEM", {"unit": "BAL"}), "svc_price_exp": ("ei_bsse_m_r2", "BS-PE3M", {"unit": "BAL"}),
    "con_conf": ("ei_bsbu_m_r2", "BS-CCI-BAL", {}), "con_empl_exp": ("ei_bsbu_m_r2", "BS-CEME-BAL", {}),
    "con_price_exp": ("ei_bsbu_m_r2", "BS-CPE-BAL", {}), "con_orders": ("ei_bsbu_m_r2", "BS-COB-BAL", {}),
    "cons_savings_exp": ("ei_bsco_m", "BS-SV-NY", {"unit": "BAL"}),     # (appended after the delay correction)
}
STS = {      # key: (dataset, filters, delay days after the month)  -> stored as the index level; drivers use its y/y %
    "reg": ("sts_rb_m", dict(indic_bt="REG", nace_r2="B-S_X_O_S94", s_adj="SCA", unit="I21"), 45),
    "bkrt": ("sts_rb_m", dict(indic_bt="BKRT", nace_r2="B-S_X_O_S94", s_adj="SCA", unit="I21"), 95),
    "constr": ("sts_copr_m", dict(indic_bt="PRD", nace_r2="F", s_adj="SCA", unit="I21"), 60),
    "svc_turn": ("sts_setu_m", dict(indic_bt="NETTUR", nace_r2="H-N_X_K", s_adj="SCA", unit="I21"), 100),
    "transp_turn": ("sts_setu_m", dict(indic_bt="NETTUR", nace_r2="H", s_adj="SCA", unit="I21"), 100),
    "wareh_turn": ("sts_setu_m", dict(indic_bt="NETTUR", nace_r2="H52", s_adj="SCA", unit="I21"), 100),
    "water_tr_turn": ("sts_setu_m", dict(indic_bt="NETTUR", nace_r2="H50", s_adj="SCA", unit="I21"), 100),
    "exp_vol": ("ei_eteu27_2020_m", dict(stk_flow="EXP", unit="IVOL-SA", partner="WORLD", indic="ET-T"), 80),
    "imp_vol": ("ei_eteu27_2020_m", dict(stk_flow="IMP", unit="IVOL-SA", partner="WORLD", indic="ET-T"), 80),
    "exp_vol_xeu": ("ei_eteu27_2020_m", dict(stk_flow="EXP", unit="IVOL-SA", partner="EXT_EU27_2020", indic="ET-T"), 80),
    "imp_vol_xeu": ("ei_eteu27_2020_m", dict(stk_flow="IMP", unit="IVOL-SA", partner="EXT_EU27_2020", indic="ET-T"), 80),
    "imp_uv_xeu": ("ei_eteu27_2020_m", dict(stk_flow="IMP", unit="IVU-NSA", partner="EXT_EU27_2020", indic="ET-T"), 80),
    "imp_price": ("sts_inpi_m", dict(indic_bt="PRC_IMP", cpa2_1="CPA_B-D", s_adj="NSA", unit="I21"), 35),
    "imp_price_nrg": ("sts_inpi_m", dict(indic_bt="PRC_IMP", cpa2_1="CPA_MIG_NRG_X_E", s_adj="NSA", unit="I21"), 35),
}
ATTR = "Source: Eurostat (business and consumer surveys: European Commission, DG ECFIN). Adapted by the forecasting engine (y/y changes)."


def _fetch(ds: str, flt: dict) -> pd.DataFrame:
    from forecasting_engine.eurostat import parse_jsonstat
    j = _get(HS.ESTAT_BASE + ds, params=dict(format="JSON", lang="en", sinceTimePeriod=SINCE, **flt)).json()
    d = parse_jsonstat(j).dropna(subset=["value"])
    d = d[d["time"].astype(str).str.match(r"^\d{4}-\d{2}$")]
    return pd.DataFrame({"date": pd.to_datetime(d["time"] + "-01", utc=True), "geo": d["geo"].astype(str).map(lambda g: GEO.get(g, g)),
                         "value": d["value"].astype(float)}).set_index("date")


def load(log=print) -> dict:
    out = {}
    for k, (ds, ind, flt) in {**SURVEY}.items():
        try:
            d = _fetch(ds, dict(indic=ind, s_adj="SA", **flt))
            HS.put_parquet(f"lab/raw/econ_{k}.parquet", d)
            out[k] = len(d)
            log(f"econ {k}: {d['geo'].nunique()} geos, {len(d)} rows, last {d.index.max().date()}")
        except Exception as e:   # noqa: BLE001
            log(f"econ {k}: FAILED {type(e).__name__}: {str(e)[:160]}")
    for k, (ds, flt, _lag) in STS.items():
        try:
            d = _fetch(ds, flt)
            HS.put_parquet(f"lab/raw/econ_{k}.parquet", d)
            out[k] = len(d)
            log(f"econ {k}: {d['geo'].nunique()} geos, {len(d)} rows, last {d.index.max().date()}")
        except Exception as e:   # noqa: BLE001
            log(f"econ {k}: FAILED {type(e).__name__}: {str(e)[:160]}")
    return out


def wide(k: str) -> pd.DataFrame | None:
    d = HS.get_parquet(f"lab/raw/econ_{k}.parquet")
    if d is None or not len(d):
        return None
    d = d.reset_index()
    w = d.pivot_table(index="date", columns="geo", values="value", aggfunc="last").sort_index()
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC") if pd.DatetimeIndex(w.index).tz is not None else pd.DatetimeIndex(w.index).tz_localize("UTC")
    return w


def yoy(w: pd.DataFrame) -> pd.DataFrame:
    w = w.where(w > 0)
    return (100 * np.log(w / w.shift(12))).clip(-150, 150)


# which drivers each family gets (appended in this order; never reorder: the champion is scored on its first N drivers)
COMMON = ["svc_conf", "ret_conf", "con_conf", "ind_prod_exp", "ind_orders", "ind_export_orders", "cons_unemp_exp", "cons_major_purch",
          "exp_vol", "imp_vol", "reg", "bkrt", "constr"]
FAMILY = {
    "ip_m": COMMON + ["ind_prod_trend", "ind_stocks", "ind_empl_exp", "exp_vol_xeu", "imp_price_nrg", "svc_turn"],
    "retail_m": COMMON + ["ret_sales_past", "ret_sales_exp", "cons_fin_exp", "cons_econ_exp", "svc_turn", "cons_price_exp"],
    "unemp_m": COMMON + ["ind_empl_exp", "svc_empl_exp", "ret_empl_exp", "con_empl_exp", "svc_demand_exp", "svc_turn"],
    "hicp_m": COMMON + ["ind_price_exp", "ret_price_exp", "svc_price_exp", "con_price_exp", "cons_price_exp", "cons_price_past", "imp_price",
                        "imp_price_nrg", "imp_uv_xeu"],
    "ports_m": COMMON + ["exp_vol_xeu", "imp_vol_xeu", "imp_uv_xeu", "transp_turn", "wareh_turn", "water_tr_turn", "imp_price_nrg",
                         "ind_prod_trend"],
}
GROUP = {"exp_vol": "trade", "imp_vol": "trade", "exp_vol_xeu": "trade", "imp_vol_xeu": "trade", "imp_uv_xeu": "trade",
         "transp_turn": "transport", "wareh_turn": "transport", "water_tr_turn": "transport", "imp_price_nrg": "energy"}


def drivers(fid: str) -> list:
    from forecasting_engine.lab.panel import Driver
    out = []
    for k in FAMILY.get(fid, []):
        w = wide(k)
        if w is None:
            continue
        if k in SURVEY:
            ds, ind, _ = SURVEY[k]
            out.append(Driver(f"es_{k}", GROUP.get(k, "economy"), w, MONTH,
                              note=f"Eurostat {ds} {ind} (EC business and consumer surveys, SA balance), released at the end of the month"))
        else:
            ds, flt, lag = STS[k]
            out.append(Driver(f"es_{k}_yoy", GROUP.get(k, "economy"), yoy(w), MONTH + lag * DAY,
                              note=f"Eurostat {ds} {flt}, y/y log change %, known month + {lag} days"))
    return out


# -----------------------------------------------------------------------------------------------------
# ECB Data Portal (data-api.ecb.europa.eu), ECB reuse policy: free reuse incl. commercial with the source cited ("Source: ECB").
# BSI and MIR monthly statistics come out about 4-5 weeks after the month (August 2026 was up on 29 Sep / 2 Oct): month + 35 days used.
ECB2 = {"loans_nfc_g": ("BSI", "M..N.A.A20T.A.I.U2.2240.Z01.A", "MFI loans to non-financial corporations (adjusted), annual growth %"),
        "loans_hh_g": ("BSI", "M..N.A.A20T.A.I.U2.2250.Z01.A", "MFI loans to households (adjusted), annual growth %"),
        "overnight_dep_g": ("BSI", "M..N.A.L21.A.I.U2.2300.Z01.A", "overnight deposits of the non-MFI private sector, annual growth %"),
        "ccob_nfc": ("MIR", "M..B.A2I.AM.R.A.2240.EUR.N", "composite cost of borrowing, new loans to corporations, %"),
        "ccob_hh_house": ("MIR", "M..B.A2C.AM.R.A.2250.EUR.N", "composite cost of borrowing, new house-purchase loans to households, %")}
ECB2_LAG = 40
FAMILY2 = {f: list(ECB2) for f in ("ip_m", "retail_m", "unemp_m", "hicp_m", "ports_m")}


def load_ecb(log=print) -> dict:
    import io
    out = {}
    for k, (flow, key, _) in ECB2.items():
        try:
            r = _get(f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}", params={"format": "csvdata", "startPeriod": "2000-01",
                                                                                         "detail": "dataonly"})
            d = pd.read_csv(io.BytesIO(r.content), usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
            d = d[d["REF_AREA"].str.len() == 2]
            d = pd.DataFrame({"date": pd.to_datetime(d["TIME_PERIOD"] + "-01", utc=True), "geo": d["REF_AREA"].map(lambda g: GEO.get(g, g)),
                              "value": d["OBS_VALUE"].astype(float)}).set_index("date")
            HS.put_parquet(f"lab/raw/econ_{k}.parquet", d)
            out[k] = len(d)
            log(f"econ {k}: {d['geo'].nunique()} geos, {len(d)} rows, last {d.index.max().date()}")
        except Exception as e:   # noqa: BLE001
            log(f"econ {k}: FAILED {type(e).__name__}: {str(e)[:160]}")
    return out


def drivers2(fid: str) -> list:
    from forecasting_engine.lab.panel import Driver
    out = []
    for k in FAMILY2.get(fid, []):
        w = wide(k)
        if w is not None:
            out.append(Driver(f"ecb_{k}", "economy", w, MONTH + ECB2_LAG * DAY, note=f"ECB {ECB2[k][0]} {ECB2[k][1]}: {ECB2[k][2]}; "
                                                                                    f"known month + {ECB2_LAG} days"))
    return out


# appended after the x1/x2 delays were corrected, so every family's driver list (and with it the forecast-cache
# key, core.memo_method) differs from the runs made with the first, too-short delays
FAMILY3 = {f: ["cons_savings_exp"] for f in ("ip_m", "retail_m", "unemp_m", "hicp_m", "ports_m")}


def drivers3(fid: str) -> list:
    from forecasting_engine.lab.panel import Driver
    out = []
    for k in FAMILY3.get(fid, []):
        w = wide(k)
        if w is not None:
            out.append(Driver(f"es_{k}", "economy", w, MONTH, note=f"Eurostat {SURVEY[k][0]} {SURVEY[k][1]} (EC consumer survey, SA balance)"))
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] == "load":
        load()
        load_ecb()
    elif a[0] == "load_ecb":
        load_ecb()
    elif a[0] == "relag":                     # set the corrected (longer) delays on this module's own first x1/x2 entries (names es_* / ecb_*)
        import json
        new = {f"es_{k}": MONTH for k in SURVEY} | {f"es_{k}_yoy": MONTH + v[2] * DAY for k, v in STS.items()} | \
            {f"ecb_{k}": MONTH + ECB2_LAG * DAY for k in ECB2}
        for f in a[1:]:
            man = json.loads(HS.get_bytes(f"lab/frozen/{f}/manifest.json"))
            ch = []
            for d in man["drivers"]:
                if d["name"] in new and abs(d["avail_h"] - new[d["name"]]) > 1e-6:
                    ch.append((d["name"], round(d["avail_h"] / DAY, 1), round(new[d["name"]] / DAY, 1)))
                    d["avail_h"] = new[d["name"]]
                    d["note"] = d.get("note", "") + f"; delay corrected 20 Oct 2026 to {new[d['name']] / DAY:.0f} days (worst-case reporter)"
            HS.put_bytes(f"lab/frozen/{f}/manifest.json", json.dumps(man, indent=1).encode(), "application/json")
            print(f, ch)
    elif a[0] == "extend":
        from forecasting_engine.lab import panel as PN
        for f in a[1:]:
            print(PN.extend(f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
