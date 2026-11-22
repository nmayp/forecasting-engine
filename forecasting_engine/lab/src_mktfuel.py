"""Markets and fuel inputs, futures curves, refined-product spot and futures prices, positioning in energy and currency
futures, central-bank policy rates and yield curves, each stamped at the moment it is really public, for commod_d, fx_d, diesel_w and gasoline_w.

    python -m forecasting_engine.lab.src_mktfuel [eia] [cot] [bis] [ycurve]     # downloads to the store (default: all)

| Store name | Content | Source, real publication lag | Licence |
|---|---|---|---|
| lab/raw/eia_daily.parquet | long: date, id, value. NYMEX settlement prices republished by EIA: WTI light sweet crude futures months 1-4 (RCLC1-4, USD/bbl), Henry Hub natural gas futures 1-4 (RNGC1-4, USD/MMBtu), RBOB gasoline and No. 2 heating oil futures month 1 (USD/gal); spot: Brent (RBRTE), NY Harbor and US Gulf Coast ULSD, Gulf Coast conventional gasoline and kerosene-type jet fuel, Mont Belvieu propane (USD/gal) | EIA petroleum / natural gas daily price history. The futures end on 5 Apr 2024 (EIA stopped republishing them), so features use the spot series only (dnav hist_xls ...d.xls). EIA republishes them WEEKLY (Wednesday, data to the Tuesday): 14 days used since the lag audit of 20 Oct 2026 (36 h before) | EIA: public domain, "Source: U.S. Energy Information Administration" (prices originate from exchanges / Refinitiv as republished by EIA) |
| lab/raw/cftc_cot_m1.parquet | long: date (report Tuesday), market, kind (mm / swap / prod / lev / am / dealer), net % of open interest. Disaggregated futures: RBOB gasoline (111659), NY Harbor ULSD (022651), and swap-dealer / producer nets of WTI, Brent, Henry Hub. Traders in Financial Futures: EUR, JPY, GBP, CHF, CAD, AUD, MXN, BRL, ZAR (CME), leveraged funds, asset managers, dealers | CFTC Commitments of Traders, Tuesday positions released Friday 15:30 ET (Monday after a holiday): Tuesday + 6 days 21 h | US Government work, public domain |
| lab/raw/bis_cbpol.parquet | long: date, cc, value: central-bank policy rates, daily (BIS WS_CBPOL), euro area (XM) and the 20 fx_d currencies' countries | BIS SDMX API. A rate is public at its announcement, which is on or before the effective date stamped: end of day + 12 h used | BIS terms of permitted use of statistics: unrestricted incl. commercial, cite "Source: BIS", not misleading / no endorsement, no extra charge for the BIS statistics themselves |
| lab/raw/mktfuel_yc.parquet | long: date, id, value: euro area AAA government spot rates 2y, 10y (ECB YC dataflow, Svensson) | ECB Data Portal, published about 12:00 CET the next TARGET day: end of day + 12 h | ECB statistics reuse policy (free reuse incl. commercial, cite the ECB, do not modify) |
| lab/raw/fred_m1.parquet | long: date, id, value: US 10y TIPS real yield (DFII10), 3-month Treasury bill (DGS3MO), 5y breakeven (T5YIE) | FRED (Federal Reserve H.15 / US Treasury), end of day + 12 h | US Government data, public domain |
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS
from forecasting_engine.lab.sources import _get

EIA_PET = {"RCLC1": "wti_c1", "RCLC2": "wti_c2", "RCLC3": "wti_c3", "RCLC4": "wti_c4", "RBRTE": "brent_spot",
           "EER_EPMRR_PE1_Y35NY_DPG": "rbob_c1", "EER_EPD2F_PE1_Y35NY_DPG": "ho_c1",
           "EER_EPD2DXL0_PF4_Y35NY_DPG": "ulsd_nyh", "EER_EPD2DXL0_PF4_RGC_DPG": "ulsd_gulf", "EER_EPMRU_PF4_RGC_DPG": "gasoline_gulf",
           "EER_EPJK_PF4_RGC_DPG": "jet_gulf", "EER_EPLLPA_PF4_Y44MB_DPG": "propane_mb"}
EIA_NG = {"RNGC1": "ng_c1", "RNGC2": "ng_c2", "RNGC3": "ng_c3", "RNGC4": "ng_c4"}
D_LAG_H = 36.0                       # end of day + 12 h (stamps at 00:00): BIS policy rates (public at the announcement), FRED H.15 / Treasury
#                                      yields (the Treasury publishes its curve the same day)
EIA_LAG_H = 14 * 24.0                # lag audit 20 Oct 2026: EIA daily spot prices are released WEEKLY (panel.EIA_SPOT_LAG_H; was D_LAG_H)
YC_LAG_H = 5 * 24.0 + 12             # lag audit 20 Oct 2026: ECB AAA curve, 12:00 CET the next TARGET day (Easter: 5 days 11 h; was D_LAG_H)
COT_LAG_H = 6 * 24.0 + 21
COT_URL = "https://publicreporting.cftc.gov/resource/{ds}.json"
DISAGG = {"111659": "rbob", "022651": "ulsd", "067651": "wti", "06765T": "brent", "023651": "ng"}
TFF = {"099741": "USD", "097741": "JPY", "096742": "GBP", "092741": "CHF", "090741": "CAD", "232741": "AUD", "095741": "MXN", "102741": "BRL",
       "122741": "ZAR"}                # column = the fx_d series it bears on (EUR futures: EUR vs USD)
BIS_CC = {"XM": "EUR", "US": "USD", "GB": "GBP", "JP": "JPY", "CH": "CHF", "CN": "CNY", "PL": "PLN", "CZ": "CZK", "HU": "HUF", "SE": "SEK",
          "NO": "NOK", "DK": "DKK", "RO": "RON", "TR": "TRY", "AU": "AUD", "CA": "CAD", "BR": "BRL", "MX": "MXN", "IN": "INR", "KR": "KRW",
          "ZA": "ZAR"}
FRED_M1 = ("DFII10", "DGS3MO", "T5YIE")
YC = {"SR_2Y": "ea_2y", "SR_10Y": "ea_10y"}
GAL_L = 3.785411784


# ------------------------------------------------------------------------------------------------------------------------------ loaders
def _eia_daily(sid: str, area: str) -> pd.Series:
    r = _get(f"https://www.eia.gov/dnav/{area}/hist_xls/{sid}d.xls", ua=False)
    x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
    x.columns = ["date", "value"] + list(x.columns[2:])
    s = pd.Series(pd.to_numeric(x["value"], errors="coerce").to_numpy(), index=pd.to_datetime(x["date"], errors="coerce", utc=True)).dropna()
    return s[s.index.notna()].sort_index()


def eia_load(log=print) -> dict:
    rows = []
    for ids, area in ((EIA_PET, "pet"), (EIA_NG, "ng")):
        for sid, nm in ids.items():
            s = _eia_daily(sid, area)
            rows.append(pd.DataFrame({"date": s.index, "id": nm, "value": s.to_numpy()}))
            log(f"eia {nm}: {len(s)} days {s.index.min().date()} - {s.index.max().date()}")
    d = pd.concat(rows)
    HS.put_parquet("lab/raw/eia_daily.parquet", d.set_index("date"))
    return dict(series=len(rows), rows=len(d))


def _cot_pull(ds: str, code: str, cols: list[str]) -> pd.DataFrame:
    r = _get(COT_URL.format(ds=ds), ua=False, params={"cftc_contract_market_code": code, "$limit": 50000, "$order": "report_date_as_yyyy_mm_dd",
                                                       "$select": ",".join(["report_date_as_yyyy_mm_dd", "open_interest_all"] + cols)})
    d = pd.DataFrame(r.json())
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d.pop("report_date_as_yyyy_mm_dd"), utc=True)
    for c in d.columns:
        if c != "date":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _nets(d: pd.DataFrame, nm: str, kinds: dict) -> list[pd.DataFrame]:
    return [pd.DataFrame({"date": d["date"], "market": nm, "kind": k,
                          "value": ((d[lo] - d[sh]) / d["open_interest_all"] * 100).where(d["open_interest_all"] > 0)})
            for k, (lo, sh) in kinds.items() if lo in d and sh in d]


def cot_load(log=print) -> dict:
    parts = []
    kinds = {"mm": ("m_money_positions_long_all", "m_money_positions_short_all"), "swap": ("swap_positions_long_all", "swap__positions_short_all"),
             "prod": ("prod_merc_positions_long", "prod_merc_positions_short")}
    for code, nm in DISAGG.items():
        d = _cot_pull("72hh-3qpy", code, [c for v in kinds.values() for c in v])
        parts += _nets(d, nm, kinds)
        log(f"cftc disagg {nm}: {len(d)} weeks")
    kinds = {"lev": ("lev_money_positions_long", "lev_money_positions_short"), "am": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
             "dealer": ("dealer_positions_long_all", "dealer_positions_short_all")}
    for code, nm in TFF.items():
        d = _cot_pull("gpe5-46if", code, [c for v in kinds.values() for c in v])
        parts += _nets(d, nm, kinds)
        log(f"cftc tff {nm}: {len(d)} weeks")
    d = pd.concat(parts).dropna()
    HS.put_parquet("lab/raw/cftc_cot_m1.parquet", d.set_index("date"))
    return dict(rows=len(d), kinds=sorted(d["kind"].unique()))


def bis_load(log=print) -> dict:
    r = _get(f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.{'+'.join(BIS_CC)}/all", ua=False, params={"startPeriod": "1998-01-01", "format": "csv"})
    x = pd.read_csv(io.BytesIO(r.content), usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    d = pd.DataFrame({"date": pd.to_datetime(x["TIME_PERIOD"], utc=True), "cc": x["REF_AREA"], "value": pd.to_numeric(x["OBS_VALUE"], errors="coerce")}).dropna()
    log(f"bis policy rates: {d['cc'].nunique()} areas, {len(d)} rows; missing {sorted(set(BIS_CC) - set(d['cc']))}")
    HS.put_parquet("lab/raw/bis_cbpol.parquet", d.set_index("date"))
    return dict(rows=len(d))


def ycurve_load(log=print) -> dict:
    rows = []
    for k, nm in YC.items():
        r = _get(f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{k}", ua=False, params={"format": "csvdata"})
        x = pd.read_csv(io.BytesIO(r.content), usecols=["TIME_PERIOD", "OBS_VALUE"])
        rows.append(pd.DataFrame({"date": pd.to_datetime(x["TIME_PERIOD"], utc=True), "id": nm, "value": pd.to_numeric(x["OBS_VALUE"], errors="coerce")}))
        log(f"ecb yc {nm}: {len(x)} days")
    for sid in FRED_M1:
        x = pd.read_csv(io.BytesIO(_get("https://fred.stlouisfed.org/graph/fredgraph.csv", ua=False, params={"id": sid}).content))
        x.columns = ["date", "value"]
        rows.append(pd.DataFrame({"date": pd.to_datetime(x["date"], utc=True), "id": sid, "value": pd.to_numeric(x["value"], errors="coerce")}))
        log(f"fred {sid}: {len(x)} rows")
    d = pd.concat(rows).dropna()
    HS.put_parquet("lab/raw/mktfuel_yc.parquet", d[d["id"].isin(YC.values())].set_index("date"))
    HS.put_parquet("lab/raw/fred_m1.parquet", d[d["id"].isin(FRED_M1)].set_index("date"))
    return dict(rows=len(d))


# ------------------------------------------------------------------------------------------------------------------------------ features
def _wide(name: str, key: str) -> pd.DataFrame | None:
    d = HS.get_parquet(name)
    if d is None:
        return None
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    return d.pivot_table(index="date", columns=key, values="value", aggfunc="last").sort_index()


def _clean(x):
    return x.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def commod_features(f: pd.DataFrame) -> list[tuple]:
    """(name, group, frame, avail_h, note) for commod_d. f: the panel's FRED dailies. Per-series frames use the commod_d series ids."""
    e = _wide("lab/raw/eia_daily.parquet", "id")
    if e is None:
        return []
    nt = "EIA daily prices (NYMEX settlements / spot, public domain)"
    # EIA stopped republishing NYMEX futures after 5 Apr 2024 (RCLC*, RNGC*, RBOB / heating oil futures end there): a curve feature would be
    # empty for most of the holdout, so only the spot series (still updated) are used. The futures stay in the store for back-tests.
    wti = f["DCOILWTICO"].reindex(e.index)
    out = []
    brent = e["brent_spot"]
    for k, base, lab in (("ulsd_nyh", brent, "Brent"), ("ulsd_gulf", brent, "Brent"), ("jet_gulf", brent, "Brent"), ("gasoline_gulf", wti, "WTI")):
        out.append((f"crack_{k}", "fuel", _clean((e[k] * 42 - base).to_frame("*")), EIA_LAG_H, f"{nt}: {k} x 42 minus {lab}, USD/bbl"))
    out.append(("propane_vs_wti", "fuel", _clean((e["propane_mb"] * 42 / wti * 100).to_frame("*")), EIA_LAG_H, f"{nt}: Mont Belvieu propane x 42 as % of WTI"))
    return out


def fuel_features(fx: pd.DataFrame) -> list[tuple]:
    """Product prices in EUR per litre (what the pump price is built on) and their 5-day change, for diesel_w / gasoline_w."""
    e = _wide("lab/raw/eia_daily.parquet", "id")
    if e is None:
        return []
    usd = fx["USD"].reindex(e.index).ffill()
    nt = "EIA daily product prices (public domain) / ECB USD per EUR"
    out = []
    for k in ("ulsd_nyh", "ulsd_gulf", "gasoline_gulf", "jet_gulf"):
        eur_l = (e[k] / GAL_L / usd).dropna()
        out.append((f"{k}_eur_l", "fuel", _clean(eur_l.to_frame("*")), EIA_LAG_H, f"{nt}: {k} in EUR/l"))
        out.append((f"{k}_eur_l_chg5", "fuel", _clean((eur_l - eur_l.shift(5)).to_frame("*")), EIA_LAG_H, f"{nt}: {k} EUR/l, change over 5 trading days"))
    return out


def cot_features(markets: tuple, kinds: tuple, cols: bool = False) -> list[tuple]:
    d = HS.get_parquet("lab/raw/cftc_cot_m1.parquet")
    if d is None:
        return []
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    out = []
    for k in kinds:
        x = d[(d["kind"] == k) & d["market"].isin(markets)].pivot_table(index="date", columns="market", values="value", aggfunc="last").sort_index()
        from forecasting_engine.lab import panel as PN
        x = PN.cot_mask(x)                                     # lag audit 20 Oct 2026: reports published weeks late (shutdowns) dropped
        if x.empty:
            continue
        nt = f"CFTC Commitments of Traders (public domain): {k} net long, % of open interest (Tuesday positions)"
        if cols:
            out.append((f"cot_{k}_net", "markets", _clean(x), COT_LAG_H, f"{nt}, per currency future vs USD (USD column: the EUR future)"))
            out.append((f"cot_{k}_net_chg4", "markets", _clean(x - x.shift(4)), COT_LAG_H, f"{nt}, change over 4 weeks"))
        else:
            for m in x.columns:
                out.append((f"cot_{k}_{m}", "markets", _clean(x[[m]].rename(columns={m: "*"})), COT_LAG_H, f"{nt}, {m}"))
    return out


def fx_features(f: pd.DataFrame) -> list[tuple]:
    """fx_d: policy-rate differentials (per currency), currency-futures positioning (per currency), euro and US curves (global)."""
    out = []
    b = _wide("lab/raw/bis_cbpol.parquet", "cc")
    if b is not None:
        b = b.resample("D").last().ffill()
        cc = [c for c in BIS_CC if c in b and c != "XM"]
        diff = pd.DataFrame({BIS_CC[c]: b[c] - b["XM"] for c in cc})
        nt = "BIS central-bank policy rates (Source: BIS), country minus euro area, pp"
        out += [("policy_rate_diff", "policy", _clean(diff), D_LAG_H, nt),
                ("policy_rate_diff_chg90", "policy", _clean(diff - diff.shift(90)), D_LAG_H, f"{nt}, change over 90 days"),
                ("policy_rate_own_chg30", "policy", _clean(pd.DataFrame({BIS_CC[c]: b[c] - b[c].shift(30) for c in cc})), D_LAG_H,
                 "BIS policy rates (Source: BIS): the country's own change over 30 days, pp")]
    y = _wide("lab/raw/mktfuel_yc.parquet", "id")
    if y is not None:
        us2 = f["DGS2"].dropna()
        out += [("ea_curve_10_2", "markets", _clean((y["ea_10y"] - y["ea_2y"]).to_frame("*")), YC_LAG_H, "ECB AAA euro area yield curve: 10y minus 2y spot, pp"),
                ("us_ea_2y", "markets", _clean((us2 - y["ea_2y"].reindex(us2.index).ffill()).to_frame("USD")), YC_LAG_H,
                 "FRED DGS2 minus ECB AAA 2y spot, pp (USD column)")]
    u = _wide("lab/raw/fred_m1.parquet", "id")
    if u is not None:
        out += [("us_real10y", "markets", _clean(u[["DFII10"]].rename(columns={"DFII10": "*"})), D_LAG_H, "FRED DFII10 (H.15), 10y TIPS real yield"),
                ("us_bill3m", "markets", _clean(u[["DGS3MO"]].rename(columns={"DGS3MO": "*"})), D_LAG_H, "FRED DGS3MO, 3-month Treasury bill"),
                ("us_breakeven5y", "markets", _clean(u[["T5YIE"]].rename(columns={"T5YIE": "*"})), D_LAG_H, "FRED T5YIE, 5y breakeven inflation")]
    return out + cot_features(tuple(TFF.values()), ("lev", "am", "dealer"), cols=True)


if __name__ == "__main__":
    what = sys.argv[1:] or ["eia", "cot", "bis", "ycurve"]
    fns = dict(eia=eia_load, cot=cot_load, bis=bis_load, ycurve=ycurve_load)
    for w in what:
        print(w, fns[w](), flush=True)
    sys.exit(0)
