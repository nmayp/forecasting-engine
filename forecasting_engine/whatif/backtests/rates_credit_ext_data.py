"""Extra inputs for the rates -> credit follow-up (docs/methods.md).

    python -m forecasting_engine.whatif.backtests.rates_credit_ext_data   -> results/cascade/rates_credit_ext.csv.gz   (key, date, value; monthly)

| Key | Source | Content | Publication lag |
|---|---|---|---|
| jk_mp, jk_cbi | Jarocinski and Karadi (2020), updated shocks, github.com/marekjarocinski/jkshocks_update_ecb `shocks_ecb_mpd_me_m.csv` (CC BY 4.0, code BSD-3) | monthly sum of ECB monetary-policy shock (MP_pm: policy news, sign-restricted) and central-bank-information shock (CBI_pm), units: % p.a. on the OIS 1Y | 0 (shock is known on the day of the Governing Council meeting) |
| jk_pc1 | same file, pc1_hf | total OIS surprise factor (= MP + CBI) | 0 |
| mpd_ois1y, mpd_ois3m | ECB Euro Area Monetary Policy event-study Database EA-MPD, Altavilla, Brugnolini, Gurkaynak, Motto, Ragusa (2019, J. Monetary Econ.), ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx, sheet "Monetary Event Window", OIS_1Y / OIS_3M in basis points -> % | monthly sum of the surprises | 0 |
| eur3m, eur6m, eur12m | ECB Data Portal FM/M.U2.EUR.RT.MM.EURIBOR{3,6,1Y}MD_.HSTA | Euribor, monthly average (market-implied expected path of the policy rate plus a term / credit premium) | 0 (observable daily) |
Credit: "Source: ECB Data Portal; ECB EA-MPD (Altavilla et al. 2019); Jarocinski and Karadi (2020)".
"""
from __future__ import annotations

import io
import sys

import pandas as pd
import requests

from forecasting_engine.whatif.backtests.rates_credit_data import UA, ecb

OUT = __import__("pathlib").Path(__file__).resolve().parents[3] / "results" / "cascade" / "rates_credit_ext.csv.gz"
JK = "https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_ecb/main/shocks_ecb_mpd_me_m.csv"
MPD = "https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx"


def main():
    parts = []
    j = pd.read_csv(io.StringIO(requests.get(JK, headers=UA, timeout=60).text))
    j["date"] = pd.to_datetime(dict(year=j.year, month=j.month, day=1))
    for src, k in (("MP_pm", "jk_mp"), ("CBI_pm", "jk_cbi"), ("pc1_hf", "jk_pc1")):
        parts.append(pd.DataFrame(dict(key=k, date=j.date, value=pd.to_numeric(j[src], errors="coerce").fillna(0.0))))
    x = pd.read_excel(io.BytesIO(requests.get(MPD, headers=UA, timeout=120).content), sheet_name="Monetary Event Window")
    x["date"] = pd.to_datetime(x["date"]).dt.to_period("M").dt.to_timestamp()
    for c, k in (("OIS_1Y", "mpd_ois1y"), ("OIS_3M", "mpd_ois3m")):
        s = pd.to_numeric(x[c], errors="coerce").fillna(0.0).groupby(x["date"]).sum() / 100.0     # bp -> %
        s = s.reindex(pd.date_range("1999-01-01", s.index.max(), freq="MS"), fill_value=0.0)
        parts.append(pd.DataFrame(dict(key=k, date=s.index, value=s.to_numpy())))
    for k, code in (("eur3m", "EURIBOR3MD_"), ("eur6m", "EURIBOR6MD_"), ("eur12m", "EURIBOR1YD_")):
        d = ecb("FM", f"M.U2.EUR.RT.MM.{code}.HSTA", "1999-01")
        parts.append(pd.DataFrame(dict(key=k, date=d["date"], value=d["value"])))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(OUT, index=False, compression="gzip")
    print(df.groupby("key").agg(n=("value", "size"), first=("date", "min"), last=("date", "max")))


def main_market():
    """Nowcast inputs, appended to the existing file without refetching the shocks: Euribor 1m, euro-area AAA yield curve 2y and 10y
    (ECB YC, Svensson spot, daily; Bund-like, not the German curve), EUR short-term rate (EST, daily, from Oct 2019): monthly means (complete months)."""
    d = pd.read_csv(OUT)
    d = d[~d.key.isin(["eur1m", "yc2y", "yc10y", "estr"])]
    parts = []
    e1 = ecb("FM", "M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA", "1999-01")
    parts.append(pd.DataFrame(dict(key="eur1m", date=e1["date"], value=e1["value"])))
    for k, tenor in (("yc2y", "SR_2Y"), ("yc10y", "SR_10Y")):
        y = ecb("YC", f"B.U2.EUR.4F.G_N_A.SV_C_YM.{tenor}", "2004-09")
        m = y.set_index("date")["value"].resample("MS").mean()
        parts.append(pd.DataFrame(dict(key=k, date=m.index, value=m.to_numpy())))
    r = requests.get("https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT", params={"format": "csvdata", "startPeriod": "2019-10-01", "detail": "dataonly"}, headers=UA, timeout=120)
    r.raise_for_status()
    x = pd.read_csv(io.BytesIO(r.content))
    m = pd.Series(pd.to_numeric(x["OBS_VALUE"]).to_numpy(), index=pd.to_datetime(x["TIME_PERIOD"])).resample("MS").mean()
    parts.append(pd.DataFrame(dict(key="estr", date=m.index, value=m.to_numpy())))
    n = pd.concat(parts, ignore_index=True)
    n["date"] = pd.to_datetime(n["date"]).dt.strftime("%Y-%m-%d")
    pd.concat([d, n.dropna()], ignore_index=True).to_csv(OUT, index=False, compression="gzip")
    print(n.groupby("key").agg(n=("value", "size"), first=("date", "min"), last=("date", "max")))


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT)
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot(index="date", columns="key", values="value").sort_index()


if __name__ == "__main__":
    sys.exit(main_market() if "market" in sys.argv[1:] else main())
