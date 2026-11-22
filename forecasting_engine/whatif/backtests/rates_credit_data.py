"""Data for the interest-rate -> lending -> investment what-if (docs/methods.md).

    python -m forecasting_engine.whatif.backtests.rates_credit_data          # fetches ~100 small series -> results/cascade/rates_credit_data.csv.gz

One long file (key, geo, date, value), committed so a restarted container can rebuild every number without the network. Values are the current
vintage, unmodified (ECB reuse policy: statistics not modified, source cited).

| Key | Source (URL pattern) | Content | Publication lag used (days after the END of the reference month / quarter) |
|---|---|---|---|
| dfr, mro | ECB Data Portal FM/B.U2.EUR.4F.KR.DFR.LEV, FM/B.U2.EUR.4F.KR.MRR_FR.LEV (+MRR_MBR for variable-rate tenders) | policy rates, daily step function | 0 (announced) |
| ccob_nfc | ECB MIR M.<c>.B.A2I.AM.R.A.2240.EUR.N | composite cost of borrowing, new loans to non-financial corporations, % | 40 (release about day 30-33; 40 is the lab's conservative rule, src_econ.ECB2_LAG) |
| ccob_hh | ECB MIR M.<c>.B.A2C.AM.R.A.2250.EUR.N | composite cost of borrowing, new loans to households for house purchase, % | 40 |
| vol_nfc | ECB MIR M.<c>.B.A2A.A.B.0 + .1 .2240.EUR.N | new business volume, loans to corporations (excl. revolving), up to + over EUR 1M, EUR m | 40 |
| vol_hh | ECB MIR M.<c>.B.A2C.A.B.A.2250.EUR.N | new business volume, loans to households for house purchase, EUR m | 40 |
| lg_nfc, lg_hh | ECB BSI M.<c>.N.A.A20T.A.I.U2.2240.Z01.A, ...2250... | MFI loans to NFC / households, annual growth %, adjusted for sales and securitisation | 40 |
| constr | Eurostat sts_copr_m PRD F SCA I21 | production in construction, index 2021=100 | 60 (src_econ) |
| permits | Eurostat sts_cobp_m BPRM_SQM CPA_F41001_41002 SCA I21 | building permits, useful floor area, index 2021=100 | 75 (assumption: slowest reporters, checked against the release calendar only roughly) |
| gfcf | Eurostat namq_10_gdp P51G SCA CLV_I15 | gross fixed capital formation, chain-linked volume index 2015=100, quarterly | 70 (3rd estimate with expenditure breakdown) |

Credit: "Source: ECB Data Portal; Eurostat". Countries: euro area (U2 = changing composition) and AT BE DE ES FI FR GR IE IT NL PT.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[3] / "results" / "cascade" / "rates_credit_data.csv.gz"
UA = {"User-Agent": "forecasting-engine/0.1"}
ECB = "https://data-api.ecb.europa.eu/service/data/"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
GEO = ["U2", "AT", "BE", "DE", "ES", "FI", "FR", "GR", "IE", "IT", "NL", "PT"]
LAG = dict(ccob_nfc=40, ccob_hh=40, vol_nfc=40, vol_hh=40, lg_nfc=40, lg_hh=40, constr=60, permits=75, gfcf=70)
MIR = {"ccob_nfc": "M.{g}.B.A2I.AM.R.A.2240.EUR.N", "ccob_hh": "M.{g}.B.A2C.AM.R.A.2250.EUR.N",
       "vol_nfc0": "M.{g}.B.A2A.A.B.0.2240.EUR.N", "vol_nfc1": "M.{g}.B.A2A.A.B.1.2240.EUR.N", "vol_hh": "M.{g}.B.A2C.A.B.A.2250.EUR.N"}
BSI = {"lg_nfc": "M.{g}.N.A.A20T.A.I.U2.2240.Z01.A", "lg_hh": "M.{g}.N.A.A20T.A.I.U2.2250.Z01.A"}


def ecb(flow: str, key: str, since="1999-01") -> pd.DataFrame:
    for t in range(3):
        r = requests.get(f"{ECB}{flow}/{key}", params={"format": "csvdata", "startPeriod": since, "detail": "dataonly"}, headers=UA, timeout=120)
        if r.status_code in (404, 400):
            return pd.DataFrame(columns=["geo", "date", "value"])
        if r.ok:
            break
        time.sleep(2 + 3 * t)
    r.raise_for_status()
    d = pd.read_csv(io.BytesIO(r.content))
    return pd.DataFrame({"geo": d["REF_AREA"], "date": pd.to_datetime(d["TIME_PERIOD"].astype(str)), "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")}).dropna()


def eurostat(ds: str, params: str) -> pd.DataFrame:
    r = requests.get(f"{EUROSTAT}{ds}?format=JSON&lang=en&{params}", headers=UA, timeout=300)
    r.raise_for_status()
    d = r.json()
    ids, size = d["id"], d["size"]
    cats = {k: list(d["dimension"][k]["category"]["index"]) for k in ids}
    import itertools
    rows = []
    for i, combo in enumerate(itertools.product(*[range(s) for s in size])):
        v = d["value"].get(str(i))
        if v is not None:
            rows.append((cats["geo"][combo[ids.index("geo")]], cats["time"][combo[ids.index("time")]], v))
    out = pd.DataFrame(rows, columns=["geo", "t", "value"])
    out["date"] = pd.PeriodIndex(out["t"].str.replace(r"-Q(\d)", r"Q\1", regex=True), freq="M" if "Q" not in str(out["t"].iloc[0]) else "Q").to_timestamp()
    return out[["geo", "date", "value"]]


def policy() -> pd.DataFrame:
    """Daily policy rates from their change dates: deposit facility, main refinancing (fixed rate; from 2000 variable-rate tenders: minimum bid rate)."""
    rows = []
    for key, name in (("FM/B.U2.EUR.4F.KR.DFR.LEV", "dfr"), ("FM/B.U2.EUR.4F.KR.MRR_FR.LEV", "mro_fr"), ("FM/B.U2.EUR.4F.KR.MRR_MBR.LEV", "mro_mbr")):
        d = ecb(*key.split("/", 1), since="1999-01-01")
        if len(d):
            rows.append(d.assign(key=name, geo="U2")[["key", "geo", "date", "value"]])
    return pd.concat(rows)


def main():
    parts = [policy()]
    geos = "+".join(GEO)
    for k, key in MIR.items():
        parts.append(ecb("MIR", key.format(g=geos), "2003-01").assign(key=k))
        print(k, len(parts[-1]))
    for k, key in BSI.items():
        parts.append(ecb("BSI", key.format(g=geos), "1999-01").assign(key=k))
        print(k, len(parts[-1]))
    eg = "&".join(f"geo={'EA20' if g == 'U2' else g}" for g in GEO)
    for k, ds, p in (("constr", "sts_copr_m", "indic_bt=PRD&nace_r2=F&s_adj=SCA&unit=I21&sinceTimePeriod=2000-01"),
                     ("permits", "sts_cobp_m", "indic_bt=BPRM_SQM&cpa2_1=CPA_F41001_41002&s_adj=SCA&unit=I21&sinceTimePeriod=2000-01"),
                     ("gfcf", "namq_10_gdp", "na_item=P51G&s_adj=SCA&unit=CLV_I15&sinceTimePeriod=2000-Q1")):
        d = eurostat(ds, f"{p}&{eg}")
        d["geo"] = d["geo"].replace({"EA20": "U2"})
        parts.append(d.assign(key=k))
        print(k, len(d))
    df = pd.concat(parts, ignore_index=True)
    # total NFC volume = up to + over EUR 1M
    v = df[df.key.isin(["vol_nfc0", "vol_nfc1"])].pivot_table(index=["geo", "date"], columns="key", values="value")
    v = v.dropna().sum(axis=1).rename("value").reset_index().assign(key="vol_nfc")
    df = pd.concat([df[~df.key.isin(["vol_nfc0", "vol_nfc1"])], v], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["key", "geo", "date", "value"]].to_csv(OUT, index=False, compression="gzip")
    print(df.groupby("key").agg(n=("value", "size"), geos=("geo", "nunique"), first=("date", "min"), last=("date", "max")))


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT)
    d["date"] = pd.to_datetime(d["date"])
    return d


if __name__ == "__main__":
    sys.exit(main())
