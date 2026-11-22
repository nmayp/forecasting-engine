"""Layer-index panel for the cross-layer nowcasting round: every other layer of the lab as a source on a daily grid, as it was KNOWN
on each day (the ragged edge is kept: a layer with a 60-day delay is 60 days stale, a weekly layer with 2 days is 2 days stale).

    python -m forecasting_engine.lab.xl_panel          # builds ~/.cache/forecasting_engine_xl/zidx.parquet (a few minutes)

Per layer and series: the trailing mean over a window of the layer's own native grid (short: 28 days / 4 weeks / 1 month; long: 91 days / 13 weeks /
3 months), its change against the same window a year before (this removes the season), divided by the expanding standard deviation of that change
(causal), clipped at 5. Per layer: the mean over its series (at least 30 % of them present). The value is stamped when published (period start +
the layer's publication delay) and carried forward on the daily grid. Layers built on ENTSO-E values that are licensed for back-testing only (price_h,
load_h, res_h, load_d) are left out, so nothing derived here depends on restricted data.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine import histories as HS

CACHE = Path(os.environ.get("FE_XL_CACHE", str(Path.home() / ".cache" / "forecasting_engine_xl")))
RESTRICTED = {"price_h", "load_h", "res_h", "load_d", "ecb_w"}
WIN = {"D": (28, 91), "W": (4, 13), "M": (1, 3)}
YY = {"D": 364, "W": 52, "M": 12}
RULE = {"D": "D", "W": "7D", "M": "MS"}
LIMIT = {"D": 60, "W": 90, "M": 200}
GRID = pd.date_range("2004-01-01", "2026-12-31", freq="D", tz="UTC")


def _families() -> list[str]:
    from forecasting_engine.lab import core as C
    return [f["id"] for f in C.registry()["families"] if f.get("available") and f["id"] not in RESTRICTED and f["freq"] in WIN]


def _load(fid: str):
    f = CACHE / "Y" / f"{fid}.parquet"
    mf = CACHE / "Y" / f"{fid}.json"
    if f.exists() and mf.exists():
        man = json.loads(mf.read_text())
        y = pd.read_parquet(f).astype("float64")
    else:
        man = json.loads(HS.get_bytes(f"lab/frozen/{fid}/manifest.json"))
        raw = HS.get_bytes(f"lab/frozen/{fid}/Y.parquet")
        (CACHE / "Y").mkdir(parents=True, exist_ok=True)
        f.write_bytes(raw)
        mf.write_text(json.dumps({k: man[k] for k in ("freq", "avail_h", "label", "layer")}))
        y = pd.read_parquet(io.BytesIO(raw)).astype("float64")
    y.index = pd.DatetimeIndex(y.index)
    y.index = (y.index.tz_localize("UTC") if y.index.tz is None else y.index.tz_convert("UTC")).as_unit("ns")
    return man, y


def _regular(y: pd.DataFrame, freq: str) -> pd.DataFrame:
    y = y[~y.index.duplicated()].sort_index()
    if freq == "M":
        ix = pd.date_range(y.index[0], y.index[-1], freq="MS", tz="UTC")
    else:
        ix = pd.date_range(y.index[0], y.index[-1], freq=RULE[freq], tz="UTC")
    return y.reindex(ix)


PERIOD_D = {"D": 1.0, "W": 7.0, "M": 30.44}


def layer_index(y: pd.DataFrame, freq: str, avail_h: float, win: int, data_time: bool = False) -> pd.Series:
    """The layer's change index (see the module doc), as a series stamped when published (data_time: stamped at the end of its period)."""
    y = _regular(y, freq)
    minp = max(1, int(np.ceil(0.6 * win)))
    L = y.rolling(win, min_periods=minp).mean()
    d = L - L.shift(YY[freq])
    sd = d.expanding(min_periods=24).std().replace(0, np.nan)
    z = (d / sd).clip(-5, 5)
    ok = z.notna().sum(axis=1) >= max(1, int(np.ceil(0.3 * z.shape[1])))
    s = z.mean(axis=1).where(ok).dropna()
    if data_time:
        return pd.Series(s.to_numpy(), index=pd.DatetimeIndex(s.index + pd.Timedelta(days=PERIOD_D[freq])))
    kt = s.index + pd.Timedelta(hours=float(avail_h))
    return pd.Series(s.to_numpy(), index=pd.DatetimeIndex(kt))


def to_grid(s: pd.Series, freq: str) -> pd.Series:
    d = s.copy()
    d.index = d.index.floor("D")
    d = d.groupby(level=0).last()
    # information known at the START of day D: what was published before D 00:00 (so a value stamped during D is available on D + 1)
    d.index = d.index + pd.Timedelta(days=1)
    return d.reindex(GRID.union(d.index)).ffill(limit=LIMIT[freq]).reindex(GRID)


def to_data_grid(s: pd.Series, freq: str) -> pd.Series:
    d = s.copy()
    d.index = d.index.floor("D")
    d = d.groupby(level=0).last()
    return d.reindex(GRID.union(d.index)).ffill(limit=LIMIT[freq]).reindex(GRID)


def build() -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    fids = _families()
    with ThreadPoolExecutor(8) as ex:
        loaded = list(ex.map(lambda f: (f, _try_load(f)), fids))
    cols, dcols, ecols = {}, {}, {}
    for fid, got in loaded:
        if got is None:
            print("skip", fid, flush=True)
            continue
        man, y = got
        fr = man["freq"]
        if len(y.dropna(how="all")) < YY[fr] + 30:
            print("short", fid, flush=True)
            continue
        for tag, w in zip(("s", "l"), WIN[fr]):
            try:
                cols[f"{fid}:{tag}"] = to_grid(layer_index(y, fr, float(man["avail_h"]), w), fr)
                sd_ = layer_index(y, fr, float(man["avail_h"]), w, data_time=True)
                dcols[f"{fid}:{tag}"] = to_data_grid(sd_, fr)
                e_day = pd.Series(((sd_.index.floor("D") - GRID[0]).days).astype(float), index=layer_index(y, fr, float(man["avail_h"]), w).index)
                ecols[f"{fid}:{tag}"] = to_grid(e_day, fr)
            except Exception as e:  # noqa: BLE001
                print("fail", fid, tag, type(e).__name__, str(e)[:80], flush=True)
        print("ok", fid, fr, y.shape, flush=True)
    Z = pd.DataFrame(cols)
    CACHE.mkdir(parents=True, exist_ok=True)
    Z.astype("float32").to_parquet(CACHE / "zidx.parquet")
    pd.DataFrame(dcols)[Z.columns].astype("float32").to_parquet(CACHE / "zdata.parquet")
    pd.DataFrame(ecols)[Z.columns].astype("float32").to_parquet(CACHE / "zend.parquet")
    return Z


def _try_load(fid: str):
    try:
        return _load(fid)
    except Exception:  # noqa: BLE001
        return None


_Z = None
_ZD = None
_ZE = None


STORE = ("zidx", "zdata", "zend")


def _fetch() -> bool:
    """The three panel files from the history store (lab/xl/, written by publish()) if they are not on this machine."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for n in STORE:
        f = CACHE / f"{n}.parquet"
        if not f.exists():
            try:
                b = HS.get_bytes(f"lab/xl/{n}.parquet")
            except Exception:  # noqa: BLE001
                b = None
            if not b:
                return False
            f.write_bytes(b)
    return True


def publish() -> None:
    """Copy the built panel files to the history store so that VMs and other machines need not rebuild them (a few minutes with the cache cold)."""
    for n in STORE:
        HS.put_bytes(f"lab/xl/{n}.parquet", (CACHE / f"{n}.parquet").read_bytes())


def zidx() -> pd.DataFrame:
    global _Z
    if _Z is None:
        f = CACHE / "zidx.parquet"
        if not f.exists() and not _fetch():
            build()
        _Z = pd.read_parquet(f)
    return _Z


def zdata() -> pd.DataFrame:
    global _ZD
    zidx()
    if _ZD is None:
        _ZD = pd.read_parquet(CACHE / "zdata.parquet")
    return _ZD


def zend() -> pd.DataFrame:
    global _ZE
    zidx()
    if _ZE is None:
        _ZE = pd.read_parquet(CACHE / "zend.parquet")
    return _ZE


if __name__ == "__main__":
    import sys
    if "--publish" in sys.argv:
        publish()
    else:
        Z = build()
        print(Z.shape, Z.notna().mean().round(2).describe())
