"""GIFT-Eval (Salesforce, arXiv 2410.10393) evaluation harness without gluonts.

Re-implements the official protocol of github.com/SalesforceAIResearch/gift-eval (src/gift_eval/data.py,
notebooks/naive.ipynb) on the native HF `datasets` layout stored in $GIFT_ROOT (default data/gift_eval/; the official dataset repository, see README).

Protocol (official):
  * 97 (dataset, term) configurations: every dataset/frequency has a `short` term, the 20 datasets
    listed in MED_LONG also have `medium` (10x) and `long` (15x).
  * prediction_length = term multiplier (1/10/15) * base length by frequency: PRED_LENGTH_MAP
    (M 12, W 8, D 30, H 48, T 48, S 60); M4 sets use M4_PRED_LENGTH_MAP (A 6, Q 8, M 18, W 13, D 14, H 48).
  * test split: the last `windows` non-overlapping windows of the series,
    windows = min(max(1, ceil(0.1 * min_series_length / pred_len)), 20) (M4: 1). Window k's context is
    everything before it (so contexts overlap earlier test labels, as in the official code).
  * multivariate datasets are scored as independent univariate series (to_univariate).
  * metrics aggregated over all (series, window, step) cells (axis=None), NaN labels masked:
      MASE[0.5]  = mean |y - median_forecast| / seasonal_error(context, seasonality)
      CRPS ~ mean_weighted_sum_quantile_loss over tau 0.1..0.9
           = mean_tau  sum 2*|(y - q_tau) * (1[y < q_tau] - tau)| / sum |y|
  * seasonality = gluonts get_seasonality(freq): S 3600, T 1440, H 24, D 1, W 1, M 12, B 5, Q 4, else 1
    (divided by the multiple in "5T"/"10S"; 1 if not divisible).

The seasonal-naive baseline mirrors statsforecast.SeasonalNaive with `level` intervals (Gaussian,
sigma = sqrt(nansum(resid^2)/(n - m)), sigma_h = sigma*sqrt(floor(h/m)+1)), NaN context forward filled.

    python -m forecasting_engine.gifteval.harness selftest [config ...]
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterator

import numpy as np

GIFT = Path(os.environ.get("GIFT_ROOT", "data/gift_eval"))
OFFICIAL = GIFT / "_official"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TEST_SPLIT, MAX_WINDOW = 0.1, 20
MULT = {"short": 1, "medium": 10, "long": 15}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
SEASONALITY = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "B": 5, "Q": 4}
_ALIAS = {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "MIN": "T", "s": "S", "us": "U"}
PRETTY = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
          "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}

# (directory, subdir or None) for the 55 short-term configs; MED_LONG also get medium and long.
SHORT = ("m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H "
         "electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D "
         "us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing "
         "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
         "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T "
         "SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
         "jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H "
         "bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T "
         "bizitobs_l2c/H").split()
MED_LONG = set(("electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
                "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T "
                "jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application "
                "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H").split())


def _props() -> dict:
    import json
    return json.loads((OFFICIAL / "dataset_properties.json").read_text())


def config_name(ds: str, term: str) -> str:
    """Leaderboard name, e.g. 'saugeen/D/short' (matches all_results.csv)."""
    key = ds.split("/")[0].lower()
    key = PRETTY.get(key, key)
    freq = ds.split("/")[1] if "/" in ds else _props()[key]["frequency"]
    return f"{key}/{freq}/{term}"


def all_configs() -> list[tuple[str, str]]:
    """The 97 official (dataset path, term) pairs."""
    out = []
    for ds in SHORT:
        for term in ("short", "medium", "long"):
            if term != "short" and ds not in MED_LONG:
                continue
            out.append((ds, term))
    return out


def _base_freq(freq: str) -> tuple[str, int]:
    m = re.match(r"^(\d*)([A-Za-z]+)", freq)
    n = int(m.group(1) or 1)
    base = m.group(2)
    base = _ALIAS.get(base, base)
    base = {"MS": "M", "QS": "Q", "AS": "A", "YS": "A", "SM": "M"}.get(base, base)
    return base, n


def prediction_length(ds: str, freq: str, term: str) -> int:
    base, _ = _base_freq(freq)           # weekly anchors like W-WED -> 'W' (regex stops at '-')
    table = M4_PRED_LENGTH_MAP if "m4" in ds else PRED_LENGTH_MAP
    return MULT[term] * table[base]


def seasonality(freq: str) -> int:
    base, n = _base_freq(freq)
    s, rem = divmod(SEASONALITY.get(base, 1), n)
    return s if not rem else 1


class GiftTask:
    """One (dataset, term) evaluation task. Iterating yields the 5 protocol fields in order:
    contexts, targets, prediction_length, frequency, seasonality (`.name` is the leaderboard id).
    contexts: list of 1-D float arrays (one per series x window), targets: (n, prediction_length)."""

    def __init__(self, name, contexts, targets, pred_len, freq, season, windows, n_series):
        self.name, self.contexts, self.targets = name, contexts, targets
        self.prediction_length, self.frequency, self.seasonality = pred_len, freq, season
        self.windows, self.n_series = windows, n_series

    def __iter__(self):
        return iter((self.contexts, self.targets, self.prediction_length, self.frequency, self.seasonality))


def load_task(ds: str, term: str, root: Path = GIFT) -> GiftTask:
    import datasets
    hf = datasets.load_from_disk(str(root / ds)).with_format("numpy")
    freq = str(hf[0]["freq"])
    pl = prediction_length(ds, freq, term)
    series = []
    for row in hf:
        t = np.asarray(row["target"], dtype=np.float32)
        if t.ndim == 1:
            series.append(t)
        else:
            series.extend(t)                       # to_univariate: each variate its own series
    min_len = min(len(s) for s in series)
    windows = 1 if "m4" in ds else min(max(1, math.ceil(TEST_SPLIT * min_len / pl)), MAX_WINDOW)
    ctx, tgt = [], []
    for s in series:
        T = len(s)
        for k in range(windows):
            cut = T - pl * (windows - k)
            if cut < 1:
                continue
            ctx.append(s[:cut])
            tgt.append(s[cut:cut + pl])
    return GiftTask(config_name(ds, term), ctx, np.stack(tgt), pl, freq, seasonality(freq), windows, len(series))


def iter_configs(root: Path = GIFT, only: list[str] | None = None) -> Iterator[GiftTask]:
    """Iterate the official configurations (optionally filtered by leaderboard names like
    'm4_weekly/W/short'), yielding GiftTask objects that unpack as
    (contexts, targets, prediction_length, frequency, seasonality)."""
    for ds, term in all_configs():
        if only and config_name(ds, term) not in only:
            continue
        yield load_task(ds, term, root)


# ----------------------------------------------------------------------------- scoring
def seasonal_error(ctx: np.ndarray, m: int) -> float:
    if len(ctx) <= m:
        return np.nan
    return float(np.nanmean(np.abs(ctx[m:] - ctx[:-m])))


def score(task: GiftTask | tuple, quantile_forecasts: np.ndarray) -> dict:
    """quantile_forecasts: (n, 9, prediction_length) for QUANTILES (0.5 is the point forecast).
    Returns MASE and CRPS (mean weighted sum quantile loss), official aggregation (axis=None)."""
    ctxs, y, pl, _, m = tuple(task)
    q = np.asarray(quantile_forecasts, dtype=np.float64)
    assert q.shape == (len(ctxs), len(QUANTILES), pl), q.shape
    y = np.asarray(y, dtype=np.float64)
    valid = ~np.isnan(y)
    # MASE (0.5 quantile as point forecast)
    se = np.array([seasonal_error(c, m) for c in ctxs])
    with np.errstate(divide="ignore", invalid="ignore"):
        ase = np.abs(y - q[:, 4, :]) / se[:, None]
    ase = np.where(valid & np.isfinite(ase), ase, np.nan)
    mase = float(np.nanmean(ase))
    # weighted quantile loss
    ys = np.where(valid, y, 0.0)
    denom = np.abs(ys).sum()
    wql = []
    for i, tau in enumerate(QUANTILES):
        qi = q[:, i, :]
        ql = 2 * np.abs((ys - qi) * ((ys < qi).astype(float) - tau))
        wql.append(np.where(valid, ql, 0.0).sum() / denom)
    return {"MASE": mase, "CRPS": float(np.mean(wql))}


# ----------------------------------------------------------------------------- baseline
def seasonal_naive_quantiles(ctxs: list[np.ndarray], pl: int, m: int,
                             quantiles=QUANTILES) -> np.ndarray:
    """statsforecast SeasonalNaive with Gaussian intervals, as used in the official notebook."""
    z = np.array([NormalDist().inv_cdf(t) for t in quantiles])
    out = np.empty((len(ctxs), len(quantiles), pl))
    k = np.floor(np.arange(pl) / m)
    for i, c in enumerate(ctxs):
        c = np.asarray(c, dtype=np.float32).astype(np.float64)
        if np.isnan(c).any():                                # LastValueImputation
            idx = np.where(~np.isnan(c), np.arange(len(c)), 0)
            np.maximum.accumulate(idx, out=idx)
            c = c[idx]
            if np.isnan(c[0]):
                c = np.nan_to_num(c, nan=0.0)
        mm = min(m, len(c))
        last = c[-mm:]
        mean = np.tile(last, math.ceil(pl / mm))[:pl]
        n = len(c) - mm
        res = c[mm:] - c[:-mm] if n > 0 else np.zeros(0)
        sigma = math.sqrt(np.nansum(res ** 2) / n) if n > 0 else 0.0
        out[i] = mean[None, :] + z[:, None] * (sigma * np.sqrt(k + 1))[None, :]
    return out


def published(model: str = "Seasonal_Naive") -> dict[str, dict]:
    """Official leaderboard numbers from the repo's results/<model>/all_results.csv (saved under
    data/gift_eval/_official/)."""
    f = OFFICIAL / {"Seasonal_Naive": "seasonal_naive_all_results.csv", "Naive": "naive_all_results.csv"}[model]
    with open(f) as fh:
        return {r["dataset"]: {"MASE": float(r["eval_metrics/MASE[0.5]"]),
                               "CRPS": float(r["eval_metrics/mean_weighted_sum_quantile_loss"])}
                for r in csv.DictReader(fh)}


def selftest(names: list[str] | None = None) -> None:
    names = names or ["m4_weekly/W/short", "hospital/M/short", "m4_quarterly/Q/short",
                      "us_births/M/short", "saugeen/M/short"]
    pub = published()
    print(f"{'config':28s} {'n':>6s} {'MASE ours':>10s} {'published':>10s} {'CRPS ours':>10s} {'published':>10s}")
    for task in iter_configs(only=names):
        ctxs, y, pl, freq, m = task
        s = score(task, seasonal_naive_quantiles(ctxs, pl, m))
        p = pub[task.name]
        print(f"{task.name:28s} {len(ctxs):6d} {s['MASE']:10.4f} {p['MASE']:10.4f} "
              f"{s['CRPS']:10.4f} {p['CRPS']:10.4f}  (pl={pl}, season={m}, windows={task.windows})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest(sys.argv[2:] or None)
    elif sys.argv[1:2] == ["list"]:
        print(len(all_configs()), "configurations")
        for ds, t in all_configs():
            print(config_name(ds, t))
