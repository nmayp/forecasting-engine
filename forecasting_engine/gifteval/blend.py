"""Helpers for the validation-selected blend: fast scoring, member subsets, combiners, prediction files.

Predictions are read from $GIFT_PREDS (default `preds/`), laid out as `<split>/<config with / as __>/<member>.npy`, each an array
(n_series, 9 quantiles, prediction_length) as written by `forecast.py`.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np

from forecasting_engine.gifteval import harness as H

MEMBERS = ["Toto-2.0", "Chronos-2", "TiRex-2", "TimesFM-2.5"]
PREDS = Path(os.environ.get("GIFT_PREDS", "preds"))
OUT = Path(os.environ.get("GIFT_OUT", "out"))


def safe(name: str) -> str:
    return name.replace("/", "__")


def load_pred(split: str, name: str, member: str):
    f = PREDS / split / safe(name) / f"{member}.npy"
    return np.load(f) if f.exists() else None


class FastScorer:
    """Same MASE / CRPS as harness.score, with the seasonal errors and masks precomputed once."""

    def __init__(self, ctxs, y, pl, m):
        self.y = np.asarray(y, dtype=np.float64)
        self.valid = ~np.isnan(self.y)
        self.se = np.array([H.seasonal_error(c, m) for c in ctxs])
        self.ys = np.where(self.valid, self.y, 0.0)
        self.denom = np.abs(self.ys).sum()

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ase = np.abs(self.y - q[:, 4, :]) / self.se[:, None]
        ase = np.where(self.valid & np.isfinite(ase), ase, np.nan)
        mase = float(np.nanmean(ase))
        wql = []
        for i, tau in enumerate(H.QUANTILES):
            qi = q[:, i, :]
            ql = 2 * np.abs((self.ys - qi) * ((self.ys < qi).astype(float) - tau))
            wql.append(np.where(self.valid, ql, 0.0).sum() / self.denom)
        return {"MASE": mase, "CRPS": float(np.mean(wql))}


def combine(arr, comb):
    if comb == "mean":
        return np.mean(arr, axis=0)
    if comb == "median":
        return np.median(arr, axis=0)
    s = np.sort(arr, axis=0)
    return np.mean(s[1:-1], axis=0)           # trimmed mean: drop the lowest and highest member


def combos(members):
    out = []
    for r in range(1, len(members) + 1):
        for sub in itertools.combinations(members, r):
            for comb in ("mean", "median", "trimmed_mean"):
                if comb == "trimmed_mean" and len(sub) < 3:
                    continue
                if len(sub) == 1 and comb != "mean":
                    continue
                out.append((sub, comb))
    return out


def ckey(sub, comb):
    return "|".join(sub) + "#" + comb
