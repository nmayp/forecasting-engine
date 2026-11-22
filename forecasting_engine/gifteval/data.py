"""Data layer reproducing the official GIFT-Eval test_data (gift_eval.data.Dataset) without gluonts.

A Split holds *units* (one per row x window; a multivariate row keeps its variates together, as the authors'
Chronos-2 / Toto notebooks feed them) plus the flat univariate view used by scoring (position order identical
to gift_eval_harness.load_task for the test split: (row, variate, window)).
Validation split: the same construction on every series truncated before the test region
(T_val = T - prediction_length * windows), rows subsampled to 300 (seed 0). No test labels are touched.
"""
import math
from pathlib import Path
import numpy as np

from forecasting_engine.gifteval import harness as H


class Split:
    pass


def build_split(ds: str, term: str, split: str, root: Path, max_rows: int = 300, seed: int = 0) -> Split:
    import datasets
    hf = datasets.load_from_disk(str(root / ds)).with_format("numpy")
    freq = str(hf[0]["freq"])
    pl = H.prediction_length(ds, freq, term)
    n_rows = len(hf)
    first = np.asarray(hf[0]["target"])
    V = first.shape[0] if first.ndim > 1 else 1
    lens = [np.asarray(hf[i]["target"]).shape[-1] for i in range(n_rows)] if n_rows < 50000 else None
    if lens is None:
        raise RuntimeError("too many rows")
    min_len = min(lens)
    windows = 1 if "m4" in ds else min(max(1, math.ceil(H.TEST_SPLIT * min_len / pl)), H.MAX_WINDOW)

    rows = np.arange(n_rows)
    if split == "val" and n_rows > max_rows:
        rows = np.sort(np.random.RandomState(seed).choice(n_rows, max_rows, replace=False))

    units = []
    for r in rows:
        rec = hf[int(r)]
        tgt = np.asarray(rec["target"], dtype=np.float32)
        tgt = tgt.reshape(1, -1) if tgt.ndim == 1 else tgt
        pf = None
        if "past_feat_dynamic_real" in rec and np.asarray(rec["past_feat_dynamic_real"]).size:
            pf = np.asarray(rec["past_feat_dynamic_real"], dtype=np.float32)
            pf = pf.reshape(1, -1) if pf.ndim == 1 else pf
        T = tgt.shape[1]
        Tcut = T if split == "test" else T - pl * windows
        for k in range(windows):
            cut = Tcut - pl * (windows - k)
            if cut < 1:
                continue
            assert cut + pl <= Tcut <= T
            units.append(dict(row=int(r), k=k, cut=cut, tgt=tgt, pf=pf))

    n_units = len(units)
    positions = np.zeros((n_units, V), dtype=np.int64)
    for i, u in enumerate(units):
        if split == "test":
            positions[i] = (u["row"] * V + np.arange(V)) * windows + u["k"]
        else:
            positions[i] = i * V + np.arange(V)
    n_flat = n_units * V
    ctxs = [None] * n_flat
    tgts = np.zeros((n_flat, pl), dtype=np.float32)
    for i, u in enumerate(units):
        for v in range(V):
            p = positions[i, v]
            ctxs[p] = u["tgt"][v, :u["cut"]]
            tgts[p] = u["tgt"][v, u["cut"]:u["cut"] + pl]
    s = Split()
    s.name = H.config_name(ds, term)
    s.units, s.positions, s.V, s.pl, s.windows, s.freq = units, positions, V, pl, windows, freq
    s.ctxs, s.tgts, s.n_flat, s.n_rows = ctxs, tgts, n_flat, len(rows)
    s.season = H.seasonality(freq)
    return s


def to_task(s: Split):
    return H.GiftTask(s.name, s.ctxs, s.tgts, s.pl, s.freq, s.season, s.windows, s.n_rows)
