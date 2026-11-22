"""GIFT-Eval harness maths on synthetic series (the real datasets are not shipped), the blend helpers, and the shipped results file."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from forecasting_engine.gifteval import blend as B
from forecasting_engine.gifteval import harness as H

ROOT = Path(__file__).resolve().parents[1]


def test_prediction_lengths_and_seasonality():
    assert H.seasonality("H") == 24 and H.seasonality("M") == 12 and H.seasonality("D") == 1
    assert H.prediction_length("electricity/H", "H", "short") == 48
    assert H.prediction_length("electricity/H", "H", "medium") == 480
    assert H.prediction_length("m4_monthly", "M", "short") == 18
    assert len(H.all_configs()) == 97


def test_fast_scorer_matches_harness_score():
    rng = np.random.default_rng(0)
    n, pl, m = 6, 12, 4
    ctxs = [np.cumsum(rng.normal(size=60)).astype(np.float32) + 20 for _ in range(n)]
    y = (np.stack([c[-1] + np.cumsum(rng.normal(size=pl)) for c in ctxs])).astype(np.float32)
    q = np.sort(np.stack([y + rng.normal(0, 1, y.shape) + k for k in np.linspace(-1, 1, 9)], axis=1), axis=1)
    task = H.GiftTask("syn", ctxs, y, pl, "Q", m, 1, n)
    a = H.score(task, q)
    b = B.FastScorer(ctxs, y, pl, m)(q)
    assert a["MASE"] == pytest.approx(b["MASE"], rel=1e-9) and a["CRPS"] == pytest.approx(b["CRPS"], rel=1e-9)


def test_seasonal_naive_repeats_last_season():
    c = np.tile(np.arange(4, dtype=np.float32), 10)
    q = H.seasonal_naive_quantiles([c], 8, 4)
    assert q.shape == (1, 9, 8)
    assert np.allclose(q[0, 4], np.tile(np.arange(4), 2))               # the median is the last season repeated


def test_combiners_and_member_subsets():
    a = np.stack([np.full((2, 9, 3), v) for v in (1.0, 2.0, 10.0)])
    assert np.allclose(B.combine(a, "mean"), 13 / 3) and np.allclose(B.combine(a, "median"), 2.0) and np.allclose(B.combine(a, "trimmed_mean"), 2.0)
    cs = B.combos(B.MEMBERS)
    assert len(cs) == 4 + 6 * 2 + 4 * 3 + 1 * 3                          # singles; pairs (mean, median); triples and the full set (mean, median, trimmed)
    assert B.ckey(("A", "B"), "mean") == "A|B#mean"


def test_shipped_result_file():
    rows = list(csv.DictReader(open(ROOT / "results" / "gifteval" / "all_results.csv")))
    assert len(rows) == 97 and {r["model"] for r in rows} == {"FM-Blend"}
    assert all(math.isfinite(float(r["eval_metrics/MASE[0.5]"])) for r in rows)
