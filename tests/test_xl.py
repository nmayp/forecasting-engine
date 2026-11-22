"""Cross-layer round (xl_panel, xl_lib): the panel is causal and the gap is what the target has not seen."""
import numpy as np
import pandas as pd

from forecasting_engine.lab import xl_lib as XLB
from forecasting_engine.lab import xl_panel as XP


def _series(n=120, seed=0):
    ix = pd.date_range("2010-01-01", periods=n, freq="MS", tz="UTC")
    return pd.DataFrame({"a": np.random.default_rng(seed).standard_normal(n).cumsum(), "b": np.random.default_rng(seed + 1).standard_normal(n).cumsum()}, index=ix)


def test_layer_index_is_causal():
    y = _series()
    base = XP.to_grid(XP.layer_index(y, "M", 24 * 30.44 + 24 * 20, 3), "M")
    y2 = y.copy()
    y2.iloc[-6:] += 100.0                                   # change the last six months
    new = XP.to_grid(XP.layer_index(y2, "M", 24 * 30.44 + 24 * 20, 3), "M")
    cut = pd.Timestamp("2019-06-01", tz="UTC")             # long before the changed months were published
    assert np.allclose(base[:cut].dropna().to_numpy(), new[:cut].dropna().to_numpy())


def test_to_grid_waits_for_publication():
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-03-10 10:00", "2020-04-10 10:00"], tz="UTC"))     # stamped when published
    g = XP.to_grid(s, "M")
    assert np.isnan(g[pd.Timestamp("2020-03-10", tz="UTC")])                  # published later that day: not known at the start of the day
    assert g[pd.Timestamp("2020-03-11", tz="UTC")] == 1.0
    assert g[pd.Timestamp("2020-04-11", tz="UTC")] == 2.0


def test_data_time_grid_and_end_dates():
    y = _series()
    sd = XP.layer_index(y, "M", 24 * 30.44, 1, data_time=True)
    sk = XP.layer_index(y, "M", 24 * 30.44 + 24 * 10, 1)
    assert (sk.index - sd.index).min() >= pd.Timedelta(days=9)             # published after the period it describes ends
    assert (XP.to_data_grid(sd, "M").dropna().index >= sd.index[0].floor("D")).all()


def test_loadings_use_only_the_past(monkeypatch):
    rng = np.random.default_rng(0)
    Z = pd.DataFrame(rng.standard_normal((len(XP.GRID), 8)), index=XP.GRID, columns=[f"l{i}:s" for i in range(8)]).astype("float32")
    monkeypatch.setattr(XP, "_Z", Z)
    monkeypatch.setattr(XP, "_ZD", Z.copy())
    monkeypatch.setattr(XP, "_ZE", Z.copy())
    XLB.set_extra("zz_m", None)
    rho = pd.Timestamp("2018-01-01", tz="UTC").value
    a = XLB.loadings("zz_m", rho, 2)
    Z2 = Z.copy()
    Z2.iloc[XLB._day(rho):] += 50.0                                        # everything from rho on changes
    monkeypatch.setattr(XP, "_Z", Z2)
    monkeypatch.setattr(XP, "_ZD", Z2.copy())
    monkeypatch.setattr(XP, "_ZE", Z2.copy())
    XLB.set_extra("zz_m", None)
    b = XLB.loadings("zz_m", rho, 2)
    assert np.allclose(a[0], b[0])                                         # the loadings estimated up to rho do not move
