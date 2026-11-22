"""History store round trip and the Eurostat JSON-stat parser, offline."""
import numpy as np
import pandas as pd

from forecasting_engine import eurostat, histories as HS


def test_parquet_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("FE_HISTORY_STORE", str(tmp_path))
    df = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    HS.put_parquet("x/y.parquet", df)
    back = HS.get_parquet("x/y.parquet")
    assert back is not None and np.allclose(back["v"], df["v"])
    assert HS.get_parquet("missing.parquet") is None
    assert HS.list_names("x") == ["x/y.parquet"]


def test_restricted_flag(monkeypatch):
    monkeypatch.delenv("FE_ALLOW_RESTRICTED", raising=False)
    assert HS.restricted_ok() is False
    monkeypatch.setenv("FE_ALLOW_RESTRICTED", "1")
    assert HS.restricted_ok() is True


def test_entsoe_token_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("ENTSOE_TOKEN", raising=False)
    HS._TOKEN.clear()
    try:
        HS._token()
        raise AssertionError("expected EntsoeError")
    except HS.EntsoeError as e:
        assert "ENTSOE_TOKEN" in str(e)


def test_parse_jsonstat():
    j = {"id": ["geo", "time"], "size": [2, 2], "dimension": {"geo": {"category": {"index": {"DE": 0, "FR": 1}}}, "time": {"category": {"index": ["2024-01", "2024-02"]}}},
         "value": {"0": 1.0, "1": 2.0, "3": 4.0}}
    d = eurostat.parse_jsonstat(j)
    assert len(d) == 3 and d.loc[d["geo"].eq("FR") & d["time"].eq("2024-02"), "value"].iloc[0] == 4.0
