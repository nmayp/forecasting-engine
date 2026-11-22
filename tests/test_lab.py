"""Forecasting lab: what is known at each origin, windows that do not leak, the registry written once, the champion rule,
and the restricted families' values withheld from the public report."""
import json

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.lab import core as C
from forecasting_engine.lab import panel as PN
from forecasting_engine.lab import report as RP


def _fam(fid="syn_d", n=700, J=3, freq="D"):
    rng = np.random.default_rng(1)
    ix = pd.date_range("2018-01-01", periods=n, freq=freq, tz="UTC")
    t = np.arange(n)
    drv = rng.normal(0, 1, n)                     # a leading signal: it moves every series 10 days later
    Y = pd.DataFrame({cc: 50 + 5 * np.sin(t * 2 * np.pi / 7) + 3 * np.roll(drv, 10) + rng.normal(0, 1, n) for cc in ("ES", "FR", "DE", "IT")[:J]}, index=ix)
    D = [PN.Driver("signal", "markets", pd.DataFrame({"*": drv}, index=ix), 12.0)]
    return PN.Family(fid, "Synthetic", "markets", freq, "u", Y, 26.0, 0, 7, (7,), drivers=D)


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    fam = _fam()
    monkeypatch.setattr(C, "LAB", tmp_path)
    monkeypatch.setattr(C, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(C, "TRIALS", tmp_path / "trials.jsonl")
    monkeypatch.setattr(C, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(C, "_DESIGNS", {})
    monkeypatch.setattr(C, "MAX_EVAL", 40)
    monkeypatch.setattr(C, "GBM", dict(C.GBM, max_iter=60))
    monkeypatch.setattr(PN, "FAMILIES", ("syn_d",))
    monkeypatch.setattr(PN, "load", lambda fid: fam)
    from forecasting_engine import histories as HS
    monkeypatch.setattr(HS, "get_bytes", lambda name: json.dumps(dict(sha256={})).encode())
    C.register()
    return fam


def test_day_ahead_values_are_known_the_day_before():
    st = pd.DatetimeIndex(["2024-03-06 00:00", "2024-03-06 23:00"], tz="UTC")
    kt = C.known_times(st, 0, 12)
    assert (pd.to_datetime(kt, utc=True) == pd.Timestamp("2024-03-05 12:00", tz="UTC")).all()
    assert pd.Timestamp(C.known_times(st[:1], 26.0, None)[0], tz="UTC") == pd.Timestamp("2024-03-07 02:00", tz="UTC")


def test_targets_are_after_what_is_known(lab):
    d = C.Design(lab)
    ok = np.where(d.ok_origin)[0]
    assert (d.R[ok, 0] > d.L[ok]).all()                                 # never forecast a value already known
    assert (d.kt[np.clip(d.L[ok], 0, None)] <= d.tau[ok]).all()          # what is used was known at the origin


def test_development_truth_ends_before_the_holdout(lab):
    reg = C.registry()["families"][0]
    d = C.Design(lab)
    dev = C._idx(d, reg["dev_origins"])
    ho = pd.Timestamp(reg["holdout_start"]).value
    assert (d.kt[d.R[dev, -1]] < ho).all() and len(C._idx(d, reg["holdout_origins"])) >= 20


def test_the_registry_is_written_once(lab):
    with pytest.raises(SystemExit):
        C.register()


def test_drivers_are_read_as_known_at_the_origin(lab):
    dr = lab.drivers[0]
    tau = np.array([pd.Timestamp("2019-01-10 12:00", tz="UTC").value])
    v = C._driver_known(dr, "*", tau)
    # stamped 2019-01-10 00:00, known 12 h later: the value of the 10th is known at 12:00 on the 10th, not the 11th's
    assert v[0] == dr.data["*"].loc[pd.Timestamp("2019-01-10", tz="UTC")]


def test_loop_rule_reference_first_then_both_windows(lab):
    C.run("ref", ["syn_d"], "0", holdout=True)
    assert C.champion("syn_d") == "ref"
    r = C.run("gbm_all", ["syn_d"], "1")[0]
    assert "holdout" not in r and r["dev"]["pooled_skill"] > 0.05        # the driver leads the series: learnable
    h = C.run("gbm_all", ["syn_d"], "1", holdout=True)[0]
    assert h["accepted"] == (h["holdout"]["pooled_skill"] > 0 and h["dev"]["pooled_skill"] > 0)
    rep = RP.report()
    f = rep["families"][0]
    assert rep["trials"] == 3 and rep["holdout_looks"] == 2 and f["n_series"] == 3


def test_restricted_families_publish_scores_not_values(lab, monkeypatch):
    monkeypatch.setattr(RP, "RESTRICTED", {"syn_d"})
    C.run("ref", ["syn_d"], "0", holdout=True)
    f = RP.report()["families"][0]
    assert f["values_withheld"] and f["actual"] is None and f["best"]["paths"] is None and f["best"]["holdout"]["pooled_skill"] == 0.0




def test_dev_blocks_and_gate_rule():
    t = np.arange(50)
    b = C.dev_blocks(t)
    assert len(b) == 4 and np.array_equal(np.concatenate(b), t)          # consecutive, every origin once
    assert len(C.dev_blocks(t[:30])) == 2 and len(C.dev_blocks(t[:5])) == 1  # fewer blocks when development is short
    g = C.gate(0.10, [0.2, 0.1, 0.0, 0.3], 0.05, [0.1, 0.1, 0.1, 0.1])  # 2 of 4 wins (the tie counts against): no majority
    assert not g["qualified"] and g["blocks_won"] == 2
    assert C.gate(0.10, [0.2, 0.2, 0.0, 0.3], 0.05, [0.1, 0.1, 0.1, 0.1])["qualified"]
    assert not C.gate(0.05, [0.2, 0.2, 0.2, 0.2], 0.05, [0.1] * 4)["qualified"]   # a tie on the whole window counts against


def test_holdout_look_only_after_the_dev_gate(lab, monkeypatch):
    monkeypatch.setattr(C, "_PERSIST", {"on": False})
    monkeypatch.setattr(C, "_MEMO", {})

    def _truth(d, eo):
        return np.stack([d.truth(j)[eo] for j in range(d.J)], axis=1)        # (n, J, H)

    def good(d, refs, eo, tf):                                               # perfect everywhere (a leak, fine for a test)
        return _truth(d, eo)

    def split(d, refs, eo, tf):                                              # perfect on the first block, 10 % worse than the reference after
        y, rf = _truth(d, eo), C.m_ref(d, refs, eo, None)
        P = y + 1.1 * (rf - y)
        P[C.dev_blocks(d.tau[eo])[0]] = y[C.dev_blocks(d.tau[eo])[0]]
        return np.where(np.isfinite(P), P, rf)

    monkeypatch.setitem(C.METHODS, "t_good", good)
    monkeypatch.setitem(C.METHODS, "t_split", split)
    C.run("ref", ["syn_d"], "0", holdout=True)
    s = C.run("t_split", ["syn_d"], "1", holdout=True)[0]
    nb = s["n_blocks"]
    assert nb >= 2 and len(s["dev_blocks"]) == nb and s["vs_dev_blocks"] == [0.0] * nb
    assert s["dev"]["pooled_skill"] > 0 and s["blocks_won"] == 1           # better on the whole window, not on most blocks
    assert s["qualified"] is False and s["accepted"] is False and "holdout" not in s and "holdout_paths" not in s
    g = C.run("t_good", ["syn_d"], "1", holdout=True)[0]
    assert g["qualified"] and g["blocks_won"] == nb and g["holdout"]["pooled_skill"] > 0.9 and g["accepted"]
    assert C.champion("syn_d") == "t_good"
    s2 = C.run("t_split", ["syn_d"], "2", holdout=True)[0]                   # the champion's blocks are recomputed (memo_method)
    assert s2["qualified"] is False and s2["vs_dev"] > 0.9 and "holdout" not in s2
    rows = C.ledger()
    rows.append(dict(rows[1], accepted=True))                                # an accepted row must also be qualified
    monkeypatch.setattr(C, "ledger", lambda: rows)
    assert C.champion("syn_d") == "t_good"
    f = RP.report()["families"][0]
    assert C.holdout_looks(rows, "syn_d") == 2 and f["holdout_looks"] == 2 and f["gated_out"] == 3
