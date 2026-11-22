"""Cascade benchmark: event catalogue, link ablation, and the aggregation over the shipped cell results."""
import json
from pathlib import Path

from forecasting_engine.cascade import cascade as CA

ROOT = Path(__file__).resolve().parents[1]


def test_event_catalogue():
    ev = CA.load_events()
    assert len(ev) == 30
    splits = {e["split"] for e in ev}
    assert {"dev", "check"} <= splits
    for e in ev:
        assert e["onset"] and e["cells"], e["id"]
        for c in e["cells"]:
            assert c["family"] and c["series"], e["id"]
    assert sum(len(e["cells"]) for e in ev if e["split"] in ("dev", "check")) >= 160


def test_links_off_removes_cross_layer_inputs():
    assert CA.links_off("chronos2x", "energy") == "chronos2"
    assert CA.links_off("tirex2c", "energy") == "tirex2"
    t = CA.links_off("gbm_only-network+energy", "energy")
    assert "network" not in t and "energy" in t
    b = CA.links_off("blend:chronos2x|gbm_all_reg", "transport")
    assert b.startswith("blend:chronos2|gbm_only-")


def test_aggregate_shipped_cells():
    rows = []
    for name in ("cells_dev", "cells_chk_a", "cells_chk_b", "cells_chk_c"):
        rows += json.loads((ROOT / "results" / "cascade" / f"{name}.json").read_text())["rows"]
    dev = CA.aggregate(rows, "dev")
    chk = CA.aggregate(rows, "check")
    assert dev["events"] == 21 and dev["cells"] > 100
    assert chk["events"] == 8 and chk["cells"] > 30
    for tab in (dev["table"], chk["table"]):
        assert {"K_full", "K_off", "chronos2"} <= set(tab)
        assert -1 < tab["K_full"]["skill"] < 1
