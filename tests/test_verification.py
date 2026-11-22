import json
from pathlib import Path

from forecasting_engine.lab import core as C
from forecasting_engine.lab import scoreboard as S

V = Path(__file__).resolve().parents[1] / "verification"


def test_champions_match_the_scoreboard():
    champs = {x["family"]: x for x in json.loads((V / "champions.json").read_text())["layers"]}
    assert len(champs) == len(C.registry()["families"])
    for r in S.rows():
        if r["engine"] is not None:
            assert abs(champs[r["family"]]["holdout_skill"] - r["engine"]) < 1e-9


def test_manifest_has_hashes_and_delays():
    layers = json.loads((V / "manifest.json").read_text())["layers"]
    assert layers and all(len(x["sha256"]) > 0 and "drivers" in x for x in layers)
    assert all(len(h) == 64 for x in layers for h in x["sha256"].values())


def test_models_listed():
    names = [m["name"] for m in json.loads((V / "models.json").read_text())["models"]]
    assert {"Chronos-2", "TiRex-2", "Toto 2.0"} <= set(names)
