"""The shipped ledger and benchmark rows reproduce the published headline counts."""
from forecasting_engine.lab import scoreboard as SB


def test_headline_counts():
    rs = SB.rows()
    assert len(rs) == 66                                                    # 66 families, 65 of them with all three open models scored
    scored = [r for r in rs if r["margin"] is not None]
    assert len(scored) == 65
    assert sum(r["margin"] > 0 for r in scored) == 43                       # ahead of the best of Chronos-2, TiRex-2, Toto 2.0
    assert sum(r["margin_c2"] > 0 for r in rs if r["margin_c2"] is not None) == 58   # ahead of Chronos-2 alone


def test_known_rows_and_table():
    rs = {r["family"]: r for r in SB.rows()}
    assert abs(rs["load_h"]["engine"] - 0.519) < 0.001 and rs["load_h"]["best_fm"] == "Toto 2.0"
    assert abs(rs["load_h"]["margin"] - 0.174) < 0.002
    assert rs["health_w"]["margin"] < 0                                        # a lost layer stays lost
    md = SB.markdown(list(rs.values()))
    assert "| load_h | energy |" in md and "43 of 65" in md
