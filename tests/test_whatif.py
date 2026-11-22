"""Back-tested what-if mechanisms (forecasting_engine.whatif): heat -> load, strikes -> air traffic, rates -> cost of borrowing,
oil -> pump prices -> HICP, chokepoints -> freight. Offline: the mechanisms read only forecasting_engine/whatif/data/*.json.

Every result must carry status, as_of, ONE back-test line, caveats and p10-p90 ranges (meta.whatif); the failed links must not be wired."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from forecasting_engine.whatif import mech_kit as MK  # noqa: E402
from forecasting_engine.whatif import registry as R  # noqa: E402

WHATIF_KEYS = ("mechanism", "status", "status_note", "as_of", "run_date", "tested_scope", "backtest", "caveats", "ranges")


def run(sid, req):
    out = R.run(sid, req, {})
    assert out.get("available", True) is not False, out.get("reason")
    return out


def check_whatif(out, sid, status):
    wi = out["meta"]["whatif"]
    assert all(wi.get(k) for k in WHATIF_KEYS), [k for k in WHATIF_KEYS if not wi.get(k)]
    assert wi["status"] == status == out["meta"]["status"] and wi["status"] in R.STATUSES
    assert "\n" not in wi["backtest"] and len(wi["backtest"]) < 1200            # one line
    assert len(wi["caveats"]) >= 3 and "p10-p90" in wi["ranges"]
    assert out["validation"]["note"] == wi["backtest"] and out["validation"]["rule"]
    assert out["frames"] and all("values" in f for f in out["frames"])
    for name, s in out["series"].items():
        n = len(s["t_h"])
        assert n == len(s["p10"]) == len(s["p50"]) == len(s["p90"]), name
        assert s["unit"]
        for a, b, c in zip(s["p10"], s["p50"], s["p90"]):
            assert a <= b + 1e-9 and b <= c + 1e-9, (sid, name, a, b, c)
    json.dumps(out)
    return wi


def test_registered_with_examples_and_statuses():
    EX = R.EX
    ids = {s["id"]: s for s in R.catalog()}
    want = {"heat_load": "screening", "strike_air": "experimental", "rates_credit": "screening", "oil_pump_hicp": "screening", "choke_freight": "experimental"}
    for sid, st in want.items():
        assert sid in ids and ids[sid]["status"] == st, sid
        assert sid in EX.EXAMPLES
        out = run(sid, EX.EXAMPLES[sid]["request"])
        check_whatif(out, sid, st)


# ------------------------------------------------------------------------------------------------ kit
def test_bands():
    b = MK.mult_band([10.0, -4.0, 0.0], 0.2)
    assert b["p10"][0] < 10 < b["p90"][0] and b["p10"][1] < -4 < b["p90"][1] and b["p10"][2] == b["p90"][2] == 0
    assert abs(b["p90"][0] / 10 - math.exp(MK.Z90 * 0.2)) < 1e-12
    r = MK.ratio_band([0.5], {"p10": 0.6, "p50": 0.9, "p90": 1.2})
    assert np.allclose([r["p10"][0], r["p50"][0], r["p90"][0]], [0.3, 0.45, 0.6])


def test_num_list_validation():
    assert list(MK.num_list(3, "x", 3)) == [3, 3, 3]
    for bad in ([], "a", [1, float("nan")], [1, 2], True):
        with pytest.raises(ValueError):
            MK.num_list(bad, "x", 3)
    with pytest.raises(ValueError):
        MK.num_list([100], "x", 1, 0, 50)


# ------------------------------------------------------------------------------------------------ heat -> load
HEAT = {"area": {"country": ["ES", "IT"]}, "scenario": {"kind": "custom", "start": "2026-07-14", "tmean_c": [26, 28, 30, 31, 32, 30, 27]}}


def test_heat_load_response_and_ranges():
    out = run("heat_load", HEAT)
    wi = check_whatif(out, "heat_load", "screening")
    assert "ES" in out["summary"] and "IT" in out["summary"]
    es = out["summary"]["ES"]
    assert es["peak_excess_pct"]["p10"] < es["peak_excess_pct"]["p50"] < es["peak_excess_pct"]["p90"]
    assert es["peak_day"] == "2026-07-18"                                           # the hottest day
    s = out["series"]["ES_excess_pct"]
    assert s["p50"][4] == max(s["p50"]) and s["p50"][0] < s["p50"][4]
    assert len(out["series"]["ES_hourly_excess_pct"]["t_h"]) == 7 * 24
    assert "licence" in wi and "open question" in wi["licence"]                     # the ENTSO-E question stays visible
    assert "load_d" not in json.dumps(out["summary"])
    assert out["meta"]["sources"] and "restricted" in " ".join(wi["caveats"] + [wi["licence"]])


def test_heat_load_matches_the_degree_day_formula():
    P = MK.params("heat_load")["countries"]["ES"]
    n = 10
    ref = [P["tc"] + 3] * n
    path = [P["tc"] + 8] * n                                                       # constant, so the inertia filter is flat
    out = run("heat_load", {"area": {"country": ["ES"]}, "scenario": {"kind": "custom", "tmean_c": path, "reference_c": ref}, "params": {"hourly": False}})
    want = 100 * math.expm1(P["sc"] * 5)
    assert abs(out["series"]["ES_excess_pct"]["p50"][-1] - want) < 0.05
    # zero excess when the path is the reference; negative when cooler
    z = run("heat_load", {"area": {"country": ["ES"]}, "scenario": {"kind": "custom", "tmean_c": ref, "reference_c": ref}, "params": {"hourly": False}})
    assert max(abs(x) for x in z["series"]["ES_excess_pct"]["p50"]) < 1e-9
    c = run("heat_load", {"area": {"country": ["ES"]}, "scenario": {"kind": "custom", "tmean_c": [P["tc"]] * n, "reference_c": ref}, "params": {"hourly": False}})
    assert c["series"]["ES_excess_pct"]["p50"][-1] < 0


def test_heat_load_baseline_anomaly_and_errors():
    out = run("heat_load", {"area": {"country": ["FR"]}, "scenario": {"kind": "custom", "tmean_c": [24, 26, 28], "baseline_mw": 50000, "start": "2026-07-01"}, "params": {"hourly": False}})
    mw, pc = out["series"]["FR_excess_mw"]["p50"], out["series"]["FR_excess_pct"]["p50"]
    assert all(abs(a - 500 * b) < 1e-6 * max(1, abs(a)) + 1e-3 for a, b in zip(mw, pc))
    assert out["summary"]["FR"]["cumulative_excess_gwh"]["p50"] is not None
    an = run("heat_load", {"area": {"country": ["DE"]}, "scenario": {"kind": "anomaly", "anomaly_c": 6, "days": 5, "start": "2026-07-14"}, "params": {"hourly": False}})
    assert len(an["series"]["DE_excess_pct"]["t_h"]) == 5 and an["series"]["DE_excess_pct"]["p50"][0] > 0
    for bad in ({"area": {"country": ["ZZ"]}, "scenario": HEAT["scenario"]}, {"area": {"country": ["ES"]}, "scenario": {"kind": "custom", "tmean_c": [90]}},
                {"area": {"country": ["ES"]}, "scenario": {"kind": "forecast"}}, {"area": {"lat": 40, "lon": 0, "radius_km": 5}, "scenario": HEAT["scenario"]}):
        assert R.run("heat_load", bad, {}).get("available") is False


def test_heat_load_warns_outside_tested_range():
    out = run("heat_load", {"area": {"country": ["ES"]}, "scenario": {"kind": "custom", "tmean_c": [38, 40], "start": "2026-07-14"}})
    assert any("beyond the range" in w for w in out["meta"]["warnings"])


# ------------------------------------------------------------------------------------------------ strikes -> air traffic
STRIKE = {"area": {"country": ["BE"]}, "scenario": {"kind": "custom", "start": "2026-11-24", "days": 3, "intensity": 0.75}}


def test_strike_air_share_is_k_times_intensity_and_relative_only():
    out = run("strike_air", STRIKE)
    wi = check_whatif(out, "strike_air", "experimental")
    s = out["series"]["BE_flights_lost_share"]
    P = MK.params("strike_air")
    q = P["ratio_realised_over_predicted"]["announced"]
    assert abs(s["p50"][0] - 0.75 * q["p50"]) < 1e-3 and s["p10"][0] < s["p50"][0] < s["p90"][0]
    assert set(out["series"]) == {"BE_flights_lost_share"}                         # no flight counts without a baseline
    assert "not shown" in out["summary"]["BE"]["flights_lost_total"]
    assert "EUROCONTROL" in wi["licence"] and any("Ports" in x for x in wi["links_not_wired"])
    assert "experimental" in wi["status_note"] or "licence" in wi["status_note"]


def test_strike_air_baseline_dates_cap_and_untested_country():
    req = {"area": {"country": ["FR"]}, "scenario": {"kind": "custom", "dates": ["2026-11-03", "2026-11-05"], "intensity": [0.25, 1.0]}, "params": {"baseline_flights": 8000, "intensity_basis": "estimate"}}
    out = run("strike_air", req)
    s = out["series"]["FR_flights_lost_share"]
    assert len(s["t_h"]) == 3 and s["p50"][1] == 0 and max(s["p90"]) <= 0.98                # the day between is not a strike day; capped at 0.98
    assert out["series"]["FR_flights_lost"]["p50"][0] == pytest.approx(8000 * s["p50"][0])
    assert out["summary"]["FR"]["flights_lost_total"]["p50"] > 0
    est = MK.params("strike_air")["ratio_realised_over_predicted"]
    assert est["estimate"]["p90"] - est["estimate"]["p10"] > est["announced"]["p90"] - est["announced"]["p10"]      # a guessed intensity has a wider band
    o2 = run("strike_air", {"area": {"country": ["SE"]}, "scenario": {"kind": "custom", "days": 1, "intensity": 0.5}})
    assert any("not among the tested" in w for w in o2["meta"]["warnings"])
    for bad in ({"area": {"country": ["BE"]}, "scenario": {"kind": "custom", "days": 2}},
                {"area": {"country": ["BE"]}, "scenario": {"kind": "custom", "days": 2, "intensity": 1.5}},
                {"area": {"country": ["BE"]}, "scenario": {"kind": "custom", "days": 2, "intensity": 0.5}, "params": {"intensity_basis": "x"}},
                {"area": {"country": ["BE"]}, "scenario": {"kind": "custom", "days": 400, "intensity": 0.5}}):
        assert R.run("strike_air", bad, {}).get("available") is False


# ------------------------------------------------------------------------------------------------ rates -> cost of borrowing
def _rates(path, **kw):
    sc = {"kind": "custom", "rate_path_pct": path}
    sc.update(kw)
    return run("rates_credit", {"area": {"country": ["DE", "IT"]}, "scenario": sc})


def test_rates_direction_size_and_flags():
    cut = _rates([2.25, 2.0, 2.0, 1.75, 1.75, 1.5])
    hike = _rates([2.75, 3.0, 3.25, 3.5, 3.5, 3.5], pricing="surprise")
    wi = check_whatif(cut, "rates_credit", "screening")
    f_cut, f_hike = cut["summary"]["DE"]["firms"], hike["summary"]["DE"]["firms"]
    assert f_cut["effect_of_your_path_vs_policy_held_pp"] < 0 < f_hike["effect_of_your_path_vs_policy_held_pp"]
    assert f_cut["change_at_end_pp"]["p10"] < f_cut["change_at_end_pp"]["p50"] < f_cut["change_at_end_pp"]["p90"]
    assert f_cut["status"] == "screening" and cut["summary"]["DE"]["house_purchase"]["status"] == "experimental"
    # long-run pass-through close to complete, as published (0.7-1.0)
    assert 0.6 < f_cut["long_run_pass_through"] < 1.1
    text = " ".join(wi["caveats"])
    assert "late" in text or "later" in text                                       # the 4-6 month timing failure
    assert "unpriced" in text or "priced" in text                                  # the unpriced-move failure
    assert any("loan volumes" in x for x in wi["links_not_wired"])
    assert any("Euribor" in w for w in cut["meta"]["warnings"])                    # a cut the market has not priced is flagged
    assert "timing" in wi["backtest"].lower()


def test_rates_steps_input_and_errors():
    a = _rates([2.0] * 12)
    b = run("rates_credit", {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "steps": [{"month": "2026-10", "rate_pct": 2.0}]}})
    assert a["summary"]["DE"]["firms"]["end_of_path_pct"] == b["summary"]["DE"]["firms"]["end_of_path_pct"]
    c = run("rates_credit", {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "steps": [{"month": "2027-01", "change_pp": -0.5}], "months": 6}})
    assert len(c["series"]["DE_firms_rate_pct"]["t_h"]) == 6
    for bad in ({"area": {"country": ["DE"]}, "scenario": {"kind": "custom"}}, {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "rate_path_pct": [30]}},
                {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "rate_path_pct": [2], "months": 24}}, {"area": {"country": ["GB"]}, "scenario": {"kind": "custom", "rate_path_pct": [2]}},
                {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "rate_path_pct": [2], "pricing": "x"}}):
        assert R.run("rates_credit", bad, {}).get("available") is False


# ------------------------------------------------------------------------------------------------ oil -> pump prices -> HICP
OIL = {"area": {"country": ["DE", "ES", "MT"]}, "scenario": {"kind": "custom", "product_change_pct": [0, 10, 20, 30, 30, 30, 30, 30], "gas_delta_eur_mwh": [0, 20, 20, 20], "elec_delta_eur_mwh": [0, 30, 30, 30]}}


def test_oil_pump_hicp_direction_size_and_flags():
    up = run("oil_pump_hicp", OIL)
    wi = check_whatif(up, "oil_pump_hicp", "screening")
    dn = run("oil_pump_hicp", {"area": OIL["area"], "scenario": {"kind": "custom", "product_change_pct": [0, -10, -20, -30, -30, -30, -30, -30]}})
    d_up, d_dn = up["summary"]["DE"], dn["summary"]["DE"]
    assert d_up["diesel_pump"]["effect_vs_prices_flat_pct"] > 5 > -5 > d_dn["diesel_pump"]["effect_vs_prices_flat_pct"]
    assert d_up["hicp_fuel"]["cumulative_effect_pct"]["p50"] > 0 > d_dn["hicp_fuel"]["cumulative_effect_pct"]["p50"]
    assert d_up["hicp_fuel"]["cumulative_effect_pct"]["p10"] < d_up["hicp_fuel"]["cumulative_effect_pct"]["p90"]
    # pump price: the long-run response to a wholesale change is close to complete (0.6-1.1 EUR per EUR, with the tax factor), the flat run is below the scenario
    flat, scen = up["series"]["DE_diesel_pump_if_prices_flat_eur_l"]["p50"], up["series"]["DE_diesel_pump_eur_l"]["p50"]
    prod = MK.params("oil_pump_hicp")["pump"]["diesel"]["product"]["x_last_eur_l"]
    ratio = (scen[-1] - flat[-1]) / (0.30 * prod)
    assert 0.6 < ratio < 1.5, ratio
    assert up["summary"]["DE"]["hicp_elc_gas"]["status"] == "experimental" and d_up["hicp_fuel"]["status"] == "screening"
    assert "headline_contribution_pp" in d_up and wi["links_not_wired"]
    assert any("Excise" in c or "excise" in c for c in wi["caveats"]) and "IMF" in wi["licence"]
    # the electricity-gas band is the widest (0.11 of country-months within +-20 %)
    g = d_up["hicp_elc_gas"]["cumulative_effect_pct"]
    f = d_up["hicp_fuel"]["cumulative_effect_pct"]
    assert (g["p90"] / g["p50"]) > (f["p90"] / f["p50"])


def test_oil_pump_hicp_inputs_and_errors():
    by_brent = run("oil_pump_hicp", {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "brent_usd_bbl": [150, 160, 170, 170], "brent_now_usd_bbl": 130}})
    assert by_brent["summary"]["DE"]["diesel_pump"]["effect_vs_prices_flat_pct"] > 0
    pump = run("oil_pump_hicp", {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "pump_change_pct": [0, 5, 10, 10]}})
    s1, s2 = pump["summary"]["DE"]["hicp_fuel"]["cumulative_effect_pct"], by_brent["summary"]["DE"]["hicp_fuel"]["cumulative_effect_pct"]
    assert (s1["p90"] / s1["p50"]) < (s2["p90"] / s2["p50"])                       # given the pump path the band is narrower (0.44 vs 0.20 within +-20 %)
    gas_only = run("oil_pump_hicp", {"area": {"country": ["IT"]}, "scenario": {"kind": "custom", "gas_delta_eur_mwh": 30}})
    assert "diesel_pump" not in gas_only["summary"]["IT"] and gas_only["summary"]["IT"]["hicp_elc_gas"]["cumulative_effect_pct"]["p50"] > 0
    for bad in ({"area": {"country": ["DE"]}, "scenario": {"kind": "custom"}}, {"area": {"country": ["US"]}, "scenario": {"kind": "custom", "product_change_pct": 5}},
                {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "product_change_pct": -99}}, {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "product_change_pct": 5, "months": 9}},
                {"area": {"country": ["DE"]}, "scenario": {"kind": "custom", "product_change_pct": [1] * 40}}):
        assert R.run("oil_pump_hicp", bad, {}).get("available") is False


# ------------------------------------------------------------------------------------------------ chokepoints -> freight
CHOKE = {"area": {"lat": 30.0, "lon": 32.55, "radius_km": 30}, "scenario": {"kind": "custom", "lane": "suez", "transit_cut_pct": 55, "weeks": 26}}


def test_choke_freight_lane_size_and_scope():
    out = run("choke_freight", CHOKE)
    wi = check_whatif(out, "choke_freight", "experimental")
    P = MK.params("choke_freight")
    d = 0.55
    ln_end = P["beta"] * P["phi"]["suez"] * d * (1 - 0.5 ** 26)                  # EWMA of a constant cut has converged
    s = out["series"]["freight_rate_multiplier"]
    assert abs(math.log(s["p50"][-1]) - ln_end) < 0.02 and s["p10"][-1] < s["p50"][-1] < s["p90"][-1]
    assert 2.5 < s["p50"][-1] < 4.0                                               # about x3: the Red Sea Shanghai-Rotterdam rise was x3.1-x3.3
    assert any("import prices" in x for x in wi["links_not_wired"]) and "x the pre-onset" in s["unit"]
    assert "Drewry" in wi["licence"] and any("10 weeks" in c for c in wi["caveats"])
    comp = run("choke_freight", dict(CHOKE, scenario=dict(CHOKE["scenario"], index="wci")))
    assert comp["series"]["freight_rate_multiplier"]["p50"][-1] < s["p50"][-1]    # the composite carries 0.35 of the lane
    pan = run("choke_freight", dict(CHOKE, scenario=dict(CHOKE["scenario"], lane="panama", index="wci_rtm")))
    assert all(abs(x - 1) < 1e-9 for x in pan["series"]["freight_rate_multiplier"]["p50"]) and any("no exposure" in w for w in pan["meta"]["warnings"])
    zero = run("choke_freight", dict(CHOKE, scenario=dict(CHOKE["scenario"], transit_cut_pct=0)))
    assert all(abs(x - 1) < 1e-9 for x in zero["series"]["freight_rate_multiplier"]["p90"])
    for bad in (dict(CHOKE, scenario={"kind": "custom", "lane": "bosphorus", "transit_cut_pct": 50}), dict(CHOKE, scenario={"kind": "custom", "lane": "suez"}),
                dict(CHOKE, scenario={"kind": "custom", "lane": "suez", "transit_cut_pct": 120}), dict(CHOKE, scenario={"kind": "custom", "lane": "suez", "transit_cut_pct": 10, "weeks": 99})):
        assert R.run("choke_freight", bad, {}).get("available") is False


# ------------------------------------------------------------------------------------------------ failed links stay, marked
FAILED = ("ports_strike", "rates_real", "epidemic_activity", "price_industry", "freight_prices")


















# ------------------------------------------------------------------------------------------------ optional inputs (23 Oct 2026)
def test_oil_product_and_crack_paths_are_optional_and_defaults_unchanged():
    base = {"area": {"country": ["DE"]}}
    sc = {"kind": "custom", "brent_usd_bbl": [135, 140, 150, 150, 150, 150], "brent_now_usd_bbl": 130}
    plain = run("oil_pump_hicp", base | {"scenario": sc})["summary"]["DE"]
    assert run("oil_pump_hicp", base | {"scenario": sc | {"diesel_crack_change_usd_bbl": None}})["summary"]["DE"] == plain      # an absent / null key changes nothing
    dsl = run("oil_pump_hicp", base | {"scenario": sc | {"diesel_crack_change_usd_bbl": [0, 5, 10, 20, 20, 20]}})
    s = dsl["summary"]["DE"]
    assert s["diesel_pump"]["effect_vs_prices_flat_pct"] > plain["diesel_pump"]["effect_vs_prices_flat_pct"] + 2           # diesel margin up -> diesel pump up
    assert s["gasoline_pump"] == plain["gasoline_pump"]                                                                     # petrol untouched
    assert any("Crack-spread" in w for w in dsl["meta"]["warnings"]) and "diesel_crack_change_usd_bbl" in dsl["meta"]["whatif"]["inputs_used"]["wholesale_product_paths"]
    both = run("oil_pump_hicp", base | {"scenario": {"kind": "custom", "crack_change_usd_bbl": 20}})["summary"]["DE"]         # crack alone: Brent unchanged
    assert both["diesel_pump"]["effect_vs_prices_flat_pct"] > 1 and both["gasoline_pump"]["effect_vs_prices_flat_pct"] > 1
    own = run("oil_pump_hicp", base | {"scenario": {"kind": "custom", "diesel_change_pct": [0, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "gasoline_change_pct": 0}})["summary"]["DE"]
    assert own["diesel_pump"]["effect_vs_prices_flat_pct"] > 4 and abs(own["gasoline_pump"]["effect_vs_prices_flat_pct"]) < 1e-6
    for bad in ({"crack_change_usd_bbl": 1000}, {"diesel_change_pct": -99}, {"gasoline_crack_change_usd_bbl": [1] * 40}):
        assert R.run("oil_pump_hicp", base | {"scenario": {"kind": "custom"} | bad}, {}).get("available") is False


def test_heat_load_wind_solar_are_context_only():
    base = {"area": {"country": ["ES"]}, "scenario": dict(HEAT["scenario"])}
    ref = run("heat_load", base)
    sc = base["scenario"] | {"baseline_mw": 30000, "wind_mw": 3000, "solar_mw": [5000] * len(HEAT["scenario"]["tmean_c"])}
    out = run("heat_load", {"area": base["area"], "scenario": sc})
    # the demand answer is exactly what it was without them (the mechanism does not use generation); the MW scale comes from the baseline now
    assert out["series"]["ES_excess_pct"] == ref["series"]["ES_excess_pct"]
    g = out["summary"]["ES"]["generation_context"]
    assert g["status"] == "context" and g["wind_mean_mw"] == 3000 and g["residual_load_mean_mw"] < 30000 + 5000
    assert "ES_residual_load_context_mw" in out["series"] and any("context only" in w for w in out["meta"]["warnings"])
    nobase = run("heat_load", {"area": base["area"], "scenario": base["scenario"] | {"wind_mw": 3000}})
    assert "residual_load_mean_mw" not in nobase["summary"]["ES"]["generation_context"]
    assert R.run("heat_load", {"area": base["area"], "scenario": base["scenario"] | {"wind_mw": -5}}, {}).get("available") is False


# ------------------------------------------------------------------------------------------------ calibrated ranges and exceedance
def test_calibrated_band_and_exceedance_helpers():
    P = MK.params("oil_pump_hicp")
    m = MK.cal_mode(P, "oil_pump", "chain")
    assert m and m["meta"]["dev"]["coverage_after"] > m["meta"]["dev"]["coverage_before"]            # the calibration is what the result quotes
    c = np.array([0.0, 0.05, 0.2, -0.2])
    b, s = MK.cal_band(c, m, "DE")
    assert np.all(b["p10"] <= b["p50"]) and np.all(b["p50"] <= b["p90"]) and np.all(s > 0)
    assert (b["p90"] - b["p10"])[2] > (b["p90"] - b["p10"])[0]                                      # wider for a larger shock
    assert (b["p90"] - b["p10"])[0] > 0                                                              # a zero effect still has a band (the multiplicative band had none)
    p = MK.exceed_prob(c, s, m, "DE")
    assert np.all((p >= 0) & (p <= 1)) and p[2] > p[0] and p[3] > p[0]                              # a bigger effect, either sign, is more likely to leave the normal band
    far = MK.cal_band(np.zeros(8), m, "DE")[1]                                                        # beyond the tested steps the spread grows (sqrt rule), never shrinks
    assert far[-1] >= far[3] - 1e-12
    assert MK.cal_mode(P, "nope") is None and MK.cal_mode({}, "oil_pump") is None


def test_results_carry_calibrated_ranges_and_exceedance():
    base = {"area": {"country": ["DE"]}}
    o = run("oil_pump_hicp", base | {"scenario": {"kind": "custom", "brent_usd_bbl": [135, 140, 150, 150, 150, 150], "brent_now_usd_bbl": 130}})
    wi = o["meta"]["whatif"]
    assert "CALIBRATED" in wi["ranges"] and set(wi["calibration"]) == {"pump", "hicp_fuel"}
    assert wi["calibration"]["pump"]["dev"]["coverage_after"] > 0.7 and wi["calibration"]["pump"]["check"]["n_events"] >= 1
    pe = o["series"]["DE_diesel_pump_p_leaves_normal_band"]
    assert all(0 <= x <= 1 for x in pe["p50"]) and pe["p10"] == pe["p50"] == pe["p90"]
    assert 0 <= o["summary"]["DE"]["diesel_pump"]["prob_leaves_normal_band_at_week_n"] <= 1
    assert 0 <= o["summary"]["DE"]["hicp_fuel"]["prob_leaves_normal_band_at_end"] <= 1
    # a bigger oil move: wider absolute band and a higher chance of leaving the normal band
    big = run("oil_pump_hicp", base | {"scenario": {"kind": "custom", "brent_usd_bbl": [160, 190, 220, 220, 220, 220], "brent_now_usd_bbl": 130}})
    a, b = o["summary"]["DE"]["diesel_pump"], big["summary"]["DE"]["diesel_pump"]
    assert b["prob_leaves_normal_band_at_week_n"] > a["prob_leaves_normal_band_at_week_n"]
    assert (b["change_at_week_n_eur_l"]["p90"] - b["change_at_week_n_eur_l"]["p10"]) > (a["change_at_week_n_eur_l"]["p90"] - a["change_at_week_n_eur_l"]["p10"])
    # rates and heat: calibrated bands, non-zero width where the old multiplicative band collapsed, exceedance series present
    cut = _rates([2.25, 2.0, 2.0, 1.75, 1.75, 1.5])
    assert "CALIBRATED" in cut["meta"]["whatif"]["ranges"] and "DE_firms_rate_p_leaves_normal_band" in cut["series"]
    f = cut["summary"]["DE"]["firms"]
    assert f["change_at_end_pp"]["p10"] < f["change_at_end_pp"]["p50"] < f["change_at_end_pp"]["p90"] and 0 <= f["prob_leaves_normal_band_at_end"] <= 1
    h = run("heat_load", HEAT)
    assert "CALIBRATED" in h["meta"]["whatif"]["ranges"] and "ES_p_load_leaves_normal_band" in h["series"]
    zero = run("heat_load", {"area": {"country": ["ES"]}, "scenario": {"kind": "custom", "start": "2026-07-14", "tmean_c": [15, 15, 15]}})["series"]["ES_excess_pct"]
    assert max(zero["p90"]) - min(zero["p10"]) > 0                                                   # no heat, still an honest band


# ------------------------------------------------------------------------------------------------ registry switch
def test_whatif_off_switch(monkeypatch):
    monkeypatch.setenv("FE_WHATIF_OFF", "heat_load;oil_pump_hicp strike_air")
    assert R.whatif_off() == {"heat_load", "oil_prices", "strike_air"}
    monkeypatch.setenv("FE_WHATIF_OFF", "none")
    assert R.whatif_off() == set()
    monkeypatch.delenv("FE_WHATIF_OFF")
    assert R.whatif_off() == set()


def test_all_five_are_listed_with_examples():
    ids = {s["id"] for s in R.catalog()}
    assert {"heat_load", "strike_air", "rates_credit", "oil_pump_hicp", "choke_freight"} <= ids
    assert not R.LOAD_ERRORS
    assert R.run("nope", {})["available"] is False
