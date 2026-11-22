"""`choke_freight`: a transit cut at a shipping chokepoint (Suez / Bab el-Mandeb, Panama) -> the container freight rate on the Asia-Europe lane. Link 1 ONLY of
docs/methods.md (forecasting_engine/lab/choke_mech.py).

    d_t = share of container transits lost against the pre-onset mean (0-1);  K_t = e * phi * EWMA(d_t), EWMA a = 0.5 (one-week half-life)
    ln(R_t / R_0) = beta * K_t        beta = 10 (set from the 2021 supply crunch, NOT fitted), phi = 0.21 Suez (longer route absorbs capacity) / 0.15 Panama (assumed),
                                      e = share of the freight index that sails the lane: 1 Shanghai-Rotterdam, 0.35 WCI composite, 0.25 SCFI composite (assumed)

The answer is the freight-rate MULTIPLIER against the pre-onset level (x): no Drewry or SCFI level is stored or shown (their terms forbid redistribution). It says "which direction and what
order of magnitude", not a rate forecast. Links 2 and 3 (freight -> import prices -> goods HICP) FAILED their check and are not here: the chain stops at freight.

Request: area: any point (e.g. the chokepoint: {"lat": 30.0, "lon": 32.55, "radius_km": 30}); the lane is chosen by the scenario;
  scenario {"kind": "custom", "lane": "suez" | "panama", "transit_cut_pct": number | [weekly % of container transits lost vs the pre-onset mean], "weeks": n (default 26, max 52),
            "index": "wci_rtm" (Shanghai-Rotterdam, default) | "wci" | "scfi"}
"""
from __future__ import annotations

import numpy as np

from forecasting_engine.whatif import kit as K
from forecasting_engine.whatif import mech_kit as MK
from forecasting_engine.whatif.registry import Simulator, register

ID = "choke_freight"
VERSION = "1.0 (link 1, 22 Oct 2026)"
MAX_WEEKS = 52
PARAMS = {}
SOURCES = ["Chokepoint mechanism (docs/methods.md), link 1: transits -> freight; back-tested on IMF PortWatch transits and Drewry WCI (back-test only, no level shown)"]


def ewma(x: np.ndarray, a: float) -> np.ndarray:
    out, z = np.zeros(len(x)), 0.0
    for i, v in enumerate(x):
        z = a * z + (1 - a) * v
        out[i] = z
    return out


def run(req: dict, ctx: dict) -> dict:
    tm = K.Timer()
    P = MK.params("choke_freight")
    sc = dict(req.get("scenario") or {})
    if str(sc.get("kind") or "custom") != "custom":
        raise ValueError("scenario.kind must be custom")
    lane = str(sc.get("lane") or "suez").lower()
    if lane not in P["phi"]:
        raise ValueError("scenario.lane must be suez or panama (Bosphorus has no container lane in the freight indices)")
    idx = str(sc.get("index") or "wci_rtm").lower()
    if idx not in P["exposure"]:
        raise ValueError("scenario.index must be wci_rtm, wci or scfi")
    cut = sc.get("transit_cut_pct")
    if cut is None:
        raise ValueError("scenario.transit_cut_pct is required: the % of container transits lost against the pre-onset mean, per week")
    n = int(sc.get("weeks") or (len(cut) if isinstance(cut, (list, tuple)) else 26))
    if not 1 <= n <= MAX_WEEKS:
        raise ValueError(f"scenario.weeks must be 1-{MAX_WEEKS}")
    d = MK.num_list(cut, "transit_cut_pct", n, 0, 100, max_n=MAX_WEEKS) / 100.0
    e = P["exposure"][idx][lane]
    warnings = []
    if e == 0:
        warnings.append(f"the {idx} index has no exposure to the {lane} lane in this mechanism: the rate does not move")
    K_t = e * P["phi"][lane] * ewma(d, P["ewma_a"])
    ln_c = P["beta"] * K_t
    f = P["size_factor"]
    ln = {"p10": ln_c * f["p10"], "p50": ln_c, "p90": ln_c * f["p90"]}
    mult = {k: np.expm1(v) * 100 for k, v in ln.items()}                      # % above the pre-onset level
    t_h = np.arange(n) * 168.0
    series = {"freight_rate_change_pct": MK.ser("% above the pre-onset freight rate (index " + idx + ")", t_h, mult),
              "freight_rate_multiplier": MK.ser("x the pre-onset freight rate", t_h, {k: np.exp(v) for k, v in ln.items()}),
              "effective_capacity_lost": MK.ser("share of the lane's effective capacity absorbed", t_h, {"p10": K_t, "p50": K_t, "p90": K_t})}
    i = int(np.argmax(ln_c))
    mean_ln = float(np.mean(ln_c))
    summary = dict(lane=lane, index=idx, weeks=n, mean_transit_cut_pct=MK.rnd(100 * float(np.mean(d)), 1),
                   peak_freight_multiplier=MK.qd3(np.exp(ln["p10"][i]), np.exp(ln["p50"][i]), np.exp(ln["p90"][i])), peak_week=int(i + 1),
                   window_mean_ln_rise=MK.qd3(float(np.mean(ln["p10"])), mean_ln, float(np.mean(ln["p90"]))),
                   end_freight_multiplier=MK.qd3(np.exp(ln["p10"][-1]), np.exp(ln["p50"][-1]), np.exp(ln["p90"][-1])),
                   beta=P["beta"], phi=P["phi"][lane], exposure_share=e,
                   next_links="freight -> import prices -> goods HICP is NOT offered: the links failed their check (import prices did not follow freight in 2024)")
    moments = [MK.moment(i * 168.0, f"Freight peaks at x{np.exp(ln_c[i]):.2f} the pre-onset rate in week {i + 1} (p10-p90 x{np.exp(ln['p10'][i]):.2f} to x{np.exp(ln['p90'][i]):.2f}); the back-test timing of the largest move is off by about 10 weeks", min(1.0, float(ln_c[i]) / 1.5))]
    bt = P["backtest"]
    l26, l52, c26, c52, mae = bt["lane_26w"], bt["lane_52w"], bt["composite_26w"], bt["composite_52w"], bt["mae"]
    line = (f"One check event (Red Sea from Dec 2023, Suez container transits -54 %, four overlapping cells, looked at once; beta and phi were not blind): Shanghai-Rotterdam rise {l26['realised_ln']:.2f} realised vs {l26['predicted_ln']:.2f} predicted "
            f"(size {l26['size']:.2f}; 52 weeks {l52['size']:.2f}), error {mae['mechanism']:.2f} vs {mae['no_change']:.2f} for no change and {mae['chronos2']:.2f} / {mae['tirex2']:.2f} for Chronos-2 / TiRex-2, direction right {bt['direction']}; "
            f"composite index under-predicted (size {c26['size']:.2f} / {c52['size']:.2f}); timing FAILS (median {bt['timing_weeks']} weeks); size over beta 7-15 x phi 0.15-0.27 runs {f['min']:.2f}-{f['max']:.2f} of the central")
    wi = MK.whatif_block(
        mechanism="Chokepoint container transit cut -> Asia-Europe container freight rate (link 1 of 3)", status="experimental",
        status_note="the Shanghai-Rotterdam lane passed skill, direction and size on its one check event, but the pre-declared rules fail on composite size and on timing, beta and phi were set by someone who knew the outcome, and the development events cannot test them",
        as_of=P["as_of"], tested_scope="Suez / Bab el-Mandeb container transits -> Shanghai-Rotterdam and composite container freight, weekly, 26 and 52 weeks from the onset, one event (Red Sea 2023-24)",
        backtest=line,
        ranges=(f"p10-p90 = the central ln rise x {f['p10']:.2f} / x {f['p90']:.2f}: the spread of the predicted size over beta 7-15 and phi 0.15-0.27 ({f['n']} combinations, the Red Sea back-test); it is a sensitivity range, not a calibrated error band, "
                "and the timing error is not in it"),
        caveats=[
            "Direction and order of magnitude only: beta (the rate rise per unit of capacity absorbed) and phi were written down by someone who knew the Red Sea outcome and could not be tested on 2021-23 (no development event has an exposed container lane with a transit cut).",
            "Size is not robust: over beta 7-15 and phi 0.15-0.27 the Shanghai-Rotterdam size ratio runs 0.48-1.84 and only 4 of 9 combinations sit in 0.8-1.25.",
            "Timing fails: the median offset of the largest move is about 10 weeks. Freight also moves with demand and congestion (2021 container crunch, the June-July 2024 peak season), which a transit-only model cannot know.",
            "Composite indices are under-predicted (size 0.44-0.50): only 35 % of the WCI composite is assumed to sail the Suez lane while in 2024 all lanes rose as diverted tonnage absorbed fleet capacity.",
            "One check event with overlapping windows: every pooled number is a single event. The Panama and Bosphorus what-ifs have no check event.",
            "Freight levels are shown only as a multiplier against the pre-onset rate: Drewry and SCFI terms forbid redistribution of their values."],
        licence="Back-tests used IMF PortWatch transits (back-test only; ask the IMF before a paid product) and Drewry World Container Index prints (terms forbid redistribution): no level of either is stored or shown.",
        not_tested=["Delivery times and delays", "Bulk and tanker freight (no free index)", "Lane-level rates other than Shanghai-Rotterdam", "Transit cuts longer than 52 weeks"],
        links_not_wired=["Freight -> EU import prices (failed: eps about 0 in 14 years of data; import prices did not move in 2024)", "Import prices -> goods HICP (failed / not detectable)"])
    rule = ("Pre-declared rules per link: skill above no change and above the open models; direction >= 0.80 of moved cells; median size 0.8-1.25; median timing <= 4 weeks. Link 1: skill, direction pass; "
            "size fails on the composite (0.73 median over 4 cells), timing fails (10.5 weeks). Link 2 and 3 failed.")
    cases = [{"name": "Red Sea, Shanghai-Rotterdam 26 weeks", "metric": "predicted / realised ln rise", "observed": 1.0, "simulated": l26["size"], "error": round(l26["size"] - 1, 2)},
             {"name": "Red Sea, Shanghai-Rotterdam 52 weeks", "metric": "predicted / realised ln rise", "observed": 1.0, "simulated": l52["size"], "error": round(l52["size"] - 1, 2)},
             {"name": "Red Sea, composite 26 weeks", "metric": "predicted / realised ln rise", "observed": 1.0, "simulated": c26["size"], "error": round(c26["size"] - 1, 2)},
             {"name": "Red Sea, composite 52 weeks", "metric": "predicted / realised ln rise", "observed": 1.0, "simulated": c52["size"], "error": round(c52["size"] - 1, 2)}]
    return MK.result(ID, VERSION, tm, wi, series, summary, MK.timeline(t_h, {"freight_rate_change_pct": mult["p50"], "effective_capacity_lost": K_t}), moments, MK.validation(rule, line, cases),
                     assumptions=[f"Transit cut: mean {100 * float(np.mean(d)):.0f} % of container transits over {n} weeks, smoothed with a one-week half-life.",
                                  f"beta {P['beta']:g}, phi {P['phi'][lane]:g} ({lane}), index share on the lane {e:g} (assumed for composites)."], warnings=warnings, sources=SOURCES)


register(Simulator(id=ID, name="Chokepoint closure to freight rates", domain="supply", hazards=[], run=run, scenarios=["custom"], params=PARAMS, status="experimental",
                   description="A container-transit cut at Suez or Panama -> the Asia-Europe container freight rate as a multiplier of the pre-onset level, with a sensitivity range (experimental, one check event, direction and order of magnitude only)",
                   max_horizon_h=24 * 7 * MAX_WEEKS))
