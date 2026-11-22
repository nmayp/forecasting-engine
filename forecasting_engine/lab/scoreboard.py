"""The benchmark scoreboard: every family's champion against the other engines on the same locked holdout, from the ledger and
results/lab/bench.json (docs/methods.md is written from it).

    python -m forecasting_engine.lab.scoreboard            # prints the table and rewrites docs/methods.md

Skill is pooled skill against the strict reference on the holdout (1 = perfect, 0 = as good as the reference, negative = worse). Each
foundation model counts its best fair variant (own history; every layer as covariates; with cross-country context; causal parents only).
Rows marked "_prefix" in bench.json were run before the cross-learning batch fix and are left out. Since 19 Oct 2026 a layer is
won only if the engine beats the BEST of Chronos-2, TiRex-2 and Toto 2.0.
"""
from __future__ import annotations

import json
import sys

from forecasting_engine.lab import core as C

FMS = (("Chronos-2", "chronos-2"), ("TiRex-2", "tirex-2"), ("Toto 2.0", "toto-2"))     # (name, bench.json tag prefix): the bar
EXTRA = (("TimesFM 2.5", "timesfm-2.5"), ("FlowState", "flowstate"))                    # shown for the stack's diversity, not the bar
LAYER = {"price_h": "energy", "load_h": "energy", "res_h": "energy", "load_d": "energy", "diesel_w": "fuel", "gasoline_w": "fuel",
         "ip_m": "economy", "retail_m": "economy", "unemp_m": "economy", "hicp_m": "economy", "fx_d": "markets", "commod_d": "markets",
         "unrest_w": "society", "kev_d": "cyber", "cpi_long_m": "economy", "ip_long_m": "economy", "ur_long_m": "economy",
         "ports_m": "transport", "health_w": "health", "air_d": "transport", "security_w": "security", "info_w": "information",
         "tourism_m": "tourism", "water_w": "water", "wildfire_w": "environment", "aq_d": "environment", "hicp_ea_m": "economy",
         "ecb_w": "policy", "choke_w": "transport", "wiki_w": "attention", "jobs_w": "labour", "asylum_m": "migration",
         "milk_m": "agriculture", "space_d": "space", "carbon_w": "emissions", "regulation_w": "regulation",
         "credit_m": "credit", "freight_d": "freight", "constr_m": "construction", "firms_m": "firms", "housing_m": "housing", "sent_m": "sentiment",
         "services_m": "services", "insolv_m": "insolvency", "research_m": "research", "water_m": "water demand", "transit_d": "transit", "c311_d": "city services",
         "spend_w": "consumer spending",
         "rasff_w": "food safety", "ted_w": "public procurement", "hmrc_m": "tax receipts", "traffic_d": "road traffic",
         "mort_w": "mortality", "bike_d": "cycling", "reg_d": "vehicle registrations", "ev_m": "vehicle electrification", "agri_w": "agri-food prices",
         "agricom_m": "agricultural commodities", "quake_w": "earthquakes", "arxiv_m": "academic preprints",
         "births_m": "births", "fire911_w": "emergency calls", "sec_w": "corporate disclosure", "library_m": "library loans", "launch_w": "space launches"}


LAGFIX_TAG = {"chronos2": "chronos-2_univariate", "chronos2x": "chronos-2_covariates_cross", "toto2": "toto-2_univariate", "toto2x": "toto-2_cross", "tirex2": "tirex-2_univariate",
              "tirex2x": "tirex-2_cross", "tirex2c": "tirex-2_covariates", "timesfm25": "timesfm-2.5_univariate", "flowstate": "flowstate_univariate"}


FIX_ROUNDS = ("lagfix", "leakfix")


def _lagfix_fms(led: list[dict], fid: str) -> dict:
    """Open-model rows re-scored in round "lagfix" (lag audit, 20 Oct 2026: the family's own target delay changed, so the bench.json
    scores of these models were made on values not yet published) or "leakfix" (21 Oct 2026: unemp_m / ur_long_m target delay; the GDELT
    targets of unrest_w, security_w, info_w rebuilt keyed on DATEADDED): method -> latest row."""
    out = {}
    for r in led:
        if r.get("family") == fid and r.get("round") in FIX_ROUNDS and r.get("rescore") and r.get("method") in LAGFIX_TAG and r.get("holdout"):
            out[r["method"]] = r
    return out


def rows() -> list[dict]:
    led = C.ledger()
    b = json.loads((C.LAB / "bench.json").read_text()) if (C.LAB / "bench.json").exists() else {}
    out = []
    for f in [x["id"] for x in C.registry()["families"]]:
        cr = C.champion_row(f)                          # with its re-scored (lag-corrected) scores when a rescore row exists
        ch = cr["method"] if cr else None
        k = (cr.get("holdout") or {}).get("pooled_skill") if cr else None
        bb = b.get(f, {})
        fix = _lagfix_fms(led, f)                       # open models re-scored after the publication-lag audit replace their bench.json rows
        if fix:
            bb = {t: v for t, v in bb.items() if not any(t.startswith(p) for p in {LAGFIX_TAG[m].split("_")[0] for m in fix})} | {
                LAGFIX_TAG[m]: dict(holdout=x["holdout"]) for m, x in fix.items()}
        fms = {}
        for name, pre in FMS + EXTRA:
            sc = {t: v["holdout"].get("pooled_skill") for t, v in bb.items()
                  if t.startswith(pre) and not t.endswith("_prefix") and isinstance(v, dict) and v.get("holdout")}
            sc = {t: v for t, v in sc.items() if v is not None}
            fms[name] = max(sc.items(), key=lambda x: x[1]) if sc else (None, None)
        scored = {n: v for n, v in fms.items() if v[1] is not None and n in dict(FMS)}
        ex = {n: fms[n] for n, _ in EXTRA if fms[n][1] is not None}
        xname = max(ex, key=lambda n: ex[n][1]) if ex else None
        bname = max(scored, key=lambda n: scored[n][1]) if scored else None
        best = scored[bname][1] if bname else None
        bolt = next((v["holdout"].get("pooled_skill") for t, v in bb.items() if "bolt" in t and isinstance(v, dict) and v.get("holdout")), None)
        c2 = fms["Chronos-2"]
        out.append(dict(family=f, layer=LAYER.get(f, ""), champion=ch, engine=k, looks=C.holdout_looks(led, f), chronos2=c2[1], chronos2_variant=c2[0], bolt=bolt,
                        fms={n: dict(skill=v[1], variant=v[0]) for n, v in fms.items()}, best_fm=bname, best=best,
                        other=xname, other_skill=ex[xname][1] if xname else None,
                        margin_c2=None if k is None or c2[1] is None else round(k - c2[1], 4),
                        margin=None if k is None or best is None else round(k - best, 4)))
    return out


def _p(x) -> str:
    return "" if x is None else f"{x * 100:.1f} %"


def markdown(rs: list[dict]) -> str:
    for r in rs:                                        # provisional until Chronos-2, TiRex-2 and Toto 2.0 all have a score
        r["provisional"] = any(r["fms"][n]["skill"] is None for n, _ in FMS)
    prov = [r for r in rs if r["provisional"] and r["margin"] is not None]
    rs_all, rs = rs, [r for r in rs if not r["provisional"]]
    scored = [r for r in rs if r["margin"] is not None]
    win = sum(r["margin"] > 0 for r in scored)
    full = scored
    c2 = [r for r in rs if r["margin_c2"] is not None]
    lines = ["# Benchmarks: the engine against open foundation models", "",
             "Same locked holdout, same strict reference, same scoring for every engine (lab v2). Skill = pooled skill against the reference "
             "(0 = as good as the reference). Each foundation model is its best fair variant: own history, every layer as covariates as known "
             "at the origin (where the model takes covariates: Chronos-2 and TiRex-2; Toto 2.0 has no covariate input yet), cross-country "
             "context, or the causal parents only. A layer counts as won only if the engine beats the best of the three. "
             "Most champions are ensembles that include open models (Toto 2.0, FlowState, Chronos-2 or TiRex-2 forecasts as members, next to "
             "the engine's own trees); a blend that contains the best open model counts as a win only if it beats that model alone on the holdout. "
             "Generated by `python -m forecasting_engine.lab.scoreboard`.", "",
             f"Engine ahead of the best of Chronos-2, TiRex-2 and Toto 2.0 on {win} of {len(scored)} layers with scores "
             f"(layers with all three models scored; {len(prov)} provisional layers are excluded from this count: {', '.join(r['family'] for r in prov) or 'none'}); ahead of Chronos-2 alone on {sum(r['margin_c2'] > 0 for r in c2)} of {len(c2)}.", "",
             "The last column is the better of two more open models (TimesFM 2.5, own history or + known-future covariates; FlowState, own "
             "history), kept for the stack's diversity; it is not part of the bar.", "",
             "Looks = holdout looks the family has used so far (each one a selection on the holdout; since 20 Oct 2026 a candidate gets one "
             "only after beating the champion on development and on most of its time blocks). Many looks make a small holdout lead weaker.", "",
             "| Family | Layer | Engine | Looks | Chronos-2 | TiRex-2 | Toto 2.0 | Best model | Margin vs best | TimesFM 2.5 / FlowState |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs_all:
        cells = []
        for n, pre in FMS:
            x = r["fms"][n]
            v = f" ({x['variant'].replace(pre + '_', '')})" if x["variant"] else ""
            cells.append(f"{_p(x['skill'])}{v}")
        m = "" if r["margin"] is None else f"{r['margin'] * 100:+.1f} pts" + (" (provisional: not counted)" if r["provisional"] else "")
        o = f"{_p(r['other_skill'])} ({r['other']})" if r["other"] else ""
        lines.append(f"| {r['family']} | {r['layer']} | {_p(r['engine'])} | {r['looks']} | {' | '.join(cells)} | {r['best_fm'] or ''} | {m} | {o} |")
    lines += ["", "Chronos-Bolt, the grid operators (next-day load and wind/solar) and LEAR (next-day price, scored on the next-day cells "
              "only) are in `results/lab/bench.json` (`chronos-bolt-small_zero_shot`, `operators_*`, `lear*`, `engine_champion next day only`)."]
    return "\n".join(lines) + "\n"


DOC = C.ROOT / "results" / "scoreboard.md"


def main(argv=None) -> int:
    rs = rows()
    md = markdown(rs)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
