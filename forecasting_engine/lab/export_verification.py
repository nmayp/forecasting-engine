"""Writes the files in verification/ from the shipped ledger and registry, so the champion of every layer, the third-party models and the
data each layer was scored on can be checked without the research tree.

    python -m forecasting_engine.lab.export_verification
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from forecasting_engine.lab import core as C

OUT = Path(__file__).resolve().parents[2] / "verification"
PANEL = Path(__file__).resolve().parent / "panel.py"

# How a method string is read (forecasting_engine/lab/core.py, function `method`). Weights are rules, not stored constants.
GRAMMAR = {
    "blend:a|b": "the mean of the members' forecasts",
    "median:a|b": "the per-point median of the members' forecasts",
    "trimN:a|b": "the mean after dropping the N members with the worst recent errors (default 1)",
    "adapt:a|b": "a mean weighted by each member's recent errors, using only forecasts whose outcomes were already known (latest 12)",
    "rgd:m / rg:m": "the method with regime features (rgd: every driver's regime, rg: own and cross-series)",
    "anchor-x:m": "the method learns corrections to the forecast of x (a foundation model or driver)",
    "recon:m": "reconciled with the hourly sibling's chained forecast",
    "xcovid:m": "the method with the 2020-21 period handled separately",
    "gbm_*": "gradient-boosted trees; the name lists the driver layers used (gbm_only-energy+weather) and the size (_big, _xl)",
    "chronos2 / chronos2x / toto2 / toto2x / tirex2 / tirex2x / tirex2c / flowstate / timesfm25": "open foundation models as members; x = with cross-series context, c = with covariates",
    "ref / naive / seasonal": "the strict reference: the better of 'no change' and 'same as last season'",
}

# Where each input of a layer comes from (panel.BUILD keys). Licence notes: see forecasting_engine/lab/sources.py and docs/methods.md.
SOURCES = {
    "load": "ENTSO-E Transparency Platform, actual load", "load_da": "ENTSO-E Transparency Platform, day-ahead load forecast",
    "price": "ENTSO-E Transparency Platform, day-ahead price", "gen": "ENTSO-E Transparency Platform, generation by type",
    "net_imp": "ENTSO-E Transparency Platform, cross-border flows", "outage": "ENTSO-E Transparency Platform, generation unavailability",
    "temp": "population-weighted daily temperature table (ERA5-type; not shipped)", "hdd": "population-weighted daily temperature table (ERA5-type; not shipped)",
    "cdd": "population-weighted daily temperature table (ERA5-type; not shipped)",
    "fx": "ECB Data Portal, euro reference rates", "ecb_rates": "ECB Data Portal, policy and money-market rates",
    "fred": "FRED (St. Louis Fed): Brent, WTI, Henry Hub, Treasury yields, dollar index, VIX, IMF commodity prices, policy uncertainty",
    "strike": "GDELT 2.0 events (BigQuery export), strikes", "protest": "GDELT 2.0 events (BigQuery export), protests",
    "violence": "GDELT 2.0 events (BigQuery export), violence", "coerce": "GDELT 2.0 events (BigQuery export), coercion",
    "tone": "GDELT 2.0 events (BigQuery export), tone", "events": "GDELT 2.0 events (BigQuery export), event counts",
    "kev": "CISA Known Exploited Vulnerabilities catalogue", "ip": "Eurostat, industrial production", "retail": "Eurostat, retail trade volume",
    "unemp": "Eurostat, unemployment rate", "hicp": "Eurostat, HICP inflation", "esi": "Eurostat, economic sentiment indicator",
    "consumer_conf": "Eurostat, consumer confidence", "industry_conf": "Eurostat, industry confidence", "ppi": "Eurostat, producer prices",
    "stringency": "Oxford COVID-19 Government Response Tracker", "hicp_index": "Eurostat, HICP index", "hicp_energy_index": "Eurostat, HICP energy index",
    "diesel": "EU Weekly Oil Bulletin, diesel pump price", "gasoline": "EU Weekly Oil Bulletin, petrol (95) pump price",
}
RESTRICTED = {"load", "load_da", "price", "gen", "net_imp", "outage"}   # free token, but the data may not be redistributed

MODELS = [
    dict(name="Chronos-2", maker="Amazon", hf="amazon/chronos-2", package="chronos-forecasting", package_version=None, revision=None, licence="Apache-2.0",
         note="package version and weights revision were not recorded in the code"),
    dict(name="TiRex-2", maker="NX-AI", hf="NX-AI/TiRex-2", package="tirex-2", package_version="0.3.1", revision=None, licence="Apache-2.0"),
    dict(name="Toto 2.0", maker="Datadog", hf="Datadog/Toto-2.0-1B", package="toto-2", package_version="2.0.0", revision=None, licence="Apache-2.0"),
    dict(name="TimesFM 2.5", maker="Google", hf="google/timesfm-2.5-200m-pytorch", package="timesfm", package_version="2.0.2", revision=None, licence="Apache-2.0",
         note="the 2.5 code; the 3.0 weights are non-commercial and were never used"),
    dict(name="FlowState", maker="IBM", hf="ibm-granite/granite-timeseries-flowstate-r1", package="granite-tsfm", package_version="0.3.10 (installed --no-deps)",
         revision="r1.1", licence="Apache-2.0"),
]


# Families whose target is read through a variable key in panel.build_family (P.get(k)), so the key is not visible to a text scan.
TARGET_KEY = {"ip_m": "ip", "retail_m": "retail", "unemp_m": "unemp", "hicp_m": "hicp"}


def builder_lines() -> dict[str, tuple[int, list[str]]]:
    """Family id -> (line of its branch in panel.build_family, the panel.BUILD keys that branch reads)."""
    src = PANEL.read_text().split("\n")
    start = next(i for i, l in enumerate(src) if l.startswith("def build_family"))
    end = next(i for i, l in enumerate(src) if l.startswith("def extra_drivers"))
    out: dict[str, tuple[int, list]] = {}
    cur: list[str] = []
    for i in range(start, end):
        m = re.match(r"    (?:el)?if fid (?:==|in) (.*?):\s*$", src[i])
        if m:
            cur = re.findall(r'"([a-z0-9_]+)"', m.group(1))
            for c in cur:
                out.setdefault(c, (i + 1, []))
            continue
        for c in cur:
            out[c][1].extend(re.findall(r'P\.get\("([a-z0-9_]+)"', src[i]))
    return {k: (v[0], sorted(set(v[1]))) for k, v in out.items()}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    reg = C.registry()
    led = C.ledger()
    b = builder_lines()
    champs, manifest = [], []
    for f in reg["families"]:
        fid = f["id"]
        cr = C.champion_row(fid)
        meth = cr["method"] if cr else None
        champs.append(dict(
            family=fid, label=f["label"], layer=f["layer"], champion=meth,
            members=[x for x in re.split(r"[|]", re.sub(r"^[a-z0-9]+:", "", meth or "")) if x] if meth else [],
            round=cr.get("round") if cr else None, utc=cr.get("utc") if cr else None,
            dev_skill=(cr.get("dev") or {}).get("pooled_skill") if cr else None,
            holdout_skill=(cr.get("holdout") or {}).get("pooled_skill") if cr else None,
            holdout_looks=C.holdout_looks(led, fid),
        ))
        line, keys = b.get(fid, (None, []))
        keys = sorted(set(keys) | ({TARGET_KEY[fid]} if fid in TARGET_KEY else set()))
        manifest.append(dict(
            family=fid, label=f["label"], layer=f["layer"], frequency=f["freq"], unit=f["unit"], horizon_steps=f["H"], series=f["series"],
            data_from=f["start"][:10], data_to=f["end"][:10], dev_origins=[f["dev_origins"][0][:10], f["dev_origins"][-1][:10]] if f["dev_origins"] else None,
            holdout_origins=[f["holdout_origins"][0][:10], f["holdout_origins"][-1][:10]] if f["holdout_origins"] else None,
            n_dev_origins=len(f["dev_origins"]), n_holdout_origins=len(f["holdout_origins"]), strict_reference=f["reference"],
            inputs_note="Shared inputs read by the builder; a layer may also read its own source modules (forecasting_engine/lab/src_*.py), see the builder.",
            inputs=[dict(key=k, source=SOURCES.get(k, "see the builder"), redistribution_restricted=k in RESTRICTED) for k in keys],
            builder=f"forecasting_engine/lab/panel.py, build_family, line {line}" if line else "see forecasting_engine/lab/panel.py",
            drivers=[dict(name=d["name"], group=d["group"], published_after_hours=d["avail_h"]) for d in f["drivers"]],
            sha256=f["sha256"],
        ))
    (OUT / "champions.json").write_text(json.dumps(dict(
        note="The champion of each layer on the locked holdout, as recorded in results/lab/trials.jsonl. Weights are rules (see grammar), not stored constants.",
        grammar=GRAMMAR, layers=champs), indent=1))
    (OUT / "models.json").write_text(json.dumps(dict(
        note="Third-party models used as members and as the bar. Their weights are not copied here; load them by id. A null means it was not recorded.",
        models=MODELS), indent=1))
    (OUT / "manifest.json").write_text(json.dumps(dict(
        note="What each layer was scored on: its series, the origin split, the strict reference, every driver with its publication delay, and the SHA-256 of "
             "each series file as frozen at the time (the target is Y, drivers are X_*). A re-download will differ where a source revises its history.",
        layers=manifest), indent=1))
    print(len(champs), "layers written to", OUT)


if __name__ == "__main__":
    main()
