"""Common interface of the what-if mechanisms. Each mechanism registers one `Simulator`; `run()` calls it and never raises.

    from forecasting_engine.whatif import registry as R
    R.catalog()                                   # the five mechanisms with status, parameters and an example request
    R.run("heat_load", R.EX.EXAMPLES["heat_load"]["request"])

A what-if is not a forecast: the caller gives the driver path (temperatures, a strike calendar, a policy-rate path, an oil path, a transit
cut) and gets the mechanism's response with p10 / p50 / p90 ranges.

Request keys:
    area      {"country": [ISO-2, ...]} (or a point for the chokepoint mechanism)
    scenario  {"kind": "custom" | ..., ...}
    horizon_h hours to simulate (the mechanism caps it)
    params    mechanism-specific options (documented in `params`)

Result keys (JSON-serialisable):
    meta       {sim, version, run_s, status, assumptions, warnings, sources, whatif: {mechanism, status, as_of, backtest, caveats, ranges, ...}}
    frames     time steps for animation or tables
    series     {name: {"unit", "t_h", "p10", "p50", "p90"}}
    summary    {metric: value | {p10, p50, p90}}
    validation {cases, rule, adopted, note}

Status words: validated = passed its pre-registered rule on held-out events; screening = plausible order of magnitude; experimental = failed part
of its rule or not yet tested on held-out events.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable

from forecasting_engine.whatif import examples as EX

STATUSES = ("validated", "screening", "experimental")


@dataclass
class SimResult(dict):
    """A plain dict with the documented keys; dataclass only for type hints."""


@dataclass
class Simulator:
    id: str
    name: str
    domain: str
    run: Callable[[dict, dict], dict]
    hazards: list = field(default_factory=list)
    scenarios: list = field(default_factory=lambda: ["forecast", "custom"])
    params: dict = field(default_factory=dict)      # name -> {"default", "unit", "doc"}
    status: str = "screening"
    description: str = ""
    max_horizon_h: int = 168

    def info(self) -> dict:
        out = {"id": self.id, "name": self.name, "domain": self.domain, "hazards": self.hazards, "scenarios": self.scenarios,
               "params": self.params, "status": self.status, "description": self.description, "max_horizon_h": self.max_horizon_h}
        ex = EX.EXAMPLES.get(self.id)
        if ex:
            out["example"] = {k: ex[k] for k in ("request", "title", "description")}
        if self.id in EX.SCENARIO_KEYS:
            out["scenario_keys"] = EX.SCENARIO_KEYS[self.id]
        return out


_REG: dict[str, Simulator] = {}


def register(sim: Simulator) -> Simulator:
    if sim.status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    _REG[sim.id] = sim
    return sim


def catalog() -> list[dict]:
    _load_all()
    return [s.info() for s in sorted(_REG.values(), key=lambda s: (s.domain, s.id))]


def get(sim_id: str) -> Simulator | None:
    _load_all()
    return _REG.get(sim_id)


def run(sim_id: str, req: dict, ctx: dict | None = None) -> dict:
    """Run one mechanism; never raises (returns {available: False, reason})."""
    s = get(sim_id)
    if s is None:
        return {"available": False, "reason": f"unknown simulator {sim_id}"}
    req = dict(req)
    req["horizon_h"] = min(int(req.get("horizon_h") or s.max_horizon_h), s.max_horizon_h)
    try:
        out = s.run(req, ctx or {})
    except Exception as e:  # fail soft: the caller sees the reason
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}
    out.setdefault("meta", {}).setdefault("sim", s.id)
    out["meta"].setdefault("status", s.status)
    if out.get("available", True) is not False:
        out.setdefault("moments", [])
        out.setdefault("validation", {"cases": [], "rule": "", "adopted": False,
                                      "note": f"no validation block in this result; status '{s.status}'"})
    return out


# The mechanisms register themselves on import.
MODULES = [
    "forecasting_engine.whatif.mech_heat_load", "forecasting_engine.whatif.mech_strike_air", "forecasting_engine.whatif.mech_rates",
    "forecasting_engine.whatif.mech_oil_prices", "forecasting_engine.whatif.mech_choke_freight",
]
_loaded = False
_MODULE_OF = {"heat_load": "heat_load", "strike_air": "strike_air", "rates_credit": "rates", "oil_pump_hicp": "oil_prices", "choke_freight": "choke_freight"}


def whatif_off() -> set[str]:
    """Module suffixes of the mechanisms switched off by FE_WHATIF_OFF (ids or suffixes, separated by commas, semicolons or spaces). Default: none."""
    raw = os.environ.get("FE_WHATIF_OFF", "")
    return {_MODULE_OF.get(x, x) for x in re.split(r"[,;\s]+", raw) if x and x != "none"}


def _load_all():
    global _loaded
    if _loaded:
        return
    import importlib
    off = whatif_off()
    for m in MODULES:
        if m.rsplit("mech_", 1)[1] in off:
            continue
        try:
            importlib.import_module(m)
        except Exception as e:  # one broken mechanism must not hide the others
            LOAD_ERRORS[m] = f"{type(e).__name__}: {e}"
    _loaded = True


LOAD_ERRORS: dict[str, str] = {}
