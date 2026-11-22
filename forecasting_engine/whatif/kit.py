"""Small helpers shared by the what-if mechanisms: number rounding, a timer and the result `meta` block."""
from __future__ import annotations

import math
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def _f(x, nd: int = 4):
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{nd}g}") if abs(x) < 1e15 else x


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def s(self) -> float:
        return round(time.time() - self.t0, 3)


def meta(sim: str, version: str, timer: Timer, status: str, assumptions=(), warnings=(), sources=(), **extra) -> dict:
    m = {"sim": sim, "version": version, "run_s": timer.s, "status": status, "assumptions": list(assumptions), "warnings": list(warnings),
         "sources": list(sources)}
    m.update(extra)
    return m
