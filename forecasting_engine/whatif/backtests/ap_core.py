"""Shared pieces of the "Asymmetric and pooled links" work (docs/methods.md, section "Asymmetric and pooled links").

1. Asymmetric response: a shock series dx is split into its rises and falls (optionally with a dead band c: moves smaller than c count as zero),
   each with its own lag kernel, so a link can have different pass-through and different speed up and down.
2. Hierarchical (partial-pooling) estimation over units (countries): country coefficients theta_c ~ N(theta_pool, tau^2); the pooled mean is a
   precision-weighted ridge towards the prior and tau^2 is learned from the data by the DerSimonian-Laird moment estimator (a random-effects
   model; tau^2 = 0 means full pooling, tau^2 large means no pooling). "fixed" pooling (the legacy rule: tau = prior/2) is kept for comparison.
No fitted quantity here looks beyond the data public at the origin.
"""
from __future__ import annotations

import numpy as np


def split_updown(dx, c: float = 0.0):
    """dx: pandas Series of changes. Returns (up, down): up = dx where dx > c else 0; down = dx where dx < -c else 0 (NaN stays NaN)."""
    up = dx.where(~(dx <= c), 0.0)
    dn = dx.where(~(dx >= -c), 0.0)
    return up, dn


def geo_kernel(rho: float, K: int) -> np.ndarray:
    w = np.array([(1 - rho) * rho ** k for k in range(K + 1)]) if rho > 0 else np.r_[1.0, np.zeros(K)]
    return w / w.sum()


def kernel_series(d, rho: float, delay: int, K: int):
    """Distributed lag of a change series d (pandas Series): sum_k w_k d(t - k - delay), weights sum to 1."""
    w = geo_kernel(rho, K)
    return sum(w[k] * d.shift(k + delay) for k in range(K + 1))


def dl_tau2(est: np.ndarray, var: np.ndarray) -> float:
    """DerSimonian-Laird between-unit variance of coefficient estimates est (one per unit) with sampling variances var."""
    est, var = np.asarray(est, float), np.asarray(var, float)
    ok = np.isfinite(est) & np.isfinite(var) & (var > 0)
    est, var = est[ok], var[ok]
    if len(est) < 3:
        return float("nan")
    w = 1.0 / var
    mu = float((w * est).sum() / w.sum())
    Q = float((w * (est - mu) ** 2).sum())
    den = float(w.sum() - (w ** 2).sum() / w.sum())
    return max(0.0, (Q - (len(est) - 1)) / den) if den > 0 else float("nan")


def pooled_fit(stats: dict, th0: np.ndarray, sd0: np.ndarray, eb: bool, fixed_sd: np.ndarray, tau_floor: float = 1e-3, theta_max: float = 0.5, clip: bool = True):
    """stats: {unit: (XtX (k,k), Xty (k,), n, yty)} from within-unit demeaned data.
    Pool: ridge towards th0 with prior sd sd0. Units: shrunk towards the pool with sd tau (eb: DL moment estimate per coefficient; else fixed_sd).
    Returns (theta {unit: (k,)}, pool (k,), tau (k,), s2)."""
    k = len(th0)
    XtX = sum(v[0] for v in stats.values())
    Xty = sum(v[1] for v in stats.values())
    n = sum(v[2] for v in stats.values())
    yty = sum(v[3] for v in stats.values())
    ols = np.linalg.lstsq(XtX + 1e-12 * np.eye(k), Xty, rcond=None)[0]
    s2 = max((yty - 2 * ols @ Xty + ols @ XtX @ ols) / max(n - k, 1), 1e-12)
    P0 = np.diag(1.0 / sd0 ** 2)
    pool = np.linalg.solve(XtX / s2 + P0, Xty / s2 + P0 @ th0)
    tau = np.array(fixed_sd, float)
    if eb:
        est, var = {}, {}
        for u, (A, b, nn, _) in stats.items():
            try:
                Ai = np.linalg.inv(A + 1e-12 * np.eye(k))
            except np.linalg.LinAlgError:
                continue
            est[u], var[u] = Ai @ b, s2 * np.diag(Ai)
        tau = np.empty(k)
        for j in range(k):
            t2 = dl_tau2(np.array([est[u][j] for u in est]), np.array([var[u][j] for u in est]))
            tau[j] = fixed_sd[j] if not np.isfinite(t2) else max(np.sqrt(t2), tau_floor)
    if clip:
        pool = np.clip(pool, 0, theta_max)          # the pool is clipped before it is used as the units' prior mean (as in fuel_fare.fit_theta)
    out = {}
    for u, (A, b, nn, _) in stats.items():
        Pt = np.diag(1.0 / tau ** 2)
        th = np.linalg.solve(A / s2 + Pt, b / s2 + Pt @ pool)
        out[u] = np.clip(th, 0, theta_max) if clip else th
    return out, pool, tau, s2
