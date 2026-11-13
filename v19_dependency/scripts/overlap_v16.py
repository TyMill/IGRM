"""Overlapping-policy fixed-horizon safety certificates.

A deliberately restricted known-q, independently queried state-label experiment.
It does NOT establish an adaptive-VOI acquisition efficiency theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta


@dataclass(frozen=True)
class DesignResult:
    n_real: tuple[float, ...]
    n_integer: tuple[int, ...]
    objective_real: float
    residual_max: float


def margin(q: np.ndarray, r: np.ndarray, a: tuple[int, ...], tau: float) -> float:
    return float(np.sum(q[list(a)] * (tau - r[list(a)])))


def variance_proxy(q: np.ndarray, n: np.ndarray, a: tuple[int, ...]) -> float:
    if np.any(n[list(a)] <= 0):
        return float("inf")
    return float(np.sum(q[list(a)] ** 2 / n[list(a)]))


def simultaneous_radius(q: np.ndarray, n: np.ndarray, a: tuple[int, ...], family_size: int, alpha: float) -> float:
    return sqrt(0.5 * log(2 * family_size / alpha) * variance_proxy(q, n, a))


def shared_allocation(q: np.ndarray, family: Sequence[tuple[int, ...]], target_margins: Sequence[float], alpha: float = 0.05) -> DesignResult:
    """Minimum continuous label count satisfying all Hoeffding margin constraints.

    An oracle *design* if target margins are unknown; it never observes simulator
    truth when used as a test-time *certifier*.
    """
    q = np.asarray(q, dtype=float)
    family = tuple(tuple(a) for a in family)
    if not np.isclose(np.sum(q), 1) or np.any(q <= 0):
        raise ValueError("q must lie in the positive probability simplex")
    if len(family) != len(target_margins) or not family:
        raise ValueError("invalid family / target margins")
    L = log(2 * len(family) / alpha)
    limits = np.asarray(target_margins, dtype=float) ** 2 / (2 * L)
    if np.any(limits <= 0):
        raise ValueError("strictly positive design margins required")
    K = len(q)
    # Unused states have zero quota. Optimize only the union.
    active = sorted(set().union(*map(set, family)))
    pos = {j: i for i, j in enumerate(active)}
    fidx = [[pos[j] for j in a] for a in family]
    qa = q[active]
    # Start feasible by making every state's own variance contribution small.
    x0 = np.full(len(active), 2 * len(active) * max(float(np.sum(q[list(a)]**2)) for a in family) / min(limits))

    def objective(x):
        return float(x.sum())

    def cons(x, ids, a, lim):
        return lim - float(np.sum(q[list(a)] ** 2 / x[ids]))

    constraints = [{"type": "ineq", "fun": lambda x, ids=ids, a=a, lim=lim: cons(x, ids, a, lim)} for ids, a, lim in zip(fidx, family, limits)]
    result = minimize(objective, x0, method="SLSQP", bounds=[(1e-6, None)] * len(active), constraints=constraints,
                      options={"maxiter": 3000, "ftol": 1e-10})
    if not result.success:
        # SLSQP can fail at large scales; use closed form or a high-accuracy
        # numerical solver for the reported design rather than silent fallback.
        raise RuntimeError(f"SLSQP failed: {result.message}")
    n = np.zeros(K)
    n[active] = result.x
    integer = np.ceil(n).astype(int)
    residual = max(float(variance_proxy(q, integer, a) - lim) for a, lim in zip(family, limits))
    assert residual <= 1e-9
    return DesignResult(tuple(map(float, n)), tuple(map(int, integer)), float(n.sum()), residual)


def certifier(q: np.ndarray, errors: np.ndarray, n: np.ndarray, family: Sequence[tuple[int, ...]], tau: float, gamma: float, alpha: float = 0.05, method: str = "shared-hoeffding") -> tuple[int, ...] | None:
    """Select maximum-coverage certified mask using ONLY queried errors and q."""
    M = len(family)
    candidates = []
    for a in family:
        a = tuple(a)
        if any(n[j] <= 0 for j in a):
            continue
        cov = float(q[list(a)].sum())
        if cov < gamma:
            continue
        hat_r = errors / np.maximum(n, 1)
        hat_margin = margin(q, hat_r, a, tau)
        if method == "shared-hoeffding":
            threshold = simultaneous_radius(q, n, a, M, alpha)
            acceptable = hat_margin >= threshold
        elif method == "cp-box":
            # Bonferroni simultaneous upper bounds; fixed n only.
            upper = np.ones(len(q))
            for j in range(len(q)):
                if n[j] > 0:
                    k = int(errors[j]); nj = int(n[j])
                    upper[j] = 1.0 if k == nj else float(beta.ppf(1 - alpha / len(q), k + 1, nj - k))
            acceptable = margin(q, upper, a, tau) >= 0
        else:
            raise ValueError(method)
        if acceptable:
            candidates.append((cov, a))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: (pair[0], tuple(-j for j in pair[1])))[1]


def closed_form_three_state(alpha: float = 0.05, r: float = 0.08, tau: float = 0.15) -> dict[str, float]:
    q = np.array([1/3] * 3)
    g = 2 * (tau - r) / 3
    s = 9 * g ** 2 / (2 * log(4 / alpha))
    x = (1 + 1 / sqrt(2)) / s
    y = (1 + sqrt(2)) / s
    return {"margin": g, "s": s, "outer_each": x, "central": y, "shared_total": 2 * x + y,
            "uniform_shared_total": 6 / s, "isolated_total": 8 / s}
