"""Finite-horizon simultaneous certifier for a categorical stream.

This is a constructive, fixed-horizon *analysis baseline* independent of
IGRM v0.10's adaptive value-of-information heuristic.  It never sees the
true q/r arrays when making its certification decision.
"""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Certificate:
    mask: tuple[int, ...]
    coverage_lower: float
    numerator_upper: float
    observed_labels: int


def radii(n: int, quotas: np.ndarray, alpha_q: float, alpha_r: float):
    quotas = np.asarray(quotas, dtype=np.int64)
    k = len(quotas)
    if n <= 0 or k < 1 or np.any(quotas <= 0) or not 0 < alpha_q < 1 or not 0 < alpha_r < 1:
        raise ValueError("invalid sample size, quota, or confidence allocation")
    dq = math.sqrt(((k+1) * math.log(2.0) + math.log(1.0 / alpha_q)) / (2.0 * n))
    dr = np.sqrt(math.log(2.0 * k / alpha_r) / (2.0 * quotas))
    return dq, dr


def certify(states: np.ndarray, queried_losses: list[np.ndarray], quotas: np.ndarray,
            tau: float = .15, coverage_target: float = .75,
            alpha_q: float = .025, alpha_r: float = .025,
            family: list[tuple[int, ...]] | None = None) -> Certificate | None:
    states = np.asarray(states, dtype=np.int64)
    quotas = np.asarray(quotas, dtype=np.int64)
    k = len(quotas)
    if not 0 < tau < 1 or not 0 < coverage_target <= 1 or np.any(states < 0) or np.any(states >= k):
        raise ValueError("invalid target risk/coverage or state index")
    if len(queried_losses) != k or any(len(queried_losses[i]) > quotas[i] for i in range(k)):
        raise ValueError("loss arrays must reflect only queried labels")
    if any(len(queried_losses[i]) < quotas[i] for i in range(k)):
        return None  # shortage: never silently estimate from fewer labels
    if np.any(np.bincount(states, minlength=k) < quotas):
        return None  # physically unavailable observations
    if any(np.any(~np.isin(np.asarray(queried_losses[i]), [0, 1])) for i in range(k)):
        raise ValueError("losses must be observed Bernoulli 0/1 outcomes")
    if np.sum(quotas) > len(states):
        raise ValueError("budget cannot exceed target horizon")
    dq, dr = radii(len(states), quotas, alpha_q, alpha_r)
    qhat = np.bincount(states, minlength=k) / len(states)
    rhat = np.asarray([np.mean(np.asarray(queried_losses[i], dtype=float)) for i in range(k)])
    if np.any((rhat < 0) | (rhat > 1)):
        raise ValueError("losses must be Bernoulli")
    if family is None:
        family = [tuple(c) for nsel in range(1, k+1) for c in itertools.combinations(range(k), nsel)]
    winner = None
    for mask in family:
        if not mask or any(i < 0 or i >= k for i in mask):
            raise ValueError("invalid mask")
        idx = np.array(mask, dtype=int)
        L = float(np.sum(qhat[idx]) - dq)
        U = float(np.dot(qhat[idx], rhat[idx]) + dq + np.dot(qhat[idx], dr[idx]))
        if L >= coverage_target and U <= tau * L:
            new = Certificate(tuple(mask), L, U, int(np.sum(quotas)))
            if winner is None or (new.coverage_lower > winner.coverage_lower + 1e-12) or (abs(new.coverage_lower - winner.coverage_lower) <= 1e-12 and new.mask < winner.mask):
                winner = new
    return winner


def sample_prefix_labels(states: np.ndarray, losses: np.ndarray, quotas: np.ndarray):
    """Simulated offline prefix querying; losses of unqueried arrivals are not exposed to certify()."""
    states, losses = np.asarray(states), np.asarray(losses)
    quotas = np.asarray(quotas)
    result = [[] for _ in quotas]
    for s, y in zip(states, losses):
        if len(result[s]) < quotas[s]:
            result[s].append(int(y))
    return [np.asarray(x, dtype=int) for x in result]
