"""Finite-horizon, hard-budget, data-adaptive overlap certification.

Research code. Known target state law q; stationary Bernoulli conditional losses;
queries depend only on states and previously observed losses. Prefixes at n=2**j
are covered by simultaneous alpha-spent CP intervals. Stage arrival blocks are
preallocated from a Chernoff bound, so the proof does not pretend skipped arrivals
can be queried retroactively.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, pi, sqrt
from typing import Sequence

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class AdaptiveResult:
    selected: tuple[int, ...] | None
    labels: int
    per_state: tuple[int, ...]
    completed_stages: int
    eliminated_masks: int
    stage_failure: bool
    certifier: str
    stage_lengths: tuple[int, ...]
    certified_at: int | None


def validate(q: np.ndarray, family: Sequence[tuple[int, ...]], tau: float, gamma: float) -> tuple[tuple[int, ...], ...]:
    K = len(q)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any(q <= 0) or not abs(float(np.sum(q)) - 1.0) <= 1e-12:
        raise ValueError('q must be a positive probability vector')
    if not 0 < tau < 1 or not 0 < gamma <= 1:
        raise ValueError('invalid tau/gamma')
    fam = tuple(tuple(int(i) for i in a) for a in family)
    if not fam or len(set(fam)) != len(fam):
        raise ValueError('nonempty distinct policy family required')
    if any(not a or len(set(a)) != len(a) or min(a) < 0 or max(a) >= K for a in fam):
        raise ValueError('invalid mask')
    return fam


def tail_alpha(delta: float, K: int, stage: int) -> float:
    """One-sided tail spend; total of two tails over K and all stages <=delta."""
    if not 0 < delta < 1 or K < 1 or stage < 0:
        raise ValueError('invalid spending parameters')
    return 3.0 * delta / (pi**2 * K * (stage + 1)**2)


def dyadic_bounds(k: np.ndarray, n: np.ndarray, stage: int, delta: float,
                  *, method: str = 'cp') -> tuple[np.ndarray, np.ndarray]:
    """Simultaneous two-sided state intervals at a dyadic prefix.

    Inputs must be counts of first 2**stage queried outcomes in each evaluated
    state, not data-dependent arbitrary prefixes.
    """
    k = np.asarray(k, dtype=np.int64); n = np.asarray(n, dtype=np.int64)
    if k.shape != n.shape or k.ndim != 1 or (k < 0).any() or (k > n).any() or (n < 0).any():
        raise ValueError('invalid binomial counts')
    K = len(n); a = tail_alpha(delta, K, stage)
    lower = np.zeros(K, dtype=float); upper = np.ones(K, dtype=float)
    for i in range(K):
        ni = int(n[i]); ki = int(k[i])
        if ni == 0:
            continue
        if ni != 2**stage:
            raise ValueError('only dyadic completed prefixes may be evaluated')
        p = ki / ni
        if method == 'cp':
            lower[i] = 0.0 if ki == 0 else float(beta.ppf(a, ki, ni - ki + 1))
            upper[i] = 1.0 if ki == ni else float(beta.ppf(1 - a, ki + 1, ni - ki))
        elif method == 'hoeffding':
            e = sqrt(log(1 / a) / (2 * ni))
            lower[i] = max(0.0, p - e)
            upper[i] = min(1.0, p + e)
        else:
            raise ValueError('unknown method')
    return lower, upper


def epsilon(stage: int, K: int, delta: float) -> float:
    return sqrt(log(1.0 / tail_alpha(delta, K, stage)) / (2.0 * (2**stage)))


def phase_length(increment: int, q_min: float, stage: int, delta_arrival: float, K: int) -> int:
    """Chernoff-block sufficient length for every state (union over stages)."""
    if increment < 1 or not 0 < q_min <= 1 or not 0 < delta_arrival < 1:
        raise ValueError('invalid phase arguments')
    failure_share = 6 * delta_arrival / (pi**2 * K * (stage + 1)**2)
    t = log(1 / failure_share)
    z = sqrt(2 * t)
    y = (z + sqrt(z * z + 4 * increment)) / 2
    return ceil(y * y / q_min)


def design_phases(K: int, q_min: float, H: int, budget: int,
                  delta_arrival: float) -> tuple[int, ...]:
    """Largest dyadic last stage whose worst-case labels and phase blocks fit."""
    phases = []
    for j in range(40):
        n_j = 2**j
        if K * n_j > budget:
            break
        increment = 1 if j == 0 else 2**(j-1)
        h_j = phase_length(increment, q_min, j, delta_arrival, K)
        if sum(phases) + h_j > H:
            break
        phases.append(h_j)
    return tuple(phases)


def mask_margin_from_limits(q: np.ndarray, r_bounds: np.ndarray,
                            a: tuple[int, ...], tau: float) -> float:
    idx = list(a)
    return float(np.sum(q[idx] * (tau - r_bounds[idx])))


def choose_best(family: Sequence[tuple[int, ...]], q: np.ndarray) -> tuple[int, ...] | None:
    if not family:
        return None
    return max(family, key=lambda a: (float(q[list(a)].sum()), tuple(-i for i in a)))


def run_adaptive(states: np.ndarray, hidden_losses: np.ndarray, q: np.ndarray,
                 family: Sequence[tuple[int, ...]], *, tau: float = .15,
                 gamma: float = .1, budget_fraction: float = .2,
                 delta_cert: float = .025, delta_arrival: float = .025,
                 method: str = 'cp', max_stages: int | None = None,
                 query_all_states: bool = False) -> AdaptiveResult:
    """Run stream acquisition; never accesses a loss before deciding its query.

    Every stage operates on its own predetermined stream block. At stage j it
    requests at most Delta_j=2**(j-1) fresh labels per still-active state. Masks
    are certified/pruned only after all active-state quotas in the stage are met.
    """
    states = np.asarray(states, dtype=int)
    hidden_losses = np.asarray(hidden_losses, dtype=np.int8)
    q = np.asarray(q, dtype=float)
    if states.ndim != 1 or len(states) != len(hidden_losses) or not np.isin(hidden_losses, [0, 1]).all():
        raise ValueError('stream dimensions or loss values invalid')
    fam = validate(q, family, tau, gamma)
    H, K = len(states), len(q)
    if np.any(states < 0) or np.any(states >= K) or not 0 < budget_fraction <= 1:
        raise ValueError('invalid states/budget')
    B = int(np.floor(budget_fraction * H))
    # A state outside every candidate policy cannot influence a certificate.
    # Fix from the v1.7 independent audit: do not let irrelevant rare states
    # determine the arrival schedule or consume the pathwise label budget.
    support0 = set(range(K)) if query_all_states else set(i for a in fam for i in a)
    blocks = design_phases(len(support0), float(min(q[list(support0)])), H, B, delta_arrival)
    if max_stages is not None:
        blocks = blocks[:max_stages]
    if not blocks:
        return AdaptiveResult(None, 0, (0,) * K, 0, 0, False, method, blocks, None)
    counts = np.zeros(K, dtype=int)
    errors = np.zeros(K, dtype=int)
    active = list(fam)
    best = None
    cert_at = None
    pos = 0; completed = 0; eliminated = 0; failed = False
    for j, length in enumerate(blocks):
        if not active:
            break
        increment = 1 if j == 0 else 2**(j-1)
        current = states[pos:pos + length]
        # The queried indices are chosen from state arrivals BEFORE looking at losses.
        support = set(range(K)) if query_all_states else set(i for a in active for i in a)
        active_support = set(i for a in active for i in a)
        new_indices = []
        for i in sorted(support):
            loc = np.flatnonzero(current == i)
            if len(loc) < increment:
                failed = True
            # Query each state's first Delta_j arrivals; this can be executed
            # causally online. Any partial phase still consumes those labels.
            new_indices.extend((pos + loc[:increment]).tolist())
        for idx in new_indices:
            i = int(states[idx]); counts[i] += 1; errors[i] += int(hidden_losses[idx])
        pos += length
        assert counts.sum() <= B
        if failed:
            break
        eval_counts = np.asarray([counts[i] if i in active_support else 0 for i in range(K)], dtype=int)
        eval_errors = np.asarray([errors[i] if i in active_support else 0 for i in range(K)], dtype=int)
        L, U = dyadic_bounds(eval_errors, eval_counts, j, delta_cert, method=method)
        certified = [a for a in active if q[list(a)].sum() >= gamma and
                     mask_margin_from_limits(q, U, a, tau) >= 0.0]
        new_best = choose_best(certified, q)
        if new_best is not None and (best is None or q[list(new_best)].sum() > q[list(best)].sum()):
            best = new_best; cert_at = j
        new_active = [a for a in active if mask_margin_from_limits(q, L, a, tau) >= 0.0]
        eliminated += len(active) - len(new_active)
        active = new_active
        completed += 1
        if best is not None and (not active or q[list(best)].sum() >= max(q[list(a)].sum() for a in active)):
            break
    return AdaptiveResult(best, int(counts.sum()), tuple(int(n) for n in counts),
                          completed, eliminated, failed, method, blocks, cert_at)


def fixed_uniform_cp(states: np.ndarray, hidden_losses: np.ndarray, q: np.ndarray,
                     family: Sequence[tuple[int, ...]], *, tau: float, gamma: float,
                     n_labels: int, alpha: float, seed: int) -> AdaptiveResult:
    """Strong fixed-budget baseline: label indices drawn independently of outcomes.

    Since the label indices are selected *before* observing any losses, ordinary
    fixed-sample Bonferroni CP intervals are valid, without anytime spending.
    """
    fam = validate(np.asarray(q, dtype=float), family, tau, gamma)
    H = len(states); K = len(q)
    if not 0 <= n_labels <= H:
        raise ValueError('invalid fixed budget')
    rng = np.random.default_rng(seed)
    ix = rng.choice(H, size=n_labels, replace=False)
    n = np.bincount(states[ix], minlength=K)
    e = np.bincount(states[ix], weights=hidden_losses[ix], minlength=K).astype(int)
    upper = np.ones(K, dtype=float)
    for i in range(K):
        if n[i]:
            upper[i] = 1.0 if e[i] == n[i] else float(beta.ppf(1 - alpha / K, e[i] + 1, n[i] - e[i]))
    certified = [a for a in fam if q[list(a)].sum() >= gamma and
                 mask_margin_from_limits(q, upper, a, tau) >= 0.0]
    return AdaptiveResult(choose_best(certified, q), n_labels,
                          tuple(int(x) for x in n), 1, 0, False, 'uniform-cp-fixed', (), 0 if certified else None)
