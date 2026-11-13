"""Finite-horizon unknown-witness certificate under iid categorical arrivals.

Research code, v1.5. Uses independent screening and confirmation phases.
No experiment may substitute these simulator parameters for deployment labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, exp, floor, log
import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class Design:
    K: int
    r_safe: float
    r_other: float
    tau: float
    b: float
    alpha: float
    beta: float
    m0: int
    n0: int
    m: int
    k: int
    n1: int
    pilot_misidentification_bound: float
    pilot_arrival_failure_bound: float
    confirmation_arrival_failure: float
    confirmation_type1_bound: float
    confirmation_power: float

    @property
    def n(self) -> int:
        return self.n0 + self.n1

    @property
    def queries_max(self) -> int:
        return self.K * self.m0 + self.m

    @property
    def recovery_lower_bound(self) -> float:
        return 1 - (
            self.pilot_misidentification_bound
            + self.pilot_arrival_failure_bound
            + self.confirmation_arrival_failure
            + 1 - self.confirmation_power
        )


def largest_binomial_safe_k(m: int, tau: float, alpha: float) -> int:
    """Largest integer k with P[Bin(m,tau) <= k] <= alpha."""
    # binomial ppf returns the smallest k at which cdf reaches alpha.
    k = int(binom.ppf(alpha, m, tau))
    while k >= 0 and binom.cdf(k, m, tau) > alpha:
        k -= 1
    while k + 1 <= m and binom.cdf(k + 1, m, tau) <= alpha:
        k += 1
    return k


def design_unknown_witness(
    K: int = 8, r_safe: float = 0.08, r_other: float = 0.95,
    tau: float = 0.15, b: float = 0.20,
    alpha: float = 0.05, beta: float = 0.10,
    max_n: int = 400_000,
) -> Design:
    if not (K >= 2 and 0 < r_safe < tau < 0.5 and 2 * tau < r_other <= 1):
        raise ValueError('Requires K>=2, 0<r_safe<tau<0.5, r_other>2tau')
    if not (0 < b <= 1 and 0 < alpha < 1 - beta < 1):
        raise ValueError('Invalid budget or error probabilities')
    q = 1 / K
    delta = r_other - r_safe
    m0 = ceil(2.0 * log(4 * K / beta) / delta**2)
    misid = K * exp(-m0 * delta**2 / 2)
    if misid > beta / 4 + 1e-12:
        raise AssertionError('screening error allocation violated')

    n0 = K * m0
    while K * binom.cdf(m0 - 1, n0, q) > beta / 4:
        n0 += 1
        if n0 > max_n:
            raise ValueError('Pilot arrival requirement too expensive')
    pilot_arrival = K * binom.cdf(m0 - 1, n0, q)

    m = 1
    while True:
        k = largest_binomial_safe_k(m, tau, alpha)
        power = 0 if k < 0 else float(binom.cdf(k, m, r_safe))
        if power >= 1 - beta / 4:
            break
        m += 1
        if m > max_n:
            raise ValueError('Confirmation sample requirement too expensive')

    # n1 is a separate, fresh confirmation epoch. Budget is across both stages.
    n1 = max(m, floor((K * m0 + m) / b) - n0)
    while (
        binom.cdf(m - 1, n1, q) > beta / 4
        or floor(b * (n0 + n1)) < K * m0 + m
    ):
        n1 += 1
        if n0 + n1 > max_n:
            raise ValueError('Not enough observations under the given budget')
    conf_arrival = float(binom.cdf(m - 1, n1, q))
    return Design(K, r_safe, r_other, tau, b, alpha, beta,
                  m0, n0, m, k, n1, misid, pilot_arrival,
                  conf_arrival, float(binom.cdf(k, m, tau)), power)


def trial(d: Design, rng: np.random.Generator, rates: np.ndarray | None = None, witness: int = 0) -> tuple[bool, int, int, int]:
    """Return (certified, chosen_state, actual_query_count, true_risk_fail)."""
    if rates is None:
        rates = np.full(d.K, d.r_other)
        rates[witness] = d.r_safe
    rates = np.asarray(rates)
    if rates.shape != (d.K,):
        raise ValueError('Rate vector has incorrect shape')
    counts0 = rng.multinomial(d.n0, np.full(d.K, 1/d.K))
    actual0 = int(np.minimum(counts0, d.m0).sum())
    if np.any(counts0 < d.m0):
        return False, -1, actual0, 0
    pilot_errors = rng.binomial(d.m0, rates)
    chosen = int(np.argmin(pilot_errors))
    count1 = int(rng.binomial(d.n1, 1/d.K))
    m_actual = min(d.m, count1)
    if m_actual < d.m:
        return False, chosen, actual0 + m_actual, 0
    confirmation_errors = int(rng.binomial(d.m, rates[chosen]))
    cert = confirmation_errors <= d.k
    unsafe = int(cert and rates[chosen] > d.tau)
    return bool(cert), chosen, actual0 + d.m, unsafe


def monte_carlo(d: Design, repetitions: int = 2000, seed: int = 20261112, witness: int = 3):
    rng = np.random.default_rng(seed)
    safe = np.array([trial(d, rng, witness=witness) for _ in range(repetitions)], dtype=object)
    null_rates = np.full(d.K, d.r_other)
    null_rates[witness] = d.tau + 0.005
    null = np.array([trial(d, rng, null_rates, witness=witness) for _ in range(repetitions)], dtype=object)
    return {
        'safe_world_power': float(np.mean(safe[:,0].astype(bool))),
        'safe_world_correct_recovery': float(np.mean(safe[:,0].astype(bool) & (safe[:,1].astype(int) == witness))),
        'null_false_cert_rate': float(np.mean(null[:,0].astype(bool))),
        'mean_queries_safe': float(np.mean(safe[:,2].astype(int))),
        'max_queries_safe': int(max(safe[:,2])),
        'repetitions': repetitions,
    }


def kl_bernoulli(p: float, q: float) -> float:
    from math import log
    return p * log(p/q) + (1-p) * log((1-p)/(1-q))


def lower_horizons(d: Design):
    num = kl_bernoulli(1-d.beta, d.alpha)
    den = kl_bernoulli(d.r_safe, d.tau)
    q = 1 / d.K
    return ceil(num/(min(d.b,q)*den)), ceil(num/(d.b*q*den))


if __name__ == '__main__':
    import json
    from dataclasses import asdict
    d = design_unknown_witness()
    print(json.dumps({'design':asdict(d), 'horizon':d.n, 'total_label_budget':floor(d.b*d.n),
                      'queries_max':d.queries_max, 'guaranteed_power':d.recovery_lower_bound,
                      'necessary_horizons':lower_horizons(d), 'simulation':monte_carlo(d)}, indent=2))


def discovery_information_lp(d: Design):
    """Real-valued necessary horizon from pairwise transcript KL constraints.

    Variables are n and the expected query counts u_i in the safe world j=0.
    This is a lower bound, not an attainable protocol.
    """
    from scipy.optimize import linprog
    K=d.K
    D_safe=kl_bernoulli(d.r_safe,d.tau)
    D_rh=kl_bernoulli(d.r_safe,d.r_other)
    D_hr=kl_bernoulli(d.r_other,d.r_safe)
    risk_kl=kl_bernoulli(1-d.beta,d.alpha)
    ident_kl=kl_bernoulli(1-d.beta,d.beta)
    A=[];rhs=[]
    # Variables x=[n,u_0,...,u_{K-1}].
    for i in range(K):
        row=np.zeros(K+1);row[0]=-1/K;row[1+i]=1
        A.append(row);rhs.append(0)
    row=np.zeros(K+1);row[0]=-d.b;row[1:]=1
    A.append(row);rhs.append(0)
    row=np.zeros(K+1);row[1]=-D_safe
    A.append(row);rhs.append(-risk_kl)
    for k in range(1,K):
        row=np.zeros(K+1);row[1]=-D_rh;row[1+k]=-D_hr
        A.append(row);rhs.append(-ident_kl)
    c=np.zeros(K+1);c[0]=1
    sol=linprog(c,A_ub=np.array(A),b_ub=np.array(rhs),bounds=[(0,None)]*(K+1),method='highs')
    if not sol.success: raise RuntimeError(sol.message)
    return {'necessary_horizon_real':float(sol.x[0]),
            'expected_labels':sol.x[1:].tolist(),
            'risk_KL_requirement':risk_kl,
            'identity_KL_requirement':ident_kl,
            'risk_divergence':D_safe,
            'identity_divergences':(D_rh,D_hr)}
