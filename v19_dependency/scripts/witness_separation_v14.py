"""Witness-state arrival-constrained certificate; independent research diagnostic.

This is a *known-witness* construction, NOT the adaptive VOI v0.10 policy.
It uses binomial tails, no simulator labels or fitted hyperparameters.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import binom


def binary_kl(p: float, q: float) -> float:
    if not (0 < p < 1 and 0 < q < 1):
        raise ValueError('p, q must lie in (0,1)')
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def np_cutoff(m: int, tau: float, alpha: float) -> int:
    """Largest integer k satisfying P_tau(Bin(m,tau)<=k) <= alpha."""
    if m < 1 or not (0 < tau < 1 and 0 < alpha < 1):
        raise ValueError('Invalid binomial test parameters')
    lo, hi = -1, m
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binom.cdf(mid, m, tau) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class WitnessCertificate:
    alpha: float
    beta: float
    q: float
    r: float
    tau: float
    b: float
    n: int
    m: int
    k: int
    arrival_probability: float
    conditional_power: float
    true_power: float
    false_certificate_probability_at_boundary: float
    targeted_information_lower_n: float
    uniform_information_lower_n: float

    @property
    def actual_label_budget_rate(self) -> float:
        return self.m / self.n


def certificate(n: int, m: int, *, q: float, r: float, tau: float,
                alpha: float, beta: float, b: float) -> WitnessCertificate:
    if n < 1 or m < 1 or m > math.floor(b * n):
        raise ValueError('The requested quota exceeds the hard label budget')
    if not (0 < q <= 1 and 0 < r < tau < .5 and 0 < b <= 1 and 0 < alpha < 1 - beta < 1):
        raise ValueError('Parameter assumptions violated')
    k = np_cutoff(m, tau, alpha)
    p_arr = float(binom.sf(m - 1, n, q))
    p_good = float(binom.cdf(k, m, r))
    p_bad = float(binom.cdf(k, m, tau))
    info = binary_kl(r, tau)
    kappa = binary_kl(1.0 - beta, alpha)
    return WitnessCertificate(
        alpha=alpha, beta=beta, q=q, r=r, tau=tau, b=b, n=n, m=m, k=k,
        arrival_probability=p_arr,
        conditional_power=p_good,
        true_power=p_arr * p_good,
        false_certificate_probability_at_boundary=p_arr * p_bad,
        targeted_information_lower_n=kappa / (min(q, b) * info),
        uniform_information_lower_n=kappa / (b * q * info),
    )


def smallest_horizon_for_m(m: int, *, q: float, b: float, required_arrival_probability: float) -> int | None:
    if not 0 < required_arrival_probability < 1:
        return None
    lo = max(m, math.ceil(m / b))
    hi = max(lo, math.ceil(1.5 * m / q))
    while binom.sf(m - 1, hi, q) < required_arrival_probability:
        hi *= 2
        if hi > 100_000_000:
            return None
    while lo < hi:
        mid = (lo + hi) // 2
        if binom.sf(m - 1, mid, q) >= required_arrival_probability:
            hi = mid
        else:
            lo = mid + 1
    return lo


def search_exact_certificate(*, q: float, r: float, tau: float,
                             alpha: float, beta: float, b: float,
                             max_m: int = 1500) -> WitnessCertificate:
    """Exact finite-parameter search, restricted to 1 <= m <= max_m.

    Searches fixed-quota, one-sided binomial tests. It does not optimize over all
    possible adaptive/sequential procedures.
    """
    best = None
    for m in range(1, max_m + 1):
        k = np_cutoff(m, tau, alpha)
        if k < 0:
            continue
        p_good = float(binom.cdf(k, m, r))
        if p_good < 1.0 - beta:
            continue
        required_p_arr = (1.0 - beta) / p_good
        if required_p_arr >= 1.0:
            continue
        n = smallest_horizon_for_m(m, q=q, b=b, required_arrival_probability=required_p_arr)
        if n is None:
            continue
        candidate = certificate(n, m, q=q, r=r, tau=tau, alpha=alpha, beta=beta, b=b)
        if candidate.true_power < 1.0 - beta - 1e-12:
            continue
        if best is None or (candidate.n, candidate.m) < (best.n, best.m):
            best = candidate
    if best is None:
        raise RuntimeError('No certificate found in quota search range')
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/witness_frontier_v14.csv')
    p.add_argument('--figure', default='figures/witness_frontier_v14.pdf')
    p.add_argument('--max-m', type=int, default=1500)
    args = p.parse_args()
    q, r, tau, b, beta = 1/8, .08, .15, .20, .10
    alpha_values = [.05, .01, .001, .0001]
    rows=[]
    for alpha in alpha_values:
        cert=search_exact_certificate(q=q,r=r,tau=tau,b=b,beta=beta,alpha=alpha,max_m=args.max_m)
        row={
            'alpha':cert.alpha,'beta':cert.beta,'q_witness':q,'risk_true':r,'risk_threshold':tau,
            'label_rate_limit':b,'n_targeted':cert.n,'m_targeted':cert.m,'test_errors_max':cert.k,
            'power_exact':cert.true_power,'false_certificate_at_tau_exact':cert.false_certificate_probability_at_boundary,
            'arrival_probability':cert.arrival_probability,
            'necessary_targeted_horizon':cert.targeted_information_lower_n,
            'necessary_uniform_horizon':cert.uniform_information_lower_n,
            'uniform_lower_over_targeted_achievable':cert.uniform_information_lower_n/cert.n,
            'targeted_over_necessary_targeted':cert.n/cert.targeted_information_lower_n,
        }
        rows.append(row)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7.6,4.8))
        xx=range(len(rows))
        ax.plot(xx,[x['necessary_targeted_horizon'] for x in rows],marker='o',label='Necessary: targeted / any adaptive')
        ax.plot(xx,[x['n_targeted'] for x in rows],marker='s',label='Sufficient: exact witness binomial test')
        ax.plot(xx,[x['necessary_uniform_horizon'] for x in rows],marker='^',label='Necessary: state-blind uniform querying')
        ax.set_xticks(list(xx),[str(x['alpha']) for x in rows])
        ax.set_xlabel('False-certification budget alpha')
        ax.set_ylabel('Target-arrival horizon n')
        ax.set_title('Known witness: information and availability bounds')
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        fig.tight_layout()
        path=Path(args.figure);path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    except ImportError:
        pass
    print(f'Wrote {len(rows)} rows to {dest}')
    for row in rows:
        print(f"alpha={row['alpha']:.4g} n={row['n_targeted']} m={row['m_targeted']} "
              f"power={row['power_exact']:.6f}, false={row['false_certificate_at_tau_exact']:.6f}, "
              f"uniform must n>={math.ceil(row['necessary_uniform_horizon'])}")


if __name__=='__main__':
    main()
