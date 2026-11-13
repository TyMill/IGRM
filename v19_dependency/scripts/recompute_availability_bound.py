#!/usr/bin/env python3
"""Independent numerical check of the *necessary* arrival/label information barrier.

Does not produce or simulate target labels; results are analytic for the two-state
example, with an optional SLSQP confirmation of the least-favorable alternative.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
OUT = ROOT / 'data'
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

def kl(a: float, b: float) -> float:
    assert 0 < a < 1 and 0 < b < 1
    return a * math.log(a / b) + (1-a) * math.log((1-a)/(1-b))

def binary_kl(a: float, b: float) -> float:
    return kl(a,b)

def composite_info(n: float, B: float, *, q1=.6, r1=.14, tau=.15) -> float:
    """Exact max-min info for q=(.6,.4), r=(.14,.40), c=.6, tau=.15."""
    return min(B, n*q1) * kl(r1, tau)

def pair_info(n: float, B: float, *, q1=.6, r1=.14, s1=.16) -> float:
    return min(B, n*q1) * kl(r1, s1)

def numeric_least_favorable(x1=2400, x2=1000):
    q = np.array([.6,.4]); r=np.array([.14,.4]); tau=.15
    # Policy family for c=.6: {state1} and {state1,state2}.
    def objective(s):
        return x1*kl(r[0],s[0]) + x2*kl(r[1],s[1])
    constraints=[
        {'type':'ineq','fun':lambda s: s[0]-tau},
        {'type':'ineq','fun':lambda s: np.dot(q,s-tau)},
    ]
    result=minimize(objective, [.16,.4], method='SLSQP',
                    bounds=[(1e-8,1-1e-8)]*2,
                    constraints=constraints, options={'ftol':1e-12, 'maxiter':3000})
    if not result.success and abs(result.fun - x1*kl(.14,.15))>1e-7:
        raise RuntimeError(f'optimization failed: {result.message}')
    return result

def main():
    alpha=.05; beta=.10; b=.2
    kappa=kl(1-beta,alpha)
    d_simple=kl(.14,.16); d_composite=kl(.14,.15)
    pair_n=kappa/(b*d_simple)
    comp_n=kappa/(b*d_composite)
    pair_labels=kappa/d_simple
    comp_labels=kappa/d_composite
    # At all relevant horizons q1=.6 > b=.2 so budget, not availability,
    # is the limiting constraint. For a hypothetical unrestricted budget,
    # q1 caps the number of labels per arrival.
    B_construct=11081; n_construct=55405; k_construct=1600
    false_positive=float(binom.cdf(k_construct,B_construct,.15))
    conditional_power=float(binom.cdf(k_construct,B_construct,.14))
    availability_failure_upper=math.exp(-2*n_construct*(.6-B_construct/n_construct)**2)
    assert false_positive < alpha and conditional_power*(1-availability_failure_upper)>1-beta
    values={
        'constructive_horizon':n_construct,
        'constructive_queries':B_construct,
        'constructive_accept_errors_at_most':k_construct,
        'constructive_false_certification_bound':false_positive,
        'constructive_conditional_power':conditional_power,
        'constructive_arrival_failure_upper':availability_failure_upper,
        'constructive_lower_power_bound':conditional_power*(1-availability_failure_upper),
        'alpha':alpha,'beta':beta,'q':[.6,.4],'r':[.14,.4],
        'pair_s':[.16,.4],'c':.6,'tau':.15,'label_rate_budget':b,
        'kappa':kappa,'kl_014_016':d_simple,'kl_014_015':d_composite,
        'pair_required_expected_labels':pair_labels,
        'pair_required_arrivals_availability_only':pair_labels/.6,
        'pair_required_arrivals_with_rate_budget':pair_n,
        'composite_required_expected_labels':comp_labels,
        'composite_required_arrivals_availability_only':comp_labels/.6,
        'composite_required_arrivals_with_rate_budget':comp_n,
        'observations_at_n12000_B2400':composite_info(12000,2400),
        'simple_pair_info_at_n12000_B2400':pair_info(12000,2400),
        'warning':'information bounds are necessary, not sufficient; does not prove novel priority'
    }
    res=numeric_least_favorable()
    values['least_favorable_numeric_s']=list(map(float,res.x))
    values['least_favorable_numeric_objective']=float(res.fun)
    values['least_favorable_analytic_objective']=float(2400*d_composite)
    values['least_favorable_abs_error']=float(abs(res.fun-2400*d_composite))
    assert abs(res.fun-2400*d_composite)<1e-6, values
    assert 7705 < pair_n < 7707
    assert 29732 < comp_n < 29734
    assert composite_info(12000,2400) < kappa
    assert composite_info(35000,7000) >= kappa
    assert comp_n > pair_n
    (OUT/'availability_bound_v12.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
    horizons=np.arange(1000,37001,250)
    with (OUT/'arrival_budget_frontier_v12.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['n_arrivals','budget_20pct','max_kl_single_pair','max_kl_unsafe_family','required_kl'])
        for n in horizons:
            budget=math.floor(b*n)
            writer.writerow([n,budget,pair_info(n,budget),composite_info(n,budget),kappa])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(horizons,[pair_info(n,math.floor(b*n)) for n in horizons],label='Single comparison (14% vs 16%)',lw=2)
    ax.plot(horizons,[composite_info(n,math.floor(b*n)) for n in horizons],label='All unsafe alternatives (boundary 15%)',lw=2)
    ax.axhline(kappa, color='black',ls='--',lw=1.3,label='Required information')
    ax.axvline(12000,color='grey',linestyle=':',linewidth=1.1)
    ax.text(12300,kappa*.12,'12,000 arrivals',fontsize=8,color='grey')
    ax.set(xlabel='Target arrivals (n)',ylabel='Maximum expected log-likelihood information (nats)',title='Necessary information under a 20% label budget')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(frameon=False,loc='upper left',fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG/'availability_barrier_v12.pdf')
    fig.savefig(FIG/'availability_barrier_v12.png',dpi=180)
    print(json.dumps(values,indent=2))

if __name__=='__main__':
    main()
