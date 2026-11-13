"""v1.9 finite-horizon one-sided CP spending; no outcome-peeking or oracle q.

This is a PRE-SPECIFIED error-spending improvement to v1.8, not a new CS or
an algorithm proven statistically optimal. Safety via union-bound over
predeclared checkpoints, states, and independent unlabeled q pilot.

Upper risk bounds use sum_j alpha_upper[j]=delta_cert/K PER STATE.
Lower elimination bounds use a SEPARATE budget delta_elim; undercoverage of
these lowers may affect recovery/power, but not false certification safety.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import floor, pi
from typing import Sequence
import numpy as np
from scipy.stats import beta
from scripts.adaptive_overlap_v17 import validate, design_phases
from scripts.unknown_q_v18 import (UnknownQResult, pilot_q_box,
    robust_coverage, robust_margin)


def terminal_weights(J: int, terminal_share: float = 0.8) -> np.ndarray:
    """Fixed (non-data-dependent) finite-horizon alpha-spending schedule.

    For J>1, terminal receives terminal_share, earlier checkpoints split the
    remainder proportional to 1/(j+1)^2. No unused alpha can be reassigned.
    """
    if not isinstance(J, int) or J < 1 or not np.isfinite(terminal_share) or not 0 < terminal_share < 1:
        raise ValueError('invalid stages/terminal share')
    if J == 1:
        return np.array([1.0])
    j = np.arange(1, J, dtype=float)
    early = j**(-2)
    weights = np.concatenate(((1 - terminal_share) * early / early.sum(),
                              [terminal_share]))
    assert np.all(weights > 0) and np.isclose(weights.sum(), 1, atol=1e-14)
    return weights


def finite_cp_bounds(errors: np.ndarray, counts: np.ndarray, stage: int, *,
                     J: int, K: int, delta_cert: float = .025,
                     delta_elim: float = .025, terminal_share: float = .8):
    """Simultaneous per-state one-sided CP upper and lower at dyadic checkpoints.

    Alpha upper budget is delta_cert over all K states and J checkpoints.
    Lower-bound budget is delta_elim, independent of the certification budget.
    Stage count is chosen from a pilot without looking at queried losses.
    """
    e = np.asarray(errors,dtype=np.int64); n=np.asarray(counts,dtype=np.int64)
    if not (0 < delta_cert < 1 and 0 < delta_elim < 1) or K < 1 or not (0 <= stage < J):
        raise ValueError('invalid error budget/checkpoint')
    if e.shape != n.shape or e.shape != (K,) or np.any(n < 0) or np.any(e < 0) or np.any(e > n):
        raise ValueError('invalid counts')
    positive = n > 0
    if np.any(n[positive] != (1 << stage)):
        raise ValueError('non-dyadic checkpoint')
    weights=terminal_weights(J,terminal_share)
    a_upper=delta_cert*weights[stage]/K
    # Spending for lower confidence bounds is infinite-horizon valid:
    # sum_{stage >=0} 6/(pi^2 (stage+1)^2)=1.
    a_lower=(6.0/(pi*pi)) * delta_elim / (K * (stage+1)**2)
    lo=np.zeros(K); up=np.ones(K)
    for i in range(K):
        ni=int(n[i]); ki=int(e[i])
        if ni:
            if ki > 0:
                lo[i]=float(beta.ppf(a_lower,ki,ni-ki+1))
            if ki < ni:
                up[i]=float(beta.ppf(1.0-a_upper,ki+1,ni-ki))
    return lo,up


def run_unknown_q_finite_cp(states: np.ndarray, hidden_losses: np.ndarray, K: int,
        family: Sequence[tuple[int,...]], *, pilot_n:int=4096,tau:float=.15,
        gamma:float=.25,budget_fraction:float=.2,delta_q:float=.025,
        delta_cert:float=.025,delta_elim:float=.025,delta_arrival:float=.025,
        terminal_share:float=.8,max_stages:int|None=None,
        query_all_states:bool=False) -> UnknownQResult:
    """A label-blind pilot designs a finite checkpoint horizon; one-sided CP certifies.

    The true q and r are not inputs. Every queried loss is inspected only
    after a query index is chosen from the visible state arrival and past.
    """
    states=np.asarray(states,dtype=int);losses=np.asarray(hidden_losses,dtype=np.int8)
    if states.ndim!=1 or losses.shape!=states.shape or not np.isin(losses,[0,1]).all():
        raise ValueError('invalid stream')
    if not (K>=1 and 0<pilot_n<len(states)) or not (0<budget_fraction<=1):
        raise ValueError('invalid K/pilot/budget')
    if np.any(states<0) or np.any(states>=K) or not (0<delta_q<1 and 0<delta_arrival<1):
        raise ValueError('invalid states/error levels')
    if not (0<delta_cert<1 and 0<delta_elim<1):
        raise ValueError('invalid cert/elimination budgets')
    fam=validate(np.full(K,1/K),family,tau,gamma)
    qhat,ql,qu=pilot_q_box(states[:pilot_n],K,delta_q)
    initial_support=set(range(K)) if query_all_states else {i for a in fam for i in a}
    B=int(floor(budget_fraction*len(states)))
    if not initial_support or min(ql[list(initial_support)]) <= 0:
        blocks=()
    else:
        blocks=design_phases(len(initial_support),float(min(ql[list(initial_support)])),
                     len(states)-pilot_n,B,delta_arrival)
    if max_stages is not None:
        if max_stages < 0:raise ValueError('negative stage cap')
        blocks=blocks[:max_stages]
    counts=np.zeros(K,dtype=int);errors=np.zeros(K,dtype=int)
    active=list(fam)
    best=None; best_cl=-1.;cert_at=None
    position=pilot_n;completed=0;stage_failed=False
    J=len(blocks)
    for j,length in enumerate(blocks):
        if not active:break
        delta=1 if j==0 else 2**(j-1)
        block=states[position:position+length]
        active_support={i for a in active for i in a}
        support=set(range(K)) if query_all_states else active_support
        new_ix=[]
        for i in sorted(support):
            loc=np.flatnonzero(block==i)
            if len(loc)<delta:stage_failed=True
            new_ix.extend((position+loc[:delta]).tolist())
        for ix in new_ix:
            i=int(states[ix]); counts[i]+=1;errors[i]+=int(losses[ix])
        position+=length
        if counts.sum()>B:raise AssertionError('pathwise label budget exceeded')
        if stage_failed:break
        n=np.array([counts[i] if i in active_support else 0 for i in range(K)])
        k=np.array([errors[i] if i in active_support else 0 for i in range(K)])
        lo,up=finite_cp_bounds(k,n,j,J=J,K=K,delta_cert=delta_cert,
                  delta_elim=delta_elim,terminal_share=terminal_share)
        # Strict comparison; no negative numerical slack on the acceptance side.
        for a in active:
            cl,_=robust_coverage(a,ql,qu)
            if cl>=gamma and robust_margin(a,up,tau,ql,qu)>=0.:
                if cl>best_cl:
                    best=a;best_cl=cl;cert_at=j
        active=[a for a in active if robust_margin(a,lo,tau,ql,qu,maximize=True)>=0.]
        completed+=1
        if best is not None and (not active or best_cl>=max(robust_coverage(a,ql,qu)[1] for a in active)):
            break
    return UnknownQResult(best,int(counts.sum()),tuple(map(int,counts)),completed,
         stage_failed,tuple(map(float,qhat)),tuple(map(float,ql)),tuple(map(float,qu)),
         pilot_n,max(0.,best_cl),tuple(blocks),cert_at)
