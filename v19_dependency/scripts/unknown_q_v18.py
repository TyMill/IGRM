"""Pilot-estimated target state law + sequential shared-label risk certificates.

No true-q argument is accepted by the procedure. The first n_pilot states are
observed without querying their labels. Binomial marginal CP intervals are
jointly valid by Bonferroni; subsequent label-dependent selection maximizes
worst-case population coverage over their simplex/box intersection.

Stationary independent categorical states; conditional Bernoulli labels;
predictable queries; deterministic epoch H and pilot size.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import floor
from typing import Sequence
import numpy as np
from scipy.stats import beta
from scripts.adaptive_overlap_v17 import (AdaptiveResult, validate, dyadic_bounds,
    design_phases, phase_length)


@dataclass(frozen=True)
class UnknownQResult:
    selected: tuple[int,...] | None
    labels: int
    per_state: tuple[int,...]
    completed_stages: int
    stage_failure: bool
    qhat: tuple[float,...]
    qlower: tuple[float,...]
    qupper: tuple[float,...]
    pilot_n: int
    certified_lower_coverage: float
    phase_lengths: tuple[int,...]
    certified_at: int | None


def pilot_q_box(states: np.ndarray, K: int, delta_q: float):
    """Simultaneous 1-d marginal Clopper--Pearson intervals for iid states."""
    if states.ndim != 1 or len(states) < 1 or K < 1 or not 0 < delta_q < 1:
        raise ValueError('invalid pilot settings')
    if (states < 0).any() or (states >= K).any():
        raise ValueError('invalid state index')
    n = len(states)
    k = np.bincount(states,minlength=K)
    a = delta_q/(2*K)
    l = np.array([0. if ki==0 else beta.ppf(a,ki,n-ki+1) for ki in k],dtype=float)
    u = np.array([1. if ki==n else beta.ppf(1-a,ki+1,n-ki) for ki in k],dtype=float)
    qhat=k/n
    # Confidence box intersects simplex (qhat belongs to it).
    assert np.all(l <= qhat+1e-13) and np.all(u >= qhat-1e-13)
    return qhat,l,u


def linear_box_simplex_extreme(coeff: np.ndarray, l: np.ndarray, u: np.ndarray,
                               *, maximize: bool = False) -> float:
    """Exact greedy solution of a linear function on {l<=q<=u,sum(q)=1}."""
    coeff=np.asarray(coeff,float);l=np.asarray(l,float);u=np.asarray(u,float)
    if coeff.shape!=l.shape or coeff.shape!=u.shape or coeff.ndim!=1:
        raise ValueError('dimension mismatch')
    if not np.isfinite(coeff).all() or (l<0).any() or (l>u).any() or (u>1).any():
        raise ValueError('invalid simplex box')
    remaining=1.-float(l.sum())
    if remaining < -1e-9 or remaining > float((u-l).sum())+1e-9:
        raise ValueError('empty simplex box')
    remaining=max(0.,remaining)
    q=l.copy()
    for i in np.argsort(-coeff if maximize else coeff,kind='stable'):
        add=min(float(u[i]-l[i]),remaining)
        q[i]+=add;remaining-=add
        if remaining <= 1e-12:
            break
    if remaining>1e-8:
        raise ValueError('unallocated mass')
    return float(coeff@q)


def robust_coverage(a: tuple[int,...],l: np.ndarray,u: np.ndarray):
    c=np.zeros(len(l));c[list(a)]=1.
    return (linear_box_simplex_extreme(c,l,u),
            linear_box_simplex_extreme(c,l,u,maximize=True))


def robust_margin(a: tuple[int,...], rbound: np.ndarray, tau: float,
                  l: np.ndarray,u: np.ndarray, *, maximize: bool=False):
    c=np.zeros(len(l)); c[list(a)]=tau-rbound[list(a)]
    return linear_box_simplex_extreme(c,l,u,maximize=maximize)


def run_unknown_q(states: np.ndarray, hidden_losses: np.ndarray, K: int,
                  family: Sequence[tuple[int,...]], *, pilot_n: int=4096,
                  tau: float=.15, gamma: float=.25,budget_fraction: float=.2,
                  delta_q: float=.025,delta_cert: float=.025,
                  delta_arrival: float=.025,max_stages: int|None=None,
                  query_all_states: bool=False) -> UnknownQResult:
    """Two-source certificate: unlabeled q pilot, dyadic Bernoulli CP for r.

    On E_q ∩ E_r, any certified mask has true risk <=tau and coverage >=gamma.
    Unconditionally, false-certification probability <=delta_q+delta_cert.
    Available states after pilot are independent of the pilot box.
    """
    states=np.asarray(states,int); losses=np.asarray(hidden_losses,np.int8)
    if states.ndim!=1 or losses.shape!=states.shape or not np.isin(losses,[0,1]).all():
        raise ValueError('invalid stream')
    if not (K>=1 and pilot_n>=1 and pilot_n<len(states)):
        raise ValueError('invalid K/pilot length')
    if (states<0).any() or (states>=K).any() or not (0<budget_fraction<=1):
        raise ValueError('invalid states/budget')
    if not(0<delta_q<1 and 0<delta_cert<1 and 0<delta_arrival<1):
        raise ValueError('invalid error budget')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    qhat,l,u=pilot_q_box(states[:pilot_n],K,delta_q)
    support=set(range(K)) if query_all_states else set(i for a in fam for i in a)
    if not support or min(l[list(support)])<=0:
        blocks=()
    else:
        # Stage design uses a LOWER confidence bound on state arrival rates.
        # Number of requested labels is bounded by |support|2^J, not K2^J.
        blocks=design_phases(len(support),float(min(l[list(support)])),
                             len(states)-pilot_n,int(floor(budget_fraction*len(states))),delta_arrival)
    if max_stages is not None:blocks=blocks[:max_stages]
    counts=np.zeros(K,dtype=int); errors=np.zeros(K,dtype=int)
    active=list(fam);best=None;best_lower=-1.;cert_at=None
    pos=pilot_n; completed=0;fail=False
    for j,length in enumerate(blocks):
        if not active:break
        step=1 if j==0 else 2**(j-1)
        current=states[pos:pos+length]
        active_support=set(i for a in active for i in a)
        support_j=set(range(K)) if query_all_states else active_support
        new_ix=[]
        for i in sorted(support_j):
            ix=np.flatnonzero(current==i)
            if len(ix)<step:fail=True
            new_ix.extend((pos+ix[:step]).tolist())
        for ix in new_ix:
            i=int(states[ix]);counts[i]+=1;errors[i]+=int(losses[ix])
        pos+=length
        assert int(counts.sum())<=floor(budget_fraction*len(states))
        if fail:break
        n=np.array([counts[i] if i in active_support else 0 for i in range(K)])
        k=np.array([errors[i] if i in active_support else 0 for i in range(K)])
        lo,up=dyadic_bounds(k,n,j,delta_cert,method='cp')
        for a in active:
            cl,_=robust_coverage(a,l,u)
            if cl>=gamma and robust_margin(a,up,tau,l,u)>=0:
                if cl>best_lower:
                    best=a;best_lower=cl;cert_at=j
        active=[a for a in active if robust_margin(a,lo,tau,l,u,maximize=True)>=0]
        completed+=1
        if best is not None and (not active or best_lower>=max(robust_coverage(a,l,u)[1] for a in active)):
            break
    return UnknownQResult(best,int(counts.sum()),tuple(int(x) for x in counts),
                          completed,fail,tuple(map(float,qhat)),tuple(map(float,l)),tuple(map(float,u)),
                          pilot_n,max(0.,best_lower),tuple(blocks),cert_at)


def fixed_unknown_q_cp(states: np.ndarray, hidden_losses: np.ndarray,
                       K: int, family: Sequence[tuple[int,...]], *,
                       pilot_n: int, n_labels: int, tau: float=.15,
                       gamma: float=.25,delta_q: float=.025,
                       delta_cert: float=.025,seed: int=0) -> UnknownQResult:
    """Strong comparator with the SAME unknown q pilot and a fresh fixed CP sample.

    Uses one-sided Bonferroni CP intervals for conditional errors; selection
    shares the state-specific intervals across all masks. Safety <=dq+dc.
    """
    states=np.asarray(states,int);loss=np.asarray(hidden_losses,np.int8)
    if states.ndim!=1 or states.shape!=loss.shape or not np.isin(loss,[0,1]).all():
        raise ValueError('invalid stream')
    if not(0<pilot_n<len(states) and 0<=n_labels<=len(states)-pilot_n):
        raise ValueError('invalid pilot/budget')
    if (states<0).any() or (states>=K).any() or not(0<delta_cert<1):
        raise ValueError('invalid q/certification')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    qhat,l,u=pilot_q_box(states[:pilot_n],K,delta_q)
    rng=np.random.default_rng(seed)
    ix=rng.choice(np.arange(pilot_n,len(states)),size=n_labels,replace=False)
    count=np.bincount(states[ix],minlength=K)
    err=np.bincount(states[ix],weights=loss[ix],minlength=K).astype(int)
    upper=np.ones(K)
    for i in range(K):
        ni=int(count[i]);ki=int(err[i])
        if ni and ki<ni:
            upper[i]=float(beta.ppf(1-delta_cert/K,ki+1,ni-ki))
    candidates=[]
    for a in fam:
        cl,_=robust_coverage(a,l,u)
        if cl>=gamma and robust_margin(a,upper,tau,l,u)>=0:
            candidates.append((cl,a))
    if candidates:
        cl,A=max(candidates,key=lambda x:(x[0],tuple(-i for i in x[1])))
    else:
        cl,A=0.,None
    return UnknownQResult(A,int(n_labels),tuple(int(x) for x in count),1,False,
                          tuple(map(float,qhat)),tuple(map(float,l)),tuple(map(float,u)),
                          pilot_n,float(cl),(),0 if A else None)
