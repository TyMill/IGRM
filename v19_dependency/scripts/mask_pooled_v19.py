"""Direct mask-conditional risk CP with shareable labels and unknown q.

KEY ASSUMPTION: while mask A is active, query *every* arrival S in A.
Querying only a nonuniform subset within A invalidates the iid Bernoulli(R_A)
mask-conditional samples. Querying all states in the union of active masks
preserves this property despite predictable mask elimination.

The finite mask-wise checkpoints and their error allocations are chosen using
only an independent unlabeled pilot, never queried labels. This is a known
union-bound/fixed-checkpoint construction, NOT novel general CS theory.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import floor
from typing import Sequence
import numpy as np
from scipy.stats import beta
from scripts.adaptive_overlap_v17 import validate
from scripts.unknown_q_v18 import (pilot_q_box,robust_coverage)
from scripts.finite_horizon_cp_v19 import terminal_weights

@dataclass(frozen=True)
class MaskPooledResult:
    selected: tuple[int,...]|None
    labels: int
    per_state: tuple[int,...]
    per_mask: tuple[int,...]
    pilot_n: int
    lower_coverage: float
    eliminated_masks: int
    exhausted_masks: int
    certified_at_labels: int|None
    stopped_on_budget: bool


def mask_checkpoints(target: int) -> tuple[int,...]:
    if not isinstance(target,int) or target<1:raise ValueError('positive target required')
    checkpoints=[]
    step=16
    while step<target:
        checkpoints.append(step)
        step*=2
    if not checkpoints or checkpoints[-1]!=target:
        checkpoints.append(target)
    return tuple(checkpoints)


def mask_terminal_targets(family:Sequence[tuple[int,...]],ql:np.ndarray,qu:np.ndarray,
                          B:int, fraction:float=.9)->dict:
    if not 0 < fraction <=1 or B<1:raise ValueError('invalid budget/target fraction')
    out={}
    for a in family:
        cl,_=robust_coverage(a,ql,qu)
        n=max(1,min(B,int(floor(fraction*B*cl))))
        out[a]=n
    return out


def cp_one_sided(k:int,n:int,a:float)->tuple[float,float]:
    if not (0<a<1 and 0<=k<=n and n>=1):raise ValueError('invalid CP inputs')
    lo=0. if k==0 else float(beta.ppf(a,k,n-k+1))
    up=1. if k==n else float(beta.ppf(1-a,k+1,n-k))
    return lo,up


def run_mask_pooled(states:np.ndarray,hidden_losses:np.ndarray,K:int,
                    family:Sequence[tuple[int,...]],*,pilot_n:int=4096,
                    tau:float=.15,gamma:float=.25,budget_fraction:float=.2,
                    delta_q:float=.025,delta_cert:float=.025,
                    delta_elim:float=.025,terminal_share:float=.8,
                    target_fraction:float=.9,query_all_states:bool=False,
                    max_labels:int|None=4096)->MaskPooledResult:
    s=np.asarray(states,dtype=int);loss=np.asarray(hidden_losses,dtype=np.int8)
    if s.ndim!=1 or loss.shape!=s.shape or not np.isin(loss,[0,1]).all():
        raise ValueError('invalid stream')
    if not (K>=1 and 0<pilot_n<len(s) and 0<budget_fraction<=1):
        raise ValueError('invalid pilot/K/budget')
    if np.any(s<0) or np.any(s>=K) or not (0<delta_q<1 and 0<delta_cert<1 and 0<delta_elim<1):
        raise ValueError('invalid state/alpha')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    _,ql,qu=pilot_q_box(s[:pilot_n],K,delta_q)
    B=int(floor(budget_fraction*len(s)))
    if max_labels is not None:
        if max_labels<1:raise ValueError('invalid max_labels')
        B=min(B,max_labels)
    coverage={a:robust_coverage(a,ql,qu) for a in fam}
    active={a for a in fam if coverage[a][0]>=gamma}
    targets=mask_terminal_targets(fam,ql,qu,B,target_fraction)
    checkpts={a:mask_checkpoints(targets[a]) for a in active}
    upper_spending={a:terminal_weights(len(checkpts[a]),terminal_share) for a in active}
    # Separate lower-bound budget. Earlier checkpoints receive nonzero weight.
    lower_spending={a:(np.arange(1,len(checkpts[a])+1,dtype=float)**-2) for a in active}
    for a in active:lower_spending[a]/=lower_spending[a].sum()
    M=len(fam)
    next_checkpoint={a:0 for a in active}
    mask_n={a:0 for a in fam};mask_k={a:0 for a in fam}
    per_state=np.zeros(K,dtype=int)
    nquery=0;best=None;best_cl=-1.;best_at=None;eliminated=0;exhausted=0
    for t in range(pilot_n,len(s)):
        if not active or nquery>=B:break
        # All states in each ACTIVE mask are queried. This uniform-within-mask
        # acquisition rule is necessary for direct Bernoulli(R_A) inference.
        active_support={i for a in active for i in a}
        i=int(s[t])
        if not query_all_states and i not in active_support:
            continue
        # Query decision has now been made, using ONLY i and past labels.
        y=int(loss[t]);nquery+=1;per_state[i]+=1
        for a in list(active):
            if i not in a:continue
            mask_n[a]+=1;mask_k[a]+=y
            checkpoint_id=next_checkpoint[a]
            if mask_n[a] != checkpts[a][checkpoint_id]:continue
            n=mask_n[a];k=mask_k[a]
            alpha_u=delta_cert*upper_spending[a][checkpoint_id]/M
            alpha_l=delta_elim*lower_spending[a][checkpoint_id]/M
            _,up=cp_one_sided(k,n,alpha_u)
            lo,_=cp_one_sided(k,n,alpha_l)
            if up<=tau:
                cl=coverage[a][0]
                if cl>best_cl:
                    best=a;best_cl=cl;best_at=nquery
                active.remove(a)
            elif lo>tau:
                eliminated+=1;active.remove(a)
            elif checkpoint_id+1>=len(checkpts[a]):
                exhausted+=1;active.remove(a)
            else:
                next_checkpoint[a]+=1
        if best is not None and (not active or best_cl>=max(coverage[a][1] for a in active)):
            break
    return MaskPooledResult(best,nquery,tuple(map(int,per_state)),
        tuple(mask_n[a] for a in fam),pilot_n,max(0.,best_cl),eliminated,exhausted,
        best_at,nquery>=B)


def fixed_mask_pooled(states:np.ndarray,loss:np.ndarray,K:int,
                  family:Sequence[tuple[int,...]],*,pilot_n:int=4096,
                  n_labels:int=4096,tau:float=.15,gamma:float=.25,
                  delta_q:float=.025,delta_cert:float=.025,seed:int=0)->MaskPooledResult:
    s=np.asarray(states,int);loss=np.asarray(loss,np.int8)
    if s.ndim!=1 or loss.shape!=s.shape or not np.isin(loss,[0,1]).all() or np.any(s<0) or np.any(s>=K):
        raise ValueError('invalid stream')
    if not(0<pilot_n<len(s) and 0<=n_labels<=len(s)-pilot_n and 0<delta_cert<1):
        raise ValueError('invalid budget')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    _,ql,qu=pilot_q_box(s[:pilot_n],K,delta_q)
    rng=np.random.default_rng(seed)
    ix=rng.choice(np.arange(pilot_n,len(s)),size=n_labels,replace=False)
    sub_s=s[ix];sub_l=loss[ix];K=len(ql)
    cover={a:robust_coverage(a,ql,qu) for a in fam}
    best=None;best_cl=-1.;per_mask=[]
    for a in fam:
        mask=np.isin(sub_s,a)
        n=int(mask.sum());k=int(sub_l[mask].sum());per_mask.append(n)
        if n==0 or cover[a][0]<gamma:continue
        _,up=cp_one_sided(k,n,delta_cert/len(fam))
        if up<=tau and cover[a][0]>best_cl:
            best=a;best_cl=cover[a][0]
    return MaskPooledResult(best,n_labels,tuple(np.bincount(sub_s,minlength=K).astype(int)),
            tuple(per_mask),pilot_n,max(0.,best_cl),0,0,
            n_labels if best is not None else None,n_labels>0)
