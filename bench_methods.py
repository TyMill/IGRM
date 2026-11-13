"""Independent safe sequential benchmark for IGRM v1.9.

Models: pilot q confidence box, i.i.d. Bernoulli losses conditional on states,
mask-complete queries while active. E-processes are PRIOR WORK, not proposed
novelty. Source: fixed-mixture likelihood ratios and predictable plug-in bets.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'v19_dependency'))
from math import floor,log,sqrt
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scripts.mask_pooled_v19 import (MaskPooledResult,mask_terminal_targets,
    mask_checkpoints,cp_one_sided)
from scripts.unknown_q_v18 import (pilot_q_box,robust_coverage,
    linear_box_simplex_extreme)
from scripts.adaptive_overlap_v17 import validate

# Fixed before the held-out benchmark; no dependence on true risks.
ALTERNATIVES = np.array([.04,.08,.11,.125,.137,.145,.149],dtype=float)
WEIGHTS = np.array([.05,.08,.12,.18,.22,.21,.14],dtype=float)
assert np.isclose(WEIGHTS.sum(),1.0)


def evidence_one_mask(losses, tau=.15, mode='mixture'):
    """E-values at all prefixes for a Bernoulli H0:p >= tau.

    Under H0, each fixed p_alt<tau LR is a supermartingale. Fixed mixtures
    and predictable p_alt,t based only on past mask labels preserve this.
    """
    if not 0<tau<1 or mode not in ('mixture','betting'):raise ValueError('inputs')
    alts=ALTERNATIVES*(tau/.15)
    if mode=='mixture':
        logs=np.log(WEIGHTS)
        out=[]
        for y in losses:
            logs=logs+(np.log(alts/tau) if y else np.log((1-alts)/(1-tau)))
            out.append(float(logsumexp(logs)))
        return out
    z=0.;k=0;n=0;out=[]
    for y in losses:
        # Predictable beta(1,1) estimate. Every bet is safe at or above tau.
        alt=min(tau, (k+1)/(n+2))
        if y: z+=log(max(alt,1e-12)/tau)
        else: z+=log((1-alt)/(1-tau))
        k+=int(y);n+=1;out.append(z)
    return out


def run_mask_eprocess(states,hidden_losses,K,family,*,mode='mixture',
                      pilot_n=4096,tau=.15,gamma=.25,
                      budget_fraction=.2,delta_q=.025,delta_cert=.025,
                      delta_elim=.025,target_fraction=.9,max_labels=4096):
    """Drop-in counterpart of mask-pooled CP with no alpha spending for certification.

    Query every arrival in active mask support, so every mask's queried
    prefix is the prefix of its own iid Bernoulli conditional loss sequence.
    CP lower elimination uses the unchanged v1.9 predetermined checkpoints.
    """
    s=np.asarray(states,dtype=int); y=np.asarray(hidden_losses,dtype=np.int8)
    if s.ndim!=1 or y.shape!=s.shape or np.any(s<0) or np.any(s>=K) or not np.isin(y,[0,1]).all():
        raise ValueError('invalid stream')
    if not(0<pilot_n<len(s) and 0<delta_q<1 and 0<delta_cert<1
           and 0<delta_elim<1 and 0<budget_fraction<=1):raise ValueError('parameters')
    if mode not in ('mixture','betting'):raise ValueError('mode')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    _,ql,qu=pilot_q_box(s[:pilot_n],K,delta_q)
    B=min(int(floor(budget_fraction*len(s))),int(max_labels))
    coverage={a:robust_coverage(a,ql,qu) for a in fam}
    active={a for a in fam if coverage[a][0]>=gamma}
    targets=mask_terminal_targets(fam,ql,qu,B,target_fraction)
    checkpts={a:mask_checkpoints(targets[a]) for a in active}
    lower_spending={a:(np.arange(1,len(checkpts[a])+1,dtype=float)**-2) for a in active}
    for a in active:lower_spending[a]/=lower_spending[a].sum()
    M=len(fam)
    next_checkpoint={a:0 for a in active}
    mask_n={a:0 for a in fam};mask_k={a:0 for a in fam}
    per_state=np.zeros(K,dtype=int)
    # for mixture: log E(a) = log(sum weights*LRs). For betting: one wealth scalar.
    logwealth={a:WEIGHTS.copy() if mode=='mixture' else 0. for a in active}
    alts=ALTERNATIVES*(tau/.15)
    factors_0=(1-alts)/(1-tau); factors_1=alts/tau
    threshold=log(M/delta_cert);threshold_e=M/delta_cert
    nquery=0; best=None;best_cl=-1.;best_at=None;eliminated=0;exhausted=0
    for t in range(pilot_n,len(s)):
        if not active or nquery>=B:break
        support={i for a in active for i in a}
        i=int(s[t])
        if i not in support:continue
        label=int(y[t]);nquery+=1;per_state[i]+=1
        for a in tuple(active):
            if i not in a:continue
            n=mask_n[a];k=mask_k[a]
            if mode=='mixture':
                # Wealth is bounded by the stopping threshold times 1/(1-tau),
                # so direct arithmetic is stable before certification.
                logwealth[a]*=(factors_1 if label else factors_0)
                log_e=log(float(logwealth[a].sum())) if logwealth[a].sum()>0 else -float('inf')
            else:
                alt=min(tau,(k+1)/(n+2))
                inc=log(max(alt,1e-12)/tau) if label else log((1-alt)/(1-tau))
                logwealth[a]+=inc
                log_e=logwealth[a]
            mask_n[a]+=1;mask_k[a]+=label
            if log_e>=threshold:
                cl=coverage[a][0]
                if cl>best_cl:
                    best=a;best_cl=cl;best_at=nquery
                active.remove(a)
                continue
            j=next_checkpoint[a]
            if mask_n[a]!=checkpts[a][j]:continue
            lo,_=cp_one_sided(mask_k[a],mask_n[a],delta_elim*lower_spending[a][j]/M)
            if lo>tau:
                eliminated+=1;active.remove(a)
            elif j+1>=len(checkpts[a]):
                exhausted+=1;active.remove(a)
            else:next_checkpoint[a]+=1
        if best is not None and (not active or best_cl>=max(coverage[a][1] for a in active)):
            break
    return MaskPooledResult(best,nquery,tuple(map(int,per_state)),
        tuple(mask_n[a] for a in fam),pilot_n,max(0.,best_cl),eliminated,
        exhausted,best_at,nquery>=B)


def fixed_joint_weighted_hoeffding(states,hidden_losses,K,family,*,
                                  pilot_n=4096,n_labels=4096,tau=.15,gamma=.25,
                                  delta_q=.025,delta_cert=.025,seed=0):
    """Safe FIXED joint (non-coordinatewise) stratified Hoeffding comparator.

    Conditional on the randomly chosen state counts, labels in each state are
    independent Bernoulli; for any fixed q in pilot box, simultaneous
    weighted deviations are bounded by sqrt(log(M/dc)*sum(q_i^2/n_i)/2).
    Supremum linear bias over q and upper bound on variance term make it safe.
    """
    s=np.asarray(states,dtype=int);y=np.asarray(hidden_losses,dtype=np.int8)
    if s.ndim!=1 or y.shape!=s.shape or np.any(s<0) or np.any(s>=K) or not np.isin(y,[0,1]).all():
        raise ValueError('invalid stream')
    if not(0<pilot_n<len(s) and 0<=n_labels<=len(s)-pilot_n):raise ValueError('inputs')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    _,ql,qu=pilot_q_box(s[:pilot_n],K,delta_q)
    rng=np.random.default_rng(seed)
    ix=rng.choice(np.arange(pilot_n,len(s)),size=n_labels,replace=False)
    ns=np.bincount(s[ix],minlength=K)
    ks=np.bincount(s[ix],weights=y[ix],minlength=K)
    rh=np.divide(ks,ns,out=np.zeros(K),where=ns>0)
    best=None;best_cl=-1.;per_mask=[]
    for a in fam:
        lcover,_=robust_coverage(a,ql,qu)
        per_mask.append(int(ns[list(a)].sum()))
        if lcover<gamma or np.any(ns[list(a)]==0):continue
        d=np.zeros(K);d[list(a)]=rh[list(a)]-tau
        worst=linear_box_simplex_extreme(d,ql,qu,maximize=True)
        radius=sqrt(.5*log(len(fam)/delta_cert)*sum(qu[i]**2/ns[i] for i in a))
        if worst+radius<=0 and lcover>best_cl:
            best=a;best_cl=lcover
    return MaskPooledResult(best,n_labels,tuple(map(int,ns)),tuple(per_mask),pilot_n,
                            max(0.,best_cl),0,0,n_labels if best else None,True)
