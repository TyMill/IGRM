"""Pre-submission independently implemented comparators for IGRM v1.9.

Two PRIOR-ART constructions (not novel statistical methods):
  1. Fixed-sample exact binomial tests with Holm stepdown for mask family.
  2. Precommitted hybrid: anytime-valid predictable betting for early approval,
     plus a single terminal CP test using the remaining alpha.

Conditions: frozen iid (S,Y); non-anticipating mask-complete labeling while a
mask remains active; pilot for q using no labels; bounded optional stopping.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import floor,log
import numpy as np
from scipy.stats import binom
from bench_methods import MaskPooledResult
from scripts.mask_pooled_v19 import cp_one_sided,mask_checkpoints,mask_terminal_targets
from scripts.adaptive_overlap_v17 import validate
from scripts.unknown_q_v18 import pilot_q_box,robust_coverage


def fixed_mask_holm(states,hidden_losses,K,family,*,pilot_n=4096,
                    n_labels=4096,tau=.15,gamma=.25,delta_q=.025,
                    delta_cert=.025,seed=0):
    """Uniform fixed labels, exact one-sided p-values, Holm FWER correction."""
    s=np.asarray(states,dtype=int);y=np.asarray(hidden_losses,dtype=np.int8)
    if s.ndim!=1 or y.shape!=s.shape or not np.isin(y,[0,1]).all() or np.any(s<0) or np.any(s>=K):
        raise ValueError('invalid stream')
    if not (0<pilot_n<len(s) and 0<n_labels<=len(s)-pilot_n and 0<delta_cert<1 and 0<delta_q<1):
        raise ValueError('invalid parameters')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    _,ql,qu=pilot_q_box(s[:pilot_n],K,delta_q)
    rng=np.random.default_rng(seed)
    ix=rng.choice(np.arange(pilot_n,len(s)),size=n_labels,replace=False)
    ns=np.bincount(s[ix],minlength=K)
    candidates=[]; per_mask=[]
    for mask in fam:
        midx=np.isin(s[ix],mask);n=int(midx.sum()); k=int(y[ix][midx].sum())
        per_mask.append(n)
        cl,_=robust_coverage(mask,ql,qu)
        if cl>=gamma and n>0:
            # Under p >= tau, lower-tail p-value at boundary tau is superuniform.
            candidates.append((float(binom.cdf(k,n,tau)),mask,cl))
    candidates.sort(key=lambda z:z[0])
    certified=[]
    for rank,(pv,mask,cl) in enumerate(candidates):
        if pv <= delta_cert/(len(candidates)-rank):
            certified.append((cl,mask))
        else:
            break
    best=None;best_cl=0.
    if certified:
        best_cl,best=max(certified,key=lambda z:(z[0],-len(z[1])))
    return MaskPooledResult(best,int(n_labels),tuple(np.bincount(s[ix],minlength=K).astype(int)),
        tuple(per_mask),pilot_n,float(best_cl),0,0,
        n_labels if best else None,True)


def run_hybrid_betting_terminal_cp(states,hidden_losses,K,family,*,
                   pilot_n=4096,tau=.15,gamma=.25,
                   budget_fraction=.2,delta_q=.025,delta_cert=.025,
                   delta_elim=.025,early_share=.1,target_fraction=.9,max_labels=4096):
    """Anytime predictable betting with strictly precommitted CP endpoint.

    Bonferroni FWER: per-mask early betting error <= delta_cert*early_share/M
    by Ville and terminal CP error <= delta_cert*(1-early_share)/M, so their
    union across masks plus pilot q error <= delta_cert+delta_q.
    Elimination uses distinct lower CP budgets and affects only power.
    """
    s=np.asarray(states,dtype=int); y=np.asarray(hidden_losses,dtype=np.int8)
    if s.ndim!=1 or y.shape!=s.shape or not np.isin(y,[0,1]).all() or np.any(s<0) or np.any(s>=K):
        raise ValueError('invalid stream')
    if not (0<pilot_n<len(s) and 0<delta_q<1 and 0<delta_cert<1 and
            0<delta_elim<1 and 0<budget_fraction<=1 and 0<early_share<1):
        raise ValueError('invalid inputs')
    fam=validate(np.ones(K)/K,family,tau,gamma)
    _,ql,qu=pilot_q_box(s[:pilot_n],K,delta_q)
    B=min(int(floor(budget_fraction*len(s))),int(max_labels))
    cover={a:robust_coverage(a,ql,qu) for a in fam}
    active={a for a in fam if cover[a][0]>=gamma}
    targets=mask_terminal_targets(fam,ql,qu,B,target_fraction)
    chk={a:mask_checkpoints(targets[a]) for a in active}
    lower_w={a:np.arange(1,len(chk[a])+1,dtype=float)**-2 for a in active}
    for a in active:lower_w[a]/=lower_w[a].sum()
    next_j={a:0 for a in active}
    M=len(fam); log_threshold=log(M/(delta_cert*early_share))
    per_state=np.zeros(K,dtype=int)
    mask_n={a:0 for a in fam}; mask_k={a:0 for a in fam}
    logwealth={a:0. for a in active}
    nq=0;best=None;best_cl=-1.;best_at=None;eliminated=0;exhausted=0
    for t in range(pilot_n,len(s)):
        if not active or nq>=B:break
        sup={i for a in active for i in a}
        st=int(s[t])
        if st not in sup:continue
        yy=int(y[t]);nq+=1;per_state[st]+=1
        for a in tuple(active):
            if st not in a:continue
            n=mask_n[a];k=mask_k[a]
            # Predictable alternative calculated BEFORE current label, always <=tau.
            alt=min(tau,(k+1)/(n+2))
            if yy:
                logwealth[a]+=log(max(alt,1e-12)/tau)
            else:
                logwealth[a]+=log((1-alt)/(1-tau))
            mask_n[a]+=1;mask_k[a]+=yy
            if logwealth[a]>=log_threshold:
                cl=cover[a][0]
                if cl>best_cl:
                    best=a;best_cl=cl;best_at=nq
                active.remove(a)
                continue
            j=next_j[a]
            if mask_n[a]!=chk[a][j]:continue
            # CP at ONLY the precommitted terminal checkpoint, not interim.
            if j+1==len(chk[a]):
                _,up=cp_one_sided(mask_k[a],mask_n[a],
                     delta_cert*(1-early_share)/M)
                if up<=tau:
                    cl=cover[a][0]
                    if cl>best_cl:
                        best=a;best_cl=cl;best_at=nq
                    active.remove(a);continue
            lo,_=cp_one_sided(mask_k[a],mask_n[a],delta_elim*lower_w[a][j]/M)
            if lo>tau:
                eliminated+=1;active.remove(a)
            elif j+1>=len(chk[a]):
                exhausted+=1;active.remove(a)
            else:
                next_j[a]+=1
        if best is not None and (not active or best_cl>=max(cover[a][1] for a in active)):
            break
    return MaskPooledResult(best,nq,tuple(map(int,per_state)),
        tuple(mask_n[a] for a in fam),pilot_n,max(0.,best_cl),
        eliminated,exhausted,best_at,nq>=B)
