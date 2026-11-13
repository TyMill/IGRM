import numpy as np
import pytest
from scipy.optimize import linprog
from scripts.unknown_q_v18 import (pilot_q_box,linear_box_simplex_extreme, robust_coverage,
    robust_margin,run_unknown_q)


def test_box_contains_empirical_mass():
    st=np.repeat([0,1,2,3],[1000,2000,3000,4000])
    h,l,u=pilot_q_box(st,4,.05)
    assert np.all(l<=h) and np.all(h<=u)
    assert abs(h.sum()-1)<1e-14

@pytest.mark.parametrize('seed',range(12))
def test_exact_box_lp(seed):
    rng=np.random.default_rng(seed)
    K=4;q=rng.dirichlet(np.ones(K))
    l=np.maximum(0,q-rng.uniform(.02,.14,K))
    u=np.minimum(1,q+rng.uniform(.02,.14,K))
    c=rng.normal(size=K)
    o=linprog(c,A_eq=np.ones((1,K)),b_eq=[1.],bounds=list(zip(l,u)),method='highs')
    assert o.success
    assert abs(linear_box_simplex_extreme(c,l,u)-o.fun)<1e-10
    o=linprog(-c,A_eq=np.ones((1,K)),b_eq=[1.],bounds=list(zip(l,u)),method='highs')
    assert o.success
    assert abs(linear_box_simplex_extreme(c,l,u,maximize=True)+o.fun)<1e-10


def test_robust_mass_bounds_for_true_q():
    q=np.array([.1,.2,.3,.4]); l=q-.01;u=q+.01
    for A in [(0,),(0,1),(1,2,3)]:
        lo,hi=robust_coverage(A,l,u)
        assert lo-1e-12<=q[list(A)].sum()<=hi+1e-12


def test_robust_margin_bounds_for_true_q():
    q=np.array([.1,.2,.3,.4]);l=q-.01;u=q+.01
    true_r=np.array([.02,.10,.3,.9]);upper=np.minimum(1,true_r+.02)
    for A in [(0,1),(0,1,2),(2,3)]:
        ml=robust_margin(A,upper,.15,l,u)
        gt=(q[list(A)]*(.15-true_r[list(A)])).sum()
        assert ml<=gt+1e-12


def test_no_oracle_q_parameter_and_budget():
    rng=np.random.default_rng(17)
    states=rng.choice(4,size=30000,p=[.1,.2,.3,.4]);losses=(rng.random(30000)<np.array([.02,.04,.6,.6])[states]).astype(int)
    out=run_unknown_q(states,losses,4,((0,1),(1,2),(2,3),(0,1,2)),pilot_n=4096)
    assert out.labels<=6000
    assert out.labels==sum(out.per_state)
    assert len(out.qhat)==4
    if out.selected is not None:
        true_q=np.array([.1,.2,.3,.4]);r=np.array([.02,.04,.6,.6])
        A=list(out.selected)
        assert (true_q[A]*r[A]).sum()/true_q[A].sum()<=.15


def test_pilot_never_consumes_labels():
    s=np.tile([0,1],6000);loss=np.zeros_like(s)
    a=run_unknown_q(s,loss,2,((0,),(1,)),pilot_n=4000,max_stages=1)
    assert a.labels<=2
    assert a.pilot_n==4000


def test_unknown_q_detects_invalid():
    with pytest.raises(ValueError):
        run_unknown_q(np.array([0,1,0]),np.array([0,0,0]),2,((0,),),pilot_n=3)


def test_audit_regression_irrelevant_rare_state_no_longer_blocks_phases():
    from scripts.adaptive_overlap_v17 import run_adaptive
    q=np.array([.4999995,.4999995,.000001])
    states=np.tile([0,1],15000)
    losses=np.zeros(len(states),dtype=np.int8)
    result=run_adaptive(states,losses,q,((0,1),),tau=.15,gamma=.25)
    assert result.selected==(0,1)
    assert result.labels>0 and result.labels<=6000


def test_pilot_labels_are_not_inspected():
    rng=np.random.default_rng(123)
    states=rng.choice(3,30000,p=[.2,.3,.5]); loss=np.zeros(len(states),dtype=np.int8)
    alt=loss.copy();alt[:4096]=1
    family=((0,),(0,1),(1,2))
    a=run_unknown_q(states,loss,3,family,pilot_n=4096)
    b=run_unknown_q(states,alt,3,family,pilot_n=4096)
    assert a==b


def test_robust_q_box_rejects_naively_safe_policy_under_uncertain_mass():
    # A pilot favouring the low-risk state can make plug-in risk appear safe.
    # Proper robust certification cannot ignore the high-risk alternatives.
    from scripts.unknown_q_v18 import robust_margin
    states=np.array([0]*14+[1]*6,dtype=int)
    h,l,u=pilot_q_box(states,2,.05)
    r=np.array([.01,.30]);tau=.15
    naive=float(h@(tau-r))
    robust=robust_margin((0,1),r,tau,l,u)
    assert naive>0
    assert robust<0


def test_q_box_width_hoeffding_inversion_bound():
    import math
    rng=np.random.default_rng(222)
    for n in [20,100,1000]:
        s=rng.choice(4,n,p=[.1,.2,.3,.4]);_,l,u=pilot_q_box(s,4,.05)
        eps=math.sqrt(math.log(2*4/.05)/(2*n))
        assert np.all(u-l<=2*eps+1e-10)
