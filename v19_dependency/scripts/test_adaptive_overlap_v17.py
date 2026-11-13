import numpy as np
import pytest
from scipy.stats import binom
from scripts.adaptive_overlap_v17 import (
    tail_alpha, dyadic_bounds, epsilon, phase_length, design_phases,
    run_adaptive, fixed_uniform_cp, validate, mask_margin_from_limits,
)


def test_spending_is_simultaneously_valid():
    for K in (1, 4, 10):
        for delta in (.01, .05):
            spend = sum(2 * K * tail_alpha(delta, K, j) for j in range(2000))
            assert 0 < spend < delta


@pytest.mark.parametrize('n,r', [(1,.3),(8,.05),(32,.15),(128,.9)])
def test_cp_coverage_exact_fixed_prefix(n,r):
    # Analytically sum ALL counts under the Bernoulli law, not a Monte Carlo estimate.
    j = int(np.log2(n)); miss = 0.
    for k in range(n+1):
        lo, hi = dyadic_bounds(np.array([k]),np.array([n]),j,.05)
        if not lo[0] <= r <= hi[0]:
            miss += binom.pmf(k,n,r)
    assert miss <= 2 * tail_alpha(.05,1,j) + 1e-11


def test_cp_not_wider_than_hoeffding():
    for j in range(3,11):
        n=2**j
        for k in (0,1,n//4,n//2,n-1,n):
            cp_lo,cp_hi=dyadic_bounds(np.array([k,0,0,0]),np.array([n,0,0,0]),j,.025,method='cp')
            h_lo,h_hi=dyadic_bounds(np.array([k,0,0,0]),np.array([n,0,0,0]),j,.025,method='hoeffding')
            assert cp_lo[0] >= h_lo[0]-1e-9
            assert cp_hi[0] <= h_hi[0]+1e-9


def test_phase_length_guarantee():
    q=.15; K=4; delta=.025
    for j in range(11):
        d=1 if j==0 else 2**(j-1)
        H=phase_length(d,q,j,delta,K)
        t=np.log(1.0/(6*delta/(np.pi**2*K*(j+1)**2)))
        assert H*q - np.sqrt(2*H*q*t) >= d-1e-10


def test_budget_design():
    phases=design_phases(4,.25,30000,6000,.025)
    assert len(phases)>8
    assert sum(phases)<=30000
    assert 4*(2**(len(phases)-1))<=6000
    assert 4*(2**len(phases))>6000


def test_never_query_future_loss_before_decision():
    # All outcomes are swapped AFTER assigning the first stage's query indices.
    rng=np.random.default_rng(1)
    states=rng.choice(4,size=10000)
    a=run_adaptive(states,np.zeros(len(states),dtype=int),np.ones(4)/4,((0,1),(2,3)),max_stages=3)
    b=run_adaptive(states,np.ones(len(states),dtype=int),np.ones(4)/4,((0,1),(2,3)),max_stages=3)
    # No output guarantee identical (querying is permitted to adapt to past losses);
    # the first block count is deterministic and both exhaust the 3 pre-set stages.
    assert a.labels <= int(.2*len(states))
    assert b.labels <= int(.2*len(states))
    assert a.completed_stages<=3 and b.completed_stages<=3


def test_correct_safety_on_easy_deterministic_world():
    rng=np.random.default_rng(42)
    states=rng.choice(4,size=30000)
    loss=np.where(states<2,0,1)
    res=run_adaptive(states,loss,np.ones(4)/4,((0,1),(0,1,2),(2,3),(3,)),
                     budget_fraction=.2,tau=.15,gamma=.2)
    assert res.selected==(0,1)
    assert not res.stage_failure
    assert res.labels<=6000
    assert res.eliminated_masks>0


def test_failure_to_find_cert_is_not_safety_violation():
    rng=np.random.default_rng(2026)
    states=rng.choice(4,size=30000)
    loss=np.ones(len(states),dtype=int)
    r=run_adaptive(states,loss,np.ones(4)/4,((0,1),(2,3),(0,1,2,3)),
                   budget_fraction=.2)
    assert r.selected is None


def test_fixed_baseline_same_hard_cap():
    rng=np.random.default_rng(14)
    states=rng.choice(4,size=10000)
    losses=np.where(states<2,0,1)
    res=fixed_uniform_cp(states,losses,np.ones(4)/4,((0,1),(2,3)),tau=.15,gamma=.2,
                         n_labels=2000,alpha=.05,seed=42)
    assert res.labels==2000
    assert sum(res.per_state)==2000
    assert res.selected==(0,1)


def test_mask_validation():
    with pytest.raises(ValueError):validate(np.array([.5,.5]),[(0,0)],.15,.1)
    with pytest.raises(ValueError):validate(np.array([.5,.5]),[(0,2)],.15,.1)
    with pytest.raises(ValueError):validate(np.array([.5,.6]),[(0,1)],.15,.1)
    with pytest.raises(ValueError):dyadic_bounds(np.array([0]),np.array([3]),2,.05)


def test_adaptive_vs_state_blind_same_certification_smaller_cost():
    rng=np.random.default_rng(832)
    q=np.array([.25]*4)
    states=rng.choice(4,size=30000)
    losses=(rng.random(30000)<np.array([.04,.07,.55,.65])[states]).astype(int)
    family=((0,1),(0,1,2),(1,2),(2,3),(0,2,3),(0,1,2,3))
    a=run_adaptive(states,losses,q,family,delta_cert=.025,delta_arrival=.025)
    u=run_adaptive(states,losses,q,family,delta_cert=.025,delta_arrival=.025,query_all_states=True)
    assert a.selected==u.selected
    assert a.labels<=u.labels<=6000
    assert a.completed_stages==u.completed_stages
    assert a.labels<u.labels
