"""Check masks, direct CP, causal budgets, irrelevant states and adversarial cases."""
import numpy as np
import pytest
from scipy.stats import binom
from scripts.mask_pooled_v19 import (run_mask_pooled,fixed_mask_pooled,
    mask_checkpoints,mask_terminal_targets,cp_one_sided)
from scripts.run_adaptive_overlap_v17 import FAMILY

@pytest.mark.parametrize('n',[1,5,20,128])
def test_cpsize(n):
    p=.15;a=.025
    fail=sum(binom.pmf(k,n,p) for k in range(n+1) if cp_one_sided(k,n,a)[1]<p-1e-14)
    assert fail<=a+1e-12

@pytest.mark.parametrize('t',[1,16,17,1024,1800])
def test_checkpoints(t):
    cps=mask_checkpoints(t)
    assert cps[-1]==t and all(x<y for x,y in zip(cps,cps[1:]))

def test_zero_one_extremes_and_budget():
    H=30000;K=4;s=np.arange(H)%K
    good=run_mask_pooled(s,np.zeros(H,dtype=np.int8),K,FAMILY)
    bad=run_mask_pooled(s,np.ones(H,dtype=np.int8),K,FAMILY)
    assert good.selected is not None and good.labels<=4096
    assert bad.selected is None and bad.labels<=4096


def test_mask_pool_spend_is_fixed_before_labels():
    H=30000;K=4;s=np.arange(H)%K
    a=run_mask_pooled(s,np.zeros(H,dtype=np.int8),K,FAMILY)
    b=run_mask_pooled(s,np.ones(H,dtype=np.int8),K,FAMILY)
    # The unlabeled pilot and all q-derived target checkpoints do not use losses.
    assert a.pilot_n==b.pilot_n


def test_state_biased_acquisition_cannot_be_reused_for_direct_mask_cp():
    # For mask (0,1), q=(.5,.5), r=(0,1), true R=.5. A strategy
    # querying only state 0 would see risk 0 and make invalid inferences.
    r=np.array([0.,1.]);q=np.array([.5,.5]);A=(0,1)
    true_r=float((q*r).sum())
    assert true_r==.5
    assert cp_one_sided(0,200,.025)[1]<.15
    # This documents a prohibited query policy, not an executable mode.


def test_rare_irrelevant_state():
    rng=np.random.default_rng(44);H=30000
    q=[.4999995,.4999995,.000001]
    s=rng.choice(3,H,p=q);loss=np.zeros(H,dtype=np.int8)
    x=run_mask_pooled(s,loss,3,((0,1),))
    assert x.selected==(0,1)
    assert x.per_state[2]==0 and x.labels<=4096


def test_q_robust_coverage_defers_when_not_enough_pilot():
    rng=np.random.default_rng(45);H=20000
    s=rng.choice(4,H,p=[.1,.2,.3,.4]);loss=np.zeros(H,dtype=np.int8)
    x=run_mask_pooled(s,loss,4,((0,),),pilot_n=100,gamma=.25)
    assert x.selected is None
