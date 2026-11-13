"""Regression and proof-obligation checks for finite-horizon CP spending."""
import numpy as np
import pytest
from scipy.stats import binom
from scripts.finite_horizon_cp_v19 import terminal_weights, finite_cp_bounds, run_unknown_q_finite_cp
from scripts.run_adaptive_overlap_v17 import FAMILY

@pytest.mark.parametrize('J',[1,2,3,7,11,17])
def test_spending_sums(J):
    for t in [.1,.5,.8,.95]:
        w=terminal_weights(J,t)
        assert np.isclose(w.sum(),1.,atol=1e-13)
        assert np.all(w>0)
        if J>1:assert np.isclose(w[-1],t)

@pytest.mark.parametrize('a',[.025,.05])
def test_upper_tail_exact_at_each_checkpoint(a):
    """Inversion of CP: P_p(U<p) <= assigned alpha (enumerated for n<=16)."""
    p=.15; J=5;K=2
    for j in range(J):
        n=1<<j
        ass=a*terminal_weights(J,.8)[j]/K
        fail=0.
        for k in range(n+1):
            U=finite_cp_bounds(np.array([k,0]),np.array([n,0]),j,J=J,K=K,
                    delta_cert=a)[1][0]
            if U<p-1e-14:fail+=binom.pmf(k,n,p)
        assert fail<=ass+1e-12

def test_no_slack_in_certification():
    H=30000;K=4;pilot=4096
    s=np.arange(H)%K;loss=np.zeros(H,dtype=np.int8)
    good=run_unknown_q_finite_cp(s,loss,K,FAMILY,pilot_n=pilot)
    assert good.labels<=int(.2*H) and good.selected is not None
    assert good.certified_lower_coverage>=.25
    bad=run_unknown_q_finite_cp(s,np.ones(H,dtype=np.int8),K,FAMILY,pilot_n=pilot)
    assert bad.selected is None

def test_finite_cp_is_stronger_at_final_dyadic_than_v18():
    from scripts.adaptive_overlap_v17 import dyadic_bounds
    J=11; j=10; n=1<<j; k=145
    U_old=dyadic_bounds(np.array([k,0,0,0]),np.array([n,0,0,0]),j,.025)[1][0]
    U_new=finite_cp_bounds(np.array([k,0,0,0]),np.array([n,0,0,0]),j,J=J,K=4,
        delta_cert=.025)[1][0]
    assert U_new<U_old

@pytest.mark.parametrize('wrong',[-.1,0,1,2])
def test_invalid_terminal_share(wrong):
    with pytest.raises(ValueError):terminal_weights(4,wrong)

def test_rare_irrelevant_state():
    rng=np.random.default_rng(12);H=30000
    s=rng.choice(3,H,p=[.4999995,.4999995,.000001]);loss=(rng.random(H)<.02).astype('i1')
    x=run_unknown_q_finite_cp(s,loss,3,((0,1),),pilot_n=4096)
    assert x.completed_stages>0 and x.selected==(0,1)
