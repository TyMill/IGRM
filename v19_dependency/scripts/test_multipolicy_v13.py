import numpy as np
import pytest
from multipolicy_certifier_v13 import radii, certify, sample_prefix_labels

Q = np.array([.5, .3, .2]); R=np.array([.03,.06,.9])
QUOTAS=np.array([6000,5000,3000]); N=80000

def test_budget_and_positive_radii():
    dq, dr = radii(N,QUOTAS,.025,.025)
    assert np.sum(QUOTAS) <= .2*N
    assert 0 < dq < .01
    assert np.all((dr > 0)&(dr < .04))

def test_margin_criterion_is_satisfied_by_known_oracle():
    dq, dr = radii(N,QUOTAS,.025,.025)
    C=.8; M=.5*.03+.3*.06
    rhs=2*(1+.15)*dq+2*(.5*dr[0]+.3*dr[1])+2*dq*max(dr[0],dr[1])
    assert C > .75+2*dq
    assert .15*C-M > rhs

def test_certifies_safe_two_state_mask_simulation():
    rng=np.random.default_rng(1234)
    states = rng.choice(3,size=N,p=Q)
    losses = (rng.random(N)<R[states]).astype(int)
    queried = sample_prefix_labels(states,losses,QUOTAS)
    cert = certify(states,queried,QUOTAS)
    assert cert is not None
    true_coverage=sum(Q[list(cert.mask)])
    true_risk=sum((Q*R)[list(cert.mask)])/true_coverage
    assert true_coverage >= cert.coverage_lower >= .75
    assert true_risk <= .15
    assert cert.observed_labels==14000
    assert 2 not in cert.mask

def test_unmet_state_quota_abstains():
    states=np.zeros(100, dtype=int)
    labels=[np.zeros(10,dtype=int),np.zeros(0,dtype=int)]
    assert certify(states,labels,np.array([10,10]),coverage_target=.5) is None

def test_no_false_certification_on_large_observed_error():
    states = np.tile([0,1,2], 30000)[:N]
    queried = [np.ones(int(m),dtype=int) for m in QUOTAS]
    assert certify(states,queried,QUOTAS) is None

@pytest.mark.parametrize('seed', range(10))
def test_random_realizations_preserve_safety(seed):
    rng=np.random.default_rng(seed)
    n=16000; quotas=np.array([1100,950,800])
    states=rng.choice(3,size=n,p=Q)
    losses=(rng.random(n)<R[states]).astype(int)
    cert=certify(states,sample_prefix_labels(states,losses,quotas),quotas)
    if cert is not None:
        true_cov=float(Q[list(cert.mask)].sum())
        true_risk=float((Q*R)[list(cert.mask)].sum()/true_cov)
        assert true_cov>=.75 and true_risk<=.15


def test_forged_labels_cannot_override_arrival_shortage():
    states=np.zeros(100, dtype=int)
    labels=[np.zeros(10,dtype=int),np.zeros(10,dtype=int)]
    assert certify(states,labels,np.array([10,10]),coverage_target=.5) is None

def test_nonbinary_labels_are_rejected():
    states=np.tile([0,1],100)
    labels=[np.full(10,.2),np.zeros(10)]
    with pytest.raises(ValueError):
        certify(states, labels, np.array([10,10]),coverage_target=.5)
