import numpy as np
import pytest

from scripts.overlap_v16 import (
    closed_form_three_state, shared_allocation, margin,
    simultaneous_radius, certifier, variance_proxy,
)


def test_closed_form_vs_solver():
    c = closed_form_three_state()
    d = shared_allocation(np.ones(3)/3, [(0,1),(1,2)], [c['margin']]*2)
    assert abs(d.objective_real-c['shared_total']) < 1e-3
    assert d.residual_max < 1e-9


def test_integer_feasibility():
    q=np.ones(3)/3
    c=closed_form_three_state()
    d=shared_allocation(q,[(0,1),(1,2)],[c['margin']]*2)
    n=np.asarray(d.n_integer)
    assert 2 * simultaneous_radius(q,n,(0,1),2,.05) <= c['margin']+1e-10
    assert 2 * simultaneous_radius(q,n,(1,2),2,.05) <= c['margin']+1e-10


def test_reuse_saves_labels_vs_isolated():
    c=closed_form_three_state()
    assert c['shared_total'] < c['uniform_shared_total'] < c['isolated_total']
    assert abs(c['shared_total']/c['isolated_total']-(3+2**.5*2)/8) < 1e-10


def test_mask_risk_relation():
    q=np.ones(3)/3; r=np.array([.1,.15,.2]); a=(0,1,2)
    assert margin(q,r,a,.15) == pytest.approx(0, abs=1e-14)


def test_variance_decreases_on_more_labels():
    q=np.ones(4)/4
    assert variance_proxy(q,np.array([100]*4),(0,1)) > variance_proxy(q,np.array([200]*4),(0,1))


def test_abstains_without_labels():
    q=np.ones(3)/3
    assert certifier(q,np.zeros(3),np.zeros(3),[(0,1)],.15,.1) is None


def test_certifier_never_uses_r_truth():
    q=np.ones(4)/4; n=np.full(4,2000)
    a=certifier(q,np.array([0,0,0,2000]),n,[(0,1),(2,3)],.15,.1)
    assert a == (0,1)


def test_cp_unlabeled_state():
    q=np.ones(3)/3; n=np.array([100,100,0]); errors=np.array([0,0,0])
    assert certifier(q,errors,n,[(0,1),(1,2)],.15,.1,method='cp-box') == (0,1)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        certifier(np.ones(2)/2,np.zeros(2),np.ones(2),[(0,1)],.15,.1,method='wrong')


def test_invalid_distribution_rejected():
    with pytest.raises(ValueError):
        shared_allocation(np.array([.2,.2]),[(0,1)],[.01])


def test_known_probability_mask_selection():
    q=np.ones(4)/4; n=np.full(4,20000)
    errors=np.array([int(.1*20000),int(.1*20000),int(.1*20000),int(.3*20000)])
    a=certifier(q,errors,n,[(0,1),(1,2),(0,1,2),(2,3)],.15,.1)
    assert a==(0,1,2)


def test_no_false_cert_when_error_obvious():
    q=np.ones(3)/3; n=np.full(3,30000); k=np.full(3,10000)
    assert certifier(q,k,n,[(0,1),(1,2)],.15,.1) is None
