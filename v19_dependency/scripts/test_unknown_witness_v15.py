import math
import numpy as np
import pytest
from scipy.stats import binom
from scripts.unknown_witness_v15 import (
  design_unknown_witness, largest_binomial_safe_k, lower_horizons, monte_carlo, trial
)

@pytest.fixture(scope='module')
def d(): return design_unknown_witness()

def test_design_regime(d):
    assert d.K == 8 and d.m0 == 16 and d.n0 == 226
    assert d.m == 266 and d.k == 30 and d.n1 == 2374
    assert d.n == 2600 and d.queries_max == 394

def test_four_error_allocations(d):
    assert d.pilot_misidentification_bound <= d.beta/4
    assert d.pilot_arrival_failure_bound <= d.beta/4
    assert d.confirmation_arrival_failure <= d.beta/4
    assert 1 - d.confirmation_power <= d.beta/4
    assert d.recovery_lower_bound >= 1 - d.beta

def test_exact_calibration(d):
    assert binom.cdf(d.k, d.m, d.tau) <= d.alpha
    assert binom.cdf(d.k+1, d.m, d.tau) > d.alpha
    for unsafe_r in [d.tau, .16, .24, .5, .9, 1.]:
        assert binom.cdf(d.k,d.m,unsafe_r) <= d.alpha+1e-12

def test_hard_budget(d):
    assert d.queries_max <= math.floor(d.b*d.n)
    rng=np.random.default_rng(32)
    for _ in range(500):
        for rates in (None,np.linspace(d.tau,.99,d.K)):
            cert,j,n_lab,unsafe=trial(d,rng,rates)
            assert n_lab <= d.queries_max
            assert (not cert) or (0<=j<d.K)

def test_information_bound(d):
    low, uniform = lower_horizons(d)
    assert low < d.n < uniform
    assert uniform >= 4222

def test_determinism(d):
    assert monte_carlo(d,300,77)==monte_carlo(d,300,77)

def test_unknown_witness_all_locations(d):
    rng=np.random.default_rng(888)
    for witness in range(d.K):
        for _ in range(20):
            cert,j,n_lab,unsafe=trial(d,rng,witness=witness)
            assert unsafe==0
            assert n_lab<=d.queries_max

def test_invalid_params():
    for kwargs in [dict(K=1),dict(r_safe=.2),dict(r_other=.2),dict(alpha=.95),dict(b=0)]:
        with pytest.raises(ValueError): design_unknown_witness(**kwargs)

def test_discovery_lp():
    from scripts.unknown_witness_v15 import discovery_information_lp
    d=design_unknown_witness()
    ans=discovery_information_lp(d)
    assert ans['necessary_horizon_real']>0
    assert ans['necessary_horizon_real'] < d.n
    u=np.array(ans['expected_labels'])
    n=ans['necessary_horizon_real']
    assert u.sum() <= d.b*n+1e-8
    assert np.all(u <= n/d.K+1e-8)
    assert u[0]*ans['risk_divergence']>=ans['risk_KL_requirement']-1e-8
    a,c=ans['identity_divergences']
    for val in u[1:]:assert u[0]*a+val*c>=ans['identity_KL_requirement']-1e-8
