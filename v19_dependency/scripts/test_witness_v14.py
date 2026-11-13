"""Independent tests for the known-witness subproblem; no VOI optimality claim."""
import math

import pytest
from scipy.stats import binom

from scripts.witness_separation_v14 import (
    binary_kl, certificate, np_cutoff, search_exact_certificate,
)


def test_cutoff_validity_and_maximality():
    for m in (50, 120, 190, 1000):
        for a in (.05,.01,.001):
            k = np_cutoff(m,.15,a)
            assert binom.cdf(k,m,.15) <= a + 1e-14
            if k < m:
                assert binom.cdf(k+1,m,.15) > a - 1e-14


def test_kl_bernoulli_is_nonnegative():
    for p in (.01,.05,.08,.2,.9):
        assert binary_kl(p,.15)>=0
    assert binary_kl(.15,.15) == pytest.approx(0,abs=1e-14)


def test_concrete_separation_and_budget():
    c = certificate(1743,190,q=1/8,r=.08,tau=.15,alpha=.05,beta=.1,b=.2)
    assert c.k == 20
    assert c.true_power >= .9
    assert c.false_certificate_probability_at_boundary <= .05
    assert c.m <= math.floor(c.b*c.n)
    assert math.ceil(c.uniform_information_lower_n) == 4222
    assert c.n < c.uniform_information_lower_n


def test_exact_search_reproduces_the_key_case():
    best=search_exact_certificate(q=1/8,r=.08,tau=.15,b=.2,
                                  alpha=.05,beta=.1,max_m=350)
    assert best.n==1710
    assert best.m==182


def test_under_unsafe_alternative_error_below_alpha():
    c = certificate(1743,190,q=1/8,r=.08,tau=.15,alpha=.05,beta=.1,b=.2)
    for unsafe in (.15, .151, .2, .4, .99):
        assert c.arrival_probability*binom.cdf(c.k,c.m,unsafe) <= .05 + 1e-12


def test_no_magic_if_budget_smaller():
    with pytest.raises(ValueError):
        certificate(100,30,q=.125,r=.08,tau=.15,alpha=.05,beta=.1,b=.2)


def test_separation_ratio():
    K,b=8,.2
    q=1/K
    assert min(q,b)/(b*q)==pytest.approx(5.)
