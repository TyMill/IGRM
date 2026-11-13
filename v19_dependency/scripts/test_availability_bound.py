"""Independent deterministic checks for the two-state theorem example."""
import math
from scipy.stats import binom
from scripts.recompute_availability_bound import kl, composite_info, pair_info, numeric_least_favorable


def test_binary_kl_positive():
    assert kl(.14,.15)>0
    assert kl(.14,.16)>kl(.14,.15)


def test_necessary_budget_frontier():
    kappa=kl(.9,.05)
    assert composite_info(12000,2400)<kappa
    assert composite_info(30000,6000)>kappa
    assert composite_info(30000,18000)==18000*kl(.14,.15)


def test_rare_state_availability_cap():
    assert math.isclose(composite_info(1000,1000,q1=.1),100*kl(.14,.15))
    assert math.isclose(pair_info(1000,1000,q1=.1),100*kl(.14,.16))


def test_closed_form_least_favorable():
    result=numeric_least_favorable()
    assert abs(result.fun-2400*kl(.14,.15))<1e-8
    assert abs(result.x[0]-.15)<1e-7


def test_constructive_binomial_rule():
    B=11081;n=55405;k=1600
    assert B<=.2*n
    false_approve=binom.cdf(k,B,.15)
    success_given_labels=binom.cdf(k,B,.14)
    # Bound the failure of obtaining B state-1 arrivals using Hoeffding.
    shortage_bound=math.exp(-2*n*(.6-B/n)**2)
    assert false_approve<.05
    assert (1-shortage_bound)*success_given_labels>.9


def test_necessity_does_not_imply_sufficiency():
    kappa=kl(.9,.05)
    # The single pair is separable at n=12000 while the composite unsafe family is not.
    assert pair_info(12000,2400)>kappa
    assert composite_info(12000,2400)<kappa
