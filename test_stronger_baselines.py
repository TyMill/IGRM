import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from stronger_baselines import fixed_mask_holm,run_hybrid_betting_terminal_cp
from scripts.mask_pooled_v19 import fixed_mask_pooled

def test_holm_not_worse_than_bonferroni_identical_fixed_sample():
    q=np.array([.25,.25,.25,.25]);r=np.array([.02,.04,.16,.3])
    fam=((0,1),(0,1,2),(1,2),(2,3),(0,1,2,3))
    for seed in range(10):
        rng=np.random.default_rng(seed)
        s=rng.choice(4,size=18000,p=q);y=(rng.random(len(s))<r[s]).astype(np.int8)
        a=fixed_mask_holm(s,y,4,fam,pilot_n=1500,n_labels=2048,seed=seed)
        b=fixed_mask_pooled(s,y,4,fam,pilot_n=1500,n_labels=2048,seed=seed)
        # Holm with at most same eligible masks is at least as powerful as
        # globally Bonferroni at alpha/M, with monotone p-value equivalence.
        assert a.lower_coverage+1e-12>=b.lower_coverage
        assert a.labels==b.labels==2048

def test_hybrid_hard_budget_valid_format():
    rng=np.random.default_rng(1)
    s=rng.choice(4,12000,p=[.2,.3,.3,.2]);y=(rng.random(len(s))<np.array([.02,.03,.1,.3])[s]).astype(np.int8)
    a=run_hybrid_betting_terminal_cp(s,y,4,((0,1),(0,1,2),(1,2),(2,3)),pilot_n=1500,max_labels=1000)
    assert 0<=a.labels<=1000
    assert sum(a.per_state)==a.labels
    assert a.lower_coverage<=1

def test_hybrid_and_fixed_reject_obvious_bad():
    s=np.tile(np.arange(4),4000)
    y=np.ones(len(s),dtype=np.int8)
    fam=((0,1),(0,1,2),(1,2),(2,3))
    a=run_hybrid_betting_terminal_cp(s,y,4,fam,pilot_n=1500,max_labels=1000)
    b=fixed_mask_holm(s,y,4,fam,pilot_n=1500,n_labels=1000)
    assert a.selected is None and b.selected is None
