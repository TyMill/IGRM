"""Reproduce an *empirical* mask-complete type-I stress test, not proof.

A=(0,1), q=(1/2,1/2), r=(0.001,0.301), R_A=0.151>0.15.
Even though losses strongly depend on the state, complete querying within
A provides iid Bernoulli(R_A) mask-level losses in the target epoch.
"""
from pathlib import Path
import json
import numpy as np
from scripts.mask_pooled_v19 import run_mask_pooled

def main(repeats=2000, seed=27182818, output='data/heterogeneous_unsafe_stress_v19.json'):
    false=0;max_labels=0
    for j in range(repeats):
        rng=np.random.default_rng(seed+j)
        s=rng.choice(2,size=2500,p=[.5,.5])
        r=np.array([.001,.301])
        loss=(rng.random(2500)<r[s]).astype(np.int8)
        x=run_mask_pooled(s,loss,2,((0,1),),pilot_n=300,budget_fraction=.25,
                 max_labels=512,delta_q=.025,delta_cert=.025)
        false+=x.selected is not None
        max_labels=max(max_labels,x.labels)
    result={'seed':seed,'repeats':repeats,'q':[.5,.5],'r':[.001,.301],
        'true_mask_risk':.151,'tau':.15,'unsafe_certifications':false,
        'rate':false/repeats,'max_labels':max_labels,
        'note':'Monte Carlo check, NOT a proof of validity; ground truth is slightly unsafe and heterogeneous.'}
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': main()
