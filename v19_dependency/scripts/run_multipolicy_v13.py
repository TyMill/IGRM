#!/usr/bin/env python3
"""Small numerical sanity check of fixed-quota theorem v1.3.

Only simulated world parameters are used to score the selected mask, never
by the selection rule. Do not interpret this experiment as empirical proof of
finite-sample coverage guarantees or original algorithmic superiority.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from multipolicy_certifier_v13 import certify, sample_prefix_labels, radii

ROOT = Path(__file__).resolve().parents[1]
q=np.array([.5,.3,.2]);r=np.array([.03,.06,.9]);n=80000; quotas=np.array([6000,5000,3000])
rows=[]
for seed in range(100):
    rng=np.random.default_rng(91000+seed)
    states=rng.choice(3,size=n,p=q)
    losses=(rng.random(n)<r[states]).astype(np.int8)
    labels=sample_prefix_labels(states,losses,quotas)
    result=certify(states,labels,quotas)
    if result is None:
        rows.append(dict(seed=seed,certified=0,mask='',true_coverage=0.,true_risk='',coverage_lower='',unsafe_certificate=0,labels=sum(map(len,labels))))
    else:
        true_c=float(q[list(result.mask)].sum())
        true_risk=float(np.sum((q*r)[list(result.mask)])/true_c)
        rows.append(dict(seed=seed,certified=1,mask=''.join(str(i+1) for i in result.mask),true_coverage=true_c,true_risk=true_risk,coverage_lower=result.coverage_lower,unsafe_certificate=int(true_c<.75 or true_risk>.15),labels=result.observed_labels))
assert len(rows)==100
p=ROOT/'data/multipolicy_v13_100seeds.csv'
with p.open('w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
dq, dr=radii(n,quotas,.025,.025)
summary={'scenario':'synthetic, stationary, 3 states','seeds':100,'horizon':n,'quotas':quotas.tolist(),'hard_label_budget':int(quotas.sum()),'alpha_q':.025,'alpha_r':.025,'dq':float(dq),'d_i':dr.tolist(),'certified_runs':sum(int(x['certified']) for x in rows),'unsafe_certifications_observed':sum(int(x['unsafe_certificate']) for x in rows),'mean_certified_true_coverage':float(np.mean([float(x['true_coverage']) for x in rows])),'all_realized_masks':sorted(set(x['mask'] for x in rows)), 'interpretation':'Sanity check only; 100 seeds cannot establish 95% validity, novelty, or VOI superiority.'}
(ROOT/'data/multipolicy_v13_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
