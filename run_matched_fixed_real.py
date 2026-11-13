from pathlib import Path
import sys
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R),str(R/'v19_dependency')]
from scripts.run_mask_pooled_v19 import FAMILY
from scripts.mask_pooled_v19 import fixed_mask_pooled
from stronger_baselines import fixed_mask_holm
base=pd.read_csv(R/'results/diamonds/frozen_heldout_annotations.csv')
q=np.bincount(base.state,minlength=4)/len(base)
r=np.bincount(base.state,weights=base.loss,minlength=4)/np.bincount(base.state,minlength=4)
rows=[]
for k in range(24):
 rng=np.random.default_rng(891740+k*31); order=rng.permutation(len(base))
 s=base.state.to_numpy()[order]; z=base.loss.to_numpy(dtype=np.int8)[order]
 for n in [1843,2048]:
  for method,f in [('fixed_cp',fixed_mask_pooled),('fixed_holm',fixed_mask_holm)]:
   res=f(s,z,4,FAMILY,pilot_n=2048,n_labels=n,seed=82000+k)
   a=res.selected;cov=0 if a is None else float(q[list(a)].sum())
   risk=np.nan if a is None else float(np.dot(q[list(a)],r[list(a)])/cov)
   rows.append(dict(order_id=k,method=method,n_labels=n,coverage=cov,risk=risk))
out=R/'results/diamonds/fixed_budget_sensitivity.csv';pd.DataFrame(rows).to_csv(out,index=False)
print(pd.DataFrame(rows).groupby(['method','n_labels']).coverage.agg(['mean','std','count']).to_string())
