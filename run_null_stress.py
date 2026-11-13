"""Finite-horizon empirical null calibration, not a proof of type-I control."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R),str(R/'v19_dependency')]
from stronger_baselines import run_hybrid_betting_terminal_cp,fixed_mask_holm
from scripts.run_mask_pooled_v19 import FAMILY
from scripts.mask_pooled_v19 import run_mask_pooled

def run(n=180):
 rows=[]
 for p in [.15,.151]:
  for rep in range(n):
   rng=np.random.default_rng(4100000+rep+int(p*100000)*100000)
   s=rng.integers(0,4,12000)
   y=(rng.random(12000)<p).astype(np.int8)
   for method,func in [('v19',run_mask_pooled),('hybrid',run_hybrid_betting_terminal_cp),('fixed_holm',fixed_mask_holm)]:
    opts={'pilot_n':1500}
    if method=='fixed_holm':opts|={'n_labels':2048,'seed':517131+rep}
    else:opts|={'max_labels':2048}
    res=func(s,y,4,FAMILY,**opts)
    rows.append((p,rep,method,int(res.selected is not None),res.labels))
  print('completed null risk',p,flush=True)
 import pandas as pd
 df=pd.DataFrame(rows,columns=['true_p','rep','method','certified','labels'])
 out=R/'results/null_stress_new.csv';df.to_csv(out,index=False)
 print(df.groupby(['true_p','method']).agg(n=('rep','count'),false_certs=('certified','sum'),mean_labels=('labels','mean')).to_string())
 return df
if __name__=='__main__':run()
