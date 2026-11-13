"""Fair unknown-q fixed-CP baselines on the exact paired streams."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.unknown_q_v18 import fixed_unknown_q_cp
from scripts.run_adaptive_overlap_v17 import WORLDS,FAMILY

def run():
    records=[];H=30000;n0=4096;seed=20261112;repeats=200
    for widx,(name,spec) in enumerate(WORLDS.items()):
        q=np.array(spec['q']);r=np.array(spec['r'])
        for rep in range(repeats):
            rng=np.random.default_rng(seed+1000003*widx+31*rep)
            s=rng.choice(4,H,p=q); loss=(rng.random(H)<r[s]).astype(np.int8)
            for n_labels in [4096,6000]:
                x=fixed_unknown_q_cp(s,loss,4,FAMILY,pilot_n=n0,n_labels=n_labels,
                    seed=seed+widx*500001+rep)
                a=x.selected
                cov=0. if a is None else float(q[list(a)].sum())
                risk=None if a is None else float((q[list(a)]*r[list(a)]).sum()/cov)
                records.append({'world':name,'repeat':rep,'n_labels':n_labels,'coverage':cov,
                                'labels':x.labels,'unsafe':int(risk is not None and risk>.15)})
    path=Path('data/unknown_q_fixed_baselines_v18.csv')
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    d=pd.DataFrame(records)
    summary=d.groupby(['world','n_labels'])[['coverage','unsafe']].mean().reset_index()
    summary.to_csv('data/unknown_q_fixed_baselines_summary_v18.csv',index=False)
    print(summary.round(5).to_string(index=False))
if __name__=='__main__':run()
