"""Predeclared pilot-size sensitivity, same state/loss stream in all arms."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import pandas as pd
from scripts.unknown_q_v18 import run_unknown_q
from scripts.run_adaptive_overlap_v17 import WORLDS,FAMILY


def run(out='data/pilot_sensitivity_v18.csv',repeats=200,H=30000,seed=20261112):
    sizes=[1024,4096,8192]
    worlds=['separated','near_boundary','rare_safe']
    rows=[]
    for name in worlds:
        widx=list(WORLDS).index(name)
        q=np.array(WORLDS[name]['q']);r=np.array(WORLDS[name]['r'])
        for rep in range(repeats):
            rng=np.random.default_rng(seed+widx*1000003+rep*31)
            st=rng.choice(4,H,p=q);los=(rng.random(H)<r[st]).astype(np.int8)
            for n0 in sizes:
                x=run_unknown_q(st,los,4,FAMILY,pilot_n=n0,tau=.15,gamma=.25)
                A=x.selected
                cov=0. if A is None else float(q[list(A)].sum())
                risk=None if A is None else float((q[list(A)]*r[list(A)]).sum()/cov)
                rows.append({'world':name,'repeat':rep,'pilot_n':n0,'coverage':cov,'labels':x.labels,
                             'certified':int(A is not None),'unsafe':int(risk is not None and risk>.15),
                             'stages':x.completed_stages,'stage_failure':int(x.stage_failure)})
    out=Path(out)
    with out.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    d=pd.DataFrame(rows);summary=d.groupby(['world','pilot_n'],sort=False)[['coverage','labels','certified','unsafe','stages','stage_failure']].mean().reset_index()
    summary.to_csv(out.with_name('pilot_sensitivity_summary_v18.csv'),index=False)
    print('rows',len(rows),'sha256',hashlib.sha256(out.read_bytes()).hexdigest());print(summary.round(5).to_string(index=False))
if __name__=='__main__':run()
