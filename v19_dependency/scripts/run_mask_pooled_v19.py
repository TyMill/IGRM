"""Held-out paired experiment for mask-level sequential certification (v1.9).

Primary analysis uses a fresh deterministic seed (314159265) distinct from all
v1.8 runs and the initial 100-seed exploratory pilot. All methods receive the
same unlabeled pilot, same hidden stream, same delta_q=.025, delta_cert=.025.
Four queried-label strategies have the same hard 4096-label budget; the 6000
control is explicitly larger. No oracle q or r is available to any procedure.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scripts.mask_pooled_v19 import run_mask_pooled,fixed_mask_pooled
from scripts.finite_horizon_cp_v19 import run_unknown_q_finite_cp
from scripts.unknown_q_v18 import run_unknown_q,fixed_unknown_q_cp
from scripts.run_adaptive_overlap_v17 import WORLDS,FAMILY

EXTENDED={**WORLDS,
    'near_boundary_asymmetric':{'q':[.15,.35,.25,.25],
            'r':[.105,.14,.16,.19]},
    'rare_irrelevant':{'q':[.4999995,.4999995,.000001],
            'r':[.02,.03,.9]}}
ORDER=('statewise_v18','statewise_finite_cp_v19','statewise_fixed_cp_4096',
       'mask_sequential_v19','mask_sequential_allstates_v19',
       'mask_fixed_cp_4096','mask_fixed_cp_6000')

def run(output:Path,repeats:int=50,rep_start:int=0,
        H:int=30000,pilot_n:int=4096,seed:int=314159265):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for wid,(world,spec) in enumerate(EXTENDED.items()):
        q=np.array(spec['q'],float);r=np.array(spec['r'],float);K=len(q)
        family=FAMILY if K==4 else ((0,1),)
        oracle=max((q[list(a)].sum() for a in family if
            (q[list(a)]*(r[list(a)]-.15)).sum()<=0.0),default=0.)
        for rep in range(rep_start,rep_start+repeats):
            seed_i=seed+wid*1000003+31*rep
            rng=np.random.default_rng(seed_i)
            states=rng.choice(K,H,p=q)
            losses=(rng.random(H)<r[states]).astype(np.int8)
            ixseed=seed+wid*500001+rep
            methods=[
                ('statewise_v18',run_unknown_q(states,losses,K,family,pilot_n=pilot_n)),
                ('statewise_finite_cp_v19',run_unknown_q_finite_cp(states,losses,K,family,pilot_n=pilot_n)),
                ('statewise_fixed_cp_4096',fixed_unknown_q_cp(states,losses,K,family,pilot_n=pilot_n,
                                                      n_labels=4096,seed=ixseed)),
                ('mask_sequential_v19',run_mask_pooled(states,losses,K,family,pilot_n=pilot_n)),
                ('mask_sequential_allstates_v19',run_mask_pooled(states,losses,K,family,pilot_n=pilot_n,query_all_states=True)),
                ('mask_fixed_cp_4096',fixed_mask_pooled(states,losses,K,family,pilot_n=pilot_n,
                                                       n_labels=4096,seed=ixseed)),
                ('mask_fixed_cp_6000',fixed_mask_pooled(states,losses,K,family,pilot_n=pilot_n,
                                                       n_labels=6000,seed=ixseed)),
            ]
            for method,x in methods:
                a=x.selected
                cov=0. if a is None else float(q[list(a)].sum())
                risk=None if a is None else float((q[list(a)]*r[list(a)]).sum()/cov)
                rows.append({'world':world,'rep':rep,'method':method,
                    'coverage':cov,'labels':x.labels,'selected':'' if a is None else ','.join(map(str,a)),
                    'risk':'' if risk is None else risk,
                    'unsafe':int(risk is not None and risk>.15),
                    'oracle_coverage':oracle,'pilot_n':pilot_n})
    path=output/'mask_pooled_heldout_v19.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={'protocol':'locked before running held-out seed 314159265',
        'seed':seed,'repeats':repeats,'rep_start':rep_start,'H':H,'pilot_n':pilot_n,
        'b':.2,'tau':.15,'gamma':.25,'delta_q':.025,'delta_cert':.025,
        'delta_elim':.025,'terminal_weight':.8,'target_fraction':.9,
        'max_sequential_labels':4096,'fixed_budgets':[4096,6000],
        'worlds':EXTENDED, 'methods':ORDER,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (output/'mask_heldout_manifest_v19.json').write_text(json.dumps(manifest,indent=2))
    print('rows',len(rows),'sha256',manifest['sha256'])
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data')
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--rep-start',type=int,default=0)
    a=p.parse_args()
    run(Path(a.output),a.repeats,a.rep_start)
