"""Pre-specified paired v1.9 protocol (same streams as v1.8, same unknown-q pilot).

Primary new method: finite_horizon_080 with terminal weight .80 (defined
before observing benchmark). Sensitivity: .50 terminal weight, not optimized
against held-out experimental results. Old v1.8 and unknown-q fixed CP are
recomputed from the same frozen random generator, budgets and safety levels.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scripts.unknown_q_v18 import run_unknown_q,fixed_unknown_q_cp
from scripts.finite_horizon_cp_v19 import run_unknown_q_finite_cp
from scripts.run_adaptive_overlap_v17 import WORLDS,FAMILY

def run(output:Path,repeats:int=200,H:int=30000,pilot_n:int=4096,seed:int=20261112,rep_start:int=0):
    output.mkdir(parents=True,exist_ok=True)
    worlds={**WORLDS,'rare_irrelevant':{'q':[.4999995,.4999995,.000001], 'r':[.02,.03,.9]}}
    rows=[]
    for wid,(world,spec) in enumerate(worlds.items()):
        q=np.array(spec['q']);r=np.array(spec['r']);K=len(q)
        fam=FAMILY if K==4 else ((0,1),)
        print('START',world,flush=True)
        for rep in range(rep_start,rep_start+repeats):
            if rep%50==0:print('PROGRESS',world,rep,flush=True)
            local_seed=seed+wid*1000003+rep*31
            rng=np.random.default_rng(local_seed)
            states=rng.choice(K,size=H,p=q)
            losses=(rng.random(H)<r[states]).astype(np.int8)
            methods=[
                ('unknownq_dyadic_v18',run_unknown_q(states,losses,K,fam,pilot_n=pilot_n)),
                ('unknownq_terminal_080',run_unknown_q_finite_cp(states,losses,K,fam,pilot_n=pilot_n,terminal_share=.8)),
                ('unknownq_terminal_050',run_unknown_q_finite_cp(states,losses,K,fam,pilot_n=pilot_n,terminal_share=.5)),
                ('unknownq_fixed_cp_4096',fixed_unknown_q_cp(states,losses,K,fam,pilot_n=pilot_n,n_labels=4096,seed=seed+wid*500001+rep)),
                ('unknownq_fixed_cp_6000',fixed_unknown_q_cp(states,losses,K,fam,pilot_n=pilot_n,n_labels=6000,seed=seed+wid*500001+rep)),
            ]
            for name,x in methods:
                a=x.selected
                cov=0. if a is None else float(q[list(a)].sum())
                risk=None if a is None else float((q[list(a)]*r[list(a)]).sum()/cov)
                rows.append({'world':world,'seed_index':rep,'method':name,
                    'coverage':cov,'labels':x.labels,
                    'risk':'' if risk is None else risk,
                    'unsafe':int(risk is not None and risk>.15),
                    'selected':'' if a is None else ','.join(map(str,a)),
                    'completed_stages':x.completed_stages,
                    'cert_stage':'' if x.certified_at is None else x.certified_at,
                    'stage_failure':int(x.stage_failure),'pilot_n':pilot_n})
    csvpath=output/'finite_horizon_paired_v19.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for world in worlds:
        for method in [x[0] for x in methods]:
            group=[x for x in rows if x['world']==world and x['method']==method]
            summary.append({'world':world,'method':method,'repeats':repeats,
                'mean_coverage':float(np.mean([x['coverage'] for x in group])),
                'mean_labels':float(np.mean([x['labels'] for x in group])),
                'positive_coverage_fraction':float(np.mean([x['coverage']>0 for x in group])),
                'unsafe_count':sum(x['unsafe'] for x in group),
                'stage_failures':sum(x['stage_failure'] for x in group)})
    sumcsv=output/'finite_horizon_summary_v19.csv'
    with sumcsv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    manifest={'seed':seed,'H':H,'pilot_n':pilot_n,'b':.2,'tau':.15,'gamma':.25,
        'delta_q':.025,'delta_cert':.025,'delta_elim':.025,
        'delta_arrival':.025,'terminal_weight_primary':.8,'terminal_sensitivity':.5,
        'fixed_budgets':[4096,6000], 'repeats':repeats, 'rep_start':rep_start,
        'worlds':worlds,
        'sha256_csv':hashlib.sha256(csvpath.read_bytes()).hexdigest(),
        'sha256_summary':hashlib.sha256(sumcsv.read_bytes()).hexdigest()}
    (output/'finite_horizon_manifest_v19.json').write_text(json.dumps(manifest,indent=2))
    print('rows',len(rows),'sha256',manifest['sha256_csv'])
    for s in summary:print(s)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data');p.add_argument('--repeats',type=int,default=200);p.add_argument('--H',type=int,default=30000);p.add_argument('--pilot-n',type=int,default=4096)
    p.add_argument('--rep-start',type=int,default=0)
    a=p.parse_args();run(Path(a.output),a.repeats,a.H,a.pilot_n,rep_start=a.rep_start)
