"""Matched paired experiment for known/unknown q, honest benchmark and rare irrelevant state."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scripts.unknown_q_v18 import run_unknown_q
from scripts.adaptive_overlap_v17 import run_adaptive, fixed_uniform_cp
from scripts.run_adaptive_overlap_v17 import WORLDS,FAMILY


def main(out:Path,repeats:int=100,H:int=30000,pilot_n:int=4096,seed:int=20261112):
    out.mkdir(parents=True,exist_ok=True)
    worlds={**WORLDS,
        'rare_irrelevant':{'q':[.4999995,.4999995,.000001],'r':[.02,.03,.9]}}
    rows=[]
    for widx,(name,spec) in enumerate(worlds.items()):
        q=np.array(spec['q']); r=np.array(spec['r']);K=len(q)
        fam=FAMILY if K==4 else ((0,1),)
        for rep in range(repeats):
            rng=np.random.default_rng(seed+1000003*widx+31*rep)
            st=rng.choice(K,H,p=q); losses=(rng.random(H)<r[st]).astype(np.int8)
            base=run_adaptive(st,losses,q,fam,tau=.15,gamma=.25,delta_cert=.025,delta_arrival=.025)
            unk=run_unknown_q(st,losses,K,fam,pilot_n=pilot_n,tau=.15,gamma=.25,delta_q=.025,delta_cert=.025,delta_arrival=.025)
            fixed=fixed_uniform_cp(st,losses,q,fam,tau=.15,gamma=.25,n_labels=int(.2*H),alpha=.05,seed=seed+widx*500001+rep)
            for method,res in [('known_q_adaptive',base),('estimated_q_adaptive',unk),('known_q_fixed_cp',fixed)]:
                A=res.selected
                cov=0 if A is None else float(q[list(A)].sum())
                risk=None if A is None else float((q[list(A)]*r[list(A)]).sum()/cov)
                rows.append({'world':name,'rep':rep,'method':method,'labels':res.labels,
                    'coverage':cov,'risk':risk if risk is not None else '',
                    'unsafe':int(risk is not None and risk>.15),
                    'failed_stage':int(res.stage_failure),
                    'selected':'' if A is None else ','.join(map(str,A)),
                    'pilot_n':pilot_n if method=='estimated_q_adaptive' else 0})
    with (out/'unknown_q_v18.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    summary=[]
    for name in worlds:
        for method in ['known_q_adaptive','estimated_q_adaptive','known_q_fixed_cp']:
            group=[x for x in rows if x['world']==name and x['method']==method]
            summary.append({'world':name,'method':method,'repeats':repeats,
                'mean_coverage':np.mean([x['coverage'] for x in group]),
                'mean_labels':np.mean([x['labels'] for x in group]),
                'unsafe_certifications':sum(x['unsafe'] for x in group),
                'stage_failures':sum(x['failed_stage'] for x in group)})
    with (out/'unknown_q_summary_v18.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    manifest={'seed':seed,'H':H,'pilot_n':pilot_n,'delta_q':.025,'delta_cert':.025,
              'delta_arrival':.025,'b':.2,'gamma':.25,'repeats':repeats,'worlds':worlds,
              'results_sha256':hashlib.sha256((out/'unknown_q_v18.csv').read_bytes()).hexdigest(),
              'summary':summary}
    (out/'unknown_q_manifest_v18.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'rows':len(rows),'sha256':manifest['results_sha256'],'summary':summary},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data');p.add_argument('--repeats',type=int,default=100);p.add_argument('--H',type=int,default=30000);p.add_argument('--pilot-n',type=int,default=4096)
    a=p.parse_args();main(Path(a.output),a.repeats,a.H,a.pilot_n)
