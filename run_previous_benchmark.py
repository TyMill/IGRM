"""Locked, independently seeded re-comparison on original seven v1.9 scenarios.

This is a BENCHMARK of prior-art e-processes vs v1.9, not new theory.
The world families were previously studied; fresh rep indices test stochastic
replicability but do not count as truly new problem families.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from bench_methods import run_mask_eprocess,fixed_joint_weighted_hoeffding
from scripts.mask_pooled_v19 import run_mask_pooled,fixed_mask_pooled
from scripts.run_mask_pooled_v19 import EXTENDED,FAMILY

METHODS=('mask_cp_sequential_v19','mask_e_mixture','mask_e_predictable_betting',
         'mask_cp_fixed_4096','weighted_joint_hoeffding_fixed_4096')

def run(out:Path,repeat_start=1000,repeats=40,worlds=None):
    out.mkdir(parents=True,exist_ok=True)
    worlds=list(EXTENDED) if worlds is None else worlds
    rows=[];start=time.monotonic()
    for world in worlds:
        wid=list(EXTENDED).index(world)
        spec=EXTENDED[world];q=np.asarray(spec['q'],float);r=np.asarray(spec['r'],float)
        K=len(q);fam=FAMILY if K==4 else ((0,1),)
        oracle=max((float(q[list(a)].sum()) for a in fam
                    if np.sum(q[list(a)]*(r[list(a)]-.15))<=0),default=0.)
        for rep in range(repeat_start,repeat_start+repeats):
            # Existing v1.9 RNG schedule with non-overlapping rep indices.
            seed=314159265+wid*1000003+31*rep
            rng=np.random.default_rng(seed)
            states=rng.choice(K,30000,p=q)
            losses=(rng.random(30000)<r[states]).astype(np.int8)
            fixed_seed=314159265+wid*500001+rep
            methods=(
                ('mask_cp_sequential_v19',run_mask_pooled(states,losses,K,fam)),
                ('mask_e_mixture',run_mask_eprocess(states,losses,K,fam,mode='mixture')),
                ('mask_e_predictable_betting',run_mask_eprocess(states,losses,K,fam,mode='betting')),
                ('mask_cp_fixed_4096',fixed_mask_pooled(states,losses,K,fam,
                                                      n_labels=4096,seed=fixed_seed)),
                ('weighted_joint_hoeffding_fixed_4096',fixed_joint_weighted_hoeffding(
                    states,losses,K,fam,n_labels=4096,seed=fixed_seed)),
            )
            for name,res in methods:
                a=res.selected
                cov=0. if a is None else float(q[list(a)].sum())
                risk='' if not a else float(np.dot(q[list(a)],r[list(a)])/cov)
                rows.append({'world':world,'rep':rep,'method':name,
                             'coverage':cov,'labels':res.labels,
                             'selected':'' if not a else ','.join(map(str,a)),
                             'risk':risk,'unsafe':int(bool(a) and risk>.15),
                             'oracle_coverage':oracle,
                             'pilot_n':4096,'label_cap':4096,
                             'delta_q':.025,'delta_cert':.025})
        print('completed',world,'repeats',repeats,'elapsed_s',round(time.monotonic()-start,1),flush=True)
    csv_path=out/'benchmark_paired.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for world in worlds:
        for method in METHODS:
            a=[z for z in rows if z['world']==world and z['method']==method]
            summary.append({'world':world,'method':method,'n_repeats':len(a),
                'mean_coverage':float(np.mean([x['coverage'] for x in a])),
                'mean_labels':float(np.mean([x['labels'] for x in a])),
                'positive_certifications':sum(x['coverage']>0 for x in a),
                'unsafe_certifications':sum(x['unsafe'] for x in a),
                'oracle_coverage':a[0]['oracle_coverage']})
    summary_path=out/'benchmark_summary.csv'
    with summary_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    manifest={'status':'PRELIMINARY RESEARCH BENCHMARK; prior art baselines',
        'seed_schedule':'314159265+world_index*1000003+31*rep',
        'repeat_start':repeat_start,'repeats':repeats,'worlds':worlds,
        'H':30000,'pilot_n':4096,'tau':.15,'gamma':.25,'budget_fraction':.2,
        'actual_hard_label_cap':4096,'delta_q':.025,'delta_cert':.025,
        'delta_elim':.025,'fixed_mask_alt_probs':[float(x) for x in __import__('bench_methods').ALTERNATIVES],
        'fixed_mask_alt_weights':[float(x) for x in __import__('bench_methods').WEIGHTS],
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (csv_path,summary_path)},
        'v19_source_files_sha256':{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(
            (Path(__file__).resolve().parent/'v19_dependency/scripts').glob('*.py'))},
        'note':'All methods see the same hidden stream and unlabeled pilot. '
               'No method is passed true q/r. Fixed comparators choose exactly 4096 labels '
               'uniformly without replacement. No tuning on repeat_start: fresh replicates '
               'of worlds previously seen in v1.9. Different validity techniques apply.'}
    (out/'BENCHMARK_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('rows',len(rows),'sha256',manifest['files_sha256']['benchmark_paired.csv'])
    return rows,summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    p.add_argument('--rep-start',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=40)
    p.add_argument('--worlds',default='all')
    x=p.parse_args()
    worlds=list(EXTENDED) if x.worlds=='all' else x.worlds.split(',')
    if any(w not in EXTENDED for w in worlds):raise ValueError('unknown world')
    run(Path(x.output),x.rep_start,x.repeats,worlds)
