"""Held-out paired synthetic overlap experiment with strong nonadaptive baseline.

The generator knows q,r; certifiers see only q, visible states and queried losses.
True population risk is used only for post-hoc evaluation. No tuning on these data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.adaptive_overlap_v17 import design_phases, fixed_uniform_cp, run_adaptive


FAMILY = ((0, 1), (0, 1, 2), (1, 2), (2, 3), (0, 2, 3), (0, 1, 2, 3))
WORLDS = {
    'separated': {'q': [.25, .25, .25, .25], 'r': [.04, .07, .55, .65]},
    'multiple_safe': {'q': [.25, .25, .25, .25], 'r': [.04, .07, .13, .45]},
    'near_boundary': {'q': [.25, .25, .25, .25], 'r': [.125, .145, .165, .19]},
    'all_unsafe': {'q': [.25, .25, .25, .25], 'r': [.20, .25, .30, .35]},
    'rare_safe': {'q': [.10, .20, .30, .40], 'r': [.02, .04, .60, .60]},
}


def run(out: Path, *, repeats: int = 200, H: int = 30000, b: float = .20,
        seed: int = 20261112) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    K = 4
    records = []
    design = {}
    for wid, (world, spec) in enumerate(WORLDS.items()):
        q = np.array(spec['q']); r = np.array(spec['r'])
        oracle = max((q[list(a)].sum() for a in FAMILY if
                      (q[list(a)] * (r[list(a)] - .15)).sum() <= 0), default=0.)
        phases = design_phases(K, float(q.min()), H, int(b*H), .025)
        cap = K * 2**(len(phases)-1)
        design[world] = {'num_stages': len(phases), 'phase_block_lengths': phases,
                         'max_adaptive_labels': cap, 'total_stream_length': sum(phases),
                         'full_label_budget': int(b*H), 'oracle_coverage': float(oracle)}
        for rep in range(repeats):
            local_seed = seed + wid*1000003 + rep*31
            rng = np.random.default_rng(local_seed)
            states = rng.choice(K, size=H, p=q)
            hidden_loss = (rng.random(H) < r[states]).astype(np.int8)
            strategies = {
                'adaptive_cp': run_adaptive(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             budget_fraction=b,delta_cert=.025,
                                             delta_arrival=.025,method='cp'),
                'state_blind_dyadic_cp': run_adaptive(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             budget_fraction=b,delta_cert=.025,
                                             delta_arrival=.025,method='cp',query_all_states=True),
                'adaptive_hoeffding': run_adaptive(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             budget_fraction=b,delta_cert=.025,
                                             delta_arrival=.025,method='hoeffding'),
                'uniform_cp_matched': fixed_uniform_cp(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             n_labels=cap,alpha=.05,seed=local_seed+70001),
                'uniform_cp_full': fixed_uniform_cp(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             n_labels=int(b*H),alpha=.05,seed=local_seed+70001),
                'uniform_cp_matched_equal_delta': fixed_uniform_cp(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             n_labels=cap,alpha=.025,seed=local_seed+70001),
                'uniform_cp_full_equal_delta': fixed_uniform_cp(states,hidden_loss,q,FAMILY,tau=.15,gamma=.25,
                                             n_labels=int(b*H),alpha=.025,seed=local_seed+70001),
            }
            for name, outcome in strategies.items():
                mask = outcome.selected
                cov = 0. if mask is None else float(q[list(mask)].sum())
                risk = None if mask is None else float(np.sum(q[list(mask)]*r[list(mask)])/cov)
                records.append({
                    'world': world, 'repeat': rep, 'strategy': name,
                    'labels': outcome.labels,
                    'arrivals_until_decision': int(sum(outcome.stage_lengths[:outcome.completed_stages+int(outcome.stage_failure)])) if outcome.stage_lengths else H,
                    'completed_stages': outcome.completed_stages,
                    'eliminated_masks': outcome.eliminated_masks,
                    'stage_failure': int(outcome.stage_failure),
                    'selected_mask': '' if mask is None else ','.join(map(str,mask)),
                    'certified_coverage': cov, 'population_risk': '' if risk is None else risk,
                    'unsafe_certification': int(risk is not None and risk > .15+1e-12),
                    'oracle_coverage': float(oracle),
                    'full_budget': int(b*H), 'adaptive_worst_case_cap': cap,
                })
    fields=list(records[0]); path=out/'adaptive_overlap_v17.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(records)
    summary=[]
    for world in WORLDS:
        for strategy in ('adaptive_cp','state_blind_dyadic_cp','adaptive_hoeffding','uniform_cp_matched','uniform_cp_full',
                         'uniform_cp_matched_equal_delta','uniform_cp_full_equal_delta'):
            rows=[x for x in records if x['world']==world and x['strategy']==strategy]
            summary.append({
                'world':world,'strategy':strategy,'replicates':repeats,
                'mean_coverage':float(np.mean([x['certified_coverage'] for x in rows])),
                'mean_labels':float(np.mean([x['labels'] for x in rows])),
                'mean_arrivals_until_decision':float(np.mean([x['arrivals_until_decision'] for x in rows])),
                'sd_coverage':float(np.std([x['certified_coverage'] for x in rows],ddof=1)),
                'mean_eliminated':float(np.mean([x['eliminated_masks'] for x in rows])),
                'positive_coverage_frequency':float(np.mean([x['certified_coverage']>0 for x in rows])),
                'unsafe_certifications':int(sum(x['unsafe_certification'] for x in rows)),
                'stage_failures':int(sum(x['stage_failure'] for x in rows)),
                'oracle_coverage':float(rows[0]['oracle_coverage']),
            })
    with (out/'adaptive_overlap_summary_v17.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    data={'protocol':{'seed':seed,'repeats':repeats,'H':H,'b':b,'tau':.15,'gamma':.25,
                      'delta_cert':.025,'delta_arrival':.025,'family':FAMILY,
                      'independence':'paired states/losses across methods; same fixed design; separate random uniform index selection',
                      'CP_confidence':'dyadic alpha-spending for adaptive; fixed Bonferroni CP for fixed uniform'},
          'worlds':WORLDS,'design':design,'summary':summary,
          'sha256_csv':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'adaptive_overlap_summary_v17.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(records),'sha256':data['sha256_csv'],'design':design,'summary':summary},indent=2))
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data')
    parser.add_argument('--repeats',type=int,default=200)
    parser.add_argument('--H',type=int,default=30000)
    parser.add_argument('--seed',type=int,default=20261112)
    args=parser.parse_args()
    run(Path(args.output),repeats=args.repeats,H=args.H,seed=args.seed)
