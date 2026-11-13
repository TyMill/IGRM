"""Locked, paired challenge of earlier benchmark with strong standard comparators."""
from pathlib import Path
import os,sys,csv,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT))
from stronger_baselines import fixed_mask_holm,run_hybrid_betting_terminal_cp
from scripts.run_mask_pooled_v19 import EXTENDED,FAMILY

METHODS=('fixed_mask_holm','hybrid_betting_terminal_cp')
def run(reps=200):
 rows=[];tic=time.time()
 for wid,(world,spec) in enumerate(EXTENDED.items()):
  q=np.asarray(spec['q'],float);r=np.asarray(spec['r'],float); K=len(q);fam=FAMILY if K==4 else ((0,1),)
  for rep in range(1000,1000+reps):
   seed=314159265+wid*1000003+31*rep
   rng=np.random.default_rng(seed)
   s=rng.choice(K,30000,p=q)
   y=(rng.random(30000)<r[s]).astype(np.int8)
   seedfix=314159265+wid*500001+rep
   methods=(('fixed_mask_holm',fixed_mask_holm(s,y,K,fam,seed=seedfix)),
            ('hybrid_betting_terminal_cp',run_hybrid_betting_terminal_cp(s,y,K,fam)))
   for meth,res in methods:
    a=res.selected
    cov=0 if a is None else float(q[list(a)].sum())
    risk='' if a is None else float(np.dot(q[list(a)],r[list(a)])/cov)
    rows.append(dict(world=world,rep=rep,method=meth,coverage=cov,labels=res.labels,
        selected='' if a is None else ','.join(map(str,a)),risk=risk,
        unsafe=int(bool(a) and risk>.15),oracle_coverage=max(
        (float(q[list(A)].sum()) for A in fam if np.sum(q[list(A)]*(r[list(A)]-.15))<=0),default=0.),
        pilot_n=4096,label_cap=4096,delta_q=.025,delta_cert=.025))
  print('completed',world,'elapsed',round(time.time()-tic,1),flush=True)
 out=ROOT/'results';out.mkdir(exist_ok=True)
 df=pd.DataFrame(rows); p=out/'stronger_200.csv';df.to_csv(p,index=False)
 prior=pd.read_csv(ROOT/'prior_benchmark/benchmark_paired.csv')
 joined=pd.concat([prior,df],ignore_index=True)
 assert len(joined)==len(EXTENDED)*reps*7
 assert joined.groupby(['world','rep','method']).size().max()==1
 joined.to_csv(out/'all_paired_7_methods.csv',index=False)
 summary=joined.groupby(['world','method']).agg(mean_coverage=('coverage','mean'),mean_labels=('labels','mean'),n=('rep','nunique'),unsafe=('unsafe','sum')).reset_index()
 summary.to_csv(out/'summary_7_methods.csv',index=False)
 sha=lambda pp:hashlib.sha256(pp.read_bytes()).hexdigest()
 manifest={'seed':'314159265+world_id*1000003+31*rep; rep=1000..1199',
 'datasets':'Seven existing synthetic world definitions from v1.9; no world-level external holdout',
 'method_family':'Existing five baselines + precommitted hybrid and exact Holm masks',
 'hard_label_cap':4096,'pilot_n':4096,'tau':.15,'gamma':.25,'delta_q':.025,
 'delta_cert':.025,'hybrid_early_share':.1,'hybrid_target_fraction':.9,
 'fixed_holm':'Holm stepdown exact one-sided Bernoulli binomial p-values for pilot-eligible masks',
 'limitations':'Two new methods chosen after looking at v1.9 summaries. No preregistered confirmatory tuning. Correlated replicate worlds.',
 'sha256':{a.name:sha(a) for a in [p,out/'all_paired_7_methods.csv',out/'summary_7_methods.csv']}}
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 for w in ['near_boundary','near_boundary_asymmetric','separated','rare_safe','rare_irrelevant']:
  print('\n',w);print(summary[summary.world==w].sort_values('mean_coverage',ascending=False).to_string(index=False))
 return summary
if __name__=='__main__':run()
