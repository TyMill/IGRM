"""Large public, real-label case study using plotnine's 53,940 diamonds.

Original labels: observed real sale prices, frozen regression model; no synthetic
Bernoulli draws and no bootstrap with replacement. One held-out 13,485-item
finite evaluation pool; 24 permutations are *order-sensitivity replicates* of
the same fixed pool, NOT independent scientific experiments.
"""
from pathlib import Path
import os,sys,hashlib,json,time
import pandas as pd
import numpy as np
from plotnine.data import diamonds
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT),str(ROOT/'v19_dependency')]
from scripts.run_mask_pooled_v19 import FAMILY
from scripts.mask_pooled_v19 import run_mask_pooled,fixed_mask_pooled
from bench_methods import run_mask_eprocess
from stronger_baselines import run_hybrid_betting_terminal_cp,fixed_mask_holm

LOSS_THRESHOLD=.15
TAU=.15
PILOT=2048
BUDGET=2048
REPEATS=24
METHOD_NAMES=('seq_cp_v19','e_betting','e_mixture','hybrid_e_cp','fixed_cp','fixed_holm')

def run():
 tic=time.time();out=ROOT/'results'/'diamonds';out.mkdir(parents=True,exist_ok=True)
 df=diamonds.copy();X=pd.get_dummies(df.drop(columns='price'),columns=['cut','color','clarity'],dtype=float)
 y=np.log(df['price'].to_numpy().astype(float))
 idx=np.arange(len(df))
 tr,tmp=train_test_split(idx,test_size=.35,random_state=20261113)
 dv,ev=train_test_split(tmp,test_size=5/7,random_state=20261114)
 model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=18,max_depth=6,
      learning_rate=.08,l2_regularization=5,random_state=20261113)
 model.fit(X.iloc[tr],y[tr])
 pred_dev=model.predict(X.iloc[dv]);pred_ev=model.predict(X.iloc[ev])
 edges=np.quantile(pred_dev,[.25,.50,.75])
 state=np.searchsorted(edges,pred_ev,side='right').astype(np.int8)
 losses=(np.abs(pred_ev-y[ev])>LOSS_THRESHOLD).astype(np.int8)
 q=np.bincount(state,minlength=4).astype(float)/len(state)
 r=np.bincount(state,weights=losses,minlength=4)/np.bincount(state,minlength=4)
 oracle=max((float(q[list(a)].sum()) for a in FAMILY if np.sum(q[list(a)]*(r[list(a)]-TAU))<=0),default=0.)
 pd.DataFrame({'heldout_row_id':ev,'state':state,'loss':losses,
               'predicted_log_price':pred_ev,'true_log_price':y[ev]}).to_csv(out/'frozen_heldout_annotations.csv',index=False)
 rows=[]
 for k in range(REPEATS):
  rng=np.random.default_rng(891740+k*31)
  order=rng.permutation(len(state))
  s=state[order];z=losses[order]
  fixed_seed=82000+k
  runs=(('seq_cp_v19',run_mask_pooled(s,z,4,FAMILY,pilot_n=PILOT,max_labels=BUDGET)),
        ('e_betting',run_mask_eprocess(s,z,4,FAMILY,mode='betting',pilot_n=PILOT,max_labels=BUDGET)),
        ('e_mixture',run_mask_eprocess(s,z,4,FAMILY,mode='mixture',pilot_n=PILOT,max_labels=BUDGET)),
        ('hybrid_e_cp',run_hybrid_betting_terminal_cp(s,z,4,FAMILY,pilot_n=PILOT,max_labels=BUDGET)),
        ('fixed_cp',fixed_mask_pooled(s,z,4,FAMILY,pilot_n=PILOT,n_labels=BUDGET,seed=fixed_seed)),
        ('fixed_holm',fixed_mask_holm(s,z,4,FAMILY,pilot_n=PILOT,n_labels=BUDGET,seed=fixed_seed)))
  for method,res in runs:
   a=res.selected;cov=0 if not a else float(q[list(a)].sum());risk=np.nan if not a else float(np.dot(q[list(a)],r[list(a)])/cov)
   rows.append(dict(order_id=k,method=method,selected='' if a is None else ','.join(map(str,a)),
       selected_heldout_coverage=cov,selected_heldout_risk=risk,
       exceeds_heldout_threshold=int(bool(a) and risk>TAU),labels=res.labels,
       certified_lower_coverage=res.lower_coverage,oracle_empirical_coverage=oracle))
 results=pd.DataFrame(rows);results.to_csv(out/'diamonds_order_sensitivity.csv',index=False)
 summary=results.groupby('method').agg(n_orderings=('order_id','nunique'),
    mean_heldout_coverage=('selected_heldout_coverage','mean'),mean_labels=('labels','mean'),
    positive=('selected_heldout_coverage',lambda z:int((z>0).sum())),
    empirical_unsafe=('exceeds_heldout_threshold','sum')).reset_index()
 summary.to_csv(out/'diamonds_summary.csv',index=False)
 met={'source':'plotnine.data.diamonds, ggplot2 diamonds public dataset; no network access needed',
   'n_total':len(df),'n_train':len(tr),'n_dev':len(dv),'n_eval':len(ev),
   'source_file_sha256':hashlib.sha256(Path(__import__('plotnine').__file__).parent.joinpath('data/diamonds.csv').read_bytes()).hexdigest(),
   'model':'HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=18,max_depth=6,learning_rate=.08,l2_regularization=5,random_state=20261113)',
   'features':list(X.columns),'split_seed_train':20261113,'split_seed_dev':20261114,
   'error_metric':'binary indicator |log(actual_price)-log(predicted_price)|>.15',
   'loss_threshold':LOSS_THRESHOLD,'risk_threshold_tau':TAU,'gamma':.25,
   'dev_predicted_price_log_quartiles':edges.tolist(),
   'dev_error_rate':float(np.mean(np.abs(pred_dev-y[dv])>LOSS_THRESHOLD)),
   'heldout_log_mae':float(mean_absolute_error(y[ev],pred_ev)),
   'heldout_q_by_state':q.tolist(),'heldout_r_by_state':r.tolist(),
   'heldout_empirical_oracle_coverage':oracle,'pilot_n':PILOT,
   'label_cap':BUDGET,'delta_q':.025,'delta_cert':.025,
   'order_permutations':REPEATS,'permutation_seed_formula':'891740+31*k, k=0..23',
   'primary_order':'k=0','fixed_label_seed_formula':'82000+k',
   'limitation':'All 24 rows per method use the SAME 13,485 actual test records. Orderings are not statistically independent. Population-level iid guarantees do NOT automatically hold conditionally on this finite dataset. Sample was randomly split from an observational, non-temporal collection; this is no deployment or drift validation.',
   'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'frozen_heldout_annotations.csv',out/'diamonds_order_sensitivity.csv',out/'diamonds_summary.csv')}}
 (out/'PROTOCOL.json').write_text(json.dumps(met,indent=2))
 print('time',round(time.time()-tic,1),'s');print(json.dumps({k:v for k,v in met.items() if k not in ('features','model','limitation')},indent=2));print(summary.to_string(index=False))
 print('PRIMARY ORDER',results[results.order_id==0].to_string(index=False))
if __name__=='__main__':run()
