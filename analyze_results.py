from pathlib import Path
import hashlib
import numpy as np,pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent
src=ROOT/'results/all_paired_7_methods.csv'
df=pd.read_csv(src)
print('rows',len(df),'unique keys',df[['world','rep','method']].drop_duplicates().shape[0]);print('unsafe by method',df.groupby('method').unsafe.sum().to_dict());print('max labels',df.labels.max())
P=df.pivot(index=['world','rep'],columns='method',values='coverage')
L=df.pivot(index=['world','rep'],columns='method',values='labels')
comp=[('hybrid_betting_terminal_cp','mask_cp_sequential_v19'),('fixed_mask_holm','mask_cp_fixed_4096'),('fixed_mask_holm','hybrid_betting_terminal_cp')]
rows=[]
for world in df.world.unique():
 for x,y in comp:
  dif=(P.loc[world][x]-P.loc[world][y]).to_numpy()
  labeldiff=(L.loc[world][x]-L.loc[world][y]).to_numpy()
  n=len(dif);m=dif.mean();s=dif.std(ddof=1)
  h=stats.t.ppf(.975,n-1)*s/np.sqrt(n)
  rows.append(dict(world=world,first=x,second=y,mean_cov_diff_pp=100*m,
    descriptive_ci_lo_pp=100*(m-h),descriptive_ci_hi_pp=100*(m+h),
    wins=int((dif>1e-12).sum()),losses=int((dif < -1e-12).sum()),ties=int((abs(dif)<=1e-12).sum()),
    mean_label_diff=labeldiff.mean()))
res=pd.DataFrame(rows);res.to_csv(ROOT/'results/paired_effects.csv',index=False)
print(res[res.world.isin(['near_boundary','near_boundary_asymmetric','separated','rare_safe','rare_irrelevant'])].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print('SHA historical')
for f in ['/mnt/data/final_v10_30seeds.csv', '/mnt/data/IGRM_Benchmark_Review_Work/results/full_200/benchmark_paired.csv']:
 p=Path(f);print(p.name,hashlib.sha256(p.read_bytes()).hexdigest())
print('SHA strengthened',hashlib.sha256(src.read_bytes()).hexdigest())
