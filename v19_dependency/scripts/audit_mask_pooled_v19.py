"""Self-contained deterministic audit of held-out v1.9 matched experiment."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]

def run():
    d=pd.read_csv(ROOT/'data/mask_pooled_heldout_v19.csv')
    required={'world','rep','method','coverage','labels','unsafe','oracle_coverage'}
    assert required<=set(d)
    assert len(d)==9800
    assert not d.duplicated(['world','rep','method']).any()
    assert d.groupby('world').rep.nunique().eq(200).all()
    assert d.groupby(['world','rep']).method.nunique().eq(7).all()
    assert (d.labels>=0).all() and (d.labels<=6000).all()
    assert (d.query("method != 'mask_fixed_cp_6000'").labels<=4096).all()
    assert not d.unsafe.any()
    assert (d.query('coverage > 0').coverage>=.25-1e-12).all()
    assert (d.coverage<=d.oracle_coverage+1e-12).all()
    summary=d.groupby(['world','method'],sort=True).agg(
        mean_coverage=('coverage','mean'),mean_labels=('labels','mean'),
        coverage_positive_rate=('coverage',lambda x:float((x>0).mean())),
        unsafe_count=('unsafe','sum'),n=('rep','size')).reset_index()
    summary.to_csv(ROOT/'data/mask_pooled_summary_v19.csv',index=False,float_format='%.17g')
    pairs=[]
    for world in sorted(d.world.unique()):
        pivot=d[d.world==world].pivot(index='rep',columns='method',values=['coverage','labels'])
        for metric,baseline in [('coverage','statewise_v18'),
                ('coverage','statewise_fixed_cp_4096'),
                ('coverage','mask_fixed_cp_4096'),
                ('labels','statewise_v18'),
                ('labels','mask_sequential_allstates_v19')]:
            x=pivot[(metric,'mask_sequential_v19')]-pivot[(metric,baseline)]
            n=len(x);m=float(x.mean());sem=float(x.std(ddof=1)/np.sqrt(n));crit=float(t.ppf(.975,n-1))
            pairs.append({'world':world,'metric':metric,'baseline':baseline,
                'mean_difference':m,'lower_95_descriptive_t':m-crit*sem,
                'upper_95_descriptive_t':m+crit*sem,
                'wins':int((x>0).sum()),'losses':int((x<0).sum()),
                'ties':int((x==0).sum()),'repeats':n})
    pd.DataFrame(pairs).to_csv(ROOT/'data/mask_pooled_paired_v19.csv',index=False,float_format='%.17g')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    frozen={'v10_mac':'38ea2b41b12986ac200828a72a53b0afd7d5732ff921f3ed8b0e2bf89ed31c43',
            'v17':'11933471b900b22361ecfba2706fee916af71539152a5ce336dd1d27e9f5149e',
            'v18':'6ec07469958f91009ed81bb0692fab4ea55bdbddbb7a670ce2624ac81b7884f2'}
    for key,file in [('v10_mac','final_v10_30seeds.csv'),('v17','adaptive_overlap_v17.csv'),('v18','unknown_q_v18.csv')]:
        assert sha(ROOT/'data'/file)==frozen[key]
    report={'row_count':len(d),'worlds':sorted(d.world.unique()),'repeats_per_world':200,
        'methods_per_seed':7,'unsafe_count':int(d.unsafe.sum()),
        'sha256_heldout':sha(ROOT/'data/mask_pooled_heldout_v19.csv'),
        'sha256_summary':sha(ROOT/'data/mask_pooled_summary_v19.csv'),
        'sha256_paired':sha(ROOT/'data/mask_pooled_paired_v19.csv'),
        'frozen_shas':frozen,'notes':["Paired t intervals are descriptive across synthetic seeds, not familywise simultaneous inference.",
        "Results are comparisons in finite synthetic worlds, not proof of new statistical theory.",
        "A mask requires full within-mask querying while active; arbitrary VOI state-skewed querying invalidates direct pooled Bernoulli CP."]}
    (ROOT/'data/mask_pooled_audit_v19.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':run()
