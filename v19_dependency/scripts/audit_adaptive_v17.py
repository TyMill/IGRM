"""Independent recomputation of main v1.7 tables from the exported row CSV."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def audit() -> dict:
    data=ROOT/'data/adaptive_overlap_v17.csv'
    p=pd.read_csv(data,keep_default_na=False)
    summary=json.loads((ROOT/'data/adaptive_overlap_summary_v17.json').read_text())
    worlds=tuple(summary['worlds'])
    strategies=tuple(s['strategy'] for s in summary['summary'] if s['world']==worlds[0])
    repeats=summary['protocol']['repeats']
    assert len(p)==len(worlds)*len(strategies)*repeats
    assert p.groupby(['world','strategy']).size().eq(repeats).all()
    assert p.unsafe_certification.sum()==0
    assert p.stage_failure.sum()==0
    assert p.labels.le(p.full_budget).all()
    for obj in summary['summary']:
        sub=p[(p.world==obj['world'])&(p.strategy==obj['strategy'])]
        for col,typ in [('certified_coverage','mean_coverage'),('labels','mean_labels'),
                        ('arrivals_until_decision','mean_arrivals_until_decision')]:
            assert abs(sub[col].mean()-obj[typ])<1e-12
        assert int(sub.unsafe_certification.sum())==obj['unsafe_certifications']
    gap_means={}
    for w in worlds:
        sub=p[p.world==w]
        piv=sub.pivot(index='repeat',columns='strategy',values='labels')
        masks=sub.pivot(index='repeat',columns='strategy',values='selected_mask')
        assert (masks.adaptive_cp==masks.state_blind_dyadic_cp).all(),w
        d=piv.state_blind_dyadic_cp-piv.adaptive_cp
        assert d.ge(0).all()
        gap_means[w]=round(float(d.mean()),4)
    return {'csv_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
            'canonical_v10_sha256':hashlib.sha256((ROOT/'data/final_v10_30seeds.csv').read_bytes()).hexdigest(),
            'rows':len(p),'unique_streams':len(worlds)*repeats,
            'unsafe_certifications':int(p.unsafe_certification.sum()),
            'stage_failures':int(p.stage_failure.sum()),
            'same_masks_adaptive_vs_stateblind':True,'mean_saved_labels':gap_means}

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
