"""Compact publication figures built only from frozen v1.7 summary CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
s=pd.read_csv(ROOT/'data/adaptive_overlap_summary_v17.csv')
scenarios=['separated','multiple_safe','near_boundary','all_unsafe','rare_safe']
labels=['Separated','Several safe','Near boundary','All unsafe','Rare safe']
methods=['adaptive_cp','uniform_cp_matched_equal_delta','uniform_cp_full_equal_delta']
method_labels=['Adaptive CP','Fixed CP (4096, same level)','Fixed CP (6000, same level)']

fig,ax=plt.subplots(figsize=(8.9,3.8))
x=np.arange(len(scenarios));width=.24
for i,(method,label) in enumerate(zip(methods,method_labels)):
    y=[100*s[(s.world==w)&(s.strategy==method)].iloc[0].mean_coverage for w in scenarios]
    ax.bar(x+(i-1)*width,y,width,label=label)
ax.set_xticks(x,labels)
ax.set_ylabel('Mean certified population coverage (%)')
ax.set_ylim(0,83)
ax.set_yticks([0,20,40,60,80])
ax.grid(axis='y',alpha=.18)
ax.set_axisbelow(True)
ax.legend(ncol=3,fontsize=8,frameon=False,loc='upper right')
fig.tight_layout(pad=.6)
for suffix in ['pdf','png']:
    fig.savefig(ROOT/'figures'/f'adaptive_coverage_v17.{suffix}',bbox_inches='tight',dpi=220)
plt.close(fig)

fig,ax=plt.subplots(figsize=(8.9,3.8))
cost_methods=['adaptive_cp','state_blind_dyadic_cp','uniform_cp_matched','uniform_cp_full']
cost_labels=['Adaptive CP','State-blind dyadic CP','Fixed CP (4096)','Fixed CP (6000)']
width=.19
for i,(method,label) in enumerate(zip(cost_methods,cost_labels)):
    y=[s[(s.world==w)&(s.strategy==method)].iloc[0].mean_labels for w in scenarios]
    ax.bar(x+(i-1.5)*width,y,width,label=label)
ax.set_xticks(x,labels)
ax.set_ylabel('Mean queried labels')
ax.set_ylim(0,6750)
ax.set_yticks([0,2000,4000,6000])
ax.grid(axis='y',alpha=.18)
ax.set_axisbelow(True)
ax.legend(ncol=2,fontsize=8,frameon=False,loc='upper right')
fig.tight_layout(pad=.6)
for suffix in ['pdf','png']:
    fig.savefig(ROOT/'figures'/f'adaptive_label_cost_v17.{suffix}',bbox_inches='tight',dpi=220)
plt.close(fig)
