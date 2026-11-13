"""Simple vector graphics from the frozen held-out audit CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data/mask_pooled_summary_v19.csv')
worlds=['near_boundary','near_boundary_asymmetric']
methods=['statewise_v18','statewise_fixed_cp_4096','mask_sequential_v19','mask_fixed_cp_4096']
labels=['Statewise seq.','Statewise fixed','Mask seq.','Mask fixed']
vals=np.array([[float(d[(d.world==w)&(d.method==m)].mean_coverage.iloc[0])*100 for m in methods] for w in worlds])
x=np.arange(len(worlds));width=.18
fig,ax=plt.subplots(figsize=(7.2,3.15))
for j,name in enumerate(labels):
    ax.bar(x+(j-1.5)*width,vals[:,j],width,label=name)
ax.set_xticks(x,['Near boundary','Asymmetric near boundary'])
ax.set_ylabel('Certified population coverage (%)')
ax.set_ylim(0,31)
ax.legend(loc='upper left',ncol=2,fontsize=8,frameon=False)
ax.grid(axis='y',alpha=.18)
fig.tight_layout()
for ext in ('png','pdf'):
    fig.savefig(ROOT/'figures'/f'mask_near_boundary_v19.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
# Label cost comparison (easy scenarios)
worlds=['separated','all_unsafe','rare_safe']
methods=['statewise_v18','mask_sequential_allstates_v19','mask_sequential_v19']
labels=['Statewise v1.8','Mask seq. all states','Mask seq. active support']
vals=np.array([[float(d[(d.world==w)&(d.method==m)].mean_labels.iloc[0]) for m in methods] for w in worlds])
x=np.arange(len(worlds));width=.25
fig,ax=plt.subplots(figsize=(7.2,3.1))
for j,name in enumerate(labels):
    ax.bar(x+(j-1)*width,vals[:,j],width,label=name)
ax.set_xticks(x,['Separated risks','All unsafe','Rare safe subset'])
ax.set_ylabel('Average queried labels')
ax.set_ylim(0,3400)
ax.legend(loc='upper right',fontsize=8,frameon=False)
ax.grid(axis='y',alpha=.18)
fig.tight_layout()
for ext in ('png','pdf'):
    fig.savefig(ROOT/'figures'/f'mask_label_cost_v19.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
print('plots saved')
