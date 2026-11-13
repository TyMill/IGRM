from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
figs=R/'figures';figs.mkdir(exist_ok=True)
d=pd.read_csv(R/'results/summary_7_methods.csv')
scens=['near_boundary','near_boundary_asymmetric']
meth=['mask_cp_sequential_v19','hybrid_betting_terminal_cp','mask_e_predictable_betting','mask_cp_fixed_4096','fixed_mask_holm']
labels=['Seq. CP v1.9','Hybrid bet + CP','Betting e-process','Fixed CP','Fixed Holm']
fig,ax=plt.subplots(figsize=(9,4.6));import numpy as np
X=np.arange(len(scens));width=.15
for j,(m,lbl) in enumerate(zip(meth,labels)):
 vals=[float(d[(d.world==s)&(d.method==m)].mean_coverage.iloc[0])*100 for s in scens]
 ax.bar(X+(j-2)*width,vals,width,label=lbl)
ax.set_xticks(X,['Symmetric near-boundary','Asymmetric near-boundary']);ax.set_ylabel('Mean certified population coverage (%)');ax.set_title('Stronger synthetic benchmarks: n=200 paired streams per scenario');ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figs/'synthetic_coverage.png',dpi=160);plt.close(fig)
d=pd.read_csv(R/'results/diamonds/diamonds_summary.csv');meth=['seq_cp_v19','hybrid_e_cp','e_betting','fixed_cp','fixed_holm'];names=['Seq. CP v1.9','Hybrid bet + CP','Betting e-process','Fixed CP (2048)','Fixed Holm (2048)']
fig,ax=plt.subplots(figsize=(9,4.6));ax.bar(names,[float(d[d.method==m].mean_heldout_coverage.iloc[0])*100 for m in meth]);ax.set_ylabel('Mean selected heldout coverage (%)');ax.set_title('Diamonds: 13,485 real held-out labels, 24 order permutations');ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figs/'diamonds_coverage.png',dpi=160);plt.close(fig)
