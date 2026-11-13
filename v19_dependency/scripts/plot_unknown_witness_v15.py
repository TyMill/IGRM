from pathlib import Path
import json
import matplotlib.pyplot as plt
from scripts.unknown_witness_v15 import design_unknown_witness, lower_horizons

folder = Path(__file__).resolve().parents[1]
d=design_unknown_witness()
lo,uni=lower_horizons(d)
labels=['General information\nnecessary','Known witness\nsufficient (v1.4)','Unknown witness\nsufficient (v1.5)','State-blind random\nnecessary']
vals=[lo,1710,d.n,uni]
fig,ax=plt.subplots(figsize=(8.6,4.1))
bars=ax.bar(range(4),vals,width=.68,edgecolor='black',linewidth=.6)
ax.set_xticks(range(4),labels)
ax.set_ylabel('Target stream horizon (observations)')
ax.set_title('Unknown-witness identification with a 20% label budget')
ax.set_ylim(0,4750)
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+75,f'{v:,}',ha='center',va='bottom',fontsize=10)
ax.grid(axis='y',alpha=.19)
fig.text(.5,.005,'K=8; alpha=0.05; beta=0.10. Different bounds and constructions, not comparable algorithms.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(folder/'figures/unknown_witness_v15.pdf');fig.savefig(folder/'figures/unknown_witness_v15.png',dpi=180)
plt.close(fig)
