from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
p=Path('data/unknown_q_summary_v18.csv'); d=pd.read_csv(p)
names=['separated','multiple_safe','near_boundary','all_unsafe','rare_safe','rare_irrelevant']
labels=['Separated','Multiple safe','Near threshold','All unsafe','Rare safe','Rare irrelevant']
styles=[('known_q_adaptive','Known q: adaptive'),('estimated_q_adaptive','Estimated q: adaptive'),('known_q_fixed_cp','Known q: fixed CP')]
fig,ax=plt.subplots(figsize=(10.4,4.1))
w=.26
for idx,(key,name) in enumerate(styles):
    y=[float(d[(d.world==world)&(d.method==key)].mean_labels.iloc[0]) for world in names]
    ax.bar([j+(idx-1)*w for j in range(len(names))],y,width=w,label=name)
ax.set_xticks(range(len(labels)),labels,rotation=17,ha='right')
ax.set_ylabel('Mean queried labels');ax.set_ylim(0,6800)
ax.legend(loc='upper right',fontsize=8,frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig('figures/unknown_q_label_cost_v18.pdf');fig.savefig('figures/unknown_q_label_cost_v18.png',dpi=170)
plt.close(fig)
