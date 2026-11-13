from pathlib import Path
import json
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
x=json.loads((root/'data/overlap_summary_v16.json').read_text())
fig,ax=plt.subplots(figsize=(6.2,3.2))
c=x['design_closed_form']
labels=['Shared optimized','Shared uniform','Isolated pools']
values=[c['shared_total'],c['uniform_shared_total'],c['isolated_total']]
bars=ax.bar(labels,values)
for bar,v in zip(bars,values):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+35,f'{v:.0f}',ha='center',va='bottom',fontsize=10)
ax.set_ylabel('Continuous label budget')
ax.set_ylim(0,4050)
ax.set_title('Overlapping policy design: two masks, three states')
fig.tight_layout()
fig.savefig(root/'figures/overlap_allocation_v16.pdf')
fig.savefig(root/'figures/overlap_allocation_v16.png',dpi=150)
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,3.6))
worlds=['mixed_safe_unsafe','boundary_safe','all_unsafe']
for method in ['shared-hoeffding','cp-box']:
 for n in [1000,4000]:
  vals=[next(z['mean_certified_coverage'] for z in x['summary'] if z['world']==w and z['n_per_state']==n and z['method']==method) for w in worlds]
  ax.plot(range(3),vals,marker='o',label=f'{method}, n_i={n}')
ax.set_xticks(range(3),['Mixed','Near threshold','All unsafe'])
ax.set_ylim(0,0.82)
ax.set_ylabel('Mean certified population coverage')
ax.set_title('Synthetically queried labels: 300 seeds / world')
ax.legend(fontsize=8,ncol=2)
fig.tight_layout()
fig.savefig(root/'figures/overlap_recovery_v16.pdf')
fig.savefig(root/'figures/overlap_recovery_v16.png',dpi=150)
plt.close(fig)
