"""Render a monochrome boundary figure and compact reusable result table."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"output/disaster_route_inspection_information_v1"
records=json.loads((OUT/"results.json").read_text())
lookup={(r['rho'],r['q'],r['budget'],r['fee'],r['policy']):r['exact']['reward'] for r in records}
fig,axes=plt.subplots(1,2,figsize=(8.4,3.1),layout="constrained")
styles=[(0,":"),(.5,"--"),(1,"-")]
for rho,style in styles:
    qs=[.5,.65,.8,1]
    gains=[lookup[(rho,q,10,1,'best_static_or_one_update')]-lookup[(rho,q,10,1,'static')] for q in qs]
    axes[0].plot(qs,gains,color="black",linestyle=style,marker="o",markerfacecolor="white",markersize=4,label=f"rho = {rho:g}")
    fees=[0,1,2,3]
    differences=[lookup[(rho,1,12,fee,'forced_one_update')]-lookup[(rho,1,12,fee,'static')] for fee in fees]
    axes[1].plot(fees,differences,color="black",linestyle=style,marker="o",markerfacecolor="white",markersize=4,label=f"rho = {rho:g}")
axes[0].set(title="Optional update: budget 10, fee 1",xlabel="Known sensor correctness q",ylabel="Expected detection gain over static",xticks=[.5,.65,.8,1],ylim=(-.03,.53))
axes[1].set(title="Forced update: budget 12, q = 1",xlabel="Extra update fee (proxy units)",ylabel="Expected detection gain over static",xticks=[0,1,2,3],ylim=(-.54,.05))
for ax in axes:
    ax.axhline(0,color="black",linewidth=.5)
    ax.legend(frameon=False,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
(OUT/"figures").mkdir(exist_ok=True)
fig.savefig(OUT/"figures/information_boundary.png",dpi=220,facecolor="white")
fig.savefig(OUT/"figures/information_boundary.pdf",facecolor="white")
plt.close(fig)
with (OUT/"results.csv").open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['rho','q','budget','fee','policy','expected_correct_detections','mean_observations','mean_cost'])
    writer.writeheader()
    for r in records:
        writer.writerow({k:r[k] for k in ['rho','q','budget','fee','policy']}|{
            'expected_correct_detections':r['exact']['reward'],'mean_observations':r['exact']['observations'],'mean_cost':r['exact']['cost']})
print(json.dumps({'rows':len(records),'figure':str(OUT/'figures/information_boundary.png')}))
