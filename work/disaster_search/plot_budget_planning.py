"""Descriptive planning controls and public utility-resolution diagnostic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_xbd_mini import atomic_json,sha


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--previous",type=Path,required=True)
    args=parser.parse_args()
    new=json.loads((args.run/"summary.json").read_text())["event_macro_means"]
    old=json.loads((args.previous/"summary.json").read_text())["event_macro_means"]
    def value(policy,scope="development_fit",cost=2):
        return next(r["distinct_found"] for r in new+old if r["policy"]==policy and r["fit_scope"]==scope and r["scenario"]=="mixed_survey" and r["upgrade_cost"]==cost and r["budget"]==32 and r["threshold"]==.1)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    policies=["proxy_greedy","two_step","batch_tail","prior_survey"]
    x=list(range(4));width=.36
    axes[0].bar([i-width/2 for i in x],[value(p) for p in policies],width,label="All development fit",color="#94a3b8")
    axes[0].bar([i+width/2 for i in x],[value(p,scope="event_holdout") for p in policies],width,label="Other 2 dev events fit",color="#334155")
    axes[0].set(xticks=x,xticklabels=["Greedy","Two-step","Batch tail","Raw change"],ylabel="Distinct buildings / scene (event macro)",title="A  Primary setting: budget32, cost2")
    axes[0].legend(frameon=False,fontsize=8,loc="lower left")
    for policy,label,color in [("proxy_greedy","Greedy count","#dc2626"),("two_step","Two-step","#d97706"),("batch_tail","Batch tail","#2563eb"),("prior_survey","Raw change survey","#475569")]:
        axes[1].plot([1,2,4],[value(policy,cost=c) for c in [1,2,4]],marker="o",label=label,color=color,linewidth=2)
    axes[1].set(xticks=[1,2,4],xlabel="Upgrade cost",title="B  Cost sensitivity; budget32")
    axes[1].legend(frameon=False,fontsize=8,loc="lower left")
    diagnostic=json.loads((args.run/"utility_resolution.json").read_text())
    ties=[r["top_score_tie_cells"] for r in diagnostic["per_scene"] if r["fit_scope"]=="development_fit"]
    axes[2].hist(ties,bins=[0,8,16,24,32,40,48,56,64],color="#0f766e",edgecolor="white")
    axes[2].axvline(sum(ties)/len(ties),color="#111827",linestyle="--",label="Mean = 16.1 cells")
    axes[2].set(xlabel="Cells sharing the highest initial count utility",ylabel="Development scenes",title="C  Public score ties (post-hoc diagnostic)")
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.suptitle("Budget planning improves its greedy proxy, but does not close the simple-ranking gap",fontsize=13)
    directory=args.run/"figures";directory.mkdir(exist_ok=True)
    fig.savefig(directory/"planning_and_resolution.png",dpi=180);fig.savefig(directory/"planning_and_resolution.pdf");plt.close(fig)
    atomic_json(directory/"figure_receipt.json",{"runner_sha256":sha(Path(__file__).read_bytes()),
        "inputs_sha256":{str(p):sha(p.read_bytes()) for p in [args.run/"summary.json",args.previous/"summary.json",args.run/"utility_resolution.json"]},
        "scope":"Development only. Mixed quality, threshold10%. No error bars or independent-event inference. Ties panel uses no outcome labels.",
        "files":{p.name:sha(p.read_bytes()) for p in directory.iterdir() if p.suffix in {".pdf",".png"}}})


if __name__=="__main__":main()
