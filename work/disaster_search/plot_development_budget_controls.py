"""Render descriptive development controls; no confidence or novelty claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_xbd_mini import atomic_json,sha


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads((args.run/"summary.json").read_text())
    rows=summary["event_macro_means"]
    def value(policy,scope="development_fit",cost=2,budget=32,threshold=.1):
        return next(r["distinct_found"] for r in rows if r["policy"]==policy and r["fit_scope"]==scope and r["scenario"]=="mixed_survey" and r["upgrade_cost"]==cost and r["budget"]==budget and r["threshold"]==threshold)
    labels={"prior_survey":"Raw change, survey only","count_survey":"Count, survey only",
            "count_negative":"Count, negative upgrade only","count_all":"Count, all upgrades"}
    colors={"prior_survey":"#64748b","count_survey":"#2563eb","count_negative":"#059669","count_all":"#dc2626"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figure,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    for policy in labels:
        axes[0].plot([8,16,32],[value(policy,budget=b) for b in [8,16,32]],marker="o",label=labels[policy],color=colors[policy],linewidth=2)
        axes[1].plot([1,2,4],[value(policy,cost=c) for c in [1,2,4]],marker="o",color=colors[policy],linewidth=2)
    axes[0].set(xlabel="Observation budget",ylabel="Distinct buildings per scene (event macro)",title="A  Budget curve; upgrade cost = 2",xticks=[8,16,32])
    axes[1].set(xlabel="Native upgrade cost",title="B  Cost sensitivity; budget = 32",xticks=[1,2,4])
    policies=["prior_survey","count_survey","count_negative","count_all"]
    x=list(range(len(policies)));width=.35
    axes[2].bar([i-width/2 for i in x],[value(p) for p in policies],width,label="Fit on all 72",color="#94a3b8")
    axes[2].bar([i+width/2 for i in x],[value(p,scope="event_holdout") for p in policies],width,label="Fit on other 2 dev events",color="#334155")
    axes[2].set(xticks=x,xticklabels=["Raw\nchange","Count\nsurvey","Count\nnegative","Count\nall"],title="C  Development event transfer")
    axes[2].legend(frameon=False,fontsize=8,loc="lower left")
    axes[0].legend(frameon=False,fontsize=8,loc="upper left")
    for ax in axes:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.set_ylim(bottom=0)
    figure.suptitle("Development diagnostics: revisits find buildings but can lose net budget value",fontsize=13)
    directory=args.run/"figures";directory.mkdir(exist_ok=True)
    figure.savefig(directory/"budget_and_cost.png",dpi=180)
    figure.savefig(directory/"budget_and_cost.pdf")
    plt.close(figure)
    atomic_json(directory/"figure_receipt.json",{"source_summary_sha256":sha((args.run/"summary.json").read_bytes()),
        "runner_sha256":sha(Path(__file__).read_bytes()),"scope":"72 development scenes, mixed quality, threshold10%; no independent validation or confidence intervals.",
        "aggregation":"Repeated seeds averaged within scenes; scene means within events, then three events equally. Mexico retained with zero matches.",
        "files":{p.name:sha(p.read_bytes()) for p in directory.iterdir() if p.suffix in {".png",".pdf"}}})


if __name__=="__main__":main()
