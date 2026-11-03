"""Evidence/stop decision and standalone plots for the external reference pilot."""
from collections import defaultdict
import json
from pathlib import Path

from build_xbd_mini import atomic_json,sha


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);args=p.parse_args();root=args.root
    get=lambda n:json.loads((root/n).read_text())
    rows=get("sensor_area_opportunity.json");sensor=[]
    for event in sorted({r["event"] for r in rows}):
        part=[r for r in rows if r["event"]==event and r["domain"]=="all_published_pixels"]
        ref=sum(r["reference_pixels"] for r in part)
        sensor.append({"event":event,"scenes":len(part),"reference_positive_scenes":sum(r["reference_pixels"]>0 for r in part),"reference_pixels":ref,
            "native_covered_reference_pixels":sum(r["native"]["covered_reference_pixels"] for r in part),
            "blurred_covered_reference_pixels":sum(r["blurred"]["covered_reference_pixels"] for r in part),
            "either_mode_positive_scenes":sum(r["either_mode"]["covered_reference_pixels"]>0 for r in part)})
    summary=get("area_summary.json");main=[r for r in summary["event_macro_rows"] if r["budget"]==8 and r["domain"]=="all_published_pixels"]
    by_policy={r["policy"]:r for r in main};ratio=by_policy["prior_survey"]["covered_reference_pixels"]/by_policy["random_exact_expectation"]["covered_reference_pixels"]
    evidence={"sensor_by_event":sensor,"primary_macro_per_scene":main,"raw_survey_over_random_exact_area_ratio":ratio,
        "detectable_reference_events":sum(r["native_covered_reference_pixels"]+r["blurred_covered_reference_pixels"]>0 for r in sensor),
        "detectable_reference_scenes":sum(r["either_mode_positive_scenes"] for r in sensor),
        "reference_positive_scenes":sum(r["reference_positive_scenes"] for r in sensor),
        "reference_recall_is_not_physical_damage_accuracy":True,"new_method_selected":False}
    atomic_json(root/"evidence_summary.json",evidence)
    atomic_json(root/"next_step_decision.json",{"decision":"STOP_SEARCH_STRATEGY_TUNING_ON_THIS_SENSOR_REFERENCE_COMBINATION",
        "reason":"Only1of3events and1of18scenes has any fixed-cache overlap with the published severe reference; all blurred overlap is zero. The pilot cannot support a cross-disaster search-method claim.",
        "current_controls":"Raw survey and negative gate coincide in main area reward and are below exact random expectation. Do not turn random finite-order variation into an algorithm contribution.",
        "next":"Offline fixed-weight input-context/scale and reference-correspondence diagnostic before any new search method, paid GPU or target calibration.",
        "human_annotation_resources":"Unavailable; keep published-reference proxy language, no fabricated instance gold.",
        "old_outputs_preserved":True,"new_fit_enabled":False,"new_search_variant_enabled":False})
    atomic_json(root/"prospective_context_diagnostic.json",{"status":"PROSPECTIVE_NOT_EXECUTED",
        "question":"Does the128px crop interface, rather than a usable search rule, explain loss of published-reference overlap?",
        "fixed_cohort":"All18 pairs, including all zero-reference and zero-opportunity scenes; no favorable-scene replacement.",
        "stage1":"Same fixed checkpoint on whole512x512 native pairs,18forward calls. Compare full-scene output with mosaic of16cached native128px crops and their reference overlaps.",
        "positive_control":"Before blaming domain shift, reproduce a prespecified already-used development crop whose old canonical prediction was positive; verify crop/tensor/code/model lineage. Not a new evaluation sample.",
        "information_warning":"Whole512px pairs contain unpurchased high-resolution context. This is an offline diagnostic with full-scene information, not a legal budget8 search policy or a claimed cost saving.",
        "possible_followup":"Only if evidence supports an input-context effect, preregister a separate comparison of128-to512 scaling and free-coarse-context plus paid-native-cell input. Do not train or tune to these labels.",
        "interpretation":"A difference can implicate context/scale; it does not independently validate reference semantics or establish a novel method.",
        "source_evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),"paid_gpu_or_training":False})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.8),layout="constrained")
    names=["Turkey","Pakistan","Texas"]
    native=[100*r["native_covered_reference_pixels"]/r["reference_pixels"] if r["reference_pixels"] else 0 for r in sensor]
    axes[0].bar(names,native,color=["#537d90","#537d90","#bdc5c8"])
    for i,r in enumerate(sensor):axes[0].text(i,native[i]+1.1,f"{native[i]:.2f}%" if r["reference_pixels"] else "No reference target",ha="center",fontsize=8)
    axes[0].set_ylim(0,45);axes[0].set_ylabel("Published severe pixels covered (%)");axes[0].set_title("Full native-crop coverage opportunity")
    palette={"random_exact_expectation":"#26828e","random_survey":"#8bb7ba","prior_survey":"#b55737","prior_gate":"#82619b"}
    labels={"random_exact_expectation":"Random exact expectation","random_survey":"Random, 10 fixed orders","prior_survey":"RGB change survey","prior_gate":"RGB negative gate"}
    for policy in ["random_exact_expectation","random_survey","prior_survey","prior_gate"]:
        part=sorted([r for r in summary["event_macro_rows"] if r["domain"]=="all_published_pixels" and r["policy"]==policy],key=lambda r:r["budget"])
        axes[1].plot([r["budget"] for r in part],[r["covered_reference_pixels"] for r in part],label=labels[policy],color=palette[policy],marker="o",linestyle="--" if policy in ["prior_gate","random_survey"] else "-")
    axes[1].set_xticks([4,8,16]);axes[1].set_xlabel("Observation cost budget");axes[1].set_ylabel("Covered reference pixels / scene");axes[1].set_title("Three-event macro, all18 scenes");axes[1].legend(fontsize=7,loc="upper left")
    shown=["random_exact_expectation","prior_survey","prior_gate"]
    axes[2].bar(["Random\nexact","RGB\nsurvey","RGB\ngate"],[by_policy[k]["reference_false_positive_pixels"] for k in shown],color=[palette[k] for k in shown])
    axes[2].set_ylabel("Reference-relative false pixels / scene");axes[2].set_title("Primary budget 8, upgrade cost 2")
    for ax in axes:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("EBD reference-area feasibility: 1 detectable scene; descriptive proxy only",fontsize=12)
    folder=root/"figures";folder.mkdir(exist_ok=True)
    for suffix in ["png","pdf"]:fig.savefig(folder/("area_feasibility."+suffix),dpi=180)
    plt.close(fig)
    atomic_json(folder/"figure_receipt.json",{"source_summary_sha256":sha((root/"area_summary.json").read_bytes()),
        "evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),"generator_sha256":sha(Path(__file__).read_bytes()),
        "plots_not_independent_efficacy":True})
    print(json.dumps({"detectable_events":evidence["detectable_reference_events"],"detectable_scenes":evidence["detectable_reference_scenes"],"raw_over_random_exact":ratio}))


if __name__=="__main__":main()
