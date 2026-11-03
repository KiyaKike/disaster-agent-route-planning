"""Separate surrogate gains, action changes and real development discoveries."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_xbd_mini import atomic_json, sha
from run_utility_tie_controls import CONFIG_FIELDS


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.run/"protocol.json").read_text())
    audit=json.loads((args.run/"audit.json").read_text())
    report=json.loads((args.run/"report.json").read_text())
    if audit["stage"]!="SERIAL_TWO_STAGE_PRIMARY_FULL_REPLAY_PASSED":raise ValueError("Full replay required")
    summary=json.loads((args.run/"summary.json").read_text())
    simple_path=Path(protocol["sources"]["simple_scenes"]["path"]).with_name("summary.json")
    simple=json.loads(simple_path.read_text())
    def primary(r):return (r["scenario"],r["upgrade_cost"],r["budget"],r["threshold"])==("mixed_survey",2,32,.1)
    reference_policies=["prior_survey","prior_gate","count_survey","count_negative","count_all"]
    main_rows=[r for r in summary["event_macro_means"]+simple["event_macro_means"] if primary(r)
        and r["policy"] in reference_policies+["serial_one_stage","serial_two_stage"]]
    random_rows=[r for r in main_rows if r["tie_rule"]=="fixed_random"]
    lookup={(r["fit_scope"],r["policy"]):r for r in random_rows}
    effects=json.loads((args.run/"paired_effects.json").read_text())
    main_effects=[r for r in effects["event_macro_effects"] if r["threshold"]==.1]
    control={}
    action_counts=Counter()
    action_groups=defaultdict(Counter)
    with gzip.open(args.run/"trajectories.jsonl.gz","rt") as stream:
        for line in stream:
            row=json.loads(line)
            key=tuple(row[k] for k in CONFIG_FIELDS if k!="policy")
            if row["policy"]=="serial_one_stage":
                control[key]=row["trace"]
            else:
                original=control.pop(key)
                def actions(trace):return [{k:v for k,v in a.items() if k!="utility"} for a in trace]
                changed=actions(original)!=actions(row["trace"])
                fields={"paired_trajectories":1,"changed_actual_action_sequences":int(changed),
                    "selected_actions_with_positive_proxy_gain":row["planning_diagnostics"]["selected_actions_with_positive_two_stage_gain"]}
                action_counts.update(fields)
                action_groups[row["fit_scope"],row["tie_rule"]].update(fields)
    if control or action_counts["paired_trajectories"]!=8640:raise ValueError("Incomplete pair comparison")
    actions={"totals":dict(action_counts),"by_fit_and_tie":[{"fit_scope":k[0],"tie_rule":k[1],**v} for k,v in sorted(action_groups.items())]}
    atomic_json(args.run/"action_comparison.json",actions)
    relevant=[r for r in main_effects if r["tie_rule"]=="fixed_random" and r["comparison"] in
        ["serial_two_stage_vs_serial_one_stage","serial_two_stage_vs_prior_survey","serial_two_stage_vs_prior_gate"]]
    not_advancing=any(r["delta_found"]<=0 for r in relevant)
    event_rows=[r for r in summary["event_means"]+simple["event_means"] if primary(r)
        and r["policy"] in ["prior_survey","prior_gate","serial_one_stage","serial_two_stage"] and r["tie_rule"]=="fixed_random"]
    evidence={"stage":"SERIAL_TWO_STAGE_PRIMARY_EVIDENCE_COMPLETE","primary_random_tie_macro":random_rows,
        "primary_all_ties_macro":main_rows,"primary_random_tie_event_means":event_rows,
        "primary_all_ties_paired_effects":main_effects,"action_comparison":actions,
        "strong_control_structure":"Previously omitted exploratory batch and exact branch marginalization implemented; serial cost/noisy feedback adaptation, not original MF-ENS reproduction.",
        "new_method_selected":False,"independent_validation_claim":False,
        "simple_reference_summary_sha256":sha(simple_path.read_bytes())}
    atomic_json(args.run/"evidence_summary.json",evidence)
    decision={"stage":"SERIAL_MULTIFIDELITY_PRIMARY_CONTROL_COMPLETED",
        "do_not_advance_this_planner_as_new_method":not_advancing,
        "interpretation":"Record development evidence without tuning exploratory batch size or deeper lookahead. If real discovery fails the simple/one-stage controls, surrogate value gains do not justify continuing this planner." if not_advancing else
            "Only provisional development improvement; no independent efficacy or method selection. Data admission and fuller baseline robustness remain open.",
        "two_stage_structure_implementation_gate":"PASS_SAME_INTERFACE_ENGINEERING_CONTROL",
        "original_mfens_faithful_reproduction_gate":"NOT_CLAIMED_DIFFERENT_NOISY_SERIAL_TASK",
        "full_grid_robustness_gate":"NOT_TESTED_ONLY_PRESPECIFIED_PRIMARY_SETTING",
        "independent_data_gate":"NOT_ADMITTED_PER_METADATA_AUDIT",
        "new_method_selected":False,"old_stopped_variants_remain_stopped":True,
        "new_model_fit_enabled":False,"validation_or_confirmation_scoring_enabled":False,"paid_gpu_enabled":False,
        "next_work":["Freeze the one/two-stage controls and all positive/negative results; do not tune another planner on these72.",
            "Prepare a concrete independent-data protocol for any geography/event generalization claim, retaining the SoCal/Woolsey overlap evidence and missing footprints.",
            "Before a new method, settle observation/utility shift and the defensible evaluation scope; fuller literature baseline and cost/quality robustness cannot be declared solved by this primary-only adapter."],
        "basis_evidence_sha256":sha((args.run/"evidence_summary.json").read_bytes()),
        "basis_audit_sha256":sha((args.run/"audit.json").read_bytes())}
    atomic_json(args.run/"next_step_decision.json",decision)
    # Neutral static research figure; avoid picking a favorable fit or tie.
    selected=["prior_survey","prior_gate","serial_one_stage","serial_two_stage"]
    labels=["Raw survey","Raw negative gate","One-stage batch tail","Two-stage exploration"]
    fig,axes=plt.subplots(1,2,figsize=(10.8,4),constrained_layout=True)
    y=np.arange(4)
    maximum=max(lookup[s,p]["distinct_found"] for s in ["development_fit","event_holdout"] for p in selected)*1.16
    for ax,scope,title in zip(axes,["development_fit","event_holdout"],["All-development source fit","Leave one development event out"]):
        vals=[lookup[scope,p]["distinct_found"] for p in selected]
        ax.barh(y,vals,color=["#67768b","#314e73","#58a0a0","#cc874e"],height=.64)
        ax.set_yticks(y,labels)
        ax.invert_yaxis()
        ax.set_xlim(0,maximum)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel("Distinct buildings found (event macro mean)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x",alpha=.16)
        ax.set_axisbelow(True)
        for i,v in enumerate(vals):ax.text(v+.05,i,f"{v:.3f}",va="center",fontsize=9)
    fig.suptitle("Serial two-stage multifidelity control: development primary setting",fontsize=12)
    figures=args.run/"figures"
    figures.mkdir(exist_ok=True)
    for suffix in ["png","pdf"]:fig.savefig(figures/("serial_two_stage_primary."+suffix),dpi=180)
    plt.close(fig)
    atomic_json(figures/"figure_receipt.json",{"basis_evidence_sha256":sha((args.run/"evidence_summary.json").read_bytes()),
        "scope":"Mixed,C2,B32,threshold10%, established random ties, development only; no error bars treating seeds as events.",
        "files":{p.name:sha(p.read_bytes()) for p in figures.iterdir() if p.suffix in [".png",".pdf"]}})
    chinese={"prior_survey":"原始变化只初查","prior_gate":"原始变化阴性门控","count_survey":"数量只初查",
        "count_negative":"数量阴性复核","count_all":"数量全部复核","serial_one_stage":"一阶段批次尾项",
        "serial_two_stage":"两阶段探索批次适配"}
    table=["| 策略 | 全开发拟合 | 开发事件留出 |","|---|---:|---:|"]
    for p in reference_policies+["serial_one_stage","serial_two_stage"]:
        table.append(f"| {chinese[p]} | {lookup['development_fit',p]['distinct_found']:.6f} | {lookup['event_holdout',p]['distinct_found']:.6f} |")
    full_gain=lookup['development_fit','serial_two_stage']['distinct_found']-lookup['development_fit','serial_one_stage']['distinct_found']
    held_gain=lookup['event_holdout','serial_two_stage']['distinct_found']-lookup['event_holdout','serial_one_stage']['distinct_found']
    held_raw_gap=lookup['event_holdout','serial_two_stage']['distinct_found']-lookup['event_holdout','prior_gate']['distinct_found']
    lines=["# 串行成本下的两阶段多保真对照", "",
        "已补齐此前缺失的探索批次、分支枚举和按结果更新的最终成本批次；这是MF-ENS-inspired同接口适配，不是原文忠实复现，也不是提出新方法。", "",
        "主要配置固定为混合质量、预算32、升级成本2。下表为10%匹配阈值和原有随机同分规则，5质量种子×10顺序先在场景内平均，再按事件等权。开发事件留出仍是已反复检查的开发数据。", "",*table,"",
        f"两阶段相对一阶段的全开发/留出增量为{full_gain:+.6f}/{held_gain:+.6f}栋/场景；留出相对原始阴性门控为{held_raw_gap:+.6f}。保留这个小幅增益，但不足以支持把增加规划深度作为新方法推进。", "",
        "## 机制及核验", "",
        "保持来源数量/声明参数不变。在假设当前动作后，从当前最终批次之外按预期交换收益挑选最多2个未访问格；探索成员在结果出现前固定。每个探索都付初查成本，再精确枚举原图终止/模糊阴性/模糊阳性分支，按结果重算剩余预算批次；空探索也可选。只执行第一步真实付费动作，随后重新规划。", "",
        "关闭探索时直接调用旧一阶段值及原同分规则，8,640条轨迹与历史batch_tail的动作、观察、效用逐项完全一致；17,280条新轨迹、51,840条阈值评分全部重放，9,216个固定掩膜哈希核验。212项测试通过，包括小池穷举、成本扣除、非适应批次和私有字段隔离。", "",
        f"探索启用后，在8,640对轨迹中有{action_counts['changed_actual_action_sequences']:,}对实际动作变化；被选动作中{action_counts['selected_actions_with_positive_proxy_gain']:,}步的两阶段代理增量为正。代理增量、动作变化和真实去重发现分别保存，不能互相替代。", "",
        "## 逐事件与同分敏感性", ""]
    held={(r['event'],r['policy']):r for r in event_rows if r['fit_scope']=='event_holdout'}
    lines += ["| 开发留出事件 | 原始只初查 | 阴性门控 | 一阶段 | 两阶段 |","|---|---:|---:|---:|---:|"]
    for event in sorted({e for e,p in held}):
        lines.append("| "+event+" | "+" | ".join(f"{held[event,p]['distinct_found']:.6f}" for p in selected)+" |")
    tie_effects={(r['fit_scope'],r['tie_rule']):r for r in main_effects if r['comparison']=='serial_two_stage_vs_serial_one_stage'}
    lines += ["", "| 同分规则 | 两阶段−一阶段：全开发 | 两阶段−一阶段：开发留出 |", "|---|---:|---:|"]
    for rule in ['row_major','raw_change','fixed_random']:
        lines.append(f"| {rule} | {tie_effects['development_fit',rule]['delta_found']:+.6f} | {tie_effects['event_holdout',rule]['delta_found']:+.6f} |")
    lines += ["", "全开发拟合在行列/变化同分下变为负增益；留出时三规则都是很小的正增益，但仍低于原始只初查和阴性门控。不能把随机规则的一点正向结果写成稳健优势。"]
    lines += ["", "所有3种同分规则和5/10/25%阈值均保留在summary及paired_effects中，未根据结果换规则或主配置。Mexico无缓存可检出目标，仍保留在分母中。具体继续/停止判定见next_step_decision.json。", "",
        "## 限制与下一步", "",
        "两阶段只保证在这个近似代理里可选择不做探索，不保证真实发现增加。原文准确高保真与并行时间预算、只计高保真奖励及kNN传播在当前任务中均有差异；本适配同格计数仍忽略跨格建筑重复，末段仍不规划其他未访问格的后续升级。", "",
        "本轮只检验原定主要配置，不声明成本/质量/预算全网格稳健性。地理独立准入仍未通过，SoCal/Woolsey重合及缺范围证据保持保留；不能因对照完成就开启保留集评分。", "",
        "没有新拟合、感知推理、影像下载、控制器训练或GPU费用。保存的planning_s含缓存复用，只作执行记账，不是冷缓存速度比较。旧停止变体不恢复；下一阶段先锁独立数据与观察/效用偏移的研究范围，不在这72场景上调探索上限或继续加深规划。", "",
        "算法定义、原文机制对应及所有差异见adaptation.md。原始论文PDF/文本路径和SHA256、代码哈希均在protocol.json；源文件快照及完成文件哈希见completion_receipt.json。", ""]
    (args.run/"README.md").write_text("\n".join(lines))
    print(json.dumps({"stage":evidence["stage"],"do_not_advance_as_method":not_advancing,"action_comparison":dict(action_counts),
        "primary_random_tie_effects":relevant}),flush=True)


if __name__=="__main__":
    main()
