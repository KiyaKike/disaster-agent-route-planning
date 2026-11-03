"""Report the pre-specified calibration decision without tuning another map."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_rank_calibration_control import without_utility
from build_xbd_mini import atomic_json, sha
from run_utility_tie_controls import CONFIG_FIELDS, is_primary


def compare_actions(path):
    source, counts, groups = {}, Counter(), defaultdict(Counter)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[k] for k in CONFIG_FIELDS if k != "policy")
            if row["policy"] == "rank_negative_unprojected":
                source[key] = row["trace"]
            elif row["policy"] == "rank_negative_isotonic":
                original = source.pop(key)
                changed = without_utility(original) != without_utility(row["trace"])
                label = tuple(row[k] for k in ["fit_scope", "scenario", "upgrade_cost", "budget", "tie_rule"])
                groups[label]["compared_trajectories"] += 1
                groups[label]["changed_action_sequences"] += changed
                counts["compared_trajectories"] += 1
                counts["changed_action_sequences"] += changed
                if (row["scenario"], row["upgrade_cost"], row["budget"]) == ("mixed_survey", 2, 32):
                    counts["primary_all_ties_compared"] += 1
                    counts["primary_all_ties_changed"] += changed
    if source or counts["compared_trajectories"] != 120960:
        raise ValueError("Incomplete matched action comparison")
    return {"counts": dict(counts), "by_setting": [
        {**dict(zip(["fit_scope", "scenario", "upgrade_cost", "budget", "tie_rule"], k)), **v}
        for k, v in sorted(groups.items())]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    audit = json.loads((args.run / "audit.json").read_text())
    if audit["stage"] != "DEVELOPMENT72_RANK_CALIBRATION_FULL_REPLAY_PASSED":
        raise ValueError("Successful full replay required")
    refs_path = Path(protocol["sources"]["references"]["path"]).with_name("summary.json")
    old = json.loads(refs_path.read_text())
    new = json.loads((args.run / "summary.json").read_text())
    policies = ["prior_survey", "prior_gate", "count_survey", "count_negative",
        "rank_survey", "rank_negative_unprojected", "rank_negative_isotonic"]
    all_macro = old["event_macro_means"] + new["event_macro_means"]
    main_rows = [r for r in all_macro if r["policy"] in policies and is_primary(r)]
    primary = [r for r in main_rows if r["tie_rule"] == "fixed_random"]
    lookup = {(r["fit_scope"], r["policy"]): r for r in primary}
    effects = json.loads((args.run / "paired_effects.json").read_text())
    primary_effects = [r for r in effects["event_macro_effects"] if is_primary(r)]
    held = {r["comparison"]: r for r in primary_effects if r["fit_scope"] == "event_holdout" and r["tie_rule"] == "fixed_random"}
    over_unprojected = held["rank_negative_isotonic_vs_rank_negative_unprojected"]["delta_found"]
    over_raw_gate = held["rank_negative_isotonic_vs_prior_gate"]["delta_found"]
    stopped = over_unprojected <= 0 or over_raw_gate <= 0
    actions = compare_actions(args.run / "trajectories.jsonl.gz")
    atomic_json(args.run / "price_projection_action_effects.json", actions)
    event_rows = [r for r in old["event_means"] + new["event_means"] if r["policy"] in policies
        and is_primary(r) and r["tie_rule"] == "fixed_random"]
    evidence = {"stage": "RANK_CALIBRATION_CONTROL_EVIDENCE_COMPLETE",
        "primary_random_tie_event_macro": primary, "primary_all_ties_event_macro": main_rows,
        "primary_random_tie_by_event": event_rows, "primary_all_ties_paired_effects": primary_effects,
        "isotonic_vs_unprojected_action_comparison": actions["counts"],
        "held_primary_isotonic_minus_unprojected": over_unprojected,
        "held_primary_isotonic_minus_raw_gate": over_raw_gate,
        "descriptive_only": True, "new_method_selected": False,
        "validation_or_confirmation_scored": False,
        "reference_summary_sha256": sha(refs_path.read_bytes())}
    atomic_json(args.run / "evidence_summary.json", evidence)
    decision = {"stage": "RANK_CALIBRATION_CONTROL_COMPLETE", "variant_stopped": stopped,
        "decision": "Stop this isotonic survey-price projection variant; retain raw survey/raw negative gate as simple controls and record raw-order restoration as baseline repair." if stopped else
            "Only a provisional development gain; require event/tie assessment, strong baseline and independent data admission before choosing any method.",
        "basis_evidence_sha256": sha((args.run / "evidence_summary.json").read_bytes()),
        "basis_audit_sha256": sha((args.run / "audit.json").read_bytes()),
        "not_universal_impossibility": "This rejects this fixed source-only isotonic survey-price variant, not every calibration or revisit method.",
        "old_positive_revisit_policy_still_stopped": True,
        "do_not_tune_another_mapping_on_these72": True, "new_method_selected": False,
        "next_work": ["Freeze the simple raw baselines and the negative findings; stop adding heuristics on the same development72.",
            "Complete evaluation-data admission: source/license evidence, full image footprint/spatial overlap, checkpoint training exposure, usable event diversity. Use metadata admission before any reserved policy scores.",
            "Specify and implement a budget-compatible strong literature baseline before any method claim; the current batch-tail adapter is not full MF-ENS.",
            "Only after data and baseline gates, freeze one candidate and the independent validation protocol. Do not automatically score validation/confirmation now."],
        "validation_or_confirmation_access_enabled": False, "new_inference_calls": 0, "paid_gpu_needed": False}
    atomic_json(args.run / "next_step_decision.json", decision)

    # Standard static research figure; primary B32/cost2/10%, both fit scopes.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    selected = ["prior_survey", "prior_gate", "count_negative", "rank_negative_unprojected", "rank_negative_isotonic"]
    labels = ["Raw survey", "Raw negative gate", "Old count negative", "Raw order + old prices", "Raw order + isotonic prices"]
    y = np.arange(len(selected))
    for ax, scope, title in zip(axes, ["development_fit", "event_holdout"], ["All-development source fit", "Leave one development event out"]):
        values = [lookup[scope, p]["distinct_found"] for p in selected]
        ax.barh(y, values, color=["#596579", "#304e74", "#aeb8c4", "#52a1a3", "#d2874b"], height=.64)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.15)
        ax.set_xlabel("Distinct buildings found (event macro mean)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            ax.text(value + .06, i, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("Rank restoration helps; isotonic survey pricing adds no primary gain", fontsize=13)
    figures = args.run / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(figures / ("rank_repair_and_price_control." + suffix), dpi=180)
    plt.close(fig)
    atomic_json(figures / "figure_receipt.json", {"source_evidence_sha256": sha((args.run / "evidence_summary.json").read_bytes()),
        "scope": "Development only; mixed quality, cost2, B32, threshold10%, established random ties; seeds averaged within scene.",
        "files": {p.name: sha(p.read_bytes()) for p in figures.iterdir() if p.suffix in [".png", ".pdf"]}})

    table = ["| 策略 | 全开发拟合 | 开发事件留出 |", "|---|---:|---:|"]
    chinese = {"prior_survey": "原始变化只初查", "prior_gate": "原始变化阴性门控", "count_survey": "旧数量只初查",
        "count_negative": "旧数量阴性复核", "rank_survey": "保留原始排序、只初查",
        "rank_negative_unprojected": "保留原始排序＋原数量报价", "rank_negative_isotonic": "保留原始排序＋单调报价"}
    for p in policies:
        table.append(f"| {chinese[p]} | {lookup['development_fit', p]['distinct_found']:.6f} | {lookup['event_holdout', p]['distinct_found']:.6f} |")
    held_events = {(r["event"], r["policy"]): r for r in event_rows if r["fit_scope"] == "event_holdout"}
    event_table = ["| 开发留出事件 | 原始只初查 | 原始阴性门控 | 保序单调报价＋阴性复核 |", "|---|---:|---:|---:|"]
    for event in sorted({e for e, p in held_events}):
        event_table.append("| " + event + " | " + " | ".join(f"{held_events[event, p]['distinct_found']:.6f}"
            for p in ["prior_survey", "prior_gate", "rank_negative_isotonic"]) + " |")
    lines = ["# 保持排序的效用校准对照", "", "结论：停止本轮固定的单调报价变体；恢复原始排序记为基线修复，不能作为新方法。旧阳性复核方案继续停止。", "",
        "主要设置仍为混合质量、预算32、升级成本2、10%可见轮廓阈值。以下采用提前固定的随机同分规则，5个质量种子和10个同分顺序先在场景内平均，场景在事件内平均，再对3事件等权。", "",
        *table, "", "这里的“开发事件留出”仍是已反复检查过的开发数据，不是独立验证。Mexico无缓存可检出目标，保留在所有分母内。", "",
        "## 检验的区别", "",
        "三组新策略都严格沿原始变化分数初查。只初查组必须逐动作复现旧原始排序；未投影阴性组使用冻结的原数量报价；投影阴性组只把初查报价按来源统计做单调投影，阴性升级报价完全不变。", "",
        "投影分别作用于原图/模糊的4档收缩均值，按来源已知格数+8加权；各留出模型仅使用另外两个开发事件的冻结参数，不读取被留出事件结果拟合。单调是这次对照的约束，未证明适用于真实事件。", "",
        f"留出主要设置下，单调组相对原始只初查为 {lookup['event_holdout', 'rank_negative_isotonic']['distinct_found'] - lookup['event_holdout', 'prior_survey']['distinct_found']:+.6f} 栋/场景；相对未投影组 {over_unprojected:+.6f}，相对原始阴性门控 {over_raw_gate:+.6f}。这触发协议中预定的停止条件。", "",
        *event_table, "", "相对原始只初查，留出Harvey少1.363333栋/场景，Santa Rosa多0.072500，Mexico均为零。不能把某个事件的弱增益写成跨事件改善。", "",
        "全部预算/质量/成本/阈值/同分结果保存在summary及paired_effects中，没有改用成本1或另一个同分规则作为主结果。动作是否相同另外逐序列比较，不能由均值相等推断。", "",
        "敏感性核查：主要设置的三种同分规则下，投影与未投影的动作都完全相同，发现和误报差均为零。其他配置会改变动作；全开发拟合、混合质量、成本1时，投影相对未投影在预算8/16/32分别少0.177222/0.324444/0.421111栋/场景。", "",
        "## 核验", "",
        f"完成362,880条轨迹、1,088,640条阈值评分；每条轨迹完整重放决策、预算、实际质量和付费观察，全部初查子序列保持原始排序。120,960条只初查轨迹与旧原始策略的动作/观察完全一致；来源投影另用四档全部分段枚举独立核对。9,216个固定掩膜哈希通过。189项测试通过。", "",
        f"两种阴性报价在全部配置中比较 {actions['counts']['compared_trajectories']:,} 对轨迹，动作改变 {actions['counts']['changed_action_sequences']:,} 对；在主要配置的全部同分/拟合范围中改变 {actions['counts']['primary_all_ties_changed']:,} 对。详见price_projection_action_effects.json。", "",
        "没有新感知推理、影像下载、付费GPU或验证/确认评分；本轮新增的拟合仅是4套来源聚合参数的单调投影。不能把它写成“没有新拟合”。", "",
        "## 接下来", "",
        "先冻结原始初查/原始阴性门控基线和这些负结果，不再在同一72样本上调映射。优先补齐数据准入证据与预算兼容强基线：来源许可、完整影像地理重叠、权重暴露和有效事件多样性仍未解决；原批量尾部适配不是完整MF-ENS。只有这两项过关，再锁一个候选进入独立验证。", "",
        "这只是否定当前固定校准变体，不是否定所有复核或校准方法。现阶段尚未选定有独立证据支持的新方法。", "",
        "复现入口：work/disaster_search/run_rank_calibration_control.py、audit_rank_calibration_control.py、summarize_rank_calibration_control.py。协议、来源与代码哈希见protocol.json，完成文件哈希见completion_receipt.json。", ""]
    (args.run / "README.md").write_text("\n".join(lines))
    print(json.dumps({"stage": evidence["stage"], "variant_stopped": stopped,
        "held_primary_isotonic_minus_unprojected": over_unprojected,
        "held_primary_isotonic_minus_raw_gate": over_raw_gate, "action_comparison": actions["counts"]}), flush=True)


if __name__ == "__main__":
    main()
