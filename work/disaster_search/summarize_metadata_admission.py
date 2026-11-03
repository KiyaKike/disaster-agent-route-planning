"""Make a bounded admission decision, retaining weak and missing evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from build_xbd_mini import atomic_json, sha


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    audit = json.loads((args.run / "audit.json").read_text())
    ledger = json.loads((args.run / "candidate_ledger.json").read_text())
    probe = json.loads((args.run / "geometry_duplicate_probe.json").read_text())
    assert audit["stage"] == "OFFLINE_METADATA_ADMISSION_FULL_REPLAY_PASSED"
    assert not probe["pixel_hash_equality_verified"] and not report["whole_216_cohort_independently_admitted"]
    # Descriptive area bins are added AFTER the screen; they never change the
    # frozen1e-6 screening threshold, membership, admission, or any policy score.
    bins = [1e-4, 1e-3, .01, .1, .99]
    labels = ["below_0.01_percent", "0.01_to_0.1_percent", "0.1_to_1_percent",
              "1_to_10_percent", "10_to_99_percent", "99_percent_or_more"]
    distributions = {}
    for field in ["training_inferred_overlaps", "training_padded_inferred_overlaps", "selection_test_inferred_overlaps"]:
        pairs = [x["fraction_of_smaller_inferred_post_range"] for r in ledger for x in (r[field] or [])]
        counts = Counter(labels[int(np.searchsorted(bins, x, side="right"))] for x in pairs)
        distributions[field] = {"candidate_flags": sum(bool(r[field]) for r in ledger), "pairs": len(pairs),
            "area_bin_pair_counts": {label: counts[label] for label in labels},
            "fraction_quantiles_min_median_p90_p99_max": np.quantile(pairs, [0, .5, .9, .99, 1]).tolist() if pairs else [],
            "thresholds_are_descriptive_not_new_admission_rules": True}
    by_split = defaultdict(list)
    for r in ledger:
        by_split[r["split"]].append(r)
    splits = []
    for split, rows in sorted(by_split.items()):
        splits.append({"split": split, "candidates": len(rows),
            "usable_inferred_ranges": sum(r["usable_inferred_range"] for r in rows),
            "unknown_inferred_ranges": sum(not r["usable_inferred_range"] for r in rows),
            "original_training_envelope_flags": sum(bool(r["old_training_annotation_envelope_overlap_ids"]) for r in rows),
            "inferred_training_overlap_flags": sum(bool(r["training_inferred_overlaps"]) for r in rows),
            "old_or_new_training_screen_flags": sum(bool(r["old_training_annotation_envelope_overlap_ids"]) or bool(r["training_inferred_overlaps"]) for r in rows),
            "inferred_selection_test_overlap_flags": sum(bool(r["selection_test_inferred_overlaps"]) for r in rows)})
    evidence = {"stage": "METADATA_ADMISSION_EVIDENCE_SUMMARIZED", "split_summary": splits,
        "area_distributions": distributions, "by_split_and_event": report["by_split_and_event"],
        "near_complete_spatial_pairs": probe["pairs"], "candidate_ranges_unknown": report["candidate_unknown_inferred_ranges"],
        "training_ranges_unknown": report["training_missing_or_rejected_inferred_ranges"],
        "cross_split_inferred_pair_count": len(report["inferred_cross_policy_split_overlaps"]),
        "official_full_footprints_verified": False, "pixel_duplicate_confirmed": False,
        "independent_validation_admitted": False, "validation_or_confirmation_scores_run": False,
        "basis_report_sha256": sha((args.run / "report.json").read_bytes()),
        "basis_probe_sha256": sha((args.run / "geometry_duplicate_probe.json").read_bytes())}
    atomic_json(args.run / "evidence_summary.json", evidence)
    decision = {"stage": "METADATA_SCREEN_FINISHED_INDEPENDENT_GEOGRAPHY_NOT_ADMITTED",
        "conclusion": "Fixed216 cohort cannot currently support unseen-geography or unseen-event perception claims. Near-complete SoCal/Woolsey spatial overlap is a concrete cross-name risk, not a pixel-duplicate confirmation.",
        "current_supported_scope": "Development-only fixed-detector observation-budget policy engineering with documented exposure and synthetic observation quality; no new method or independent efficacy established.",
        "cohort_membership_unchanged": True, "no_automatic_scene_removal_or_replacement": True,
        "validation_or_confirmation_scoring_enabled": False,
        "new_controller_training_enabled": False, "paid_gpu_enabled": False,
        "stopped_variants_remain_stopped": ["count_positive_revisit", "source_only_isotonic_survey_price_projection"],
        "next_executable_work": "Specify and implement one budget-compatible strong literature baseline on the frozen development interface, with explicit adaptations for noisy native feedback, same-cell upgrades and global distinct-building evaluation. Existing batch-tail is not full MF-ENS. Do not train another heuristic on these72 or score reserved cohorts before method/data scope lock.",
        "independent_data_requirements": ["Official footprint/scene metadata and complete overlap coverage, not inferred transforms alone.",
            "Image byte/pixel duplicate evidence, including the pre-input in the3 SoCal/Woolsey cases; equal encoded length is insufficient.",
            "Prespecified detector training/selection scope compatible with the intended geography/event claim.",
            "Original source permission/attribution evidence; pinned repackager declarations alone do not clear redistribution.",
            "Separate data protocol for any new independent regions; no performance-based replacement of the fixed216."],
        "basis_evidence_sha256": sha((args.run / "evidence_summary.json").read_bytes()),
        "basis_audit_sha256": sha((args.run / "audit.json").read_bytes())}
    atomic_json(args.run / "next_step_decision.json", decision)
    table = ["| 分组 | 候选 | 可重建范围 | 未知范围 | 训练范围相交提示 | 选权重test相交提示 |",
        "|---|---:|---:|---:|---:|---:|"]
    names = {"development": "开发", "validation": "验证", "confirmation": "最终确认"}
    for r in splits:
        table.append(f"| {names[r['split']]} | {r['candidates']} | {r['usable_inferred_ranges']} | {r['unknown_inferred_ranges']} | {r['inferred_training_overlap_flags']} | {r['inferred_selection_test_overlap_flags']} |")
    pairs_table = ["| 验证候选 | 作者训练池场景 | 推断重合比例 | 像素一致性 |", "|---|---|---:|---|"]
    for r in probe["pairs"]:
        pairs_table.append(f"| {r['candidate_id']} | {r['training_id']} | {100*r['fraction_of_smaller_inferred_post_range']:.6f}% | 未核对 |")
    dist = distributions["training_inferred_overlaps"]
    text = ["# 固定216候选的离线元数据准入审计", "",
        "结论：空间风险证据已补充，但当前候选仍不能宣称地理独立或感知器未见事件泛化；不启用验证/确认评分。保留全部固定候选，不按本次风险或后续成绩自动换图。", "",
        "## 本轮完成了什么", "",
        "仅重放缓存的XY/WGS84轮廓、顶点数和图像尺寸，尝试用固定规则重建post影像范围。另对作者的训练池和选权重test池做相同重建，所有损伤与数量列在读取前拦截。没有读新影像、下载数据、训练控制器或运行感知/策略。", "",
        f"固定216候选中159个通过几何一致性与覆盖检查，57个记为未知；作者6,177个训练场景中3,627个可重建，2,550个未知。未通过的包括无轮廓、点数不足或轮廓覆盖不足，不能改记为独立。", "",
        *table, "",
        "“相交提示”是推断范围筛查，不是已确认图像重复；无提示也不是通过隔离。原建筑包络筛查另行保留，不能因新拟合缺失就撤销旧风险。当前未检出可重建候选之间的跨分组相交，但缺失范围和推断误差仍在，不能据此宣布分组地理独立。", "",
        "## 三对跨事件名称的高重合风险", "", *pairs_table, "",
        "这三对的完整XY轮廓集合不相同，post PNG字节长度也不同；三对pre PNG字节长度分别相同，但档案偏移不同。没有读取PNG内容，不能由长度相同证明重复影像。证据支持相同地理范围风险，不支持像素完全重复或标签重复的结论。", "",
        f"全部训练相交提示共{dist['candidate_flags']}个候选、{dist['pairs']}对；其中{dist['area_bin_pair_counts']['below_0.01_percent']}对的交叠小于较小范围的0.01%，大量只是极窄边界缝。不能把所有提示写成同等程度的数据泄漏。", "",
        "2像素向外扩展仅作邻接敏感性筛查，不是置信界。内部几何交叉预测和低残差不能替代官方地理定位精度；post轮廓也没有独立验证pre影像足迹。面积分箱为筛查完成后的描述，不改变预定阈值或准入规则。", "",
        "## 权重、许可与事件数", "",
        "作者模型卡写明train 2,799＋tier3 3,378，合计6,177；TACO额外收录2,991个tier3场景，本轮未把额外场景算进作者训练池。候选名称与作者训练/选权重test名称无交集，但9个候选事件均在训练池中出现。未取得逐优化步骤训练清单，不把公开训练配方等同于该清单。冻结模型卡、配置和权重哈希均核对。", "",
        "来源缓存卡片声明CC-BY-NC-SA-4.0，模型卡声明Apache-2.0；这里只整理固定声明，没有重新核查原发布方完整条款，也未确认影像再分发权限。", "",
        "有效事件数沿用已有开发感知报告：Harvey、Santa Rosa有缓存检出，Mexico有标注目标但无缓存检出。仍保留Mexico；没有读取验证/确认的损伤构成来挑选事件。", "",
        "## 核验与复现", "",
        "7,326套几何拟合全部重放；648项候选—数据池筛查和216条候选记录逐项重核，201项测试通过。主审计及高重合补充探查均为零网络、零新影像载荷、零损伤列读取；各读取范围/哈希与代码快照已保存。补充探查明确在主筛查后按所有≥99%空间重合对冻结，未使用任何策略成绩。", "",
        "```bash", "work/disaster_search/.venv_inference/bin/python work/disaster_search/run_metadata_admission_audit.py --out output/disaster_search_metadata_admission_v2",
        "work/disaster_search/.venv_inference/bin/python work/disaster_search/audit_metadata_admission.py --run output/disaster_search_metadata_admission_v2",
        "work/disaster_search/.venv_inference/bin/python work/disaster_search/probe_metadata_geometry_duplicates.py --run output/disaster_search_metadata_admission_v2",
        "work/disaster_search/.venv_inference/bin/python work/disaster_search/summarize_metadata_admission.py --run output/disaster_search_metadata_admission_v2", "```", "",
        "运行器要求新输出目录，缓存缺失会在零网络预算处停止，不会自动下载或替换样本。全量旧试验输出和旧协议保持不变。", "",
        "## 下一步执行边界", "",
        "下一项是预算兼容的强基线接口与实现，明确原文真值查询和当前有误差的原图反馈、同格升级、建筑去重目标之间的差异；现有批量尾项不是完整MF-ENS。只在开发接口内做工程核验，不恢复已停止的数量阳性复核/单调报价变体。", "",
        "若后续要声称未见地理/灾害泛化，应另立独立地区的数据与训练暴露协议，补官方范围和像素重复证据；不能靠本轮删掉风险场景、换入阳性图来修复。固定感知器下的策略比较与感知器泛化需分别界定。", ""]
    (args.run / "README.md").write_text("\n".join(text))
    print(json.dumps({"stage": evidence["stage"], "near_complete_training_spatial_pairs": len(probe["pairs"]),
        "candidate_ranges_unknown": evidence["candidate_ranges_unknown"], "independent_validation_admitted": False,
        "training_overlap_distribution": dist}), flush=True)


if __name__ == "__main__":
    main()
