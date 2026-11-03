"""Describe the fixed input experiment and register a bounded next diagnostic."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from plan_search_v2 import freeze


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);args=p.parse_args();root=args.root
    get=lambda n:json.loads((root/n).read_text());protocol=get("protocol.json");summary=get("summary.json");audit=get("audit.json")
    primary=[r for r in summary["totals"] if r["domain"]=="all_published_pixels"]
    by_mode={r["input_mode"]:r for r in primary};old=by_mode["crop128_mosaic_native"];new=by_mode["whole512_native"]
    event_rows=[r for r in summary["event_rows"] if r["domain"]=="all_published_pixels"]
    events=sorted({r["event"] for r in event_rows});pairs=[r for r in get("paired_changes.json") if r["domain"]=="all_published_pixels"]
    reopened=[e for e in events if next(r for r in event_rows if r["event"]==e and r["input_mode"]=="crop128_mosaic_native")["covered_reference_pixels"]==0 and next(r for r in event_rows if r["event"]==e and r["input_mode"]=="whole512_native")["covered_reference_pixels"]>0]
    passed=bool(reopened) and new["covered_reference_pixels"]>old["covered_reference_pixels"] and new["pooled_reference_precision"]>=old["pooled_reference_precision"]
    evidence={"primary_domain":"all_published_pixels","native_crop":old,"native_whole":new,"reopened_events":reopened,
        "followup_screen_passed":passed,"gained_reference_pixels":sum(r["gained_reference_pixels"] for r in pairs),
        "lost_reference_pixels":sum(r["lost_reference_pixels"] for r in pairs),"preserved_reference_pixels":sum(r["preserved_reference_pixels"] for r in pairs),
        "reference_coverage_increased_scenes":sum(r["gained_reference_pixels"]>r["lost_reference_pixels"] for r in pairs),
        "reference_coverage_decreased_scenes":sum(r["gained_reference_pixels"]<r["lost_reference_pixels"] for r in pairs),
        "reference_coverage_equal_scenes":sum(r["gained_reference_pixels"]==r["lost_reference_pixels"] for r in pairs),
        "input_interface_changes_prediction":True,"unique_cause_identified":False,"new_search_method_selected":False,
        "area_is_published_reference_proxy":True,"whole_input_is_not_budget8_policy":True}
    atomic_json(root/"evidence_summary.json",evidence)
    atomic_json(root/"next_step_decision.json",{"decision":"PROCEED_TO_BOUNDED_INPUT_INFORMATION_DIAGNOSTIC" if passed else "STOP_AND_REASSESS_SENSOR_REFERENCE_INTERFACE",
        "basis":"Predeclared event-recovery, total-overlap and pooled-reference-precision screen.","screen_passed":passed,
        "all18_retained":True,"search_tuning_still_stopped":True,"stage2_executed":False,"paid_gpu_needed":False,
        "cause_limit":"Input size, available context, boundary padding and model internal resolution may jointly change outputs; current experiment does not isolate them.",
        "reference_limit":"Published semantic pixels have not received new human or geographic validation;18 pairs remain development/feasibility only."})
    next_protocol={"status":"PROSPECTIVE_NOT_EXECUTED","client_report_date":"2026-10-26",
        "question":"Can a large-input interface recover reference overlap using only free coarse RGB and one purchased128px cell?",
        "basis_protocol_sha256":sha((root/"protocol.json").read_bytes()),"basis_evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),
        "fixed_ids":protocol["selected_ids"],"cohort_rule":"Same18 pairs, all16 cells each, including all zeros. No replacements or label-selected ROIs.",
        "checkpoint":{"model_id":protocol["model_id"],"revision":protocol["model_revision"],"weights_sha256":protocol["weights_sha256"],"config_sha256":protocol["config_sha256"],"code_commit":protocol["code_commit"]},
        "cached_comparison":"Native128px cell, existing288 caches; no new inference.",
        "new_conditions":[
            {"name":"patch128_to512","calls":288,"input":"Purchased native128x128 RGB cell only, upsample512x512 BILINEAR for each date.","output":"Predict512 binary severe mask using fixed decoder, reduce to128 with NEAREST; recompute >=4px declaration from canonical128 mask."},
            {"name":"blank512_plus_native_cell","calls":288,"input":"512x512 RGB canvas filled with (124,116,104), then paste purchased128px native cell at its original grid position for each date.","output":"Keep only exact128px ROI prediction; ignore every output pixel outside purchased cell."},
            {"name":"free_coarse512_plus_native_cell","calls":288,"input":"Free64x64 BOX coarse RGB per date, upsample512 BILINEAR, then paste purchased128px native cell at its original position.","output":"Keep only exact128px ROI prediction; ignore every output pixel outside purchased cell."},
            {"name":"free_coarse512_only","calls":18,"input":"Same free64px BOX RGB upsampled512 BILINEAR; no native-cell paste.","output":"Whole-scene coarse-only diagnostic, and fixed per-cell ROIs for marginal-information comparison. Do not pretend coarse-only pixels require paid observations."}],
        "positive_controls":"Replay same2 locked positive control crops; any unequal pixel stops before new18-scene conditions.",
        "max_new_forward_calls":884,"no_hidden_warmup":True,"precision":"Same fixed model, tensor normalization, eval mode, seed and CPU/MPS settings as stage1.",
        "input_boundary":"Coarse-context transform accepts only two64px coarse images and the selected pair of128px patches, not a512px high-resolution scene. Model output and boolean for each paid observation may depend on those only.",
        "required_preflight":"Synthetic geometry, resize inversion, identical coarse context outside ROI across cell modes, no unseen-native access, and boolean threshold tests. Lock implementation hashes before model calls.",
        "domains":protocol["domains"],"evaluation":"All18 scene mosaics, all3 events and pooled severe-reference coverage/precision/false pixels, plus paired gains/losses. Report null recall for zero-reference events; no label fitting.",
        "native_only_scope":"This diagnostic covers native observations only. Mixed native/blur budget8 search cannot be scored until an equally frozen blurred-cache extension exists.",
        "engineering_followup_rule":"Proceed to a separate mixed-quality cache extension only if free-coarse+cell restores positive overlap in both reference-positive events, improves pooled overlap with no pooled reference-precision drop vs cached native128, exceeds blank+cell covered reference pixels overall, and adds reference coverage beyond corresponding free-coarse-only ROI in both events. Report every condition regardless of rule outcome; no statistical or novelty claim.",
        "no_controller_fitting_or_paid_gpu_or_download":True,"search_policy_runs":0,"human_annotation_resources":"Unavailable","instances_created":0}
    freeze(root/"prospective_information_diagnostic.json",next_protocol)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    folder=root/"figures";folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout="constrained")
    x=np.arange(3);colors=["#b7774e","#327c88"]
    for index,mode in enumerate(["crop128_mosaic_native","whole512_native"]):
        rows=[next(r for r in event_rows if r["event"]==e and r["input_mode"]==mode) for e in events]
        axes[0].bar(x+(index-.5)*.32,[100*r["pooled_area_recall"] if r["pooled_area_recall"] is not None else 0 for r in rows],width=.32,color=colors[index],label=["16 cached native128 crops","One native512 scene"][index])
    axes[0].set_xticks(x,["Turkey","Pakistan","Texas"]);axes[0].set_ylim(0,61);axes[0].text(2,3,"No severe reference\nrecall N/A",ha="center",fontsize=8)
    axes[0].set_ylabel("Published severe-pixel recall (%)");axes[0].set_title("All18 fixed pairs; each event has6");axes[0].legend(fontsize=8)
    groups=["Covered reference","Reference-relative false"]
    for index,row in enumerate([old,new]):
        values=[row["covered_reference_pixels"],row["reference_false_positive_pixels"]]
        bars=axes[1].bar(np.arange(2)+(index-.5)*.32,values,width=.32,color=colors[index])
        for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+90,f"{value:,}",ha="center",fontsize=8)
    axes[1].set_xticks([0,1],groups);axes[1].set_ylim(0,6900);axes[1].set_ylabel("Pixels pooled across18 scenes");axes[1].set_title("More overlap and fewer reference-relative false pixels")
    for ax in axes:ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.suptitle("Fixed checkpoint input diagnostic | full scene is offline, outside budget8",fontsize=11)
    for ext in ["png","pdf"]:fig.savefig(folder/("context_comparison."+ext),dpi=170)
    plt.close(fig)
    # All18 scene overlays, rather than selecting the two scenes that improved.
    refs={r["sample_id"]:r for r in json.loads(Path("output/disaster_search_external_area18_v1/reference_manifest.json").read_text())}
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};whole={r["sample_id"]:r for r in get("whole_observations.json")}
    for event in events:
        ids=[sid for sid in protocol["selected_ids"] if refs[sid]["event"]==event]
        fig,axes=plt.subplots(6,4,figsize=(12,17),layout="constrained")
        for row,sid in enumerate(ids):
            with Image.open(images[sid,"post"]["path"]) as im:post=np.asarray(im.convert("RGB"))
            with Image.open(refs[sid]["path"]) as im:reference=np.asarray(im)>=3
            native=np.zeros((512,512),dtype=np.bool_)
            for obs in protocol["previous_native_observations"]:
                if obs["sample_id"]!=sid:continue
                r=int(obs["cell_id"][1:3]);c=int(obs["cell_id"][4:6])
                with np.load(obs["cache_file"],allow_pickle=False) as z:native[r*128:(r+1)*128,c*128:(c+1)*128]=z["severe_mask"]
            with np.load(whole[sid]["cache_file"],allow_pickle=False) as z:full=z["severe_mask"]
            for col,mask in enumerate([None,reference,native,full]):
                canvas=post.copy()
                if mask is not None:canvas[mask]=(0.3*canvas[mask]+.7*np.array([40,190,210] if col==1 else [242,122,54])).astype(np.uint8)
                axes[row,col].imshow(canvas);axes[row,col].axis("off")
                if col==0:axes[row,col].set_title(sid,fontsize=8)
                elif row==0:axes[row,col].set_title(["","Published severe ref","Cached native128 mosaic","Native512 whole"][col],fontsize=9)
        fig.suptitle(event+" | all6 pairs, published reference proxy",fontsize=12)
        fig.savefig(folder/(event+"_overlays.png"),dpi=110);plt.close(fig)
    atomic_json(folder/"figure_receipt.json",{"summary_sha256":sha((root/"summary.json").read_bytes()),"protocol_sha256":sha((root/"protocol.json").read_bytes()),"generator_sha256":sha(Path(__file__).read_bytes()),"all18_overlay_pairs_included":True,"human_review":False})
    fmt=lambda x:"null" if x is None else f"{100*x:.2f}%"
    table=[]
    for e in events:
        a=next(r for r in event_rows if r["event"]==e and r["input_mode"]=="crop128_mosaic_native");b=next(r for r in event_rows if r["event"]==e and r["input_mode"]=="whole512_native")
        table.append(f"| {e} | {a['reference_pixels']:,} | {a['covered_reference_pixels']:,}／{fmt(a['pooled_area_recall'])} | {b['covered_reference_pixels']:,}／{fmt(b['pooled_area_recall'])} | {a['reference_false_positive_pixels']:,} → {b['reference_false_positive_pixels']:,} |")
    text=f"""# 固定感知器输入诊断：128 裁块与 512 整图

日期：2026-10-26。状态：20 次本机前向、72 个域评分和 36 个配对变化完成并重放。下一轮输入信息对照仅登记，尚未执行。

## 当前结论

相同权重及原始 RGB 下，输入接口改变了预测：全部 18 对的严重参考像素覆盖从 **915 增至 4,796**，合并参考召回从 **4.78% 增至 25.03%**，参考相对精度从 **13.55% 增至 81.00%**。此前零重叠的土耳其事件恢复了 3,486 个重叠像素。预先登记的有界后续诊断条件通过。

这支持先查输入接口，再谈搜索策略。不能把收益唯一归因于上下文：输入尺寸、模型内部尺度、边界填充和可用上下文同时变化，尚未被分别控制。整图含未付费高分辨率信息，只作离线全信息诊断，**不是预算 8 搜索成绩**。

## 全部事件结果

| 事件，各 6 对 | 严重参考像素 | 16 格原图裁块覆盖／召回 | 整图覆盖／召回 | 参考相对误报像素 |
|---|---:|---:|---:|---:|
"""+"\n".join(table)+f"""

合计参考相对误报从 5,840 降至 1,125。巴基斯坦参考精度从 69.0045% 略降至 68.9111%，没有用总体提升掩盖逐事件差异。得州没有严重参考目标，召回为 null；不计成成功。

逐像素配对新增 3,882、丢失 1、保留 914 个参考像素，净增加 3,881。仅两个场景覆盖增加，其余 16 个不变；有重叠的场景从 1/18 增至 2/18，事件从 1/3 增至 2/3。有严重参考的 5 个场景中仍有 3 个全程零重叠，不能称为跨事件问题已经解决。完整逐图结果在 `scene_scores.json`，没有替换、删除或只汇报有利影像。

双时相非全零域中，参考像素从 19,159 减为 19,158，整图覆盖为 4,795，裁块仍为 915；方向相同。这个域检查不能认证配准或参考语义。

![输入诊断对比]({(folder/'context_comparison.png').resolve()})

## 接入和审计

先按已知开发阳性观察的身份哈希，在 Harvey、Santa Rosa 各固定一个旧裁块。两者新旧预测分别为 3,943／1,379 个严重像素，均逐像素完全一致。它们是接入回归对照，不能计入新的外部评估。

之后对固定 18 对分别做一次未经缩放的 512×512 原图推理，比较旧 16 个 128×128 原图掩膜的几何拼接。预测过程不读参考 PNG，评价器单独读取类别 3／4；数据构成此前已知，属于开发诊断，并非盲测。

本机 MPS 共 2 次对照＋18 次整图前向，整图循环约 6.34 秒；无隐藏预热、训练、付费 GPU 或新下载。审计重新构造输入张量、核对全部缓存及 288 个旧原图格，用另写的整数计数核对所有 72 个评分及 36 个变化；{audit['old_frozen_sources_unchanged']} 份旧冻结源码不变。270 项测试通过，其中本阶段新增 8 项。整图推理的调用数和耗时不能与旧裁块推理总时直接解释为成本优势；数据采购的信息范围不同。

三张事件叠加图均包含该事件全部六对，青色为发布严重参考、橙色为模型严重输出，见 `figures/*_overlays.png`。代理查看可作几何和展示检查，不等于有独立人工标注。

## 下一轮已经锁定的工作

`prospective_information_diagnostic.json` 固定同一 18 对、每对全部 16 格，并要求先锁实现哈希，再产生新输出：

1. 缓存 128 原图格作对照；同一格放大到 512，检查输入尺度影响。
2. 只把已付费的 128 原图格放进固定空白 512 画布，控制大画布与边界影响。
3. 把该格放进免费 64 BOX 粗图重建的 512 画布，检查粗上下文是否提供额外作用。
4. 增加仅免费粗图对照，区分原图格带来的增量与免费信息本身可达到的覆盖。

三个逐格条件各 288 次，仅粗图 18 次，加原两个回归对照，最多 **884 次本机前向**。画布条件只保留已付费格的预测；变换函数只接收粗图和已选局部，不能偷偷使用其余原图。保留全部事件、零目标场景及两个评价域，不按标签调参数。

下一轮先作原图输入诊断，不把它改称主要混合质量实验。只有免费粗图＋局部通过预先写明的参考覆盖、精度、空白画布和粗图增量门槛，才另立模糊缓存扩展；原图／模糊两套测量均冻结后，才能重跑预算 8 的简单搜索对照。当前没有新控制器或路径最优结论。

## 解释边界与复核

EBD 发布语义参考尚无本项目新增人工复核，不是逐栋实例真值或现场准确率。模型与标注构建均有 xBD 来源关联，原地理独立性／影像许可链限制继续保留。18 对永久作可行性／开发数据；旧验证、确认及 216 候选状态未改变。

核心文件：`protocol.json`、`positive_control_results.json`、`inference_report.json`、`summary.json`、`evidence_summary.json`、`audit.json`、`completion_receipt.json`。原区域预算搜索的失败结果保留在上一阶段目录，整图改进不会追溯替换旧成绩。完成目录及源码快照冻结；重放时使用新的输出审计目录。
"""
    (root/"README.md").write_text(text)
    print(json.dumps({"followup_screen_passed":passed,"covered_reference_pixels_before":old["covered_reference_pixels"],"covered_reference_pixels_after":new["covered_reference_pixels"],"stage2_executed":False}))


if __name__=="__main__":main()
