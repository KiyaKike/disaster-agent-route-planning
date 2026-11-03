"""Close the fixed blur candidate and document the next metadata-only replication."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_quality256_core import QUALITIES,mosaic
from run_external_context_diagnostic import checked_mask


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--area",type=Path,required=True);args=p.parse_args();root=args.root
    read=lambda name:json.loads((root/name).read_text());protocol=read("protocol.json");summary=read("summary.json");decision=read("next_step_decision.json");audit=read("audit.json");report=read("inference_report.json")
    rows=[r for r in summary["event_rows"] if r["domain"]=="all_published_pixels"];events=sorted({r["event"] for r in rows});by_event={(r["event"],r["input_mode"]):r for r in rows}
    totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]=="all_published_pixels"};changes=[r for r in read("paired_changes.json") if r["domain"]=="all_published_pixels"]
    feedback=read("block_feedback.json");matrix=[[sum(r["native_declared_target"]==a and r["blurred_declared_target"]==b for r in feedback) for b in [False,True]] for a in [False,True]]
    observations=read("observations.json")
    evidence={"primary_screen_passed":decision["primary_screen_passed"],"full16_area_native":totals[QUALITIES[0]],"full16_area_blurred":totals[QUALITIES[1]],
        "gained_reference_pixels":sum(r["gained_reference_pixels"] for r in changes),"lost_reference_pixels":sum(r["lost_reference_pixels"] for r in changes),"preserved_reference_pixels":sum(r["preserved_reference_pixels"] for r in changes),
        "blurred_masks_all_zero":all(r["predicted_severe_pixels"]==0 for r in observations),"blurred_boolean_positives":sum(r["declared_target"] for r in observations),
        "blurred_predicted_building_pixels":sum(r["predicted_building_pixels"] for r in observations),"native_vs_blur_boolean_matrix":matrix,
        "matrix_is_prediction_comparison_not_ground_truth_confusion":True,"native_only_footprint_evidence_preserved":True,
        "budget8_revisit_feasible":False,"new_search_method_selected":False,"new_search_runs":0,"physical_accuracy_claim":False}
    atomic_json(root/"evidence_summary.json",evidence)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    folder=root/"figures";folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.1),layout="constrained");x=np.arange(3)
    for i,quality in enumerate(QUALITIES):
        values=[by_event[e,quality]["covered_reference_pixels"] for e in events]
        bars=axes[0].bar(x+(i-.5)*.32,values,width=.32,color=["#2d7e88","#ca9966"][i],label=["Cached native256","Fixed blurred256"][i])
        for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+70,str(value),ha="center",fontsize=8)
    axes[0].set_xticks(x,["Turkey","Pakistan","Texas"]);axes[0].set_ylim(0,5000);axes[0].set_title("Purchased full16-area,6 scenes/event")
    axes[0].set_ylabel("Published severe-reference pixels covered");axes[0].legend(fontsize=8);axes[0].text(2,500,"No severe reference\nrecall N/A",ha="center",fontsize=8);axes[0].grid(axis="y",alpha=.15);axes[0].set_axisbelow(True)
    axes[1].imshow(np.asarray(matrix),cmap="Blues",vmin=0,vmax=72)
    for a in range(2):
        for b in range(2):axes[1].text(b,a,str(matrix[a][b]),ha="center",va="center",fontsize=18,color="white" if matrix[a][b]>35 else "#243238")
    axes[1].set_xticks([0,1],["Blur negative","Blur positive"]);axes[1].set_yticks([0,1],["Native negative","Native positive"]);axes[1].set_title("Boolean model outputs on72 blocks\nPrediction comparison, not truth labels",fontsize=10)
    fig.suptitle("Fixed256 blur quality check | 18 development pairs | no new search runs",fontsize=11)
    for ext in ["png","pdf"]:fig.savefig(folder/("quality256_comparison."+ext),dpi=175)
    plt.close(fig)
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};refs={r["sample_id"]:r for r in json.loads((args.area/"reference_manifest.json").read_text())}
    for event in events:
        ids=[sid for sid in protocol["selected_ids"] if refs[sid]["event"]==event];fig,axes=plt.subplots(6,4,figsize=(12,17),layout="constrained")
        for r,sid in enumerate(ids):
            with Image.open(images[sid,"post"]["path"]) as im:post=np.asarray(im.convert("RGB"))
            with Image.open(refs[sid]["path"]) as im:ref=np.asarray(im)>=3
            native=mosaic({row["block_id"]:checked_mask(Path(row["cache_file"]),row["cache_sha256"],(256,256)) for row in protocol["previous_native256_observations"] if row["sample_id"]==sid})
            blurred=mosaic({row["block_id"]:checked_mask(Path(row["cache_file"]),row["cache_sha256"],(256,256)) for row in observations if row["sample_id"]==sid})
            for c,mask in enumerate([None,ref,native,blurred]):
                canvas=post.copy()
                if mask is not None:canvas[mask]=(.3*canvas[mask]+.7*np.array([40,190,210] if c==1 else [242,122,54])).astype(np.uint8)
                axes[r,c].imshow(canvas);axes[r,c].axis("off")
                if c==0:axes[r,c].set_title(sid,fontsize=8)
                elif r==0:axes[r,c].set_title(["","Published severe ref","Cached native256","Fixed blurred256"][c],fontsize=9)
        fig.suptitle(event+" | full16-area mosaics, all6 pairs | not budget8 search",fontsize=11)
        fig.savefig(folder/(event+"_native_blur.png"),dpi=110);plt.close(fig)
    atomic_json(folder/"figure_receipt.json",{"summary_sha256":sha((root/"summary.json").read_bytes()),"evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),"generator_sha256":sha(Path(__file__).read_bytes()),
        "all18_pairs_both_qualities_in_overlays":True,"all_overlays_full16_area":True,"human_annotation":False})
    table=[]
    for event in events:
        a=by_event[event,QUALITIES[0]];b=by_event[event,QUALITIES[1]]
        table.append(f"| {event} | {a['reference_pixels']:,} | {a['covered_reference_pixels']:,} | {b['covered_reference_pixels']:,} | {a['reference_false_positive_pixels']:,}／{b['reference_false_positive_pixels']:,} |")
    replication=read("prospective_native_replication36.json") if (root/"prospective_native_replication36.json").exists() else None
    next_text=f"""已读取既有ZIP目录元数据，固定 **新增36对**：三个事件各追加12对，沿用原身份哈希排序的第7–18名，排除原18对，没有按图像外观、标签或检出结果选样。新清单、72张RGB及36份发布灾后掩膜的队列已保存，但下载全部禁用；选中压缩载荷共{replication['selected_compressed_payload_bytes']:,}字节，未来单独精确范围提取上限48MiB。本轮新增载荷读取仍为0。

`prospective_native_replication36.json` 固定原图128／256同面积对照，主预算8、均匀随机精确期望、全部新36对及双域；新增758次本机前向的上限包括576个128格、144个256块、36个免费粗图基线和2个旧接入对照。现阶段没有执行这些推理。下载实现及后续执行源码需在产生新结果前另行冻结。

新36对先独立报告；要求有效发现机会扩展到至少6个新场景并覆盖至少两个事件，同时检查面积增量、参考精度及事件方向。若未达到预设条件，报告失败或证据不足，不能补挑阳性、把旧18并入救分或继续搜索变体。即使成功，也只是同来源、同事件的前瞻测量复验，不能代替地理／人工独立确认。""" if replication else "下一步需先明确新增原图样本的固定身份和复验范围，再读取新载荷；当前没有新数据或搜索策略结果。"
    text=f"""# 256观察块的固定模糊质量检查

日期：2026-10-26。状态：74次本机前向及全部测量重放完成。**固定模糊256没有保留严重参考重叠，按事先条件停止这版混合质量候选。原图256的测量结果保持有效。**

## 完整结果

以下为每场景四块全部覆盖、购买16单位原图面积的测量诊断，不是预算8搜索成绩。

| 事件，每个6对 | 严重参考像素 | 原图256覆盖 | 模糊256覆盖 | 原图／模糊参考相对误报 |
|---|---:|---:|---:|---:|
"""+"\n".join(table)+f"""

模糊72个严重预测掩膜全部为0：参考重叠0、严重布尔阳性0；原图覆盖5,353个参考像素，全部在模糊条件下丢失，保留／新增均为0。没有严重预测时参考精度为null，不能把零误报解释成高准确率。得州没有严重参考，召回同样为null。

全零特指严重损伤掩膜：模糊输入仍有{evidence['blurred_predicted_building_pixels']:,}个模型建筑定位像素，不是全部模型张量均为空。当前没有进一步定位类别边界、训练域或标签语义的唯一原因，也没有据此调阈值、减轻模糊或换另一个输入条件。

72个块中，原图阴性67个、阳性5个；模糊后72个均阴性。这是模型预测之间的变化，不是按人工真值统计的灵敏度／特异度。双时相非零域同样零重叠，门槛也失败；原图在该域仍覆盖5,352个参考像素。

![质量对照]({(folder/'quality256_comparison.png').resolve()})

## 输入与验证

每个已付费256 RGB块只做一次冻结退化：GaussianBlur(1.5)→128 BOX→256 BILINEAR，再用原归一化、同一检查点、解码及至少4像素阳性阈值。退化函数只接收已选256块，不接收512原图或额外上下文。它是同一影像的合成退化，不是新无人机视角、独立重访或实测传感器质量。

模型前先核对全部72组真实输入：退化RGB字节、归一化张量与旧 `degrade_image`／`prepare_tensor(...,blurred)` 完全一致。两个旧开发阳性128裁块再次逐像素复现；随后仅新增72个模糊256观察，旧72个原图256缓存不重跑。调用均先登记预约，无隐藏预热。

314项测试通过，新增12项覆盖退化字节、张量相等、不重复退化、块外信息不可用、几何、固定阈值、参考门槛及免费基线去重。独立审计用另写的Pillow退化和整数计数重建74个张量、72个模糊掩膜，重放72个场景域评分、36个配对变化、288个块域评分和72对布尔反馈；旧36个原图场景域结果保持一致。{audit['old_frozen_sources_unchanged']}份旧冻结源码不变。

本机MPS循环约{report['loop_seconds']:.2f}秒，没有训练、付费GPU、新下载、策略拟合或保留集评分。三张事件叠加图保留全部18对及两个质量，青色为发布严重参考、橙色为严重预测；代理查看不代表人工语义或配准认证。

## 研究范围的决定

停止当前“合成模糊＋原图升级”的混合质量候选，不在这18对上继续调模糊、阈值或画布来追结果。原图256此前在同面积预算8下相对128的测量改善保留；它仍只有2/18有效场景，不能直接上升为新搜索方法。

成本限制也保持：256初查成本4，按原2倍系数的原图升级成本8，主预算8无法执行4＋8。因此本轮不能产生该预算的复核收益结论，更不通过更换主预算制造增益。

后续待验证的问题收缩为：**原图质量下，观察单元大小与区域选择能否在同等付费面积下带来稳定增量。** 先扩大有效样本重复固定对照，测量可靠后才考虑算法贡献。

## 下一步已做的准备

{next_text}

当前18对永久为可行性／开发数据。发布EBD语义参考不是本项目新增人工逐栋真值；原始影像来源、许可链及地理独立性局限继续保留，未创建实例标签。旧216候选、旧验证／确认集状态不变。

复核入口：`protocol.json`、`prepared_inputs`协议字段、`calls/`、`positive_control_results.json`、`inference_report.json`、`summary.json`、`paired_changes.json`、`block_feedback.json`、`audit.json`、`next_step_decision.json`及完成收据。完成输出冻结；需重放时用新的审计输出目录。
"""
    (root/"README.md").write_text(text)
    print(json.dumps({"primary_screen_passed":decision["primary_screen_passed"],"blurred_severe_masks_all_zero":evidence["blurred_masks_all_zero"],"native_result_preserved":True,"new36_payloads_read":0}))


if __name__=="__main__":main()
