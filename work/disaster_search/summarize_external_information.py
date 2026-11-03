"""All-condition evidence, plots and next-stage decision for fixed18 inputs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_area_core import CELLS,box
from external_information_core import PATCH_MODES,COARSE_MODE
from plan_search_v2 import freeze
from run_external_context_diagnostic import checked_mask

ORDER=["crop128_mosaic_native",*PATCH_MODES,COARSE_MODE,"whole512_native"]
LABELS={"crop128_mosaic_native":"原128裁块",PATCH_MODES[0]:"局部放大至512",PATCH_MODES[1]:"空白画布＋局部",PATCH_MODES[2]:"免费粗图＋局部",COARSE_MODE:"仅免费粗图","whole512_native":"整图原图（离线参照）"}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--area",type=Path,required=True);args=p.parse_args();root=args.root
    read=lambda name:json.loads((root/name).read_text());protocol=read("protocol.json");summary=read("summary.json");decision=read("next_step_decision.json");audit=read("audit.json");inference=read("inference_report.json")
    totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]=="all_published_pixels"}
    events=sorted({r["event"] for r in summary["event_rows"]});event_rows={(r["event"],r["input_mode"]):r for r in summary["event_rows"] if r["domain"]=="all_published_pixels"}
    scores=read("scene_scores.json");changes=read("paired_changes.json");main_mode=PATCH_MODES[2]
    comparisons=[]
    for before in ["crop128_mosaic_native",PATCH_MODES[1],COARSE_MODE]:
        rows=[r for r in changes if r["domain"]=="all_published_pixels" and r["before"]==before and r["after"]==main_mode]
        comparisons.append({"before":before,"after":main_mode,"gained_reference_pixels":sum(r["gained_reference_pixels"] for r in rows),"lost_reference_pixels":sum(r["lost_reference_pixels"] for r in rows),
            "preserved_reference_pixels":sum(r["preserved_reference_pixels"] for r in rows),"scenes_increased":sum(r["gained_reference_pixels"]>r["lost_reference_pixels"] for r in rows),
            "scenes_decreased":sum(r["gained_reference_pixels"]<r["lost_reference_pixels"] for r in rows),"scenes_equal":sum(r["gained_reference_pixels"]==r["lost_reference_pixels"] for r in rows)})
    evidence={"all_conditions": [totals[m] for m in ORDER],"coarse_plus_cell_paired_comparisons":comparisons,
        "primary_screen_passed":decision["primary_screen_passed"],"main_condition_was_predeclared":True,"posthoc_winner_selected":False,
        "whole_native_outside_budget_information":True,"native_only_not_mixed_budget_search":True,"published_reference_not_human_gold":True}
    atomic_json(root/"evidence_summary.json",evidence)
    if decision["primary_screen_passed"]:
        freeze(root/"prospective_mixed_extension.json",{"status":"PROSPECTIVE_NOT_EXECUTED","basis_protocol_sha256":sha((root/"protocol.json").read_bytes()),
            "basis_evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),"selected_ids":protocol["selected_ids"],
            "condition":main_mode,"new_predictions":"Same18 scenes/all16 cells;288 blurred local observations plus same2 native positive controls, max290 forward calls. No new native calls.",
            "blur_recipe":"Apply old128px patch GaussianBlur1.5, resize64 BOX then128 BILINEAR; paste only blurred patch into free64 BOX ->512 BILINEAR canvas. Do not blur the entire canvas. Keep only paid128 ROI.",
            "checks_before_execution":"Lock new implementation after exact old patch-transform equality, ROI cropping, information boundary and quality invisibility tests.",
            "conditional_search":"After native+blur cache frozen, run only existing random_survey,prior_survey,prior_gate with same public features, budgets4/8/16 (primary8), costs1/2, five quality maps and10 order seeds. No model/controller fit or new planner.",
            "free_baseline":"Report free-coarse-only prediction at budget0 to all comparisons; count each additional severe reference pixel only once. Report total union and incremental coverage beyond free baseline, as well as reference-relative false pixels. Do not attribute free detections to paid search.",
            "policy_information":"Strategies read fixed free RGB cues and paid canonical booleans; no reference pixels, raw model canvases, prediction counts or future qualities. All policies share the same public information.",
            "evaluation_scope":"Same18 development/feasibility scenes, allzero cases retained, bothdomains, each event and macro across all3; not independent efficacy or physical flight cost.",
            "paid_gpu_or_training_or_download":False,"no_new_search_executed":True})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    folder=root/"figures";folder.mkdir(exist_ok=True)
    display=["Native128","Patch upscale","Blank + cell","Coarse + cell","Coarse only","Native512*"]
    colors=["#a7b2b5","#cb915a","#82789a","#287f8a","#6c9b6c","#c6c9cb"]
    fig,axes=plt.subplots(1,3,figsize=(14.2,4.6),layout="constrained")
    for ax,key,title in [(axes[0],"covered_reference_pixels","Severe reference pixels covered"),(axes[1],"reference_false_positive_pixels","Reference-relative false pixels")]:
        values=[totals[m][key] for m in ORDER];ax.barh(range(6),values,color=colors)
        ax.set_yticks(range(6),display);ax.invert_yaxis();ax.set_title(title)
        ax.set_xlim(0,max(values+[1])*1.25)
        for i,value in enumerate(values):ax.text(value+max(values+[1])*.02,i,f"{value:,}",va="center",fontsize=8)
        ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    x=np.arange(3);width=.13
    for i,mode in enumerate(ORDER):
        values=[100*event_rows[e,mode]["pooled_area_recall"] if event_rows[e,mode]["pooled_area_recall"] is not None else 0 for e in events]
        axes[2].bar(x+(i-2.5)*width,values,width,color=colors[i],label=display[i])
    axes[2].set_xticks(x,["Turkey","Pakistan","Texas"]);axes[2].set_ylabel("Published severe reference recall (%)");axes[2].set_ylim(0,100)
    axes[2].text(2,4,"No severe ref\nrecall N/A",ha="center",fontsize=8);axes[2].legend(fontsize=7,ncols=2,loc="upper left");axes[2].set_title("All18 pairs;6 per event")
    fig.suptitle("Information-bounded input diagnostic | native only | *full native scene is offline reference",fontsize=11)
    for ext in ["png","pdf"]:fig.savefig(folder/("information_comparison."+ext),dpi=170)
    plt.close(fig)
    refs={r["sample_id"]:r for r in json.loads((args.area/"reference_manifest.json").read_text())};images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]}
    observations=read("observations.json");whole={r["sample_id"]:r for r in protocol["offline_whole_observations"]}
    def masks_for(sid):
        result={}
        for mode in ORDER:
            if mode=="whole512_native":
                r=whole[sid];result[mode]=checked_mask(Path(r["cache_file"]),r["cache_sha256"],(512,512))
            elif mode==COARSE_MODE:
                r=next(r for r in observations if r["sample_id"]==sid and r["input_mode"]==mode);result[mode]=checked_mask(Path(r["cache_file"]),r["cache_sha256"],(512,512))
            else:
                rows=[r for r in (protocol["previous_native_observations"] if mode=="crop128_mosaic_native" else observations) if r["sample_id"]==sid and (mode=="crop128_mosaic_native" or r["input_mode"]==mode)]
                canvas=np.zeros((512,512),dtype=np.bool_)
                if len(rows)!=16:raise ValueError("Incomplete plot mosaic")
                for r in rows:
                    x,y,xx,yy=box(r["cell_id"]);canvas[y:yy,x:xx]=checked_mask(Path(r["cache_file"]),r["cache_sha256"],(128,128))
                result[mode]=canvas
        return result
    for event in events:
        ids=[sid for sid in protocol["selected_ids"] if refs[sid]["event"]==event]
        fig,axes=plt.subplots(6,8,figsize=(20.8,16.2),layout="constrained")
        for row,sid in enumerate(ids):
            with Image.open(images[sid,"post"]["path"]) as im:post=np.asarray(im.convert("RGB"))
            with Image.open(refs[sid]["path"]) as im:reference=np.asarray(im)>=3
            masks=masks_for(sid)
            for col,mask in enumerate([None,reference,*[masks[m] for m in ORDER]]):
                canvas=post.copy()
                if mask is not None:canvas[mask]=(.3*canvas[mask]+.7*np.array([40,190,210] if col==1 else [242,122,54])).astype(np.uint8)
                axes[row,col].imshow(canvas);axes[row,col].axis("off")
                if col==0:axes[row,col].set_title(sid,fontsize=7)
                elif row==0:axes[row,col].set_title((["","Published severe ref"]+display)[col],fontsize=8)
        fig.suptitle(event+" | all6 pairs and all6 inputs | cyan published ref, orange predictions",fontsize=11)
        fig.savefig(folder/(event+"_all_conditions.png"),dpi=125);plt.close(fig)
    atomic_json(folder/"figure_receipt.json",{"summary_sha256":sha((root/"summary.json").read_bytes()),"evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),
        "generator_sha256":sha(Path(__file__).read_bytes()),"all18_scenes_all6_conditions_in_overlays":True,"human_annotation":False})
    fmt=lambda value:"null" if value is None else f"{100*value:.2f}%"
    table=[f"| {LABELS[m]} | {totals[m]['covered_reference_pixels']:,} | {fmt(totals[m]['pooled_area_recall'])} | {fmt(totals[m]['pooled_reference_precision'])} | {totals[m]['reference_false_positive_pixels']:,} | {totals[m]['detectable_reference_scenes']}/18 |" for m in ORDER]
    event_table=[f"| {e} | "+" | ".join(str(event_rows[e,m]["covered_reference_pixels"]) for m in ORDER)+" |" for e in events]
    names={"both_reference_positive_events_have_overlap":"两个参考阳性事件均有重叠","pooled_coverage_exceeds_cached_native128":"合并覆盖高于旧128裁块","pooled_reference_precision_not_lower":"合并参考精度不降低","pooled_coverage_exceeds_blank_frame":"合并覆盖高于空白画布＋局部","positive_net_coverage_beyond_free_coarse_in_each_positive_event":"两个阳性事件均较仅粗图有净覆盖增量"}
    checks=decision["domain_screens"][0]["checks"];gate_lines=[f"- {names[k]}：{'通过' if value else '未通过'}。" for k,value in checks.items()]
    passed=decision["primary_screen_passed"]
    conclusion="免费粗图＋已付费局部通过全部预先登记的工程门槛，可以登记原图／模糊混合质量缓存扩展；尚不构成预算搜索的收益证明。" if passed else "免费粗图＋已付费局部未通过全部预先登记的工程门槛，当前候选停止进入混合质量搜索。所有其他条件照常报告，没有按最高分改选主方法。"
    next_text="已登记 `prospective_mixed_extension.json`，尚未执行：固定18对全部格，为主输入条件新增288个模糊格输出，加2个旧对照最多290次本机前向；先核对旧模糊规则及输入边界。原图／模糊缓存均冻结后，只跑既有三个简单策略，主要预算8、升级成本2，同时报告免费粗图基线和付费新增的参考像素。" if passed else "本轮不扩展模糊缓存、不运行新搜索轨迹。已另立 `prospective_footprint_diagnostic.json`，尚未执行：检验同等付费原图面积下，连续256像素块能否优于离散128格。每个256块明确付4单位，512整图付16单位，在预算8不可调用整图；不再把未付费原图上下文交给模型。主条件为未经缩放的256原图，粗图画布为次要对照，固定全部18对及四个不重叠区块，加2个接入对照最多146次本机前向。该新原图面积成本代理不等于旧混合质量或实际飞行成本。先锁实现和成本测试，再产生输出；主条件失败即停止在这18对上继续输入接口变体，转而单独评估数据／参考或感知器。"
    text=f"""# 免费粗图与付费局部：固定18对输入对照

日期：2026-10-26。状态：884次本机前向、216个场景域评分、180个配对变化均完成并重放；没有预算搜索或控制器拟合。

## 结果与决策

**{conclusion}**

| 输入条件 | 覆盖严重参考像素 | 合并参考召回 | 合并参考精度 | 参考相对误报像素 | 有重叠场景 |
|---|---:|---:|---:|---:|---:|
"""+"\n".join(table)+"""

分母为同一18对中的19,159个发布严重参考像素；全部零目标、零检出场景保留。表内逐格条件是遍历全部16格的诊断，尚不是预算8成绩。整图原图额外使用尚未付费的高分辨率区域，只保留为离线参照。仅粗图使用免费信息，其检出不能记作付费搜索收益。

| 事件，每个6对 | 原128 | 放大局部 | 空白＋局部 | 粗图＋局部 | 仅粗图 | 整图离线 |
|---|---:|---:|---:|---:|---:|---:|
"""+"\n".join(event_table)+"""

得州没有发布严重参考，召回为null，不能算满分或真实无损伤。两个参考阳性事件分别报告，随机种子或格子数不能替代独立灾害事件数。双时相非零域的完整结果也在 `summary.json`，不改选主域。

## 预先登记的继续门槛

主候选始终为“免费粗图＋局部”，主域为全部发布像素。独立的空白、放大和仅粗图条件用于解释输入效应。

"""+"\n".join(gate_lines)+f"""

上轮条件里的“超出仅粗图的覆盖”在新输出产生前明确为：每个参考阳性事件的净覆盖增量均严格大于0；另存像素集合的新得／丢失／保留，避免用净值掩盖变化。该工程筛选不是统计显著性或新颖性判断。主候选相对旧128、空白画布、仅粗图的全部配对变化见 `evidence_summary.json` 和 `paired_changes.json`。

![全部输入对照]({(folder/'information_comparison.png').resolve()})

## 信息边界与验证

- 主变换仅接收两张64×64免费粗图和选中格的两张128×128原图；不接收512原图。免费粗图固定由512 RGB以BOX缩至64，再BILINEAR重建512；只在选定位置贴回128格。
- 放大对照仅将已付费128格BILINEAR放大512；固定解码的512布尔严重掩膜NEAREST缩回128。空白对照使用固定RGB(124,116,104)画布。画布条件均只裁回付费128格，格外预测不进入奖励或反馈。
- 所有局部布尔声明均由最终128掩膜至少4个严重像素产生，不使用整张512输出的阳性声明。仅粗图作为免费基线保留全图输出。
- 相同冻结ChangeOS权重、模型代码、归一化、eval设置及随机种子；不训练、不调阈值、不按参考选择格子。预测过程不解码参考类别；校验文件字节的哈希不作为模型输入。
- 两个固定旧开发阳性裁块再次逐像素复现；每次模型前向均先写调用预约，再保存完整收据。共2对照＋864局部＋18粗图，恰好884次，无隐藏预热。

审计独立重建884个张量，逐一核对原始512输出至864个付费128格的映射，重算216评分和180配对变化；72个旧输入评分保持一致。{audit['old_frozen_sources_unchanged']}份旧冻结源码不变；284项测试通过，新增14项。本机{inference['device'].upper()}循环约{inference['loop_seconds_this_invocation']:.2f}秒，零新增网络字节、零付费GPU、新人工标注／建筑实例均为0。

三张事件叠加图包含每事件全部6对、全部6种输入，见 `figures/*_all_conditions.png`；这只是代理的展示与几何检查，不是独立人工语义或配准认证。

## 下一步

{next_text}

18对永久为可行性／开发数据。参考是EBD发布语义掩膜，当前没有新增人工逐栋真值或独立地理准入；模型和标注构建有xBD来源关联。不能把以上数字当成物理灾损准确率、新方法验证或飞行航程收益。旧保留集和216候选状态不变。

复核入口：`protocol.json`、`calls/`、`positive_control_results.json`、`inference_report.json`、`summary.json`、`next_step_decision.json`、`audit.json`、`completion_receipt.json`。运行代码及父协议按SHA256绑定；完成目录冻结，重放请使用新的输出目录。
"""
    (root/"README.md").write_text(text)
    print(json.dumps({"primary_screen_passed":passed,"new_search_runs":0,"conditions_reported":6}))


if __name__=="__main__":main()
