"""Report fixed native-area comparisons, preserving event failures and primary scope."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_footprint_core import MODES,NATIVE128,NATIVE256,CONTEXT256,FULL512,units
from evaluate_external_footprint import load_scene
from plan_search_v2 import freeze

LABELS={NATIVE128:"128原图格，成本1",NATIVE256:"256连续原图块，成本4",CONTEXT256:"免费粗图＋256块，成本4",FULL512:"512整图，成本16"}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--area",type=Path,required=True);args=p.parse_args();root=args.root
    read=lambda name:json.loads((root/name).read_text());protocol=read("protocol.json");summary=read("summary.json");decision=read("next_step_decision.json");audit=read("audit.json");inference=read("inference_report.json")
    main={(r["input_mode"],r["policy"]):r for r in summary["totals"] if r["domain"]=="all_published_pixels" and r["budget"]==8}
    a=main[NATIVE256,"random_exact"];b=main[NATIVE128,"random_exact"];events=sorted({r["event"] for r in summary["event_rows"]})
    event_rows={(r["event"],r["input_mode"],r["policy"]):r for r in summary["event_rows"] if r["domain"]=="all_published_pixels" and r["budget"]==8}
    opportunities=read("full_opportunities.json")
    full=[]
    for mode in MODES:
        rows=[r for r in opportunities if r["domain"]=="all_published_pixels" and r["input_mode"]==mode]
        full.append({"input_mode":mode,"covered_reference_pixels":sum(r["covered_reference_pixels"] for r in rows),
            "reference_false_positive_pixels":sum(r["reference_false_positive_pixels"] for r in rows),"detectable_scenes":sum(r["incremental_covered_reference_pixels"]>0 for r in rows),
            "detectable_events":len({r["event"] for r in rows if r["incremental_covered_reference_pixels"]>0})})
    evidence={"primary_native_only_budget8":a,"cached128_comparison":b,"primary_random_coverage_ratio":a["event_macro_incremental_covered_pixels_per_scene"]/b["event_macro_incremental_covered_pixels_per_scene"],
        "primary_screen_passed":decision["primary_screen_passed"],"full_native_area16_opportunities":full,
        "ranked256_zero_reward_events":[e for e in events if event_rows[e,NATIVE256,"raw_change_rank"]["incremental_covered_reference_pixels"]==0],
        "all18_retained":True,"old_mixed_quality_scores_not_compared":True,"secondary_not_promoted":True,
        "measurement_feasibility_only":True,"new_search_method_selected":False}
    atomic_json(root/"evidence_summary.json",evidence)
    if decision["primary_screen_passed"]:
        freeze(root/"prospective_native256_quality_check.json",{"status":"PROSPECTIVE_NOT_EXECUTED","client_report_date":"2026-10-26",
            "basis_protocol_sha256":sha((root/"protocol.json").read_bytes()),"basis_evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),
            "basis_decision_sha256":sha((root/"next_step_decision.json").read_bytes()),"selected_ids":protocol["selected_ids"],"blocks":protocol["prospective_protocol"]["blocks"],"checkpoint":protocol["checkpoint"],
            "question":"Does the fixed synthetic blur preserve usable severe-reference coverage for native256 observation footprints?",
            "fixed_primary_condition":NATIVE256,"secondary_context256_mixed_extension_enabled":False,
            "new_inference":"Same18 pairs x4 fixed blocks =72 blurred256 calls, preceded by same2 locked128 native positive controls. Reuse72 native256 caches unchanged.",
            "max_new_forward_calls":74,"no_hidden_warmup":True,
            "blur_recipe":"For each purchased256 RGB date apply exactly existing degrade_image(...,blurred):GaussianBlur1.5 ->128 BOX ->256 BILINEAR. Same normalization/checkpoint/decoder; no additional resize or free-coarse canvas.",
            "canonical_output":"256 boolean severe mask, >=4px declaration from full purchased256 footprint. No predictions from unpurchased pixels.",
            "required_preflight":"Verify byte-for-byte equality of blur inputs to frozen shared helper,4-block geometry, canonical256 output, allowed information and fixed code hashes before model calls.",
            "evaluation":"Full16-area native/blur per-scene/event coverage, false pixels, gained/lost/preserved reference pixels; bothdomains; all18 includingzero targets. No new planner, utility fit or adaptive search scores.",
            "engineering_measurement_screen":"Advance mixed-quality feasibility only if blurred256 retains nonzero published severe-reference overlap in BOTH reference-positive events. Keep native-only positive result separate if this fails; do not retune blur or select another input condition.",
            "area_cost_consequence":"Initial256 block observation costs4 native128-equivalent area units; at mixed quality native/blur known only after acquisition. A subsequent native upgrade costs8 under the prior2x upgrade multiplier. At budget8,4+8 is infeasible: no revisit benefit can be claimed. Any later full search protocol must state this and cannot switch primary budget to manufacture a gain.",
            "blur_limitation":"Same original pixels with synthetic degradation; not a new UAV view, independent observation or measured sensor quality.",
            "free_baseline":"Reuse free-coarse-only predictions and keep free detections out of paid incremental credit.",
            "cohort_status":"These18 remain feasibility/development. No new human annotation, independent geography or confirmed physical accuracy.",
            "no_training_paid_gpu_download_or_reserved_scores":True,"new_search_runs":0})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    folder=root/"figures";folder.mkdir(exist_ok=True);shown=[NATIVE128,NATIVE256,CONTEXT256]
    colors={NATIVE128:"#b97f56",NATIVE256:"#247e88",CONTEXT256:"#8c81a0",FULL512:"#9da8aa"}
    fig,axes=plt.subplots(1,3,figsize=(13.1,4.1),layout="constrained")
    values=[main[m,"random_exact"]["event_macro_incremental_covered_pixels_per_scene"] for m in shown]
    bars=axes[0].bar(["128 cells","256 blocks","Coarse +256"],values,color=[colors[m] for m in shown])
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+3,f"{value:.2f}",ha="center",fontsize=9)
    axes[0].set_ylim(0,185);axes[0].set_ylabel("Expected NEW severe-reference pixels / scene");axes[0].set_title("Budget8, exact uniform random")
    x=np.arange(3);width=.22
    for i,mode in enumerate(shown):
        vals=[event_rows[e,mode,"random_exact"]["incremental_covered_reference_pixels"]/6 for e in events]
        axes[1].bar(x+(i-1)*width,vals,width,color=colors[mode],label=["128 cells","256 blocks","Coarse +256"][i])
    axes[1].set_xticks(x,["Turkey","Pakistan","Texas"]);axes[1].set_title("Same area budget;6 pairs per event");axes[1].set_ylabel("Expected NEW reference pixels / scene");axes[1].legend(fontsize=8)
    axes[1].text(2,12,"No severe ref\nrecall N/A",ha="center",fontsize=8)
    for mode in [NATIVE128,NATIVE256,FULL512]:
        rows=sorted([r for r in summary["totals"] if r["domain"]=="all_published_pixels" and r["input_mode"]==mode and r["policy"]=="random_exact" and r["affordable"]],key=lambda r:r["budget"])
        axes[2].plot([r["budget"] for r in rows],[r["event_macro_incremental_covered_pixels_per_scene"] for r in rows],marker="o",color=colors[mode],label={NATIVE128:"128 random",NATIVE256:"256 random",FULL512:"512: cost16 only"}[mode])
    rows=sorted([r for r in summary["totals"] if r["domain"]=="all_published_pixels" and r["input_mode"]==NATIVE256 and r["policy"]=="raw_change_rank"],key=lambda r:r["budget"])
    axes[2].plot([r["budget"] for r in rows],[r["event_macro_incremental_covered_pixels_per_scene"] for r in rows],"--",marker="x",color=colors[NATIVE256],label="256 raw-change rank")
    axes[2].set_xticks([4,8,16]);axes[2].set_xlabel("Purchased128-cell-equivalent native area");axes[2].set_ylabel("NEW reference pixels / scene,3-event macro");axes[2].set_title("Native only; rank is secondary");axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.suptitle("Contiguous observation footprint | native quality | fixed18 feasibility pairs",fontsize=11)
    for ext in ["png","pdf"]:fig.savefig(folder/("footprint_comparison."+ext),dpi=175)
    plt.close(fig)
    scenes={sid:load_scene(sid,root,args.area,protocol) for sid in protocol["selected_ids"]};images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]}
    for event in events:
        ids=[sid for sid in protocol["selected_ids"] if scenes[sid][0]==event];fig,axes=plt.subplots(6,6,figsize=(17,16.4),layout="constrained")
        for row,sid in enumerate(ids):
            _,ref,_,_,masks=scenes[sid]
            with Image.open(images[sid,"post"]["path"]) as im:post=np.asarray(im.convert("RGB"))
            mosaics=[]
            for mode in MODES:
                canvas=np.zeros((512,512),dtype=np.bool_)
                for unit in units(mode):
                    xx,yy,x1,y1=unit["bounds"];canvas[yy:y1,xx:x1]=masks[mode][unit["unit_id"]]
                mosaics.append(canvas)
            for col,mask in enumerate([None,ref,*mosaics]):
                canvas=post.copy()
                if mask is not None:canvas[mask]=(.3*canvas[mask]+.7*np.array([40,190,210] if col==1 else [242,122,54])).astype(np.uint8)
                axes[row,col].imshow(canvas);axes[row,col].axis("off")
                if col==0:axes[row,col].set_title(sid,fontsize=7)
                elif row==0:axes[row,col].set_title(["","Published severe ref","Native128 mosaic","Native256 mosaic","Coarse +256 mosaic","Native512"][col],fontsize=8)
        fig.suptitle(event+" | all6 pairs | FULL16-area mosaics, not budget8 acquisitions",fontsize=11)
        fig.savefig(folder/(event+"_full_coverage.png"),dpi=120);plt.close(fig)
    atomic_json(folder/"figure_receipt.json",{"summary_sha256":sha((root/"summary.json").read_bytes()),"evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),"generator_sha256":sha(Path(__file__).read_bytes()),
        "all18_scenes_and4_modes_in_overlays":True,"overlays_are_area16_full_coverage_not_budget8":True,"human_annotation":False})
    fmt=lambda value:"null" if value is None else f"{100*value:.2f}%"
    table=[]
    for mode in [NATIVE128,NATIVE256,CONTEXT256]:
        random=main[mode,"random_exact"];rank=main[mode,"raw_change_rank"]
        table.append(f"| {LABELS[mode]} | {random['event_macro_incremental_covered_pixels_per_scene']:.3f} | {fmt(random['reference_precision'])} | {random['event_macro_false_pixels_per_scene']:.3f} | {rank['event_macro_incremental_covered_pixels_per_scene']:.3f} |")
    event_table=[]
    for event in events:
        before=event_rows[event,NATIVE128,"random_exact"];after=event_rows[event,NATIVE256,"random_exact"];rank=event_rows[event,NATIVE256,"raw_change_rank"]
        event_table.append(f"| {event} | {before['incremental_covered_reference_pixels']:.1f} | {after['incremental_covered_reference_pixels']:.1f} | {fmt(after['area_recall'])} | {rank['incremental_covered_reference_pixels']:.1f} |")
    gate=decision["domain_screens"][0];gate_names={"both_reference_positive_events_have_incremental_opportunity":"两个参考阳性事件均有新增覆盖机会","budget8_exact_random_macro_incremental_coverage_exceeds128":"预算8精确随机的事件宏平均新增覆盖高于128","budget8_pooled_reference_precision_not_below128":"预算8合并参考精度不低于128"}
    gate_lines=[f"- {gate_names[key]}：{'通过' if value else '未通过'}。" for key,value in gate["checks"].items()]
    result="主条件未经缩放的256连续块通过全部预设测量门槛。下一步可以检查模糊质量，尚未选定新的搜索方法。" if decision["primary_screen_passed"] else "主条件未通过全部测量门槛，停止在这18对上继续输入接口变体。次要条件不替代主条件。"
    next_text="已登记 `prospective_native256_quality_check.json`，尚未执行：同18对全部4块补72个256模糊输出，加2个旧阳性对照，最多74次本机前向。完全沿用GaussianBlur1.5／缩半BOX／还原BILINEAR规则，先锁实现再出结果；只做质量测量，不训练或运行新规划。只有模糊256在两个参考阳性事件均保留重叠机会，才继续另立混合质量可行性协议。" if decision["primary_screen_passed"] else "保持当前和全部历史结果，下一轮另立数据／参考或固定感知器评估，先解决测量可用性，再开发规划算法。"
    text=f"""# 连续256块：同等原图面积成本的观察对照

日期：2026-10-26。状态：146次本机前向、2,160个固定排序记录、4,320个排序域评分、432个随机精确期望和7,740个付费单位均核验完成。**{result}**

## 主结果：原图质量，面积预算8

预算8购买8个128格或2个256块。全部18对、三个事件等权平均的新增严重参考覆盖，精确均匀随机下从 **25.417升至148.694像素／场景**，约{evidence['primary_random_coverage_ratio']:.2f}倍；合并参考精度从 **13.55%升至55.14%**。这里是同一固定感知器的观察单元变化，不是新选点算法。

| 观察输入 | 随机精确新增覆盖／场景 | 随机合并参考精度 | 随机参考相对误报／场景 | 固定变化排序新增覆盖／场景 |
|---|---:|---:|---:|---:|
"""+"\n".join(table)+"""

主条件始终为未经缩放的原生256块，主对照为精确均匀随机。免费粗图＋256是已登记的次要条件，未因为其参考精度较高而改选主方法。免费粗图基线在这18对预测0个严重像素，仍逐像素纳入所有策略；免费检出不计入付费新增奖励。

固定变化排序虽有更高的总体面积，256主要设置全部覆盖都来自土耳其，巴基斯坦覆盖为0，低于该事件的随机期望652.5。因此当前结果支持更大观察块的测量可用性，尚不支持该排序对两个灾害都有效。

| 事件，每个6对；表内为事件合计像素 | 128随机期望 | 256随机期望 | 256随机参考召回 | 256固定排序 |
|---|---:|---:|---:|---:|
"""+"\n".join(event_table)+f"""

得州无发布严重参考，召回为null，不计成满分。上述随机小数是离散采样的期望，不是部分物理像素。精度为期望合并覆盖数／期望合并预测数，不是随机轨迹精度的平均值。

## 成本、范围与继续门槛

新合同按购买原图面积计费：128×128成本1，256×256成本4，512×512成本16。这是受控信息面积代理，不是GPU推理时间、实际航程或实测采集延迟。免费64 BOX粗图规则不变；画布对照只保留付费256区域的输出。预算8没有购买完整512图，所有整图获取动作严格为0。

本轮只有原图质量，不能直接与上一轮原图／模糊混合分数比较，也没有通过换主预算或图像尺寸免费扩大信息范围。预算4／8／16、全部事件和两个有效域均在 `summary.json` 保留。512整图仅在预算16显示为可用，预算4／8的0新增只表示无法购买，不能据此称它检出能力为0。

"""+"\n".join(gate_lines)+f"""

双时相非零域中，256主预算精确随机覆盖为148.667像素／场景、参考精度55.15%，门槛方向相同。该检查不能认证配准或标签语义。

![观察面积对照]({(folder/'footprint_comparison.png').resolve()})

## 改善的实际覆盖面

全部16单位原图面积都买到时，256拼接重叠5,353个严重参考像素：土耳其4,048、巴基斯坦1,305；128为915，512整图为4,796。整图输出不是逐像素上界，模型输出随输入接口变化不保证单调。

三种可发现输入仍都只在2/18场景有重叠；5个含严重参考的场景仍有3个零重叠。改善没有扩大有效场景数，不构成独立跨灾害验证。叠加图包括每事件全部6对及四种输入，均为买足16单位面积的展示，不冒充预算8实际获取图。

## 本轮完成与审计

同两个旧开发阳性128裁块再次逐像素一致；18对×4块×2条件，共144次新观察，合计146次前向。本机MPS循环约{inference['loop_seconds']:.2f}秒；没有隐藏预热、新下载、模型／控制器训练或付费GPU。预测不解码参考标签，评价器单独使用EBD发布语义类别3／4。

302项测试通过，新增18项覆盖四块分区、真实像素归属、4／16计费、整图预算限制、精确期望与穷举一致、免费基线去重及无目标null。独立审计重建146个张量和144个256付费输出，按另一套几何及整数并集计数重放2,160条排序记录、4,320域评分和7,740个付费单位；432随机评分另用组合计数的包含概率重算。864个场景策略域均值及144个全覆盖评分核对一致，{audit['old_frozen_sources_unchanged']}份旧冻结源码不变。

原始变化排序只读固定粗图特征，块效用按4个成员格的变化均值，同分取10个冻结格顺序中的最小成员名次；没有参考信息、未来质量或局部反馈驱动重排。精确随机无须挑选随机种子，依据不相交付费单元的包含概率计算，不用理想真值选点。公布的10个同分顺序在场景内平均，不当成10个独立灾害。

## 下一步

{next_text}

面积计费也带来一个限制：若将原有2倍升级系数按面积延续，256初查成本4、原图升级成本8，预算8无法支付4＋8。后续不能宣称主预算8下存在复核收益；需要先明确测量与成本合同，不能为了看到增益改主预算。

18对永久为可行性／开发，未取得独立地理或新增人工语义准入。发布面积参考不等于建筑实例真值或现场物理准确率；旧216候选和保留集状态不变。早前免费粗图＋128候选继续停止，没有用本轮256结果追溯修改。

复核入口：`protocol.json`、`calls/`、`positive_control_results.json`、`inference_report.json`、`ranked_acquisitions.jsonl.gz`、`full_opportunities.json`、`summary.json`、`audit.json`、`next_step_decision.json`及完成收据。源码、检查点和上游输出均绑定哈希；完成目录冻结，重放请用新的输出审计目录。
"""
    (root/"README.md").write_text(text)
    print(json.dumps({"primary_screen_passed":decision["primary_screen_passed"],"random128":b["event_macro_incremental_covered_pixels_per_scene"],"random256":a["event_macro_incremental_covered_pixels_per_scene"],"new_quality_check_executed":False}))


if __name__=="__main__":main()
