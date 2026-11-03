"""Report new36 separately; close or advance only the preregistered measurement screen."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from evaluate_native_replication36 import MODES,load_scene
from external_footprint_core import NATIVE128,NATIVE256,units


def read(path):return json.loads(Path(path).read_text())


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    protocol=read(root/'protocol.json');summary=read(root/'summary.json');decision=read(root/'next_step_decision.json');audit=read(root/'audit.json');inference=read(root/'inference_report.json');extraction=read(root/'extraction_report.json')
    events=sorted({r['event'] for r in summary['event_rows']});main={(r['input_mode'],r['policy']):r for r in summary['totals'] if r['domain']=='all_published_pixels' and r['budget']==8}
    event={(r['event'],r['input_mode'],r['policy']):r for r in summary['event_rows'] if r['domain']=='all_published_pixels' and r['budget']==8}
    opportunities=read(root/'full_opportunities.json');quality=read(root/'rgb_quality.json');images=protocol['rgb_inputs'];grouped=defaultdict(list)
    for r in quality:grouped[r['pair_pixel_sha256']].append(r['sample_id'])
    new_pair_duplicates=[ids for ids in grouped.values() if len(ids)>1]
    old_images=read('output/disaster_search_external_rgb18_v1/rgb_manifest.json');old_lookup={(r['sample_id'],r['mode']):r for r in old_images};old_pairs={}
    for sid in sorted({r['sample_id'] for r in old_images}):
        pixels=[]
        for date in ['pre','post']:
            with Image.open(old_lookup[sid,date]['path']) as im:pixels.append(im.convert('RGB').tobytes())
        old_pairs[sha(pixels[0]+pixels[1])]=sid
    pixel_matches=[{'new_sample_id':r['sample_id'],'old_sample_id':old_pairs[r['pair_pixel_sha256']]} for r in quality if r['pair_pixel_sha256'] in old_pairs]
    reference=read(root/'reference_manifest.json');references={r['sample_id']:r for r in reference};lookup={(r['sample_id'],r['mode']):r for r in images}
    domain_notes=[];scenes={sid:load_scene(root,sid,protocol) for sid in protocol['selected_ids']}
    for sid,(_,target,free,domains,_) in scenes.items():domain_notes.append({'sample_id':sid,'reference_pixels':int(target.sum()),'reference_outside_both_dates_nonzero':int((target&~domains['both_dates_nonzero']).sum()),'free_predicted_pixels':int(free.sum())})
    data={'new_pairs':36,'new_RGB':72,'new_references':36,'all_selected_retained':True,
        'RGB_images_at_least25percent_zero':sum(z>=.25 for r in quality for z in r['zero_rgb_fraction'].values()),
        'maximum_zero_rgb_fraction':max(z for r in quality for z in r['zero_rgb_fraction'].values()),'new_pair_pixel_duplicates':new_pair_duplicates,
        'new_vs_old18_pair_pixel_exact_matches':pixel_matches,'reference_positive_scenes':sum(r['severe_reference_pixels']>0 for r in reference),
        'severe_reference_pixels':sum(r['severe_reference_pixels'] for r in reference),'reference_outside_both_dates_nonzero':sum(r['reference_outside_both_dates_nonzero'] for r in domain_notes),
        'domain_per_scene':domain_notes,'geographic_independence':'UNVERIFIED','human_review':'UNAVAILABLE'}
    atomic_json(root/'data_quality_summary.json',data)
    a=main[NATIVE256,'random_exact'];b=main[NATIVE128,'random_exact'];ratio=a['event_macro_incremental_covered_pixels_per_scene']/b['event_macro_incremental_covered_pixels_per_scene'] if b['event_macro_incremental_covered_pixels_per_scene'] else None
    evidence={'new36_only_primary256':a,'new36_only128_comparison':b,'coverage_ratio_256_over128':ratio,
        'primary_screen_passed':decision['primary_screen_passed'],'evidence_sufficient':decision['evidence_sufficient'],
        'old18_pooled':False,'old18_rescue_enabled':False,'native128_cost':1,'native256_cost':4,'primary_budget':8,
        'new_method_selected':False,'physical_cost_or_accuracy_claim':False,'data_quality_sha256':sha((root/'data_quality_summary.json').read_bytes())}
    atomic_json(root/'evidence_summary.json',evidence)
    next_scope={'status':'PROSPECTIVE_SCOPE_ONLY_NOT_EXECUTED','basis_decision_sha256':sha((root/'next_step_decision.json').read_bytes()),
        'current_replication_passed':decision['primary_screen_passed'],'no_more_same_source_checkpoint_variants_after_failure':not decision['primary_screen_passed'],
        'recommended_scope':'Measurement controls and data/reference admission before any method claim.' if decision['primary_screen_passed'] else 'Separate assessment of fixed detector sensitivity and published reference/data suitability before any planner or further source-cohort variants.',
        'mechanical_next_actions':['Use existing caches to describe class/false-pixel correspondence and source/reference provenance; preserve every failed screen.','Prepare a separate fixed detector or separately sourced reference study only after stating its questions, data limits and prospective decision rules.'],
        'enabled_training':False,'enabled_new_data_acquisition':False,'enabled_new_inference':False,'enabled_planner_tuning':False,'old_reserved_sets_enabled':False,
        'human_annotations_available':False,'reason_no_new_method_yet':'These54 remain same-source/event development material; new36 primary decision is not an independent efficacy or novelty test.'}
    atomic_json(root/'next_scope.json',next_scope)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    folder=root/'figures';folder.mkdir(exist_ok=True);colors={NATIVE128:'#b9855c',NATIVE256:'#287d87'}
    fig,axes=plt.subplots(1,3,figsize=(13.3,4.1),layout='constrained')
    vals=[main[m,'random_exact']['event_macro_incremental_covered_pixels_per_scene'] for m in MODES];bars=axes[0].bar(['Native128','Native256'],vals,color=[colors[m] for m in MODES]);maxv=max(vals+[1])
    for bar,val in zip(bars,vals):axes[0].text(bar.get_x()+bar.get_width()/2,val+maxv*.03,f'{val:.2f}',ha='center')
    axes[0].set_ylim(0,maxv*1.3);axes[0].set_ylabel('Expected NEW reference pixels / scene');axes[0].set_title('NEW36 only | budget8 | exact random')
    x=np.arange(3)
    for i,m in enumerate(MODES):axes[1].bar(x+(i-.5)*.32,[event[e,m,'random_exact']['incremental_covered_reference_pixels']/12 for e in events],width=.32,color=colors[m],label=['128 cells','256 blocks'][i])
    axes[1].set_xticks(x,['Turkey','Pakistan','Texas']);axes[1].set_ylabel('Expected NEW reference pixels / scene');axes[1].set_title('All12 pairs/event retained');axes[1].legend(fontsize=8)
    count=[sum(r['domain']=='all_published_pixels' and r['input_mode']==NATIVE256 and r['event']==e and r['incremental_covered_reference_pixels']>0 for r in opportunities) for e in events]
    bars=axes[2].bar(['Turkey','Pakistan','Texas'],count,color=colors[NATIVE256]);axes[2].axhline(2,color='#a78566',linestyle='--',linewidth=1,label='2/event in at least2 events')
    for bar,val in zip(bars,count):axes[2].text(bar.get_x()+bar.get_width()/2,val+.15,str(val),ha='center')
    axes[2].set_ylim(0,13);axes[2].set_ylabel('Detectable NEW scenes /12');axes[2].set_title('Native256 full16-area opportunity');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Native footprint replication | fixed36 new pairs | published semantic reference proxy',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(folder/('native36_replication.'+ext),dpi=175)
    plt.close(fig)
    for e in events:
        ids=[sid for sid in protocol['selected_ids'] if scenes[sid][0]==e]
        for part in range(2):
            selected=ids[part*6:(part+1)*6];fig,axes=plt.subplots(6,5,figsize=(15,17),layout='constrained')
            for row,sid in enumerate(selected):
                _,target,free,_,masks=scenes[sid]
                with Image.open(lookup[sid,'post']['path']) as im:post=np.asarray(im.convert('RGB'))
                mosaics=[]
                for m in MODES:
                    canvas=free.copy()
                    for u in units(m):
                        xx,yy,x1,y1=u['bounds'];canvas[yy:y1,xx:x1]|=masks[m][u['unit_id']]
                    mosaics.append(canvas)
                for col,mask in enumerate([None,target,free,*mosaics]):
                    canvas=post.copy()
                    if mask is not None:canvas[mask]=(.3*canvas[mask]+.7*np.array([40,190,210] if col==1 else [242,122,54])).astype(np.uint8)
                    axes[row,col].imshow(canvas);axes[row,col].axis('off')
                    if col==0:axes[row,col].set_title(sid,fontsize=8)
                    elif row==0:axes[row,col].set_title(['','Published severe ref','FREE coarse baseline','Free +native128 union','Free +native256 union'][col],fontsize=8)
            fig.suptitle(e+f' | new pairs {part*6+1}..{part*6+6} | full16-area, not budget8 actions',fontsize=11)
            fig.savefig(folder/(e+f'_part{part+1}.png'),dpi=110);plt.close(fig)
    atomic_json(folder/'figure_receipt.json',{'summary_sha256':sha((root/'summary.json').read_bytes()),'data_quality_sha256':sha((root/'data_quality_summary.json').read_bytes()),'generator_sha256':sha(Path(__file__).read_bytes()),
        'all36_pairs_all_inputs_in_overlays':True,'overlays_are_full16_area_not_budget8':True,'human_annotation':False})
    fmt=lambda v:'null' if v is None else f'{100*v:.2f}%'
    table=[]
    for m in MODES:
        random=main[m,'random_exact'];rank=main[m,'raw_change_rank'];table.append(f"| {'128格，成本1' if m==NATIVE128 else '256块，成本4'} | {random['event_macro_incremental_covered_pixels_per_scene']:.3f} | {fmt(random['reference_precision'])} | {random['event_macro_false_pixels_per_scene']:.3f} | {rank['event_macro_incremental_covered_pixels_per_scene']:.3f} |")
    event_table=[]
    for e in events:
        old=event[e,NATIVE128,'random_exact'];new=event[e,NATIVE256,'random_exact'];event_table.append(f"| {e} | {new['reference_pixels']:,} | {old['incremental_covered_reference_pixels']:.2f} | {new['incremental_covered_reference_pixels']:.2f} | {fmt(new['reference_precision'])} | {sum(r['event']==e and r['domain']=='all_published_pixels' and r['input_mode']==NATIVE256 and r['incremental_covered_reference_pixels']>0 for r in opportunities)}/12 |")
    checks=decision['domain_screens'][0]['checks'];labels={'at_least_two_reference_positive_events':'至少两个含严重参考的事件','at_least_six_detectable_new_scenes':'至少六个有256新增重叠的新场景',
        'at_least_two_detectable_scenes_in_each_of_two_events':'至少两个事件各有至少两个可发现新场景','budget8_random_macro_incremental_coverage_increases':'预算8精确随机宏平均新增覆盖提升',
        'budget8_pooled_reference_precision_does_not_fall':'预算8合并参考精度不降低','every_reference_positive_event_nondecreasing':'每个参考阳性事件覆盖均不降低','strict_event_gains_in_at_least_two_reference_positive_events':'至少两个参考阳性事件严格提升'}
    gate='\n'.join(f"- {labels[k]}：{'通过' if v else '未通过'}。" for k,v in checks.items())
    conclusion='新增36对通过预先登记的测量复验门槛；仍未形成新的搜索算法或独立效果结论。' if decision['primary_screen_passed'] else '新增36对未能建立预先登记的测量复验结论；按协议停止在这批来源／检查点上追加输入或选点变体。'
    text=f"""# 新增36对：原图128／256同面积测量复验

日期：2026-10-27。状态：108个原始成员提取、758次固定前向、全部预算对照及独立审计完成。**{conclusion}**

## 新36对的主结果

仅使用新增36对、每事件12对。旧18对不参与本轮主统计或继续条件。面积预算8购买8个128格或2个256块；免费64 BOX粗图基线对所有对照共享。主条件是原图256，主对照为固定128，质量均为原图。

| 观察单元 | 精确随机新增参考覆盖／场景 | 合并参考精度 | 参考相对误报／场景 | 固定变化排序新增覆盖／场景 |
|---|---:|---:|---:|---:|
"""+'\n'.join(table)+"""

先在事件内平均全部12个场景，再对三个事件等权平均。随机期望按不相交付费区域的包含概率计算；精度为期望覆盖总数／期望预测总数，不是轨迹精度均值。排序仅作次要对照，使用原粗图变化和原10个同分顺序，不替代主条件。没有新控制器、反馈规划或参数拟合。

| 事件，每个12对；覆盖为事件合计像素 | 发布严重参考像素 | 128随机新增期望 | 256随机新增期望 | 256参考精度 | 256可发现新场景 |
|---|---:|---:|---:|---:|---:|
"""+'\n'.join(event_table)+f"""

以上参考是EBD发布语义类别3／4，尚无本项目人工复核；无参考目标的事件召回为null，不算满分。参考相对误报不是人工确认的真实误报。全部预算4／8／16、全部事件、两评价域以及每场景结果都保留在 `summary.json`、`scene_scores.json`。

## 预设门槛逐项结果

{gate}

有效场景与事件数量是工程证据充足性条件，不是统计功效或显著性计算。主域是全部发布像素，非零域只作敏感性检查。没有删掉零目标、全黑、低检出或不利场景，没有改预算、阈值、排序或门槛；未将原18对合并救分。即使数值改善，也不能越过未满足的证据条件。

![新36复验]({(folder/'native36_replication.png').resolve()})

## 样本与数据检查

沿用原来源v2、种子和身份哈希顺序，每事件取第7–18名，与旧18身份无交集。72张RGB和36份发布灾后语义掩膜由324个精确ZIP范围提取，实际读取 **{extraction['received_bytes']:,}字节**，低于48MiB；压缩内容29,635,418字节。成员名、边界、CRC、SHA256、PNG尺寸／模式及整数类别均逐一复核，没有整包、其他影像或其他标签载荷访问。来源使用继续按既有本地研究范围与 `ATTRIBUTION.txt` 记录。

新RGB中{data['RGB_images_at_least25percent_zero']}张至少25%全通道零值，最大{100*data['maximum_zero_rgb_fraction']:.2f}%；完整像素对重复组{len(new_pair_duplicates)}组，与旧18完全相同的像素对{len(pixel_matches)}对。严重参考阳性场景{data['reference_positive_scenes']}/36，严重参考共{data['severe_reference_pixels']:,}像素，其中{data['reference_outside_both_dates_nonzero']:,}个在双时相非零域之外。全部保留；这些结构检查不认证实际配准、地理独立或语义标签正确性。

六张叠加图覆盖全部36对，青色发布参考、橙色模型输出，单独列出免费粗图基线；128／256展示的是购买完整16单位面积后的并集，不是预算8的实际动作。代理查看不能代替人工标注。

## 执行与独立审计

先通过330项测试，再冻结提取代码、范围白名单及48MiB上限；模型、评价器、门槛和审计实现也在新输出前留存 `implementation_seal.json`。推理前先复现两个旧开发阳性128裁块，均逐像素一致。随后36个免费粗图＋576个原图128格＋144个原图256块，共 **758次** 本机前向，无隐藏预热。模型／代码／归一化／随机种子沿用旧固定版本。预测不解码参考像素，评价器单独读取标签。

本机{inference['device'].upper()}推理循环约{inference['loop_seconds']:.2f}秒；不训练、不租GPU。独立审计重放324个网络范围、原PNG与CRC，重建758个张量及全部掩膜，按另一套面积几何、整数并集计数与组合包含概率重算 **2,160条排序记录、4,320域评分、12,600个付费动作、432项随机精确评分**；864个场景域策略均值、144个全覆盖评分一致。{audit['old_frozen_sources_unchanged']}份旧冻结源码保持一致，旧验证／确认集未评分。

## 接下来如何推进

{('先完成来源与参考独立准入和测量稳健性，再为任何新方法另立问题、强对照与协议。' if decision['primary_screen_passed'] else '先单独评估固定感知器与发布参考／数据的适用性，当前不继续本来源／检查点上的输入参数、区域选择或规划变体。')} 具体范围保存在 `next_scope.json`，仅为前瞻工作范围，未启用新下载、训练、推理或规划实验。原图18对的探索结果保留为历史记录；本轮不将它们当成新36结果的验证。

更早停止的合成模糊／复核与免费粗图＋128候选继续停止，未通过变更退化或预算恢复。原面积合同依然是信息面积代理，不是飞行航程、成像质量或真实采集延迟。累计54对都属于同来源／事件的开发测量材料，不代表地理／人工独立确认，也不是逐栋实例真值。新方法尚未选定，旧216候选与保留集状态不变。

复核入口：`extraction_protocol.json`、`extraction_report.json`、`rgb_manifest.json`、`reference_manifest.json`、`protocol.json`、`calls/`、`inference_report.json`、`full_opportunities.json`、`summary.json`、`next_step_decision.json`、`audit.json`及完成收据。完成目录和源码快照冻结；重放请使用新的审计输出目录。
"""
    (root/'README.md').write_text(text)
    print(json.dumps({'new36_primary_screen_passed':decision['primary_screen_passed'],'evidence_sufficient':decision['evidence_sufficient'],'primary128':b['event_macro_incremental_covered_pixels_per_scene'],'primary256':a['event_macro_incremental_covered_pixels_per_scene']}))


if __name__=='__main__':main()
