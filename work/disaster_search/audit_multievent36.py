"""Independently replay source admission, inputs, per-event counts and budgets."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import shutil
import numpy as np
import cv2
from PIL import Image
from prepare_multievent36 import TACO,AUTHOR,OLD_COHORT
from run_multievent36 import OUT,PARENT,ROOT,HERE,SOURCE,read,digest,checked,rawsha,save,verify
from multievent36_core import select,EVENTS
from audit_strongbaseline24 import independent_reference
from audit_strongbaseline_local24 import independent_metrics,compare


def audit():
    p=verify();selection=read(OUT/'selection_protocol.json');selected=read(OUT/'selected.json')
    splits={s:set((AUTHOR/f'{s}_samples_event_split.txt').read_text().splitlines()) for s in ['train','val','test']}
    assert selected==select(read(TACO/'identity_rows.json'),splits,set(selection['blocked_ids']),set(selection['blocked_events']))
    assert len(selected)==36 and Counter(r['event'] for r in selected)=={e:12 for e in EVENTS}
    old_reserved={r['sample_id'] for r in read(OLD_COHORT)};assert not {r['sample_id'] for r in selected}&old_reserved
    queue=read(OUT/'RGB_queue.json');rgb=read(OUT/'RGB_manifest.json');ledger=read(OUT/'network_ledger.json')
    assert len(queue)==len(rgb)==72 and len({(r['sample_id'],r['mode']) for r in rgb})==72
    assert sum(r['bytes'] for r in queue)==106554040
    for row in ledger:
        assert row['status']=='COMPLETE' and row['received']==row['bytes']
        if row['role']=='new_RGB':assert any(q['offset']<=row['offset']<row['offset']+row['bytes']<=q['offset']+q['bytes'] for q in queue)
        else:assert row['role'] in ['source_metadata','transient_damage']
    assert sum(r['bytes'] for r in ledger)<=384*1024*1024
    assert sum(r['bytes'] for r in ledger if r['role']=='new_RGB')==106554040
    for image in rgb:
        checked(image['path'],image['sha256'])
        with Image.open(image['path']) as im:im.verify()
        with Image.open(image['path']) as im:
            assert im.mode=='RGB' and im.size==(1024,1024);a=np.asarray(im)
        assert rawsha(a)==image['pixel_sha256_RGB'] and float(np.all(a==0,axis=2).mean())==image['zero_RGB_fraction']
    uses=read(OUT/'label_read_receipts.json');private=[(r['offset'],r['offset']+r['bytes']) for r in uses if r['transient_damage']]
    for use in uses:
        if use['transient_damage']:assert use['cache_file'] is None and not use['persisted']
    for file in (OUT/'metadata_ranges').glob('*.bin'):
        offset,count=map(int,file.stem.split('_'));assert not any(max(offset,a)<min(offset+count,b) for a,b in private)
    jobs={r['sample_id']:r for r in read(OUT/'rgb_jobs.json')};control=read(OUT/'old_control.json');rows=read(OUT/'observations.json');manifest=read(OUT/'cache_manifest.json')
    assert len(rows)==2308 and len(manifest)==576
    expected=[('old_control',control['job']['sample_id'],control['cache']['tile_id'],i) for i in range(4)]
    expected += [('production',sid,tile,i) for sid in p['selected_ids'] for tile in range(16) for i in range(4)]
    current=None;images=None;coarse_orders={r['sample_id']:r for r in read(OUT/'public_static_orders.json')}
    for index,(row,wanted) in enumerate(zip(rows,expected),1):
        assert row['index']==index and row['status']=='COMPLETE' and (row['role'],row['sample_id'],row['tile_id'],row['flip_id'])==wanted
        assert row==read(OUT/'calls'/f'{index:04d}.json') and row['protocol_sha256']==digest(OUT/'inference_protocol.json')
        if current!=row['sample_id']:
            current=row['sample_id'];job=control['job'] if row['role']=='old_control' else jobs[current]
            images=[cv2.imread(item['path'],cv2.IMREAD_COLOR) for item in job['images']]
        tile=row['tile_id'];x,y=tile%4*256,tile//4*256;a=np.concatenate([im[y:y+256,x:x+256] for im in images],2).astype(np.float32);a/=127;a-=1;a=a.transpose(2,0,1)[None].copy()
        axes=[(),(2,),(3,),(2,3)][row['flip_id']]
        if axes:a=np.flip(a,axis=axes).copy()
        assert list(a.shape)==row['input_shape'] and rawsha(a)==row['input_sha256']
    with np.load(OUT/'old_control_mean.npz',allow_pickle=False) as now,np.load(control['cache']['path'],allow_pickle=False) as old:
        a=now['mean_probabilities'];error=float(np.max(np.abs(a-old['mean_probabilities'])))
        assert error<=1e-5 and np.array_equal(np.where(a[0]>.38,a[1:].argmax(0)+1,0),old['prediction'])
    refs={r['sample_id']:r for r in read(OUT/'reference_manifest.json')};scenes={r['sample_id']:r for r in read(OUT/'scene_scores.json')}
    tiles={(r['sample_id'],r['tile_id']):r for r in read(OUT/'tile_scores.json')};policies={(r['sample_id'],r['budget'],r['policy']):r for r in read(OUT/'policy_scores.json')}
    assert len(tiles)==576 and len(scenes)==36 and len(policies)==432
    subsets={k:np.array(list(combinations(range(16),k)),np.int16) for k in [2,4,8,16]};enumerated=0
    for sid in p['selected_ids']:
        job=jobs[sid];cp=[np.asarray(Image.open(item['path']).resize((64,64),Image.Resampling.BOX)) for item in job['images']]
        diff=np.abs(cp[0].astype(np.int16)-cp[1].astype(np.int16)).astype(np.float64)/255
        scores=[float(diff[i//4*16:(i//4+1)*16,i%4*16:(i%4+1)*16].mean()) for i in range(16)]
        order=sorted(range(16),key=lambda i:(-scores[i],i));assert coarse_orders[sid]['order']==order
        np.testing.assert_allclose(scores,coarse_orders[sid]['scores_row_major'],atol=1e-12,rtol=0)
        truth=independent_reference(read(refs[sid]['label_path']));np.testing.assert_array_equal(truth,np.asarray(Image.open(refs[sid]['reference_file'])))
        assembled=np.zeros((1024,1024),np.uint8);local=[]
        ms=sorted([r for r in manifest if r['sample_id']==sid],key=lambda r:r['tile_id']);assert [r['tile_id'] for r in ms]==list(range(16))
        for item in ms:
            checked(item['path'],item['sha256']);i=item['tile_id'];x,y=i%4*256,i//4*256
            with np.load(item['path'],allow_pickle=False) as cache:
                a=cache['mean_probabilities'];assert a.dtype==np.float32 and a.shape==(5,256,256) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
                assert rawsha(a)==item['mean_probability_sha256'];pred=np.where(a[0]>.38,a[1:].argmax(0)+1,0).astype(np.uint8)
                np.testing.assert_array_equal(pred,cache['prediction']);assert rawsha(pred)==item['prediction_sha256']
            t=truth[y:y+256,x:x+256];v=t!=255;target=np.isin(t,[3,4]);found=np.isin(pred,[3,4]);assembled[y:y+256,x:x+256]=pred
            c={'tp':int((v&target&found).sum()),'fp':int((v&~target&found).sum()),'reference':int((v&target).sum()),'valid_pixels':int(v.sum())}
            compare(tiles[sid,i],c);local.append(c)
        np.testing.assert_array_equal(assembled,np.asarray(Image.open(OUT/'assembled'/(sid+'.png'))))
        tp=np.array([c['tp'] for c in local]);fp=np.array([c['fp'] for c in local]);reference=sum(c['reference'] for c in local)
        compare(scenes[sid],independent_metrics(int(tp.sum()),int(fp.sum()),reference));assert scenes[sid]['detectable_tiles']==int((tp>0).sum())
        oracle=sorted(range(16),key=lambda i:(-int(tp[i]),i))
        for budget in [8,16,32,64]:
            k=budget//4;sub=subsets[k];enumerated+=len(sub);summed=tp[sub].sum(1)
            random=policies[sid,budget,'random_exact_expected_counts'];compare(random,independent_metrics(float(summed.mean()),float(fp[sub].sum(1).mean()),reference))
            assert abs(random['scene_any_tp_probability']-float((summed>0).mean()))<1e-12
            for name,ranks in [('static_coarse_change',order),('oracle_true_gain',oracle)]:
                row=policies[sid,budget,name];chosen=ranks[:k];compare(row,independent_metrics(int(tp[chosen].sum()),int(fp[chosen].sum()),reference))
                assert row['trace']==[{'tile_id':i,'cost':4,'cumulative_cost':4*(j+1)} for j,i in enumerate(chosen)] and row['spent']==budget
            assert policies[sid,budget,'oracle_true_gain']['tp']==int(summed.max())
    totals=read(OUT/'event_totals.json');assert len(totals)==36
    for row in totals:
        group=[r for r in policies.values() if r['event']==row['event'] and r['budget']==row['budget'] and r['policy']==row['policy']];assert len(group)==12
        compare(row,independent_metrics(sum(r['tp'] for r in group),sum(r['fp'] for r in group),sum(r['reference'] for r in group)))
        assert row['mean_TP_per_scene']==row['tp']/12
    for row in read(OUT/'macro_totals.json'):
        group=[r for r in totals if r['budget']==row['budget'] and r['policy']==row['policy']];assert len(group)==3
        assert row['equal_event_mean_TP_per_scene']==sum(r['mean_TP_per_scene'] for r in group)/3
        expected=sum(r['recall'] for r in group)/3 if all(r['recall'] is not None for r in group) else None
        compare(row,{'macro_recall':expected})
    summary=read(OUT/'summary.json')
    for event in summary['events']:
        group=[r for r in scenes.values() if r['event']==event['event']];assert len(group)==12
        compare(event['all_grid'],independent_metrics(sum(r['tp'] for r in group),sum(r['fp'] for r in group),sum(r['reference'] for r in group)))
        assert event['reference_positive_scenes']==sum(r['reference']>0 for r in group) and event['local_detectable_scenes']==sum(r['tp']>0 for r in group)
    parent=read(PARENT/'completion_receipt.json')
    for n,h in parent['files'].items():checked(PARENT/n,h)
    for n,h in parent['external_files'].items():checked(n,h)
    save(OUT/'audit_report.json',{'status':'PASS','new_RGB_verified':72,'inputs_reconstructed':2308,'local_caches_decoded':576,'references_rebuilt':36,'tile_scores_replayed':576,
        'policy_scores_replayed':432,'event_aggregates_replayed':36,'macro_aggregates_replayed':12,'random_subsets_enumerated':enumerated,
        'old_control_probability_max_abs_difference':error,'old_frozen_sources_verified':63,'original_UIDs_recovered':False,'new_audit_model_calls':0,
        'old_reserved_outcomes_returned':False,'unselected_shared_damage_bytes_persisted':False})
    print({'audit':'PASS','random_subsets':enumerated,'events':3,'scenes':36},flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(OUT/'summary.json');totals=read(OUT/'event_totals.json');runtime=read(OUT/'runtime.json');quality=read(OUT/'RGB_quality_report.json');admission=read(OUT/'data_admission.json')
    assert read(OUT/'audit_report.json')['status']=='PASS'
    (OUT/'figures').mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    colors={'random_exact_expected_counts':'#6B7280','static_coarse_change':'#2563EB','oracle_true_gain':'#B45309'}
    labels={'random_exact_expected_counts':'Exact random','static_coarse_change':'Coarse-change static','oracle_true_gain':'Oracle bound'}
    for ax,event in zip(axes,EVENTS):
        item=next(r for r in s['events'] if r['event']==event)
        for policy in colors:
            group=sorted([r for r in totals if r['event']==event and r['policy']==policy],key=lambda r:r['budget'])
            ax.plot([r['budget']/64*100 for r in group],[r['recall']*100 if r['recall'] is not None else np.nan for r in group],'-o',color=colors[policy],label=labels[policy])
        ax.set(title=f"{event}\npositive {item['reference_positive_scenes']}/12;detectable {item['local_detectable_scenes']}/12",xlabel='Acquired native area (%)',ylim=(-1,101));ax.axvline(25,color='#555',linestyle=':',linewidth=1);ax.grid(alpha=.2)
    axes[0].set_ylabel('Severe-reference recall (%)');axes[0].legend(fontsize=8)
    fig.suptitle('New36 event-held-out measurement | fixed native256 | no adaptive method')
    fig.savefig(OUT/'figures/multievent_budget.png',dpi=180);fig.savefig(OUT/'figures/multievent_budget.pdf');plt.close(fig)
    def pct(v):return 'NA' if v is None else f'{100*v:.2f}%'
    table=[]
    for e in s['events']:
        primary={r['policy']:r for r in s['primary_per_event'] if r['event']==e['event']}
        table.append(f"| {e['event']} | {e['reference_positive_scenes']}/12 | {e['local_detectable_scenes']}/12 | {pct(e['all_grid']['recall'])} | {pct(primary['random_exact_expected_counts']['recall'])} | {pct(primary['static_coarse_change']['recall'])} | {pct(primary['oracle_true_gain']['recall'])} |")
    text='\n'.join(table);received=admission['network_bytes_received']
    (OUT/'README.md').write_text(f'''# 新三事件 36 对：局部预算测量复验

2026-10-28。已按预先固定身份哈希各选12对，覆盖尼泊尔洪水、巽他海啸、塔斯卡卢萨龙卷风。三个事件不在作者训练/验证清单，不在旧216保留队列事件或本地已有命名影像/结果事件中。本批永久作为开发测量复验，不升级为最终确认集。

## 主预算16：只观察25%原图面积

| 事件 | 有严重参考 | 局部可发现 | 全网格局部召回 | 随机精确期望 | 静态粗图排序 | Oracle诊断上界 |
|---|---:|---:|---:|---:|---:|---:|
{text}

每个场景16个原生256块，每块原图配对面积成本4，主预算16选4块；次预算8/32/64。排序仅用免费64BOX粗图变化，免费粗图是仿真假设。每个取得的块按作者4翻转推理，主预算每条路径16次前向，计算与面积分别报告。模型阈值、BGR/127-1、解码和先前SantaRosa合同一致。

![三事件面积预算曲线](figures/multievent_budget.png)

Oracle知道所有真实局部收益，只是相同动作/成本的离线上界。均匀随机为全部子集的精确期望；精度/F1若查看JSON，是期望计数之比，不是非线性指标的期望。未观察的严重参考仍算漏检。没有运行新反馈策略，没有把SantaRosa并入救分。

## 来源、质量与准入

作者完整训练/验证/测试清单与TACO身份路径集合逐一相等，共11034对。TACO三个事件候选池分别619/148/343，按SHA256(seed|event|ID)取前12，不读损伤、建筑数、几何或预测挑样本。未采用原镜像较少的身份池；没有先筛出“有建筑/有损伤”场景。

36对72张原图，载荷106554040字节；含所需元数据/共享标签的新网络接收合计 {received:,} 字节，低于预设384MiB上限，RGB低于256MiB。新RGB都核对PNG完整性、尺寸、编码、SHA和像素哈希；与已有本地1024原图对及本批精确影像对重复检查通过。全零RGB至少25%的影像 {quality['flagged_images25pct']} 张，最大零值比例 {quality['max_zero_RGB_fraction']:.2%}；仅标记，所有36对保留，无删图、无像素过滤。精确零值不是经人工验证的云/无效掩膜。

损伤列可能与旧保留/未选样本共享物理Parquet列。可信提取器只返回固定36行；共享损伤字节瞬时读取，不缓存、不输出其他样本结果。派生多边形保持原坐标及旧半像素容差，未知及冲突参考忽略；没有恢复原始UID或创造人工标注。标签来源为发布参考，缺少原始灾前定位真值，不是官方mixedF1、实例AP或真实现场识别率。

## 运行与复核

363项测试通过，新增6项身份/准入隔离测试。2308次MPS前向：4次旧256控制、2304次新局部推理。前向累计 {runtime['forward_seconds']:.2f} 秒，含缓存运行循环 {runtime['loop_seconds']:.2f} 秒。没有训练、付费GPU或旧保留集评分。

独立重建2308个输入、576份局部概率、36份参考栅格，重放576块评分、432策略预算评分、36事件汇总及12宏汇总；逐场景和预算穷举全部子集验证均匀随机及Oracle最优。63份旧冻结源码不变。

## 判断与下一步

满足每事件至少6个局部可发现场景的事件数：{s['events_with_at_least6_local_detectable_scenes']}/3。前瞻筛查：`{s['screen']}`。这是工程测量密度筛查，不是发表门槛，也不表示其他事件没有损伤。结果不足时停止在当前跨灾害合同上增加规划变体；不能因这轮结果补挑阳性、并入SantaRosa或重选阈值。若充分，再审查一个反馈机制的研究差异并设计独立确认数据。

事件必须分别报告；参考严重像素为0时召回为NA。三个事件中任一无阳性时，主宏召回为null，不把它暗中记0或删去；主总体奖励同时保留等事件平均TP/场景。样本量小、自然损伤分布未做平衡；没有独立地理/全训练池像素重复证明，作者清单也不能独立还原权重训练历史。像素面积是观察成本与奖励代理，不是统一地面平方米、建筑数量或实际飞行航线成本。

可复核：`selection_protocol.json`、`measurement_contract_before_labels.json`、`inference_protocol.json`、`RGB_quality_report.json`、`data_admission.json`、`summary.json`、`event_totals.json`、`macro_totals.json`、`audit_report.json`。
''')
    save(OUT/'prospective_next_scope.json',{'status':'NOT_EXECUTED','screen':s['screen'],'next':'Review per-event measurement failure and damage prevalence before any policy variants' if s['events_with_at_least6_local_detectable_scenes']<3 else 'Review one feedback mechanism and fresh confirmation design before implementation',
        'new_inference_enabled':False,'new_data_payloads_enabled':False,'training':False,'gpu_rental':False,'old_reserved_sets_enabled':False})
    print({'report':'WRITTEN'},flush=True)


def complete():
    p=verify();assert read(OUT/'audit_report.json')['status']=='PASS';v=read(OUT/'visual_inspection.json');assert v['budget_plot_viewed'] and v['all_three_RGB_sheets_viewed']
    for n,h in v['files'].items():checked(OUT/n,h)
    snapshots=OUT/'source_snapshots';snapshots.mkdir()
    for n,h in p['source_code_sha256'].items():checked(HERE/n,h);shutil.copyfile(HERE/n,snapshots/n)
    selection=read(OUT/'selection_protocol.json');external={**selection['external_files'],**p['external_files'],**{str(HERE/n):h for n,h in p['source_code_sha256'].items()},**{str(SOURCE/n):h for n,h in p['upstream_files_sha256'].items()}}
    for use in read(OUT/'metadata_read_receipts.json')+read(OUT/'label_read_receipts.json'):
        if use['cache_file']:
            file=Path(use['cache_file']);external[str(file)]=digest(file);external[str(file.with_suffix('.json'))]=digest(file.with_suffix('.json'))
    external.update(read(OUT/'RGB_quality_report.json')['old_RGB_files_checked'])
    for n,h in external.items():checked(n,h)
    receipt={'status':'COMPLETE','client_report_date':'2026-10-28','source_code_sha256':p['source_code_sha256'],'external_files':external,
        'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completion_receipt.json'},
        'new_events':3,'new_scenes':36,'new_RGB':72,'new_forward_calls':2308,'new_tests':6,'tests_passed':363,'inputs_reconstructed':2308,
        'local_caches_verified':576,'policy_scores_replayed':432,'old_frozen_sources_verified':63,'original_UIDs_recovered':False,
        'training':False,'gpu_rented':False,'old_reserved_sets_scored':False,'new_human_annotations':0,'new_adaptive_method_run':False,'independent_efficacy_claim':False}
    save(OUT/'completion_receipt.json',receipt)
    for n,h in receipt['files'].items():checked(OUT/n,h)
    print({'status':'COMPLETE','receipt_sha256':digest(OUT/'completion_receipt.json'),'output_files':len(receipt['files']),'source_snapshots':len(p['source_code_sha256'])},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','report','complete'],required=True);args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'audit':audit,'report':report,'complete':complete}[args.stage]()
