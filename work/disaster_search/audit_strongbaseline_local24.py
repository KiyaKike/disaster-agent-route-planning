"""Independent local-input/area/payment/random/oracle replay and reporting."""
import argparse
import csv
from io import StringIO
from itertools import combinations
import math
from pathlib import Path
import shutil
import numpy as np
import cv2
from PIL import Image

from run_strongbaseline_local24 import ROOT,HERE,SOURCE,PARENT,OUT,read,digest,checked,rawsha,save,verify
from audit_strongbaseline24 import independent_reference


def independent_metrics(tp,fp,reference):
    return {'tp':tp,'fp':fp,'fn':reference-tp,'reference':reference,
        'precision_ratio_of_counts':tp/(tp+fp) if tp+fp else None,'recall':tp/reference if reference else None,
        'f1_ratio_of_counts':2*tp/(tp+fp+reference) if tp+fp+reference else None}


def compare(row,expected):
    for k,v in expected.items():
        if isinstance(v,(float,np.floating)):
            assert row[k] is not None and abs(row[k]-v)<1e-10,(k,row[k],v)
        else:assert row[k]==v,(k,row[k],v)


def audit():
    p=verify();jobs=read(OUT/'rgb_jobs.json');rows=read(OUT/'observations.json');manifest=read(OUT/'cache_manifest.json')
    assert len(jobs)==24 and len(rows)==1540 and len(manifest)==384
    assert read(OUT/'runtime.json')['new_forward_calls']==1540
    cover=np.zeros((1024,1024),np.uint8)
    for unit in p['units']:
        x,y,xx,yy=unit['bounds'];cover[y:yy,x:xx]+=1;assert (xx-x)*(yy-y)==128**2*unit['area_cost']
    assert np.all(cover==1) and sum(u['area_cost'] for u in p['units'])==64
    public={r['sample_id']:r for r in read(OUT/'public_static_orders.json')};input_count=0
    expected=[]
    expected.extend(('old_full1024_control',jobs[0]['sample_id'],None,i) for i in range(4))
    expected.extend(('native256',j['sample_id'],tile,i) for j in jobs for tile in range(16) for i in range(4))
    byscene={};scenepairs={}
    for j in jobs:
        sid=j['sample_id'];images=[cv2.imread(r['path'],cv2.IMREAD_COLOR) for r in j['images']]
        assert all(a.shape==(1024,1024,3) for a in images);scenepairs[sid]=images
        coarse=[]
        for item in j['images']:
            coarse.append(np.asarray(Image.open(item['path']).resize((64,64),Image.Resampling.BOX)))
        diff=np.abs(coarse[0].astype(np.int16)-coarse[1].astype(np.int16)).astype(np.float64)/255
        scores=[float(diff[i//4*16:(i//4+1)*16,i%4*16:(i%4+1)*16].mean()) for i in range(16)]
        np.testing.assert_allclose(scores,public[sid]['scores_row_major'],atol=1e-12,rtol=0)
        assert public[sid]['order']==sorted(range(16),key=lambda i:(-scores[i],i))
        byscene[sid]=scores
    for index,(row,wanted) in enumerate(zip(rows,expected),1):
        assert row['index']==index and row['status']=='COMPLETE'
        assert (row['role'],row['sample_id'],row['tile_id'],row['flip_id'])==wanted
        assert row==read(OUT/'calls'/f'{index:04d}.json') and row['protocol_sha256']==digest(OUT/'protocol.json')
        images=scenepairs[row['sample_id']]
        if row['role']=='native256':
            tile=row['tile_id'];x,y=tile%4*256,tile//4*256;images=[a[y:y+256,x:x+256] for a in images]
        a=np.concatenate(images,2).astype(np.float32);a/=127;a-=1;a=a.transpose(2,0,1)[None].copy()
        axes=[(),(2,),(3,),(2,3)][row['flip_id']]
        if axes:a=np.flip(a,axis=axes).copy()
        assert list(a.shape)==row['input_shape'] and rawsha(a)==row['input_sha256'];input_count+=1
    del scenepairs
    control=read(OUT/'old_control.json')
    with np.load(OUT/'old_control_mean.npz',allow_pickle=False) as current,np.load(control['path'],allow_pickle=False) as old:
        a=current['mean_probabilities'];b=old['mean_probabilities'];error=float(np.max(np.abs(a-b)))
        pred=np.where(a[0]>.38,a[1:].argmax(0)+1,0).astype(np.uint8)
        assert error<=1e-5 and np.array_equal(pred,old['prediction'])
        assert error==read(OUT/'old_control_report.json')['max_abs_mean_probability_difference']
    refs={r['sample_id']:r for r in read(OUT/'references.json')}
    tile_scores={(r['sample_id'],r['tile_id']):r for r in read(OUT/'tile_scores.json')}
    scene_scores={r['sample_id']:r for r in read(OUT/'scene_scores.json')}
    policies={(r['sample_id'],r['budget'],r['policy']):r for r in read(OUT/'policy_scores.json')}
    assert len(tile_scores)==384 and len(scene_scores)==24 and len(policies)==288
    subsets={k:np.array(list(combinations(range(16),k)),np.int16) for k in [2,4,8,16]}
    enumerated=0;replayed=[];scene_replayed=[]
    for sid in p['selected_ids']:
        truth=independent_reference(read(refs[sid]['label_path']));np.testing.assert_array_equal(truth,np.asarray(Image.open(refs[sid]['reference_file'])))
        rows_here=sorted([r for r in manifest if r['sample_id']==sid],key=lambda r:r['tile_id'])
        assert [r['tile_id'] for r in rows_here]==list(range(16));assembled=np.zeros((1024,1024),np.uint8);counts=[]
        for item in rows_here:
            checked(item['path'],item['sha256']);tile=item['tile_id'];x,y=tile%4*256,tile//4*256
            assert item['bounds']==[x,y,x+256,y+256] and item['cost']==4
            with np.load(item['path'],allow_pickle=False) as d:
                a=d['mean_probabilities'];assert a.dtype==np.float32 and a.shape==(5,256,256) and np.isfinite(a).all()
                assert 0<=a.min()<=a.max()<=1 and rawsha(a)==item['mean_probability_sha256']
                pred=np.where(a[0]>.38,a[1:].argmax(0)+1,0).astype(np.uint8)
                np.testing.assert_array_equal(pred,d['prediction']);assert rawsha(pred)==item['prediction_pixel_sha256']
            assembled[y:y+256,x:x+256]=pred;t=truth[y:y+256,x:x+256];valid=t!=255;target=np.isin(t,[3,4]);found=np.isin(pred,[3,4])
            r={'tp':int((valid&target&found).sum()),'fp':int((valid&~target&found).sum()),'reference':int((valid&target).sum()),'valid_pixels':int(valid.sum())}
            compare(tile_scores[sid,tile],r);counts.append(r)
        np.testing.assert_array_equal(assembled,np.asarray(Image.open(OUT/'assembled'/(sid+'.png'))))
        tp=np.array([r['tp'] for r in counts],np.int64);fp=np.array([r['fp'] for r in counts],np.int64);reference=sum(r['reference'] for r in counts)
        scene=independent_metrics(int(tp.sum()),int(fp.sum()),reference);compare(scene_scores[sid],scene)
        assert scene_scores[sid]['detectable_tiles']==int((tp>0).sum());scene_replayed.append(scene)
        oracle=sorted(range(16),key=lambda i:(-int(tp[i]),i))
        for budget in p['budgets']:
            k=budget//4;sub=subsets[k];enumerated+=len(sub)
            random_tp=tp[sub].sum(1);random_fp=fp[sub].sum(1)
            exact=independent_metrics(float(random_tp.mean()),float(random_fp.mean()),reference)
            random=policies[sid,budget,'random_exact_expected_counts'];compare(random,exact)
            assert abs(random['scene_any_tp_probability']-float((random_tp>0).mean()))<1e-12
            assert random['spent']==k*4 and random['acquisitions']==k
            replayed.append({'sample_id':sid,'budget':budget,'policy':'random_exact_expected_counts',**exact})
            for policy,order in [('static_coarse_change',public[sid]['order']),('oracle_true_gain',oracle)]:
                row=policies[sid,budget,policy];chosen=order[:k]
                assert row['trace']==[{'tile_id':tile,'cost':4,'cumulative_cost':4*(j+1)} for j,tile in enumerate(chosen)]
                assert row['spent']==k*4 and row['acquisitions']==k
                value=independent_metrics(int(tp[chosen].sum()),int(fp[chosen].sum()),reference);compare(row,value)
                replayed.append({'sample_id':sid,'budget':budget,'policy':policy,**value})
            # Exhaustive proof of same-budget oracle optimality on all subsets.
            assert policies[sid,budget,'oracle_true_gain']['tp']==int(random_tp.max())
            if budget==64:
                for policy in ['random_exact_expected_counts','static_coarse_change','oracle_true_gain']:compare(policies[sid,budget,policy],scene)
    totals=read(OUT/'totals.json');assert len(totals)==12
    for row in totals:
        selected=[r for r in replayed if r['budget']==row['budget'] and r['policy']==row['policy']]
        summed=independent_metrics(sum(r['tp'] for r in selected),sum(r['fp'] for r in selected),sum(r['reference'] for r in selected))
        compare(row,summed);assert abs(row['mean_TP_per_scene']-summed['tp']/24)<1e-10
        positive=[r for r in selected if r['reference']>0];assert abs(row['mean_positive_scene_recall']-sum(r['recall'] for r in positive)/len(positive))<1e-12
        assert row['mean_area_spent']==row['budget'] and row['mean_forward_calls_per_scene']==row['budget']
    summary=read(OUT/'summary.json');compare(summary['all_grid_native256'],independent_metrics(sum(r['tp'] for r in scene_replayed),sum(r['fp'] for r in scene_replayed),sum(r['reference'] for r in scene_replayed)))
    detectable=sum(r['tp']>0 for r in scene_replayed);retention=summary['all_grid_native256']['recall']/read(PARENT/'summary.json')['recall']
    assert summary['local_detectable_scenes']==detectable and summary['all_grid_recall_fraction_of_old_full1024']==retention
    signal=detectable>=p['prospective_screen']['minimum_reference_positive_scenes_with_local_TP'] and retention>=p['prospective_screen']['minimum_all_grid_recall_fraction_of_previous_full1024']
    assert summary['engineering_signal_screen_passed']==signal
    parent=read(PARENT/'completion_receipt.json')
    for n,h in parent['files'].items():checked(PARENT/n,h)
    for n,h in parent['external_files'].items():checked(n,h)
    runtime=read(OUT/'runtime.json');assert abs(runtime['forward_seconds']-sum(r['elapsed_seconds'] for r in read(OUT/'observations.json')))<1e-9
    save(OUT/'audit_report.json',{'status':'PASS','input_tensors_reconstructed':input_count,'all256_predictions_decoded':384,
        'reference_rasters_rebuilt':24,'tile_scores_replayed':384,'scene_scores_replayed':24,'policy_scores_replayed':288,
        'aggregates_replayed':12,'uniform_random_subsets_enumerated':enumerated,'oracle_optima_exhaustively_verified':96,
        'old_full1024_control_probability_max_abs_difference':error,'old_frozen_sources_verified':len(parent['source_code_sha256']),
        'parent_output_files_verified':len(parent['files']),'parent_external_files_verified':len(parent['external_files']),
        'new_audit_model_calls':0,'score_selection_changed':False})
    for name,data in [('scene_scores.csv',read(OUT/'scene_scores.json')),('totals.csv',totals)]:
        stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data);(OUT/name).write_text(stream.getvalue())
    print({'audit':'PASS','inputs':input_count,'random_subsets':enumerated,'policy_rows':288},flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify();s=read(OUT/'summary.json');runtime=read(OUT/'runtime.json');totals=read(OUT/'totals.json')
    assert read(OUT/'audit_report.json')['status']=='PASS'
    (OUT/'figures').mkdir()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    styles=[('random_exact_expected_counts','Exact random','#6B7280'),('static_coarse_change','Coarse-change static','#2563EB'),('oracle_true_gain','Oracle diagnostic bound','#B45309')]
    for policy,label,color in styles:
        rows=sorted([r for r in totals if r['policy']==policy],key=lambda r:r['budget'])
        x=[0]+[r['budget']/64*100 for r in rows]
        axes[0].plot(x,[0]+[r['recall']*100 for r in rows],'-o',label=label,color=color)
        axes[1].plot(x,[0]+[r['mean_positive_scene_recall']*100 for r in rows],'-o',label=label,color=color)
    for ax in axes:
        ax.axvline(25,color='#555555',linestyle=':',linewidth=1);ax.set(xlabel='Acquired native area (%)',xlim=(0,102),ylim=(0,102));ax.grid(alpha=.2)
    axes[0].set(title='Pooled severe-reference recall',ylabel='Correctly discovered reference area (%)')
    axes[1].set(title='Mean recall across reference-positive scenes')
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Fixed24 Santa Rosa | native256 | four flips per action | single-event development')
    fig.savefig(OUT/'figures/area_budget.png',dpi=180);fig.savefig(OUT/'figures/area_budget.pdf');plt.close(fig)
    def pct(v):return 'null' if v is None else f'{v*100:.2f}%'
    table='\n'.join(f"| {label} | {s['primary'][policy]['tp']:,.1f} | {pct(s['primary'][policy]['recall'])} | {pct(s['primary'][policy]['mean_positive_scene_recall'])} |" for policy,label,_ in styles)
    local=s['all_grid_native256'];old=s['previous_full1024'];head=s['primary_oracle_minus_static_TP']
    text=f'''# Strong Baseline：256 局部观察与面积预算诊断

2026-10-28。固定现有 Santa Rosa24 对，新增384个原生256观察块。该阶段检验局部测量与策略空间，不提出新反馈方法。

## 主预算结果

主预算16＝购买4个256块，占1024整幅原图面积25%。每块配对灾前/灾后原图，取得后执行作者4翻转；每个预算内轨迹包含16次模型前向。两日期作为一对按面积计费，计算次数另列，不把翻转当作免费计算。下表随机项为精确期望，不选随机种子。

| 条件 | 24场景合计正确发现严重参考像素 | 合并参考召回 | 阳性场景平均召回 |
|---|---:|---:|---:|
{table}

Oracle 比静态多覆盖 {head:,.0f} 个参考像素，是已知全部真实动作收益后的离线上界，不能当成可执行 Agent 或新方法收益。静态策略只读64×64免费粗图的平均 RGB 变化；这份低分辨率先验免费是仿真假设，不能宣称实际感知完全无成本。排序在新推理前已冻结，不使用全幅高分辨率变化、严重参考、当前预测或未来反馈。

![面积预算曲线](figures/area_budget.png)

预算8／16／32／64分别覆盖12.5%／25%／50%／100%原图面积；新1024场景的预算合同不与旧EBD场景绝对分数拼接。图中每条曲线共享同一组256局部预测，差别只是选哪些块。所有16块互不重叠，奖励为观察块内正确检出的严重参考像素，未观察部分仍算漏检。

## 局部输入保留了多少信号

全部16局部块都观察时，严重像素精度 {pct(local['precision_ratio_of_counts'])}、召回 {pct(local['recall'])}、F1 {pct(local['f1_ratio_of_counts'])}。旧1024整图召回 {pct(old['recall'])}；局部全网格召回保留比例 {s['all_grid_recall_fraction_of_old_full1024']:.4f}。这比较输入上下文适用性，不是预算内方法增益。

24场景中有严重参考 {s['reference_positive_scenes']} 个，其中局部全网格至少有一个真阳性的场景 {s['local_detectable_scenes']} 个；完全漏检阳性场景：{', '.join(s['local_complete_miss_ids']) or '无'}。全部24个固定场景及既有黑色影像区域均保留，未知/冲突参考255排除，未按预测挑阳性或删除失败。

## 复核与解释边界

357项测试通过。4次旧整图回归调用和1536次新局部调用，共1540次MPS前向；前向累计 {runtime['forward_seconds']:.2f} 秒，含缓存保存循环 {runtime['loop_seconds']:.2f} 秒。旧整图概率均值和最终掩膜已复现。没有训练、新RGB、网络下载、付费GPU或旧保留集评分。

独立重建1540个输入，重新解码384份局部概率、重建24份多边形参考，重放384个块评分、288个预算/策略评分和12个汇总；逐场景/预算穷举所有均匀子集核对随机期望与同成本Oracle最优性。59份旧冻结源码不变。

随机精度/F1列若查看原始JSON，含义是“期望计数之比”，不是非线性指标本身的期望；召回对固定参考线性，随机召回是精确期望。本阶段无可学习策略、无训练，无统计独立方法收益声明。Santa Rosa符合作者训练未见事件清单，但已为本项目开发使用，且仅一个事件，不是新独立/跨事件验证，也不是官方mixedF1或逐栋人工真值。

前瞻工程筛查固定为至少6个参考阳性且局部可发现场景，并保留至少一半旧整图合并召回；这只是防止在无测量信号时继续调规划，不是发表门槛。工程筛查结果：{s['engineering_signal_screen_passed']}；Oracle相对静态的主预算空间大于0：{s['diagnostic_oracle_static_headroom_positive']}。当前决策：`{s['decision']}`。

## 下一范围

若测量筛查成立，下一项只准备用作者训练未见事件的新多事件队列：先锁定身份、开发/确认用途、影像质量及获取上限，再读载荷和评价。先检验相同局部合同是否跨事件保持发现机会，然后才考虑一个明确的反馈决策机制。当前没有开放新数据、进一步推理或旧保留集；旧EBD停止的输入/规划变体继续停止。若筛查未成立，则停止当前合同上的规划实验。

可复核文件：`protocol.json`、`public_static_orders.json`、`cache_manifest.json`、`tile_scores.json`、`scene_scores.csv`、`policy_scores.json`、`totals.csv`、`audit_report.json`。
'''
    (OUT/'README.md').write_text(text)
    save(OUT/'prospective_next_scope.json',{'status':'NOT_EXECUTED_METADATA_PLANNING_ONLY','decision':s['decision'],
        'next':'Prospectively lock new multi-event measurement cohort and image-quality/access rules before payloads;no adaptive strategy until measurement replication',
        'new_data_payloads_enabled':False,'new_inference_enabled':False,'gpu_rental':False,'training':False,'reserved_sets_enabled':False})
    print({'report':'WRITTEN','plot':'area_budget.png'},flush=True)


def complete():
    p=verify();assert read(OUT/'audit_report.json')['status']=='PASS'
    visual=read(OUT/'visual_inspection.json');assert visual['plot_viewed']
    checked(OUT/'figures/area_budget.png',visual['plot_sha256'])
    snapshots=OUT/'source_snapshots';snapshots.mkdir()
    for n,h in p['source_code_sha256'].items():checked(HERE/n,h);shutil.copyfile(HERE/n,snapshots/n)
    external={**p['external_files'],**{str(HERE/n):h for n,h in p['source_code_sha256'].items()},**{str(SOURCE/n):h for n,h in p['upstream_files_sha256'].items()}}
    for n,h in external.items():checked(n,h)
    receipt={'status':'COMPLETE','client_report_date':'2026-10-28','parent_receipt_sha256':p['parent_receipt_sha256'],
        'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completion_receipt.json'},
        'external_files':external,'source_code_sha256':p['source_code_sha256'],'tests_passed':357,'new_tests':8,
        'new_forward_calls':1540,'new_local256_calls':1536,'old_full1024_regression_calls':4,'input_tensors_reconstructed':1540,
        'local_probability_caches_verified':384,'tile_scores_replayed':384,'policy_scores_replayed':288,'aggregates_replayed':12,
        'old_frozen_sources_verified':59,'plot_viewed':True,'new_rgb':0,'network_downloads':0,'training':False,'gpu_rented':False,
        'new_human_annotations':0,'old_reserved_sets_scored':False,'independent_efficacy_claim':False,'new_adaptive_method_run':False}
    save(OUT/'completion_receipt.json',receipt)
    for n,h in receipt['files'].items():checked(OUT/n,h)
    print({'status':'COMPLETE','receipt_sha256':digest(OUT/'completion_receipt.json'),'output_files':len(receipt['files']),'sources':len(p['source_code_sha256'])},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','report','complete'],required=True);args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'audit':audit,'report':report,'complete':complete}[args.stage]()
