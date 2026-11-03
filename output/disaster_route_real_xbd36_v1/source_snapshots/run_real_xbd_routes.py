"""Real xBD imagery/model-cache route replay;reference labels evaluator-only."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from run_route_loop import ROOT,HERE,read,save,digest,checked
from real_xbd_route_core import (POLICIES,BUDGETS,HOME,xy_grid,public_features,execute,evaluate)

SOURCE=ROOT/'output/disaster_search_multievent36_v1'
OUT=ROOT/'output/disaster_route_real_xbd36_v1'
SOURCES=('real_xbd_route_core.py','test_route_real_xbd.py','run_real_xbd_routes.py','real_xbd_viewer.html')
NAMES={'pre_static':'灾前纹理静态','post_static':'灾后线索静态','online_frozen':'在线重算但无反馈传播','online_neighbor':'在线邻域反馈基线'}


def verify_old():
    r=read(SOURCE/'completion_receipt.json')
    for n,h in r['files'].items():checked(SOURCE/n,h)
    for n,h in r.get('external_files',{}).items():checked(n,h)
    for n,h in r['source_code_sha256'].items():checked(ROOT/'work/disaster_search'/n,h)
    for stage in ('disaster_route_loop_v1','disaster_route_factor_v2','disaster_route_snapshot_v3','disaster_route_trigger_v4','disaster_route_dual_map_demo_v1'):
        base=ROOT/'output'/stage
        for n,h in read(base/'completion_receipt.json')['files'].items():checked(base/n,h)


def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['source_sha256'].items():checked(HERE/n,h)
    for n,h in p['input_sha256'].items():checked(n,h)
    return p


def plan():
    assert not (OUT/'protocol.json').exists();verify_old()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_route*.py','-v'],capture_output=True,text=True)
    log=result.stdout+result.stderr;(OUT/'test_log.txt').write_text(log);assert result.returncode==0 and 'Ran 77 tests' in log
    inputs={str(SOURCE/n):digest(SOURCE/n) for n in ('selected.json','RGB_manifest.json','rgb_jobs.json','cache_manifest.json','reference_manifest.json','completion_receipt.json')}
    inputs[str(OUT/'capture_metadata.json')]=digest(OUT/'capture_metadata.json')
    p={'client_date':'2026-11-02','stage':'REAL_XBD36_IMAGE_GRID_ROUTE_REPLAY_V1',
       'source_sha256':{n:digest(HERE/n) for n in SOURCES},'input_sha256':inputs,'test_log_sha256':digest(OUT/'test_log.txt'),
       'selection':'All existing36,three events12each,no replacement or new outcome-based selection;permanently development data.',
       'route_count':432,'budgets_s_proxy':BUDGETS,'primary_budget_s_proxy':240,'policies':POLICIES,
       'prior':'Free cached64BOXpre/post RGB;pretexture gradient midrank;post equal-weight pretexture rank and change rank. Utilities .05+.95rank,not calibrated target probabilities.',
       'geometry':'Each1024chip is a separate4x4inspection grid;centers of native256tiles;hypothetical depot bottom-middle. Never connect unrelated chips.',
       'units':'Distance in256px grid widths;20proxy seconds per grid width,20per observation,1per actual solve. Published sensor GSD not used as PNG pixel spacing;no real meters or verified flight-time claim.',
       'planning':'Common greedy-seeded insertion+2opt using unitless utilities. Pre-plan1 charged to all;post-static/online pay additional1post revision;online pays each subsequent attempt.',
       'feedback':'Only after paid local observation reveal fixed detector severe-pixel count. Online_neighbor averages current post utility with binary severe-presence bits of paid4-neighbors;untuned baseline,not a claimed new method.',
       'control':'Online_frozen uses same rescheduling and fees without propagation;separates feedback from solving again.',
       'model':'Reuse all576 frozen native256 author4flip caches;unchanged threshold. No new inference or random synthetic observation errors.',
       'reference':'Published semantic masks,major/destroyed pixels;unknown255 ignored. No reference label,TP,FP or unseen local prediction passed to planner.',
       'metrics':['Correctly detected severe-reference pixels and pooled event recall','Covered severe-reference pixels','False positives','First arrival/detection withnull/failure rate','Grid distance and proxytime','Planning calls and paid observations'],
       'opportunity_gate':'Keep Sunda zero-detectable cases and every other failure. No family/threshold tuning on this36 and no universal success claim;all-tiles detector ceiling is not a same-budget oracle.',
       'examples':'Per-event previously fixed selection_rank1;full36 remain available in viewer;do not replace unfavorable examples.',
       'claim_scope':'Real-image/model/reference replay with idealized grid motion. Not new independent disasters,real continuous-area flight or field validation.',
       'source_urls':['https://arxiv.org/abs/1911.09296','https://huggingface.co/datasets/isp-uv-es/xview2-taco/tree/44e219dc1d88c9d0903d6b478d7f2f3c435e574f'],
       'gpu_rented':False,'new_training':False,'new_model_calls':0,'new_real_annotations':0}
    save(OUT/'protocol.json',p);save(OUT/'selection_receipt.json',{'protocol_sha256':digest(OUT/'protocol.json'),'routes_executed':0})
    print('FROZEN:77 tests,36 real pairs,432 route replays',flush=True)


def inventory():
    jobs={r['sample_id']:r for r in read(SOURCE/'rgb_jobs.json')}
    caches={(r['sample_id'],r['tile_id']):r for r in read(SOURCE/'cache_manifest.json')}
    refs={r['sample_id']:r for r in read(SOURCE/'reference_manifest.json')}
    return jobs,caches,refs


def prediction(caches,sid,i):
    r=caches[sid,i];checked(r['path'],r['sha256'])
    with np.load(r['path'],allow_pickle=False) as cache:
        pred=cache['prediction'].copy();p=cache['mean_probabilities']
        np.testing.assert_array_equal(pred,np.where(p[0]>.38,p[1:].argmax(0)+1,0).astype(np.uint8))
    return pred


def make_features(jobs,sid):
    images=[np.asarray(Image.open(SOURCE/'coarse'/f'{sid}_{mode}.png')) for mode in ('pre','post')]
    for mode,image in zip(('pre','post'),images):
        row=next(r for r in jobs[sid]['images'] if r['mode']==mode);checked(row['path'],row['sha256'])
        exact=np.asarray(Image.open(row['path']).resize((64,64),Image.Resampling.BOX));np.testing.assert_array_equal(image,exact)
    return public_features(*images)


def tile_counts(caches,refs,sid):
    r=refs[sid];checked(r['reference_file'],r['reference_sha256']);truth=np.asarray(Image.open(r['reference_file']));counts=[]
    for i in range(16):
        x,y=i%4*256,i//4*256;t=truth[y:y+256,x:x+256];pred=prediction(caches,sid,i)
        valid=t!=255;severe=np.isin(t,[3,4]);found=np.isin(pred,[3,4])
        counts.append({'reference':int((valid&severe).sum()),'tp':int((valid&severe&found).sum()),
                       'fp':int((valid&~severe&found).sum()),'predicted_severe_pixels':int(found.sum())})
    return counts


def run():
    verify();assert not (OUT/'routes.json').exists();jobs,caches,refs=inventory();routes=[];metrics=[];features=[];private_counts={}
    for s in read(SOURCE/'selected.json'):
        sid=s['sample_id'];f=make_features(jobs,sid);features.append({'sample_id':sid,'event':s['event'],**f})
        # Evaluator is deliberately separate from observation callback/public features.
        cs=tile_counts(caches,refs,sid);private_counts[sid]=cs
        for budget in BUDGETS:
            for policy in POLICIES:
                seen=[]
                def observe(i,paid_s):
                    if i in seen:raise ValueError('Repeated cached acquisition')
                    seen.append(i)
                    return int(np.isin(prediction(caches,sid,i),[3,4]).sum())
                r=execute(f,policy,budget,observe);assert seen==r['state']['visited']
                metadata={'sample_id':sid,'event':s['event'],'budget_s_proxy':budget,'policy':policy}
                r.update(metadata);routes.append(r);metrics.append({**metadata,**evaluate(r,cs)})
        print(json.dumps({'scenes':len(features),'routes':len(routes)}),flush=True) if len(features)%12==0 else None
    save(OUT/'public_features.json',features);save(OUT/'private_tile_scores.json',private_counts)
    save(OUT/'routes.json',routes);save(OUT/'metrics.json',metrics)
    summary=[]
    for event in sorted({r['event'] for r in metrics}):
        for budget in BUDGETS:
            for policy in POLICIES:
                group=[r for r in metrics if r['event']==event and r['budget_s_proxy']==budget and r['policy']==policy];assert len(group)==12
                total=sum(r['reference_severe_pixels'] for r in group);tp=sum(r['true_positive_pixels'] for r in group);fp=sum(r['false_positive_pixels'] for r in group)
                summary.append({'event':event,'budget_s_proxy':budget,'policy':policy,'scenes':12,'reference_severe_pixels':total,
                    'true_positive_pixels':tp,'false_positive_pixels':fp,'recall':tp/total if total else None,
                    'precision':tp/(tp+fp) if tp+fp else None,'coverage_recall':sum(r['covered_reference_pixels'] for r in group)/total if total else None,
                    'reference_positive_scenes':sum(r['reference_severe_pixels']>0 for r in group),
                    'all_tile_detectable_scenes':sum(r['all_tiles_detector_tp_ceiling']>0 for r in group),
                    'all_tiles_detector_recall_ceiling':sum(r['all_tiles_detector_tp_ceiling'] for r in group)/total if total else None,
                    **{'mean_'+k:float(np.mean([r[k] for r in group])) for k in ('observations','planning_calls','total_distance_grid','total_time_s_proxy','first_detection_or_budget_s_proxy')},
                    'no_correct_detection_rate':sum(r['no_correct_detection'] for r in group)/12})
    save(OUT/'event_summary.json',summary)
    print('EXECUTED:432 routes,real cached observations,no synthetic targets',flush=True)


def audit():
    verify();jobs,caches,refs=inventory();features={r['sample_id']:r for r in read(OUT/'public_features.json')}
    counts=read(OUT/'private_tile_scores.json');rows=read(OUT/'metrics.json');routes=read(OUT/'routes.json');lookup={(r['sample_id'],r['budget_s_proxy'],r['policy']):r for r in rows}
    assert len(routes)==len(lookup)==432;xy=xy_grid();steps_checked=plans_checked=0
    for sid in features:
        rebuilt=make_features(jobs,sid)
        for k in rebuilt:assert features[sid][k]==rebuilt[k]
        assert counts[sid]==tile_counts(caches,refs,sid)
    for r in routes:
        sid=r['sample_id'];s=r['state'];f=features[sid];elapsed=distance=0.;pos=HOME;visited=[];pending=[];cursor=0
        for k in range(len(s['history'])+1):
            while cursor<len(s['planning_events']) and s['planning_events'][cursor]['after_observations']==k:
                e=s['planning_events'][cursor];assert abs(e['elapsed_before_s_proxy']-elapsed)<1e-7;elapsed+=1
                assert abs(e['elapsed_after_s_proxy']-elapsed)<1e-7 and e['position_index']==pos
                weights=np.array(f['pre_utility'] if e['reason']=='pre_plan' else f['post_utility'])
                if r['policy']=='online_neighbor' and e['reason']!='pre_plan':
                    for j in range(16):
                        bits=[h['predicted_severe_pixels']>0 for h in s['history'][:k] if abs(h['tile_id']//4-j//4)+abs(h['tile_id']%4-j%4)==1]
                        if bits:weights[j]=(weights[j]+sum(bits))/(len(bits)+1)
                np.testing.assert_allclose(weights,e['utility'],atol=1e-12,rtol=0)
                pending=list(e['planned_route']);assert not set(pending)&set(visited) and len(pending)==len(set(pending))
                nodes=[pos,*pending,HOME];cost=sum(math.hypot(*(xy[a]-xy[b])) for a,b in zip(nodes,nodes[1:]))*20+len(pending)*20
                assert elapsed+cost<=r['budget_s_proxy']+1e-7
                cursor+=1;plans_checked+=1
            if k==len(s['history']):break
            h=s['history'][k];i=h['tile_id'];assert pending.pop(0)==i and i not in visited
            leg=math.hypot(*(xy[i]-xy[pos]));arrival=elapsed+20*leg;end=arrival+20
            for name,value in [('elapsed_before_s_proxy',elapsed),('leg_grid',leg),('arrival_s_proxy',arrival),('observation_complete_s_proxy',end)]:assert abs(h[name]-value)<1e-7
            assert h['predicted_severe_pixels']==counts[sid][i]['predicted_severe_pixels']
            elapsed=end;distance+=leg;pos=i;visited.append(i);steps_checked+=1
        back=math.hypot(*(xy[pos]-xy[HOME]));elapsed+=back*20;distance+=back
        assert abs(elapsed-s['elapsed_s_proxy'])<1e-7 and abs(distance-s['distance_grid'])<1e-7 and visited==s['visited'] and s['closed'] and not pending
        assert elapsed<=r['budget_s_proxy']+1e-7
        score=evaluate(r,counts[sid]);expected=lookup[sid,r['budget_s_proxy'],r['policy']]
        for key,value in score.items():assert expected[key]==value
        # Deterministic public-only rerun checks every action without using reference.
        replay=execute(f,r['policy'],r['budget_s_proxy'],lambda i,t:counts[sid][i]['predicted_severe_pixels'])
        assert replay['state']==s and replay['frames']==r['frames']
    for group in read(OUT/'event_summary.json'):
        rs=[r for r in rows if all(r[k]==group[k] for k in ('event','budget_s_proxy','policy'))];assert len(rs)==12
        ref=sum(r['reference_severe_pixels'] for r in rs);tp=sum(r['true_positive_pixels'] for r in rs)
        assert ref==group['reference_severe_pixels'] and tp==group['true_positive_pixels']
        assert group['recall']==(tp/ref if ref else None)
    verify_old();save(OUT/'audit_report.json',{'status':'PASS','real_pairs':36,'all_local_caches_checked':576,'routes_replayed':432,
        'paid_observations_replayed':steps_checked,'paid_plans_replayed':plans_checked,'budget_violations':0,
        'reference_labels_passed_to_planner':False,'new_model_calls':0,'random_target_fields_used':False,
        'distance_scope':'Image-grid units,not verified ground meters','new_independent_validation':False})
    print(json.dumps(read(OUT/'audit_report.json')),flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import base64,io
    verify();assert read(OUT/'audit_report.json')['status']=='PASS';figures=OUT/'figures';assets=OUT/'assets';figures.mkdir(exist_ok=True);assets.mkdir(exist_ok=True)
    summary=read(OUT/'event_summary.json');events=sorted({r['event'] for r in summary});colors=('#64748b','#2563eb','#d97706','#12805c')
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for col,event in enumerate(events):
        for row,metric in enumerate(('coverage_recall','recall')):
            ax=axes[row,col]
            for policy,color in zip(POLICIES,colors):
                rs=sorted([r for r in summary if r['event']==event and r['policy']==policy],key=lambda r:r['budget_s_proxy'])
                ax.plot([r['budget_s_proxy'] for r in rs],[100*r[metric] if r[metric] is not None else np.nan for r in rs],'-o',c=color,label=policy,markersize=4)
            if row==1:ax.axhline(100*next(r['all_tiles_detector_recall_ceiling'] for r in summary if r['event']==event),c='#b91c1c',ls=':',label='All-tile detector ceiling')
            ax.set(title=event,xlabel='Proxy budget units',ylim=(-2,102),xticks=BUDGETS);ax.grid(alpha=.2)
        axes[0,col].set_ylabel('Covered severe reference (%)');axes[1,col].set_ylabel('Correctly detected reference (%)')
    axes[1,2].legend(fontsize=7);fig.suptitle('REAL xBD36 | fixed detector replay | image-grid movement,not physical flight\nAll3 events retained,including zero detector opportunity;240 is primary')
    fig.savefig(figures/'real_route_budget.png',dpi=170);plt.close(fig)
    # Browser assets are recorded after execution and never supplied to the agent.
    jobs,caches,refs=inventory();features={r['sample_id']:r for r in read(OUT/'public_features.json')};routes=read(OUT/'routes.json');selected=read(SOURCE/'selected.json')
    data={'events':events,'policies':NAMES,'budgets':BUDGETS,'cases':[]}
    for item in selected:
        sid=item['sample_id'];pred_full=np.zeros((1024,1024),bool)
        for i in range(16):pred_full[i//4*256:(i//4+1)*256,i%4*256:(i%4+1)*256]=np.isin(prediction(caches,sid,i),[3,4])
        truth=np.asarray(Image.open(refs[sid]['reference_file']));overlay=np.zeros((1024,1024,4),np.uint8);overlay[np.isin(truth,[3,4])]=[230,75,45,180]
        pred_overlay=np.zeros_like(overlay);pred_overlay[pred_full]=[230,145,30,190]
        for role,array in [('reference',overlay),('prediction',pred_overlay)]:Image.fromarray(array).resize((512,512),Image.Resampling.NEAREST).save(assets/f'{sid}_{role}.png')
        for mode in ('pre','post'):
            im=Image.open(next(r['path'] for r in jobs[sid]['images'] if r['mode']==mode))
            im.resize((512,512),Image.Resampling.BOX).save(assets/f'{sid}_{mode}.jpg',quality=88)
            shutil.copyfile(SOURCE/'coarse'/f'{sid}_{mode}.png',assets/f'{sid}_{mode}_coarse.png')
        data['cases'].append({'sample_id':sid,'event':item['event'],'selection_rank':item['selection_rank'],'features':features[sid],
            'traces':{f"{r['policy']}|{r['budget_s_proxy']}":{'state':r['state'],'frames':r['frames']} for r in routes if r['sample_id']==sid}})
    save(OUT/'public_replay.json',data)
    (OUT/'demo.html').write_text((HERE/'real_xbd_viewer.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')))
    rows=[r for r in summary if r['budget_s_proxy']==240]
    table='\n'.join(f"| {r['event']} | {NAMES[r['policy']]} | {100*r['coverage_recall']:.2f}% | {100*r['recall']:.2f}% | {r['mean_observations']:.2f} | {r['mean_planning_calls']:.2f} |" for r in rows)
    (OUT/'README.md').write_text(f'''# 真实xBD36：双层地图Agent的路线回放

2026-11-02。[打开真实数据回放](demo.html)。此页面需保留相邻assets目录。全部36对真实前后影像、固定模型的576块预测及发布语义参考来自既有缓存；没有随机生成目标、随机误检漏检、新训练或新模型推理。每幅影像是独立任务，不能将36块散布影像拼接成连续灾区航线。

## 能说明什么

这轮检验真实灾前/灾后图像线索、真实识别误差、付费观察与路线选择的关系。浏览器先前的合成例只证明接口可运行，尚不能满足真实部署需求；本轮也仍是影像网格回放，不是实际无人机、地面搜救或真实通行实验。

第一层由64像素粗图构造灾前纹理代理和灾后变化排名，二者均非已校准灾损概率。第二层只在支付移动及观察成本后写入固定模型的局部严重预测数量。在线邻域反馈是未经调参的4邻域简单基线；同时保留相同重算但无传播对照。发布参考标签只供评价，不参与路线决策。

两种双层组织继续共享信息：先验/线索＋沿途证据；或几何代价＋语义搜索状态。切换组织不自动改变路线，不作为新方法收益。

## 240主预算结果

| 事件 | 策略 | 覆盖严重参考 | 正确检出严重参考 | 平均观察格 | 平均规划次数 |
|---|---|---:|---:|---:|---:|
{table}

覆盖指访问区域包含多少严重发布参考像素；正确检出还要求固定模型预测与参考重叠。海啸全部局部块的真阳性上限仍为0，路线不能补救模型完全漏检的目标。零参考场景不删除，其召回为null，另报失败率。36对是已使用开发数据，不是独立确认。

![分事件预算结果](figures/real_route_budget.png)

## 预算与地理限制

每幅1024图像分16个256格，起终点假设在底边中部。距离单位为256像素格宽；移动一格宽计20代理时间单位、观察20、每次求解1。预算120/240/360，240主比较，所有移动、观察、规划和返程均计费。灾前规划1计入所有方法，灾后/在线另付1修订；每次在线重算另付1。

补读了发布拍摄GSD和时间，仅新增84,672字节公共元数据，没有读取新的损伤或几何列。传感器GSD不是已经核实的PNG地理变换，故未据此报告真实米数；20时间单位也不是实测飞行时间。道路断裂、障碍、地形、禁飞、通信和真实平台约束仍缺。不能从本报告推断现场可执行性。

## 核验与复现

77项路线测试通过，其中新增10项。432条完整路线的费用、观察、效用更新和评分重放，576份模型缓存按原阈值重新解码。全36及三个事件逐一报告，不补挑阳性、不把旧SantaRosa加入救分，不在本批调邻域、权重或阈值。缓存离线生成是评测装置；在线策略只能取得已付格的预测，未访问格不免费暴露。

离线浏览器记录包含完整已执行轨迹和评价图层，界面按时间轴渲染前缀；评价参考开关不进入Agent状态。真实影像、预测、发布参考三者分开标识，发布参考不是新的人工现场真值。新增图片只是既有缓存的展示缩略图，未改变感知输入。

原始来源：[xBD论文](https://arxiv.org/abs/1911.09296)、[固定TACO版本](https://huggingface.co/datasets/isp-uv-es/xview2-taco/tree/44e219dc1d88c9d0903d6b478d7f2f3c435e574f)。数据与展示衍生材料遵循CC-BY-NC-SA-4.0；影像归Maxar Open Data，原数据归xBD作者/DIU/CMU SEI。
''')
    print('REPORT_AND_REAL_DATA_VIEWER_READY',flush=True)


def seal():
    p=verify();verify_old();assert read(OUT/'audit_report.json')['status']=='PASS'
    assert read(OUT/'visual_inspection.json')['browser_and_figure_verified']
    snapshots=OUT/'source_snapshots';snapshots.mkdir(exist_ok=True)
    for name in (*SOURCES,'acquire_xbd_route_metadata.py','acquire_xbd_route_metadata_resumed.py','finish_xbd_capture_metadata.py'):
        shutil.copyfile(HERE/name,snapshots/name)
    receipt={'status':'COMPLETE','client_date':'2026-11-02','real_pairs':36,'routes':432,'tests_passed':77,'new_tests':10,
      'new_model_calls':0,'new_training':False,'gpu_rented':False,'new_manual_labels':0,'physical_flight_validated':False,
      'budget_violations':0,'source_sha256':p['source_sha256'],
      'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file()}}
    save(OUT/'completion_receipt.json',receipt);print('SEALED '+digest(OUT/'completion_receipt.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('plan','run','audit','report','seal'));args=p.parse_args()
    assert not (OUT/'completion_receipt.json').exists()
    {'plan':plan,'run':run,'audit':audit,'report':report,'seal':seal}[args.command]()
