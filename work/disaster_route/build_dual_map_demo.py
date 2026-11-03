"""Freeze,execute and verify an information-equivalent two-view agent demo."""
import argparse
import json
import math
import shutil
import subprocess
import sys

import numpy as np

from run_route_loop import ROOT,HERE,DATA,maps,read,save,digest,checked
from run_route_trigger import OUT as PARENT,verify_previous,verify as verify_parent
from route_core import Config
from route_factor_core import Planner
from route_snapshot_core import PrivateWorld,fingerprint
from dual_map_core import DualMapAgent,MODES,POLICIES

OUT=ROOT/'output/disaster_route_dual_map_demo_v1'
SOURCES=('dual_map_core.py','test_route_dual_map.py','build_dual_map_demo.py','dual_map_viewer.html')
SEED=2026110200


def old_integrity():
    verify_parent();verify_previous()
    for n,h in read(PARENT/'completion_receipt.json')['files'].items():checked(PARENT/n,h)


def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['source_sha256'].items():checked(HERE/n,h)
    for n,h in p['input_sha256'].items():checked(n,h)
    checked(OUT/'test_log.txt',p['test_log_sha256'])
    return p


def plan():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists();old_integrity()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_route*.py','-v'],capture_output=True,text=True)
    log=result.stdout+result.stderr;(OUT/'test_log.txt').write_text(log)
    assert result.returncode==0 and 'Ran 67 tests' in log
    inputs={str(p):digest(p) for p in (DATA,PARENT/'synthetic_worlds.json',PARENT/'completion_receipt.json')}
    inputs.update(read(ROOT/'output/disaster_route_loop_v1/protocol.json')['external_sha256'])
    protocol={'client_date':'2026-11-02','stage':'DUAL_MAP_AGENT_FUNCTIONAL_DEMO_V1',
       'source_sha256':{n:digest(HERE/n) for n in SOURCES},'input_sha256':inputs,'test_log_sha256':digest(OUT/'test_log.txt'),
       'purpose':'Implement and compare two information-equivalent map organizations in the complete pre/post-disaster paid inspection loop,not establish algorithmic efficacy.',
       'modes':{'prior_evidence':'Prior/cue layer + paid observation evidence,with shared geometry.',
                'geometry_semantic':'Straight-line geometric cost/budget feasibility + semantic search state retaining the same prior,evidence and fusion.'},
       'selection':{'regions':list(maps()),'seed':SEED,'prior_mode':'aligned','noise':'moderate','budget_s':1200,
                    'reused_worlds':2,'policies':POLICIES,'route_executions':12,'no_outcome_based_selection':True},
       'agent_actions':'Pre-plan,receive post cues,optionally post-plan,move and observe,update maps,optional every5 replan,return.',
       'costs':{'speed_m_s':5,'observe_s':20,'planning_call_s':1,
           'initial_planning':'All policies pay1 pre-plan;post_static and online_5 additionally pay1 post-plan. Both are charged to the same1200s budget;the pre plan is not a free trial.',
           'online':'Every5 observations replan if another paid inspection+return after planning is affordable;all actual calls charged.',
           'scope':'Planar metric movement,no terrain/road/flight safety validation.'},
       'source_truth':'Real PS geometry and change cue;preprior uniform placeholder,postprior synthetic rank construction;targets and observed bits reused simulation,not verified real damage.',
       'functional_gates':['No paid evidence before completed inspection','Prior immutable under online observation','Two modes produce identical actions and costs with identical information',
                           'Snapshot restore preserves both maps and no repeated prefix charge','All tours return within budget','No private target or future-bit arrays in browser data'],
       'figures':'Both regions,online_5,pre_plan/post_plan/after10 replan;both map organizations. No simulated true target locations drawn.',
       'claim_gate':'Two map organizations are not two new planners. No dual-map improvement claim without a future information-fair comparison.',
       'new_training':False,'new_perception_calls':0,'gpu_rented':False,'new_real_labels':0}
    save(OUT/'protocol.json',protocol);save(OUT/'selection_receipt.json',{'protocol_sha256':digest(OUT/'protocol.json'),'new_routes_executed':0})
    print('FROZEN:67 tests,12 functional routes,2 map organizations',flush=True)


def worlds():
    return {w['region']:w for w in read(PARENT/'synthetic_worlds.json') if w['mode']=='aligned' and w['seed']==SEED}


def private(g,w):
    target=set(w['target_ids']);labels=np.array([sid in target for sid in g.ids],bool)
    bits=np.array(w['uniform_observation_draws'])<np.where(labels,.8,.1)
    return PrivateWorld('dual-map-demo|'+g.region,labels,bits,(.8,.1))


def run():
    verify();assert not (OUT/'routes.json').exists()
    mm=maps();ww=worlds();planner=Planner();routes=[];snapshots=[];metrics=[];exports=[]
    for region,g in mm.items():
        world=private(g,ww[region])
        for policy in POLICIES:
            for mode in MODES:
                a=DualMapAgent(g,world,mode,policy,planner);a.prepare();snapshot=None
                while a.session.state['remaining_route']:
                    a.advance()
                    if len(a.maps.records)==10 and snapshot is None:
                        snapshot=json.loads(json.dumps(a.snapshot()))
                        exports.append({'region':region,'policy':policy,'mode':mode,'after_observations':10,'view':a.view()})
                r=a.finish()
                assert snapshot is not None
                b=DualMapAgent.restore(g,snapshot,{world.token:world},planner)
                assert b.finish()==r
                routes.append(r);snapshots.append({'region':region,'policy':policy,'mode':mode,'snapshot':snapshot})
                s=r['final_public'];h=s['history'];correct=[x for x in h if world.targets[x['cell_index']] and x['observed_positive']]
                metrics.append({'region':region,'policy':policy,'mode':mode,'observations':len(h),
                    'total_time_s':s['elapsed_s'],'total_distance_m':s['distance_m'],'planning_calls':len(s['planning_events']),
                    'simulated_correct_detections':len(correct),'simulated_false_positives':sum(not world.targets[x['cell_index']] and x['observed_positive'] for x in h)})
    assert len(routes)==len(snapshots)==12
    save(OUT/'routes.json',routes);save(OUT/'agent_snapshots.json',snapshots);save(OUT/'map_views_after10.json',exports);save(OUT/'metrics.json',metrics)
    # Browser data contains public-only completed histories,not evaluator labels.
    original=read(DATA);public={'schema':1,'description':'Recorded functional replay,not live vehicle control','regions':{}}
    for region,g in mm.items():
        traces={}
        for policy in POLICIES:
            a,b=[next(r for r in routes if r['region']==region and r['policy']==policy and r['mode']==mode) for mode in MODES]
            assert a['final_public']==b['final_public'] and a['frames']==b['frames'] and a['maps']==b['maps']
            traces[policy]={'steps':a['final_public']['history'],'frames':a['frames'],'map_modes_action_equal':True}
        public['regions'][region]={'ids':g.ids,'xy_m':g.xy.tolist(),'blocks':g.blocks.tolist(),'start_index':g.start_index,
            'prior_pre':g.prior_pre.tolist(),'prior_post':g.prior_post.tolist(),
            'cue':[float(x) if np.isfinite(x) else None for x in g.cue],
            'images':{'pre':original[region]['images']['灾前'],'post':original[region]['images']['灾后']},'traces':traces}
    save(OUT/'public_demo.json',public)
    (OUT/'demo.html').write_text((HERE/'dual_map_viewer.html').read_text().replace('__DEMO_DATA__',json.dumps(public,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')))
    print('EXECUTED:12 full routes,12 snapshot continuations,6 mode parity checks',flush=True)


def near(a,b):
    assert abs(a-b)<1e-7,(a,b)


def audit():
    verify();old_integrity();mm=maps();ww=worlds();routes=read(OUT/'routes.json');assert len(routes)==12
    public=read(OUT/'public_demo.json');solver=Planner();observations=plans=frames=0
    def bayes(q,bit):
        hot=.1+.7*.6;cold=.1+.7*.02
        if not bit:hot,cold=1-hot,1-cold
        return q*hot/(q*hot+(1-q)*cold)
    for r in routes:
        g=mm[r['region']];w=private(g,ww[r['region']]);s=r['final_public'];used=[];position=g.start_index;elapsed=travel=0.;cursor=0;pending=[]
        q=g.prior_post.copy()
        for j in range(len(s['history'])+1):
            es=[]
            while cursor<len(s['planning_events']) and s['planning_events'][cursor]['after_observations']==j:
                es.append(s['planning_events'][cursor]);cursor+=1
            if j==0:assert len(es)==1+int(r['policy']!='pre_static')
            elif es:assert len(es)==1 and r['policy']=='online_5' and j%5==0
            for e in es:
                near(e['elapsed_before_s'],elapsed);assert e['position_index']==position
                elapsed+=1;near(e['elapsed_after_s'],elapsed)
                belief=g.prior_pre.copy() if e['map_phase']=='pre' else q.copy()
                mask=np.zeros(len(g.ids),bool);mask[used]=True
                route,_=solver.solve(g,'insertion_2opt',g.xy[position],mask,elapsed,belief,Config(max_visits=len(g.ids)),0.,len(g.ids)-len(used))
                assert e['planned_route']==route;pending=list(route);plans+=1
            if j==len(s['history']):break
            h=s['history'][j];i=h['cell_index'];assert pending.pop(0)==i and i not in used
            leg=math.hypot(*(g.xy[i]-g.xy[position]));arrival=elapsed+leg/5;end=arrival+20
            for name,value in [('elapsed_before_step_s',elapsed),('leg_m',leg),('arrival_s',arrival),('observation_complete_s',end)]:near(h[name],value)
            assert h['observed_positive']==bool(w.observations[i]);b=int(g.blocks[i]);near(h['block_belief_before'],q[b]);q[b]=bayes(q[b],h['observed_positive']);near(h['block_belief_after'],q[b])
            assert end+math.hypot(*(g.xy[i]-g.xy[g.start_index]))/5<=1200+1e-8
            used.append(i);position=i;elapsed=end;travel+=leg;observations+=1
        assert not pending and used==s['visited_indices']
        back=math.hypot(*(g.xy[position]-g.xy[g.start_index]));elapsed+=back/5;travel+=back
        near(elapsed,s['elapsed_s']);near(travel,s['distance_m']);near(back,s['return_m'])
        assert s['closed'] and s['position_index']==g.start_index and elapsed<=1200+1e-8
        near(elapsed,travel/5+20*len(used)+len(s['planning_events']))
        records=[{k:h[k] for k in ('step','cell_index','cell_id','block_index','observed_positive','observation_complete_s')} for h in s['history']]
        assert records==r['maps']['records']
        np.testing.assert_allclose(q,r['final_public']['belief_hot'],rtol=0,atol=1e-12)
        for f in r['frames']:
            k=f['observations'];assert k<=len(used)
            prior=g.prior_pre if f['phase']=='pre' else g.prior_post;belief=prior.copy()
            for h in s['history'][:k]:belief[h['block_index']]=bayes(belief[h['block_index']],h['observed_positive'])
            np.testing.assert_allclose(belief,f['block_hot_posterior'],rtol=0,atol=1e-12)
            payload={'schema':1,'geometry_key':r['maps']['geometry_key'],'phase':f['phase'],'assumed_likelihood':(.8,.1),'records':records[:k]}
            assert fingerprint(payload)==f['map_state_sha256']
            spent=sum(h['leg_m']/5+20 for h in s['history'][:k])+f['planning_calls']
            if f['closed']:spent+=back/5
            near(spent,f['elapsed_s']);frames+=1
        display=public['regions'][g.region]['traces'][r['policy']]
        assert display['frames']==r['frames'] and display['steps']==s['history']
    for region in mm:
        for policy in POLICIES:
            a,b=[next(r for r in routes if r['region']==region and r['policy']==policy and r['mode']==mode) for mode in MODES]
            assert a['final_public']==b['final_public'] and a['maps']==b['maps'] and a['frames']==b['frames']
    # Structural allow-list rejects hidden target/future-bit fields in browser data.
    assert set(public)=={'schema','description','regions'}
    for region,v in public['regions'].items():
        assert set(v)=={'ids','xy_m','blocks','start_index','prior_pre','prior_post','cue','images','traces'}
        for p,t in v['traces'].items():assert set(t)=={'steps','frames','map_modes_action_equal'}
    for view in read(OUT/'map_views_after10.json'):
        v=view['view'];ev=v['layers']['observation_evidence'] if view['mode']=='prior_evidence' else v['layers']['semantic_search']['evidence']
        assert sum(x is not None for x in ev['signals'])==10 and ev['confirmed_targets'] is None
        if view['mode']=='geometry_semantic':
            assert v['layers']['geometry_cost']['road_passability'] is None and v['layers']['geometry_cost']['terrain_obstacles'] is None
    for meta in read(OUT/'agent_snapshots.json'):
        g=mm[meta['region']];w=private(g,ww[meta['region']]);a=DualMapAgent.restore(g,meta['snapshot'],{w.token:w},solver)
        r=next(r for r in routes if all(r[k]==meta[k] for k in ('region','mode','policy')));assert a.finish()==r
    save(OUT/'audit_report.json',{'status':'PASS','routes':12,'paid_observations':observations,'paid_planning_calls':plans,
         'public_frames_checked':frames,'map_organization_parities':6,'snapshot_continuations':12,'budget_violations':0,
         'private_target_or_future_bit_arrays_in_browser':False,'new_algorithm_effect_established':False,
         'old_frozen_outputs_unchanged':True})
    print(json.dumps(read(OUT/'audit_report.json')),flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    verify();assert read(OUT/'audit_report.json')['status']=='PASS';figures=OUT/'figures';figures.mkdir(exist_ok=True)
    public=read(OUT/'public_demo.json')
    for region,g in public['regions'].items():
        xy=np.array(g['xy_m']);xy-=xy.min(0);blocks=np.array(g['blocks']);t=g['traces']['online_5'];steps=t['steps']
        selected=[next(f for f in t['frames'] if f['reason']==reason) for reason in ('pre_plan','post_plan')]
        selected.append(next(f for f in t['frames'] if f['reason']=='online_replan_5' and f['observations']==10))
        for mode in MODES:
            fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
            for col,f in enumerate(selected):
                seen={h['cell_index']:h['observed_positive'] for h in steps[:f['observations']]}
                prior=np.array(g['prior_pre'] if f['phase']=='pre' else g['prior_post'])
                for row in range(2):
                    ax=axes[row,col]
                    if mode=='prior_evidence' and row==0:values=prior[blocks];cmap='Greens';vmax=.4;label='Prior / cues'
                    elif mode=='prior_evidence':values=np.array([2 if seen.get(i) is True else 1 if i in seen else 0 for i in range(len(xy))]);cmap=ListedColormap(['#e1e6df','#557cac','#d38a32']);vmax=2;label='Paid evidence'
                    elif row==0:
                        move=np.linalg.norm(xy-xy[f['position_index']],axis=1);home=np.linalg.norm(xy-xy[g['start_index']],axis=1)
                        values=(f['elapsed_s']+1+move/5+20+home/5<=1200).astype(float);values[list(seen)]=0;cmap=ListedColormap(['#e1e6df','#a4c8b0']);vmax=1;label='Geometric budget feasibility'
                    else:values=.02+.58*np.array(f['block_hot_posterior'])[blocks];cmap='Greens';vmax=.6;label='Fused semantic expectation'
                    ax.scatter(xy[:,0],xy[:,1],c=values,cmap=cmap,vmin=0,vmax=vmax,s=14,marker='s')
                    prefix=[g['start_index']]+[h['cell_index'] for h in steps[:f['observations']]]
                    suffix=[f['position_index'],*f['remaining_route'],g['start_index']]
                    ax.plot(xy[suffix,0],xy[suffix,1],'--',c='#26845d',linewidth=1)
                    ax.plot(xy[prefix,0],xy[prefix,1],c='#20342d',linewidth=1.5)
                    ax.scatter(*xy[g['start_index']],marker='*',s=75,c='#172c23');ax.scatter(*xy[f['position_index']],s=25,c='#d77929')
                    ax.set_aspect('equal');ax.set(title=f"{label}\n{f['elapsed_s']:.1f}s | {f['observations']} paid observations",xlabel='Local easting (m)')
                    if col==0:ax.set_ylabel('Local northing (m)')
            en='Port' if region=='口岸' else 'South'
            fig.suptitle(f"FUNCTIONAL SIMULATION | {en} | {mode}\nPre plan / post update / after10 observations + replan | solid=paid prefix,dashed=current suffix incl.return")
            fig.savefig(figures/f'{en.lower()}_{mode}.png',dpi=160);plt.close(fig)
    metrics=read(OUT/'metrics.json');table='\n'.join(f"| {r['region']} | {r['policy']} | {r['observations']} | {r['planning_calls']} | {r['total_distance_m']:.0f} | {r['total_time_s']:.1f} |" for r in metrics if r['mode']=='prior_evidence')
    (OUT/'README.md').write_text(f'''# 灾前灾后Agent搜查：两种双层地图原型

2026-11-02。[打开可切换回放](demo.html)。无需服务端或网络，直接用浏览器打开本目录的demo.html。

已经接通同一Agent的灾前规划、接收灾后线索、执行付费观察、更新地图、每5次重规划和返程。界面可选两区、三种路线策略、两种地图组织，并逐步回放或跳转灾前、灾后、第10次观察和最终返程。点击格子查看当时可用线索、后验和观测状态。

| 方案 | 第一层 | 第二层 | 决策输入 |
|---|---|---|---|
| 1 | 搜索先验/线索 | 沿途付费观测证据 | 两层融合信念＋共享几何与预算 |
| 2 | 平面几何代价/预算可行性 | 保留同一先验与证据的语义搜索状态 | 相同融合信念＋相同几何与预算 |

两种组织保留完全相同的信息，由同一求解器产生逐动作相同的路线。这是接口公平性与功能一致性验证，**不是两种新算法的效果比较，也不说明双层地图已提高搜查收益**。两种组织可以在以后组合使用：第一种管理搜索记忆，第二种向路线规划提供代价与语义接口。

## 已实际完成

使用V4已经用过的两区第一个种子2026110200、先验正确、中等模拟误差、1200秒案例。每区三种策略各执行两种地图模式，共12条闭合路线；没有按发现结果换例。三种策略为保持灾前路线、灾后修订一次后固定、沿途每5次观察重规划。

67项测试通过，其中新增14项。全部12条路线、两模式的6组逐动作一致性、12个第10次观察后的状态恢复与继续执行均已核验。证据只在移动和20秒观察完成后写入；未知格保持未知，阳性不升级成独立真值，阴性不等于排除目标。原始先验在沿途观察时不被覆盖，后验由当前先验和付费证据重新构造。

所有策略支付1秒灾前规划；灾后静态和在线策略再支付1秒灾后修订。后续调用也逐次收费，全部包含在同一1200秒预算中，返程另按实际距离计费。因此这批功能案例的准备费约定与V4不同，不能混合绝对分数排名。以下只报告几何和费用，不用两例正确发现量宣称方法优劣。

| 区域 | 策略 | 观察格 | 付费规划次数 | 往返米 | 总秒 |
|---|---|---:|---:|---:|---:|
{table}

## 两层地图与路线阶段

四张图分别展示两区的两种组织，列为灾前计划、灾后修订、第10次观察后重规划；两行是各自两层。黑色实线为已经支付的路径，绿色虚线为当前余下路线及返程。地图不画未来观测或模拟目标真值位置。

![口岸方案1](figures/port_prior_evidence.png)
![口岸方案2](figures/port_geometry_semantic.png)
![南侧方案1](figures/south_prior_evidence.png)
![南侧方案2](figures/south_geometry_semantic.png)

## 输入边界与下一项

PS几何、已有前后影像和变化分数来自既有缓存。灾前仍使用均匀占位先验，尚未完成真实建筑/人口暴露图；灾后热点先验由变化排名人为构造，不是已校准灾损概率。观测及目标沿用模拟世界，固定误差率0.8/0.1为已知模拟假设，未调用新识别器。

方案2目前是直线移动代价和预算可行性，真实道路通行、地形障碍、禁飞及飞行动力学没有接入；缺失值保留null。界面不把预算可行性写成已验证通行或飞行安全。实际无人机、VLM或LLM工具调度尚未接通，当前Agent为可恢复的确定性闭环控制原型。

这一步建立状态表示、付费更新和可解释回放。下一步优先补有来源的灾前结构信息与几何通行约束，在同样信息和求解器下再冻结效果实验；单层充分统计量基线不能通过故意丢失信息被削弱。已有基线、负例和感知汇报保持。
''')
    print('REPORT_READY:4 figures + self-contained HTML replay',flush=True)


def seal():
    p=verify();old_integrity();assert read(OUT/'audit_report.json')['status']=='PASS'
    visual=read(OUT/'visual_inspection.json');assert visual['four_figures_viewed'] and visual['browser_interactions_passed']
    dest=OUT/'source_snapshots';dest.mkdir(exist_ok=True)
    for n in SOURCES:shutil.copyfile(HERE/n,dest/n)
    receipt={'status':'COMPLETE','client_date':'2026-11-02','purpose':'FUNCTIONAL_INTERFACE_AND_REPLAY_NOT_EFFICACY',
        'tests_passed':67,'new_tests':14,'route_executions':12,'mode_action_parities':6,'snapshot_continuations':12,
        'budget_violations':0,'new_training':False,'new_perception_calls':0,'gpu_rented':False,'new_real_labels':0,
        'source_sha256':p['source_sha256'],'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file()}}
    save(OUT/'completion_receipt.json',receipt);print('SEALED '+digest(OUT/'completion_receipt.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('plan','run','audit','report','seal'));args=parser.parse_args()
    assert not (OUT/'completion_receipt.json').exists()
    {'plan':plan,'run':run,'audit':audit,'report':report,'seal':seal}[args.command]()
