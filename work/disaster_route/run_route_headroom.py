"""Frozen diagnostic of solver/prior/perception headroom,not a new method.

Public exact routing uses only existing coarse-image utility. The two
hindsight routes explicitly use evaluator knowledge and are never deployed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from route_headroom_core import exact_catalogue,solve_catalogue,subset_rewards,audit_bellman
from real_xbd_route_core import xy_grid,BUDGETS,HOME

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
SOURCE=ROOT/'output/disaster_route_real_xbd36_v1'
OUT=ROOT/'output/disaster_route_headroom_v1'
SOURCES=['route_headroom_core.py','test_route_headroom.py','run_route_headroom.py']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def open_output():
    if (OUT/'completion_receipt.json').exists():raise ValueError('Sealed diagnostic; use new version')


def verify():
    p=read(OUT/'protocol.json')
    for path,h in p['inputs_sha256'].items():assert sha(path)==h,path
    for f,h in p['sources_sha256'].items():assert sha(HERE/f)==h,f
    return p


def plan():
    OUT.mkdir(exist_ok=True);open_output()
    assert not (OUT/'protocol.json').exists()
    for rel,h in read(SOURCE/'completion_receipt.json')['files'].items():assert sha(SOURCE/rel)==h,rel
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_route*.py','-v'],capture_output=True,text=True)
    text=result.stdout+result.stderr;(OUT/'test_log.txt').write_text(text)
    assert result.returncode==0 and 'Ran 99 tests' in text,text[-2000:]
    inputs=[SOURCE/n for n in ['completion_receipt.json','protocol.json','public_features.json','private_tile_scores.json','metrics.json','routes.json']]
    inputs.append(HERE/'real_xbd_route_core.py')
    save(OUT/'protocol.json',{'client_date':'2026-11-02','stage':'DECISIVE_HEADROOM_DIAGNOSTIC_ON_REUSED36',
        'selection':'All36 original real-route development cases,three events12each;no outcome-based filtering or retuning.',
        'budgets_s_proxy':BUDGETS,'primary_budget_s_proxy':240,
        'geometry':'Unchanged4x4image grid,16native256tiles,home16bottom-middle;no physical-flight claim.',
        'cost':'20proxy units per grid width+20per observed tile+2fixed planning units(pre1+post1);closed return charged.',
        'exact_solver':'Held-Karp minimum closed distance for every one of65536subsets;choose reward-maximizing feasible subset. Ties shortest distance then lowest mask. Exact under this fixed additive16-node model.',
        'routes':{'public_exact':'Maximize unchanged post_utility;no labels or unseen predictions. Engineering stronger static control.',
                  'reference_coverage_oracle':'Maximize all severe reference pixels in visited disjoint tiles;perfect knowledge of published labels,evaluator only.',
                  'detector_tp_oracle':'Maximize actual fixed-detector TP pixels in visited tiles;perfect knowledge of predictions+reference,evaluator only.'},
        'interpretation':'Oracle headroom establishes what is possible with complete hindsight,not how much online observation can attain. Public-utility optimality does not imply reference-score optimality. No claimed causal decomposition or new method.',
        'timing':'Keep prior synthetic fixed solve charge2 for comparability. Catalogue actual wall time reported separately,shared across cases;not measured on-board runtime or proof of real-time operation.',
        'zero_cases':'All retained;recall null if no reference;zero model opportunity kept as zero when reference exists.',
        'continuation_rule':'If hindsight gain exists,check actual observable signal usefulness next. Do not infer positive online-information value from hindsight gap alone.',
        'inputs_sha256':{str(p):sha(p) for p in inputs},'sources_sha256':{f:sha(HERE/f) for f in SOURCES},
        'new_model_calls':0,'gpu_rental':False,'new_training':False,'new_independent_confirmation':False})
    (OUT/'source_snapshots').mkdir(exist_ok=True)
    for f in SOURCES:shutil.copyfile(HERE/f,OUT/'source_snapshots'/f)
    print('FROZEN:99 tests;unchanged36;three budgets;public exact plus two evaluator-only oracles',flush=True)


def catalogue():
    p=OUT/'exact_catalogue.npz'
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def run():
    open_output();verify();assert not (OUT/'solutions.json').exists()
    start=time.perf_counter();cat=exact_catalogue(xy_grid(),HOME);wall=time.perf_counter()-start
    np.savez_compressed(OUT/'exact_catalogue.npz',**cat)
    fs=read(SOURCE/'public_features.json');solutions=[]
    # All public plans are fixed before loading evaluator score values.
    for f in fs:
        for b in BUDGETS:
            s=solve_catalogue(cat,f['post_utility'],b)
            solutions.append({'sample_id':f['sample_id'],'event':f['event'],'budget_s_proxy':b,'kind':'public_exact',**s})
    save(OUT/'public_exact_plans_before_private_scores.json',solutions)
    private=read(SOURCE/'private_tile_scores.json')
    for f in fs:
        cs=private[f['sample_id']]
        for b in BUDGETS:
            for kind,key in [('reference_coverage_oracle','reference'),('detector_tp_oracle','tp')]:
                s=solve_catalogue(cat,[r[key] for r in cs],b)
                solutions.append({'sample_id':f['sample_id'],'event':f['event'],'budget_s_proxy':b,'kind':kind,**s})
    save(OUT/'solutions.json',solutions)
    save(OUT/'runtime.json',{'catalogue_wall_s':wall,'catalogue_shared_across_cases':True,'catalogue_subsets':len(cat['closed']),
        'open_path_states':int(HOME*(1<<(HOME-1))),'new_model_calls':0,'note':'Actual offline wall time is not substituted for fixed legacy planning charge.'})
    print('SOLVED',len(solutions),'routes;catalogue wall seconds',round(wall,3),flush=True)


def scores(route,cs):
    visited=route['route'];total=sum(c['reference'] for c in cs);covered=sum(cs[i]['reference'] for i in visited);tp=sum(cs[i]['tp'] for i in visited)
    return {'reference':total,'covered_reference':covered,'tp':tp,'fp':sum(cs[i]['fp'] for i in visited),
            'coverage':covered/total if total else None,'recall':tp/total if total else None,
            'all_tile_tp':sum(c['tp'] for c in cs)}


def audit():
    open_output();verify();cat=catalogue();states=audit_bellman(cat)
    fs={r['sample_id']:r for r in read(SOURCE/'public_features.json')};private=read(SOURCE/'private_tile_scores.json')
    ss=read(OUT/'solutions.json');ms=read(SOURCE/'metrics.json');old=read(SOURCE/'routes.json')
    table=[]
    for s in ss:
        cs=private[s['sample_id']];route=s['route'];mask=s['subset_mask'];assert sum(1<<i for i in route)==mask
        assert len(route)==len(set(route)) and len(route)==s['observations']
        xy=xy_grid();nodes=[HOME,*route,HOME]
        dist=sum(float(np.hypot(*(xy[a]-xy[b]))) for a,b in zip(nodes,nodes[1:]))
        assert abs(dist-s['distance_grid'])<1e-9
        assert abs(dist*20+len(route)*20+2-s['elapsed_s_proxy'])<1e-8
        assert s['elapsed_s_proxy']<=s['budget_s_proxy']+1e-8
        weights=fs[s['sample_id']]['post_utility'] if s['kind']=='public_exact' else [c['reference' if s['kind']=='reference_coverage_oracle' else 'tp'] for c in cs]
        # Independent per-node vector sum over masks validates global reward maximum.
        values=np.zeros(len(cat['closed']));masks=np.arange(len(values))
        for j,w in enumerate(weights):values[(masks&(1<<j))!=0]+=w
        feasible=cat['closed']*20+cat['cardinality']*20+2<=s['budget_s_proxy']+1e-8
        assert abs(values[feasible].max()-s['objective'])<1e-9
        assert abs(sum(weights[i] for i in route)-s['objective'])<1e-9
        baseline=next(m for m in ms if m['sample_id']==s['sample_id'] and m['budget_s_proxy']==s['budget_s_proxy'] and m['policy']=='post_static')
        baseline_route=next(r['state']['visited'] for r in old if r['sample_id']==s['sample_id'] and r['budget_s_proxy']==s['budget_s_proxy'] and r['policy']=='post_static')
        if s['kind']=='public_exact':assert s['objective']+1e-9>=sum(weights[i] for i in baseline_route)
        elif s['kind']=='reference_coverage_oracle':assert s['objective']>=baseline['covered_reference_pixels']
        else:assert s['objective']>=baseline['true_positive_pixels']
        table.append({**s,**scores(s,cs)})
    for sid in fs:
        for kind in ('public_exact','reference_coverage_oracle','detector_tp_oracle'):
            vals=[next(s['objective'] for s in ss if s['sample_id']==sid and s['kind']==kind and s['budget_s_proxy']==b) for b in BUDGETS]
            assert all(a<=b+1e-9 for a,b in zip(vals,vals[1:]))
    assert len(ss)==324
    save(OUT/'scored_solutions.json',table)
    save(OUT/'audit.json',{'status':'PASS','routes_checked':len(ss),'valid_open_path_states_bellman_checked':states,
        'all_subsets_evaluated':len(cat['closed']),'budget_violations':0,'oracle_below_matched_baseline':0,
        'public_exact_plans':108,'hindsight_oracle_plans':216,'private_labels_used_in_public_planner':False,
        'new_model_calls':0,'new_independent_confirmation':False})
    print(json.dumps(read(OUT/'audit.json')),flush=True)


def report():
    open_output();verify();assert read(OUT/'audit.json')['status']=='PASS'
    rows=read(OUT/'scored_solutions.json');legacy=read(SOURCE/'metrics.json');fs=read(SOURCE/'public_features.json')
    summary=[]
    for event in sorted({r['event'] for r in fs}):
        for b in BUDGETS:
            baseline=[r for r in legacy if r['event']==event and r['budget_s_proxy']==b and r['policy']=='post_static']
            total=sum(r['reference_severe_pixels'] for r in baseline)
            entry={'event':event,'budget_s_proxy':b,'cases':12,'reference':total,
                   'positive_cases':sum(r['reference_severe_pixels']>0 for r in baseline),
                   'legacy_post_coverage':sum(r['covered_reference_pixels'] for r in baseline)/total if total else None,
                   'legacy_post_recall':sum(r['true_positive_pixels'] for r in baseline)/total if total else None}
            for kind in ('public_exact','reference_coverage_oracle','detector_tp_oracle'):
                group=[r for r in rows if r['event']==event and r['budget_s_proxy']==b and r['kind']==kind]
                assert len(group)==12
                entry[kind+'_coverage']=sum(r['covered_reference'] for r in group)/total if total else None
                entry[kind+'_recall']=sum(r['tp'] for r in group)/total if total else None
            entry['all_tile_detector_recall']=sum(r['all_tiles_detector_tp_ceiling'] for r in baseline)/total if total else None
            entry['routing_tp_headroom_pp']=100*(entry['detector_tp_oracle_recall']-entry['legacy_post_recall']) if total else None
            entry['coverage_headroom_pp']=100*(entry['reference_coverage_oracle_coverage']-entry['legacy_post_coverage']) if total else None
            summary.append(entry)
    save(OUT/'event_summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=[r for r in summary if r['budget_s_proxy']==240]
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True);x=np.arange(3);width=.23
    for ax,fields,title in [(axes[0],[('legacy_post_coverage','Existing post-static'),('public_exact_coverage','Exact same public utility'),('reference_coverage_oracle_coverage','Hindsight coverage oracle')],'Covered severe reference'),
                            (axes[1],[('legacy_post_recall','Existing post-static'),('public_exact_recall','Exact same public utility'),('detector_tp_oracle_recall','Hindsight detector-TP oracle')],'Correctly detected severe reference')]:
        for k,(field,label) in enumerate(fields):ax.bar(x+(k-1)*width,[r[field]*100 for r in primary],width,label=label)
        ax.set_xticks(x,['Nepal flood','Sunda tsunami','Tuscaloosa tornado']);ax.set_ylim(0,105);ax.set_ylabel('Published reference pixels (%)');ax.set_title(title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same240 proxy budget | all36 development scenes | hindsight is a diagnostic,not a deployable method')
    (OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/headroom.png',dpi=160);plt.close(fig)
    pct=lambda v:'null' if v is None else f'{100*v:.2f}%'
    lines=['# 现有真实影像路线的提升空间诊断','',
        '2026年11月2日。目的：先判断当前预算路线还有多大改善空间，再决定是否值得开发新的在线方法。沿用全部36对开发影像，不新增数据、模型调用或GPU支出。该诊断不证明双层地图或在线观测有效。','',
        '16格任务可用动态规划穷举全部65,536个访问子集的最短闭合路线。所有对照仍计移动、观察、返程及灾前1＋灾后1代理规划费用；主预算240，另保留120/360。最短路线与最优子集在这一固定小规模加性模型内可核验，不能推广为现场最优航线。','',
        '| 事件 | 原灾后静态正确召回 | 同公开线索精确求解 | 同预算检出Oracle | 全格模型检出上限 | 原参考覆盖 → 覆盖Oracle |',
        '|---|---:|---:|---:|---:|---:|']
    for r in primary:
        lines.append(f"| {r['event']} | {pct(r['legacy_post_recall'])} | {pct(r['public_exact_recall'])} | {pct(r['detector_tp_oracle_recall'])} | {pct(r['all_tile_detector_recall'])} | {pct(r['legacy_post_coverage'])} → {pct(r['reference_coverage_oracle_coverage'])} |")
    lines += ['',
        '公开线索精确求解只看原64像素粗图生成的效用，不看标签或未访问预测。它优化效用分数，并不保证提高真实参考召回。两个Oracle分别提前知道全部严重参考，或全部实际模型真阳性；仅用于事后诊断，始终与Agent策略分开。','',
        '覆盖Oracle回答预算下最多能访问多少参考损伤面积；检出Oracle回答固定感知器下最多能正确检出多少参考像素。二者目标不同，路线可不同。全格模型上限取消了观察预算，不能与同预算Oracle混称。','',
        '有Oracle差距仅说明理想选点可能改善，不能说明现有沿途信号足以识别这些更好的点。继续条件是再检验实际观测对未访问区域的预测或行动价值，而非恢复已停止的邻域/同建筑调参。','',
        '规划求解的实际CPU耗时单独见runtime.json；收费仍使用旧代理约定，不是实时性测试。三事件阳性少且数据反复用于开发，不作泛化或显著性结论。','',
        '[图表](figures/headroom.png) · [核验](audit.json) · [逐案例评分](scored_solutions.json) · [固定协议](protocol.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(primary,ensure_ascii=False),flush=True)


def seal():
    open_output();verify();assert read(OUT/'audit.json')['status']=='PASS';assert read(OUT/'visual_inspection.json')['status']=='PASS'
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion_receipt.json'}
    save(OUT/'completion_receipt.json',{'status':'COMPLETE_DIAGNOSTIC_ONLY','files':files,'new_model_calls':0,'gpu_rental':False,
                                     'new_method_established':False,'online_information_value_established':False})
    print('SEALED',len(files),'files',sha(OUT/'completion_receipt.json'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plan','run','audit','report','seal']);a=ap.parse_args();globals()[a.command]()
