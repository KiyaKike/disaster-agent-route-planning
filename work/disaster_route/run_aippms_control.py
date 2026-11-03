"""Pinned-source audit and bounded corrected ISRS mechanism reference.

No native Julia execution or paper-table claim. New output version;all prior
real-data/perception results remain immutable and are not pooled with simulation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from aippms_control_core import (Layout,SENSORS,ExactOneSense,exact_evaluation,run_episode,
                                 accuracy,posterior,actions,update,manhattan,cost)

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
VENDOR=HERE/'vendor/sbo_aippms_24db0a6'
OUT=ROOT/'output/disaster_route_aippms_control_v1'
REV='24db0a618a98f7c0c9bed470d25cc68fe6dc9b50'
SOURCE_URL='https://github.com/sisl/SBO_AIPPMS/tree/'+REV
SOURCES=['aippms_control_core.py','test_route_aippms_control.py','run_aippms_control.py']
MODELS=('informative','uninformative');BUDGETS=(12,20);SEEDS=tuple(range(2026110300,2026110304))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def open_output():
    if (OUT/'completion_receipt.json').exists():raise ValueError('Sealed output;use new version')


def source_audit():
    records=[];tree=read(VENDOR/'tree.json');lookup={f['path']:f for f in tree['tree'] if f['type']=='blob'}
    for p in VENDOR.rglob('*'):
        if not p.is_file() or p.name=='tree.json':continue
        rel=p.relative_to(VENDOR).as_posix();b=p.read_bytes();git=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert git==lookup[rel]['sha'] and len(b)==lookup[rel]['size'],rel
        records.append({'path':rel,'bytes':len(b),'git_blob_sha1':git,'sha256':sha(p)})
    f='InformationRockSample/AIPPMS/src/envs/InfSearchRockSample.jl';s=(VENDOR/f).read_text()
    start=s.index('function belief_update_location_states_sensor(');end=s.index('## NOTE: mode_density',start);function=s[start:end]
    assert function.count('obs_location_states')==1 # Declaration only,no read in body.
    assert 'for (i, loc) in enumerate(env.location_states)' in function and 'if loc_type == loc' in function
    assert 'normalize(new_distribution)' in function
    lines=s.splitlines();located=lambda token:next(i for i,line in enumerate(lines,1) if token in line)
    common=(VENDOR/'InformationRockSample/AIPPMS/src/envs/common.jl').read_text()
    paths={f['path'] for f in tree['tree']}
    result={'scope':'Static audit of this pinned ISRS source only;not a claim about published paper results or every version.',
      'repo':'sisl/SBO_AIPPMS','commit':REV,'source_url':SOURCE_URL,'files':records,
      'julia_in_path':shutil.which('julia'),'project_or_manifest_present':any(Path(p).name in ('Project.toml','Manifest.toml') for p in paths),
      'findings':[{'severity':'CRITICAL_FOR_FAIR_BASELINE','file':f,'line':located('if loc_type == loc'),
                   'finding':'Sensor belief updater uses hidden env.location_states for likelihood;observation argument occurs in declaration only.',
                   'correction':'Python reference conditions on delivered observation and uses sum-normalized Bayes probabilities.'},
                  {'severity':'MODEL_CONSISTENCY','file':'InformationRockSample/AIPPMS/src/envs/common.jl','line':next(i for i,l in enumerate(common.splitlines(),1) if 'sample_location_states(pomdp.env' in l),
                   'finding':'Sensor generator samples from fixed environment object,rather than the sampled rollout state passed to generate_o.',
                   'correction':'Planning simulation observations are generated from its root-sampled hypothetical world;real environment separately uses actual truth.'}],
      'native_reproduction':'NOT_RUN;no Julia runtime and repository provides no pinned dependency manifest. No global runtime/package installation performed.',
      'port_scope':'Controlled5x5three-rock Python mechanism reference with corrected observation handling;not author table replication.'}
    save(OUT/'upstream_source_audit.json',result)
    (OUT/'UPSTREAM_LICENSE.txt').write_bytes((VENDOR/'LICENSE').read_bytes())
    return result


def plan():
    OUT.mkdir(exist_ok=True);open_output();assert not (OUT/'protocol.json').exists()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_route*.py','-v'],capture_output=True,text=True)
    log=result.stdout+result.stderr;(OUT/'test_log.txt').write_text(log)
    assert result.returncode==0 and 'Ran 113 tests' in log,log[-2000:]
    audit=source_audit()
    inputs={str(VENDOR/f['path']):f['sha256'] for f in audit['files']};inputs[str(VENDOR/'tree.json')]=sha(VENDOR/'tree.json')
    old=[]
    for name in ['disaster_route_loop_v1','disaster_route_factor_v2','disaster_route_snapshot_v3','disaster_route_trigger_v4','disaster_route_dual_map_demo_v1',
                 'disaster_route_real_xbd36_v1','disaster_route_structure_pilot_v1','disaster_route_headroom_v1']:
        p=ROOT/'output'/name/'completion_receipt.json';old.append(p)
    old.append(ROOT/'output/感知检查_阶段汇报_20261028/归档校验.json')
    save(OUT/'protocol.json',{'client_date':'2026-11-02','stage':'CORRECTED_AIPPMS_ISRS_MECHANISM_REFERENCE',
      'native_paper_table_reproduction':False,'source_commit':REV,'source_url':SOURCE_URL,
      'layout':{'grid':[5,5],'home':[0,0],'rocks':[[0,2],[2,4],[3,3]],'beacons':[[0,0],[2,2],[4,4]],
                'selection':'Rock layout is pinned author default5x5three-rock. Fixed3beacons are a declared controlled addition;no outcome tuning.'},
      'worlds':'All8good/bad configurations,equal independent prior0.5;world IDs0..7,no filtering.',
      'sensors':{'near':{'cost':.5,'efficiency':2.5},'far':{'cost':2,'efficiency':10},
                 'informative':'P(correct)=.5*(1+2^(-4*EuclideanDistance/efficiency)) from pinned author ISRS code.',
                 'uninformative':'P(correct)=.5 for each rock regardless of distance;known zero-information negative control.',
                 'footprint':'Paid beacon reading returns noisy bits about all rocks,including unvisited ones;unlike current local-image-only replay.'},
      'budgets':BUDGETS,'primary_budget':20,'cost_units':'Source energy units,not meters or seconds;4-neighbor move1,sensing.5/2,return paid.',
      'reward':'First good-rock visit+10,bad or consumed-rock visit-10,other movement/sensing0. This follows pinned code;not severe-pixel recall or number of rescued people.',
      'policies':['exact_no_sense','exact_one_sense','pomcp_random','pomcp_gcb'],
      'exact_reference':'Exact finite-energy DP,at most0or1sensor action. All hidden worlds and all possible observed sensor outputs evaluated exhaustively. Not an upper bound on unrestricted repeated sensing.',
      'pomcp_reference':{'queries_per_decision':100,'depth':5,'rollout':'Continue feasible GCB-softmax or random actions to early stop/return;GCB uses exact expected mode improvement,paper Eq5.',
                         'uct_scale':10,'discount':1,'tree_reuse':False,'seeds':SEEDS,
                         'differences':'Python POMCP-style reference,not BasicPOMCP.jl execution. UCT10/discount1/early stop at home and exact rather than sampled IG declared;no post-outcome tuning.'},
      'sensor_randomness':'Separate seeded RNG from planner;matched environment seed for each world/condition/budget across POMCP variants. Same nth sensor action uses same random stream,but actions/ranges may differ.',
      'planning_time':'Actual CPU wall time reported separately,not charged as source energy;not deployment timing or identical prior route-time contract.',
      'continuation':'If exact zero-information reference gains value,fail implementation. Informative gain only validates this remote-sensing mechanism,not real disaster noise handling or new method.',
      'source_sha256':{f:sha(HERE/f) for f in SOURCES},'inputs_sha256':inputs,'old_receipt_sha256':{str(p):sha(p) for p in old},
      'new_model_calls':0,'gpu_rental':False,'new_training':False,'real_disaster_efficacy':False})
    (OUT/'source_snapshots').mkdir(exist_ok=True)
    for f in SOURCES:shutil.copyfile(HERE/f,OUT/'source_snapshots'/f)
    print('FROZEN:113tests;corrected paper-based reference;all8worlds;2sensor models;2budgets;256POMCPepisodes',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for path,h in {**p['inputs_sha256'],**p['old_receipt_sha256']}.items():assert sha(path)==h,path
    for f,h in p['source_sha256'].items():assert sha(HERE/f)==h,f
    return p


def run():
    open_output();verify();assert not (OUT/'episodes.json').exists()
    exact_rows=[];episodes=[];stats=[];(OUT/'partial').mkdir(exist_ok=True)
    for model in MODELS:
        layout=Layout(sensor_model=model)
        for budget in BUDGETS:
            key=f'{model}_{budget}';part=OUT/'partial'/f'{key}.json'
            if part.exists():
                d=read(part);assert d['protocol_sha256']==sha(OUT/'protocol.json')
                exact_rows+=d['exact'];episodes+=d['episodes'];stats.append(d['runtime']);continue
            planner=ExactOneSense(layout);local_exact=[];local_episodes=[];start=time.perf_counter()
            for sense in (0,1):
                e=exact_evaluation(layout,budget,sense,planner)
                local_exact.append({'sensor_model':model,'budget':budget,'policy':'exact_one_sense' if sense else 'exact_no_sense',**e})
            exact_wall=time.perf_counter()-start
            del planner
            start=time.perf_counter()
            for world_id,truth in enumerate(itertools.product((False,True),repeat=3)):
                for seed in SEEDS:
                    env_seed=seed+world_id*1000+budget*100000
                    for policy in ('pomcp_random','pomcp_gcb'):
                        e=run_episode(layout,truth,budget,policy,env_seed,seed+world_id*10000)
                        local_episodes.append({'sensor_model':model,'budget':budget,'world_id':world_id,'truth':list(truth),
                            'environment_seed':env_seed,'planner_seed':seed+world_id*10000,**e})
            runtime={'sensor_model':model,'budget':budget,'exact_wall_s':exact_wall,'pomcp_wall_s':time.perf_counter()-start}
            save(part,{'protocol_sha256':sha(OUT/'protocol.json'),'exact':local_exact,'episodes':local_episodes,'runtime':runtime})
            exact_rows+=local_exact;episodes+=local_episodes;stats.append(runtime)
            print(json.dumps({'condition':key,'exact_values':[(r['policy'],r['expected_reward']) for r in local_exact],
                             'pomcp_episodes':len(local_episodes),'wall':runtime}),flush=True)
    save(OUT/'exact_evaluations.json',exact_rows);save(OUT/'episodes.json',episodes);save(OUT/'runtime.json',stats)
    print('REFERENCE_COMPLETE',len(episodes),'POMCPepisodes',len(exact_rows),'exact condition-policy evaluations',flush=True)


def audit():
    open_output();verify();exact=read(OUT/'exact_evaluations.json');episodes=read(OUT/'episodes.json');branches=steps=0
    for e in exact:
        total=sum(b['probability'] for b in e['branches']);assert abs(total-1)<1e-9
        assert abs(sum(b['probability']*b['reward'] for b in e['branches'])-e['bellman_value'])<1e-8
        assert all(b['returned'] and b['energy_used']<=e['budget']+1e-9 for b in e['branches'])
        if e['sensor_model']=='uninformative':assert e['expected_reward']==0 and e['sense_probability']==0
        branches+=len(e['branches'])
    for e in episodes:
        layout=Layout(sensor_model=e['sensor_model']);pos=layout.home;left=e['budget'];consumed=0;prior=(.5,)*3
        truth=e['truth'];reward=0.;senses=moves=0
        for h in e['history']:
            a=(h['action'][0],tuple(h['action'][1]) if h['action'][0]=='move' else h['action'][1])
            assert h['before']=={'position':list(pos),'remaining_energy':left,'consumed_mask':consumed,'belief':list(prior)}
            assert a in actions(layout,pos,int(round(left*2)))
            left-=cost(a)/2
            if a[0]=='move':
                pos=a[1];r=0.
                if pos in layout.rocks:
                    j=layout.rocks.index(pos);r=-10. if consumed&(1<<j) or not truth[j] else 10.;consumed|=1<<j
                prior=tuple(0. if consumed&(1<<j) else p for j,p in enumerate(prior));moves+=1
            else:
                # Independently recompute Bayes likelihoods from observed bits.
                sensor=a[1];q=accuracy(layout,pos,sensor);obs=h['observation'];post=[]
                for j,p in enumerate(prior):
                    if consumed&(1<<j):post.append(0.);continue
                    yes=p*(q[j] if obs[j] else 1-q[j]);no=(1-p)*(1-q[j] if obs[j] else q[j]);post.append(yes/(yes+no))
                prior=tuple(post);r=0.;senses+=1
            assert h['reward']==r and h['after_position']==list(pos) and h['after_remaining_energy']==left
            assert all(abs(a-b)<1e-12 for a,b in zip(h['after_belief'],prior))
            assert left>=manhattan(pos,layout.home);reward+=r;steps+=1
        assert pos==layout.home and reward==e['total_reward'] and senses==e['senses'] and moves==e['moves']
        assert e['planning_simulations']==100*(len(e['history'])+1)
        # Full deterministic replay checks planner and private sensor RNG consistency.
        replay=run_episode(layout,truth,e['budget'],e['policy'],e['environment_seed'],e['planner_seed'])
        assert replay=={k:e[k] for k in replay}
    assert len(episodes)==256
    save(OUT/'audit.json',{'status':'PASS','pomcp_episodes_replayed':len(episodes),'paid_actions_replayed':steps,
                         'exact_branches_checked':branches,'budget_or_return_violations':0,'zero_information_exact_gain':0,
                         'native_julia_execution':False,'paper_table_replication':False,'new_model_calls':0,'real_disaster_efficacy':False})
    print(json.dumps(read(OUT/'audit.json')),flush=True)


def report():
    open_output();verify();assert read(OUT/'audit.json')['status']=='PASS'
    exact=read(OUT/'exact_evaluations.json');episodes=read(OUT/'episodes.json');summary=[]
    for model in MODELS:
        for b in BUDGETS:
            for policy in ('exact_no_sense','exact_one_sense','pomcp_random','pomcp_gcb'):
                if policy.startswith('exact'):
                    r=next(e for e in exact if e['sensor_model']==model and e['budget']==b and e['policy']==policy)
                    mean=r['expected_reward'];energy=r['expected_energy'];sensing=r['sense_probability'];n=len(r['branches'])
                    role='Exhaustive expectation,branches dependent;not independent samples.'
                else:
                    rs=[r for r in episodes if r['sensor_model']==model and r['budget']==b and r['policy']==policy]
                    mean=sum(r['total_reward'] for r in rs)/len(rs);energy=sum(r['energy_used'] for r in rs)/len(rs)
                    sensing=sum(r['senses'] for r in rs)/len(rs);n=len(rs);role='8worlds x4seed runs;not32independent maps.'
                summary.append({'sensor_model':model,'budget':b,'policy':policy,'expected_or_sample_mean_reward':mean,
                                'mean_energy':energy,'sensing_probability_or_mean_calls':sensing,'records':n,'record_role':role})
    save(OUT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    policies=('exact_no_sense','exact_one_sense','pomcp_random','pomcp_gcb');labels=('Exact no sensing','Exact <=1 sensing','POMCP random','POMCP GCB')
    for ax,model in zip(axes,MODELS):
        for k,(p,label) in enumerate(zip(policies,labels)):
            rr=[next(r for r in summary if r['sensor_model']==model and r['budget']==b and r['policy']==p) for b in BUDGETS]
            ax.bar(np.arange(2)+(k-1.5)*.19,[r['expected_or_sample_mean_reward'] for r in rr],.19,label=label)
        ax.axhline(0,color='#334155',lw=.8);ax.set_title(model+' sensor');ax.set_xticks([0,1],[str(b) for b in BUDGETS]);ax.set_xlabel('Energy budget (source units)');ax.set_ylabel('Task reward (+10 / -10)');ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Corrected paper-based ISRS mechanism reference | not native Julia/table replication')
    (OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/control_results.png',dpi=160);plt.close(fig)
    lines=['# AIPPMS 信息价值受控验证','',
      '2026年11月2日。本轮完成固定作者源码审计及修正后的小规模Python参考实现。**没有运行作者Julia环境，没有复现论文原表格，也不是灾后真实数据效果实验。**','',
      '## 为什么没有原样运行仓库','',
      f'已固定[作者仓库]({SOURCE_URL})的提交`{REV}`。本机没有Julia，仓库没有Project.toml/Manifest.toml，脚本还含作者本机绝对路径。未安装全局运行时或大量依赖。更关键的是，ISRS信念更新在接收观测后，似然判断使用隐藏的`env.location_states`，而传入观测仅出现在函数参数中。该版本不能原样作为信息公平基线；这是固定源码的静态审计结论，不据此判断论文发表结果。具体位置和修正见`upstream_source_audit.json`。','',
      '## 实际验证的机制','',
      '沿用源码默认5×5三岩石布局、近/远传感成本0.5/2及距离衰减公式，明确增加三个固定信标，并保留全部8种隐藏好坏配置。移动消耗1能源单位，传感只在信标进行，返回起点也消耗预算；预算12/20，20为主预算。首次好岩石+10，坏或已消耗岩石-10，遵循该版本源码，不是损伤像素召回或找到人员数。','',
      '公开状态仅含位置、预算、已消耗对象与观测后验；真实状态仅由环境执行器持有。贝叶斯更新使用实际收到的观测并按总概率归一化；规划模拟从当前信念抽取假想世界，再从该假想世界生成观测。','',
      '设置有信息传感器与已知正确率0.5的无信息传感器。无感知/最多一次感知采用精确有限预算动态规划，并穷举真实世界和全部观测分支核对期望值；最多一次感知并不是可重复感知策略的全局上界。POMCP参考采用每次100模拟、树深5、随机/GCB rollout；UCT尺度10、折扣1、允许回家提前结束、精确期望模式增益均为明确实现选择，不是作者库逐字移植，未根据结果调参。','',
      '## 结果','',
      '| 传感模型 | 预算 | 策略 | 精确期望或样本平均奖励 | 平均消耗能源 | 感知概率或平均调用 |','|---|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['sensor_model']} | {r['budget']} | {r['policy']} | {r['expected_or_sample_mean_reward']:.4f} | {r['mean_energy']:.3f} | {r['sensing_probability_or_mean_calls']:.3f} |")
    lines += ['',
      '精确结果为全分支加权期望；POMCP是每条件8种世界×4种固定种子的32次运行，不能写成32个独立场景，也不能和精确分支数作样本量比较。有限模拟的POMCP可能低于受限精确参考，甚至对无信息观测付费，不据此修改搜索深度或择优报告。','',
      '## 对灾前灾后路线研究的意义','',
      '这个任务中，信标观测直接覆盖尚未访问的岩石，因此即使岩石相互独立，也能帮助后续选点。信息价值不要求任意相邻区域共享损伤状态。它与当前256像素局部缓存任务的关键差别是观测范围：当前任务只在已到达格获得当地预测，尚未建立对其他候选点的可靠信息通道。','',
      '本轮仅验证已知观测模型下的行动机制。下一步先核实灾害任务中的真实信息通道：同一次可用观测能否覆盖未访问候选点，或者在独立场景中是否存在可验证的语义关联。没有证据时不扩展同邻格传播；没有真实新视角时不把重复缓存当作复核。','',
      '双层地图继续保存结构先验与已付证据，但收益由观测与决策机制检验；模拟奖励不并入旧真实影像结果。无需GPU或新训练。','',
      '![受控结果](figures/control_results.png)','',
      '[固定协议](protocol.json) · [费用与重放核验](audit.json) · [源代码审计](upstream_source_audit.json) · [完整结果](summary.json)','',
      '来源：[AIPPMS论文](https://arxiv.org/abs/2003.09746)、[SBO-AIPPMS论文](https://arxiv.org/abs/2209.07660)。作者源码MIT许可见`UPSTREAM_LICENSE.txt`。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'next_scope.json',{'status':'NOT_EXECUTED','next':'Identify and test actual disaster-task observation footprint or independent semantic association before new online update method.','native_replication_remaining':True,'real_data_information_value_established':False,'new_training':False,'gpu_rental':False})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def seal():
    open_output();verify();assert read(OUT/'audit.json')['status']=='PASS';assert read(OUT/'visual_inspection.json')['status']=='PASS'
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion_receipt.json'}
    save(OUT/'completion_receipt.json',{'status':'COMPLETE_CORRECTED_MECHANISM_REFERENCE_ONLY','files':files,
                                     'native_julia_execution':False,'paper_table_replication':False,'new_model_calls':0,'gpu_rental':False,'real_disaster_efficacy':False})
    print('SEALED',len(files),'files',sha(OUT/'completion_receipt.json'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plan','run','audit','report','seal']);a=ap.parse_args();globals()[a.command]()
