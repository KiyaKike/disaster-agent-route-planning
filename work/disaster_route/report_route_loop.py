"""Scientific route figures, teacher-readable findings and completion receipt."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from run_route_loop import ROOT,HERE,OUT,read,digest,checked,save,verify

COLORS={'pre_static':'#6B7280','post_static':'#2563EB','online':'#B45309','online_frozen':'#8B5CF6'}
LABELS={'pre_static':'Pre static','post_static':'Post static','online':'Online feedback','online_frozen':'Replan no update'}
REGION={'口岸':'Port','南侧建筑区':'South'}
MODES={'aligned':'Aligned cue','reversed':'Reversed cue','independent':'Uninformative cue'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify();assert read(OUT/'audit_report.json')['status']=='PASS'
    means=read(OUT/'condition_means.json');paired=read(OUT/'paired_comparisons.json');public=read(OUT/'public_geometry.json');worlds=read(OUT/'synthetic_worlds.json')
    routes=[json.loads(line) for line in (OUT/'routes.jsonl').open()];runtime=read(OUT/'runtime.json');summary=read(OUT/'summary.json')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    # Predeclared first-seed moderate-noise example in each region, no best-case pick.
    for region,g in public.items():
        fig,axes=plt.subplots(1,3,figsize=(12,4.6),constrained_layout=True);xy=np.array(g['xy_m']);offset=xy.min(0);positions=xy-offset
        world=next(w for w in worlds if w['region']==region and w['mode']=='aligned' and w['seed']==2026102800)
        target=np.array([sid in set(world['target_ids']) for sid in g['ids']])
        for ax,policy in zip(axes,['pre_static','post_static','online']):
            r=next(r for r in routes if r['world_id']==world['world_id'] and r['noise']=='moderate' and r['budget_s']==1200 and r['policy']==policy)
            ax.scatter(positions[:,0],positions[:,1],s=3,c='#DDDDDD',label='Candidate cells')
            ax.scatter(positions[target,0],positions[target,1],s=12,c='#C74440',alpha=.6,label='Simulated targets')
            lookup={sid:i for i,sid in enumerate(g['ids'])};path=positions[[lookup[sid] for sid in r['closed_route_ids']]]
            ax.plot(path[:,0],path[:,1],'-',c=COLORS[policy],linewidth=1.3,alpha=.9)
            home=positions[lookup[g['start_id']]];ax.scatter(*home,marker='*',s=140,c='#111111',zorder=8,label='Start and return')
            true_indices=[s['cell_index'] for s in r['steps'] if target[s['cell_index']] and s['observed_positive']]
            if true_indices:ax.scatter(positions[true_indices,0],positions[true_indices,1],s=34,facecolors='none',edgecolors='#15803D',linewidths=1.2,label='Correct detections')
            ax.set(title=f"{LABELS[policy]}\n{r['evaluation']['detected_target_cells']} detections | {r['total_time_s']:.0f}s | {r['total_distance_m']:.0f}m",xlabel='Local easting (m)')
            ax.set_aspect('equal');ax.grid(alpha=.15)
        axes[0].set_ylabel('Local northing (m)');axes[0].legend(fontsize=7,loc='upper left')
        fig.suptitle(f'SIMULATION on {REGION[region]} geometry | fixed first seed | aligned prior | sensitivity .8, FPR .1')
        fig.savefig(figures/f'route_example_{REGION[region].lower()}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharey='row',constrained_layout=True)
    for row,(region,g) in enumerate(public.items()):
        for col,mode in enumerate(['aligned','reversed','independent']):
            ax=axes[row,col]
            for policy in ['pre_static','post_static','online','online_frozen']:
                rs=sorted([r for r in means if r['region']==region and r['mode']==mode and r['noise']=='moderate' and r['policy']==policy],key=lambda r:r['budget_s'])
                ax.plot([r['budget_s']/60 for r in rs],[r['detected_target_cells'] for r in rs],'-o',label=LABELS[policy],c=COLORS[policy],linewidth=1.4,markersize=4)
            ax.set_title(f'{REGION[region]} | {MODES[mode]}');ax.set_xlabel('Total budget including return (min)');ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Mean detected synthetic target cells')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('SIMULATION | paired10 seeds | fixed moderate observation noise | all travel + inspection + return charged')
    fig.savefig(figures/'budget_comparison.png',dpi=170);fig.savefig(figures/'budget_comparison.pdf');plt.close(fig)
    # Include all18 primary region/prior/noise conditions, including losses.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True,sharex=True)
    for ax,region in zip(axes,public):
        group=[r for r in paired if r['region']==region and r['budget_s']==1200 and r['comparator']=='post_static']
        names=[MODES[r['mode']]+' / '+r['noise'] for r in group];values=[r['mean_detected_delta'] for r in group]
        y=np.arange(len(group));ax.barh(y,values,color=['#15803D' if v>0 else '#B91C1C' if v<0 else '#777777' for v in values])
        ax.set_yticks(y,names,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='#555555',linewidth=.8);ax.grid(axis='x',alpha=.2)
        ax.set_title(REGION[region]);ax.set_xlabel('Paired mean online - post static detections')
    fig.suptitle('SIMULATION | 20min budget | every fixed prior/noise condition retained')
    fig.savefig(figures/'all_primary_deltas.png',dpi=170);plt.close(fig)
    # Real geometry output: two static routes, no true/synthetic target display.
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),constrained_layout=True)
    for ax,(region,g) in zip(axes,public.items()):
        xy=np.array(g['xy_m']);xy-=xy.min(0);lookup={sid:i for i,sid in enumerate(g['ids'])}
        cue=np.array([v if v is not None else np.nan for v in g['coarse_change_cue']]);known=np.isfinite(cue)
        ax.scatter(xy[~known,0],xy[~known,1],s=5,c='#DDDDDD');ax.scatter(xy[known,0],xy[known,1],s=6,c=cue[known],cmap='Greys',vmin=0,vmax=1,alpha=.55)
        for r in read(OUT/'PS_geometry_static_routes.json'):
            if r['region']!=region:continue
            path=xy[[lookup[sid] for sid in r['closed_route_ids']]];ax.plot(path[:,0],path[:,1],c=COLORS[r['policy']],linewidth=1.5,label=LABELS[r['policy']])
        home=xy[lookup[g['start_id']]];ax.scatter(*home,marker='*',s=110,c='#111111');ax.set_aspect('equal');ax.grid(alpha=.2)
        ax.set(title=REGION[region],xlabel='Local easting (m)',ylabel='Local northing (m)');ax.legend(fontsize=8)
    fig.suptitle('PS GEOMETRY ONLY | closed planar routes | no real target detection or flight validity claim')
    fig.savefig(figures/'PS_geometry_routes.png',dpi=170);plt.close(fig)
    primary=[r for r in means if r['budget_s']==1200 and r['noise']=='moderate' and r['policy']!='online_frozen']
    table='\n'.join(f"| {r['region']} | {MODES[r['mode']]} | {r['policy']} | {r['detected_target_cells']:.2f} | {r['first_detection_or_budget_s']:.1f} | {r['no_detection_rate']:.0%} | {r['total_distance_m']:.0f} | {r['total_time_s']:.1f} |" for r in primary)
    delta=summary['primary_condition_counts'];totalsteps=read(OUT/'audit_report.json')['paid_observation_steps_checked']
    text=f'''# 灾前灾后巡查路线闭环第一轮实验

2026-10-28。已把已有PS静态演示接成含移动、观察、返程和规划延迟的闭环实验；感知检查归档保留，本轮没有继续追识别精度。**路线执行与预算核验已完成，目标发现效果目前只在清楚标记的合成仿真中测量。**

## 主线与三种路线

1. 灾前静态：只用几何和均匀热点先验形成固定巡查路线。这是最小灾前基线，尚未接入真实建筑暴露/人口图，不能称为完整灾前风险规划。
2. 灾后一次性：读取已有PS变化线索构成的固定先验，用同一个求解器一次性生成闭合路线；沿途记录观测，但不改变路径。
3. 在线重规划：初始信息和灾后一次性相同，每次付费观察后按固定噪声模型更新当前空间块信念，重算剩余可行路线并走第一步。

另保留“重算但不更新信念”控制，用来区分观测更新和单纯重算/额外规划成本。在线不是保证更优的新方法；这里是一个可检验的基线。核心求解器统一为“期望目标概率/移动加观察时间”的贪心启发式，不宣称路线全局最优。

## 路线成本与可行性

口岸、南侧建筑区分开运行，不跨两裁切之间的空隙连线。所有方法共享每区同一起终点、5m/s、每格20s、最多30次观察。时间预算600/1200/1800s，主预算1200s。每次选择都为回到起点预留时间，返程包含在最终总距离和耗时中。

规划延迟固定为仿真1s/次：静态付一次，在线每次执行重规划付一次；它是实验假设，不是无人机上的实测延迟。求解器实际CPU耗时另报，不混入仿真1s。总时钟等于移动＋观察＋已付规划延迟，全部记录在逐步轨迹中。

本轮2160条仿真路线预算均可行；独立核对{totalsteps}次付费观察及全部返程。在线规划接口只接收公开几何、当前信念和已观测的一位反馈，不能查询未访问目标或未来观测。静态策略也获得同一模拟观测流，但按定义不用于改路。

## 感知固定与仿真边界

真实感知器保留既有Strong Baseline事件划分检查点，不更换权重、不训练、不新增图像推理。路线仿真采用固定模拟观测器，灵敏度/误报率分别为理想1/0、中等.8/.1、较差.6/.25。它们不是从遥感像素精度倒推的无人机性能，也不是新的实测感知成绩。

模拟目标格与观测存于单独的 `synthetic_worlds.json`；没有写入既有PS真实目标空表。3×3网格共享隐藏热点状态，用于构造有空间相关性的合成环境。灾后变化分数仅通过预先固定的块排名构成.05–.35热点概率，未知块.2；**这个变换仅用于仿真，不表示变化分数已经校准为真实灾损概率。**

先验正确、先验顺序反向、先验无信息三个环境各10个固定种子，两个区域共60个目标场。每个目标场的目标、观测随机数在方法、预算和噪声条件间共享。模型正确且似然已知的条件本来更利于Bayes更新，所以不能只展示它。反向和无信息条件、所有失败与无发现轨迹均保留；10种子并不是10次真实灾害复验。

## 同预算的结果

主预算1200s下，18个区域/先验/噪声条件的在线减灾后静态平均发现数：正向{delta['positive']}个、相同{delta['zero']}个、负向{delta['negative']}个。它是全部仿真条件的计数，不是显著性检验或跨灾害效果证明。

![主预算所有条件](figures/all_primary_deltas.png)

下表显示中等噪声、1200s的三种主路线，所有区域和先验条件都保留。发现单位为合成目标格；首次正确发现时间在未发现时原值为null，并明确计入失败率。表中的“首次发现或预算”对失败填预算1200s作为受限时间指标，越小越好；不能把失败填0或只比较成功者的时间。

| 区域 | 仿真先验 | 路线 | 平均发现目标格 | 首次发现或预算 秒 | 无发现率 | 往返米 | 总耗时秒 |
|---|---|---|---:|---:|---:|---:|---:|
{table}

![预算曲线](figures/budget_comparison.png)

预先固定的示例是每区首种子、先验正确、中等噪声、1200s；没有挑选最好例子。下图目标全部为合成标签，只借用了PS网格几何，不应当成该地区真实灾损图。

![口岸仿真路线](figures/route_example_port.png)
![南侧仿真路线](figures/route_example_south.png)

## 真实PS几何案例

![PS几何静态路线](figures/PS_geometry_routes.png)

真实案例只导出灾前均匀静态与灾后线索静态的闭合几何路线，以及距离、耗时和未校准变化线索的访问汇总。没有真实观测反馈就不伪造一条“已经在线适应”的实地路线。对应 `PS_geometry_static_routes.geojson` 可在GIS中查看；它是平面巡查路线原型，未加入地形、禁飞区、障碍、转弯半径、相机覆盖和电池动力学，不能直接执行飞行。PS真实目标覆盖与首次到达目标时间维持null。

## 实现与复核

新增12项有针对性的闭环测试通过，涵盖返程预留、费用闭合、未来信息隔离、固定静态路线不受反馈影响、Bayes更新方向、无发现时间处理、世界配对与无标签公开接口。另独立重建60个目标世界，逐步重放2160条路线、216份条件汇总和162份配对对照。旧感知冻结代码及PS真实目标空表哈希不变；汇报材料仍在原处。

本机CPU执行仿真循环约{runtime['loop_seconds']:.2f}s，累计求解器计算约{runtime['sum_planner_wall_seconds']:.2f}s。没有网络下载、付费GPU、新训练、新感知调用或旧保留集评分。

## 下一步

先使用本轮轨迹明确在线何时改善、何时被错误反馈或转场成本拖累，核对结果是否只来自有利生成假设；不按本轮最佳条件换主预算或调噪声。路线闭环通过后，下一阶段优先把允许通行/可飞区域、合理相机观察范围或固定感知缓存接入同一个时间与位置接口，每次只验证一个新增约束。真实目标资料不足时，继续分开报告仿真机制与PS几何案例；不把仿真发现量当成实地救援效果。新颖性和论文主张尚未建立。

主要产物：`protocol.json`、`routes.jsonl`、`metrics.csv`、`condition_means.json`、`paired_comparisons.json`、`PS_geometry_static_routes.geojson`、`audit_report.json`。
'''
    (OUT/'README.md').write_text(text)
    save(OUT/'prospective_next_scope.json',{'status':'NOT_EXECUTED','next':'Analyze paired route failures under fixed assumptions,then integrate one explicit traversal/observation constraint without changing sensor weights.',
        'new_perception_calls_enabled':False,'training_enabled':False,'gpu_rental':False,'physical_flight_authorized':False,
        'real_targets_available':False,'old_perception_archive_preserved':True})
    print({'report':'WRITTEN','figures':5},flush=True)


def complete():
    p=verify();assert read(OUT/'audit_report.json')['status']=='PASS';visual=read(OUT/'visual_inspection.json');assert visual['all_five_figures_viewed']
    for n,h in visual['files'].items():checked(OUT/n,h)
    snapshots=OUT/'source_snapshots';snapshots.mkdir()
    for n,h in p['source_sha256'].items():checked(HERE/n,h);shutil.copyfile(HERE/n,snapshots/n)
    external={**p['external_sha256'],**{str(HERE/n):h for n,h in p['source_sha256'].items()},
              **{str(ROOT/'work/disaster_search'/n):h for n,h in p['old_perception_sources_sha256'].items()}}
    for n,h in external.items():checked(n,h)
    receipt={'status':'COMPLETE','client_report_date':'2026-10-28','source_code_sha256':p['source_sha256'],
        'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completion_receipt.json'},'external_files':external,
        'simulation_routes':2160,'synthetic_worlds':60,'realPS_geometry_static_routes':4,'new_tests_passed':12,'budget_violations':0,
        'all_five_figures_viewed':True,'new_perception_calls':0,'training':False,'gpu_rented':False,'new_real_target_labels':0,
        'old_reserved_sets_scored':False,'physical_flight_validation':False,'new_method_efficacy_established':False}
    save(OUT/'completion_receipt.json',receipt)
    for n,h in receipt['files'].items():checked(OUT/n,h)
    print({'status':'COMPLETE','receipt_sha256':digest(OUT/'completion_receipt.json'),'output_files':len(receipt['files'])},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['report','complete'],required=True);a=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'report':report,'complete':complete}[a.stage]()
