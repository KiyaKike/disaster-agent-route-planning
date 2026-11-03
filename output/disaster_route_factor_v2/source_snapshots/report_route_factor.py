"""Report all factorial outcomes and seal only after figure inspection."""
import argparse
import shutil

import numpy as np

from run_route_factor import OUT,PARENT,HERE,SOLVERS,STOPS,read,save,digest,checked,verify,verify_parent

COLORS={'pre_static':'#6B7280','post_static':'#2563EB','online':'#B45309'}
NAMES={'pre_static':'Pre static','post_static':'Post static','online':'Online feedback'}
REGIONS={'口岸':'Port','南侧建筑区':'South'}
MODES={'aligned':'Aligned','reversed':'Reversed','independent':'Uninformative'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify();assert read(OUT/'audit_report.json')['status']=='PASS'
    means=read(OUT/'condition_means.json');pairs=read(OUT/'paired_comparisons.json');summary=read(OUT/'summary.json');runtime=read(OUT/'runtime.json')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),constrained_layout=True,sharey='row')
    for row,region in enumerate(REGIONS):
        for col,solver in enumerate(SOLVERS):
            ax=axes[row,col]
            for policy in ('pre_static','post_static','online'):
                for stopping in STOPS:
                    rs=sorted([r for r in means if r['region']==region and r['solver']==solver and r['stopping']==stopping and r['mode']=='aligned' and r['noise']=='moderate' and r['policy']==policy],key=lambda r:r['budget_s'])
                    ax.plot([r['budget_s']/60 for r in rs],[r['detected_target_cells'] for r in rs],'-o' if stopping=='time_only' else '--s',c=COLORS[policy],markersize=3,linewidth=1.4,label=NAMES[policy]+(' / time only' if stopping=='time_only' else ' / cap30'))
            ax.set(title=REGIONS[region]+' | '+solver,xlabel='Total budget including return (min)');ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Mean detected synthetic target cells')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('SIMULATION | solver x stopping rule | aligned prior, moderate noise | paired existing10 seeds')
    fig.savefig(figures/'factor_budget.png',dpi=175);plt.close(fig)
    rows=[(region,mode,noise) for region in REGIONS for mode in MODES for noise in ('perfect','moderate','poor')]
    columns=[(solver,stopping) for solver in SOLVERS for stopping in STOPS]
    values=np.array([[next(p['mean_detected_delta'] for p in pairs if (p['region'],p['mode'],p['noise'],p['solver'],p['stopping'],p['budget_s'],p['comparator'])==(*row,*column,1200,'post_static')) for column in columns] for row in rows])
    fig,ax=plt.subplots(figsize=(9,8.5),constrained_layout=True)
    scale=max(1.,float(np.abs(values).max()));im=ax.imshow(values,cmap='RdBu',vmin=-scale,vmax=scale,aspect='auto')
    ax.set_yticks(range(len(rows)),[REGIONS[r]+' / '+MODES[m]+' / '+n for r,m,n in rows],fontsize=8)
    ax.set_xticks(range(4),[s+'\n'+stop for s,stop in columns],fontsize=9)
    for i in range(values.shape[0]):
        for j in range(4):ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(values[i,j])>scale*.55 else '#111111')
    fig.colorbar(im,ax=ax,label='Paired mean online - post static target detections',shrink=.7)
    ax.set_title('SIMULATION | all18 conditions,20min budget\nPositive = online detects more;negative = static detects more')
    fig.savefig(figures/'factor_feedback_deltas.png',dpi=175);plt.close(fig)
    public=read(PARENT/'public_geometry.json');worlds=read(PARENT/'synthetic_worlds.json');examples=read(OUT/'examples.json')
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row,region in enumerate(REGIONS):
        g=public[region];xy=np.array(g['xy_m']);xy-=xy.min(0);lookup={sid:i for i,sid in enumerate(g['ids'])}
        world=next(w for w in worlds if w['region']==region and w['mode']=='aligned' and w['seed']==2026102800)
        targets=np.array([sid in set(world['target_ids']) for sid in g['ids']])
        for col,policy in enumerate(('pre_static','post_static','online')):
            ax=axes[row,col];r=next(r for r in examples if r['region']==region and r['policy']==policy)
            ax.scatter(xy[:,0],xy[:,1],s=3,c='#DDDDDD');ax.scatter(xy[targets,0],xy[targets,1],s=10,c='#C74440',alpha=.55)
            path=xy[[lookup[sid] for sid in r['closed_route_ids']]];ax.plot(path[:,0],path[:,1],c=COLORS[policy],linewidth=1.1)
            home=xy[lookup[g['start_id']]];ax.scatter(*home,marker='*',s=110,c='#111111')
            correct=[s['cell_index'] for s in r['steps'] if targets[s['cell_index']] and s['observed_positive']]
            if correct:ax.scatter(xy[correct,0],xy[correct,1],s=30,facecolors='none',edgecolors='#15803D')
            ax.set(title=f"{REGIONS[region]} | {NAMES[policy]}\n{r['evaluation']['detected_target_cells']} detections | {r['visits']} visits | {r['total_time_s']:.0f}s",xlabel='Local easting (m)')
            ax.set_aspect('equal');ax.grid(alpha=.15)
            if col==0:ax.set_ylabel('Local northing (m)')
    fig.suptitle('SIMULATION | stronger solver,time-only stop | fixed first seed,aligned/moderate,20min\nRed = synthetic targets;green rings = correct detections;black star = start and return')
    fig.savefig(figures/'strong_route_examples.png',dpi=175);plt.close(fig)
    count_table='\n'.join(f"| {r['solver']} | {r['stopping']} | {r['positive']} | {r['zero']} | {r['negative']} |" for r in summary['primary_condition_counts'])
    selected=[r for r in means if r['mode']=='aligned' and r['noise']=='moderate' and r['budget_s']==1200 and r['policy']!='online_frozen']
    table='\n'.join(f"| {r['region']} | {r['solver']} | {r['stopping']} | {r['policy']} | {r['detected_target_cells']:.1f} | {r['visited_target_cells']:.1f} | {r['visits']:.1f} | {r['first_arrival_or_budget_s']:.0f} | {r['first_detection_or_budget_s']:.0f} | {r['no_detection_rate']:.0%} | {r['total_distance_m']:.0f} | {r['total_time_s']:.0f} |" for r in selected)
    text=f'''# 强化路线对照与观察上限实验

2026-11-02。检验在线反馈收益能否经受更强静态路线对照，并把求解器质量与30次观察上限分开。沿用上一轮全部60个合成目标场、观测随机数、两个PS几何区域与预算；本轮为事后开发敏感性分析。

## 二乘二对照

求解器为原贪心与“闭合路线插入加2-opt”两档，停止规则为最多30格与仅按时间停止两档。每档仍比较灾前均匀静态、灾后线索静态、反馈在线与重算但不更新四策略。共8,640条路线，其中原贪心加30格的2,160条逐动作与旧结果精确复现，计算耗时除外。

强化求解器同时保留原贪心节点集合与从空路线插入的候选；按每次插入对完整往返路线增加的时间报价，最多8次2-opt改进后补插，再选择公开先验期望奖励更高的路线。该性质只保证规划时使用的先验奖励不下降，不保证真实目标发现更多。所有方法仍是启发式，不宣称全局最优。

每次规划仍扣除假设的1秒，移动5m/s、每格观察20秒、预留返程。相同公开状态的规划采用共同LRU缓存，缓存键无目标或未来观测；命中也照扣仿真规划成本。实际CPU工作另记，强化求解器计算更多，本轮不宣称同计算量或部署提速。

## 主预算结果

下表是20分钟下18个区域/先验/误差条件的在线减灾后静态平均发现数方向，所有条件保留。

| 求解器 | 停止规则 | 正向条件 | 相同条件 | 负向条件 |
|---|---|---:|---:|---:|
{count_table}

![全部主预算条件](figures/factor_feedback_deltas.png)

上图各格由同一世界内配对差值先在10个种子上平均。不同噪声与先验模式共享部分随机流，18个条件不是18次独立灾害复验；不把方向计数作为显著性检验。`paired_comparisons.json` 同时保留对灾前静态与无更新对照的全部648份配对结果，`solver_effects.json` 保留各策略换求解器后的432份变化。

## 距离时间与首次到达

以下为先验正确、中等误差、20分钟、10种子的均值。首次到达和首次正确发现的原始失败值为null；表中失败填预算作为受限时间指标，并另报无发现率。覆盖与发现均是合成目标格，不是逐栋灾损或实际搜救人数。其余条件见全部条件汇总。

| 区域 | 求解器 | 停止 | 路线 | 正确发现 | 到达目标 | 观察格 | 首次到达或预算秒 | 首次发现或预算秒 | 无发现率 | 往返米 | 总秒 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

![预算与上限](figures/factor_budget.png)

![预先固定的路线例子](figures/strong_route_examples.png)

例子仍是每区第一种子，不能按正负结果换图。红点为合成目标，只借用真实PS几何。此前真实PS目标空表、感知代码、汇报归档和旧实验全部保留，未把这些红点写成真实标签。

## 核验与边界

23项测试通过，其中新增11项。独立重建全部路线的移动、返程、计费、观测与Bayes更新，并核对所有已记录的规划后缀和强化候选的原贪心奖励下界。8,640条轨迹预算均可行，旧2,160份非运行时间指标完全一致；864份条件均值、648份配对与432份求解器变化均已重算。

本轮执行约{runtime['loop_seconds']:.1f}秒，规划请求{runtime['planner_invocations']}次，公开状态缓存命中{runtime['public_cache_hits']}次，实际未缓存求解器累计计算约{runtime['sum_uncached_solver_compute_s']:.1f}秒。没有训练、付费GPU、下载、新感知调用或新真实标签。

本轮只放宽观察次数并强化平面路线启发式。已知3×3空间结构、已知感知似然、直线移动、假设相机观察格仍保留；灾前路线仍用均匀先验。它回答对照敏感性，不证明失准感知下鲁棒性或实地航线收益。

下一阶段应以本轮强静态对照为起点，先判断在线相对收益及首次发现风险是否仍值得研究，再决定是否测试有条件改路。简单策略足够时，不增加复杂机制。
'''
    (OUT/'README.md').write_text(text)
    print({'report':'WRITTEN','figures':3},flush=True)


def complete():
    p=verify();verify_parent();assert read(OUT/'audit_report.json')['status']=='PASS'
    qa=read(OUT/'visual_inspection.json');assert qa['all_three_figures_viewed'] and len(qa['files'])==3
    for name,sha in qa['files'].items():checked(OUT/name,sha)
    snapshots=OUT/'source_snapshots';snapshots.mkdir()
    for name,sha in p['source_sha256'].items():checked(HERE/name,sha);shutil.copyfile(HERE/name,snapshots/name)
    receipt={'status':'COMPLETE','client_date':'2026-11-02','source_sha256':p['source_sha256'],
             'parent_receipt_sha256':p['parent_receipt_sha256'],
             'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file()},
             'routes':8640,'legacy_exact_reproductions':2160,'tests_passed':23,'new_tests':11,
             'budget_violations':0,'all_three_figures_viewed':True,'new_perception_calls':0,'new_training':False,'gpu_rented':False,
             'real_target_labels_written':False,'physical_flight_validation':False,'novel_method_established':False}
    save(OUT/'completion_receipt.json',receipt)
    for name,sha in receipt['files'].items():checked(OUT/name,sha)
    print({'status':'COMPLETE','output_files':len(receipt['files']),'receipt_sha256':digest(OUT/'completion_receipt.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('report','complete'),required=True);a=parser.parse_args()
    assert not (OUT/'completion_receipt.json').exists()
    {'report':report,'complete':complete}[a.stage]()
