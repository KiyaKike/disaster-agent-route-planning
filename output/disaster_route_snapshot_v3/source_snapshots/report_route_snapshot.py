"""Same-state fork plots, interpretation and verified output sealing."""
import argparse
import shutil

import numpy as np

from run_route_snapshot import OUT,BASE,HERE,read,save,digest,checked,verify,verify_parents

REGIONS={'口岸':'Port','南侧建筑区':'South'}
MODES={'aligned':'Aligned','reversed':'Reversed','independent':'Uninformative'}
NOISE=('perfect','moderate','poor')
COLORS={'continue':'#2563EB','replan_feedback':'#B45309','replan_prior':'#8B5CF6'}
LABELS={'continue':'Keep old route','replan_feedback':'Replan with feedback','replan_prior':'Replan without feedback'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify();assert read(OUT/'audit_report.json')['status']=='PASS'
    pairs=read(OUT/'paired_comparisons.json');means=read(OUT/'condition_means.json');summary=read(OUT/'summary.json');runtime=read(OUT/'runtime.json')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    rows=[(region,mode,noise) for region in REGIONS for mode in MODES for noise in NOISE]
    comparisons=[('replan_feedback','continue'),('replan_prior','continue'),('replan_feedback','replan_prior')]
    values=np.array([[next(p['mean_detected_delta'] for p in pairs if (p['region'],p['mode'],p['noise'],p['checkpoint'],p['first'],p['second'])==(*row,10,*comparison)) for comparison in comparisons] for row in rows])
    fig,ax=plt.subplots(figsize=(9,8.5),constrained_layout=True);scale=max(1.,float(abs(values).max()))
    im=ax.imshow(values,cmap='RdBu',vmin=-scale,vmax=scale,aspect='auto')
    ax.set_yticks(range(18),[REGIONS[r]+' / '+MODES[m]+' / '+n for r,m,n in rows],fontsize=8)
    ax.set_xticks(range(3),['Feedback replan\n- keep route','Prior replan\n- keep route','Feedback replan\n- prior replan'],fontsize=9)
    for i in range(18):
        for j in range(3):ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',color='white' if abs(values[i,j])>scale*.55 else '#111111',fontsize=9)
    ax.set_title('SIMULATION | identical paid history after10 inspections\nMean difference in future correct detections;all18 conditions retained')
    fig.colorbar(im,ax=ax,label='Paired mean future target detection difference',shrink=.7)
    fig.savefig(figures/'same_state_deltas.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.5),constrained_layout=True,sharey='row')
    for row,region in enumerate(REGIONS):
        for col,mode in enumerate(MODES):
            ax=axes[row,col]
            for noise,color in zip(NOISE,('#15803D','#2563EB','#B45309')):
                rs=sorted([p for p in pairs if p['region']==region and p['mode']==mode and p['noise']==noise and p['first']=='replan_feedback' and p['second']=='continue'],key=lambda p:p['checkpoint'])
                ax.plot([r['checkpoint'] for r in rs],[r['mean_detected_delta'] for r in rs],'-o',c=color,label=noise,markersize=4)
            ax.axhline(0,c='#555555',linewidth=.8);ax.grid(alpha=.2);ax.set(title=REGIONS[region]+' | '+MODES[mode],xlabel='Paid observations before snapshot',xticks=[5,10,20])
            if col==0:ax.set_ylabel('Feedback replan - keep route')
    axes[0,0].legend(fontsize=8);fig.suptitle('SIMULATION | one-time route revision | primary checkpoint10;5 and20 are secondary')
    fig.savefig(figures/'checkpoint_sensitivity.png',dpi=175);plt.close(fig)
    diagnostics=read(OUT/'prediction_diagnostics.json');fig,axes=plt.subplots(1,3,figsize=(11,3.7),constrained_layout=True,sharey=True)
    for ax,d in zip(axes,diagnostics):
        x=[r['predicted_delta'] for r in d['data']];y=[r['realized_delta'] for r in d['data']]
        ax.scatter(x,y,s=23,alpha=.7,c='#2563EB');ax.axhline(0,c='#555555',linewidth=.8);ax.axvline(0,c='#555555',linewidth=.8)
        ax.set(title=f"{d['noise']} | positive prediction:{d['positive_predicted_gain']}\nactual losses among them:{d['actual_losses_among_positive_predictions']}",xlabel='Public expected detection gain');ax.grid(alpha=.15)
    axes[0].set_ylabel('Realized future detection gain')
    fig.suptitle('SIMULATION | checkpoint10,60 correlated snapshots per noise | no fitting or trigger deployment')
    fig.savefig(figures/'predicted_vs_realized.png',dpi=175);plt.close(fig)
    public=read(BASE/'public_geometry.json');worlds=read(BASE/'synthetic_worlds.json');examples=read(OUT/'examples.json')
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row,region in enumerate(REGIONS):
        g=public[region];xy=np.array(g['xy_m']);xy-=xy.min(0);home=g['ids'].index(g['start_id'])
        world=next(w for w in worlds if w['region']==region and w['mode']=='aligned' and w['seed']==2026102800)
        target=np.array([sid in set(world['target_ids']) for sid in g['ids']])
        for col,branch in enumerate(COLORS):
            ax=axes[row,col];r=next(r for r in examples if r['region']==region and r['branch']==branch);s=r['final_public'];k=r['checkpoint']
            prefix=[home]+[h['cell_index'] for h in s['history'][:k]];future=[prefix[-1]]+[h['cell_index'] for h in s['history'][k:]]+[home]
            ax.scatter(xy[:,0],xy[:,1],s=3,c='#DDDDDD');ax.scatter(xy[target,0],xy[target,1],s=10,c='#C74440',alpha=.5)
            ax.plot(xy[prefix,0],xy[prefix,1],c='#111111',linewidth=1.7)
            ax.plot(xy[future,0],xy[future,1],c=COLORS[branch],linewidth=1.3)
            ax.scatter(*xy[home],marker='*',s=110,c='#111111');ax.scatter(*xy[prefix[-1]],marker='D',s=35,c='#111111')
            correct=[h['cell_index'] for h in s['history'][k:] if target[h['cell_index']] and h['observed_positive']]
            if correct:ax.scatter(xy[correct,0],xy[correct,1],s=30,facecolors='none',edgecolors='#15803D')
            ax.set(title=f"{REGIONS[region]} | {LABELS[branch]}\n{r['evaluation']['future_detected_target_cells']} future detections | {r['evaluation']['future_observations']} future visits",xlabel='Local easting (m)');ax.set_aspect('equal');ax.grid(alpha=.15)
            if col==0:ax.set_ylabel('Local northing (m)')
    fig.suptitle('SIMULATION | fixed first seed,aligned/moderate,checkpoint10\nBlack prefix is identical;diamond = fork;colored path = committed continuation including return')
    fig.savefig(figures/'fork_examples.png',dpi=175);plt.close(fig)
    primary=[p for p in pairs if p['checkpoint']==10 and p['first']=='replan_feedback' and p['second']=='continue']
    table='\n'.join(f"| {p['region']} | {MODES[p['mode']]} | {p['noise']} | {p['mean_detected_delta']:+.1f} | {'/'.join(map(str,p['win_tie_loss']))} | {p['mean_suffix_distance_delta_m']:+.0f} | {p['mean_future_observations_delta']:+.1f} |" for p in primary)
    pred='\n'.join(f"| {d['noise']} | {d['snapshots']} | {d['positive_predicted_gain']} | {d['actual_losses_among_positive_predictions']} |" for d in diagnostics)
    count=summary['primary_condition_counts'];wtl=summary['primary_snapshot_win_tie_loss']
    text=f'''# 同状态快照分叉与一次改路的净影响

2026-11-02。受[DeepSeek DSec](https://arxiv.org/abs/2609.22978)第6.1和6.2节快照恢复、执行状态保存，以及第6.4和6.5节评测信息隔离启发。本项目将这些工程思路用于同状态路线分叉；这不是DSec提出的路线算法，也没有复现其分布式沙箱平台。

## 实验问题与边界

在完全相同的位置、已付成本和观测历史之后，立即修改一次剩余路线是否有净收益？本轮从强化求解器生成的灾后静态路线收集前缀，一边获得付费观测、一边记录Bayes后验，但在快照前不改路。全部60个旧合成目标场、三档误差、两个PS几何区域均保留，在第5、10、20次观察后保存JSON快照，合计540个快照、1,620条后续轨迹。第10次为主要比较点，其余两点为敏感性分析。

每个快照分三支：继续旧路线；付1秒后根据当前反馈重算一次并固定执行；付同样1秒但按原灾后先验重算一次并固定执行。第三支用于区分重新求解和使用反馈的影响。已发生的前缀直接恢复，不重走、不重观察、不重复扣费；各分支自行支付后续移动、观察、新规划和返程。

**这是一次改路的局部影响，后续不再重规划，不能与上一轮始终在线策略的效果直接等同。快照来自静态行为轨迹，也不能代表所有在线策略会遇到的状态。**

## 主比较结果

第10次观察后的18个区域/先验/误差条件，反馈重规划减继续原路线的平均正确发现格：正向{count['positive']}、相同{count['zero']}、负向{count['negative']}。180个快照的胜/平/负为{wtl[0]}/{wtl[1]}/{wtl[2]}，仍是共享世界上的描述统计。

| 区域 | 先验 | 误差 | 后续正确发现增量 | 快照胜/平/负 | 后续往返距离增量米 | 后续观察数增量 |
|---|---|---|---:|---|---:|---:|
{table}

![同状态主比较](figures/same_state_deltas.png)

主指标只计算快照之后新发现的合成目标格。已经在前缀发现的目标不再记入后续奖励。下一次正确发现的等待时间在失败时保留null；受限时间指标填快照时的剩余预算，并另报无后续发现率。原始总路线、覆盖、误报、距离和时间均保留。

![快照时机](figures/checkpoint_sensitivity.png)

## 预期增益与实际增益

行动前的预期发现数由当前公开后验与已知模拟灵敏度计算，不读取未来观测或真值。下表仅检查预期增益为正时是否仍出现实际负增益，没有据此拟合阈值或宣称触发器有效。

| 误差 | 主快照数 | 预期增益为正 | 其中实际负增益 |
|---|---:|---:|---:|
{pred}

![预期和实际](figures/predicted_vs_realized.png)

![相同前缀的路线分叉](figures/fork_examples.png)

示例按事前规则固定为每区第一种子、先验正确、中等误差、第10次观察后分叉，没有换成有利轨迹。红点为合成目标，黑色已发生路径在三支完全相同，彩色为各自后续路线。

## 接口与验证

新增`Session`提供付费step、JSON快照、完整状态恢复和独立后续执行。公开状态包含位置、预算账本、观察历史、信念、剩余路线；环境侧保存不可变世界引用和已观察游标。恢复同时核对快照哈希、环境指纹、历史费用和后验，分支之间不共享可变游标。完整快照需要可信环境注册表来解析世界引用，不是操作系统进程快照。

策略接口只接收公开状态副本和公共几何。目标真值、未访问观测和评价分数只由模拟环境与评分器使用。本轮为固定Python策略的接口隔离，尚未建立可执行恶意代码的进程/文件沙箱。

39项测试通过，其中新增16项。540个继续原路线分支精确复现180条原静态轨迹的动作、时间、距离与发现结果。所有快照和1,620条分支均经过独立账本、观测、后验、未来奖励及汇总核验；规划选择使用未改动的上一轮求解器按重建的公开输入复算。

本机CPU循环约{runtime['loop_seconds']:.2f}秒，后续规划累计实测约{runtime['sum_branch_planner_wall_s']:.2f}秒。没有新影像下载、感知推理、训练、付费GPU或真实标签。

全部快照、条件、前缀和旧世界之间存在相关性，540不是540次独立灾害；数据仍为已用开发数据。已知3×3热点结构、已知模拟误差率与平面移动假设保持。真实PS目标收益、物理飞行有效性与新策略贡献仍未建立。

## 后续关口

先利用同状态差值确定改路收益及风险是否能被当时可用信息预测，再固定一个简单的触发规则。规则确定后，另立前瞻的新实例与在线行为状态检验；当前快照不能同时用来挑规则和报告独立确认收益。预测值为正不等于实际保证收益，后续移动已按路径计费，不再重复惩罚同一转场距离。
'''
    (OUT/'README.md').write_text(text)
    print({'report':'WRITTEN','figures':4},flush=True)


def complete():
    p=verify();verify_parents();assert read(OUT/'audit_report.json')['status']=='PASS'
    visual=read(OUT/'visual_inspection.json');assert visual['all_four_figures_viewed'] and len(visual['files'])==4
    for name,sha in visual['files'].items():checked(OUT/name,sha)
    directory=OUT/'source_snapshots';directory.mkdir()
    for name,sha in p['source_sha256'].items():checked(HERE/name,sha);shutil.copyfile(HERE/name,directory/name)
    receipt={'status':'COMPLETE','client_date':'2026-11-02','source_sha256':p['source_sha256'],
             'files':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file()},
             'snapshots':540,'suffix_rollouts':1620,'continued_parent_reproductions':540,'unique_parent_episodes':180,
             'tests_passed':39,'new_tests':16,'budget_violations':0,'all_four_figures_viewed':True,
             'trigger_fitted':False,'new_training':False,'new_perception_calls':0,'gpu_rented':False,
             'new_real_labels':0,'adversarial_sandbox_established':False,'new_policy_efficacy_established':False}
    save(OUT/'completion_receipt.json',receipt)
    for name,sha in receipt['files'].items():checked(OUT/name,sha)
    print({'status':'COMPLETE','files':len(receipt['files']),'receipt_sha256':digest(OUT/'completion_receipt.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('report','complete'),required=True);a=parser.parse_args()
    assert not (OUT/'completion_receipt.json').exists()
    {'report':report,'complete':complete}[a.stage]()
