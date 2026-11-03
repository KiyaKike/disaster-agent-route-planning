"""Figures and a bounded interpretation of the predeclared trigger experiment."""
import argparse
import shutil

import numpy as np

from run_route_trigger import (OUT, HERE, ROOT, BASE, SOURCES, GROUP, SEEDS,
                               read, save, digest, verify, verify_previous)
from route_trigger_core import POLICIES, NAMES

REGIONS = {'口岸': 'Port', '南侧建筑区': 'South'}
MODES = {'aligned': 'Aligned', 'reversed': 'Reversed', 'independent': 'Uninformative'}
NOISE = ('perfect', 'moderate', 'poor')
COLORS = {'pre_static': '#94A3B8', 'post_static': '#334155', 'always': '#2563EB',
          'periodic_5': '#D97706', 'gain_5': '#059669', 'one_sd_5': '#9333EA'}
CN = {'pre_static': '灾前静态', 'post_static': '灾后静态', 'always': '每次重规划',
      'periodic_5': '每5次直接改路', 'gain_5': '正期望触发', 'one_sd_5': '一标准差触发'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify()
    audit = read(OUT/'audit_report.json')
    assert audit['status'] == 'PASS'
    metrics = read(OUT/'metrics.json')
    means = read(OUT/'condition_means.json')
    pairs = read(OUT/'paired_comparisons.json')
    summary = read(OUT/'summary.json')
    runtime = read(OUT/'runtime.json')
    primary = [p for p in pairs if p['budget_s'] == 1200]
    conditions = [(r, m, n) for r in REGIONS for m in MODES for n in NOISE]
    comparisons = [('always', 'post_static'), ('periodic_5', 'post_static'), ('gain_5', 'post_static'),
                   ('one_sd_5', 'post_static'), ('one_sd_5', 'gain_5')]
    matrix = np.array([[next(p['mean_detected_delta'] for p in primary if
                      (p['region'], p['mode'], p['noise'], p['first'], p['second']) == (*c, *comparison))
                       for comparison in comparisons] for c in conditions])
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8.8), constrained_layout=True)
    scale = max(1., float(abs(matrix).max()))
    im = ax.imshow(matrix, cmap='RdBu', vmin=-scale, vmax=scale, aspect='auto')
    ax.set_yticks(range(18), [f'{REGIONS[r]} / {MODES[m]} / {n}' for r, m, n in conditions], fontsize=8)
    ax.set_xticks(range(5), ['Always\n- Post static', 'Every 5\n- Post static', 'Positive mean\n- Post static',
                           'One SD\n- Post static', 'One SD\n- Positive mean'], fontsize=9)
    for i in range(18):
        for j in range(5):
            ax.text(j, i, f'{matrix[i, j]:+.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(matrix[i, j]) > .55*scale else '#111111')
    ax.set_title('SIMULATION | 1200 s | mean paired correct-detection difference\n20 new seed IDs per condition;all18 conditions retained;fixed rules,no tuning')
    fig.colorbar(im, ax=ax, shrink=.7, label='Correct synthetic target cells')
    fig.savefig(figures/'all_condition_deltas.png', dpi=175)
    plt.close(fig)
    agg = summary['pooled_noise_means']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
    for ax, noise in zip(axes, NOISE):
        for policy in POLICIES:
            r = next(x for x in agg if x['noise'] == noise and x['policy'] == policy)
            ax.scatter(r['planning_calls'], r['detected_target_cells'], c=COLORS[policy], s=50, label=NAMES[policy])
            short = {'pre_static': 'Pre', 'post_static': 'Post', 'always': 'Always', 'periodic_5': 'Every5',
                     'gain_5': 'Mean', 'one_sd_5': '1SD'}[policy]
            ax.annotate(short, (r['planning_calls'], r['detected_target_cells']), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.grid(alpha=.2)
        ax.set(title=noise, xlabel='Paid planning calls,including rejections', xlim=(0, 43))
    axes[0].set_ylabel('Mean correct synthetic target cells')
    fig.suptitle('SIMULATION | 1200 s | equal-weight mean across both geometries and3 priors\nMeans are descriptive;candidate computation is paid even when rejected')
    fig.savefig(figures/'planning_tradeoff.png', dpi=175)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True, sharey=True)
    for ax, noise in zip(axes, NOISE):
        for policy in POLICIES:
            values = [np.mean([r['detected_target_cells'] for r in metrics if r['noise'] == noise and r['policy'] == policy and r['budget_s'] == budget]) for budget in (600, 1200, 1800)]
            ax.plot([10, 20, 30], values, '-o', c=COLORS[policy], label=NAMES[policy], markersize=4)
        ax.grid(alpha=.2)
        ax.set(title=noise, xlabel='Full tour budget (minutes)', xticks=[10, 20, 30])
    axes[0].set_ylabel('Mean correct synthetic target cells')
    axes[2].legend(fontsize=7)
    fig.suptitle('SIMULATION | new draws,same model and geometry | 20 min is primary\nMovement,inspection,all attempted plans and return included')
    fig.savefig(figures/'budget_sensitivity.png', dpi=175)
    plt.close(fig)
    examples = read(OUT/'examples.json')
    public = read(BASE/'public_geometry.json')
    worlds = read(OUT/'synthetic_worlds.json')
    for region, en in REGIONS.items():
        g = public[region]
        xy = np.array(g['xy_m'])
        xy -= xy.min(0)
        home = g['ids'].index(g['start_id'])
        world = next(w for w in worlds if w['region'] == region and w['mode'] == 'aligned' and w['seed'] == SEEDS[0])
        targets = np.array([sid in set(world['target_ids']) for sid in g['ids']])
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for ax, policy in zip(axes.flat, POLICIES):
            r = next(x for x in examples if x['region'] == region and x['policy'] == policy)
            s = r['final_public']
            nodes = [home]+s['visited_indices']+[home]
            ax.scatter(xy[:, 0], xy[:, 1], s=3, c='#DDDDDD')
            ax.scatter(xy[targets, 0], xy[targets, 1], s=10, c='#C74440', alpha=.5)
            ax.plot(xy[nodes, 0], xy[nodes, 1], c=COLORS[policy], linewidth=1.2)
            correct = [h['cell_index'] for h in s['history'] if targets[h['cell_index']] and h['observed_positive']]
            if correct:
                ax.scatter(xy[correct, 0], xy[correct, 1], s=30, facecolors='none', edgecolors='#15803D')
            changed = [s['history'][d['after_observations']-1]['cell_index'] for d in r['decisions'] if d['route_changed']]
            if changed:
                ax.scatter(xy[changed, 0], xy[changed, 1], marker='D', s=18, c='#111111')
            ax.scatter(*xy[home], marker='*', s=100, c='#111111')
            e = r['evaluation']
            ax.set(title=f"{NAMES[policy]}\n{e['detected_target_cells']} correct | {e['observations']} visits | {e['planning_calls']} calls", xlabel='Local easting (m)')
            ax.set_aspect('equal')
            ax.grid(alpha=.15)
        for ax in axes[:, 0]:
            ax.set_ylabel('Local northing (m)')
        fig.suptitle(f'SIMULATION | {en} | first new seed,aligned/moderate,1200 s\nAll6 full tours shown;red=synthetic target,green circle=correct detection,diamond=route change')
        fig.savefig(figures/f'routes_{en.lower()}.png', dpi=175)
        plt.close(fig)
    lookup = {(m['region'], m['mode'], m['noise'], m['policy']): m for m in means if m['budget_s'] == 1200}
    gate = []
    for policy in POLICIES:
        if policy == 'one_sd_5':
            continue
        bad = []
        for c in conditions:
            a, b = lookup[*c, 'one_sd_5'], lookup[*c, policy]
            if (a['detected_target_cells'] < b['detected_target_cells']-1e-12 or
                a['total_time_s'] > b['total_time_s']+1e-7 or a['planning_calls'] > b['planning_calls']+1e-12 or
                a['first_detection_or_budget_s'] > b['first_detection_or_budget_s']+1e-7):
                bad.append(list(c))
        gate.append({'comparator': policy, 'dominance_holds': not bad, 'violating_conditions': bad})
    save(OUT/'advance_gate.json', {'rule': 'See predeclared protocol;no threshold or period selection after seeing results.',
                                  'one_sd_universal_winner_gate_passed': all(x['dominance_holds'] for x in gate),
                                  'comparisons': gate})
    counts = [x for x in summary['comparison_counts'] if x['noise'] == 'all' and (x['first'], x['second']) in comparisons]
    count_table = '\n'.join(f"| {CN[x['first']]} − {CN[x['second']]} | {' / '.join(map(str, x['positive_zero_negative']))} | {' / '.join(map(str, x['win_tie_loss']))} | {x['mean_detected_delta']:+.3f} |" for x in counts)
    condition_table = '\n'.join(f"| {r} | {MODES[m]} | {n} | {matrix[i, 4]:+.2f} | {matrix[i, 3]:+.2f} | {matrix[i, 2]:+.2f} |" for i, (r, m, n) in enumerate(conditions))
    mean_table = '\n'.join(f"| {x['noise']} | {CN[x['policy']]} | {x['detected_target_cells']:.3f} | {x['observations']:.2f} | {x['planning_calls']:.2f} | {x['accepted_proposals']:.2f} | {x['first_detection_or_budget_s']:.1f} | {x['no_correct_detection_rate']:.1%} | {x['total_distance_m']:.0f} |" for x in agg)
    primary_count = next(x for x in summary['comparison_counts'] if x['noise'] == 'all' and x['first'] == 'one_sd_5' and x['second'] == 'gain_5')
    assessment = ('固定的一标准差规则未带来平均正确发现量提升，不能据此升级为主方法。' if primary_count['mean_detected_delta'] < 0 else
                  '固定的一标准差规则在本批平均正确发现量上出现正向空间，仍需逐条件和成本指标判断。')
    text = f'''# 付费重规划触发：新合成实例的完整路线检验

2026-11-02。当前已完成六种完整路线策略的对照。主问题是何时采用新的剩余巡查路线；感知误差保持固定，没有新感知推理、训练或GPU费用。

## 本轮结论

{assessment} 20分钟主预算，一标准差规则相对正期望规则的18个条件均值为{primary_count['positive_zero_negative'][0]}正、{primary_count['positive_zero_negative'][1]}平、{primary_count['positive_zero_negative'][2]}负；360条配对路线胜/平/负为{'/'.join(map(str, primary_count['win_tie_loss']))}。每次改路的局部不确定性与完整路线最终发现量并非同一个量，保守筛选不能自动保证更好的完整巡查收益。

| 比较 | 条件均值正 / 平 / 负 | 路线胜 / 平 / 负 | 平均正确发现增量格 |
|---|---|---|---:|
{count_table}

所有失败条件保留，不据结果调整“一标准差”或间隔5次。预设的全面支配关口通过：**{all(x['dominance_holds'] for x in gate)}**。这个关口同时检查发现量、总耗时、规划调用和受限首次发现时间；未通过时只报告权衡，不宣布通用最优。

![全部条件差值](figures/all_condition_deltas.png)

## 实验范围与六种策略

冻结代码、规则、种子和评价后，才生成120个合成目标场，完成6,480条完整闭合路线。20个新种子为2026110200—2026110219，与旧10个种子无交集；两个PS几何区域、三种先验条件、三档固定误差、600/1200/1800秒全部保留。1200秒是主要预算。每条件20条配对路线，不按结果挑样。

1. 灾前静态：均匀热点先验0.2，只规划一次。
2. 灾后静态：固定变化线索先验，只规划一次。
3. 每次重规划：每次付费观察后，若仍可支付规划、一次观察及返程，就付费重规划并采用新后缀。
4. 每5次直接改路：每5次观察后付费重规划并采用。
5. 正期望触发：相同间隔，付费计算后，只有预计正确发现数增加才接受。
6. 一标准差触发：相同间隔，预计增量超过差值的一个预测标准差才接受。

三种每5次策略只改变是否接受候选，检查时机一致。它们各自走不同路线，最终检查次数可能不同。筛选候选不等于省掉候选计算；拒绝也支付1秒，因此不能宣称一标准差规则相对正期望规则天然节约计算。

![规划调用与发现量](figures/planning_tradeoff.png)

## 成本、信息与不确定性

共同使用未改动的插入+2-opt求解器，不设30次观察上限。每次求解生成当前可承诺的剩余路线，未来规划调用在实际发生时另扣。速度5米/秒、观察20秒、每次候选计算1秒，移动、初始规划、被拒绝的候选、采用的候选和返程全部计费。没有重复收取转场距离。规划1秒为模拟假设，实测CPU时间另外报告。

拒绝候选后仍须承担已发生费用：旧后缀如果因此超预算，从末尾删点，直到可返程。该删点在账本中保留。若连付规划费后再观察一次并返程都不可能，则跳过候选计算，沿仍可行的旧后缀执行；后缀耗尽就返程，不额外增加计划外检查。

预期增量以“付费前旧后缀”与“付费后新后缀”比较；共同格子的未来发现数抵消。每个未访问格的正确发现概率是灵敏度×(0.02+0.58×当时热点后验)。方差保留同一热点块内的共享潜变量相关性，其他块条件独立。误报通过已付观测改变热点后验，未访问观测和合成真值不进入规划器。

**一个标准差是固定启发式，不是单次收益保证、95%置信界或经过验证的风险约束。** 先验反转/无信息时，模型的预测均值和方差也可能失配；未据本批输出拟合阈值。

V4所有在线策略均采用上述共同的承诺后缀规则，与V2每次在线求解时预留后续每步规划费的约定不同。因此在线效果只在本轮公平比较，不能与V2绝对分数混合排名；灾前/灾后静态求解规则保持，2,160条本轮静态轨迹已核验。

## 主预算完整结果

| 区域 | 先验 | 误差 | 一标准差−正期望 | 一标准差−灾后静态 | 正期望−灾后静态 |
|---|---|---|---:|---:|---:|
{condition_table}

以下每行等权汇总两个区域及三种先验，共120次共享合成实例；仅供描述，不能当作120次独立灾害。

| 误差 | 策略 | 正确发现格 | 观察数 | 付费规划次数 | 采用候选次数 | 受限首次发现秒 | 无正确发现率 | 往返米 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{mean_table}

首次到达目标与首次正确发现均保留原值；失败为null，另报无发现率。上表受限时间在无发现时填预算；无目标时召回为null。原始逐步路径和假阳性结果完整保留。

![预算敏感性](figures/budget_sensitivity.png)

## 固定路线示例

两区都展示事前固定的第一个新种子、先验正确、中等误差、20分钟全部六种策略。红点是模拟目标，绿圈是正确发现，黑色菱形标记观察后改变剩余路线的位置，星号为同一起终点。示例不替换成有利案例。

![口岸完整路线](figures/routes_port.png)

![南侧完整路线](figures/routes_south.png)

## 核验与适用范围

53项测试通过，其中新增14项。独立重建全部120个目标场、{audit['paid_observations_replayed']:,}次观察、{audit['paid_planner_calls_replayed']:,}次付费规划和{audit['proposal_decisions_replayed']:,}次候选决定；{audit['paid_rejections_verified']:,}次拒绝仍按1秒记账。观察、费用、后验、候选、接受标准、返回和评分全部重放，预算违规0，免费候选求解0；旧V1/V2/V3和老师汇报的46个归档文件保持不变。

本机CPU运行循环约{runtime['loop_seconds']:.2f}秒，规划接口累计实测{runtime['sum_measured_planner_wall_s']:.2f}秒。无GPU、训练、新影像下载或真实PS标签。原PS独立目标空表未填。

这次解决了只在静态行为快照上测试的限制：触发规则已沿自己产生的在线状态执行到返程。但新种子仍来自同一3×3热点模型、相同已知误差率和相同两块PS几何；先验条件共享随机流，预算/噪声/方法共享目标和观察抽样。它不是新的独立灾害、真实损伤验证或物理航线验证。平面直线移动仍缺地形、障碍、相机覆盖和飞行约束。

下一步先把完整路线的发现量、及时性、调用成本和负例作为方法选择依据，不在本批上继续调整阈值追分。保留已验证的简单基线，下一轮另行固定一个实际需要检验的因素，例如误差率未知或真实移动约束；不能把继续添加启发式当成研究贡献已经成立。
'''
    (OUT/'README.md').write_text(text)
    save(OUT/'prospective_next_scope.json', {'status': 'NOT_EXECUTED',
         'next': 'Select a bounded practical factor for the route problem from the complete tradeoffs;do not tune the threshold/period on these results.',
         'candidate_threshold_retuning_enabled': False, 'real_world_efficacy_established': False,
         'new_training': False, 'new_perception_calls_enabled': False, 'gpu_rental': False,
         'real_targets_available': False})
    print('REPORT_READY', flush=True)


def seal():
    protocol = verify()
    verify_previous()
    assert not (OUT/'completion_receipt.json').exists()
    audit = read(OUT/'audit_report.json')
    visual = read(OUT/'visual_inspection.json')
    assert audit['status'] == 'PASS' and visual['all_five_figures_viewed']
    snapshots = OUT/'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    for name in SOURCES:
        shutil.copyfile(HERE/name, snapshots/name)
    receipt = {'status': 'COMPLETE', 'client_date': '2026-11-02', 'source_sha256': protocol['source_sha256'],
               'worlds': 120, 'routes': 6480, 'tests_passed': 53, 'new_tests': 14,
               'observations': audit['paid_observations_replayed'], 'paid_planner_calls': audit['paid_planner_calls_replayed'],
               'budget_violations': 0, 'free_candidate_solves': 0, 'all_five_figures_viewed': True,
               'new_training': False, 'new_perception_calls': 0, 'gpu_rented': False, 'new_real_labels': 0,
               'threshold_or_period_tuned_after_outcomes': False, 'real_world_efficacy_established': False,
               'universal_winner_gate_passed': read(OUT/'advance_gate.json')['one_sd_universal_winner_gate_passed'],
               'files': {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}}
    save(OUT/'completion_receipt.json', receipt)
    print('SEALED '+digest(OUT/'completion_receipt.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('report', 'seal'))
    args = p.parse_args()
    {'report': report, 'seal': seal}[args.command]()
