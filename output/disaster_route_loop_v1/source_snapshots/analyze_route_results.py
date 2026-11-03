"""Post-run diagnostics of the frozen route experiment; no new runs or tuning."""
import hashlib
import json
from pathlib import Path
import statistics

from run_route_loop import OUT, read, save, verify

DATE = '2026-11-01'
POLICIES = ('pre_static', 'post_static', 'online')
LABELS = {'pre_static': '灾前静态', 'post_static': '灾后一次性', 'online': '在线重规划'}
MODES = {'aligned': '先验正确', 'reversed': '先验反向', 'independent': '先验无信息'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values)


def analyze():
    verify()
    assert read(OUT / 'audit_report.json')['status'] == 'PASS'
    assert not (OUT / 'completion_receipt.json').exists()
    metrics = read(OUT / 'metrics.json')
    routes = [json.loads(line) for line in (OUT / 'routes.jsonl').read_text().splitlines()]
    lookup = {(r['world_id'], r['noise'], r['budget_s'], r['policy']): r for r in metrics}
    route_lookup = {(r['world_id'], r['noise'], r['budget_s'], r['policy']): r for r in routes}
    primary = [r for r in metrics if r['budget_s'] == 1200 and r['policy'] == 'online']
    comparisons = {}
    losses = []
    for comparator in ('post_static', 'pre_static', 'online_frozen'):
        deltas = []
        conditions = []
        for region in ('口岸', '南侧建筑区'):
            for mode in MODES:
                for noise in ('perfect', 'moderate', 'poor'):
                    selected = [r for r in primary if (r['region'], r['mode'], r['noise']) == (region, mode, noise)]
                    assert len(selected) == 10
                    ds = [r['detected_target_cells'] - lookup[r['world_id'], noise, 1200, comparator]['detected_target_cells'] for r in selected]
                    deltas.extend(ds)
                    conditions.append({'region': region, 'mode': mode, 'noise': noise, 'mean_detected_delta': mean(ds)})
        comparisons[comparator] = {
            'conditions': conditions,
            'condition_win_tie_loss': [sum((c['mean_detected_delta'] > 0, c['mean_detected_delta'] == 0, c['mean_detected_delta'] < 0)[i] for c in conditions) for i in range(3)],
            'paired_run_win_tie_loss': [sum((d > 0, d == 0, d < 0)[i] for d in deltas) for i in range(3)],
            'warning': 'These counts share seeds and observation worlds across noise settings; they are descriptive, not independent replications or a significance test.',
        }
    for r in primary:
        key = (r['world_id'], r['noise'], 1200)
        other = lookup[*key, 'post_static']
        delta = r['detected_target_cells'] - other['detected_target_cells']
        if delta >= 0:
            continue
        a, b = route_lookup[*key, 'online'], route_lookup[*key, 'post_static']
        common = 0
        for x, y in zip(a['steps'], b['steps']):
            if x['cell_id'] != y['cell_id']:
                break
            common += 1
        losses.append({
            'world_id': r['world_id'], 'region': r['region'], 'mode': r['mode'], 'noise': r['noise'], 'seed': r['seed'],
            'online_detected': r['detected_target_cells'], 'post_static_detected': other['detected_target_cells'], 'delta': delta,
            'online_distance_m': r['total_distance_m'], 'post_static_distance_m': other['total_distance_m'],
            'shared_initial_visits': common,
            'last_online_observation_before_divergence': a['steps'][common-1] if common else None,
            'next_online_cell': a['steps'][common]['cell_id'] if common < len(a['steps']) else None,
            'next_static_cell': b['steps'][common]['cell_id'] if common < len(b['steps']) else None,
        })
    assert len(losses) == comparisons['post_static']['paired_run_win_tie_loss'][2]
    caps = []
    for region in ('口岸', '南侧建筑区'):
        for budget in (600, 1200, 1800):
            for policy in POLICIES + ('online_frozen',):
                rs = [r for r in metrics if (r['region'], r['budget_s'], r['policy']) == (region, budget, policy)]
                caps.append({'region': region, 'budget_s': budget, 'policy': policy, 'runs': len(rs),
                             'at_30_visit_cap': sum(r['visits'] == 30 for r in rs),
                             'mean_unused_budget_s': mean(budget-r['total_time_s'] for r in rs)})
    diagnostic = {'client_completion_date': DATE, 'analysis_type': 'POST_HOC_DESCRIPTIVE_NO_NEW_EXPERIMENT',
                  'inputs_sha256': {n: sha(OUT/n) for n in ('protocol.json', 'metrics.json', 'routes.jsonl', 'audit_report.json')},
                  'primary_budget_s': 1200, 'comparisons': comparisons,
                  'all_primary_online_losses_to_post_static': losses, 'visit_cap_diagnostics': caps,
                  'cross_mode_pairing': 'Each region/seed reuses RNG streams across the three prior modes; 60 stored worlds are not 60 independent real events.',
                  'no_new_runs': True, 'no_parameters_changed': True}
    save(OUT / 'post_run_diagnostics.json', diagnostic)
    table = []
    for region in ('口岸', '南侧建筑区'):
        for mode in MODES:
            for policy in POLICIES:
                rs = [r for r in metrics if (r['region'], r['mode'], r['noise'], r['budget_s'], r['policy']) == (region, mode, 'moderate', 1200, policy)]
                assert len(rs) == 10
                visited_recall = [r['visited_target_recall'] for r in rs if r['visited_target_recall'] is not None]
                recall = mean(visited_recall) if visited_recall else None
                recall_text = f'{recall:.1%}' if recall is not None else '未知'
                first = mean(r['first_target_arrival_s'] if r['first_target_arrival_s'] is not None else 1200 for r in rs)
                no_arrival = mean(r['first_target_arrival_s'] is None for r in rs)
                table.append(f"| {region} | {MODES[mode]} | {LABELS[policy]} | {mean(r['visited_target_cells'] for r in rs):.1f} | {recall_text} | {mean(r['detected_target_cells'] for r in rs):.1f} | {first:.1f} | {no_arrival:.0%} | {mean(r['first_detection_or_budget_s'] for r in rs):.1f} | {mean(r['false_positive_observations'] for r in rs):.1f} | {mean(r['total_distance_m'] for r in rs):.0f} | {mean(r['total_time_s'] for r in rs):.1f} |")
    text = f'''# 路线实验结果复盘与下一步

2026-11-01接续完成；协议、种子和实验源码仍保留2026-10-28冻结版本。本页是已有结果的事后描述分析，没有新增仿真或调参。

三种主路线的移动、观察、规划延迟与返程闭环已完成。2,160条路线、53,552次观察全部通过独立重放。识别模型与之前的感知汇报保留；本轮CPU仿真循环约5.54秒，无需租GPU。

20分钟预算下，在线相对灾后一次性路线的18个条件均值均为正；相对灾前均匀静态为17正、1平。但180个共享世界/噪声配对中，在线相对灾后一次性的胜/平/负为128/37/15，相对灾前静态为125/22/33。噪声设置共享世界，不能把这些配对当成180次独立灾害。所有15条主预算负例保留在 `post_run_diagnostics.json`，含路径首次分叉和此前观测。

中等误差、先验正确时，口岸的平均正确发现从灾后静态3.6格升到在线5.9格，南侧从3.4格升到6.7格。南侧在线的总路程约2810米，灾后静态约2128米；总耗时约1182秒与1027秒。它使用更多可用时间换取更多发现，并非路线同时更短、更快。

首次发现和漏掉整场目标的风险需要单独判断。口岸上述条件中，在线无发现率30%，灾后静态20%；失败计为预算的首次正确发现指标约471秒与498秒，平均略改善，但失败率更高。南侧在线同指标约253秒，灾前静态约239秒，无发现率10%与0%。所以当前不应给出“在线在所有目标上最优”的结论。30分钟、中等误差、口岸先验反向时，在线平均发现2.5格，低于灾前静态2.7格。

观测变差会削弱收益：南侧先验正确时，在线减灾后静态的20分钟平均发现增量，从理想观察6.4格降至中等误差3.3格、较差误差0.9格。“重算但不更新”对照在全部18个主条件也低于在线；这支持在当前生成模型下反馈更新有作用。它不能证明现实中的空间相关性、固定似然或灾损概率校准成立。

三种合成先验环境共享部分随机流，不是60次独立灾害；所有环境仍采用已知3×3热点结构和已知感知似然。灾前均匀先验只是最小基线，尚未用真实灾前建筑暴露图。求解器是局部贪心，灾前路线有明显长距离折返。静态路线在20至30分钟间的平台主要受30次观察上限影响；不能据此说追加电量无价值。这些限制都先于方法创新需要处理。

## 同预算完整指标

下表均为中等误差、20分钟预算、每条件10种子的均值。覆盖率先按每世界目标总数计算再平均；无目标时为null、不计入覆盖率均值。目标格为合成单位，不代表建筑数量。首次到达未到达时原值保留null，表中明确填1200秒为受限时间指标，并另报未到达率；首次正确发现也采用失败填预算的指标。两项时间不能与只保留成功者的条件均值混用。

| 区域 | 先验条件 | 路线 | 到达目标格 | 目标覆盖率 | 正确发现格 | 首次到达或预算秒 | 未到达率 | 首次正确发现或预算秒 | 误报格 | 往返米 | 总秒 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
''' + '\n'.join(table) + '''

## 接下来按顺序推进

1. **先补强路线对照。** 在新版本中给三种路线统一接入考虑闭合路线增量距离的插入/局部改进求解器，保留当前贪心作对照；保持目标场、观测与主预算，预先固定操作与计算时限。先判断收益能否超过更合理的静态规划，不先增加反馈花样。
2. **分开检查观察上限。** 当前30次上限结果原样保留；另立明确的“只按时间停止”设置，核对20至30分钟的预算收益是否被上限截断，不把新设置覆盖为旧实验成绩。
3. **接入一个实际路线约束。** 优先明确任务载体、相机观察范围与允许移动区域，再加入可得的一个约束；没有已核实的障碍/禁行图时不虚构。用户确认载体前可先完成纯算法对照，不依赖新数据或GPU。
4. **再补真实案例证据。** 从已有灾前影像提取或接入可追溯的建筑位置形成暴露权重；固定感知缓存只用于观测代理。没有真实PS目标标注，就继续把真实几何可行性与合成目标效果分开汇报。

本轮结论是路线实验管线已接通、反馈在规定合成条件下有收益与风险；研究贡献及真实灾后巡查收益尚未确认。感知检查归档仍在 `../感知检查_阶段汇报_20261028/`。
'''
    (OUT / '结果复盘与下一步.md').write_text(text)
    report = OUT / 'README.md'
    report_text = report.read_text().replace('2026-10-28。', '协议于2026-10-28冻结；2026-11-01接续完成核验与整理。', 1)
    if '[结果复盘与下一步]' not in report_text:
        report_text = report_text.replace('## 同预算的结果', '详细指标、全部主预算负例与后续优先级见 [结果复盘与下一步](结果复盘与下一步.md)。\n\n## 同预算的结果', 1)
    report.write_text(report_text)
    save(OUT / 'resumption_note.json', {'protocol_client_date': '2026-10-28', 'client_completion_date': DATE,
         'note': 'Frozen experimental sources and protocol are unchanged. Reporting resumed on the current client date. Added descriptive diagnostics; no new experiments.',
         'additional_analysis_source_sha256': sha(__file__)})
    save(OUT / 'prospective_next_scope.json', {'status': 'NOT_EXECUTED',
         'priority_order': ['Strengthen the shared closed-route static/online solver, retaining the original greedy control.',
                            'Separately evaluate time-only stopping instead of the inherited 30-visit cap.',
                            'Integrate one verified traversal or observation constraint after vehicle and available geometry are specified.',
                            'Build a traceable pre-disaster exposure prior; keep geometry and synthetic efficacy separate.'],
         'new_perception_calls_enabled': False, 'training_enabled': False, 'gpu_rental': False,
         'physical_flight_authorized': False, 'real_targets_available': False,
         'old_perception_archive_preserved': True, 'client_date': DATE})
    print({'diagnostics': 'WRITTEN', 'primary_negative_pairs_retained': len(losses), 'new_experiments': 0})


if __name__ == '__main__':
    analyze()
