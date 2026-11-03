"""Replay all fees, decisions, posterior updates and scores independently."""
from collections import Counter
import gzip
import json
import math

import numpy as np

from run_route_trigger import (OUT, BASE, HERE, ROOT, SEEDS, BUDGETS, GROUP, FIELDS, PAIRS,
                               maps, read, save, checked, verify, verify_previous)
from route_core import Config
from route_factor_core import Planner
from route_snapshot_core import PUBLIC_KEYS


def near(a, b):
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        assert b is not None and abs(a-b)<1e-7, (a, b)
    else:
        assert a == b, (a, b)


def distance(g, a, b):
    return math.hypot(float(g.xy[a, 0]-g.xy[b, 0]), float(g.xy[a, 1]-g.xy[b, 1]))


def closed_cost(g, position, route):
    nodes = [position, *route, g.start_index]
    return sum(distance(g, a, b) for a, b in zip(nodes, nodes[1:]))/5+20*len(route)


def main():
    verify()
    verify_previous()
    maps_ = maps()
    worlds = {w['world_id']: w for w in read(OUT/'synthetic_worlds.json')}
    assert len(worlds) == 120
    assert set(SEEDS).isdisjoint(w['seed'] for w in read(BASE/'synthetic_worlds.json'))
    world_count = Counter()
    for w in worlds.values():
        g = maps_[w['region']]
        assert w['seed'] in SEEDS and w['world_id'] == f"{w['region']}/{w['mode']}/{w['seed']}"
        assert w['generator_seed'] == w['seed']+(100000 if w['region'] == '南侧建筑区' else 0)
        rng = np.random.default_rng(w['generator_seed'])
        prob = g.prior_post.copy()
        if w['mode'] == 'reversed':
            order = np.argsort(prob, kind='stable')
            prob[order] = prob[order[::-1]]
        elif w['mode'] == 'independent':
            prob[:] = .2
        else:
            assert w['mode'] == 'aligned'
        np.testing.assert_array_equal(prob, w['true_hot_probabilities'])
        hot = rng.random(len(prob)) < prob
        targets = rng.random(len(g.ids)) < np.where(hot[g.blocks], .6, .02)
        assert w['hot_block_indices'] == np.flatnonzero(hot).tolist()
        assert w['target_ids'] == [g.ids[i] for i in np.flatnonzero(targets)]
        np.testing.assert_array_equal(rng.random(len(g.ids)), w['uniform_observation_draws'])
        world_count[w['region'], w['mode']] += 1
    assert len(world_count) == 6 and all(n == 20 for n in world_count.values())
    metrics = read(OUT/'metrics.json')
    lookup = {(r['world_id'], r['noise'], r['budget_s'], r['policy']): r for r in metrics}
    assert len(lookup) == len(metrics) == 6480
    likelihoods = {'perfect': (1., 0.), 'moderate': (.8, .1), 'poor': (.6, .25)}
    solver = Planner()
    steps_checked = calls_checked = decisions_checked = rejected_checked = static_checked = 0
    seen = set()
    for line in gzip.open(OUT/'routes.jsonl.gz', 'rt', encoding='utf-8'):
        r = json.loads(line)
        key = (r['world_id'], r['noise'], r['budget_s'], r['policy'])
        assert key not in seen
        seen.add(key)
        w = worlds[r['world_id']]
        assert all(r[k] == w[k] for k in ('region', 'mode', 'seed'))
        g = maps_[w['region']]
        s = r['final_public']
        assert set(s) == PUBLIC_KEYS and s['region'] == w['region']
        assert s['closed'] and not s['remaining_route']
        assert r['budget_s'] == s['budget_s'] and r['budget_s'] in BUDGETS
        assert (s['speed_m_s'], s['observe_s'], s['planning_latency_s']) == (5., 20., 1.)
        se, fp = likelihoods[r['noise']]
        target_ids = set(w['target_ids'])
        target = np.array([sid in target_ids for sid in g.ids], bool)
        bits = np.array(w['uniform_observation_draws']) < np.where(target, se, fp)
        belief = g.prior_pre.copy() if r['policy'] == 'pre_static' else g.prior_post.copy()
        position = g.start_index
        elapsed = travel = 0.
        used, pending, skips, decisions = [], [], [], []
        event_cursor = 0
        for j in range(len(s['history'])+1):
            dynamic = r['policy'] not in ('pre_static', 'post_static')
            due = dynamic and j>0 and (r['policy'] == 'always' or j % 5 == 0)
            affordable = any(i not in used and elapsed+1+distance(g, position, i)/5+20
                             +distance(g, i, g.start_index)/5 <= r['budget_s']+1e-9
                             for i in range(len(g.ids))) if due else False
            needs_event = j == 0 or (due and affordable)
            if due and not affordable:
                skips.append(j)
            events = []
            while event_cursor < len(s['planning_events']) and s['planning_events'][event_cursor]['after_observations'] == j:
                events.append(s['planning_events'][event_cursor])
                event_cursor += 1
            assert len(events) == int(needs_event), (key, j, len(events), needs_event)
            if events:
                e = events[0]
                near(e['elapsed_before_s'], elapsed)
                assert e['position_index'] == position and e['belief_source'] == 'feedback'
                before = elapsed
                elapsed += 1
                near(e['elapsed_after_s'], elapsed)
                used_array = np.zeros(len(g.ids), bool)
                used_array[used] = True
                expected, _ = solver.solve(g, 'insertion_2opt', g.xy[position], used_array, elapsed, belief,
                    Config(budget_s=r['budget_s'], max_visits=len(g.ids)), 0., len(g.ids)-len(used))
                assert e['planned_route'] == expected
                assert elapsed+closed_cost(g, position, expected) <= r['budget_s']+1e-7
                if j == 0:
                    assert 'acceptance' not in e
                    pending = expected
                else:
                    d = e['acceptance']
                    assert d['after_observations'] == j and d['old_route'] == pending
                    near(d['elapsed_before_s'], before)
                    assert d['proposed_route'] == expected
                    old, new = set(pending), set(expected)
                    assert len(old) == len(pending) and len(new) == len(expected)
                    assert not (old | new) & set(used)
                    # Independent covariance calculation from per-cell marginals.
                    nodes = sorted(old ^ new)
                    weights = np.array([1 if i in new else -1 for i in nodes])
                    q = np.array([belief[g.blocks[i]] for i in nodes])
                    probabilities = se*(.02+.58*q)
                    mean = float(np.dot(weights, probabilities))
                    variance = float(np.sum(probabilities*(1-probabilities)))
                    for a in range(len(nodes)):
                        for b in range(a+1, len(nodes)):
                            if g.blocks[nodes[a]] == g.blocks[nodes[b]]:
                                variance += 2*weights[a]*weights[b]*q[a]*(1-q[a])*(.58*se)**2
                    variance = max(0., variance)
                    near(d['mean_gain'], mean)
                    near(d['variance_gain'], variance)
                    near(d['sd_gain'], math.sqrt(variance))
                    assert (d['added_cells'], d['removed_cells'], d['shared_cells']) == (len(new-old), len(old-new), len(new & old))
                    accepted = (r['policy'] in ('always', 'periodic_5') or
                                (mean > 1e-12 if r['policy'] == 'gain_5' else mean-math.sqrt(variance) > 1e-12))
                    assert d['accepted'] == accepted
                    retained = list(pending)
                    while retained and elapsed+closed_cost(g, position, retained) > r['budget_s']+1e-8:
                        retained.pop()
                    assert d['retained_feasible_route'] == retained
                    assert d['dropped_tail_if_rejected'] == len(pending)-len(retained)
                    chosen = expected if accepted else retained
                    assert chosen == d['selected_route'] and d['route_changed'] == (chosen != pending)
                    if not accepted:
                        rejected_checked += 1
                    decisions.append(d)
                    decisions_checked += 1
                    pending = chosen
                calls_checked += 1
            if j == len(s['history']):
                break
            h = s['history'][j]
            assert pending and h['cell_index'] == pending.pop(0)
            i = h['cell_index']
            assert i not in used and h['cell_id'] == g.ids[i] and h['step'] == j+1 and h['block_index'] == int(g.blocks[i])
            leg = distance(g, position, i)
            arrival = elapsed+leg/5
            end = arrival+20
            for name, expected in (('elapsed_before_step_s', elapsed), ('leg_m', leg),
                                   ('arrival_s', arrival), ('observation_complete_s', end)):
                near(h[name], expected)
            assert end+distance(g, i, g.start_index)/5 <= r['budget_s']+1e-7
            assert h['observed_positive'] == bool(bits[i])
            b = int(g.blocks[i])
            near(h['block_belief_before'], belief[b])
            a, c = fp+(se-fp)*.6, fp+(se-fp)*.02
            if not bits[i]:
                a, c = 1-a, 1-c
            belief[b] = belief[b]*a/(belief[b]*a+(1-belief[b])*c)
            near(h['block_belief_after'], belief[b])
            used.append(i)
            position = i
            elapsed = end
            travel += leg
            steps_checked += 1
        assert not pending and event_cursor == len(s['planning_events'])
        assert decisions == r['decisions'] and skips == r['skipped_unaffordable_checkpoints']
        assert used == s['visited_indices']
        np.testing.assert_allclose(belief, s['belief_hot'], rtol=0, atol=1e-12)
        back = distance(g, position, g.start_index)
        elapsed += back/5
        travel += back
        near(back, s['return_m'])
        near(elapsed, s['elapsed_s'])
        near(travel, s['distance_m'])
        near(elapsed, travel/5+20*len(used)+len(s['planning_events']))
        assert s['position_index'] == g.start_index and elapsed <= r['budget_s']+1e-7
        if not dynamic:
            assert len(s['planning_events']) == 1 and not decisions
            assert used == s['planning_events'][0]['planned_route']
            static_checked += 1
        arrived = [h for h in s['history'] if target[h['cell_index']]]
        detected = [h for h in arrived if h['observed_positive']]
        total = int(target.sum())
        score = {'detected_target_cells': len(detected), 'visited_target_cells': len(arrived),
                 'true_target_cells': total, 'observations': len(used),
                 'detected_target_recall': len(detected)/total if total else None,
                 'visited_target_recall': len(arrived)/total if total else None,
                 'false_positive_observations': sum(not target[h['cell_index']] and h['observed_positive'] for h in s['history']),
                 'first_target_arrival_s': arrived[0]['arrival_s'] if arrived else None,
                 'first_correct_detection_s': detected[0]['observation_complete_s'] if detected else None,
                 'first_detection_or_budget_s': detected[0]['observation_complete_s'] if detected else r['budget_s'],
                 'no_correct_detection': not bool(detected), 'total_distance_m': travel, 'total_time_s': elapsed,
                 'planning_calls': len(s['planning_events']), 'planning_attempts': len(decisions),
                 'accepted_proposals': sum(d['accepted'] for d in decisions),
                 'rejected_proposals': sum(not d['accepted'] for d in decisions),
                 'route_changes': sum(d['route_changed'] for d in decisions),
                 'tail_drops_on_rejection': sum(d['dropped_tail_if_rejected'] for d in decisions if not d['accepted']),
                 'skipped_unaffordable_checkpoints': len(skips)}
        assert set(score) == set(r['evaluation'])
        for name, value in score.items():
            near(r['evaluation'][name], value)
            near(lookup[key][name], value)
    assert len(seen) == 6480 and static_checked == 2160
    means = read(OUT/'condition_means.json')
    assert len(means) == 324
    for m in means:
        rs = [r for r in metrics if all(r[k] == m[k] for k in GROUP)]
        assert len(rs) == m['runs'] == 20
        for name in FIELDS:
            near(m[name], sum(r[name] for r in rs)/20)
        near(m['no_correct_detection_rate'], sum(r['no_correct_detection'] for r in rs)/20)
    pairs = read(OUT/'paired_comparisons.json')
    assert len(pairs) == 486
    for p in pairs:
        rs = sorted([r for r in metrics if r['policy'] == p['first'] and all(r[k] == p[k] for k in GROUP[:-1])], key=lambda r: r['seed'])
        other = [lookup[r['world_id'], r['noise'], r['budget_s'], p['second']] for r in rs]
        ds = [a['detected_target_cells']-b['detected_target_cells'] for a, b in zip(rs, other)]
        assert ds == p['individual_detected_deltas']
        near(p['mean_detected_delta'], sum(ds)/20)
        near(p['median_detected_delta'], float(np.median(ds)))
        assert p['win_tie_loss'] == [sum(d>0 for d in ds), sum(d == 0 for d in ds), sum(d<0 for d in ds)]
        for name in FIELDS:
            if name != 'detected_target_cells':
                near(p['mean_'+name+'_delta'], sum(a[name]-b[name] for a, b in zip(rs, other))/20)
    summary = read(OUT/'summary.json')
    assert len(summary['comparison_counts']) == 36 and len(summary['pooled_noise_means']) == 18
    for row in summary['comparison_counts']:
        ps = [p for p in pairs if p['budget_s'] == 1200 and p['first'] == row['first'] and p['second'] == row['second']
              and (row['noise'] == 'all' or p['noise'] == row['noise'])]
        assert row['conditions'] == len(ps)
        assert row['positive_zero_negative'] == [sum(p['mean_detected_delta']>0 for p in ps), sum(p['mean_detected_delta'] == 0 for p in ps), sum(p['mean_detected_delta']<0 for p in ps)]
        assert row['win_tie_loss'] == np.sum([p['win_tie_loss'] for p in ps], axis=0).tolist()
        near(row['mean_detected_delta'], sum(p['mean_detected_delta'] for p in ps)/len(ps))
    for row in summary['pooled_noise_means']:
        rs = [r for r in metrics if r['budget_s'] == 1200 and r['noise'] == row['noise'] and r['policy'] == row['policy']]
        assert len(rs) == row['runs'] == 120
        for name in FIELDS:
            near(row[name], sum(r[name] for r in rs)/120)
        near(row['no_correct_detection_rate'], sum(r['no_correct_detection'] for r in rs)/120)
    examples = read(OUT/'examples.json')
    assert len(examples) == 12
    for r in examples:
        assert r['seed'] == SEEDS[0] and r['mode'] == 'aligned' and r['noise'] == 'moderate' and r['budget_s'] == 1200
        assert r['evaluation'] == {k: lookup[r['world_id'], r['noise'], r['budget_s'], r['policy']][k] for k in r['evaluation']}
    runtime = read(OUT/'runtime.json')
    assert runtime['observations'] == steps_checked and runtime['planning_calls'] == calls_checked
    archive = ROOT/'output/感知检查_阶段汇报_20261028/归档校验.json'
    for name, sha in read(archive)['files'].items():
        checked(archive.parent/name, sha)
    verify_previous()
    save(OUT/'audit_report.json', {'status': 'PASS', 'worlds_regenerated': 120, 'routes_replayed': 6480,
         'paid_observations_replayed': steps_checked, 'paid_planner_calls_replayed': calls_checked,
         'proposal_decisions_replayed': decisions_checked, 'paid_rejections_verified': rejected_checked,
         'static_trajectories_checked': static_checked, 'condition_means_checked': 324, 'paired_comparisons_checked': 486,
         'budget_violations': 0, 'free_candidate_solves': 0, 'advisor_archive_files_unchanged': len(read(archive)['files']),
         'old_frozen_outputs_and_sources_unchanged': True,
         'boundary': 'Full-policy tests on new draws of the same model and geometry;not external/real-disaster confirmation.',
         'replay': 'Worlds,history,costs,posterior,acceptance,moment covariance and scores independently reconstructed. Candidate choices use frozen V2 solver with rebuilt public inputs.'})
    print(json.dumps({'audit': 'PASS', 'routes': 6480, 'observations': steps_checked,
                      'paid_calls': calls_checked, 'paid_rejections': rejected_checked}), flush=True)


if __name__ == '__main__':
    assert not (OUT/'completion_receipt.json').exists()
    main()
