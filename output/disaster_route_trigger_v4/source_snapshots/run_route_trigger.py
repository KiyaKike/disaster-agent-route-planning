"""Prospectively freeze a bounded full-policy test on new simulated seed IDs."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import subprocess
import sys
import time

import numpy as np

from run_route_loop import ROOT, HERE, DATA, maps, read, save, digest, checked
from run_route_snapshot import OUT as SNAPSHOT, BASE, verify as verify_snapshot, verify_parents
from route_core import NOISE, WORLD_MODES, simulate_targets
from route_factor_core import Planner
from route_snapshot_core import PrivateWorld
from route_trigger_core import POLICIES, execute, score

OUT = ROOT/'output/disaster_route_trigger_v4'
SOURCES = ('route_trigger_core.py', 'test_route_trigger.py', 'run_route_trigger.py',
           'audit_route_trigger.py', 'report_route_trigger.py')
SEEDS = tuple(range(2026110200, 2026110220))
BUDGETS = (600, 1200, 1800)
GROUP = ('region', 'mode', 'noise', 'budget_s', 'policy')
PAIRS = (('always', 'post_static'), ('periodic_5', 'post_static'), ('gain_5', 'post_static'),
         ('one_sd_5', 'post_static'), ('one_sd_5', 'gain_5'), ('one_sd_5', 'periodic_5'),
         ('one_sd_5', 'always'), ('gain_5', 'periodic_5'), ('post_static', 'pre_static'))
FIELDS = ('detected_target_cells', 'visited_target_cells', 'observations', 'false_positive_observations',
          'first_detection_or_budget_s', 'total_distance_m', 'total_time_s', 'planning_calls',
          'planning_attempts', 'accepted_proposals', 'rejected_proposals', 'route_changes',
          'tail_drops_on_rejection', 'skipped_unaffordable_checkpoints')


def verify_previous():
    verify_snapshot()
    verify_parents()
    for name, sha in read(SNAPSHOT/'completion_receipt.json')['files'].items():
        checked(SNAPSHOT/name, sha)


def verify():
    p = read(OUT/'protocol.json')
    for name, sha in p['source_sha256'].items():
        checked(HERE/name, sha)
    for name, sha in p['input_sha256'].items():
        checked(name, sha)
    checked(OUT/'test_log.txt', p['test_log_sha256'])
    return p


def plan():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    verify_previous()
    old_seeds = {w['seed'] for w in read(BASE/'synthetic_worlds.json')}
    assert not set(SEEDS) & old_seeds
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                            '-p', 'test_route*.py', '-v'], capture_output=True, text=True)
    (OUT/'test_log.txt').write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0 and 'Ran 53 tests' in tests.stdout+tests.stderr
    inputs = {str(p): digest(p) for p in (DATA, SNAPSHOT/'completion_receipt.json',
             SNAPSHOT/'prospective_next_scope.json', BASE/'public_geometry.json')}
    for name, sha in read(BASE/'protocol.json')['external_sha256'].items():
        inputs[name] = sha
    p = {'stage': 'PAID_REPLANNING_TRIGGER_FULL_POLICY_V4', 'client_date': '2026-11-02',
         'source_sha256': {n: digest(HERE/n) for n in SOURCES}, 'input_sha256': inputs,
         'test_log_sha256': digest(OUT/'test_log.txt'),
         'question': 'Do simple paid mean/uncertainty acceptance rules improve complete inspection tours on new simulation draws?',
         'selection': {'seed_ids': SEEDS, 'prior_seed_ids': sorted(old_seeds), 'seed_ids_disjoint': True,
                       'worlds': 120, 'routes': 6480, 'regions': list(maps()), 'prior_modes': WORLD_MODES,
                       'noise': NOISE, 'budgets_s': BUDGETS, 'primary_budget_s': 1200,
                       'same_generator_and_geometry_as_development': True,
                       'meaning': 'Prospective new Monte Carlo draws after rule freeze,not new disasters or external validation.'},
         'policies': {'pre_static': 'Uniform prior .2;one initial plan then fixed route.',
                      'post_static': 'Postcue prior;one initial plan then fixed route.',
                      'always': 'After every paid observation,attempt and accept a new posterior suffix if another paid inspection is affordable.',
                      'periodic_5': 'Same,but attempt and accept only after every5 observations.',
                      'gain_5': 'Same paid checkpoints;accept proposal iff model expected future correct-detection gain over pre-attempt old suffix >1e-12.',
                      'one_sd_5': 'Same paid checkpoints;accept iff mean gain minus one model predictive SD >1e-12. No fitted multiplier.'},
         'shared_solver': 'Unchanged V2 insertion+2opt,time_only. Every call plans a committed feasible suffix with future-call cost0;future attempts pay when made. This common execution convention differs from V2 always-online future-fee reservation,so comparisons are within V4.',
         'costs': {'speed_m_s': 5, 'observe_s': 20, 'planning_latency_s': 1,
                   'rule': 'Charge initial and every attempted candidate call,including rejects. No free trial solve. Actual CPU wall time reported separately. Every tour returns within budget.',
                   'rejection': 'After paying,keep the old suffix with deterministic tail removal until it remains return-feasible;no fee refund.',
                   'unaffordable': 'Skip a checkpoint only if no unvisited cell can be inspected and returned from after a1s attempt;keep feasible old suffix.',
                   'ending': 'When remaining suffix is empty,return;do not add extra off-schedule proposals.'},
         'forecast': {'mean': 'Difference of sums sensitivity*(.02+.58*q_block) over proposed-after-fee vs old-before-fee suffix;shared cells cancel.',
                      'variance': 'Latent-hot/cold mixture:within-block conditional cell variance plus shared-latent covariance. Known sensitivity and FP;posterior from paid bits only.',
                      'boundary': 'One SD is a heuristic,not a one-sided confidence/risk guarantee. Mean/variance may be misspecified under reversed or uninformative priors. No duplicate transfer penalty.'},
         'primary': 'one_sd_5 minus gain_5,correct synthetic target detections per18 region/prior/noise conditions at1200s;report all conditions and paired wins/ties/losses.',
         'secondary': ['all dynamic methods vs post_static', 'vs always and periodic_5', 'pre vs post static',
                       '600/1800 sensitivity', 'coverage', 'first arrival/detection and failure rate', 'distance/time',
                       'attempts vs accepts vs route changes vs tail drops'],
         'advance_gate': 'Call no universal winner unless it weakly dominates all comparators on mean detections across all18 primary conditions,with no worse mean total time,planning calls or restricted first-detection time. Otherwise report tradeoffs;do not tune threshold or period in this run.',
         'pairing': 'Shared targets and per-cell uniform draws across policies,budgets and noise. Prior modes share generator streams. Conditions are not independent disasters;no unqualified significance claim.',
         'information': 'Fixed Python policy interfaces receive geometry,paid public history and known likelihood. Private targets and future bits stay in environment/scorer. No adversarial process sandbox.',
         'examples': 'Both regions,first new seed2026110200,aligned/moderate,1200s;all6 methods,without favorable-case replacement.',
         'failure_handling': 'No target denominator ->null recall;no discovery ->null first time,report failure rate and budget-restricted first-detection time.',
         'no_posthoc_changes': True, 'new_training': False, 'new_perception_calls': 0,
         'gpu_rental': False, 'new_real_labels': 0}
    save(OUT/'protocol.json', p)
    save(OUT/'selection_receipt.json', {'protocol_sha256': digest(OUT/'protocol.json'),
                                      'world_generation_executed': False, 'route_execution_started': False})
    print(json.dumps({'protocol': 'FROZEN', 'tests': 53, 'new_seed_ids': len(SEEDS), 'routes': 6480}), flush=True)


def generate_worlds():
    worlds = []
    for region, g in maps().items():
        for mode in WORLD_MODES:
            for seed in SEEDS:
                generator_seed = seed+(100000 if region == '南侧建筑区' else 0)
                w = simulate_targets(g, mode, generator_seed)
                worlds.append({'world_id': f'{region}/{mode}/{seed}', 'region': region, 'mode': mode,
                               'seed': seed, 'generator_seed': generator_seed,
                               'synthetic_truth_not_realPS': True,
                               'target_ids': [g.ids[i] for i in np.flatnonzero(w['targets'])],
                               'hot_block_indices': np.flatnonzero(w['hot_blocks']).tolist(),
                               'true_hot_probabilities': w['true_hot_probabilities'].tolist(),
                               'uniform_observation_draws': w['uniform_observation_draws'].tolist()})
    return worlds


def run():
    verify()
    assert not (OUT/'routes.jsonl.gz').exists() and not (OUT/'synthetic_worlds.json').exists()
    started = time.monotonic()
    mm = maps()
    worlds = generate_worlds()
    save(OUT/'synthetic_worlds.json', worlds)
    planner = Planner()
    metrics, examples = [], []
    with gzip.open(OUT/'routes.jsonl.gz', 'wt', encoding='utf-8') as f:
        for wi, w in enumerate(worlds):
            g = mm[w['region']]
            target_ids = set(w['target_ids'])
            target = np.array([sid in target_ids for sid in g.ids], bool)
            draw = np.array(w['uniform_observation_draws'])
            for noise, likelihood in NOISE.items():
                se, fp = likelihood
                token = hashlib.sha256((w['world_id']+'|'+noise).encode()).hexdigest()
                private = PrivateWorld(token, target, draw < np.where(target, se, fp), likelihood)
                for budget in BUDGETS:
                    for policy in POLICIES:
                        r = execute(g, private, policy, planner, budget)
                        metadata = {k: w[k] for k in ('world_id', 'region', 'mode', 'seed')}
                        r.update(metadata, noise=noise)
                        r['evaluation'] = score(r, private)
                        f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
                        metrics.append({**metadata, 'noise': noise, 'budget_s': budget, 'policy': policy,
                                        **r['evaluation'], 'measured_planner_wall_s': r['measured_planner_wall_s']})
                        if w['seed'] == SEEDS[0] and w['mode'] == 'aligned' and noise == 'moderate' and budget == 1200:
                            examples.append(r)
            if (wi+1) % 20 == 0:
                print(json.dumps({'worlds': wi+1, 'routes': len(metrics),
                                  'elapsed_s': round(time.monotonic()-started, 1)}), flush=True)
    assert len(worlds) == 120 and len(metrics) == 6480 and len(examples) == 12
    save(OUT/'metrics.json', metrics)
    save(OUT/'examples.json', examples)
    save(OUT/'runtime.json', {'status': 'COMPLETE', 'worlds': 120, 'routes': 6480,
                             'loop_seconds': time.monotonic()-started,
                             'observations': sum(r['observations'] for r in metrics),
                             'planning_calls': sum(r['planning_calls'] for r in metrics),
                             'sum_measured_planner_wall_s': sum(r['measured_planner_wall_s'] for r in metrics),
                             'gpu_rented': False, 'new_perception_calls': 0, 'new_training': False})


def summarize():
    verify()
    rows = read(OUT/'metrics.json')
    lookup = {(r['world_id'], r['noise'], r['budget_s'], r['policy']): r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in GROUP)].append(r)
    means, pairs = [], []
    for group, rs in sorted(groups.items()):
        assert len(rs) == 20
        means.append({**dict(zip(GROUP, group)), 'runs': 20,
                      **{k: float(np.mean([r[k] for r in rs])) for k in FIELDS},
                      'no_correct_detection_rate': sum(r['no_correct_detection'] for r in rs)/20})
        if group[-1] != 'post_static':
            continue
        base = sorted(rs, key=lambda r: r['seed'])
        for first, second in PAIRS:
            a = [lookup[r['world_id'], r['noise'], r['budget_s'], first] for r in base]
            b = [lookup[r['world_id'], r['noise'], r['budget_s'], second] for r in base]
            delta = [x['detected_target_cells']-y['detected_target_cells'] for x, y in zip(a, b)]
            pairs.append({**dict(zip(GROUP[:-1], group[:-1])), 'first': first, 'second': second,
                          'paired_runs': 20, 'individual_detected_deltas': delta,
                          'mean_detected_delta': float(np.mean(delta)), 'median_detected_delta': float(np.median(delta)),
                          'win_tie_loss': [sum(d>0 for d in delta), sum(d == 0 for d in delta), sum(d<0 for d in delta)],
                          **{'mean_'+k+'_delta': float(np.mean([x[k]-y[k] for x, y in zip(a, b)]))
                             for k in FIELDS if k != 'detected_target_cells'},
                          'mean_no_detection_rate_delta': sum(int(x['no_correct_detection'])-int(y['no_correct_detection']) for x, y in zip(a, b))/20})
    summaries = []
    for first, second in PAIRS:
        for noise in ('all', *NOISE):
            ps = [p for p in pairs if p['budget_s'] == 1200 and p['first'] == first and p['second'] == second
                  and (noise == 'all' or p['noise'] == noise)]
            summaries.append({'first': first, 'second': second, 'noise': noise, 'conditions': len(ps),
                              'positive_zero_negative': [sum(p['mean_detected_delta']>0 for p in ps),
                                                         sum(p['mean_detected_delta'] == 0 for p in ps),
                                                         sum(p['mean_detected_delta']<0 for p in ps)],
                              'win_tie_loss': np.sum([p['win_tie_loss'] for p in ps], axis=0).tolist(),
                              'mean_detected_delta': float(np.mean([p['mean_detected_delta'] for p in ps]))})
    aggregates = []
    for policy in POLICIES:
        for noise in NOISE:
            rs = [r for r in rows if r['budget_s'] == 1200 and r['policy'] == policy and r['noise'] == noise]
            aggregates.append({'policy': policy, 'noise': noise, 'runs': len(rs),
                               **{k: float(np.mean([r[k] for r in rs])) for k in FIELDS},
                               'no_correct_detection_rate': sum(r['no_correct_detection'] for r in rs)/len(rs)})
    save(OUT/'condition_means.json', means)
    save(OUT/'paired_comparisons.json', pairs)
    save(OUT/'summary.json', {'primary_budget_s': 1200, 'comparison_counts': summaries, 'pooled_noise_means': aggregates,
                            'boundary': 'Equal-weight summaries over shared synthetic worlds;not independent disaster-level evidence.'})
    print(json.dumps({'summarized': True, 'condition_means': len(means), 'comparisons': len(pairs)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('plan', 'run', 'summarize'))
    args = p.parse_args()
    assert not (OUT/'completion_receipt.json').exists()
    {'plan': plan, 'run': run, 'summarize': summarize}[args.command]()
