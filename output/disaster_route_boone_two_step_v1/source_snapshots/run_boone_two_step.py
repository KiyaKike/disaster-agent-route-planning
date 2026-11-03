"""Fixed two-visit sequential stress test of the frozen Boone Pearson kernel."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_boone_context_experiment import dump, sha
from run_boone_strength_control import matched_spatial


METHODS = ('pearson', 'matched_spatial', 'spatial', 'no_update')
SELECTORS = ('score_per_distance', 'score')  # distance-aware is primary


def update_scores(current, anchor, feedback, visited, xy, similarity, method):
    mask = np.array([i not in visited for i in range(len(current))])
    spatial = np.exp(-np.sum((xy-xy[anchor])**2, axis=1)/(2*50**2))
    residual = float(feedback-current[anchor])
    hypothetical = np.clip(current+.5*spatial*similarity[anchor]*residual, 0, 1)
    desired = float(np.abs(hypothetical[mask]-current[mask]).mean())
    gain = .5
    if method == 'pearson':
        proposed = hypothetical
    elif method == 'matched_spatial':
        proposed, gain, desired = matched_spatial(current, spatial, residual, mask, hypothetical)
    elif method == 'spatial':
        proposed = np.clip(current+.5*spatial*residual, 0, 1)
    elif method == 'no_update':
        proposed, gain = current.copy(), 0.
    else:
        raise ValueError(method)
    updated = current.copy()
    updated[mask] = proposed[mask]
    actual = float(np.abs(updated[mask]-current[mask]).mean())
    if method == 'matched_spatial':
        assert abs(actual-desired) < 1e-12
    return updated, {'anchor_index': anchor, 'feedback': int(feedback),
                     'anchor_score_before': float(current[anchor]), 'residual': residual,
                     'visited_indices': sorted(visited), 'gain': gain,
                     'hypothetical_pearson_mean_change': desired, 'actual_mean_change': actual,
                     'scores_before': current.tolist(), 'scores_after': updated.tolist()}


def run_episode(p0, xy, similarity, start, method, selector, reveal):
    """The policy can access labels only through reveal on its actual path."""
    scores, current, visited = p0.copy(), start, {start}
    feedback = reveal(start)
    observations = [{'index': start, 'label': int(feedback), 'initial_anchor': True}]
    updates, actions, route, hits, distance = [], [], [start], 0, 0.
    for step in (1, 2):
        scores, update = update_scores(scores, current, feedback, visited, xy, similarity, method)
        updates.append(update)
        options = [i for i in range(len(scores)) if i not in visited]
        distances = {i: float(np.linalg.norm(xy[i]-xy[current])) for i in options}
        utilities = {i: float(scores[i])/(max(distances[i], 1e-6) if selector=='score_per_distance' else 1.)
                     for i in options}
        target = min(options, key=lambda i: (-utilities[i], i))
        distance += distances[target]
        actions.append({'step': step, 'from_index': current, 'target_index': target,
                        'candidate_indices': options, 'candidate_utilities': {str(i): utilities[i] for i in options},
                        'score': float(scores[target]), 'leg_m': distances[target],
                        'observed_history_indices': sorted(visited)})
        current = target
        visited.add(target)
        route.append(target)
        feedback = reveal(target)
        observations.append({'index': target, 'label': int(feedback), 'initial_anchor': False})
        hits += feedback
    back = float(np.linalg.norm(xy[current]-xy[start]))
    return {'method': method, 'selector': selector, 'start_index': start,
            'route_indices': route, 'new_visit_hits': int(hits),
            'outbound_distance_m': distance, 'return_distance_m': back,
            'closed_distance_m': distance+back, 'new_observations': 2,
            'observations': observations, 'updates': updates, 'actions': actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory')
    read = lambda p: json.loads(p.read_text())
    source = root/'output/disaster_route_boone_context_run_v2'
    structure = root/'output/disaster_route_boone_structure_check_v1'
    refpath = root/'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json'
    f = read(source/'satellite_features.json')
    ids, xy = f['building_ids'], np.array(f['xy_m'])
    similarity = np.array(read(structure/'structure_features.json')['similarity'])
    initial = read(source/'vlm_initial_scores.json')
    p0 = np.array([initial[u] for u in ids])
    labels = {r['building_id']: r['view_binary'] for r in read(refpath)['rows']}
    distances = np.linalg.norm(xy[:, None]-xy[None, :], axis=-1)
    # Independent all-12-anchor episodes; no memory crosses episode boundaries.
    episodes = []
    for selector in SELECTORS:
        for start in range(len(ids)):
            for method in METHODS:
                accesses = []
                def reveal(index):
                    assert index not in accesses
                    accesses.append(index)
                    return labels[ids[index]]['uas']
                row = run_episode(p0, xy, similarity, start, method, selector, reveal)
                assert accesses == row['route_indices']
                row.update(start_display=f'B{start+1:02d}',
                           route_display=[f'B{i+1:02d}' for i in row['route_indices']],
                           label_access_indices=accesses,
                           noaa_evaluator_hits=sum(labels[ids[i]]['crewed'] for i in row['route_indices'][1:]))
                episodes.append(row)
    # First update must exactly reproduce the frozen single-anchor experiment.
    frozen = {}
    for r in read(structure/'prediction_rows.json'):
        frozen.setdefault((r['anchor_id'], r['method']), {})[r['target_id']] = r['score']
    first_checks = 0
    for row in episodes:
        if row['method'] not in ('pearson', 'matched_spatial'):
            continue
        method = 'S5_pearson_change' if row['method']=='pearson' else 'C5_matched_pearson'
        for i, uid in enumerate(ids):
            if i != row['start_index']:
                assert abs(row['updates'][0]['scores_after'][i]-frozen[ids[row['start_index']], method][uid]) < 1e-12
                first_checks += 1
    summary, paired = [], []
    for selector in SELECTORS:
        for method in METHODS:
            subset = [r for r in episodes if r['selector']==selector and r['method']==method]
            summary.append({'selector': selector, 'method': method,
                            'two_step_hit_sum': sum(r['new_visit_hits'] for r in subset),
                            'step1_hit_sum': sum(r['observations'][1]['label'] for r in subset),
                            'step2_hit_sum': sum(r['observations'][2]['label'] for r in subset),
                            'outbound_distance_sum_m': sum(r['outbound_distance_m'] for r in subset),
                            'closed_distance_sum_m': sum(r['closed_distance_m'] for r in subset),
                            'noaa_evaluator_hit_sum': sum(r['noaa_evaluator_hits'] for r in subset)})
        for start in range(len(ids)):
            pair = {r['method']: r for r in episodes if r['selector']==selector and r['start_index']==start}
            a, b = pair['pearson'], pair['matched_spatial']
            paired.append({'selector': selector, 'start_display': f'B{start+1:02d}',
                           'pearson_route': a['route_display'], 'matched_route': b['route_display'],
                           'pearson_hits': a['new_visit_hits'], 'matched_hits': b['new_visit_hits'],
                           'hit_delta': a['new_visit_hits']-b['new_visit_hits'],
                           'outbound_distance_delta_m': a['outbound_distance_m']-b['outbound_distance_m'],
                           'closed_distance_delta_m': a['closed_distance_m']-b['closed_distance_m']})
    verdicts = []
    for selector in SELECTORS:
        rows = [r for r in paired if r['selector']==selector]
        net = sum(r['hit_delta'] for r in rows)
        losses = sum(r['hit_delta']<0 for r in rows)
        verdicts.append({'selector': selector, 'net_hits': net,
                         'wins': sum(r['hit_delta']>0 for r in rows),
                         'ties': sum(r['hit_delta']==0 for r in rows), 'losses': losses,
                         'user_gate_pass': net>1 and losses==0})
    primary = next(r for r in verdicts if r['selector']=='score_per_distance')
    out.mkdir(parents=True)
    dump(out/'summary.json', {'table': summary, 'paired_verdicts': verdicts,
                             'primary_selector': 'score_per_distance',
                             'decision': 'ELIGIBLE_FOR_EXTERNAL_VALIDATION' if primary['user_gate_pass'] else 'STOP_BOONE_VISUAL_KERNEL_SEARCH',
                             'episodes': len(episodes), 'new_model_calls': 0,
                             'fixed_travel_budget_test': False})
    dump(out/'episodes.json', episodes)
    dump(out/'paired_results.json', paired)
    dump(out/'geometry.json', {'building_ids': ids, 'xy_m': xy.tolist(), 'distance_matrix_m': distances.tolist(),
                              'positive_pairs': [{'a': f'B{a+1:02d}', 'b': f'B{b+1:02d}', 'distance_m': float(distances[a,b])}
                                                 for a,b in ((2,6),(2,8),(6,8))],
                              'coordinate_scope': 'Original UTM footprint-vertex centers; approximate straight-line travel, not surveyed flight paths'})
    dump(out/'execution_recipe.json', {
        'protocol': 'All12 initial anchors;2 new visits; initial anchor hit excluded; source labels not field truth',
        'primary_selector': 'score_per_distance', 'diagnostic_selector': 'score',
        'update': 'Sequential current score + gain * spatial * optional Pearson gate * (new source label - current anchor score)',
        'space_scale_m': 50, 'pearson_gain': .5, 'feature_kernel_refit': False,
        'matching': 'At every update use own current state, own visited mask and newly paid label; match own hypothetical Pearson magnitude',
        'minimum_distance_denominator_m': 1e-6, 'tie_break': 'sorted building ID',
        'closed_distance': 'A0->A1->A2->A0, reported separately from outbound; no fixed travel budget',
        'user_stop_rule': 'Primary cumulative gain<=1 OR any lower-hit paired episode => stop; no tuning or region expansion on failure',
        'initial_evidence': 'A0 source label supplied equally before2-visit comparison; no initial-anchor hit credit',
        'unvisited_labels': 'Policy reads only actual A0,A1,A2 labels; A2 label arrives after final choice',
        'script_sha256': sha(Path(__file__)),
        'input_hashes': {str(p.relative_to(root)): sha(p) for p in
                         (source/'satellite_features.json', source/'vlm_initial_scores.json',
                          structure/'structure_features.json', refpath)}})
    dump(out/'first_update_checks.json', {'frozen_target_scores_matched': first_checks})
    with (out/'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    print((out/'summary.json').read_text())


if __name__ == '__main__':
    main()
