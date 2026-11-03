"""Independently replay each saved action from paid labels and score equations."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_boone_context_experiment import dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    read = lambda p: json.loads(p.read_text())
    f = read(root/'output/disaster_route_boone_context_run_v2/satellite_features.json')
    ids, xy = f['building_ids'], np.array(f['xy_m'])
    initial = read(root/'output/disaster_route_boone_context_run_v2/vlm_initial_scores.json')
    p0 = np.array([initial[u] for u in ids])
    similarity = np.array(read(root/'output/disaster_route_boone_structure_check_v1/structure_features.json')['similarity'])
    references = read(root/'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json')
    labels = {r['building_id']: r['view_binary'] for r in references['rows']}
    episodes = read(out/'episodes.json')
    second_feedback_diagnostic = []
    update_checks = 0
    for row in episodes:
        visited, scores = {row['start_index']}, p0.copy()
        position, distance, hits = row['start_index'], 0., 0
        assert len(set(row['route_indices'])) == 3
        assert row['label_access_indices'] == row['route_indices']
        assert len(row['observations'])==3 and len(row['updates'])==2
        for step, (action, update) in enumerate(zip(row['actions'], row['updates'])):
            assert update['visited_indices'] == sorted(visited)
            assert action['observed_history_indices'] == sorted(visited)
            candidates = sorted(set(range(len(ids)))-visited)
            assert action['candidate_indices'] == candidates
            np.testing.assert_allclose(update['scores_before'], scores, atol=1e-12, rtol=0)
            y = labels[ids[position]]['uas']
            residual = y-scores[position]
            assert abs(residual-update['residual']) < 1e-12
            assert row['observations'][step]['label']==y
            expected = scores.copy()
            gated = scores.copy()
            for i in candidates:
                d = math.dist(xy[position], xy[i])
                w = math.exp(-d*d/(2*50**2))
                gated[i] = min(1., max(0., scores[i]+.5*w*similarity[position,i]*residual))
                scale = .5*similarity[position,i] if row['method']=='pearson' else update['gain']
                expected[i] = min(1., max(0., scores[i]+scale*w*residual))
            np.testing.assert_allclose(expected, update['scores_after'], atol=1e-12, rtol=0)
            if row['method']=='matched_spatial':
                assert abs(np.abs(expected[candidates]-scores[candidates]).mean()-
                           np.abs(gated[candidates]-scores[candidates]).mean()) < 1e-12
            def choose(values):
                return min(candidates, key=lambda i: (-values[i]/(max(math.dist(xy[position],xy[i]),1e-6)
                                                                  if row['selector']=='score_per_distance' else 1.), i))
            selected = choose(expected)
            assert selected == action['target_index']
            if step==1:
                no_feedback_selected = choose(scores)
                second_feedback_diagnostic.append({'selector': row['selector'], 'method': row['method'],
                                                  'start_display': row['start_display'],
                                                  'with_second_feedback': f'B{selected+1:02d}',
                                                  'without_second_feedback': f'B{no_feedback_selected+1:02d}',
                                                  'choice_changed': selected!=no_feedback_selected})
            distance += math.dist(xy[position], xy[selected])
            hits += labels[ids[selected]]['uas']
            visited.add(selected)
            position, scores = selected, expected
            update_checks += 1
        assert abs(distance-row['outbound_distance_m']) < 1e-10
        assert abs(distance+math.dist(xy[position],xy[row['start_index']])-row['closed_distance_m']) < 1e-10
        assert hits == row['new_visit_hits']
        assert row['observations'][2]['label']==labels[ids[position]]['uas']
    summary = read(out/'summary.json')
    for row in summary['table']:
        subset = [r for r in episodes if (r['selector'],r['method'])==(row['selector'],row['method'])]
        assert len(subset)==12
        assert row['two_step_hit_sum']==sum(r['new_visit_hits'] for r in subset)
        assert row['two_step_hit_sum']==row['step1_hit_sum']+row['step2_hit_sum']
        assert abs(row['closed_distance_sum_m']-sum(r['closed_distance_m'] for r in subset)) < 1e-9
    receipt_count = 0
    for name in ('disaster_route_boone_context_run_v2', 'disaster_route_boone_change6_v1',
                 'disaster_route_boone_strength_control_v1', 'disaster_route_boone_structure_check_v1'):
        folder = root/'output'/name
        for path, digest in read(folder/'completion_receipt.json')['files'].items():
            assert sha(folder/path)==digest
            receipt_count += 1
    for name, digest in read(out/'execution_recipe.json')['input_hashes'].items():
        assert sha(root/name)==digest
    dump(out/'second_feedback_diagnostic.json', second_feedback_diagnostic)
    dump(out/'audit.json', {'status':'PASS','episodes_replayed':len(episodes),
                           'score_updates_and_decisions_recomputed':update_checks,
                           'paid_label_reads_checked':sum(len(r['observations']) for r in episodes),
                           'old_sealed_files_unchanged':receipt_count})
    print((out/'audit.json').read_text())
    for selector in ('score_per_distance', 'score'):
        for method in ('pearson', 'matched_spatial'):
            rows=[r for r in second_feedback_diagnostic if (r['selector'],r['method'])==(selector,method)]
            print(selector,method,'second-feedback changed choices',sum(r['choice_changed'] for r in rows))


if __name__ == '__main__':
    main()
