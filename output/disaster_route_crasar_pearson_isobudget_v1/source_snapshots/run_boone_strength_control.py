"""Match clipped update magnitude while removing target-specific visual gating."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_boone_context_experiment import independent_ap
from run_boone_context_experiment import dump, metrics, sha


def matched_spatial(p0, spatial, residual, target_mask, gated_scores):
    """Match mean absolute *clipped* score change, without target labels."""
    desired = float(np.mean(np.abs(gated_scores[target_mask] - p0[target_mask])))
    if desired < 1e-15:
        return p0.copy(), 0., desired
    lo, hi = 0., .5
    maximum = np.clip(p0 + hi * spatial * residual, 0, 1)
    if np.mean(np.abs(maximum[target_mask] - p0[target_mask])) < desired - 1e-12:
        raise ValueError('Gated magnitude exceeds the original spatial update')
    for _ in range(80):
        mid = (lo + hi) / 2
        scores = np.clip(p0 + mid * spatial * residual, 0, 1)
        if np.mean(np.abs(scores[target_mask] - p0[target_mask])) < desired:
            lo = mid
        else:
            hi = mid
    gain = (lo + hi) / 2
    scores = np.clip(p0 + gain * spatial * residual, 0, 1)
    return scores, gain, desired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new result directory')
    source = root / 'output/disaster_route_boone_context_run_v2'
    change = root / 'output/disaster_route_boone_change6_v1'
    reference = root / 'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json'
    read = lambda p: json.loads(p.read_text())
    f = read(source / 'satellite_features.json')
    ids, xy = f['building_ids'], np.array(f['xy_m'])
    initial = read(source / 'vlm_initial_scores.json')
    p0 = np.array([initial[uid] for uid in ids])
    labels = {r['building_id']: r['view_binary'] for r in read(reference)['rows']}
    frozen = {}
    original_rows = read(change / 'prediction_rows.json')
    for row in original_rows:
        frozen.setdefault((row['anchor_id'], row['method']), {})[row['target_id']] = row['score']
    pairs = [('S2_visual_spatial', 'C2_matched36'),
             ('S3_change6', 'C3_matched6'),
             ('S4_change6_old_tau', 'C4_matched6_old_tau')]
    records, predictions, diagnostics, ranks = [], list(original_rows), [], []
    for a, anchor in enumerate(ids):
        mask = np.arange(len(ids)) != a
        targets = [uid for uid in ids if uid != anchor]
        spatial = np.exp(-np.sum((xy - xy[a])**2, axis=1) / (2 * 50**2))
        residual = labels[anchor]['uas'] - p0[a]
        current = {method: frozen[anchor, method].copy() for method in
                   ('S0_no_update', 'S1_spatial', 'S2_visual_spatial', 'S3_change6', 'S4_change6_old_tau')}
        for gated, control in pairs:
            gated_scores = np.array([p0[i] if i == a else current[gated][uid]
                                     for i, uid in enumerate(ids)])
            scores, gain, desired = matched_spatial(p0, spatial, residual, mask, gated_scores)
            actual = float(np.mean(np.abs(scores[mask] - p0[mask])))
            assert abs(actual - desired) < 1e-12
            current[control] = dict(zip(targets, map(float, scores[mask])))
            diagnostics.append({'anchor_id': anchor, 'anchor_display': f'B{a+1:02d}',
                                'gated_method': gated, 'control_method': control,
                                'gain': gain, 'desired_mean_abs_change': desired,
                                'control_mean_abs_change': actual, 'absolute_error': abs(actual-desired)})
            predictions.extend({'anchor_id': anchor, 'target_id': uid, 'method': control,
                                'score': current[control][uid]} for uid in targets)
        orders = {m: sorted(targets, key=lambda uid: (-s[uid], uid)) for m, s in current.items()}
        for method, scores in current.items():
            for view in ('uas', 'crewed'):
                records.append({'anchor_id': anchor, 'anchor_display': f'B{a+1:02d}',
                                'method': method, 'reference_source': view,
                                **metrics([scores[u] for u in targets], [labels[u][view] for u in targets])})
        for uid in targets:
            ranks.append({'anchor_id': anchor, 'anchor_display': f'B{a+1:02d}',
                          'target_id': uid, 'target_display': f'B{ids.index(uid)+1:02d}',
                          **{m+'_rank': orders[m].index(uid)+1 for m in current},
                          **{m+'_score': current[m][uid] for m in current}})
    table = []
    for view in ('uas', 'crewed'):
        for method in current:
            rows = [r for r in records if r['reference_source'] == view and r['method'] == method]
            table.append({'reference_source': view, 'method': method,
                          **{k: float(np.mean([r[k] for r in rows])) for k in
                             ('ap', 'brier', 'log_loss', 'top3_positive')}})
    # Recompute aggregates independently from serialized predictions.
    independent = {}
    for a in ids:
        targets = [u for u in ids if u != a]
        for method in current:
            scores = {r['target_id']: r['score'] for r in predictions
                      if r['anchor_id'] == a and r['method'] == method}
            p = np.array([scores[u] for u in targets])
            for view in ('uas', 'crewed'):
                y = np.array([labels[u][view] for u in targets])
                pc = np.clip(p, 1e-6, 1-1e-6)
                independent.setdefault((view, method), []).append({
                    'ap': independent_ap(p, y), 'brier': float(np.mean((p-y)**2)),
                    'log_loss': float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))),
                    'top3_positive': float(y[np.argsort(-p, kind='stable')[:3]].sum())})
    for row in table:
        for key in ('ap', 'brier', 'log_loss', 'top3_positive'):
            assert abs(row[key] - np.mean([r[key] for r in independent[row['reference_source'], row['method']]])) < 1e-12
    per_anchor_delta = []
    for gated, control in pairs:
        for a in ids:
            selected = {r['method']: r for r in records if r['anchor_id'] == a and r['reference_source'] == 'uas'}
            per_anchor_delta.append({'anchor_id': a, 'anchor_display': f'B{ids.index(a)+1:02d}',
                                    'gated_method': gated, 'control_method': control,
                                    'gated_ap': selected[gated]['ap'], 'control_ap': selected[control]['ap'],
                                    'gated_minus_control_ap': selected[gated]['ap']-selected[control]['ap']})
    old_hashes = 0
    for directory in (source, change):
        receipt = read(directory/'completion_receipt.json')
        for name, digest in receipt['files'].items():
            assert sha(directory/name) == digest, str(directory/name)
            old_hashes += 1
    out.mkdir(parents=True)
    dump(out/'summary.json', {'table': table, 'anchor_cases': 12, 'new_model_calls': 0,
                             'new_route_runs': 0, 'cost': 'CPU only; no API or GPU rental'})
    dump(out/'prediction_rows.json', predictions)
    dump(out/'per_anchor_scores.json', records)
    dump(out/'per_anchor_deltas.json', per_anchor_delta)
    dump(out/'strength_matches.json', diagnostics)
    dump(out/'target_rank_changes.json', ranks)
    dump(out/'execution_recipe.json', {
        'control': 'Per-anchor scalar spatial gain matching mean absolute clipped target-score change',
        'gain_range': [0, .5], 'bisection_steps': 80, 'space_scale_m': 50,
        'frozen_gated_methods': [p[0] for p in pairs],
        'target_labels_used_to_choose_gain': False,
        'feature_use': 'Visual target identity removed; aggregate update strength retained',
        'design_status': 'Development diagnostic following inspected bandwidth results; no parameter search',
        'script_sha256': sha(Path(__file__)),
        'input_hashes': {str(p.relative_to(root)): sha(p) for p in
                         (source/'satellite_features.json', source/'vlm_initial_scores.json',
                          change/'prediction_rows.json', reference)}})
    dump(out/'audit.json', {'status': 'PASS', 'independent_aggregate_checks': len(table)*4,
                           'strength_matches': len(diagnostics),
                           'max_mean_strength_error': max(r['absolute_error'] for r in diagnostics),
                           'old_sealed_files_unchanged': old_hashes})
    with (out/'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'table': [r for r in table if r['reference_source']=='uas'],
                      'audit': read(out/'audit.json')}, indent=2))


if __name__ == '__main__':
    main()
