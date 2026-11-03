"""Audit completed local-VLM records and summarize how histories move scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def independent_ap(p, y):
    order = np.argsort(-p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p) - 1]
    cumulative = np.cumsum(sorted_y)[ends]
    increments = np.diff(np.r_[0, cumulative])
    return float(np.sum(increments * cumulative / (ends + 1)) / sum(y)) if sum(y) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    out = args.output
    summary = read(out / 'summary.json')
    assert summary['anchor_cases'] == 12
    assert summary['forward_calls'] == 276
    manifest = read(out / 'crop_manifest.json')
    crop_map = {r['building_id']: r['crops'] for r in manifest['rows']}
    ids = sorted(crop_map)
    calls = [read(p) for p in sorted((out / 'calls').glob('*.json'))]
    assert len(calls) == 276
    input_checks = Counter()
    for c in calls:
        anchor, target = c['anchor_id'], c['target_id']
        assert target in ids
        expected = [crop_map[target][source]['path'] for source in ('pre', 'post')]
        if anchor is not None:
            assert anchor in ids and anchor != target
            if not c['label_only_history']:
                expected = [crop_map[anchor][source]['path'] for source in ('pre', 'post', 'uas')] + expected
        assert c['image_paths'] == expected
        assert c['prompt'].count('<image>') == len(expected)
        assert 0 <= c['score'] <= 1
        assert 0 < c['answer_set_probability_mass'] <= 1.00001
        input_checks['all'] += 1
        input_checks['initial' if anchor is None else 'label_only' if c['label_only_history'] else 'full_history'] += 1
    for row in manifest['rows']:
        for crop in row['crops'].values():
            assert hashlib.sha256((out / crop['path']).read_bytes()).hexdigest() == crop['sha256']
    refs = read(args.workspace / 'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json')
    by_id = {r['building_id']: r['view_binary'] for r in refs['rows']}
    predictions = read(out / 'prediction_rows.json')
    assert len(predictions) == 660
    groups = {}
    for row in predictions:
        group = groups.setdefault((row['anchor_id'], row['method']), {})
        assert row['target_id'] not in group
        group[row['target_id']] = row['score']
    assert len(groups) == 60
    recomputed = {}
    for (anchor, method), scores in groups.items():
        target_ids = sorted(scores)
        assert set(target_ids) == set(ids) - {anchor}
        p = np.array([scores[uid] for uid in target_ids])
        for source in ('uas', 'crewed'):
            y = np.array([by_id[uid][source] for uid in target_ids], dtype=float)
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            m = {'brier': float(np.mean((p-y)**2)),
                 'log_loss': float(-np.mean(y*np.log(pc) + (1-y)*np.log(1-pc))),
                 'ap': independent_ap(p, y),
                 'top3_positive': float(y[np.argsort(-p, kind='stable')[:3]].sum())}
            recomputed.setdefault((source, method), []).append(m)
    for row in summary['table']:
        items = recomputed[(row['reference_source'], row['method'])]
        for key in ('brier', 'log_loss', 'ap', 'top3_positive'):
            assert abs(row[key] - np.mean([m[key] for m in items])) < 1e-10
    initial = read(out / 'vlm_initial_scores.json')
    shifts = []
    for (anchor, method), scores in sorted(groups.items()):
        if method not in ('V1_vlm_history', 'V2_vlm_label_only'):
            continue
        delta = np.array([scores[uid] - initial[uid] for uid in sorted(scores)])
        shifts.append({'anchor_id': anchor, 'method': method,
                       'anchor_uas_view_label': by_id[anchor]['uas'],
                       'mean_shift': float(np.mean(delta)),
                       'std_shift': float(np.std(delta)),
                       'all_targets_move_same_direction': bool(np.all(delta > 0) or np.all(delta < 0)),
                       'min_shift': float(delta.min()), 'max_shift': float(delta.max())})
    (out / 'score_shift_diagnostic.json').write_text(json.dumps(shifts, indent=2) + '\n')
    audit = {'status': 'PASS', 'checked_calls': dict(input_checks),
             'crop_hashes_verified': 60, 'prediction_rows': len(predictions),
             'independent_summary_metric_checks': len(summary['table']) * 4,
             'full_history_same_direction_anchor_cases': sum(
                 r['all_targets_move_same_direction'] for r in shifts if r['method'] == 'V1_vlm_history'),
             'minimum_answer_probability_mass': min(c['answer_set_probability_mass'] for c in calls),
             'scope': 'Input access, crop hashes and metric audit; not independent georegistration or field-truth validation'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
