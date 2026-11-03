"""Independently recompute ranking metrics and footprint Pearson correlations."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_boone_context_experiment import independent_ap
from run_boone_context_experiment import dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    read = lambda p: json.loads(p.read_text())
    reference = read(root/'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json')
    labels = {r['building_id']: r['view_binary'] for r in reference['rows']}
    groups = {}
    for row in read(out/'prediction_rows.json'):
        group = groups.setdefault((row['anchor_id'], row['method']), {})
        assert row['target_id'] not in group
        group[row['target_id']] = row['score']
    results, per_case_checks = {}, 0
    saved = {(r['anchor_id'], r['method'], r['reference_source']): r
             for r in read(out/'per_anchor_scores.json')}
    for (anchor, method), scores in groups.items():
        assert set(scores) == set(labels)-{anchor}
        order = sorted(scores, key=lambda u: (-scores[u], u))
        p = np.array([scores[u] for u in order])
        for view in ('uas', 'crewed'):
            y = np.array([labels[u][view] for u in order])
            hits = int(y[:3].sum())
            dcg = sum(int(y[i])/math.log2(i+2) for i in range(3))
            ideal = sum(1/math.log2(i+2) for i in range(min(3, int(y.sum()))))
            clipped = np.clip(p, 1e-6, 1-1e-6)
            record = {'ap': independent_ap(p, y), 'brier': float(np.mean((p-y)**2)),
                      'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
                      'top3_positive': hits, 'precision_at3': hits/3, 'ndcg_at3': dcg/ideal}
            for key, value in record.items():
                assert abs(value-saved[anchor, method, view][key]) < 1e-12
                per_case_checks += 1
            assert saved[anchor, method, view]['top3_ids'] == order[:3]
            results.setdefault((view, method), []).append(record)
    aggregate_checks = 0
    for row in read(out/'summary.json')['table']:
        entries = results[row['reference_source'], row['method']]
        for key in ('ap', 'brier', 'log_loss', 'top3_positive', 'precision_at3', 'ndcg_at3'):
            assert abs(row[key]-np.mean([r[key] for r in entries])) < 1e-12
            aggregate_checks += 1
        assert row['top3_hit_sum'] == sum(r['top3_positive'] for r in entries)
        aggregate_checks += 1
    manifest = read(root/'output/disaster_route_boone_context_run_v2/crop_manifest.json')
    images = {s: np.asarray(Image.open(root/manifest['source_images'][s]['path']).convert('RGB'))
              for s in ('pre', 'post')}
    public = read(root/'output/disaster_route_boone_context_pilot_v1/public_case_index.json')
    geometry = {r['building_id']: r['roi_polygon_px'] for r in public['geometry']}
    features = read(out/'structure_features.json')['rows']
    for row in features:
        mask = Image.new('L', (images['pre'].shape[1], images['pre'].shape[0]), 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in geometry[row['building_id']]], fill=255)
        pixels = {s: image[np.asarray(mask)>0].astype(float) for s, image in images.items()}
        correlations = [float(np.corrcoef(pixels['pre'][:, c], pixels['post'][:, c])[0, 1]) for c in range(3)]
        np.testing.assert_allclose(correlations, row['channel_correlations'], atol=1e-12, rtol=0)
        assert abs(1-np.mean(correlations)-row['pearson_change']) < 1e-12
    frozen_count = 0
    for name in ('disaster_route_boone_context_run_v2', 'disaster_route_boone_change6_v1',
                 'disaster_route_boone_strength_control_v1'):
        folder = root/'output'/name
        for path, digest in read(folder/'completion_receipt.json')['files'].items():
            assert sha(folder/path) == digest
            frozen_count += 1
    recipe = read(out/'execution_recipe.json')
    for name, digest in recipe['input_hashes'].items():
        assert sha(root/name) == digest
    dump(out/'audit.json', {'status': 'PASS', 'prediction_groups': len(groups),
                           'per_anchor_metric_checks': per_case_checks,
                           'aggregate_metric_checks': aggregate_checks,
                           'independent_numpy_correlation_buildings': len(features),
                           'old_sealed_files_unchanged': frozen_count,
                           'max_strength_matching_error': max(r['absolute_error'] for r in read(out/'strength_matches.json'))})
    print((out/'audit.json').read_text())


if __name__ == '__main__':
    main()
