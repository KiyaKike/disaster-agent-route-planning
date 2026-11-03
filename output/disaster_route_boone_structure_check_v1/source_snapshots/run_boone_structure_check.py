"""One fixed Pearson-change ablation and top-three audit, without model calls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_boone_context_experiment import dump, feature_kernel, masked_crop, metrics, sha
from run_boone_strength_control import matched_spatial


def pearson_change(pre, post):
    """Mean channelwise Pearson change on footprint pixels, no padded background."""
    pre, post = np.asarray(pre, dtype=float), np.asarray(post, dtype=float)
    x, y = pre-pre.mean(axis=0), post-post.mean(axis=0)
    denominator = np.linalg.norm(x, axis=0)*np.linalg.norm(y, axis=0)
    valid = denominator > 1e-12
    if not np.any(valid):
        raise ValueError('No textured channel; Pearson correlation is undefined')
    correlations = np.sum(x[:, valid]*y[:, valid], axis=0)/denominator[valid]
    return float(1-np.mean(np.clip(correlations, -1, 1))), correlations.tolist()


def top_metrics(scores, labels, k=3):
    """Binary relevance, log2 rank discount, deterministic ID tie-break."""
    order = sorted(scores, key=lambda u: (-scores[u], u))
    hits = int(sum(labels[u] for u in order[:k]))
    positives = int(sum(labels[u] for u in order))
    discounts = 1/np.log2(np.arange(2, k+2))
    dcg = sum(labels[u]*discounts[i] for i, u in enumerate(order[:k]))
    idcg = float(discounts[:min(k, positives)].sum())
    return {'top3_positive': hits, 'precision_at3': hits/k,
            'ndcg_at3': float(dcg/idcg) if idcg else None, 'top3_ids': order[:k]}


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
    previous = root/'output/disaster_route_boone_strength_control_v1'
    refpath = root/'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json'
    publicpath = root/'output/disaster_route_boone_context_pilot_v1/public_case_index.json'
    f, manifest = read(source/'satellite_features.json'), read(source/'crop_manifest.json')
    ids, xy = f['building_ids'], np.array(f['xy_m'])
    geometry = {r['building_id']: r for r in read(publicpath)['geometry']}
    images = {}
    for name in ('pre', 'post'):
        record = manifest['source_images'][name]
        path = root/record['path']
        assert sha(path) == record['sha256']
        images[name] = Image.open(path).convert('RGB')
    features, correlations = [], []
    for uid in ids:
        pixels = [masked_crop(images[s], geometry[uid]['roi_polygon_px'])[1] for s in ('pre', 'post')]
        change, channel_correlations = pearson_change(*pixels)
        features.append([change])
        correlations.append({'building_id': uid, 'display_id': f'B{ids.index(uid)+1:02d}',
                             'channel_correlations': channel_correlations, 'pearson_change': change})
    _, standardized, similarity, tau = feature_kernel(features)
    # Feature and kernel construction above has no reference-label access.
    labels = {r['building_id']: r['view_binary'] for r in read(refpath)['rows']}
    initial = read(source/'vlm_initial_scores.json')
    p0 = np.array([initial[u] for u in ids])
    predictions = read(previous/'prediction_rows.json')
    matches = []
    for a, anchor in enumerate(ids):
        mask = np.arange(len(ids)) != a
        spatial = np.exp(-np.sum((xy-xy[a])**2, axis=1)/(2*50**2))
        residual = labels[anchor]['uas']-p0[a]
        scores = np.clip(p0+.5*spatial*similarity[a]*residual, 0, 1)
        control, gain, magnitude = matched_spatial(p0, spatial, residual, mask, scores)
        error = abs(float(np.abs(control[mask]-p0[mask]).mean())-magnitude)
        assert error < 1e-12
        matches.append({'anchor_id': anchor, 'gain': gain, 'desired_mean_abs_change': magnitude,
                        'absolute_error': error})
        for method, values in [('S5_pearson_change', scores), ('C5_matched_pearson', control)]:
            predictions.extend({'anchor_id': anchor, 'target_id': uid, 'method': method,
                                'score': float(values[i])} for i, uid in enumerate(ids) if i != a)
    groups = {}
    for row in predictions:
        group = groups.setdefault((row['anchor_id'], row['method']), {})
        assert row['target_id'] not in group
        group[row['target_id']] = row['score']
    methods = sorted({r['method'] for r in predictions})
    records, ranks, comparisons = [], [], []
    for a in ids:
        for method in methods:
            scores = groups[a, method]
            targets = sorted(scores)
            assert set(targets) == set(ids)-{a}
            for view in ('uas', 'crewed'):
                reference = {u: labels[u][view] for u in targets}
                records.append({'anchor_id': a, 'anchor_display': f'B{ids.index(a)+1:02d}',
                                'method': method, 'reference_source': view,
                                **metrics([scores[u] for u in targets], [reference[u] for u in targets]),
                                **top_metrics(scores, reference)})
            order = sorted(targets, key=lambda u: (-scores[u], u))
            ranks.extend({'anchor_id': a, 'anchor_display': f'B{ids.index(a)+1:02d}',
                          'target_id': u, 'target_display': f'B{ids.index(u)+1:02d}',
                          'method': method, 'rank': order.index(u)+1, 'score': scores[u]} for u in targets)
        for gated, control in [('S2_visual_spatial', 'C2_matched36'),
                               ('S3_change6', 'C3_matched6'),
                               ('S4_change6_old_tau', 'C4_matched6_old_tau'),
                               ('S5_pearson_change', 'C5_matched_pearson')]:
            for view in ('uas', 'crewed'):
                pair = {r['method']: r for r in records if r['anchor_id']==a and r['reference_source']==view}
                comparisons.append({'anchor_id': a, 'anchor_display': f'B{ids.index(a)+1:02d}',
                                    'reference_source': view, 'gated_method': gated, 'control_method': control,
                                    **{key+'_delta': pair[gated][key]-pair[control][key] for key in
                                       ('ap', 'top3_positive', 'precision_at3', 'ndcg_at3')},
                                    'gated_top3': pair[gated]['top3_ids'], 'control_top3': pair[control]['top3_ids']})
    table = []
    for view in ('uas', 'crewed'):
        for method in methods:
            rows = [r for r in records if r['reference_source']==view and r['method']==method]
            table.append({'reference_source': view, 'method': method,
                          **{key: float(np.mean([r[key] for r in rows])) for key in
                             ('ap', 'brier', 'log_loss', 'top3_positive', 'precision_at3', 'ndcg_at3')},
                          'top3_hit_sum': sum(r['top3_positive'] for r in rows)})
    out.mkdir(parents=True)
    dump(out/'summary.json', {'table': table, 'anchor_cases': 12, 'new_model_calls': 0,
                             'new_route_runs': 0, 'tau': tau})
    dump(out/'prediction_rows.json', predictions)
    dump(out/'per_anchor_scores.json', records)
    dump(out/'per_anchor_comparisons.json', comparisons)
    dump(out/'target_ranks.json', ranks)
    dump(out/'strength_matches.json', matches)
    dump(out/'structure_features.json', {'rows': correlations, 'standardized_features': standardized.tolist(),
                                       'similarity': similarity.tolist(), 'tau': tau})
    dump(out/'execution_recipe.json', {
        'representation': '1 - mean(per-RGB-channel Pearson(pre,post)) on original footprint pixels',
        'invariance': 'Independent positive affine intensity change per channel; not registration invariant',
        'kernel': 'Gaussian on standardized1D feature; std floor0.05; median positive distance',
        'feature_count': 1, 'space_scale_m': 50, 'residual_gain': .5,
        'feature_uses_uas_or_noaa': False, 'target_labels_used_to_choose_gain': False,
        'new_model_calls': 0, 'script_sha256': sha(Path(__file__)),
        'top_metrics_status': 'Retrospective development diagnostics, not new independent test endpoints',
        'ndcg_definition': 'Binary relevance; discount1/log2(rank+1); ideal uses unvisited positives',
        'input_hashes': {str(p.relative_to(root)): sha(p) for p in
                         (source/'satellite_features.json', source/'vlm_initial_scores.json',
                          source/'crop_manifest.json', previous/'prediction_rows.json', refpath, publicpath)},
        'source_images': manifest['source_images'],
        'cross_source_disagreement_status': 'Not run: no frozen legal NOAA prior; target NOAA pixels QA/evaluator-only under existing main protocol'})
    with (out/'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'tau': tau, 'uas_table': [r for r in table if r['reference_source']=='uas']}, indent=2))


if __name__ == '__main__':
    main()
