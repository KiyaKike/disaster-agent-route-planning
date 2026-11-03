"""Rebuild references, tensors and all pixel scores without more model calls."""
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json, sha
from run_changeos_pilot import crop_pair, prepare_tensor
from run_perception_pair72 import ROOT, HERE, DATA, LEGACY, PARENT, checked, digest, read, verify_protocol, load_pair

OUT = ROOT / 'output/disaster_search_perception_pair72_v1'


def independent_truth(document):
    layers = [np.zeros((1024, 1024), np.uint8) for _ in range(5)]
    mapping = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3, 'un-classified': 4}
    for f in document['features']['xy']:
        text = f['wkt'][10:-2]
        points = np.asarray([[float(v) for v in s.split()] for s in text.split(',')])
        cv2.fillPoly(layers[mapping[f['properties']['subtype']]], [np.rint(points).astype(np.int32)], 1)
    count = sum(layers[:4]); truth = np.zeros((1024, 1024), np.uint8)
    for k in range(4): truth[layers[k] != 0] = k + 1
    truth[(count > 1) | (layers[4] != 0)] = 255
    return truth


def independent_vote(loc, damage):
    # skimage provides a second connected-component implementation.
    from skimage.measure import label
    components = label(loc, connectivity=2, background=0)
    table = np.zeros(int(components.max()) + 1, np.uint8)
    for i in range(1, len(table)):
        selected = damage[components == i]
        table[i] = 1 + np.argmax([int((selected == g).sum()) * w for g, w in zip((1, 2, 3, 4), (8, 38, 25, 11))])
    return table[components]


def independently_score(truth, prediction):
    valid = truth != 255; gt = ((truth == 3) | (truth == 4)) & valid
    pred = ((prediction == 3) | (prediction == 4)) & valid
    tp = int((gt & pred).sum()); fp = int((~gt & pred & valid).sum())
    fn = int((gt & ~pred).sum()); tn = int((~gt & ~pred & valid).sum())
    def ratio(a, b): return a / b if b else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'reference_severe_pixels': tp + fn,
            'predicted_severe_pixels_on_valid_domain': tp + fp,
            'precision': ratio(tp, tp+fp), 'recall': ratio(tp, tp+fn),
            'f1': ratio(2*tp, 2*tp+fp+fn), 'iou': ratio(tp, tp+fp+fn),
            'false_severe_background_pixels': int((pred & (truth == 0)).sum()),
            'false_severe_other_building_pixels': int((pred & ((truth == 1) | (truth == 2))).sum()),
            'false_severe_per_million_valid_pixels': ratio(fp*1e6, int(valid.sum())), 'evaluated_pixels': int(valid.sum())}


def check_values(row, expected):
    for k, v in expected.items():
        if isinstance(v, float):
            if row[k] is None or abs(row[k] - v) > 1e-12: raise ValueError('Metric differs: ' + k)
        elif row[k] != v: raise ValueError('Metric differs: ' + k)


def main():
    if (OUT / 'completion_receipt.json').exists(): raise ValueError('Completed output frozen')
    protocol = read(OUT / 'protocol.json'); verify_protocol(OUT, protocol)
    checked(DATA / 'manifest.json', protocol['data_manifest_sha256'])
    checked(PARENT / 'completion_receipt.json', protocol['parent_receipt_sha256'])
    for name, value in read(PARENT / 'completion_receipt.json')['source_code_sha256'].items(): checked(HERE / name, value)
    jobs = {r['sample_id']: r for r in read(OUT / 'rgb_jobs.json')}
    references = read(OUT / 'reference_manifest.json'); rows = read(OUT / 'observations.json')
    if len(rows) != 578 or len(jobs) != 72 or len(references) != 72: raise ValueError('Incomplete inventory')
    group = defaultdict(list); controls = []; input_count = 0
    expected_order = [('swint', 'regression_control', c['sample_id'], None) for c in protocol['positive_controls']]
    expected_order += [(m, 'production', sid, tile) for m in ('swint', 'r34') for sid in sorted(jobs) for tile in range(4)]
    for index, (row, expected) in enumerate(zip(rows, expected_order), 1):
        if (row['model_id'], row['role'], row['sample_id'], row.get('tile_id')) != expected or row['index'] != index:
            raise ValueError('Job order differs')
        if row != read(OUT / 'calls' / f'{index:03d}.json') or row['status'] != 'COMPLETE' or row['protocol_sha256'] != digest(OUT / 'protocol.json'):
            raise ValueError('Call ledger differs')
        if row['role'] == 'regression_control':
            c = protocol['positive_controls'][len(controls)]
            pair = crop_pair(*load_pair(c), c['cell_id'])
        else:
            pair = load_pair(jobs[row['sample_id']]); tile = row['tile_id']
            x0, y0 = (tile % 2)*512, (tile//2)*512
            bounds = [x0, y0, x0+512, y0+512]
            if row['bounds'] != bounds: raise ValueError('Wrong native bounds')
            pair = [im.crop(bounds) for im in pair]
        if row['model_id'] == 'swint': tensor = prepare_tensor(*pair, 'native').numpy()
        else:
            # Recreate official normalization independently of the new adapter.
            joined = np.concatenate([np.asarray(im) for im in pair], axis=2).astype(np.float32)
            joined = (joined - np.asarray([123.675, 116.28, 103.53]*2, np.float32)) / np.asarray([58.395, 57.12, 57.375]*2, np.float32)
            tensor = joined.transpose(2, 0, 1)[None].copy()
        if list(tensor.shape) != row['input_shape'] or sha(tensor.tobytes()) != row['input_tensor_sha256']:
            raise ValueError('Input tensor differs')
        input_count += 1
        checked(row['cache_file'], row['cache_sha256'])
        with np.load(row['cache_file'], allow_pickle=False) as d: loc, damage = d['building'], d['classes']
        if loc.dtype != np.bool_ or damage.dtype != np.uint8 or loc.shape != tuple(row['input_shape'][-2:]) or damage.shape != loc.shape:
            raise ValueError('Invalid cached arrays')
        if row['role'] == 'regression_control':
            checked(c['legacy_cache_file'], c['legacy_cache_sha256'])
            with np.load(c['legacy_cache_file'], allow_pickle=False) as d: old = d['severe_mask']
            if not np.array_equal(loc & (damage >= 3), old): raise ValueError('Control no longer identical')
            controls.append(row['sample_id'])
        else: group[row['sample_id'], row['model_id']].append((row['tile_id'], loc, damage))
    old_by_scene = defaultdict(list)
    checked(LEGACY / 'observations.json', protocol['legacy128_secondary']['observations_sha256'])
    for r in read(LEGACY / 'observations.json'):
        if r['mode'] == 'native': old_by_scene[r['sample_id']].append(r)
    score_index = {(r['sample_id'], r['model_id'], r['decoder']): r for r in read(OUT / 'scene_scores.json')}
    pooled = defaultdict(lambda: np.zeros((5, 5), np.int64)); reference_count = 0; scores_count = 0; legacy_count = 0
    for ref in references:
        sid = ref['sample_id']; checked(ref['label_path'], ref['label_sha256'])
        truth = independent_truth(read(ref['label_path'])); reference_count += 1
        checked(ref['reference_file'], ref['reference_sha256'])
        if sha(truth.tobytes()) != ref['reference_pixel_sha256'] or not np.array_equal(truth, np.asarray(Image.open(ref['reference_file']))):
            raise ValueError('Raster reference differs')
        for model_id in ('swint', 'r34', 'legacy_swint128'):
            loc = np.zeros((1024, 1024), bool); damage = np.zeros((1024, 1024), np.uint8)
            if model_id == 'legacy_swint128':
                old = old_by_scene[sid]
                if len(old) != 64: raise ValueError('Old inventory differs')
                for r in old:
                    checked(r['cache_file'], r['cache_sha256']); legacy_count += 1
                    with np.load(r['cache_file'], allow_pickle=False) as d: mask = d['severe_mask']
                    y, x = int(r['cell_id'][1:3])*128, int(r['cell_id'][4:6])*128
                    damage[y:y+128, x:x+128] = mask.astype(np.uint8)*3
                maps = {'raw_pixel': damage}
            else:
                for tile, local, classes in group[sid, model_id]:
                    x, y = (tile % 2)*512, (tile//2)*512
                    loc[y:y+512, x:x+512] = local; damage[y:y+512, x:x+512] = classes
                maps = {'raw_pixel': damage*loc.astype(np.uint8), 'common_object_vote': independent_vote(loc, damage)}
            for decoder, pred in maps.items():
                saved = score_index[sid, model_id, decoder]; check_values(saved, independently_score(truth, pred))
                cm = np.asarray([[int(((truth == i) & (pred == j)).sum()) for j in range(5)] for i in range(5)], np.int64)
                if saved['confusion_5x5'] != cm.tolist(): raise ValueError('Five-class confusion differs')
                pooled[model_id, decoder, ref['event']] += cm; scores_count += 1
        if reference_count % 12 == 0: print(json.dumps({'references_rebuilt': reference_count, 'scores_replayed': scores_count}), flush=True)
    events = read(OUT / 'event_scores.json'); totals = read(OUT / 'summary.json')['totals']
    from perception_pair72_core import metrics
    # Counts above were computed independently; aggregate ratios are checked here.
    for event in events: check_values(event, metrics(pooled[event['model_id'], event['decoder'], event['event']]))
    for total in totals:
        selected = [e for e in events if e['model_id'] == total['model_id'] and e['decoder'] == total['decoder']]
        cm = sum((np.asarray(e['confusion_5x5']) for e in selected), np.zeros((5, 5), np.int64))
        check_values(total, {**metrics(cm), 'equal_event_macro_f1': float(np.mean([e['f1'] for e in selected]))})
    audit = {'status': 'PASS', 'references_independently_rasterized': reference_count, 'input_tensors_reconstructed': input_count,
             'pixel_identical_old_controls': len(controls), 'legacy128_masks_verified': legacy_count,
             'scene_scores_independently_replayed': scores_count, 'event_scores_checked': len(events), 'total_scores_checked': len(totals),
             'old_frozen_sources_verified': len(read(PARENT / 'completion_receipt.json')['source_code_sha256']),
             'new_model_calls_by_audit': 0, 'new_fitting': False, 'gpu_rented': False,
             'auditor_sha256': digest(Path(__file__))}
    atomic_json(OUT / 'audit.json', audit); print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
