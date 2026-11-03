"""Frozen paired-checkpoint diagnostic on already-open xBD development72 only.

Stages: plan, predict, evaluate. Prediction reads the locked RGB-only job list;
post polygons and cached legacy predictions are read only by plan/evaluation.
No training, GPU rental, threshold sweep, planner or reserved-set access.
"""
import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze
from run_changeos_pilot import prepare_tensor, crop_pair, sync, CODE_COMMIT, WEIGHTS_SHA256, CONFIG_SHA256
from extract_external_area_masks import verify_parent
from perception_pair72_core import MODELS, TILES, rasterize_post, prepare_r34, decode_output, object_vote, confusion, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'output/disaster_search_hold_development72_v2'
LEGACY = ROOT / 'output/disaster_search_hold_development72_perception_v1'
PARENT = ROOT / 'output/disaster_search_native36_v1'
SWIN = HERE / 'attachments/external/huggingface/changeos_swint'
R34 = HERE / 'attachments/external/changeos_r34_v02'
DATA_SHA = 'a70646afabf0822b3adea609835ab95c1cfdb2e7f834d6d8173d38fa4b042346'
R34_SHA = '3aa0520284ba5b2358b695732902295ef5fbc225b75ff1885a9eb41609c038d2'
SOURCE_FILES = ['run_perception_pair72.py', 'perception_pair72_core.py', 'test_perception_pair72.py',
                'run_changeos_pilot.py', 'build_xbd_mini.py', 'plan_search_v2.py', 'extract_external_area_masks.py']


def read(path): return json.loads(Path(path).read_text())
def digest(path): return sha(Path(path).read_bytes())


def checked(path, expected):
    if digest(path) != expected: raise ValueError('File changed: ' + str(path))


def plan(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        protocol = read(out / 'protocol.json'); verify_protocol(out, protocol)
        print('Existing protocol verified; not regenerated', flush=True)
        return
    parent_sha = verify_parent(PARENT)
    parent = read(PARENT / 'completion_receipt.json')
    for name, value in parent['source_code_sha256'].items(): checked(HERE / name, value)
    checked(DATA / 'manifest.json', DATA_SHA)
    manifest = read(DATA / 'manifest.json')
    if len(manifest['samples']) != 72 or {s['split'] for s in manifest['samples']} != {'development'}:
        raise ValueError('Only the existing full development72 cohort is permitted')
    for row in manifest['files']: checked(DATA / row['path'], row['sha256'])
    checked(SWIN / 'model.safetensors', WEIGHTS_SHA256); checked(SWIN / 'config.json', CONFIG_SHA256)
    checked(R34 / 'changeos_r34.pt', R34_SHA)
    if (R34 / 'changeos_r34.pt').stat().st_size != 142087496: raise ValueError('Unexpected checkpoint size')
    references = []; rgb_jobs = []
    (out / 'reference').mkdir(exist_ok=True)
    for sample in sorted(manifest['samples'], key=lambda s: s['sample_id']):
        sid = sample['sample_id']; files = sample['files']
        document = read(DATA / files['post_label'])
        if document['metadata']['height'] != 1024 or document['metadata']['width'] != 1024:
            raise ValueError('Reference geometry differs')
        truth, audit = rasterize_post(document)
        file = out / 'reference' / (sid + '.png')
        Image.fromarray(truth).save(file)
        references.append({'sample_id': sid, 'event': sample['event'], 'label_path': str((DATA / files['post_label']).resolve()),
                           'label_sha256': digest(DATA / files['post_label']), 'reference_file': str(file.resolve()),
                           'reference_sha256': digest(file), 'reference_pixel_sha256': sha(truth.tobytes()), **audit})
        images = []
        for mode in ('pre', 'post'):
            path = DATA / files[mode + '_image']
            with Image.open(path) as im:
                if im.mode != 'RGB' or im.size != (1024, 1024): raise ValueError('Expected native RGB1024')
            images.append({'mode': mode, 'path': str(path.resolve()), 'sha256': digest(path)})
        rgb_jobs.append({'sample_id': sid, 'event': sample['event'], 'images': images})
    preview = []
    for event in sorted({r['event'] for r in references}):
        for positive in (False, True):
            candidates = [r for r in references if r['event'] == event and
                          ((r['pixels_by_id']['3'] + r['pixels_by_id']['4'] > 0) == positive)]
            if candidates: preview.append(min(r['sample_id'] for r in candidates))
    freeze(out / 'reference_manifest.json', references)
    freeze(out / 'rgb_jobs.json', rgb_jobs)
    log = (out / 'test_log.txt').read_text()
    if 'Ran 338 tests' not in log or not log.rstrip().endswith('OK'): raise ValueError('Run the338-test suite before freezing')
    protocol = {'stage': 'XBD_DEVELOPMENT72_PAIRED_CHECKPOINT_PIXEL_DIAGNOSTIC', 'client_date': '2026-10-28',
        'parent_receipt_sha256': parent_sha, 'data_directory': str(DATA), 'data_manifest_sha256': DATA_SHA,
        'samples': 72, 'events': 3, 'rgb_jobs_sha256': digest(out / 'rgb_jobs.json'),
        'reference_manifest_sha256': digest(out / 'reference_manifest.json'),
        'source_code_sha256': {n: digest(HERE / n) for n in SOURCE_FILES},
        'test_log_sha256': digest(out / 'test_log.txt'), 'tests_passed_before_production': 338,
        'models': {'swint': {'directory': str(SWIN), 'model.safetensors': WEIGHTS_SHA256, 'config.json': CONFIG_SHA256,
                            'torchange_commit': CODE_COMMIT, 'source': 'https://huggingface.co/EVER-Z/torchange_example_changeos_swint_on_xview2_best42k'},
                   'r34': {'directory': str(R34), 'changeos_r34.pt': R34_SHA,
                           'official_wrapper.py': digest(R34 / 'official_wrapper.py'), 'README.upstream.md': digest(R34 / 'README.upstream.md'),
                           'source_commit': '7936b84aceb9c4b89e959690f3da28749b38d10c',
                           'source': 'https://github.com/Z-Zheng/ChangeOS/releases/download/v0.2/changeos_r34.pt',
                           'format': 'author-published TorchScript,5damage channels and1localization logit',
                           'training_sample_membership': 'not supplied with release; event/sample exposure cannot be excluded',
                           'license_file_at_pinned_root': 'LICENSE returned404; no redistribution-rights claim'}},
        'input': {'native_tile_size': 512, 'scene_size': 1024, 'tile_bounds_xyxy': TILES, 'resize': False,
                  'tta': False, 'batch_size': 1, 'normalization': 'Model-specific official RGB ImageNet float32 normalization'},
        'primary_decoder': 'Swin loc>.5 probability; R34 loc>0 logit; damage argmax over5 channels; raw classes multiplied by predicted building mask',
        'secondary_decoder': 'Common scene-level8-connected weighted damage vote, weights[8,38,25,11]; no ground-truth objects used',
        'reference': 'Existing post polygons; np.round vertices and cv2.fillPoly; background0,grades1..4,unknown255. Conflicting known class overlaps and all unknown footprints ignored. Existing +/-0.5pixel boundary tolerance; OpenCV clips only at raster boundary.',
        'primary_metrics': 'Severe classes3or4 pooled pixel precision/recall/F1 and FP; primary comparison equal-event macroF1; per-event results mandatory',
        'not_official_xview2_score': 'Post-reference only; missing original pre-localization polygons; do not call this official mixedF1 or instanceAP',
        'legacy128_secondary': {'directory': str(LEGACY), 'observations_sha256': digest(LEGACY / 'observations.json'),
                                'interpretation': 'Existing Swin128 native masks; severe-only metrics under same new post reference. Context differs; not a model comparison.'},
        'call_cap': 578, 'call_inventory': {'old_swin128_regression_controls': 2, 'swint512': 288, 'r34512': 288},
        'forward_order': '2oldSwincontrols, all72Swin scenes sortedID, then all72R34 scenes sortedID, four row-major tiles each; no unlogged warmup',
        'positive_controls': read(PARENT / 'protocol.json')['positive_controls'],
        'preview_ids': preview, 'preview_selection': 'For each event, firstID with and without severe reference; fixed before predictions; qualitative only',
        'prospective_decision': 'Report both checkpoints and per-event failures. A higher development macroF1 can inform a working detector only; no threshold fitting, new method claim or reserved-set use. Do not restart EBD input/planner variants.',
        'limitations': ['Already-open development data and prior experiments; training exposure expected for Swin, unknown forR34; no independent generalization claim.',
                       'Same ChangeOS family, different backbones/checkpoint generations; not a cross-architecture strong-baseline study.',
                       '512tiling is an offline perception diagnostic; not legal unpaid context for the old128 budgeted search.',
                       'Derived TACO IDs preserve source polygons but are not original UIDs; no new human annotations.'],
        'paid_gpu': False, 'training': False, 'new_imagery': False, 'reserved_sets_enabled': False}
    freeze(out / 'protocol.json', protocol)
    print(json.dumps({'frozen': str(out / 'protocol.json'), 'production_calls': 578, 'reference_scenes': 72}), flush=True)


def verify_protocol(out, protocol):
    for n, value in protocol['source_code_sha256'].items(): checked(HERE / n, value)
    for n, key in [('rgb_jobs.json', 'rgb_jobs_sha256'), ('reference_manifest.json', 'reference_manifest_sha256'), ('test_log.txt', 'test_log_sha256')]:
        checked(out / n, protocol[key])
    for model_id in MODELS:
        row = protocol['models'][model_id]
        for n, value in row.items():
            if n in ('model.safetensors', 'config.json', 'changeos_r34.pt', 'official_wrapper.py', 'README.upstream.md'):
                checked(Path(row['directory']) / n, value)


def load_pair(job):
    result = []
    for row in job['images']:
        checked(row['path'], row['sha256'])
        with Image.open(row['path']) as im: result.append(im.convert('RGB'))
    return result


def load_cache(row):
    checked(row['cache_file'], row['cache_sha256'])
    with np.load(row['cache_file'], allow_pickle=False) as d:
        building, classes = d['building'], d['classes']
    size = row['input_shape'][-2:]
    if building.dtype != np.bool_ or classes.dtype != np.uint8 or list(building.shape) != size or classes.shape != building.shape:
        raise ValueError('Invalid prediction cache')
    if not set(np.unique(classes)).issubset({0, 1, 2, 3, 4}): raise ValueError('Invalid damage class')
    return building, classes


def predict(out, protocol):
    for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY']: os.environ[key] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    import torch
    from safetensors.torch import load_model
    from torchange.models.changeos import ChangeOS
    direct = importlib.metadata.distribution('torchange').read_text('direct_url.json')
    if not direct or CODE_COMMIT not in direct: raise ValueError('Model source commit mismatch')
    torch.set_num_threads(4); torch.manual_seed(20261025)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    for n in ('calls', 'cache'): (out / n).mkdir(exist_ok=True)
    jobs = read(out / 'rgb_jobs.json'); rows = []; index = 0; new_calls = 0
    protocol_sha = digest(out / 'protocol.json'); started = time.perf_counter()
    def forward(model, model_id, identity, pair):
        nonlocal index, new_calls
        index += 1
        if index > protocol['call_cap']: raise ValueError('Forward cap exceeded')
        tensor = prepare_tensor(*pair, 'native') if model_id == 'swint' else prepare_r34(*pair)
        identity = {**identity, 'index': index, 'model_id': model_id, 'protocol_sha256': protocol_sha,
                    'input_shape': list(tensor.shape), 'input_tensor_sha256': sha(tensor.numpy().tobytes())}
        record = out / 'calls' / f'{index:03d}.json'
        if record.exists():
            row = read(record)
            if row['status'] != 'COMPLETE' or any(row[k] != v for k, v in identity.items()):
                raise ValueError('Incomplete or different reservation; do not silently repeat forward')
            return row, *load_cache(row)
        freeze(record, {**identity, 'status': 'RESERVED'})
        t = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor.to(device)); sync(device)
            building, classes = decode_output(output, model_id, tuple(tensor.shape[-2:]))
        new_calls += 1
        cache = out / 'cache' / f'{index:03d}.npz'
        np.savez_compressed(cache, building=building, classes=classes)
        row = {**identity, 'status': 'COMPLETE', 'device': str(device), 'elapsed_seconds': time.perf_counter() - t,
               'cache_file': str(cache.resolve()), 'cache_sha256': digest(cache)}
        atomic_json(record, row)
        return row, building, classes
    for model_id in MODELS:
        if model_id == 'swint':
            config = read(SWIN / 'config.json')['config']; config['encoder']['params']['weights'] = None
            model = ChangeOS(config).eval()
            missing, unexpected = load_model(model, str(SWIN / 'model.safetensors'), strict=True)
            if missing or unexpected: raise ValueError('Strict checkpoint load failed')
            model.requires_grad_(False)
        else:
            model = torch.jit.load(str(R34 / 'changeos_r34.pt'), map_location='cpu').eval()
        model.to(device); sync(device)
        if model_id == 'swint':
            for control in protocol['positive_controls']:
                pair = crop_pair(*load_pair(control), control['cell_id'])
                row, loc, classes = forward(model, model_id, {'role': 'regression_control', 'sample_id': control['sample_id'],
                                                             'cell_id': control['cell_id']}, pair)
                if row['input_tensor_sha256'] != control['prepared_tensor_sha256']: raise ValueError('Old control input differs')
                checked(control['legacy_cache_file'], control['legacy_cache_sha256'])
                with np.load(control['legacy_cache_file'], allow_pickle=False) as d: old = d['severe_mask']
                if not np.array_equal(old, loc & (classes >= 3)): raise ValueError('Old positive control mismatch')
                rows.append(row)
        for i, job in enumerate(jobs):
            pair = load_pair(job)
            for tile, bounds in enumerate(TILES):
                row, _, _ = forward(model, model_id, {'role': 'production', 'sample_id': job['sample_id'],
                    'event': job['event'], 'tile_id': tile, 'bounds': list(bounds)}, [im.crop(bounds) for im in pair])
                rows.append(row)
            print(json.dumps({'model': model_id, 'scene_complete': i + 1, 'calls_complete': index, 'device': str(device)}), flush=True)
        del model; gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()
    if len(rows) != 578 or index != 578: raise ValueError('Incomplete call inventory')
    atomic_json(out / 'observations.json', rows)
    atomic_json(out / 'inference_report.json', {'status': 'COMPLETE', 'calls': 578, 'new_calls_this_invocation': new_calls,
        'device': str(device), 'loop_seconds': time.perf_counter() - started, 'sum_recorded_call_seconds': sum(r['elapsed_seconds'] for r in rows),
        'paid_gpu': False, 'training': False, 'new_network_by_prediction': 0, 'labels_decoded_by_prediction': False,
        'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'torchange', 'numpy', 'Pillow']}})


def assemble(rows):
    loc = np.zeros((1024, 1024), bool); classes = np.zeros((1024, 1024), np.uint8)
    if len(rows) != 4 or {r['tile_id'] for r in rows} != {0, 1, 2, 3}: raise ValueError('Tiles missing or repeated')
    for row in rows:
        building, damage = load_cache(row); x0, y0, x1, y1 = TILES[row['tile_id']]
        loc[y0:y1, x0:x1] = building; classes[y0:y1, x0:x1] = damage
    return loc, classes


def legacy_masks(protocol):
    path = LEGACY / 'observations.json'; checked(path, protocol['legacy128_secondary']['observations_sha256'])
    grouped = defaultdict(list)
    for r in read(path):
        if r['mode'] == 'native': grouped[r['sample_id']].append(r)
    masks = {}
    for sid, rows in grouped.items():
        if len(rows) != 64 or len({r['cell_id'] for r in rows}) != 64: raise ValueError('Old128 inventory differs')
        prediction = np.zeros((1024, 1024), np.uint8)
        for r in rows:
            checked(r['cache_file'], r['cache_sha256'])
            with np.load(r['cache_file'], allow_pickle=False) as d: mask = d['severe_mask']
            if mask.shape != (128, 128) or mask.dtype != np.bool_: raise ValueError('Legacy shape differs')
            y, x = int(r['cell_id'][1:3]) * 128, int(r['cell_id'][4:6]) * 128
            prediction[y:y+128, x:x+128] = mask.astype(np.uint8) * 3
        masks[sid] = prediction
    if len(masks) != 72: raise ValueError('Legacy cohort differs')
    return masks


def evaluate(out, protocol):
    references = read(out / 'reference_manifest.json'); rows = read(out / 'observations.json')
    if len(rows) != 578 or any(r['status'] != 'COMPLETE' for r in rows): raise ValueError('Incomplete predictions')
    grouped = defaultdict(list)
    for row in rows:
        if row['role'] == 'production': grouped[row['sample_id'], row['model_id']].append(row)
    legacy = legacy_masks(protocol); scores = []; pooled = defaultdict(lambda: np.zeros((5, 5), np.int64))
    for ref in references:
        checked(ref['reference_file'], ref['reference_sha256'])
        truth = np.asarray(Image.open(ref['reference_file']))
        for model_id in ('swint', 'r34', 'legacy_swint128'):
            if model_id == 'legacy_swint128':
                maps = {'raw_pixel': legacy[ref['sample_id']]}
            else:
                loc, classes = assemble(grouped[ref['sample_id'], model_id])
                maps = {'raw_pixel': classes * loc.astype(np.uint8), 'common_object_vote': object_vote(loc, classes)}
            for decoder, prediction in maps.items():
                cm = confusion(truth, prediction); summary = metrics(cm)
                scores.append({'sample_id': ref['sample_id'], 'event': ref['event'], 'model_id': model_id,
                    'decoder': decoder, **summary, 'confusion_5x5': cm.tolist(),
                    'full_grade_confusion_valid': model_id != 'legacy_swint128'})
                pooled[model_id, decoder, ref['event']] += cm
    if len(scores) != 360: raise ValueError('Score inventory differs')
    event_scores = []; totals = []
    for (model_id, decoder, event), cm in sorted(pooled.items()):
        matching = [r for r in scores if r['model_id'] == model_id and r['decoder'] == decoder and r['event'] == event]
        event_scores.append({'model_id': model_id, 'decoder': decoder, 'event': event, 'scenes': len(matching),
                             **metrics(cm), 'scenes_with_any_reference_overlap': sum(r['tp'] > 0 for r in matching),
                             'reference_positive_scenes': sum(r['reference_severe_pixels'] > 0 for r in matching),
                             'confusion_5x5': cm.tolist()})
    for model_id, decoder in sorted({(r['model_id'], r['decoder']) for r in event_scores}):
        events = [r for r in event_scores if r['model_id'] == model_id and r['decoder'] == decoder]
        cm = sum((np.asarray(r['confusion_5x5'], np.int64) for r in events), np.zeros((5, 5), np.int64))
        totals.append({'model_id': model_id, 'decoder': decoder, **metrics(cm),
                       'equal_event_macro_f1': float(np.mean([r['f1'] for r in events])),
                       'reference_positive_scenes': sum(r['reference_positive_scenes'] for r in events),
                       'scenes_with_any_reference_overlap': sum(r['scenes_with_any_reference_overlap'] for r in events)})
    atomic_json(out / 'scene_scores.json', scores); atomic_json(out / 'event_scores.json', event_scores)
    atomic_json(out / 'summary.json', {'status': 'COMPLETE', 'scenes': 72, 'models': 2, 'totals': totals,
        'primary': 'raw_pixel/equal_event_macro_f1', 'secondary_vote': 'same published weighted vote forbothmodels',
        'not_official_mixed_f1': True, 'legacy_context_comparison_secondary': True,
        'independent_efficacy_established': False, 'new_method_established': False, 'gpu_rented': False})
    print(json.dumps(totals), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'output/disaster_search_perception_pair72_v1')
    p.add_argument('--stage', choices=['plan', 'predict', 'evaluate'], required=True)
    args = p.parse_args(); out = args.out.resolve()
    if (out / 'completion_receipt.json').exists(): raise ValueError('Completed output frozen')
    if args.stage == 'plan': plan(out); return
    protocol = read(out / 'protocol.json'); verify_protocol(out, protocol)
    if args.stage == 'predict': predict(out, protocol)
    else: evaluate(out, protocol)


if __name__ == '__main__': main()
