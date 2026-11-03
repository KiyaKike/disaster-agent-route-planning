"""Small local-VLM experiment on fixed Boone crops; no training or external API."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_crop(image, polygon):
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], fill=255)
    box = mask.getbbox()
    if box is None:
        raise ValueError('Empty footprint')
    masked = Image.new('RGB', image.size, (128, 128, 128))
    masked.paste(image, mask=mask)
    original = np.asarray(image)[np.asarray(mask) > 0].astype(np.float64) / 255
    crop = masked.crop(box)
    side = max(crop.size)
    square = Image.new('RGB', (side, side), (128, 128, 128))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    # Nearest neighbor keeps masked background exact at display resolution.
    square = square.resize((256, 256), Image.Resampling.NEAREST)
    outside = np.asarray(mask) == 0
    assert np.all(np.asarray(masked)[outside] == 128)
    return square, original, list(box), int((~outside).sum())


def prepare(root, out):
    pilot = root / 'output/disaster_route_boone_context_pilot_v1'
    public = json.loads((pilot / 'public_case_index.json').read_text())
    region = root / 'output/disaster_route_crasar_region_rank2_v1'
    noaa = root / 'output/disaster_route_noaa_entry_v1'
    paths = {'pre': region / 'pre_roi.png', 'post': region / 'post_roi.png',
             'uas': region / 'uas_roi.png',
             'noaa_raw_qa_only': noaa / 'crewed_raw_georef_roi.png',
             'noaa_aligned_qa_only': noaa / 'crewed_aligned_roi.png'}
    images = {k: Image.open(p).convert('RGB') for k, p in paths.items()}
    assert len({im.size for im in images.values()}) == 1
    geometry = {g['building_id']: g for g in public['geometry']}
    ids = sorted(geometry)
    rows, features, centers = [], [], []
    for i, uid in enumerate(ids):
        g = geometry[uid]
        points = np.asarray(g['polygon_utm'], dtype=np.float64)
        # Fixed footprint-vertex center; sufficient for this development kernel.
        centers.append(points[:-1].mean(axis=0).tolist())
        crops, pixels = {}, {}
        for source, image in images.items():
            crop, values, box, area = masked_crop(image, g['roi_polygon_px'])
            rel = f'crops/{uid}/{source}.png'
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path)
            crops[source] = {'path': rel, 'sha256': sha(path),
                             'roi_box': box, 'mask_pixels': area}
            pixels[source] = values
        pre, post = pixels['pre'], pixels['post']
        hist = np.concatenate([np.histogram(post[:, c], bins=8, range=(0, 1))[0]
                               / len(post) for c in range(3)])
        # Only pre/post satellite pixels enter the transfer features.
        delta = post - pre
        f = np.concatenate([hist, post.mean(axis=0), post.std(axis=0),
                            delta.mean(axis=0), delta.std(axis=0)])
        features.append(f)
        rows.append({'building_id': uid, 'display_id': f'B{i+1:02d}', 'crops': crops})
    features = np.asarray(features)
    standardized = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), .05)
    distances = np.linalg.norm(standardized[:, None] - standardized[None, :], axis=-1)
    tau = float(np.median(distances[distances > 0]))
    similarity = np.exp(-distances ** 2 / (2 * tau ** 2))
    dump(out / 'crop_manifest.json', {'rows': rows, 'source_images': {
        k: {'path': str(p.relative_to(root)), 'sha256': sha(p)} for k, p in paths.items()},
        'scope': 'Footprint-masked crops; no independent registration certification'})
    dump(out / 'satellite_features.json', {
        'building_ids': ids, 'xy_m': centers, 'features': features.tolist(),
        'standardized_features': standardized.tolist(), 'similarity': similarity.tolist(),
        'tau': tau, 'feature_scope': '24 post RGB histogram bins + 6 post mean/std + 6 pre-post delta mean/std',
        'normalization': 'Unlabelled full satellite cohort; std floor0.05; median positive pairwise distance',
        'uses_uas_or_noaa_pixels': False})
    for start in (0, 6):
        sheet = Image.new('RGB', (720, 6 * 170 + 30), 'white')
        draw = ImageDraw.Draw(sheet)
        for col, source in enumerate(('pre', 'post', 'uas', 'noaa_raw_qa_only')):
            draw.text((col * 180 + 8, 8), source, fill='black')
        for ri, row in enumerate(rows[start:start + 6]):
            for col, source in enumerate(('pre', 'post', 'uas', 'noaa_raw_qa_only')):
                im = Image.open(out / row['crops'][source]['path']).resize((144, 144))
                x, y = col * 180 + 8, 30 + ri * 170
                sheet.paste(im, (x, y))
                draw.text((x, y + 146), row['display_id'] + ' ' + row['building_id'][:6], fill='black')
        sheet.save(out / f'crop_panel_{start//6+1}.png')
    return public, ids


def grouped_ap(scores, labels):
    positives = sum(labels)
    if not positives:
        return None
    groups = {}
    for score, label in zip(scores, labels):
        groups.setdefault(float(score), []).append(label)
    hit = seen = 0
    ap = 0.
    for score in sorted(groups, reverse=True):
        block = groups[score]
        new = sum(block)
        hit += new
        seen += len(block)
        ap += (new / positives) * (hit / seen)
    return ap


def metrics(predictions, labels):
    pairs = [(float(p), int(y)) for p, y in zip(predictions, labels) if y is not None]
    ps, ys = zip(*pairs)
    eps = 1e-6
    order = sorted(range(len(ps)), key=lambda i: (-ps[i], i))
    return {'brier': sum((p-y)**2 for p, y in pairs) / len(pairs),
            'log_loss': -sum(y*math.log(max(eps, min(1-eps, p)))
                            +(1-y)*math.log(max(eps, min(1-eps, 1-p)))
                            for p, y in pairs) / len(pairs),
            'ap': grouped_ap(ps, ys), 'top3_positive': sum(ys[i] for i in order[:3])}


def score_statistics(p0, anchor, feedback, xy, similarity):
    distance = np.linalg.norm(xy - xy[anchor], axis=1)
    spatial = np.exp(-distance**2 / (2 * 50**2))
    residual = feedback - p0[anchor]
    return {'S0_no_update': p0.copy(),
            'S1_spatial': np.clip(p0 + .5 * spatial * residual, 0, 1),
            'S2_visual_spatial': np.clip(p0 + .5 * spatial * similarity[anchor] * residual, 0, 1)}


def run_vlm(root, out, revision, limit):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForVision2Seq
    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float32
    started = time.time()
    processor = AutoProcessor.from_pretrained(MODEL, revision=revision)
    # Crops are small; avoid the default2048px resize and many redundant tiles.
    processor.image_processor.size = {'longest_edge': 512}
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL, revision=revision, torch_dtype=dtype,
        attn_implementation='eager').to(device).eval()
    manifest = json.loads((out / 'crop_manifest.json').read_text())
    rows = manifest['rows']
    ids = [r['building_id'] for r in rows]
    refs = json.loads((root / 'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json').read_text())
    labels = {r['building_id']: r['view_binary'] for r in refs['rows']}
    # SmolVLM often emits a leading-space token. Include both surface forms.
    answer_forms = ('A', ' A', 'B', ' B')
    answer_ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in answer_forms]
    assert all(len(x) == 1 for x in answer_ids), answer_ids
    options = [x[0] for x in answer_ids]
    assert len(set(options)) == 4
    dump(out / 'runtime.json', {'model': MODEL, 'revision': revision,
         'device': device, 'torch': torch.__version__, 'transformers': transformers.__version__,
         'score': 'Class probability over bare and leading-space A versus B token variants',
         'answer_token_forms': list(answer_forms),
         'answer_token_ids': options, 'dtype': 'float32', 'training': False,
         'processor_longest_edge': 512, 'image_splitting': False,
         'external_api_calls': 0, 'gpu_rental': False})

    def infer(target, anchor=None, label_only=False):
        target_id = ids[target]
        key = f'prior_{target:02d}' if anchor is None else f'history_{anchor:02d}_{target:02d}'
        if label_only:
            key = f'label_only_{anchor:02d}_{target:02d}'
        cache = out / 'calls' / f'{key}.json'
        if cache.exists():
            return json.loads(cache.read_text())['score']
        content, images, image_paths = [], [], []

        def add_image(row, source, description):
            rel = row['crops'][source]['path']
            content.extend([{'type': 'text', 'text': description}, {'type': 'image'}])
            images.append(Image.open(out / rel).convert('RGB'))
            image_paths.append(rel)

        content.append({'type': 'text', 'text':
            'Assess visible roof damage in the TARGET building from satellite images. '
            'The images show only individual building footprints. '
            'A means visible roof damage; B means no visible roof damage. '})
        if anchor is not None:
            r = rows[anchor]
            if not label_only:
                add_image(r, 'pre', 'HISTORY building: pre-disaster satellite.')
                add_image(r, 'post', 'HISTORY building: post-disaster satellite.')
                add_image(r, 'uas', 'HISTORY building: paid UAS image, captured on the next catalogue day.')
            label = 'A (damage)' if labels[ids[anchor]]['uas'] == 1 else 'B (no damage)'
            content.append({'type': 'text', 'text':
                f'The published UAS-view annotation of the HISTORY building is {label}. '
                'This is a view-specific observation, not error-free field truth. '
                'Use the history only if its visual pattern transfers to the TARGET. '})
        add_image(rows[target], 'pre', 'TARGET building: pre-disaster satellite.')
        add_image(rows[target], 'post', 'TARGET building: post-disaster satellite.')
        content.append({'type': 'text', 'text':
            'Which label best describes the TARGET roof? Reply with exactly A or B.'})
        messages = [{'role': 'user', 'content': content}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors='pt').to(device)
        call_started = time.time()
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            restricted = torch.softmax(logits[options], dim=-1)
        score = float(restricted[:2].sum().cpu())
        record = {'target_id': target_id, 'anchor_id': None if anchor is None else ids[anchor],
                  'label_only_history': label_only,
                  'score': score, 'answer_set_probability_mass': float(probs[options].sum().cpu()),
                  'unrestricted_top_token': processor.tokenizer.decode([int(logits.argmax().cpu())]),
                  'seconds': time.time()-call_started, 'input_tokens': int(inputs['input_ids'].shape[-1]),
                  'image_paths': image_paths, 'prompt': prompt, 'revision': revision}
        assert all('/uas.png' not in p or (anchor is not None and ids[anchor] in p) for p in image_paths)
        dump(cache, record)
        print(f'{key} score={score:.5f} {record["seconds"]:.2f}s', flush=True)
        del inputs, logits, probs, restricted
        return score

    p0 = np.array([infer(i) for i in range(len(ids))])
    dump(out / 'vlm_initial_scores.json', dict(zip(ids, p0.tolist())))
    features = json.loads((out / 'satellite_features.json').read_text())
    xy, similarity = np.asarray(features['xy_m']), np.asarray(features['similarity'])
    results, aggregate, detail = [], {}, []
    count = len(ids) if limit is None else min(limit, len(ids))
    # Model inputs get only one anchor record; held-out target labels are used below.
    for anchor in range(count):
        predictions = score_statistics(p0, anchor, labels[ids[anchor]]['uas'], xy, similarity)
        targets = [i for i in range(len(ids)) if i != anchor]
        history_scores = p0.copy()
        for target in targets:
            history_scores[target] = infer(target, anchor)
        predictions['V1_vlm_history'] = history_scores
        label_scores = p0.copy()
        for target in targets:
            label_scores[target] = infer(target, anchor, label_only=True)
        predictions['V2_vlm_label_only'] = label_scores
        for method, scores in predictions.items():
            for target in targets:
                detail.append({'anchor_id': ids[anchor], 'target_id': ids[target],
                               'method': method, 'score': float(scores[target])})
            for source in ('uas', 'crewed'):
                m = metrics([scores[i] for i in targets], [labels[ids[i]][source] for i in targets])
                row = {'anchor_id': ids[anchor], 'method': method, 'reference_source': source, **m}
                results.append(row)
                aggregate.setdefault((source, method), []).append(m)
    table = []
    for (source, method), data in aggregate.items():
        table.append({'reference_source': source, 'method': method,
                      **{k: float(np.mean([m[k] for m in data if m[k] is not None]))
                         for k in ('brier', 'log_loss', 'ap', 'top3_positive')}})
    dump(out / 'per_anchor_scores.json', results)
    dump(out / 'prediction_rows.json', detail)
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    calls = [json.loads(p.read_text()) for p in sorted((out / 'calls').glob('*.json'))]
    summary = {'status': 'LOCAL_SMALL_VLM_COMPARISON_COMPLETE', 'unique_buildings': len(ids),
               'anchor_cases': count, 'target_rows_per_method': count*(len(ids)-1),
               'model': MODEL, 'revision': revision, 'table': table,
               'forward_calls': len(calls), 'measured_forward_seconds': sum(c['seconds'] for c in calls),
               'min_answer_probability_mass': min(c['answer_set_probability_mass'] for c in calls),
               'elapsed_seconds': time.time()-started, 'external_api_cost': 0, 'gpu_rental': False,
               'reference_scope': 'Published view-label agreement on one development region; not common field truth',
               'vlm_scope': 'Small local checkpoint feasibility trial; not a conclusion about all VLMs',
               'new_route_claims': False}
    dump(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--revision')
    parser.add_argument('--limit-anchors', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Freeze the executable recipe before inspecting prediction results.
    recipe = {'model': MODEL, 'revision': args.revision,
              'spatial_scale_m': 50, 'residual_gain': .5,
              'features': '24 RGB histogram + post mean/std + pre/post delta mean/std',
              'feature_std_floor': .05, 'visual_tau': 'median positive unlabelled distance',
              'score': 'next-token class probability combining bare and leading-space A/B tokens',
              'label_only_ablation': True,
              'processor_longest_edge': 512, 'image_splitting': False,
              'anchors': args.limit_anchors or 12,
              'script_sha256': sha(Path(__file__)), 'seed': 0}
    previous = out / 'execution_recipe.json'
    if previous.exists():
        if json.loads(previous.read_text()) != recipe:
            raise ValueError('Recipe differs; use a new output directory')
    else:
        dump(previous, recipe)
    if not (out / 'crop_manifest.json').exists():
        prepare(args.workspace.resolve(), out)
    if not args.prepare_only:
        if not args.revision:
            raise ValueError('Pin a model revision')
        run_vlm(args.workspace.resolve(), out, args.revision, args.limit_anchors)


if __name__ == '__main__':
    main()
