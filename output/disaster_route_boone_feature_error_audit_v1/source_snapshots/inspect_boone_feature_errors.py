"""Post-hoc attribution of the fixed spatial versus visual-spatial AP difference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(path.read_text())


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def decompose(scores, labels):
    order = sorted(scores, key=lambda uid: (-scores[uid], uid))
    positives = sum(labels[uid] for uid in order)
    if not positives:
        raise ValueError('This audit expects known positive references')
    contributions = {}
    hits = 0
    for rank, uid in enumerate(order, 1):
        if labels[uid]:
            hits += 1
            contributions[uid] = hits / rank / positives
    return order, contributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep prior diagnostics unchanged; use a new directory')
    root, out = args.workspace.resolve(), args.output.resolve()
    run = root / 'output/disaster_route_boone_context_run_v2'
    crop_manifest = read(run / 'crop_manifest.json')
    crop_rows = {r['building_id']: r for r in crop_manifest['rows']}
    ids = sorted(crop_rows)
    names = {uid: crop_rows[uid]['display_id'] for uid in ids}
    by_name = {name: uid for uid, name in names.items()}
    ref_path = root / 'output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json'
    labels = {r['building_id']: r['view_binary']['uas'] for r in read(ref_path)['rows']}
    predictions = {}
    for row in read(run / 'prediction_rows.json'):
        predictions.setdefault((row['anchor_id'], row['method']), {})[row['target_id']] = row['score']
    f = read(run / 'satellite_features.json')
    feature_index = {uid: i for i, uid in enumerate(f['building_ids'])}
    similarity, xy = np.array(f['similarity']), np.array(f['xy_m'])
    cases, targets = [], []
    target_contribution = {uid: np.zeros(2) for uid in ids if labels[uid]}
    for anchor in ids:
        spatial = predictions[anchor, 'S1_spatial']
        visual = predictions[anchor, 'S2_visual_spatial']
        order_s, c_s = decompose(spatial, labels)
        order_v, c_v = decompose(visual, labels)
        ap_s, ap_v = sum(c_s.values()), sum(c_v.values())
        for uid in c_s:
            target_contribution[uid] += np.array([c_s[uid], c_v[uid]]) / len(ids)
        cases.append({'anchor_id': anchor, 'anchor_name': names[anchor],
                      'anchor_uas_label': labels[anchor], 'ap_spatial': ap_s,
                      'ap_visual': ap_v, 'ap_delta': ap_v - ap_s,
                      'spatial_order': [names[uid] for uid in order_s],
                      'visual_order': [names[uid] for uid in order_v]})
        for uid in order_s:
            ia, it = feature_index[anchor], feature_index[uid]
            targets.append({'anchor_id': anchor, 'anchor_name': names[anchor],
                            'target_id': uid, 'target_name': names[uid],
                            'target_uas_label': labels[uid],
                            'distance_m': float(np.linalg.norm(xy[ia]-xy[it])),
                            'visual_similarity': float(similarity[ia, it]),
                            'initial_score': predictions[anchor, 'S0_no_update'][uid],
                            'spatial_score': spatial[uid], 'visual_score': visual[uid],
                            'spatial_rank': order_s.index(uid)+1,
                            'visual_rank': order_v.index(uid)+1,
                            'ap_contribution_spatial': c_s.get(uid, 0),
                            'ap_contribution_visual': c_v.get(uid, 0)})
    summary_table = read(run / 'summary.json')['table']
    expected = {r['method']: r['ap'] for r in summary_table if r['reference_source']=='uas'}
    macro_s = float(np.mean([c['ap_spatial'] for c in cases]))
    macro_v = float(np.mean([c['ap_visual'] for c in cases]))
    assert abs(macro_s-expected['S1_spatial']) < 1e-12
    assert abs(macro_v-expected['S2_visual_spatial']) < 1e-12
    attribution = [{'target_id': uid, 'target_name': names[uid],
                    'spatial_macro_ap_contribution': float(value[0]),
                    'visual_macro_ap_contribution': float(value[1]),
                    'delta': float(value[1]-value[0])}
                   for uid, value in target_contribution.items()]
    assert abs(sum(r['delta'] for r in attribution)-(macro_v-macro_s)) < 1e-12
    summary = {'analysis_type': 'post_hoc_error_audit_not_new_experiment',
               'spatial_ap': macro_s, 'visual_ap': macro_v, 'delta': macro_v-macro_s,
               'improved_anchor_cases': sum(c['ap_delta'] > 1e-12 for c in cases),
               'worse_anchor_cases': sum(c['ap_delta'] < -1e-12 for c in cases),
               'tied_anchor_cases': sum(abs(c['ap_delta']) <= 1e-12 for c in cases),
               'positive_target_contributions': attribution,
               'reference': 'Published UAS-view labels', 'new_model_calls': 0,
               'input_hashes': {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                                for name in ['prediction_rows.json', 'satellite_features.json',
                                             'crop_manifest.json', 'summary.json']}}
    out.mkdir(parents=True)
    dump(out / 'summary.json', summary)
    dump(out / 'per_anchor.json', cases)
    with (out / 'target_rank_changes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(targets[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(targets)
    fonts = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(fonts), 17) if fonts.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(fonts), 14) if fonts.exists() else font
    for anchor_name, selected in [('B11', ['B11', 'B09', 'B10', 'B02']),
                                   ('B04', ['B04', 'B01', 'B07', 'B09']),
                                   ('B12', ['B12', 'B11', 'B09'])]:
        anchor = by_name[anchor_name]
        canvas = Image.new('RGB', (880, 70 + 194*len(selected)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), f'Anchor {anchor_name}: negative UAS-view feedback; diagnostic images only', font=font, fill='black')
        for i, title in enumerate(['Pre satellite', 'Post satellite', 'UAS (evaluation view)']):
            draw.text((278+i*198, 43), title, font=small, fill='black')
        for ri, name in enumerate(selected):
            uid = by_name[name]
            y = 70 + ri*194
            draw.line((10, y-4, 870, y-4), fill='black')
            draw.text((14, y+4), f'{name}  UAS label={labels[uid]}', font=font, fill='black')
            draw.text((14, y+30), uid[:12], font=small, fill='black')
            if uid == anchor:
                draw.text((14, y+58), 'Revealed anchor', font=font, fill='black')
            else:
                row = next(t for t in targets if t['anchor_id']==anchor and t['target_id']==uid)
                draw.text((14, y+58), f'Rank: {row["spatial_rank"]} -> {row["visual_rank"]}', font=font, fill='black')
                draw.text((14, y+87), f'Similarity: {row["visual_similarity"]:.3f}', font=small, fill='black')
                draw.text((14, y+112), f'Spatial score: {row["spatial_score"]:.4f}', font=small, fill='black')
                draw.text((14, y+137), f'Visual score: {row["visual_score"]:.4f}', font=small, fill='black')
            for col, source in enumerate(('pre', 'post', 'uas')):
                info = crop_rows[uid]['crops'][source]
                path = run / info['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
                im = Image.open(path).convert('RGB').resize((180, 180))
                canvas.paste(im, (278+col*198, y))
        canvas.save(out / f'case_{anchor_name}_panel.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
