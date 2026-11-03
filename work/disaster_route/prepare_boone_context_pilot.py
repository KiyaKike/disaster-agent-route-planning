"""Prepare a development-only case index, without images, API calls or predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def binary_label(label):
    if label == 'no damage':
        return 0
    if label in {'minor damage', 'major damage', 'destroyed'}:
        return 1
    return None


def build_cases(geometry):
    # Selection uses the old geometric gate only, never labels or predictions.
    ids = sorted(g['building_id'] for g in geometry if g['eligible'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate building IDs')
    return ids, [
        {'case_id': f'anchor_{i:02d}', 'anchor_id': uid,
         'target_ids': [other for other in ids if other != uid],
         'feedback_source': 'published_UAS_view_annotation_and_image',
         'reference_scope': 'development_archive_source_label_prediction'}
        for i, uid in enumerate(ids)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    region = root / 'output/disaster_route_crasar_region_rank2_v1'
    noaa = root / 'output/disaster_route_noaa_entry_v1'
    sources = [region / 'public_geometry.json',
               region / 'public_planning_inputs_any_damage.json',
               noaa / 'private_view_reference_comparison.json',
               noaa / 'entry_assessment.json']
    geometry, planning, references, assessment = map(load, sources)
    ids, cases = build_cases(geometry)
    if len(ids) != 12:
        raise ValueError('This protocol fixes the old 12-building cohort')
    if set(planning['building_ids']) != set(ids):
        raise ValueError('Prior score cohort differs')
    prior = dict(zip(planning['building_ids'], planning['prior_damage_score']))
    by_id = {r['building_id']: r for r in references}
    selected = [by_id[uid] for uid in ids]
    if not all(r['existing_geometry_eligible'] for r in selected):
        raise ValueError('Reference cohort differs')
    # No labels or future-source images are placed in the public case index.
    public = {'development_only': True,
              'geometry': [{k: g[k] for k in ('building_id', 'roi_polygon_px',
                                             'polygon_utm')}
                           for g in geometry if g['building_id'] in ids],
              'legacy_damage_pixel_fraction': prior,
              'legacy_score_is_building_damage_probability': False,
              'vlm_no_history_scores': None,
              'cases': cases}
    evaluation = {
        'scope': 'Evaluator-only view-specific references; not field truth',
        'rows': [{'building_id': uid,
                  'view_labels': by_id[uid]['view_labels'],
                  'view_binary': {view: binary_label(label) for view, label
                                  in by_id[uid]['view_labels'].items()}}
                 for uid in ids]}
    summary = {
        'unique_regions': 1, 'unique_buildings': len(ids),
        'overlapping_anchor_cases': len(cases),
        'target_prediction_rows_not_independent': sum(len(c['target_ids']) for c in cases),
        'positive_labels_by_source': {
            view: sum(binary_label(r['view_labels'][view]) == 1 for r in selected)
            for view in ('satellite', 'uas', 'crewed')},
        'label_histograms': {view: dict(Counter(r['view_labels'][view] for r in selected))
                             for view in ('satellite', 'uas', 'crewed')},
        'legacy_score_range': [min(prior.values()), max(prior.values())],
        'legacy_score_semantics': 'Damage model pixel fraction; not building probability',
        'common_field_truth_established': assessment['common_field_truth_established'],
        'common_state_established': assessment['common_state_established'],
        'status': 'PROTOCOL_AND_CASE_INDEX_ONLY',
        'new_model_calls': 0, 'gpu_rental': False,
        'method_comparison_performed': False,
        'next': 'Prepare footprint-masked crops and inspect correspondence; freeze visual statistical baseline before model comparison',
        'input_sha256': {p.relative_to(root).as_posix():
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    assert all(c['anchor_id'] not in c['target_ids'] for c in cases)
    args.output.mkdir(parents=True)
    for name, data in [('public_case_index.json', public),
                       ('evaluator_only_view_references.json', evaluation),
                       ('preparation_summary.json', summary)]:
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
