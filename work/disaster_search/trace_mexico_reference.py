"""Cross-check exactly3 already-used development scenes against published masks.

No new RGB, model calls, training, source-object selection or reserved-set score.
Parquet mask columns for shared row groups are transient; only selected masks
may be retained. This does not recover original polygon JSON or original UIDs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_xbd_mini import atomic_json, sha
from extract_external_area_masks import verify_parent
from plan_search_v2 import freeze
from run_perception_pair72 import ROOT, HERE, digest, read, checked, assemble
from mexico_reference_core import footer_metadata, column_interval, allowed_read, selected_rows, decode_mask, object_mask, mask_correspondence

PARENT = ROOT / 'output/disaster_search_perception_pair72_v1'
NAMES = ROOT / 'output/disaster_search_data_admission_v1/shard_names'
OUT = ROOT / 'output/disaster_search_mexico_trace3_v1'
REPO = 'EVER-Z/torchange_xView2'
REVISION = 'cda47e44377258222166dac63a85068e8bdca446'
CAP = 8 * 1024 * 1024
SOURCE_NAMES = ['trace_mexico_reference.py', 'mexico_reference_core.py', 'test_mexico_reference.py',
                'run_perception_pair72.py', 'perception_pair72_core.py', 'build_xbd_mini.py',
                'plan_search_v2.py', 'extract_external_area_masks.py']


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists(): raise ValueError('Plan already frozen')
    parent_sha = verify_parent(PARENT)
    failures = read(PARENT / 'mexico_failure_description.json')['objects']
    wanted = sorted(r['sample_id'] for r in failures)
    if wanted != ['mexico-earthquake_00000081', 'mexico-earthquake_00000145', 'mexico-earthquake_00000162']:
        raise ValueError('Only all3 previous failures are permitted')
    locations = defaultdict(list)
    for path in sorted(NAMES.glob('hold-*.json')):
        info = read(path)
        if info['revision'] != REVISION: raise ValueError('Unpinned author metadata')
        for i, name in enumerate(info['image_names']):
            sid = name.removeprefix('hold/images/')
            if sid in wanted:
                locations[sid].append({'sample_id': sid, 'image_name': name, 'row_index': i, 'shard_path': info['path'],
                    'shard_size': info['remote_size'], 'source_names_file': str(path), 'source_names_sha256': digest(path),
                    'shard_rows': info['rows']})
    if set(locations) != set(wanted) or any(len(v) != 1 for v in locations.values()): raise ValueError('Ambiguous source identity')
    test_log = (OUT / 'test_log.txt').read_text()
    if 'Ran 341 tests' not in test_log or not test_log.rstrip().endswith('OK'): raise ValueError('341-test suite required')
    protocol = {'stage': 'THREE_FIXED_MEXICO_FAILURE_REFERENCE_CROSSCHECK', 'client_date': '2026-10-28',
        'parent_receipt_sha256': parent_sha, 'selected_ids': wanted, 'selection': 'All3 severe source objects in the prior failed event; no replacements',
        'source_repo': REPO, 'revision': REVISION, 'selected_locations': [locations[s][0] for s in wanted],
        'allowed_projected_columns': ['image_name', 't2_mask'], 'max_network_bytes': CAP,
        'shared_column_contract': 'A selected row group can physically contain other labels. Decode transiently in trusted extraction and return only these3 exact IDs; do not print, summarize or retain unselected mask payloads.',
        'forbidden': ['RGB column reads', 't1_mask reads', 'new scenes', 'new model calls', 'training', 'GPU rental', 'policy tuning', 'reserved-set scoring'],
        'class_encoding': '0background,1no-damage,2minor,3major,4destroyed;255ignored; no adaptive remapping',
        'comparisons': 'Full-scene severe IoU and exact class agreement on shared known pixels; published grade distribution inside each fixed original derived footprint. No threshold fitting or efficacy gate.',
        'qualitative_crops': 'One160x160 native crop centered at each fixed footprint bounding-box center, clipped to scene; displayed2x nearest, same bounds forbothdates. Only target outline overlaid; no relabelling.',
        'interpretation': 'Cross-representation consistency for the same underlying xBD release; not a second independent human truth, original JSON UID bridge or damage-observability verdict.',
        'tests_passed': 341, 'test_log_sha256': digest(OUT / 'test_log.txt'),
        'source_code_sha256': {n: digest(HERE / n) for n in SOURCE_NAMES}}
    freeze(OUT / 'protocol.json', protocol)
    print(json.dumps({'status': 'PLANNED', 'samples': 3, 'payload_cap': CAP, 'new_inference': 0}), flush=True)


class Transfer:
    def __init__(self):
        self.path = OUT / 'transfer_ledger.json'
        self.rows = read(self.path) if self.path.exists() else []
        self.reserved = sum(r['bytes'] for r in self.rows)

    def get(self, location, start, count, role):
        if not 0 <= start < start+count <= location['shard_size'] or count > 1024*1024: raise ValueError('Invalid bounded request')
        if self.reserved + count > CAP: raise ValueError('Network cap exceeded')
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{location['shard_path']}"
        self.reserved += count
        row = {'shard': location['shard_path'], 'offset': start, 'bytes': count, 'role': role, 'status': 'RESERVED',
               'at_utc': datetime.now(timezone.utc).isoformat()}
        self.rows.append(row); atomic_json(self.path, self.rows)
        request = Request(url, headers={'Range': f'bytes={start}-{start+count-1}', 'Accept-Encoding': 'identity'})
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f"bytes {start}-{start+count-1}/{location['shard_size']}":
                    raise ValueError('Exact range not honored; do not consume body')
                raw = response.read(count+1)
            if len(raw) != count: raise ValueError('Incomplete response')
            row.update(status='COMPLETE', received=len(raw), sha256=sha(raw)); atomic_json(self.path, self.rows)
            return raw
        except Exception as error:
            row.update(status='FAILED', error_type=type(error).__name__); atomic_json(self.path, self.rows)
            raise


class ColumnReader(io.RawIOBase):
    def __init__(self, transfer, location, allowed, forbidden):
        self.transfer, self.location = transfer, location
        self.allowed, self.forbidden = allowed, forbidden; self.position = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.location['shard_size']+offset
        if not 0 <= position <= self.location['shard_size']: raise ValueError('Seek outside file')
        self.position = position; return position
    def read(self, count=-1):
        if count == 0: return b''
        if count < 0: raise ValueError('No unbounded read')
        allowed_read(self.position, count, self.allowed, self.forbidden)
        chunks = []; remaining = count
        while remaining:
            n = min(remaining, 1024*1024)
            chunks.append(self.transfer.get(self.location, self.position, n, 'transient_selected_group_mask_or_identity'))
            self.position += n; remaining -= n
        return b''.join(chunks)
    def readinto(self, buffer):
        raw = self.read(len(buffer)); buffer[:len(raw)] = raw; return len(raw)


def fetch(protocol):
    import pyarrow.parquet as pq
    transfer = Transfer(); grouped = defaultdict(list)
    for location in protocol['selected_locations']: grouped[location['shard_path']].append(location)
    (OUT / 'published_masks').mkdir(exist_ok=True); outputs = []; metadata_records = []
    for shard, locations in sorted(grouped.items()):
        location = locations[0]; size = location['shard_size']
        tail = transfer.get(location, size-8, 8, 'parquet_trailer')
        footer_size = struct.unpack('<I', tail[:4])[0]
        if tail[4:] != b'PAR1' or not 0 < footer_size <= 256*1024: raise ValueError('Unexpected footer')
        footer = transfer.get(location, size-8-footer_size, footer_size, 'parquet_footer_metadata_only')
        metadata = footer_metadata(footer, tail)
        if metadata.num_rows != location['shard_rows']: raise ValueError('Source row count changed')
        selected_groups = []; allowed = []; forbidden = []; offset = 0
        for group_id in range(metadata.num_row_groups):
            group = metadata.row_group(group_id)
            wanted_group = any(offset <= r['row_index'] < offset+group.num_rows for r in locations)
            if wanted_group: selected_groups.append(group_id)
            for j in range(group.num_columns):
                column = group.column(j); interval = column_interval(column)
                if column.path_in_schema.split('.')[0] in ('image_name', 't2_mask') and wanted_group: allowed.append(interval)
                else: forbidden.append(interval)
            offset += group.num_rows
        if sum(b-a for a,b in allowed) + transfer.reserved > CAP: raise ValueError('Projected column bytes exceed remaining cap')
        reader = ColumnReader(transfer, location, allowed, forbidden)
        parquet = pq.ParquetFile(reader, metadata=metadata, pre_buffer=False, buffer_size=0)
        table = parquet.read_row_groups(selected_groups, columns=['image_name', 't2_mask'], use_threads=False)
        selected = selected_rows(table, [r['image_name'] for r in locations]); shared_rows = table.num_rows
        del table, parquet
        for row in locations:
            result = selected[row['image_name']]; raw = result['bytes']
            if raw is None or not 0 < len(raw) <= 1024*1024: raise ValueError('Missing selected PNG payload')
            array = decode_mask(raw); path = OUT / 'published_masks' / (row['sample_id']+'.png')
            if path.exists() and path.read_bytes() != raw: raise ValueError('Cached selected mask changed')
            path.write_bytes(raw)
            outputs.append({'sample_id': row['sample_id'], 'shard_path': shard, 'image_name': row['image_name'],
                            'path': str(path), 'sha256': sha(raw), 'pixel_sha256': sha(array.tobytes()),
                            'publisher_path_field': result.get('path'), 'selected_row_index_in_full_shard': row['row_index']})
        metadata_records.append({'shard_path': shard, 'rows': metadata.num_rows, 'selected_row_groups': selected_groups,
            'shared_projected_rows': shared_rows, 'selected_rows_returned': len(locations),
            'allowed_ranges': allowed, 'forbidden_ranges': forbidden, 'footer_sha256_not_persisted': sha(footer),
            'no_footer_column_statistics_exposed': True})
        print(json.dumps({'shard': shard, 'selected_masks': len(outputs), 'received': sum(r.get('received',0) for r in transfer.rows)}), flush=True)
    if len(outputs) != 3: raise ValueError('Incomplete selected inventory')
    atomic_json(OUT / 'published_mask_manifest.json', outputs)
    atomic_json(OUT / 'extraction_report.json', {'status': 'COMPLETE', 'metadata': metadata_records,
        'selected_masks': 3, 'received_bytes': sum(r.get('received',0) for r in transfer.rows), 'reserved_bytes': transfer.reserved,
        'request_count': len(transfer.rows), 'new_RGB_payloads': 0, 'new_model_calls': 0,
        'unselected_mask_payloads_persisted': False, 'unselected_outcomes_returned_or_scored': False})


def compare_and_render(protocol):
    refs = {r['sample_id']: r for r in read(PARENT / 'reference_manifest.json')}
    jobs = {r['sample_id']: r for r in read(PARENT / 'rgb_jobs.json')}
    old_rows = read(PARENT / 'observations.json')
    failures = {r['sample_id']: r for r in read(PARENT / 'mexico_failure_description.json')['objects']}
    palette = np.array([[0,0,0], [220,224,226], [236,185,65], [243,119,48], [203,43,53]], np.uint8)
    sheet = Image.new('RGB', (1600, 1100), 'white'); draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    for col, label in enumerate(['Pre RGB + target outline', 'Post RGB + target outline', 'Published reference', 'Swin512 raw', 'R34-512 raw']): draw.text((col*320+8, 7), label, fill='black', font=font)
    results = []
    for i, maskrow in enumerate(sorted(read(OUT / 'published_mask_manifest.json'), key=lambda r:r['sample_id'])):
        sid = maskrow['sample_id']; ref = refs[sid]; failure = failures[sid]
        checked(maskrow['path'], maskrow['sha256']); published = decode_mask(Path(maskrow['path']).read_bytes())
        checked(ref['reference_file'], ref['reference_sha256']); truth = np.asarray(Image.open(ref['reference_file']))
        checked(ref['label_path'], ref['label_sha256']); document = read(ref['label_path'])
        feature = next(f for f in document['features']['xy'] if f['properties']['uid'] == failure['derived_uid'])
        footprint, points = object_mask(feature)
        correspondence = mask_correspondence(truth, published, footprint)
        center = (points.min(0)+points.max(0))/2
        left, top = [max(0, min(864, int(round(v))-80)) for v in center]
        bounds = [left, top, left+160, top+160]
        panels = []
        for image in jobs[sid]['images']:
            checked(image['path'], image['sha256'])
            with Image.open(image['path']) as im: panels.append(im.convert('RGB'))
        def colored(a):
            rgb = palette[np.minimum(a,4)]; rgb[a==255] = [115,105,158]
            return Image.fromarray(rgb)
        panels.append(colored(published)); prediction_details = {}
        for model in ('swint', 'r34'):
            selected = [r for r in old_rows if r['role']=='production' and r['sample_id']==sid and r['model_id']==model]
            loc, classes = assemble(selected)
            panels.append(colored(classes * loc.astype(np.uint8)))
            prediction_details[model] = {'building_fraction_inside_footprint': float(loc[footprint].mean()),
                'ungated_class_counts_inside_footprint': {str(k): int((classes[footprint]==k).sum()) for k in range(5)}}
        shifted = [((float(x)-left)*2, (float(y)-top)*2) for x,y in points]
        y = 36+i*342
        draw.text((8,y), sid+' | native160x160,display2x | source major-damage', fill='black', font=font)
        for col, panel in enumerate(panels):
            crop = panel.crop(bounds).resize((320,320), Image.Resampling.NEAREST)
            ImageDraw.Draw(crop).line(shifted, fill=(0,255,255), width=2)
            sheet.paste(crop,(col*320,y+20))
        results.append({'sample_id': sid, 'derived_uid': failure['derived_uid'], 'source_object_index': feature['properties']['source_object_index'],
            'source_polygon_sha256_round6': feature['properties']['polygon_sha256_round6'], 'reference_grade': feature['properties']['subtype'],
            'crop_bounds_xyxy': bounds, 'footprint_pixels': int(footprint.sum()), 'correspondence': correspondence,
            'prediction_description': prediction_details, 'human_damage_reassessment': None,
            'original_json_object_identity_verified': False})
    draw.text((8,1074),'Cyan=preserved source target outline. Display crops only; no new model inputs, relabelling or independent human review.',fill='black',font=font)
    (OUT / 'figures').mkdir(exist_ok=True); sheet.save(OUT / 'figures/failure_closeups.png')
    atomic_json(OUT / 'crosscheck.json', {'status': 'COMPLETE', 'scope': 'Same3development references, descriptive cross-representation comparison only', 'results': results,
        'original_json_recovered': False, 'original_UIDs_recovered': False, 'independent_human_annotations_created': 0,
        'new_model_calls': 0, 'training': False, 'gpu_rented': False})
    print(json.dumps(results), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--stage', choices=['plan','fetch','compare'], required=True)
    args = p.parse_args()
    if (OUT / 'completion_receipt.json').exists(): raise ValueError('Completed output frozen')
    if args.stage == 'plan': plan(); return
    protocol = read(OUT / 'protocol.json')
    for n,h in protocol['source_code_sha256'].items(): checked(HERE/n,h)
    for r in protocol['selected_locations']: checked(r['source_names_file'],r['source_names_sha256'])
    checked(PARENT/'completion_receipt.json',protocol['parent_receipt_sha256'])
    if args.stage == 'fetch': fetch(protocol)
    else: compare_and_render(protocol)


if __name__ == '__main__': main()
