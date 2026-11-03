"""Strict byte ranges and fixed-object summaries for a source-label cross-check."""
import io
import struct

import cv2
import numpy as np
from PIL import Image


def footer_metadata(footer, trailer):
    import pyarrow.parquet as pq
    if len(trailer) != 8 or trailer[4:] != b'PAR1' or struct.unpack('<I', trailer[:4])[0] != len(footer):
        raise ValueError('Malformed Parquet footer')
    return pq.ParquetFile(io.BytesIO(b'PAR1' + footer + trailer)).metadata


def column_interval(column):
    starts = [v for v in (column.dictionary_page_offset, column.data_page_offset) if v is not None and v >= 4]
    if not starts or column.total_compressed_size <= 0: raise ValueError('Invalid column location')
    start = min(starts)
    return start, start + column.total_compressed_size


def allowed_read(start, count, allowed, forbidden):
    end = start + count
    if count <= 0 or not any(a <= start < end <= b for a, b in allowed):
        raise ValueError('Read outside selected metadata/mask intervals')
    if any(max(start, a) < min(end, b) for a, b in forbidden):
        raise ValueError('Image or unrequested column interval was requested')


def selected_rows(table, wanted_names):
    if set(table.column_names) != {'image_name', 't2_mask'} or len(wanted_names) != len(set(wanted_names)):
        raise ValueError('Unexpected projected columns or repeated selection')
    names = table.column('image_name').to_pylist()
    if any(names.count(name) != 1 for name in wanted_names): raise ValueError('Missing or duplicate fixed identity')
    result = {name: table.slice(names.index(name), 1).to_pylist()[0]['t2_mask'] for name in wanted_names}
    return result


def decode_mask(raw):
    with Image.open(io.BytesIO(raw)) as im:
        if im.format != 'PNG' or im.size != (1024, 1024) or im.mode not in ('L', 'P') or getattr(im, 'n_frames', 1) != 1:
            raise ValueError('Expected original-size single-channel PNG label')
        array = np.array(im)
    if array.dtype != np.uint8 or not set(np.unique(array)).issubset({0, 1, 2, 3, 4, 255}):
        raise ValueError('Unknown mask encoding; do not remap silently')
    return array


def object_mask(feature):
    text = feature['wkt']
    if not text.startswith('POLYGON ((') or not text.endswith('))'): raise ValueError('Simple source ring required')
    points = np.asarray([[float(v) for v in p.split()] for p in text[10:-2].split(',')])
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all(): raise ValueError('Bad ring')
    mask = np.zeros((1024, 1024), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool), points


def mask_correspondence(reference, published, footprint):
    valid = (reference != 255) & (published != 255)
    a, b = (reference >= 3) & (reference <= 4) & valid, (published >= 3) & (published <= 4) & valid
    union = int((a | b).sum())
    domain = footprint & valid
    return {'same_class_fraction_on_common_known_domain': float((reference[valid] == published[valid]).mean()),
            'known_pixel_disagreements': int(((reference != published) & valid).sum()),
            'reference_severe_pixels_common_domain': int(a.sum()), 'published_severe_pixels_common_domain': int(b.sum()),
            'severe_intersection_pixels': int((a & b).sum()), 'severe_union_pixels': union,
            'severe_iou': float((a & b).sum() / union) if union else None,
            'footprint_common_valid_pixels': int(domain.sum()),
            'published_classes_inside_fixed_footprint': {str(k): int((published[domain] == k).sum()) for k in (0, 1, 2, 3, 4, 255)},
            'published_ignored_pixels_inside_fixed_footprint': int((footprint & (published == 255)).sum())}
