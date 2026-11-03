"""Post-polygon pixel diagnostic. Not the official xView2 mixed score."""
from collections import Counter

import cv2
import numpy as np

GRADES = {'no-damage': 1, 'minor-damage': 2, 'major-damage': 3,
          'destroyed': 4, 'un-classified': 255}
MODELS = ('swint', 'r34')
TILES = [(x, y, x + 512, y + 512) for y in (0, 512) for x in (0, 512)]


def rasterize_post(document, size=(1024, 1024)):
    """Round vertices, fill inclusive OpenCV polygons; ignore ambiguous overlaps."""
    height, width = size
    bits = np.zeros(size, np.uint8)
    unknown = np.zeros(size, np.uint8)
    grades = Counter()
    boundary_objects = 0
    for feature in document['features']['xy']:
        grade = GRADES[feature['properties']['subtype']]
        wkt = feature['wkt']
        if not wkt.startswith('POLYGON ((') or not wkt.endswith('))'):
            raise ValueError('Only simple source polygon rings are supported')
        points = np.asarray([tuple(map(float, s.strip().split()))
                             for s in wkt[10:-2].split(',')], np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('Malformed polygon')
        if not np.array_equal(points[0], points[-1]):
            raise ValueError('Unclosed polygon')
        if (points < -.5).any() or (points[:, 0] > width + .5).any() or (points[:, 1] > height + .5).any():
            raise ValueError('Polygon exceeds existing half-pixel boundary tolerance')
        boundary_objects += int((points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any())
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        selected = mask.astype(bool)
        if grade == 255:
            unknown[selected] = 1
        else:
            bits[selected] |= (1 << (grade - 1))
        grades[feature['properties']['subtype']] += 1
    conflict = (bits != 0) & ((bits & (bits - np.uint8(1))) != 0)
    truth = np.zeros(size, np.uint8)
    for grade in (1, 2, 3, 4):
        truth[bits == (1 << (grade - 1))] = grade
    truth[(unknown != 0) | conflict] = 255
    return truth, {'objects_by_grade': dict(grades), 'objects': sum(grades.values()),
                   'boundary_objects': boundary_objects, 'conflicting_known_pixels': int(conflict.sum()),
                   'unknown_polygon_pixels': int(unknown.sum()), 'ignored_union_pixels': int((truth == 255).sum()),
                   'pixels_by_id': {str(k): int((truth == k).sum()) for k in (0, 1, 2, 3, 4, 255)}}


def prepare_r34(pre, post):
    import torch
    arrays = [np.asarray(im, dtype=np.float32).transpose(2, 0, 1) for im in (pre, post)]
    array = np.concatenate(arrays, axis=0)[None].copy()
    mean = np.asarray([123.675, 116.28, 103.53] * 2, np.float32)[None, :, None, None]
    std = np.asarray([58.395, 57.12, 57.375] * 2, np.float32)[None, :, None, None]
    return torch.from_numpy((array - mean) / std)


def decode_output(output, model_id, size):
    import torch
    if model_id == 'swint':
        loc, damage = output['t1_semantic_prediction'], output['change_prediction']
    elif model_id == 'r34':
        loc, damage = output
    else:
        raise ValueError('Unknown model')
    if tuple(loc.shape) != (1, 1, *size) or tuple(damage.shape) != (1, 5, *size):
        raise ValueError('Unexpected prediction shape')
    if not torch.isfinite(loc).all() or not torch.isfinite(damage).all():
        raise ValueError('Non-finite output')
    if model_id == 'swint':
        if not torch.allclose(damage.sum(1), torch.ones_like(damage[:, 0]), atol=1e-4):
            raise ValueError('Swin output must be probabilities')
        if not ((loc >= 0) & (loc <= 1)).all():
            raise ValueError('Swin localization must be probabilities')
    building = (loc[0, 0] > (.5 if model_id == 'swint' else 0)).cpu().numpy()
    classes = damage.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return building, classes


def object_vote(building, classes):
    """Common secondary weighted component vote from the official wrapper."""
    count, components = cv2.connectedComponents(building.astype(np.uint8), connectivity=8)
    votes = np.stack([np.bincount(components.ravel(), weights=(classes == grade).ravel(), minlength=count) * weight
                      for grade, weight in zip((1, 2, 3, 4), (8, 38, 25, 11))], axis=1)
    chosen = votes.argmax(axis=1).astype(np.uint8) + 1
    chosen[0] = 0
    return chosen[components]


def confusion(truth, prediction):
    if truth.shape != prediction.shape or not set(np.unique(truth)).issubset({0, 1, 2, 3, 4, 255}):
        raise ValueError('Unexpected reference')
    if prediction.dtype != np.uint8 or not set(np.unique(prediction)).issubset({0, 1, 2, 3, 4}):
        raise ValueError('Unexpected prediction')
    valid = truth != 255
    return np.bincount(truth[valid].astype(np.int64) * 5 + prediction[valid], minlength=25).reshape(5, 5)


def metrics(cm):
    cm = np.asarray(cm, np.int64)
    if cm.shape != (5, 5) or (cm < 0).any():
        raise ValueError('Expected a nonnegative 5x5 confusion matrix')
    tp = int(cm[3:, 3:].sum())
    fp = int(cm[:3, 3:].sum())
    fn = int(cm[3:, :3].sum())
    tn = int(cm[:3, :3].sum())
    def ratio(a, b): return a / b if b else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'reference_severe_pixels': tp + fn,
            'predicted_severe_pixels_on_valid_domain': tp + fp,
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn), 'iou': ratio(tp, tp + fp + fn),
            'false_severe_background_pixels': int(cm[0, 3:].sum()),
            'false_severe_other_building_pixels': int(cm[1:3, 3:].sum()),
            'false_severe_per_million_valid_pixels': ratio(fp * 1e6, int(cm.sum())),
            'evaluated_pixels': int(cm.sum())}
