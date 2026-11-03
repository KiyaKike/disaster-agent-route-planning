"""Infer post-image ranges from paired outline vertices, never certify footprints.

These affine reconstructions provide additional overlap SCREENING. They are
not official geotransforms or evidence of exact pixel/scene independence.
"""
from pathlib import Path
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from audit_hold_taco import CachedRanges, assert_no_protected_reads

COLUMNS = ["ml:polygon_xy", "ml:polygon_len", "ml:polygon_lnglat",
           "ml:polygon_lnglat_len", "ml:height", "ml:width"]
LIMITS = {"min_distinct_vertices": 6, "min_rings": 2, "min_axis_span_fraction": .25,
    "min_convex_hull_area_fraction": .10, "max_normalized_design_condition": 100.,
    "max_in_sample_inverse_residual_px": 1., "max_p95_in_sample_inverse_residual_px": .5,
    "max_even_odd_ring_cross_prediction_px": 1., "padding_px": 2.,
    "min_relative_overlap_area": 1e-6}


class OfflineGeometryRanges(CachedRanges):
    """Read-only cache replay; reject protected requests BEFORE touching cache."""
    def __init__(self, root):
        if not Path(root).is_dir():
            raise ValueError("Existing range cache required")
        super().__init__(Path(root), max_network_bytes=0)
        self.protected = []
        self.allowed = None

    def read(self, offset, size):
        assert_no_protected_reads([{"offset": offset, "bytes": size}], self.protected)
        if self.allowed is not None and not covered(offset, size, self.allowed):
            raise ValueError("Read outside explicitly allowed geometry/footer intervals")
        return super().read(offset, size)


def covered(offset, size, intervals):
    end = offset + size
    cursor = offset
    for a, b in sorted((r["offset"], r["offset"] + r["bytes"]) for r in intervals):
        if a <= cursor < b:
            cursor = max(cursor, b)
        if cursor >= end:
            return True
    return size == 0


def signed_area(points):
    points = np.asarray(points, float)
    if len(points) < 3:
        return 0.
    points = points - points[0]
    return float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                        points[:, 1] * np.roll(points[:, 0], -1)) / 2)


def polygon_overlap(a, b):
    """Convex quadrilateral clipping; ratio relative to smaller polygon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    origin = a[0].copy()
    a, b = a - origin, b - origin
    area_a, area_b = abs(signed_area(a)), abs(signed_area(b))
    if min(area_a, area_b) <= 0:
        raise ValueError("Nondegenerate polygons required")
    if signed_area(b) < 0:
        b = b[::-1]
    result = list(a)
    for p, q in zip(b, np.roll(b, -1, axis=0)):
        if not result:
            break
        previous = result[-1]
        clipped = []
        def distance(r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        ds = distance(previous)
        for current in result:
            de = distance(current)
            if (ds >= 0) != (de >= 0):
                clipped.append(previous + (current - previous) * ds / (ds - de))
            if de >= 0:
                clipped.append(current)
            previous, ds = current, de
        result = clipped
    intersection = abs(signed_area(result)) if result else 0.
    return min(1., intersection / min(area_a, area_b))


def bounds(points):
    a = np.asarray(points)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def candidate_intersections(box, boxes):
    b = np.asarray(boxes)
    return np.flatnonzero((b[:, 0] < box[2]) & (b[:, 2] > box[0]) &
                          (b[:, 1] < box[3]) & (b[:, 3] > box[1]))


def fit_geometry(flat_xy, xy_lengths, flat_geo, geo_lengths, width, height):
    """Fixed criteria on geometry only. Failure is UNKNOWN, never independent."""
    def fail(reason, **detail):
        return {"status": reason, "usable_inferred_range": False,
                "official_footprint_verified": False, **detail}
    xy, geo = np.asarray(flat_xy, float), np.asarray(flat_geo, float)
    xl, gl = np.asarray(xy_lengths, int), np.asarray(geo_lengths, int)
    if not len(xy) or not len(geo):
        return fail("MISSING_GEOMETRY")
    if not width or not height or width <= 0 or height <= 0:
        return fail("INVALID_DIMENSIONS")
    if not np.array_equal(xl, gl) or len(xy) != len(geo) or len(xy) != 2 * int(xl.sum()) or (xl < 3).any():
        return fail("UNALIGNED_RING_LAYOUT")
    xy, geo = xy.reshape(-1, 2), geo.reshape(-1, 2)
    if not np.isfinite(xy).all() or not np.isfinite(geo).all():
        return fail("NONFINITE_COORDINATES")
    if (np.abs(geo[:, 0]) > 180).any() or (np.abs(geo[:, 1]) > 90).any():
        return fail("INVALID_WGS84_RANGE")
    # Keep original ordering, including closure vertices. No best-match rotation
    # or fitted re-pairing of XY/WGS84 rings is allowed.
    normalized = xy / [width, height] - .5
    if len(xl) < LIMITS["min_rings"] or len(np.unique(xy, axis=0)) < LIMITS["min_distinct_vertices"]:
        return fail("INSUFFICIENT_GEOMETRY")
    design = np.column_stack((normalized, np.ones(len(xy))))
    condition = float(np.linalg.cond(design))
    try:
        hull_area = float(ConvexHull(normalized).volume)
    except QhullError:
        return fail("DEGENERATE_GEOMETRY")
    spans = np.ptp(normalized, axis=0).tolist()
    quality = {"axis_span_fraction": spans, "convex_hull_area_fraction": hull_area,
               "normalized_design_condition": condition}
    if min(spans) < LIMITS["min_axis_span_fraction"] or hull_area < LIMITS["min_convex_hull_area_fraction"]:
        return fail("INSUFFICIENT_IMAGE_COVERAGE", **quality)
    if not math.isfinite(condition) or condition > LIMITS["max_normalized_design_condition"]:
        return fail("ILL_CONDITIONED_GEOMETRY", **quality)
    origin = geo.mean(axis=0)
    coefficients, _, rank, _ = np.linalg.lstsq(design, geo - origin, rcond=None)
    if rank != 3 or abs(np.linalg.det(coefficients[:2])) < 1e-16:
        return fail("SINGULAR_GEOTRANSFORM", **quality)
    inverse = np.linalg.inv(coefficients[:2])
    def errors(d, g, coeff):
        # Inverse residuals in PIXEL units, not unscaled degree distances.
        return np.linalg.norm((d @ coeff - (g - origin)) @ inverse * [width, height], axis=1)
    residual = errors(design, geo, coefficients)
    quality.update(max_inverse_residual_px=float(residual.max()),
        p95_inverse_residual_px=float(np.quantile(residual, .95)))
    if quality["max_inverse_residual_px"] > LIMITS["max_in_sample_inverse_residual_px"] or quality["p95_inverse_residual_px"] > LIMITS["max_p95_in_sample_inverse_residual_px"]:
        return fail("AFFINE_RESIDUAL_TOO_LARGE", **quality)
    # Deterministic object-level cross-prediction; never split a ring's vertices
    # across fit/check. This is geometric consistency, not independent accuracy.
    ring_index = np.repeat(np.arange(len(xl)), xl)
    cross = []
    for parity in [0, 1]:
        training = ring_index % 2 == parity
        test = ~training
        d = design[training]
        if np.linalg.matrix_rank(d) != 3:
            return fail("CROSS_PREDICTION_UNIDENTIFIABLE", **quality)
        coeff = np.linalg.lstsq(d, geo[training] - origin, rcond=None)[0]
        cross.append(float(errors(design[test], geo[test], coeff).max()))
    quality["even_odd_ring_max_cross_prediction_px"] = max(cross)
    if max(cross) > LIMITS["max_even_odd_ring_cross_prediction_px"]:
        return fail("CROSS_PREDICTION_TOO_LARGE", **quality)
    def corners(pad):
        p = np.array([[-pad, -pad], [width + pad, -pad],
                      [width + pad, height + pad], [-pad, height + pad]], float)
        return (np.column_stack((p / [width, height] - .5, np.ones(4))) @ coefficients + origin).tolist()
    polygon, padded = corners(0), corners(LIMITS["padding_px"])
    return {"status": "INFERRED_AFFINE_RANGE_ONLY", "usable_inferred_range": True,
        "official_footprint_verified": False, "pixel_domain": [0, 0, width, height],
        "affine_origin_wgs84": origin.tolist(), "normalized_pixel_to_wgs84_coefficients": coefficients.tolist(),
        "inferred_corners_wgs84": polygon, "padded_inferred_corners_wgs84": padded,
        "inferred_bounds_wgs84": bounds(polygon), "padded_inferred_bounds_wgs84": bounds(padded), **quality}
