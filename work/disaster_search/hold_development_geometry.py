"""Explicit subpixel boundary tolerance, with original XY coordinates retained.

This is a data-adapter exception, not a clipping or discovery-rule change.
The geometry-only audit preceded this rule and exposed no damage values.
"""
from audit_hold_taco import REVISION
from build_xbd_mini import sha
from materialize_hold_probes import grade_name
from prepare_xbd_sample import polygon_centroid
from verify_taco_polygon_bridge import ring_key,split_rings

TOLERANCE_PIXELS=.5


def build_features(sid,row):
    rings=split_rings(row["ml:polygon_xy"],row["ml:polygon_len"])
    if len(rings) != len(row["ml:damage"]):raise ValueError("Object/grade arrays differ in length")
    width,height=row["ml:width"],row["ml:height"]
    seen,features=set(),[]
    for index,(ring,code) in enumerate(zip(rings,row["ml:damage"])):
        fingerprint=ring_key(ring)
        if fingerprint in seen:raise ValueError("Duplicate per-object polygon; do not double-count")
        seen.add(fingerprint)
        if ring[-1] != ring[0]:raise ValueError("Unclosed source polygon")
        wkt="POLYGON (("+", ".join(f"{x:.17g} {y:.17g}" for x,y in ring)+"))"
        cx,cy=polygon_centroid(wkt)
        if not 0<=cx<=width or not 0<=cy<=height:raise ValueError("Polygon centroid outside image")
        if any(not (-TOLERANCE_PIXELS<=x<=width+TOLERANCE_PIXELS and -TOLERANCE_PIXELS<=y<=height+TOLERANCE_PIXELS) for x,y in ring):
            raise ValueError("Source polygon exceeds explicit half-pixel tolerance; investigate without dropping")
        uid="derived-taco-"+sha(f"{REVISION}|{sid}|{index}".encode())
        features.append({"wkt":wkt,"properties":{"uid":uid,"uid_origin":"DERIVED_SOURCE_OBJECT_INDEX",
            "source_object_index":index,"polygon_sha256_round6":fingerprint,"subtype":grade_name(code)}})
    return features
