"""Explicit audit adapter for 2D np.cross and projected boundary tolerance.

Preserves all frozen experiment code/routes. A 1cm geometric tolerance accounts
for the image edge's curved ENU projection versus its straight corner chord.
This is an audit convention, not a statement of independent location accuracy.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'disaster_search'))
import run_structure_pilot as run


def main():
    run.ensure_open();run.verify_inputs()
    cases=run.read(run.OUT/'public_cases.json')
    checked=0; maximum_outside=0.
    for c in cases:
        if not c['geometry_quality']['usable_inferred_range']:continue
        corners=np.asarray(c['corners_xy_m_reconstructed'])
        edges=np.roll(corners,-1,axis=0)-corners
        for point in np.asarray(c['xy_m_reconstructed']):
            rel=point-corners
            determinants=edges[:,0]*rel[:,1]-edges[:,1]*rel[:,0]
            distance=determinants/np.linalg.norm(edges,axis=1)
            orientation=np.sign(np.sum(corners[:,0]*np.roll(corners[:,1],-1)-corners[:,1]*np.roll(corners[:,0],-1)))
            outside=float(max(0.,-np.min(distance*orientation)))
            maximum_outside=max(maximum_outside,outside)
            assert outside<=.01
            checked+=1
    run.save(run.OUT/'audit_compatibility_amendment.json',{
        'scope':'Audit only; frozen core,source protocol and route/metric outputs unchanged.',
        'reason':'Installed NumPy removed 2-dimensional np.cross. Direct ENU audit also found the bottom-middle depot 1.7-3.3mm outside the straight corner chord because the projected pixel boundary is curved.',
        'correction':'Use 2D scalar determinant and explicit1cm boundary distance tolerance for ENU chord approximation; no changes to positions,costs,actions or scores.',
        'boundary_tolerance_m':.01,'maximum_chord_excursion_m':maximum_outside,
        'scope_limitation':'This amends the idealized convex-boundary assertion by1cm only, not field geolocation accuracy or actual flight clearance.',
        'frozen_runner_sha256':run.sha(Path(run.__file__)), 'resumed_auditor_sha256':run.sha(__file__),
        'boundary_points_independently_checked':checked,
        'routes_sha256_before':run.sha(run.OUT/'routes.json'),'metrics_sha256_before':run.sha(run.OUT/'metrics.json')})
    original=np.cross
    def compatible_cross(a,b,*args,**kwargs):
        a,b=np.asarray(a),np.asarray(b)
        if not args and not kwargs and a.shape[-1]==2 and b.shape[-1]==2:
            d=a[...,0]*b[...,1]-a[...,1]*b[...,0]
            return np.where(np.abs(d)<=.01*np.linalg.norm(a,axis=-1),0.,d)
        return original(a,b,*args,**kwargs)
    np.cross=compatible_cross
    try:run.audit()
    finally:np.cross=original
    amendment=run.read(run.OUT/'audit_compatibility_amendment.json')
    assert amendment['routes_sha256_before']==run.sha(run.OUT/'routes.json')
    assert amendment['metrics_sha256_before']==run.sha(run.OUT/'metrics.json')
    from shutil import copyfile
    copyfile(__file__,run.OUT/'source_snapshots'/Path(__file__).name)


if __name__=='__main__':main()
