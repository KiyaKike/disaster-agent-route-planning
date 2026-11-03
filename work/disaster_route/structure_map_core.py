"""Development-only pre-building/evidence maps with reconstructed metric geometry.

No cached prediction or damage label is an input to build_public_case/plan_route.
All utilities and footprint coverage weights are uncalibrated, not probabilities.
"""
import copy
import math
import re

import numpy as np
from PIL import Image, ImageDraw

from inferred_scene_geometry import fit_geometry

POLICIES = ('pre_static', 'post_static', 'online_frozen', 'online_neighbor',
            'online_structure', 'online_structure_guard')
SPEED = 5.
TURN_RATE = math.pi / 4
OBSERVE = 20.
PLAN = 1.
RETURN_MARGIN = 20.
HOME = 64


def ring(wkt):
    m = re.fullmatch(r'POLYGON\s*\(\(([^()]+)\)\)', wkt.strip(), re.I)
    if not m:
        raise ValueError('Only original simple closed polygons supported')
    a = np.array([list(map(float, p.split())) for p in m[1].split(',')])
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 4 or not np.isfinite(a).all() or not np.array_equal(a[0], a[-1]):
        raise ValueError('Invalid closed ring')
    return a


def raster(points):
    im = Image.new('1', (1024, 1024), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in points], fill=1)
    return np.asarray(im, bool)


def ecef(lonlat):
    """WGS84 ellipsoid, zero ellipsoidal height; no sensor GSD conversion."""
    lon, lat = np.deg2rad(np.asarray(lonlat, float)).T
    a = 6378137.; e2 = 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.column_stack((n*np.cos(lat)*np.cos(lon), n*np.cos(lat)*np.sin(lon), n*(1-e2)*np.sin(lat)))


def local_enu(lonlat, origin):
    lon, lat = np.deg2rad(origin)
    rot = np.array([[-np.sin(lon), np.cos(lon), 0],
                    [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)]])
    return (ecef(lonlat) - ecef([origin])[0]) @ rot.T


def midrank(a):
    a = np.asarray(a)
    return np.array([(np.sum(a < x) + (np.sum(a == x)-1)/2)/(len(a)-1) for x in a])


def build_public_case(pre_label, pre_coarse, post_coarse, sample_id):
    """Public pre footprints + shared 64px RGB only; reject post subtype input."""
    if pre_label['metadata']['img_name'] != sample_id + '_pre_disaster.png':
        raise ValueError('Not the selected pre-disaster label')
    if (pre_label['metadata']['width'], pre_label['metadata']['height']) != (1024, 1024):
        raise ValueError('Unexpected image size')
    lookup = {}
    for kind in ('xy', 'lng_lat'):
        rows = pre_label['features'][kind]
        if any('subtype' in f['properties'] for f in rows):
            raise ValueError('Damage subtype cannot enter public pre map')
        lookup[kind] = {f['properties']['uid']: f for f in rows}
        if len(lookup[kind]) != len(rows):
            raise ValueError('Duplicate pre building identity')
    if set(lookup['xy']) != set(lookup['lng_lat']):
        raise ValueError('Unpaired pre footprint identities')
    uids = sorted(lookup['xy'])
    xy_rings = [ring(lookup['xy'][u]['wkt']) for u in uids]
    geo_rings = [ring(lookup['lng_lat'][u]['wkt']) for u in uids]
    if any(len(x) != len(g) for x,g in zip(xy_rings, geo_rings)):
        raise ValueError('Unpaired vertices')
    quality = fit_geometry(np.concatenate(xy_rings).ravel(), [len(p) for p in xy_rings],
                           np.concatenate(geo_rings).ravel(), [len(p) for p in geo_rings], 1024, 1024)
    base = {'sample_id':sample_id, 'geometry_quality':quality, 'pre_building_count':len(uids)}
    if not quality['usable_inferred_range']:
        return base, None
    origin = quality['affine_origin_wgs84']
    coef = np.array(quality['normalized_pixel_to_wgs84_coefficients'])
    px = np.array([[i%8*128+64, i//8*128+64] for i in range(64)] + [[512,1024]],float)
    def project(points):
        return np.column_stack((np.asarray(points)/1024-.5, np.ones(len(points)))) @ coef + origin
    lonlat = project(px)
    xy = local_enu(lonlat, origin)
    corners_px = np.array([[0,0],[1024,0],[1024,1024],[0,1024]])
    masks = [raster(p) for p in xy_rings]
    areas = np.array([int(m.sum()) for m in masks])
    if np.any(areas == 0):
        raise ValueError('Empty visible pre footprint')
    pixels = np.array([[int(m[r*128:(r+1)*128,c*128:(c+1)*128].sum())
                        for r in range(8) for c in range(8)] for m in masks])
    shares = pixels / areas[:,None]
    mass = shares.sum(0)
    for im in (pre_coarse, post_coarse):
        if im.shape != (64,64,3) or im.dtype != np.uint8:
            raise ValueError('Only shared 64px RGB allowed for initial cues')
    delta = np.abs(pre_coarse.astype(float)-post_coarse.astype(float)).mean(2)/255
    change = [float(delta[r*8:(r+1)*8,c*8:(c+1)*8].mean()) for r in range(8) for c in range(8)]
    risk = .5 + .5*midrank(change)
    pub = {**base, 'geometry_scope':'Reconstructed from original pre XY/WGS84 building vertices; internal consistency only, no official geotransform or independent position accuracy.',
           'xy_m_reconstructed':xy.tolist(), 'lonlat_reconstructed':lonlat.tolist(),
           'corners_xy_m_reconstructed':local_enu(project(corners_px),origin).tolist(),
           'corners_lonlat_reconstructed':project(corners_px).tolist(), 'origin_wgs84':origin,
           'pre_building_uids':uids, 'pre_building_areas_px':areas.tolist(), 'pre_building_tile_pixels':pixels.tolist(),
           'building_mass':mass.tolist(), 'post_change':change, 'post_rank_utility':risk.tolist(),
           'pre_pixel_rings':[p.tolist() for p in xy_rings], 'pre_wgs84_rings':[p.tolist() for p in geo_rings],
           'road_passability':None,'terrain_obstacles':None,'flight_permission':None,
           'time_scope':'Assumed 5m/s,45deg/s yaw,20s observation,1s solve and20s retained return margin; not measured flight time.'}
    return pub, masks


class EvidenceMemory:
    def __init__(self, public, masks):
        self.public = public
        self.masks = masks
        self.records = []

    def ingest(self, tile, mask, elapsed, paid_history):
        if not paid_history or len(paid_history) != len(self.records)+1:
            raise ValueError('Evidence requires the next paid history entry')
        h = paid_history[-1]
        if h['tile_id'] != tile or h['observation_complete_s_model'] != elapsed or h['observed']:
            raise ValueError('Evidence not associated with a new completed paid observation')
        if any(r['tile_id'] == tile for r in self.records) or mask.shape != (128,128) or mask.dtype != bool:
            raise ValueError('Duplicate observation or invalid cached mask')
        r,c = divmod(tile,8)
        hit = [int((m[r*128:(r+1)*128,c*128:(c+1)*128] & mask).sum()) for m in self.masks]
        self.records.append({'tile_id':tile,'observed_at_s_model':elapsed,
                             'predicted_severe_pixels':int(mask.sum()),'severe_pixels_on_pre_buildings':hit})
        h['observed'] = True

    def weights(self, policy, phase):
        p = self.public
        parts = np.array(p['pre_building_tile_pixels'],float)
        areas = np.array(p['pre_building_areas_px'],float)
        shares = parts/areas[:,None]
        mass = shares.sum(0)
        risk = np.ones(64) if phase == 'pre' else np.array(p['post_rank_utility'])
        if policy == 'online_neighbor' and phase == 'post':
            updated = risk.copy()
            for j in range(64):
                neighbors = [r for r in self.records if abs(j//8-r['tile_id']//8)+abs(j%8-r['tile_id']%8)==1]
                if neighbors:
                    updated[j] = (risk[j]+sum(r['predicted_severe_pixels']>0 for r in neighbors))/(1+len(neighbors))
            return mass*updated
        if policy not in ('online_structure','online_structure_guard') or not self.records or phase == 'pre':
            return mass*risk
        chosen = [r['tile_id'] for r in self.records]
        observed_pixels = parts[:,chosen].sum(1)
        severe_pixels = np.array([r['severe_pixels_on_pre_buildings'] for r in self.records]).sum(0)
        coverage = observed_pixels/areas
        signal = np.divide(severe_pixels,observed_pixels,out=np.zeros_like(areas),where=observed_pixels>0)
        strength = .5 if policy == 'online_structure_guard' else 1.
        # Same pre-object only, bounded by its actually observed footprint fraction.
        # Guard retains at least half of each original building/cell score.
        update = (1-strength*coverage[:,None])*risk[None,:] + strength*coverage[:,None]*signal[:,None]
        return (shares*update).sum(0)


def angle_delta(a,b):
    return abs((a-b+math.pi)%(2*math.pi)-math.pi)


def leg(xy, a, b, yaw):
    d = xy[b]-xy[a]
    length = float(np.linalg.norm(d))
    heading = math.atan2(d[1],d[0]) if length > 1e-12 else yaw
    turn = angle_delta(heading,yaw)/TURN_RATE
    return length/SPEED+turn,length,heading,turn


def suffix_cost(xy, position, yaw, route):
    cost = 0.
    for i in list(route)+[HOME]:
        dt,_,yaw,_ = leg(xy,position,i,yaw)
        cost += dt + (OBSERVE if i != HOME else 0)
        position = i
    return cost


def plan_route(public, position, yaw, visited, elapsed, budget, weights):
    """Shared reward/next-leg greedy + up to8 deterministic best 2-opt passes."""
    xy = np.array(public['xy_m_reconstructed']); used = set(visited)
    path = []; pos = position; head = yaw; clock = elapsed
    while True:
        candidates = []
        for j in range(64):
            if j in used or weights[j] <= 0:
                continue
            dt,_,h,_ = leg(xy,pos,j,head)
            back = leg(xy,j,HOME,h)[0]
            if clock+dt+OBSERVE+back+RETURN_MARGIN <= budget+1e-8:
                candidates.append((-(weights[j]/(dt+OBSERVE)),j,dt,h))
        if not candidates:
            break
        _,j,dt,head = min(candidates)
        path.append(j);used.add(j);pos=j;clock+=dt+OBSERVE
    for _ in range(8):
        before = suffix_cost(xy,position,yaw,path)
        best = (before,tuple(path))
        for a in range(len(path)):
            for b in range(a+2,len(path)+1):
                candidate = path[:a]+path[a:b][::-1]+path[b:]
                value = suffix_cost(xy,position,yaw,candidate)
                if value < best[0]-1e-9:
                    best=(value,tuple(candidate))
        if best[0] >= before-1e-9:
            break
        path=list(best[1])
    return path


def execute(public, masks, policy, budget, observe):
    if policy not in POLICIES:
        raise ValueError('Unknown policy')
    xy=np.array(public['xy_m_reconstructed']);mem=EvidenceMemory(public,masks)
    state={'policy':policy,'budget_s_model':budget,'elapsed_s_model':0.,'distance_m_reconstructed':0.,
           'turn_s_model':0.,'position':HOME,'yaw':math.pi/2,'visited':[],'history':[],'plans':[],
           'remaining':[],'closed':False}
    frames=[];phase='pre'
    def capture(reason):
        frames.append({'reason':reason,'phase':phase,'position':state['position'],'visited':list(state['visited']),
                       'elapsed_s_model':state['elapsed_s_model'],'remaining':list(state['remaining']),
                       'weights':mem.weights(policy,phase).tolist(),'records':copy.deepcopy(mem.records)})
    def plan(reason):
        state['elapsed_s_model']+=PLAN
        w=mem.weights(policy,phase)
        state['remaining']=plan_route(public,state['position'],state['yaw'],state['visited'],state['elapsed_s_model'],budget,w)
        state['plans'].append({'reason':reason,'after_observations':len(state['history']),
                               'elapsed_s_model':state['elapsed_s_model'],'weights':w.tolist(),'route':list(state['remaining'])})
        capture(reason)
    capture('initialized');plan('pre_plan');phase='post';capture('post_cues_received')
    if policy!='pre_static':plan('post_plan')
    while state['remaining']:
        j=state['remaining'].pop(0)
        dt,dist,head,turn=leg(xy,state['position'],j,state['yaw'])
        arrival=state['elapsed_s_model']+dt;end=arrival+OBSERVE
        if end+leg(xy,j,HOME,head)[0]+RETURN_MARGIN>budget+1e-8:
            raise ValueError('Observation/return not affordable')
        state['position']=j;state['yaw']=head;state['elapsed_s_model']=end
        state['distance_m_reconstructed']+=dist;state['turn_s_model']+=turn
        state['visited'].append(j)
        state['history'].append({'tile_id':j,'arrival_s_model':arrival,'observation_complete_s_model':end,'observed':False})
        prediction=observe(j,end)  # No unseen tile prediction is passed to memory/planner.
        mem.ingest(j,prediction,end,state['history']);capture('paid_observation')
        if policy.startswith('online'):
            affordable=any(k not in state['visited'] and end+PLAN+suffix_cost(xy,j,head,[k])+RETURN_MARGIN<=budget+1e-8
                           for k in range(64) if public['building_mass'][k]>0)
            if affordable:plan('online_replan')
    dt,dist,head,turn=leg(xy,state['position'],HOME,state['yaw'])
    state['elapsed_s_model']+=dt;state['distance_m_reconstructed']+=dist;state['turn_s_model']+=turn
    state['position']=HOME;state['yaw']=head;state['closed']=True
    if state['elapsed_s_model']+RETURN_MARGIN>budget+1e-8:
        raise ValueError('Return margin violated')
    capture('returned')
    return {'state':state,'evidence':mem.records,'frames':frames}
