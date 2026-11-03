"""Qualification before routing: geometry, observed image support and private labels."""
import sys,json,hashlib,datetime
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from probe_crasar_region import ROOT,O,dump
sys.path.insert(0,str(ROOT/'work/gyirong_imagery_search/geo_libs'))
from pyproj import Transformer
from affine import Affine
import cv2

def image_translation(a,b):
    # Diagnostic only: held-out image features; never refit georeferencing to labels.
    imgs=[];features=[]
    sift=cv2.SIFT_create(nfeatures=3000)
    for rgb in [a,b]:
        rgb=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        mask=cv2.erode((gray>0).astype('uint8')*255,np.ones((19,19),np.uint8))
        kp,desc=sift.detectAndCompute(gray,mask);features.append((kp,desc))
    if any(f[1] is None for f in features):return {'status':'INSUFFICIENT_FEATURES'}
    pairs=cv2.BFMatcher().knnMatch(features[0][1],features[1][1],k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    if len(good)<4:return {'status':'INSUFFICIENT_MATCHES','matches':len(good)}
    x=np.array([features[0][0][m.queryIdx].pt for m in good]);y=np.array([features[1][0][m.trainIdx].pt for m in good])
    delta=(y-x)*.25
    # Translation consensus radius2m, deterministic exhaustive consensus centers.
    support=np.linalg.norm(delta[:,None]-delta[None,:],axis=2)<=2
    idx=int(np.argmax(support.sum(axis=1)));keep=support[idx]
    if keep.sum()<4:return {'status':'INSUFFICIENT_CONSENSUS','matches':len(good),'largest_translation_consensus':int(keep.sum()),'independent_absolute_accuracy':False}
    return {'status':'DIAGNOSTIC_ONLY','matches':len(good),'translation_consensus':int(keep.sum()),
            'median_translation_m':np.median(delta[keep],axis=0).tolist(),
            'median_direct_residual_m':float(np.median(np.linalg.norm(delta[keep],axis=1))),
            'independent_absolute_accuracy':False}

def run():
    p=json.loads((O/'roi_protocol.json').read_text());b=p['bounds'];px=.125
    source_protocol=json.loads((O/'protocol.json').read_text())
    dates={m:datetime.datetime.strptime(r['Date (mm/dd/yyy)'],'%m/%d/%Y').date() for m,r in source_protocol['images'].items()}
    lag=(dates['uas']-dates['post']).days
    # Freeze evaluation definitions before accessing damage labels.
    contract={'primary_target':'UAS-view major damage or destroyed building; not injured persons or ground-survey truth',
      'eligible_geometry':'Shared IDs whose centroids lie inside fixed ROI; rasterized polygon at least90% inside ROI and at least90% supported by nonblack RGB in every source.',
      'coverage':'RGB-nonzero image support proxy, not proof of visibility or camera footprint',
      'eligibility_uses_labels':False,'minimum_positive_targets_to_define_recall':1,
      'gates':'If zero eligible primary targets, do not compare severe-search efficacy or claim a method contribution.',
      'routing_if_eligible':{'actions':'4x4 disjoint32m image access windows; no measured UAV camera footprint',
       'public_features':'Published footprint geometry, fixed model outputs on PRE/POST satellite pair; never reference labels.',
       'paid_observation':'Fixed model PRE/UAS output only within accessed window; unrevealed UAS data private.',
       'cost':'Euclidean motion plus observation and planning charges, paid return; all timing assumptions reported separately from measured distances.',
       'baselines':['distance first','damage cue first','uncertainty first','exact subset static on same public utility'],
       'candidate':'Observation-value/cost heuristic must use public inputs only; uncertainty scores are uncalibrated. No learned update without separate calibration data.',
       'metrics':['unique severe buildings effectively checked','time to first correctly checked severe building','distance_m','elapsed_proxy_s'],
       'fairness':'Same starts,budgets,observation footprints,detector and return constraints; full subset enumeration/static optimum and planning wall time reported.'},
      'data_scope':f'One training development region. No held-out generalization; UAS minus satellite time gap{lag}days; no flight telemetry.'}
    cp=O/'evaluation_protocol.json'
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:dump(cp,contract)
    imgs={m:np.array(Image.open(O/f'{m}_roi.png')) for m in ['pre','post','uas']}
    rows={m:json.loads((O/f'{m}_annotations_train.json').read_text()) for m in imgs}
    by={m:{r['building_id']:r for r in rows[m]} for m in rows}
    tx=Transformer.from_crs(4326,p['crs'],always_xy=True)
    public=[];private=[];consistency={}
    meta=json.loads((O/'image_metadata.json').read_text())
    for m in rows:
        inv=~Affine(*meta[m]['transform'][:6]);to_source=Transformer.from_crs(4326,meta[m]['crs'],always_xy=True)
        residual=[]
        for r in rows[m]:
            expected=np.array([inv*to_source.transform(v['lon'],v['lat']) for v in r['EPSG:4326']])
            given=np.array([[v['x'],v['y']] for v in r['pixels']])
            assert expected.shape==given.shape
            residual.extend(np.linalg.norm(expected-given,axis=1).tolist())
        consistency[m]={'median_vertex_residual_native_px':float(np.median(residual)),
                        'max_vertex_residual_native_px':float(max(residual)),
                        'scope':'Internal polygon-coordinate consistency only, not image registration or survey accuracy'}
    for uid in p['shared_building_centroids_inside']:
        r=by['uas'][uid];xy=np.array([tx.transform(v['lon'],v['lat']) for v in r['EPSG:4326']])
        pix=np.column_stack(((xy[:,0]-b[0])/px,(b[3]-xy[:,1])/px))
        maskim=Image.new('1',(1024,1024));ImageDraw.Draw(maskim).polygon([tuple(v) for v in pix],fill=1)
        mask=np.array(maskim);area=abs(float(np.sum(xy[:-1,0]*xy[1:,1]-xy[1:,0]*xy[:-1,1])))/2
        inside=min(1.,float(mask.sum()*px*px/area))
        cov={m:float(np.mean(np.any(im[mask]!=0,axis=1))) for m,im in imgs.items()}
        eligible=inside>=.9 and min(cov.values())>=.9
        public.append({'building_id':uid,'polygon_utm':xy.tolist(),'roi_polygon_px':pix.tolist(),
            'inside_fraction_raster_proxy':inside,'image_support_fraction':cov,'eligible':eligible})
        private.append({'building_id':uid,'eligible':eligible,'labels':{m:by[m][uid]['label'] for m in by},
                        'primary_target':by['uas'][uid]['label'] in ['major damage','destroyed']})
    dump(O/'public_geometry.json',public);dump(O/'private_reference.json',private)
    hist={m:dict(Counter(r['labels'][m] for r in private if r['eligible'])) for m in rows}
    targets=sum(r['eligible'] and r['primary_target'] for r in private)
    audit={'region':source_protocol['images']['uas']['Orthomosaic'],'capture_dates_author_catalog':{m:d.isoformat() for m,d in dates.items()},
       'temporal_scope':f'Published PRE/POST roles; UAS minus POST satellite gap{lag}days. Dates are author metadata,not flight logs.',
       'roi_area_m2':128**2,'rgb_nonzero_support_fraction':{m:float(np.mean(np.any(im!=0,axis=2))) for m,im in imgs.items()},
       'shared_ids_all':p['shared_buildings_all'],'centroids_inside':len(public),'geometry_eligible':sum(r['eligible'] for r in public),
       'private_eligible_label_histograms':hist,'eligible_primary_targets':targets,
       'internal_coordinate_checks':consistency,'image_feature_diagnostics':{f'{m}_to_uas':image_translation(imgs[m],imgs['uas']) for m in ['pre','post']},
       'task_gate':'NO_PRIMARY_TARGETS' if not targets else 'REGISTRATION_AND_SENSOR_REVIEW_REQUIRED',
       'flight_execution_gate':'NOT_ESTABLISHED:no trajectory,pose,altitude,camera footprint,measured service/energy cost',
       'new_model_calls':0,'method_gain_established':False,'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
    dump(O/'qualification.json',audit)
    # Natural-color data panel; all surrounding annotation is white/black.
    panel=Image.new('RGB',(1536,574),'white');draw=ImageDraw.Draw(panel)
    titles=[f'{m.upper()} {dates[m].isoformat()}' for m in imgs]
    for i,(m,title) in enumerate(zip(imgs,titles)):
        im=Image.fromarray(imgs[m]).resize((512,512));d=ImageDraw.Draw(im)
        for k,r in enumerate(public):
            q=np.array(r['roi_polygon_px'])/2;d.line([tuple(v) for v in q],fill='white',width=2)
            x,y=q.mean(axis=0);d.text((x,y),str(k+1),fill='white',stroke_width=1,stroke_fill='black')
        panel.paste(im,(i*512,30));draw.text((i*512+10,10),title,fill='black')
    draw.text((10,552),'Same128m ROI; published local shifts applied; black pixels are absent image support. Outlines use shared footprint geometry.',fill='black')
    panel.save(O/'region_audit_panel.png')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
