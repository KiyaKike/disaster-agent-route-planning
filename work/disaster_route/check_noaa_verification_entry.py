"""One fixed NOAA entry:actual pixels and view-label semantics,not route efficacy."""
import sys,json,threading,shutil,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tifffile
from PIL import Image,ImageDraw
from probe_crasar_region import ROOT,REV,RemoteTIFF,open_request,url
from crasar_alignment_core import nearest_shift_indices
sys.path.insert(0,str(ROOT/'work/gyirong_imagery_search/geo_libs'))
import rasterio
from rasterio.windows import Window
from affine import Affine
from pyproj import Transformer
import cv2

O=ROOT/'output/disaster_route_noaa_entry_v1'
P=ROOT/'output/disaster_route_crasar_region_rank2_v1'
S=ROOT/'output/disaster_route_verification_screen_v1'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def fetch(path,dest,cap):
    if dest.exists():return dest.read_bytes()
    with open_request(url(path)) as r:b=r.read(cap+1)
    if len(b)>cap:raise RuntimeError('Annotation cap exceeded')
    dest.write_bytes(b);return b

def run():
    O.mkdir(exist_ok=True,parents=True)
    if (O/'completion_receipt.json').exists():raise RuntimeError('Sealed output')
    entry=json.loads((S/'real_entry_metadata.json').read_text());roi=json.loads((P/'roi_protocol.json').read_text())
    protocol={'revision':REV,'scene':entry['scene'],'path':entry['path'],'selection':'Previously fixed first metadata candidate,Boone;existing128m ROI unchanged;no replacement by damage or model score',
      'claims_to_check':['Actual image support','Published local alignment and independent image matching diagnostic','Shared building IDs and view-specific label disagreement','Capture-day ordering and ground-reference availability'],
      'rules':'No new model calls,threshold tuning,route runs or new human truth. Published view labels used only for diagnostic comparison,never as agent observations.',
      'payload_cap_bytes':48*1024*1024,'annotation_cap_bytes':12*1024*1024,
      'acceptance':'A real acquired view is not automatically a calibration anchor or independent same-state truth;full UAV reinspection remains unsupported without timing/pose/error/ground-reference evidence.',
      'output_grid':{'crs':roi['crs'],'bounds':roi['bounds'],'pixels':1024},'gpu_rental':False}
    if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol
    else:dump(O/'protocol.json',protocol)
    name=entry['source']['Orthomosaic']
    a_path=f'train/annotations/CREWED/building_damage_assessment/{name}.json'
    v_path=f'train/annotations/CREWED/building_alignment_adjustments/{name}.json'
    # Geometry/adjustments needed to evaluate actual aligned support;label results are not used for ROI selection.
    annotations=json.loads(fetch(a_path,O/'crewed_annotations_train.json',protocol['annotation_cap_bytes']))
    vectors=np.array(json.loads(fetch(v_path,O/'crewed_alignment.json',1000000)),float)
    ranges=O/'ranges';ranges.mkdir(exist_ok=True)
    for f in (S/'metadata_entry/ranges').glob('*.bin'):
        if not (ranges/f.name).exists():shutil.copyfile(f,ranges/f.name)
    reader=RemoteTIFF.__new__(RemoteTIFF);reader.path=entry['path'];reader.folder=ranges;reader.size=entry['logical_file_bytes'];reader.pos=0
    reader.blocksize=65536;reader.blocks={int(f.stem):f.read_bytes() for f in ranges.glob('*.bin') if f.stat().st_size==min(65536,reader.size-int(f.stem)*65536)}
    reader.cap=protocol['payload_cap_bytes'];reader.bytes=sum(map(len,reader.blocks.values()));reader.lock=threading.Lock();reader.calls=[];reader.name=reader.path
    sparse=O/'crewed_partial.sparse.tif';reader.sparse(sparse)
    with rasterio.open(sparse) as ds:
        transform=ds.transform;crs=str(ds.crs);height,width=ds.height,ds.width
        tags=ds.tags();bands=ds.count
    assert bands==3
    bounds=roi['bounds'];rr,cc=np.mgrid[:1024,:1024]
    x=bounds[0]+(cc+.5)*.125;y=bounds[3]-(rr+.5)*.125
    to_source=Transformer.from_crs(roi['crs'],crs,always_xy=True);sx,sy=to_source.transform(x,y)
    qx,qy=(~transform)*(sx,sy);idx=nearest_shift_indices(vectors[:,0],qx,qy)
    shift=vectors[:,1]-vectors[:,0];dx=shift[idx,0];dy=shift[idx,1]
    cx,cy=qx+dx,qy+dy
    col0=max(0,math.floor(min(qx.min(),cx.min()))-3);col1=min(width,math.ceil(max(qx.max(),cx.max()))+3)
    row0=max(0,math.floor(min(qy.min(),cy.min()))-3);row1=min(height,math.ceil(max(qy.max(),cy.max()))+3)
    assert col1>col0 and row1>row0
    with tifffile.TiffFile(reader) as tf:
        page=tf.pages[0];assert not page.is_tiled and page.planarconfig==1
        strip_indices=range(row0//page.rowsperstrip,math.ceil(row1/page.rowsperstrip))
        wanted=set()
        for i in strip_indices:
            start=page.dataoffsets[i];end=start+page.databytecounts[i];wanted.update(range(start//65536,(end-1)//65536+1))
        source_metadata={'shape':list(page.shape),'rowsperstrip':page.rowsperstrip,'compression':str(page.compression),
            'crs':crs,'transform':list(transform),'tags':tags,'native_window':[col0,row0,col1-col0,row1-row0],
            'required_strips':len(strip_indices),'required_blocks':len(wanted),'additional_blocks':len(wanted-set(reader.blocks)),
            'applied_alignment_vector_indices':np.unique(idx).tolist(),'alignment_vectors':len(vectors)}
    dump(O/'source_metadata.json',source_metadata)
    groups=[]
    for k in sorted(wanted-set(reader.blocks)):
        if groups and k==groups[-1][-1]+1 and len(groups[-1])<16:groups[-1].append(k)
        else:groups.append([k])
    print(json.dumps({'strips':len(strip_indices),'download_blocks':sum(map(len,groups)),'window':source_metadata['native_window']}),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(reader.get_group,groups))
    reader.sparse(sparse)
    # Validate every requested strip payload exists in byte cache before any raster read.
    assert wanted<=set(reader.blocks)
    with rasterio.open(sparse) as ds:raw=np.moveaxis(ds.read([1,2,3],window=Window(col0,row0,col1-col0,row1-row0)),0,-1)
    maps={}
    for key,px,py in [('raw_georef',qx,qy),('aligned',cx,cy)]:
        xx=(px-col0-.5).astype('float32');yy=(py-row0-.5).astype('float32')
        assert xx.min()>=0 and yy.min()>=0 and xx.max()<raw.shape[1]-1 and yy.max()<raw.shape[0]-1
        im=cv2.remap(raw,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        Image.fromarray(im).save(O/f'crewed_{key}_roi.png');maps[key]=im
    dump(O/'acquisition.json',{'cached_bytes':reader.bytes,'logical_file_bytes':reader.size,'full_image_downloaded':False,
      'range_requests_this_run':reader.calls,'decoded_window_shape':list(raw.shape),'source':entry['source'],
      'nonblack_fraction':{k:float(np.any(v!=0,axis=2).mean()) for k,v in maps.items()},
      'decoding_verified':'All required source strip ranges cached,read only bounded source pixel window;partial file never represents complete imagery'})
    geometry=json.loads((P/'public_geometry.json').read_text());refs=json.loads((P/'private_reference.json').read_text())
    crewed={r['building_id']:r for r in annotations};existing={r['building_id']:r for r in refs}
    known=['no damage','minor damage','major damage','destroyed'];records=[]
    for g in geometry:
        uid=g['building_id'];maskim=Image.new('1',(1024,1024));ImageDraw.Draw(maskim).polygon([tuple(p) for p in g['roi_polygon_px']],fill=1);mask=np.array(maskim)
        c=crewed.get(uid)
        records.append({'building_id':uid,'existing_geometry_eligible':g['eligible'],'crewed_id_exists':c is not None,
          'crewed_image_support_fraction':float(np.any(maps['aligned'][mask]!=0,axis=1).mean()),
          'view_labels':{'satellite':existing[uid]['labels']['post'],'uas':existing[uid]['labels']['uas'],'crewed':c['label'] if c else None},
          'label_reference_scope':'Independent published image-view annotations,not common physical ground truth'})
    usable=[r for r in records if r['existing_geometry_eligible'] and r['crewed_id_exists'] and r['crewed_image_support_fraction']>=.9]
    labelcheck={}
    for view in ['satellite','uas']:
        both=[r for r in usable if r['view_labels'][view] in known and r['view_labels']['crewed'] in known]
        labelcheck[view]={'joint_known_count':len(both),'disagreements':sum(r['view_labels'][view]!=r['view_labels']['crewed'] for r in both),
          'pairs':dict(__import__('collections').Counter(r['view_labels'][view]+' -> '+r['view_labels']['crewed'] for r in both))}
    from qualify_crasar_region import image_translation
    feature_checks={}
    for mode in ['pre','post','uas']:
        other=np.array(Image.open(P/f'{mode}_roi.png'))
        feature_checks[mode+'_to_crewed_raw']=image_translation(other,maps['raw_georef'])
        feature_checks[mode+'_to_crewed_aligned']=image_translation(other,maps['aligned'])
    result={'status':'ACTUAL_ENTRY_DIAGNOSTIC_COMPLETE','fixed_scene':entry['scene'],
      'existing_roi_center_buildings':len(geometry),'existing_geometry_eligible':sum(g['eligible'] for g in geometry),
      'crewed_annotation_records':len(annotations),'shared_ids_with_roi':sum(r['crewed_id_exists'] for r in records),'crewed_supported_existing_eligible':len(usable),
      'view_label_comparison':labelcheck,'image_alignment_diagnostics':feature_checks,
      'catalogue_dates':{'pre_satellite':'2022-01-21','post_satellite':'2022-09-30','crewed_NOAA':'2022-09-30','uas':'2022-10-01'},
      'absolute_capture_times_established':False,'common_state_established':False,'error_independence_established':False,
      'common_field_truth_established':False,'sensor_calibration_anchor_established':False,'new_UAS_revisit_validated':False,
      'scope':'Archive satellite/crewed pixel support and view-label diagnostic only;not a causal prior-error/detector-failure attribution test or route efficacy test',
      'new_model_calls':0,'training':False,'gpu_rental':False}
    dump(O/'private_view_reference_comparison.json',records);dump(O/'entry_assessment.json',result)
    panel=Image.new('RGB',(1536,426),'white');draw=ImageDraw.Draw(panel)
    images=[('Satellite 2022-09-30',np.array(Image.open(P/'post_roi.png'))),('NOAA raw georeference',maps['raw_georef']),('NOAA local adjustment',maps['aligned']),('UAS 2022-10-01',np.array(Image.open(P/'uas_roi.png')))]
    for i,(title,array) in enumerate(images):
        im=Image.fromarray(array).resize((384,384));d=ImageDraw.Draw(im)
        for j,g in enumerate(geometry,1):
            q=np.array(g['roi_polygon_px'])*.375;d.line([tuple(v) for v in q],fill='white',width=1);a,b=q.mean(axis=0);d.text((a,b),str(j),fill='white',stroke_width=1,stroke_fill='black')
        panel.paste(im,(i*384,25));draw.text((i*384+6,8),title,fill='black')
    draw.text((8,412),'Same fixed128m ROI. Published image-view labels are not a common field ground truth. No new detector inference.',fill='black')
    panel.save(O/'entry_panel.png')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
