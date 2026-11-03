"""Read a bounded georeferenced ROI; sparse files never represent full imagery."""
import sys,json,urllib.request,math,threading,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import tifffile
from probe_crasar_region import ROOT,O,REV,url,dump,open_request
sys.path.insert(0,str(ROOT/'work/gyirong_imagery_search/geo_libs'))
import rasterio
from rasterio.warp import reproject,Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer
from affine import Affine
from crasar_alignment_core import nearest_shift_indices,corrected_image_transform


def fetch(path,target,limit):
    if target.exists():return target.read_bytes()
    with open_request(url(path)) as r:data=r.read(limit+1)
    if len(data)>limit:raise RuntimeError('Transfer cap exceeded')
    target.write_bytes(data);return data


def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Output is sealed')
    p=json.loads((O/'protocol.json').read_text());meta=json.loads((O/'image_metadata.json').read_text())
    records={};annotations={};geometry={}
    for mode,source in [('pre','SATELLITE'),('post','SATELLITE'),('uas','UAS')]:
        name=p['images'][mode]['Orthomosaic']
        path=f'train/annotations/{source}/building_damage_assessment/{name}.json'
        annotations[mode]=json.loads(fetch(path,O/f'{mode}_annotations_train.json',12000000))
        adjust=f'train/annotations/{source}/building_alignment_adjustments/{name}.json'
        records[mode]={'annotation_path':path,'alignment_path':adjust}
        records[mode]['alignment_vectors']=len(json.loads(fetch(adjust,O/f'{mode}_alignment.json',1000000)))
    shared=set.intersection(*[{r['building_id'] for r in annotations[m]} for m in annotations])
    assert shared
    # ROI selection uses shared building geometry only, never labels or predictions.
    a={r['building_id']:r for r in annotations['uas']}
    roi_crs=meta['uas']['crs']
    assert roi_crs.startswith('EPSG:326') or roi_crs.startswith('EPSG:327'),'Metric UTM required'
    to_utm=Transformer.from_crs(4326,roi_crs,always_xy=True)
    centers={uid:np.mean([to_utm.transform(v['lon'],v['lat']) for v in a[uid]['EPSG:4326']],axis=0) for uid in sorted(shared)}
    center=np.mean(list(centers.values()),axis=0)
    bounds=[float(center[0]-64),float(center[1]-64),float(center[0]+64),float(center[1]+64)]
    roi={'selection':'128m square centered at mean geographic centroids of all shared buildings; no damage or image-score filtering',
         'crs':roi_crs,'bounds':bounds,'size_m':128,'output_pixels':1024,'shared_buildings_all':len(shared),
         'shared_building_centroids_inside':[uid for uid,c in centers.items() if bounds[0]<=c[0]<=bounds[2] and bounds[1]<=c[1]<=bounds[3]],
         'public_image_access_unit':'4x4 disjoint 32m square image windows,not measured camera footprints',
         'motion':'Subsequent distance can be georeferenced;flight/observation durations still unmeasured assumptions.'}
    dump(O/'roi_protocol.json',roi)
    transform=from_bounds(*bounds,1024,1024)
    # Read TIFF sub-IFD metadata by the same bounded range reader, reusing saved blocks.
    from probe_crasar_region import RemoteTIFF
    readers={}
    for mode in ['pre','post','uas']:
      if meta[mode]['total_file_bytes']<8000000 and mode!='uas':
        fetch(meta[mode]['path'],O/f'{mode}_complete.tif',8000000);continue
      # Derive payload padding from the published local field before any pixel read.
      if meta[mode]['crs']==roi_crs:
        vectors=np.array(json.loads((O/f'{mode}_alignment.json').read_text()))
        rr,cc=np.mgrid[:1024,:1024];gx,gy=transform*(cc+.5,rr+.5)
        qx,qy=(~Affine(*meta[mode]['transform'][:6]))*(gx,gy)
        used=np.unique(nearest_shift_indices(vectors[:,0],qx,qy))
        maximum_shift=float(np.max(np.abs(vectors[used,1]-vectors[used,0]))*max(meta[mode]['pixel_size']))
        margin=max(16,math.ceil(maximum_shift+2))
      else:raise RuntimeError('Large-source range extraction requires common projected CRS')
      records[mode]['fetch_margin_m']=margin
      reader=RemoteTIFF.__new__(RemoteTIFF)
      reader.path=meta[mode]['path'];reader.folder=O/f'ranges_{mode}';reader.pos=0;reader.size=meta[mode]['total_file_bytes']
      reader.blocks={int(f.stem):f.read_bytes() for f in reader.folder.glob('*.bin') if f.stat().st_size==min(65536,reader.size-int(f.stem)*65536)};reader.cap=int(os.environ.get('CRASAR_IMAGE_RANGE_CAP_BYTES',32*1024*1024))
      reader.bytes=sum(len(x) for x in reader.blocks.values());reader.blocksize=65536;reader.calls=[];reader.name=reader.path;reader.lock=threading.Lock()
      with tifffile.TiffFile(reader) as tf:
        pages=list(tf.pages)+list(tf.pages[0].pages or [])
        # Select finest overview no coarser than 0.25m per pixel, bounded ROI payload.
        candidates=[q for q in pages if meta[mode]['width']/q.imagewidth*meta[mode]['pixel_size'][0]<=max(.25,meta[mode]['pixel_size'][0]*1.001)]
        page=min(candidates,key=lambda q:q.imagewidth)
        factor=meta[mode]['width']/page.imagewidth;res=factor*meta[mode]['pixel_size'][0]
        assert meta[mode]['crs']==roi_crs,'Large-source range extraction requires common projected CRS'
        b=meta[mode]['bounds'];c0=max(0,int((bounds[0]-margin-b[0])/res));c1=min(page.imagewidth,math.ceil((bounds[2]+margin-b[0])/res))
        r0=max(0,int((b[3]-bounds[3]-margin)/res));r1=min(page.imagelength,math.ceil((b[3]-bounds[1]+margin)/res))
        if page.is_tiled:
          cols=math.ceil(page.imagewidth/page.tilewidth)
          indices=[rr*cols+cc for rr in range(r0//page.tilelength,math.ceil(r1/page.tilelength)) for cc in range(c0//page.tilewidth,math.ceil(c1/page.tilewidth))]
        else:indices=list(range(r0//page.rowsperstrip,math.ceil(r1/page.rowsperstrip)))
        wanted=set()
        for idx in indices:
            start=page.dataoffsets[idx];end=start+page.databytecounts[idx]
            wanted.update(range(start//65536,(end-1)//65536+1))
        selected={'overview_factor':factor,'effective_pixel_size_m':res,'shape':[page.imagelength,page.imagewidth],
                  'tile_indices':indices,'additional_blocks':len(wanted-set(reader.blocks))}
      print(mode,'ROI source',json.dumps({k:v for k,v in selected.items() if k!='tile_indices'}),'chunks',len(indices),'cached_blocks',len(reader.blocks),flush=True)
    # Distinct byte ranges are independent; accounting is protected by a lock.
      groups=[]
      for number in sorted(wanted-set(reader.blocks)):
        if groups and number==groups[-1][-1]+1 and len(groups[-1])<16:groups[-1].append(number)
        else:groups.append([number])
      with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(reader.get_group,groups))
      reader.sparse(O/f'{mode}_metadata_only.sparse.tif')
      records[mode]['roi_source']=selected;readers[mode]=reader
    for mode in ['pre','post','uas']:
        path=O/(f'{mode}_complete.tif' if mode not in readers else f'{mode}_metadata_only.sparse.tif')
        with rasterio.open(path) as ds:
            out=np.zeros((3,1024,1024),dtype=np.uint8)
            records[mode]['source_band_count']=ds.count
            records[mode]['preview_band_mapping']=[1,2,3] if ds.count>=3 else [1,1,1]
            shifts=json.loads((O/f'{mode}_alignment.json').read_text())
            tx=Transformer.from_crs(roi_crs,ds.crs,always_xy=True)
            origins=np.array([v[0] for v in shifts])
            rr,cc=np.mgrid[:1024,:1024];gx,gy=transform*(cc+.5,rr+.5)
            sx,sy=tx.transform(gx,gy);qx,qy=(~ds.transform)*(sx,sy)
            nearest=nearest_shift_indices(origins,qx,qy)
            used=np.unique(nearest)
            records[mode]['applied_alignment_vector_indices']=used.tolist()
            records[mode]['alignment_rule']='Nearest published vector at each canonical destination pixel; discontinuities retained,not smoothed.'
            native_transform=ds.transform
            # Explicit overview level prevents reads from sparse, unacquired full-resolution tiles.
            selected=records[mode].get('roi_source',{'overview_factor':1})
            options={}
            if selected['overview_factor']>1.1:options['OVERVIEW_LEVEL']=ds.overviews(1).index(int(round(selected['overview_factor'])))
            with rasterio.open(path,**options) as source:
              from rasterio.vrt import WarpedVRT
              for idx in used:
                shift=np.array(shifts[idx][1])-origins[idx]
                # Fetched margin must include shifts; never silently access unacquired tiles.
                native_resolution=meta[mode]['pixel_size'][0]
                if meta[mode]['crs']==roi_crs and np.max(np.abs(shift))*native_resolution>records[mode].get('fetch_margin_m',16)-2:raise RuntimeError('Alignment exceeds fetched margin')
                corrected=corrected_image_transform(native_transform,source.transform,shift)
                temp=np.zeros_like(out)
                with WarpedVRT(source,src_transform=corrected) as aligned:
                  for band in range(3):
                    reproject(rasterio.band(aligned,band+1 if source.count>=3 else 1),temp[band],src_transform=aligned.transform,src_crs=aligned.crs,
                              dst_transform=transform,dst_crs=roi_crs,resampling=Resampling.bilinear)
                mask=nearest==idx;out[:,mask]=temp[:,mask]
        im=np.moveaxis(out,0,-1);Image.fromarray(im).save(O/f'{mode}_roi.png')
        records[mode]['all_zero_pixel_fraction']=float(np.mean(np.all(im==0,axis=2)))
    dump(O/'roi_acquisition.json',{'records':records,'range_bytes':{m:r.bytes for m,r in readers.items()},'uas_total_file_bytes':meta['uas']['total_file_bytes'],'full_uas_downloaded':False})
    print(json.dumps({'shared':len(shared),'roi_buildings':len(roi['shared_building_centroids_inside']),'download_MB':{m:r.bytes/1e6 for m,r in readers.items()},'zero_fractions':{m:r['all_zero_pixel_fraction'] for m,r in records.items()}},ensure_ascii=False))


if __name__=='__main__':run()
