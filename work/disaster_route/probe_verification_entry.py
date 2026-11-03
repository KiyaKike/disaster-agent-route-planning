"""One outcome-blind first candidate:GeoTIFF headers only,no new labels/pixels."""
import sys,json,hashlib
from pathlib import Path
import tifffile
from probe_crasar_region import RemoteTIFF,ROOT
sys.path.insert(0,str(ROOT/'work/gyirong_imagery_search/geo_libs'))
import rasterio
from rasterio.warp import transform_bounds

O=ROOT/'output/disaster_route_verification_screen_v1'
def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Sealed')
    catalogue=json.loads((ROOT/'output/disaster_route_research_admission_v1/verification_catalogue.json').read_text())
    chosen=next(c for c in catalogue if c['additional_crewed_same_date_as_primary_post'])
    row=next(v for v in chosen['additional_files_same_date_as_primary_post'] if v['Source']=='Crewed Aircraft')
    path='train/imagery/CREWED/'+row['Orthomosaic']
    inventory=json.loads((ROOT/'output/disaster_route_crasar_feasibility_20261103/file_inventory.json').read_text());assert path in inventory
    entry=O/'metadata_entry';entry.mkdir(exist_ok=True)
    reader=RemoteTIFF(path,entry/'ranges',cap=2*1024*1024)
    with tifffile.TiffFile(reader) as tf:
        p=tf.pages[0];header={'shape':list(p.shape),'compression':str(p.compression),'tiled':p.is_tiled}
    sparse=entry/'header_only.sparse.tif';reader.sparse(sparse)
    with rasterio.open(sparse) as ds:
        header.update(crs=str(ds.crs),bands=ds.count,bounds=list(ds.bounds),resolution=list(ds.res))
        roi=json.loads((ROOT/'output/disaster_route_crasar_region_rank2_v1/roi_protocol.json').read_text())
        rb=transform_bounds(roi['crs'],ds.crs,*roi['bounds'])
        coverage=ds.bounds.left<=rb[0] and ds.bounds.bottom<=rb[1] and ds.bounds.right>=rb[2] and ds.bounds.top>=rb[3]
    result={'status':'HEADER_ONLY_CANDIDATE_CHECK','selection':'First existing training candidate in previously frozen ordering with additional crewed imagery on primary POST date;no label or method-score selection',
      'scene':chosen['scene'],'source':row,'path':path,'header':header,'logical_file_bytes':reader.size,'header_bytes_read':reader.bytes,
      'previous128mROI_within_georeferenced_bounds':coverage,'actual_nonblank_coverage_checked':False,
      'relation':'NOAA crewed and primary satellite dates2022-09-30;UAS2022-10-01. Day equality is not simultaneous capture or error independence.',
      'admissible_scope':'Potential archived satellite/crewed evidence reveal only,pending real pixel coverage,registration,label semantics and timing checks',
      'same_state_independent_UAS_reinspection':False,'calibration_anchor_established':False,'common_ground_truth_established':False,
      'new_annotations_read':False,'new_pixel_payloads_requested':False,'model_calls':0,
      'provenance':'GeoTIFF header fields only;metadata range blocks may contain incidental neighboring bytes,not decoded image observations'}
    (O/'real_entry_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
