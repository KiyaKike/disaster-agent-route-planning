"""Read only pinned public capture metadata for the existing36 xBD scenes."""
import argparse
import json
import math
from pathlib import Path
import sys
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/disaster_search'))
import pyarrow.parquet as pq
from audit_hold_taco import MetadataView,column_ranges,URL,SIZE
from inferred_scene_geometry import OfflineGeometryRanges
from run_route_loop import read,save,digest

SOURCE=ROOT/'output/disaster_search_multievent36_v1'
CACHE=ROOT/'output/disaster_search_hold_taco_v1'
OUT=ROOT/'output/disaster_route_real_xbd36_v1'
COLS=['ml:source_file','ml:pre_gsd_m','ml:post_gsd_m','ml:pre_capture_date','ml:post_capture_date','internal:current_id']


def plan():
    OUT.mkdir(exist_ok=True);assert not (OUT/'metadata_protocol.json').exists()
    entry=read(CACHE/'cozip_index.json')['METADATA/sample.parquet']
    p=pq.ParquetFile(MetadataView(OfflineGeometryRanges(CACHE/'ranges'),entry))
    allowed=column_ranges(p,entry,set(COLS));protected=column_ranges(p,entry,{'ml:damage','ml:n_buildings','ml:n_buildings_pre','ml:polygon_xy','ml:polygon_len','ml:polygon_lnglat','ml:polygon_lnglat_len'})
    save(OUT/'metadata_protocol.json',{'client_date':'2026-11-02','url':URL,'archive_size':SIZE,'columns':COLS,
        'allowed_new_ranges':allowed,'protected_ranges':protected,'entry':entry,'max_new_network_bytes':2*1024*1024,
        'selected_sha256':digest(SOURCE/'selected.json'),'source_sha256':digest(__file__),
        'purpose':'Public GSD and dates for existing36 route cases;no new imagery,damage,counts or geometry columns.'})
    print('CAPTURE_METADATA_PROTOCOL_FROZEN',flush=True)


def fetch():
    p=read(OUT/'metadata_protocol.json');assert p['source_sha256']==digest(__file__) and p['selected_sha256']==digest(SOURCE/'selected.json')
    assert not (OUT/'capture_metadata.json').exists()
    old=OfflineGeometryRanges(CACHE/'ranges');old.protected=p['protected_ranges'];ledger=[]
    (OUT/'metadata_ranges').mkdir(exist_ok=True)
    class Reader:
        def read(self,offset,size):
            try:return old.read(offset,size)
            except ValueError as e:
                if str(e)!='Metadata network budget exceeded':raise
            assert any(r['offset']<=offset<=offset+size<=r['offset']+r['bytes'] for r in p['allowed_new_ranges'])
            assert not any(max(offset,r['offset'])<min(offset+size,r['offset']+r['bytes']) for r in p['protected_ranges'])
            assert sum(r['bytes'] for r in ledger)+size<=p['max_new_network_bytes']
            with urlopen(Request(URL,headers={'Range':f'bytes={offset}-{offset+size-1}','Accept-Encoding':'identity'}),timeout=30) as response:
                assert response.status==206 and response.headers.get('Content-Range')==f'bytes {offset}-{offset+size-1}/{SIZE}'
                data=response.read(size+1)
            assert len(data)==size
            path=OUT/'metadata_ranges'/f'{offset}_{size}.bin';path.write_bytes(data)
            ledger.append({'offset':offset,'bytes':size,'sha256':digest(path)});save(OUT/'metadata_network_ledger.json',ledger)
            return data
    rows=pq.ParquetFile(MetadataView(Reader(),p['entry'])).read(columns=COLS,use_threads=False).to_pylist()
    lookup={r['internal:current_id']:r for r in rows};result=[]
    for s in read(SOURCE/'selected.json'):
        r=lookup[s['source_parent_id']];assert r['ml:source_file'].startswith(s['sample_id']+'_')
        pre,post=map(float,(r['ml:pre_gsd_m'],r['ml:post_gsd_m']))
        assert math.isfinite(pre) and math.isfinite(post) and 0<pre<10 and 0<post<10
        result.append({'sample_id':s['sample_id'],'event':s['event'],'source_parent_id':s['source_parent_id'],
            'pre_gsd_m':pre,'post_gsd_m':post,'pre_capture_date':str(r['ml:pre_capture_date']),
            'post_capture_date':str(r['ml:post_capture_date']),
            'metric_scope':'Approximate within-chip distances from released post-image GSD,not an official geotransform or traversability map.'})
    save(OUT/'capture_metadata.json',result)
    if not ledger:save(OUT/'metadata_network_ledger.json',ledger)
    save(OUT/'metadata_receipt.json',{'status':'PASS','selected_rows':len(result),'new_network_bytes':sum(r['bytes'] for r in ledger),
         'public_metadata_only':True,'damage_or_polygon_columns_read':False,'new_imagery_bytes':0,
         'selected_post_gsd_min_m':min(r['post_gsd_m'] for r in result),'selected_post_gsd_max_m':max(r['post_gsd_m'] for r in result),
         'capture_metadata_sha256':digest(OUT/'capture_metadata.json')})
    print(json.dumps(read(OUT/'metadata_receipt.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('plan','fetch'));args=parser.parse_args()
    {'plan':plan,'fetch':fetch}[args.command]()
