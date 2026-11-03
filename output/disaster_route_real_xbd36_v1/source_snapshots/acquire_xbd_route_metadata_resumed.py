"""Read one metadata column at a time;keep the original narrow range allowlist.

PyArrow merged adjacent column reads across non-requested capture columns.
This adapter avoids that merged request rather than broadening access.
"""
import math
from urllib.request import Request,urlopen

import acquire_xbd_route_metadata as base


def main():
    p=base.read(base.OUT/'metadata_protocol.json')
    assert p['source_sha256']==base.digest(base.__file__) and p['selected_sha256']==base.digest(base.SOURCE/'selected.json')
    assert not (base.OUT/'capture_metadata.json').exists()
    base.save(base.OUT/'metadata_amendment_01.json',{'scope':'One projected column per Parquet read to prevent read coalescing over unrequested columns.',
        'allowed_or_protected_ranges_changed':False,'parent_protocol_sha256':base.digest(base.OUT/'metadata_protocol.json'),
        'adapter_source_sha256':base.digest(__file__)})
    old=base.OfflineGeometryRanges(base.CACHE/'ranges');old.protected=p['protected_ranges'];ledger=[]
    cache=base.OUT/'metadata_ranges';cache.mkdir(exist_ok=True)
    class Reader:
        def read(self,offset,size):
            try:return old.read(offset,size)
            except ValueError as error:
                if str(error)!='Metadata network budget exceeded':raise
            assert any(r['offset']<=offset<=offset+size<=r['offset']+r['bytes'] for r in p['allowed_new_ranges']),(offset,size)
            assert not any(max(offset,r['offset'])<min(offset+size,r['offset']+r['bytes']) for r in p['protected_ranges'])
            assert sum(r['bytes'] for r in ledger)+size<=p['max_new_network_bytes']
            path=cache/f'{offset}_{size}.bin'
            with urlopen(Request(base.URL,headers={'Range':f'bytes={offset}-{offset+size-1}','Accept-Encoding':'identity'}),timeout=30) as response:
                assert response.status==206 and response.headers.get('Content-Range')==f'bytes {offset}-{offset+size-1}/{base.SIZE}'
                data=response.read(size+1)
            assert len(data)==size;path.write_bytes(data)
            ledger.append({'offset':offset,'bytes':size,'sha256':base.digest(path)});base.save(base.OUT/'metadata_network_ledger.json',ledger)
            return data
    columns={}
    for name in base.COLS:
        columns[name]=base.pq.ParquetFile(base.MetadataView(Reader(),p['entry'])).read(columns=[name],use_threads=False).column(0).to_pylist()
    index={v:i for i,v in enumerate(columns['internal:current_id'])};result=[]
    for sample in base.read(base.SOURCE/'selected.json'):
        i=index[sample['source_parent_id']];sid=sample['sample_id'];assert columns['ml:source_file'][i].startswith(sid+'_')
        pre,post=float(columns['ml:pre_gsd_m'][i]),float(columns['ml:post_gsd_m'][i])
        assert math.isfinite(pre) and math.isfinite(post) and 0<pre<10 and 0<post<10
        result.append({'sample_id':sid,'event':sample['event'],'source_parent_id':sample['source_parent_id'],
            'pre_gsd_m':pre,'post_gsd_m':post,'pre_capture_date':str(columns['ml:pre_capture_date'][i]),
            'post_capture_date':str(columns['ml:post_capture_date'][i]),
            'metric_scope':'Approximate within-chip distances from released post GSD,not official geotransform or traversability.'})
    base.save(base.OUT/'capture_metadata.json',result);base.save(base.OUT/'metadata_network_ledger.json',ledger)
    base.save(base.OUT/'metadata_receipt.json',{'status':'PASS','selected_rows':len(result),'new_network_bytes':sum(r['bytes'] for r in ledger),
        'public_metadata_only':True,'damage_or_polygon_columns_read':False,'new_imagery_bytes':0,
        'selected_post_gsd_min_m':min(r['post_gsd_m'] for r in result),'selected_post_gsd_max_m':max(r['post_gsd_m'] for r in result),
        'capture_metadata_sha256':base.digest(base.OUT/'capture_metadata.json')})
    print(base.json.dumps(base.read(base.OUT/'metadata_receipt.json')),flush=True)


if __name__=='__main__':main()
