"""Complete selected metadata extraction from verified cached public ranges.

The source_file column stores canonical sample IDs,without a PNG suffix.
No new network requests and no target columns are used by this completion step.
"""
import math
import acquire_xbd_route_metadata as base


def main():
    assert not (base.OUT/'capture_metadata.json').exists()
    p=base.read(base.OUT/'metadata_protocol.json');ledger=base.read(base.OUT/'metadata_network_ledger.json')
    old=base.OfflineGeometryRanges(base.CACHE/'ranges');old.protected=p['protected_ranges']
    receipts={(r['offset'],r['bytes']):r['sha256'] for r in ledger}
    class Reader:
        def read(self,offset,size):
            path=base.OUT/'metadata_ranges'/f'{offset}_{size}.bin'
            if path.exists():
                assert (offset,size) in receipts and base.digest(path)==receipts[offset,size]
                return path.read_bytes()
            return old.read(offset,size)
    columns={name:base.pq.ParquetFile(base.MetadataView(Reader(),p['entry'])).read(columns=[name],use_threads=False).column(0).to_pylist() for name in base.COLS}
    index={value:i for i,value in enumerate(columns['internal:current_id'])};rows=[]
    for s in base.read(base.SOURCE/'selected.json'):
        i=index[s['source_parent_id']];assert columns['ml:source_file'][i]==s['sample_id']
        pre,post=float(columns['ml:pre_gsd_m'][i]),float(columns['ml:post_gsd_m'][i])
        assert math.isfinite(pre) and math.isfinite(post) and 0<pre<10 and 0<post<10
        rows.append({'sample_id':s['sample_id'],'event':s['event'],'source_parent_id':s['source_parent_id'],
            'pre_gsd_m':pre,'post_gsd_m':post,'pre_capture_date':str(columns['ml:pre_capture_date'][i]),'post_capture_date':str(columns['ml:post_capture_date'][i]),
            'metric_scope':'Approximate image-plane ground distance from released post GSD;not an official geotransform or traversability map.'})
    base.save(base.OUT/'capture_metadata.json',rows)
    base.save(base.OUT/'metadata_amendment_02.json',{'scope':'Source identity is canonical sample ID,not ID plus PNG suffix.','identity_changed':False,
        'new_network_bytes':0,'source_sha256':base.digest(__file__)})
    base.save(base.OUT/'metadata_receipt.json',{'status':'PASS','selected_rows':len(rows),'new_network_bytes':sum(r['bytes'] for r in ledger),
        'public_metadata_only':True,'damage_or_polygon_columns_read':False,'new_imagery_bytes':0,
        'selected_post_gsd_min_m':min(r['post_gsd_m'] for r in rows),'selected_post_gsd_max_m':max(r['post_gsd_m'] for r in rows),
        'capture_metadata_sha256':base.digest(base.OUT/'capture_metadata.json')})
    print(base.json.dumps(base.read(base.OUT/'metadata_receipt.json')),flush=True)


if __name__=='__main__':main()
