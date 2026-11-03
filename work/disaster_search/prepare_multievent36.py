"""Lock new identities, fetch only their RGB, then return only their source labels."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import io
import json
from pathlib import Path
import re
import threading
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image

from run_strongbaseline24 import ROOT,HERE,read,digest,checked,rawsha,save
from audit_hold_taco import URL,SIZE,REVISION,MetadataView,column_ranges
from build_xbd_mini import atomic_json,sha
from materialize_hold_probes import CHILD_COLUMNS,extract_allowed_rows,CLASSES
from hold_development_geometry import build_features
from perception_pair72_core import rasterize_post
from multievent36_core import select,rgb_quality,EVENTS,SEED

OUT=ROOT/'output/disaster_search_multievent36_v1'
PARENT=ROOT/'output/disaster_search_strongbaseline_local24_v1'
TACO=ROOT/'output/disaster_search_hold_taco_v1'
AUTHOR=ROOT/'output/disaster_search_mexico_trace3_v1/prospective_sources'
OLD_COHORT=ROOT/'output/disaster_search_v2_design/cohort_selection.json'


def put(name,obj):
    p=OUT/name
    if p.exists():
        if read(p)!=obj:raise ValueError('Frozen artifact differs: '+name)
    else:save(p,obj)


class Transfer:
    def __init__(self):
        self.path=OUT/'network_ledger.json';self.rows=read(self.path) if self.path.exists() else [];self.lock=threading.Lock()
    def get(self,offset,count,role):
        if not 0<=offset<offset+count<=SIZE or not 0<count<=1024*1024:raise ValueError('Invalid exact range')
        with self.lock:
            used=sum(r['bytes'] for r in self.rows)
            if used+count>384*1024*1024:raise ValueError('Total transfer cap exceeded')
            if role=='new_RGB' and sum(r['bytes'] for r in self.rows if r['role']==role)+count>256*1024*1024:raise ValueError('RGB cap exceeded')
            row={'offset':offset,'bytes':count,'role':role,'status':'RESERVED'};self.rows.append(row);atomic_json(self.path,self.rows)
        try:
            with urlopen(Request(URL,headers={'Range':f'bytes={offset}-{offset+count-1}','Accept-Encoding':'identity'}),timeout=30) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{offset+count-1}/{SIZE}':raise ValueError('Exact range not honored')
                data=response.read(count+1)
            if len(data)!=count:raise ValueError('Incomplete exact range')
            with self.lock:row.update(status='COMPLETE',received=len(data),sha256=sha(data));atomic_json(self.path,self.rows)
            return data
        except Exception as error:
            with self.lock:row.update(status='FAILED',error_type=type(error).__name__);atomic_json(self.path,self.rows)
            raise


class HybridRanges:
    def __init__(self,transfer,allowed,protected=()):
        self.transfer,self.allowed,self.protected=transfer,allowed,list(protected);self.used=[]
        self.root=OUT/'metadata_ranges';self.root.mkdir(exist_ok=True)
    def read(self,offset,size):
        if not any(a<=offset<=offset+size<=b for a,b in self.allowed):raise ValueError('Read outside authorized metadata or RGB')
        pieces=[]
        while size:
            n=min(size,1024*1024);private=any(max(offset,p['offset'])<min(offset+n,p['offset']+p['bytes']) for p in self.protected)
            raw=None;cached=None
            if not private:
                for root in [TACO/'ranges',self.root]:
                    file=root/f'{offset}_{n}.bin';receipt=file.with_suffix('.json')
                    if file.exists():
                        r=read(receipt);raw=file.read_bytes()
                        if len(raw)!=n or sha(raw)!=r['sha256'] or r['url']!=URL or r['offset']!=offset or r['bytes']!=n:raise ValueError('Pinned range cache changed')
                        cached=str(file);break
            if raw is None:
                raw=self.transfer.get(offset,n,'transient_damage' if private else 'source_metadata')
                if not private:
                    file=self.root/f'{offset}_{n}.bin';file.write_bytes(raw)
                    save(file.with_suffix('.json'),{'url':URL,'offset':offset,'bytes':n,'sha256':sha(raw)})
            self.used.append({'offset':offset,'bytes':n,'sha256':sha(raw),'transient_damage':private,'cache_file':cached,
                             'persisted':not private})
            pieces.append(raw);offset+=n;size-=n
        return b''.join(pieces)


def verify_selection():
    p=read(OUT/'selection_protocol.json')
    for n,h in p['source_code_sha256'].items():checked(HERE/n,h)
    for n,h in p['external_files'].items():checked(n,h)
    checked(OUT/'selected.json',p['selected_sha256']);checked(OUT/'test_log.txt',p['test_log_sha256'])
    return p


def plan():
    import pyarrow.parquet as pq
    parent=read(PARENT/'completion_receipt.json')
    assert digest(PARENT/'completion_receipt.json')=='0dd225efdbf5272ca592433bf31bbcbc786591b4005d068ed5ee796c086250f7'
    for n,h in parent['files'].items():checked(PARENT/n,h)
    for n,h in parent['external_files'].items():checked(n,h)
    identities=read(TACO/'identity_rows.json');index=read(TACO/'cozip_index.json')
    splits={s:set((AUTHOR/f'{s}_samples_event_split.txt').read_text().splitlines()) for s in ['train','val','test']}
    source_paths={r['ml:split_original']+'/images/'+r['ml:source_file'] for r in identities}
    assert len(identities)==11034 and source_paths==set.union(*splits.values())
    payloads=defaultdict(list);pattern=re.compile(r'([a-z][a-z0-9-]*_\d{8})')
    for root in [HERE/'attachments',ROOT/'output']:
        for file in root.rglob('*'):
            if not file.is_file() or file.suffix not in ['.png','.jpg','.jpeg','.npz','.json']:continue
            if root==ROOT/'output' and not str(file).startswith(str(ROOT/'output/disaster_search')):continue
            for sid in pattern.findall(file.name):payloads[sid].append(str(file))
    old=read(OLD_COHORT);blocked_ids=set(payloads)|{r['sample_id'] for r in old};blocked_events={r['event'] for r in old}|{s.rsplit('_',1)[0] for s in payloads}
    selected=select(identities,splits,blocked_ids,blocked_events);assert len(selected)==36
    put('selected.json',selected)
    sources={**parent['source_code_sha256'],**{n:digest(HERE/n) for n in ['multievent36_core.py','test_multievent36.py','prepare_multievent36.py']}}
    external={str(PARENT/'completion_receipt.json'):digest(PARENT/'completion_receipt.json'),str(OLD_COHORT):digest(OLD_COHORT)}
    for n in ['identity_rows.json','cozip_index.json','collection.json']:external[str(TACO/n)]=digest(TACO/n)
    for s in splits:external[str(AUTHOR/f'{s}_samples_event_split.txt')]=digest(AUTHOR/f'{s}_samples_event_split.txt')
    assert 'Ran 363 tests' in (OUT/'test_log.txt').read_text() and (OUT/'test_log.txt').read_text().rstrip().endswith('OK')
    damage=next(t for t in read(TACO/'collection.json')['ml:contract']['targets'] if t['name']=='damage');assert damage['classes']==CLASSES and damage['ignore_index']==4
    p={'stage':'IDENTITY_ONLY_MULTIEVENT36_SELECTION_BEFORE_NEW_PAYLOADS','client_date':'2026-10-28','source_repo':'isp-uv-es/xview2-taco','source_revision':REVISION,'source_archive_size':SIZE,
        'source_code_sha256':sources,'external_files':external,'selected_sha256':digest(OUT/'selected.json'),'test_log_sha256':digest(OUT/'test_log.txt'),
        'seed':SEED,'events':list(EVENTS),'samples_per_event':12,'pool_sizes':{e:sum(r['ml:disaster']==e for r in identities) for e in EVENTS},
        'selection':'SHA256(seed|event|sample_id),first12per event;full author/TACO11034identity equality;no geometry,counts,damage or predictions used',
        'blocked_ids':sorted(blocked_ids),'blocked_events':sorted(blocked_events),'old_reserved_ids_accessible':False,
        'quality_rules':['Require valid single PNG1024x1024RGB both dates;stop on invalid payload without replacement','Keep all36,including empty reference and large zero-RGB regions','Flag exact-zero RGB fraction>=25%;no masking or deletion;not a validated nodata classifier','Check exact pair duplicates withinnew and against existinglocalRGB;stop inference on duplicate,no substitution'],
        'access_order':'Freeze identities+rules;read children metadata;fetch exactly72RGB;RGB integrity/quality/duplicate checks;then trusted extractor returns only36source label rows;then freeze inference protocol before new predictions',
        'max_new_RGB_bytes':256*1024*1024,'max_total_network_bytes':384*1024*1024,
        'shared_label_contract':'ml:damage column may physically include oldreserved/unselected rows;trusted extractor transiently projects shared source but returns only36fixed IDs;no unselected outcomes retained/printed/summarized;never persist shared damage bytes.',
        'metadata_only_child_interval':[index['METADATA/children.parquet']['offset'],index['METADATA/children.parquet']['offset']+index['METADATA/children.parquet']['size']],
        'measurement_contract':'Same native256,cost4,primarybudget16,secondary8/32/64,free64BOXcoarse,authorfourflips;do not fit thresholds or replaceevent/scenes after outcomes',
        'purpose':'New-to-this-project measurement replication development;not final independent confirmation;author-event-heldout by published lists,not independently proven training history/geographic independence',
        'training':False,'gpu_rental':False,'reserved_sets_enabled':False,'adaptive_strategy_enabled':False}
    put('selection_protocol.json',p)
    put('prior_payload_inventory.json',dict(payloads))
    transfer=Transfer();ranges=HybridRanges(transfer,[p['metadata_only_child_interval']]);entry=index['METADATA/children.parquet']
    children=pq.ParquetFile(MetadataView(ranges,entry)).read(columns=CHILD_COLUMNS,use_threads=False).to_pylist()
    lookup=defaultdict(list)
    for row in children:lookup[row['internal:parent_id']].append(row)
    queue=[]
    for sample in selected:
        for mode in ['pre','post']:
            found=[r for r in lookup[sample['source_parent_id']] if r['internal:relative_path'].endswith('/'+mode+'.png')]
            assert len(found)==1;item=found[0]
            assert 0<item['internal:size']<=8*1024*1024
            queue.append({'sample_id':sample['sample_id'],'event':sample['event'],'mode':mode,'offset':item['internal:offset'],'bytes':item['internal:size']})
    assert len(queue)==72 and sum(r['bytes'] for r in queue)<=p['max_new_RGB_bytes']
    put('RGB_queue.json',queue);save(OUT/'metadata_read_receipts.json',ranges.used)
    print({'selection':'FROZEN','events':p['pool_sizes'],'pairs':36,'RGB_bytes':sum(r['bytes'] for r in queue)},flush=True)


def rgb():
    p=verify_selection();queue=read(OUT/'RGB_queue.json');assert len(queue)==72
    transfer=Transfer();records=[]
    def fetch(row):
        path=OUT/'samples'/row['sample_id']/(row['sample_id']+'_'+row['mode']+'_disaster.png');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('No unchecked RGB restart')
        parts=[];off=row['offset'];remaining=row['bytes']
        while remaining:
            n=min(remaining,1024*1024);parts.append(transfer.get(off,n,'new_RGB'));off+=n;remaining-=n
        raw=b''.join(parts)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format!='PNG' or im.mode!='RGB' or im.size!=(1024,1024):raise ValueError('Source image contract failed')
            a=np.asarray(im).copy()
        quality=rgb_quality(a);path.write_bytes(raw)
        return {**row,'path':str(path),'sha256':sha(raw),'pixel_sha256_RGB':rawsha(a),**quality}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(fetch,r) for r in queue]
        for f in as_completed(jobs):
            records.append(f.result());save(OUT/'RGB_manifest.json',sorted(records,key=lambda r:(r['sample_id'],r['mode'])))
            if len(records)%12==0:print({'RGB_complete':len(records),'total':72},flush=True)
    assert len(records)==72
    pairs={sid:tuple(next(r['pixel_sha256_RGB'] for r in records if r['sample_id']==sid and r['mode']==mode) for mode in ['pre','post']) for sid in {r['sample_id'] for r in records}}
    duplicate=[sorted([sid for sid,x in pairs.items() if x==value]) for value in set(pairs.values()) if sum(x==value for x in pairs.values())>1]
    # Existing canonical source PNGs only, not derived masks/coarse images.
    prior=read(OUT/'prior_payload_inventory.json');old_pairs={};old_files={}
    for sid,paths in prior.items():
        hashes={}
        for mode in ['pre','post']:
            found=[Path(n) for n in paths if n.endswith('_'+mode+'_disaster.png')]
            if found:
                file=found[0]
                with Image.open(file) as im:
                    if im.mode=='RGB' and im.size==(1024,1024):hashes[mode]=rawsha(np.asarray(im));old_files[str(file)]=digest(file)
        if len(hashes)==2:old_pairs[sid]=(hashes['pre'],hashes['post'])
    overlap=[{'new':sid,'old':oid} for sid,v in pairs.items() for oid,w in old_pairs.items() if v==w]
    save(OUT/'RGB_quality_report.json',{'status':'PASS' if not duplicate and not overlap else 'DUPLICATE_BLOCKED','images':72,'pairs':36,
        'flagged_images25pct':sum(r['zero_RGB_at_least25pct_flag'] for r in records),'max_zero_RGB_fraction':max(r['zero_RGB_fraction'] for r in records),
        'duplicate_pairs_within_new':duplicate,'duplicate_pairs_with_existing':overlap,'old_RGB_files_checked':old_files,
        'primary_samples_deleted':0,'pixels_masked':0,'labels_read_by_RGB_stage':False})
    if duplicate or overlap:raise ValueError('Exact pixel pair duplicate;fixed cohort retained,do not substitute')


def labels():
    import pyarrow.parquet as pq
    p=verify_selection();assert read(OUT/'RGB_quality_report.json')['status']=='PASS'
    index=read(TACO/'cozip_index.json');entry=index['METADATA/sample.parquet'];transfer=Transfer()
    ranges=HybridRanges(transfer,[(entry['offset'],entry['offset']+entry['size'])])
    parquet=pq.ParquetFile(MetadataView(ranges,entry));ranges.protected=column_ranges(parquet,entry,{'ml:damage'})
    selected=read(OUT/'selected.json');identities=read(TACO/'identity_rows.json')
    rows=extract_allowed_rows(parquet,identities,[r['sample_id'] for r in selected])
    refs=[];jobs=[];rgbrows=read(OUT/'RGB_manifest.json');(OUT/'reference').mkdir()
    for sample in selected:
        sid=sample['sample_id'];row=rows[sid]
        assert (row['ml:width'],row['ml:height'])==(1024,1024)
        features=build_features(sid,row)
        document={'metadata':{'width':1024,'height':1024,'disaster':sample['event'],'img_name':sid+'_post_disaster.png'},
                  'features':{'xy':features},'provenance':{'revision':REVISION,'original_uid_retained':False,'identity':'DERIVED_SOURCE_OBJECT_INDEX'}}
        label=OUT/'samples'/sid/(sid+'_post_disaster.json');save(label,document)
        truth,qa=rasterize_post(document);path=OUT/'reference'/(sid+'.png');Image.fromarray(truth).save(path)
        refs.append({'sample_id':sid,'event':sample['event'],'label_path':str(label),'label_sha256':digest(label),'reference_file':str(path),'reference_sha256':digest(path),**qa})
        images=[{'mode':mode,'path':next(r['path'] for r in rgbrows if r['sample_id']==sid and r['mode']==mode),'sha256':next(r['sha256'] for r in rgbrows if r['sample_id']==sid and r['mode']==mode)} for mode in ['pre','post']]
        jobs.append({'sample_id':sid,'event':sample['event'],'images':images})
    save(OUT/'reference_manifest.json',refs);save(OUT/'rgb_jobs.json',jobs);save(OUT/'label_read_receipts.json',ranges.used)
    for use in ranges.used:
        if use['transient_damage']:assert not use['persisted'] and use['cache_file'] is None
    report={'status':'PASS','fixed_pairs':36,'selected_damage_rows_returned':36,'unselected_outcomes_exposed_or_persisted':False,
        'original_uids_recovered':False,'new_human_annotations':0,'source_revision':REVISION,
        'geometry_rule':'Existing halfpixel tolerance,original coordinates retained;unknown/conflicting pixels ignored;no new clipping/relabel rule',
        'by_event':{e:{'scenes':sum(r['event']==e for r in refs),'reference_positive_scenes':sum(r['event']==e and r['pixels_by_id']['3']+r['pixels_by_id']['4']>0 for r in refs),'severe_pixels':sum(r['pixels_by_id']['3']+r['pixels_by_id']['4'] for r in refs if r['event']==e)} for e in EVENTS},
        'network_bytes_received':sum(r.get('received',0) for r in read(OUT/'network_ledger.json')),'training':False,'gpu_rental':False,'new_model_calls':0}
    save(OUT/'data_admission.json',report);print(report,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','rgb','labels'],required=True);args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'plan':plan,'rgb':rgb,'labels':labels}[args.stage]()
