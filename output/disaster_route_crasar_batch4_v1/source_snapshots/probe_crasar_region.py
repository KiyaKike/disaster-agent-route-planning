"""Bounded metadata qualification of a real before/after/UAS region."""
import csv,io,json,os,sys,time,urllib.parse,urllib.request,hashlib,threading
from pathlib import Path
import numpy as np
import tifffile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/gyirong_imagery_search/geo_libs'))
import rasterio
from pyproj import Transformer
O=Path(os.environ.get('CRASAR_REGION_OUTPUT',str(ROOT/'output/disaster_route_crasar_region_v1')))
F=ROOT/'output/disaster_route_crasar_feasibility_20261103'
REV=json.loads((F/'feasibility.json').read_text())['revision']


def dump(p,x):
    if (O/'completion_receipt.json').exists():raise RuntimeError('Output is sealed; choose a new output directory')
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def url(path):return f'https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/resolve/{REV}/'+urllib.parse.quote(path)
def date(r):
    import datetime
    return datetime.datetime.strptime(r['Date (mm/dd/yyy)'],'%m/%d/%Y').date().isoformat()


class RemoteTIFF(io.RawIOBase):
    """Range-only reader. Every downloaded block is saved for reproducibility."""
    def __init__(self,path,folder,cap=16*1024*1024):
        self.path=path;self.folder=folder;folder.mkdir(exist_ok=True,parents=True)
        self.pos=0;self.size=None;self.blocks={};self.cap=cap;self.bytes=0;self.blocksize=65536;self.calls=[]
        self.name=path
        self.lock=threading.Lock()
        self.get(0)
    def get(self,number):
        if number in self.blocks:return self.blocks[number]
        start=number*self.blocksize;end=start+self.blocksize-1
        req=urllib.request.Request(url(self.path),headers={'Range':f'bytes={start}-{end}'})
        # Query separates cache entries whose CDN ignores Range in its cache key.
        req.full_url += f'?range_start={start}'
        with urllib.request.urlopen(req,timeout=45) as r:
            content_range=r.headers.get('Content-Range','');status=r.status
            if status!=206 or not content_range.startswith(f'bytes {start}-'):
                raise RuntimeError(f'Range not honored: {status} {content_range}')
            data=r.read(self.blocksize+1)
            self.size=int(content_range.split('/')[-1])
        if len(data)>self.blocksize:raise RuntimeError('Oversized range response')
        with self.lock:
            self.bytes+=len(data)
            if self.bytes>self.cap:raise RuntimeError('Metadata transfer cap exceeded')
            self.blocks[number]=data;(self.folder/f'{number:012d}.bin').write_bytes(data)
            self.calls.append({'start':start,'length':len(data)})
        return data
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        end=min(self.size,self.pos+n);chunks=[]
        while self.pos<end:
            k,offset=divmod(self.pos,self.blocksize);b=self.get(k);take=min(end-self.pos,len(b)-offset)
            if take<=0:break
            chunks.append(b[offset:offset+take]);self.pos+=take
        return b''.join(chunks)
    def sparse(self,dest):
        with dest.open('wb') as f:
            f.truncate(self.size)
            for k,data in self.blocks.items():f.seek(k*self.blocksize);f.write(data)


def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Output is sealed')
    O.mkdir(parents=True,exist_ok=True)
    catalogue=ROOT/'output/disaster_route_crasar_region_v1/statistics.csv'
    if not (O/'statistics.csv').exists():(O/'statistics.csv').write_bytes(catalogue.read_bytes())
    rows=list(csv.DictReader((O/'statistics.csv').open()))
    paths=set(json.loads((F/'file_inventory.json').read_text()))
    uas=sorted([r for r in rows if r['Source']=='sUAS' and r['Train/Test']=='Train'],key=lambda r:r['Orthomosaic'])
    candidates=[]
    for u in uas:
        sats=[r for r in rows if r['Source']=='Satellite' and r['Train/Test']=='Train' and r['Orthomosaic'].startswith(u['Orthomosaic']+'_')]
        pre=sorted([r for r in sats if r['Pre/Post Event']=='PRE'],key=lambda r:(date(r),r['Orthomosaic']),reverse=True)
        post=sorted([r for r in sats if r['Pre/Post Event']=='POST'],key=lambda r:(date(r),r['Orthomosaic']))
        if pre and post and all('train/imagery/'+source+'/'+r['Orthomosaic'] in paths for source,r in [('UAS',u),('SATELLITE',pre[0]),('SATELLITE',post[0])]):
            candidates.append((u,pre[0],post[0]))
    rank=int(os.environ.get('CRASAR_REGION_RANK','0'))
    u,pre,post=candidates[rank]
    public_fields=['Orthomosaic','Train/Test','Source','Event','Date (mm/dd/yyy)','Pre/Post Event','GSD (m/px)']
    protocol={'date':'2026-11-03','revision':REV,'selection':'First lexicographic training UAS scene with published PRE and POST satellite metadata and all three files. Latest reported PRE;earliest POST then filename. No damage classes/scores used.',
              'original_pecan_pair':'Both POST;cannot supply before/after condition.',
              'eligible_train_scenes':len(candidates),'selected_zero_based_rank':rank,'images':{name:{k:r[k] for k in public_fields} for name,r in [('uas',u),('pre',pre),('post',post)]},
              'scope':'First geographic/temporal qualification. Capture dates are author catalog claims,not independently verified flight logs.',
              'metadata_transfer_cap_per_image_bytes':16*1024*1024,'test_files_opened':False}
    dump(O/'protocol.json',protocol)
    results={}
    for name,source,r in [('pre','SATELLITE',pre),('post','SATELLITE',post),('uas','UAS',u)]:
        path='train/imagery/'+source+'/'+r['Orthomosaic'];reader=RemoteTIFF(path,O/('ranges_'+name))
        with tifffile.TiffFile(reader) as tf:
            page=tf.pages[0]
            tags={str(k):str(v.value)[:1000] for k,v in page.tags.items() if k in (270,306,33550,33922,34735,34737)}
            base={'shape':list(page.shape),'is_tiled':page.is_tiled,'compression':str(page.compression),'pages':len(tf.pages),'tags':tags,'total_file_bytes':reader.size}
        sparse=O/f'{name}_metadata_only.sparse.tif';reader.sparse(sparse)
        with rasterio.open(sparse) as ds:
            tx=Transformer.from_crs(ds.crs,4326,always_xy=True)
            corners=[tx.transform(x,y) for x,y in [(ds.bounds.left,ds.bounds.bottom),(ds.bounds.right,ds.bounds.top)]]
            base.update({'crs':str(ds.crs),'transform':list(ds.transform),'bounds':list(ds.bounds),'geographic_diagonal':corners,'bands':ds.count,'width':ds.width,'height':ds.height,'pixel_size':list(ds.res)})
        base.update({'downloaded_bytes':reader.bytes,'range_requests':reader.calls,'path':path})
        results[name]=base;dump(O/'image_metadata.json',results)
        print(name,json.dumps({k:base[k] for k in ['crs','shape','pixel_size','total_file_bytes','downloaded_bytes']},ensure_ascii=False),flush=True)
    dump(O/'metadata_stage.json',{'status':'METADATA_READ','images':results})


if __name__=='__main__':run()
