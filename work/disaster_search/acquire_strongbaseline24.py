"""Bounded public-author checkpoint download; no credentials or model execution."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/disaster_search_strongbaseline24_v1'
DEST=ROOT/'work/disaster_search/attachments/external/disasteradaptivenet_fada4e7'
EXPECTED=309003425


class Form(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={};self.active=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a.get('action');self.active=True
        if tag=='input' and self.active and 'name' in a:self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


def main():
    plan=json.loads((OUT/'acquisition_plan.json').read_text())
    assert plan['expected_bytes']==EXPECTED
    target=DEST/'strong_baseline_event.pt'
    if target.exists():raise ValueError('Existing checkpoint is not replaced')
    partial=target.with_suffix('.pt.part')
    if partial.exists():raise ValueError('Prior partial download requires explicit audit')
    url='https://drive.usercontent.google.com/download?'+urlencode({'id':plan['drive_id'],'export':'download','confirm':'t'})
    ledger=[];start=time.monotonic()
    for attempt in range(2):
        row={'request_index':attempt+1,'host':urlparse(url).hostname,'at_utc':datetime.now(timezone.utc).isoformat()}
        ledger.append(row)
        try:
            with urlopen(Request(url,headers={'User-Agent':'disaster-baseline-reproduction','Accept-Encoding':'identity'}),timeout=30) as response:
                row.update(status=response.status,content_type=response.headers.get('Content-Type'),content_length=response.headers.get('Content-Length'))
                if response.status!=200:raise ValueError('Unexpected download status')
                if 'text/html' in (row['content_type'] or ''):
                    body=response.read(512*1024+1);row['html_bytes']=len(body)
                    if len(body)>512*1024:raise ValueError('Confirmation HTML too large')
                    form=Form();form.feed(body.decode('utf-8'))
                    if not form.action or urlparse(form.action).scheme!='https' or urlparse(form.action).hostname!='drive.usercontent.google.com':
                        raise ValueError('No recognized public download form')
                    if form.fields.get('id')!=plan['drive_id']:raise ValueError('Confirmation changed file ID')
                    url=form.action+'?'+urlencode(form.fields)
                    continue
                if row['content_length'] is not None and int(row['content_length'])!=EXPECTED:raise ValueError('Binary length differs from author metadata')
                count=0;sha=hashlib.sha256();last=0
                with partial.open('xb') as file:
                    while True:
                        data=response.read(min(1024*1024,EXPECTED+1-count))
                        if not data:break
                        count+=len(data)
                        if count>EXPECTED:raise ValueError('Binary byte cap exceeded')
                        sha.update(data);file.write(data)
                        if count-last>=64*1024*1024:
                            print(json.dumps({'downloaded_bytes':count,'expected_bytes':EXPECTED}),flush=True);last=count
                if count!=EXPECTED:raise ValueError('Incomplete checkpoint')
                with partial.open('rb') as file:assert file.read(4)==b'PK\x03\x04','Expected PyTorch ZIP checkpoint'
                partial.rename(target);row.update(received_bytes=count,sha256=sha.hexdigest(),result='COMPLETE')
                receipt={'status':'COMPLETE','path':str(target),'bytes':count,'sha256':sha.hexdigest(),'elapsed_seconds':time.monotonic()-start,
                    'source_drive_id':plan['drive_id'],'source_commit':plan['source_commit'],'requests':ledger,
                    'checksum_scope':'Locally computed content SHA256; no publisher checksum returned by Drive connector.',
                    'new_model_calls':0,'training':False,'gpu_rental':False}
                (OUT/'checkpoint_acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print(json.dumps(receipt),flush=True);return
        finally:
            (OUT/'download_request_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    raise ValueError('No binary response after bounded public confirmation')


if __name__=='__main__':main()
