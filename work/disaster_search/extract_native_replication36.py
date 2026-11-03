"""Exact selected72 RGB+36 semantic masks, immutable cohort and48MiB cap."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import threading
from urllib.request import Request,urlopen

from PIL import Image

from build_xbd_mini import atomic_json,sha
from extract_external_area_masks import verify_parent,decode_reference
from extract_external_rgb18 import central_infos,parse_local_header,decompress_rgb
from plan_independent_feasibility import EVENTS,pair_inventory,rank_hash
from plan_search_v2 import freeze

CAP=48*1024*1024


def read(path):return json.loads(Path(path).read_text())


def members_from_metadata(quality,metadata):
    prospect=read(quality/"prospective_native_replication36.json");queue=read(quality/"native_replication36_queue_disabled.json")
    old=set(prospect["old18_ids_excluded"]);wanted=[]
    for event in EVENTS:
        path=metadata/"sources"/(event+"_index.json")
        if sha(path.read_bytes())!=prospect["inventory_sha256"][event]:raise ValueError("Inventory changed")
        pairs=pair_inventory(event,read(path));wanted.extend(sorted(pairs,key=lambda sid:(rank_hash(event,sid),sid))[6:18])
    if [r["sample_id"] for r in prospect["selected_pairs"]]!=wanted or len(set(wanted))!=36 or old&set(wanted):raise ValueError("New36 selection differs")
    if len(queue)!=108 or any(r["enabled"] is not False for r in queue):raise ValueError("Parent queue changed")
    reports=read(metadata/"archive_index_report.json")["archives"];members=[]
    for report in reports:
        event=report["archive"].removesuffix(".zip");raw=(metadata/"archive_metadata"/event/f"{report['central_directory_offset']}_{report['central_directory_bytes']}.bin").read_bytes()
        if sha(raw)!=report["index_sha256"]:raise ValueError("Archive directory changed")
        infos=central_infos(raw,report["member_entries"]);lookup={i.filename:i for i in infos};offsets=sorted({i.header_offset for i in infos}|{report["central_directory_offset"]})
        for row in queue:
            if row["event"]!=event:continue
            sid=row["sample_id"];mode=row["mode"];meta=row["member"]
            name=f"{event}/masks/{sid}_post_disaster.png" if mode=="published_post_mask" else f"{event}/images/{sid}_{mode}_disaster.png"
            if sid not in wanted or mode not in ["pre","post","published_post_mask"] or meta["path"]!=name:raise ValueError("Unplanned member")
            info=lookup[name]
            if info.compress_size!=meta["compressed_bytes"] or info.file_size!=meta["uncompressed_bytes"] or f"{info.CRC:08x}"!=meta["crc32_unverified"]:raise ValueError("Member metadata differs")
            members.append({**meta,"sample_id":sid,"event":event,"mode":mode,"archive_url":row["archive_url"],"archive_size":row["archive_size"],
                "local_header_offset":info.header_offset,"flag_bits":info.flag_bits,"next_header_or_directory":next(o for o in offsets if o>info.header_offset)})
    if len(members)!=108 or len({(m['sample_id'],m['mode']) for m in members})!=108:raise ValueError("Incomplete whitelist")
    if sum(m["compressed_bytes"] for m in members)!=prospect["selected_compressed_payload_bytes"]:raise ValueError("Payload inventory differs")
    return sorted(members,key=lambda m:(m["sample_id"],m["mode"]))


class SelectedRangeCache:
    def __init__(self,root,members,cap=CAP):
        self.root=root;root.mkdir(exist_ok=True,parents=True);self.cap=cap;self.allowed={m["path"]:m for m in members}
        self.file=root/"transfer_ledger.json";self.ledger=read(self.file) if self.file.exists() else []
        self.reserved=sum(r["reserved_bytes"] for r in self.ledger);self.lock=threading.RLock();self.halted=threading.Event()

    def read(self,member,offset,count,role):
        if member["path"] not in self.allowed or self.allowed[member["path"]]!=member:raise ValueError("Member not in exact whitelist")
        if role not in ["header30","name_extra","rgb_compressed","mask_compressed"]:raise ValueError("Unknown range role")
        if not member["local_header_offset"]<=offset<offset+count<=member["next_header_or_directory"] or count>4*1024*1024:raise ValueError("Range crosses selected member")
        if role=="header30" and (offset!=member["local_header_offset"] or count!=30):raise ValueError("Header interval differs")
        if role=="name_extra" and (offset!=member["local_header_offset"]+30 or count>len(member["path"].encode())+4096):raise ValueError("Name interval differs")
        if role.endswith("compressed"):
            expected="mask_compressed" if member["mode"]=="published_post_mask" else "rgb_compressed"
            if role!=expected or count!=member["compressed_bytes"]:raise ValueError("Wrong payload role/length")
        identity={"url":member["archive_url"],"member":member["path"],"offset":offset,"bytes":count,"role":role};key=sha(json.dumps(identity,sort_keys=True).encode())
        raw=self.root/(key+".bin");receipt=self.root/(key+".json")
        with self.lock:
            if raw.exists()!=receipt.exists():raise ValueError("Incomplete range cache")
            if raw.exists():
                saved=read(receipt);data=raw.read_bytes()
                if any(saved[k]!=v for k,v in identity.items()) or saved["sha256"]!=sha(data) or len(data)!=count:raise ValueError("Range bytes changed")
                return data,saved
            if self.halted.is_set():raise ValueError("Extraction halted after an earlier error")
            if self.reserved+count>self.cap:raise ValueError("Persistent48MiB cap exceeded")
            row={**identity,"reserved_bytes":count,"received_bytes":0,"status":"RESERVED","at_utc":datetime.now(timezone.utc).isoformat()}
            self.ledger.append(row);self.reserved+=count;atomic_json(self.file,self.ledger)
        try:
            request=Request(member["archive_url"],headers={"Range":f"bytes={offset}-{offset+count-1}","Accept-Encoding":"identity"})
            with urlopen(request,timeout=30) as response:
                expected=f"bytes {offset}-{offset+count-1}/{member['archive_size']}"
                if response.status!=206 or response.headers.get("Content-Range")!=expected or response.headers.get("Content-Encoding","identity")!="identity" or response.headers.get("Content-Length") not in [None,str(count)]:raise ValueError("Exact Range not honored; response body not read")
                chunks=bytearray()
                while len(chunks)<count:
                    part=response.read(min(65536,count-len(chunks)))
                    if not part:raise ValueError("Truncated exact response")
                    chunks.extend(part)
                    with self.lock:row["received_bytes"]=len(chunks)
                data=bytes(chunks)
            saved={**identity,"sha256":sha(data),"http_status":206,"content_range":expected}
            with self.lock:
                raw.write_bytes(data);atomic_json(receipt,saved);row["status"]="EXACT_RANGE_RECEIVED";atomic_json(self.file,self.ledger)
            return data,saved
        except Exception as error:
            with self.lock:
                self.halted.set();row.update(status="STOPPED_NO_FALLBACK",error_type=type(error).__name__,error=str(error));atomic_json(self.file,self.ledger)
            raise


def extract_member(member,reader,data_root):
    header,a=reader.read(member,member["local_header_offset"],30,"header30");n,extra,start=parse_local_header(header,member)
    name,b=reader.read(member,member["local_header_offset"]+30,n+extra,"name_extra")
    if name[:n].decode("utf-8" if member["flag_bits"]&2048 else "cp437")!=member["path"]:raise ValueError("Local name differs")
    role="mask_compressed" if member["mode"]=="published_post_mask" else "rgb_compressed"
    packed,c=reader.read(member,start,member["compressed_bytes"],role);png=decompress_rgb(packed,member)
    record={"sample_id":member["sample_id"],"event":member["event"],"mode":member["mode"],"sha256":sha(png),"bytes":len(png),"source_member":member["path"],"ranges":[a,b,c],"crc32_verified":member["crc32_unverified"]}
    if member["mode"]=="published_post_mask":
        labels,encoding=decode_reference(png);record.update(encoding=encoding,class_pixels={str(k):int((labels==k).sum()) for k in range(5)},severe_reference_pixels=int((labels>=3).sum()))
        target=data_root/"reference_masks"/(member["sample_id"]+".png")
    else:
        with Image.open(io.BytesIO(png)) as im:
            if im.format!="PNG" or im.mode!="RGB" or im.size!=(512,512) or getattr(im,"n_frames",1)!=1:raise ValueError("Wrong RGB encoding/shape")
            im.load();record.update(shape=[512,512,3],pixel_sha256_rgb=sha(im.tobytes()))
        target=data_root/"images"/(member["sample_id"]+"_"+member["mode"]+".png")
    target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists() and sha(target.read_bytes())!=sha(png):raise ValueError("PNG cache changed")
    if not target.exists():target.write_bytes(png)
    record["path"]=str(target.resolve());return record


def plan(args):
    parents={name:verify_parent(getattr(args,name)) for name in ["quality","metadata","rgb_parent"]};members=members_from_metadata(args.quality,args.metadata)
    log=(args.out/"test_log.txt").read_text()
    if "Ran 330 tests" not in log or not log.rstrip().endswith("OK"):raise ValueError("330-test preflight required")
    use=read(args.rgb_parent/"source_use_record.json")
    if not use["local_research_rgb_copy_enabled"] or use["commercial_use_or_publication_of_raw_images_enabled"]:raise ValueError("Source-use scope differs")
    for name in ["source_use_record.json","ATTRIBUTION.txt"]:
        target=args.out/name;content=(args.rgb_parent/name).read_bytes()
        if target.exists() and target.read_bytes()!=content:raise ValueError("Attribution changed")
        if not target.exists():target.write_bytes(content)
    protocol={"stage":"NEW36_EXACT_RGB_AND_PUBLISHED_REFERENCE_EXTRACTION","client_report_date":"2026-10-27","parent_receipt_sha256":parents,
        "prospective_replication_sha256":sha((args.quality/"prospective_native_replication36.json").read_bytes()),"members":members,"max_transfer_bytes":CAP,
        "data_root":str(args.data.resolve()),"thread_workers":4,"selected_pairs":36,"RGB_members":72,"reference_members":36,
        "range_roles":["header30","name_extra","rgb_compressed","mask_compressed"],"stop_on_error_no_replacement_no_whole_archive":True,
        "source_code_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in ["extract_native_replication36.py","native_replication36_core.py","test_native_replication36.py","extract_external_rgb18.py","extract_external_area_masks.py","plan_independent_feasibility.py"]},
        "preflight_tests":330,"test_log_sha256":sha((args.out/"test_log.txt").read_bytes()),
        "no_fit_no_blur_no_new_planner":True,"reference_is_published_semantic_not_instance_gold":True}
    freeze(args.out/"extraction_protocol.json",protocol);return members


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["quality","metadata","rgb_parent","out","data"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--execute",action="store_true");args=p.parse_args();args.out.mkdir(exist_ok=True)
    if (args.out/"completion_receipt.json").exists():raise ValueError("Completed output frozen")
    members=plan(args)
    if not args.execute:print(json.dumps({"members":108,"payload_read":False,"transfer_cap_bytes":CAP}));return
    reader=SelectedRangeCache(args.data/"ranges",members);records=[];failure=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(extract_member,m,reader,args.data):m for m in members}
        for future in as_completed(tasks):
            try:
                row=future.result();records.append(row);atomic_json(args.out/"extracted_members_partial.json",sorted(records,key=lambda r:(r['sample_id'],r['mode'])))
                print(json.dumps({"sample_id":row["sample_id"],"mode":row["mode"],"members_complete":len(records)}),flush=True)
            except Exception as error:
                reader.halted.set()
                if failure is None:failure={"member":tasks[future]["path"],"error":str(error),"error_type":type(error).__name__}
                for task in tasks:task.cancel()
    atomic_json(args.out/"extraction_report.json",{"status":"COMPLETE" if failure is None and len(records)==108 else "STOPPED_INCOMPLETE","members_complete":len(records),
        "reserved_bytes":reader.reserved,"received_bytes":sum(r['received_bytes'] for r in reader.ledger),"exact_ranges":sum(r['status']=='EXACT_RANGE_RECEIVED' for r in reader.ledger),"failure":failure,"transfer_cap":CAP,"inference_calls":0})
    if failure is not None:raise ValueError("Extraction stopped without replacement: "+str(failure))
    atomic_json(args.out/"rgb_manifest.json",sorted([r for r in records if r['mode']!='published_post_mask'],key=lambda r:(r['sample_id'],r['mode'])))
    atomic_json(args.out/"reference_manifest.json",sorted([r for r in records if r['mode']=='published_post_mask'],key=lambda r:r['sample_id']))


if __name__=="__main__":main()
