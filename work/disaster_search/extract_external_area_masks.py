"""Separate fixed18 semantic-reference extraction. Never invents instances."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json, sha
from extract_external_rgb18 import central_infos, parse_local_header, decompress_rgb as decompress_png
from plan_search_v2 import freeze

TRANSFER_CAP = 4 * 1024 * 1024


def decode_reference(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (512,512) or getattr(image,"n_frames",1) != 1:
            raise ValueError("Reference must be single512x512PNG")
        array = np.asarray(image)
        encoding = image.mode
        if array.ndim == 3 and array.shape == (512,512,3) and np.array_equal(array[:,:,0],array[:,:,1]) and np.array_equal(array[:,:,0],array[:,:,2]):
            array = array[:,:,0]
            encoding += "_identical_channels_integer_ids"
        if array.ndim != 2 or array.dtype.kind not in "ui" or not set(np.unique(array)).issubset({0,1,2,3,4}):
            raise ValueError("Unreleased reference encoding; no remapping or unknown-to-zero")
        return array.astype(np.uint8), encoding


class MaskRangeReader:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True,exist_ok=True)
        self.ledger_path = root/"transfer_ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
        self.reserved = sum(r["bytes"] for r in self.ledger)

    def read(self, member, offset, count, role):
        low, high = member["local_header_offset"],member["next_header_or_directory"]
        if role not in ["reference_header30","reference_name_extra","published_damage_mask_compressed"] or not low <= offset < offset+count <= high:
            raise ValueError("Not a bounded selected reference interval")
        if role == "reference_header30" and (offset != low or count != 30):
            raise ValueError("Wrong fixed header interval")
        if role == "reference_name_extra" and (offset != low+30 or count > len(member["path"].encode())+4096):
            raise ValueError("Wrong variable header interval")
        if role == "published_damage_mask_compressed" and count != member["compressed_bytes"]:
            raise ValueError("Wrong reference payload size")
        identity = {"url":member["archive_url"],"member":member["path"],"offset":offset,"bytes":count,"role":role}
        key = sha(json.dumps(identity,sort_keys=True).encode())
        raw,receipt = self.root/(key+".bin"),self.root/(key+".json")
        if raw.exists() != receipt.exists():raise ValueError("Incomplete cache pair")
        if raw.exists():
            saved=json.loads(receipt.read_text());data=raw.read_bytes()
            if any(saved[k]!=v for k,v in identity.items()) or len(data)!=count or sha(data)!=saved["sha256"]:
                raise ValueError("Reference range cache changed")
            return data,{**saved,"cache_hit":True}
        if self.reserved+count>TRANSFER_CAP:raise ValueError("Reference transfer cap exceeded")
        record={**identity,"status":"RESERVED","received":0,"at_utc":datetime.now(timezone.utc).isoformat()}
        self.ledger.append(record);self.reserved+=count;atomic_json(self.ledger_path,self.ledger)
        try:
            request=Request(member["archive_url"],headers={"Range":f"bytes={offset}-{offset+count-1}","Accept-Encoding":"identity"})
            with urlopen(request,timeout=25) as response:
                expected=f"bytes {offset}-{offset+count-1}/{member['archive_size']}"
                if response.status!=206 or response.headers.get("Content-Range")!=expected or response.headers.get("Content-Encoding","identity")!="identity" or response.headers.get("Content-Length") not in [None,str(count)]:
                    raise ValueError("Exact reference Range not honored; no body read")
                body=bytearray()
                while len(body)<count:
                    chunk=response.read(min(65536,count-len(body)))
                    if not chunk:raise ValueError("Truncated reference interval")
                    body.extend(chunk);record["received"]=len(body)
                data=bytes(body)
            saved={**identity,"sha256":sha(data),"status":206,"content_range":expected}
            raw.write_bytes(data);atomic_json(receipt,saved);record["status"]="EXACT_RANGE_RECEIVED"
            return data,{**saved,"cache_hit":False}
        except Exception as error:
            record.update(status="STOPPED_NO_BROADENING",error_type=type(error).__name__,error=str(error));raise
        finally:atomic_json(self.ledger_path,self.ledger)


def selected_members(rgb, metadata):
    queue=json.loads((rgb/"published_mask_queue_metadata_only.json").read_text())
    reports=json.loads((metadata/"archive_index_report.json").read_text())["archives"]
    source=json.loads((metadata/"sources/ebd_figshare_v2.json").read_text())
    archives={f["name"]:f for f in source["files"]}
    members=[]
    for report in reports:
        event=report["archive"].removesuffix(".zip")
        raw=(metadata/"archive_metadata"/event/f"{report['central_directory_offset']}_{report['central_directory_bytes']}.bin").read_bytes()
        if sha(raw)!=report["index_sha256"]:raise ValueError("Frozen directory changed")
        infos=central_infos(raw,report["member_entries"]);by_name={i.filename:i for i in infos}
        offsets=sorted(set(i.header_offset for i in infos)|{report["central_directory_offset"]})
        for row in queue:
            if row["event"]!=event:continue
            name=row["member"]["path"]
            if name!=f"{event}/masks/{row['sample_id']}_post_disaster.png":raise ValueError("Unplanned reference identity")
            info=by_name[name]
            if info.file_size!=row["member"]["uncompressed_bytes"] or info.compress_size!=row["member"]["compressed_bytes"] or f"{info.CRC:08x}"!=row["member"]["crc32_unverified"]:raise ValueError("Mask metadata changed")
            members.append({**row["member"],"sample_id":row["sample_id"],"event":event,
                "archive_url":archives[report["archive"]]["download_url"],"archive_size":report["published_size"],
                "local_header_offset":info.header_offset,"next_header_or_directory":next(o for o in offsets if o>info.header_offset),"flag_bits":info.flag_bits})
    if len(members)!=18 or len({m["sample_id"] for m in members})!=18:raise ValueError("Exactly18 fixed reference masks required")
    return sorted(members,key=lambda m:m["sample_id"])


def verify_parent(path):
    receipt=json.loads((path/"completion_receipt.json").read_text())
    for name,digest in receipt["files"].items():
        if sha((path/name).read_bytes())!=digest:raise ValueError("Frozen parent changed: "+name)
    for name,digest in receipt.get("external_files",{}).items():
        if sha(Path(name).read_bytes())!=digest:raise ValueError("Frozen external file changed")
    return sha((path/"completion_receipt.json").read_bytes())


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["rgb","metadata","out","data"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    parent_hash=verify_parent(args.rgb);verify_parent(args.metadata)
    members=selected_members(args.rgb,args.metadata)
    freeze(args.out/"mask_extraction_protocol.json",{"stage":"FIXED18_PUBLISHED_SEMANTIC_REFERENCE_ONLY",
        "rgb_completion_sha256":parent_hash,"members":members,"extractor_sha256":sha(Path(__file__).read_bytes()),
        "data_root":str(args.data.resolve()),"max_new_transfer_bytes":TRANSFER_CAP,
        "released_integer_values":[0,1,2,3,4],"accepted_layout":"Single integer channel or identical RGB integer channels, no palette/color remapping",
        "use":"Published semantic reference; no new human labels or building instance IDs",
        "no_replacement_or_full_archive_fallback":True,"inference_enabled":False})
    if not args.execute:print(json.dumps({"planned":18,"payload_bytes":0}));return
    reader=MaskRangeReader(args.data/"ranges");rows=[]
    for m in members:
        header,a=reader.read(m,m["local_header_offset"],30,"reference_header30")
        n,extra,start=parse_local_header(header,m)
        variable,b=reader.read(m,m["local_header_offset"]+30,n+extra,"reference_name_extra")
        if variable[:n].decode("utf-8" if m["flag_bits"]&2048 else "cp437")!=m["path"]:raise ValueError("Local reference identity differs")
        packed,c=reader.read(m,start,m["compressed_bytes"],"published_damage_mask_compressed")
        data=decompress_png(packed,m);array,encoding=decode_reference(data)
        target=args.data/"reference_masks"/(m["sample_id"]+".png");target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target.read_bytes())!=sha(data):raise ValueError("Reference file changed")
        if not target.exists():target.write_bytes(data)
        rows.append({"sample_id":m["sample_id"],"event":m["event"],"path":str(target.resolve()),"sha256":sha(data),"bytes":len(data),
            "encoding":encoding,"class_pixels":{str(k):int((array==k).sum()) for k in range(5)},"severe_reference_pixels":int((array>=3).sum()),
            "source_member":m["path"],"ranges":[a,b,c]})
        atomic_json(args.out/"reference_manifest_partial.json",rows)
        print(json.dumps({"sample_id":m["sample_id"],"complete":len(rows),"encoding":encoding}),flush=True)
    atomic_json(args.out/"reference_manifest.json",rows)
    atomic_json(args.out/"reference_extraction_report.json",{"reference_masks":len(rows),"network_body_bytes_received":sum(r["received"] for r in reader.ledger),
        "reserved_bytes":reader.reserved,"reference_label_payload_read":True,"new_inference":0,"building_instances_created":0,"new_human_annotations":0})


if __name__=="__main__":main()
