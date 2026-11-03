"""Record semantic-reference/RGB interface checks; no human validation claim."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from build_xbd_mini import atomic_json,sha
from extract_external_area_masks import decode_reference
from plan_search_v2 import freeze


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb",type=Path,required=True);p.add_argument("--root",type=Path,required=True);args=p.parse_args()
    rgb=json.loads((args.rgb/"rgb_manifest.json").read_text());refs=json.loads((args.root/"reference_manifest.json").read_text())
    freeze(args.root/"reference_inspection_protocol.json",{"scope":"Reference format/identity/RGB-border diagnostics, not label accuracy or registration certification",
        "rgb_manifest_sha256":sha((args.rgb/"rgb_manifest.json").read_bytes()),"reference_manifest_sha256":sha((args.root/"reference_manifest.json").read_bytes()),
        "inspector_sha256":sha(Path(__file__).read_bytes()),
        "primary_domain":"All512x512 published reference pixels, no outcome-based exclusion",
        "diagnostic_domain":"Pixels with any nonzero RGB channel at both dates; sensitivity only, not certified nodata or a better-score selection",
        "no_alignment_shift_or_scene_replacement":True})
    images={(r["sample_id"],r["mode"]):r for r in rgb};events=defaultdict(list);rows=[]
    for ref in refs:
        sid=ref["sample_id"]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference changed")
        label,mode=decode_reference(Path(ref["path"]).read_bytes());severe=label>=3
        pair=[]
        for when in ["pre","post"]:
            record=images[sid,when]
            if sha(Path(record["path"]).read_bytes())!=record["sha256"]:raise ValueError("RGB changed")
            with Image.open(record["path"]) as im:pair.append(im.copy())
        pre,post=[np.asarray(im) for im in pair];pre_nonzero=np.any(pre>0,axis=2);post_nonzero=np.any(post>0,axis=2)
        rows.append({"sample_id":sid,"event":ref["event"],"class_pixels":ref["class_pixels"],"severe_reference_pixels":int(severe.sum()),
            "severe_reference_on_post_all_zero_rgb":int((severe&~post_nonzero).sum()),
            "severe_reference_outside_both_dates_nonzero":int((severe&~(pre_nonzero&post_nonzero)).sum()),
            "primary_domain":"all published pixels","instance_gold":False,"registration_accuracy":"UNVERIFIED"})
        overlay=post.copy();overlay[severe]=(overlay[severe]*.5+np.array([255,38,49])*.5).astype(np.uint8)
        events[ref["event"]].append((sid,pair[0],pair[1],Image.fromarray(overlay),int(severe.sum())))
    for event,part in events.items():
        sheet=Image.new("RGB",(810,len(part)*292+50),"#f3f4f1");draw=ImageDraw.Draw(sheet)
        draw.text((10,10),event+" | PRE / POST / POST + PUBLISHED SEVERE REFERENCE",fill="black")
        for i,(sid,pre,post,overlay,count) in enumerate(part):
            top=35+i*292
            for j,im in enumerate([pre,post,overlay]):
                title=sid if j<2 else f"Published severe pixels: {count}"
                draw.text((10+j*268,top),title,fill="black");sheet.paste(im.resize((256,256),Image.Resampling.BOX),(10+j*268,top+20))
        folder=args.root/"reference_previews";folder.mkdir(exist_ok=True);sheet.save(folder/(event+".jpg"),quality=92)
    totals={event:{"scenes":len(part),"reference_positive_scenes":sum(r["severe_reference_pixels"]>0 for r in rows if r["event"]==event),
        "severe_reference_pixels":sum(r["severe_reference_pixels"] for r in rows if r["event"]==event)} for event,part in events.items()}
    atomic_json(args.root/"reference_quality_report.json",{"stage":"REFERENCE_SCHEMA_AND_IDENTITY_CHECKED_NOT_HUMAN_GOLD","rows":rows,"events":totals,
        "reference_positive_events":sum(r["severe_reference_pixels"]>0 for r in totals.values()),"new_building_instances":0,"new_inference":0})
    atomic_json(args.root/"reference_interface_gate.json",{"technical_interface_passed":True,"selected_ids":sorted(r["sample_id"] for r in refs),
        "reference_manifest_sha256":sha((args.root/"reference_manifest.json").read_bytes()),"rgb_manifest_sha256":sha((args.rgb/"rgb_manifest.json").read_bytes()),
        "human_or_geographic_independence_passed":False,"purpose":"Fixed-detector descriptive reference-area diagnostic only"})
    print(json.dumps({"events":totals,"severe_on_post_zero_rgb":sum(r["severe_reference_on_post_all_zero_rgb"] for r in rows)},ensure_ascii=False))


if __name__=="__main__":main()
