"""Native128/256 fixed new36 replication with758 bounded model calls."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_area_core import public_features
from external_footprint_core import NATIVE128,NATIVE256,units
from extract_external_area_masks import verify_parent
from plan_search_v2 import freeze
from run_changeos_pilot import crop_pair,prepare_tensor,predict_summary,sync,CODE_COMMIT
from run_external_context_diagnostic import checked_mask
from evaluate_native_replication36 import MODES,COARSE,evaluate


def read(path):return json.loads(Path(path).read_text())


def plan(args):
    parent_hash=verify_parent(args.quality);prospect=read(args.quality/"prospective_native_replication36.json");extract=read(args.out/"extraction_protocol.json")
    if sha((args.quality/"prospective_native_replication36.json").read_bytes())!=extract["prospective_replication_sha256"]:raise ValueError("Prospective cohort changed")
    for name,digest in extract["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Extraction code changed")
    report=read(args.out/"extraction_report.json")
    if report["status"]!="COMPLETE" or report["members_complete"]!=108 or report["exact_ranges"]!=324 or report["received_bytes"]>prospect["new_transfer_cap_bytes"]:raise ValueError("Extraction not complete")
    images=read(args.out/"rgb_manifest.json");references=read(args.out/"reference_manifest.json");selected=sorted(r["sample_id"] for r in prospect["selected_pairs"])
    if len(images)!=72 or len(references)!=36 or set(r["sample_id"] for r in references)!=set(selected):raise ValueError("New36 manifests incomplete")
    if {r['sample_id'] for r in images}&set(prospect['old18_ids_excluded']):raise ValueError("Old18 overlap")
    if len({(r['sample_id'],r['mode']) for r in images})!=72:raise ValueError("Incomplete RGB pair")
    log=(args.out/"test_log.txt").read_text()
    if "Ran 330 tests" not in log or not log.rstrip().endswith("OK") or sha((args.out/"test_log.txt").read_bytes())!=extract['test_log_sha256']:raise ValueError("330-test evidence differs")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((args.model/name).read_bytes())!=prospect["checkpoint"][key]:raise ValueError("Checkpoint changed")
    lookup={(r['sample_id'],r['mode']):r for r in images};features={};quality=[];pair_hashes={}
    for sid in selected:
        pair=[];zeros={}
        for date in ['pre','post']:
            row=lookup[sid,date]
            if sha(Path(row['path']).read_bytes())!=row['sha256']:raise ValueError("RGB bytes differ")
            with Image.open(row['path']) as im:pair.append(im.convert('RGB'))
            zeros[date]=float(np.all(np.asarray(pair[-1])==0,axis=2).mean())
        features[sid]=public_features(*pair);pair_hashes[sid]=sha(pair[0].tobytes()+pair[1].tobytes())
        quality.append({'sample_id':sid,'event':lookup[sid,'pre']['event'],'zero_rgb_fraction':zeros,'pair_pixel_sha256':pair_hashes[sid]})
    freeze(args.out/'public_features.json',features);freeze(args.out/'rgb_quality.json',quality)
    previous=read(args.quality/'protocol.json');names=['run_native_replication36.py','evaluate_native_replication36.py','native_replication36_core.py','test_native_replication36.py',
        'external_area_core.py','external_footprint_core.py','evaluate_external_footprint.py','run_changeos_pilot.py','run_external_context_diagnostic.py','build_xbd_mini.py','plan_search_v2.py']
    protocol={'stage':'NEW36_NATIVE_AREA_MEASUREMENT_REPLICATION','client_report_date':'2026-10-27','quality_completion_sha256':parent_hash,
        'prospective_protocol_sha256':sha((args.quality/'prospective_native_replication36.json').read_bytes()),'prospective_protocol':prospect,
        'extraction_protocol_sha256':sha((args.out/'extraction_protocol.json').read_bytes()),'selected_ids':selected,'rgb_inputs':images,'positive_controls':previous['positive_controls'],
        'reference_manifest_sha256':sha((args.out/'reference_manifest.json').read_bytes()),'public_features_sha256':sha((args.out/'public_features.json').read_bytes()),
        'checkpoint':prospect['checkpoint'],'model_directory':str(args.model.resolve()),'max_forward_calls':758,
        'call_inventory':{'old_native128_controls':2,'new_native128':576,'new_native256':144,'free_coarse_only':36},
        'forward_order':'2 locked native128 controls; each sorted new36 scene, free64BOX->512BILINEAR first,16native128 cells then4native256 blocks. No warmup.',
        'source_code_sha256':{n:sha(Path(__file__).with_name(n).read_bytes()) for n in names},'tests_passed':330,
        'evaluation_rule':'Frozen new36-only exact-random budget8 screens. Allbudgets4/8/16 and bothdomains retained. Old18 never pooled into decision.',
        'new_training':False,'paid_gpu':False,'new_blur':False,'reference_pixels_decoded_by_predict':False,'adaptive_search_runs':0}
    freeze(args.out/'protocol.json',protocol);return protocol


def predict(args,protocol):
    for key in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
    os.environ['WANDB_MODE']='disabled'
    import torch
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model
    direct=importlib.metadata.distribution('torchange').read_text('direct_url.json')
    if not direct or CODE_COMMIT not in direct:raise ValueError('Unpinned model code')
    torch.set_num_threads(4);torch.manual_seed(20261025);config=read(args.model/'config.json')['config'];config['encoder']['params']['weights']=None
    model=ChangeOS(config).eval();missing,unexpected=load_model(model,str(args.model/'model.safetensors'),strict=True)
    if missing or unexpected:raise ValueError('Strict model load failed')
    model.requires_grad_(False);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device);sync(device)
    for name in ['cache','calls','scene_records']:(args.out/name).mkdir(exist_ok=True)
    index=0;new_calls=0;start=time.perf_counter()
    def forward(job,pair,mode,unit=None,expected=None):
        nonlocal index,new_calls
        index+=1
        if index>758:raise ValueError('Forward cap exceeded')
        tensor=prepare_tensor(*pair,'native');digest=sha(tensor.numpy().tobytes())
        if expected is not None and digest!=expected:raise ValueError('Control tensor differs')
        identity={'index':index,'job':job,'input_mode':mode,'unit_id':unit,'input_tensor_shape':list(tensor.shape),'input_tensor_sha256':digest}
        file=args.out/'calls'/f'call_{index:03d}.json';cache=args.out/'cache'/(job+'.npz');size=512 if mode==COARSE else 256 if mode==NATIVE256 else 128
        if file.exists():
            saved=read(file)
            if saved['status']!='COMPLETE' or any(saved.get(k)!=v for k,v in identity.items()):raise ValueError('Unfinished or different reservation')
            return saved,checked_mask(Path(saved['cache_file']),saved['cache_sha256'],(size,size))
        freeze(file,{**identity,'status':'RESERVED'});t=time.perf_counter()
        with torch.inference_mode():output=model(tensor.to(device));sync(device);summary,mask=predict_summary(output)
        new_calls+=1
        if mask.dtype!=np.bool_ or mask.shape!=(size,size):raise ValueError('Output shape differs')
        np.savez_compressed(cache,severe_mask=mask)
        row={**identity,'status':'COMPLETE',**summary,'elapsed_seconds':time.perf_counter()-t,'cache_file':str(cache.resolve()),'cache_sha256':sha(cache.read_bytes())}
        atomic_json(file,row);del output,tensor;return row,mask
    controls=[]
    for control in protocol['positive_controls']:
        pair=[]
        for r in control['images']:
            if sha(Path(r['path']).read_bytes())!=r['sha256']:raise ValueError('Control RGB differs')
            with Image.open(r['path']) as im:pair.append(im.convert('RGB'))
        row,mask=forward(control['sample_id']+'_positive_control',crop_pair(*pair,control['cell_id']),'positive_control',expected=control['prepared_tensor_sha256'])
        old=checked_mask(Path(control['legacy_cache_file']),control['legacy_cache_sha256'],(128,128));different=int((mask!=old).sum())
        controls.append({'observation_id':control['observation_id'],'exact_mask_equal':different==0,'different_pixels_from_legacy':different,**row});atomic_json(args.out/'positive_control_results.json',controls)
        if different:raise ValueError('Control mismatch; stop before new36')
    images={(r['sample_id'],r['mode']):r for r in protocol['rgb_inputs']};observations=[]
    for sid in protocol['selected_ids']:
        pair=[]
        for date in ['pre','post']:
            image=images[sid,date]
            if sha(Path(image['path']).read_bytes())!=image['sha256']:raise ValueError('RGB changed')
            with Image.open(image['path']) as im:pair.append(im.convert('RGB'))
        coarse=tuple(im.resize((64,64),Image.Resampling.BOX).resize((512,512),Image.Resampling.BILINEAR) for im in pair);rows=[]
        row,_=forward(sid+'_free_coarse',coarse,COARSE,'scene');rows.append({'sample_id':sid,'event':images[sid,'pre']['event'],**row})
        for mode in MODES:
            for unit in units(mode):
                selected=tuple(im.crop(unit['bounds']) for im in pair);row,_=forward(sid+'_'+unit['unit_id']+'_'+mode,selected,mode,unit['unit_id'])
                rows.append({'sample_id':sid,'event':images[sid,'pre']['event'],**row})
        atomic_json(args.out/'scene_records'/(sid+'.json'),rows);observations.extend(rows)
        print(json.dumps({'scene':sid,'scenes_complete':len(observations)//21,'completed_or_reused_calls':index,'device':str(device)}),flush=True)
    if index!=758 or len(observations)!=756:raise ValueError('Inventory incomplete')
    atomic_json(args.out/'observations.json',observations);atomic_json(args.out/'inference_report.json',{'status':'COMPLETE','total_forward_calls':758,'new_calls_this_invocation':new_calls,
        'positive_control_calls':2,'native128_calls':576,'native256_calls':144,'free_coarse_calls':36,'device':str(device),'loop_seconds':time.perf_counter()-start,
        'new_network_bytes_by_prediction':0,'new_fitting':False,'paid_gpu':False,'reference_pixels_decoded_by_prediction':False,
        'observations_sha256':sha((args.out/'observations.json').read_bytes())})


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['quality','model','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['plan','predict','evaluate'],required=True);args=p.parse_args()
    if (args.out/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    protocol=plan(args)
    if args.stage=='predict':predict(args,protocol)
    elif args.stage=='evaluate':evaluate(args.out,protocol)
    else:print(json.dumps({'planned_forward_calls':758,'new36_only':True,'new_inference':0}))


if __name__=='__main__':main()
