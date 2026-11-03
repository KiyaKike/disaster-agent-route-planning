"""Replay selected ZIP bytes, native inputs, random inclusion and area-paid ranks."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_external_footprint import geometry,independent_score,random_by_combinatorics,independent_rank
from build_xbd_mini import atomic_json,sha
from evaluate_external_footprint import mean_scores
from external_footprint_core import NATIVE128,NATIVE256,BUDGETS,ORDER_SEEDS,aggregate
from external_area_core import public_features
from extract_external_area_masks import verify_parent,decode_reference
from extract_external_rgb18 import parse_local_header,decompress_rgb
from extract_native_replication36 import members_from_metadata
from run_changeos_pilot import crop_pair,prepare_tensor
from run_external_context_diagnostic import checked_mask
from native_replication36_core import decide_replication
from evaluate_native_replication36 import MODES,COARSE


def read(path):return json.loads(Path(path).read_text())


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','quality','metadata']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.root;protocol=read(root/'protocol.json');extract=read(root/'extraction_protocol.json')
    parents=[Path('output')/('disaster_search_'+name+'_v1') for name in ['serial_multifidelity','independent_protocol','external_rgb18','external_area18','context18','information18','footprint18','quality256']]
    serial=read(parents[0]/'completion_receipt.json');sources={Path(n).name:d for n,d in serial['files'].items() if n.startswith('source_snapshots/') and n.endswith('.py')}
    for parent in parents:
        verify_parent(parent);sources.update(read(parent/'completion_receipt.json').get('source_code_sha256',{}))
    if verify_parent(args.quality)!=protocol['quality_completion_sha256']:raise ValueError('Quality receipt changed')
    for name,digest in sources.items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError('Old source changed: '+name)
    for owner in [extract,protocol]:
        for name,digest in owner['source_code_sha256'].items():
            if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError('Frozen new source changed')
    if sha((root/'extraction_protocol.json').read_bytes())!=protocol['extraction_protocol_sha256'] or sha((args.quality/'prospective_native_replication36.json').read_bytes())!=protocol['prospective_protocol_sha256']:raise ValueError('Prospective protocol changed')
    log=(root/'test_log.txt').read_bytes()
    if sha(log)!=extract['test_log_sha256'] or b'Ran 330 tests' not in log or not log.rstrip().endswith(b'OK'):raise ValueError('Test evidence differs')
    members=members_from_metadata(args.quality,args.metadata)
    if members!=extract['members']:raise ValueError('Selected ZIP whitelist differs')
    rows=read(root/'rgb_manifest.json')+read(root/'reference_manifest.json');by_member={r['source_member']:r for r in rows};data_root=Path(extract['data_root']);total=0;intervals=set()
    for member in members:
        row=by_member[member['path']];parts=[]
        for receipt in row['ranges']:
            identity={k:receipt[k] for k in ['url','member','offset','bytes','role']};key=sha(json.dumps(identity,sort_keys=True).encode());raw=(data_root/'ranges'/(key+'.bin')).read_bytes();saved=read(data_root/'ranges'/(key+'.json'))
            if identity['url']!=member['archive_url'] or identity['member']!=member['path'] or any(saved[k]!=v for k,v in identity.items()):raise ValueError('Range source differs')
            if sha(raw)!=receipt['sha256'] or sha(raw)!=saved['sha256'] or len(raw)!=identity['bytes']:raise ValueError('Range cache changed')
            if saved['http_status']!=206 or saved['content_range']!=f"bytes {identity['offset']}-{identity['offset']+identity['bytes']-1}/{member['archive_size']}":raise ValueError('Nonexact range receipt')
            parts.append(raw);intervals.add((identity['member'],identity['offset'],identity['bytes'],identity['role']));total+=len(raw)
        header,variable,packed=parts;n,extra,start=parse_local_header(header,member)
        wanted=[(member['local_header_offset'],30,'header30'),(member['local_header_offset']+30,n+extra,'name_extra'),(start,member['compressed_bytes'],'mask_compressed' if member['mode']=='published_post_mask' else 'rgb_compressed')]
        if [(r['offset'],r['bytes'],r['role']) for r in row['ranges']]!=wanted or variable[:n].decode('utf-8' if member['flag_bits']&2048 else 'cp437')!=member['path']:raise ValueError('Unexpected range/name')
        png=decompress_rgb(packed,member)
        if png!=Path(row['path']).read_bytes() or sha(png)!=row['sha256']:raise ValueError('PNG replay differs')
        if member['mode']=='published_post_mask':
            labels,encoding=decode_reference(png)
            if row['encoding']!=encoding or row['class_pixels']!={str(i):int((labels==i).sum()) for i in range(5)}:raise ValueError('Reference decode differs')
        else:
            with Image.open(row['path']) as im:
                if im.mode!='RGB' or im.size!=(512,512) or sha(im.tobytes())!=row['pixel_sha256_rgb']:raise ValueError('RGB pixel encoding differs')
    ledger=read(data_root/'ranges/transfer_ledger.json')
    if len(ledger)!=324 or len(intervals)!=324 or total!=sum(r['received_bytes'] for r in ledger) or sum(r['reserved_bytes'] for r in ledger)>48*1024*1024 or any(r['status']!='EXACT_RANGE_RECEIVED' for r in ledger):raise ValueError('Network inventory differs')
    if {(r['member'],r['offset'],r['bytes'],r['role']) for r in ledger}!=intervals:raise ValueError('Unplanned range access')
    for name,key in [('model.safetensors','weights_sha256'),('config.json','config_sha256')]:
        if sha((Path(protocol['model_directory'])/name).read_bytes())!=protocol['checkpoint'][key]:raise ValueError('Model changed')
    calls=[read(p) for p in sorted((root/'calls').glob('call_*.json'))];observations=read(root/'observations.json');by_job={r['job']:r for r in observations}
    if len(calls)!=758 or len(list((root/'cache').glob('*.npz')))!=758 or len(observations)!=756 or len(by_job)!=756:raise ValueError('Model call inventory differs')
    index=0;tensors=0
    def check(job,pair,mode,unit=None):
        nonlocal index,tensors
        row=calls[index];index+=1;tensor=prepare_tensor(*pair,'native')
        if row['status']!='COMPLETE' or row['index']!=index or row['job']!=job or row['input_mode']!=mode or row['unit_id']!=unit:raise ValueError('Call identity differs')
        if sha(tensor.numpy().tobytes())!=row['input_tensor_sha256'] or list(tensor.shape)!=row['input_tensor_shape']:raise ValueError('Input reconstruction differs')
        size=512 if mode==COARSE else 256 if mode==NATIVE256 else 128;mask=checked_mask(Path(row['cache_file']),row['cache_sha256'],(size,size));tensors+=1
        if int(mask.sum())!=row['predicted_severe_pixels'] or (int(mask.sum())>=4)!=row['declared_target']:raise ValueError('Mask summary differs')
        if mode!='positive_control' and any(by_job[job].get(k)!=v for k,v in row.items()):raise ValueError('Call receipt differs from observation')
        return row,mask
    controls=read(root/'positive_control_results.json')
    if len(controls)!=2:raise ValueError('Controls missing')
    for previous,current in zip(protocol['positive_controls'],controls):
        pair=[]
        for r in previous['images']:
            if sha(Path(r['path']).read_bytes())!=r['sha256']:raise ValueError('Old control changed')
            with Image.open(r['path']) as im:pair.append(im.convert('RGB'))
        row,mask=check(previous['sample_id']+'_positive_control',crop_pair(*pair,previous['cell_id']),'positive_control')
        old=checked_mask(Path(previous['legacy_cache_file']),previous['legacy_cache_sha256'],(128,128))
        if not np.array_equal(mask,old) or not current['exact_mask_equal'] or current['different_pixels_from_legacy']!=0 or any(current.get(k)!=v for k,v in row.items()):raise ValueError('Control regression')
    images={(r['sample_id'],r['mode']):r for r in protocol['rgb_inputs']};references={r['sample_id']:r for r in read(root/'reference_manifest.json')};features=read(root/'public_features.json');scenes={}
    for sid in protocol['selected_ids']:
        pair=[]
        for date in ['pre','post']:
            r=images[sid,date]
            with Image.open(r['path']) as im:pair.append(im.convert('RGB'))
        if features[sid]!=public_features(*pair):raise ValueError('Public cues differ')
        coarse=tuple(im.resize((64,64),Image.Resampling.BOX).resize((512,512),Image.Resampling.BILINEAR) for im in pair);scene_rows=[]
        _,free=check(sid+'_free_coarse',coarse,COARSE,'scene');scene_rows.append(by_job[sid+'_free_coarse']);masks={mode:{} for mode in MODES}
        for mode in MODES:
            for uid,(x,y,width),cost in geometry(mode):
                patches=tuple(im.crop((x,y,x+width,y+width)) for im in pair);job=sid+'_'+uid+'_'+mode;_,mask=check(job,patches,mode,uid);masks[mode][uid]=mask;scene_rows.append(by_job[job])
        if scene_rows!=read(root/'scene_records'/(sid+'.json')):raise ValueError('Scene record differs')
        labels,_=decode_reference(Path(references[sid]['path']).read_bytes());target=labels>=3
        domains={'all_published_pixels':np.ones((512,512),dtype=np.bool_),'both_dates_nonzero':(np.asarray(pair[0]).max(2)>0)&(np.asarray(pair[1]).max(2)>0)}
        scenes[sid]=(references[sid]['event'],target,free,domains,masks)
    if index!=758:raise ValueError('Missing forward calls')
    previous_ids=set(protocol['prospective_protocol']['old18_ids_excluded'])
    if len(scenes)!=36 or set(scenes)&previous_ids:raise ValueError('New36-only scope violated')
    grouped=defaultdict(list);seen=set();count=0;actions=0
    evaluation=read(root/'evaluation_report.json')
    if sha((root/'ranked_acquisitions.jsonl.gz').read_bytes())!=evaluation['trajectory_sha256']:raise ValueError('Rank bytes changed')
    with gzip.open(root/'ranked_acquisitions.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);sid=row['sample_id'];mode=row['input_mode'];budget=row['budget'];seed=row['order_seed'];event,ref,free,domains,masks=scenes[sid];key=(sid,mode,budget,seed)
            if key in seen or mode not in MODES or budget not in BUDGETS or seed not in ORDER_SEEDS or row['event']!=event:raise ValueError('Unplanned ranked record')
            seen.add(key);trace=independent_rank(features[sid],mode,budget,seed);lookup={uid:(bounds,cost) for uid,bounds,cost in geometry(mode)};paid=np.zeros_like(ref);spent=0
            for a in trace:
                uid=a['unit_id'];(x,y,width),cost=lookup[uid];paid[y:y+width,x:x+width]|=masks[mode][uid];spent+=cost;a['declared_target']=int(masks[mode][uid].sum())>=4
                if a['cost']!=cost or a['cumulative_cost']!=spent or spent>budget:raise ValueError('Paid area differs')
            if trace!=row['trace']:raise ValueError('Rank replay differs')
            for domain,mask in domains.items():
                score=independent_score(paid,ref,free,mask,spent,len(trace),True)
                if score!=row['scores'][domain]:raise ValueError('Pixel union replay differs')
                grouped[sid,event,domain,mode,budget].append(score)
            count+=1;actions+=len(trace)
    if count!=2160 or actions!=12600:raise ValueError('Rank/action count differs')
    scores=[];opportunities=[]
    for sid,(event,ref,free,domains,masks) in scenes.items():
        for mode in MODES:
            paid=np.zeros_like(ref)
            for uid,(x,y,width),cost in geometry(mode):paid[y:y+width,x:x+width]|=masks[mode][uid]
            for domain,mask in domains.items():opportunities.append({'sample_id':sid,'event':event,'domain':domain,'input_mode':mode,**independent_score(paid,ref,free,mask,16,len(geometry(mode)),True)})
            for budget in BUDGETS:
                for domain,mask in domains.items():scores.append({'sample_id':sid,'event':event,'domain':domain,'input_mode':mode,'budget':budget,'policy':'random_exact',**random_by_combinatorics(masks[mode],mode,budget,ref,free,mask)})
    for (sid,event,domain,mode,budget),rows in grouped.items():scores.append({'sample_id':sid,'event':event,'domain':domain,'input_mode':mode,'budget':budget,'policy':'raw_change_rank',**mean_scores(rows)})
    if scores!=read(root/'scene_scores.json') or opportunities!=read(root/'full_opportunities.json') or aggregate(scores)!=read(root/'summary.json'):raise ValueError('Exact-random/aggregation replay differs')
    if decide_replication(aggregate(scores),opportunities)!=read(root/'next_step_decision.json'):raise ValueError('Screen replay differs')
    report=read(root/'inference_report.json')
    if report['total_forward_calls']!=758 or report['status']!='COMPLETE' or report['observations_sha256']!=sha((root/'observations.json').read_bytes()):raise ValueError('Inference report differs')
    atomic_json(root/'audit.json',{'status':'NEW36_RANGES_INPUTS_AREA_UNIONS_AND_EXPECTATIONS_REPLAYED','old_frozen_sources_unchanged':len(sources),
        'selected_new_pairs':36,'old18_overlap':0,'extracted_members_verified':108,'exact_ranges_replayed':324,'network_bytes_verified':total,
        'forward_calls':758,'input_tensors_reconstructed':tensors,'positive_controls_pixel_identical':2,'native128_masks_verified':576,'native256_masks_verified':144,'coarse_masks_verified':36,
        'ranked_records_replayed':count,'ranked_domain_scores_replayed':count*2,'paid_acquisitions_verified':actions,'exact_random_scores_replayed':432,
        'scene_domain_policy_means_replayed':len(scores),'full_coverage_scores_replayed':len(opportunities),'tests_passed':330,'new_tests':16,
        'primary_screen_passed':read(root/'next_step_decision.json')['primary_screen_passed'],'new_neural_calls_by_audit':0,'new_fitting':False,'paid_gpu':False,
        'old18_pooled':False,'human_annotations_created':0,'building_instances_created':0,'old_reserved_sets_scored':False,'independent_efficacy_claim':False})
    print(json.dumps(read(root/'audit.json')))


if __name__=='__main__':main()
