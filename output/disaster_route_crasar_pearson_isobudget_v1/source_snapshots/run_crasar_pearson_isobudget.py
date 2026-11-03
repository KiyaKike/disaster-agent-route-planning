"""Freeze Boone Pearson operator; evaluate other fixed regions under hard budgets."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from crasar_pearson_budget_core import POLICIES, SPEED, OBSERVE, PLAN, budget_episode
from route_headroom_core import exact_catalogue, audit_bellman, subset_rewards, reconstruct
from run_boone_context_experiment import MODEL, dump, feature_kernel, masked_crop, sha
from run_boone_structure_check import pearson_change


REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
BUDGETS = (60,100,140)
SCENES = ('case_rank03','case_rank04','case_rank05','case_rank06')


def prepare(root,out):
    read = lambda p: json.loads(p.read_text())
    batch = root/'output/disaster_route_crasar_batch4_v1'
    status = []
    for name in SCENES:
        src, dst = batch/name, out/name
        dst.mkdir()
        metadata = read(src/'image_metadata.json')
        protocol = read(src/'protocol.json')
        row = {'case':name,'scene':protocol['images']['uas']['Orthomosaic']}
        # Do not use exported RGB display mode to override the original sensor gate.
        bands = {k:v['bands'] for k,v in metadata.items()}
        row['original_bands'] = bands
        if any(n<3 for n in bands.values()):
            row.update(status='EXCLUDED_ORIGINAL_NON_RGB',reason='Original fixed RGB gate; no replacement')
            status.append(row)
            continue
        geometry = sorted([g for g in read(src/'public_geometry.json') if g['eligible']],key=lambda g:g['building_id'])
        images = {s:Image.open(src/f'{s}_roi.png').convert('RGB') for s in ('pre','post')}
        rows, features, centers = [], [], []
        for g in geometry:
            uid = g['building_id']
            pixels, crops = {}, {}
            for s in ('pre','post'):
                crop, values, box, area = masked_crop(images[s],g['roi_polygon_px'])
                rel = f'crops/{uid}/{s}.png'
                path = dst/rel
                path.parent.mkdir(parents=True,exist_ok=True)
                crop.save(path)
                pixels[s] = values
                crops[s] = {'path':rel,'sha256':sha(path),'roi_box':box,'mask_pixels':area}
            change, channel = pearson_change(pixels['pre'],pixels['post'])
            features.append([change])
            points = np.asarray(g['polygon_utm'],float)
            centers.append(points[:-1].mean(axis=0).tolist())
            rows.append({'building_id':uid,'crops':crops,'channel_correlations':channel,'pearson_change':change})
        _,z,similarity,tau = feature_kernel(features)
        bounds = read(src/'roi_protocol.json')['bounds']
        xy = centers+[[float((bounds[0]+bounds[2])/2),float(bounds[1])]]
        public = {'building_ids':[r['building_id'] for r in rows],'xy_m':xy,
                  'similarity':similarity.tolist(),'pearson_features':features,
                  'standardized_features':z.tolist(),'tau':tau,
                  'depot':'Original ROI south midpoint; no known source label at start',
                  'source_hashes':{p:sha(src/p) for p in ('pre_roi.png','post_roi.png','public_geometry.json','roi_protocol.json','image_metadata.json')}}
        dump(dst/'public_inputs.json',public)
        dump(dst/'crop_manifest.json',{'rows':rows,'source':str(src.relative_to(root))})
        row.update(status='PREPARED',buildings=len(rows),tau=tau)
        status.append(row)
    dump(out/'region_status.json',status)
    return status


def infer_priors(out,status):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForVision2Seq
    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    processor.image_processor.size = {'longest_edge':512}
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float32,
                attn_implementation='eager',local_files_only=True).to(device).eval()
    options = [processor.tokenizer.encode(s,add_special_tokens=False) for s in ('A',' A','B',' B')]
    assert all(len(v)==1 for v in options)
    options = [v[0] for v in options]
    dump(out/'perception_runtime.json',{'model':MODEL,'revision':REVISION,'device':device,
          'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float32',
          'answer_tokens':options,'processor_longest_edge':512,'image_splitting':False,
          'prompt':'Identical no-history prompt as Boone v2','new_training':False,
          'local_files_only':True,'external_api_calls':0,'gpu_rental':False})
    count = 0
    for region in status:
        if region['status']!='PREPARED':
            continue
        dst = out/region['case']
        rows = json.loads((dst/'crop_manifest.json').read_text())['rows']
        priors = {}
        for row in rows:
            uid = row['building_id']
            content = [{'type':'text','text':
                'Assess visible roof damage in the TARGET building from satellite images. '
                'The images show only individual building footprints. '
                'A means visible roof damage; B means no visible roof damage. '}]
            images, paths = [], []
            for s,description in [('pre','TARGET building: pre-disaster satellite.'),('post','TARGET building: post-disaster satellite.')]:
                content.extend([{'type':'text','text':description},{'type':'image'}])
                path = row['crops'][s]['path']
                paths.append(path)
                images.append(Image.open(dst/path).convert('RGB'))
            content.append({'type':'text','text':'Which label best describes the TARGET roof? Reply with exactly A or B.'})
            prompt = processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True)
            inputs = processor(text=prompt,images=images,return_tensors='pt').to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1].float()
                restricted = torch.softmax(logits[options],dim=-1)
                score = float(restricted[:2].sum().cpu())
                mass = float(torch.softmax(logits,dim=-1)[options].sum().cpu())
            priors[uid] = score
            dump(dst/f'calls/prior_{uid}.json',{'target_id':uid,'score':score,'answer_set_probability_mass':mass,
                 'image_paths':paths,'prompt':prompt,'revision':REVISION,'seconds':time.perf_counter()-started})
            count += 1
            print(f"{region['case']} initial {count}: {score:.5f}",flush=True)
            del inputs,logits,restricted
        dump(dst/'vlm_initial_scores.json',priors)
    return count


def evaluate(root,out,status):
    read = lambda p: json.loads(p.read_text())
    summaries, routes, comparisons, solver_records = [], [], [], []
    for region in status:
        if region['status']!='PREPARED':
            continue
        dst = out/region['case']
        public = read(dst/'public_inputs.json')
        ids, xy = public['building_ids'], np.array(public['xy_m'])
        p0_dict = read(dst/'vlm_initial_scores.json')
        p0 = np.array([p0_dict[u] for u in ids])
        similarity = np.array(public['similarity'])
        refs = {r['building_id']:r for r in read(root/'output/disaster_route_crasar_batch4_v1'/region['case']/'private_reference.json')}
        known = {'no damage','minor damage','major damage','destroyed'}
        labels = [None if refs[u]['labels']['uas'] not in known else int(refs[u]['labels']['uas']!='no damage') for u in ids]
        severe = [int(refs[u]['labels']['uas'] in ('major damage','destroyed')) for u in ids]
        started = time.perf_counter()
        cat = exact_catalogue(xy,len(ids))
        states = audit_bellman(cat)
        solver_records.append({'case':region['case'],'enumerated_subsets':len(cat['closed']),
                               'bellman_states_verified':states,'precompute_and_audit_wall_seconds':time.perf_counter()-started})
        truth_weights = np.array([int(v==1) for v in labels])
        oracle_rewards = subset_rewards(truth_weights)
        for budget in BUDGETS:
            costs = cat['closed']/SPEED+cat['cardinality']*OBSERVE+PLAN
            feasible = np.flatnonzero(costs<=budget+1e-9)
            ceiling = int(oracle_rewards[feasible].max())
            for policy in POLICIES:
                accesses = []
                def reveal(i):
                    assert i not in accesses
                    accesses.append(i)
                    return labels[i]
                row = budget_episode(p0,xy,similarity,policy,budget,reveal,catalogue=cat)
                assert accesses==row['route']
                row.update(case=region['case'],scene=region['scene'],label_access_indices=accesses,
                           route_building_ids=[ids[i] for i in row['route']])
                hits = sum(labels[i]==1 for i in row['route'])
                first = next((o['observation_complete_proxy_s'] for o in row['observations'] if o['source_label']==1),None)
                row.update(source_positive_checks=hits,first_positive_check_proxy_s=first,
                           unknown_source_checks=sum(labels[i] is None for i in row['route']),
                           source_severe_checks=sum(severe[i] for i in row['route']))
                assert hits<=ceiling
                routes.append(row)
                summaries.append({'case':region['case'],'scene':region['scene'],'budget_proxy_s':budget,
                                  'policy':policy,'positive_checks':hits,'source_positive_total':int(sum(v==1 for v in labels)),
                                  'severe_checks':row['source_severe_checks'],'unknown_checks':row['unknown_source_checks'],
                                  'visits':len(row['route']),'distance_m':row['distance_m'],
                                  'elapsed_proxy_s':row['elapsed_proxy_s'],'first_positive_check_proxy_s':first,
                                  'hindsight_ceiling':ceiling})
            subset = {r['policy']:r for r in summaries if r['case']==region['case'] and r['budget_proxy_s']==budget}
            for control in POLICIES:
                if control=='pearson':continue
                comparisons.append({'case':region['case'],'budget_proxy_s':budget,'control':control,
                                    'pearson_checks':subset['pearson']['positive_checks'],
                                    'control_checks':subset[control]['positive_checks'],
                                    'delta':subset['pearson']['positive_checks']-subset[control]['positive_checks']})
        dump(dst/'routes.json',[r for r in routes if r['case']==region['case']])
        region['status']='REPLAY_COMPLETE'
    dump(out/'region_status.json',status)
    dump(out/'summary.json',{'table':summaries,'main_budget_proxy_s':100,'routes':len(routes),
                           'scope':'Boone-external previously inspected training regions; building-label-assisted archive replay',
                           'physical_field_truth':False,'same_event':True,'budget_violations':0})
    dump(out/'paired_comparisons.json',comparisons)
    dump(out/'solver_certificates.json',solver_records)
    with (out/'summary.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(summaries[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({'main_results':[r for r in summaries if r['budget_proxy_s']==100]},ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root,out = args.workspace.resolve(),args.output.resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True)
    # Save the complete fixed method and endpoint contract before predictions/routes.
    dump(out/'protocol.json',{'regions':list(SCENES),'primary_endpoint':'Any-damage UAS-view source-positive checks',
         'secondary_endpoint':'Severe UAS-view source checks; unknown labels separately counted',
         'budgets_proxy_s':list(BUDGETS),'main_budget_proxy_s':100,'speed_mps':SPEED,
         'observation_s':OBSERVE,'planning_update_s':PLAN,'return_paid':True,
         'online_planning_charge':'1s for every decision attempt, including final no-feasible decision; exact static1s once',
         'unit':'One building footprint per paid archive observation; not original32m-window sensor replay',
         'no_free_initial_anchor':True,'priors':'Same frozen SmolVLM no-history prompt and token scoring as Boone',
         'model':MODEL,'revision':REVISION,'feature':'Frozen1-mean RGB-channel Pearson correlation',
         'feature_scale_rule':'Unlabelled per-region standardization floor0.05 and median-positive-distance Gaussian; no label tuning',
         'space_scale_m':50,'residual_gain':.5,'selection':'current score / outgoing straight-line distance; feasibility includes return',
         'feedback':'Paid UAS author-view binary label; unknown skips update, never converted into a negative',
         'policies':list(POLICIES),'strong_static':'Exact Held-Karp all-subset public-p0 objective; Bellman certificate',
         'limitations':['Same-event train/development regions previously inspected','No independent registration certificate',
                        'Point observation and10s service are assumptions, not measured camera coverage',
                        'No site truth or field sensor accuracy claim'],
         'locked_source_hashes':{name:sha(root/'work/disaster_route'/name) for name in
                                ('run_boone_two_step.py','run_boone_structure_check.py','run_boone_strength_control.py',
                                 'crasar_pearson_budget_core.py','run_crasar_pearson_isobudget.py')},
         'parent_receipt_sha256':sha(root/'output/disaster_route_boone_two_step_v1/completion_receipt.json')})
    status = prepare(root,out)
    count = infer_priors(out,status)
    dump(out/'inference_count.json',{'initial_satellite_forward_calls':count,'new_history_calls':0,'api_cost':0,'gpu_rental':False})
    evaluate(root,out,status)


if __name__=='__main__':
    main()
