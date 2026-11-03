"""Execute the four fixed metadata-ranked training regions; preserve every failure."""
import os,sys,json,time,subprocess,hashlib,shutil,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'work/disaster_route'
B=ROOT/'output/disaster_route_crasar_batch4_v1'
NAMES=['1001-Ft-Myers-Beach-DIRT.geo.tif','1001-Iona-Point.geo.tif','1001-Kennedy-Green-Mobile-Homes.geo.tif','1001-McGregor-College-Pkwy-South.1.geo.tif']

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def invoke(folder,rank,script):
    env=dict(os.environ,CRASAR_REGION_OUTPUT=str(folder),CRASAR_REGION_RANK=str(rank),CRASAR_BATCH_PREDECLARED='1')
    # Iona's published UAS has no usable pyramid; this is a transport cap amendment,not case selection.
    if rank==3:env['CRASAR_IMAGE_RANGE_CAP_BYTES']=str(64*1024*1024)
    t=time.monotonic()
    with (folder/(script+'.log')).open('a') as log:
        result=subprocess.run([sys.executable,str(HERE/script)],env=env,stdout=log,stderr=subprocess.STDOUT)
    return {'script':script,'exit_code':result.returncode,'wall_seconds':time.monotonic()-t}

def acquire(rank,name):
    folder=B/f'case_rank{rank+1:02d}';folder.mkdir(exist_ok=True)
    old=folder/'stage_status.json'
    history=json.loads(old.read_text()).get('history',[]) if old.exists() else []
    for script,artifact in [('probe_crasar_region.py','metadata_stage.json'),('extract_crasar_region.py','roi_acquisition.json'),('qualify_crasar_region.py','qualification.json')]:
        if (folder/artifact).exists():continue
        r=invoke(folder,rank,script);history.append(r)
        dump(folder/'stage_status.json',{'rank':rank,'scene':name,'history':history,'last_stage':script,'exit_code':r['exit_code']})
        if r['exit_code']:return {'rank':rank,'scene':name,'folder':str(folder),'status':'ACQUISITION_OR_QUALIFICATION_FAILED','history':history}
        if script=='probe_crasar_region.py':assert json.loads((folder/'protocol.json').read_text())['images']['uas']['Orthomosaic']==name
    return {'rank':rank,'scene':name,'folder':str(folder),'status':'QUALIFIED_FOR_DIAGNOSTIC_REVIEW','history':history}

def run():
    B.mkdir(parents=True,exist_ok=True)
    if (B/'completion_receipt.json').exists():raise RuntimeError('Batch sealed')
    attempts=B/'attempts';attempts.mkdir(exist_ok=True)
    previous=B/'execution_status.json'
    if previous.exists():shutil.copyfile(previous,attempts/f'previous_execution_{len(list(attempts.glob("*.json")))+1:03d}.json')
    parent=ROOT/'output/disaster_route_crasar_region_rank2_v1/next_batch_protocol.json'
    protocol={'parent_protocol':str(parent),'parent_protocol_sha256':sha(parent),'scene_names':NAMES,'zero_based_ranks':[2,3,4,5],
      'selection':'Exactly the four previously fixed scenes,one128m center ROI each;all failures retained;no outcome-based replacement',
      'primary':'UAS major/destroyed','secondary':'UAS minor/major/destroyed,predeclared for this batch,exploratory development',
      'evaluation':'Same frozen detector,128x128 per32m window,thresholds.38/.8/.1,budgets60/100/140,5m/s,10s observation,1s planning,return paid',
      'quality_scope':'Geometry support and image registration separate. Routing permitted only as explicitly provisional image-access diagnostics unless independent registration passes.',
      'resource_caps':'Two independent scene acquisition workers,eight range threads per source,32MiB cached image ranges per source,12MB annotation limit',
      'prediction_increment':'Leave-one-region-out prediction of unvisited fixed-model damage fraction,using prior features versus same plus paid-prefix features;candidate rows exclude prefix and shared-building overlaps;no event-generalization claim',
      'outcome_selection':False,'training_new_detector':False,'gpu_rental':False}
    p=B/'batch_protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:dump(p,protocol)
    src=B/'source_snapshots';src.mkdir(exist_ok=True)
    for script in ['probe_crasar_region.py','extract_crasar_region.py','qualify_crasar_region.py','crasar_alignment_core.py','run_crasar_fixed_perception.py','evaluate_crasar_routes.py','crasar_route_controls.py','route_headroom_core.py','run_crasar_batch.py']:
        target=src/script
        if not target.exists():shutil.copyfile(HERE/script,target)
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(acquire,rank,name):(rank,name) for rank,name in zip([2,3,4,5],NAMES)}
        for job in as_completed(jobs):
            rank,name=jobs[job]
            try:r=job.result()
            except Exception as e:r={'rank':rank,'scene':name,'status':'FAILED','error':str(e),'folder':str(B/f'case_rank{rank+1:02d}')}
            results.append(r);dump(B/'acquisition_status.json',sorted(results,key=lambda r:r['rank']))
            print(json.dumps(r,ensure_ascii=False),flush=True)
    for r in sorted(results,key=lambda r:r['rank']):
        if r['status']!='QUALIFIED_FOR_DIAGNOSTIC_REVIEW':continue
        folder=Path(r['folder']);q=json.loads((folder/'qualification.json').read_text())
        if not q['geometry_eligible']:r['status']='NO_ELIGIBLE_BUILDINGS';continue
        metadata=json.loads((folder/'image_metadata.json').read_text())
        if any(v['bands']<3 for v in metadata.values()):r['status']='RGB_SENSOR_GATE_FAILED';continue
        dates=q['capture_dates_author_catalog']
        if not dates['pre']<dates['post']<=dates['uas']:r['status']='TEMPORAL_GATE_FAILED';continue
        for script,artifact in [('run_crasar_fixed_perception.py','perception_runtime.json'),('evaluate_crasar_routes.py','experiment_summary_any_damage.json')]:
            if (folder/artifact).exists():continue
            stage=invoke(folder,r['rank'],script);r['history'].append(stage)
            if stage['exit_code']:r['status']='INFERENCE_OR_REPLAY_FAILED';break
        else:r['status']='PROVISIONAL_REPLAY_COMPLETE'
        dump(B/'execution_status.json',sorted(results,key=lambda r:r['rank']))
        print(json.dumps({'scene':r['scene'],'status':r['status']},ensure_ascii=False),flush=True)
    dump(B/'execution_status.json',sorted(results,key=lambda r:r['rank']))

if __name__=='__main__':run()
