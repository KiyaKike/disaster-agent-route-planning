"""Metadata-only admission audit for paid independent verification; no label mining."""
import csv,json,datetime,hashlib
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'output/disaster_route_research_admission_v1'
CAT=ROOT/'output/disaster_route_crasar_region_v1/statistics.csv'
INV=ROOT/'output/disaster_route_crasar_feasibility_20261103/file_inventory.json'
REV='4c2732a7ab9658517ca8caa69bfee9e1f87ca532'

def date(row):return datetime.datetime.strptime(row['Date (mm/dd/yyy)'],'%m/%d/%Y').date()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def run():
    O.mkdir(exist_ok=True,parents=True)
    if (O/'completion_receipt.json').exists():raise RuntimeError('Admission audit is sealed')
    rows=list(csv.DictReader(CAT.open()));paths=set(json.loads(INV.read_text()))
    # No BDA counts,damage labels,model scores or test annotations used for selection.
    candidates=[]
    for u in sorted((r for r in rows if r['Source']=='sUAS' and r['Train/Test']=='Train'),key=lambda r:r['Orthomosaic']):
        scene=u['Orthomosaic'];linked=[r for r in rows if r['Train/Test']=='Train' and r['Orthomosaic'].startswith(scene+'_')]
        sat=[r for r in linked if r['Source']=='Satellite']
        pre=sorted((r for r in sat if r['Pre/Post Event']=='PRE'),key=lambda r:(date(r),r['Orthomosaic']),reverse=True)
        post=sorted((r for r in sat if r['Pre/Post Event']=='POST'),key=lambda r:(date(r),r['Orthomosaic']))
        if not pre or not post:continue
        def exists(r):
            source={'sUAS':'UAS','Satellite':'SATELLITE','Crewed Aircraft':'CREWED'}[r['Source']]
            return f"train/imagery/{source}/{r['Orthomosaic']}" in paths
        if not all(exists(r) for r in [u,pre[0],post[0]]):continue
        alternatives=[r for r in linked if r['Pre/Post Event']=='POST' and r['Orthomosaic']!=post[0]['Orthomosaic'] and exists(r)]
        same_primary=[r for r in alternatives if date(r)==date(post[0])]
        same_uas=[r for r in alternatives if date(r)==date(u)]
        def public(r):return {k:r[k] for k in ['Orthomosaic','Source','Event','Date (mm/dd/yyy)','Pre/Post Event','Platform / Provider']}
        candidates.append({'rank_zero_based':len(candidates),'scene':scene,'event':u['Event'],'uas_date':date(u).isoformat(),
          'selected_pre':public(pre[0]),'selected_post':public(post[0]),
          'independent_uas_sessions_established':False,'same_exact_scene_uas_orthomosaics':1,
          'additional_post_files':list(map(public,alternatives)),
          'additional_files_same_date_as_primary_post':list(map(public,same_primary)),
          'additional_files_same_date_as_uas':list(map(public,same_uas)),
          'additional_crewed_same_date_as_primary_post':sum(r['Source']=='Crewed Aircraft' for r in same_primary),
          'additional_crewed_same_date_as_uas':sum(r['Source']=='Crewed Aircraft' for r in same_uas),
          'qualification':'Metadata candidates only. Day equality is not simultaneity,spatial coverage,independent error or field truth.'})
    summary={'dataset_revision':REV,'catalogue_sha256':sha(CAT),'inventory_sha256':sha(INV),
      'selection':'Same24training scene order as previous protocol;all counts use only file/source/date metadata',
      'eligible_scene_count':len(candidates),'events':dict(Counter(r['event'] for r in candidates)),
      'scenes_with_additional_post_file':sum(bool(r['additional_post_files']) for r in candidates),
      'scenes_with_additional_file_same_day_as_primary_post':sum(bool(r['additional_files_same_date_as_primary_post']) for r in candidates),
      'scenes_with_additional_file_same_day_as_uas':sum(bool(r['additional_files_same_date_as_uas']) for r in candidates),
      'scenes_with_crewed_same_day_as_primary_post':sum(r['additional_crewed_same_date_as_primary_post']>0 for r in candidates),
      'scenes_with_crewed_same_day_as_uas':sum(r['additional_crewed_same_date_as_uas']>0 for r in candidates),
      'source_geometry_coverage_checked_for_new_views':False,'error_independence_established':False,'same_state_field_truth_established':False,
      'flight_reinspection_supported':False,'annotations_or_new_imagery_downloaded':False,'new_model_calls':0,
      'existing_inspected_scenes':[r for r in candidates if r['rank_zero_based']<6]}
    dump(O/'verification_catalogue.json',candidates);dump(O/'data_feasibility.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='existing_inspected_scenes'},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
