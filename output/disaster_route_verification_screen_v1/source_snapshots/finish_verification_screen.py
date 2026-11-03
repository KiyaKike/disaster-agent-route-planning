"""Validate the screening conclusion and seal its reproducible evidence."""
import json,hashlib,shutil
from pathlib import Path
from run_verification_screen import ROOT,HERE,O

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Sealed')
    s=json.loads((O/'summary.json').read_text());r=json.loads((O/'results.json').read_text())
    assert len(r)==108 and s['candidate_matches_exact_bayes_conditions']==108
    for row in r:
        assert row['policies']['exact_bayes_reference']['values']==row['policies']['disagreement_screen_candidate']['values']
    assert sha(O/'problem_protocol.md')==json.loads((O/'protocol.json').read_text())['problem_sha256']
    assert (O/'tests.txt').read_text().rstrip().endswith('OK')
    entry=json.loads((O/'real_entry_metadata.json').read_text())
    assert entry['status']=='HEADER_ONLY_CANDIDATE_CHECK' and not entry['calibration_anchor_established']
    verified={}
    for name in ['disaster_route_crasar_batch4_v1','disaster_route_research_admission_v1']:
        p=ROOT/'output'/name;receipt=json.loads((p/'completion_receipt.json').read_text())
        for file,h in receipt['files'].items():assert sha(p/file)==h,(name,file)
        verified[name]=len(receipt['files'])
    snapshot=O/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for name in ['verification_screen_core.py','run_verification_screen.py','test_verification_screen.py','probe_verification_entry.py','finish_verification_screen.py','probe_crasar_region.py']:
        shutil.copyfile(HERE/name,snapshot/name)
    audit={'status':'COMPLETE_RESEARCH_SCREEN_NOT_NEW_METHOD','conditions':len(r),'exact_bayes_equalities':108,
      'independent_branch_enumerations':s['independent_branch_enumerations'],'independent_lp_checks':s['independent_numerical_lp_checks'],
      'tests_passed':6,'full_policy_space_optimality':False,'scope':'Only frozen at-most-one-calibration and two-terminal-route family',
      'real_data_entry_header_verified':True,'real_verification_or_calibration_validated':False,
      'new_model_calls':0,'new_training':False,'gpu_rental':False,'previous_frozen_files_verified':verified,
      'decision':s['decision'],'new_method_claim':False,'new_theorem_claim':False}
    (O/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and '.sparse.tif' not in p.name and p.name!='completion_receipt.json'}
    (O/'completion_receipt.json').write_text(json.dumps({'status':'COMPLETE_CANDIDATE_STOPPED_NO_METHOD_CONTRIBUTION','files':files,
      'excluded_ephemeral':'Header-only sparse TIFF is a partial logical-size container,not a complete image'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'audit':audit,'sealed_files':len(files)},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
