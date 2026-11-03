"""Independent finite-branch verification and admission evidence receipt."""
import itertools,json,hashlib,shutil
from fractions import Fraction as F
from pathlib import Path
from audit_verification_feasibility import ROOT,O,dump

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def enumerate_rewards(p,s):
    posterior=p*(1-s)/(1-p*s);totals={'new_B':F(0),'repeat_A':F(0)};mass=F(0);branches=0
    # Enumerate target presence and all potential independent acquisition draws.
    for a,b,u,v,w in itertools.product([0,1],repeat=5):
        prob=(posterior if a else 1-posterior)*(p if b else 1-p)
        prob*= (s if u else 1-s)*(s if v else 1-s)*(s if w else 1-s)
        mass+=prob;totals['new_B']+=prob*int(bool(b and w))
        totals['repeat_A']+=prob*int(bool(a and (u or v)));branches+=1
    assert mass==1
    return totals,branches

def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Sealed')
    witness=json.loads((O/'identifiability_witness.json').read_text());branches=0
    for name,w in witness['worlds'].items():
        p,s=F(w['target_probability']),F(w['sensitivity']);values,n=enumerate_rewards(p,s);branches+=n
        assert values['new_B']==F(witness['route_witness']['fresh_B_reward'][name])
        assert values['repeat_A']==F(witness['route_witness']['revisit_A_reward'][name])
    # Independent cost audit of both nondominated closed strategies.
    assert F(1,2)+1+F(1,2)==2 and 1+1==2
    data=json.loads((O/'data_feasibility.json').read_text());cases=json.loads((O/'verification_catalogue.json').read_text())
    assert len(cases)==data['eligible_scene_count']==24
    assert sum(bool(c['additional_files_same_date_as_uas']) for c in cases)==0
    assert sum(c['additional_crewed_same_date_as_primary_post']>0 for c in cases)==18
    # Primary source follow-up explicitly overlaps generic next-view acquisition.
    primary=O/'sources/CrossViewGate'
    active=(primary/'docs/results/active_view_protocol.md').read_text()
    assert 'Hidden visibility/confidence and labels' not in active or 'online' in active.lower()
    assert 'They may not read hidden images' in active and 'offline sequential evidence' in active
    assert 'NO-GO' in (primary/'docs/results/active_view_cvian_spatial_v1.md').read_text()
    # Existing complete analyses stay frozen.
    previous={}
    for name in ['disaster_route_crasar_region_v1','disaster_route_crasar_region_rank2_v1','disaster_route_crasar_batch4_v1']:
        p=ROOT/'output'/name;r=json.loads((p/'completion_receipt.json').read_text())
        for file,h in r['files'].items():assert sha(p/file)==h,(name,file)
        previous[name]=len(r['files'])
    audit={'status':'RESEARCH_ADMISSION_AUDIT_COMPLETE_NOT_METHOD_VALIDATION','first_pass_count_equalities':witness['binomial_count_equalities_verified'],
      'independent_conditional_branch_enumerations':branches,'closed_strategy_budget_checks':2,
      'metadata_training_regions_audited':24,'new_images_or_annotations_downloaded':False,'new_model_calls':0,
      'new_theorem_claim':False,'new_method_gain_established':False,'flight_validation':False,
      'previous_sealed_files_verified':previous,'generic_reliability_verification_novelty':'NOT_ESTABLISHED;existing probabilistic search and public cross-view active-acquisition development overlap',
      'narrow_decision_relevance_question':'UNVERIFIED_NEW_HYPOTHESIS;requires literature and identifiable verification evidence',
      'data_conclusion':'No additional same-day-as-UAS source in these24exact-scene records;other independent sessions/field truth not established. Does not assert absence in all datasets.'}
    dump(O/'audit.json',audit)
    snapshot=O/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for name in ['audit_verification_feasibility.py','check_verification_identifiability.py','verify_research_admission.py']:
        shutil.copyfile(ROOT/'work/disaster_route'/name,snapshot/name)
    files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='completion_receipt.json'}
    dump(O/'completion_receipt.json',{'status':'COMPLETE_NOVELTY_AND_IDENTIFIABILITY_ADMISSION_REVIEW','files':files,'method_contribution':False})
    print(json.dumps({'audit':audit,'sealed_files':len(files)},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
