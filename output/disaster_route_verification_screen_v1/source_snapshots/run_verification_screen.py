"""Freeze protocol,execute finite exact screening and independently verify it."""
import json,hashlib,shutil,subprocess,sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from verification_screen_core import solve,MODELS

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'work/disaster_route'
O=ROOT/'output/disaster_route_verification_screen_v1'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,F):return {'exact':str(x),'value':float(x)}
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x

def replay(policy,world,rho,c,q):
    """Independent target/noise/probe enumeration;planner does not see these states."""
    p,s=MODELS[world];a_prob=p*(1-s)/(1-p*s)
    mass=F(0);value=F(0);branches=0;maxcost=F(0)
    for h in range(2):
        ph=(q if h==world else 1-q) if policy['probe'] else F(int(h==0))
        if not ph:continue
        kind,m=policy['tails'][h];fee=1+(c+F(1,4) if policy['probe'] else 0)
        if kind=='stop':mass+=ph;maxcost=max(maxcost,fee);branches+=1;continue
        if kind=='B':
            for target,detector in product([0,1],repeat=2):
                prob=ph*(p if target else 1-p)*(s if detector else 1-s)
                mass+=prob;value+=prob*int(bool(target and detector));branches+=1
            maxcost=max(maxcost,fee+F(1,2)+1+F(1,2));continue
        for sticky in range(2):
            pmode=rho if sticky else 1-rho
            for target in range(2):
                pt=a_prob if target else 1-a_prob
                for detections in product([0,1],repeat=m):
                    pd=F(int(not any(detections))) if sticky else F(1)
                    if not sticky:
                        for y in detections:pd*=s if y else 1-s
                    prob=ph*pmode*pt*pd;mass+=prob
                    value+=prob*int(bool(target and any(detections)));branches+=1
                    used=next((i+1 for i,y in enumerate(detections) if target and y),m)
                    if prob:maxcost=max(maxcost,fee+used)
    assert mass==1
    return value,maxcost,branches

def check_lp(policies,oracle,exact):
    vals=np.array([[float(p['values'][m]) for p in policies] for m in range(2)])
    # oracle-value <= regret;nonnegative mixture sums1,regret nonnegative.
    objective=np.r_[np.zeros(len(policies)),1.]
    r=linprog(objective,A_ub=np.column_stack((-vals,-np.ones(2))),b_ub=-np.array(list(map(float,oracle))),
        A_eq=np.array([np.r_[np.ones(len(policies)),0.]]),b_eq=[1.],bounds=[(0,None)]*(len(policies)+1),method='highs')
    assert r.success and abs(r.fun-float(exact['worst_regret']))<1e-8

def run():
    O.mkdir(exist_ok=True,parents=True)
    if (O/'completion_receipt.json').exists():raise RuntimeError('Already sealed')
    protocol=HERE/'verification_problem_v1.md'
    frozen=O/'problem_protocol.md'
    if frozen.exists():assert sha(frozen)==sha(protocol)
    else:shutil.copyfile(protocol,frozen)
    dump(O/'protocol.json',{'status':'FROZEN_BEFORE_COMPUTATION','problem_sha256':sha(frozen),'grid':{'budget':[3,4,5],'rho':['0','1/2','1'],'probe_cost':['0','1/2','1','2'],'probe_accuracy':['1/2','4/5','1']},
      'primary':[4,'0','0','1'],'new_method_claim':False,'real_observation_model':False,'scope':'Exact within two-tail,at-most-one-calibration policy family;not unrestricted POMDP optimality'})
    results=[];branches=lpchecks=common=equal=benefit=always_harm=noinfo=sticky=0;primary=None
    for budget,rho,c,q in product([3,4,5],[F(0),F(1,2),F(1)],[F(0),F(1,2),F(1),F(2)],[F(1,2),F(4,5),F(1)]):
        r=solve(budget,rho,c,q);base=r['policies']['static_mixture'];bayes=r['policies']['exact_bayes_reference'];candidate=r['policies']['disagreement_screen_candidate']
        assert candidate['values']==bayes['values'];equal+=1;common+=r['common_optimal_static']
        def mean(p):return sum(p['values'])/2
        gain=mean(bayes)-mean(base);assert gain>=0;benefit+=gain>0
        always_harm+=mean(r['policies']['always_probe'])<mean(base)
        if q==F(1,2):assert gain==0;noinfo+=1
        if rho==1:
            assert all(p['values']==(F(0),F(0)) for p in r['static_strategies'] if p['tails'][0][0]=='A');sticky+=1
        for policy in r['static_strategies']+r['probe_strategies']:
            for world in range(2):
                observed,cost,count=replay(policy,world,rho,c,q);branches+=count
                assert observed==policy['values'][world] and cost<=budget and policy['cost_cap']<=budget
        for key,ps in [('robust_no_probe',r['static_strategies']),('robust_with_optional_probe',r['static_strategies']+r['probe_strategies'])]:
            check_lp(ps,r['oracle'],r[key]);lpchecks+=1
        assert r['robust_with_optional_probe']['worst_regret']<=r['robust_no_probe']['worst_regret']
        r['bayes_gain_vs_static']=gain
        if (budget,rho,c,q)==(4,F(0),F(0),F(1)):primary=r
        results.append(serial(r))
    assert primary is not None and len(results)==108
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_verification_screen.py','-v'],capture_output=True,text=True)
    (O/'tests.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    summary={'status':'COMPLETE_MECHANISM_SCREEN_NO_DISTINCT_METHOD','conditions':108,'candidate_matches_exact_bayes_conditions':equal,
      'common_static_optimum_conditions':common,'paid_calibration_improves_mixture_reward_conditions':benefit,
      'always_calibrate_worse_than_static_conditions':always_harm,'uninformative_probe_zero_gain_conditions':noinfo,
      'fully_correlated_revisit_zero_new_detection_conditions':sticky,'independent_branch_enumerations':branches,
      'independent_numerical_lp_checks':lpchecks,'tests_passed':6,
      'interpretation':'Decision disagreement is real,but this screen is only a computationally equivalent writing of standard Bayesian value-of-information within the frozen policy class. No independent algorithmic gain established. Grid counts are not disaster frequencies.',
      'decision':'STOP_GENERIC_DISAGREEMENT_GATE_AS_NEW_METHOD','new_theorem_claim':False,'real_disaster_efficacy':False,
      'new_model_calls':0,'gpu_rental':False,'primary':serial(primary)}
    dump(O/'results.json',results);dump(O/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='primary'},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
