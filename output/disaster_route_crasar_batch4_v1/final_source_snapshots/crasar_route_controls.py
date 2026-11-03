"""Budget-complete controls for published image-window replay, not flight control.

Only public arrays enter planning. Reference labels are evaluated elsewhere.
Coverage is a disjoint-window fraction of each building, so repeat observations
cannot earn a second reward. No calibration or feedback propagation is implied.
"""
import time
import numpy as np
from route_headroom_core import exact_catalogue,reconstruct

def validate(xy,coverage,damage,uncertainty,speed,observe,plan,budget):
    xy=np.asarray(xy,float);c=np.asarray(coverage,float)
    d=np.asarray(damage,float);u=np.asarray(uncertainty,float)
    n=len(xy)-1
    if xy.shape!=(n+1,2) or c.ndim!=2 or c.shape[0]!=n or d.shape!=(c.shape[1],) or u.shape!=d.shape:
        raise ValueError('Nodes,window-building coverage and building score shapes must match')
    if any(not np.isfinite(a).all() for a in [xy,c,d,u]):raise ValueError('Finite public inputs required')
    if np.any((c<0)|(c>1)) or np.any(c.sum(axis=0)>1+1e-7) or np.any((d<0)|(d>1)) or np.any((u<0)|(u>1)):
        raise ValueError('Disjoint coverage fractions and unit-interval score proxies required')
    if speed<=0 or min(observe,plan,budget)<0:raise ValueError('Invalid budget contract')
    if budget<plan:raise ValueError('Budget cannot pay initial planning charge')
    return xy,c,d,u

def coverage_utility(fraction,weight):
    # Continuous effective coverage: sensitivity to incomplete observations stays visible.
    return float(np.dot(np.minimum(1.,fraction),weight))

def replay(path,xy,c,weight,speed,observe,plan,budget):
    n=len(xy)-1;current=n;elapsed=plan;distance=0.;seen=np.zeros(c.shape[1]);events=[]
    if len(set(path))!=len(path) or any(not 0<=j<n for j in path):raise ValueError('Unique valid windows required')
    for j in path:
        leg=float(np.linalg.norm(xy[current]-xy[j]));elapsed+=leg/speed+observe;distance+=leg
        seen+=c[j];events.append({'window':int(j),'arrival_proxy_s':float(elapsed-observe),
            'observation_complete_proxy_s':float(elapsed),'utility':coverage_utility(seen,weight)})
        current=j
    back=float(np.linalg.norm(xy[current]-xy[n]));distance+=back;elapsed+=back/speed
    if elapsed>budget+1e-7:raise AssertionError('Closed route exceeds budget')
    return {'route':list(map(int,path)),'distance_m':distance,'elapsed_proxy_s':elapsed,'observation_count':len(path),
        'events':events,'public_utility_proxy':coverage_utility(seen,weight),'returned_home':True,
        'planning_charge_proxy_s':plan,'observed_building_fraction':seen.tolist()}

def plan_route(policy,xy,coverage,damage,uncertainty,*,speed,observe,plan,budget,catalogue=None):
    xy,c,d,u=validate(xy,coverage,damage,uncertainty,speed,observe,plan,budget)
    n=len(xy)-1;start=time.perf_counter()
    # This fixed product is a heuristic proxy, never claimed to estimate true information gain.
    weights=d*u
    if policy=='static_exact':
        cat=catalogue if catalogue is not None else exact_catalogue(xy,n)
        costs=cat['closed']/speed+cat['cardinality']*observe+plan
        feasible=np.flatnonzero(costs<=budget+1e-9)
        # Windows are disjoint, hence coverage utility is additive and Held-Karp applies.
        from route_headroom_core import subset_rewards
        reward=subset_rewards(c@weights);best=reward[feasible].max()
        ties=feasible[np.isclose(reward[feasible],best,atol=1e-10,rtol=0)]
        mask=int(min(ties,key=lambda m:(cat['closed'][m],int(m))))
        path=reconstruct(cat,mask)
        certificate={'solver':'Held-Karp all subsets','enumerated_subsets':len(cat['closed']),
            'feasible_subsets':len(feasible),'objective_upper_bound':float(best),'objective_gap':0.}
    else:
        if policy not in ['distance','damage','uncertainty','value_cost_proxy']:raise ValueError('Unknown policy')
        current=n;elapsed=plan;seen=np.zeros(c.shape[1]);path=[]
        while True:
            options=[]
            for j in range(n):
                if j in path or c[j].sum()==0:continue
                leg=float(np.linalg.norm(xy[current]-xy[j]));back=float(np.linalg.norm(xy[j]-xy[n]))
                if elapsed+leg/speed+observe+back/speed>budget+1e-9:continue
                if policy=='distance':priority=-leg
                elif policy=='damage':priority=float(c[j]@d)
                elif policy=='uncertainty':priority=float(c[j]@u)
                else:
                    # Incremental closed-route cost includes changed return, not just outbound motion.
                    oldreturn=float(np.linalg.norm(xy[current]-xy[n]))
                    increment=(leg+back-oldreturn)/speed+observe
                    gain=coverage_utility(seen+c[j],weights)-coverage_utility(seen,weights)
                    priority=gain/max(increment,1e-12)
                    if gain<=1e-12:continue
                options.append((-priority,leg,j))
            if not options:break
            _,leg,j=min(options);elapsed+=leg/speed+observe;path.append(j);seen+=c[j];current=j
        certificate={'solver':'greedy feasible insertion','objective_gap':None}
    result=replay(path,xy,c,weights,speed,observe,plan,budget)
    result.update(policy=policy,solver_wall_seconds=time.perf_counter()-start,certificate=certificate,
        utility_scope='damage score × uncertainty score × newly accessed footprint fraction; uncalibrated proxy',
        flight_cost_measured=False)
    return result
