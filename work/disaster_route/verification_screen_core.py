"""Exact restricted decision experiment,not a new route-planning method."""
from fractions import Fraction as F
from itertools import product,combinations

MODELS=((F(1,5),F(1)),(F(4,5),F(1,4)))
EXTRA_PLAN=F(1,4)

def payoff(action,world,rho):
    p,s=MODELS[world]
    if action[0]=='stop':return F(0)
    if action[0]=='B':return p*s
    posterior=p*(1-s)/(1-p*s)
    return posterior*(1-rho)*(1-(1-s)**action[1])

def tail_cost(action):
    return F(0) if action[0]=='stop' else F(2) if action[0]=='B' else F(action[1])

def actions(remaining):
    if remaining<0:return []
    out=[('stop',0)]
    if remaining>=1:out.append(('A',int(remaining)))
    if remaining>=2:out.append(('B',1))
    return out

def strategies(budget,c,q,rho):
    remaining=F(budget)-1
    static=[{'name':f'{a[0]}{a[1]}','probe':False,'tails':(a,a),'cost_cap':1+tail_cost(a),
             'values':tuple(payoff(a,m,rho) for m in range(2))} for a in actions(remaining)]
    probe=[]
    for tails in product(actions(remaining-c-EXTRA_PLAN),repeat=2):
        vals=tuple(sum((q if h==m else 1-q)*payoff(tails[h],m,rho) for h in range(2)) for m in range(2))
        probe.append({'name':f'Q:{tails[0]}|{tails[1]}','probe':True,'tails':tails,
                      'cost_cap':1+c+EXTRA_PLAN+max(map(tail_cost,tails)),'values':vals})
    return static,probe

def best(policies,weights=(F(1,2),F(1,2))):
    return min(policies,key=lambda p:(-sum(w*v for w,v in zip(weights,p['values'])),p['cost_cap'],p['name']))

def robust_mix(policies,oracle):
    candidates=[]
    def add(i,j,t):
        values=tuple(t*policies[i]['values'][k]+(1-t)*policies[j]['values'][k] for k in range(2))
        regret=max(oracle[k]-values[k] for k in range(2))
        candidates.append((regret,i,j,t,values))
    for i in range(len(policies)):add(i,i,F(1))
    for i,j in combinations(range(len(policies)),2):
        a,b=policies[i]['values'],policies[j]['values']
        denom=(a[0]-b[0])-(a[1]-b[1])
        if not denom:continue
        t=(oracle[0]-oracle[1]-b[0]+b[1])/denom
        if 0<=t<=1:add(i,j,t)
    regret,i,j,t,values=min(candidates,key=lambda z:(z[0],z[1],z[2],z[3]))
    return {'kind':'generic_minimax_regret','components':[(policies[i],t),(policies[j],1-t)],
            'values':values,'worst_regret':regret,'probe_probability':t*policies[i]['probe']+(1-t)*policies[j]['probe'],
            'cost_cap':max(policies[i]['cost_cap'] if t else F(0),policies[j]['cost_cap'] if t<1 else F(0))}

def solve(budget,rho,c,q):
    static,probe=strategies(budget,c,q,rho);allp=static+probe
    oracle=tuple(max(p['values'][m] for p in allp) for m in range(2))
    common=[p for p in static if all(p['values'][m]==max(s['values'][m] for s in static) for m in range(2))]
    bayes=best(allp)
    candidate=best(common) if common else bayes
    policies={'static_mixture':best(static),'fixed_model0':best(allp,(F(1),F(0))),
      'fixed_model1':best(allp,(F(0),F(1))),'always_probe':best(probe) if probe else best(static),
      'exact_bayes_reference':bayes,'disagreement_screen_candidate':candidate}
    return {'budget':budget,'rho':rho,'c':c,'q':q,'common_optimal_static':bool(common),'oracle':oracle,
            'policies':policies,'static_strategies':static,'probe_strategies':probe,
            'robust_no_probe':robust_mix(static,oracle),'robust_with_optional_probe':robust_mix(allp,oracle)}
