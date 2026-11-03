"""Exact observational-equivalence witness; not a new theorem or empirical result."""
from fractions import Fraction as F
from math import comb
from audit_verification_feasibility import O,dump

def repeated_distribution(p,s):
    # Same unchanged target,independent acquisition noise conditional on target.
    return {'00':1-p+p*(1-s)**2,'01':p*(1-s)*s,'10':p*s*(1-s),'11':p*s*s}

def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Audit sealed')
    worlds={'prior_overestimates_damage':(F(1,5),F(1)),
            'prior_high_but_detector_misses':(F(4,5),F(1,4))}
    a,b=worlds.values();assert a[0]*a[1]==b[0]*b[1]==F(1,5)
    checked=0
    for n in range(1,13):
        # Every first-pass bit string has equal probability; binomial counts also match.
        for k in range(n+1):
            pa=comb(n,k)*(a[0]*a[1])**k*(1-a[0]*a[1])**(n-k)
            pb=comb(n,k)*(b[0]*b[1])**k*(1-b[0]*b[1])**(n-k)
            assert pa==pb;checked+=1
    negative_posteriors={name:p*(1-s)/(1-p*s) for name,(p,s) in worlds.items()}
    assert list(negative_posteriors.values())==[F(0),F(3,4)]
    independent={name:repeated_distribution(p,s) for name,(p,s) in worlds.items()}
    assert all(sum(v.values())==1 for v in independent.values())
    tv=sum(abs(independent[list(worlds)[0]][key]-independent[list(worlds)[1]][key]) for key in ['00','01','10','11'])/2
    assert tv==F(3,10)
    # A closed-route witness after the same initial negative at home A.
    # Unit service cost;B is distance1/2 from A,so B visit+return costs2.
    # Remaining budget2 also permits two independent additional acquisitions at A.
    # Pay only once for finding the target,not once per positive detection.
    revisit_reward={name:negative_posteriors[name]*(1-(1-s)**2) for name,(p,s) in worlds.items()}
    fresh_reward={name:p*s for name,(p,s) in worlds.items()}
    assert list(revisit_reward.values())==[F(0),F(21,64)]
    assert all(v==F(1,5) for v in fresh_reward.values())
    mix=(F(21,64)-F(1,5))/F(21,64)
    minimax_regret=F(1,5)*mix
    assert mix==F(41,105) and minimax_regret==F(41,525)
    result={'status':'EXACT_EQUIVALENCE_WITNESS_VERIFIED','assumptions':['No false positives','Unknown homogeneous target prevalence p and detector sensitivity s','Independent sites,one observation per site,no informative quality side-channel','Target static within independent revisit demonstration'],
      'worlds':{name:{'target_probability':str(p),'sensitivity':str(s),'first_pass_positive_probability':str(p*s),'target_probability_after_negative':str(negative_posteriors[name])} for name,(p,s) in worlds.items()},
      'first_pass_observation_laws_equal':True,'binomial_count_equalities_verified':checked,'sample_sizes_checked':[1,12],
      'formula':'P(Z=1)=p*s;first-pass outcomes alone identify the product,not its factors. This equality is algebraic for every sample size,not only the finite checks.',
      'new_independent_revisit_law':{name:{k:str(v) for k,v in d.items()} for name,d in independent.items()},
      'independent_revisit_total_variation':str(tv),
      'route_witness':{'remaining_budget':2,'home':'A','distance_A_B':'1/2','service_cost':1,'return_cost_counted':True,
          'feasible_nondominated_strategies':['visit B once and return','independently reacquire A up to twice,stop after first positive'],
          'reward':'Probability of discovering one genuine target;repeated positives not double-counted',
          'fresh_B_reward':{k:str(v) for k,v in fresh_reward.items()},'revisit_A_reward':{k:str(v) for k,v in revisit_reward.items()},
          'best_strategy_by_world':{list(worlds)[0]:'fresh B',list(worlds)[1]:'revisit A'},
          'best_world_blind_minimax_repeat_probability':str(mix),'minimax_regret_vs_world_known_reference':str(minimax_regret),
          'scope':'Exact two-location witness conditional on initial negative;unknown-world strategies cannot choose a different first action on identical history. No real-disaster or new-theorem claim.'},
      'duplicate_existing_image':'Deterministic repeated access supplies identical Z,not an independent sensing draw,and does not break the witness equivalence',
      'boundaries':['Known sensitivity or calibrated prevalence can remove this ambiguity','Reference anchors,informative quality features or appropriately independent repeat sensing may distinguish worlds','Conditional independence and unchanged target cannot be assumed from same-day metadata','Does not prove all real disaster data non-identifiable','Does not prove an algorithmic novelty or route gain'],
      'empirical_disaster_claim':False,'new_theorem_claim':False,'new_model_calls':0}
    dump(O/'identifiability_witness.json',result);print(result['status'],checked,'equalities,TV=',tv)

if __name__=='__main__':run()
