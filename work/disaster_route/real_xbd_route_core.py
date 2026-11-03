"""Image-grid inspection routes with real cached detector observations.

Utilities are ranking scores,not damage probabilities. Distance is in256px grid
widths;time is an explicit budget proxy,not measured flight time or GSD-derived m.
"""
import copy
import numpy as np

from route_core import Config
from route_factor_core import insertion_fill,two_opt,path_distance

POLICIES=('pre_static','post_static','online_frozen','online_neighbor')
BUDGETS=(120,240,360)
HOME=16
MOVE_S_PER_GRID=20.
OBSERVE_S=20.
PLAN_S=1.


def xy_grid():
    return np.array([[i%4+.5,3.5-i//4] for i in range(16)]+[[2.,0.]],float)


def midrank(values):
    a=np.asarray(values,float)
    if a.shape!=(16,) or not np.isfinite(a).all():raise ValueError('16 finite public features required')
    return np.array([(sum(a<x)+(sum(a==x)-1)/2)/15 for x in a])


def public_features(pre,post):
    if any(x.shape!=(64,64,3) or x.dtype!=np.uint8 for x in (pre,post)):raise ValueError('Only public64RGB inputs allowed')
    gray=pre.astype(float).mean(2)/255;dy,dx=np.gradient(gray);edges=np.hypot(dx,dy)
    change=np.abs(pre.astype(float)-post.astype(float)).mean(2)/255
    edge=np.array([edges[i//4*16:(i//4+1)*16,i%4*16:(i%4+1)*16].mean() for i in range(16)])
    cue=np.array([change[i//4*16:(i//4+1)*16,i%4*16:(i%4+1)*16].mean() for i in range(16)])
    return {'pre_texture':edge.tolist(),'post_change':cue.tolist(),
            'pre_utility':(.05+.95*midrank(edge)).tolist(),
            'post_utility':(.05+.95*(midrank(edge)+midrank(cue))/2).tolist(),
            'meaning':'Texture/change rank utilities,not building exposure labels or calibrated damage probabilities.'}


def utility_from_evidence(base,history):
    """Untuned four-neighbor baseline;only completed paid predictions enter it."""
    out=np.asarray(base,float).copy()
    for j in range(16):
        neighbors=[h for h in history if abs(h['tile_id']//4-j//4)+abs(h['tile_id']%4-j%4)==1]
        if neighbors:out[j]=(out[j]+sum(h['predicted_severe_pixels']>0 for h in neighbors))/(1+len(neighbors))
    return out


def weighted_route(position,visited,elapsed,weights,budget):
    xy=xy_grid();distances=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    rewards=np.r_[weights,0.];used=np.zeros(17,bool);used[HOME]=True;used[visited]=True
    config=Config(budget_s=budget,speed_m_s=1/MOVE_S_PER_GRID,observe_s=OBSERVE_S,planning_latency_s=PLAN_S,max_visits=16)
    seed=[];pos=position;clock=elapsed;mask=used.copy()
    while True:
        valid=[i for i in range(16) if not mask[i] and clock+distances[pos,i]*MOVE_S_PER_GRID+OBSERVE_S+distances[i,HOME]*MOVE_S_PER_GRID<=budget+1e-9]
        if not valid:break
        i=min(valid,key=lambda i:(-rewards[i]/(distances[pos,i]*MOVE_S_PER_GRID+OBSERVE_S),i))
        clock+=distances[pos,i]*MOVE_S_PER_GRID+OBSERVE_S;seed.append(i);mask[i]=True;pos=i
    candidates=[]
    for initial in (seed,[]):
        route=insertion_fill(distances,position,HOME,used,rewards,elapsed,config,0.,16-len(visited),initial)
        route=two_opt(distances,position,HOME,route)
        route=insertion_fill(distances,position,HOME,used,rewards,elapsed,config,0.,16-len(visited),route)
        candidates.append(route)
    return min(candidates,key=lambda r:(-float(rewards[r].sum()),path_distance(distances,position,HOME,r),tuple(r)))


def execute(features,policy,budget,observe):
    if policy not in POLICIES or budget<=2:raise ValueError('Invalid route configuration')
    xy=xy_grid();state={'policy':policy,'budget_s_proxy':budget,'elapsed_s_proxy':0.,'distance_grid':0.,
        'position_index':HOME,'visited':[],'remaining_route':[],'history':[],'planning_events':[],'closed':False}
    frames=[];phase='pre'
    def weights():
        base=features['pre_utility'] if phase=='pre' else features['post_utility']
        return utility_from_evidence(base,state['history']) if policy=='online_neighbor' and phase=='post' else np.array(base)
    def capture(reason):
        frames.append({'reason':reason,'phase':phase,'observations':len(state['history']),
            **{k:copy.deepcopy(state[k]) for k in ('position_index','elapsed_s_proxy','distance_grid','remaining_route','closed')},
            'planning_calls':len(state['planning_events']),'utility':weights().tolist()})
    def plan(reason):
        reserve=np.linalg.norm(xy[state['position_index']]-xy[HOME])*MOVE_S_PER_GRID
        if state['elapsed_s_proxy']+PLAN_S+reserve>budget+1e-8:raise ValueError('No plan/return reserve')
        before=state['elapsed_s_proxy'];state['elapsed_s_proxy']+=PLAN_S;utility=weights()
        route=weighted_route(state['position_index'],state['visited'],state['elapsed_s_proxy'],utility,budget)
        state['remaining_route']=route
        state['planning_events'].append({'reason':reason,'after_observations':len(state['history']),
            'elapsed_before_s_proxy':before,'elapsed_after_s_proxy':state['elapsed_s_proxy'],
            'position_index':state['position_index'],'utility':utility.tolist(),'planned_route':list(route)})
        capture(reason)
    capture('initialized');plan('pre_plan');phase='post';capture('post_cues_received')
    if policy!='pre_static':plan('post_plan')
    while state['remaining_route']:
        i=state['remaining_route'].pop(0);before=state['elapsed_s_proxy']
        leg=float(np.linalg.norm(xy[i]-xy[state['position_index']]))
        arrival=before+leg*MOVE_S_PER_GRID;end=arrival+OBSERVE_S
        if end+np.linalg.norm(xy[i]-xy[HOME])*MOVE_S_PER_GRID>budget+1e-8:raise ValueError('No paid observation/return reserve')
        if i in state['visited']:raise ValueError('Repeated observation')
        state['elapsed_s_proxy']=end;state['distance_grid']+=leg;state['position_index']=i
        # Cached model output is revealed here only,after travel and observation.
        pixels=int(observe(i,end))
        if not 0<=pixels<=256*256:raise ValueError('Invalid detector observation')
        state['visited'].append(i);state['history'].append({'step':len(state['history'])+1,'tile_id':i,
            'elapsed_before_s_proxy':before,'leg_grid':leg,'arrival_s_proxy':arrival,
            'observation_complete_s_proxy':end,'predicted_severe_pixels':pixels})
        capture('paid_observation')
        if policy in ('online_frozen','online_neighbor'):
            feasible=any(j not in state['visited'] and end+PLAN_S+np.linalg.norm(xy[j]-xy[i])*MOVE_S_PER_GRID+OBSERVE_S
                +np.linalg.norm(xy[j]-xy[HOME])*MOVE_S_PER_GRID<=budget+1e-9 for j in range(16))
            if feasible:plan('online_replan')
    back=float(np.linalg.norm(xy[state['position_index']]-xy[HOME]));state['distance_grid']+=back
    state['elapsed_s_proxy']+=back*MOVE_S_PER_GRID;state['position_index']=HOME;state['closed']=True;state['return_grid']=back
    if state['elapsed_s_proxy']>budget+1e-8:raise ValueError('Budget violation')
    capture('returned');return {'state':state,'frames':frames}


def evaluate(route,tile_counts):
    """Evaluator-only published-reference scores;never supplied to a planner."""
    s=route['state'];history=s['history'];chosen=s['visited'];reference=sum(c['reference'] for c in tile_counts)
    tp=sum(tile_counts[i]['tp'] for i in chosen);fp=sum(tile_counts[i]['fp'] for i in chosen)
    arrived=[h for h in history if tile_counts[h['tile_id']]['reference']>0]
    detected=[h for h in history if tile_counts[h['tile_id']]['tp']>0]
    return {'reference_severe_pixels':reference,'covered_reference_pixels':sum(tile_counts[i]['reference'] for i in chosen),
        'true_positive_pixels':tp,'false_positive_pixels':fp,'recall':tp/reference if reference else None,
        'precision':tp/(tp+fp) if tp+fp else None,'coverage_recall':sum(tile_counts[i]['reference'] for i in chosen)/reference if reference else None,
        'first_target_arrival_s_proxy':arrived[0]['arrival_s_proxy'] if arrived else None,
        'first_correct_detection_s_proxy':detected[0]['observation_complete_s_proxy'] if detected else None,
        'first_detection_or_budget_s_proxy':detected[0]['observation_complete_s_proxy'] if detected else s['budget_s_proxy'],
        'no_correct_detection':not bool(detected),'observations':len(chosen),'planning_calls':len(s['planning_events']),
        'total_distance_grid':s['distance_grid'],'total_time_s_proxy':s['elapsed_s_proxy'],
        'all_tiles_detector_tp_ceiling':sum(c['tp'] for c in tile_counts)}
