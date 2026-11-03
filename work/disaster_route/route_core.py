"""Budget-feasible geometric inspection routes and a fixed noisy observation model.

The planner receives public geometry and beliefs, never targets or future signals.
These are planar inspection routes, not obstacle/terrain-validated flight paths.
"""
from dataclasses import dataclass
import math
import numpy as np

POLICIES=('pre_static','post_static','online','online_frozen')
NAMES={'pre_static':'灾前均匀先验静态','post_static':'灾后线索一次性规划',
       'online':'观测反馈在线重规划','online_frozen':'在线重算但不更新信念'}
WORLD_MODES=('aligned','reversed','independent')
NOISE={'perfect':(1.0,0.0),'moderate':(.8,.1),'poor':(.6,.25)}
HOT_TARGET_RATE=.6
COLD_TARGET_RATE=.02


@dataclass(frozen=True)
class Config:
    budget_s:float=1200
    speed_m_s:float=5
    observe_s:float=20
    planning_latency_s:float=1
    max_visits:int=30
    def __post_init__(self):
        values=[self.budget_s,self.speed_m_s,self.observe_s,self.planning_latency_s]
        if any(not math.isfinite(x) or x<=0 for x in values) or type(self.max_visits) is not int or self.max_visits<1:
            raise ValueError('Positive finite costs and visit cap required')


@dataclass(frozen=True)
class PublicMap:
    region:str
    ids:tuple
    xy:np.ndarray
    lonlat:np.ndarray
    blocks:np.ndarray
    block_ids:tuple
    prior_pre:np.ndarray
    prior_post:np.ndarray
    cue:np.ndarray
    start_index:int


def public_map(region,cells):
    allowed={'cell_id','area_m2','row','col','centroid_easting_m','centroid_northing_m','centroid_lon','centroid_lat','change_cue_score'}
    # Explicit public projection prevents labels/properties from becoming planner inputs.
    rows=[{k:r[k] for k in allowed} for r in cells if r['area_m2']>=1800]
    rows.sort(key=lambda r:r['cell_id'])
    if not rows or len({r['cell_id'] for r in rows})!=len(rows):raise ValueError('Unique eligible public cells required')
    xy=np.array([[r['centroid_easting_m'],r['centroid_northing_m']] for r in rows],np.float64)
    lonlat=np.array([[r['centroid_lon'],r['centroid_lat']] for r in rows],np.float64)
    if not np.isfinite(xy).all() or not np.isfinite(lonlat).all():raise ValueError('Finite metric and geographic positions required')
    block_keys=[(r['row']//3,r['col']//3) for r in rows];keys=tuple(sorted(set(block_keys)))
    block=np.array([keys.index(k) for k in block_keys],np.int64)
    cue=np.array([r['change_cue_score'] if r['change_cue_score'] is not None else np.nan for r in rows],np.float64)
    if np.any(np.isfinite(cue)&((cue<0)|(cue>1))):raise ValueError('Cue outside original[0,1] range')
    # Rank is an explicitly synthetic prior construction, not empirical calibration.
    scores=[]
    for j in range(len(keys)):
        values=cue[(block==j)&np.isfinite(cue)];scores.append(float(np.mean(values)) if len(values) else None)
    post=np.full(len(keys),.2,np.float64);known=sorted([j for j,v in enumerate(scores) if v is not None],key=lambda j:(scores[j],keys[j]))
    for j in known:
        low=sum(scores[k]<scores[j] for k in known);ties=sum(scores[k]==scores[j] for k in known)
        percentile=(low+(ties-1)/2)/(len(known)-1) if len(known)>1 else .5
        post[j]=.05+.3*percentile
    pre=np.full(len(keys),.2,np.float64)
    origin=np.array([(xy[:,0].min()+xy[:,0].max())/2,xy[:,1].min()])
    start=min(range(len(rows)),key=lambda i:(float(np.linalg.norm(xy[i]-origin)),rows[i]['cell_id']))
    for a in [xy,lonlat,block,pre,post,cue]:a.flags.writeable=False
    return PublicMap(region,tuple(r['cell_id'] for r in rows),xy,lonlat,block,keys,pre,post,cue,start)


def greedy_route(geometry,position,visited,elapsed,belief,config,latency,visit_limit=None):
    """One common reward-per-leg-time heuristic, with a return reserve at every step.

    latency is the per-future-observation planning cost of this execution mode.
    The returned suffix is feasible including all its stops, latency and return.
    """
    n=len(geometry.ids)
    if visited.shape!=(n,) or visited.dtype!=bool or belief.shape!=(len(geometry.block_ids),):raise ValueError('Invalid public planning state')
    if not np.isfinite(belief).all() or np.any((belief<0)|(belief>1)):raise ValueError('Invalid belief')
    if elapsed<0 or latency<0 or not math.isfinite(elapsed):raise ValueError('Invalid elapsed time or latency')
    limit=config.max_visits if visit_limit is None else visit_limit
    if type(limit) is not int or limit<0:raise ValueError('Bad suffix visit limit')
    position=np.asarray(position,np.float64).copy();used=visited.copy();time=float(elapsed);route=[]
    home=geometry.xy[geometry.start_index]
    returns=np.linalg.norm(geometry.xy-home,axis=1)/config.speed_m_s
    rewards=COLD_TARGET_RATE+(HOT_TARGET_RATE-COLD_TARGET_RATE)*belief[geometry.blocks]
    for _ in range(limit):
        move=np.linalg.norm(geometry.xy-position,axis=1)/config.speed_m_s
        action=move+config.observe_s+latency
        feasible=(~used)&(time+action+returns<=config.budget_s+1e-9)
        if not feasible.any():break
        utility=np.where(feasible,rewards/action,-np.inf)
        # ID order is fixed on PublicMap; numpy first-argmax supplies deterministic ties.
        index=int(np.argmax(utility));time+=float(action[index]);position=geometry.xy[index].copy()
        used[index]=True;route.append(index)
    if time+float(np.linalg.norm(position-home))/config.speed_m_s>config.budget_s+1e-8:raise ValueError('Route cannot return within budget')
    return route


def update_belief(prior,positive,sensitivity,false_positive):
    if not 0<=prior<=1 or not isinstance(positive,(bool,np.bool_)) or not 0<=false_positive<sensitivity<=1:raise ValueError('Invalid fixed observation likelihood')
    a=false_positive+(sensitivity-false_positive)*HOT_TARGET_RATE
    b=false_positive+(sensitivity-false_positive)*COLD_TARGET_RATE
    lh_hot=a if positive else 1-a;lh_cold=b if positive else 1-b
    den=prior*lh_hot+(1-prior)*lh_cold
    if den==0:raise ValueError('Impossible observation')
    return prior*lh_hot/den


def simulate_targets(geometry,mode,seed):
    if mode not in WORLD_MODES or type(seed) is not int:raise ValueError('Unknown frozen simulation condition')
    rng=np.random.default_rng(seed);post=geometry.prior_post.copy()
    if mode=='aligned':prob=post
    elif mode=='reversed':
        # Preserve the same pool of block probabilities, assign in opposite cue order.
        order=np.argsort(post,kind='stable');prob=np.empty_like(post);prob[order]=post[order[::-1]]
    else:prob=np.full_like(post,.2)
    latent=rng.random(len(prob))<prob
    rates=np.where(latent[geometry.blocks],HOT_TARGET_RATE,COLD_TARGET_RATE)
    targets=rng.random(len(geometry.ids))<rates
    return {'targets':targets,'hot_blocks':latent,'true_hot_probabilities':prob,'uniform_observation_draws':rng.random(len(targets))}


def signals(world,noise):
    if noise not in NOISE:raise ValueError('Unknown fixed noise scenario')
    sensitivity,fp=NOISE[noise];p=np.where(world['targets'],sensitivity,fp)
    return world['uniform_observation_draws']<p


def execute(geometry,policy,config,observe):
    """The sole environment interface is observe(index) after paid travel+inspection.

    Static policies receive the same observations but do not use them to change route.
    observe() returns one bit, never hidden target labels or unvisited signals.
    """
    import time as clock
    if policy not in POLICIES:raise ValueError('Unknown route policy')
    belief=geometry.prior_pre.copy() if policy=='pre_static' else geometry.prior_post.copy()
    visited=np.zeros(len(geometry.ids),bool);home=geometry.xy[geometry.start_index];position=home.copy()
    elapsed=0.;travel=0.;planning_calls=0;planning_wall=0.;events=[];steps=[];static=[]
    def plan(latency,limit,initial=False):
        nonlocal planning_calls,planning_wall
        t=clock.perf_counter();answer=greedy_route(geometry,position,visited,elapsed,belief,config,latency,limit)
        planning_wall+=clock.perf_counter()-t;planning_calls+=1
        return answer
    if policy in ('pre_static','post_static'):
        if config.planning_latency_s+np.linalg.norm(position-home)/config.speed_m_s<=config.budget_s:
            elapsed+=config.planning_latency_s;static=plan(0,config.max_visits,True)
            events.append({'after_observations':0,'at_cell':geometry.ids[geometry.start_index],'plan_latency_s':config.planning_latency_s,
                'remaining_budget_s':config.budget_s-elapsed,'first_five_suffix_ids':[geometry.ids[i] for i in static[:5]],'planned_suffix_visits':len(static)})
    while len(steps)<config.max_visits:
        if policy in ('pre_static','post_static'):
            if len(steps)>=len(static):break
            index=static[len(steps)];latency=0.
        else:
            # No paid final replan if even the next inspection+return is impossible.
            move=np.linalg.norm(geometry.xy-position,axis=1)/config.speed_m_s
            reserve=np.linalg.norm(geometry.xy-home,axis=1)/config.speed_m_s
            feasible=(~visited)&(elapsed+config.planning_latency_s+move+config.observe_s+reserve<=config.budget_s+1e-9)
            if not feasible.any():break
            suffix=plan(config.planning_latency_s,config.max_visits-len(steps))
            if not suffix:raise ValueError('Feasibility and suffix planner disagree')
            index=suffix[0];latency=config.planning_latency_s
            events.append({'after_observations':len(steps),'at_cell':geometry.ids[geometry.start_index] if not steps else steps[-1]['cell_id'],
                'plan_latency_s':latency,'remaining_budget_s':config.budget_s-elapsed-latency,
                'first_five_suffix_ids':[geometry.ids[i] for i in suffix[:5]],'planned_suffix_visits':len(suffix)})
        before=elapsed;elapsed+=latency;leg=float(np.linalg.norm(geometry.xy[index]-position));arrival=elapsed+leg/config.speed_m_s
        completion=arrival+config.observe_s
        if completion+float(np.linalg.norm(geometry.xy[index]-home))/config.speed_m_s>config.budget_s+1e-8:raise ValueError('No return reserve')
        # observe is called only here, after explicitly charging movement and observation.
        positive=bool(observe(index));block=int(geometry.blocks[index]);prior=float(belief[block])
        if policy=='online':belief[block]=update_belief(prior,positive,*observe.likelihood)
        elapsed=completion;travel+=leg;visited[index]=True;position=geometry.xy[index].copy()
        steps.append({'step':len(steps)+1,'cell_index':index,'cell_id':geometry.ids[index],'block_index':block,
            'elapsed_before_step_s':before,'planning_latency_s':latency,'leg_m':leg,'arrival_s':arrival,'observation_complete_s':completion,
            'cumulative_travel_m':travel,'return_reserve_s':float(np.linalg.norm(position-home))/config.speed_m_s,
            'observed_positive':positive,'block_belief_before':prior,'block_belief_after':float(belief[block])})
    return_m=float(np.linalg.norm(position-home));total=elapsed+return_m/config.speed_m_s
    if total>config.budget_s+1e-8:raise ValueError('Closed tour over budget')
    return {'policy':policy,'name':NAMES[policy],'region':geometry.region,'start_id':geometry.ids[geometry.start_index],
        'budget_s':config.budget_s,'visits':len(steps),'steps':steps,'planning_events':events,'planning_calls':planning_calls,
        'simulated_planning_latency_s':config.planning_latency_s*planning_calls,'measured_planning_wall_s':planning_wall,
        'return_m':return_m,'total_distance_m':travel+return_m,'total_time_s':total,
        'closed_route_ids':[geometry.ids[geometry.start_index]]+[s['cell_id'] for s in steps]+[geometry.ids[geometry.start_index]],
        'movement_seconds':(travel+return_m)/config.speed_m_s,'observation_seconds':len(steps)*config.observe_s}


class Observer:
    def __init__(self,observations,likelihood):self._observations=observations;self.likelihood=likelihood;self.accessed=[]
    def __call__(self,index):
        if index in self.accessed:raise ValueError('Repeated observation')
        self.accessed.append(index);return bool(self._observations[index])


def evaluate(route,targets):
    if targets.dtype!=bool:raise ValueError('Boolean synthetic target truth required')
    visits=[s['cell_index'] for s in route['steps']]
    if len(visits)!=len(set(visits)):raise ValueError('Duplicate inspection')
    correct=[s for s in route['steps'] if targets[s['cell_index']] and s['observed_positive']]
    arrived=[s for s in route['steps'] if targets[s['cell_index']]]
    fp=sum(not targets[s['cell_index']] and s['observed_positive'] for s in route['steps'])
    total=int(targets.sum())
    return {'true_target_cells':total,'visited_target_cells':len(arrived),'detected_target_cells':len(correct),'false_positive_observations':int(fp),
        'visited_target_recall':len(arrived)/total if total else None,'detected_target_recall':len(correct)/total if total else None,
        'first_target_arrival_s':arrived[0]['arrival_s'] if arrived else None,
        'first_correct_detection_s':correct[0]['observation_complete_s'] if correct else None,
        'first_detection_or_budget_s':correct[0]['observation_complete_s'] if correct else route['budget_s'],
        'first_detection_censored':not bool(correct)}
