"""Serializable inspection state with evaluator-private deterministic observations.

This is an interface boundary for fixed Python policies, not an adversarial sandbox.
The planner receives geometry and a copied public state, never a world or snapshot.
"""
import copy
import hashlib
import json

import numpy as np

from route_core import Config, update_belief

PUBLIC_KEYS = {'region','budget_s','speed_m_s','observe_s','planning_latency_s','position_index',
               'elapsed_s','distance_m','visited_indices','belief_hot','remaining_route',
               'planning_events','history','closed','return_m'}
BRANCHES = ('continue','replan_feedback','replan_prior')


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pack(public, environment):
    payload={'schema':1,'public':copy.deepcopy(public),'environment':copy.deepcopy(environment)}
    return {'payload':payload,'sha256':fingerprint(payload)}


class PrivateWorld:
    def __init__(self, token, targets, observations, likelihood):
        self.token=token
        self.targets=np.asarray(targets,bool).copy()
        self.observations=np.asarray(observations,bool).copy()
        if self.targets.shape!=self.observations.shape or self.targets.ndim!=1:
            raise ValueError('Invalid private world')
        self.targets.flags.writeable=False
        self.observations.flags.writeable=False
        self.likelihood=tuple(likelihood)
        se,fp=self.likelihood
        if not 0<=fp<se<=1:raise ValueError('Invalid likelihood')
        self.sha256=fingerprint({'token':token,'targets':self.targets.tolist(),
                                 'observations':self.observations.tolist(),'likelihood':self.likelihood})


class PrivateEnvironment:
    def __init__(self, world, seen=()):
        self.world=world
        self.seen=set(seen)

    def observe(self,index):
        if index in self.seen:raise ValueError('Repeated paid observation')
        self.seen.add(index)
        return bool(self.world.observations[index])

    def token(self):
        return {'world_token':self.world.token,'world_sha256':self.world.sha256,'seen':sorted(self.seen)}


def choose_suffix(g, public, likelihood, planner, belief_source):
    """Public-only decision interface. Called after the planning fee is charged."""
    if set(public)!=PUBLIC_KEYS:raise ValueError('Unexpected public state fields')
    if belief_source not in ('feedback','prior'):raise ValueError('Unknown planning belief')
    used=np.zeros(len(g.ids),bool);used[public['visited_indices']]=True
    belief=np.array(public['belief_hot']) if belief_source=='feedback' else g.prior_post.copy()
    config=Config(budget_s=public['budget_s'],speed_m_s=public['speed_m_s'],
                  observe_s=public['observe_s'],planning_latency_s=public['planning_latency_s'],max_visits=len(g.ids))
    route,stats=planner.solve(g,'insertion_2opt',g.xy[public['position_index']],used,
                              public['elapsed_s'],belief,config,0.,len(g.ids)-len(public['visited_indices']))
    return list(route),stats


class Session:
    def __init__(self,g,world,budget_s=1200):
        Config(budget_s=budget_s)
        if len(world.targets)!=len(g.ids):raise ValueError('Geometry/world mismatch')
        self.g=g
        self.environment=PrivateEnvironment(world)
        self.state={'region':g.region,'budget_s':float(budget_s),'speed_m_s':5.,'observe_s':20.,'planning_latency_s':1.,
                    'position_index':g.start_index,'elapsed_s':0.,'distance_m':0.,'visited_indices':[],
                    'belief_hot':g.prior_post.tolist(),'remaining_route':[],'planning_events':[],'history':[],
                    'closed':False,'return_m':0.}
        self.planner_wall_s=0.

    def public(self):
        return copy.deepcopy(self.state)

    def _route_distance(self,route):
        nodes=[self.state['position_index'],*route,self.g.start_index]
        return sum(float(np.linalg.norm(self.g.xy[a]-self.g.xy[b])) for a,b in zip(nodes,nodes[1:]))

    def set_route(self,route):
        s=self.state;route=list(route)
        if s['closed']:raise ValueError('Already returned')
        if any(type(i) is not int or not 0<=i<len(self.g.ids) for i in route):raise ValueError('Invalid cell')
        if len(set(route))!=len(route) or set(route)&set(s['visited_indices']):raise ValueError('Duplicate inspection')
        if s['elapsed_s']+self._route_distance(route)/s['speed_m_s']+len(route)*s['observe_s']>s['budget_s']+1e-8:
            raise ValueError('Suffix cannot return within budget')
        s['remaining_route']=route

    def plan(self,planner,belief_source='feedback'):
        s=self.state
        if s['closed']:raise ValueError('Already returned')
        if s['elapsed_s']+s['planning_latency_s']+self._route_distance([])/s['speed_m_s']>s['budget_s']+1e-8:
            raise ValueError('Cannot afford planning and return')
        before=s['elapsed_s'];s['elapsed_s']+=s['planning_latency_s']
        route,stats=choose_suffix(self.g,self.public(),self.environment.world.likelihood,planner,belief_source)
        self.set_route(route)
        s['planning_events'].append({'after_observations':len(s['history']),'elapsed_before_s':before,
            'elapsed_after_s':s['elapsed_s'],'position_index':s['position_index'],'belief_source':belief_source,
            'planned_route':list(route)})
        self.planner_wall_s+=stats['wall_s']

    def step(self):
        s=self.state
        if s['closed'] or not s['remaining_route']:raise ValueError('No next inspection')
        index=s['remaining_route'][0]
        if index in self.environment.seen:raise ValueError('Repeated inspection')
        before=s['elapsed_s'];leg=float(np.linalg.norm(self.g.xy[index]-self.g.xy[s['position_index']]))
        arrival=before+leg/s['speed_m_s'];end=arrival+s['observe_s']
        reserve=float(np.linalg.norm(self.g.xy[index]-self.g.xy[self.g.start_index]))/s['speed_m_s']
        if end+reserve>s['budget_s']+1e-8:raise ValueError('No return reserve')
        # Only now has this action's travel and observation time been charged.
        s['elapsed_s']=end
        bit=self.environment.observe(index)
        block=int(self.g.blocks[index]);prior=s['belief_hot'][block]
        posterior=update_belief(prior,bit,*self.environment.world.likelihood)
        s['belief_hot'][block]=posterior;s['position_index']=index;s['distance_m']+=leg
        s['visited_indices'].append(index);s['remaining_route'].pop(0)
        record={'step':len(s['history'])+1,'cell_index':index,'cell_id':self.g.ids[index],'block_index':block,
                'elapsed_before_step_s':before,'leg_m':leg,'arrival_s':arrival,'observation_complete_s':end,
                'observed_positive':bit,'block_belief_before':prior,'block_belief_after':posterior}
        s['history'].append(record)
        return copy.deepcopy(record)

    def close(self):
        s=self.state
        if s['closed']:raise ValueError('Return already charged')
        back=float(np.linalg.norm(self.g.xy[s['position_index']]-self.g.xy[self.g.start_index]))
        if s['elapsed_s']+back/s['speed_m_s']>s['budget_s']+1e-8:raise ValueError('Return over budget')
        s['elapsed_s']+=back/s['speed_m_s'];s['distance_m']+=back;s['return_m']=back
        s['position_index']=self.g.start_index;s['remaining_route']=[];s['closed']=True

    def snapshot(self):
        return pack(self.state,self.environment.token())

    @classmethod
    def restore(cls,g,snapshot,registry):
        payload=snapshot['payload']
        if snapshot['sha256']!=fingerprint(payload) or payload['schema']!=1:raise ValueError('Corrupt snapshot')
        s=copy.deepcopy(payload['public']);env=payload['environment']
        if set(s)!=PUBLIC_KEYS or s['region']!=g.region:raise ValueError('Invalid public schema')
        world=registry[env['world_token']]
        if world.sha256!=env['world_sha256']:raise ValueError('Changed private environment')
        session=cls(g,world,s['budget_s']);session.state=s;session.environment=PrivateEnvironment(world,env['seen'])
        session.validate()
        return session

    def validate(self):
        s=self.state;g=self.g;world=self.environment.world
        if (s['speed_m_s'],s['observe_s'],s['planning_latency_s'])!=(5.,20.,1.):raise ValueError('Changed costs')
        used=[];position=g.start_index;elapsed=travel=0.;belief=g.prior_post.copy();events=s['planning_events'];event_index=0
        for j in range(len(s['history'])+1):
            while event_index<len(events) and events[event_index]['after_observations']==j:
                e=events[event_index]
                if abs(e['elapsed_before_s']-elapsed)>1e-7 or e['position_index']!=position:raise ValueError('Bad planning ledger')
                elapsed+=1.
                if abs(e['elapsed_after_s']-elapsed)>1e-7:raise ValueError('Bad planning fee')
                event_index+=1
            if j==len(s['history']):break
            h=s['history'][j];i=h['cell_index']
            if type(i) is not int or not 0<=i<len(g.ids) or i in used or h['cell_id']!=g.ids[i] or h['step']!=j+1 or h['block_index']!=int(g.blocks[i]):raise ValueError('Bad history identity')
            leg=float(np.linalg.norm(g.xy[i]-g.xy[position]));arrival=elapsed+leg/5;end=arrival+20
            for key,value in [('elapsed_before_step_s',elapsed),('leg_m',leg),('arrival_s',arrival),('observation_complete_s',end)]:
                if abs(h[key]-value)>1e-7:raise ValueError('Bad motion ledger')
            bit=bool(world.observations[i]);b=int(g.blocks[i])
            if h['observed_positive']!=bit or abs(h['block_belief_before']-belief[b])>1e-12:raise ValueError('Bad observed history')
            belief[b]=update_belief(float(belief[b]),bit,*world.likelihood)
            if abs(h['block_belief_after']-belief[b])>1e-12:raise ValueError('Bad posterior history')
            if end+float(np.linalg.norm(g.xy[i]-g.xy[g.start_index]))/5>s['budget_s']+1e-8:raise ValueError('Historic return infeasible')
            used.append(i);position=i;elapsed=end;travel+=leg
        if event_index!=len(events) or used!=s['visited_indices'] or set(used)!=self.environment.seen:raise ValueError('Bad observation cursor')
        np.testing.assert_allclose(belief,s['belief_hot'],atol=1e-12,rtol=0)
        if s['closed']:
            back=float(np.linalg.norm(g.xy[position]-g.xy[g.start_index]));elapsed+=back/5;travel+=back
            if abs(s['return_m']-back)>1e-7 or s['remaining_route']:raise ValueError('Bad return state')
            position=g.start_index
        if position!=s['position_index'] or abs(elapsed-s['elapsed_s'])>1e-7 or abs(travel-s['distance_m'])>1e-7:
            raise ValueError('Bad accumulated state')
        if elapsed>s['budget_s']+1e-8:raise ValueError('Snapshot over budget')
        if not s['closed']:self.set_route(s['remaining_route'])


def public_expected_detections(g,public,route,likelihood):
    belief=np.array(public['belief_hot']);se,_=likelihood
    return float(sum(se*(.02+.58*belief[g.blocks[i]]) for i in route))


def finish_branch(g,snapshot,registry,branch,planner):
    if branch not in BRANCHES:raise ValueError('Unknown branch')
    session=Session.restore(g,snapshot,registry);before=session.public();k=len(before['history'])
    if branch!='continue':session.plan(planner,'feedback' if branch=='replan_feedback' else 'prior')
    chosen=list(session.state['remaining_route'])
    expectation=public_expected_detections(g,before,chosen,session.environment.world.likelihood)
    while session.state['remaining_route']:session.step()
    session.close()
    return {'branch':branch,'fork_observation_count':k,'fork_elapsed_s':before['elapsed_s'],'fork_planning_calls':len(before['planning_events']),
            'planned_suffix_indices':chosen,'public_expected_future_detections':expectation,
            'final_public':session.public(),'branch_planner_wall_s':session.planner_wall_s}


def score_branch(result,world):
    """Evaluator-only; its return value is never supplied to a planner or step()."""
    s=result['final_public'];k=result['fork_observation_count'];future=s['history'][k:]
    detected=[h for h in future if world.targets[h['cell_index']] and h['observed_positive']]
    arrived=[h for h in future if world.targets[h['cell_index']]]
    all_detected=[h for h in s['history'] if world.targets[h['cell_index']] and h['observed_positive']]
    remaining=s['budget_s']-result['fork_elapsed_s']
    next_detection=detected[0]['observation_complete_s']-result['fork_elapsed_s'] if detected else None
    prefix_distance=sum(h['leg_m'] for h in s['history'][:k])
    return {'future_detected_target_cells':len(detected),'future_visited_target_cells':len(arrived),
            'future_false_positive_observations':sum(not world.targets[h['cell_index']] and h['observed_positive'] for h in future),
            'future_observations':len(future),'total_detected_target_cells':len(all_detected),
            'next_target_arrival_delay_s':arrived[0]['arrival_s']-result['fork_elapsed_s'] if arrived else None,
            'next_detection_delay_s':next_detection,'no_future_detection':not bool(detected),
            'next_detection_or_remaining_budget_s':next_detection if next_detection is not None else remaining,
            'suffix_distance_m':s['distance_m']-prefix_distance,'suffix_elapsed_s':s['elapsed_s']-result['fork_elapsed_s'],
            'total_distance_m':s['distance_m'],'total_time_s':s['elapsed_s'],
            'additional_planning_calls':len(s['planning_events'])-result['fork_planning_calls']}
