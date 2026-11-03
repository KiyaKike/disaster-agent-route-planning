"""Two public map organizations over one paid inspection-agent state.

A: prior/cues + observation evidence,with shared geometry.
B: geometric cost + semantic search state (including prior and evidence).
These are information-equivalent interfaces,not two new routing algorithms.
"""
import copy
from dataclasses import replace

import numpy as np

from route_core import Config, update_belief
from route_snapshot_core import Session, fingerprint
from route_trigger_core import can_propose

MODES = ('prior_evidence', 'geometry_semantic')
POLICIES = ('pre_static', 'post_static', 'online_5')


def geometry_key(g):
    return fingerprint({'region':g.region,'ids':g.ids,'xy':g.xy.tolist(),'blocks':g.blocks.tolist(),
                        'pre':g.prior_pre.tolist(),'post':g.prior_post.tolist(),
                        'cue':[float(x) if np.isfinite(x) else None for x in g.cue],'start':g.start_index})


class DualMapMemory:
    def __init__(self,g,likelihood):
        se,fp=likelihood
        if not 0<=fp<se<=1:raise ValueError('Invalid assumed observation model')
        self.g=g;self.likelihood=tuple(likelihood);self.phase='pre';self.records=[]
        self.key=geometry_key(g)

    def receive_post_cues(self):
        if self.phase!='pre' or self.records:raise ValueError('Post update must precede paid observations and occur once')
        self.phase='post'

    def ingest(self,public):
        """Append only completed observations already present in the paid ledger."""
        if public['region']!=self.g.region or len(public['history'])!=len(public['visited_indices']):
            raise ValueError('Mismatched paid history')
        projected=[]
        for h,i in zip(public['history'],public['visited_indices']):
            if h['cell_index']!=i or type(i) is not int or not 0<=i<len(self.g.ids):raise ValueError('Bad paid cell')
            if h['cell_id']!=self.g.ids[i] or h['block_index']!=int(self.g.blocks[i]):raise ValueError('Bad spatial registration')
            if type(h['observed_positive']) is not bool:raise ValueError('Observation must be a bit')
            if h['observation_complete_s']>public['elapsed_s']+1e-8 or abs(h['observation_complete_s']-h['arrival_s']-20)>1e-7:
                raise ValueError('Observation not paid/completed')
            projected.append({k:h[k] for k in ('step','cell_index','cell_id','block_index','observed_positive','observation_complete_s')})
        if len(set(public['visited_indices']))!=len(projected):raise ValueError('Repeated observation')
        if projected[:len(self.records)]!=self.records or len(projected)<len(self.records):raise ValueError('Evidence history was changed')
        self.records=copy.deepcopy(projected)

    def prior(self):
        return (self.g.prior_pre if self.phase=='pre' else self.g.prior_post).copy()

    def posterior(self):
        q=self.prior()
        for r in self.records:
            b=r['block_index'];q[b]=update_belief(float(q[b]),r['observed_positive'],*self.likelihood)
        return q

    def evidence(self):
        signals=[None]*len(self.g.ids);times=[None]*len(self.g.ids)
        counts=np.zeros((len(self.g.block_ids),2),int)
        for r in self.records:
            signals[r['cell_index']]=r['observed_positive'];times[r['cell_index']]=r['observation_complete_s']
            counts[r['block_index'],int(r['observed_positive'])]+=1
        return {'signals':signals,'observed_at_s':times,'block_negative_positive_counts':counts.tolist(),
                'paid_records':copy.deepcopy(self.records),'confirmed_targets':None,
                'meaning':'Observed bits,not verified targets;unvisited=None,negative does not certify clearance.'}

    def snapshot(self):
        return {'schema':1,'geometry_key':self.key,'phase':self.phase,'assumed_likelihood':self.likelihood,
                'records':copy.deepcopy(self.records)}

    @classmethod
    def restore(cls,g,payload,public):
        if payload['schema']!=1 or payload['geometry_key']!=geometry_key(g):raise ValueError('Changed map registration/inputs')
        result=cls(g,payload['assumed_likelihood'])
        if payload['phase']=='post':result.receive_post_cues()
        elif payload['phase']!='pre':raise ValueError('Unknown source phase')
        result.ingest(public)
        if result.records!=payload['records']:raise ValueError('Evidence/snapshot mismatch')
        return result

    def export(self,mode,public):
        if mode not in MODES:raise ValueError('Unknown map organization')
        if public['visited_indices']!=[r['cell_index'] for r in self.records]:raise ValueError('Unsynchronized evidence cursor')
        g=self.g;q=self.posterior();prior={'phase':self.phase,'block_hot_prior':self.prior().tolist(),
                'change_cue':[float(x) if np.isfinite(x) else None for x in g.cue] if self.phase=='post' else [None]*len(g.ids),
                'provenance':{'geometry':'Existing PS grid','pre_prior':'Uniform .2 placeholder,not building exposure',
                              'post_prior':'Synthetic rank construction from PS change cue,not calibrated damage probability'}}
        ev=self.evidence();fusion={'block_hot_posterior':q.tolist(),'cell_target_expectation':(.02+.58*q[g.blocks]).tolist(),
                                 'model':'Known synthetic3x3hot/cold model;no real-world calibration established'}
        common={'mode':mode,'geometry_key':self.key,'phase':self.phase,'assumed_likelihood':list(self.likelihood),
                'position_index':public['position_index'],'elapsed_s':public['elapsed_s'],'budget_s':public['budget_s'],
                'visited_indices':list(public['visited_indices'])}
        if mode=='prior_evidence':
            return {**common,'layers':{'prior_cues':prior,'observation_evidence':ev},'fusion':fusion}
        move=np.linalg.norm(g.xy-g.xy[public['position_index']],axis=1)
        home=np.linalg.norm(g.xy-g.xy[g.start_index],axis=1)
        feasible=public['elapsed_s']+1+move/5+20+home/5<=public['budget_s']+1e-9
        feasible[public['visited_indices']]=False
        geo={'xy_m':g.xy.tolist(),'start_index':g.start_index,'distance_from_current_m':move.tolist(),
             'distance_to_depot_m':home.tolist(),'budget_feasible_after_planning':feasible.tolist(),
             'road_passability':None,'terrain_obstacles':None,
             'meaning':'Straight-line geometric cost/return feasibility,not validated road/flight passability'}
        return {**common,'layers':{'geometry_cost':geo,'semantic_search':{'prior':prior,'evidence':ev,'fusion':fusion}}}


def planner_inputs(g,view,public):
    """Read either map organization through an explicit public-only adapter."""
    if view['geometry_key']!=geometry_key(g) or view['visited_indices']!=public['visited_indices']:
        raise ValueError('Map/planner state mismatch')
    if view['mode']=='prior_evidence':fusion=view['fusion']
    elif view['mode']=='geometry_semantic':fusion=view['layers']['semantic_search']['fusion']
    else:raise ValueError('Unknown map organization')
    belief=np.array(fusion['block_hot_posterior'],float)
    if belief.shape!=(len(g.block_ids),) or not np.isfinite(belief).all() or np.any((belief<0)|(belief>1)):
        raise ValueError('Invalid map belief')
    used=np.zeros(len(g.ids),bool);used[view['visited_indices']]=True
    return belief,used


class DualMapAgent:
    def __init__(self,g,world,mode,policy,planner,budget_s=1200):
        if mode not in MODES or policy not in POLICIES:raise ValueError('Unknown agent configuration')
        self.g=g;self.mode=mode;self.policy=policy;self.planner=planner
        self.maps=DualMapMemory(g,world.likelihood)
        # Pre planning uses only the uniform pre prior. Post sources arrive explicitly.
        self.session=Session(replace(g,prior_post=g.prior_pre),world,budget_s)
        self.frames=[];self.skipped=[];self.stage='initialized';self.capture('initialized')

    def view(self):return self.maps.export(self.mode,self.session.public())

    def capture(self,reason):
        s=self.session.public();self.frames.append({'frame':len(self.frames),'reason':reason,'phase':self.maps.phase,
            'observations':len(s['history']),'position_index':s['position_index'],'elapsed_s':s['elapsed_s'],
            'distance_m':s['distance_m'],'planning_calls':len(s['planning_events']),
            'remaining_route':list(s['remaining_route']),'closed':s['closed'],
            'block_hot_posterior':self.maps.posterior().tolist(),'map_state_sha256':fingerprint(self.maps.snapshot())})

    def plan(self,reason):
        s=self.session.state
        if s['closed']:raise ValueError('Already closed')
        reserve=float(np.linalg.norm(self.g.xy[s['position_index']]-self.g.xy[self.g.start_index]))/5
        if s['elapsed_s']+1+reserve>s['budget_s']+1e-9:raise ValueError('Cannot pay planning and return')
        before=s['elapsed_s'];s['elapsed_s']+=1
        view=self.view();belief,used=planner_inputs(self.g,view,self.session.public())
        cfg=Config(budget_s=s['budget_s'],max_visits=len(self.g.ids))
        route,stats=self.planner.solve(self.g,'insertion_2opt',self.g.xy[s['position_index']],used,s['elapsed_s'],belief,cfg,0.,len(self.g.ids)-int(used.sum()))
        self.session.set_route(route);s['belief_hot']=belief.tolist()
        s['planning_events'].append({'after_observations':len(s['history']),'elapsed_before_s':before,
            'elapsed_after_s':s['elapsed_s'],'position_index':s['position_index'],'belief_source':'feedback',
            'planned_route':list(route),'map_phase':self.maps.phase,'reason':reason,
            'map_state_sha256':fingerprint(self.maps.snapshot())})
        self.session.planner_wall_s+=stats['wall_s'];self.capture(reason)

    def prepare(self):
        if self.stage!='initialized':raise ValueError('Already prepared')
        self.plan('pre_plan');self.maps.receive_post_cues();self.session.g=self.g
        self.session.state['belief_hot']=self.maps.posterior().tolist();self.capture('post_cues_received')
        if self.policy!='pre_static':self.plan('post_plan')
        self.stage='executing'

    def advance(self):
        if self.stage!='executing' or not self.session.state['remaining_route']:raise ValueError('No next action')
        self.session.step();self.maps.ingest(self.session.public());self.capture('paid_observation')
        if self.policy=='online_5' and len(self.maps.records)%5==0:
            if can_propose(self.g,self.session.public()):self.plan('online_replan_5')
            else:self.skipped.append(len(self.maps.records));self.capture('unaffordable_replan_skipped')

    def finish(self):
        if self.stage=='initialized':self.prepare()
        while self.session.state['remaining_route']:self.advance()
        self.session.close();self.stage='closed';self.capture('returned')
        return self.result()

    def result(self):
        return {'region':self.g.region,'mode':self.mode,'policy':self.policy,'stage':self.stage,
                'final_public':self.session.public(),'maps':self.maps.snapshot(),'frames':copy.deepcopy(self.frames),
                'skipped_unaffordable_replans':list(self.skipped)}

    def snapshot(self):
        payload={'schema':1,'mode':self.mode,'policy':self.policy,'stage':self.stage,
                 'session':self.session.snapshot(),'maps':self.maps.snapshot(),
                 'frames':copy.deepcopy(self.frames),'skipped':list(self.skipped)}
        return {'payload':payload,'sha256':fingerprint(payload)}

    @classmethod
    def restore(cls,g,snapshot,registry,planner):
        p=snapshot['payload']
        if snapshot['sha256']!=fingerprint(p) or p['schema']!=1:raise ValueError('Corrupt agent snapshot')
        phase=p['maps']['phase'];sg=replace(g,prior_post=g.prior_pre) if phase=='pre' else g
        session=Session.restore(sg,p['session'],registry)
        memory=DualMapMemory.restore(g,p['maps'],session.public())
        np.testing.assert_allclose(session.state['belief_hot'],memory.posterior(),atol=1e-12,rtol=0)
        obj=cls.__new__(cls);obj.g=g;obj.mode=p['mode'];obj.policy=p['policy'];obj.stage=p['stage'];obj.planner=planner
        if obj.mode not in MODES or obj.policy not in POLICIES:raise ValueError('Unknown restored configuration')
        obj.session=session;obj.maps=memory;obj.frames=copy.deepcopy(p['frames']);obj.skipped=list(p['skipped'])
        return obj
