"""Corrected small ISRS mechanism reference based on pinned AIPPMS source.

MIT upstream: sisl/SBO_AIPPMS commit24db0a6;see vendor/LICENSE.
Python reimplementation,NOT a native Julia/table replication. Hidden truth lives
only in Environment or simulated worlds sampled from public belief. Bayesian
updates use the delivered sensor observation,not upstream env.location_states.
"""
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
import random


@dataclass(frozen=True)
class Layout:
    side: int = 5
    rocks: tuple = ((0,2),(2,4),(3,3))  # Original5x5default positions,zero-based.
    beacons: tuple = ((0,0),(2,2),(4,4)) # Explicit controlled-layout addition.
    sensor_model: str = 'informative'

    @property
    def home(self):return (0,0)


SENSORS=(('near',1,2.5),('far',4,10.)) # Cost ticks;one tick=.5energy.


def manhattan(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])


def neighbors(layout,pos):
    return tuple(sorted((x,y) for x,y in ((pos[0]-1,pos[1]),(pos[0]+1,pos[1]),(pos[0],pos[1]-1),(pos[0],pos[1]+1))
                        if 0<=x<layout.side and 0<=y<layout.side))


def accuracy(layout,pos,sensor):
    eff=SENSORS[sensor][2]
    if layout.sensor_model not in ('informative','uninformative'):raise ValueError('Unknown sensor model')
    return tuple(.5 if layout.sensor_model=='uninformative' else .5*(1+2**(-4*math.dist(pos,r)/eff)) for r in layout.rocks)


def actions(layout,pos,remaining):
    result=[]
    for nxt in neighbors(layout,pos):
        if 2+2*manhattan(nxt,layout.home)<=remaining:result.append(('move',nxt))
    if pos in layout.beacons:
        for k,s in enumerate(SENSORS):
            if s[1]+2*manhattan(pos,layout.home)<=remaining:result.append(('sense',k))
    if pos==layout.home:result.append(('stop',None))
    return tuple(result)


def cost(action):
    return 2 if action[0]=='move' else SENSORS[action[1]][1] if action[0]=='sense' else 0


def movement(layout,pos,consumed,probs,action):
    if action[0]!='move':raise ValueError('Movement action required')
    nxt=action[1];reward=0.;new=consumed
    if nxt in layout.rocks:
        j=layout.rocks.index(nxt)
        reward=-10. if consumed&(1<<j) else 20*probs[j]-10
        new|=1<<j
    return nxt,new,reward


def posterior(prior,correct,observed_positive):
    if not 0<=prior<=1 or not .5<=correct<=1:raise ValueError('Invalid probability')
    good=prior*(correct if observed_positive else 1-correct)
    bad=(1-prior)*(1-correct if observed_positive else correct)
    if good+bad<=0:raise ValueError('Zero-probability observation')
    return good/(good+bad)


def update(layout,prior,consumed,pos,sensor,observation):
    if pos not in layout.beacons or len(observation)!=len(prior) or any(type(x) is not bool for x in observation):
        raise ValueError('Actual beacon observation required')
    q=accuracy(layout,pos,sensor)
    return tuple(0. if consumed&(1<<j) else posterior(p,q[j],o) for j,(p,o) in enumerate(zip(prior,observation)))


def outcomes(layout,prior,consumed,pos,sensor):
    q=accuracy(layout,pos,sensor)
    for obs in itertools.product((False,True),repeat=len(prior)):
        probability=1.
        for j,(p,o) in enumerate(zip(prior,obs)):
            p=0. if consumed&(1<<j) else p
            positive=p*q[j]+(1-p)*(1-q[j])
            probability*=positive if o else 1-positive
        if probability>0:yield probability,obs,update(layout,prior,consumed,pos,sensor,obs)


class ExactOneSense:
    """Exact finite-energy reference:zero or at most one paid remote sensing.

    This restriction is a declared diagnostic,not the unrestricted AIPPMS optimum.
    Additional sensing is available to the POMCP variants,which need not dominate
    this restricted exact reference at finite planning compute.
    """
    def __init__(self,layout):
        self.layout=layout;self.calls=0
        self.value=lru_cache(maxsize=None)(self._value)

    def _value(self,pos,remaining,consumed,prior,sensing_left):
        self.calls+=1
        best=-math.inf;choice=None
        for action in actions(self.layout,pos,remaining):
            if action[0]=='stop':v=0.
            elif action[0]=='move':
                nxt,new,reward=movement(self.layout,pos,consumed,prior,action)
                v=reward+self.value(nxt,remaining-2,new,prior,sensing_left)[0]
            elif sensing_left:
                v=sum(p*self.value(pos,remaining-cost(action),consumed,post,0)[0]
                      for p,obs,post in outcomes(self.layout,prior,consumed,pos,action[1]))
            else:continue
            # Actions ordered move,sense,stop. Prefer stop on exact-zero ties;
            # then motion over unnecessary sensing when information is worthless.
            priority=0 if action[0]=='stop' else 1 if action[0]=='move' else 2
            old_priority=0 if choice and choice[0]=='stop' else 1 if choice and choice[0]=='move' else 2
            if v>best+1e-10 or (abs(v-best)<=1e-10 and priority<old_priority):best,choice=v,action
        if choice is None:raise ValueError('No feasible action,including return')
        return best,choice

    def action(self,pos,remaining,consumed,prior,sensing_left):
        return self.value(pos,remaining,consumed,tuple(prior),sensing_left)[1]


def predictive_sample(layout,prior,consumed,rng):
    return tuple(False if consumed&(1<<j) else rng.random()<p for j,p in enumerate(prior))


def simulate_observation(layout,truth,consumed,pos,sensor,rng):
    q=accuracy(layout,pos,sensor)
    return tuple((False if consumed&(1<<j) else good) if rng.random()<q[j] else not(False if consumed&(1<<j) else good)
                 for j,good in enumerate(truth))


def transition(layout,pos,remaining,consumed,prior,truth,action,rng):
    if action not in actions(layout,pos,remaining):raise ValueError('Infeasible action')
    if action[0]=='stop':return pos,remaining,consumed,prior,('stop',),0.
    if action[0]=='move':
        nxt,new,_=movement(layout,pos,consumed,prior,action)
        reward=0.
        if nxt in layout.rocks:
            j=layout.rocks.index(nxt);reward=-10. if consumed&(1<<j) or not truth[j] else 10.
        # The next task state of any visited rock is bad,regardless of its initial state.
        post=tuple(0. if new&(1<<j) else p for j,p in enumerate(prior))
        return nxt,remaining-2,new,post,('visit',),reward
    obs=simulate_observation(layout,truth,consumed,pos,action[1],rng)
    post=update(layout,prior,consumed,pos,action[1],obs)
    return pos,remaining-cost(action),consumed,post,obs,0.


def expected_mode_gain(layout,pos,consumed,prior,sensor):
    # Exact expected per-location mode increase,paper Eq5;25nodes as upstream.
    total=0.
    q=accuracy(layout,pos,sensor)
    for j,p in enumerate(prior):
        if consumed&(1<<j):continue
        positive=p*q[j]+(1-p)*(1-q[j])
        expected=0.
        for o,prob in ((True,positive),(False,1-positive)):
            if prob>0:
                post=posterior(p,q[j],o);expected+=prob*max(post,1-post)
        total+=expected-max(p,1-p)
    return max(0.,total/(layout.side**2))


class POMCP:
    """POMCP-style root belief sampling,observation tree and GCB/random rollout.

    Corrected Python reference with explicit UCT10 and discount1;not a bytewise
    port of BasicPOMCP.jl. Queries100/depth5 match pinned trial script settings.
    Exact factorized belief updates accompany observation histories.
    """
    def __init__(self,layout,seed,rollout='gcb',queries=100,depth=5):
        self.layout=layout;self.rng=random.Random(seed);self.rollout_kind=rollout
        self.queries=queries;self.depth=depth;self.tree={};self.simulations=0

    def rollout_action(self,pos,remaining,consumed,prior):
        aa=actions(self.layout,pos,remaining)
        if self.rollout_kind=='random':return self.rng.choice(aa)
        ratios=[]
        for a in aa:
            if a[0]=='move':v=movement(self.layout,pos,consumed,prior,a)[2]/1.
            elif a[0]=='sense':v=expected_mode_gain(self.layout,pos,consumed,prior,a[1])/(cost(a)/2)
            else:v=0.
            ratios.append(v)
        largest=max(ratios);weights=[math.exp(v-largest) for v in ratios]
        return self.rng.choices(aa,weights=weights,k=1)[0]

    def rollout(self,pos,remaining,consumed,prior,truth):
        total=0.
        while True:
            a=self.rollout_action(pos,remaining,consumed,prior)
            if a[0]=='stop':return total
            pos,remaining,consumed,prior,obs,r=transition(self.layout,pos,remaining,consumed,prior,truth,a,self.rng)
            total+=r

    def simulate(self,pos,remaining,consumed,prior,truth,history,depth):
        if depth<=0:return self.rollout(pos,remaining,consumed,prior,truth)
        aa=actions(self.layout,pos,remaining)
        if history not in self.tree:
            self.tree[history]={a:[0,0.] for a in aa}
            return self.rollout(pos,remaining,consumed,prior,truth)
        node=self.tree[history];untried=[a for a in aa if node[a][0]==0]
        if untried:a=self.rng.choice(untried)
        else:
            visits=sum(v[0] for v in node.values())
            a=max(aa,key=lambda a:node[a][1]+10*math.sqrt(math.log(visits)/node[a][0]))
        if a[0]=='stop':value=0.
        else:
            nxt,left,new,post,obs,r=transition(self.layout,pos,remaining,consumed,prior,truth,a,self.rng)
            value=r+self.simulate(nxt,left,new,post,truth,history+(a,obs),depth-1)
        stat=node[a];stat[0]+=1;stat[1]+=(value-stat[1])/stat[0]
        return value

    def action(self,pos,remaining,consumed,prior,sensing_left=None):
        self.tree={}
        for _ in range(self.queries):
            truth=predictive_sample(self.layout,prior,consumed,self.rng)
            self.simulate(pos,remaining,consumed,prior,truth,(),self.depth);self.simulations+=1
        root=self.tree[()]
        return max((a for a,v in root.items() if v[0]),key=lambda a:root[a][1])


class Environment:
    """Private truth access restricted to paid action execution."""
    def __init__(self,layout,truth,budget,seed):
        if len(truth)!=len(layout.rocks):raise ValueError('One hidden bit per rock')
        self.layout=layout;self._truth=tuple(truth);self.budget=2*budget
        self.rng=random.Random(seed);self.sensor_read_index=0

    def step(self,pos,remaining,consumed,prior,action):
        return transition(self.layout,pos,remaining,consumed,prior,self._truth,action,self.rng)


def run_episode(layout,truth,budget,policy,seed=0,planner_seed=0,exact=None):
    env=Environment(layout,truth,budget,seed);pos=layout.home;remaining=2*budget;consumed=0
    prior=(.5,)*len(layout.rocks);sensing_left=1 if policy=='exact_one_sense' else 0
    planner=(exact or ExactOneSense(layout)) if policy.startswith('exact') else POMCP(layout,planner_seed,policy.removeprefix('pomcp_'))
    history=[];total=0.;senses=0;moves=0
    while True:
        a=planner.action(pos,remaining,consumed,prior,sensing_left)
        before={'position':list(pos),'remaining_energy':remaining/2,'consumed_mask':consumed,'belief':list(prior)}
        if a[0]=='stop':
            assert pos==layout.home;break
        nxt,left,new,post,obs,r=env.step(pos,remaining,consumed,prior,a)
        history.append({'before':before,'action':[a[0],list(a[1]) if a[0]=='move' else a[1]],
                        'observation':list(obs),'reward':r,'after_position':list(nxt),'after_remaining_energy':left/2,'after_belief':list(post)})
        pos,remaining,consumed,prior=nxt,left,new,post;total+=r
        if a[0]=='sense':senses+=1;sensing_left=0
        else:moves+=1
        assert remaining>=2*manhattan(pos,layout.home)
    return {'policy':policy,'total_reward':total,'senses':senses,'moves':moves,'energy_used':budget-remaining/2,
            'returned':pos==layout.home,'remaining_energy':remaining/2,'history':history,
            'planning_simulations':getattr(planner,'simulations',0)}


def exact_evaluation(layout,budget,allow_sense,planner):
    """Enumerate every actual hidden world AND every possible paid sensor output.

    No Monte Carlo noise;cross-checks analytic Bellman value against truth-world
    execution. At most one sensor makes branch enumeration small and exact.
    """
    branches=[]
    for truth in itertools.product((False,True),repeat=len(layout.rocks)):
        def branch(pos,remaining,consumed,prior,sensing_left,prob,total,history):
            a=planner.action(pos,remaining,consumed,prior,sensing_left)
            if a[0]=='stop':
                assert pos==layout.home
                branches.append({'truth':list(truth),'probability':prob/(2**len(truth)),'reward':total,
                                 'history':history,'energy_used':budget-remaining/2,'returned':True})
            elif a[0]=='move':
                nxt,new,_=movement(layout,pos,consumed,prior,a);r=0.
                if nxt in layout.rocks:
                    j=layout.rocks.index(nxt);r=-10. if consumed&(1<<j) or not truth[j] else 10.
                post=tuple(0. if new&(1<<j) else p for j,p in enumerate(prior))
                branch(nxt,remaining-2,new,post,sensing_left,prob,total+r,history+[{'action':['move',list(nxt)],'reward':r}])
            else:
                q=accuracy(layout,pos,a[1])
                for obs in itertools.product((False,True),repeat=len(truth)):
                    likelihood=1.
                    for j,o in enumerate(obs):
                        state=False if consumed&(1<<j) else truth[j]
                        likelihood*=q[j] if o==state else 1-q[j]
                    if likelihood:
                        post=update(layout,prior,consumed,pos,a[1],obs)
                        branch(pos,remaining-cost(a),consumed,post,0,prob*likelihood,total,
                               history+[{'action':['sense',a[1]],'observation':list(obs),'reward':0.}])
        branch(layout.home,2*budget,0,(.5,)*len(layout.rocks),allow_sense,1.,0.,[])
    assert abs(sum(b['probability'] for b in branches)-1)<1e-10
    expected=sum(b['probability']*b['reward'] for b in branches)
    value=planner.value(layout.home,2*budget,0,(.5,)*len(layout.rocks),allow_sense)[0]
    assert abs(expected-value)<1e-8
    return {'expected_reward':expected,'bellman_value':value,'branches':branches,
            'sense_probability':sum(b['probability'] for b in branches if any(h['action'][0]=='sense' for h in b['history'])),
            'expected_energy':sum(b['probability']*b['energy_used'] for b in branches)}
