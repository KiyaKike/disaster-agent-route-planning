"""Exact small-grid routing diagnostics; hindsight oracles are evaluator-only.

Held-Karp enumerates minimum closed tour length for every visited subset.
The same feasible subset catalogue can optimize public utility or private
reference/TP rewards without changing geometry,observation cost or budget.
"""
import itertools
import math

import numpy as np


def exact_catalogue(xy,home):
    xy=np.asarray(xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all() or home!=len(xy)-1:
        raise ValueError('Finite2D nodes followed by one home required')
    n=home
    if not 1<=n<=18:raise ValueError('Small diagnostic only')
    distances=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    count=1<<n;dp=np.full((count,n),np.inf);parent=np.full((count,n),-1,np.int16)
    cardinality=np.zeros(count,np.uint8)
    for mask in range(1,count):
        cardinality[mask]=cardinality[mask>>1]+(mask&1)
        bits=[j for j in range(n) if mask&(1<<j)]
        for j in bits:
            previous=mask^(1<<j)
            if not previous:dp[mask,j]=distances[home,j];continue
            values=dp[previous]+distances[:n,j]
            k=int(np.argmin(values));dp[mask,j]=values[k];parent[mask,j]=k
    last=np.argmin(dp+distances[:n,home],axis=1).astype(np.int16)
    closed=dp[np.arange(count),last]+distances[last,home];closed[0]=0.;last[0]=-1
    return {'distances':distances,'dp':dp,'parent':parent,'closed':closed,'last':last,'cardinality':cardinality}


def reconstruct(cat,mask):
    if mask==0:return []
    path=[];j=int(cat['last'][mask]);left=mask
    while left:
        if j<0 or not left&(1<<j):raise ValueError('Invalid predecessor table')
        path.append(j);k=int(cat['parent'][left,j]);left^=1<<j;j=k
    return path[::-1]


def subset_rewards(weights):
    weights=np.asarray(weights,float)
    if weights.ndim!=1 or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Finite nonnegative additive rewards required')
    out=np.zeros(1<<len(weights))
    for j,w in enumerate(weights):
        step=1<<j
        for start in range(step,len(out),2*step):out[start:start+step]=out[start-step:start]+w
    return out


def solve_catalogue(cat,weights,budget,move=20.,observe=20.,plan=2.):
    if len(weights)!=cat['dp'].shape[1] or min(budget,move,observe,plan)<0:
        raise ValueError('Invalid diagnostic contract')
    rewards=subset_rewards(weights)
    cost=cat['closed']*move+cat['cardinality']*observe+plan
    feasible=np.flatnonzero(cost<=budget+1e-8)
    if not len(feasible):raise ValueError('Budget cannot pay fixed planning charge')
    best=float(rewards[feasible].max())
    ties=feasible[np.isclose(rewards[feasible],best,rtol=0,atol=1e-10)]
    # Deterministic tie resolution: shortest closed tour,then lowest subset mask.
    distance=float(cat['closed'][ties].min())
    ties=ties[np.isclose(cat['closed'][ties],distance,rtol=0,atol=1e-10)]
    mask=int(ties.min());route=reconstruct(cat,mask)
    return {'route':route,'subset_mask':mask,'objective':float(rewards[mask]),
            'distance_grid':float(cat['closed'][mask]),'elapsed_s_proxy':float(cost[mask]),
            'observations':int(cat['cardinality'][mask]),'feasible_subsets':len(feasible),'planning_calls':int(plan)}


def audit_bellman(cat):
    """Verify optimality certificate using each possible final edge separately."""
    dp=cat['dp'];d=cat['distances'];n=dp.shape[1];count=len(dp);states=0
    if not np.isinf(dp[0]).all():raise AssertionError('Empty open path should be infinite')
    allm=np.arange(count)
    for j in range(n):
        present=allm[(allm&(1<<j))!=0];prev=present^(1<<j)
        expected=np.full(len(present),np.inf)
        for k in range(n):
            if k!=j:expected=np.minimum(expected,dp[prev,k]+d[k,j])
        expected[prev==0]=d[n,j]
        np.testing.assert_allclose(dp[present,j],expected,rtol=0,atol=1e-10)
        absent=allm[(allm&(1<<j))==0]
        assert np.isinf(dp[absent,j]).all()
        states+=len(present)
    closed=np.min(dp+d[:n,n],axis=1);closed[0]=0
    np.testing.assert_allclose(closed,cat['closed'],rtol=0,atol=1e-10)
    return states


def brute_force(xy,home,weights,budget,move=20.,observe=20.,plan=2.):
    """Independent permutation enumeration for unit tests on tiny maps."""
    n=home;best=0.
    for k in range(n+1):
        for path in itertools.permutations(range(n),k):
            nodes=(home,)+path+(home,)
            length=sum(math.dist(xy[a],xy[b]) for a,b in zip(nodes,nodes[1:]))
            if length*move+k*observe+plan<=budget+1e-8:
                best=max(best,sum(weights[j] for j in path))
    return best
