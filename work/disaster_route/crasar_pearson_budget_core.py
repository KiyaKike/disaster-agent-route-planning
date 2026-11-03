"""Building-level paid evidence replay with a hard return-inclusive time budget."""
import math
import time

import numpy as np

from route_headroom_core import reconstruct, subset_rewards
from run_boone_two_step import update_scores


POLICIES = ('distance', 'static_exact', 'no_update', 'spatial', 'matched_spatial', 'pearson')
SPEED, OBSERVE, PLAN = 5., 10., 1.


def select_static(cat, p0, budget):
    costs = cat['closed']/SPEED + cat['cardinality']*OBSERVE + PLAN
    feasible = np.flatnonzero(costs<=budget+1e-9)
    rewards = subset_rewards(p0)
    best = float(rewards[feasible].max())
    ties = feasible[np.isclose(rewards[feasible],best,atol=1e-10,rtol=0)]
    mask = int(min(ties,key=lambda m:(cat['closed'][m],int(m))))
    return reconstruct(cat,mask), {'subset_mask':mask,'public_objective':float(rewards[mask]),
                                 'upper_bound':best,'gap':0.,'subsets':len(costs),
                                 'feasible_subsets':len(feasible)}


def budget_episode(p0, xy, similarity, policy, budget, reveal, catalogue=None):
    """xy contains building centers followed by one depot; no free anchor label."""
    started = time.perf_counter()
    p0, xy = np.asarray(p0,float), np.asarray(xy,float)
    home = len(p0)
    if xy.shape!=(home+1,2) or policy not in POLICIES or budget<PLAN:
        raise ValueError('Invalid public inputs or budget')
    scores, current, elapsed, distance = p0.copy(), home, 0., 0.
    visited, evidence, decisions, updates = set(), [], [], []
    planning_calls, last_feedback, certificate = 0, None, None
    planned = None
    if policy=='static_exact':
        planned, certificate = select_static(catalogue,p0,budget)
        elapsed, planning_calls = PLAN, 1
    while len(visited)<home:
        if planned is not None:
            if not planned:
                break
            target = planned.pop(0)
            candidates = [target]
        else:
            back_now = math.dist(xy[current],xy[home])/SPEED
            if elapsed+PLAN+back_now>budget+1e-9:
                break
            elapsed += PLAN
            planning_calls += 1
            # Newly paid evidence may update the map only after its own visit.
            if current!=home and last_feedback is not None and policy in ('pearson','spatial','matched_spatial'):
                scores, update = update_scores(scores,current,last_feedback,visited,xy[:home],similarity,policy)
                update['decision_number'] = planning_calls
                updates.append(update)
            candidates = [i for i in range(home) if i not in visited and
                          elapsed+math.dist(xy[current],xy[i])/SPEED+OBSERVE+
                          math.dist(xy[i],xy[home])/SPEED<=budget+1e-9]
            if not candidates:
                decisions.append({'decision_number':planning_calls,'elapsed_proxy_s':elapsed,
                                  'observed_indices':sorted(visited),'feasible_indices':[], 'target':None})
                break
            if policy=='distance':
                target = min(candidates,key=lambda i:(math.dist(xy[current],xy[i]),i))
            else:
                target = min(candidates,key=lambda i:(-scores[i]/max(math.dist(xy[current],xy[i]),1e-6),i))
        leg = math.dist(xy[current],xy[target])
        before_return = elapsed+leg/SPEED+OBSERVE
        reserved_return = math.dist(xy[target],xy[home])/SPEED
        assert before_return+reserved_return<=budget+1e-8
        decisions.append({'decision_number':planning_calls,'elapsed_proxy_s':elapsed,
                          'observed_indices':sorted(visited),'feasible_indices':candidates,'target':target,
                          'position_before':current,'scores':scores.tolist(),
                          'return_reserve_s':reserved_return})
        arrival = elapsed+leg/SPEED
        elapsed, distance, current = before_return, distance+leg, target
        visited.add(target)
        last_feedback = reveal(target)
        if last_feedback not in (None,0,1):
            raise ValueError('A paid source label must be binary or unknown')
        evidence.append({'index':target,'source_label':last_feedback,'arrival_proxy_s':arrival,
                         'observation_complete_proxy_s':elapsed,'leg_m':leg,
                         'score_before_observation':float(scores[target])})
    back = math.dist(xy[current],xy[home])
    distance += back
    elapsed += back/SPEED
    assert elapsed<=budget+1e-8
    assert len({r['index'] for r in evidence})==len(evidence)
    return {'policy':policy,'budget_proxy_s':budget,'route':[r['index'] for r in evidence],
            'distance_m':distance,'return_distance_m':back,'elapsed_proxy_s':elapsed,
            'planning_calls':planning_calls,'observations':evidence,'decisions':decisions,'updates':updates,
            'returned_home':True,'budget_slack_s':budget-elapsed,'certificate':certificate,
            'solver_wall_seconds':time.perf_counter()-started}
