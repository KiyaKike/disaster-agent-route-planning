"""Paid candidate evaluation and predeclared route acceptance heuristics.

The one-standard-deviation rule is a model-based heuristic, not a safety bound.
All dynamic policies use the same committed-suffix solver and pay each attempt,
including rejected proposals. No private world is passed to a decision function.
"""
import math
from dataclasses import replace

import numpy as np

from route_snapshot_core import PUBLIC_KEYS, Session

POLICIES = ('pre_static', 'post_static', 'always', 'periodic_5', 'gain_5', 'one_sd_5')
NAMES = {'pre_static': 'Pre static', 'post_static': 'Post static', 'always': 'Always replan',
         'periodic_5': 'Every 5 visits', 'gain_5': 'Positive mean / 5', 'one_sd_5': 'Mean > 1 SD / 5'}
PERIOD = 5
EPS = 1e-12


def public_geometry(g, policy):
    return replace(g, prior_post=g.prior_pre) if policy == 'pre_static' else g


def route_cost(g, public, route):
    nodes = [public['position_index'], *route, g.start_index]
    distance = sum(float(np.linalg.norm(g.xy[a]-g.xy[b])) for a, b in zip(nodes, nodes[1:]))
    return distance/public['speed_m_s'] + len(route)*public['observe_s']


def feasible_old_suffix(g, public, route):
    """A paid rejected attempt may require dropping tail stops to preserve return."""
    route = list(route)
    while route and public['elapsed_s']+route_cost(g, public, route) > public['budget_s']+1e-8:
        route.pop()
    if public['elapsed_s']+route_cost(g, public, route) > public['budget_s']+1e-8:
        raise ValueError('Planning spent return reserve')
    return route


def can_propose(g, public):
    used = np.zeros(len(g.ids), bool)
    used[public['visited_indices']] = True
    move = np.linalg.norm(g.xy-g.xy[public['position_index']], axis=1)/public['speed_m_s']
    home = np.linalg.norm(g.xy-g.xy[g.start_index], axis=1)/public['speed_m_s']
    return bool(np.any((~used) & (public['elapsed_s']+public['planning_latency_s']+move
                                +public['observe_s']+home <= public['budget_s']+1e-9)))


def gain_moments(g, public, old_route, proposal, likelihood):
    """Moments of the future correct-detection count difference under the belief.

Shared cells cancel. Conditional on a block's hot/cold latent variable, unvisited
cell detections are independent Bernoulli(se * target_rate). Correlation between
different cells in the same block is retained through the latent variable.
The result can be misspecified when the supplied prior/model is wrong.
"""
    if set(public) != PUBLIC_KEYS:
        raise ValueError('Unexpected public fields')
    se, fp = likelihood
    if not 0 <= fp < se <= 1:
        raise ValueError('Invalid public likelihood')
    q = np.asarray(public['belief_hot'], float)
    if q.shape != (len(g.block_ids),) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Invalid belief')
    visited = set(public['visited_indices'])
    for route in (old_route, proposal):
        if len(route) != len(set(route)) or visited & set(route):
            raise ValueError('Only unique unvisited future cells allowed')
        if any(type(i) is not int or not 0 <= i < len(g.ids) for i in route):
            raise ValueError('Invalid future cell')
    old, new = set(old_route), set(proposal)
    added, removed = sorted(new-old), sorted(old-new)
    plus = np.bincount(g.blocks[added], minlength=len(q))
    minus = np.bincount(g.blocks[removed], minlength=len(q))
    signed, count = plus-minus, plus+minus
    cold, hot = .02*se, .6*se
    conditional = (1-q)*cold*(1-cold)+q*hot*(1-hot)
    latent = q*(1-q)*(hot-cold)**2
    mean = float(np.sum(signed*((1-q)*cold+q*hot)))
    variance = float(np.sum(count*conditional+signed**2*latent))
    return {'mean_gain': mean, 'variance_gain': variance, 'sd_gain': math.sqrt(max(0., variance)),
            'added_cells': len(added), 'removed_cells': len(removed), 'shared_cells': len(new & old)}


def accept_proposal(policy, moments):
    if policy in ('always', 'periodic_5'):
        return True
    if policy == 'gain_5':
        return moments['mean_gain'] > EPS
    if policy == 'one_sd_5':
        return moments['mean_gain']-moments['sd_gain'] > EPS
    raise ValueError('No candidate rule for static policy')


def paid_attempt(session, planner, policy):
    before = session.public()
    if not can_propose(session.g, before):
        raise ValueError('No affordable candidate inspection')
    old = list(before['remaining_route'])
    # The only solver call charges 1s even if its proposal is later rejected.
    session.plan(planner, 'feedback')
    after = session.public()
    proposed = list(after['remaining_route'])
    moments = gain_moments(session.g, before, old, proposed, session.environment.world.likelihood)
    accepted = accept_proposal(policy, moments)
    retained = feasible_old_suffix(session.g, after, old)
    chosen = proposed if accepted else retained
    session.set_route(chosen)
    decision = {'after_observations': len(before['history']), 'elapsed_before_s': before['elapsed_s'],
                'old_route': old, 'proposed_route': proposed, 'retained_feasible_route': retained,
                'selected_route': chosen, 'accepted': accepted, 'route_changed': chosen != old,
                'dropped_tail_if_rejected': len(old)-len(retained), **moments}
    # Keep the planned proposal and actual selected suffix explicit in the ledger.
    session.state['planning_events'][-1]['acceptance'] = decision
    return decision


def execute(g, world, policy, planner, budget_s=1200):
    if policy not in POLICIES:
        raise ValueError('Unknown trigger policy')
    g = public_geometry(g, policy)
    session = Session(g, world, budget_s)
    session.plan(planner, 'feedback')
    decisions, skipped = [], []
    dynamic = policy not in ('pre_static', 'post_static')
    while session.state['remaining_route']:
        session.step()
        due = dynamic and (policy == 'always' or len(session.state['history']) % PERIOD == 0)
        if due:
            if can_propose(g, session.public()):
                decisions.append(paid_attempt(session, planner, policy))
            else:
                skipped.append(len(session.state['history']))
    session.close()
    return {'policy': policy, 'budget_s': budget_s, 'final_public': session.public(),
            'decisions': decisions, 'skipped_unaffordable_checkpoints': skipped,
            'measured_planner_wall_s': session.planner_wall_s}


def score(result, world):
    """Evaluator-only: never returned to the route decision functions."""
    s = result['final_public']
    target = world.targets
    h = s['history']
    arrived = [x for x in h if target[x['cell_index']]]
    detected = [x for x in arrived if x['observed_positive']]
    decisions = result['decisions']
    total = int(target.sum())
    return {'detected_target_cells': len(detected), 'visited_target_cells': len(arrived),
            'true_target_cells': total, 'observations': len(h),
            'detected_target_recall': len(detected)/total if total else None,
            'visited_target_recall': len(arrived)/total if total else None,
            'false_positive_observations': sum(not target[x['cell_index']] and x['observed_positive'] for x in h),
            'first_target_arrival_s': arrived[0]['arrival_s'] if arrived else None,
            'first_correct_detection_s': detected[0]['observation_complete_s'] if detected else None,
            'first_detection_or_budget_s': detected[0]['observation_complete_s'] if detected else s['budget_s'],
            'no_correct_detection': not bool(detected), 'total_distance_m': s['distance_m'],
            'total_time_s': s['elapsed_s'], 'planning_calls': len(s['planning_events']),
            'planning_attempts': len(decisions), 'accepted_proposals': sum(d['accepted'] for d in decisions),
            'rejected_proposals': sum(not d['accepted'] for d in decisions),
            'route_changes': sum(d['route_changed'] for d in decisions),
            'tail_drops_on_rejection': sum(d['dropped_tail_if_rejected'] for d in decisions if not d['accepted']),
            'skipped_unaffordable_checkpoints': len(result['skipped_unaffordable_checkpoints'])}
