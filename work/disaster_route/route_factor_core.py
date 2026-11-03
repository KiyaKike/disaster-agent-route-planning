"""Public-information route solvers for the solver x stopping-rule experiment."""
import time
from collections import OrderedDict

import numpy as np

from route_core import Config, POLICIES, NAMES, greedy_route, update_belief


def path_distance(distances, start, home, route):
    nodes = [start, *route, home]
    return float(sum(distances[a, b] for a, b in zip(nodes, nodes[1:])))


def two_opt(distances, start, home, route, max_moves=8):
    """At most eight best improving reversals; preserve the selected node set."""
    route = list(route)
    for _ in range(max_moves):
        m = len(route)
        if m < 2:
            break
        nodes = np.array([start, *route, home])
        left, first = nodes[:-2], nodes[1:-1]
        last, right = nodes[1:-1], nodes[2:]
        delta = distances[left[:, None], last[None, :]] + distances[first[:, None], right[None, :]]
        delta -= distances[left, first][:, None] + distances[last, right][None, :]
        delta[~np.triu(np.ones((m, m), bool), 1)] = np.inf
        i, j = np.unravel_index(int(np.argmin(delta)), delta.shape)
        if delta[i, j] >= -1e-8:
            break
        route[i:j+1] = reversed(route[i:j+1])
    return route


def insertion_fill(distances, start, home, visited, rewards, elapsed, config, latency, limit, seed=()):
    """Insert the largest expected reward per marginal closed-route time."""
    route = list(seed)
    used = visited.copy()
    if route:
        used[route] = True
    total_distance = path_distance(distances, start, home, route)
    while len(route) < limit:
        candidates = np.flatnonzero(~used)
        if not len(candidates):
            break
        left = np.array([start, *route], dtype=int)
        right = np.array([*route, home], dtype=int)
        delta = distances[candidates[:, None], left[None, :]] + distances[candidates[:, None], right[None, :]]
        delta -= distances[left, right][None, :]
        delta = np.maximum(delta, 0.)
        extra = delta / config.speed_m_s + config.observe_s + latency
        current = elapsed + total_distance / config.speed_m_s + len(route) * (config.observe_s + latency)
        valid = current + extra <= config.budget_s + 1e-9
        if not valid.any():
            break
        utility = np.where(valid, rewards[candidates, None] / extra, -np.inf)
        row, slot = np.unravel_index(int(np.argmax(utility)), utility.shape)
        index = int(candidates[row])
        route.insert(int(slot), index)
        used[index] = True
        total_distance += float(delta[row, slot])
    return route


class Planner:
    """Identical memoization rules for both solvers; cache keys contain no truth."""
    def __init__(self, max_cache=20000):
        self.cache = OrderedDict()
        self.geometry = {}
        self.max_cache = max_cache

    def solve(self, g, solver, position, visited, elapsed, belief, config, latency, limit):
        if solver not in ('greedy', 'insertion_2opt'):
            raise ValueError('Unknown solver')
        key = (g.region, solver, position.tobytes(), visited.tobytes(), float(elapsed), belief.tobytes(),
               config.budget_s, config.speed_m_s, config.observe_s, latency, limit)
        before = time.perf_counter()
        if key in self.cache:
            result = self.cache.pop(key)
            self.cache[key] = result
            return list(result), {'cache_hit': True, 'wall_s': time.perf_counter()-before, 'solver_compute_s': 0.}
        if solver == 'greedy':
            answer = greedy_route(g, position, visited, elapsed, belief, config, latency, limit)
        else:
            answer = self.strong(g, position, visited, elapsed, belief, config, latency, limit)
        wall = time.perf_counter()-before
        self.cache[key] = tuple(answer)
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return answer, {'cache_hit': False, 'wall_s': wall, 'solver_compute_s': wall}

    def strong(self, g, position, visited, elapsed, belief, config, latency, limit):
        if not limit:
            return []
        if g.region not in self.geometry:
            distance = np.linalg.norm(g.xy[:, None, :] - g.xy[None, :, :], axis=2)
            self.geometry[g.region] = (distance, {point.tobytes(): i for i, point in enumerate(g.xy)})
        distances, indices = self.geometry[g.region]
        start, home = indices[position.tobytes()], g.start_index
        rewards = .02 + .58 * belief[g.blocks]
        seed = greedy_route(g, position, visited, elapsed, belief, config, latency, limit)
        candidates = []
        for initial in (seed, ()):
            route = insertion_fill(distances, start, home, visited, rewards, elapsed, config, latency, limit, initial)
            route = two_opt(distances, start, home, route)
            route = insertion_fill(distances, start, home, visited, rewards, elapsed, config, latency, limit, route)
            candidates.append(route)
        # Retaining the old greedy node set guarantees no lower planned prior reward.
        return min(candidates, key=lambda route: (-float(rewards[route].sum()), path_distance(distances, start, home, route), tuple(route)))


def execute(g, policy, config, observe, solver, planner):
    if policy not in POLICIES:
        raise ValueError('Unknown policy')
    static_policy = policy in ('pre_static', 'post_static')
    belief = g.prior_pre.copy() if policy == 'pre_static' else g.prior_post.copy()
    visited = np.zeros(len(g.ids), bool)
    home = g.xy[g.start_index]
    position = home.copy()
    elapsed = travel = planning_wall = compute_wall = 0.
    calls = hits = 0
    steps, events, static = [], [], []

    def plan(latency, limit):
        nonlocal planning_wall, compute_wall, calls, hits
        route, stats = planner.solve(g, solver, position, visited, elapsed, belief, config, latency, limit)
        calls += 1
        hits += int(stats['cache_hit'])
        planning_wall += stats['wall_s']
        compute_wall += stats['solver_compute_s']
        events.append({'after_observations': len(steps), 'at_cell': g.ids[g.start_index] if not steps else steps[-1]['cell_id'],
                       'plan_latency_s': config.planning_latency_s,
                       'remaining_budget_s': config.budget_s-elapsed-(0 if static_policy else latency),
                       'first_five_suffix_ids': [g.ids[i] for i in route[:5]], 'planned_suffix_visits': len(route),
                       'planned_suffix_indices': route,
                       'planned_expected_targets': float((.02+.58*belief[g.blocks[route]]).sum()) if route else 0.,
                       'cached_public_state': stats['cache_hit']})
        return route

    if static_policy and config.planning_latency_s <= config.budget_s:
        elapsed += config.planning_latency_s
        static = plan(0., config.max_visits)
    while len(steps) < config.max_visits:
        if static_policy:
            if len(steps) >= len(static):
                break
            index, latency = static[len(steps)], 0.
        else:
            move = np.linalg.norm(g.xy-position, axis=1)/config.speed_m_s
            reserve = np.linalg.norm(g.xy-home, axis=1)/config.speed_m_s
            feasible = (~visited) & (elapsed+config.planning_latency_s+move+config.observe_s+reserve <= config.budget_s+1e-9)
            if not feasible.any():
                break
            suffix = plan(config.planning_latency_s, config.max_visits-len(steps))
            if not suffix:
                raise ValueError('Feasible action missing')
            index, latency = suffix[0], config.planning_latency_s
        before = elapsed
        elapsed += latency
        leg = float(np.linalg.norm(g.xy[index]-position))
        arrival = elapsed+leg/config.speed_m_s
        completion = arrival+config.observe_s
        reserve = float(np.linalg.norm(g.xy[index]-home))/config.speed_m_s
        if completion+reserve > config.budget_s+1e-8:
            raise ValueError('No return reserve')
        positive = bool(observe(index))
        block = int(g.blocks[index])
        prior = float(belief[block])
        if policy == 'online':
            belief[block] = update_belief(prior, positive, *observe.likelihood)
        elapsed, position = completion, g.xy[index].copy()
        travel += leg
        visited[index] = True
        steps.append({'step': len(steps)+1, 'cell_index': index, 'cell_id': g.ids[index], 'block_index': block,
                      'elapsed_before_step_s': before, 'planning_latency_s': latency, 'leg_m': leg,
                      'arrival_s': arrival, 'observation_complete_s': completion, 'cumulative_travel_m': travel,
                      'return_reserve_s': reserve, 'observed_positive': positive,
                      'block_belief_before': prior, 'block_belief_after': float(belief[block])})
    return_m = float(np.linalg.norm(position-home))
    total = elapsed+return_m/config.speed_m_s
    if total > config.budget_s+1e-8:
        raise ValueError('Over budget')
    return {'policy': policy, 'name': NAMES[policy], 'region': g.region, 'solver': solver,
            'start_id': g.ids[g.start_index], 'budget_s': config.budget_s, 'visits': len(steps),
            'steps': steps, 'planning_events': events, 'planning_calls': calls,
            'simulated_planning_latency_s': config.planning_latency_s*calls,
            'measured_planning_wall_s': planning_wall, 'uncached_solver_compute_s': compute_wall, 'public_plan_cache_hits': hits,
            'return_m': return_m, 'total_distance_m': travel+return_m, 'total_time_s': total,
            'closed_route_ids': [g.ids[g.start_index]]+[s['cell_id'] for s in steps]+[g.ids[g.start_index]],
            'movement_seconds': (travel+return_m)/config.speed_m_s, 'observation_seconds': len(steps)*config.observe_s}
