"""Exact controlled inspection study, not a physical UAV or disaster-data model.

An observation completes the local inspection. Reward is one correctly detected
positive, paid once. A public generative distribution is distinct from the hidden
world used only by the independent exhaustive evaluator.
"""
from dataclasses import dataclass
from itertools import permutations, product
import math

HOME = (0, 0)
SITES = ((1, 0), (2, 0), (0, 2))
INITIAL_PLAN_COST = 1
OBSERVATION_COST = 1
WORLDS = tuple(product((0, 1), repeat=3))


def distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def world_distribution(rho):
    """All marginals .5. Mixture: independent, or (Z,Z,1-Z)."""
    if not 0 <= rho <= 1:
        raise ValueError("rho must be a mixture weight")
    return tuple((1-rho)/8 + (rho/2 if x[0] == x[1] and x[2] == 1-x[0] else 0)
                 for x in WORLDS)


def routes(available):
    yield ()
    for n in range(1, len(available)+1):
        yield from permutations(available, n)


def closed_cost(route, start=HOME):
    pos = start
    total = 0
    for i in route:
        total += distance(pos, SITES[i]) + OBSERVATION_COST
        pos = SITES[i]
    return total + distance(pos, HOME)


def expected_reward(route, belief, q):
    return sum(q * sum(p*x[i] for p, x in zip(belief, WORLDS)) for i in route)


def best_route(belief, q, available, remaining, start=HOME):
    feasible = [r for r in routes(available) if closed_cost(r, start) <= remaining]
    if not feasible:
        raise ValueError("Return is infeasible")
    route = min(feasible, key=lambda r: (-round(expected_reward(r, belief, q), 12), closed_cost(r, start), r))
    return route, expected_reward(route, belief, q)


def posterior(belief, q, site, observation):
    weights = [p * (q if x[site] == observation else 1-q) for p, x in zip(belief, WORLDS)]
    probability = sum(weights)
    if probability == 0:
        raise ValueError("Impossible observation")
    return probability, tuple(p/probability for p in weights)


@dataclass(frozen=True)
class Policy:
    kind: str
    route: tuple = ()
    first: int | None = None
    tails: tuple = ((), ())
    extra_fee: int = 0
    predicted_value: float = 0.


def static_policy(belief, q, budget, fee=0):
    route, value = best_route(belief, q, (0, 1, 2), budget-INITIAL_PLAN_COST-fee)
    return Policy("static", route=route, extra_fee=fee, predicted_value=value)


def forced_one_update(belief, q, budget, fee):
    candidates = []
    for first in range(3):
        prefix_cost = INITIAL_PLAN_COST + distance(HOME, SITES[first]) + OBSERVATION_COST + fee
        remaining = budget-prefix_cost
        if remaining < distance(SITES[first], HOME):
            continue
        value = expected_reward((first,), belief, q)
        tails = []
        for y in (0, 1):
            probability, post = posterior(belief, q, first, y)
            tail, tail_value = best_route(post, q, tuple(i for i in range(3) if i != first), remaining, SITES[first])
            tails.append(tail)
            value += probability*tail_value
        candidates.append(Policy("one_update", first=first, tails=tuple(tails), extra_fee=fee, predicted_value=value))
    if not candidates:
        return Policy("static", extra_fee=fee, predicted_value=0.)
    return min(candidates, key=lambda p: (-round(p.predicted_value, 12), p.first, p.tails))


def make_policies(rho, q, budget, fee):
    if not .5 <= q <= 1:
        raise ValueError("Sensor correctness must be in [.5,1]")
    belief = world_distribution(rho)
    static = static_policy(belief, q, budget)
    charged_static = static_policy(belief, q, budget, fee)
    forced = forced_one_update(belief, q, budget, fee)
    # Exact known-model diagnostic, not a learned gate or unrestricted optimum.
    selected = forced if forced.predicted_value > static.predicted_value+1e-12 else static
    return {"static": static, "static_with_same_extra_fee": charged_static,
            "forced_one_update": forced, "best_static_or_one_update": selected}


def exhaustive_evaluate(policy, rho, q, budget):
    """Independent world/observation replay; planner never receives true x."""
    total = {"reward": 0., "false_positives": 0., "target_coverage": 0.,
             "cost": 0., "observations": 0., "extra_updates": 0.}
    probability_sum = 0.
    leaves = 0
    for world, prior in zip(WORLDS, world_distribution(rho)):
        for observations in WORLDS:
            p = prior
            for x, y in zip(world, observations):
                p *= q if x == y else 1-q
            if p == 0:
                continue
            leaves += 1
            probability_sum += p
            if policy.kind == "static":
                route = policy.route
            else:
                route = (policy.first,) + policy.tails[observations[policy.first]]
            if len(route) != len(set(route)):
                raise AssertionError("Repeated inspection reward")
            # A separate cost calculation from primitive execution steps.
            cost = INITIAL_PLAN_COST + policy.extra_fee
            pos = HOME
            reward = fp = covered = 0
            for i in route:
                cost += abs(pos[0]-SITES[i][0]) + abs(pos[1]-SITES[i][1]) + OBSERVATION_COST
                pos = SITES[i]
                reward += int(world[i] == 1 and observations[i] == 1)
                fp += int(world[i] == 0 and observations[i] == 1)
                covered += world[i]
            cost += abs(pos[0]-HOME[0]) + abs(pos[1]-HOME[1])
            if cost > budget:
                raise AssertionError("Movement/observation/planning/return budget violation")
            for key, value in {"reward":reward, "false_positives":fp, "target_coverage":covered,
                               "cost":cost, "observations":len(route),
                               "extra_updates":int(policy.kind == "one_update")}.items():
                total[key] += p*value
    if not math.isclose(probability_sum, 1, abs_tol=1e-12):
        raise AssertionError("World/observation mass is not normalized")
    if not math.isclose(total["reward"], policy.predicted_value, abs_tol=1e-10):
        raise AssertionError("Planner and independent exhaustive evaluator disagree")
    return {**total, "exact_positive_probability_leaves":leaves, "probability_sum":probability_sum}
