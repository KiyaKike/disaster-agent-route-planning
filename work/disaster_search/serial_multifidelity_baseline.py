"""Cost-compatible two-stage MF-ENS-inspired control, NOT faithful MF-ENS.

After a putative action: optional batch of currently unseen surveys outside
the current optimal batch, then an outcome-adapted costed batch. Each fantasy
survey and upgrade is charged. No future cache, truth or instance ID is used.
"""
from functools import lru_cache
import json
import math

from cross_cell_count_smoke import CELLS
from run_budget_planning import EPS
from run_utility_tie_controls import expanded_values

POLICIES = ["serial_one_stage", "serial_two_stage"]


def decrement(counts, group, status):
    result = list(counts)
    index = 3 * group + status
    if result[index] <= 0:
        raise ValueError("Unavailable class action")
    result[index] -= 1
    return tuple(result)


class TwoStageModel:
    """Exchangeability-compressed finite state lookahead under frozen means.

    Only the tail optimization has an exact claim. Overall lookahead remains
    approximate: exploratory membership uses individual exchange values, and
    no future survey-upgrade chains occur inside the final batch itself.
    """
    def __init__(self, classes, native_probability, cost, cache_size=50000):
        self.classes = tuple((tuple(reward), float(prob)) for reward, prob in classes)
        self.native = float(native_probability)
        self.cost = cost
        if cost not in [1, 2, 4] or not 0 <= self.native <= 1:
            raise ValueError("Frozen costs and quality mixture required")
        for reward, probability in self.classes:
            if len(reward) != 3 or not 0 <= probability <= 1 or reward[0] is None:
                raise ValueError("Invalid public class")
            if any(v is not None and (not math.isfinite(v) or v < 0) for v in reward):
                raise ValueError("Nonnegative finite expected count required")
        # Bounded caches, one instance per scene/config; no unbounded session data.
        self.tail = lru_cache(maxsize=cache_size)(self._tail)
        self.after_surveys = lru_cache(maxsize=cache_size)(self._after_surveys)
        self.future = lru_cache(maxsize=cache_size)(self._future)
        self.q = lru_cache(maxsize=cache_size)(self._q)

    def _tail(self, counts, budget):
        if budget < 0 or len(counts) != 3 * len(self.classes) or any(n < 0 for n in counts):
            raise ValueError("Invalid count state/budget")
        survey, upgrade = [], []
        for group, (rewards, _) in enumerate(self.classes):
            for status, reward in enumerate(rewards):
                if reward is None:
                    continue
                target = survey if status == 0 or self.cost == 1 else upgrade
                target.extend((reward, group, status) for _ in range(counts[3 * group + status]))
        # Stable class/status ordering only resolves final-batch exact ties;
        # online action ties use the existing matched cell permutations.
        survey.sort(key=lambda r: (-r[0], r[1], r[2]))
        upgrade.sort(key=lambda r: (-r[0], r[1], r[2]))
        sp, up = [0.], [0.]
        for row in survey:
            sp.append(sp[-1] + row[0])
        for row in upgrade:
            up.append(up[-1] + row[0])
        best, chosen = -1., (0,) * len(counts)
        for k in range(min(len(upgrade), budget // self.cost) + 1):
            n = min(len(survey), budget - k * self.cost)
            value = up[k] + sp[n]
            # Ascending k means exact ties prefer fewer cost>1 upgrades.
            if value > best:
                best = value
                selected = [0] * len(counts)
                for _, group, status in survey[:n] + upgrade[:k]:
                    selected[3 * group + status] += 1
                chosen = tuple(selected)
        return best, chosen

    def survey_branches(self, counts, group):
        remaining = decrement(counts, group, 0)
        positive = self.classes[group][1]
        result = []
        for status, probability in [(3, self.native), (1, (1 - self.native) * (1 - positive)),
                                    (2, (1 - self.native) * positive)]:
            if not probability:
                continue
            branch = list(remaining)
            if status != 3:
                branch[3 * group + status] += 1
            result.append((probability, tuple(branch)))
        return result

    def _after_surveys(self, counts, batch, budget):
        """All exploration memberships fixed before ANY fantasy result."""
        if not batch:
            return self.tail(counts, budget)[0]
        group, rest = batch[0], batch[1:]
        return sum(p * self.after_surveys(branch, rest, budget) for p, branch in self.survey_branches(counts, group))

    def exchange_score(self, counts, group, budget):
        # Query cost is accounted for using budget-1. Compare with the same
        # candidate removed; only its newly unlocked conditional upgrade can
        # improve the costed final batch. This adapts MF-ENS's p* exchange.
        if budget <= self.cost or self.native == 1:
            return 0.
        without = decrement(counts, group, 0)
        base = self.tail(without, budget - 1)[0]
        revealed = sum(p * self.tail(branch, budget - 1)[0] for p, branch in self.survey_branches(counts, group))
        return max(0., revealed - base)

    def exploration_order(self, counts, budget):
        maximum = min(self.cost, max(0, budget - self.cost))
        if not maximum or self.native == 1:
            return ()
        _, selected = self.tail(counts, budget)
        candidates = []
        for group in range(len(self.classes)):
            outside = counts[3 * group] - selected[3 * group]
            if outside <= 0:
                continue
            score = self.exchange_score(counts, group, budget)
            if score > EPS:
                candidates.extend((score, group) for _ in range(min(outside, maximum)))
        candidates.sort(key=lambda v: (-v[0], v[1]))
        return tuple(g for _, g in candidates[:maximum])

    def _future(self, counts, budget):
        best, _ = self.tail(counts, budget)
        chosen = ()
        batch = self.exploration_order(counts, budget)
        immediate = 0.
        for length, group in enumerate(batch, 1):
            immediate += self.classes[group][0][0]
            prefix = batch[:length]
            value = immediate + self.after_surveys(counts, prefix, budget - length)
            if value > best + EPS:
                best, chosen = value, prefix
        # Empty prefix always available: optional exploration has nonnegative
        # surrogate improvement, without guaranteeing real discovery gains.
        return best, chosen

    def _q(self, counts, remaining, group, status):
        cost = 1 if status == 0 else self.cost
        reward = self.classes[group][0][status]
        if reward is None or cost > remaining:
            raise ValueError("Ineligible hypothetical action")
        budget = remaining - cost
        branches = self.survey_branches(counts, group) if status == 0 else [(1., decrement(counts, group, status))]
        one_stage = reward + sum(p * self.tail(s, budget)[0] for p, s in branches)
        two_stage = reward + sum(p * self.future(s, budget)[0] for p, s in branches)
        expected_exploration = sum(p * len(self.future(s, budget)[1]) for p, s in branches)
        gain = max(0., two_stage - one_stage)
        return one_stage, gain if gain > EPS else 0., expected_exploration


def compress_inputs(inputs, cost):
    keys = [(tuple(reward), probability) for reward, probability in zip(inputs["reward"], inputs["blur_positive"])]
    unique = sorted(set(keys), key=lambda x: json.dumps(x, separators=(",", ":")))
    membership = [unique.index(key) for key in keys]
    return TwoStageModel(unique, inputs["native_survey_probability"], cost), membership


def compressed_state(state, membership, groups):
    counts = [0] * (3 * groups)
    for status, group in zip(state, membership):
        if status != 3:
            counts[3 * group + status] += 1
    return tuple(counts)


def choose(state, inputs, remaining, cost, ranks, policy, model, membership):
    if policy not in POLICIES:
        raise ValueError("Unknown serial control")
    counts = compressed_state(state, membership, len(model.classes))
    original = expanded_values(state, inputs, remaining, cost, "batch_tail")
    best = None
    for i, original_q, greedy in original:
        gain, expected_batch = 0., 0.
        if policy == "serial_two_stage":
            compressed_q, gain, expected_batch = model.q(counts, remaining, membership[i], state[i])
            if abs(compressed_q - original_q) > 1e-9:
                raise ValueError("Compressed one-stage value differs from frozen baseline")
        value = original_q + gain
        # Anchor h=0 to the frozen arithmetic and original tie semantics.
        if best is None or value > best[1] + EPS or (abs(value - best[1]) <= EPS and
                (greedy > best[2] or (greedy == best[2] and ranks[i] < ranks[best[0]]))):
            best = (i, value, greedy, gain, expected_batch)
    return best


def run_episode(environment, inputs, budget, cost, ranks, policy, model, membership):
    state, trace, spent = [0] * len(membership), [], 0
    diagnostics = {"selected_actions_with_positive_two_stage_gain": 0,
        "sum_selected_surrogate_gain": 0., "sum_selected_expected_fantasy_surveys": 0.}
    while spent < budget:
        selected = choose(state, inputs, budget - spent, cost, ranks, policy, model, membership)
        if selected is None:
            break
        i, q, _, gain, expected_batch = selected
        kind = "survey" if state[i] == 0 else "upgrade"
        action_cost = 1 if kind == "survey" else cost
        result = environment.observe(CELLS[i], kind)
        if result["cost"] != action_cost:
            raise ValueError("Paid environment cost differs")
        spent += action_cost
        trace.append({"cell_id": CELLS[i], "kind": kind, "mode": result["mode"],
            "declared_target": result["declared_target"], "cost": action_cost,
            "cumulative_cost": spent, "utility": q})
        diagnostics["selected_actions_with_positive_two_stage_gain"] += gain > EPS
        diagnostics["sum_selected_surrogate_gain"] += gain
        diagnostics["sum_selected_expected_fantasy_surveys"] += expected_batch
        state[i] = 3 if result["mode"] == "native" else 2 if result["declared_target"] else 1
    return trace, diagnostics
