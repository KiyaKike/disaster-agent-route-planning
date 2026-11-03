import itertools
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cache_budget_oracle import evaluate, solve


def cell(native, blurred=0, mode="blurred"):
    return {"native": native, "blurred": blurred, "survey_mode": mode}


class CacheOracleTests(unittest.TestCase):
    def test_matches_exhaustive_unique_coverage_with_prerequisites(self):
        rng = random.Random(122)
        for cost in [1, 2, 4]:
            for _ in range(4):
                cells = [cell(rng.randrange(1 << 7), rng.randrange(1 << 7), rng.choice(["native", "blurred"])) for i in range(5)]
                for allow in [False, True]:
                    for budget in [1, 3, 5]:
                        options = [[0, 1, 2] if allow and c["survey_mode"] == "blurred" else [0, 1] for c in cells]
                        feasible = []
                        for levels in itertools.product(*options):
                            if sum(0 if k == 0 else 1 if k == 1 else 1 + cost for k in levels) <= budget:
                                feasible.append(evaluate(cells, levels, budget, cost, allow)["found"])
                        optimum = max(feasible)
                        result = solve(cells, budget, cost, allow)
                        self.assertTrue(result["certified_optimal"])
                        self.assertEqual((result["lower"], result["upper"]), (optimum, optimum))

    def test_upgrade_requires_initial_survey_and_full_price(self):
        cells = [cell(0b111, 0)]
        self.assertEqual(solve(cells, 2, 2)["upper"], 0)
        self.assertEqual(solve(cells, 3, 2)["lower"], 3)
        self.assertEqual(solve(cells, 32, 2, False)["upper"], 0)
        with self.assertRaises(ValueError):
            evaluate([cell(3, 0, "native")], [2], 3, 2, True)

    def test_keeps_blurred_only_matches_and_cross_cell_union(self):
        cells = [cell(0b10, 0b01), cell(0b10, 0, "native")]
        result = solve(cells, 3, 2)
        self.assertEqual(result["lower"], 2)
        duplicated = solve([cell(7, 0, "native"), cell(7, 0, "native")], 2, 2)
        self.assertEqual(duplicated["lower"], 3)

    def test_timeout_without_incumbent_is_not_called_exact(self):
        cells = [cell(1, 0, "native"), cell(2, 0, "native"), cell(4, 0, "native")]
        fake = SimpleNamespace(status=1, message="time limit", x=None, mip_dual_bound=-2.7)
        with patch("cache_budget_oracle.milp", return_value=fake):
            r = solve(cells, 1, 2)
        self.assertEqual((r["lower"], r["upper"]), (1, 2))
        self.assertFalse(r["certified_optimal"])
        self.assertEqual(r["certificate"], "unclosed_dual_or_union_bound")

    def test_timeout_without_dual_uses_valid_union_upper(self):
        cells = [cell(1, 0, "native"), cell(2, 0, "native")]
        fake = SimpleNamespace(status=1, message="time limit", x=None, mip_dual_bound=None)
        with patch("cache_budget_oracle.milp", return_value=fake):
            r = solve(cells, 1, 2)
        self.assertEqual((r["lower"], r["upper"]), (1, 2))

    def test_bad_dual_cannot_silently_override_feasible_solution(self):
        cells = [cell(1, 0, "native"), cell(2, 0, "native")]
        fake = SimpleNamespace(status=1, message="time limit", x=None, mip_dual_bound=0.)
        with patch("cache_budget_oracle.milp", return_value=fake):
            with self.assertRaises(ValueError):
                solve(cells, 1, 2)

    def test_empty_and_zero_budget_are_certified_without_solver(self):
        for cells, budget in [([cell(0)], 32), ([cell(7)], 0)]:
            r = solve(cells, budget, 2)
            self.assertEqual((r["lower"], r["upper"]), (0, 0))
            self.assertFalse(r["solver_called"])


if __name__ == "__main__":
    unittest.main()
