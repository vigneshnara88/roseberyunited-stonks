import unittest

from stonks_solver import HOME_YEAR, simulate, solve_case


def score(case, actions):
    result = simulate(case, actions)
    if not result.ok:
        raise AssertionError(result.reason)
    return result.capital - case["capital"]


class StonksSolverTests(unittest.TestCase):
    def test_sample(self):
        case = {
            "energy": 2,
            "capital": 500,
            "timeline": {
                "2037": {"Apple": {"price": 100, "qty": 10}},
                "2036": {"Apple": {"price": 10, "qty": 50}},
            },
        }
        actions = solve_case(case)
        self.assertEqual(actions, ["j-2037-2036", "b-Apple-50",
                                   "j-2036-2037", "s-Apple-50"])
        self.assertEqual(score(case, actions), 4500)

    def test_no_profit_returns_empty_valid_plan(self):
        case = {
            "energy": 8,
            "capital": 100,
            "timeline": {
                str(HOME_YEAR): {"A": {"price": 10, "qty": 100}},
                "2035": {"A": {"price": 10, "qty": 100}},
            },
        }
        self.assertEqual(solve_case(case), [])


    def test_intermediate_sale_can_reinvest(self):
        case = {
            "energy": 4,
            "capital": 10,
            "timeline": {
                "2037": {"A": {"price": 10, "qty": 10},
                         "B": {"price": 50, "qty": 0}},
                "2036": {"A": {"price": 1, "qty": 10},
                         "B": {"price": 50, "qty": 10}},
                "2035": {"A": {"price": 5, "qty": 0},
                         "B": {"price": 1, "qty": 100}},
            },
        }
        actions = solve_case(case)
        self.assertGreaterEqual(score(case, actions), 2490)


    def test_direct_non_home_peak(self):
        case = {
            "energy": 4,
            "capital": 100,
            "timeline": {
                "2037": {"A": {"price": 5, "qty": 0}},
                "2036": {"A": {"price": 100, "qty": 0}},
                "2035": {"A": {"price": 1, "qty": 100}},
            },
        }
        actions = solve_case(case)
        self.assertEqual(score(case, actions), 9900)


if __name__ == "__main__":
    unittest.main()
