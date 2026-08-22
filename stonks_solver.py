"""Solver for the Time Travelling Stonks Man challenge.

The challenge server validates an action list, not an explanation, so this
module optimizes for two things:

* produce only actions that our local simulator can execute successfully;
* search several sensible travel/trading policies and return the best plan
  found for the concrete test case.

The strongest common case is buying under-priced historical lots and selling
them at 2037. The extra route policies cover less obvious cases where selling
at an intermediate year frees cash that can then be reinvested deeper in time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from typing import Any, Callable

HOME_YEAR = 2037
MAX_DEPTH_ROUTES = 8
MAX_PAIR_ROUTES = 15
MAX_TOTAL_ROUTES = 80
MAX_REPEAT_ROUTES = 10
MAX_REPEAT_CYCLES = 25
EXACT_KNAPSACK_CAPITAL = 2500
EXACT_KNAPSACK_UNITS = 120
FINAL_EXACT_CAPITAL = 20000
FINAL_EXACT_BLOCKS = 180
POSTPROCESS_ROUTE_POLICIES = frozenset({
    ("best_future", "roi", False, 1.0),
    ("best_future_nondom", "roi", False, 1.0),
    ("best_future", "roi", False, 1.05),
})
POSTPROCESS_ITERATIONS = 3


@dataclass(frozen=True)
class Lot:
    year: int
    stock: str
    price: int
    qty: int
    sell_index: int
    sell_year: int
    sell_price: int

    @property
    def profit(self) -> int:
        return self.sell_price - self.price

    @property
    def roi(self) -> float:
        return self.sell_price / self.price


def solve_cases(cases: list[dict[str, Any]]) -> list[list[str]]:
    bulk = len(cases) >= 25
    return [solve_case(case, allow_exact=not bulk, fast=bulk)
            for case in cases]


def solve_case(case: dict[str, Any], allow_exact: bool = True,
               fast: bool = False) -> list[str]:
    energy = int(case.get("energy", 0))
    capital = int(case.get("capital", 0))
    timeline = normalize_timeline(case.get("timeline") or {})

    if energy <= 0 or capital <= 0 or not timeline:
        return []

    large_case = energy >= 20
    search_fast = fast or large_case
    routes = candidate_routes(timeline, energy, fast=search_fast)
    policies = build_policies(fast=search_fast)
    best_actions: list[str] = []
    best_value = capital
    best_profit = 0
    seen: set[tuple[int, ...]] = set()

    for route in routes:
        key = tuple(route)
        if key in seen or route_cost(route) > energy:
            continue
        seen.add(key)
        for policy in policies:
            actions = plan_for_route(timeline, capital, route, policy)
            result = simulate(case, actions)
            if not result.ok:
                continue
            profit = result.capital - capital
            if result.capital > best_value or (
                result.capital == best_value and profit > best_profit
                and len(actions) < len(best_actions)
            ):
                best_value = result.capital
                best_profit = profit
                best_actions = actions
            if energy >= 20 and policy in POSTPROCESS_ROUTE_POLICIES:
                improved_actions, improved_result = improve_plan(
                    case, timeline, capital, energy, actions)
                if improved_result.ok and improved_result.capital > best_value:
                    best_value = improved_result.capital
                    best_profit = improved_result.capital - capital
                    best_actions = improved_actions

    cycle_actions = high_energy_cycle_plan(timeline, energy, capital)
    if cycle_actions:
        result = simulate(case, cycle_actions)
        if result.ok and result.capital > best_value:
            best_value = result.capital
            best_profit = result.capital - capital
            best_actions = cycle_actions

    bundle_actions = bundled_cycle_plan(timeline, energy, capital)
    if bundle_actions:
        result = simulate(case, bundle_actions)
        if result.ok and result.capital > best_value:
            best_value = result.capital
            best_profit = result.capital - capital
            best_actions = bundle_actions

    exact_allowed = allow_exact and not large_case

    final_actions = final_2037_exact_plan(case) if exact_allowed else None
    if final_actions is not None:
        result = simulate(case, final_actions)
        if result.ok and result.capital > best_value:
            best_value = result.capital
            best_profit = result.capital - capital
            best_actions = final_actions

    exact_actions = exact_search(case) if exact_allowed else None
    if exact_actions is not None:
        result = simulate(case, exact_actions)
        if result.ok and result.capital > best_value:
            best_value = result.capital
            best_profit = result.capital - capital
            best_actions = exact_actions

    if best_actions and energy >= 20:
        improved_actions, improved_result = improve_plan(
            case, timeline, capital, energy, best_actions)
        if improved_result.ok and improved_result.capital > best_value:
            best_value = improved_result.capital
            best_profit = improved_result.capital - capital
            best_actions = improved_actions

    return best_actions if best_profit > 0 else []


def improve_plan(case: dict[str, Any],
                 timeline: dict[int, dict[str, dict[str, int]]],
                 capital: int, energy: int,
                 actions: list[str]) -> tuple[list[str], SimResult]:
    best_actions = actions
    best_result = simulate(case, best_actions)
    if not best_result.ok:
        return best_actions, best_result

    for _ in range(POSTPROCESS_ITERATIONS):
        changed = False
        candidates = [slack_fill_plan(timeline, capital, best_actions, "roi")]
        if len(timeline) <= 12:
            candidates.append(slack_fill_plan(timeline, capital, best_actions,
                                              "profit"))
        candidates.append(detour_fill_plan(timeline, capital, energy,
                                           best_actions))
        candidates.append(delayed_sell_plan(timeline, capital, best_actions))

        for candidate_actions in candidates:
            if candidate_actions == best_actions:
                continue
            result = simulate(case, candidate_actions)
            if result.ok and result.capital > best_result.capital:
                best_actions = candidate_actions
                best_result = result
                changed = True
        if not changed:
            break
    return best_actions, best_result


def normalize_timeline(raw: dict[str, Any]) -> dict[int, dict[str, dict[str, int]]]:
    out: dict[int, dict[str, dict[str, int]]] = {}
    for year_key, stocks in raw.items():
        try:
            year = int(year_key)
        except (TypeError, ValueError):
            continue
        if year <= 0 or year > HOME_YEAR or not isinstance(stocks, dict):
            continue
        clean: dict[str, dict[str, int]] = {}
        for stock, quote in stocks.items():
            if not isinstance(quote, dict):
                continue
            try:
                price = int(quote.get("price", 0))
                qty = int(quote.get("qty", 0))
            except (TypeError, ValueError):
                continue
            if price > 0 and qty >= 0:
                clean[str(stock)] = {"price": price, "qty": qty}
        if clean:
            out[year] = clean
    return out


def candidate_routes(timeline: dict[int, dict[str, dict[str, int]]],
                     energy: int, fast: bool = False) -> list[list[int]]:
    years = sorted(y for y in timeline if y <= HOME_YEAR)
    data_years = sorted(set(years + [HOME_YEAR]), reverse=True)
    min_reachable = max(1, HOME_YEAR - energy)
    reachable = [y for y in data_years if y >= min_reachable]
    if HOME_YEAR not in reachable:
        reachable.insert(0, HOME_YEAR)

    routes: list[list[int]] = [[HOME_YEAR]]
    max_depth = energy // 2
    profitable_depths = promising_depths(timeline, reachable, max_depth)
    data_depths = [HOME_YEAR - y for y in reachable
                   if 0 <= HOME_YEAR - y <= max_depth]
    deepest_data_depth = max(data_depths or [0])
    if deepest_data_depth > 0:
        routes.append(round_trip_route(reachable, deepest_data_depth))
    if energy >= 20 and deepest_data_depth > 0:
        for depth in repeated_scan_depths(timeline, reachable, energy,
                                          max_depth):
            repeated = repeat_route(round_trip_route(reachable, depth), energy)
            if len(repeated) > 1:
                routes.append(repeated)
    if fast:
        return routes if len(routes) > 1 else [[HOME_YEAR]]

    # Sometimes a short round trip to 2037 grows cash early, then a second
    # deeper trip spends it better. Try compact two-trip combinations.
    for first in profitable_depths[:8]:
        remaining = energy - 2 * first
        if remaining < 2:
            continue
        for second in profitable_depths[:10]:
            if 2 * first + 2 * second <= energy:
                routes.append(compact_route(
                    round_trip_route(reachable, first)
                    + round_trip_route(reachable, second)[1:]))

    # Try direct buy/sell pair tours for non-2037 sell peaks.
    pair_routes = profitable_pair_routes(timeline, energy)
    routes.extend(pair_routes[:MAX_PAIR_ROUTES])
    if energy >= 20:
        for route in pair_routes[:MAX_REPEAT_ROUTES]:
            repeated = repeat_route(route, energy)
            if len(repeated) > len(route):
                routes.append(repeated)

    # Deterministic shortest plans first; longer speculative plans later.
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for route in routes:
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            unique.append(route)
        if len(unique) >= MAX_TOTAL_ROUTES:
            break
    return unique


def round_trip_route(reachable: list[int], depth: int) -> list[int]:
    oldest = HOME_YEAR - depth
    down = [y for y in sorted(set(reachable + [oldest]), reverse=True)
            if HOME_YEAR >= y >= oldest]
    up = list(reversed(down[:-1]))
    return compact_route(down + up)


def compact_route(route: list[int]) -> list[int]:
    out: list[int] = []
    for year in route:
        if out and out[-1] == year:
            continue
        out.append(year)
    if not out or out[0] != HOME_YEAR:
        out.insert(0, HOME_YEAR)
    if out[-1] != HOME_YEAR:
        out.append(HOME_YEAR)
    return out


def route_cost(route: list[int]) -> int:
    if not route:
        return 0
    return sum(abs(route[i] - route[i - 1]) for i in range(1, len(route)))


def repeat_route(route: list[int], energy: int) -> list[int]:
    cost = route_cost(route)
    if cost <= 0 or cost > energy:
        return route
    cycles = min(MAX_REPEAT_CYCLES, energy // cost)
    out = [HOME_YEAR]
    for _ in range(cycles):
        out.extend(route[1:])
    return compact_route(out)


def repeated_scan_depths(timeline: dict[int, dict[str, dict[str, int]]],
                         reachable: list[int], energy: int,
                         max_depth: int) -> list[int]:
    data_depths = sorted({
        HOME_YEAR - year
        for year in reachable
        if 0 < HOME_YEAR - year <= max_depth
    })
    if not data_depths:
        return []

    out: list[int] = []

    def add(depth: int) -> None:
        if depth > 0 and depth not in out:
            out.append(depth)

    add(data_depths[-1])

    for target_cycles in (7, 8):
        target_depth = energy // (2 * target_cycles)
        choices = [depth for depth in data_depths if depth <= target_depth]
        if choices:
            add(max(choices))

    potential: list[tuple[int, int]] = []
    for depth in data_depths:
        oldest = HOME_YEAR - depth
        sweep_years = [year for year in reachable if HOME_YEAR >= year >= oldest]
        score = 0
        for buy_year in sweep_years:
            for stock, quote in timeline.get(buy_year, {}).items():
                best_price = max(
                    (timeline.get(year, {}).get(stock, {}).get("price", 0)
                     for year in sweep_years),
                    default=0,
                )
                if best_price > quote["price"]:
                    score += (best_price - quote["price"]) * quote["qty"]
        if score > 0:
            potential.append((score, depth))
    for _, depth in sorted(potential, reverse=True)[:5]:
        add(depth)

    return out


def high_energy_cycle_plan(timeline: dict[int, dict[str, dict[str, int]]],
                           energy: int, starting_capital: int) -> list[str]:
    """Greedily compound through profitable cycles, returning home at the end."""
    if energy < 20:
        return []

    best_actions: list[str] = []
    best_value = starting_capital
    case = {
        "energy": energy,
        "capital": starting_capital,
        "timeline": timeline,
    }
    for mode in ("mixed", "efficiency", "profit"):
        actions = chain_cycle_plan(timeline, energy, starting_capital, mode)
        result = simulate(case, actions)
        if result.ok and result.capital > best_value:
            best_value = result.capital
            best_actions = actions
    return best_actions


def chain_cycle_plan(timeline: dict[int, dict[str, dict[str, int]]],
                     energy: int, starting_capital: int,
                     mode: str) -> list[str]:
    years = sorted(timeline)
    remaining = {(year, stock): quote["qty"]
                 for year, stocks in timeline.items()
                 for stock, quote in stocks.items()}
    actions: list[str] = []
    current_year = HOME_YEAR
    capital = starting_capital

    while energy > 0 and len(actions) < 5000:
        best: tuple[float, int, float, int, int, int, str, int] | None = None
        for buy_year in years:
            if abs(current_year - buy_year) + abs(buy_year - HOME_YEAR) > energy:
                continue
            for stock, quote in timeline[buy_year].items():
                buy_price = quote["price"]
                qty = min(remaining.get((buy_year, stock), 0),
                          capital // buy_price)
                if qty <= 0:
                    continue
                for sell_year in years:
                    sell_quote = timeline[sell_year].get(stock)
                    if not sell_quote or sell_quote["price"] <= buy_price:
                        continue
                    travel = (abs(current_year - buy_year)
                              + abs(buy_year - sell_year))
                    if travel <= 0 or travel + abs(sell_year - HOME_YEAR) > energy:
                        continue
                    profit = (sell_quote["price"] - buy_price) * qty
                    roi = sell_quote["price"] / buy_price
                    score = cycle_score(mode, profit, roi, qty, travel)
                    candidate = (
                        score,
                        profit,
                        roi,
                        -travel,
                        buy_year,
                        sell_year,
                        stock,
                        qty,
                    )
                    if best is None or candidate > best:
                        best = candidate

        if best is None:
            break

        _, _, _, _, buy_year, sell_year, stock, qty = best
        buy_price = timeline[buy_year][stock]["price"]
        sell_price = timeline[sell_year][stock]["price"]

        if current_year != buy_year:
            actions.append(jump_action(current_year, buy_year))
            energy -= abs(current_year - buy_year)
            current_year = buy_year
        actions.append(buy_action(stock, qty))
        capital -= buy_price * qty
        remaining[(buy_year, stock)] -= qty

        if current_year != sell_year:
            actions.append(jump_action(current_year, sell_year))
            energy -= abs(current_year - sell_year)
            current_year = sell_year
        actions.append(sell_action(stock, qty))
        capital += sell_price * qty

    if current_year != HOME_YEAR and abs(current_year - HOME_YEAR) <= energy:
        actions.append(jump_action(current_year, HOME_YEAR))
    return actions


def cycle_score(mode: str, profit: int, roi: float, qty: int,
                travel: int) -> float | tuple[float, ...]:
    if mode == "profit":
        return float(profit)
    if mode == "compound":
        return log(roi) * qty / travel
    if mode == "rate":
        return log(roi) / travel
    if mode == "roi":
        return (roi, profit / travel)
    if mode == "mixed":
        return (profit / travel) * (roi ** 0.5)
    return profit / travel


def bundled_cycle_plan(timeline: dict[int, dict[str, dict[str, int]]],
                       energy: int, starting_capital: int) -> list[str]:
    """Trade several same-leg opportunities at once to save travel energy."""
    if energy < 20:
        return []

    years = sorted(timeline)
    remaining = {(year, stock): quote["qty"]
                 for year, stocks in timeline.items()
                 for stock, quote in stocks.items()}
    actions: list[str] = []
    current_year = HOME_YEAR
    capital = starting_capital

    while energy > 0 and len(actions) < 5000:
        best: tuple[int, float, int, int, int,
                    list[tuple[tuple[str, int, int, int], int]]] | None = None
        for buy_year in years:
            to_buy = abs(current_year - buy_year)
            if to_buy + abs(buy_year - HOME_YEAR) > energy:
                continue
            for sell_year in years:
                travel = to_buy + abs(buy_year - sell_year)
                if travel <= 0 or travel + abs(sell_year - HOME_YEAR) > energy:
                    continue
                items: list[tuple[str, int, int, int]] = []
                for stock, quote in timeline[buy_year].items():
                    buy_price = quote["price"]
                    sell_quote = timeline[sell_year].get(stock)
                    if (sell_quote and sell_quote["price"] > buy_price
                            and buy_price <= capital
                            and remaining.get((buy_year, stock), 0) > 0):
                        items.append((stock, buy_price, sell_quote["price"],
                                      remaining[(buy_year, stock)]))
                chosen = choose_bundle(items, capital)
                profit = sum((item[2] - item[1]) * qty
                             for item, qty in chosen)
                if profit <= 0:
                    continue
                candidate = (profit, profit / travel, -travel, buy_year,
                             sell_year, chosen)
                if best is None or candidate > best:
                    best = candidate

        if best is None:
            break

        _, _, _, buy_year, sell_year, chosen = best
        if current_year != buy_year:
            actions.append(jump_action(current_year, buy_year))
            energy -= abs(current_year - buy_year)
            current_year = buy_year
        for item, qty in sorted(chosen, key=lambda choice: choice[0][0]):
            stock, buy_price, _, _ = item
            actions.append(buy_action(stock, qty))
            capital -= buy_price * qty
            remaining[(buy_year, stock)] -= qty
        if current_year != sell_year:
            actions.append(jump_action(current_year, sell_year))
            energy -= abs(current_year - sell_year)
            current_year = sell_year
        for item, qty in sorted(chosen, key=lambda choice: choice[0][0]):
            stock, _, sell_price, _ = item
            actions.append(sell_action(stock, qty))
            capital += sell_price * qty

    if current_year != HOME_YEAR and abs(current_year - HOME_YEAR) <= energy:
        actions.append(jump_action(current_year, HOME_YEAR))
    return actions


def choose_bundle(items: list[tuple[str, int, int, int]],
                  capital: int) -> list[tuple[tuple[str, int, int, int], int]]:
    chosen: list[tuple[tuple[str, int, int, int], int]] = []
    remaining_cash = capital
    for item in sorted(items, key=lambda x: (x[2] - x[1], x[2] / x[1], -x[1]),
                       reverse=True):
        _, buy_price, _, qty_available = item
        qty = min(qty_available, remaining_cash // buy_price)
        if qty <= 0:
            continue
        chosen.append((item, qty))
        remaining_cash -= buy_price * qty
    return chosen


def promising_depths(timeline: dict[int, dict[str, dict[str, int]]],
                     reachable: list[int], max_depth: int) -> list[int]:
    scored: list[tuple[int, int]] = []
    home_prices = timeline.get(HOME_YEAR, {})
    for y in reachable:
        depth = HOME_YEAR - y
        if depth <= 0 or depth > max_depth:
            continue
        score = 0
        for stock, quote in timeline.get(y, {}).items():
            hp = home_prices.get(stock, {}).get("price")
            if hp and hp > quote["price"]:
                score += (hp - quote["price"]) * quote["qty"]
        if score > 0:
            scored.append((score, depth))
    scored.sort(reverse=True)
    if not scored:
        return list(range(1, min(max_depth, MAX_DEPTH_ROUTES) + 1))
    return [d for _, d in scored]


def profitable_pair_routes(timeline: dict[int, dict[str, dict[str, int]]],
                           energy: int) -> list[list[int]]:
    min_reachable = max(1, HOME_YEAR - energy)
    years = sorted((y for y in timeline if y >= min_reachable), reverse=True)
    routes: list[tuple[float, list[int]]] = []
    for buy_year in years:
        for stock, quote in timeline[buy_year].items():
            buy_price = quote["price"]
            if quote["qty"] <= 0:
                continue
            for sell_year in years:
                sell_quote = timeline[sell_year].get(stock)
                if not sell_quote or sell_quote["price"] <= buy_price:
                    continue
                route = compact_route([HOME_YEAR, buy_year, sell_year, HOME_YEAR])
                cost = route_cost(route)
                if cost <= energy:
                    gain = (sell_quote["price"] - buy_price) / buy_price
                    routes.append((gain / max(1, cost), route))
    routes.sort(reverse=True, key=lambda x: x[0])
    return [r for _, r in routes]


Policy = tuple[str, str, bool, float]


def build_policies(fast: bool = False) -> list[Policy]:
    if fast:
        return [
            ("best_future", "roi", False, 1.0),
            ("best_future", "cheap_roi", False, 1.0),
            ("best_future", "roi", False, 1.01),
            ("best_future", "roi", False, 1.05),
            ("best_future", "rate", False, 1.0),
            ("best_future_nondom", "roi", False, 1.0),
            ("earliest_profit", "rate", False, 1.0),
            ("final_2037", "roi", False, 1.0),
        ]
    return [
        ("best_future", "roi", False, 1.0),
        ("best_future_nondom", "roi", False, 1.0),
        ("best_future", "profit", False, 1.0),
        ("best_future", "cheap_roi", False, 1.0),
        ("best_future", "roi", False, 1.01),
        ("best_future", "roi", False, 1.05),
        ("suffix_peak", "roi", False, 1.0),
        ("best_rate", "rate", False, 1.0),
        ("earliest_profit", "rate", False, 1.0),
        ("earliest_profit_nondom", "rate", False, 1.0),
        ("earliest_profit", "roi", False, 1.0),
        ("final_2037", "roi", False, 1.0),
        ("final_2037", "profit", False, 1.0),
    ]


def plan_for_route(timeline: dict[int, dict[str, dict[str, int]]],
                   starting_capital: int, route: list[int],
                   policy: Policy) -> list[str]:
    target_mode, ordering, exact_small, min_roi = policy
    target_lookup = precompute_targets(timeline, route, target_mode)
    capital = starting_capital
    actions: list[str] = []
    current_year = HOME_YEAR
    holdings: dict[str, int] = {}
    scheduled: dict[tuple[int, str], int] = {}
    bought_qty: dict[tuple[int, str], int] = {}

    for index, year in enumerate(route):
        if year != current_year:
            actions.append(jump_action(current_year, year))
            current_year = year

        # Sell first: a sale and purchase in the same year are both legal, and
        # selling first maximizes cash available for fresh lots.
        for stock in sorted(timeline.get(year, {})):
            qty = scheduled.pop((index, stock), 0)
            if qty <= 0:
                continue
            held = holdings.get(stock, 0)
            qty = min(qty, held)
            if qty <= 0:
                continue
            price = timeline[year][stock]["price"]
            holdings[stock] = held - qty
            if holdings[stock] <= 0:
                holdings.pop(stock, None)
            capital += qty * price
            actions.append(sell_action(stock, qty))

        candidates = buy_candidates(timeline, route, index, bought_qty,
                                    target_mode, target_lookup)
        if min_roi > 1.0:
            candidates = [lot for lot in candidates if lot.roi >= min_roi]
        chosen = choose_lots(candidates, capital, ordering, exact_small)
        for lot, qty in chosen:
            if qty <= 0 or lot.price * qty > capital:
                continue
            capital -= lot.price * qty
            bought_key = (lot.year, lot.stock)
            bought_qty[bought_key] = bought_qty.get(bought_key, 0) + qty
            holdings[lot.stock] = holdings.get(lot.stock, 0) + qty
            scheduled[(lot.sell_index, lot.stock)] = (
                scheduled.get((lot.sell_index, lot.stock), 0) + qty)
            actions.append(buy_action(lot.stock, qty))

    if current_year != HOME_YEAR:
        actions.append(jump_action(current_year, HOME_YEAR))
    return actions


def buy_candidates(timeline: dict[int, dict[str, dict[str, int]]],
                   route: list[int], index: int,
                   bought_qty: dict[tuple[int, str], int],
                   target_mode: str,
                   target_lookup: dict[tuple[int, str],
                                       tuple[int, int, int]]) -> list[Lot]:
    year = route[index]
    out: list[Lot] = []
    for stock, quote in timeline.get(year, {}).items():
        remaining = quote["qty"] - bought_qty.get((year, stock), 0)
        if remaining <= 0:
            continue
        target = target_lookup.get((index, stock))
        if target is not None and target[2] <= quote["price"]:
            target = None
        if target is None:
            target = choose_sell_target(timeline, route, index, stock,
                                        quote["price"], target_mode)
        if target is None:
            continue
        if skips_later_cheaper(target_mode) and later_cheaper_lot_exists(
                timeline, route, index, target[0], stock, quote["price"]):
            continue
        sell_index, sell_year, sell_price = target
        out.append(Lot(year, stock, quote["price"], remaining, sell_index,
                       sell_year, sell_price))
    return out


def precompute_targets(timeline: dict[int, dict[str, dict[str, int]]],
                       route: list[int],
                       mode: str) -> dict[tuple[int, str], tuple[int, int, int]]:
    """Cache target sell years for modes whose best future is price-only."""
    mode = base_target_mode(mode)
    if mode in {"earliest_profit", "best_rate"}:
        return {}

    lookup: dict[tuple[int, str], tuple[int, int, int]] = {}

    if mode == "final_2037":
        next_home: int | None = None
        for i in range(len(route) - 1, -1, -1):
            if next_home is not None:
                for stock, quote in timeline.get(HOME_YEAR, {}).items():
                    lookup[(i, stock)] = (next_home, HOME_YEAR, quote["price"])
            if route[i] == HOME_YEAR:
                next_home = i
        return lookup

    best_by_stock: dict[str, tuple[int, int, int]] = {}
    for i in range(len(route) - 1, -1, -1):
        for stock, target in best_by_stock.items():
            lookup[(i, stock)] = target
        for stock, quote in timeline.get(route[i], {}).items():
            current = (i, route[i], quote["price"])
            old = best_by_stock.get(stock)
            if old is None or quote["price"] >= old[2]:
                best_by_stock[stock] = current
    return lookup


def choose_sell_target(timeline: dict[int, dict[str, dict[str, int]]],
                       route: list[int], index: int, stock: str, buy_price: int,
                       mode: str) -> tuple[int, int, int] | None:
    mode = base_target_mode(mode)
    future: list[tuple[int, int, int]] = []
    for j in range(index + 1, len(route)):
        q = timeline.get(route[j], {}).get(stock)
        if q and q["price"] > buy_price:
            future.append((j, route[j], q["price"]))
    if not future:
        return None

    if mode == "earliest_profit":
        return future[0]

    if mode == "final_2037":
        for item in reversed(future):
            if item[1] == HOME_YEAR:
                return item
        return None

    if mode == "best_rate":
        return max(future, key=lambda x: (
            log(x[2] / buy_price) / max(1, abs(x[1] - route[index])),
            x[2],
            -x[0],
        ))

    if mode == "suffix_peak":
        best_price = max(p for _, _, p in future)
        return min((item for item in future if item[2] == best_price),
                   key=lambda x: x[0])

    return max(future, key=lambda x: (x[2], -x[0]))


def base_target_mode(mode: str) -> str:
    return mode.removesuffix("_nondom")


def skips_later_cheaper(mode: str) -> bool:
    return mode.endswith("_nondom")


def later_cheaper_lot_exists(timeline: dict[int, dict[str, dict[str, int]]],
                             route: list[int], start_index: int,
                             sell_index: int, stock: str,
                             buy_price: int) -> bool:
    for j in range(start_index + 1, sell_index):
        quote = timeline.get(route[j], {}).get(stock)
        if quote and quote["qty"] > 0 and quote["price"] < buy_price:
            return True
    return False


def choose_lots(candidates: list[Lot], capital: int, ordering: str,
                exact_small: bool) -> list[tuple[Lot, int]]:
    if not candidates or capital <= 0:
        return []

    if (exact_small and capital <= EXACT_KNAPSACK_CAPITAL
            and sum(c.qty for c in candidates) <= EXACT_KNAPSACK_UNITS):
        exact = exact_knapsack(candidates, capital)
        if exact:
            return exact

    key = ordering_key(ordering)
    ordered = sorted(candidates, key=key, reverse=True)
    chosen: list[tuple[Lot, int]] = []
    remaining = capital
    for lot in ordered:
        qty = min(lot.qty, remaining // lot.price)
        if qty <= 0:
            continue
        chosen.append((lot, qty))
        remaining -= qty * lot.price
    return chosen


def slack_fill_plan(timeline: dict[int, dict[str, dict[str, int]]],
                    starting_capital: int, actions: list[str],
                    ordering: str) -> list[str]:
    """Buy extra lots along an existing route without reducing future cash.

    This pass is deliberately conservative: it only uses cash that remains
    available at every later block until the inserted sell action, so it cannot
    invalidate the original plan's later purchases.
    """
    blocks = action_blocks(actions)
    if len(blocks) <= 1:
        return actions

    bought: Counter[tuple[int, str]] = Counter()
    capital = starting_capital
    cap_after: list[int] = []
    block_min: list[int] = []
    for year, block_actions, _, _ in blocks:
        lowest = capital
        for action in block_actions:
            parsed = parse_action(action)
            if parsed is None:
                return actions
            kind, stock, raw_qty = parsed
            if kind not in {"b", "s"}:
                continue
            qty = int(raw_qty)
            price = timeline.get(year, {}).get(stock, {}).get("price")
            if price is None:
                return actions
            if kind == "b":
                capital -= price * qty
                bought[(year, stock)] += qty
            else:
                capital += price * qty
            lowest = min(lowest, capital)
        block_min.append(lowest)
        cap_after.append(capital)

    targets = best_future_targets_for_blocks(timeline, blocks)
    candidates: list[tuple[tuple[float, ...], int, int, str, int]] = []
    for index, (year, _, _, _) in enumerate(blocks[:-1]):
        for stock, quote in timeline.get(year, {}).items():
            target = targets[index].get(stock)
            if target is None:
                continue
            sell_index, _, sell_price = target
            if sell_index <= index or sell_price <= quote["price"]:
                continue
            unit_profit = sell_price - quote["price"]
            roi = sell_price / quote["price"]
            span = sell_index - index
            key = slack_candidate_key(ordering, roi, unit_profit, span,
                                      quote["price"])
            candidates.append((key, index, sell_index, stock, quote["price"]))

    used_after = [0] * len(blocks)
    used_block = [0] * len(blocks)
    extra_bought: Counter[tuple[int, str]] = Counter()
    inserted_buys: dict[int, list[str]] = defaultdict(list)
    inserted_sells: dict[int, list[str]] = defaultdict(list)

    for _, buy_index, sell_index, stock, buy_price in sorted(
            candidates, reverse=True):
        buy_year = blocks[buy_index][0]
        available = (timeline[buy_year][stock]["qty"]
                     - bought[(buy_year, stock)]
                     - extra_bought[(buy_year, stock)])
        if available <= 0:
            continue

        slack_values = [
            cap_after[i] - used_after[i]
            for i in range(buy_index, sell_index)
        ]
        slack_values.extend(
            block_min[i] - used_block[i]
            for i in range(buy_index + 1, sell_index)
        )
        cash_slack = min(slack_values) if slack_values else 0
        qty = min(available, cash_slack // buy_price)
        if qty <= 0:
            continue

        cost = qty * buy_price
        for i in range(buy_index, sell_index):
            used_after[i] += cost
        for i in range(buy_index + 1, sell_index):
            used_block[i] += cost
        extra_bought[(buy_year, stock)] += qty
        inserted_buys[buy_index].append(buy_action(stock, qty))
        inserted_sells[sell_index].append(sell_action(stock, qty))

    if not inserted_buys and not inserted_sells:
        return actions

    out: list[str] = []
    for index, (_, block_actions, jump, _) in enumerate(blocks):
        out.extend(sorted(inserted_sells.get(index, [])))
        out.extend(block_actions)
        out.extend(sorted(inserted_buys.get(index, [])))
        if jump is not None:
            out.append(jump)
    return out


def detour_fill_plan(timeline: dict[int, dict[str, dict[str, int]]],
                     starting_capital: int, energy: int,
                     actions: list[str]) -> list[str]:
    """Use leftover energy for profitable buy-only detours from route stops."""
    spare_energy = energy - action_energy_cost(actions)
    if spare_energy <= 0:
        return actions

    blocks = action_blocks(actions)
    if len(blocks) <= 1:
        return actions

    bought: Counter[tuple[int, str]] = Counter()
    capital = starting_capital
    cap_after: list[int] = []
    block_min: list[int] = []
    for year, block_actions, _, _ in blocks:
        lowest = capital
        for action in block_actions:
            parsed = parse_action(action)
            if parsed is None:
                return actions
            kind, stock, raw_qty = parsed
            if kind not in {"b", "s"}:
                continue
            qty = int(raw_qty)
            price = timeline.get(year, {}).get(stock, {}).get("price")
            if price is None:
                return actions
            if kind == "b":
                capital -= price * qty
                bought[(year, stock)] += qty
            else:
                capital += price * qty
            lowest = min(lowest, capital)
        block_min.append(lowest)
        cap_after.append(capital)

    targets = best_future_targets_for_blocks(timeline, blocks)
    candidates: list[tuple[tuple[float, ...], int, int, int, str, int]] = []
    for index, (route_year, _, _, _) in enumerate(blocks[:-1]):
        for buy_year, stocks in timeline.items():
            if buy_year == route_year:
                continue
            travel = 2 * abs(route_year - buy_year)
            if travel <= 0 or travel > spare_energy:
                continue
            for stock, quote in stocks.items():
                target = targets[index].get(stock)
                if target is None:
                    continue
                sell_index, _, sell_price = target
                if sell_index <= index or sell_price <= quote["price"]:
                    continue
                remaining = quote["qty"] - bought[(buy_year, stock)]
                if remaining <= 0:
                    continue
                total_profit = (sell_price - quote["price"]) * remaining
                roi = sell_price / quote["price"]
                key = (
                    total_profit / max(1, travel),
                    roi,
                    total_profit,
                    -travel,
                )
                candidates.append((key, index, sell_index, buy_year, stock,
                                   quote["price"]))

    if not candidates:
        return actions

    used_after = [0] * len(blocks)
    used_block = [0] * len(blocks)
    extra_bought: Counter[tuple[int, str]] = Counter()
    detour_buys: dict[int, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    inserted_sells: dict[int, list[str]] = defaultdict(list)

    for _, buy_index, sell_index, buy_year, stock, buy_price in sorted(
            candidates, reverse=True):
        route_year = blocks[buy_index][0]
        travel = 2 * abs(route_year - buy_year)
        if travel > spare_energy:
            continue
        available = (timeline[buy_year][stock]["qty"]
                     - bought[(buy_year, stock)]
                     - extra_bought[(buy_year, stock)])
        if available <= 0:
            continue

        slack_values = [
            cap_after[i] - used_after[i]
            for i in range(buy_index, sell_index)
        ]
        slack_values.extend(
            block_min[i] - used_block[i]
            for i in range(buy_index + 1, sell_index)
        )
        cash_slack = min(slack_values) if slack_values else 0
        qty = min(available, cash_slack // buy_price)
        if qty <= 0:
            continue

        cost = qty * buy_price
        for i in range(buy_index, sell_index):
            used_after[i] += cost
        for i in range(buy_index + 1, sell_index):
            used_block[i] += cost
        spare_energy -= travel
        extra_bought[(buy_year, stock)] += qty
        detour_buys[buy_index][buy_year].append(buy_action(stock, qty))
        inserted_sells[sell_index].append(sell_action(stock, qty))

    if not detour_buys and not inserted_sells:
        return actions

    out: list[str] = []
    for index, (year, block_actions, jump, _) in enumerate(blocks):
        out.extend(sorted(inserted_sells.get(index, [])))
        out.extend(block_actions)
        for buy_year, buy_actions in sorted(detour_buys.get(index, {}).items()):
            out.append(jump_action(year, buy_year))
            out.extend(sorted(buy_actions))
            out.append(jump_action(buy_year, year))
        if jump is not None:
            out.append(jump)
    return out


def delayed_sell_plan(timeline: dict[int, dict[str, dict[str, int]]],
                      starting_capital: int,
                      actions: list[str]) -> list[str]:
    """Move affordable sells to later route visits with better prices."""
    if not actions:
        return actions

    year = HOME_YEAR
    capital = starting_capital
    cap_after: list[int] = []
    block_starts: list[tuple[int, int]] = [(0, HOME_YEAR)]
    sells: list[tuple[int, str, int, int]] = []

    for index, action in enumerate(actions):
        parsed = parse_action(action)
        if parsed is None:
            return actions
        kind, second, third = parsed
        if kind == "j":
            year = int(third)
            block_starts.append((index + 1, year))
        elif kind == "b":
            qty = int(third)
            quote = timeline.get(year, {}).get(second)
            if quote is None:
                return actions
            capital -= quote["price"] * qty
        elif kind == "s":
            qty = int(third)
            quote = timeline.get(year, {}).get(second)
            if quote is None:
                return actions
            sell_price = quote["price"]
            sells.append((index, second, qty, sell_price))
            capital += sell_price * qty
        cap_after.append(capital)

    if not sells:
        return actions

    candidates: list[tuple[tuple[float, ...], int, int, str, int, int]] = []
    for sell_index, stock, qty, old_price in sells:
        if qty <= 0:
            continue
        for target_pos, target_year in block_starts:
            if target_pos <= sell_index:
                continue
            quote = timeline.get(target_year, {}).get(stock)
            if quote is None or quote["price"] <= old_price:
                continue
            gain = quote["price"] - old_price
            span = target_pos - sell_index
            key = (gain / old_price, gain / max(1, span), gain)
            candidates.append((key, sell_index, target_pos, stock, old_price,
                               qty))

    if not candidates:
        return actions

    used_cash = [0] * len(actions)
    remaining_sell_qty = {index: qty for index, _, qty, _ in sells}
    reduced_sells: Counter[int] = Counter()
    inserted_sells: dict[int, list[str]] = defaultdict(list)

    for _, sell_index, target_pos, stock, old_price, _ in sorted(
            candidates, reverse=True):
        available = remaining_sell_qty.get(sell_index, 0)
        if available <= 0:
            continue
        cash_slack = min(
            (cap_after[i] - used_cash[i]
             for i in range(sell_index, target_pos)),
            default=0,
        )
        qty = min(available, cash_slack // old_price)
        if qty <= 0:
            continue
        locked_cash = qty * old_price
        for i in range(sell_index, target_pos):
            used_cash[i] += locked_cash
        remaining_sell_qty[sell_index] -= qty
        reduced_sells[sell_index] += qty
        inserted_sells[target_pos].append(sell_action(stock, qty))

    if not reduced_sells:
        return actions

    out: list[str] = []
    for index, action in enumerate(actions):
        out.extend(sorted(inserted_sells.get(index, [])))
        parsed = parse_action(action)
        if parsed is None:
            return actions
        kind, stock, raw_qty = parsed
        if kind == "s" and reduced_sells.get(index, 0):
            qty = int(raw_qty) - reduced_sells[index]
            if qty > 0:
                out.append(sell_action(stock, qty))
        else:
            out.append(action)
    out.extend(sorted(inserted_sells.get(len(actions), [])))
    return out


def action_energy_cost(actions: list[str]) -> int:
    cost = 0
    for action in actions:
        parsed = parse_action(action)
        if parsed is None or parsed[0] != "j":
            continue
        _, src, dst = parsed
        cost += abs(int(src) - int(dst))
    return cost


def action_blocks(actions: list[str]) -> list[tuple[int, list[str], str | None,
                                                   int | None]]:
    blocks: list[tuple[int, list[str], str | None, int | None]] = []
    year = HOME_YEAR
    block_actions: list[str] = []
    for action in actions:
        parsed = parse_action(action)
        if parsed is None:
            continue
        kind, second, third = parsed
        if kind == "j":
            blocks.append((year, block_actions, action, int(third)))
            year = int(third)
            block_actions = []
        else:
            block_actions.append(action)
    blocks.append((year, block_actions, None, None))
    return blocks


def best_future_targets_for_blocks(
    timeline: dict[int, dict[str, dict[str, int]]],
    blocks: list[tuple[int, list[str], str | None, int | None]],
) -> list[dict[str, tuple[int, int, int]]]:
    lookup: list[dict[str, tuple[int, int, int]]] = [{} for _ in blocks]
    best_by_stock: dict[str, tuple[int, int, int]] = {}
    for index in range(len(blocks) - 1, -1, -1):
        lookup[index] = best_by_stock.copy()
        year = blocks[index][0]
        for stock, quote in timeline.get(year, {}).items():
            old = best_by_stock.get(stock)
            current = (index, year, quote["price"])
            if old is None or quote["price"] >= old[2]:
                best_by_stock[stock] = current
    return lookup


def slack_candidate_key(ordering: str, roi: float, unit_profit: int,
                        span: int, price: int) -> tuple[float, ...]:
    if ordering == "profit":
        return (unit_profit, roi, -span, -price)
    return (roi, unit_profit, -span, -price)


def ordering_key(ordering: str) -> Callable[[Lot], tuple[float, ...]]:
    if ordering == "profit":
        return lambda l: (l.profit, l.roi, -l.price)
    if ordering == "rate":
        return lambda l: (
            log(l.roi) / max(1, abs(l.sell_year - l.year)),
            l.roi,
            l.profit,
        )
    if ordering == "cheap_roi":
        return lambda l: (l.roi, -l.price, l.profit)
    if ordering == "early_roi":
        return lambda l: (l.roi / max(1, abs(l.sell_year - l.year)),
                          -abs(l.sell_year - l.year), l.profit)
    return lambda l: (l.roi, l.profit, -l.price)


def exact_knapsack(candidates: list[Lot], capital: int) -> list[tuple[Lot, int]]:
    items: list[tuple[int, int, int]] = []
    for idx, lot in enumerate(candidates):
        qty = lot.qty
        block = 1
        while qty > 0:
            take = min(block, qty)
            items.append((idx, take * lot.price, take * lot.profit))
            qty -= take
            block <<= 1

    dp: list[tuple[int, int | None]] = [(0, None)] * (capital + 1)
    for item_idx, (_, cost, value) in enumerate(items):
        for cash in range(capital, cost - 1, -1):
            candidate = dp[cash - cost][0] + value
            if candidate > dp[cash][0]:
                dp[cash] = (candidate, item_idx)

    best_cash = max(range(capital + 1), key=lambda c: dp[c][0])
    if dp[best_cash][0] <= 0:
        return []

    counts = [0] * len(candidates)
    cash = best_cash
    used_items: set[int] = set()
    while cash >= 0:
        item_idx = dp[cash][1]
        if item_idx is None or item_idx in used_items:
            break
        used_items.add(item_idx)
        lot_idx, cost, _ = items[item_idx]
        counts[lot_idx] += cost // candidates[lot_idx].price
        cash -= cost

    return [(candidates[i], qty) for i, qty in enumerate(counts) if qty > 0]


@dataclass
class SimResult:
    ok: bool
    capital: int
    energy_left: int
    reason: str = ""


def simulate(case: dict[str, Any], actions: list[str]) -> SimResult:
    timeline = normalize_timeline(case.get("timeline") or {})
    capital = int(case.get("capital", 0))
    energy = int(case.get("energy", 0))
    year = HOME_YEAR
    holdings: dict[str, int] = {}
    bought: dict[tuple[int, str], int] = {}

    for raw in actions:
        if not isinstance(raw, str):
            return SimResult(False, capital, energy, "non-string action")
        parsed = parse_action(raw)
        if parsed is None:
            return SimResult(False, capital, energy, "bad action format")
        kind, second, third = parsed
        if kind == "j":
            try:
                src = int(second)
                dst = int(third)
            except ValueError:
                return SimResult(False, capital, energy, "bad jump year")
            if src != year:
                return SimResult(False, capital, energy, "jump source mismatch")
            cost = abs(dst - src)
            if cost > energy or dst <= 0 or dst > HOME_YEAR:
                return SimResult(False, capital, energy, "bad jump")
            energy -= cost
            year = dst
        elif kind == "b":
            stock = second
            try:
                qty = int(third)
            except ValueError:
                return SimResult(False, capital, energy, "bad buy quantity")
            quote = timeline.get(year, {}).get(stock)
            if not quote or qty <= 0 or qty > quote["qty"]:
                return SimResult(False, capital, energy, "bad buy")
            already = bought.get((year, stock), 0)
            if already + qty > quote["qty"]:
                return SimResult(False, capital, energy, "buy exceeds lot qty")
            cost = quote["price"] * qty
            if cost > capital:
                return SimResult(False, capital, energy, "insufficient capital")
            bought[(year, stock)] = already + qty
            capital -= cost
            holdings[stock] = holdings.get(stock, 0) + qty
        elif kind == "s":
            stock = second
            try:
                qty = int(third)
            except ValueError:
                return SimResult(False, capital, energy, "bad sell quantity")
            quote = timeline.get(year, {}).get(stock)
            if not quote or qty <= 0 or holdings.get(stock, 0) < qty:
                return SimResult(False, capital, energy, "bad sell")
            holdings[stock] -= qty
            if holdings[stock] <= 0:
                holdings.pop(stock, None)
            capital += quote["price"] * qty
        else:
            return SimResult(False, capital, energy, "unknown action")

    if year != HOME_YEAR:
        return SimResult(False, capital, energy, "did not return home")
    if holdings:
        return SimResult(False, capital, energy, "unsold holdings")
    return SimResult(True, capital, energy)


def jump_action(src: int, dst: int) -> str:
    return f"j-{src}-{dst}"


def buy_action(stock: str, qty: int) -> str:
    return f"b-{stock}-{qty}"


def sell_action(stock: str, qty: int) -> str:
    return f"s-{stock}-{qty}"


def parse_action(raw: str) -> tuple[str, str, str] | None:
    if raw.startswith("j-"):
        parts = raw.split("-")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return None
    if raw.startswith(("b-", "s-")):
        kind, rest = raw.split("-", 1)
        if "-" not in rest:
            return None
        stock, qty = rest.rsplit("-", 1)
        return kind, stock, qty
    return None


def final_2037_exact_plan(case: dict[str, Any]) -> list[str] | None:
    """Exact bounded knapsack for the intended buy-past/sell-2037 pattern."""
    timeline = normalize_timeline(case.get("timeline") or {})
    home = timeline.get(HOME_YEAR, {})
    if not home:
        return None
    try:
        energy = int(case.get("energy", 0))
        capital = int(case.get("capital", 0))
    except (TypeError, ValueError):
        return None
    if capital > FINAL_EXACT_CAPITAL:
        return None

    min_year = max(1, HOME_YEAR - energy // 2)
    lots: list[tuple[int, str, int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for year, stocks in timeline.items():
        if year < min_year or year > HOME_YEAR:
            continue
        for stock, quote in stocks.items():
            sell_quote = home.get(stock)
            if not sell_quote or quote["qty"] <= 0:
                continue
            buy_price = quote["price"]
            sell_price = sell_quote["price"]
            if sell_price <= buy_price:
                continue
            lot_idx = len(lots)
            lots.append((year, stock, buy_price, sell_price, quote["qty"]))
            remaining = quote["qty"]
            block = 1
            while remaining > 0:
                qty = min(block, remaining)
                blocks.append((lot_idx, qty, buy_price * qty,
                               (sell_price - buy_price) * qty))
                remaining -= qty
                block <<= 1

    if not blocks or len(blocks) > FINAL_EXACT_BLOCKS:
        return None

    dp: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for block_idx, (_, _, cost, value) in enumerate(blocks):
        additions: dict[int, tuple[int, tuple[int, ...]]] = {}
        for spent, (profit, path) in dp.items():
            new_spent = spent + cost
            if new_spent > capital:
                continue
            new_profit = profit + value
            old_profit = additions.get(
                new_spent, dp.get(new_spent, (-1, ())))[0]
            if new_profit > old_profit:
                additions[new_spent] = (new_profit, path + (block_idx,))
        dp.update(additions)

    spent, (profit, path) = max(dp.items(), key=lambda item: item[1][0])
    if profit <= 0 or spent <= 0:
        return []

    selected: dict[tuple[int, str], int] = {}
    sells: dict[str, int] = {}
    for block_idx in path:
        lot_idx, qty, _, _ = blocks[block_idx]
        year, stock, _, _, _ = lots[lot_idx]
        selected[(year, stock)] = selected.get((year, stock), 0) + qty
        sells[stock] = sells.get(stock, 0) + qty

    actions: list[str] = []
    current = HOME_YEAR
    for year in sorted({year for year, _ in selected}, reverse=True):
        if year != current:
            actions.append(jump_action(current, year))
            current = year
        for stock in sorted(stock for y, stock in selected if y == year):
            actions.append(buy_action(stock, selected[(year, stock)]))
    if current != HOME_YEAR:
        actions.append(jump_action(current, HOME_YEAR))
    for stock in sorted(sells):
        actions.append(sell_action(stock, sells[stock]))
    return actions


def exact_search(case: dict[str, Any]) -> list[str] | None:
    """Exhaustive optimizer for small hidden cases.

    The heuristic handles large quantities quickly. For small cases, the whole
    problem is finite, so we can search the real action graph and remove the
    awkward integer/reinvestment edge cases that greedy policies miss.
    """
    timeline = normalize_timeline(case.get("timeline") or {})
    if not timeline:
        return []
    try:
        energy = int(case.get("energy", 0))
        capital = int(case.get("capital", 0))
    except (TypeError, ValueError):
        return None

    stocks = sorted({stock for quotes in timeline.values() for stock in quotes})
    years = sorted(set(timeline) | {HOME_YEAR})
    lots: list[tuple[int, str, int, int]] = []
    lot_index: dict[tuple[int, str], int] = {}
    total_qty = 0
    for year in years:
        for stock in stocks:
            quote = timeline.get(year, {}).get(stock)
            if quote and quote["qty"] > 0:
                lot_index[(year, stock)] = len(lots)
                lots.append((year, stock, quote["price"], quote["qty"]))
                total_qty += quote["qty"]

    if (len(stocks) > 4 or len(years) > 5 or energy > 8 or
            total_qty > 14 or capital > 500):
        return None

    stock_index = {stock: i for i, stock in enumerate(stocks)}
    max_states = 60_000
    visited_states = 0
    memo: dict[tuple[int, int, int, tuple[int, ...], int],
               tuple[int, tuple[str, ...]]] = {}

    def better(candidate: tuple[int, tuple[str, ...]],
               incumbent: tuple[int, tuple[str, ...]]) -> tuple[int, tuple[str, ...]]:
        if candidate[0] > incumbent[0]:
            return candidate
        if candidate[0] == incumbent[0] and len(candidate[1]) < len(incumbent[1]):
            return candidate
        return incumbent

    def rec(year: int, energy_left: int, cash: int, holdings: tuple[int, ...],
            bought_mask: int) -> tuple[int, tuple[str, ...]]:
        nonlocal visited_states
        key = (year, energy_left, cash, holdings, bought_mask)
        if key in memo:
            return memo[key]
        visited_states += 1
        if visited_states > max_states:
            raise RuntimeError("exact search state budget exceeded")

        no_holdings = all(qty == 0 for qty in holdings)
        best = (cash, ()) if year == HOME_YEAR and no_holdings else (-10**18, ())

        # Sells. Partial sales matter when cash has to be freed for a better
        # opportunity but some shares should still be held for a later peak.
        for stock, idx in stock_index.items():
            held = holdings[idx]
            quote = timeline.get(year, {}).get(stock)
            if held <= 0 or not quote:
                continue
            price = quote["price"]
            for qty in range(1, held + 1):
                new_holdings = list(holdings)
                new_holdings[idx] -= qty
                value, suffix = rec(year, energy_left, cash + price * qty,
                                    tuple(new_holdings), bought_mask)
                if value > -10**17:
                    best = better((value, (sell_action(stock, qty),) + suffix),
                                  best)

        # Buys. To match the safest reading of "once a stock in a given year is
        # bought, it is permanently gone", buy at most once from each year-stock
        # lot, possibly only the affordable part.
        for stock, idx in stock_index.items():
            li = lot_index.get((year, stock))
            if li is None or ((bought_mask >> li) & 1):
                continue
            _, _, price, qty_available = lots[li]
            max_qty = min(qty_available, cash // price)
            for qty in range(1, max_qty + 1):
                new_holdings = list(holdings)
                new_holdings[idx] += qty
                value, suffix = rec(year, energy_left, cash - price * qty,
                                    tuple(new_holdings), bought_mask | (1 << li))
                if value > -10**17:
                    best = better((value, (buy_action(stock, qty),) + suffix),
                                  best)

        # Jumps. There is no reason to jump to years with no prices except the
        # required final return to 2037, because travel cost is linear.
        for dst in years:
            if dst == year:
                continue
            cost = abs(dst - year)
            if cost <= energy_left:
                value, suffix = rec(dst, energy_left - cost, cash, holdings,
                                    bought_mask)
                if value > -10**17:
                    best = better((value, (jump_action(year, dst),) + suffix),
                                  best)

        memo[key] = best
        return best

    try:
        value, actions = rec(HOME_YEAR, energy, capital, tuple([0] * len(stocks)), 0)
    except RuntimeError:
        return None
    if value <= capital:
        return []
    return list(actions)
