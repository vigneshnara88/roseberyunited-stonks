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

from dataclasses import dataclass
from math import log
from typing import Any, Callable

HOME_YEAR = 2037


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
    return [solve_case(case) for case in cases]


def solve_case(case: dict[str, Any]) -> list[str]:
    energy = int(case.get("energy", 0))
    capital = int(case.get("capital", 0))
    timeline = normalize_timeline(case.get("timeline") or {})

    if energy <= 0 or capital <= 0 or not timeline:
        return []

    routes = candidate_routes(timeline, energy)
    policies = build_policies()
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

    exact_actions = exact_search(case)
    if exact_actions is not None:
        result = simulate(case, exact_actions)
        if result.ok and result.capital > best_value:
            best_value = result.capital
            best_profit = result.capital - capital
            best_actions = exact_actions

    return best_actions if best_profit > 0 else []


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
                     energy: int) -> list[list[int]]:
    years = sorted(y for y in timeline if y <= HOME_YEAR)
    data_years = sorted(set(years + [HOME_YEAR]), reverse=True)
    min_reachable = max(1, HOME_YEAR - energy)
    reachable = [y for y in data_years if y >= min_reachable]
    if HOME_YEAR not in reachable:
        reachable.insert(0, HOME_YEAR)

    routes: list[list[int]] = [[HOME_YEAR]]
    max_depth = energy // 2
    depths = sorted({HOME_YEAR - y for y in reachable
                     if 0 <= HOME_YEAR - y <= max_depth})

    for depth in depths:
        if depth == 0:
            continue
        routes.append(round_trip_route(reachable, depth))

    if max_depth > 0:
        routes.append(round_trip_route(reachable, max_depth))

    # Sometimes a short round trip to 2037 grows cash early, then a second
    # deeper trip spends it better. Try compact two-trip combinations.
    profitable_depths = promising_depths(timeline, reachable, max_depth)
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
    routes.extend(pair_routes[:80])

    # Deterministic shortest plans first; longer speculative plans later.
    routes.sort(key=lambda r: (route_cost(r), len(r), tuple(r)))
    return routes


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
        return sorted({d for d in range(1, max_depth + 1)})
    return [d for _, d in scored]


def profitable_pair_routes(timeline: dict[int, dict[str, dict[str, int]]],
                           energy: int) -> list[list[int]]:
    years = sorted(timeline, reverse=True)
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


Policy = tuple[str, str, bool]


def build_policies() -> list[Policy]:
    sell_targets = [
        "best_future",
        "earliest_profit",
        "best_rate",
        "final_2037",
        "suffix_peak",
    ]
    orderings = [
        "roi",
        "rate",
        "profit",
        "cheap_roi",
        "early_roi",
    ]
    policies: list[Policy] = []
    for target in sell_targets:
        for ordering in orderings:
            policies.append((target, ordering, False))
            policies.append((target, ordering, True))
    return policies


def plan_for_route(timeline: dict[int, dict[str, dict[str, int]]],
                   starting_capital: int, route: list[int],
                   policy: Policy) -> list[str]:
    target_mode, ordering, exact_small = policy
    capital = starting_capital
    actions: list[str] = []
    current_year = HOME_YEAR
    holdings: dict[str, int] = {}
    scheduled: dict[tuple[int, str], int] = {}
    used_lots: set[tuple[int, str]] = set()

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

        candidates = buy_candidates(timeline, route, index, used_lots,
                                    target_mode)
        chosen = choose_lots(candidates, capital, ordering, exact_small)
        for lot, qty in chosen:
            if qty <= 0 or lot.price * qty > capital:
                continue
            capital -= lot.price * qty
            used_lots.add((lot.year, lot.stock))
            holdings[lot.stock] = holdings.get(lot.stock, 0) + qty
            scheduled[(lot.sell_index, lot.stock)] = (
                scheduled.get((lot.sell_index, lot.stock), 0) + qty)
            actions.append(buy_action(lot.stock, qty))

    if current_year != HOME_YEAR:
        actions.append(jump_action(current_year, HOME_YEAR))
    return actions


def buy_candidates(timeline: dict[int, dict[str, dict[str, int]]],
                   route: list[int], index: int,
                   used_lots: set[tuple[int, str]],
                   target_mode: str) -> list[Lot]:
    year = route[index]
    out: list[Lot] = []
    for stock, quote in timeline.get(year, {}).items():
        if quote["qty"] <= 0 or (year, stock) in used_lots:
            continue
        target = choose_sell_target(timeline, route, index, stock,
                                    quote["price"], target_mode)
        if target is None:
            continue
        sell_index, sell_year, sell_price = target
        out.append(Lot(year, stock, quote["price"], quote["qty"], sell_index,
                       sell_year, sell_price))
    return out


def choose_sell_target(timeline: dict[int, dict[str, dict[str, int]]],
                       route: list[int], index: int, stock: str, buy_price: int,
                       mode: str) -> tuple[int, int, int] | None:
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


def choose_lots(candidates: list[Lot], capital: int, ordering: str,
                exact_small: bool) -> list[tuple[Lot, int]]:
    if not candidates or capital <= 0:
        return []

    if exact_small and capital <= 40000 and sum(c.qty for c in candidates) <= 600:
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
    bought: set[tuple[int, str]] = set()

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
            if (year, stock) in bought:
                return SimResult(False, capital, energy, "duplicate lot buy")
            cost = quote["price"] * qty
            if cost > capital:
                return SimResult(False, capital, energy, "insufficient capital")
            bought.add((year, stock))
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

    if (len(stocks) > 5 or len(years) > 8 or energy > 14 or
            total_qty > 22 or capital > 5000):
        return None

    stock_index = {stock: i for i, stock in enumerate(stocks)}
    max_states = 250_000
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
