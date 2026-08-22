# Time Travelling Stonks Man Notes

Saved from the live challenge page on 2026-08-22.

## Endpoint

Expose `POST /stonks`.

Input is a JSON array of test cases. Each test case has:

- `energy`: positive integer greater than 1.
- `capital`: positive integer greater than 0.
- `timeline`: mapping of year strings to stock records.
- Each stock record has positive integer `price` and non-negative integer `qty`.

## Rules

- Start in year `2037`.
- Must return to year `2037`.
- Jump action costs `abs(from_year - to_year)` energy.
- Buying a stock at a given year consumes that year-stock quantity and cash.
- Selling requires holding that stock and uses the current year's listed price.

## Output

Return a JSON array of action arrays, one per input test case.

Action format:

- `j-<fromYear>-<toYear>`
- `b-<stockName>-<quantity>`
- `s-<stockName>-<quantity>`

## Strategy

The implementation searches valid routes under the energy budget and, for
each route, tests multiple buy/sell policies. It locally simulates every
candidate action sequence and returns the valid plan with the highest final
capital. The policies cover:

- direct historical buys sold at `2037`;
- intermediate peak sales;
- early sales that recycle capital;
- direct non-2037 buy/sell pair tours;
- two-trip plans where a shallow trip grows cash before a deeper trip.
