# Test results: pool consumed from wrong end (pick_order inverted)

Date: 2026-06-01
Branch: main

## Bug

`size_desc` processed the smallest file first (every pick_order was reversed).

`order_pool()` sorts correctly: `size_desc` returns `[biggest, ..., smallest]` with the
intended-first item at index 0. But `_process_app_once` consumed the pool with
`ctx.pool.pop()`, which removes the **last** element, so it took the smallest first.
Affected every non-random order (alphabetical/date/release too).

## Fix

`worker.py`: `ctx.pool.pop()` -> `ctx.pool.pop(0)` so the worker consumes the pool in the
processing order `order_pool` returns. (One pop per ~15s tick, so O(n) front-pop is irrelevant.)

## Test

Added `test_process_app_once_consumes_head_of_pool_first`: pool `[10, 20, 30]` -> after one
tick the head (10) is consumed, leaving `[20, 30]`. Fails with the old tail `pop()`.

## Commands

```sh
uv run ruff format optimizarr/ tests/   # unchanged
uv run ruff check optimizarr/ tests/    # All checks passed!
uv run pytest                           # 135 passed
```
