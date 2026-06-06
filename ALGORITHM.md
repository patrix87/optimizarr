# How Optimizarr decides

This is the in-depth companion to the [README](README.md): the selection algorithm, the
configuration model, and the worker loop.

The optimizer evaluates the releases available for one library item, decides whether a better one
exists, and grabs it through Radarr/Sonarr. It is built around the reality that **grabbed releases
frequently fail to download**, so *"optimized"* means *the algorithm can no longer find anything
better than the current file*, never merely *"we triggered a grab."*

Three ideas carry the whole design:

- **Two relative axes.** Each candidate is scored on just two axes — Profilarr **score** (higher
  better) and **GiB/h** (smaller better) — each min-max normalized *over the candidates the
  indexers actually returned for this movie*. There are no absolute size bands, target sizes, or
  shaped curves; "good size" is defined relative to what is on offer, which is what lets one
  profile shrink a movie that another leaves alone.
- **The forbidden quadrant.** Versus the current file a pick may be better+bigger (a real score
  upgrade), better+smaller (ideal), or a little worse+smaller (a deliberate size trade). It may
  **never** be worse *and* bigger. That falls out for free: a candidate worse on both axes must
  improve by at least one concrete threshold (`min_score_delta` or `min_size_delta_gb`) to ACT,
  which a worse+bigger candidate never can.
- **One-and-done.** A movie is optimized once: when its current (imported) file is the best pick
  for its profile it is marked *satisfied* and never re-evaluated. This removes re-evaluation loops
  that could oscillate.

---

## 1. Inclusion filters

Before scoring, candidates pass through filters that drop the obviously-bad. They are identical for
every profile (`topsis.py::apply_prefilters`):

1. **Hard rejections** — blocklisted, unparseable, wrong item, dead torrents, temporarily-rejected.
2. **Score window** — three-tier, anchored at the top of what the indexers offer (favors quality,
   only widens when the indexer is sparse). Negatives are always dropped (all floors are ≥ 0).
   - *Tier 1* `score >= max_available − score_window`. If ≥ `min_candidates` survive: accepted.
   - *Tier 2* `score >= current_file_score`. Expands to include releases at or above what you
     already have. Tried when Tier 1 yields too few.
   - *Tier 3* `score >= max(0, current_file_score − score_window)`. The full downgrade budget:
     allows trading some score for a meaningfully smaller file. Reached only as a last resort.
3. **Resolution** — keep only the profile's **target resolution**, so the GiB/h axis is comparable
   (a 1080p and a 2160p release are not). This also enforces "never drop below target".
4. **Legitimacy band** — drop anything outside the shared per-resolution `[floor, ceiling]` GiB/h
   bounds (fake/upscale below floor, bloat above ceiling). These are wide (≈ P1 / P99 of real
   sizes); the relative scoring, not a hard band, expresses taste.
5. **Outlier drop** *(`outlier_frac` = 0.5)* — drop a release whose GiB/h is below
   `outlier_frac × median(survivors)`. This is the key guard against over-compressed / upscale
   "2160p" junk that Profilarr still scores high on format tags (a real 2160p HEVC is ~4-8 GiB/h;
   junk is < 2). Without it, ~half of picks were such tiny files. The absolute floor is too low to
   catch them; this relative check is what does.

If fewer than `min_candidates` (default 6) survive all filters, the relative min-max is not
trustworthy, so the decision is **HOLD without satisfying** — the movie is retried later when more
releases appear. `min_candidates` also controls when the score-window tier is accepted vs expanded.

---

## 2. Relative scoring

Over the survivors, each axis is min-max normalized (the current file is then placed on the same
ranges, clamped, so it is comparable):

```
n_score = (score − min_score) / (max_score − min_score)          # higher score -> 1
n_size  = (max_gbh − gbh)     / (max_gbh − min_gbh)              # smaller file -> 1
```

A degenerate range (all equal) yields 1.0 for that axis (it does not discriminate). A profile's
**weights** (score + size, summing to 1.0) combine the two into a TOPSIS *closeness* — distance to
the ideal point `(1, 1)` vs the anti-ideal `(0, 0)`:

```
d_ideal = √( w_score·(1−n_score)² + w_size·(1−n_size)² )
d_anti  = √( w_score·n_score²     + w_size·n_size²     )
closeness = d_anti / (d_ideal + d_anti)        # 1 = ideal, 0 = anti-ideal
```

The weights are the **only** per-profile knob. The shipped profiles:

| Profile | score | size | leans |
| --- | --- | --- | --- |
| Remux | 0.94 | 0.06 | best score, size barely matters |
| Quality | 0.86 | 0.14 | best score, mild lean to smaller |
| Balanced | 0.56 | 0.44 | middle |
| Efficient | 0.44 | 0.56 | smaller, decent score |
| Compact | 0.22 | 0.78 | smallest legitimate file |

Because the axes are relative, the smallest survivor always gets `n_size = 1` regardless of its
absolute bitrate — so Efficient/Compact pick the lean release even when every candidate sits in
what a fixed band would call "too big". Resolution is **not** a scored axis (Profilarr folds it
into the score, and filter 3 already pins it).

---

## 3. The decision

`decision.py::decide`: resolve the profile → run the filters → relatively score the survivors →
compare to the current file.

- ACT on the best candidate (highest TOPSIS closeness) **iff** it clears at least one concrete
  threshold vs the current file: score improves by `>= min_score_delta` (default `100`) **or**
  size shrinks by `>= min_size_delta_gb` (default `0.5 GB`). When there is no current file, any
  candidate qualifies. Two optional per-app pre-filters run first: `allow_size_increase = false`
  drops anything bigger than the current file, `allow_quality_downgrade = false` drops anything
  lower-scoring.
- Otherwise **HOLD**. Two kinds:
  - **satisfiable** — there were enough candidates and none cleared either threshold: the current
    file is good enough for its profile, so mark it satisfied (permanent).
  - **insufficient** — fewer than `min_candidates`: do *not* satisfy; retry later.

The forbidden quadrant is impossible: a candidate worse on both axes (lower score, bigger size)
cannot clear either threshold and therefore never triggers ACT.

---

## 4. Configuration model

Everything lives under `[optimizer.topsis]`, layered on `defaults.toml`:

```toml
[optimizer.topsis]
score_window = 100000          # downgrade budget: keep score >= current - this
min_candidates = 2             # below this, HOLD without satisfying
outlier_frac = 0.5             # drop releases below 0.5x the movie's median GiB/h (junk guard)
default_preset = "Balanced"    # used when a profile name matches no preset keyword
min_score_delta = 100          # ACT only if pick.score - current.score >= this
min_size_delta_gb = 0.5        # ...or if current.size - pick.size >= this (in GB)

[optimizer.topsis.size_bounds]  # shared per-resolution {floor, ceiling} GiB/h legitimacy band
"2160" = { floor = 3.0, ceiling = 30.0 }
"1080" = { floor = 1.0, ceiling = 15.0 }
# … 720 / 480 …

[optimizer.topsis.presets.Efficient]
score = 0.40                   # weights (score + size, sum 1.0) — the only per-profile knob
size = 0.60
```

A profile attaches to the preset whose name is a case-insensitive substring of the profile name
(`2160p Efficient` → Efficient); `[optimizer.topsis.profiles."Exact Name"]` overrides a preset
or its weights for one profile.

---

## 5. The worker loop and state

The optimizer is a continuous, interval-driven worker. Each tick: refresh the item list if due,
build the active pool, gate on the download queue, then evaluate one item.

- **Safe refresh.** The library fetch is guarded: a failed or interrupted fetch keeps the previous
  item set and retries next tick. State is **never** pruned from the list, so a connection blip can
  never wipe it.
- **One queue fetch per tick** serves both the pace gate (`queue_max`) and the "already
  downloading?" skip, so there is no in-flight state to track and a restart needs no reconciliation.
- `process_interval_seconds` (default 15, min 10) doubles as a settle delay after a grab.
- `pick_order` only changes which items are improved first, never the per-item decision.

### Per-item state lifecycle (one-and-done)

`state.json` (keyed by item id) records, per satisfied item, **the profile it was satisfied for**.

```mermaid
stateDiagram-v2
    [*] --> Unprocessed: not in state
    Unprocessed --> Unprocessed: ACT grab posted (state unchanged)
    Unprocessed --> Satisfied: HOLD, current file optimal for its profile
    Satisfied --> Unprocessed: profile changed OR file removed
```

- A grab is **never recorded**. A grab that succeeds replaces the file; the next evaluation finds
  it optimal → satisfied. A grab that fails was never satisfied, so the item is retried, and the
  dead release is now blocklisted (so filter 1 drops it and the next-best is tried). This relies on
  Radarr/Sonarr **Failed Download Handling** (default on).
- Satisfied is **permanent**: there is no time-based re-evaluation. A satisfied movie becomes
  eligible again only if its **profile changes** (the optimal pick depends on the profile) or its
  **file is removed**. To force a full re-run, delete (or edit) `state.json`.

### Active-hours schedule

`[optimizer.schedule]` defines a per-day active window in local time (24h HH:MM). **Outside the
window the worker skips list refresh and movie evaluation, but queue import processing always
runs** (so completed downloads kept off the queue by auto-import are never blocked by the
schedule).

A window where `start >= end` crosses midnight. The default ships `23:00` to `08:00` on every day,
meaning the optimizer queries indexers and evaluates movies overnight. Omit a day (or the entire
block) to treat that day as always active. Transitions are logged once at INFO.

---

## 6. The Unmonitor job

A separate, cron-scheduled pass (`[unmonitor]`, default `0 4 * * *`) that **unmonitors** items so
the \*arr apps stop chasing upgrades off their RSS feeds. An item is unmonitored when it is
monitored, has a file, (optionally) has its quality cutoff met, and is at least `days` old against
the configured `release_type` date. This pairs with the optimizer: the optimizer keeps improving
files by `hasFile` regardless of monitored state, while Unmonitor strips the monitoring that would
otherwise have Radarr/Sonarr fighting it with fresh RSS grabs.

---

*`dry_run = true` makes both features log every would-be action without changing anything.
`tools/diagnose.py` runs the real engine over a gathered library sample
(`tools/gather_training_data.py`) and reports the pick quadrants, the total size shift, and the
full original → new-pick list.*
