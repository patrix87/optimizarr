# How Optimizarr decides

This is the in-depth companion to the [README](README.md): the selection algorithm, the guard
rails, the TOPSIS math, the configuration model, and the worker loop.

The optimizer evaluates the releases available for one library item, decides whether a better
one exists, and grabs it through Radarr/Sonarr. It is built around the reality that **grabbed
releases frequently fail to download**, so *"optimized"* means *the algorithm can no longer find
anything better than the current file*, never merely *"we triggered a grab."*

There are two big ideas:

- **Per-preset 5-point size bands.** Each preset carries its own
  `{floor, lo, target, hi, ceiling}` trapezoid per resolution (GiB/h). `floor` and `ceiling` are
  hard gates (out-of-band releases are dropped before scoring); `lo … hi` is the "good size" band
  where score drives the pick; `target` is the peak, placed at a per-profile size percentile, so
  each profile aims at a different size and the five profiles pick different releases.
- **A single closeness-gain swap rule.** Releases are scored by TOPSIS *closeness* over two axes
  (score, size; resolution is a hard guard, not an axis), and a candidate is grabbed only if it
  raises closeness past a small margin, doesn't drop resolution below target, and isn't bigger at a
  lower-or-equal score (a larger file is grabbed only on a real score upgrade). Because closeness is
  file-independent and strictly increases on every swap, the optimizer **cannot oscillate**.

---

## 1. The size model: per-preset tables, 5-point trapezoid

Each preset defines its own `[optimizer.topsis.presets.<name>.reference]` table: one
`{floor, lo, target, hi, ceiling}` entry per resolution, in GiB/h (`GB = 1024³`, so GiB). The
shipped 2160p and 1080p bands (floor / lo / target / hi / ceiling); 720p and 480p follow the same
shape, scaled down:

| preset | 2160p | 1080p |
| --- | --- | --- |
| Remux | 15 / 28 / 38 / 60 / 90 | 8 / 15 / 20 / 32 / 48 |
| Quality | 6.5 / 10.7 / 11.5 / 13.5 / 30 | 3 / 5 / 6.3 / 8.7 / 16 |
| Balanced | 5 / 8.1 / 9.3 / 11 / 20 | 2 / 3.8 / 4.5 / 5.3 / 10 |
| Efficient | 4 / 7.1 / 7.8 / 8.8 / 15 | 1.5 / 2.8 / 3.5 / 4.3 / 7.5 |
| Compact | 3.5 / 5.9 / 6.8 / 7.6 / 12 | 1.2 / 1.9 / 2.6 / 3.2 / 6 |

- **floor / ceiling**: legitimacy bounds. Outside `[floor, ceiling]` a release is dropped before
  scoring (too soft / fake below floor, bloated above ceiling), and `n_size = 0` there.
- **lo … hi**: the "good size" band. `n_size = size_shoulder` (default 0.85) at lo and hi, so
  inside the band size barely moves the decision and **score drives** the pick.
- **target**: the peak (`n_size = 1.0`), placed at a per-profile percentile of real release sizes
  (Compact ~P10, Efficient ~P30, Balanced ~P50, Quality ~P77, Remux at remux bitrates). Slide
  `target` within the band to nudge a profile lower or higher.

The size-desirability curve is a **trapezoid** (a flat-topped tent):

```
n_size(gbh) = 0                                          if gbh ≤ floor or gbh ≥ ceiling
            = shoulder·(gbh−floor)/(lo−floor)            if floor < gbh < lo    (rise to the band)
            = shoulder + (1−shoulder)·(gbh−lo)/(target−lo)   if lo ≤ gbh ≤ target
            = 1 − (1−shoulder)·(gbh−target)/(hi−target)      if target < gbh ≤ hi
            = shoulder·(ceiling−gbh)/(ceiling−hi)        if hi < gbh < ceiling  (fall past the band)
```

Each profile aims at a **different** band, so the five profiles pick different releases for the
same movie. Unlike the old one-sided curve, a too-small file is penalized (it sits on the rising
shoulder, not a plateau) — but it is never actually *grown* unless that is a genuine score upgrade
(§3). A lone tiny release among bigger same-score peers (usually a bad encode) is dropped by an
outlier prefilter: anything below `outlier_frac` (default 0.5) × the median GiB/h of the
gap-cut cluster. "A good encode has corroborating peers."

---

## 2. Scoring: TOPSIS over two axes

Resolution is **not** a scored axis: Profilarr already folds it into the score, and a hard guard
forbids dropping below the profile target (§3). So each surviving release has two normalized
attributes:

```
n_score = 1 / (1 + exp(-(score − score_center) / score_width))   # logistic (default)
n_size  = trapezoid above (the preset's floor/lo/target/hi/ceiling for this resolution)
```

Score normalizes through a **fixed** transform (a function of the release alone, so closeness
stays comparable across items and the no-oscillation guarantee in §3 holds). The default is a
**logistic** S-curve: Profilarr scores bunch tightly near the top (real releases top out around
~950k and cluster there) then fall off fast, so a logistic concentrates the `[0,1]` range where
releases actually compete. A linear ramp over `[0, 1,000,000]` wastes ~80% of the axis on scores
no release reaches, which lets a small real score difference at the top get drowned out by the
size axis; the logistic fixes that and, having no hard floor, still separates a library whose
scores all sit near one value. `score_center` (n_score = 0.5) and `score_width` (slope) are set
in `[optimizer.topsis]`; `score_norm = "linear"` restores the old fixed ramp.

A profile's **weights** (score + size, summing to 1.0) combine the axes into a TOPSIS *closeness*,
the distance to the ideal point `(1,1)` vs the anti-ideal `(0,0)`:

```
d_ideal = √( Σ wₖ·(1 − aₖ)² )
d_anti  = √( Σ wₖ·aₖ² )
closeness = d_anti / (d_ideal + d_anti)        # 1 = ideal, 0 = anti-ideal
```

Shipped weights and pick method:

| Profile | score | size | pick method |
| --- | --- | --- | --- |
| Remux | 0.94 | 0.06 | `topsis` |
| Quality | 0.86 | 0.14 | `topsis` |
| Balanced | 0.56 | 0.44 | `topsis` |
| Efficient | 0.44 | 0.56 | `topsis` |
| Compact | 0.22 | 0.78 | `topsis` |

All presets pick by `topsis` (highest closeness); the profiles diverge because their **bands** sit
at different sizes, not because of the pick method. `max_score` and `min_size` remain available for
profile overrides.

---

## 3. The decision pipeline

```mermaid
flowchart TD
    A["Item selected"] --> RP["Resolve profile → preset<br/>(name-keyword match or override)"]
    RP --> B["GET /api/v3/release"]
    B --> C["Pre-filter 1: drop hard rejections<br/>blocklisted · unparseable · wrong item · dead"]
    C --> D["Pre-filter 2: drop outside the preset's size band<br/>gbh < floor (fake/upscale) or gbh > ceiling (bloat)"]
    D --> E["Pre-filter 3: gap-cut on score<br/>keep the top cluster; cut at first drop > score_gap"]
    E --> O["Pre-filter 4: drop lone-small size outliers<br/>gbh < outlier_frac × cluster median"]
    O --> SC["Score survivors (TOPSIS closeness over score + size)"]
    SC --> GATE["SWAP RULE<br/>raise closeness ≥ min_closeness_gain,<br/>don't drop resolution below target,<br/>not bigger at a lower-or-equal score"]
    GATE --> LEFT{"any legal candidate?"}
    LEFT -->|no| HOLD["HOLD: mark satisfied"]
    LEFT -->|yes| PICK["PICK best survivor (topsis: max closeness)"]
    PICK --> ACT["ACT: POST grab {guid, indexerId}"]
```

Two app-policy pre-filters can run before scoring: `allow_size_increase = false` drops anything
bigger than the current file, and `allow_quality_downgrade = false` drops anything lower-scoring
(turning the latter off neutralizes the size-leaning profiles, which is the point of the flag).

ACT if at least one candidate passes the swap rule, else HOLD. Closeness decides both *whether*
to act (the gain margin) and, via the pick method, *which* survivor to grab.

---

## 4. The swap rule (the guard rails)

A scored candidate may replace the current file only if **both** hold:

1. **Closeness gain.** `closeness(candidate) ≥ closeness(current) + min_closeness_gain`. Closeness
   is computed from the release's own attributes and the preset (not from the current file), so it
   is a single, stable per-release number. `min_closeness_gain` (default `0.02`, per-preset
   overridable) is a hysteresis margin that avoids churning for negligible gains. An unknown
   current score is treated as the worst case, so any scored candidate is an improvement.
2. **Resolution guard.** `min(cand_res, target) ≥ min(cur_res, target)`, where `target` is the
   profile's target resolution. Resolution may rise toward the target or fall only as far as the
   target, never below it. This preserves "drop resolution to match a leaner profile" (a 1080p
   profile may move a 2160p file down to 1080p) while stopping a size-leaning preset from shaving
   resolution below what the profile asks for.

### Why there is no explicit "no senseless swap" rule

"Never swap to a bigger file for a lower (or equal) score" is **not** a separate rule: at equal
resolution such a candidate is worse on score and no better on size, so its closeness cannot
exceed the current file's, and the gain test already rejects it. The resolution guard is the only
hard constraint closeness does not already imply (a size-leaning preset could otherwise raise
closeness by dropping resolution for size).

### No oscillation, for any cycle length

Closeness is a pure function of a release's attributes and the preset, independent of which file
you currently hold. Every accepted swap raises the held file's closeness by at least
`min_closeness_gain`, so that closeness is **strictly increasing** and bounded by 1: the walk
visits each file at most once and terminates in at most `1 / min_closeness_gain` swaps. This is a
stronger guarantee than a pairwise rule can give. Two directional size/score thresholds, for
example, can stop a 2-file ping-pong yet still admit a 3-file cycle; a single monotonic quantity
cannot cycle at all. The convergence property is checked in tests by simulating the
grab/re-evaluate loop over thousands of random pools across every preset.

---

## 5. Configuration model

Everything lives under `[optimizer.topsis]`, layered on `defaults.toml`:

```toml
[optimizer.topsis]
score_ideal = 1000000          # n_score scale …
score_anti_ideal = 0
resolution_ideal = 2160        # fallback target when a profile exposes no allowed resolution
resolution_anti_ideal = 480
score_gap = 0.20               # gap-cut: keep the top score cluster within this relative drop
default_preset = "Balanced"    # used when a profile name matches no preset keyword
min_closeness_gain = 0.02      # global swap margin (per-preset overridable)

[optimizer.topsis.presets.Efficient]
score = 0.44                   # weights (score + size, sum 1.0)
size = 0.56
pick = "topsis"                # topsis | max_score | min_size
min_closeness_gain = 0.02      # optional per-preset override
[optimizer.topsis.presets.Efficient.reference]
"2160" = { floor = 4.0, lo = 7.1, target = 7.8, hi = 8.8, ceiling = 15 }   # GiB/h per resolution
"1080" = { floor = 1.5, lo = 2.8, target = 3.5, hi = 4.3, ceiling = 7.5 }
# … 720 / 480 …
```

A Radarr/Sonarr profile attaches to the preset whose name is a case-insensitive **substring** of
the profile name (`2160p Quality` → Quality). Pin or customize an exact profile with
`[optimizer.topsis.profiles."<name>"]`, which may set `preset`, `weights`, `pick`, `reference`, or
`min_closeness_gain` (anything omitted is inherited from the matched preset). Validation at load
time enforces weights-sum-to-1, `0 ≤ min_closeness_gain < 1`, `floor < lo ≤ target ≤ hi < ceiling`, and a
known `pick` method.

---

## 6. The worker loop

The optimizer is a continuous, interval-driven worker (the unmonitor job keeps its own cron).

```mermaid
flowchart TD
    Start(["Worker start"]) --> Refresh["Fetch item list<br/>(every list_refresh_minutes)"]
    Refresh --> Pool["Build active pool:<br/>has-file − satisfied(in window) − evaluated this pass"]
    Pool --> Empty{"pool empty?"}
    Empty -->|yes| Idle["Idle sleep, then re-check / refresh"]
    Idle --> Refresh
    Empty -->|no| Q["GET /api/v3/queue<br/>(pace gate + 'already downloading' set)"]
    Q --> Gate{"queue ≤ queue_max?"}
    Gate -->|no| WaitQ["Sleep process_interval_seconds"]
    WaitQ --> Q
    Gate -->|yes| Pick["Pick item: random or ordered (pick_order)"]
    Pick --> Dl{"already in queue?"}
    Dl -->|yes| Skip["Skip: already downloading"]
    Dl -->|no| Eval["Evaluate (pipeline §3)"]
    Eval --> Decide{"ACT or HOLD?"}
    Decide -->|HOLD| Sat["Mark satisfied"]
    Decide -->|ACT| Grab["POST grab: record nothing"]
    Skip --> Settle["Sleep process_interval_seconds (settle)"]
    Sat --> Settle
    Grab --> Settle
    Settle --> Empty
```

- **One queue fetch per iteration** serves both the pace gate (`queue_max`) and the "already
  downloading?" skip, so there's **no in-flight state** to track and a restart needs no
  reconciliation.
- `process_interval_seconds` (default 15, min 10) doubles as a **settle delay**: after a grab,
  Radarr needs a moment to register the release in the queue before the next `queue_max` check.
- `pick_order` only changes which items are improved first (random, or ordered by title / file
  size / date added / release date), never the per-item decision.
- A grab **records nothing**. Each item is remembered for the current **pass** so it isn't
  re-picked; a list refresh updates the candidate set but doesn't restart the pass. When the pass
  is fully covered it resets.

### Per-item state lifecycle

State (`/data/state.json`, keyed by item id) records exactly one thing, whether an item is
**satisfied**, which is what makes failure handling self-correcting.

```mermaid
stateDiagram-v2
    [*] --> Unprocessed: not in state
    Unprocessed --> Unprocessed: ACT: grab posted (state unchanged)
    Unprocessed --> Satisfied: HOLD: nothing better than current
    Satisfied --> Unprocessed: reevaluate_after_days elapsed
```

- A grab is **never recorded**. A grab that **succeeds** replaces the file; next evaluation finds
  nothing better → **satisfied** → leaves the pool. A grab that **fails** was never satisfied, so
  the item is retried, and by then the dead release is blocklisted, so pre-filter 1 drops it and
  the next-best candidate is picked. Repeated failures walk down the ranking until one sticks.
- A download **in progress** is skipped via live queue membership, never re-grabbed.

> **Dependency:** this relies on Radarr/Sonarr **Failed Download Handling** (default on) to
> blocklist dead releases. Without it, a failed grab wouldn't be de-prioritized next pass.

---

## 7. The Unmonitor job

A separate, cron-scheduled pass (`[unmonitor]`, default `0 4 * * *`) that **unmonitors** items so
the \*arr apps stop chasing upgrades off their RSS feeds just because newer releases appeared.

An item is unmonitored when **all** of these hold (`features/unmonitor/candidates.py`):

1. it is currently **monitored** (otherwise nothing to do);
2. it **has a file**: never unmonitor a wanted-but-undownloaded item (that would mean "give
   up"); only stop chasing *upgrades*;
3. if `require_cutoff_met = true`, its quality **cutoff is met** (don't stop early if it hasn't
   reached the target quality yet);
4. it is at least `days` old, measured against the configured `release_type` date
   (`digitalRelease` for Radarr, `airDateUtc` for Sonarr, by default).

This pairs naturally with the optimizer: the optimizer keeps improving files by `hasFile`
regardless of monitored state, while Unmonitor strips the monitoring that would otherwise have
Radarr/Sonarr fighting it with fresh RSS grabs.

---

*`dry_run = true` makes both features log every would-be action without changing anything.
`tools/weight_lab.py` renders how each preset scores and picks across sample releases, and
`tools/weight_lab.py --dataset <jsonl>` runs the presets against a real gathered library sample
(see `tools/gather_training_data.py`).*
