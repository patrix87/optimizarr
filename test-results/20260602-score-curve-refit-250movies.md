# Score curve re-fit on 250 movies + library-wide downgrade replay

Date: 2026-06-02

## Why

The logistic score curve and the Quality preset were first tuned on a 52-movie sample whose top
score maxed at 952,600. Re-harvested a larger random sample to confirm the fit, excluding
sub-zero releases from the analysis (they are dropped by the gap-cut before scoring anyway).

## New dataset

`tools/gather_training_data.py --app radarr --count 250` (seed 0, superset of the old sample).
250 movies, 46,635 releases (28,056 with score >= 0; 18,579 negative, excluded from the fit).

- Release scores now reach **993,060** (vs 952,600 before), p99 = 986k. Confirms scores do climb
  near the 1,000,000 "perfect" anchor; the small sample undersold the top.
- Gap-cut survivors (the releases `n_score` actually ranks), >= 0: 13,756. Still multi-tier
  (~700k / ~860k / ~920-993k), competitive spread wide.

## Re-fit

Best-fit logistic to the survivor CDF: **center 792,500, width 87,500** — essentially the same
as the small-sample fit (805k/85k). Quality 2160p target lowered 16 -> 14 per operator preference
(still HOLDs the Dune case; now also takes equal-score leaner files).

**Final adopted value: center 825,000, width 87,500.** A follow-up tool, `tools/score_curve_fit.py`,
re-fit the curve on three candidate populations: all releases >= 0 (degenerate, grid-railed),
post gap-cut (~790k, ~ the survivor fit), and the **real** candidates that survive the full
prefilters (eligible + per-item size band + score gap), which fit center **825,000**. The real
population is the right basis (it is exactly what `n_score` ranks); the size band strips the
low-score junk that survives the gap-cut, so the center sits higher. Practical impact is small
(31 score-downgrades either way in replay), but it gives marginally sharper top-tier
discrimination. Set in `[optimizer.topsis]` and the `TopsisConfig` fallback. Note: many movies
had their profiles changed recently, so current_file may be far from target; that only affects
the replay sanity-check, not the curve fit (which is over release scores).

## Library-wide replay (the real test)

Replayed `decide()` over all 250 movies with shipped defaults (allow_size_increase=true,
allow_quality_downgrade=true), counting ACT decisions where the picked release scores LOWER than
the current file (a score-downgrade).

| normalization | ACT | score-downgrades | >= 20k drop |
| --- | --- | --- | --- |
| logistic (new default) | 140 | 31 | 1 |
| linear [0, 1M] (old)   | 135 | 33 | 1 |

Downgrades by profile (logistic): **28 of 31 are `2160p Efficient`** (size-leaning, by design),
2 are `1080p Efficient`, 1 is `2160p Balanced`. **Zero on any Quality profile** — the reported
regression is gone.

The remaining Efficient downgrades are the feature working: ~0.5-1% score traded for roughly
halving the file, e.g.

| profile | title | score | size |
| --- | --- | --- | --- |
| 2160p Efficient | The Batman (2022) | 989,040 -> 984,000 | 35.8 -> 16.6 GB |
| 2160p Efficient | Kingdom of the Planet of the Apes | 991,000 -> 980,000 | 25.5 -> 13.3 GB |
| 2160p Efficient | Spider-Man: No Way Home | 993,060 -> 987,000 | 27.8 -> 14.8 GB |

## Checks

`ruff format` + `ruff check`: clean. `pytest -q`: 137 passed. Dune (949,600/46.2 GB vs
920,000/14 GB) under `2160p Quality`: HOLD.
