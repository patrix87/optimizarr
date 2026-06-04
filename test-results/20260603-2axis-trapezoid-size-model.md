# 2-axis scoring + 5-point trapezoid size model (Phase 2)

Date: 2026-06-03

## What changed

Replaced the 3-axis TOPSIS (score, resolution, size) + one-sided size curve with:

- **Two axes** (score, size). Resolution is no longer scored: Profilarr folds it into the score and
  a hard guard in decision.py already forbids dropping below the profile target. Weights are now
  `{score, size}` summing to 1.0.
- **5-point trapezoid size band** `{floor, lo, target, hi, ceiling}` (renamed `bloat` -> `ceiling`).
  `n_size` is 0 outside `[floor, ceiling]`, rises to `size_shoulder` (0.85) at lo, peaks at 1.0 at
  target, falls back to the shoulder at hi. Inside `[lo, hi]` score drives the pick; `target` sits
  at a per-profile size percentile (Compact ~P10, Efficient ~P30, Balanced ~P50, Quality ~P77,
  Remux remux-range), so the five profiles aim at different sizes and pick different releases.
- **Outlier prefilter**: drop a candidate below `outlier_frac` (0.5) x the gap-cut cluster median
  GiB/h ("a good encode has corroborating peers").
- **No-inflate-at-score-loss invariant**: decision.py forbids grabbing a release that is bigger AND
  lower-or-equal score. The old one-sided curve gave this for free; the peaked trapezoid does not,
  so it is now explicit. A larger file is grabbed only on a genuine score upgrade.
- All presets pick by `topsis` (divergence comes from band placement, not the pick method);
  `max_score`/`min_size` remain for overrides.

## Why these choices (Phase 1 experiment, tools/size_band_lab.py)

- **Relative percentile bands were rejected**: 1796 re-grabs in a resampled-set oscillation walk
  vs **0** for absolute bands. Absolute bands + outlier prefilter is provably stable.
- Shoulder swept 0.6-0.9; 0.85 gives the best in-band score-driving without disturbing where the
  size-leaning profiles land.

## Validation (real engine over 250-movie set)

- **Divergence** (distinct picks across the 5 presets, 2160p movies): 38 all-5, 22 of 4, 7 of 3,
  1 of 2, 1 of 1.
- **Score-downgrades under each item's own profile**: 19, all on size-leaning profiles (17 Efficient
  2160p, 1 Efficient 1080p, 1 Balanced) and **all shrink the file** (the intended score-for-size
  trade). **Zero** on Quality or Remux.
- **Zero grow-at-lower-score swaps** (was 52 before the invariant; the worst was a 220k-point drop
  to grow 1.9 -> 3.8 GB, now blocked).
- **Dune** (949,600 / 46.2 GB vs 920,000 / 14 GB under 2160p Quality): HOLD.
- Weights sweep confirmed raising score weight does NOT fix the grows (71 -> 63 only, and hurts
  divergence) because the current files sit below floor (n_size 0); the invariant is the right fix.

## Tests / tooling

- `pytest -q`: 137 passed (incl. the no-oscillation convergence test and the restored
  `test_bigger_at_no_score_gain_is_never_grabbed`). `ruff format` + `ruff check`: clean.
- Updated: topsis.py, config.py, decision.py, defaults.toml, ALGORITHM.md, README.md,
  tools/weight_lab.py, tools/cutoff_viz.py. New: tools/size_band_lab.py.
- Reports: reports/size_band_lab_20260602_223602.md (model choice),
  reports/score_curve_fit_20260602_162909.md (score curve).
