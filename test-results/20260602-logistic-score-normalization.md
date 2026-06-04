# Logistic score normalization (fix flat top-end of the score axis)

Date: 2026-06-02

## Problem

Linear `n_score` over `[0, 1,000,000]` made the score axis nearly flat where real releases live.
The gap-cut keeps only candidates within `score_gap` (20%) of the top, so survivors sit in
~`[800k, 1M]`, i.e. the top 0.20 of the axis, while size/resolution use the full `[0,1]`.
Effective discriminating power (`weight x range-used`) for the Quality preset was score ~0.13 vs
size ~0.20, so size out-voted score despite a 3x higher weight. This drove the 2160p Quality
downgrades (Dune: 949,600/46.2 GB wanting 920,000/14 GB).

## Data (reports/training_data_radarr.jsonl, 52 movies, 9,684 releases)

- No release reaches 1,000,000; real max is **952,600**, so linear wastes the top ~5% of the axis.
- Gap-cut survivors are **bimodal** (~720k tier and ~850-952k top tier), per-item competitive
  spread ~245k (median). The axis needs resolution across ~700k..952k, not a hard window.
- Best-fit logistic to the survivor CDF: **center 805,000, width 85,000**.

Top-end resolution (Dune is 949.6k vs 920k):

| gap | linear /1M | logistic 805k/85k |
| --- | --- | --- |
| 950k - 920k | +0.030 | **+0.051** |

## Why a fixed curve, not relative-to-candidate-list scoring

Relative (within-set) scoring would make a release's closeness depend on what the indexers
returned that minute, breaking the invariant that closeness is a fixed function of the release
alone. That invariant is what guarantees no oscillation. Relative scoring can therefore churn and
even downgrade (a weak set `{700k, 740k}` normalizes 740k to 1.0 and can replace a 920k file).
The logistic keeps the "resolution where releases cluster" benefit while staying a fixed,
release-only transform, so non-oscillation still holds.

## Change

- `optimizarr/features/optimizer/topsis.py`: `normalize_score` gains a logistic branch.
- `optimizarr/features/optimizer/config.py`: `TopsisConfig` gains `score_norm` (default
  `"logistic"`), `score_center` (805000), `score_width` (85000); `"linear"` still available.
- `optimizarr/defaults.toml`: documents and sets the above.
- `ALGORITHM.md`: scoring section updated.
- Quality preset retune from the prior change (2160p target 16 / bloat 30, weights
  0.73/0.15/0.12) is kept: the curve fixes score discrimination globally; the preset still sets
  what 2160p file sizes the profile accepts.

## Validation (real `decide()` via `default_topsis()`, "2160p Quality", Dune runtime 2.567 h)

| outcome | scenario | notes |
| --- | --- | --- |
| HOLD | 949,600/46.2 GB vs 920,000/14 GB | the report; was ACT |
| HOLD | 949,600/46.2 GB vs 935,000/20.5 GB (~8 GiB/h) | small score drop refused |
| HOLD | 949,600/46.2 GB vs 949,600/20.5 GB | same-score shrink; Quality is quality-first |
| ACT  | 920,000/30 GB vs 949,600/25 GB | genuine score upgrade, also smaller |
| ACT  | 700,000/46 GB vs 949,600/25 GB | upgrade a poor file |
| ACT  | 949,600/80 GB vs 949,600/40 GB | real bloat trim at equal score |
| HOLD | 924,600/30 GB vs 720,000/12 GB | low-tier downgrade now correctly refused |

`ruff format` + `ruff check`: clean. `pytest -q`: 137 passed.

## Tunables

`score_center` / `score_width` in `[optimizer.topsis]`. Smaller width = steeper = more
resolution at the very top (920 vs 950) but flatter tier separation; larger = gentler. The
same-score-shrink HOLD above is a Quality choice (size weight 0.12); raise the Quality `size`
weight to re-enable free same-quality shrinks (the curve keeps score safe either way).
