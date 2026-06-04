# Quality preset retune: stop downgrading high-score 2160p files

Date: 2026-06-02

## Problem

The shipped **Quality** preset was downgrading large, near-perfect 2160p files. Reported case:
Dune (2021), current file score 949,600 / 46.2 GB (~17.9 GiB/h), optimizer wanted a
920,000 / 14 GB release. Generally 2160p Quality files were dropping from ~18 GiB/h to ~8.

Root cause: at the top of the Profilarr score scale every good release normalizes to ~0.92-0.95
(linear over `[0, 1,000,000]`), so the score axis is nearly flat and the old size weight (0.20)
plus a low 2160p band (`target 11 / bloat 20`) let the size axis dominate. A 46 GB file read as
near-bloat (`n_size = 0.236`) and almost any smaller in-band release looked like a big win.

## Change ([defaults.toml](../optimizarr/defaults.toml))

| | weights (score/res/size) | 2160p floor/target/bloat |
| --- | --- | --- |
| before | 0.65 / 0.15 / 0.20 | 5 / 11 / 20 |
| after  | 0.73 / 0.15 / 0.12 | 5 / 16 / 30 |

Lower resolutions raised in step to keep the preset coherent: 1080p 1.5/7/14, 720p 0.5/2.5/6,
480p 0.25/1/3.5. Docs updated in [ALGORITHM.md](../ALGORITHM.md).

## Validation (real `decide()` via `default_topsis()`, Dune runtime 2.567 h)

| outcome | scenario | current closeness | best/pick | Δ |
| --- | --- | --- | --- | --- |
| HOLD | 949,600 / 46.2 GB  vs  920,000 / 14 GB (the report) | 0.935 | (no legal swap) | - |
| HOLD | 949,600 / 46.2 GB  vs  935,000 / 20.5 GB (~8 GiB/h) | 0.935 | (no legal swap) | - |
| ACT  | 949,600 / 46.2 GB  vs  949,600 / 20.5 GB (equal score, leaner) | 0.935 | 0.957 | +0.022 |
| ACT  | 700,000 / 46 GB  vs  1,000,000 / 30 GB (true upgrade) | 0.748 | 1.000 | +0.252 |
| ACT  | 949,600 / 80 GB (~31 GiB/h)  vs  949,600 / 40 GB (real bloat trim) | 0.720 | 0.957 | +0.237 |

Net: score-losing downgrades of high-score 2160p files now HOLD; equal-or-better-score leaner
files and genuine bloat trims still ACT.

## Test suite

`pytest -q`: 137 passed.
