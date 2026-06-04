"""Explore logistic score-curve fits on different candidate populations (read-only, offline).

The shipped `score_center` / `score_width` were fit to the pooled distribution of every release
with score >= 0. That includes a lot of releases the optimizer never actually has to choose
between (they fail the size band, or fall outside the per-item score-gap cluster). This tool
re-fits the logistic on tighter populations so we can see whether the curve should be calibrated
to what `n_score` really ranks:

  all_pos  : every release with customFormatScore >= 0 (the current shipped basis).
  gapcut   : after eligible() + filter_by_score_gap() -- the post gap-cut candidates.
  real     : after the FULL prefilters (eligible + per-item size band + score gap), using each
             item's resolved preset -- the real candidates that get scored and can be picked.

For each population it fits a logistic to the empirical CDF, then replays decide() over the whole
dataset under that fitted curve and counts score-downgrades, so the practical effect is visible.
It NEVER touches Radarr/Sonarr or writes config; it only reads the training JSONL and writes a
markdown report.

    uv run python tools/score_curve_fit.py --in reports/training_data_radarr.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.decision import decide
from optimizarr.features.optimizer.topsis import Topsis, eligible

KEY_SCORES = [600_000, 700_000, 800_000, 860_000, 900_000, 920_000, 949_600, 976_000, 993_000]


def _release_dict(r: dict) -> dict:
    """Reconstruct the Radarr-shaped release dict decide()/Topsis expect from a training row."""
    return {
        "guid": r.get("title"),
        "indexerId": 1,
        "title": r.get("title"),
        "customFormatScore": r.get("score"),
        "quality": {"quality": {"resolution": r.get("resolution") or 0}},
        "size": r.get("size_bytes") or 0,
        "rejections": r.get("rejections") or [],
        "temporarilyRejected": bool(r.get("temporarily_rejected")),
    }


def _current_dict(cf: dict | None) -> dict | None:
    if not cf or cf.get("score") is None:
        return None
    return {
        "id": 1,
        "customFormatScore": cf.get("score"),
        "size": cf.get("size_bytes") or 0,
        "quality": {"quality": {"resolution": cf.get("resolution") or 0}},
    }


def populations(rows: list[dict], topsis: Topsis) -> dict[str, list[int]]:
    """Pooled candidate scores for each population definition."""
    all_pos: list[int] = []
    gapcut: list[int] = []
    real: list[int] = []
    for r in rows:
        rels = [_release_dict(x) for x in r["releases"]]
        rt = r["runtime_h"]
        all_pos += [x["customFormatScore"] for x in rels if (x["customFormatScore"] or -1) >= 0]
        gap = topsis.filter_by_score_gap(eligible(rels))
        gapcut += [x["customFormatScore"] for x in gap]
        resolved = topsis.resolve_profile(r["profile"]["name"])
        kept, _ = topsis.apply_prefilters(rels, rt, resolved.reference)
        real += [x["customFormatScore"] for x in kept]
    return {"all_pos": sorted(all_pos), "gapcut": sorted(gapcut), "real": sorted(real)}


def _pct(a: list[int], p: float) -> float:
    if not a:
        return float("nan")
    k = (len(a) - 1) * p
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def fit_logistic(scores: list[int]) -> tuple[int, int]:
    """Grid least-squares fit of n=1/(1+exp(-(s-c)/w)) to the population's empirical CDF."""
    import bisect

    n = len(scores)
    lo, hi = scores[0], scores[-1]
    xs = list(range(int(lo), int(hi) + 1, 5000)) or [lo]
    emp = [bisect.bisect_right(scores, x) / n for x in xs]
    best: tuple[float, int, int] | None = None
    for c in range(650_000, 960_001, 2500):
        for w in range(20_000, 140_001, 2500):
            sse = sum(
                (1 / (1 + math.exp(-(x - c) / w)) - e) ** 2 for x, e in zip(xs, emp, strict=True)
            )
            if best is None or sse < best[0]:
                best = (sse, c, w)
    assert best is not None
    return best[1], best[2]


def logistic(s: float, c: float, w: float) -> float:
    return 1 / (1 + math.exp(-(s - c) / w))


def replay_downgrades(rows: list[dict], topsis: Topsis) -> tuple[int, int, int]:
    """Replay decide() over the dataset; return (ACT, score-downgrades, downgrades >= 20k)."""
    act = dn = big = 0
    for r in rows:
        cur = _current_dict(r.get("current_file"))
        if cur is None:
            continue
        d = decide(
            topsis,
            [_release_dict(x) for x in r["releases"]],
            r["runtime_h"],
            r["profile"]["name"],
            r["profile"]["target_resolution"],
            cur,
        )
        if d.action != "ACT":
            continue
        act += 1
        ps, cs = (d.pick or {}).get("score"), cur["customFormatScore"]
        if ps is not None and cs is not None and ps < cs:
            dn += 1
            if cs - ps >= 20_000:
                big += 1
    return act, dn, big


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="reports/training_data_radarr.jsonl", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.inp.read_text().splitlines() if line.strip()]
    cfg = default_topsis()
    topsis = Topsis(cfg)
    pops = populations(rows, topsis)

    lines: list[str] = []
    w = lines.append
    w("# Score-curve fit per candidate population\n")
    w(f"Source: `{args.inp}` ({len(rows)} movies)  |  generated {datetime.now():%Y-%m-%d %H:%M}\n")
    w(f"Shipped default: center={int(cfg.score_center):,} width={int(cfg.score_width):,}\n")

    w("\n## Population sizes and percentiles\n")
    w("| population | n | p25 | p50 | p75 | p90 | p95 | max |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, sc in pops.items():
        w(
            f"| {name} | {len(sc):,} | "
            + " | ".join(f"{_pct(sc, p):,.0f}" for p in (0.25, 0.5, 0.75, 0.90, 0.95, 1.0))
            + " |"
        )

    fits = {name: fit_logistic(sc) for name, sc in pops.items()}
    w("\n## Fitted logistic per population\n")
    w("| population | center | width |")
    w("| --- | ---: | ---: |")
    for name, (c, wd) in fits.items():
        w(f"| {name} | {c:,} | {wd:,} |")

    w("\n## n_score at key scores (each fit + shipped default)\n")
    header = "| score | " + " | ".join(fits) + " | shipped |"
    w(header)
    w("| ---: " + "| ---: " * (len(fits) + 1) + "|")
    for s in KEY_SCORES:
        row = f"| {s:,} | "
        row += " | ".join(f"{logistic(s, c, wd):.3f}" for c, wd in fits.values())
        row += f" | {logistic(s, cfg.score_center, cfg.score_width):.3f} |"
        w(row)

    w("\n## Replay impact (decide() over the whole dataset)\n")
    w("| curve | center | width | ACT | score-downgrades | >=20k |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    variants = {**fits, "shipped": (int(cfg.score_center), int(cfg.score_width))}
    for name, (c, wd) in variants.items():
        t = Topsis(replace(cfg, score_center=float(c), score_width=float(wd)))
        act, dn, big = replay_downgrades(rows, t)
        w(f"| {name} | {c:,} | {wd:,} | {act} | {dn} | {big} |")
    tl = Topsis(replace(cfg, score_norm="linear"))
    act, dn, big = replay_downgrades(rows, tl)
    w(f"| linear[0,1M] | - | - | {act} | {dn} | {big} |")

    out = args.out or Path("reports") / f"score_curve_fit_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
