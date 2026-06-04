"""Phase-1 experiment for the size-model redesign (read-only, offline). NO engine change.

Compares two candidate size models, both 2-axis TOPSIS (logistic n_score from the shipped config +
a 5-point trapezoid n_size, resolution dropped to a guard) over the harvested 250-movie set:

  A. absolute bands + outlier prefilter - per-resolution {floor, lo, target, hi, ceiling} GiB/h
     placed from the GLOBAL size percentiles, same for every movie; plus a relative outlier drop.
  B. relative percentile - per movie, the band is placed from THAT movie's own candidate size
     distribution, so it scales with encode difficulty.

For each model it reports, per profile: which release each profile would pick, whether the five
profiles diverge, whether the pick is the top-score release inside the band (score is driving),
whether any pick is a lone-small outlier, where picks land in the size distribution, and a
varying-candidate-set oscillation count (re-grabs when the visible candidate set is resampled).

    uv run python tools/size_band_lab.py --in reports/training_data_radarr.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.topsis import HARD_REJECT_KEYWORDS, Topsis

# profile -> (score_weight, size_weight, anchor percentile of the size distribution).
# Weights are the shipped 3-axis weights with resolution dropped and re-summed to 1.0.
PROFILES: dict[str, tuple[float, float, float]] = {
    "Remux": (0.94, 0.06, 92),
    "Quality": (0.86, 0.14, 77),
    "Balanced": (0.56, 0.44, 50),
    "Efficient": (0.44, 0.56, 30),
    "Compact": (0.22, 0.78, 10),
}
HALF_WIDTH_PCT = 15.0  # band half-width in percentile points around the anchor
OUTLIER_FRAC = 0.5  # drop a release whose gbh < OUTLIER_FRAC * median(cluster gbh)
SHOULDER = 0.75  # n_size at lo/hi


@dataclass
class Cand:
    score: int
    gbh: float
    res: int
    size_gb: float
    title: str


def _eligible(rels: list[dict]) -> list[Cand]:
    """Hard-rejection + non-negative-score filter, mapped to simple Cand records."""
    out: list[Cand] = []
    for r in rels:
        if r.get("temporarily_rejected"):
            continue
        if any(
            any(k in reason for k in HARD_REJECT_KEYWORDS) for reason in (r.get("rejections") or [])
        ):
            continue
        s = r.get("score")
        if s is None or s < 0:
            continue
        out.append(
            Cand(
                s,
                float(r.get("gbh") or 0.0),
                int(r.get("resolution") or 0),
                float(r.get("size_gb") or 0.0),
                r.get("title") or "?",
            )
        )
    return out


def _gap_cut(cands: list[Cand], gap: float) -> list[Cand]:
    """Keep the top score cluster down to the first relative drop > gap."""
    srt = sorted(cands, key=lambda c: -c.score)
    if not srt:
        return []
    kept = [srt[0]]
    for prev, cur in zip(srt, srt[1:], strict=False):
        if prev.score > 0 and (prev.score - cur.score) / prev.score > gap:
            break
        kept.append(cur)
    return kept


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _rank_pct(sorted_vals: list[float], x: float) -> float:
    """Percentile position of x within sorted_vals (0..100)."""
    if not sorted_vals:
        return 0.0
    below = sum(1 for v in sorted_vals if v < x)
    return 100.0 * below / len(sorted_vals)


def trapezoid(
    gbh: float, floor: float, lo: float, target: float, hi: float, ceiling: float
) -> float:
    """5-point trapezoid: 0 outside [floor, ceiling], SHOULDER at lo/hi, 1.0 at target."""
    if gbh <= floor or gbh >= ceiling:
        return 0.0
    if gbh < lo:
        return SHOULDER * (gbh - floor) / (lo - floor) if lo > floor else SHOULDER
    if gbh <= target:
        return SHOULDER + (1 - SHOULDER) * (gbh - lo) / (target - lo) if target > lo else 1.0
    if gbh <= hi:
        return 1.0 - (1 - SHOULDER) * (gbh - target) / (hi - target) if hi > target else 1.0
    return SHOULDER * (ceiling - gbh) / (ceiling - hi) if ceiling > hi else SHOULDER


def closeness(n_score: float, n_size: float, w_score: float, w_size: float) -> float:
    d_ideal = math.sqrt(w_score * (1 - n_score) ** 2 + w_size * (1 - n_size) ** 2)
    d_anti = math.sqrt(w_score * n_score**2 + w_size * n_size**2)
    tot = d_ideal + d_anti
    return 0.0 if tot == 0 else d_anti / tot


def band_from_pcts(
    vals_sorted: list[float], anchor: float
) -> tuple[float, float, float, float, float]:
    """5 band points (gbh) for an anchor percentile over a sorted gbh population."""
    floor = _pct(vals_sorted, 2)
    ceiling = _pct(vals_sorted, 98)
    lo = _pct(vals_sorted, max(2, anchor - HALF_WIDTH_PCT))
    target = _pct(vals_sorted, anchor)
    hi = _pct(vals_sorted, min(98, anchor + HALF_WIDTH_PCT))
    # keep strictly increasing for the trapezoid
    lo = max(lo, floor + 1e-6)
    hi = max(hi, target + 1e-6)
    ceiling = max(ceiling, hi + 1e-6)
    return floor, lo, target, hi, ceiling


def drop_outliers(cands: list[Cand]) -> list[Cand]:
    if len(cands) < 3:
        return cands
    med = statistics.median(c.gbh for c in cands)
    return [c for c in cands if c.gbh >= OUTLIER_FRAC * med]


class Model:
    """A size model: produces band points for a candidate set + profile anchor."""

    def __init__(self, name: str, global_pcts: dict[int, list[float]]):
        self.name = name
        self.global_pcts = global_pcts  # res -> sorted gbh list (for absolute model)

    def _res_pop(self, res: int) -> list[float]:
        if res in self.global_pcts:
            return self.global_pcts[res]
        # nearest defined resolution at/below, else any
        below = [r for r in sorted(self.global_pcts) if r <= res]
        key = below[-1] if below else sorted(self.global_pcts)[0]
        return self.global_pcts[key]

    def band(self, cands: list[Cand], res: int, anchor: float) -> tuple[float, ...]:
        if self.name == "absolute":
            return band_from_pcts(self._res_pop(res), anchor)
        pop = sorted(c.gbh for c in cands) or [0.0]
        return band_from_pcts(pop, anchor)


def pick(
    topsis: Topsis,
    model: Model,
    cands: list[Cand],
    res: int,
    anchor: float,
    w_score: float,
    w_size: float,
) -> Cand | None:
    if not cands:
        return None
    floor, lo, target, hi, ceiling = model.band(cands, res, anchor)
    best, best_clo = None, -1.0
    for c in cands:
        ns = topsis.normalize_score(c.score)
        nz = trapezoid(c.gbh, floor, lo, target, hi, ceiling)
        clo = closeness(ns, nz, w_score, w_size)
        if clo > best_clo:
            best, best_clo = c, clo
    return best


def in_band_top_score(cands: list[Cand], band: tuple[float, ...]) -> Cand | None:
    _floor, lo, _target, hi, _ceiling = band
    inb = [c for c in cands if lo <= c.gbh <= hi]
    return max(inb, key=lambda c: c.score) if inb else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="reports/training_data_radarr.jsonl", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.inp.read_text().splitlines() if line.strip()]
    cfg = default_topsis()
    topsis = Topsis(cfg)
    gap = cfg.score_gap

    # Per-movie candidate set = eligible + gap-cut, restricted to the profile target resolution.
    movies: list[tuple[int, list[Cand]]] = []
    res_pop: dict[int, list[float]] = {}
    for r in rows:
        tr = r["profile"]["target_resolution"] or 0
        cands = _gap_cut(_eligible(r["releases"]), gap)
        cands = [c for c in cands if c.res == tr] or cands  # fall back to all if none at tr
        cands = drop_outliers(cands)
        if len(cands) < 2:
            continue
        movies.append((tr, cands))
        for c in cands:
            res_pop.setdefault(c.res, []).append(c.gbh)
    for k in res_pop:
        res_pop[k].sort()

    models = {"absolute": Model("absolute", res_pop), "relative": Model("relative", res_pop)}

    lines: list[str] = []
    w = lines.append
    w("# Size-model experiment: absolute bands vs relative percentile\n")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    w(f"Source `{args.inp}` ({len(rows)} movies, {len(movies)} usable)  |  {stamp}\n")
    w(f"shoulder={SHOULDER}, half-width={HALF_WIDTH_PCT}pct, outlier_frac={OUTLIER_FRAC}\n")

    w("\n## Per-resolution candidate GiB/h percentiles (absolute bands placed from these)\n")
    w("| res | n | P5 | P10 | P25 | P50 | P75 | P90 | P95 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for res in sorted(res_pop, reverse=True):
        v = res_pop[res]
        w(
            f"| {res} | {len(v)} | "
            + " | ".join(f"{_pct(v, p):.1f}" for p in (5, 10, 25, 50, 75, 90, 95))
            + " |"
        )

    for mname, model in models.items():
        # picks[profile] = list of (movie_idx, pick, cands, band)
        diverge = Counter()  # distinct-pick-count per movie
        score_driving = Counter()
        outlier_hits = 0
        land_pct: dict[str, list[float]] = {p: [] for p in PROFILES}
        for tr, cands in movies:
            picks_this = {}
            for prof, (ws, wz, anchor) in PROFILES.items():
                p = pick(topsis, model, cands, tr, anchor, ws, wz)
                if p is None:
                    continue
                picks_this[prof] = p
                band = model.band(cands, tr, anchor)
                top = in_band_top_score(cands, band)
                if top is not None and p.title == top.title:
                    score_driving[prof] += 1
                pop = sorted(c.gbh for c in cands)
                land_pct[prof].append(_rank_pct(pop, p.gbh))
                med = statistics.median(c.gbh for c in cands)
                if p.gbh < OUTLIER_FRAC * med:
                    outlier_hits += 1
            diverge[len({p.title for p in picks_this.values()})] += 1

        w(f"\n## Model: {mname}\n")
        w("**Divergence** (distinct releases picked by the 5 profiles, per movie):")
        w("")
        w("| distinct picks | movies |")
        w("| ---: | ---: |")
        for k in sorted(diverge):
            w(f"| {k} | {diverge[k]} |")
        w(f"\n**Outlier picks** (pick gbh < {OUTLIER_FRAC}x median): {outlier_hits} (target: 0)\n")
        w("\n**Score-driving** (pick == top-score release in band) + **landing pct**:\n")
        w("| profile | anchor pct | score-driving | mean landing pct |")
        w("| --- | ---: | ---: | ---: |")
        for prof, (_ws, _wz, anchor) in PROFILES.items():
            n = len(land_pct[prof])
            sd = f"{100 * score_driving[prof] / n:.0f}%" if n else "-"
            ml = f"{statistics.mean(land_pct[prof]):.0f}" if land_pct[prof] else "-"
            w(f"| {prof} | {anchor:.0f} | {sd} | {ml} |")

        # Oscillation: resample candidate set, count re-grabs.
        regrabs = grabs = 0
        sample = movies if len(movies) <= 80 else rng.sample(movies, 80)
        for tr, cands in sample:
            if len(cands) < 4:
                continue
            for _prof, (ws, wz, anchor) in PROFILES.items():
                cur = rng.choice(cands)
                left: set[str] = set()
                for _ in range(40):
                    subset = [c for c in cands if rng.random() < 0.75] or cands
                    if cur.title not in {c.title for c in subset}:
                        subset = subset + [cur]
                    p = pick(topsis, model, subset, tr, anchor, ws, wz)
                    if p is None or p.title == cur.title:
                        continue
                    # swap only on a real closeness gain under the CURRENT subset
                    band = model.band(subset, tr, anchor)
                    nz_cur = trapezoid(cur.gbh, *band)
                    nz_p = trapezoid(p.gbh, *band)
                    clo_cur = closeness(topsis.normalize_score(cur.score), nz_cur, ws, wz)
                    clo_p = closeness(topsis.normalize_score(p.score), nz_p, ws, wz)
                    if clo_p < clo_cur + cfg.default_min_closeness_gain:
                        continue
                    grabs += 1
                    if p.title in left:
                        regrabs += 1
                    left.add(cur.title)
                    cur = p
        w(
            f"\n**Oscillation** (resampled-set walk, {len(sample)} movies x 5 profiles): "
            f"{grabs} grabs, **{regrabs} re-grabs** (target: 0)\n"
        )

    out = args.out or Path("reports") / f"size_band_lab_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
