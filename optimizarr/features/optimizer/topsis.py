"""TOPSIS release scorer over two RELATIVE axes.

For one library item, candidates are scored on two axes, each min-max normalized over the surviving
candidates (the current file is then placed on the same scale):
  - score: Profilarr customFormatScore, higher better -> n_score = (s - smin) / (smax - smin).
  - size:  GiB/h, smaller better -> n_size = (gmax - g) / (gmax - gmin).
Resolution is NOT a scored axis: a pre-filter keeps only the profile's target resolution.

Score window uses a three-tier strategy (see `_score_floor_tier`): anchor at the top of what the
indexers offer (Tier 1), expand toward the current file's score if too few survive (Tier 2), and
fall back to the full downgrade budget below the current file (Tier 3). This favors high-scoring
candidates and only widens the window when the indexer returns sparse results near the top.

A profile's `weights` combine the two normalized axes into a TOPSIS closeness; the only
per-profile knob is the weights. There is no absolute size table or shape, and no logistic score
curve -- everything is relative to what the indexers actually offer for this movie.
"""

from __future__ import annotations

import math

from optimizarr.features.optimizer.config import ResolvedProfile, SizeBounds, TopsisConfig

GB = 1024**3

# Rejections meaning "can't be considered at all"; everything else is advisory.
HARD_REJECT_KEYWORDS = (
    "blocklisted",
    "Unable to parse",
    "Unknown Movie",
    "Not enough seeders",
)


def _release_gbh(release: dict, runtime_h: float) -> float:
    if not runtime_h or runtime_h <= 0:
        return 0.0
    return (release.get("size", 0) / GB) / runtime_h


def _release_resolution(release: dict) -> int:
    return ((release.get("quality") or {}).get("quality") or {}).get("resolution") or 0


def eligible(releases: list[dict]) -> list[dict]:
    """Drop hard-rejected releases (blocklist, parse failure, dead torrents)."""
    keep = []
    for r in releases:
        if r.get("temporarilyRejected"):
            continue
        rejections = r.get("rejections") or []
        if any(any(k in reason for k in HARD_REJECT_KEYWORDS) for reason in rejections):
            continue
        keep.append(r)
    return keep


def _score_floor_tier(
    scores: list[int],
    current_score: int | None,
    score_window: int,
    min_pool: int,
) -> tuple[int, int]:
    """Three-tier score floor over a list of non-negative scores. Returns (floor, tier).

    Tier 1 -- anchor at top: floor = max(scores) - score_window. Keeps the window tight when
              the indexer returns many high-scoring releases.
    Tier 2 -- expand to current: floor = current_score. Tried when Tier 1 yields too few; pulls
              in releases scored at or above the current file to round out the pool.
    Tier 3 -- full budget: floor = max(0, current_score - score_window). Last resort; the
              original behavior, reached only when both tighter tiers are too sparse.
    """
    max_score = max(scores, default=0)

    t1_floor = max(0, max_score - score_window)
    if sum(1 for s in scores if s >= t1_floor) >= min_pool:
        return t1_floor, 1

    if current_score is not None:
        t2_floor = max(0, current_score)
        if sum(1 for s in scores if s >= t2_floor) >= min_pool:
            return t2_floor, 2

    t3_floor = max(0, current_score - score_window) if current_score is not None else 0
    return t3_floor, 3


def _norm(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Min-max to [0,1] (clamped). invert=True makes a smaller value score higher. A degenerate
    range (hi <= lo) means the axis does not discriminate -> 1.0 for everyone."""
    if hi <= lo:
        return 1.0
    t = (value - lo) / (hi - lo)
    if invert:
        t = 1.0 - t
    return min(1.0, max(0.0, t))


class Topsis:
    """Config-driven scorer + picker. One instance per optimizer run."""

    def __init__(self, cfg: TopsisConfig):
        self.cfg = cfg

    # ----- profile -> preset resolution -----

    def _match_preset(self, profile_name: str) -> str:
        """First preset whose name is a case-insensitive substring of the profile name;
        preset definition order breaks ties (so Remux wins over Quality in 'Remux Quality')."""
        low = profile_name.lower()
        for name in self.cfg.presets:
            if name.lower() in low:
                return name
        return self.cfg.default_preset

    def resolve_profile(self, profile_name: str | None) -> ResolvedProfile:
        """Resolve a profile name to weights, honoring an exact-name override,
        then name-keyword preset matching, then default_preset."""
        cfg = self.cfg
        override = cfg.profiles.get(profile_name) if profile_name else None
        if override and override.preset:
            base = cfg.presets[override.preset]
        elif profile_name:
            base = cfg.presets[self._match_preset(profile_name)]
        else:
            base = cfg.presets[cfg.default_preset]
        weights = override.weights if (override and override.weights) else base.weights
        return ResolvedProfile(weights=weights)

    def bounds_for(self, res: int) -> tuple[float, float]:
        """(floor, ceiling) GiB/h for a resolution; nearest defined at or below, else lowest."""
        bounds: SizeBounds = self.cfg.size_bounds
        if res in bounds:
            return bounds[res]
        keys = sorted(bounds)
        below = [k for k in keys if k <= res]
        return bounds[below[-1]] if below else bounds[keys[0]]

    # ----- pre-filters -----

    def _apply_score_window(
        self, releases: list[dict], current_score: int | None
    ) -> tuple[list[dict], int, int]:
        """Three-tier window; returns (filtered, effective_floor, tier_number)."""
        nonneg = [(r, r.get("customFormatScore") or 0) for r in releases]
        nonneg = [(r, s) for r, s in nonneg if s >= 0]
        scores = [s for _, s in nonneg]
        floor, tier = _score_floor_tier(
            scores, current_score, self.cfg.score_window, self.cfg.min_candidates
        )
        return [r for r, s in nonneg if s >= floor], floor, tier

    def filter_by_score_window(self, releases: list[dict], current_score: int | None) -> list[dict]:
        """Three-tier score window anchored at the top of what's available, expanding toward the
        downgrade budget only when too few candidates survive the tighter tiers."""
        filtered, _, _ = self._apply_score_window(releases, current_score)
        return filtered

    def filter_by_resolution(
        self, releases: list[dict], target_resolution: int | None
    ) -> list[dict]:
        """Keep only the profile's target resolution (so the GiB/h axis is comparable). When the
        profile exposes no target, keep everything."""
        if not target_resolution:
            return releases
        return [r for r in releases if _release_resolution(r) == target_resolution]

    def filter_by_size_band(self, releases: list[dict], runtime_h: float) -> list[dict]:
        """Drop releases outside the shared per-resolution [floor, ceiling] band (fake/upscale
        below floor, bloat above ceiling)."""
        keep = []
        for r in releases:
            floor, ceiling = self.bounds_for(_release_resolution(r))
            if floor <= _release_gbh(r, runtime_h) <= ceiling:
                keep.append(r)
        return keep

    def drop_size_outliers(self, releases: list[dict], runtime_h: float) -> list[dict]:
        """Drop a release whose GiB/h is a lone outlier below the surviving cluster: below
        `outlier_frac x median(cluster GiB/h)`. Disabled when outlier_frac <= 0 or < 3 survivors."""
        frac = self.cfg.outlier_frac
        if frac <= 0 or len(releases) < 3:
            return releases
        gbhs = sorted(_release_gbh(r, runtime_h) for r in releases)
        mid = len(gbhs) // 2
        median = gbhs[mid] if len(gbhs) % 2 else (gbhs[mid - 1] + gbhs[mid]) / 2
        return [r for r in releases if _release_gbh(r, runtime_h) >= frac * median]

    def apply_prefilters(
        self,
        releases: list[dict],
        runtime_h: float,
        target_resolution: int | None,
        current_score: int | None,
    ) -> tuple[list[dict], dict]:
        """Run the inclusion filters in order; return (kept, diag) with per-stage counts."""
        diag: dict[str, object] = {"input": len(releases)}
        after_hard = eligible(releases)
        diag["after_hard_rejections"] = len(after_hard)
        after_window, score_floor, window_tier = self._apply_score_window(after_hard, current_score)
        diag["after_score_window"] = len(after_window)
        diag["score_floor"] = score_floor
        diag["window_tier"] = window_tier
        after_res = self.filter_by_resolution(after_window, target_resolution)
        diag["after_resolution"] = len(after_res)
        after_band = self.filter_by_size_band(after_res, runtime_h)
        diag["after_size_band"] = len(after_band)
        kept = self.drop_size_outliers(after_band, runtime_h)
        diag["after_outlier_drop"] = len(kept)
        diag["inclusion"] = (
            f"score >= {score_floor:,} (tier {window_tier})"
            f" + target res + [floor,ceiling] band + lone-small drop"
        )
        return kept, diag

    # ----- scoring -----

    def closeness(self, n_score: float, n_size: float, weights: dict[str, float]) -> float:
        """TOPSIS closeness in [0,1] over the two axes. 1 = ideal (best score, smallest file)."""
        w = {"n_score": weights["score"], "n_size": weights["size"]}
        a = {"n_score": n_score, "n_size": n_size}
        d_ideal = math.sqrt(sum(w[k] * (1.0 - a[k]) ** 2 for k in w))
        d_anti = math.sqrt(sum(w[k] * a[k] ** 2 for k in w))
        total = d_ideal + d_anti
        return 0.0 if total == 0 else d_anti / total

    def _attrs(self, score, res, gbh, size_gb, smin, smax, gmin, gmax) -> dict:
        return {
            "n_score": _norm(score or 0, smin, smax),
            "n_size": _norm(gbh, gmin, gmax, invert=True),
            "raw": {"score": score, "resolution": res, "gbh": gbh, "size_gb": size_gb},
        }

    def score_pool(
        self,
        kept: list[dict],
        current_file: dict | None,
        runtime_h: float,
        resolved: ResolvedProfile,
    ) -> tuple[list[tuple[dict, dict, float]], dict | None]:
        """Min-max normalize the kept candidates on both axes, score each by closeness (sorted
        best-first), and place the current file on the SAME ranges. Returns (scored, current),
        where `current` carries n_score/n_size/raw/closeness (None if there is no current score).
        Ranges are built from candidates only, so the candidate ranking is current-independent."""
        rows = [
            (
                r,
                r.get("customFormatScore"),
                _release_gbh(r, runtime_h),
                _release_resolution(r),
                (r.get("size", 0) or 0) / GB,
            )
            for r in kept
        ]
        scores = [s or 0 for _r, s, _g, _res, _sz in rows]
        gbhs = [g for _r, _s, g, _res, _sz in rows]
        smin, smax = (min(scores), max(scores)) if scores else (0, 0)
        gmin, gmax = (min(gbhs), max(gbhs)) if gbhs else (0.0, 0.0)

        scored = []
        for r, s, g, res, sz in rows:
            a = self._attrs(s, res, g, sz, smin, smax, gmin, gmax)
            scored.append((r, a, self.closeness(a["n_score"], a["n_size"], resolved.weights)))
        scored.sort(key=lambda x: (-x[2], -(x[1]["raw"]["score"] or 0), x[1]["raw"]["gbh"]))

        current = self._current_attrs(current_file, runtime_h, resolved, smin, smax, gmin, gmax)
        return scored, current

    def _current_attrs(self, cf, runtime_h, resolved, smin, smax, gmin, gmax) -> dict | None:
        if not cf:
            return None
        score = cf.get("customFormatScore")
        size_gb = (cf.get("size", 0) or 0) / GB
        gbh = (size_gb / runtime_h) if (runtime_h and runtime_h > 0) else 0.0
        res = _release_resolution(cf)
        if score is None:
            return {
                "n_score": None,
                "n_size": None,
                "closeness": None,
                "raw": {"score": None, "resolution": res, "gbh": gbh, "size_gb": size_gb},
            }
        a = self._attrs(score, res, gbh, size_gb, smin, smax, gmin, gmax)
        a["closeness"] = self.closeness(a["n_score"], a["n_size"], resolved.weights)
        return a
