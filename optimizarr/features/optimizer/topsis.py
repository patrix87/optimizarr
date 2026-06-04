"""TOPSIS-based release scorer + per-profile pickers.

Multi-objective release scoring on TWO axes, each normalized to [0,1]:
  - score: Profilarr customFormatScore through a fixed transform (logistic by default), higher
           better. Resolution is NOT a scored axis — Profilarr folds it into the score, and a hard
           guard in decision.py forbids dropping below the profile target.
  - size:  GiB/h on a 5-point TRAPEZOID `{floor, lo, target, hi, ceiling}` — n_size is 0 outside
           [floor, ceiling], rises to `size_shoulder` at lo, peaks at 1.0 at target, falls back to
           `size_shoulder` at hi. The flat-ish [lo, hi] band is the profile's "good size" region:
           inside it score drives the pick; outside it the steep shoulders pull back toward the
           band. A too-small file IS now penalized (so it can be replaced by one nearer the band).

Inclusion (before scoring): drop hard rejections, drop outside the per-resolution size band
(gbh < floor or gbh > ceiling), gap-cut the score tail, then drop lone-small size outliers.

The size table is PER-PRESET (config-driven): each resolved profile carries its own table +
weights + pick + a `min_closeness_gain` swap margin. The swap rule lives in decision.py (a
closeness-gain test); closeness is a fixed function of the release alone, so the optimizer is
provably non-oscillating (every swap strictly raises it). This module scores and picks survivors.
"""

from __future__ import annotations

import math

from optimizarr.features.optimizer.config import Reference, ResolvedProfile, TopsisConfig

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
        """Resolve a profile name to weights + pick + size table + swap margin, honoring an
        exact-name override, then name-keyword preset matching, then default_preset."""
        cfg = self.cfg
        override = cfg.profiles.get(profile_name) if profile_name else None
        if override and override.preset:
            base = cfg.presets[override.preset]
        elif profile_name:
            base = cfg.presets[self._match_preset(profile_name)]
        else:
            base = cfg.presets[cfg.default_preset]
        weights = override.weights if (override and override.weights) else base.weights
        pick = override.pick if (override and override.pick) else base.pick
        reference = override.reference if (override and override.reference) else base.reference
        min_gain = (
            override.min_closeness_gain
            if (override and override.min_closeness_gain is not None)
            else base.min_closeness_gain
        )
        return ResolvedProfile(
            weights=weights, pick=pick, reference=reference, min_closeness_gain=min_gain
        )

    def reference_for(
        self, res: int, reference: Reference
    ) -> tuple[float, float, float, float, float]:
        """(floor, lo, target, hi, ceiling) for a resolution; nearest defined at or below."""
        if res in reference:
            return reference[res]
        keys = sorted(reference)
        below = [k for k in keys if k <= res]
        return reference[below[-1]] if below else reference[keys[0]]

    # ----- pre-filters -----

    def filter_by_size_band(
        self, releases: list[dict], runtime_h: float, reference: Reference
    ) -> list[dict]:
        """Drop releases outside the preset's per-resolution size band: below `floor` (fakes /
        upscales / too-soft-for-the-resolution) or above `ceiling` (bloated)."""
        keep = []
        for r in releases:
            floor, _lo, _target, _hi, ceiling = self.reference_for(
                _release_resolution(r), reference
            )
            gbh = _release_gbh(r, runtime_h)
            if floor <= gbh <= ceiling:
                keep.append(r)
        return keep

    def drop_size_outliers(self, releases: list[dict], runtime_h: float) -> list[dict]:
        """Drop a release whose GiB/h is a lone outlier below the comparable-score cluster: below
        `outlier_frac x median(cluster GiB/h)`. "A good encode has corroborating peers", so a single
        suspiciously-small release among bigger ones (likely a bad encode) is never targeted. Run on
        the gap-cut survivors so "comparable-score" holds. Disabled when outlier_frac <= 0."""
        frac = self.cfg.outlier_frac
        if frac <= 0 or len(releases) < 3:
            return releases
        gbhs = sorted(_release_gbh(r, runtime_h) for r in releases)
        mid = len(gbhs) // 2
        median = gbhs[mid] if len(gbhs) % 2 else (gbhs[mid - 1] + gbhs[mid]) / 2
        return [r for r in releases if _release_gbh(r, runtime_h) >= frac * median]

    def filter_by_score_gap(self, releases: list[dict]) -> list[dict]:
        """Keep the top score cluster: sort desc, scan high->low, cut at the first consecutive
        relative drop greater than score_gap. Negatives are always dropped."""
        nonneg = [r for r in releases if (r.get("customFormatScore") or 0) >= 0]
        if not nonneg:
            return []
        srt = sorted(nonneg, key=lambda r: -(r.get("customFormatScore") or 0))
        kept = [srt[0]]
        for prev, cur in zip(srt, srt[1:], strict=False):
            ps = prev.get("customFormatScore") or 0
            cs = cur.get("customFormatScore") or 0
            if ps > 0 and (ps - cs) / ps > self.cfg.score_gap:
                break
            kept.append(cur)
        return kept

    def apply_prefilters(
        self, releases: list[dict], runtime_h: float, reference: Reference
    ) -> tuple[list[dict], dict]:
        """Run all pre-filters in order; return (kept, diag) with per-stage counts."""
        diag: dict[str, object] = {"input": len(releases)}
        after_hard = eligible(releases)
        diag["after_hard_rejections"] = len(after_hard)
        after_band = self.filter_by_size_band(after_hard, runtime_h, reference)
        diag["after_size_band"] = len(after_band)
        after_gap = self.filter_by_score_gap(after_band)
        diag["after_score_gap"] = len(after_gap)
        kept = self.drop_size_outliers(after_gap, runtime_h)
        diag["after_outlier_drop"] = len(kept)
        diag["inclusion"] = f"size band + gap-cut (>{self.cfg.score_gap:.0%} drop) + outlier drop"
        return kept, diag

    # ----- normalization -----

    def normalize_score(self, s: float) -> float:
        cfg = self.cfg
        if cfg.score_norm == "logistic":
            # S-curve centered at score_center. Profilarr scores bunch near the top (~950k) and
            # fall off below, so a logistic puts the axis's resolution where releases actually
            # compete, instead of wasting most of [0,1] on scores no release reaches. No hard
            # cutoff: a library whose scores all sit near one value still gets spread out.
            z = (s - cfg.score_center) / cfg.score_width
            if z <= -60:
                return 0.0
            if z >= 60:
                return 1.0
            return 1.0 / (1.0 + math.exp(-z))
        if s >= cfg.score_ideal:
            return 1.0
        if s <= cfg.score_anti_ideal:
            return 0.0
        return (s - cfg.score_anti_ideal) / (cfg.score_ideal - cfg.score_anti_ideal)

    def normalize_size(
        self, gbh: float, floor: float, lo: float, target: float, hi: float, ceiling: float
    ) -> float:
        """5-point trapezoid: 0 outside [floor, ceiling], rising to `size_shoulder` at lo, peaking
        at 1.0 at target, falling back to `size_shoulder` at hi. The flat-ish [lo, hi] band is the
        "good size" region for the profile; inside it score drives the pick, and the gentle slope to
        target lets `target` placement nudge the preference. Unlike the old one-sided curve, a
        too-small file IS penalized (it can be replaced by one nearer the band)."""
        sh = self.cfg.size_shoulder
        if gbh <= floor or gbh >= ceiling:
            return 0.0
        if gbh < lo:
            return sh * (gbh - floor) / (lo - floor) if lo > floor else sh
        if gbh <= target:
            return sh + (1 - sh) * (gbh - lo) / (target - lo) if target > lo else 1.0
        if gbh <= hi:
            return 1.0 - (1 - sh) * (gbh - target) / (hi - target) if hi > target else 1.0
        return sh * (ceiling - gbh) / (ceiling - hi) if ceiling > hi else sh

    def _attrs(
        self, score: float, res: int, gbh: float, size_gb: float, resolved: ResolvedProfile
    ) -> dict:
        floor, lo, target, hi, ceiling = self.reference_for(res, resolved.reference)
        return {
            "n_score": self.normalize_score(score or 0),
            "n_size": self.normalize_size(gbh, floor, lo, target, hi, ceiling),
            "raw": {
                "score": score,
                "resolution": res,
                "gbh": gbh,
                "size_gb": size_gb,
                "reference": (floor, lo, target, hi, ceiling),
                "target": target,
            },
        }

    def attributes_for(
        self,
        release: dict,
        runtime_h: float,
        resolved: ResolvedProfile,
        target_resolution: int | None = None,
    ) -> dict:
        """Normalized [0,1] attributes + raw values for one release. (target_resolution is accepted
        for call-site symmetry but unused: resolution is a guard in decision.py, not an axis.)"""
        size_bytes = release.get("size", 0)
        return self._attrs(
            release.get("customFormatScore", 0),
            _release_resolution(release),
            _release_gbh(release, runtime_h),
            size_bytes / GB,
            resolved,
        )

    def closeness(self, attrs: dict, weights: dict[str, float]) -> float:
        """TOPSIS closeness in [0,1] over the two axes (score, size). 1 = ideal, 0 = anti-ideal."""
        w = {"n_score": weights["score"], "n_size": weights["size"]}
        d_ideal = math.sqrt(sum(w[k] * (1.0 - attrs[k]) ** 2 for k in w))
        d_anti = math.sqrt(sum(w[k] * attrs[k] ** 2 for k in w))
        total = d_ideal + d_anti
        return 0.0 if total == 0 else d_anti / total

    def _current_resolution(self, movie_file: dict) -> int:
        """The library file's resolution bucket, read from its own `quality` block (Bluray-1080p /
        WEBDL-2160p etc.) — the same nominal field candidates report, so the two are comparable.
        This is reliable even for scope (2.40:1) content, whose raw pixel height is misleadingly
        short. Returns 0 if the quality is unknown."""
        return int(((movie_file.get("quality") or {}).get("quality") or {}).get("resolution") or 0)

    def current_attributes(
        self,
        movie_file: dict,
        runtime_h: float,
        resolved: ResolvedProfile,
        target_resolution: int | None = None,
    ) -> dict | None:
        """Normalized attributes for the existing library file (None if its score is unknown)."""
        score = movie_file.get("customFormatScore")
        if score is None:
            return None
        size = movie_file.get("size", 0) or 0
        size_gb = size / GB
        gbh = (size_gb / runtime_h) if (runtime_h and runtime_h > 0) else 0.0
        res = self._current_resolution(movie_file)
        return self._attrs(score, res, gbh, size_gb, resolved)

    def closeness_for_current_file(
        self,
        movie_file: dict,
        runtime_h: float,
        resolved: ResolvedProfile,
        target_resolution: int | None = None,
    ) -> tuple[float | None, dict]:
        """Closeness for the existing library file (None if its score is unknown)."""
        attrs = self.current_attributes(movie_file, runtime_h, resolved, target_resolution)
        if attrs is None:
            size = movie_file.get("size", 0) or 0
            size_gb = size / GB
            gbh = (size_gb / runtime_h) if (runtime_h and runtime_h > 0) else 0.0
            return None, {
                "score": None,
                "resolution": self._current_resolution(movie_file),
                "gbh": gbh,
                "size_gb": size_gb,
            }
        return self.closeness(attrs, resolved.weights), attrs["raw"]

    # ----- scoring & picking -----

    def score_candidates(
        self,
        releases: list[dict],
        runtime_h: float,
        resolved: ResolvedProfile,
        target_resolution: int | None = None,
    ) -> tuple[list[tuple[dict, dict, float]], dict]:
        """Pre-filter, then return (scored: [(release, attrs, closeness)], diag), sorted best
        first by closeness (with deterministic tie-breaks)."""
        kept, diag = self.apply_prefilters(releases, runtime_h, resolved.reference)
        scored = [
            (r, a, self.closeness(a, resolved.weights))
            for r, a in (
                (r, self.attributes_for(r, runtime_h, resolved, target_resolution)) for r in kept
            )
        ]
        scored.sort(key=lambda x: (-x[2], -(x[1]["raw"]["score"] or 0), x[1]["raw"]["gbh"]))
        return scored, diag

    def select(
        self, candidates: list[tuple[dict, dict, float]], resolved: ResolvedProfile
    ) -> tuple[dict, dict, float] | None:
        """Choose one candidate by the profile's pick method. `candidates` are assumed already
        gated (every entry is a legal swap); ties break deterministically."""
        if not candidates:
            return None
        if resolved.pick == "max_score":
            return max(candidates, key=lambda x: (x[1]["raw"]["score"] or 0, -x[1]["raw"]["gbh"]))
        if resolved.pick == "min_size":
            return min(candidates, key=lambda x: (x[1]["raw"]["gbh"], -(x[1]["raw"]["score"] or 0)))
        # topsis: already sorted best-first by score_candidates
        return max(candidates, key=lambda x: x[2])
