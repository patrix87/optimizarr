"""TOPSIS-based release scorer + per-profile pickers.

Multi-objective release scoring. Three axes, each normalized to [0,1]:
  - score:      Profilarr customFormatScore, fixed scale [anti_ideal, ideal] (higher better)
  - resolution: pixel height toward the profile target (higher better, low weight — Profilarr
                already folds resolution into score, so this axis mostly avoids double-counting)
  - size:       GiB/h on a ONE-SIDED curve — n_size = 1.0 at/below the preset's `target`,
                ramping to 0 at the preset's `bloat`. Smaller than the target is never penalized,
                so nothing is ever inflated to "reach" a target.

Inclusion (before scoring): drop hard rejections, drop outside the preset's per-resolution size
band (gbh < floor = fakes/upscales, gbh > bloat = bloat), then gap-cut the score tail.

The size table `{floor, target, bloat}` is now PER-PRESET (config-driven): each resolved profile
carries its own table + weights + pick + a `min_closeness_gain` swap margin. The swap rule lives
in decision.py (a closeness-gain test); this module scores and picks among the survivors.
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

    def reference_for(self, res: int, reference: Reference) -> tuple[float, float, float]:
        """A preset's (floor, target, bloat) for a resolution; nearest-defined-at-or-below."""
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
        upscales / too-soft-for-the-resolution) or above `bloat` (bloated)."""
        keep = []
        for r in releases:
            floor, _target, bloat = self.reference_for(_release_resolution(r), reference)
            gbh = _release_gbh(r, runtime_h)
            if floor <= gbh <= bloat:
                keep.append(r)
        return keep

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
        kept = self.filter_by_score_gap(after_band)
        diag["after_score_gap"] = len(kept)
        diag["inclusion"] = f"size band + gap-cut (>{self.cfg.score_gap:.0%} drop)"
        return kept, diag

    # ----- normalization -----

    def normalize_score(self, s: float) -> float:
        cfg = self.cfg
        if s >= cfg.score_ideal:
            return 1.0
        if s <= cfg.score_anti_ideal:
            return 0.0
        return (s - cfg.score_anti_ideal) / (cfg.score_ideal - cfg.score_anti_ideal)

    def normalize_resolution(self, r: int, target: int | None = None) -> float:
        cfg = self.cfg
        ideal = target if target else cfg.resolution_ideal
        if r >= ideal:
            return 1.0
        if r <= cfg.resolution_anti_ideal:
            return 0.0
        return (r - cfg.resolution_anti_ideal) / (ideal - cfg.resolution_anti_ideal)

    def normalize_size(self, gbh: float, target: float, bloat: float) -> float:
        """One-sided cost curve: 1.0 at or below `target`, linear down to 0 at `bloat`. Smaller
        than the target is never penalized — that is what keeps the optimizer from ever inflating
        a file to reach a target."""
        if gbh <= target:
            return 1.0
        if gbh >= bloat or bloat <= target:
            return 0.0
        return (bloat - gbh) / (bloat - target)

    def _attrs(
        self,
        score: float,
        res: int,
        gbh: float,
        size_gb: float,
        resolved: ResolvedProfile,
        target_resolution: int | None,
    ) -> dict:
        floor, target, bloat = self.reference_for(res, resolved.reference)
        return {
            "n_score": self.normalize_score(score or 0),
            "n_resolution": self.normalize_resolution(res, target_resolution),
            "n_size": self.normalize_size(gbh, target, bloat),
            "raw": {
                "score": score,
                "resolution": res,
                "gbh": gbh,
                "size_gb": size_gb,
                "reference": (floor, target, bloat),
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
        """Normalized [0,1] attributes + raw values for one release."""
        size_bytes = release.get("size", 0)
        return self._attrs(
            release.get("customFormatScore", 0),
            _release_resolution(release),
            _release_gbh(release, runtime_h),
            size_bytes / GB,
            resolved,
            target_resolution,
        )

    def closeness(self, attrs: dict, weights: dict[str, float]) -> float:
        """TOPSIS closeness in [0,1]. 1 = ideal, 0 = anti-ideal."""
        w = {
            "n_score": weights["score"],
            "n_resolution": weights["resolution"],
            "n_size": weights["size"],
        }
        d_ideal = math.sqrt(sum(w[k] * (1.0 - attrs[k]) ** 2 for k in w))
        d_anti = math.sqrt(sum(w[k] * attrs[k] ** 2 for k in w))
        total = d_ideal + d_anti
        return 0.0 if total == 0 else d_anti / total

    def _current_resolution(self, movie_file: dict) -> int:
        mi = movie_file.get("mediaInfo") or {}
        res_str = mi.get("resolution") or ""
        if "x" in res_str:
            try:
                return int(res_str.split("x")[1])
            except (IndexError, ValueError):
                return 0
        return 0

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
        return self._attrs(score, res, gbh, size_gb, resolved, target_resolution)

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
