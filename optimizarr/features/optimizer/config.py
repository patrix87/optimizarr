"""Optimizer feature configuration: schema + parsing of the [optimizer] TOML section.

Tuning values come from the merged config (defaults.toml + the user's config.toml); there are no
magic defaults baked into this module. The shared loader (optimizarr.config) delegates here.

Size model (relative): there are no per-preset size tables or shapes. Inclusion filters drop the
obviously-bad (score < 0, score below `current - score_window`, wrong resolution, outside the
shared per-resolution `size_bounds` legitimacy band, lone-small outliers). The survivors are then
scored on TWO axes, each min-max normalized OVER THE SURVIVORS (+ the current file): score (higher
better) and GiB/h (smaller better). A profile's `weights` combine them into a TOPSIS closeness.
The only per-profile knob is the weights. Oscillation is prevented by one-and-done state (a movie
is satisfied once its current file is optimal and never re-evaluated), not by a closeness margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optimizarr.config import RADARR_RELEASE_TYPES, SONARR_RELEASE_TYPES

PICK_ORDERS = {
    "random",  # shuffle each pass
    "alphabetical_asc",  # A->Z by title
    "alphabetical_desc",  # Z->A by title
    "size_asc",  # smallest file first
    "size_desc",  # biggest file first
    "date_added_asc",  # oldest import first
    "date_added_desc",  # newest import first
    "release_date_asc",  # oldest release first
    "release_date_desc",  # newest release first
}
# resolution -> (floor, ceiling) GiB/h. Shared legitimacy bounds: below floor = fake/upscale,
# above ceiling = bloat. A candidate outside its resolution's band is dropped before scoring.
SizeBounds = dict[int, tuple[float, float]]


@dataclass
class Preset:
    """A named bundle: TOPSIS weights + a swap margin. The pick is always max closeness."""

    weights: dict[str, float]  # keys: score, size (sum 1.0)
    min_closeness_gain: float  # ACT only if the pick beats the current file's closeness by this


@dataclass
class ProfileOverride:
    """Exact-name override: reference a preset, or override its weights / margin."""

    preset: str | None = None
    weights: dict[str, float] | None = None
    min_closeness_gain: float | None = None


@dataclass
class ResolvedProfile:
    """Everything the scorer + decision need for one profile, after preset + override."""

    weights: dict[str, float]
    min_closeness_gain: float


@dataclass
class TopsisConfig:
    default_preset: str
    default_min_closeness_gain: float
    presets: dict[str, Preset]
    # Three-tier score window (see topsis._score_floor_tier):
    #   Tier 1: keep score >= max_available - score_window (anchor at top of what's on offer).
    #   Tier 2: if fewer than min_candidates survive, expand down to current_file_score.
    #   Tier 3: if still fewer, expand to max(0, current_file_score - score_window) (full budget).
    score_window: int = 100000
    # Minimum pool size used at two gates: (1) the score-window tier check (expand to the next
    # tier if too few survive the current one); (2) the TOPSIS gate (HOLD without satisfying if
    # fewer than this many remain after all filters -- relative min-max is untrustworthy too thin).
    min_candidates: int = 6
    # Outlier prefilter: drop a candidate whose GiB/h is below outlier_frac x the median GiB/h of
    # the surviving cluster. 0 disables it.
    outlier_frac: float = 0.5
    # Shared per-resolution {floor, ceiling} legitimacy band (GiB/h).
    size_bounds: SizeBounds = field(default_factory=dict)
    profiles: dict[str, ProfileOverride] = field(default_factory=dict)


@dataclass
class OptimizerAppConfig:
    enabled: bool = True
    min_age_days: int = 0
    release_type: list[str] = field(default_factory=list)
    # If False, releases bigger than the current file are filtered out before scoring —
    # blocks resolution upgrades too (1080p -> 2160p is always a size increase).
    allow_size_increase: bool = True
    # If False, releases with a lower score than the current file are filtered out before scoring.
    allow_quality_downgrade: bool = True
    # If True, queue items waiting for manual import don't count toward queue_max.
    ignore_completed_in_queue: bool = True
    # If True, the worker force-imports completed items rejected solely for score regression.
    auto_import_downgrades: bool = True


@dataclass
class OptimizerConfig:
    enabled: bool = False
    queue_max: int = 5
    import_max: int = 2
    pick_order: str = "random"
    process_interval_seconds: int = 15
    list_refresh_minutes: int = 15
    radarr: OptimizerAppConfig = field(default_factory=OptimizerAppConfig)
    sonarr: OptimizerAppConfig = field(default_factory=OptimizerAppConfig)
    topsis: TopsisConfig = field(default_factory=lambda: default_topsis())


# ----- parsing helpers -----


def _weights(raw: dict, where: str) -> dict[str, float]:
    w = {k: float(raw[k]) for k in ("score", "size") if k in raw}
    missing = {"score", "size"} - w.keys()
    if missing:
        raise ValueError(f"{where}: missing weight keys {sorted(missing)}")
    total = sum(w.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"{where}: weights must sum to 1.0, got {total:.3f}")
    return w


def _size_bounds(raw: dict, where: str) -> SizeBounds:
    out: SizeBounds = {}
    for res, entry in raw.items():
        try:
            res_int = int(res)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}: key {res!r} is not an integer resolution") from e
        if not isinstance(entry, dict) or not {"floor", "ceiling"} <= entry.keys():
            raise ValueError(f"{where}.{res}: expected {{floor, ceiling}}, got {entry!r}")
        floor = float(entry["floor"])
        ceiling = float(entry["ceiling"])
        if not (0.0 <= floor < ceiling):
            raise ValueError(
                f"{where}.{res}: must satisfy 0 <= floor < ceiling, got floor={floor} "
                f"ceiling={ceiling}"
            )
        out[res_int] = (floor, ceiling)
    if not out:
        raise ValueError(f"{where} is empty (define at least one resolution's {{floor, ceiling}})")
    return out


def _parse_min_gain(value: float | int, where: str) -> float:
    gain = float(value)
    if not (0.0 <= gain < 1.0):
        raise ValueError(f"{where}.min_closeness_gain must satisfy 0 <= gain < 1.0, got {gain}")
    return gain


def _parse_preset(raw: dict, where: str, default_min_gain: float) -> Preset:
    min_gain = (
        _parse_min_gain(raw["min_closeness_gain"], where)
        if "min_closeness_gain" in raw
        else default_min_gain
    )
    return Preset(weights=_weights(raw, where), min_closeness_gain=min_gain)


def _parse_profile_override(raw: dict, where: str) -> ProfileOverride:
    weights = _weights(raw["weights"], f"{where}.weights") if "weights" in raw else None
    min_gain = (
        _parse_min_gain(raw["min_closeness_gain"], where) if "min_closeness_gain" in raw else None
    )
    return ProfileOverride(preset=raw.get("preset"), weights=weights, min_closeness_gain=min_gain)


def _parse_topsis(raw: dict) -> TopsisConfig:
    default_min_gain = _parse_min_gain(raw["min_closeness_gain"], "optimizer.topsis")
    presets = {
        name: _parse_preset(p, f"presets.{name}", default_min_gain)
        for name, p in raw["presets"].items()
    }
    if not presets:
        raise ValueError("optimizer.topsis.presets is empty (defaults.toml should define them)")
    default_preset = str(raw["default_preset"])
    if default_preset not in presets:
        raise ValueError(f"default_preset {default_preset!r} is not a defined preset")
    profiles = {
        name: _parse_profile_override(o, f"profiles.{name}")
        for name, o in raw.get("profiles", {}).items()
    }
    for name, ov in profiles.items():
        if ov.preset is not None and ov.preset not in presets:
            raise ValueError(f"profiles.{name}.preset {ov.preset!r} is not a defined preset")
    score_window = int(raw["score_window"])
    if score_window < 0:
        raise ValueError(f"optimizer.topsis.score_window must be >= 0, got {score_window}")
    min_candidates = int(raw["min_candidates"])
    if min_candidates < 1:
        raise ValueError(f"optimizer.topsis.min_candidates must be >= 1, got {min_candidates}")
    outlier_frac = float(raw["outlier_frac"])
    if not (0.0 <= outlier_frac < 1.0):
        raise ValueError(f"optimizer.topsis.outlier_frac must be in [0, 1), got {outlier_frac}")
    if "size_bounds" not in raw:
        raise ValueError(
            "optimizer.topsis.size_bounds is required (per-resolution {floor, ceiling})"
        )
    return TopsisConfig(
        default_preset=default_preset,
        default_min_closeness_gain=default_min_gain,
        presets=presets,
        score_window=score_window,
        min_candidates=min_candidates,
        outlier_frac=outlier_frac,
        size_bounds=_size_bounds(raw["size_bounds"], "optimizer.topsis.size_bounds"),
        profiles=profiles,
    )


def _parse_release_types(raw: object, allowed: set[str], where: str) -> list[str]:
    if isinstance(raw, str):
        raise ValueError(
            f"{where}.release_type must be a list of strings, got string {raw!r}. "
            f'Use release_type = ["{raw}"]'
        )
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}.release_type must be a non-empty list, got {raw!r}")
    out: list[str] = []
    for entry in raw:
        entry_s = str(entry).strip()
        if entry_s not in allowed:
            raise ValueError(f"{where}.release_type entry {entry_s!r} not in {sorted(allowed)}")
        out.append(entry_s)
    return out


def _parse_optimizer_app(raw: dict, allowed: set[str], where: str) -> OptimizerAppConfig:
    return OptimizerAppConfig(
        enabled=bool(raw["enabled"]),
        min_age_days=int(raw["min_age_days"]),
        release_type=_parse_release_types(raw["release_type"], allowed, where),
        allow_size_increase=bool(raw["allow_size_increase"]),
        allow_quality_downgrade=bool(raw["allow_quality_downgrade"]),
        ignore_completed_in_queue=bool(raw["ignore_completed_in_queue"]),
        auto_import_downgrades=bool(raw["auto_import_downgrades"]),
    )


def parse_optimizer(raw: dict) -> OptimizerConfig:
    pick_order = str(raw["pick_order"]).strip()
    if pick_order not in PICK_ORDERS:
        raise ValueError(f"optimizer.pick_order={pick_order!r} not in {sorted(PICK_ORDERS)}")

    process_interval_seconds = int(raw["process_interval_seconds"])
    if process_interval_seconds < 10:
        raise ValueError(
            f"optimizer.process_interval_seconds must be >= 10, got {process_interval_seconds}"
        )

    return OptimizerConfig(
        enabled=bool(raw["enabled"]),
        queue_max=int(raw["queue_max"]),
        import_max=int(raw["import_max"]),
        pick_order=pick_order,
        process_interval_seconds=process_interval_seconds,
        list_refresh_minutes=int(raw["list_refresh_minutes"]),
        radarr=_parse_optimizer_app(raw["radarr"], RADARR_RELEASE_TYPES, "optimizer.radarr"),
        sonarr=_parse_optimizer_app(raw["sonarr"], SONARR_RELEASE_TYPES, "optimizer.sonarr"),
        topsis=_parse_topsis(raw["topsis"]),
    )


def default_topsis() -> TopsisConfig:
    """Parse the bundled defaults' TOPSIS section. For tests and tools that need a config
    without going through the full env-dependent load_config()."""
    from optimizarr.config import _load_defaults

    return _parse_topsis(_load_defaults()["optimizer"]["topsis"])
