"""Optimizer feature configuration: schema + parsing of the [optimizer] TOML section.

Tuning values (the shared size reference, presets, transition rules, score anchors) come from
the merged config (defaults.toml + the user's config.toml) — there are no magic defaults baked
into this module. The shared loader (optimizarr.config) delegates to parse_optimizer() here.

Size model: one objective `[reference]` per resolution gives `{floor, target, ceiling}` GiB/h,
shared by every profile. Each preset only carries a *relative* `size_aim` (fraction of target,
where its one-sided size curve plateaus) plus its TOPSIS weights, a `pick` method, and a
`transitions` rule set (which moves from the current file are legal). See
docs/condition-matrix-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optimizarr.config import RADARR_RELEASE_TYPES, SONARR_RELEASE_TYPES

PICK_ORDERS = {"random", "ordered"}
PICK_METHODS = {"topsis", "max_score", "min_size"}


@dataclass
class Transitions:
    """Per-profile rule set deciding which moves from the current file are legal. Magnitudes are
    on normalized/relative scales (see classify() in transitions.py). The universal forbidden
    moves are hard-coded; these flags express the per-profile differences."""

    score_slack: float  # |Δn_score| within this = "same" (noise band)
    score_much: float  # |Δn_score| beyond this = "much" higher/lower (MUST exceed score_slack)
    size_slack: float  # |relative size delta| within this = "same"
    size_much: float  # relative size delta beyond this = "much" smaller/bigger
    allow_bigger_for_score: bool  # may a bigger file be accepted for a higher score?
    bigger_needs_much_score: bool  # if so, must the gain clear score_much (vs any higher)?
    accept_score_drop: bool  # may a slightly-lower-score release be accepted at all?
    slight_drop_needs_much_smaller: bool  # if so, only when it is *much* smaller?
    accept_much_lower_score: bool  # may a much-lower-score release be accepted (Compact)?
    viability_score: int  # floor below which score drops are never accepted


@dataclass
class Preset:
    """A named bundle: TOPSIS weights + a relative size aim + a pick method + transition rules."""

    weights: dict[str, float]  # keys: score, resolution, size (sum 1.0)
    size_aim: float  # fraction of reference target where n_size stops being 1.0 (one-sided)
    pick: str  # "topsis" | "max_score" | "min_size"
    transitions: Transitions


@dataclass
class ProfileOverride:
    """Exact-name override: reference a preset, or override its weights / size_aim / pick."""

    preset: str | None = None
    weights: dict[str, float] | None = None
    size_aim: float | None = None
    pick: str | None = None


@dataclass
class ResolvedProfile:
    """Everything the picker needs for one profile, after preset + override resolution."""

    weights: dict[str, float]
    size_aim: float
    pick: str
    transitions: Transitions


@dataclass
class TopsisConfig:
    score_ideal: int
    score_anti_ideal: int
    resolution_ideal: int
    resolution_anti_ideal: int
    score_gap: float
    default_preset: str
    # resolution -> (floor, target, ceiling) GiB/h, shared by all profiles.
    reference: dict[int, tuple[float, float, float]]
    presets: dict[str, Preset]
    profiles: dict[str, ProfileOverride] = field(default_factory=dict)


@dataclass
class OptimizerAppConfig:
    enabled: bool = True
    min_age_days: int = 0
    # List of date fields the age gate checks. ALL listed dates must be at least
    # min_age_days old. Two-gate default ([release, dateAdded]) avoids touching freshly
    # released items (still being chased by Radarr/Sonarr) and freshly imported files.
    release_type: list[str] = field(default_factory=list)
    # If False, releases bigger than the current file are filtered out before scoring —
    # blocks resolution upgrades too (1080p -> 2160p is always a size increase).
    allow_size_increase: bool = True
    # If False, releases with a lower score than the current file are filtered out before
    # scoring. NOTE: turning this off neutralizes size-leaning presets (Compact/Efficient),
    # which are designed to swap a slightly-lower-score release for a meaningfully smaller one.
    allow_quality_downgrade: bool = True
    # If True, queue items waiting for manual import don't count toward queue_max — only
    # actively downloading/queued items do. Keeps the optimizer flowing when downgrades or
    # other rejected items pile up in the import-pending state.
    ignore_completed_in_queue: bool = True
    # If True, on each tick the worker scans the queue for completed items rejected solely
    # for score regression ("Not an upgrade") and force-imports them through manualimport.
    # Other rejection categories (executable/sample/mismatch) are left untouched.
    auto_import_downgrades: bool = True


@dataclass
class OptimizerConfig:
    enabled: bool = False
    queue_max: int = 5
    # When more than this many completed downloads are waiting to import (trackedDownloadState
    # importPending) or actively importing (importing), pause grabbing new releases so the
    # import backlog drains first. importBlocked is excluded: it needs manual action and would
    # never clear, freezing grabs forever.
    import_max: int = 2
    pick_order: str = "random"
    process_interval_seconds: int = 15
    list_refresh_minutes: int = 15
    reevaluate_after_days: int = 30
    radarr: OptimizerAppConfig = field(default_factory=OptimizerAppConfig)
    sonarr: OptimizerAppConfig = field(default_factory=OptimizerAppConfig)
    topsis: TopsisConfig = field(default_factory=lambda: default_topsis())


# ----- parsing helpers -----


def _weights(raw: dict, where: str) -> dict[str, float]:
    w = {k: float(raw[k]) for k in ("score", "resolution", "size") if k in raw}
    missing = {"score", "resolution", "size"} - w.keys()
    if missing:
        raise ValueError(f"{where}: missing weight keys {sorted(missing)}")
    total = sum(w.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"{where}: weights must sum to 1.0, got {total:.3f}")
    return w


def _reference(raw: dict, where: str) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for res, entry in raw.items():
        try:
            res_int = int(res)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}: key {res!r} is not an integer resolution") from e
        if not isinstance(entry, dict) or not {"floor", "target", "ceiling"} <= entry.keys():
            raise ValueError(f"{where}.{res}: expected {{floor, target, ceiling}}, got {entry!r}")
        floor = float(entry["floor"])
        target = float(entry["target"])
        ceiling = float(entry["ceiling"])
        if not (floor < target <= ceiling):
            raise ValueError(
                f"{where}.{res}: must satisfy floor < target <= ceiling, "
                f"got floor={floor} target={target} ceiling={ceiling}"
            )
        out[res_int] = (floor, target, ceiling)
    if not out:
        raise ValueError(f"{where} is empty (defaults.toml should define it)")
    return out


def _transitions(raw: dict, where: str) -> Transitions:
    def num(key: str, default: float) -> float:
        return float(raw.get(key, default))

    def flag(key: str, default: bool) -> bool:
        return bool(raw.get(key, default))

    score_slack = num("score_slack", 0.02)
    score_much = num("score_much", 0.10)
    size_slack = num("size_slack", 0.03)
    size_much = num("size_much", 0.30)
    if score_much <= score_slack:
        raise ValueError(
            f"{where}: score_much ({score_much}) must exceed score_slack ({score_slack}) — "
            f"this inequality is what guarantees no two-file oscillation"
        )
    if size_much <= size_slack:
        raise ValueError(f"{where}: size_much ({size_much}) must exceed size_slack ({size_slack})")
    return Transitions(
        score_slack=score_slack,
        score_much=score_much,
        size_slack=size_slack,
        size_much=size_much,
        allow_bigger_for_score=flag("allow_bigger_for_score", True),
        bigger_needs_much_score=flag("bigger_needs_much_score", True),
        accept_score_drop=flag("accept_score_drop", True),
        slight_drop_needs_much_smaller=flag("slight_drop_needs_much_smaller", False),
        accept_much_lower_score=flag("accept_much_lower_score", False),
        viability_score=int(raw.get("viability_score", 0)),
    )


def _parse_pick(raw: dict, where: str) -> str:
    pick = str(raw.get("pick", "topsis"))
    if pick not in PICK_METHODS:
        raise ValueError(f"{where}.pick={pick!r} not in {sorted(PICK_METHODS)}")
    return pick


def _parse_size_aim(value: float | int | str, where: str) -> float:
    aim = float(value)
    if not (0.0 < aim <= 1.0):
        raise ValueError(f"{where}.size_aim must satisfy 0 < size_aim <= 1.0, got {aim}")
    return aim


def _parse_preset(raw: dict, where: str) -> Preset:
    return Preset(
        weights=_weights(raw, where),
        size_aim=_parse_size_aim(raw.get("size_aim", 1.0), where),
        pick=_parse_pick(raw, where),
        transitions=_transitions(raw.get("transitions", {}), f"{where}.transitions"),
    )


def _parse_profile_override(raw: dict, where: str) -> ProfileOverride:
    weights = _weights(raw["weights"], f"{where}.weights") if "weights" in raw else None
    size_aim = _parse_size_aim(raw["size_aim"], where) if "size_aim" in raw else None
    pick = _parse_pick(raw, where) if "pick" in raw else None
    return ProfileOverride(preset=raw.get("preset"), weights=weights, size_aim=size_aim, pick=pick)


def _parse_topsis(raw: dict) -> TopsisConfig:
    presets = {
        name: _parse_preset(p, f"presets.{name}") for name, p in raw.get("presets", {}).items()
    }
    if not presets:
        raise ValueError("optimizer.topsis.presets is empty (defaults.toml should define them)")
    default_preset = str(raw.get("default_preset", "Balanced"))
    if default_preset not in presets:
        raise ValueError(f"default_preset {default_preset!r} is not a defined preset")
    profiles = {
        name: _parse_profile_override(o, f"profiles.{name}")
        for name, o in raw.get("profiles", {}).items()
    }
    for name, ov in profiles.items():
        if ov.preset is not None and ov.preset not in presets:
            raise ValueError(f"profiles.{name}.preset {ov.preset!r} is not a defined preset")
    return TopsisConfig(
        score_ideal=int(raw["score_ideal"]),
        score_anti_ideal=int(raw["score_anti_ideal"]),
        resolution_ideal=int(raw["resolution_ideal"]),
        resolution_anti_ideal=int(raw["resolution_anti_ideal"]),
        score_gap=float(raw["score_gap"]),
        default_preset=default_preset,
        reference=_reference(raw.get("reference", {}), "optimizer.topsis.reference"),
        presets=presets,
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


def _parse_optimizer_app(
    raw: dict, default_release_type: list[str], allowed: set[str], where: str
) -> OptimizerAppConfig:
    release_type = _parse_release_types(
        raw.get("release_type", default_release_type), allowed, where
    )
    return OptimizerAppConfig(
        enabled=bool(raw.get("enabled", True)),
        min_age_days=int(raw.get("min_age_days", 0)),
        release_type=release_type,
        allow_size_increase=bool(raw.get("allow_size_increase", True)),
        allow_quality_downgrade=bool(raw.get("allow_quality_downgrade", True)),
        ignore_completed_in_queue=bool(raw.get("ignore_completed_in_queue", True)),
        auto_import_downgrades=bool(raw.get("auto_import_downgrades", True)),
    )


def parse_optimizer(raw: dict) -> OptimizerConfig:
    pick_order = str(raw.get("pick_order", "random")).strip()
    if pick_order not in PICK_ORDERS:
        raise ValueError(f"optimizer.pick_order={pick_order!r} not in {sorted(PICK_ORDERS)}")

    process_interval_seconds = int(raw.get("process_interval_seconds", 15))
    if process_interval_seconds < 10:
        raise ValueError(
            f"optimizer.process_interval_seconds must be >= 10, got {process_interval_seconds}"
        )

    return OptimizerConfig(
        enabled=bool(raw.get("enabled", False)),
        queue_max=int(raw.get("queue_max", 5)),
        import_max=int(raw.get("import_max", 2)),
        pick_order=pick_order,
        process_interval_seconds=process_interval_seconds,
        list_refresh_minutes=int(raw.get("list_refresh_minutes", 15)),
        reevaluate_after_days=int(raw.get("reevaluate_after_days", 30)),
        radarr=_parse_optimizer_app(
            raw.get("radarr", {}),
            ["digitalRelease", "dateAdded"],
            RADARR_RELEASE_TYPES,
            "optimizer.radarr",
        ),
        sonarr=_parse_optimizer_app(
            raw.get("sonarr", {}),
            ["airDateUtc", "dateAdded"],
            SONARR_RELEASE_TYPES,
            "optimizer.sonarr",
        ),
        topsis=_parse_topsis(raw.get("topsis", {})),
    )


def default_topsis() -> TopsisConfig:
    """Parse the bundled defaults' TOPSIS section. For tests and tools that need a config
    without going through the full env-dependent load_config()."""
    from optimizarr.config import _load_defaults

    return _parse_topsis(_load_defaults()["optimizer"]["topsis"])
