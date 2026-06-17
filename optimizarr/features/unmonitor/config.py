"""Unmonitor feature configuration: schema + parsing of the [unmonitor] TOML section.

The shared loader (optimizarr.config) delegates to parse_unmonitor() here, so the unmonitor
feature owns its own config surface.
"""

from dataclasses import dataclass

from optimizarr.config import RADARR_RELEASE_TYPES, SONARR_RELEASE_TYPES


# These dataclasses carry NO default values: every default lives in defaults.toml (the single source
# of truth) and is read by parse_unmonitor() below with no Python-side fallback. To build a config
# from the bundled defaults (e.g. in tests), parse them via default_unmonitor(), not field defaults.
@dataclass
class UnmonitorAppConfig:
    days: int
    release_type: str
    require_cutoff_met: bool


@dataclass
class UnmonitorConfig:
    enabled: bool
    cron_schedule: str
    run_on_start: bool
    radarr: UnmonitorAppConfig
    sonarr: UnmonitorAppConfig


def _parse_unmonitor_app(raw: dict, allowed: set[str], where: str) -> UnmonitorAppConfig:
    release_type = str(raw["release_type"]).strip()
    if release_type not in allowed:
        raise ValueError(f"{where}.release_type={release_type!r} not in {sorted(allowed)}")
    return UnmonitorAppConfig(
        days=int(raw["days"]),
        release_type=release_type,
        require_cutoff_met=bool(raw["require_cutoff_met"]),
    )


def parse_unmonitor(raw: dict) -> UnmonitorConfig:
    return UnmonitorConfig(
        enabled=bool(raw["enabled"]),
        cron_schedule=str(raw["cron_schedule"]).strip(),
        run_on_start=bool(raw["run_on_start"]),
        radarr=_parse_unmonitor_app(raw["radarr"], RADARR_RELEASE_TYPES, "unmonitor.radarr"),
        sonarr=_parse_unmonitor_app(raw["sonarr"], SONARR_RELEASE_TYPES, "unmonitor.sonarr"),
    )


def default_unmonitor() -> UnmonitorConfig:
    """Parse the bundled defaults' [unmonitor] section. Defaults live only in defaults.toml; tests
    build a baseline from this (then dataclasses.replace to vary one field) instead of hardcoding
    values, so changing a default in defaults.toml never requires a test edit."""
    from optimizarr.config import _load_defaults

    return parse_unmonitor(_load_defaults()["unmonitor"])
