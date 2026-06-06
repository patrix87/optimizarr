import pytest

from optimizarr.config import load_config
from optimizarr.features.optimizer.config import PICK_ORDERS
from optimizarr.features.optimizer.worker import _PICK_ORDER_KEYS

_MANAGED_ENV_VARS = [
    "LOG_LEVEL",
    "RADARR_URL",
    "RADARR_API_KEY",
    "SONARR_URL",
    "SONARR_API_KEY",
]


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in _MANAGED_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def _write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return str(path)


def test_radarr_only_with_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://radarr:7878")
    monkeypatch.setenv("RADARR_API_KEY", "abc")
    path = _write(tmp_path, "")

    config = load_config(path)
    assert config.radarr is not None
    assert config.sonarr is None
    assert config.radarr.url == "http://radarr:7878"
    assert config.radarr.api_key == "abc"
    assert config.dry_run is False
    assert config.state_path == "/data/state.json"

    um = config.unmonitor
    assert um.enabled is True
    assert um.cron_schedule
    assert isinstance(um.run_on_start, bool)
    assert um.radarr.days > 0
    assert um.radarr.release_type
    assert isinstance(um.radarr.require_cutoff_met, bool)

    # per-app enabled is on by default; sonarr's app config is still parsed even with no conn
    assert config.optimizer.radarr.enabled is True
    assert config.optimizer.sonarr.enabled is True
    assert config.optimizer.enabled is True


def test_overrides_from_toml(monkeypatch, tmp_path):
    monkeypatch.setenv("SONARR_URL", "http://sonarr:8989/")
    monkeypatch.setenv("SONARR_API_KEY", "k")
    path = _write(
        tmp_path,
        """
        dry_run = true

        [unmonitor]
        cron_schedule = "*/30 * * * *"
        run_on_start = false

        [unmonitor.sonarr]
        days = 60
        release_type = "dateAdded"
        require_cutoff_met = false
        """,
    )

    config = load_config(path)
    assert config.radarr is None
    assert config.sonarr is not None
    assert config.sonarr.url == "http://sonarr:8989"  # trailing slash stripped
    assert config.dry_run is True
    assert config.unmonitor.cron_schedule == "*/30 * * * *"
    assert config.unmonitor.run_on_start is False
    assert config.unmonitor.sonarr.days == 60
    assert config.unmonitor.sonarr.release_type == "dateAdded"
    assert config.unmonitor.sonarr.require_cutoff_met is False
    assert config.optimizer.radarr.enabled is True  # per-app flag (worker still skips no-conn)


def test_rejects_when_neither_configured(tmp_path):
    with pytest.raises(ValueError, match="Neither"):
        load_config(_write(tmp_path, ""))


def test_missing_config_file_uses_defaults(monkeypatch):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    config = load_config("/nonexistent/config.toml")  # no user file -> built-in defaults
    assert config.optimizer.enabled is True
    assert config.optimizer.topsis is not None
    assert "Balanced" in config.optimizer.topsis.presets


def test_rejects_invalid_radarr_release_type(monkeypatch, tmp_path):
    # The unmonitor's release_type is still a single string — only the optimizer's takes a list.
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[unmonitor.radarr]\nrelease_type = "premiereDate"\n')
    with pytest.raises(ValueError, match="release_type"):
        load_config(path)


def test_rejects_invalid_pick_order(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer]\npick_order = "sideways"\n')
    with pytest.raises(ValueError, match="pick_order"):
        load_config(path)


@pytest.mark.parametrize("order", sorted(PICK_ORDERS))
def test_accepts_every_pick_order_through_full_load(monkeypatch, tmp_path, order):
    # Guards the regression that shipped to prod: the validator must accept every order the
    # worker can actually execute, exercised through the real load_config path (not order_pool
    # directly, which is what let the mismatch slip past the existing tests).
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, f'[optimizer]\npick_order = "{order}"\n')
    cfg = load_config(path)
    assert cfg.optimizer.pick_order == order


def test_pick_orders_match_worker_keys():
    # The validator set and the worker's sort-key table must stay in lockstep: "random" is the
    # only order handled outside the key table. If they drift, a config either rejects an order
    # the worker supports (prod error) or accepts one that KeyErrors at runtime.
    expected = {"random", *_PICK_ORDER_KEYS}
    assert expected == PICK_ORDERS


def test_rejects_process_interval_below_floor(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, "[optimizer]\nprocess_interval_seconds = 5\n")
    with pytest.raises(ValueError, match="process_interval_seconds must be >= 10"):
        load_config(path)


def test_rejects_preset_weights_not_summing_to_one(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        "[optimizer.topsis.presets.Balanced]\nscore = 0.5\nresolution = 0.3\nsize = 0.3\n",
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(path)


def test_rejects_unknown_default_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.topsis]\ndefault_preset = "Nope"\n')
    with pytest.raises(ValueError, match="default_preset"):
        load_config(path)


def test_rejects_unknown_preset_in_profile_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.topsis.profiles."X"]\npreset = "Nope"\n')
    with pytest.raises(ValueError, match="not a defined preset"):
        load_config(path)


def test_rejects_size_bounds_out_of_order(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        '[optimizer.topsis.size_bounds]\n"2160" = { floor = 40, ceiling = 10 }\n',
    )
    with pytest.raises(ValueError, match="floor < ceiling"):
        load_config(path)


def test_optimizer_app_age_gate_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    config = load_config(_write(tmp_path, ""))
    assert config.optimizer.radarr.min_age_days >= 0
    assert config.optimizer.radarr.release_type  # non-empty
    assert config.optimizer.sonarr.release_type  # non-empty
    # Per-app flags default on.
    assert config.optimizer.radarr.ignore_completed_in_queue is True
    assert config.optimizer.radarr.auto_import_downgrades is True
    assert config.optimizer.sonarr.ignore_completed_in_queue is True
    assert config.optimizer.sonarr.auto_import_downgrades is True


def test_optimizer_per_app_enabled_and_filter_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        """
        [optimizer.sonarr]
        enabled = false
        allow_size_increase = false
        ignore_completed_in_queue = false

        [optimizer.radarr]
        allow_quality_downgrade = false
        auto_import_downgrades = false
        """,
    )
    config = load_config(path)
    assert config.optimizer.sonarr.enabled is False
    assert config.optimizer.sonarr.allow_size_increase is False
    assert config.optimizer.sonarr.ignore_completed_in_queue is False
    assert config.optimizer.radarr.allow_quality_downgrade is False
    assert config.optimizer.radarr.auto_import_downgrades is False
    # untouched flags keep their defaults
    assert config.optimizer.radarr.enabled is True
    assert config.optimizer.radarr.allow_size_increase is True
    assert config.optimizer.radarr.ignore_completed_in_queue is True
    assert config.optimizer.sonarr.auto_import_downgrades is True


def test_optimizer_app_age_gate_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        """
        [optimizer.radarr]
        min_age_days = 14
        release_type = ["inCinemas"]
        """,
    )
    config = load_config(path)
    assert config.optimizer.radarr.min_age_days == 14
    assert config.optimizer.radarr.release_type == ["inCinemas"]


def test_optimizer_release_type_accepts_multi_date_list(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        """
        [optimizer.radarr]
        release_type = ["digitalRelease", "physicalRelease", "dateAdded"]
        """,
    )
    config = load_config(path)
    assert config.optimizer.radarr.release_type == [
        "digitalRelease",
        "physicalRelease",
        "dateAdded",
    ]


def test_rejects_release_type_as_string(monkeypatch, tmp_path):
    # Strict: a bare string is no longer accepted — must be a list.
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.radarr]\nrelease_type = "digitalRelease"\n')
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_config(path)


def test_rejects_empty_release_type_list(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, "[optimizer.radarr]\nrelease_type = []\n")
    with pytest.raises(ValueError, match="non-empty list"):
        load_config(path)


def test_rejects_invalid_optimizer_release_type(monkeypatch, tmp_path):
    monkeypatch.setenv("SONARR_URL", "http://x")
    monkeypatch.setenv("SONARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.sonarr]\nrelease_type = ["digitalRelease"]\n')
    with pytest.raises(ValueError, match="optimizer.sonarr.release_type"):
        load_config(path)


def test_parses_topsis_presets_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        """
        [optimizer]
        enabled = true
        queue_max = 2

        [optimizer.topsis]
        score_window = 80000

        [optimizer.topsis.profiles."2160p Remux"]
        preset = "Remux"

        [optimizer.topsis.profiles."Custom 1080p"]
        weights = { score = 0.6, size = 0.4 }
        """,
    )

    config = load_config(path)
    t = config.optimizer.topsis
    assert config.optimizer.enabled is True
    assert config.optimizer.queue_max == 2
    assert t.score_window == 80000  # user override
    assert t.min_candidates > 0
    # shipped presets survive the deep-merge
    assert {"Remux", "Quality", "Balanced", "Efficient", "Compact"} <= set(t.presets)
    assert 0 < t.presets["Compact"].weights["size"] < 1
    # shared per-resolution legitimacy bounds: floor < ceiling, both positive
    floor_2160, ceil_2160 = t.size_bounds[2160]
    assert 0 < floor_2160 < ceil_2160
    floor_1080, ceil_1080 = t.size_bounds[1080]
    assert 0 < floor_1080 < ceil_1080
    # ACT gate thresholds: axis and closeness gates
    assert t.min_score_delta >= 0
    assert t.min_size_delta_gb >= 0
    assert 0 <= t.min_closeness_gain < 1
    # overrides parse as preset-ref or explicit weights
    assert t.profiles["2160p Remux"].preset == "Remux"
    custom_weights = t.profiles["Custom 1080p"].weights
    assert custom_weights is not None and custom_weights["size"] == 0.4
    assert t.default_preset in t.presets


def test_parses_schedule(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        """
        [optimizer.schedule]
        monday    = { start = "22:30", end = "07:00" }
        saturday  = { start = "23:00", end = "09:00" }
        """,
    )
    cfg = load_config(path)
    sch = cfg.optimizer.schedule
    # 0=Monday, 5=Saturday
    assert 0 in sch and 5 in sch
    from datetime import time

    assert sch[0].start == time(22, 30) and sch[0].end == time(7, 0)
    assert sch[5].start == time(23, 0) and sch[5].end == time(9, 0)


def test_schedule_defaults_all_days(monkeypatch, tmp_path):
    # Built-in defaults define a 23:00-08:00 window for all 7 days.
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    cfg = load_config(_write(tmp_path, ""))
    assert len(cfg.optimizer.schedule) == 7
    from datetime import time

    for window in cfg.optimizer.schedule.values():
        assert window.start == time(23, 0)
        assert window.end == time(8, 0)


def test_rejects_invalid_schedule_time(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.schedule]\nmonday = { start = "25:00", end = "08:00" }\n')
    with pytest.raises((ValueError, Exception)):
        load_config(path)


def test_rejects_unknown_schedule_day(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.schedule]\nfunday = { start = "23:00", end = "08:00" }\n')
    with pytest.raises(ValueError, match="unknown day"):
        load_config(path)


def test_optimizer_import_max_default_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://radarr:7878")
    monkeypatch.setenv("RADARR_API_KEY", "abc")

    assert load_config(_write(tmp_path, "")).optimizer.import_max >= 0
    over = load_config(_write(tmp_path, "[optimizer]\nimport_max = 4\n"))
    assert over.optimizer.import_max == 4
