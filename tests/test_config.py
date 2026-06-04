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
    assert um.cron_schedule == "0 4 * * *"
    assert um.run_on_start is True
    assert um.radarr.days == 14
    assert um.radarr.release_type == "digitalRelease"
    assert um.radarr.require_cutoff_met is True

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


def test_rejects_min_closeness_gain_out_of_range(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, "[optimizer.topsis.presets.Efficient]\nmin_closeness_gain = 1.5\n")
    with pytest.raises(ValueError, match="min_closeness_gain"):
        load_config(path)


def test_rejects_unknown_pick_method(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(tmp_path, '[optimizer.topsis.presets.Efficient]\npick = "lottery"\n')
    with pytest.raises(ValueError, match="pick"):
        load_config(path)


def test_rejects_reference_out_of_order(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    path = _write(
        tmp_path,
        "[optimizer.topsis.presets.Balanced.reference]\n"
        '"2160" = { floor = 3, lo = 5, target = 20, hi = 10, ceiling = 18 }\n',
    )
    with pytest.raises(ValueError, match="floor < lo <= target <= hi < ceiling"):
        load_config(path)


def test_optimizer_app_age_gate_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://x")
    monkeypatch.setenv("RADARR_API_KEY", "k")
    config = load_config(_write(tmp_path, ""))
    assert config.optimizer.radarr.min_age_days == 14
    # Dual-gate by default: release date AND dateAdded both must pass.
    assert config.optimizer.radarr.release_type == ["digitalRelease", "dateAdded"]
    assert config.optimizer.sonarr.release_type == ["airDateUtc", "dateAdded"]
    # New per-app flags default on.
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
        score_gap = 0.30

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
    assert t.score_gap == 0.30
    # shipped presets survive the deep-merge
    assert {"Remux", "Quality", "Balanced", "Efficient", "Compact"} <= set(t.presets)
    assert t.presets["Compact"].weights["size"] == 0.78
    assert t.presets["Compact"].pick == "topsis"
    # per-preset 5-point size tables (floor, lo, target, hi, ceiling)
    assert t.presets["Balanced"].reference[2160] == (5.0, 8.1, 9.3, 11.0, 20.0)
    assert t.presets["Efficient"].reference[2160] == (4.0, 7.1, 7.8, 8.8, 15.0)
    assert t.presets["Remux"].reference[2160] == (15.0, 28.0, 38.0, 60.0, 90.0)
    # swap margin default
    assert t.default_min_closeness_gain == 0.02
    assert t.presets["Balanced"].min_closeness_gain == 0.02
    # overrides parse as preset-ref or explicit weights
    assert t.profiles["2160p Remux"].preset == "Remux"
    custom_weights = t.profiles["Custom 1080p"].weights
    assert custom_weights is not None and custom_weights["size"] == 0.4
    assert t.default_preset == "Balanced"


def test_optimizer_import_max_default_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RADARR_URL", "http://radarr:7878")
    monkeypatch.setenv("RADARR_API_KEY", "abc")

    assert load_config(_write(tmp_path, "")).optimizer.import_max == 2  # default
    over = load_config(_write(tmp_path, "[optimizer]\nimport_max = 4\n"))
    assert over.optimizer.import_max == 4
