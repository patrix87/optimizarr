from datetime import UTC, datetime, timedelta

from optimizarr.features.optimizer.state import INSUFFICIENT, SATISFIED, StateManager


def _mgr(tmp_path):
    return StateManager(str(tmp_path / "state.json"))


def test_missing_file_starts_empty(tmp_path):
    m = _mgr(tmp_path)
    assert m.get("radarr", 1) is None


def test_mark_satisfied_persists_profile(tmp_path):
    path = tmp_path / "state.json"
    m = StateManager(str(path))
    m.mark_satisfied("radarr", 42, "2160p Quality")
    assert path.exists()

    reloaded = StateManager(str(path))
    entry = reloaded.get("radarr", 42)
    assert entry is not None
    assert entry.status == SATISFIED
    assert entry.profile == "2160p Quality"


def test_is_active_one_and_done(tmp_path):
    m = _mgr(tmp_path)

    # Unprocessed -> active.
    assert m.is_active("radarr", 1, "2160p Quality", has_file=True)

    # Satisfied for its profile, with a file -> not active (one-and-done, no time re-eval).
    m.mark_satisfied("radarr", 2, "2160p Quality")
    assert not m.is_active("radarr", 2, "2160p Quality", has_file=True)

    # Profile changed -> active again (the optimal pick depends on the profile).
    assert m.is_active("radarr", 2, "2160p Efficient", has_file=True)

    # File removed -> active again (needs a fresh grab).
    assert m.is_active("radarr", 2, "2160p Quality", has_file=False)


def test_record_insufficient_counts_then_cools_down(tmp_path):
    m = _mgr(tmp_path)

    # Tries below max keep the item active and counting; retry_after stays unset.
    e1 = m.record_insufficient("radarr", 1, "2160p Quality", max_tries=3, cooldown_days=30)
    assert e1.status == INSUFFICIENT and e1.tries == 1 and e1.retry_after is None
    assert m.is_active("radarr", 1, "2160p Quality", has_file=True)

    e2 = m.record_insufficient("radarr", 1, "2160p Quality", max_tries=3, cooldown_days=30)
    assert e2.tries == 2 and e2.retry_after is None
    assert m.is_active("radarr", 1, "2160p Quality", has_file=True)

    # Reaching max_tries sets a cooldown and parks the item.
    e3 = m.record_insufficient("radarr", 1, "2160p Quality", max_tries=3, cooldown_days=30)
    assert e3.tries == 3 and e3.retry_after is not None
    assert not m.is_active("radarr", 1, "2160p Quality", has_file=True)


def test_insufficient_cooldown_expires_and_resets(tmp_path):
    m = _mgr(tmp_path)
    m.record_insufficient("radarr", 1, "2160p Quality", max_tries=1, cooldown_days=30)
    entry = m.get("radarr", 1)
    assert entry.retry_after is not None

    # Before the cooldown ends: inactive.
    before = datetime.now(UTC)
    assert not m.is_active("radarr", 1, "2160p Quality", has_file=True, now=before)

    # After it ends: active again, and the next attempt restarts the counter at 1.
    after = datetime.fromisoformat(entry.retry_after) + timedelta(seconds=1)
    assert m.is_active("radarr", 1, "2160p Quality", has_file=True, now=after)
    nxt = m.record_insufficient("radarr", 1, "2160p Quality", max_tries=3, cooldown_days=30)
    assert nxt.tries == 1 and nxt.retry_after is None


def test_insufficient_profile_change_reactivates_and_resets(tmp_path):
    m = _mgr(tmp_path)
    m.record_insufficient("radarr", 1, "2160p Quality", max_tries=3, cooldown_days=30)
    m.record_insufficient("radarr", 1, "2160p Quality", max_tries=3, cooldown_days=30)

    # A different profile is active immediately and starts a fresh count.
    assert m.is_active("radarr", 1, "2160p Efficient", has_file=True)
    e = m.record_insufficient("radarr", 1, "2160p Efficient", max_tries=3, cooldown_days=30)
    assert e.tries == 1


def test_insufficient_round_trips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    m = StateManager(str(path))
    m.record_insufficient("radarr", 7, "2160p Quality", max_tries=2, cooldown_days=30)
    m.record_insufficient("radarr", 7, "2160p Quality", max_tries=2, cooldown_days=30)

    reloaded = StateManager(str(path))
    entry = reloaded.get("radarr", 7)
    assert entry.status == INSUFFICIENT and entry.tries == 2 and entry.retry_after is not None
