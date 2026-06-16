from datetime import UTC, datetime, timedelta

from optimizarr.features.optimizer.state import (
    IN_FLIGHT,
    INSUFFICIENT,
    OPEN,
    PARKED,
    SATISFIED,
    StateManager,
)


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
    assert entry is not None and entry.retry_after is not None

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
    assert entry is not None
    assert entry.status == INSUFFICIENT and entry.tries == 2 and entry.retry_after is not None


def test_record_grab_is_in_flight_and_inactive(tmp_path):
    m = _mgr(tmp_path)
    e = m.record_grab("radarr", 1, "2160p Quality", "guidA", 555)
    assert e.status == IN_FLIGHT and e.grabbed_guid == "guidA" and e.grabbed_file_id == 555
    assert e.tried_guids == ["guidA"]
    # In flight -> not active (the worker reconciles it, it is never searched while waiting).
    assert not m.is_active("radarr", 1, "2160p Quality", has_file=True)


def test_record_grab_accumulates_tried_guids(tmp_path):
    m = _mgr(tmp_path)
    m.record_grab("radarr", 1, "2160p Quality", "a", 555)
    m.resolve_in_flight("radarr", 1, imported=False)
    m.record_grab("radarr", 1, "2160p Quality", "b", 555)
    acc = m.get("radarr", 1)
    assert acc is not None and acc.tried_guids == ["a", "b"]
    assert m.tried_guids("radarr", 1, "2160p Quality", has_file=True) == {"a", "b"}


def test_resolve_in_flight_imported_satisfies_failed_opens(tmp_path):
    m = _mgr(tmp_path)
    m.record_grab("radarr", 1, "2160p Quality", "a", 555)
    assert m.resolve_in_flight("radarr", 1, imported=True).status == SATISFIED

    m.record_grab("radarr", 2, "2160p Quality", "b", 555)
    failed = m.resolve_in_flight("radarr", 2, imported=False)
    assert failed.status == OPEN and failed.tried_guids == ["b"]
    assert m.is_active("radarr", 2, "2160p Quality", has_file=True)  # open -> active


def test_tried_guids_ignored_on_profile_change_or_no_file(tmp_path):
    m = _mgr(tmp_path)
    m.record_grab("radarr", 1, "2160p Quality", "a", 555)
    m.resolve_in_flight("radarr", 1, imported=False)
    assert m.tried_guids("radarr", 1, "2160p Efficient", has_file=True) == set()  # profile changed
    assert m.tried_guids("radarr", 1, "2160p Quality", has_file=False) == set()  # file removed


def test_park_sets_cooldown_and_clears_memory(tmp_path):
    m = _mgr(tmp_path)
    m.record_grab("radarr", 1, "2160p Quality", "a", 555)
    m.resolve_in_flight("radarr", 1, imported=False)
    e = m.park("radarr", 1, "2160p Quality", cooldown_days=30)
    assert e.status == PARKED and e.retry_after is not None and e.tried_guids == []
    assert not m.is_active("radarr", 1, "2160p Quality", has_file=True)


def test_in_flight_round_trips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    m = StateManager(str(path))
    m.record_grab("radarr", 9, "2160p Quality", "g", 42)
    entry = StateManager(str(path)).get("radarr", 9)
    assert entry is not None
    assert entry.status == IN_FLIGHT and entry.grabbed_guid == "g" and entry.grabbed_file_id == 42
    assert entry.tried_guids == ["g"]


def test_in_flight_items_snapshot(tmp_path):
    m = _mgr(tmp_path)
    m.record_grab("radarr", 1, "p", "a", 1)
    m.record_grab("radarr", 2, "p", "b", 2)
    m.mark_satisfied("radarr", 3, "p")
    ids = {iid for iid, _ in m.in_flight_items("radarr")}
    assert ids == {1, 2}


def test_existing_state_file_loads_without_new_fields(tmp_path):
    # A pre-upgrade state.json (no tried_guids/grabbed_* keys) must load with safe defaults and keep
    # satisfied entries dormant — deploying the new schema never re-grabs the library.
    import json

    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "radarr": {
                    "5": {
                        "status": "satisfied",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "profile": "2160p Quality",
                    }
                }
            }
        )
    )
    m = StateManager(str(path))
    entry = m.get("radarr", 5)
    assert entry is not None
    assert entry.status == SATISFIED and entry.tried_guids == [] and entry.grabbed_guid is None
    assert not m.is_active("radarr", 5, "2160p Quality", has_file=True)
