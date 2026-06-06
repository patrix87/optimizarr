from optimizarr.features.optimizer.state import SATISFIED, StateManager


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
