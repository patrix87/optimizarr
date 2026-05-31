import logging
import threading
from datetime import UTC, datetime

from optimizarr.arr import ArrApi, RadarrApi
from optimizarr.config import Connection
from optimizarr.features.optimizer.config import OptimizerAppConfig, OptimizerConfig, default_topsis
from optimizarr.features.optimizer.state import SATISFIED, StateManager
from optimizarr.features.optimizer.topsis import GB, Topsis
from optimizarr.features.optimizer.worker import (
    _MANUAL_IMPORT_MAX_FAILS,
    OptimizerWorker,
    _AppContext,
    _ImportSlot,
    _is_importable_downgrade,
    _is_score_regression,
    age_ok,
)
from optimizarr.http import ArrTimeout

NOW = datetime(2026, 5, 28, tzinfo=UTC)


def _release(guid="g1", score=1_000_000, resolution=2160, size_gb=14.0):
    return {
        "guid": guid,
        "indexerId": 1,
        "title": f"Movie.{resolution}p",
        "customFormatScore": score,
        "quality": {"quality": {"resolution": resolution}},
        "size": int(size_gb * GB),
        "rejections": [],
    }


def _file(score=200_000, resolution="1920x1080", size_gb=30.0):
    return {
        "id": 555,
        "customFormatScore": score,
        "size": int(size_gb * GB),
        "mediaInfo": {"resolution": resolution},
    }


# ----- age gate -----


def _radarr_api():
    return RadarrApi(Connection(name="radarr", url="http://x", api_key="k"))


def _opt_cfg(min_age_days, release_type=("digitalRelease",)):
    return OptimizerAppConfig(min_age_days=min_age_days, release_type=list(release_type))


def test_age_gate_disabled_passes_everything():
    api = _radarr_api()
    cfg = _opt_cfg(0)
    assert age_ok(api, {"digitalRelease": "2026-05-27T00:00:00Z"}, cfg, NOW)  # 1 day, still ok
    assert age_ok(api, {}, cfg, NOW)  # no date, still ok when gate off


def test_age_gate_blocks_recent_and_allows_old():
    api = _radarr_api()
    cfg = _opt_cfg(14)
    assert not age_ok(api, {"digitalRelease": "2026-05-20T00:00:00Z"}, cfg, NOW)  # 8d < 14
    assert age_ok(api, {"digitalRelease": "2026-01-01T00:00:00Z"}, cfg, NOW)  # old enough
    assert not age_ok(api, {}, cfg, NOW)  # unknown date is skipped when gating is on


def test_age_gate_date_added_reads_movie_file():
    api = _radarr_api()
    cfg = _opt_cfg(14, release_type=("dateAdded",))
    item = {
        "digitalRelease": "2026-05-27T00:00:00Z",
        "movieFile": {"dateAdded": "2026-01-01T00:00:00Z"},
    }
    assert age_ok(api, item, cfg, NOW)  # uses movieFile.dateAdded, which is old


def test_age_gate_dual_requires_all_dates_old():
    # The dual gate is the heart of the change: an item must clear BOTH dates to be
    # picked. Just-released movies (recent digitalRelease) and just-imported files
    # (recent dateAdded) are kept off-limits even when the other date is old.
    api = _radarr_api()
    cfg = _opt_cfg(14, release_type=("digitalRelease", "dateAdded"))

    # Both old -> pass.
    both_old = {
        "digitalRelease": "2026-01-01T00:00:00Z",
        "movieFile": {"dateAdded": "2026-02-01T00:00:00Z"},
    }
    assert age_ok(api, both_old, cfg, NOW)

    # Fresh release, file long on disk -> still blocked by the release-age gate.
    fresh_release = {
        "digitalRelease": "2026-05-27T00:00:00Z",
        "movieFile": {"dateAdded": "2026-01-01T00:00:00Z"},
    }
    assert not age_ok(api, fresh_release, cfg, NOW)

    # Old release, just-added file -> blocked by the file-age gate.
    fresh_file = {
        "digitalRelease": "2026-01-01T00:00:00Z",
        "movieFile": {"dateAdded": "2026-05-27T00:00:00Z"},
    }
    assert not age_ok(api, fresh_file, cfg, NOW)

    # Missing date -> the gate stays closed (conservative).
    missing_release = {"movieFile": {"dateAdded": "2026-01-01T00:00:00Z"}}
    assert not age_ok(api, missing_release, cfg, NOW)


# ----- queue classification -----


def test_is_score_regression_matches_completed_with_marker():
    record = {
        "status": "completed",
        "trackedDownloadState": "importPending",
        "statusMessages": [
            {"title": "x", "messages": ["Not an upgrade for existing movie file(s)"]}
        ],
    }
    assert _is_score_regression(record)


def test_is_score_regression_matches_live_radarr_custom_format_phrasing():
    # Verbatim message from a live Radarr v3 queue — the marker must catch this exact
    # phrasing (the older "Not an upgrade" pattern is no longer what Radarr emits).
    record = {
        "status": "completed",
        "trackedDownloadState": "importPending",
        "statusMessages": [
            {
                "title": "x",
                "messages": [
                    "Not a Custom Format upgrade for existing movie file(s). "
                    "New: [1080p Bluray] (700300) do not improve on "
                    "Existing: [1080p Bluray, x265 (Bluray)] (920600)"
                ],
            }
        ],
    }
    assert _is_score_regression(record)


def test_is_score_regression_matches_sonarr_episode_phrasing():
    record = {
        "status": "completed",
        "trackedDownloadState": "importPending",
        "statusMessages": [
            {
                "title": "x",
                "messages": ["Not a Custom Format upgrade for existing episode file(s)."],
            }
        ],
    }
    assert _is_score_regression(record)


def test_is_score_regression_ignores_still_downloading():
    record = {
        "status": "downloading",
        "trackedDownloadState": "downloading",
        "statusMessages": [{"title": "x", "messages": ["Not an upgrade"]}],
    }
    assert not _is_score_regression(record)


def test_is_score_regression_ignores_other_categories():
    # Virus/executable: NOT a downgrade — leave alone.
    record = {
        "status": "completed",
        "trackedDownloadState": "importPending",
        "statusMessages": [{"title": "x", "messages": ["Found executable in download: foo.exe"]}],
    }
    assert not _is_score_regression(record)


def test_is_importable_downgrade_accepts_no_or_score_only_rejections():
    assert _is_importable_downgrade({"rejections": []})
    assert _is_importable_downgrade(
        {"rejections": [{"reason": "Not an upgrade for existing movie file(s)"}]}
    )
    # Verbatim live-Radarr rejection on the manualimport candidate side — must accept.
    assert _is_importable_downgrade(
        {
            "rejections": [
                {
                    "reason": (
                        "Not a Custom Format upgrade for existing movie file(s). "
                        "New: [1080p Bluray] (920600) do not improve on "
                        "Existing: [1080p Bluray, x265 (Bluray)] (923200)"
                    ),
                    "type": "permanent",
                }
            ]
        }
    )
    # Mixed rejections (e.g. sample) -> not importable; needs human review.
    assert not _is_importable_downgrade(
        {
            "rejections": [
                {"reason": "Not an upgrade for existing movie file(s)"},
                {"reason": "Sample"},
            ]
        }
    )


# ----- _process_one: grab vs HOLD, and what gets persisted -----


class _ProcessAdapter(ArrApi):
    """Adapter double serving canned data to _process_one and recording grabs."""

    app = "radarr"

    def __init__(self, releases, current_file):
        self._releases = releases
        self._current = current_file
        self.grabbed: list[dict] = []

    def runtime_h(self, item):
        return 2.0

    def profile_for(self, item):
        return ("2160p Quality", 2160)

    def current_file(self, item):
        return self._current

    def current_file_id(self, item):
        return (self._current or {}).get("id")

    def releases(self, item):
        return self._releases

    def label(self, item):
        return "Movie (2024)"

    def grab(self, release):
        self.grabbed.append(release)


def _worker(state, dry_run=False):
    w = OptimizerWorker.__new__(OptimizerWorker)
    w.opt = OptimizerConfig(enabled=True)
    w.state = state
    w.topsis = Topsis(default_topsis())
    w.dry_run = dry_run
    return w


def _ctx(adapter):
    ctx = _AppContext(adapter, OptimizerAppConfig())
    ctx.items_by_id = {1: {"id": 1}}
    return ctx


def test_process_one_hold_marks_satisfied(tmp_path):
    # Current file is already excellent; the candidate is no better -> HOLD -> satisfied.
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _ProcessAdapter(
        releases=[_release(score=1_000_000, resolution=2160, size_gb=14.0)],
        current_file=_file(score=1_000_000, resolution="3840x2160", size_gb=14.0),
    )
    _worker(state)._process_one(_ctx(adapter), 1)
    entry = state.get("radarr", 1)
    assert entry is not None and entry.status == SATISFIED
    assert adapter.grabbed == []


def test_process_one_act_grabs_without_marking(tmp_path):
    # A clear upgrade is grabbed, but the item is NOT marked satisfied — it stays in the
    # pool until a later evaluation HOLDs (success) or it's retried (failure).
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _ProcessAdapter(
        releases=[_release(score=1_000_000, resolution=2160, size_gb=14.0)],
        current_file=_file(score=200_000, resolution="1920x1080", size_gb=30.0),
    )
    _worker(state)._process_one(_ctx(adapter), 1)
    assert len(adapter.grabbed) == 1
    assert state.get("radarr", 1) is None


def test_process_one_dry_run_does_not_grab(tmp_path):
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _ProcessAdapter(
        releases=[_release(score=1_000_000, resolution=2160, size_gb=14.0)],
        current_file=_file(score=200_000, resolution="1920x1080", size_gb=30.0),
    )
    _worker(state, dry_run=True)._process_one(_ctx(adapter), 1)
    assert adapter.grabbed == []
    assert state.get("radarr", 1) is None


def test_process_app_once_downgrades_search_timeout_to_warning(tmp_path, caplog):
    # A slow interactive indexer search (ArrTimeout) must be handled gracefully: logged as a
    # concise WARNING (not an ERROR traceback), the item left in `evaluated` for a later pass,
    # and never marked satisfied.
    state = StateManager(str(tmp_path / "s.json"))
    w = _worker(state)

    class _TimeoutAdapter(_ProcessAdapter):
        def queue_items(self):
            return []

        def releases(self, item):
            raise ArrTimeout("GET /api/v3/release?movieId=1 timed out after 240s")

    adapter = _TimeoutAdapter(releases=[], current_file=None)
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=False))
    ctx.items_by_id = {1: {"id": 1}}
    ctx.pool = [1]
    ctx.last_refresh = datetime.now(UTC)  # keep needs_refresh() False

    with caplog.at_level(logging.WARNING):
        handled = w._process_app_once(ctx)

    assert handled is True  # work was attempted; loop continues, no crash
    assert 1 in ctx.evaluated  # retained for retry on the next pass
    assert state.get("radarr", 1) is None  # NOT marked satisfied
    assert any("will retry on a later pass" in r.getMessage() for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_process_app_once_pauses_grabs_when_import_backlog_exceeds_max(tmp_path):
    # More than import_max (default 2) completed downloads waiting to import -> stop grabbing
    # so the backlog drains first. The pool item is left untouched (no grab).
    state = StateManager(str(tmp_path / "s.json"))
    records = [
        {
            "id": i,
            "movieId": 100 + i,
            "downloadId": f"d{i}",
            "status": "completed",
            "trackedDownloadState": "importPending",
        }
        for i in range(3)  # 3 > import_max 2
    ]
    adapter = _QueueAdapter(records)
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=False))
    ctx.items_by_id = {1: {"id": 1}}
    ctx.pool = [1]
    ctx.last_refresh = datetime.now(UTC)  # keep needs_refresh() False

    w = _worker(state)
    assert w.opt.import_max == 2
    assert w._process_app_once(ctx) is False  # gate tripped
    assert ctx.pool == [1]  # pool item not consumed -> nothing grabbed
    assert 1 not in ctx.evaluated


class _GrabQueueAdapter(_ProcessAdapter):
    """_ProcessAdapter (can grab) plus a canned queue, for gate tests that reach the grab."""

    _queue_id_field = "movieId"

    def __init__(self, records, releases, current_file):
        super().__init__(releases, current_file)
        self._records = records

    def queue_items(self):
        return self._records


def test_importblocked_does_not_count_toward_import_gate(tmp_path):
    # importBlocked is manual-only; it must NOT count toward the gate, else a stuck manual item
    # freezes grabbing forever. 5 importBlocked records (well over import_max 2) must still grab.
    state = StateManager(str(tmp_path / "s.json"))
    records = [
        {
            "id": i,
            "movieId": 100 + i,
            "status": "completed",
            "trackedDownloadState": "importBlocked",
        }
        for i in range(5)
    ]
    adapter = _GrabQueueAdapter(
        records,
        releases=[_release(score=1_000_000, resolution=2160, size_gb=14.0)],
        current_file=_file(score=200_000, resolution="1920x1080", size_gb=30.0),
    )
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=False))
    ctx.items_by_id = {1: {"id": 1}}
    ctx.pool = [1]
    ctx.last_refresh = datetime.now(UTC)

    w = _worker(state)
    assert w._process_app_once(ctx) is True  # gate not tripped
    assert len(adapter.grabbed) == 1  # it grabbed despite 5 blocked items


def test_build_pool_holds_progress_across_refresh_then_resets(tmp_path):
    # A list refresh must NOT restart the pass: items already evaluated stay excluded
    # until the whole active set is covered, then the pass resets.
    state = StateManager(str(tmp_path / "s.json"))
    w = _worker(state)
    ctx = _AppContext(_ProcessAdapter([], None), OptimizerAppConfig())
    ctx.items_by_id = {1: {"id": 1}, 2: {"id": 2}, 3: {"id": 3}}
    now = datetime.now(UTC)

    w._build_pool(ctx, now)
    assert set(ctx.pool) == {1, 2, 3}

    # Two items processed this pass; a refresh happened (evaluated preserved).
    ctx.evaluated = {1, 2}
    w._build_pool(ctx, now)
    assert ctx.pool == [3]  # only the unvisited item remains

    # Last item visited -> pool empties -> pass resets to a fresh full sweep.
    ctx.evaluated = {1, 2, 3}
    w._build_pool(ctx, now)
    assert set(ctx.pool) == {1, 2, 3}
    assert ctx.evaluated == set()


class _QueueAdapter(ArrApi):
    """Adapter double for queue/manualimport tests — records POSTed imports."""

    app = "radarr"
    _queue_id_field = "movieId"

    def __init__(self, records, candidates=None, raises=None):
        self._records = records
        self._candidates = candidates or {}
        self._raises: dict[str, str] = raises or {}
        self.imports: list[tuple[list[dict], str]] = []
        self.candidate_calls: list[str] = []

    def queue_items(self):
        return self._records

    def set_records(self, records):
        self._records = records

    def manual_import_candidates(self, download_id, *, timeout=None, retry=True):
        self.candidate_calls.append(download_id)
        if download_id in self._raises:
            raise RuntimeError(self._raises[download_id])
        return self._candidates.get(download_id, [])

    def manual_import(self, items, import_mode="auto", *, timeout=None, retry=True):
        # Record items + mode separately so the test can assert both without coupling to the
        # real method's body-wrapping behavior.
        self.imports.append((list(items), import_mode))


def _downgrade_record(download_id="dl1", movie_id=42):
    return {
        "id": 1,
        "movieId": movie_id,
        "downloadId": download_id,
        "title": "Movie.2024.2160p.WEB.x265",
        "status": "completed",
        "trackedDownloadState": "importPending",
        "statusMessages": [
            {"title": "x", "messages": ["Not an upgrade for existing movie file(s)"]}
        ],
    }


def _wait_for_slot(ctx, timeout=2.0):
    """Tests block on the slot's daemon thread so they can assert post-import state."""
    thread = ctx.import_slot._thread
    if thread is not None:
        thread.join(timeout=timeout)


def test_handle_queue_imports_force_imports_downgrades(tmp_path):
    state = StateManager(str(tmp_path / "s.json"))
    record = _downgrade_record()
    candidate = {
        "path": "/downloads/Movie.2024.mkv",
        "movie": {"id": 42},
        "quality": {"quality": {"name": "WEBDL-2160p"}},
        "rejections": [{"reason": "Not an upgrade for existing movie file(s)"}],
    }
    adapter = _QueueAdapter([record], candidates={"dl1": [candidate]})
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    w = _worker(state)
    w._handle_queue_imports(ctx)
    _wait_for_slot(ctx)
    assert adapter.imports == [([candidate], "auto")]
    # The command is async: the item is recorded as pending, NOT yet declared imported.
    assert ctx.import_slot.is_pending("dl1")


def test_handle_queue_imports_confirms_only_after_item_leaves_queue(tmp_path, caplog):
    # The premature/duplicate "auto-imported" bug: success must be logged once, and only once
    # the item actually leaves the queue, not at POST time.
    state = StateManager(str(tmp_path / "s.json"))
    record = _downgrade_record()
    candidate = {"path": "/x.mkv", "movie": {"id": 42}, "rejections": []}
    adapter = _QueueAdapter([record], candidates={"dl1": [candidate]})
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    w = _worker(state)

    with caplog.at_level(logging.INFO):
        # Tick 1: submit. Item still queued -> no "auto-imported" yet.
        w._handle_queue_imports(ctx)
        _wait_for_slot(ctx)
        assert not any("auto-imported" in r.getMessage() for r in caplog.records)

        # Tick 2: item STILL in queue (import pending) -> must NOT re-submit, still no claim.
        w._handle_queue_imports(ctx)
        _wait_for_slot(ctx)
        assert adapter.imports == [([candidate], "auto")]  # only one POST, no re-submission
        assert not any("auto-imported" in r.getMessage() for r in caplog.records)

        # Tick 3: item has left the queue -> confirmed import, logged exactly once.
        caplog.clear()
        adapter.set_records([])
        w._handle_queue_imports(ctx)
        _wait_for_slot(ctx)

    confirmed = [r for r in caplog.records if "auto-imported" in r.getMessage()]
    assert len(confirmed) == 1
    assert "Movie.2024.2160p.WEB.x265" in confirmed[0].getMessage()
    assert not ctx.import_slot.is_pending("dl1")


def test_handle_queue_imports_marks_nonimportable_skip_no_resubmit(tmp_path):
    # A non-score-regression rejection (Sample) is left alone AND skipped for the session, so
    # the slow candidate scan isn't repeated every tick.
    state = StateManager(str(tmp_path / "s.json"))
    candidate = {
        "path": "/x.mkv",
        "rejections": [
            {"reason": "Not an upgrade for existing movie file(s)"},
            {"reason": "Sample"},
        ],
    }
    adapter = _QueueAdapter([_downgrade_record()], candidates={"dl1": [candidate]})
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    w = _worker(state)

    w._handle_queue_imports(ctx)
    _wait_for_slot(ctx)
    w._handle_queue_imports(ctx)  # second tick
    _wait_for_slot(ctx)

    assert adapter.imports == []
    assert ctx.import_slot.should_skip("dl1")
    assert adapter.candidate_calls == ["dl1"]  # scanned once, then skipped


def test_handle_queue_imports_dry_run_does_not_post(tmp_path):
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _QueueAdapter(
        [_downgrade_record()],
        candidates={"dl1": [{"path": "/x.mkv", "rejections": []}]},
    )
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    w = _worker(state, dry_run=True)
    w._handle_queue_imports(ctx)
    _wait_for_slot(ctx)
    assert adapter.imports == []


def test_handle_queue_imports_skips_when_candidates_have_other_rejections(tmp_path):
    # Sample rejection alongside the downgrade -> leave alone for human review.
    state = StateManager(str(tmp_path / "s.json"))
    candidate = {
        "path": "/downloads/x.mkv",
        "rejections": [
            {"reason": "Not an upgrade for existing movie file(s)"},
            {"reason": "Sample"},
        ],
    }
    adapter = _QueueAdapter([_downgrade_record()], candidates={"dl1": [candidate]})
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    _worker(state)._handle_queue_imports(ctx)
    _wait_for_slot(ctx)
    assert adapter.imports == []


def test_handle_queue_imports_disabled_is_noop(tmp_path):
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _QueueAdapter(
        [_downgrade_record()],
        candidates={"dl1": [{"path": "/x.mkv", "rejections": []}]},
    )
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=False))
    _worker(state)._handle_queue_imports(ctx)
    _wait_for_slot(ctx)
    assert adapter.imports == []


def test_handle_queue_imports_skips_when_slot_busy(tmp_path):
    # If another import is already in flight, the tick is a no-op — no second thread.
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _QueueAdapter(
        [_downgrade_record()],
        candidates={"dl1": [{"path": "/x.mkv", "rejections": []}]},
    )
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))

    # Occupy the slot with a long-running fake thread.
    blocker = threading.Event()
    ctx.import_slot.submit("other", blocker.wait)
    try:
        _worker(state)._handle_queue_imports(ctx)
        # Slot is still the one we set; no new submission.
        assert adapter.imports == []
    finally:
        blocker.set()
        _wait_for_slot(ctx)


def test_handle_queue_imports_skips_downloadid_with_too_many_failures(tmp_path):
    # A downloadId that has hit _MANUAL_IMPORT_MAX_FAILS is dropped from the candidate
    # search entirely until worker restart, so the slot stops burning the 5-min timeout
    # on a permanently broken record.
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _QueueAdapter(
        [_downgrade_record()],
        candidates={"dl1": [{"path": "/x.mkv", "rejections": []}]},
    )
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    ctx.import_slot._fail_counts["dl1"] = _MANUAL_IMPORT_MAX_FAILS
    _worker(state)._handle_queue_imports(ctx)
    _wait_for_slot(ctx)
    assert adapter.imports == []


def test_import_slot_busy_releases_after_target_returns():
    slot = _ImportSlot()
    done = threading.Event()

    def target():
        done.wait(timeout=2)
        return True

    assert slot.submit("dl1", target)
    assert slot.busy()
    # Second submit while busy is a no-op.
    assert not slot.submit("dl2", lambda: True)
    done.set()
    # Wait for the thread to finish so busy() flips back.
    if slot._thread is not None:
        slot._thread.join(timeout=2)
    assert not slot.busy()


def test_import_slot_failure_count_increments_and_skip_kicks_in():
    slot = _ImportSlot()
    # Run a failing target enough times to hit the skip threshold.
    for _ in range(_MANUAL_IMPORT_MAX_FAILS):
        slot.submit("dl1", lambda: False)
        if slot._thread is not None:
            slot._thread.join(timeout=2)
    assert slot.should_skip("dl1")
    # A different downloadId is unaffected.
    assert not slot.should_skip("dl2")


def test_import_slot_failure_count_resets_on_success():
    slot = _ImportSlot()
    slot.submit("dl1", lambda: False)
    if slot._thread is not None:
        slot._thread.join(timeout=2)
    assert slot._fail_counts.get("dl1") == 1
    slot.submit("dl1", lambda: True)
    if slot._thread is not None:
        slot._thread.join(timeout=2)
    assert "dl1" not in slot._fail_counts


def test_run_manual_import_returns_false_on_get_failure(tmp_path):
    # If the manualimport GET raises, the slot's fail counter must tick up
    # (via the _run wrapper) by returning False.
    state = StateManager(str(tmp_path / "s.json"))
    adapter = _QueueAdapter(
        records=[_downgrade_record()],
        raises={"dl1": "simulated timeout"},
    )
    ctx = _AppContext(adapter, OptimizerAppConfig(auto_import_downgrades=True))
    w = _worker(state)
    assert w._run_manual_import(ctx, "Movie (2024)", "dl1") is False
    assert adapter.imports == []


def test_queue_active_filter_excludes_completed_when_flag_on():
    # ignore_completed_in_queue mirrors how _process_app_once computes queue_count.
    active = {"status": "downloading", "trackedDownloadState": "downloading"}
    pending = {"status": "completed", "trackedDownloadState": "importPending"}
    importing = {"status": "completed", "trackedDownloadState": "importing"}
    records = [active, pending, importing]
    assert sum(1 for r in records if ArrApi.is_queue_item_active(r)) == 1
    # When the flag is off, the worker would use len(records) instead -> all 3 count.
    assert len(records) == 3


def test_build_pool_excludes_satisfied(tmp_path):
    state = StateManager(str(tmp_path / "s.json"))
    state.mark_satisfied("radarr", 2)
    w = _worker(state)
    ctx = _AppContext(_ProcessAdapter([], None), OptimizerAppConfig())
    ctx.items_by_id = {1: {"id": 1}, 2: {"id": 2}}
    w._build_pool(ctx, datetime.now(UTC))
    assert ctx.pool == [1]  # satisfied item 2 is out of the pool
