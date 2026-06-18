"""Optimizer worker: walk the library, re-pick better releases, grab them.

"Optimized" means the algorithm can no longer find anything better than the current file
(HOLD) — never merely "we triggered a grab". The worker is deliberately simple:

  - refresh the item list on a slow interval (list_refresh_minutes);
  - on each tick, if the download queue is at/under queue_max, pick a not-yet-satisfied
    item that isn't already in the queue, evaluate it, and either grab a better release
    or mark it satisfied (HOLD);
  - a grab is never recorded. Success shows up as a HOLD on the next evaluation (→
    satisfied); failure leaves the item unsatisfied so it's retried later, with the failed
    release now blocklisted by Radarr/Sonarr's Failed Download Handling.

Downloads in progress are read live from the queue (gate + per-item skip), so there's no
in-flight bookkeeping and a restart needs no reconciliation. The per-item decision lives in
.decision; app-specific HTTP lives behind the optimizarr.arr clients; the loop here is
app-agnostic.
"""

import logging
import random
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from optimizarr.arr import ArrApi, build_client
from optimizarr.config import Config
from optimizarr.dates import age_days
from optimizarr.features.optimizer.config import OptimizerAppConfig, OptimizerConfig
from optimizarr.features.optimizer.decision import decide, format_decision
from optimizarr.features.optimizer.state import StateManager
from optimizarr.features.optimizer.topsis import Topsis
from optimizarr.http import ArrTimeout

logger = logging.getLogger("optimizarr")

# Score-regression marker — case-insensitive substring. Covers every phrasing observed
# in live Radarr/Sonarr v3 queues:
#   - "Not an upgrade for existing movie file(s)" (older Radarr)
#   - "Not a Custom Format upgrade for existing movie file(s). New: [...] do not improve
#     on Existing: [...]" (current Radarr)
#   - Sonarr's "episode file(s)" variants of both.
# The invariant substring across all of them is "upgrade for existing". Anything OTHER
# than this in statusMessages (executable / archive file / sample / mediainfo mismatch)
# is left untouched by auto-import — those get a separate handler later.
_SCORE_REGRESSION_MARKER = "upgrade for existing"

# Auto-import: the manualimport endpoint runs MediaInfo and can take 30-120s per file on
# first call per downloadId, then caches. We give it a generous timeout and disable retry
# so a single failure doesn't compound into minutes of backoff blocking the slot.
_MANUAL_IMPORT_TIMEOUT_SEC = 300
# A downloadId that has failed this many times this session is skipped to stop us from
# burning the 5-min timeout on a permanently broken record every tick. Counter is in-memory
# only — a worker restart gives every record a clean slate.
_MANUAL_IMPORT_MAX_FAILS = 3

# pick_order -> (item sort key, reverse). "random" is handled separately (shuffle). The keys
# read already-fetched item fields via the adapter (no extra HTTP); see ArrApi.label /
# file_size / date_added / release_date. label() is casefolded for case-insensitive A->Z.
_PICK_ORDER_KEYS: dict[str, tuple[Callable[[ArrApi, dict], Any], bool]] = {
    "alphabetical_asc": (lambda a, it: a.label(it).casefold(), False),
    "alphabetical_desc": (lambda a, it: a.label(it).casefold(), True),
    "size_asc": (lambda a, it: a.file_size(it), False),
    "size_desc": (lambda a, it: a.file_size(it), True),
    "date_added_asc": (lambda a, it: a.date_added(it), False),
    "date_added_desc": (lambda a, it: a.date_added(it), True),
    "release_date_asc": (lambda a, it: a.release_date(it), False),
    "release_date_desc": (lambda a, it: a.release_date(it), True),
}


def order_pool(
    pool: list[int], items_by_id: dict[int, dict], adapter: ArrApi, pick_order: str
) -> list[int]:
    """Order the active-item pool for this pass per the configured pick_order. Shuffles in
    place for "random"; otherwise returns a new list sorted by the pick_order's key."""
    if pick_order == "random":
        random.shuffle(pool)
        return pool
    key, reverse = _PICK_ORDER_KEYS[pick_order]
    return sorted(pool, key=lambda iid: key(adapter, items_by_id[iid]), reverse=reverse)


class _ImportSlot:
    """Single-slot per-app coordinator for manualimport calls.

    Spawns a daemon thread for each call so the worker's main loop isn't blocked by
    Radarr/Sonarr's slow MediaInfo parses. While a thread is alive, the slot is busy
    and the caller skips the tick. Per-downloadId failure counts let us stop retrying
    a broken record forever (cleared only on worker restart).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._fail_counts: dict[str, int] = {}
        # downloadId -> title for imports we've POSTed but not yet confirmed. The ManualImport
        # command is async (Radarr can take many minutes to actually import a large file), so a
        # POST does NOT mean "imported". We hold the id here and only declare success once the
        # item has left the queue (reconcile_completed), and we never re-submit while it's here.
        self._pending: dict[str, str] = {}
        # downloadIds we've decided not to touch again this session (non-score-regression
        # rejections). Cleared only on restart.
        self._skip: set[str] = set()

    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def should_skip(self, download_id: str) -> bool:
        with self._lock:
            return (
                download_id in self._skip
                or self._fail_counts.get(download_id, 0) >= _MANUAL_IMPORT_MAX_FAILS
            )

    def mark_skip(self, download_id: str) -> None:
        with self._lock:
            self._skip.add(download_id)

    def is_pending(self, download_id: str) -> bool:
        with self._lock:
            return download_id in self._pending

    def mark_submitted(self, download_id: str, title: str) -> None:
        with self._lock:
            self._pending[download_id] = title

    def reconcile_completed(self, queue_download_ids: set[str]) -> list[str]:
        """Drop pending downloadIds that have left the queue (those actually imported) and
        return their titles so the caller can log a *confirmed* import. An item Radarr refuses
        stays queued, so it stays pending (silent) and is never re-submitted or falsely claimed."""
        with self._lock:
            done = [(d, t) for d, t in self._pending.items() if d not in queue_download_ids]
            for d, _ in done:
                del self._pending[d]
        return [t for _, t in done]

    def submit(self, download_id: str, target: Callable[[], bool]) -> bool:
        """Spawn target() in a daemon thread if the slot is free. target() returns True on
        success. Returns True if spawned, False if already busy."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(
                target=self._run, args=(target, download_id), daemon=True
            )
            self._thread.start()
            return True

    def _run(self, target: Callable[[], bool], download_id: str) -> None:
        try:
            ok = target()
        except Exception:
            logger.exception("[manualimport] worker thread crashed for downloadId=%s", download_id)
            ok = False
        if ok:
            self._fail_counts.pop(download_id, None)
        else:
            self._fail_counts[download_id] = self._fail_counts.get(download_id, 0) + 1
            if self._fail_counts[download_id] >= _MANUAL_IMPORT_MAX_FAILS:
                logger.warning(
                    "[manualimport] downloadId=%s reached %d failures; skipping for the "
                    "remainder of this session",
                    download_id,
                    _MANUAL_IMPORT_MAX_FAILS,
                )


def age_ok(api: ArrApi, item: dict, app_cfg: OptimizerAppConfig, now: datetime) -> bool:
    """True if the item passes ALL configured release-date gates. With min_age_days <= 0 the
    gate is off. Each entry in release_type must be at least min_age_days old; a missing date
    keeps the gate closed (better to wait than to touch something whose release we can't
    verify — that's the whole point of the two-gate setup)."""
    if app_cfg.min_age_days <= 0:
        return True
    for rt in app_cfg.release_type:
        age = age_days(api.reference_date(item, rt), now)
        if age is None or age < app_cfg.min_age_days:
            return False
    return True


class _AppContext:
    """Per-app worker state: client, its config, cached item list, active pool, and the
    single-slot manualimport coordinator (one in-flight downgrade-import per app)."""

    def __init__(self, adapter: ArrApi, app_cfg: OptimizerAppConfig):
        self.adapter = adapter
        self.app_cfg = app_cfg
        self.items_by_id: dict[int, dict] = {}
        self.pool: list[int] = []
        self.evaluated: set[int] = set()
        self.last_refresh: datetime | None = None
        self.import_slot = _ImportSlot()

    def needs_refresh(self, now: datetime, list_refresh_minutes: int) -> bool:
        if self.last_refresh is None:
            return True
        age_min = (now - self.last_refresh).total_seconds() / 60
        return age_min >= list_refresh_minutes


class OptimizerWorker:
    def __init__(self, config: Config, state: StateManager):
        self.config = config
        self.opt: OptimizerConfig = config.optimizer
        self.state = state
        self.topsis = Topsis(self.opt.topsis)
        self.dry_run = config.dry_run
        self._stop = threading.Event()
        self._schedule_active: bool | None = None  # None = unknown (first tick)

        conns = {"radarr": config.radarr, "sonarr": config.sonarr}
        app_cfgs = {"radarr": self.opt.radarr, "sonarr": self.opt.sonarr}
        self.contexts: dict[str, _AppContext] = {}
        for app, conn in conns.items():
            if conn is None or not app_cfgs[app].enabled:
                continue
            self.contexts[app] = _AppContext(build_client(app, conn), app_cfgs[app])

    def stop(self) -> None:
        self._stop.set()

    # ----- per-app machinery -----

    def _refresh(self, ctx: _AppContext, now: datetime) -> None:
        adapter = ctx.adapter
        # Safe reconciliation: a failed or interrupted library fetch must NEVER clear the known
        # item set (and we never prune state from the list anyway). On error, keep the previous
        # items_by_id and retry on the next tick (last_refresh is left unchanged).
        try:
            adapter.refresh_profiles()
            # Select on hasFile alone (not monitored): the optimizer improves the existing
            # library, and the unmonitor feature deliberately strips monitoring once a file
            # exists. The age gate is the optimizer's own min_age_days.
            items = [
                it
                for it in adapter.list_items()
                if adapter.has_file(it) and age_ok(adapter, it, ctx.app_cfg, now)
            ]
        except Exception:
            logger.exception(
                "[%s] library refresh failed; keeping the previous item set", adapter.app
            )
            return
        ctx.items_by_id = {adapter.item_id(it): it for it in items}
        # NB: ctx.evaluated is intentionally NOT cleared here. A refresh only updates the
        # candidate set (new items become pickable, removed ones drop); the current pass
        # keeps its progress so a slow walk over a large library isn't restarted every
        # list_refresh_minutes. The pass resets in _build_pool once it's fully covered.
        ctx.last_refresh = now
        logger.info("[%s] list refreshed: %d items with files", adapter.app, len(items))

    def _in_active_hours(self, now_local: datetime | None = None) -> bool:
        """Return True if the current local time is inside the configured active window.
        An empty schedule means always active. A window where start >= end crosses midnight:
        e.g. start=23:00, end=08:00 is active from 23:00 that day through 08:00 the next."""
        schedule = self.opt.schedule
        if not schedule:
            return True
        now_local = now_local or datetime.now()
        t = now_local.time()
        today = now_local.weekday()  # 0=Mon, 6=Sun
        yesterday = (today - 1) % 7

        if today in schedule:
            s, e = schedule[today].start, schedule[today].end
            if s < e:  # same-day window: active between s and e
                if s <= t < e:
                    return True
            elif t >= s:  # cross-midnight: today's portion (s until midnight)
                return True

        if yesterday in schedule:
            s, e = schedule[yesterday].start, schedule[yesterday].end
            if s >= e and t < e:  # yesterday's window crosses into today (midnight until e)
                return True

        return False

    def _build_pool(self, ctx: _AppContext) -> None:
        app = ctx.adapter.app
        adapter = ctx.adapter

        def active(exclude_evaluated: bool) -> list[int]:
            out: list[int] = []
            for item_id, item in ctx.items_by_id.items():
                if exclude_evaluated and item_id in ctx.evaluated:
                    continue
                profile_name, _target_res = adapter.profile_for(item)
                if self.state.is_active(app, item_id, profile_name, adapter.has_file(item)):
                    out.append(item_id)
            return out

        ctx.pool = active(exclude_evaluated=True)
        if not ctx.pool and ctx.evaluated:
            # Every active item has been evaluated this pass — reset and start a new one.
            ctx.evaluated.clear()
            ctx.pool = active(exclude_evaluated=False)

        ctx.pool = order_pool(ctx.pool, ctx.items_by_id, ctx.adapter, self.opt.pick_order)

    def _reconcile_in_flight(self, ctx: _AppContext, queue_ids: set[int], now: datetime) -> None:
        """Resolve in-flight grabs from data already in hand (no indexer call). Once a grabbed item
        has left the queue AND its settle window has elapsed, a changed current-file id means the
        grab imported -> satisfy; an unchanged file means it failed -> open (try the next-best
        untried release next pass). The settle window guards the gap between the grab POST and the
        download appearing in the queue, so a just-grabbed item is never called failed too early."""
        adapter = ctx.adapter
        settle = timedelta(minutes=self.opt.grab.settle_minutes)
        for item_id, entry in self.state.in_flight_items(adapter.app):
            if item_id in queue_ids:
                continue  # still downloading / waiting to import -> keep waiting
            if entry.grabbed_at and now - datetime.fromisoformat(entry.grabbed_at) < settle:
                continue  # within the settle window -> not yet resolved
            item = ctx.items_by_id.get(item_id)
            if item is None:
                continue  # not in the current list (removed / age-gated) -> leave as-is
            imported = adapter.current_file_id(item) != entry.grabbed_file_id
            # Misadvertised release: it imported far below its advertised score (same profile). The
            # file is not what the release claimed, so add it to our permanent blocklist (never
            # grabbed again, even across profile changes) and re-open the item to find a genuinely
            # better release instead of marking the lie satisfied. Download FAILURES are not handled
            # here: Radarr/Sonarr already blocklist those, and the eligible() filter drops them.
            if imported and entry.grabbed_guid and self._is_misadvertised(ctx, item, entry):
                self.state.add_to_blocklist(adapter.app, item_id, entry.grabbed_guid)
                self.state.resolve_in_flight(adapter.app, item_id, imported=False)
                logger.info(
                    "[%s] %s: imported >= %d below advertised score; blocklisted release, "
                    "re-evaluating",
                    adapter.app,
                    adapter.label(item),
                    self.opt.grab.blocklist_score_drop,
                )
                continue
            self.state.resolve_in_flight(adapter.app, item_id, imported)
            logger.info(
                "[%s] %s: in-flight grab resolved -> %s",
                adapter.app,
                adapter.label(item),
                "imported, satisfied" if imported else "failed, will try next-best",
            )

    def _is_misadvertised(self, ctx: _AppContext, item: dict, entry) -> bool:
        """True if the imported file scores at least grab.blocklist_score_drop below the grabbed
        release's advertised customFormatScore, with the profile unchanged since the grab (so the
        two scores are comparable). Reads the imported file only when the check is enabled."""
        drop_min = self.opt.grab.blocklist_score_drop
        if drop_min <= 0 or entry.grabbed_score is None:
            return False
        profile_name, _ = ctx.adapter.profile_for(item)
        if entry.profile != profile_name:
            return False  # profile changed -> scores not comparable
        imported_score = (ctx.adapter.current_file(item) or {}).get("customFormatScore")
        if imported_score is None:
            return False
        return (entry.grabbed_score - imported_score) >= drop_min

    def _process_one(self, ctx: _AppContext, item_id: int) -> None:
        adapter = ctx.adapter
        item = ctx.items_by_id[item_id]
        runtime_h = adapter.runtime_h(item)
        profile_name, target_res = adapter.profile_for(item)
        has_file = adapter.has_file(item)
        current_file = adapter.current_file(item)
        releases = adapter.releases(item)
        # Releases already grabbed for this item are never grabbed again: tried (per-profile,
        # anti-oscillation) plus the permanent blocklist (releases proved broken, all profiles).
        tried = self.state.tried_guids(adapter.app, item_id, profile_name, has_file)
        blocklist = self.state.blocklisted(adapter.app, item_id)

        decision = decide(
            self.topsis,
            releases,
            runtime_h,
            profile_name,
            target_res,
            current_file,
            allow_size_increase=ctx.app_cfg.allow_size_increase,
            allow_quality_downgrade=ctx.app_cfg.allow_quality_downgrade,
            satisfied_score=self.opt.retry.satisfied_score,
            tried_guids=tried,
            blocklist=blocklist,
        )
        label = adapter.label(item)
        logger.info("%s", format_decision(adapter.app, label, decision, self.dry_run))

        if decision.action == "HOLD":
            # Satisfy (permanently) when the current file is optimal for its profile (incl. the
            # give-up case: nothing UNTRIED beats it). An insufficient-candidates HOLD instead
            # counts a retry attempt and, once exhausted, rests the item for the cooldown.
            if not self.dry_run:
                if decision.satisfy:
                    self.state.mark_satisfied(adapter.app, item_id, profile_name)
                elif decision.insufficient:
                    self._record_insufficient(ctx, item_id, profile_name, label)
            return

        # ACT. Grab cap: after grab.max_tries distinct releases without satisfying, park the item
        # for a cooldown instead of grabbing yet another (bounds downloads when the indexer keeps
        # surfacing fresh-but-doomed releases).
        if self.dry_run:
            return
        if len(tried) >= self.opt.grab.max_tries:
            self.state.park(adapter.app, item_id, profile_name, self.opt.retry.cooldown_days)
            logger.info(
                "[%s] %s: grabbed %d releases without satisfying; parking for %d days",
                adapter.app,
                label,
                len(tried),
                self.opt.retry.cooldown_days,
            )
            return

        # Record the grab as in-flight BEFORE the POST so a crash in between cannot double-grab:
        # the guid is now in tried_guids, and reconciliation will treat the (possibly never-sent)
        # grab as failed and move on to the next-best untried release.
        release = decision.release or {}
        guid = release.get("guid")
        if guid:
            self.state.record_grab(
                adapter.app,
                item_id,
                profile_name,
                guid,
                adapter.current_file_id(item),
                release.get("customFormatScore"),
            )
        adapter.grab(release)

    def _record_insufficient(
        self, ctx: _AppContext, item_id: int, profile_name: str | None, label: str
    ) -> None:
        """Count a too-few-candidates attempt for an item and log the outcome: either it will be
        retried on a later pass, or it has exhausted its tries and is rested until retry_after."""
        retry = self.opt.retry
        entry = self.state.record_insufficient(
            ctx.adapter.app, item_id, profile_name, retry.max_tries, retry.cooldown_days
        )
        if entry.retry_after is not None:
            logger.info(
                "[%s] %s: too few candidates after %d tries; resting until %s",
                ctx.adapter.app,
                label,
                entry.tries,
                entry.retry_after,
            )
        else:
            logger.info(
                "[%s] %s: too few candidates (try %d/%d); will retry on a later pass",
                ctx.adapter.app,
                label,
                entry.tries,
                retry.max_tries,
            )

    def _handle_queue_imports(self, ctx: _AppContext) -> None:
        """At most one in-flight manualimport per app per tick.

        The Radarr/Sonarr manualimport endpoint runs MediaInfo and can take minutes per
        downloadId on first call, so the actual GET/POST runs in a daemon thread via
        ctx.import_slot. The main loop only looks at the queue here, picks the first
        matching downgrade that isn't already being handled and hasn't repeatedly failed,
        and submits it. Strict scope: score-regression rejections only; other categories
        (virus / sample / mismatch) are left untouched."""
        if not ctx.app_cfg.auto_import_downgrades:
            return
        adapter = ctx.adapter
        try:
            records = adapter.queue_items()
        except Exception:
            logger.exception("[%s] queue fetch failed during auto-import scan", adapter.app)
            return

        # Confirm earlier submissions: any pending downloadId no longer in the queue actually
        # imported (Radarr removes the queue item on a real import). Log it now, confirmed,
        # instead of optimistically at POST time.
        queue_ids = {r["downloadId"] for r in records if r.get("downloadId")}
        for title in ctx.import_slot.reconcile_completed(queue_ids):
            logger.info("[%s] auto-imported downgrade %s", adapter.app, title)

        if ctx.import_slot.busy():
            return

        target = next(
            (
                r
                for r in records
                if _is_score_regression(r)
                and r.get("downloadId")
                and not ctx.import_slot.should_skip(r["downloadId"])
                and not ctx.import_slot.is_pending(r["downloadId"])  # already awaiting import
            ),
            None,
        )
        if target is None:
            return

        download_id = target["downloadId"]
        title = target.get("title") or f"queue#{target.get('id')}"
        if self.dry_run:
            logger.info(
                "[%s] would manual-import (downgrade) %s (downloadId=%s)",
                adapter.app,
                title,
                download_id,
            )
            return

        ctx.import_slot.submit(
            download_id, lambda: self._run_manual_import(ctx, title, download_id)
        )

    def _run_manual_import(self, ctx: _AppContext, title: str, download_id: str) -> bool:
        """Inside the spawned daemon thread: GET candidates, filter to importable downgrades,
        POST the ManualImport command. Returns True on a clean outcome, False on a transport
        failure so the slot's failure counter ticks up.

        Crucially it does NOT log "auto-imported" here: the command is async, so we only record
        the submission and let _handle_queue_imports confirm the import once the item leaves the
        queue. That stops the same item being re-submitted (and falsely re-announced) every tick
        while Radarr is still importing it."""
        adapter = ctx.adapter
        try:
            candidates = adapter.manual_import_candidates(
                download_id, timeout=_MANUAL_IMPORT_TIMEOUT_SEC, retry=False
            )
        except ArrTimeout as e:
            logger.warning(
                "[%s] candidate scan for %s timed out; will retry later (%s)", adapter.app, title, e
            )
            return False
        except Exception:
            logger.exception("[%s] manualimport GET failed for %s", adapter.app, title)
            return False

        importable = [c for c in candidates if _is_importable_downgrade(c)]
        if not importable:
            logger.info(
                "[%s] no importable candidates for downgrade %s (downloadId=%s); skipping",
                adapter.app,
                title,
                download_id,
            )
            ctx.import_slot.mark_skip(download_id)
            return True

        try:
            adapter.manual_import(
                importable,
                import_mode="auto",
                timeout=_MANUAL_IMPORT_TIMEOUT_SEC,
                retry=False,
            )
        except ArrTimeout as e:
            logger.warning(
                "[%s] import submit for %s timed out; will retry later (%s)", adapter.app, title, e
            )
            return False
        except Exception:
            logger.exception("[%s] manualimport POST failed for %s", adapter.app, title)
            return False

        ctx.import_slot.mark_submitted(download_id, title)
        logger.info(
            "[%s] submitted import for downgrade %s (%d file(s)); awaiting import",
            adapter.app,
            title,
            len(importable),
        )
        return True

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(seconds)

    # ----- main loop -----

    def run(self) -> None:
        if not self.contexts:
            logger.info("[optimizer] no configured apps; worker exiting")
            return
        logger.info("[optimizer] worker started for apps=%s", list(self.contexts))

        while not self._stop.is_set():
            progressed = False
            for ctx in self.contexts.values():
                if self._stop.is_set():
                    break
                if self._process_app_once(ctx):
                    progressed = True
                    self._sleep(self.opt.process_interval_seconds)

            if not progressed:
                # Nothing actionable (queue full or pool exhausted): wait one short tick.
                self._sleep(self.opt.process_interval_seconds)

    def _process_app_once(self, ctx: _AppContext, _active: bool | None = None) -> bool:
        """Do at most one unit of work for an app. Returns True if an item was processed.

        `_active` overrides the schedule check (for tests); omit to use real local time."""
        now = datetime.now(UTC)
        adapter = ctx.adapter
        active = self._in_active_hours() if _active is None else _active

        if active != self._schedule_active:
            self._schedule_active = active
            if active:
                logger.info("[optimizer] entered active hours; resuming evaluation")
            else:
                logger.info(
                    "[optimizer] outside active hours; skipping evaluation (queue imports continue)"
                )

        # List refresh and pool rebuild only happen inside active hours.
        if active and ctx.needs_refresh(now, self.opt.list_refresh_minutes):
            self._refresh(ctx, now)
            ctx.pool = []  # force rebuild below

        # Auto-import stuck downgrades always runs so the queue drains regardless of schedule.
        self._handle_queue_imports(ctx)

        if not active:
            return False

        # One queue fetch serves both the global gate and the per-item skip. The gate's count
        # optionally filters out items already past download (waiting for or doing import) —
        # those don't consume bandwidth and shouldn't block new picks.
        records = adapter.queue_items()
        queue_ids = {qid for r in records if (qid := adapter.queue_item_id(r)) is not None}
        if ctx.app_cfg.ignore_completed_in_queue:
            queue_count = sum(1 for r in records if adapter.is_queue_item_active(r))
        else:
            queue_count = len(records)
        import_count = sum(1 for r in records if adapter.is_queue_item_pending_import(r))

        # Resolve finished grabs (in_flight -> satisfied/open) before building the pool, so a
        # resolved item is correctly included/excluded this pass. No indexer call.
        self._reconcile_in_flight(ctx, queue_ids, now)

        if not ctx.pool:
            self._build_pool(ctx)
        if not ctx.pool:
            return False

        # Import backlog gate: when too many completed downloads are waiting to import, stop
        # grabbing so the backlog (which _handle_queue_imports above keeps draining) clears
        # first. Skipping the grab also skips the slow release search below, so the loop spins
        # back to the next import sooner.
        if import_count > self.opt.import_max:
            logger.debug(
                "[%s] %d imports pending > max %d; pausing grabs to drain imports",
                adapter.app,
                import_count,
                self.opt.import_max,
            )
            return False

        if queue_count > self.opt.queue_max:
            logger.debug(
                "[%s] queue %d > max %d; waiting", adapter.app, queue_count, self.opt.queue_max
            )
            return False

        # Consume from the front: order_pool returns the pool in processing order (index 0 is
        # the item the pick_order wants first, e.g. the biggest file for size_desc).
        item_id = ctx.pool.pop(0)
        if item_id in queue_ids:
            return False  # already downloading; skip and move on

        ctx.evaluated.add(item_id)  # don't re-pick within this refresh cycle
        try:
            self._process_one(ctx, item_id)
        except ArrTimeout as e:
            # Expected for slow interactive indexer searches: the item is already in
            # `evaluated`, so it's simply re-evaluated on the next full pass. Not a failure —
            # a concise warning, not an ERROR traceback.
            logger.warning("[%s] id=%d: %s; will retry on a later pass", adapter.app, item_id, e)
        except Exception:
            logger.exception("[%s] failed to process id=%d", adapter.app, item_id)
        return True


def run_optimizer(config: Config, state: StateManager) -> OptimizerWorker:
    """Construct and run the worker (blocking). Returns the worker (for tests/stop)."""
    worker = OptimizerWorker(config, state)
    worker.run()
    return worker


# ----- queue classification helpers -----


def _is_score_regression(record: dict) -> bool:
    """True iff a queue record looks like a completed download stuck purely on
    score-regression rejection. Conservative: requires (a) status=completed, (b) a state
    that signals the importer has touched it (importPending or importBlocked), and (c) at
    least one statusMessage containing the score-regression marker."""
    if (record.get("status") or "").lower() != "completed":
        return False
    if record.get("trackedDownloadState") not in {"importPending", "importBlocked"}:
        return False
    for sm in record.get("statusMessages") or []:
        for msg in sm.get("messages") or []:
            if _SCORE_REGRESSION_MARKER in (msg or "").lower():
                return True
    return False


def _is_importable_downgrade(candidate: dict) -> bool:
    """True iff a manualimport candidate has no rejections, or only score-regression
    rejections. Any other rejection reason (Sample, executable, mismatch, MediaInfo, etc.)
    blocks the auto-import — those need a deliberate human decision."""
    rejections = candidate.get("rejections") or []
    if not rejections:
        return True
    return all(_SCORE_REGRESSION_MARKER in (rj.get("reason") or "").lower() for rj in rejections)
