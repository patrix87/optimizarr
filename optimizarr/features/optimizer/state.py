"""Per-item optimizer state, persisted to JSON.

Keyed by app ("radarr"/"sonarr") then item id (movie id / episode id). Each entry records the
*profile* it pertains to (the optimal pick depends on the profile) plus `tried_guids`: every release
already grabbed for this item, which is NEVER grabbed again (the anti-oscillation core). Lifecycle:

  unprocessed              -> not in state: eligible to be evaluated
  open                     -> eligible, but carries grab memory (`tried_guids`) from prior attempts
  in_flight                -> a release was grabbed and we are awaiting its download/import outcome;
                              not evaluated (no indexer search) until it resolves
  satisfied                -> the current file is optimal for `profile` (nothing untried beats it,
                              or a grab imported successfully). Dropped from the pool; active again
                              ONLY if the profile changes or the file is removed.
  insufficient_candidates  -> too few candidate releases to trust a comparison. While `tries` is
                              below the configured max the item stays active and is retried each
                              pass; once it exhausts its tries a `retry_after` cooldown is set.
  parked                   -> grabbed `grab.max_tries` distinct releases without satisfying; rested
                              until `retry_after`, then re-evaluated fresh (tried memory cleared).

Resolution of an in_flight grab is computed by the worker from the live queue + the item's file id
(a changed file id means the grab imported -> satisfy, with no extra indexer query; an unchanged
file means it failed -> try the next-best). All writes are atomic and lock-guarded, so a crash never
yields a torn file and reconciliation re-derives in_flight outcomes idempotently after a restart.
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger("optimizarr")

SATISFIED = "satisfied"
INSUFFICIENT = "insufficient_candidates"
OPEN = "open"
IN_FLIGHT = "in_flight"
PARKED = "parked"


@dataclass
class StateEntry:
    status: str
    updated_at: str
    profile: str | None = None  # the profile the entry pertains to (invalidates on change)
    tried_guids: list[str] = field(default_factory=list)  # releases grabbed; never grab again
    tries: int = 0  # INSUFFICIENT only: consecutive too-few-candidate attempts
    retry_after: str | None = None  # INSUFFICIENT/PARKED: ISO time the cooldown ends (None = none)
    grabbed_guid: str | None = None  # IN_FLIGHT only: the release currently downloading
    grabbed_at: str | None = None  # IN_FLIGHT only: ISO grab time (settle window anchor)
    grabbed_file_id: int | None = (
        None  # IN_FLIGHT only: current file id at grab time (import probe)
    )
    grabbed_score: int | None = None  # IN_FLIGHT only: advertised customFormatScore at grab time


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StateManager:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, StateEntry]] = {"radarr": {}, "sonarr": {}}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[state] could not read %s (%s); starting empty", self.path, e)
            return
        for app, items in raw.items():
            bucket = self._data.setdefault(app, {})
            for item_id, entry in items.items():
                bucket[str(item_id)] = StateEntry(
                    status=entry["status"],
                    updated_at=entry["updated_at"],
                    profile=entry.get("profile"),
                    tried_guids=list(entry.get("tried_guids", [])),
                    tries=entry.get("tries", 0),
                    retry_after=entry.get("retry_after"),
                    grabbed_guid=entry.get("grabbed_guid"),
                    grabbed_at=entry.get("grabbed_at"),
                    grabbed_file_id=entry.get("grabbed_file_id"),
                    grabbed_score=entry.get("grabbed_score"),
                )

    def _save_locked(self) -> None:
        serializable = {
            app: {item_id: asdict(entry) for item_id, entry in items.items()}
            for app, items in self._data.items()
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(serializable, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get(self, app: str, item_id: int) -> StateEntry | None:
        return self._data.get(app, {}).get(str(item_id))

    def in_flight_items(self, app: str) -> list[tuple[int, StateEntry]]:
        """Snapshot of (item_id, entry) for every in-flight grab in `app`, for the worker to
        reconcile against the live queue. A list copy so the caller can resolve entries while
        iterating without mutating the dict under it."""
        with self._lock:
            return [
                (int(iid), entry)
                for iid, entry in self._data.get(app, {}).items()
                if entry.status == IN_FLIGHT
            ]

    def is_active(
        self,
        app: str,
        item_id: int,
        profile: str | None,
        has_file: bool,
        now: datetime | None = None,
    ) -> bool:
        """Whether an item is worth evaluating now.

        - Unprocessed or open -> active.
        - A profile change or a removed file re-opens any entry immediately (the stored decision no
          longer applies).
        - Satisfied -> inactive (one-and-done; no time-based re-activation).
        - In flight -> inactive: a grab is downloading/importing. The worker's reconciliation moves
          it to satisfied/open once it resolves, so it is never searched while waiting.
        - Insufficient (cooldown) / parked -> inactive until retry_after passes; insufficient while
          still counting tries (no retry_after) stays active."""
        entry = self.get(app, item_id)
        if entry is None:
            return True
        if not has_file or entry.profile != profile:
            return True
        if entry.status in (SATISFIED, IN_FLIGHT):
            return False
        if entry.retry_after is not None:  # INSUFFICIENT cooldown or PARKED
            now = now or datetime.now(UTC)
            return now >= datetime.fromisoformat(entry.retry_after)
        return True

    def tried_guids(self, app: str, item_id: int, profile: str | None, has_file: bool) -> set[str]:
        """Releases already grabbed for this item, honored only while the stored profile matches and
        the file is present (a profile change or a removed file starts the grab memory fresh)."""
        entry = self.get(app, item_id)
        if entry is None or not has_file or entry.profile != profile:
            return set()
        return set(entry.tried_guids)

    def mark_satisfied(self, app: str, item_id: int, profile: str | None) -> None:
        with self._lock:
            prev = self._data.get(app, {}).get(str(item_id))
            tried = list(prev.tried_guids) if (prev and prev.profile == profile) else []
            self._data.setdefault(app, {})[str(item_id)] = StateEntry(
                status=SATISFIED, updated_at=_now_iso(), profile=profile, tried_guids=tried
            )
            self._save_locked()

    def record_grab(
        self,
        app: str,
        item_id: int,
        profile: str | None,
        guid: str,
        file_id: int | None,
        score: int | None,
    ) -> StateEntry:
        """Persist that we are about to grab `guid` for this item (call BEFORE the grab POST so a
        crash in between can never cause a duplicate grab). Appends the guid to tried_guids and
        records the current file id (to detect a later import by a file-id change) and the release's
        advertised customFormatScore (to detect a misadvertised release after import)."""
        with self._lock:
            prev = self._data.get(app, {}).get(str(item_id))
            tried = list(prev.tried_guids) if (prev and prev.profile == profile) else []
            if guid not in tried:
                tried.append(guid)
            new = StateEntry(
                status=IN_FLIGHT,
                updated_at=_now_iso(),
                profile=profile,
                tried_guids=tried,
                grabbed_guid=guid,
                grabbed_at=_now_iso(),
                grabbed_file_id=file_id,
                grabbed_score=score,
            )
            self._data.setdefault(app, {})[str(item_id)] = new
            self._save_locked()
            return new

    def resolve_in_flight(self, app: str, item_id: int, imported: bool) -> StateEntry:
        """Resolve an in-flight grab: imported=True -> satisfied (the grab produced a new file);
        imported=False -> open, keeping tried_guids so the failed release is never grabbed again and
        the next-best untried one is tried on the next pass."""
        with self._lock:
            prev = self._data.get(app, {}).get(str(item_id))
            tried = list(prev.tried_guids) if prev else []
            profile = prev.profile if prev else None
            new = StateEntry(
                status=SATISFIED if imported else OPEN,
                updated_at=_now_iso(),
                profile=profile,
                tried_guids=tried,
            )
            self._data.setdefault(app, {})[str(item_id)] = new
            self._save_locked()
            return new

    def park(self, app: str, item_id: int, profile: str | None, cooldown_days: int) -> StateEntry:
        """Park an item that has grabbed grab.max_tries distinct releases without satisfying. Rest
        until retry_after, then re-evaluate fresh: tried_guids is cleared so the cooldown actually
        lets it try releases again (better ones may exist later) instead of re-parking at once."""
        with self._lock:
            retry_after = (datetime.now(UTC) + timedelta(days=cooldown_days)).isoformat()
            self._data.setdefault(app, {})[str(item_id)] = StateEntry(
                status=PARKED, updated_at=_now_iso(), profile=profile, retry_after=retry_after
            )
            self._save_locked()
            return self._data[app][str(item_id)]

    def record_insufficient(
        self, app: str, item_id: int, profile: str | None, max_tries: int, cooldown_days: int
    ) -> StateEntry:
        """Record one too-few-candidates attempt and return the updated entry.

        Counting continues only from a prior INSUFFICIENT entry for the same profile that is still
        in its active (non-cooldown) phase; otherwise the counter restarts at 1 (a fresh item, a
        changed profile, or an expired cooldown begins a new cycle). On reaching max_tries the
        entry enters a cooldown_days rest (retry_after set); below that it stays active to retry."""
        with self._lock:
            entry = self._data.get(app, {}).get(str(item_id))
            same_profile = entry is not None and entry.profile == profile
            continuing = (
                entry is not None
                and same_profile
                and entry.status == INSUFFICIENT
                and entry.retry_after is None
            )
            tries = (entry.tries + 1) if (entry is not None and continuing) else 1
            tried = list(entry.tried_guids) if (entry is not None and same_profile) else []
            retry_after = None
            if tries >= max_tries:
                retry_after = (datetime.now(UTC) + timedelta(days=cooldown_days)).isoformat()
            new = StateEntry(
                status=INSUFFICIENT,
                updated_at=_now_iso(),
                profile=profile,
                tried_guids=tried,
                tries=tries,
                retry_after=retry_after,
            )
            self._data.setdefault(app, {})[str(item_id)] = new
            self._save_locked()
            return new
