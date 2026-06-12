"""Per-item optimizer state, persisted to JSON.

Keyed by app ("radarr"/"sonarr") then item id (movie id / episode id). Each entry records the
*profile* it pertains to (the optimal pick depends on the profile). The lifecycle:

  unprocessed              -> not in state: eligible to be evaluated
  satisfied                -> the current (imported) file is optimal for `profile`; permanently
                              dropped from the pool. Active again ONLY if the profile changes or
                              the file is removed. There is no time-based re-activation.
  insufficient_candidates  -> too few candidate releases to trust a comparison. While `tries` is
                              below the configured max the item stays active and is retried each
                              pass; once it exhausts its tries a `retry_after` cooldown is set and
                              the item is left alone until then, after which it goes active again
                              (the counter resets on the next attempt).

A grab is never recorded. If it succeeds, the next evaluation HOLDs against the imported file and
marks the item satisfied; if it fails, the item was never satisfied so it stays in the pool and is
retried later (the failed release having been blocklisted by Radarr/Sonarr). Downloads in progress
are detected live from the queue, not from state, so a restart recovers with no reconciliation.
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

logger = logging.getLogger("optimizarr")

SATISFIED = "satisfied"
INSUFFICIENT = "insufficient_candidates"


@dataclass
class StateEntry:
    status: str
    updated_at: str
    profile: str | None = None  # the profile the entry pertains to (invalidates on change)
    tries: int = 0  # INSUFFICIENT only: consecutive too-few-candidate attempts
    retry_after: str | None = None  # INSUFFICIENT only: ISO time the cooldown ends (None = none)


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
                    tries=entry.get("tries", 0),
                    retry_after=entry.get("retry_after"),
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

    def is_active(
        self,
        app: str,
        item_id: int,
        profile: str | None,
        has_file: bool,
        now: datetime | None = None,
    ) -> bool:
        """Whether an item is worth evaluating now.

        - Unprocessed -> active.
        - Satisfied -> inactive while it has a file and its profile is unchanged (one-and-done,
          no time-based re-activation). A profile change or a removed file re-opens it.
        - Insufficient candidates -> a profile change or removed file re-opens it immediately;
          otherwise it stays active while still counting tries, and is inactive only during the
          retry_after cooldown (once that time passes it goes active again)."""
        entry = self.get(app, item_id)
        if entry is None:
            return True
        if not has_file or entry.profile != profile:
            return True
        if entry.status == SATISFIED:
            return False
        if entry.status == INSUFFICIENT and entry.retry_after is not None:
            now = now or datetime.now(UTC)
            return now >= datetime.fromisoformat(entry.retry_after)
        return True

    def mark_satisfied(self, app: str, item_id: int, profile: str | None) -> None:
        with self._lock:
            self._data.setdefault(app, {})[str(item_id)] = StateEntry(
                status=SATISFIED, updated_at=_now_iso(), profile=profile
            )
            self._save_locked()

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
            continuing = (
                entry is not None
                and entry.status == INSUFFICIENT
                and entry.retry_after is None
                and entry.profile == profile
            )
            tries = (entry.tries if continuing else 0) + 1
            retry_after = None
            if tries >= max_tries:
                retry_after = (datetime.now(UTC) + timedelta(days=cooldown_days)).isoformat()
            new = StateEntry(
                status=INSUFFICIENT,
                updated_at=_now_iso(),
                profile=profile,
                tries=tries,
                retry_after=retry_after,
            )
            self._data.setdefault(app, {})[str(item_id)] = new
            self._save_locked()
            return new
