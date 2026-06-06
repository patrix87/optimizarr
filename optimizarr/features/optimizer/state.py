"""Per-item optimizer state, persisted to JSON.

Keyed by app ("radarr"/"sonarr") then item id (movie id / episode id). Each satisfied entry
records the *profile* it was satisfied for. The lifecycle is one-and-done:

  unprocessed -> not in state: eligible to be evaluated
  satisfied   -> the current (imported) file is optimal for `profile`; permanently dropped from
                 the pool. It becomes active again ONLY if the profile changes or the file is
                 removed. There is no time-based re-activation; delete state.json to force a re-run.

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
from datetime import UTC, datetime

logger = logging.getLogger("optimizarr")

SATISFIED = "satisfied"


@dataclass
class StateEntry:
    status: str
    updated_at: str
    profile: str | None = None  # the profile the item was satisfied for (invalidates on change)


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

    def is_active(self, app: str, item_id: int, profile: str | None, has_file: bool) -> bool:
        """An item is active (worth evaluating) unless it is satisfied FOR ITS CURRENT PROFILE and
        still has a file. One-and-done: there is no time-based re-activation. A satisfied item
        becomes active again only if its profile changed (the optimal pick depends on the profile)
        or its file was removed (needs a fresh grab). To force a full re-run, delete state.json."""
        entry = self.get(app, item_id)
        if entry is None or entry.status != SATISFIED:
            return True
        if not has_file:
            return True
        return entry.profile != profile

    def mark_satisfied(self, app: str, item_id: int, profile: str | None) -> None:
        with self._lock:
            self._data.setdefault(app, {})[str(item_id)] = StateEntry(
                status=SATISFIED, updated_at=_now_iso(), profile=profile
            )
            self._save_locked()
