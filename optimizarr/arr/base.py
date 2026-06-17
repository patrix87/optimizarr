"""ArrApi: the per-app Radarr/Sonarr API surface that features depend on.

This is the abstraction (DIP): features (optimizer, unmonitor) talk to ArrApi methods,
never to raw `/api/v3/...` endpoints or to the concrete RadarrApi/SonarrApi. It owns all
knowledge of how to talk to a *arr app — listing items, fetching files/releases, grabbing,
reading the queue, and setting monitored state — plus normalized accessors so feature
logic can be written once for both movies and episodes.

Item *selection* (age gate, monitored, cutoff) lives in the features, not here: this layer
lists raw items and exposes accessors; each feature applies its own predicate.
"""

from urllib.parse import quote

from optimizarr.config import Connection
from optimizarr.http import ArrClient

# trackedDownloadState values that mean the download finished and is no longer consuming
# bandwidth — used to filter "active" queue items vs. ones stuck waiting for/in import.
QUEUE_INACTIVE_STATES = {"importPending", "importing", "imported", "importBlocked"}

# trackedDownloadState values for the drainable import backlog: a completed download waiting
# to be imported (importPending, includes the score-regression downgrades) or one being
# imported right now (importing). Deliberately excludes importBlocked (manual-only).
QUEUE_PENDING_IMPORT_STATES = {"importPending", "importing"}

# GET /api/v3/release?movieId=/episodeId= runs a *live* interactive indexer search and only
# returns once every indexer has answered or timed out — measured at 45-100s in the field,
# well past the 30s default. Give it a generous ceiling so a normal slow search isn't logged
# as a failure; a search that truly exceeds this still fails cleanly and the item is retried.
RELEASE_SEARCH_TIMEOUT_SEC = 240


def max_allowed_resolution(profile_items: list[dict]) -> int:
    """Max `resolution` over allowed entries in a Radarr/Sonarr quality profile's
    items[] (with nested items[]). Returns 0 if nothing is allowed."""
    best = 0
    for it in profile_items or []:
        q = it.get("quality") or {}
        if it.get("allowed") and q.get("resolution"):
            best = max(best, q["resolution"])
        for sub in it.get("items") or []:
            sq = sub.get("quality") or {}
            if sub.get("allowed") and sq.get("resolution"):
                best = max(best, sq["resolution"])
    return best


class ArrApi:
    app: str
    _queue_id_field: str  # "movieId" / "episodeId"

    def __init__(self, conn: Connection):
        self.client = ArrClient(conn.url, conn.api_key)
        self._profiles: dict[int, tuple[str, int]] = {}

    # ----- quality profiles -----

    def refresh_profiles(self) -> None:
        profiles = self.client.get("/api/v3/qualityprofile") or []
        self._profiles = {
            p["id"]: (p.get("name", str(p["id"])), max_allowed_resolution(p.get("items")))
            for p in profiles
        }

    def _profile(self, profile_id: int | None) -> tuple[str | None, int | None]:
        if profile_id is None:
            return None, None
        name, target = self._profiles.get(profile_id, (None, None))
        return name, (target or None)

    # ----- normalized accessors (concrete; share across apps) -----

    def item_id(self, item: dict) -> int:
        return item["id"]

    def monitored(self, item: dict) -> bool:
        return item.get("monitored", False)

    def has_file(self, item: dict) -> bool:
        return item.get("hasFile", False)

    def reference_date(self, item: dict, release_type: str) -> str | None:
        if release_type == "dateAdded":
            return (self._embedded_file(item) or {}).get("dateAdded")
        return item.get(release_type)

    def cutoff_met(self, item: dict) -> bool:
        return not (self._embedded_file(item) or {}).get("qualityCutoffNotMet", True)

    def current_file_id(self, item: dict) -> int | None:
        return (self._embedded_file(item) or {}).get("id")

    # Ordering keys for the optimizer's pick_order. file_size / date_added read the embedded
    # file (always present in the listing, since the pool is gated on hasFile); release_date is
    # app-specific (Radarr movies vs. Sonarr episodes carry different date fields). All three
    # are cheap reads off the already-fetched item (no extra HTTP). Missing values fall back to
    # 0 / "" so they sort first ascending (last descending).
    def file_size(self, item: dict) -> int:
        return (self._embedded_file(item) or {}).get("size") or 0

    def date_added(self, item: dict) -> str:
        return (self._embedded_file(item) or {}).get("dateAdded") or ""

    def release_date(self, item: dict) -> str:
        raise NotImplementedError

    # ----- actions (concrete) -----

    def grab(self, release: dict) -> None:
        self.client.post(
            "/api/v3/release",
            {"guid": release["guid"], "indexerId": release.get("indexerId")},
        )

    def blocklist_grab(self, item: dict, guid: str) -> bool:
        """Blocklist the grab whose release `guid` matches, so the release is never grabbed again.

        Finds the item's 'grabbed' history record carrying this guid and marks it failed, which adds
        the release to the *arr blocklist. Future searches then return it with a 'blocklisted'
        rejection, which the optimizer's inclusion filter drops. Returns True if a matching grab
        record was found and the failed/blocklist call was posted."""
        for rec in self._grabbed_history(item):
            if (rec.get("data") or {}).get("guid") == guid:
                self.client.post(f"/api/v3/history/failed/{rec['id']}", {})
                return True
        return False

    def _grabbed_history(self, item: dict) -> list[dict]:
        """The item's 'grabbed' (eventType=1) history records (app-specific endpoint)."""
        raise NotImplementedError

    def queue_items(self) -> list[dict]:
        """Full queue records (status, trackedDownloadState, statusMessages, etc.) — the
        worker computes both the queue_max count and the in-progress item-id set from this."""
        resp = self.client.get("/api/v3/queue?page=1&pageSize=1000") or {}
        return resp.get("records", []) or []

    def queue_item_id(self, record: dict) -> int | None:
        """movieId / episodeId for a queue record (None if missing)."""
        return record.get(self._queue_id_field)

    @staticmethod
    def is_queue_item_active(record: dict) -> bool:
        """True when the record represents an item still downloading or queued — i.e. NOT
        completed and waiting for / already past import. Used by `ignore_completed_in_queue`."""
        if (record.get("status") or "").lower() == "completed":
            return False
        return record.get("trackedDownloadState") not in QUEUE_INACTIVE_STATES

    @staticmethod
    def is_queue_item_pending_import(record: dict) -> bool:
        """True when a completed download is waiting to import (importPending) or actively
        importing (importing): the drainable import backlog the import gate watches.
        importBlocked is excluded on purpose: it needs manual action and never auto-clears."""
        return record.get("trackedDownloadState") in QUEUE_PENDING_IMPORT_STATES

    def manual_import_candidates(
        self,
        download_id: str,
        *,
        timeout: int | None = None,
        retry: bool = True,
    ) -> list[dict]:
        """List importable files for a downloadId (GET /api/v3/manualimport?downloadId=).
        Each candidate carries proposed movie/episode, quality, customFormats, rejections.
        Pass a longer `timeout` and `retry=False` for the worker's auto-import sweep —
        this endpoint runs MediaInfo and can take minutes on first call per downloadId."""
        return (
            self.client.get(
                f"/api/v3/manualimport?downloadId={quote(download_id)}",
                timeout=timeout,
                retry=retry,
            )
            or []
        )

    def manual_import(
        self,
        items: list[dict],
        import_mode: str = "auto",
        *,
        timeout: int | None = None,
        retry: bool = True,
    ) -> None:
        """Force-import the given manualimport candidates via the ManualImport command.

        Do NOT POST to /api/v3/manualimport for this: that endpoint only *reprocesses*
        candidates (re-evaluates rejections) and never imports. The actual import is the
        ManualImport command, and it needs each file's owning id — movieId for Radarr,
        seriesId + episodeIds for Sonarr. The GET candidate carries those only as nested
        `movie`/`series`/`episodes` objects, so each app maps them in `_manual_import_file`.
        Posting a candidate verbatim sends no top-level id, which *arr reads as 0 and
        rejects with 'Movie/Series with ID 0 does not exist'."""
        body = {
            "name": "ManualImport",
            "importMode": import_mode,
            "files": [self._manual_import_file(it) for it in items],
        }
        self.client.post("/api/v3/command", body, timeout=timeout, retry=retry)

    # ----- app-specific (subclasses implement) -----

    def _manual_import_file(self, candidate: dict) -> dict:
        """Map a manualimport GET candidate to a ManualImport command `files[]` entry,
        attaching the app-specific owning id(s) the import requires."""
        raise NotImplementedError

    def _embedded_file(self, item: dict) -> dict | None:
        """The file object carried on the item itself (movieFile / episodeFile)."""
        raise NotImplementedError

    def list_items(self) -> list[dict]:
        raise NotImplementedError

    def label(self, item: dict) -> str:
        raise NotImplementedError

    def runtime_h(self, item: dict) -> float:
        raise NotImplementedError

    def profile_for(self, item: dict) -> tuple[str | None, int | None]:
        raise NotImplementedError

    def current_file(self, item: dict) -> dict | None:
        raise NotImplementedError

    def releases(self, item: dict) -> list[dict]:
        raise NotImplementedError

    def set_monitored(self, item_ids: list[int], monitored: bool) -> None:
        raise NotImplementedError
