from optimizarr.arr import RadarrApi, SonarrApi, build_client, max_allowed_resolution
from optimizarr.arr.base import RELEASE_SEARCH_TIMEOUT_SEC
from optimizarr.config import Connection
from optimizarr.http import ArrClient


def test_max_allowed_resolution_nested():
    items = [
        {"allowed": False, "quality": {"resolution": 480}},
        {
            "allowed": True,
            "items": [
                {"allowed": True, "quality": {"resolution": 1080}},
                {"allowed": True, "quality": {"resolution": 2160}},
            ],
        },
    ]
    assert max_allowed_resolution(items) == 2160


def _conn(name="radarr"):
    return Connection(name=name, url="http://x", api_key="k")


def test_build_client_picks_app():
    assert isinstance(build_client("radarr", _conn()), RadarrApi)
    assert isinstance(build_client("sonarr", _conn("sonarr")), SonarrApi)


def test_radarr_accessors():
    api = RadarrApi(_conn())
    movie = {
        "id": 7,
        "monitored": True,
        "hasFile": True,
        "movieFile": {"id": 99, "dateAdded": "2026-01-01T00:00:00Z", "qualityCutoffNotMet": False},
        "digitalRelease": "2025-12-01T00:00:00Z",
    }
    assert api.item_id(movie) == 7
    assert api.monitored(movie) is True
    assert api.has_file(movie) is True
    assert api.current_file_id(movie) == 99
    assert api.cutoff_met(movie) is True  # qualityCutoffNotMet=False -> cutoff met
    assert api.reference_date(movie, "digitalRelease") == "2025-12-01T00:00:00Z"
    assert api.reference_date(movie, "dateAdded") == "2026-01-01T00:00:00Z"


def test_radarr_cutoff_not_met_defaults_true():
    api = RadarrApi(_conn())
    assert api.cutoff_met({"movieFile": {}}) is False  # missing -> qualityCutoffNotMet True
    assert api.cutoff_met({}) is False  # no file -> not met


def test_sonarr_accessors_use_embedded_episode_file():
    api = SonarrApi(_conn("sonarr"))
    ep = {
        "id": 3,
        "monitored": False,
        "hasFile": True,
        "episodeFile": {"id": 42, "dateAdded": "2026-02-02T00:00:00Z", "qualityCutoffNotMet": True},
        "airDateUtc": "2025-01-01T00:00:00Z",
    }
    assert api.item_id(ep) == 3
    assert api.monitored(ep) is False
    assert api.current_file_id(ep) == 42
    assert api.cutoff_met(ep) is False  # qualityCutoffNotMet True -> not met
    assert api.reference_date(ep, "airDateUtc") == "2025-01-01T00:00:00Z"
    assert api.reference_date(ep, "dateAdded") == "2026-02-02T00:00:00Z"


def test_sonarr_current_file_id_fallback():
    api = SonarrApi(_conn("sonarr"))
    assert api.current_file_id({"episodeFileId": 11}) == 11  # no episodeFile, fallback field


def test_is_queue_item_pending_import():
    f = RadarrApi.is_queue_item_pending_import
    assert f({"trackedDownloadState": "importPending"}) is True
    assert f({"trackedDownloadState": "importing"}) is True
    assert f({"trackedDownloadState": "importBlocked"}) is False  # manual-only, excluded
    assert f({"trackedDownloadState": "downloading"}) is False
    assert f({"trackedDownloadState": "imported"}) is False
    assert f({}) is False


class _RecordingClient(ArrClient):
    """Captures the GET/POST calls an adapter makes (path, timeout, retry)."""

    def __init__(self):
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []

    def post(self, path, body, *, timeout=None, retry=True):
        self.posts.append((path, body, timeout, retry))
        return None

    def get(self, path, *, timeout=None, retry=True):
        self.gets.append((path, timeout, retry))
        return []


def test_radarr_manual_import_posts_command_with_movie_id():
    api = RadarrApi(_conn())
    client = _RecordingClient()
    api.client = client
    candidate = {
        "id": 607648377,
        "path": "/downloads/Bambi.2.mkv",
        "movie": {"id": 76, "title": "Bambi II"},
        "quality": {"quality": {"name": "Bluray-1080p"}},
        "languages": [{"id": 1, "name": "English"}],
        "releaseGroup": "RSG",
        "downloadId": "abc123",
        "indexerFlags": 0,
        "rejections": [{"reason": "Not a Custom Format upgrade for existing movie file(s)."}],
    }
    api.manual_import([candidate], import_mode="auto", timeout=300, retry=False)

    (path, body, timeout, retry) = client.posts[0]
    # Must hit the command endpoint (the bare /manualimport POST only reprocesses).
    assert path == "/api/v3/command"
    assert (timeout, retry) == (300, False)
    assert body["name"] == "ManualImport"
    assert body["importMode"] == "auto"
    file = body["files"][0]
    # The owning movieId must be lifted out of the nested `movie` object — sending the
    # candidate verbatim is what made Radarr reject it with "Movie with ID 0".
    assert file["movieId"] == 76
    assert file["path"] == "/downloads/Bambi.2.mkv"
    assert file["releaseGroup"] == "RSG"
    assert file["downloadId"] == "abc123"


def test_sonarr_manual_import_posts_command_with_series_and_episode_ids():
    api = SonarrApi(_conn("sonarr"))
    client = _RecordingClient()
    api.client = client
    candidate = {
        "path": "/downloads/Show.S01E01.mkv",
        "series": {"id": 12, "title": "Show"},
        "episodes": [{"id": 501}, {"id": 502}],
        "quality": {"quality": {"name": "WEBDL-1080p"}},
        "languages": [{"id": 1, "name": "English"}],
        "releaseGroup": "GRP",
        "downloadId": "def456",
        "indexerFlags": 0,
    }
    api.manual_import([candidate])

    (path, body, _, _) = client.posts[0]
    assert path == "/api/v3/command"
    file = body["files"][0]
    assert file["seriesId"] == 12
    assert file["episodeIds"] == [501, 502]
    assert file["downloadId"] == "def456"


def test_radarr_releases_uses_generous_search_timeout():
    # The interactive indexer search routinely runs 45-100s; it must not use the 30s default.
    api = RadarrApi(_conn())
    client = _RecordingClient()
    api.client = client
    api.releases({"id": 422})

    (path, timeout, _) = client.gets[0]
    assert path == "/api/v3/release?movieId=422"
    assert timeout == RELEASE_SEARCH_TIMEOUT_SEC


def test_sonarr_releases_uses_generous_search_timeout():
    api = SonarrApi(_conn("sonarr"))
    client = _RecordingClient()
    api.client = client
    api.releases({"id": 7})

    (path, timeout, _) = client.gets[0]
    assert path == "/api/v3/release?episodeId=7"
    assert timeout == RELEASE_SEARCH_TIMEOUT_SEC


class _HistoryClient(_RecordingClient):
    """A recording client whose GET returns a canned grabbed-history list."""

    def __init__(self, history):
        super().__init__()
        self._history = history

    def get(self, path, *, timeout=None, retry=True):
        self.gets.append((path, timeout, retry))
        return self._history


def test_radarr_blocklist_grab_marks_matching_history_failed():
    api = RadarrApi(_conn())
    api.client = _HistoryClient(
        [{"id": 10, "data": {"guid": "other"}}, {"id": 11, "data": {"guid": "target"}}]
    )
    assert api.blocklist_grab({"id": 7}, "target") is True
    # Queried this movie's grabbed history, then marked the matching record failed (blocklist).
    assert any("history/movie?movieId=7" in g[0] and "eventType=1" in g[0] for g in api.client.gets)
    assert api.client.posts[0][0] == "/api/v3/history/failed/11"


def test_blocklist_grab_returns_false_and_posts_nothing_when_no_match():
    api = RadarrApi(_conn())
    api.client = _HistoryClient([{"id": 10, "data": {"guid": "other"}}])
    assert api.blocklist_grab({"id": 7}, "target") is False
    assert api.client.posts == []
