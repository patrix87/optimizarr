"""SonarrApi: Sonarr (series/episodes) implementation of ArrApi.

Items are episodes. Series metadata (title, runtime, quality profile) is fetched once per
listing and cached, since episodes reference it by seriesId.
"""

from optimizarr.arr.base import RELEASE_SEARCH_TIMEOUT_SEC, ArrApi
from optimizarr.config import Connection


class SonarrApi(ArrApi):
    app = "sonarr"
    _queue_id_field = "episodeId"

    def __init__(self, conn: Connection):
        super().__init__(conn)
        self._series_by_id: dict[int, dict] = {}

    def _embedded_file(self, item: dict) -> dict | None:
        return item.get("episodeFile")

    def _manual_import_file(self, candidate: dict) -> dict:
        return {
            "path": candidate.get("path"),
            "seriesId": (candidate.get("series") or {}).get("id"),
            "episodeIds": [e["id"] for e in (candidate.get("episodes") or []) if e.get("id")],
            "quality": candidate.get("quality"),
            "languages": candidate.get("languages"),
            "releaseGroup": candidate.get("releaseGroup"),
            "downloadId": candidate.get("downloadId"),
            "indexerFlags": candidate.get("indexerFlags"),
        }

    def _series(self, item: dict) -> dict:
        return self._series_by_id.get(item.get("seriesId") or 0, {})

    def list_items(self) -> list[dict]:
        series_list = self.client.get("/api/v3/series") or []
        self._series_by_id = {s["id"]: s for s in series_list}
        items: list[dict] = []
        for series in series_list:
            episodes = (
                self.client.get(f"/api/v3/episode?seriesId={series['id']}&includeEpisodeFile=true")
                or []
            )
            items.extend(episodes)
        return items

    def label(self, item: dict) -> str:
        title = self._series(item).get("title", "?")
        return f"{title} S{item.get('seasonNumber', 0):02d}E{item.get('episodeNumber', 0):02d}"

    def runtime_h(self, item: dict) -> float:
        return (self._series(item).get("runtime") or 0) / 60

    def release_date(self, item: dict) -> str:
        return item.get("airDateUtc") or ""

    def profile_for(self, item: dict) -> tuple[str | None, int | None]:
        return self._profile(self._series(item).get("qualityProfileId"))

    def current_file_id(self, item: dict) -> int | None:
        return (self._embedded_file(item) or {}).get("id") or item.get("episodeFileId")

    def current_file(self, item: dict) -> dict | None:
        file_id = self.current_file_id(item)
        if not file_id:
            return None
        return self.client.get(f"/api/v3/episodefile/{file_id}")

    def releases(self, item: dict) -> list[dict]:
        return (
            self.client.get(
                f"/api/v3/release?episodeId={item['id']}", timeout=RELEASE_SEARCH_TIMEOUT_SEC
            )
            or []
        )

    def set_monitored(self, item_ids: list[int], monitored: bool) -> None:
        self.client.put(
            "/api/v3/episode/monitor",
            {"episodeIds": item_ids, "monitored": monitored},
        )
