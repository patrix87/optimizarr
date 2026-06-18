"""RadarrApi: Radarr (movies) implementation of ArrApi."""

from optimizarr.arr.base import RELEASE_SEARCH_TIMEOUT_SEC, ArrApi


class RadarrApi(ArrApi):
    app = "radarr"
    _queue_id_field = "movieId"

    def _embedded_file(self, item: dict) -> dict | None:
        return item.get("movieFile")

    def _manual_import_file(self, candidate: dict) -> dict:
        return {
            "path": candidate.get("path"),
            "movieId": (candidate.get("movie") or {}).get("id"),
            "quality": candidate.get("quality"),
            "languages": candidate.get("languages"),
            "releaseGroup": candidate.get("releaseGroup"),
            "downloadId": candidate.get("downloadId"),
            "indexerFlags": candidate.get("indexerFlags"),
        }

    def list_items(self) -> list[dict]:
        return self.client.get("/api/v3/movie") or []

    def label(self, item: dict) -> str:
        return f"{item.get('title')} ({item.get('year')})"

    def runtime_h(self, item: dict) -> float:
        return (item.get("runtime") or 0) / 60

    def release_date(self, item: dict) -> str:
        # Prefer the digital release (the canonical date the age gate keys on), then physical,
        # then theatrical, so a movie still sorts even before a digital date is known.
        return (
            item.get("digitalRelease") or item.get("physicalRelease") or item.get("inCinemas") or ""
        )

    def profile_for(self, item: dict) -> tuple[str | None, int | None]:
        return self._profile(item.get("qualityProfileId"))

    def current_file(self, item: dict) -> dict | None:
        file_id = self.current_file_id(item)
        if not file_id:
            return None
        return self.client.get(f"/api/v3/movieFile/{file_id}")

    def releases(self, item: dict) -> list[dict]:
        return (
            self.client.get(
                f"/api/v3/release?movieId={item['id']}", timeout=RELEASE_SEARCH_TIMEOUT_SEC
            )
            or []
        )

    def set_monitored(self, item_ids: list[int], monitored: bool) -> None:
        self.client.put(
            "/api/v3/movie/editor",
            {"movieIds": item_ids, "monitored": monitored},
        )
