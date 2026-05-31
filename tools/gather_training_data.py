"""Gather a release-candidate training set from a LIVE deployment, for tuning the size model.

Read-only. Picks a random sample of library items, runs a real interactive indexer search for
each, and records the RAW per-release data we extract today, keyed by the item's target quality
profile (which drives scoring). It deliberately stores NO decision, closeness, gate result, or
cutoff: those are exactly what the dataset is meant to help retune, so baking them in would be
circular. It NEVER grabs, imports, unmonitors, or writes state.

Output is JSON Lines (one item per line) so a long run is crash-safe and resumable: re-run with
--resume to skip items already in the file. The sample is seeded (default --seed 0) so the same
N items are chosen across resumes.

Run (from the repo root):

    uv run --env-file .env python tools/gather_training_data.py --config config.toml --count 100

Then derive stats with tools/training_stats.py.

Options:
    --config PATH   config.toml (default: config.toml). Connection/secrets come from env (.env).
    --app NAME      radarr or sonarr (default: radarr). Must be enabled in the environment.
    --count N       items to sample (default: 100). Each is a live, rate-limited indexer search.
    --sleep SECONDS delay between items (default: 3.0), to stay gentle on indexers.
    --seed N        sample seed (default: 0). Fixed so --resume keeps the same sample.
    --out PATH      output JSONL (default: reports/training_data_<app>.jsonl).
    --resume        append, skipping items already present in --out (for restarting a long run).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from optimizarr.arr import ArrApi, build_client
from optimizarr.config import load_config
from optimizarr.features.optimizer.topsis import GB, Topsis, _release_gbh, _release_resolution


def _release_row(r: dict, runtime_h: float) -> dict:
    """RAW extracted fields for one release. No normalization, score-gap, gate, or closeness."""
    q = ((r.get("quality") or {}).get("quality")) or {}
    size = r.get("size", 0) or 0
    return {
        "title": r.get("title"),
        "indexer": r.get("indexer"),
        "protocol": r.get("protocol"),
        "score": r.get("customFormatScore"),
        "custom_formats": [c.get("name") for c in (r.get("customFormats") or []) if c.get("name")],
        "quality_name": q.get("name"),
        "quality_source": q.get("source"),
        "resolution": _release_resolution(r),
        "size_bytes": size,
        "size_gb": round(size / GB, 4),
        "gbh": round(_release_gbh(r, runtime_h), 4),
        "seeders": r.get("seeders"),
        "leechers": r.get("leechers"),
        "age_hours": r.get("ageHours"),
        "languages": [lang.get("name") for lang in (r.get("languages") or []) if lang.get("name")],
        "release_group": r.get("releaseGroup"),
        "rejections": r.get("rejections") or [],
        "temporarily_rejected": bool(r.get("temporarilyRejected")),
    }


def _current_row(topsis: Topsis, current_file: dict | None, runtime_h: float) -> dict | None:
    """RAW fields for the existing library file (the baseline for size growth/shrink)."""
    if not current_file:
        return None
    size = current_file.get("size", 0) or 0
    size_gb = size / GB
    gbh = (size_gb / runtime_h) if runtime_h and runtime_h > 0 else 0.0
    return {
        "score": current_file.get("customFormatScore"),
        "resolution": topsis._current_resolution(current_file),
        "size_bytes": size,
        "size_gb": round(size_gb, 4),
        "gbh": round(gbh, 4),
    }


def _existing_ids(out: Path) -> set[int]:
    if not out.exists():
        return set()
    ids: set[int] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def collect(api: ArrApi, item: dict, topsis: Topsis) -> dict:
    runtime_h = api.runtime_h(item)
    profile_name, target_res = api.profile_for(item)
    releases = api.releases(item)
    return {
        "app": api.app,
        "id": api.item_id(item),
        "external_id": item.get("tmdbId") or item.get("tvdbId"),
        "title": api.label(item),
        "runtime_h": round(runtime_h, 4),
        # The target profile drives scoring, so it is part of every record.
        "profile": {"name": profile_name, "target_resolution": target_res},
        "current_file": _current_row(topsis, api.current_file(item), runtime_h),
        "release_count": len(releases),
        "releases": [_release_row(r, runtime_h) for r in releases],
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--app", default="radarr", choices=["radarr", "sonarr"])
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    conn = getattr(config, args.app)
    if conn is None:
        raise SystemExit(f"{args.app} is not configured in the environment")
    topsis = Topsis(config.optimizer.topsis)
    api = build_client(args.app, conn)
    api.refresh_profiles()

    out = args.out or Path("reports") / f"training_data_{args.app}.jsonl"
    out.parent.mkdir(exist_ok=True)

    items = [it for it in api.list_items() if api.has_file(it)]
    random.Random(args.seed).shuffle(items)
    sample = items[: args.count]

    done = _existing_ids(out) if args.resume else set()
    todo = [it for it in sample if api.item_id(it) not in done]
    if args.resume and done:
        print(f"resuming: {len(done)} already collected, {len(todo)} to go")

    mode = "a" if args.resume else "w"
    written = 0
    with out.open(mode, encoding="utf-8") as f:
        for i, item in enumerate(todo, 1):
            label = api.label(item)
            print(f"[{args.app}] ({i}/{len(todo)}) searching releases for {label} ...")
            try:
                record = collect(api, item, topsis)
            except Exception as e:  # noqa: BLE001 - one bad item shouldn't kill a long run
                print(f"  [warn] {label}: {e}; skipping")
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # crash-safe: each item is durable as soon as it's fetched
            written += 1
            if i < len(todo):
                time.sleep(args.sleep)

    print(f"\nWrote {written} item(s) to {out} (total now {len(done) + written}).")


if __name__ == "__main__":
    main()
