"""Visualize the candidate funnel (incl. the score-gap cutoff) against a LIVE deployment.

Read-only. For each library item it runs a real interactive indexer search, then reproduces
the optimizer pipeline stage-by-stage and prints a table of *every* candidate release with the
reason it survived or was removed, the score-gap cutoff line, and the final pick. It NEVER
grabs, imports, unmonitors, or writes state — there is no code path here that calls an action.

It drives the real config (config.toml) and connection (.env), so the numbers match what the
worker would see.

A live `/api/v3/release` search is a slow interactive indexer call and is rate-limited, so this
processes only a small random sample by default and sleeps between items. Writes a timestamped
Markdown report under ./reports/.

Run (from the repo root):

    uv run --env-file .env python tools/cutoff_viz.py --config config.toml --limit 5

Options:
    --config PATH    Path to config.toml (default: config.toml). Layered on defaults.toml, same
                     as the worker. Connection/secrets still come from the env (.env).
    --app NAME       Which app to evaluate: radarr or sonarr (default: radarr). Must be enabled
                     in the environment (its URL + API key set).
    --limit N        Max items to evaluate (default: 5). Caps EVERY path, including --tmdb/--ids
                     (an over-long id list is truncated). Each item is a live, rate-limited
                     indexer search, hence the small default.
    --tmdb ID [ID..] Evaluate these TMDB ids (Sonarr: TVDB ids) — the id in the *arr movie URL,
                     so you can paste it straight from the browser. The recommended selector.
    --ids ID [ID...] Evaluate these *internal* Radarr/Sonarr ids (the DB primary key). A separate
                     id space from --tmdb; use this only if you specifically have the internal id.
                     --tmdb and --ids are matched independently and may be combined.
    --sleep SECONDS  Delay between items (default: 3.0), to stay gentle on the indexers.
    --seed N         Seed the random sample for a reproducible run (no effect with --tmdb/--ids).

Either selector skips the random sample (but still obeys --limit) and warns on any id that is
unknown or has no downloaded file (those are skipped) instead of failing silently. Matching by
--tmdb does not call TMDB: the id is already a field on each item from the one *arr list call.

Examples:
    # 5 random movies
    uv run --env-file .env python tools/cutoff_viz.py --config config.toml

    # specific movies by TMDB id (the number in the *arr URL), no sampling
    uv run --env-file .env python tools/cutoff_viz.py --config config.toml --tmdb 38356 1858

    # a reproducible sample of 10
    uv run --env-file .env python tools/cutoff_viz.py --config config.toml --limit 10 --seed 1
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

from optimizarr.arr import ArrApi, build_client
from optimizarr.config import load_config
from optimizarr.features.optimizer.config import OptimizerAppConfig
from optimizarr.features.optimizer.topsis import (
    Topsis,
    _release_resolution,
    eligible,
)
from optimizarr.features.optimizer.transitions import classify, is_forbidden

# Statuses, best-to-worst for display ordering of the legend only.
KEPT = "candidate"
PICK = "PICK"


def _hard_reject_reason(release: dict) -> str:
    if release.get("temporarilyRejected"):
        return "temporarily rejected"
    return "; ".join(release.get("rejections") or []) or "hard reject"


def analyze(
    topsis: Topsis,
    releases: list[dict],
    runtime_h: float,
    profile_name: str | None,
    target_res: int | None,
    current_file: dict | None,
    app_cfg: OptimizerAppConfig,
) -> dict:
    """Reproduce decide()'s pipeline, labelling each release with the stage that removed it.

    Returns a dict with the resolved profile, current-file row, per-release rows (each with raw
    attrs, closeness and a status string), and the score-gap cutoff score.
    """
    cur = current_file or {}
    cur_size = cur.get("size")
    cur_score = cur.get("customFormatScore")
    resolved = topsis.resolve_profile(profile_name)

    # --- stage membership (decide()'s exact order) ---
    after_size = [
        r
        for r in releases
        if app_cfg.allow_size_increase
        or not (isinstance(cur_size, int) and cur_size > 0)
        or r.get("size", 0) <= cur_size
    ]
    after_qual = [
        r
        for r in after_size
        if app_cfg.allow_quality_downgrade
        or cur_score is None
        or (r.get("customFormatScore") or 0) >= cur_score
    ]
    after_hard = eligible(after_qual)
    after_gbh = topsis.filter_by_gbh_floor(after_hard, runtime_h)
    after_gap = topsis.filter_by_score_gap(after_gbh)  # the gap-cut survivors == "scored"

    ids = lambda lst: {id(r) for r in lst}  # noqa: E731
    id_size, id_qual = ids(after_size), ids(after_qual)
    id_hard, id_gbh, id_gap = ids(after_hard), ids(after_gbh), ids(after_gap)

    # The score-gap cutoff line: the lowest score that survived the gap-cut (everything strictly
    # below it, among gbh/hard survivors, was dropped as the score tail).
    gap_cutoff_score = (
        min((r.get("customFormatScore") or 0) for r in after_gap) if after_gap else None
    )

    cur_attrs = topsis.current_attributes(cur, runtime_h, resolved, target_res)
    cur_clo, cur_raw = topsis.closeness_for_current_file(cur, runtime_h, resolved, target_res)
    cur_nscore = cur_attrs["n_score"] if cur_attrs is not None else 0.0
    cur_gbh = cur_raw.get("gbh", 0.0) or 0.0
    cur_res = cur_raw.get("resolution", 0) or 0

    rows = []
    legal: list[tuple[dict, dict, float]] = []
    for r in releases:
        attrs = topsis.attributes_for(r, runtime_h, resolved, target_res)
        clo = topsis.closeness(attrs, resolved.weights)
        rid = id(r)
        status = KEPT
        if rid not in id_size:
            status = "drop: bigger than current (allow_size_increase=false)"
        elif rid not in id_qual:
            status = "drop: lower score than current (allow_quality_downgrade=false)"
        elif rid not in id_hard:
            status = f"drop: hard reject ({_hard_reject_reason(r)})"
        elif rid not in id_gbh:
            floor = topsis.reference_for(_release_resolution(r))[0]
            status = f"drop: below GiB/h floor ({attrs['raw']['gbh']:.2f} < {floor:.2f})"
        elif rid not in id_gap:
            if (r.get("customFormatScore") or 0) < 0:
                status = "drop: negative score"
            else:
                status = f"drop: score-gap cut (>{topsis.cfg.score_gap:.0%} below top cluster)"
        else:
            # survived to scoring -> run the transition gate vs the current file
            deltas = classify(
                cur_nscore=cur_nscore,
                cand_nscore=attrs["n_score"],
                cur_gbh=cur_gbh,
                cand_gbh=attrs["raw"]["gbh"],
                cur_res=cur_res,
                cand_res=attrs["raw"]["resolution"],
                cand_score=int(attrs["raw"]["score"] or 0),
                t=resolved.transitions,
            )
            forbidden, reason = is_forbidden(deltas, resolved.transitions)
            if forbidden:
                status = f"drop: gate ({reason})"
            else:
                legal.append((r, attrs, clo))
        rows.append({"release": r, "attrs": attrs, "closeness": clo, "status": status})

    pick = topsis.select(legal, resolved)
    pick_id = id(pick[0]) if pick else None
    for row in rows:
        if pick_id is not None and id(row["release"]) == pick_id:
            row["status"] = PICK

    return {
        "resolved": resolved,
        "profile_name": profile_name,
        "target_res": target_res,
        "current": {"closeness": cur_clo, **cur_raw},
        "rows": rows,
        "gap_cutoff_score": gap_cutoff_score,
        "n_legal": len(legal),
    }


def _fmt_row(title: str, score, res, size_gb, gbh, clo, status) -> str:
    score_s = f"{score:,}" if score is not None else "n/a"
    clo_s = f"{clo:.3f}" if clo is not None else "n/a"
    res_s = f"{res}p" if res else "?"
    return f"| {title} | {score_s} | {res_s} | {size_gb:.2f} | {gbh:.2f} | {clo_s} | {status} |"


def render_item(api: ArrApi, item: dict, analysis: dict, gap: float) -> list[str]:
    label = api.label(item)
    prof = analysis["resolved"]
    cur = analysis["current"]
    target = analysis["target_res"]
    target_s = f"{target}p" if target else "?"
    md = [
        f"### {label}",
        "",
        f"- quality profile: `{analysis['profile_name'] or '?'}` (target {target_s})",
        f"- resolved preset: pick `{prof.pick}`, weights `{prof.weights}`",
        f"- current file: {_fmt_side(cur)}",
        f"- legal candidates after gate: **{analysis['n_legal']}**",
        "",
        "| title | score | res | size GB | GiB/h | closeness | status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    rows = sorted(
        analysis["rows"],
        key=lambda r: -(r["release"].get("customFormatScore") or 0),
    )
    cutoff = analysis["gap_cutoff_score"]
    cutoff_drawn = False
    for r in rows:
        rel = r["release"]
        raw = r["attrs"]["raw"]
        score = rel.get("customFormatScore")
        # Draw the score-gap cutoff line once, just below the last release at/above the cutoff.
        if cutoff is not None and not cutoff_drawn and (score or 0) < cutoff:
            md.append(
                f"| **─── score-gap {gap:.0%} cutoff "
                f"(min kept score = {cutoff:,}) ───** | | | | | | |"
            )
            cutoff_drawn = True
        title = rel.get("title", "?")
        # Truncate very long release titles so the table stays readable.
        if len(title) > 70:
            title = title[:67] + "..."
        md.append(
            _fmt_row(
                title,
                score,
                raw["resolution"],
                raw["size_gb"],
                raw["gbh"],
                r["closeness"],
                r["status"],
            )
        )
    md.append("")
    return md


def _fmt_side(side: dict) -> str:
    score = side.get("score")
    score_s = f"{score:,}" if score is not None else "n/a"
    clo = side.get("closeness")
    clo_s = f"{clo:.3f}" if clo is not None else "n/a"
    res = side.get("resolution") or 0
    return (
        f"score={score_s} res={res}p size={side.get('size_gb', 0):.2f}GB "
        f"({side.get('gbh', 0):.2f} GiB/h) closeness={clo_s}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.toml", help="path to config.toml")
    ap.add_argument("--app", default="radarr", choices=["radarr", "sonarr"])
    ap.add_argument("--limit", type=int, default=5, help="max items to evaluate (rate limits!)")
    ap.add_argument(
        "--tmdb",
        type=int,
        nargs="*",
        default=[],
        help="evaluate these TMDB ids (the id in the *arr movie URL); Sonarr: TVDB ids",
    )
    ap.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=[],
        help="evaluate these internal Radarr/Sonarr ids",
    )
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between items")
    ap.add_argument("--seed", type=int, help="seed the random sample for a reproducible run")
    args = ap.parse_args()

    config = load_config(args.config)
    conn = getattr(config, args.app)
    if conn is None:
        raise SystemExit(f"{args.app} is not configured in the environment")
    app_cfg = getattr(config.optimizer, args.app)
    topsis = Topsis(config.optimizer.topsis)
    api = build_client(args.app, conn)
    api.refresh_profiles()

    all_items = api.list_items()
    items = [it for it in all_items if api.has_file(it)]
    if args.tmdb or args.ids:
        # Internal id and external id (tmdb/tvdb) are separate id spaces and can collide, so they
        # are matched independently. --tmdb is the one to use: it's the id in the *arr movie URL.
        # Warn (per id space) on anything unknown or with no file, instead of silently dropping it.
        want_ext, want_internal = set(args.tmdb), set(args.ids)

        def ext_id(it: dict) -> int | None:
            return it.get("tmdbId") or it.get("tvdbId")

        items = [it for it in items if api.item_id(it) in want_internal or ext_id(it) in want_ext]

        def warn(label: str, wanted: set[int], present: set[int], with_file: set[int]) -> None:
            for missing in sorted(wanted - present):
                print(f"[warn] {label} {missing}: no such item")
            for nofile in sorted((wanted & present) - with_file):
                print(f"[warn] {label} {nofile}: exists but has no downloaded file; skipped")

        warn(
            "tmdb/tvdb id",
            want_ext,
            {e for it in all_items if (e := ext_id(it)) is not None},
            {e for it in items if (e := ext_id(it)) is not None},
        )
        warn(
            "internal id",
            want_internal,
            {api.item_id(it) for it in all_items},
            {api.item_id(it) for it in items},
        )
        # --limit caps every path: each item is a live indexer search, so an over-long id list
        # shouldn't fire dozens of them.
        if len(items) > args.limit:
            print(f"[warn] {len(items)} items selected; capping at --limit {args.limit}")
            items = items[: args.limit]
    else:
        # Random sample so repeated runs don't always show the same first movies.
        random.Random(args.seed).shuffle(items)
        items = items[: args.limit]

    header = [
        "# Candidate / score-gap cutoff visualization",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- app: `{args.app}`  url: `{conn.url}`",
        f"- score_gap (cutoff): `{topsis.cfg.score_gap:.0%}`",
        f"- items evaluated: {len(items)} (read-only; no grab/import/state writes)",
        "",
        "Status legend: `PICK` = chosen release; `candidate` = legal after the transition gate; "
        "`drop: ...` = removed, with the stage and reason. Rows are sorted by score so the "
        "score-gap cutoff line is visible.",
        "",
    ]
    body: list[str] = []
    for it in items:
        label = api.label(it)
        print(f"[{args.app}] searching releases for {label} ...")
        runtime_h = api.runtime_h(it)
        profile_name, target_res = api.profile_for(it)
        current_file = api.current_file(it)
        releases = api.releases(it)
        analysis = analyze(
            topsis, releases, runtime_h, profile_name, target_res, current_file, app_cfg
        )
        body += render_item(api, it, analysis, topsis.cfg.score_gap)
        time.sleep(args.sleep)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    out = reports / f"cutoff_viz_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text("\n".join(header + body), encoding="utf-8")
    print(f"\nWrote {out} ({len(items)} item(s)).")


if __name__ == "__main__":
    main()
