"""Pure per-item decision: given fetched data, return ACT (with the release) or HOLD.

"Optimized" means the algorithm can no longer find anything better than the current file (a
HOLD that *satisfies*), never merely "we triggered a grab". The decision is:

  1. **Filter + score** (topsis.py): drop hard rejections, drop score below the downgrade budget
     (`current - score_window`), keep only the target resolution, drop outside the legitimacy band,
     drop lone-small outliers, then min-max normalize the survivors on two relative axes (score,
     GiB/h) and combine with the profile's weights into a TOPSIS closeness.
  2. **Decide** (here): if too few candidates survived to compare, HOLD WITHOUT satisfying (retry
     later). Otherwise ACT on the best candidate iff its closeness beats the current file's by the
     margin; else HOLD and SATISFY (the current file is already optimal for its profile).

A satisfying HOLD means the current (already-imported) file is the best pick. Because a candidate
must beat the current file's closeness, a pick can never be both lower-score AND bigger than the
current file (that is worse on both axes, so lower closeness). One-and-done state (worker.py) makes
a satisfied movie permanent, which is what prevents oscillation under relative scoring.
"""

from dataclasses import dataclass

from optimizarr.features.optimizer.topsis import GB, Topsis


@dataclass
class Decision:
    action: str  # "ACT" or "HOLD"
    reason: str
    satisfy: bool = False  # HOLD only: True => current file is optimal, mark satisfied (permanent)
    profile_name: str | None = None
    current: dict | None = None  # {score, resolution, gbh, size_gb, closeness}
    pick: dict | None = None  # {score, resolution, gbh, size_gb, closeness, title}
    release: dict | None = None  # raw release to grab (ACT only)
    diag: dict | None = None


def _current_raw(cf: dict, runtime_h: float) -> dict:
    size_gb = (cf.get("size", 0) or 0) / GB
    gbh = (size_gb / runtime_h) if (runtime_h and runtime_h > 0) else 0.0
    res = ((cf.get("quality") or {}).get("quality") or {}).get("resolution") or 0
    return {"score": cf.get("customFormatScore"), "resolution": res, "gbh": gbh, "size_gb": size_gb}


def decide(
    topsis: Topsis,
    releases: list[dict],
    runtime_h: float,
    profile_name: str | None,
    target_resolution: int | None,
    current_file: dict | None,
    allow_size_increase: bool = True,
    allow_quality_downgrade: bool = True,
) -> Decision:
    """Pure decision: filter + relatively score the candidates, then ACT on the best one if it
    beats the current file's closeness, else HOLD (satisfying iff there were enough candidates to
    trust the comparison).

    Two optional pre-filters apply before scoring (per-app policy):
      - allow_size_increase=False drops releases bigger than the current file;
      - allow_quality_downgrade=False drops releases with a lower customFormatScore."""
    cur = current_file or {}
    cur_size = cur.get("size")
    if not allow_size_increase and isinstance(cur_size, int) and cur_size > 0:
        releases = [r for r in releases if r.get("size", 0) <= cur_size]
    cur_score = cur.get("customFormatScore")
    if not allow_quality_downgrade and cur_score is not None:
        releases = [r for r in releases if (r.get("customFormatScore") or 0) >= cur_score]

    resolved = topsis.resolve_profile(profile_name)
    kept, diag = topsis.apply_prefilters(releases, runtime_h, target_resolution, cur_score)

    cur_raw = _current_raw(cur, runtime_h) if current_file else None

    # Not enough to compare: HOLD but do NOT satisfy, so the movie is retried when more appear.
    if len(kept) < topsis.cfg.min_candidates:
        return Decision(
            "HOLD",
            f"too few candidates to compare ({len(kept)} < {topsis.cfg.min_candidates})",
            satisfy=False,
            profile_name=profile_name,
            current={"closeness": None, **cur_raw} if cur_raw else None,
            diag=diag,
        )

    scored, current = topsis.score_pool(kept, current_file, runtime_h, resolved)
    cur_clo = current["closeness"] if current else None
    current_info = {"closeness": cur_clo, **(current["raw"] if current else cur_raw or {})}

    margin = resolved.min_closeness_gain
    legal = [t for t in scored if cur_clo is None or t[2] > cur_clo + margin]

    if not legal:
        # Current file already beats every candidate for its profile -> optimized. Satisfy.
        best = scored[0]
        best_info = {"closeness": best[2], "title": best[0].get("title", "?"), **best[1]["raw"]}
        return Decision(
            "HOLD",
            "nothing better (current is optimal)",
            satisfy=True,
            profile_name=profile_name,
            current=current_info,
            pick=best_info,
            diag=diag,
        )

    rel, attrs, clo = legal[0]  # scored is sorted best-first, so this is the highest closeness
    pick_info = {"closeness": clo, "title": rel.get("title", "?"), **attrs["raw"]}
    return Decision(
        "ACT",
        f"best of {len(legal)} (closeness {clo:.3f})",
        satisfy=False,
        profile_name=profile_name,
        current=current_info,
        pick=pick_info,
        release=rel,
        diag=diag,
    )


def _fmt_side(side: dict | None) -> str:
    if not side:
        return "n/a"
    score = side.get("score")
    score_s = f"{score:,}" if score is not None else "n/a"
    clo = side.get("closeness")
    clo_s = f"{clo:.3f}" if clo is not None else "n/a"
    res = side.get("resolution") or 0
    res_s = f"{res}p" if res else "?"
    return (
        f"score={score_s} res={res_s} size={side.get('size_gb', 0):.1f}GB "
        f"({side.get('gbh', 0):.1f} GB/h) closeness={clo_s}"
    )


def _fmt_deltas(current: dict | None, pick: dict | None) -> str:
    if not current or not pick:
        return ""
    parts = []
    c_clo, p_clo = current.get("closeness"), pick.get("closeness")
    if c_clo is not None and p_clo is not None:
        parts.append(f"Δcloseness {p_clo - c_clo:+.3f}")
    parts.append(f"Δsize {pick.get('size_gb', 0) - current.get('size_gb', 0):+.1f}GB")
    return "  (" + ", ".join(parts) + ")" if parts else ""


def format_decision(app: str, label: str, decision: Decision, dry_run: bool) -> str:
    """Multi-line, human-readable explanation of one decision (current vs pick)."""
    profile = decision.profile_name or "?"
    if decision.action == "ACT":
        verb = "would GRAB" if dry_run else "GRAB"
        head = f"[{app}] {verb} — {label}  [profile={profile}]  ({decision.reason})"
    else:
        head = f"[{app}] HOLD — {label}  [profile={profile}]  ({decision.reason})"

    lines = [head, f"    current: {_fmt_side(decision.current)}"]
    if decision.pick:
        candidate_label = "pick" if decision.action == "ACT" else "best   "
        lines.append(
            f"    {candidate_label}: {_fmt_side(decision.pick)}"
            f"{_fmt_deltas(decision.current, decision.pick)}"
        )
        lines.append(f"    release: {decision.pick.get('title', '?')}")
    return "\n".join(lines)
