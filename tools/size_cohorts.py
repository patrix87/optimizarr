"""Size-per-hour (GiB/h) stats for specific release cohorts, from a gather_training_data.py set.

Rejected/temporarily-rejected releases are INCLUDED (no quality filtering). Two things are
dropped: remux releases (lossless rips that skew size up; --include-remux keeps them) and
releases with an unknown runtime (gbh = 0). Both drop counts are reported.

Cohorts:
  A. Profile-matched, high score: target profile resolution == result resolution, score > --score
     (default 800000). One row for 2160p, one for 1080p.
  B. By result resolution, any profile, any score: 2160p, 1080p, 720p, 480p.

Stats per cohort: n, mean, median, p10, p25, p75, p90, min, max.

Run:  uv run python tools/size_cohorts.py reports/training_data_radarr.jsonl
Writes a timestamped Markdown report under ./reports/ and prints it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from datetime import datetime
from pathlib import Path


def _load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _is_remux(r: dict) -> bool:
    """Remux releases (lossless rips, not encodes) skew size stats up, so they're excluded by
    default. Caught by the quality name (Remux-2160p/1080p) or 'remux' in the release title."""
    name = (r.get("quality_name") or "").lower()
    return "remux" in name or "remux" in (r.get("title") or "").lower()


def _pct(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0,1]) over an already-sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _summary(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": n,
        "mean": st.mean(s),
        "median": st.median(s),
        "p10": _pct(s, 0.10),
        "p25": _pct(s, 0.25),
        "p75": _pct(s, 0.75),
        "p90": _pct(s, 0.90),
        "min": s[0],
        "max": s[-1],
    }


def _row(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"| {label} | 0 |  |  |  |  |  |  |  |  |"
    return (
        f"| {label} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p10']:.2f} | "
        f"{s['p25']:.2f} | {s['p75']:.2f} | {s['p90']:.2f} | {s['min']:.2f} | {s['max']:.2f} |"
    )


HEADER = (
    "| cohort | n | mean | median | p10 | p25 | p75 | p90 | min | max |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="JSONL from gather_training_data.py")
    ap.add_argument("--score", type=float, default=800_000.0, help="group-A score-above threshold")
    ap.add_argument("--include-remux", action="store_true", help="keep remux (default: excluded)")
    args = ap.parse_args()

    def label_a(res: int) -> str:
        return f"{res}p profile + {res}p result, score > {args.score:,.0f}"

    records = _load(args.dataset)

    # cohort name -> list of gbh
    a: dict[str, list[float]] = {}  # profile-matched + high score
    b: dict[int, list[float]] = {}  # by result resolution, any profile/score
    skipped_no_runtime = 0
    skipped_remux = 0

    for rec in records:
        target = (rec.get("profile") or {}).get("target_resolution")
        for r in rec.get("releases", []):
            if not args.include_remux and _is_remux(r):
                skipped_remux += 1
                continue
            gbh = r.get("gbh") or 0
            if gbh <= 0:  # unknown runtime: cannot form a size-per-hour value
                skipped_no_runtime += 1
                continue
            res = r.get("resolution") or 0
            score = r.get("score")
            b.setdefault(res, []).append(gbh)
            if target == res and res in (2160, 1080) and score is not None and score > args.score:
                a.setdefault(label_a(res), []).append(gbh)

    md: list[str] = [
        "# Size-per-hour (GiB/h) cohorts",
        "",
        f"- dataset: `{args.dataset}`  items: {len(records)}",
        f"- remux: {'INCLUDED' if args.include_remux else f'excluded ({skipped_remux} dropped)'}",
        f"- rejected releases included; {skipped_no_runtime} dropped for unknown runtime",
        "",
        "## A. Profile-matched, score above threshold",
        "",
        HEADER,
    ]
    for res in (2160, 1080):
        md.append(_row(label_a(res), _summary(a.get(label_a(res), []))))

    md += [
        "",
        "## B. All results by resolution (any profile, any score)",
        "",
        HEADER,
    ]
    for res in (2160, 1080, 720, 480):
        md.append(_row(f"all {res}p results", _summary(b.get(res, []))))

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    out = reports / f"size_cohorts_{datetime.now():%Y%m%d_%H%M%S}.md"
    text = "\n".join(md) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
