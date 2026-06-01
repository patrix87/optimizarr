"""Size-per-hour (GiB/h) stats for specific release cohorts, from a gather_training_data.py set.

Rejected/temporarily-rejected releases are INCLUDED (no quality filtering). Dropped by default:
raw sources (remux + BR-DISK, which inflate size; --include-raw keeps them), non-H.264/H.265
codecs (AV1/XviD/DivX/VC-1/etc; --any-codec keeps them), negative-score releases
(--include-negative keeps them), and releases with an unknown runtime (gbh = 0). Each drop count
is reported.

Cohorts:
  A. Profile-matched, high score: target profile resolution == result resolution, score > --score
     (default 800000). One row for 2160p, one for 1080p.
  B. By result resolution, any profile, any score: 2160p, 1080p, 720p, 480p.

Stats per cohort: n, mean, median, p5, p10, p25, p75, p90, p95, min, max.

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


def _is_raw(r: dict) -> bool:
    """Raw, non-encode sources (remux lossless rips and full BR-DISK) inflate size stats, so
    they're dropped by default. Remux: quality name 'Remux-*' or 'remux' in the title. BR-DISK:
    Radarr's full-disc quality name."""
    name = (r.get("quality_name") or "").lower()
    title = (r.get("title") or "").lower()
    return "remux" in name or "remux" in title or "br-disk" in name


def _codec(r: dict) -> str:
    """Classify the title's video codec: 'hevc' (x265/H.265), 'avc' (x264/H.264), or 'other'
    (AV1, XviD/DivX, VC-1, MPEG-2, or unidentifiable)."""
    t = (r.get("title") or "").lower().replace(".", "")
    if "x265" in t or "h265" in t or "hevc" in t:
        return "hevc"
    if "x264" in t or "h264" in t or "avc" in t:
        return "avc"
    return "other"


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
        "p5": _pct(s, 0.05),
        "p10": _pct(s, 0.10),
        "p25": _pct(s, 0.25),
        "p75": _pct(s, 0.75),
        "p90": _pct(s, 0.90),
        "p95": _pct(s, 0.95),
        "min": s[0],
        "max": s[-1],
    }


def _row(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"| {label} | 0 |" + "  |" * 11
    return (
        f"| {label} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p5']:.2f} | "
        f"{s['p10']:.2f} | {s['p25']:.2f} | {s['p75']:.2f} | {s['p90']:.2f} | {s['p95']:.2f} | "
        f"{s['min']:.2f} | {s['max']:.2f} |"
    )


HEADER = (
    "| cohort | n | mean | median | p5 | p10 | p25 | p75 | p90 | p95 | min | max |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="JSONL from gather_training_data.py")
    ap.add_argument("--score", type=float, default=800_000.0, help="group-A score-above threshold")
    ap.add_argument("--include-raw", action="store_true", help="keep remux + BR-DISK")
    ap.add_argument(
        "--any-codec", action="store_true", help="keep all codecs (default: H.264/H.265)"
    )
    ap.add_argument("--include-negative", action="store_true", help="keep score < 0 releases")
    args = ap.parse_args()

    def label_a(res: int) -> str:
        return f"{res}p profile + {res}p result, score > {args.score:,.0f}"

    records = _load(args.dataset)

    # cohort name -> list of gbh
    a: dict[str, list[float]] = {}  # profile-matched + high score
    b: dict[int, list[float]] = {}  # by result resolution, any profile/score
    skipped_no_runtime = skipped_raw = skipped_codec = skipped_neg = 0

    for rec in records:
        target = (rec.get("profile") or {}).get("target_resolution")
        for r in rec.get("releases", []):
            if not args.include_raw and _is_raw(r):
                skipped_raw += 1
                continue
            if not args.any_codec and _codec(r) == "other":
                skipped_codec += 1
                continue
            score = r.get("score")
            if not args.include_negative and score is not None and score < 0:
                skipped_neg += 1
                continue
            gbh = r.get("gbh") or 0
            if gbh <= 0:  # unknown runtime: cannot form a size-per-hour value
                skipped_no_runtime += 1
                continue
            res = r.get("resolution") or 0
            b.setdefault(res, []).append(gbh)
            if target == res and res in (2160, 1080) and score is not None and score > args.score:
                a.setdefault(label_a(res), []).append(gbh)

    md: list[str] = [
        "# Size-per-hour (GiB/h) cohorts",
        "",
        f"- dataset: `{args.dataset}`  items: {len(records)}",
        f"- raw sources (remux + BR-DISK): "
        f"{'INCLUDED' if args.include_raw else f'excluded ({skipped_raw} dropped)'}",
        f"- codec: {'ALL' if args.any_codec else f'H.264/H.265 only ({skipped_codec} dropped)'}",
        f"- negative scores: {'INCLUDED' if args.include_negative else f'dropped ({skipped_neg})'}",
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
