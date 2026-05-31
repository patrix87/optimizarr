"""Derive size-model stats from a training set gathered by tools/gather_training_data.py.

Reads the JSONL dataset and computes, with no reference to the current decision logic:

  1. GiB/h distribution per resolution (count, mean, median, p25, p75, min, max) - the basis for
     retuning [optimizer.topsis.reference] floor/target/ceiling.
  2. GiB/h per (resolution x score bracket) - how size correlates with quality, so floors/targets
     can be reasoned about per quality tier. Brackets are fractions of --score-ideal.
  3. Size growth/shrink: current library file GiB/h vs the candidate pool, per resolution, so you
     can extrapolate whether realigning toward a new target would grow or shrink files.

Run:  uv run python tools/training_stats.py reports/training_data_radarr.jsonl

Writes a timestamped Markdown report under ./reports/ and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import datetime
from pathlib import Path

# Score brackets as fractions of score_ideal; "neg" and "unknown" are handled separately.
BRACKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, float("inf"))]


def _bracket_label(lo: float, hi: float) -> str:
    return f">={lo:.0%}" if hi == float("inf") else f"{lo:.0%}-{hi:.0%}"


# Sort order for brackets in the output (neg first, then ascending, unknown last).
BRACKET_ORDER = {_bracket_label(lo, hi): i for i, (lo, hi) in enumerate(BRACKETS)}
BRACKET_ORDER["neg"] = -1
BRACKET_ORDER["unknown"] = 99


def _load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _is_raw(r: dict) -> bool:
    """Raw, non-encode sources (remux + full BR-DISK) inflate size stats; dropped by default."""
    name = (r.get("quality_name") or "").lower()
    return "remux" in name or "remux" in (r.get("title") or "").lower() or "br-disk" in name


def _codec(r: dict) -> str:
    """Title video codec: 'hevc' (x265/H.265), 'avc' (x264/H.264), or 'other'."""
    t = (r.get("title") or "").lower().replace(".", "")
    if "x265" in t or "h265" in t or "hevc" in t:
        return "hevc"
    if "x264" in t or "h264" in t or "avc" in t:
        return "avc"
    return "other"


def _bracket(score: float | None, ideal: float) -> str:
    if score is None:
        return "unknown"
    if score < 0:
        return "neg"
    frac = score / ideal
    for lo, hi in BRACKETS:
        if lo <= frac < hi:
            return _bracket_label(lo, hi)
    return _bracket_label(*BRACKETS[-1])


def _summary(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": n,
        "mean": st.mean(s),
        "median": st.median(s),
        "p25": s[int(0.25 * (n - 1))],
        "p75": s[int(0.75 * (n - 1))],
        "min": s[0],
        "max": s[-1],
    }


def _cells(s: dict) -> str:
    """The 7 numeric cells (n, mean, median, p25, p75, min, max) without leading/trailing pipes."""
    if s["n"] == 0:
        return "0 |  |  |  |  |  | "
    return (
        f"{s['n']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p25']:.2f} | "
        f"{s['p75']:.2f} | {s['min']:.2f} | {s['max']:.2f}"
    )


def _res(res: int) -> str:
    return f"{res}p" if res else "?"


def _bucket_res(height: int) -> int:
    """Snap a raw pixel height to a standard resolution bucket. Current-file resolution comes from
    mediaInfo (e.g. 1632/858 for scope framing); candidate resolution is already standard, so this
    is a no-op there and only realigns the current-file side."""
    if not height:
        return 0
    if height >= 1601:
        return 2160
    if height >= 721:
        return 1080
    if height >= 620:
        return 720
    if height >= 400:
        return 480
    return height


def _num(value: float | None, fmt: str = ".2f") -> str:
    return format(value, fmt) if value is not None else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="JSONL from gather_training_data.py")
    ap.add_argument("--score-ideal", type=float, default=1_000_000.0, help="score == 1.0 fraction")
    ap.add_argument(
        "--include-rejected",
        action="store_true",
        help="include hard/temporarily-rejected releases (default: drop them)",
    )
    ap.add_argument("--include-raw", action="store_true", help="keep remux + BR-DISK")
    ap.add_argument(
        "--any-codec", action="store_true", help="keep all codecs (default: H.264/H.265)"
    )
    ap.add_argument("--include-negative", action="store_true", help="keep score < 0 releases")
    args = ap.parse_args()

    records = _load(args.dataset)

    rel_by_res: dict[int, list[float]] = {}
    rel_by_res_bracket: dict[tuple[int, str], list[float]] = {}
    score_by_res_bracket: dict[tuple[int, str], list[float]] = {}
    cur_by_res: dict[int, list[float]] = {}

    n_releases = n_raw = n_codec = n_neg = 0
    for rec in records:
        cur = rec.get("current_file")
        if cur and cur.get("gbh"):
            cur_by_res.setdefault(_bucket_res(cur.get("resolution") or 0), []).append(cur["gbh"])
        for r in rec.get("releases", []):
            if not args.include_raw and _is_raw(r):
                n_raw += 1
                continue
            if not args.any_codec and _codec(r) == "other":
                n_codec += 1
                continue
            score = r.get("score")
            if not args.include_negative and score is not None and score < 0:
                n_neg += 1
                continue
            if not args.include_rejected and (r.get("rejections") or r.get("temporarily_rejected")):
                continue
            gbh = r.get("gbh")
            if not gbh:
                continue
            res = r.get("resolution") or 0
            n_releases += 1
            rel_by_res.setdefault(res, []).append(gbh)
            b = _bracket(r.get("score"), args.score_ideal)
            rel_by_res_bracket.setdefault((res, b), []).append(gbh)
            if r.get("score") is not None:
                score_by_res_bracket.setdefault((res, b), []).append(r["score"])

    md: list[str] = [
        "# Training-set size stats",
        "",
        f"- dataset: `{args.dataset}`",
        f"- items: {len(records)}  releases analyzed: {n_releases}"
        f"{' (incl. rejected)' if args.include_rejected else ' (rejected dropped)'}",
        f"- raw (remux+BR-DISK): {'INCLUDED' if args.include_raw else f'dropped ({n_raw})'}; "
        f"codec: {'ALL' if args.any_codec else f'H.264/H.265 only (dropped {n_codec})'}; "
        f"negative scores: {'INCLUDED' if args.include_negative else f'dropped ({n_neg})'}",
        f"- score_ideal: {args.score_ideal:,.0f}",
        "",
        "## 1. GiB/h per resolution (candidate releases)",
        "",
        "| resolution | n | mean | median | p25 | p75 | min | max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in sorted(rel_by_res, reverse=True):
        md.append(f"| {_res(res)} | {_cells(_summary(rel_by_res[res]))} |")

    md += [
        "",
        "## 2. GiB/h per resolution x score bracket",
        "",
        "Bracket = customFormatScore as a fraction of score_ideal.",
        "",
        "| resolution | score bracket | n | mean | median | p25 | p75 | min | max | mean score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in sorted({k[0] for k in rel_by_res_bracket}, reverse=True):
        keys = sorted(
            (k for k in rel_by_res_bracket if k[0] == res),
            key=lambda k: BRACKET_ORDER.get(k[1], 50),
        )
        for k in keys:
            scores = score_by_res_bracket.get(k, [])
            mean_score = f"{st.mean(scores):,.0f}" if scores else ""
            cells = _cells(_summary(rel_by_res_bracket[k]))
            md.append(f"| {_res(res)} | {k[1]} | {cells} | {mean_score} |")

    md += [
        "",
        "## 3. Size growth / shrink (current files vs candidate pool)",
        "",
        "Per resolution: mean GiB/h of existing library files vs the candidate releases. A current "
        "mean above the pool mean means realigning toward the pool shrinks files; below it means "
        "they grow.",
        "",
        "| resolution | current n | current mean | candidate mean | delta (cand - cur) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for res in sorted(set(cur_by_res) | set(rel_by_res), reverse=True):
        cur = cur_by_res.get(res, [])
        rel = rel_by_res.get(res, [])
        cur_mean = st.mean(cur) if cur else None
        rel_mean = st.mean(rel) if rel else None
        delta = (rel_mean - cur_mean) if (cur_mean is not None and rel_mean is not None) else None
        md.append(
            f"| {_res(res)} | {len(cur)} | {_num(cur_mean)} | {_num(rel_mean)} | "
            f"{_num(delta, '+.2f')} |"
        )

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    out = reports / f"training_stats_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(records)} items, {n_releases} releases).")


if __name__ == "__main__":
    main()
