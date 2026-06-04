"""Weight lab: visualize how the presets score and pick releases under the new model.

Drives the real engine: each preset's weights + 5-point size table {floor, lo, target, hi,
ceiling} + pick, and the closeness-gain swap rule (via the real decide()). Two modes:

  - default (synthetic): curated scenarios + a retention stress test, showing every candidate's
    closeness under every preset (★ = the preset's pick; "drop" = excluded by that preset's size
    band or the score gap-cut).
  - --dataset PATH: drive a real gathered set (tools/gather_training_data.py JSONL). For a random
    sample of movies it shows, per preset, the decision and the GiB/h it would land on — the best
    sanity check for tuning the per-preset size tables.

Run:  uv run python tools/weight_lab.py
      uv run python tools/weight_lab.py --dataset reports/training_data_radarr.jsonl --limit 25
Writes a timestamped Markdown report under ./reports/ and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from optimizarr.features.optimizer.config import ResolvedProfile, default_topsis
from optimizarr.features.optimizer.decision import decide
from optimizarr.features.optimizer.topsis import GB, Topsis, eligible


def _release(title: str, score: int, res: int, size_gb: float) -> dict:
    return {
        "title": title,
        "customFormatScore": score,
        "quality": {"quality": {"resolution": res}},
        "size": int(size_gb * GB),
        "rejections": [],
    }


def _as_file(release: dict) -> dict:
    """Convert a release-shaped dict into the current-library-file shape decide() expects."""
    res = release["quality"]["quality"]["resolution"]
    return {
        "id": 1,
        "customFormatScore": release["customFormatScore"],
        "size": release["size"],
        "mediaInfo": {"resolution": f"x{res}"},
    }


@dataclass
class Scenario:
    name: str
    target_res: int
    runtime_h: float
    current: dict
    candidates: list[dict]
    note: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        name="4K: remux vs web vs lean x265",
        target_res=2160,
        runtime_h=2.0,
        current=_release("(current) mediocre 2160p", 700_000, 2160, 30.0),
        candidates=[
            _release("2160p Remux", 1_000_000, 2160, 60.0),  # 30 GiB/h
            _release("2160p WEB-DL", 950_000, 2160, 24.0),  # 12 GiB/h
            _release("2160p x265", 930_000, 2160, 8.0),  # 4 GiB/h, lean
        ],
        note="Remux takes the top-scoring remux (max_score); size-leaning presets swing to the "
        "lean encodes. No profile is pushed bigger than a real score gain warrants.",
    ),
    Scenario(
        name="Much-smaller current file must not be inflated",
        target_res=2160,
        runtime_h=2.0,
        current=_release("(current) superb tiny x265", 950_000, 2160, 7.0),  # 3.5 GiB/h
        candidates=[
            _release("2160p WEB-DL bigger", 950_000, 2160, 22.0),  # 11 GiB/h, same score
            _release("2160p remux huge", 980_000, 2160, 60.0),  # 30 GiB/h, slightly higher
        ],
        note="The current file is already tiny and good. Same-score-but-bigger is forbidden for "
        "all; only Remux may take the higher-scoring big remux (every other preset holds).",
    ),
]

RETENTION_RUNTIME_H = 2.0
RETENTION_TARGET = 2160
RETENTION_PRESET = "Compact"  # size-leaning: where small gems matter most


def _resolved(t: Topsis, name: str) -> ResolvedProfile:
    return t.resolve_profile(name)


def _included(t: Topsis, releases: list[dict], runtime_h: float, rp: ResolvedProfile) -> set[int]:
    """ids of releases that survive this preset's size band (floor..bloat) + the score gap-cut.
    Inclusion is now per-preset, since each preset carries its own size table."""
    after_band = t.filter_by_size_band(eligible(releases), runtime_h, rp.reference)
    return {id(r) for r in t.filter_by_score_gap(after_band)}


def _closeness(
    t: Topsis, rp: ResolvedProfile, release: dict, runtime_h: float, target: int
) -> float:
    return t.closeness(t.attributes_for(release, runtime_h, rp, target), rp.weights)


def render_scenario(t: Topsis, sc: Scenario) -> tuple[list[str], list[str]]:
    presets = t.cfg.presets
    rp = {name: _resolved(t, name) for name in presets}
    included = {name: _included(t, sc.candidates, sc.runtime_h, rp[name]) for name in presets}
    cur_clo = {
        name: _closeness(t, rp[name], sc.current, sc.runtime_h, sc.target_res) for name in presets
    }
    cand_clo = {
        id(r): {name: _closeness(t, rp[name], r, sc.runtime_h, sc.target_res) for name in presets}
        for r in sc.candidates
    }
    # The real decision (gate + pick) per preset.
    cur_file = _as_file(sc.current)
    decisions = {
        name: decide(t, sc.candidates, sc.runtime_h, name, sc.target_res, cur_file)
        for name in presets
    }
    winner_title = {
        name: (d.pick.get("title") if d.action == "ACT" and d.pick else None)
        for name, d in decisions.items()
    }

    md = [
        f"## {sc.name}",
        "",
        f"- target {sc.target_res}p · runtime {sc.runtime_h:g}h",
        f"- {sc.note}",
        "",
    ]
    header = ["release", "score", "res", "GiB/h", *presets]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")

    def row(r: dict, clo: dict[str, float], dropped: dict[str, bool] | None) -> str:
        gbh = (r["size"] / GB) / sc.runtime_h
        res = r["quality"]["quality"]["resolution"]
        cells = [r["title"], f"{r['customFormatScore']:,}", f"{res}p", f"{gbh:.1f}"]
        for name in presets:
            if dropped and dropped[name]:
                cells.append("drop")
            else:
                val = f"{clo[name]:.3f}"
                if winner_title[name] == r["title"]:
                    val = f"**{val} ★**"
                cells.append(val)
        return "| " + " | ".join(cells) + " |"

    md.append(row(sc.current, cur_clo, None) + "  ← current")
    for r in sc.candidates:
        md.append(row(r, cand_clo[id(r)], {name: id(r) not in included[name] for name in presets}))
    md.append("")

    md.append("**Decision per preset** (real gate + pick vs current):")
    md.append("")
    summary = [f"  {sc.name}"]
    for name in presets:
        d = decisions[name]
        title = d.pick.get("title") if d.pick else "—"
        md.append(f"- `{name}`: {d.action} — {title}  ({d.reason})")
        summary.append(f"    {name:<10} → {d.action:<4} {title}")
    md.append("")
    return md, summary


def _make_popular(rng: random.Random) -> list[dict]:
    rel: list[dict] = []

    def add(kind: str, count: int, lo: int, hi: int, size_lo: float, size_hi: float) -> None:
        for _ in range(count):
            score = min(1_000_000, rng.randint(lo, hi))
            size = round(rng.uniform(size_lo, size_hi), 1)
            rel.append(_release(f"{kind} {size}GB s={score // 1000}k", score, 2160, size))

    add("remux", 12, 880_000, 1_000_000, 45.0, 70.0)
    add("bluray", 22, 860_000, 1_000_000, 26.0, 44.0)
    add("web", 26, 850_000, 990_000, 15.0, 26.0)
    add("x265", 14, 880_000, 970_000, 6.0, 12.0)  # high score, small — the "gems"
    add("web-mid", 12, 550_000, 840_000, 14.0, 24.0)
    add("low", 4, 60_000, 480_000, 8.0, 30.0)
    add("FAKE", 4, 900_000, 1_000_000, 1.0, 2.6)  # tiny -> below the 2160 floor (3.0 GiB/h)
    rng.shuffle(rel)
    return rel


def _make_obscure() -> list[dict]:
    return [
        _release("remux 55.0GB s=700k", 700_000, 2160, 55.0),
        _release("web 18.0GB s=720k", 720_000, 2160, 18.0),
        _release("x265 7.0GB s=690k", 690_000, 2160, 7.0),  # small gem
        _release("low 9.0GB s=380k", 380_000, 2160, 9.0),
        _release("banned 10.0GB s=-30k", -30_000, 2160, 10.0),
    ]


def render_retention(t: Topsis, name: str, releases: list[dict]) -> list[str]:
    rp = _resolved(t, RETENTION_PRESET)
    rt = RETENTION_RUNTIME_H
    after_hard = eligible(releases)
    after_band = t.filter_by_size_band(after_hard, rt, rp.reference)
    kept = t.filter_by_score_gap(after_band)

    md = [f"### {name}  (preset {RETENTION_PRESET})", ""]
    md.append("| stage | releases |")
    md.append("|---|---|")
    md.append(f"| input | {len(releases)} |")
    md.append(f"| after size band (floor..bloat) | {len(after_band)} |")
    md.append(f"| after score gap-cut | {len(kept)} |")
    md.append("")

    ranked = sorted(kept, key=lambda r: -_closeness(t, rp, r, rt, RETENTION_TARGET))
    md.append("Top 8 by closeness:")
    md.append("")
    md.append("| # | release | score | size GB (GiB/h) | closeness |")
    md.append("|---|---|---|---|---|")
    for i, r in enumerate(ranked[:8], 1):
        gbh = (r["size"] / GB) / rt
        clo = _closeness(t, rp, r, rt, RETENTION_TARGET)
        md.append(
            f"| {i} | {r['title']} | {r['customFormatScore']:,} | "
            f"{r['size'] / GB:.1f} ({gbh:.1f}) | {clo:.3f} |"
        )
    md.append("")
    if kept:
        smallest = min(kept, key=lambda r: r["size"])
        by_score = sorted(kept, key=lambda r: -r["customFormatScore"])
        s_rank = by_score.index(smallest) + 1
        c_rank = ranked.index(smallest) + 1
        verb = "wins" if c_rank == 1 else "ranks"
        md.append(
            f"Smallest survivor **{smallest['title']}**: #{s_rank} of {len(kept)} by score, "
            f"#{c_rank} by closeness — survived the filter and {verb} on size."
        )
        md.append("")
    return md


# ----- real dataset mode (gather_training_data.py JSONL) -----


def _rel_from_record(r: dict) -> dict:
    return {
        "title": r.get("title") or "?",
        "customFormatScore": r.get("score") or 0,
        "quality": {"quality": {"resolution": r.get("resolution") or 0}},
        "size": r.get("size_bytes") or 0,
        "rejections": r.get("rejections") or [],
        "temporarilyRejected": bool(r.get("temporarily_rejected")),
    }


def _file_from_record(c: dict | None) -> dict | None:
    if not c:
        return None
    return {
        "id": 1,
        "customFormatScore": c.get("score"),
        "size": c.get("size_bytes") or 0,
        "quality": {"quality": {"resolution": c.get("resolution") or 0}},
    }


def render_dataset_item(t: Topsis, rec: dict) -> tuple[list[str], list[str]]:
    prof = rec.get("profile") or {}
    pname, tres = prof.get("name"), prof.get("target_resolution")
    rt = rec.get("runtime_h") or 0
    releases = [_rel_from_record(r) for r in rec.get("releases", [])]
    cur_file = _file_from_record(rec.get("current_file"))
    cur = rec.get("current_file") or {}
    cur_gbh = cur.get("gbh", 0.0) or 0.0
    title = rec.get("title", "?")

    md = [
        f"### {title}",
        "",
        f"- profile `{pname}` · target {tres}p · runtime {rt:g}h · {len(releases)} releases",
        f"- current: score={cur.get('score')} · {cur.get('resolution') or '?'}p · "
        f"{cur_gbh:.2f} GiB/h",
        "",
        "| preset | decision | pick | pick score | pick GiB/h | Δsize | Δcloseness |",
        "|---|---|---|---|---|---|---|",
    ]
    summary = [f"  {title} (cur {cur_gbh:.1f} GiB/h)"]
    for name in t.cfg.presets:
        d = decide(t, releases, rt, name, tres, cur_file)
        if d.action == "ACT" and d.pick:
            p = d.pick
            pgbh = p.get("gbh", 0.0)
            c_clo = (d.current or {}).get("closeness")
            dclo = (
                f"{p['closeness'] - c_clo:+.3f}"
                if c_clo is not None and p.get("closeness") is not None
                else ""
            )
            ttl = p.get("title", "?")
            ttl = ttl[:42] + "..." if len(ttl) > 45 else ttl
            md.append(
                f"| {name} | GRAB | {ttl} | {p.get('score') or 0:,} | {pgbh:.2f} | "
                f"{pgbh - cur_gbh:+.2f} | {dclo} |"
            )
            summary.append(f"    {name:<10} GRAB {pgbh:5.1f} GiB/h  {ttl}")
        else:
            md.append(f"| {name} | HOLD | — | | | | |")
            summary.append(f"    {name:<10} HOLD")
    md.append("")
    return md, summary


def run_dataset(t: Topsis, path: Path, limit: int, seed: int) -> tuple[list[str], list[str]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(records)
    sample = records[:limit]
    md = [
        f"# Preset lab — real library sample ({len(sample)} of {len(records)} movies)",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} · dataset `{path}`",
        "",
        "Per movie: each preset's decision and the GiB/h it would land on, via the real decide() "
        "over all gathered candidates (incl. remux). Lets you eyeball the per-preset size tables "
        "on real releases.",
        "",
    ]
    summaries: list[str] = []
    for rec in sample:
        lines, summary = render_dataset_item(t, rec)
        md.extend(lines)
        summaries.extend(summary)
    return md, summaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, help="JSONL from gather_training_data.py")
    ap.add_argument("--limit", type=int, default=25, help="movies to sample in --dataset mode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = Topsis(default_topsis())
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.dataset:
        md, summaries = run_dataset(t, args.dataset, args.limit, args.seed)
        out = Path("reports") / f"weight-lab-dataset-{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md) + "\n")
        print(f"wrote {out}")
        print("\n".join(summaries))
        return

    md: list[str] = [
        "# Preset lab",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Each preset carries its own 5-point size table {floor, lo, target, hi, ceiling} GiB/h,",
        "weights (score + size), and a pick method. A candidate is grabbed only if it raises",
        "closeness by at least min_closeness_gain (and does not drop resolution below target, and",
        "is not bigger at a lower-or-equal score). ★ = the preset's pick; 'drop' = excluded by",
        "that preset's size band (floor..ceiling) or the score gap-cut.",
        "",
        "## Presets",
        "",
        "| preset | score | size | pick | gain | 2160 (floor/lo/target/hi/ceiling) | 1080 | 720 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def _band(p, res: int) -> str:
        pts = p.reference.get(res)
        return "/".join(f"{v:g}" for v in pts) if pts else "-"

    for name, p in t.cfg.presets.items():
        w = p.weights
        md.append(
            f"| {name} | {w['score']:.2f} | {w['size']:.2f} | "
            f"{p.pick} | {p.min_closeness_gain:g} | {_band(p, 2160)} | {_band(p, 1080)} | "
            f"{_band(p, 720)} |"
        )
    md.append("")
    md.append("# Part 1 — preset comparison on curated scenarios")
    md.append("")

    summaries: list[str] = []
    for sc in SCENARIOS:
        lines, summary = render_scenario(t, sc)
        md.extend(lines)
        summaries.extend(summary)

    md.append("# Part 2 — retention at scale")
    md.append("")
    summaries.append("  --- Part 2: retention ---")
    rng = random.Random(7)
    md.extend(render_retention(t, "Large popular-title pool", _make_popular(rng)))
    md.extend(render_retention(t, "Small obscure-title pool", _make_obscure()))

    out = Path("reports") / f"weight-lab-{ts}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
