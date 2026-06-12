"""Offline quadrant report for the relative optimizer, run against a harvested dataset.

Drives the REAL engine (decide / Topsis / default_topsis) over the training JSONL, evaluating each
movie under ITS OWN profile only (Profilarr scoring is profile-specific, so cross-profile picks are
meaningless). For every ACT it classifies the pick vs the current file into one of four quadrants
(score up/down x size up/down), the key one being "score down + size up", which must never happen.

Writes two reports to reports/ (shared timestamp):
  diagnose_<ts>.md     summary (quadrants, ACT/HOLD split, total size shift) + the full
                       original -> new-pick list with a quadrant tag per row.
  diagnose_<ts>.html   a single self-contained interactive page (Summary / Picks / Candidates tabs)
                       with click-to-sort and a per-table filter. Click a movie in Picks to see all
                       its in-window candidates (and why each was/wasn't chosen). Open in a browser.

    uv run python tools/diagnose.py --in reports/training_data_radarr.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from optimizarr.features.optimizer.config import RetryConfig, default_topsis
from optimizarr.features.optimizer.decision import decide
from optimizarr.features.optimizer.topsis import HARD_REJECT_KEYWORDS, Topsis, _score_floor_tier


def _cand(r: dict) -> dict:
    return {
        "guid": r.get("title"),
        "title": r.get("title") or "?",
        "customFormatScore": r.get("score"),
        "quality": {"quality": {"resolution": r.get("resolution") or 0}},
        "size": r.get("size_bytes") or 0,
        "rejections": r.get("rejections") or [],
        "temporarilyRejected": bool(r.get("temporarily_rejected")),
    }


def _cur(cf: dict | None) -> dict | None:
    if not cf or cf.get("score") is None:
        return None
    return {
        "id": 1,
        "customFormatScore": cf.get("score"),
        "size": cf.get("size_bytes") or 0,
        "quality": {"quality": {"resolution": cf.get("resolution") or 0}},
    }


def _quadrant(ds: float, dz: float) -> str:
    s = "score+" if ds > 0 else "score-" if ds < 0 else "score="
    z = "size+" if dz > 0.05 else "size-" if dz < -0.05 else "size="
    return f"{s} {z}"


def _candidate_status(
    x: dict, t: Topsis, floor_score: int, target_res: int, pick_title: str | None
) -> str:
    """Why a release was or wasn't the pick, mirroring the engine's filters (for the drill-down)."""
    rej = x.get("rejections") or []
    if x.get("temporarily_rejected") or any(any(k in r for k in HARD_REJECT_KEYWORDS) for r in rej):
        return "rejected"
    s = x.get("score")
    if s is None or s < 0:
        return "neg-score"
    if s < floor_score:
        return "below-window"
    res = x.get("resolution") or 0
    if target_res and res != target_res:
        return "wrong-res"
    floor, ceiling = t.bounds_for(res)
    g = x.get("gbh") or 0.0
    if g < floor:
        return "below-floor"
    if g > ceiling:
        return "above-ceiling"
    return "PICK" if x.get("title") == pick_title else "eligible"


def _candidates(
    r: dict,
    t: Topsis,
    cur_score,
    target_res: int,
    pick_title: str | None,
    clo_map: dict[str, float],
) -> list[dict]:
    """In-window candidates for one movie, each annotated with status and closeness.
    Uses the same three-tier score floor as the engine so the drill-down exactly mirrors
    what got filtered. Closeness is None for candidates filtered out before scoring."""
    elig_scores = [
        x.get("score") or 0
        for x in r["releases"]
        if (x.get("score") or 0) >= 0
        and not x.get("temporarily_rejected")
        and not any(
            any(k in rj for k in HARD_REJECT_KEYWORDS) for rj in (x.get("rejections") or [])
        )
    ]
    floor_score, _ = _score_floor_tier(
        elig_scores, cur_score, t.cfg.score_window, t.cfg.min_candidates
    )
    out = []
    for x in r["releases"]:
        st = _candidate_status(x, t, floor_score, target_res, pick_title)
        if st in ("rejected", "neg-score", "below-window"):
            continue
        title = x.get("title") or "?"
        out.append(
            {
                "title": title,
                "score": x.get("score") or 0,
                "res": x.get("resolution") or 0,
                "gbh": x.get("gbh") or 0.0,
                "size_gb": x.get("size_gb") or 0.0,
                "status": st,
                "closeness": clo_map.get(title),  # None when filtered before scoring
            }
        )
    out.sort(key=lambda c: (-c["score"], c["gbh"]))
    return out


def _evaluate(rows: list[dict], t: Topsis) -> list[dict]:
    records = []
    for r in rows:
        cf = r.get("current_file")
        cur = _cur(cf)
        profile = r["profile"]["name"]
        target_res = r["profile"]["target_resolution"] or (cf or {}).get("resolution") or 0
        releases = [_cand(x) for x in r["releases"]]
        d = decide(
            t,
            releases,
            r["runtime_h"],
            profile,
            target_res,
            current_file=cur,
            satisfied_score=RetryConfig().satisfied_score,
        )

        # Run a scoring pass to get per-candidate closeness for the drill-down table.
        cur_score = (cf or {}).get("score")
        resolved = t.resolve_profile(profile)
        kept, _ = t.apply_prefilters(releases, r["runtime_h"], int(target_res or 0), cur_score)
        clo_map: dict[str, float] = {}
        if kept:
            scored_pool, _ = t.score_pool(kept, cur, r["runtime_h"], resolved)
            clo_map = {rel.get("title", "?"): clo for rel, _, clo in scored_pool}

        cur_clo = d.current.get("closeness") if d.current else None
        rec = {
            "title": r["title"],
            "profile": profile,
            "action": d.action,
            "satisfy": d.satisfy,
            "reason": d.reason,
            "cur": cf or {},
            "cur_clo": cur_clo,
            "pick": d.pick if d.action == "ACT" else None,
        }
        pick_title = d.pick.get("title") if d.action == "ACT" and d.pick else None
        rec["cands"] = _candidates(r, t, cur_score, int(target_res or 0), pick_title, clo_map)
        if rec["pick"] and cf:
            pk = rec["pick"]
            rec["dscore"] = (pk.get("score") or 0) - (cf.get("score") or 0)
            rec["dsize"] = (pk.get("size_gb") or 0) - (cf.get("size_gb") or 0)
            rec["dgbh"] = (pk.get("gbh") or 0) - (cf.get("gbh") or 0)
            pick_clo = pk.get("closeness")
            rec["dclo"] = (
                (pick_clo - cur_clo) if (pick_clo is not None and cur_clo is not None) else None
            )
            rec["quad"] = _quadrant(rec["dscore"], rec["dsize"])
        records.append(rec)
    return records


def _summary(records: list[dict]) -> dict:
    act = [r for r in records if r["action"] == "ACT"]
    quad: dict[str, int] = {}
    for r in act:
        quad[r["quad"]] = quad.get(r["quad"], 0) + 1
    return {
        "movies": len(records),
        "act": len(act),
        "hold_satisfied": sum(1 for r in records if r["action"] == "HOLD" and r["satisfy"]),
        "hold_insufficient": sum(1 for r in records if r["action"] == "HOLD" and not r["satisfy"]),
        "quad": quad,
        "size_shift_gb": sum(r.get("dsize", 0.0) for r in act),
        "bug": [r for r in act if r["dscore"] < 0 and r["dsize"] > 0.05],
    }


_MEANING = {
    "score+ size-": "better AND smaller (ideal)",
    "score+ size+": "score upgrade, bigger (justified)",
    "score+ size=": "score upgrade, same size",
    "score- size-": "traded a little score for a smaller file",
    "score= size-": "same score, smaller",
    "score- size+": "DOWNGRADE + BIGGER (must be 0)",
}


def _write_md(records: list[dict], s: dict, out: Path) -> None:
    L = [
        "# Diagnose: relative optimizer (own-profile quadrants)",
        f"\n{s['movies']} movies. ACT **{s['act']}** · HOLD-satisfied **{s['hold_satisfied']}** · "
        f"HOLD-too-few **{s['hold_insufficient']}**. Total size shift: "
        f"**{s['size_shift_gb'] / 1024:+.2f} TB**.\n",
        "## Pick quadrants (vs the current file, ACT only)\n",
        "| quadrant | count | meaning |",
        "| --- | ---: | --- |",
    ]
    for q in sorted(s["quad"], key=lambda k: -s["quad"][k]):
        L.append(f"| {q} | {s['quad'][q]} | {_MEANING.get(q, '')} |")
    L.append(f"\n**score-down & size-up (must be 0): {len(s['bug'])}**\n")
    for r in s["bug"]:
        L.append(f"- {r['title']}: Δscore {r['dscore']:+,} Δsize {r['dsize']:+.1f}GB")

    L.append("\n## Original file -> new pick (ACT only)\n")
    L.append(
        "| movie | profile | quadrant | current (score / GB / gb·h) | "
        "new pick (score / GB / gb·h) | Δscore | Δsize | Δgb·h | Δclo |"
    )
    L.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for r in records:
        if r["action"] != "ACT":
            continue
        cf, pk = r["cur"], r["pick"]
        dclo = r.get("dclo")
        dclo_s = f"{dclo:+.3f}" if dclo is not None else "-"
        L.append(
            f"| {r['title']} | {r['profile']} | {r['quad']} | "
            f"{cf.get('score', 0):,} / {cf.get('size_gb', 0):.1f} / {cf.get('gbh', 0):.2f} | "
            f"{pk.get('score', 0):,} / {pk.get('size_gb', 0):.1f} / {pk.get('gbh', 0):.2f} | "
            f"{r['dscore']:+,} | {r['dsize']:+.1f}GB | {r['dgbh']:+.2f} | {dclo_s} |"
        )
    out.write_text("\n".join(L) + "\n")


# ----- interactive HTML -----

_HEAD = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Optimizarr diagnose</title><style>
body{{font:13px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1rem;color:#1f2937}}
h1{{font-size:1.25rem;margin:.2rem 0}} h2{{font-size:1rem;margin:1.2rem 0 .3rem}}
.sub{{color:#6b7280;margin:.2rem 0 1rem}}
.tabs{{display:flex;gap:.25rem;border-bottom:2px solid #e5e7eb}}
.tabs button{{font:inherit;padding:.45rem .9rem;border:0;background:none;cursor:pointer;
color:#374151;border-bottom:2px solid transparent;margin-bottom:-2px}}
.tabs button.active{{color:#2563eb;border-bottom-color:#2563eb;font-weight:600}}
section.tab{{margin-top:1rem}}
input.filter{{display:block;margin:.6rem 0;padding:.45rem .6rem;width:min(420px,90vw);
font:inherit;border:1px solid #d1d5db;border-radius:6px}}
table.tbl{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
table.tbl th,table.tbl td{{border:1px solid #e5e7eb;padding:.25rem .5rem;text-align:left;
white-space:nowrap}}
table.tbl td.txt{{white-space:normal}}
table.tbl th{{position:sticky;top:0;background:#1f2937;color:#fff;cursor:pointer;
user-select:none;z-index:1}}
table.tbl th.num,table.tbl td.num{{text-align:right}}
table.tbl tbody tr:nth-child(even){{background:#f9fafb}}
table.tbl tbody tr:hover{{background:#eef2ff}}
.ideal{{color:#059669;font-weight:600}} .bug{{color:#dc2626;font-weight:700}} .ok{{color:#6b7280}}
td.link{{color:#2563eb;cursor:pointer;text-decoration:underline}}
.PICK{{color:#059669;font-weight:700}}
.wrong-res,.below-floor,.above-ceiling{{color:#b45309;font-weight:600}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0}}
.card{{border:1px solid #e5e7eb;border-radius:8px;padding:.7rem 1rem;min-width:140px}}
.card .big{{font-size:1.4rem;font-weight:700}}
</style></head><body><h1>Optimizarr diagnose</h1>
<div class=sub>{n} movies &middot; {ts} &middot; own-profile relative model</div>
<div class=tabs>{tabbtns}</div>
"""

_TAIL = """<script>
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('section.tab').forEach(s=>s.hidden=true);
  b.classList.add('active'); document.getElementById(b.dataset.tab).hidden=false;});
document.querySelectorAll('input.filter').forEach(inp=>{let h;inp.oninput=()=>{
  clearTimeout(h);h=setTimeout(()=>{const q=inp.value.toLowerCase();
    const rows=document.getElementById(inp.dataset.target).tBodies[0].rows;
    for(let i=0;i<rows.length;i++){
      rows[i].style.display=rows[i].textContent.toLowerCase().includes(q)?'':'none';}},150);};});
document.querySelectorAll('table.tbl th').forEach(th=>th.onclick=()=>{
  const tb=th.closest('table').tBodies[0],idx=[...th.parentNode.children].indexOf(th);
  const dir=th.dataset.dir==='asc'?-1:1;
  th.closest('table').querySelectorAll('th').forEach(h=>delete h.dataset.dir);
  th.dataset.dir=dir===1?'asc':'desc';
  const val=tr=>{const c=tr.cells[idx];
    return c.dataset.sort!==undefined?c.dataset.sort:c.textContent.trim();};
  const num=v=>v!==''&&!isNaN(Number(v));
  [...tb.rows].sort((a,b)=>{const x=val(a),y=val(b);
    return (num(x)&&num(y))?(Number(x)-Number(y))*dir:x.localeCompare(y)*dir;})
   .forEach(r=>tb.appendChild(r));});
// Click a movie -> jump to the Candidates tab filtered to that movie.
document.querySelectorAll('td.link').forEach(td=>td.onclick=()=>{
  document.querySelector('.tabs button[data-tab="cands"]').click();
  const inp=document.querySelector('input.filter[data-target="c-tbl"]');
  inp.value=td.textContent; inp.dispatchEvent(new Event('input'));});
document.querySelector('.tabs button').click();
</script></body></html>"""


def _cell(disp, sort=None, cls="") -> str:
    klass = f' class="{cls}"' if cls else ""
    ds = f' data-sort="{html.escape(str(sort))}"' if sort is not None else ""
    return f"<td{klass}{ds}>{html.escape(str(disp))}</td>"


def _tbl(tid: str, cols: list[tuple[str, bool]], rows: list[list[str]], filt: bool = True) -> str:
    head = "".join(f'<th class="{"num" if num else ""}">{html.escape(c)}</th>' for c, num in cols)
    body = "".join("<tr>" + "".join(r) + "</tr>" for r in rows)
    f = (
        f'<input class=filter data-target="{tid}" placeholder="filter {len(rows)} rows…">'
        if filt
        else ""
    )
    return (
        f'{f}<table id="{tid}" class=tbl><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _clo_cell(v: float | None, cls: str = "num") -> str:
    if v is None:
        return _cell("-", -999, cls)
    return _cell(f"{v:.3f}", v, cls)


def _write_html(records: list[dict], s: dict, out: Path) -> None:
    cards = (
        f"<div class=cards>"
        f"<div class=card>ACT<div class=big>{s['act']}</div></div>"
        f"<div class=card>satisfied (optimal)<div class=big>{s['hold_satisfied']}</div></div>"
        f"<div class=card>too few candidates<div class=big>{s['hold_insufficient']}</div></div>"
        f"<div class=card>size shift"
        f'<div class="big ideal">{s["size_shift_gb"] / 1024:+.2f} TB</div></div>'
        f'<div class=card>score-/size+ (bug)<div class="big bug">{len(s["bug"])}</div></div></div>'
    )
    qrows = [
        [_cell(q), _cell(s["quad"][q], s["quad"][q], "num"), _cell(_MEANING.get(q, ""), cls="txt")]
        for q in sorted(s["quad"], key=lambda k: -s["quad"][k])
    ]
    summary = (
        "<h2>Outcome</h2>"
        + cards
        + "<h2>Pick quadrants (ACT vs current)</h2>"
        + _tbl(
            "q-tbl", [("quadrant", False), ("count", True), ("meaning", False)], qrows, filt=False
        )
    )

    cls = {"score+ size-": "ideal", "score- size+": "bug"}
    cols = [
        ("Movie", False),
        ("Profile", False),
        ("Quadrant", False),
        ("Cur res", True),
        ("Cur score", True),
        ("Cur GB", True),
        ("Cur gb/h", True),
        ("Cur clo", True),
        ("Pick score", True),
        ("Pick GB", True),
        ("Pick gb/h", True),
        ("Pick clo", True),
        ("Δscore", True),
        ("Δsize GB", True),
        ("Δgb/h", True),
        ("Δclo", True),
    ]
    prows = []
    for r in records:
        if r["action"] != "ACT":
            continue
        cf, pk = r["cur"], r["pick"]
        cur_clo = r.get("cur_clo")
        pick_clo = pk.get("closeness") if pk else None
        dclo = r.get("dclo")
        prows.append(
            [
                _cell(r["title"], cls="txt link"),
                _cell(r["profile"]),
                _cell(r["quad"], cls=cls.get(r["quad"], "ok")),
                _cell(f"{cf.get('resolution', 0)}p", cf.get("resolution") or 0, "num"),
                _cell(f"{cf.get('score', 0):,}", cf.get("score") or 0, "num"),
                _cell(f"{cf.get('size_gb', 0):.1f}", cf.get("size_gb") or 0, "num"),
                _cell(f"{cf.get('gbh', 0):.2f}", cf.get("gbh") or 0, "num"),
                _clo_cell(cur_clo),
                _cell(f"{pk.get('score', 0):,}", pk.get("score") or 0, "num"),
                _cell(f"{pk.get('size_gb', 0):.1f}", pk.get("size_gb") or 0, "num"),
                _cell(f"{pk.get('gbh', 0):.2f}", pk.get("gbh") or 0, "num"),
                _clo_cell(pick_clo),
                _cell(f"{r['dscore']:+,}", r["dscore"], "num"),
                _cell(f"{r['dsize']:+.1f}", r["dsize"], "num"),
                _cell(f"{r['dgbh']:+.2f}", r["dgbh"], "num"),
                _cell(
                    f"{dclo:+.3f}" if dclo is not None else "-",
                    dclo if dclo is not None else -999,
                    "num",
                ),
            ]
        )
    picks = "<div class=sub>Click a movie title to see all its in-window candidates.</div>" + _tbl(
        "p-tbl", cols, prows
    )

    # Candidates drill-down: every in-window candidate for every movie, annotated.
    ccols = [
        ("Movie", False),
        ("Profile", False),
        ("Status", False),
        ("Score", True),
        ("Res", True),
        ("gb/h", True),
        ("GB", True),
        ("Closeness", True),
    ]
    crows = []
    for r in records:
        for c in r["cands"]:
            crows.append(
                [
                    _cell(r["title"], cls="txt"),
                    _cell(r["profile"]),
                    _cell(c["status"], cls=c["status"]),
                    _cell(f"{c['score']:,}", c["score"], "num"),
                    _cell(f"{c['res']}p", c["res"], "num"),
                    _cell(f"{c['gbh']:.2f}", c["gbh"], "num"),
                    _cell(f"{c['size_gb']:.1f}", c["size_gb"], "num"),
                    _clo_cell(c.get("closeness")),
                ]
            )
    cands_tab = (
        "<div class=sub>In-window candidates (score within the downgrade budget). "
        "<b>PICK</b> = chosen; wrong-res / below-floor / above-ceiling = filtered out. "
        "Closeness is relative to the surviving pool (blank when filtered before scoring).</div>"
        + _tbl("c-tbl", ccols, crows)
    )

    tabs = [
        ("summary", "Summary", summary),
        ("picks", "Picks (ACT)", picks),
        ("cands", "Candidates", cands_tab),
    ]
    tabbtns = "".join(f'<button data-tab="{i}">{lbl}</button>' for i, lbl, _ in tabs)
    head = _HEAD.format(
        n=s["movies"], ts=datetime.now().strftime("%Y-%m-%d %H:%M"), tabbtns=tabbtns
    )
    body = "".join(f'<section id="{i}" class=tab hidden>{c}</section>' for i, _, c in tabs)
    out.write_text(head + body + _TAIL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="reports/training_data_radarr.jsonl", type=Path)
    ap.add_argument("--out-dir", default="reports", type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.inp.read_text().splitlines() if line.strip()]
    t = Topsis(default_topsis())
    records = _evaluate(rows, t)
    s = _summary(records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(exist_ok=True)
    _write_md(records, s, args.out_dir / f"diagnose_{ts}.md")
    _write_html(records, s, args.out_dir / f"diagnose_{ts}.html")
    print(
        f"wrote diagnose_{ts}.{{md,html}} to {args.out_dir}/ — "
        f"ACT={s['act']} satisfied={s['hold_satisfied']} too-few={s['hold_insufficient']} "
        f"shift={s['size_shift_gb'] / 1024:+.2f}TB bug={len(s['bug'])}"
    )


if __name__ == "__main__":
    main()
