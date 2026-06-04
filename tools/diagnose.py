"""Unified diagnostic + balance report for the optimizer, run offline against a harvested dataset.

Drives the REAL engine (decide / Topsis / default_topsis) over the training JSONL and writes three
markdown reports to reports/ (shared timestamp). Read-only; never touches Radarr/Sonarr or config.

  diagnose_picks_<ts>.md  one row per movie: the current file + what each preset would pick.
                          Compact overview. Full release titles, never cropped.
  diagnose_full_<ts>.md   per-movie drill-down: the current file, every release (score / res /
                          GiB-h / size, uncropped title), and each preset's ACT/HOLD + reason +
                          deltas. This is the "why is X misbehaving" view.
  diagnose_stats_<ts>.md  aggregate balance stats per preset: ACT/HOLD, grow vs shrink, size and
                          score deltas with dispersion, library size shift (inflate guard on/off),
                          and the swaps the no-bigger-at-lower-score rule blocked.
  diagnose_<ts>.html      a single self-contained interactive page (Picks / Releases / Stats tabs)
                          with click-to-sort, per-table text filter, and no pagination. Open it in
                          a browser; markdown viewers choke on the wide tables.

    uv run python tools/diagnose.py --in reports/training_data_radarr.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.decision import decide, resolution_ok
from optimizarr.features.optimizer.topsis import Topsis


def _cand(r: dict) -> dict:
    return {
        "guid": r.get("title"),
        "indexerId": 1,
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


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _best_unconstrained(t: Topsis, releases, runtime, preset, target_res, cur):
    """Highest-closeness candidate clearing the resolution guard + margin, IGNORING the
    no-bigger-at-lower-score rule. Returns (rel, attrs, clo, cur_clo) or None."""
    resolved = t.resolve_profile(preset)
    scored, _ = t.score_candidates(releases, runtime, resolved, target_res)
    cur_clo, cur_raw = t.closeness_for_current_file(cur, runtime, resolved, target_res)
    cur_res = cur_raw.get("resolution", 0) or 0
    for rel, attrs, clo in scored:  # sorted best-first
        if not resolution_ok(cur_res, attrs["raw"]["resolution"], target_res):
            continue
        if cur_clo is not None and clo < cur_clo + resolved.min_closeness_gain:
            continue
        return rel, attrs, clo, cur_clo
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="reports/training_data_radarr.jsonl", type=Path)
    ap.add_argument("--out-dir", default="reports", type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.inp.read_text().splitlines() if line.strip()]
    t = Topsis(default_topsis())
    presets = list(t.cfg.presets)  # Remux, Quality, Balanced, Efficient, Compact

    # ----- run the engine: per movie, per preset -----
    # records[i] = {title, cur, runtime, target_res, releases_raw, picks{preset: info}}
    records = []
    for r in rows:
        cf = r.get("current_file")
        cur = _cur(cf)
        rels = [_cand(x) for x in r["releases"]]
        runtime = r["runtime_h"]
        target_res = r["profile"]["target_resolution"] or (cf or {}).get("resolution") or 0
        picks = {}
        for p in presets:
            d = decide(t, rels, runtime, p, target_res, current_file=cur)
            info = {"action": d.action, "reason": d.reason}
            if d.action == "ACT" and d.pick:
                info["pick"] = d.pick
            # would the no-inflate rule have blocked a bigger-lower-score swap?
            bu = _best_unconstrained(t, rels, runtime, p, target_res, cur) if cur else None
            if bu and cf:
                rel, attrs, clo, _cc = bu
                bigger = (rel.get("size", 0)) > (cf.get("size_bytes") or 0)
                lower = (attrs["raw"]["score"] or 0) <= (cf.get("score") or 0)
                if bigger and lower:
                    info["blocked_inflation"] = {
                        "title": rel.get("title"),
                        "score": attrs["raw"]["score"],
                        "res": attrs["raw"]["resolution"],
                        "size_gb": attrs["raw"]["size_gb"],
                        "res_upgrade": attrs["raw"]["resolution"] > (cf.get("resolution") or 0),
                    }
            picks[p] = info

        # Library-shift: decide under the movie's OWN profile, with and without the inflate guard.
        own = r["profile"]["name"]
        d_on = decide(t, rels, runtime, own, target_res, current_file=cur)
        d_off = decide(
            t, rels, runtime, own, target_res, current_file=cur, allow_larger_at_lower_score=True
        )
        records.append(
            {
                "title": r["title"],
                "profile": own,
                "target_res": target_res,
                "cur": cf,
                "releases": r["releases"],
                "picks": picks,
                "own_on": d_on.pick if d_on.action == "ACT" else None,
                "own_off": d_off.pick if d_off.action == "ACT" else None,
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(exist_ok=True)
    _write_picks(records, presets, args.out_dir / f"diagnose_picks_{ts}.md")
    _write_full(records, presets, args.out_dir / f"diagnose_full_{ts}.md")
    _write_stats(records, presets, args.out_dir / f"diagnose_stats_{ts}.md")
    _write_html(records, presets, args.out_dir / f"diagnose_{ts}.html")
    print(f"wrote diagnose_{{picks,full,stats}}_{ts}.md + diagnose_{ts}.html to {args.out_dir}/")


def _fmt_cur(cf: dict | None) -> str:
    if not cf or cf.get("score") is None:
        return "no current file"
    return (
        f"score={cf['score']:,} res={cf.get('resolution') or '?'}p "
        f"size={cf.get('size_gb', 0):.1f}GB ({cf.get('gbh', 0):.2f} GiB/h)"
    )


def _pick_cell(info: dict) -> str:
    if info["action"] != "ACT" or "pick" not in info:
        flag = " [inflation blocked]" if "blocked_inflation" in info else ""
        return f"HOLD{flag}"
    p = info["pick"]
    return f"{p.get('score', 0):,} / {p.get('size_gb', 0):.1f}GB / {p.get('gbh', 0):.2f}gbh"


def _write_picks(records, presets, out: Path) -> None:
    lines = [
        "# Diagnose: picks overview",
        f"\n{len(records)} movies. Each cell = the pick (score / size / GiB-h), or HOLD.\n",
        "| movie | profile | current | " + " | ".join(presets) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in presets) + " |",
    ]
    for rec in records:
        cells = " | ".join(_pick_cell(rec["picks"][p]) for p in presets)
        lines.append(f"| {rec['title']} | {rec['profile']} | {_fmt_cur(rec['cur'])} | {cells} |")
    out.write_text("\n".join(lines) + "\n")


def _write_full(records, presets, out: Path) -> None:
    lines = ["# Diagnose: full drill-down", ""]
    for rec in records:
        lines.append(
            f"## {rec['title']}  ·  profile: {rec['profile']} (target {rec['target_res']}p)"
        )
        lines.append(f"\n**Current file:** {_fmt_cur(rec['cur'])}\n")
        # releases (score >= 0), sorted by score desc, uncropped titles
        rels = sorted(
            (x for x in rec["releases"] if (x.get("score") or -1) >= 0),
            key=lambda x: -(x.get("score") or 0),
        )
        neg = sum(1 for x in rec["releases"] if (x.get("score") or 0) < 0)
        lines.append(f"**Releases** ({len(rels)} with score>=0, {neg} negative hidden):\n")
        lines.append("| score | res | GiB/h | size | rej | title |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for x in rels:
            rj = "Y" if (x.get("rejections") or x.get("temporarily_rejected")) else ""
            lines.append(
                f"| {x.get('score', 0):,} | {x.get('resolution') or '?'}p | "
                f"{x.get('gbh', 0):.2f} | {x.get('size_gb', 0):.1f}GB | {rj} | {x.get('title')} |"
            )
        lines.append("\n**Per-preset decision:**\n")
        lines.append("| preset | action | pick | Δscore | Δsize | note |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        cf = rec["cur"] or {}
        for p in presets:
            info = rec["picks"][p]
            note = ""
            if "blocked_inflation" in info:
                bi = info["blocked_inflation"]
                kind = "RES-UPGRADE" if bi["res_upgrade"] else "same-res"
                note = (
                    f"would inflate ({kind}): {bi['score']:,}/{bi['size_gb']:.1f}GB {bi['title']}"
                )
            if info["action"] == "ACT" and "pick" in info:
                pk = info["pick"]
                ds = (pk.get("score") or 0) - (cf.get("score") or 0)
                dz = (pk.get("size_gb") or 0) - (cf.get("size_gb") or 0)
                lines.append(
                    f"| {p} | ACT | {pk.get('title', '?')} | {ds:+,} | {dz:+.1f}GB | {note} |"
                )
            else:
                lines.append(f"| {p} | HOLD | - | - | - | {note} |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n")


def _gb(x: float) -> str:
    return f"{x / 1024:.2f} TB" if abs(x) >= 1024 else f"{x:.0f} GB"


# ----- stat computation (shared by the markdown + HTML renderers) -----


def _shift_data(records) -> dict:
    """Library size change per movie under its OWN profile, inflate guard ON vs OFF."""
    d = dict(cur=0.0, on=0.0, off=0.0, on_g=0.0, on_s=0.0, off_g=0.0, off_s=0.0, movies=0, pts=0)
    examples = []
    for rec in records:
        cf = rec["cur"]
        if not cf or cf.get("score") is None:
            continue
        cz = cf.get("size_gb") or 0
        on_gb = (rec["own_on"] or {}).get("size_gb", cz) if rec["own_on"] else cz
        off_gb = (rec["own_off"] or {}).get("size_gb", cz) if rec["own_off"] else cz
        d["cur"] += cz
        d["on"] += on_gb
        d["off"] += off_gb
        d["on_g"] += max(0.0, on_gb - cz)
        d["on_s"] += max(0.0, cz - on_gb)
        d["off_g"] += max(0.0, off_gb - cz)
        d["off_s"] += max(0.0, cz - off_gb)
        if (
            rec["own_off"]
            and off_gb > cz
            and (rec["own_off"].get("score") or 0) < (cf["score"] or 0)
            and off_gb > on_gb + 1e-6
        ):
            d["movies"] += 1
            drop = (cf["score"] or 0) - (rec["own_off"].get("score") or 0)
            d["pts"] += drop
            examples.append((drop, off_gb - cz, rec["title"]))
    d["examples"] = sorted(examples, reverse=True)
    return d


def _preset_data(records, presets) -> list[dict]:
    out = []
    for p in presets:
        act = hold = grew = shrank = 0
        ds: list[float] = []
        dsc: list[float] = []
        for rec in records:
            info = rec["picks"][p]
            cf = rec["cur"]
            if info["action"] != "ACT" or "pick" not in info:
                hold += 1
                continue
            act += 1
            if not cf:
                continue
            pk = info["pick"]
            cz, ps = cf.get("size_gb") or 0, cf.get("score") or 0
            if pk.get("size_gb", 0) > cz:
                grew += 1
            elif pk.get("size_gb", 0) < cz:
                shrank += 1
            if cz > 0:
                ds.append(100 * (pk.get("size_gb", 0) - cz) / cz)
            dsc.append((pk.get("score") or 0) - ps)
        out.append(
            dict(preset=p, act=act, hold=hold, grew=grew, shrank=shrank, dsize=ds, dscore=dsc)
        )
    return out


def _inflation_data(records, presets):
    rows = []
    ups = []
    for p in presets:
        tot = same = up = 0
        for rec in records:
            bi = rec["picks"][p].get("blocked_inflation")
            if not bi:
                continue
            tot += 1
            if bi["res_upgrade"]:
                up += 1
                ups.append((p, rec, bi))
            else:
                same += 1
        rows.append(dict(preset=p, total=tot, same=same, up=up))
    return rows, ups


def _trip(v: list[float], fmt: str) -> str:
    return "-" if not v else f"{_pct(v, 10):{fmt}} / {_pct(v, 50):{fmt}} / {_pct(v, 90):{fmt}}"


def _write_stats(records, presets, out: Path) -> None:
    lines = ["# Diagnose: balance stats", f"\n{len(records)} movies.\n"]
    sh = _shift_data(records)
    cur = sh["cur"] or 1

    def pct(x: float) -> str:
        return f"{100 * x / cur:+.1f}%"

    lines.append("## Library size shift (every movie realigned to its OWN profile)\n")
    lines.append(f"Current library (movies with a file): **{_gb(sh['cur'])}**\n")
    lines.append("| inflate guard | new total | net shift | total grown | total shrunk |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| ON (ship default) | {_gb(sh['on'])} | {_gb(sh['on'] - sh['cur'])} "
        f"({pct(sh['on'] - sh['cur'])}) | +{_gb(sh['on_g'])} | -{_gb(sh['on_s'])} |"
    )
    lines.append(
        f"| OFF (drop the rule) | {_gb(sh['off'])} | {_gb(sh['off'] - sh['cur'])} "
        f"({pct(sh['off'] - sh['cur'])}) | +{_gb(sh['off_g'])} | -{_gb(sh['off_s'])} |"
    )
    lines.append(
        f"\n**Cost of dropping the rule:** +{_gb(sh['off'] - sh['on'])} extra growth, across "
        f"{sh['movies']} movies that would grow at a lower score "
        f"(total score regression {sh['pts']:,} points).\n"
    )
    if sh["examples"]:
        lines.append("Biggest score regressions the rule prevents (drop / growth / movie):\n")
        for drop, grow, title in sh["examples"][:15]:
            lines.append(f"- −{drop:,} pts, +{grow:.1f} GB — {title}")

    lines.append("\n## Per-preset picks (vs the current file)\n")
    lines.append(
        "| preset | ACT | HOLD | grew | shrank | Δsize% p10/p50/p90 | Δscore p10/p50/p90 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for d in _preset_data(records, presets):
        lines.append(
            f"| {d['preset']} | {d['act']} | {d['hold']} | {d['grew']} | {d['shrank']} | "
            f"{_trip(d['dsize'], '+.0f')} | {_trip(d['dscore'], '+,.0f')} |"
        )

    rows, ups = _inflation_data(records, presets)
    lines.append("\n## Swaps blocked by the no-bigger-at-lower-score rule\n")
    lines.append(
        "| preset | total blocked | same-res (correctly blocked) | RES-UPGRADE (maybe allow) |"
    )
    lines.append("| --- | ---: | ---: | ---: |")
    for d in rows:
        lines.append(f"| {d['preset']} | {d['total']} | {d['same']} | {d['up']} |")
    if ups:
        lines.append("\n### Resolution-upgrade cases the rule blocks (the legitimate ones)\n")
        lines.append("| preset | movie | current | blocked candidate |")
        lines.append("| --- | --- | --- | --- |")
        for p, rec, bi in ups[:40]:
            lines.append(
                f"| {p} | {rec['title']} | {_fmt_cur(rec['cur'])} | "
                f"{bi['score']:,} / {bi['res']}p / {bi['size_gb']:.1f}GB {bi['title']} |"
            )
    out.write_text("\n".join(lines) + "\n")


# ----- interactive HTML -----

_HTML_HEAD = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Optimizarr diagnostics</title><style>
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
table.tbl th:hover{{background:#374151}} table.tbl th.num,table.tbl td.num{{text-align:right}}
table.tbl tbody tr:nth-child(even){{background:#f9fafb}}
table.tbl tbody tr:hover{{background:#eef2ff}}
.hold{{color:#9ca3af}} .holdblk{{color:#b45309;font-weight:600}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0}}
.card{{border:1px solid #e5e7eb;border-radius:8px;padding:.8rem 1rem;min-width:180px}}
.card .big{{font-size:1.4rem;font-weight:700}} .pos{{color:#dc2626}} .neg{{color:#059669}}
ul.ex li{{margin:.1rem 0}}
</style></head><body>
<h1>Optimizarr diagnostics</h1>
<div class=sub>{n} movies &middot; generated {ts}</div>
<div class=tabs>{tabbtns}</div>
"""

_HTML_TAIL = """<script>
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('section.tab').forEach(s=>s.hidden=true);
  b.classList.add('active'); document.getElementById(b.dataset.tab).hidden=false;});
document.querySelectorAll('input.filter').forEach(inp=>{let h;inp.oninput=()=>{
  clearTimeout(h);h=setTimeout(()=>{
    const q=inp.value.toLowerCase();
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
  [...tb.rows].sort((a,b)=>{const x=val(a),y=val(b),nx=parseFloat(x),ny=parseFloat(y);
    return (!isNaN(nx)&&!isNaN(ny)&&x!==''&&y!=='')?(nx-ny)*dir:x.localeCompare(y)*dir;})
   .forEach(r=>tb.appendChild(r));});
document.querySelector('.tabs button').click();
</script></body></html>"""


def _cell(disp, sort=None, cls="") -> str:
    klass = f' class="{cls}"' if cls else ""
    ds = f' data-sort="{html.escape(str(sort))}"' if sort is not None else ""
    return f"<td{klass}{ds}>{html.escape(str(disp))}</td>"


def _tbl(tid: str, cols: list[tuple[str, bool]], rows: list[list[str]]) -> str:
    head = "".join(f'<th class="{"num" if num else ""}">{html.escape(c)}</th>' for c, num in cols)
    body = "".join("<tr>" + "".join(r) + "</tr>" for r in rows)
    return (
        f'<input class=filter data-target="{tid}" placeholder="filter {len(rows)} rows…">'
        f'<table id="{tid}" class=tbl><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    )


def _write_html(records, presets, out: Path) -> None:
    initials = {p: p[0] for p in presets}

    # --- Picks tab ---
    pcols = [
        ("Movie", False),
        ("Profile", False),
        ("Cur score", True),
        ("Cur res", True),
        ("Cur GB", True),
        ("Cur GiB/h", True),
    ] + [(p, True) for p in presets]
    prows = []
    for rec in records:
        cf = rec["cur"] or {}
        cells = [
            _cell(rec["title"], cls="txt"),
            _cell(rec["profile"]),
            _cell(f"{cf.get('score'):,}" if cf.get("score") is not None else "-", cf.get("score")),
            _cell(f"{cf.get('resolution') or '?'}p", cf.get("resolution") or 0, "num"),
            _cell(f"{cf.get('size_gb', 0):.1f}", cf.get("size_gb") or 0, "num"),
            _cell(f"{cf.get('gbh', 0):.2f}", cf.get("gbh") or 0, "num"),
        ]
        for p in presets:
            info = rec["picks"][p]
            if info["action"] == "ACT" and "pick" in info:
                pk = info["pick"]
                cells.append(
                    _cell(
                        f"{pk.get('score', 0):,} / {pk.get('size_gb', 0):.1f}GB",
                        pk.get("score") or 0,
                        "num",
                    )
                )
            else:
                blk = "blocked_inflation" in info
                cells.append(
                    _cell("HOLD*" if blk else "HOLD", -1, "num holdblk" if blk else "num hold")
                )
        prows.append(cells)
    picks_html = _tbl("picks-tbl", pcols, prows)

    # --- Releases tab (flat, score >= 0) ---
    rcols = [
        ("Movie", False),
        ("Profile", False),
        ("Cur GB", True),
        ("Cur score", True),
        ("Rel score", True),
        ("Res", True),
        ("GiB/h", True),
        ("Rel GB", True),
        ("Rej", False),
        ("Picked by", False),
        ("Title", False),
    ]
    rrows = []
    for rec in records:
        cf = rec["cur"] or {}
        for x in rec["releases"]:
            if (x.get("score") or -1) < 0:
                continue
            pby = "".join(
                initials[p]
                for p in presets
                if rec["picks"][p].get("pick", {}).get("title") == x.get("title")
            )
            rrows.append(
                [
                    _cell(rec["title"], cls="txt"),
                    _cell(rec["profile"]),
                    _cell(f"{cf.get('size_gb', 0):.1f}", cf.get("size_gb") or 0, "num"),
                    _cell(f"{cf.get('score') or 0:,}", cf.get("score") or 0, "num"),
                    _cell(f"{x.get('score', 0):,}", x.get("score") or 0, "num"),
                    _cell(f"{x.get('resolution') or '?'}p", x.get("resolution") or 0, "num"),
                    _cell(f"{x.get('gbh', 0):.2f}", x.get("gbh") or 0, "num"),
                    _cell(f"{x.get('size_gb', 0):.1f}", x.get("size_gb") or 0, "num"),
                    _cell("Y" if (x.get("rejections") or x.get("temporarily_rejected")) else ""),
                    _cell(pby),
                    _cell(x.get("title"), cls="txt"),
                ]
            )
    releases_html = _tbl("rel-tbl", rcols, rrows)

    # --- Stats tab ---
    sh = _shift_data(records)
    cur = sh["cur"] or 1
    net_on, net_off = sh["on"] - sh["cur"], sh["off"] - sh["cur"]
    cards = (
        f"<div class=cards>"
        f"<div class=card>current library<div class=big>{_gb(sh['cur'])}</div></div>"
        f'<div class=card>guard ON<div class="big neg">{_gb(net_on)}</div>'
        f"{100 * net_on / cur:+.1f}%</div>"
        f'<div class=card>guard OFF<div class="big neg">{_gb(net_off)}</div>'
        f"{100 * net_off / cur:+.1f}%</div>"
        f"<div class=card>cost of dropping rule"
        f'<div class="big pos">+{_gb(sh["off"] - sh["on"])}</div>'
        f"{sh['movies']} movies · {sh['pts']:,} pts regression</div></div>"
    )
    shift_tbl = _tbl(
        "shift-tbl",
        [
            ("inflate guard", False),
            ("new total", True),
            ("net shift", True),
            ("net %", True),
            ("grown", True),
            ("shrunk", True),
        ],
        [
            [
                _cell("ON (ship default)"),
                _cell(_gb(sh["on"]), sh["on"], "num"),
                _cell(_gb(net_on), net_on, "num"),
                _cell(f"{100 * net_on / cur:+.1f}%", net_on, "num"),
                _cell("+" + _gb(sh["on_g"]), sh["on_g"], "num"),
                _cell("-" + _gb(sh["on_s"]), sh["on_s"], "num"),
            ],
            [
                _cell("OFF (drop the rule)"),
                _cell(_gb(sh["off"]), sh["off"], "num"),
                _cell(_gb(net_off), net_off, "num"),
                _cell(f"{100 * net_off / cur:+.1f}%", net_off, "num"),
                _cell("+" + _gb(sh["off_g"]), sh["off_g"], "num"),
                _cell("-" + _gb(sh["off_s"]), sh["off_s"], "num"),
            ],
        ],
    )
    ex = "".join(
        f"<li>−{d:,} pts, +{g:.1f} GB — {html.escape(t)}</li>" for d, g, t in sh["examples"][:25]
    )
    pp = _tbl(
        "pp-tbl",
        [
            ("preset", False),
            ("ACT", True),
            ("HOLD", True),
            ("grew", True),
            ("shrank", True),
            ("Δsize% p50", True),
            ("Δscore p50", True),
        ],
        [
            [
                _cell(d["preset"]),
                _cell(d["act"], d["act"], "num"),
                _cell(d["hold"], d["hold"], "num"),
                _cell(d["grew"], d["grew"], "num"),
                _cell(d["shrank"], d["shrank"], "num"),
                _cell(_trip(d["dsize"], "+.0f"), _pct(d["dsize"], 50) if d["dsize"] else 0, "num"),
                _cell(
                    _trip(d["dscore"], "+,.0f"), _pct(d["dscore"], 50) if d["dscore"] else 0, "num"
                ),
            ]
            for d in _preset_data(records, presets)
        ],
    )
    inf_rows, ups = _inflation_data(records, presets)
    inf = _tbl(
        "inf-tbl",
        [("preset", False), ("total blocked", True), ("same-res", True), ("RES-UPGRADE", True)],
        [
            [
                _cell(d["preset"]),
                _cell(d["total"], d["total"], "num"),
                _cell(d["same"], d["same"], "num"),
                _cell(d["up"], d["up"], "num"),
            ]
            for d in inf_rows
        ],
    )
    stats_html = (
        "<h2>Library size shift (every movie realigned to its own profile)</h2>"
        + cards
        + shift_tbl
        + (
            "<h2>Biggest score regressions the rule prevents</h2><ul class=ex>" + ex + "</ul>"
            if ex
            else ""
        )
        + "<h2>Per-preset picks (vs current file)</h2>"
        + pp
        + "<h2>Swaps blocked by the no-bigger-at-lower-score rule</h2>"
        + inf
    )

    tabs = [
        ("picks", "Picks", picks_html),
        ("releases", "Releases", releases_html),
        ("stats", "Stats", stats_html),
    ]
    tabbtns = "".join(f'<button data-tab="{tid}">{label}</button>' for tid, label, _ in tabs)
    head = _HTML_HEAD.format(
        n=len(records), ts=datetime.now().strftime("%Y-%m-%d %H:%M"), tabbtns=tabbtns
    )
    body = "".join(
        f'<section id="{tid}" class=tab hidden>{content}</section>' for tid, _, content in tabs
    )
    out.write_text(head + body + _HTML_TAIL)


if __name__ == "__main__":
    main()
