from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.decision import decide, format_decision
from optimizarr.features.optimizer.topsis import GB, Topsis


def _topsis() -> Topsis:
    cfg = default_topsis()
    cfg.min_candidates = 2  # decision tests use 2-3 candidates; pool-size is tested in test_topsis
    return Topsis(cfg)


def _release(guid="g1", score=900_000, resolution=2160, size_gb=20.0):
    return {
        "guid": guid,
        "title": f"Movie.{resolution}p.{guid}",
        "customFormatScore": score,
        "quality": {"quality": {"resolution": resolution}},
        "size": int(size_gb * GB),
        "rejections": [],
    }


def _file(score=200_000, resolution=2160, size_gb=30.0):
    return {
        "id": 555,
        "customFormatScore": score,
        "size": int(size_gb * GB),
        "quality": {"quality": {"resolution": resolution}},
    }


def test_act_on_clear_upgrade():
    # Current is a bloated low-score file; two good candidates exist -> ACT on the best.
    rels = [_release("a", 1_000_000, 2160, 13.0), _release("b", 950_000, 2160, 18.0)]
    d = decide(_topsis(), rels, 2.0, "2160p Quality", 2160, current_file=_file(200_000, 2160, 30.0))
    assert d.action == "ACT" and d.release is not None and d.release["guid"] == "a"
    msg = format_decision("radarr", "Movie (2024)", d, dry_run=True)
    assert "would GRAB" in msg and "Δsize" in msg


def test_too_few_candidates_holds_without_satisfying():
    d = decide(_topsis(), [_release()], 2.0, "2160p Quality", 2160, current_file=_file())
    assert d.action == "HOLD" and d.satisfy is False and d.insufficient is True


def test_too_few_candidates_satisfies_when_current_score_above_threshold():
    # Only one candidate (too few), but the current file already scores above the threshold ->
    # the file is good enough on its own, so satisfy instead of an insufficient retry.
    cur = _file(score=900_000, resolution=2160, size_gb=20.0)
    d = decide(_topsis(), [_release()], 2.0, "2160p Quality", 2160, cur, satisfied_score=800_000)
    assert d.action == "HOLD" and d.satisfy is True and d.insufficient is False


def test_too_few_candidates_below_threshold_stays_insufficient():
    cur = _file(score=700_000, resolution=2160, size_gb=20.0)
    d = decide(_topsis(), [_release()], 2.0, "2160p Quality", 2160, cur, satisfied_score=800_000)
    assert d.action == "HOLD" and d.satisfy is False and d.insufficient is True


def test_hold_and_satisfy_when_current_optimal():
    cur = _file(1_000_000, 2160, 6.0)  # best score, small
    rels = [_release("same", 1_000_000, 2160, 6.0), _release("worse", 900_000, 2160, 25.0)]
    d = decide(_topsis(), rels, 2.0, "2160p Quality", 2160, current_file=cur)
    assert d.action == "HOLD" and d.satisfy is True


def test_efficient_picks_smaller_quality_picks_higher_score():
    # Same candidate set, different profile -> different pick (the relative model's whole point).
    cur = _file(900_000, 2160, 24.0)  # 12 GiB/h
    small = _release("small", 927_000, 2160, 15.2)  # ~7.6 GiB/h, slightly lower score
    big = _release("big", 950_600, 2160, 24.9)  # ~12.5 GiB/h, top score
    eff = decide(_topsis(), [small, big], 2.0, "2160p Efficient", 2160, current_file=cur)
    qual = decide(_topsis(), [small, big], 2.0, "2160p Quality", 2160, current_file=cur)
    assert eff.action == "ACT" and eff.release["guid"] == "small"
    assert qual.action == "ACT" and qual.release["guid"] == "big"


def test_pick_never_lower_score_and_bigger_than_current():
    cur = _file(900_000, 2160, 8.0)  # already small and decent
    # Only worse-on-both candidates -> must HOLD, never grab a lower-score bigger file.
    rels = [_release("x", 850_000, 2160, 20.0), _release("y", 800_000, 2160, 25.0)]
    d = decide(_topsis(), rels, 2.0, "2160p Quality", 2160, current_file=cur)
    assert d.action == "HOLD"


def test_allow_size_increase_false_drops_bigger():
    cur = _file(400_000, 2160, 20.0)
    bigger = _release("big", 1_000_000, 2160, 30.0)  # dropped (> current)
    small = _release("small", 900_000, 2160, 13.0)
    mid = _release("mid", 850_000, 2160, 18.0)
    d = decide(
        _topsis(),
        [bigger, small, mid],
        2.0,
        "2160p Quality",
        2160,
        current_file=cur,
        allow_size_increase=False,
    )
    assert d.action == "ACT" and d.release["guid"] == "small"  # best of the two survivors


def test_allow_quality_downgrade_false_drops_lower_score():
    cur = _file(800_000, 2160, 28.0)
    higher = _release("hi", 1_000_000, 2160, 22.0)
    mid = _release("mid", 850_000, 2160, 15.0)
    lower = _release("lo", 700_000, 2160, 10.0)  # dropped (< current score)
    d = decide(
        _topsis(),
        [higher, mid, lower],
        2.0,
        "2160p Quality",
        2160,
        current_file=cur,
        allow_quality_downgrade=False,
    )
    assert d.action == "ACT" and d.release["guid"] == "hi"


def test_unknown_current_score_treated_as_upgrade():
    cur = {"id": 1, "size": int(20.0 * GB), "quality": {"quality": {"resolution": 2160}}}
    rels = [_release("a", 900_000, 2160, 10.0), _release("b", 800_000, 2160, 12.0)]
    d = decide(_topsis(), rels, 2.0, "2160p Quality", 2160, current_file=cur)
    assert d.action == "ACT"


def test_hold_when_axis_passes_but_closeness_gain_insufficient():
    # Axis gate passes (score +1000 >= min_score_delta=500) but both candidates have identical
    # size so the TOPSIS closeness barely shifts -> closeness gate blocks the grab.
    t = _topsis()
    t.cfg.min_score_delta = 500
    t.cfg.min_size_delta_gb = 0.5
    t.cfg.min_closeness_gain = 0.05
    cur = _file(900_000, 2160, 20.0)
    pick = _release("a", 901_000, 2160, 20.0)  # +1000 score, same size -> closeness gain ~0.017
    other = _release("b", 850_000, 2160, 20.0)  # lower score, same size (filler)
    d = decide(t, [pick, other], 2.0, "2160p Quality", 2160, current_file=cur)
    assert d.action == "HOLD" and d.satisfy is True
